"""Parameter-efficient fine-tuning: LoRA, adapter, and prefix-tuning layers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation.

    Args:
        rank: Rank of the low-rank decomposition matrices.
        alpha: Scaling factor; effective scale = alpha / rank.
        dropout: Dropout probability applied before the low-rank path.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear.

    Computes y = W_frozen @ x + (B @ A @ x) * (alpha / r), where A ∈ R^{r×d_in}
    and B ∈ R^{d_out×r} are trainable low-rank factors.

    Args:
        linear: The frozen base linear layer to wrap.
        config: LoRA hyper-parameters.
    """

    def __init__(self, linear: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.linear = linear
        self.config = config
        in_features = linear.in_features
        out_features = linear.out_features
        self.lora_a = nn.Linear(in_features, config.rank, bias=False)
        self.lora_b = nn.Linear(config.rank, out_features, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.scaling = config.alpha / config.rank

        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen linear + scaled low-rank residual.

        Args:
            x: Input tensor of shape ``(..., d_in)``.

        Returns:
            Output tensor of shape ``(..., d_out)``.
        """
        base = self.linear(x)
        lora = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base + lora


class AdapterLayer(nn.Module):
    """Bottleneck adapter inserted after a linear layer.

    Computes y = x + W_up(GELU(W_down(x))), acting as a residual bottleneck
    with down-projection to *bottleneck* dims and back up.

    Args:
        dim: Input / output dimensionality.
        bottleneck: Hidden bottleneck dimensionality.
        dropout: Dropout probability after the activation.
    """

    def __init__(self, dim: int, bottleneck: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: identity + bottleneck residual.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of the same shape with the adapter residual added.
        """
        return x + self.up(self.dropout(self.act(self.down(x))))


class PrefixLayer(nn.Module):
    """Prepend learned soft tokens to key/value in attention.

    For a given input sequence of length L, this layer prepends *prefix_len*
    learned vectors so that the effective key and value matrices become:
        K_new = [K_prefix ; K],  V_new = [V_prefix ; V]
    where K_prefix, V_prefix ∈ R^{prefix_len × d_model} are trainable.

    Args:
        d_model: Model hidden dimensionality.
        prefix_len: Number of soft prefix tokens to prepend.
        seed: Random seed for initialising prefix parameters.
    """

    def __init__(self, d_model: int, prefix_len: int = 10, seed: int = 42) -> None:
        super().__init__()
        self.d_model = d_model
        self.prefix_len = prefix_len

        gen = torch.Generator().manual_seed(seed)
        self.key_prefix = nn.Parameter(
            torch.randn(1, prefix_len, d_model, generator=gen) * 0.02
        )
        self.value_prefix = nn.Parameter(
            torch.randn(1, prefix_len, d_model, generator=gen) * 0.02
        )
        logger.debug(
            "PrefixLayer initialised: prefix_len=%d, d_model=%d", prefix_len, d_model
        )

    def forward(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend learned prefix vectors to key and value tensors.

        Args:
            key: Key tensor of shape ``(B, L, d_model)``.
            value: Value tensor of shape ``(B, L, d_model)``.

        Returns:
            Tuple of (key_new, value_new) each of shape
            ``(B, prefix_len + L, d_model)``.
        """
        batch_size = key.size(0)
        k_prefix = self.key_prefix.expand(batch_size, -1, -1)
        v_prefix = self.value_prefix.expand(batch_size, -1, -1)
        key_new = torch.cat([k_prefix, key], dim=1)
        value_new = torch.cat([v_prefix, value], dim=1)
        return key_new, value_new


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: tuple[str, ...] = ("classifier",),
) -> list[LoRALinear]:
    """Replace target Linear layers with LoRA wrappers.

    Args:
        model: The model whose layers will be replaced in-place.
        config: LoRA hyper-parameters.
        target_modules: Suffix patterns identifying layers to wrap.

    Returns:
        List of newly created ``LoRALinear`` wrappers.
    """
    injected: list[LoRALinear] = []
    for name, module in model.named_modules():
        if not any(name.endswith(t) for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        lora_layer = LoRALinear(module, config)
        setattr(parent, child_name, lora_layer)
        injected.append(lora_layer)
    logger.info("Injected LoRA into %d layer(s)", len(injected))
    return injected


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except classifier head.

    Args:
        model: Model whose non-classifier parameters will be frozen.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


def unfreeze_all(model: nn.Module) -> None:
    """Set ``requires_grad=True`` for every parameter in the model.

    Args:
        model: Model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the number of trainable (non-frozen) scalar parameters.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Total number of trainable scalar parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_trainable_params(model: nn.Module) -> dict[str, int | float]:
    """Return a breakdown of total, trainable, frozen parameters and compression ratio.

    Compression ratio = total / trainable.  A ratio of 100 means only 1 %
    of all parameters are trainable.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Dict with keys ``total``, ``trainable``, ``frozen``, and
        ``compression_ratio``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    compression_ratio = total / trainable if trainable > 0 else float("inf")
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "compression_ratio": compression_ratio,
    }
