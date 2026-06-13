"""Parameter-efficient fine-tuning: LoRA and adapter layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear."""

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
        base = self.linear(x)
        lora = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base + lora


class AdapterLayer(nn.Module):
    """Bottleneck adapter inserted after a linear layer."""

    def __init__(self, dim: int, bottleneck: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.dropout(self.act(self.down(x))))


def inject_lora(model: nn.Module, config: LoRAConfig, target_modules: tuple[str, ...] = ("classifier",)) -> list[LoRALinear]:
    """Replace target Linear layers with LoRA wrappers."""
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
    return injected


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except classifier head."""
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
