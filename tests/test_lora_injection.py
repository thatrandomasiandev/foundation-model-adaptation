"""Tests for LoRA injection, adapters, and parameter-efficient fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn

from fm_adapt.models.peft import (
    AdapterLayer,
    LoRAConfig,
    LoRALinear,
    count_trainable_params,
    freeze_backbone,
    inject_lora,
    trainable_parameter_count,
)
from fm_adapt.models.transformer import TransformerEncoder


def test_lora_reduces_trainable_params() -> None:
    model = TransformerEncoder(vocab_size=64, n_classes=2)
    total = model.count_parameters()
    freeze_backbone(model)
    inject_lora(model, LoRAConfig(rank=4))
    trainable = trainable_parameter_count(model)
    assert trainable < total
    assert trainable > 0


def test_lora_forward_pass() -> None:
    model = TransformerEncoder(vocab_size=64, n_classes=2, d_model=32, num_layers=1)
    freeze_backbone(model)
    inject_lora(model, LoRAConfig(rank=4))
    x = torch.randint(1, 64, (4, 8))
    logits = model(x)
    assert logits.shape == (4, 2)


def test_lora_linear_equals_frozen_at_init() -> None:
    """LoRALinear output equals W_frozen @ x when B is initialised to zero."""
    base_linear = nn.Linear(16, 8)
    lora = LoRALinear(base_linear, LoRAConfig(rank=4, alpha=8.0, dropout=0.0))
    lora.eval()

    x = torch.randn(3, 16)
    with torch.no_grad():
        lora_out = lora(x)
        base_out = base_linear(x)

    torch.testing.assert_close(lora_out, base_out, atol=1e-6, rtol=1e-5)


def test_inject_lora_compression_above_90_percent() -> None:
    """At rank=4, LoRA should reduce trainable params by >90% vs full model."""
    model = TransformerEncoder(vocab_size=128, n_classes=4, d_model=64, num_layers=2)
    total = model.count_parameters()
    freeze_backbone(model)
    inject_lora(model, LoRAConfig(rank=4))
    trainable = trainable_parameter_count(model)
    reduction = 1.0 - (trainable / total)
    assert reduction > 0.90, f"Expected >90% reduction, got {reduction:.2%}"


def test_adapter_starts_as_identity() -> None:
    """AdapterLayer output ≈ input when W_up is initialised to zero."""
    adapter = AdapterLayer(dim=32, bottleneck=8, dropout=0.0)
    nn.init.zeros_(adapter.up.weight)
    nn.init.zeros_(adapter.up.bias)
    adapter.eval()

    x = torch.randn(5, 32)
    with torch.no_grad():
        out = adapter(x)

    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-5)


def test_full_finetune_beats_lora_on_toy_task() -> None:
    """Full fine-tune achieves lower loss than LoRA on a 100-sample toy task."""
    torch.manual_seed(42)
    vocab_size, n_classes, d_model = 32, 2, 32

    x = torch.randint(1, vocab_size, (100, 10))
    y = torch.randint(0, n_classes, (100,))

    def _train(model: nn.Module, epochs: int = 30) -> float:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )
        criterion = nn.CrossEntropyLoss()
        model.train()
        loss_val = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
        return loss_val

    model_full = TransformerEncoder(
        vocab_size=vocab_size, n_classes=n_classes, d_model=d_model, num_layers=1
    )
    torch.manual_seed(42)
    loss_full = _train(model_full)

    model_lora = TransformerEncoder(
        vocab_size=vocab_size, n_classes=n_classes, d_model=d_model, num_layers=1
    )
    freeze_backbone(model_lora)
    inject_lora(model_lora, LoRAConfig(rank=4, alpha=8.0))
    torch.manual_seed(42)
    loss_lora = _train(model_lora)

    assert loss_full < loss_lora, (
        f"Full fine-tune loss ({loss_full:.4f}) should be lower than "
        f"LoRA loss ({loss_lora:.4f}) on a 100-sample toy task"
    )
