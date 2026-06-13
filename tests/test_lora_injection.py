"""Tests for LoRA injection and parameter counts."""

from __future__ import annotations

from fm_adapt.models.peft import LoRAConfig, freeze_backbone, inject_lora, trainable_parameter_count
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
    import torch

    model = TransformerEncoder(vocab_size=64, n_classes=2, d_model=32, num_layers=1)
    freeze_backbone(model)
    inject_lora(model, LoRAConfig(rank=4))
    x = torch.randint(1, 64, (4, 8))
    logits = model(x)
    assert logits.shape == (4, 2)
