"""Adaptation strategies: full fine-tune, linear probe, LoRA, DANN."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fm_adapt.adaptation.trainer import TrainConfig, TrainResult, train_classifier
from fm_adapt.data.base import AdaptationDataset, DomainShiftBundle
from fm_adapt.models.peft import LoRAConfig, freeze_backbone, inject_lora, unfreeze_all
from fm_adapt.models.transformer import TransformerEncoder
from fm_adapt.utils.device import get_device
from fm_adapt.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


AdaptationMethod = Literal["full", "linear_probe", "lora", "dann"]


@dataclass
class AdaptationConfig:
    """Unified configuration for all adaptation methods.

    Bundles method selection with PEFT hyper-parameters and training schedule
    so a single object drives the full adaptation pipeline.

    Args:
        method: Which adaptation strategy to apply.
        rank: LoRA rank (ignored for non-LoRA methods).
        alpha: LoRA scaling numerator (alpha / rank).
        prefix_len: Number of soft prefix tokens (prefix-tuning only).
        lr: Learning rate for the adaptation optimiser.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
    """

    method: AdaptationMethod = "lora"
    rank: int = 8
    alpha: float = 16.0
    prefix_len: int = 10
    lr: float = 5e-4
    epochs: int = 5
    batch_size: int = 32


@dataclass
class AdaptationResult:
    """Container for adaptation experiment outputs.

    Args:
        method: Name of the adaptation method used.
        model: The adapted model.
        train_result: Detailed training history.
        source_val_acc: Accuracy on the source validation split.
        target_val_acc: Accuracy on the target validation split.
        trainable_params: Number of trainable parameters during adaptation.
    """

    method: str
    model: nn.Module
    train_result: TrainResult
    source_val_acc: float
    target_val_acc: float
    trainable_params: int


def _freeze_backbone(model: nn.Module) -> int:
    """Freeze every parameter in the model and return the total count.

    Sets ``requires_grad = False`` on all parameters, preparing the model
    for selective unfreezing of adapter / LoRA layers.

    Args:
        model: Model whose parameters will all be frozen.

    Returns:
        Total number of scalar parameters that were frozen.
    """
    total = 0
    for param in model.parameters():
        param.requires_grad = False
        total += param.numel()
    logger.info("Froze %d parameters", total)
    return total


def _compute_flop_ratio(model: nn.Module) -> float:
    """Estimate the FLOPs ratio for adapted vs full model forward pass.

    Approximation: each trainable parameter contributes ~2 FLOPs per sample
    (one multiply, one add).  The ratio is trainable_flops / total_flops.

    Args:
        model: Model with a mix of frozen and trainable parameters.

    Returns:
        Ratio in [0, 1]; 1.0 means full fine-tuning, lower means cheaper.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total == 0:
        return 0.0
    ratio = trainable / total
    logger.debug("FLOPs ratio: %.4f (trainable=%d, total=%d)", ratio, trainable, total)
    return ratio


def _build_model(vocab_size: int, n_classes: int, seed: int) -> TransformerEncoder:
    set_torch_seed(seed)
    return TransformerEncoder(vocab_size=vocab_size, n_classes=n_classes)


def _pretrain_on_source(
    model: TransformerEncoder,
    source_train: AdaptationDataset,
    source_val: AdaptationDataset,
    config: TrainConfig,
) -> TrainResult:
    return train_classifier(
        model,
        source_train.input_ids,
        source_train.labels,
        source_val.input_ids,
        source_val.labels,
        config,
    )


class DomainClassifier(nn.Module):
    """Domain discriminator for DANN."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def train_dann(
    model: TransformerEncoder,
    source_train: AdaptationDataset,
    target_train: AdaptationDataset,
    source_val: AdaptationDataset,
    target_val: AdaptationDataset,
    config: TrainConfig | None = None,
    lambda_domain: float = 0.5,
) -> TrainResult:
    """Domain-adversarial neural network training."""
    config = config or TrainConfig()
    device = get_device(config.device)
    model = model.to(device)
    domain_clf = DomainClassifier(model.classifier.in_features).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(domain_clf.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    clf_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn = nn.CrossEntropyLoss()

    src_x = torch.tensor(source_train.input_ids, dtype=torch.long)
    src_y = torch.tensor(source_train.labels, dtype=torch.long)
    tgt_x = torch.tensor(target_train.input_ids, dtype=torch.long)
    n = min(len(src_x), len(tgt_x))
    src_x, src_y = src_x[:n], src_y[:n]
    tgt_x = tgt_x[:n]

    train_losses: list[float] = []
    val_accuracies: list[float] = []

    for _ in range(config.epochs):
        model.train()
        domain_clf.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            bx_s = src_x[idx].to(device)
            by_s = src_y[idx].to(device)
            bx_t = tgt_x[idx].to(device)

            optimizer.zero_grad()
            feat_s = model.encode(bx_s)
            feat_t = model.encode(bx_t)
            logits = model.classifier(feat_s)
            task_loss = clf_loss_fn(logits, by_s)

            rev_s = GradientReversal.apply(feat_s, lambda_domain)
            rev_t = GradientReversal.apply(feat_t, lambda_domain)
            dom_logits = domain_clf(torch.cat([rev_s, rev_t], dim=0))
            dom_labels = torch.cat(
                [torch.zeros(len(bx_s), dtype=torch.long), torch.ones(len(bx_t), dtype=torch.long)]
            ).to(device)
            domain_loss = domain_loss_fn(dom_logits, dom_labels)

            loss = task_loss + domain_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))
        val_ids = np.concatenate([source_val.input_ids, target_val.input_ids])
        val_labels = np.concatenate([source_val.labels, target_val.labels])
        from fm_adapt.adaptation.trainer import evaluate_accuracy

        val_accuracies.append(evaluate_accuracy(model, val_ids, val_labels, config.batch_size, device))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        trainable_params=trainable,
    )


def run_adaptation(
    bundle: DomainShiftBundle,
    method: AdaptationMethod = "lora",
    n_classes: int = 2,
    pretrain_epochs: int = 3,
    adapt_epochs: int = 5,
    seed: int = 42,
    lora_rank: int = 8,
) -> AdaptationResult:
    """Pretrain on source, then adapt to target with the chosen method."""
    set_torch_seed(seed)
    model = _build_model(bundle.vocab_size, n_classes, seed)

    pretrain_cfg = TrainConfig(epochs=pretrain_epochs, batch_size=32, lr=1e-3)
    _pretrain_on_source(model, bundle.source_train, bundle.source_val, pretrain_cfg)

    adapt_cfg = TrainConfig(epochs=adapt_epochs, batch_size=32, lr=5e-4)

    if method == "full":
        unfreeze_all(model)
        train_result = train_classifier(
            model,
            bundle.target_train.input_ids,
            bundle.target_train.labels,
            bundle.target_val.input_ids,
            bundle.target_val.labels,
            adapt_cfg,
        )
    elif method == "linear_probe":
        freeze_backbone(model)
        train_result = train_classifier(
            model,
            bundle.target_train.input_ids,
            bundle.target_train.labels,
            bundle.target_val.input_ids,
            bundle.target_val.labels,
            adapt_cfg,
        )
    elif method == "lora":
        freeze_backbone(model)
        inject_lora(model, LoRAConfig(rank=lora_rank))
        train_result = train_classifier(
            model,
            bundle.target_train.input_ids,
            bundle.target_train.labels,
            bundle.target_val.input_ids,
            bundle.target_val.labels,
            adapt_cfg,
        )
    elif method == "dann":
        unfreeze_all(model)
        train_result = train_dann(
            model,
            bundle.source_train,
            bundle.target_train,
            bundle.source_val,
            bundle.target_val,
            adapt_cfg,
        )
    else:
        raise ValueError(f"Unknown adaptation method: {method}")

    from fm_adapt.adaptation.trainer import evaluate_accuracy

    device = get_device()
    source_acc = evaluate_accuracy(model, bundle.source_val.input_ids, bundle.source_val.labels, device=device)
    target_acc = evaluate_accuracy(model, bundle.target_val.input_ids, bundle.target_val.labels, device=device)

    return AdaptationResult(
        method=method,
        model=model,
        train_result=train_result,
        source_val_acc=source_acc,
        target_val_acc=target_acc,
        trainable_params=train_result.trainable_params,
    )
