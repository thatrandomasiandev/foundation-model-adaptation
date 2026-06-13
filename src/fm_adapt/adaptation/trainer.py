"""Shared training utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fm_adapt.utils.device import get_device


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list[float]
    val_accuracies: list[float]
    trainable_params: int


def _make_loader(
    input_ids: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x = torch.tensor(input_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_classifier(
    model: nn.Module,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    val_ids: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train a classification model and track validation accuracy."""
    config = config or TrainConfig()
    device = get_device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    loader = _make_loader(train_ids, train_labels, config.batch_size, shuffle=True)

    train_losses: list[float] = []
    val_accuracies: list[float] = []

    for _ in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        if val_ids is not None and val_labels is not None:
            acc = evaluate_accuracy(model, val_ids, val_labels, config.batch_size, device)
            val_accuracies.append(acc)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        trainable_params=trainable,
    )


def evaluate_accuracy(
    model: nn.Module,
    input_ids: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> float:
    device = device or get_device()
    model.eval()
    loader = _make_loader(input_ids, labels, batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x).argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / max(total, 1)


def predict_proba(
    model: nn.Module,
    input_ids: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or get_device()
    model.eval()
    loader = _make_loader(input_ids, np.zeros(len(input_ids)), batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(batch_x)
            else:
                p = torch.softmax(model(batch_x), dim=-1)
            probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0)
