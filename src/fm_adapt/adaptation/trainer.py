"""Shared training utilities and the ``AdaptationTrainer`` class."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fm_adapt.utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyper-parameters.

    Args:
        epochs: Number of full passes over the training data.
        batch_size: Mini-batch size.
        lr: Peak learning rate for AdamW.
        weight_decay: L2 regularisation coefficient.
        device: Device string (``"auto"``, ``"cpu"``, ``"cuda"``).
    """

    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"


@dataclass
class TrainResult:
    """Container returned after training completes.

    Args:
        model: The trained model.
        train_losses: Per-epoch mean training loss.
        val_accuracies: Per-epoch validation accuracy (may be empty).
        trainable_params: Number of trainable parameters during training.
    """

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
    """Compute class probabilities for a batch of input sequences.

    Args:
        model: Classifier model (optionally with a ``predict_proba`` method).
        input_ids: Integer token IDs of shape ``(N, seq_len)``.
        batch_size: Inference batch size.
        device: Target device.

    Returns:
        Probability array of shape ``(N, n_classes)``.
    """
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


class AdaptationTrainer:
    """End-to-end trainer for foundation-model adaptation.

    Wraps a model and a ``TrainConfig`` into a stateful trainer with explicit
    ``train()`` / ``_train_epoch()`` / ``_eval()`` lifecycle methods.
    Loss = CrossEntropy(f(x), y) per mini-batch.

    Args:
        model: The model to train (may have partially frozen parameters).
        config: Training hyper-parameters.
        device: Device string or ``torch.device``; ``"auto"`` selects GPU
            when available.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.config = config or TrainConfig()
        self.device = get_device(device if isinstance(device, str) else str(device))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses: list[float] = []
        self.val_accuracies: list[float] = []

    def train(
        self,
        train_ids: np.ndarray,
        train_labels: np.ndarray,
        val_ids: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
    ) -> TrainResult:
        """Run the full training loop for ``config.epochs`` epochs.

        Args:
            train_ids: Training token IDs of shape ``(N, seq_len)``.
            train_labels: Training labels of shape ``(N,)``.
            val_ids: Optional validation token IDs.
            val_labels: Optional validation labels.

        Returns:
            ``TrainResult`` with the trained model, loss curve, and accuracy
            history.
        """
        train_loader = _make_loader(
            train_ids, train_labels, self.config.batch_size, shuffle=True
        )

        for epoch in range(self.config.epochs):
            epoch_loss = self._train_epoch(train_loader)
            self.train_losses.append(epoch_loss)

            if val_ids is not None and val_labels is not None:
                val_acc = self._eval(val_ids, val_labels)
                self.val_accuracies.append(val_acc)
                logger.info(
                    "Epoch %d/%d  loss=%.4f  val_acc=%.4f",
                    epoch + 1,
                    self.config.epochs,
                    epoch_loss,
                    val_acc,
                )
            else:
                logger.info(
                    "Epoch %d/%d  loss=%.4f",
                    epoch + 1,
                    self.config.epochs,
                    epoch_loss,
                )

        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return TrainResult(
            model=self.model,
            train_losses=self.train_losses,
            val_accuracies=self.val_accuracies,
            trainable_params=trainable,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        """Execute a single training epoch.

        Args:
            loader: DataLoader yielding ``(input_ids, labels)`` batches.

        Returns:
            Mean loss over all batches in the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        return epoch_loss / max(n_batches, 1)

    def _eval(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Evaluate classification accuracy on a held-out set.

        Args:
            input_ids: Validation token IDs of shape ``(N, seq_len)``.
            labels: Ground-truth labels of shape ``(N,)``.

        Returns:
            Accuracy as a float in [0, 1].
        """
        return evaluate_accuracy(
            self.model, input_ids, labels, self.config.batch_size, self.device
        )
