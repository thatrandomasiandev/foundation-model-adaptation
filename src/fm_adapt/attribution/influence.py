"""Training data attribution via gradient-based influence functions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fm_adapt.utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Container for influence-score results.

    Args:
        sample_scores: Per-training-sample influence scores.
        top_indices: Indices of the most influential training examples.
        method: Name of the attribution algorithm used.
    """

    sample_scores: np.ndarray
    top_indices: np.ndarray
    method: str


# ---------------------------------------------------------------------------
# Legacy functional API
# ---------------------------------------------------------------------------


def trac_in_scores(
    model: nn.Module,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    query_ids: np.ndarray,
    query_labels: np.ndarray,
    checkpoints: int = 3,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> AttributionResult:
    """Approximate TracIn influence: sum of gradient dot-products at checkpoints.

    score(z_train, z_query) ≈ Σ_t η_t ⟨∇ℓ(z_train, θ_t), ∇ℓ(z_query, θ_t)⟩.
    Higher score means the training example is more influential.

    Args:
        model: Classifier model to attribute.
        train_ids: Training token IDs ``(N_train, seq_len)``.
        train_labels: Training labels ``(N_train,)``.
        query_ids: Query token IDs ``(N_query, seq_len)``.
        query_labels: Query labels ``(N_query,)``.
        checkpoints: Number of gradient checkpoints to save.
        epochs: Training epochs for checkpoint collection.
        batch_size: Mini-batch size.
        lr: Learning rate.
        seed: Random seed.

    Returns:
        ``AttributionResult`` with per-sample scores and top-k indices.
    """
    torch.manual_seed(seed)
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_x = torch.tensor(train_ids, dtype=torch.long)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )

    saved_grads: list[list[torch.Tensor]] = []

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            saved_grads.append(
                [
                    p.grad.detach().clone()
                    if p.grad is not None
                    else torch.zeros_like(p)
                    for p in model.parameters()
                ]
            )
            optimizer.step()
            if len(saved_grads) >= checkpoints:
                break
        if len(saved_grads) >= checkpoints:
            break

    model.eval()
    query_x = torch.tensor(query_ids, dtype=torch.long).to(device)
    query_y = torch.tensor(query_labels, dtype=torch.long).to(device)

    query_grads: list[list[torch.Tensor]] = []
    for qx, qy in zip(query_x, query_y, strict=True):
        model.zero_grad()
        loss = criterion(model(qx.unsqueeze(0)), qy.unsqueeze(0))
        loss.backward()
        query_grads.append(
            [
                p.grad.detach().clone()
                if p.grad is not None
                else torch.zeros_like(p)
                for p in model.parameters()
            ]
        )

    scores = np.zeros(len(train_ids))
    for i in range(len(train_ids)):
        tx = train_x[i : i + 1].to(device)
        ty = train_y[i : i + 1].to(device)
        for q_grad in query_grads:
            model.zero_grad()
            loss = criterion(model(tx), ty)
            loss.backward()
            train_grad = [
                p.grad.detach() if p.grad is not None else torch.zeros_like(p)
                for p in model.parameters()
            ]
            influence = 0.0
            for ckpt_grad, t_grad in zip(saved_grads, train_grad, strict=True):
                for cg, tg in zip(ckpt_grad, t_grad, strict=True):
                    influence += float((cg.flatten() @ tg.flatten()).item())
            scores[i] += influence / max(len(saved_grads), 1)

    top_indices = np.argsort(-np.abs(scores))[: min(10, len(scores))]
    return AttributionResult(
        sample_scores=scores, top_indices=top_indices, method="trac_in"
    )


# ---------------------------------------------------------------------------
# Class-based influence estimators
# ---------------------------------------------------------------------------


class InfluenceEstimator:
    """Influence-function estimator using LiSSA for inverse-Hessian-vector products.

    Approximates I(z_train, z_query) = −∇ℓ(z_query)ᵀ H⁻¹ ∇ℓ(z_train) where
    H is the Hessian of the empirical risk and H⁻¹v is estimated via the
    LiSSA recursion (Linear time Stochastic Second-order Algorithm).

    Args:
        model: Classifier model to attribute.
        criterion: Loss function (defaults to ``CrossEntropyLoss``).
        device: Target device.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | None = None,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = get_device(device)
        self.model.to(self.device)

    def _param_grad(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute per-sample parameter gradients.

        Args:
            input_ids: Token IDs ``(1, seq_len)``.
            labels: Label ``(1,)``.

        Returns:
            List of gradient tensors, one per parameter.
        """
        self.model.zero_grad()
        logits = self.model(input_ids)
        loss = self.criterion(logits, labels)
        loss.backward()
        return [
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in self.model.parameters()
        ]

    def _ihvp_lissa(
        self,
        v: list[torch.Tensor],
        train_loader: DataLoader,
        damping: float = 0.01,
        depth: int = 100,
        scale: float = 25.0,
        seed: int = 42,
    ) -> list[torch.Tensor]:
        """Estimate H⁻¹v via the LiSSA recursion.

        Recursion: h_{t+1} = v + (1 − damping) h_t − (1/scale) ∇²ℓ · h_t,
        converging to H⁻¹v.

        Args:
            v: The vector to multiply by the inverse Hessian.
            train_loader: DataLoader over the training set for HVP estimation.
            damping: Tikhonov damping term for stability.
            depth: Number of recursion unrolling steps.
            scale: Scaling factor for the Hessian estimate.
            seed: Random seed for reproducibility.

        Returns:
            List of tensors approximating H⁻¹v.
        """
        torch.manual_seed(seed)
        h = [vi.clone() for vi in v]

        loader_iter = iter(train_loader)
        for _ in range(depth):
            try:
                batch_x, batch_y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch_x, batch_y = next(loader_iter)

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.model.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            params = [p for p in self.model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True)

            hvp_val = torch.autograd.grad(
                grads,
                params,
                grad_outputs=h[: len(params)],
                retain_graph=False,
            )

            param_idx = 0
            for i, p in enumerate(self.model.parameters()):
                if p.requires_grad:
                    h[i] = v[i] + (1 - damping) * h[i] - hvp_val[param_idx] / scale
                    param_idx += 1

        return h

    def compute_influence(
        self,
        train_ids: np.ndarray,
        train_labels: np.ndarray,
        query_ids: np.ndarray,
        query_labels: np.ndarray,
        damping: float = 0.01,
        lissa_depth: int = 100,
        batch_size: int = 32,
        seed: int = 42,
    ) -> AttributionResult:
        """Compute influence scores for every training example w.r.t. queries.

        I(z_train, z_query) = −⟨∇ℓ(z_query), H⁻¹ ∇ℓ(z_train)⟩ approximated
        with LiSSA.

        Args:
            train_ids: Training token IDs ``(N_train, seq_len)``.
            train_labels: Training labels ``(N_train,)``.
            query_ids: Query token IDs ``(N_query, seq_len)``.
            query_labels: Query labels ``(N_query,)``.
            damping: LiSSA damping.
            lissa_depth: Number of LiSSA recursion steps.
            batch_size: Batch size for Hessian estimation.
            seed: Random seed.

        Returns:
            ``AttributionResult`` with influence scores.
        """
        self.model.eval()
        train_x = torch.tensor(train_ids, dtype=torch.long)
        train_y = torch.tensor(train_labels, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
        )

        query_x = torch.tensor(query_ids, dtype=torch.long).to(self.device)
        query_y = torch.tensor(query_labels, dtype=torch.long).to(self.device)

        avg_query_grad: list[torch.Tensor] = []
        for qx, qy in zip(query_x, query_y, strict=True):
            g = self._param_grad(qx.unsqueeze(0), qy.unsqueeze(0))
            if not avg_query_grad:
                avg_query_grad = g
            else:
                avg_query_grad = [a + b for a, b in zip(avg_query_grad, g, strict=True)]
        avg_query_grad = [g / len(query_x) for g in avg_query_grad]

        ihvp = self._ihvp_lissa(
            avg_query_grad,
            train_loader,
            damping=damping,
            depth=lissa_depth,
            seed=seed,
        )

        scores = np.zeros(len(train_ids))
        for i in range(len(train_ids)):
            tx = train_x[i : i + 1].to(self.device)
            ty = train_y[i : i + 1].to(self.device)
            g = self._param_grad(tx, ty)
            score = sum(
                float((hi.flatten() @ gi.flatten()).item())
                for hi, gi in zip(ihvp, g, strict=True)
            )
            scores[i] = -score

        top_indices = np.argsort(-np.abs(scores))[: min(10, len(scores))]
        logger.info("Computed influence scores for %d training samples", len(scores))
        return AttributionResult(
            sample_scores=scores, top_indices=top_indices, method="influence_lissa"
        )

    def top_k_influential(
        self,
        train_ids: np.ndarray,
        train_labels: np.ndarray,
        query_ids: np.ndarray,
        query_labels: np.ndarray,
        k: int = 10,
        **kwargs: float | int,
    ) -> AttributionResult:
        """Return only the top-k most influential training examples.

        Convenience wrapper around ``compute_influence`` that limits the
        result to the *k* highest-magnitude scores.

        Args:
            train_ids: Training token IDs.
            train_labels: Training labels.
            query_ids: Query token IDs.
            query_labels: Query labels.
            k: Number of top influential examples to return.
            **kwargs: Forwarded to ``compute_influence``.

        Returns:
            ``AttributionResult`` whose ``top_indices`` has at most *k* entries.
        """
        result = self.compute_influence(
            train_ids, train_labels, query_ids, query_labels, **kwargs
        )
        top_k = np.argsort(-np.abs(result.sample_scores))[: min(k, len(result.sample_scores))]
        return AttributionResult(
            sample_scores=result.sample_scores,
            top_indices=top_k,
            method=result.method,
        )


class TracIn:
    """Simplified TracIn influence estimator.

    Approximates influence via gradient dot-products at checkpoints saved
    during training: score ≈ Σ_t η_t ⟨∇ℓ(z_train, θ_t), ∇ℓ(z_query, θ_t)⟩.

    Args:
        model: Classifier model.
        lr: Learning rate (used as the per-step weight η).
        criterion: Loss function (defaults to ``CrossEntropyLoss``).
        device: Target device.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        criterion: nn.Module | None = None,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.lr = lr
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = get_device(device)
        self.model.to(self.device)
        self._checkpoints: list[list[torch.Tensor]] = []

    def save_checkpoint(self) -> None:
        """Snapshot current parameter gradients as a checkpoint.

        Call this after ``loss.backward()`` during training to accumulate
        gradient checkpoints for later attribution.
        """
        grads = [
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in self.model.parameters()
        ]
        self._checkpoints.append(grads)
        logger.debug("Saved checkpoint %d", len(self._checkpoints))

    def compute_influence(
        self,
        train_ids: np.ndarray,
        train_labels: np.ndarray,
        query_ids: np.ndarray,
        query_labels: np.ndarray,
        checkpoints: int = 3,
        epochs: int = 3,
        batch_size: int = 32,
        seed: int = 42,
    ) -> AttributionResult:
        """Compute TracIn influence scores between training and query sets.

        If no checkpoints have been manually saved, this method will train
        for a few steps and collect them automatically.

        Args:
            train_ids: Training token IDs ``(N_train, seq_len)``.
            train_labels: Training labels ``(N_train,)``.
            query_ids: Query token IDs ``(N_query, seq_len)``.
            query_labels: Query labels ``(N_query,)``.
            checkpoints: Number of gradient checkpoints to collect.
            epochs: Epochs for automatic checkpoint collection.
            batch_size: Mini-batch size.
            seed: Random seed.

        Returns:
            ``AttributionResult`` with per-sample TracIn scores.
        """
        if not self._checkpoints:
            self._collect_checkpoints(
                train_ids, train_labels, checkpoints, epochs, batch_size, seed
            )

        self.model.eval()
        query_x = torch.tensor(query_ids, dtype=torch.long).to(self.device)
        query_y = torch.tensor(query_labels, dtype=torch.long).to(self.device)

        query_grads: list[list[torch.Tensor]] = []
        for qx, qy in zip(query_x, query_y, strict=True):
            self.model.zero_grad()
            loss = self.criterion(
                self.model(qx.unsqueeze(0)), qy.unsqueeze(0)
            )
            loss.backward()
            query_grads.append(
                [
                    p.grad.detach().clone()
                    if p.grad is not None
                    else torch.zeros_like(p)
                    for p in self.model.parameters()
                ]
            )

        train_x = torch.tensor(train_ids, dtype=torch.long)
        train_y = torch.tensor(train_labels, dtype=torch.long)
        scores = np.zeros(len(train_ids))

        for i in range(len(train_ids)):
            tx = train_x[i : i + 1].to(self.device)
            ty = train_y[i : i + 1].to(self.device)
            for q_grad in query_grads:
                self.model.zero_grad()
                loss = self.criterion(self.model(tx), ty)
                loss.backward()
                t_grad = [
                    p.grad.detach()
                    if p.grad is not None
                    else torch.zeros_like(p)
                    for p in self.model.parameters()
                ]
                influence = 0.0
                for cg, tg in zip(
                    self._checkpoints[0], t_grad, strict=True
                ):
                    influence += float(
                        (cg.flatten() @ tg.flatten()).item()
                    )
                scores[i] += self.lr * influence

        top_indices = np.argsort(-np.abs(scores))[: min(10, len(scores))]
        logger.info("TracIn scores computed for %d samples", len(scores))
        return AttributionResult(
            sample_scores=scores, top_indices=top_indices, method="tracin"
        )

    def _collect_checkpoints(
        self,
        train_ids: np.ndarray,
        train_labels: np.ndarray,
        n_checkpoints: int,
        epochs: int,
        batch_size: int,
        seed: int,
    ) -> None:
        """Automatically train and collect gradient checkpoints.

        Args:
            train_ids: Training token IDs.
            train_labels: Training labels.
            n_checkpoints: Target number of checkpoints to save.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            seed: Random seed.
        """
        torch.manual_seed(seed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        train_x = torch.tensor(train_ids, dtype=torch.long)
        train_y = torch.tensor(train_labels, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
        )

        for _ in range(epochs):
            for bx, by in loader:
                self.model.train()
                optimizer.zero_grad()
                loss = self.criterion(
                    self.model(bx.to(self.device)), by.to(self.device)
                )
                loss.backward()
                self.save_checkpoint()
                optimizer.step()
                if len(self._checkpoints) >= n_checkpoints:
                    return
