"""Model ensembles with mean-logit aggregation and uncertainty decomposition."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from fm_adapt.utils.device import get_device

logger = logging.getLogger(__name__)


class ModelEnsemble(nn.Module):
    """Ensemble of independently trained models.

    Aggregates predictions from *K* constituent models by averaging their
    logits.  Predictive uncertainty is decomposed into aleatoric (within-model)
    and epistemic (between-model) components.

    Args:
        models: Sequence of models to ensemble.
    """

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("ModelEnsemble requires at least one model")
        self.models = nn.ModuleList(models)

    @property
    def n_members(self) -> int:
        """Number of models in the ensemble."""
        return len(self.models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean logits across all ensemble members.

        y = (1/K) Σ_k f_k(x)

        Args:
            x: Input tensor passed to each model.

        Returns:
            Mean logits of shape matching a single model's output.
        """
        logits = torch.stack([m(x) for m in self.models], dim=0)
        return logits.mean(dim=0)

    def member_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Collect raw logits from each ensemble member.

        Args:
            x: Input tensor passed to each model.

        Returns:
            Stacked logits of shape ``(K, batch, n_classes)``.
        """
        return torch.stack([m(x) for m in self.models], dim=0)


def ensemble_predict(
    ensemble: ModelEnsemble,
    input_ids: np.ndarray,
    batch_size: int = 64,
    device: str = "auto",
) -> np.ndarray:
    """Predict class probabilities via mean-logit aggregation.

    p(y|x) = softmax((1/K) Σ_k f_k(x))

    Args:
        ensemble: A ``ModelEnsemble`` instance.
        input_ids: Token IDs of shape ``(N, seq_len)``.
        batch_size: Inference batch size.
        device: Target device string.

    Returns:
        Probability array of shape ``(N, n_classes)``.
    """
    dev = get_device(device)
    ensemble.to(dev).eval()
    x = torch.tensor(input_ids, dtype=torch.long)
    probs_list: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(dev)
            mean_logits = ensemble(batch)
            probs = torch.softmax(mean_logits, dim=-1)
            probs_list.append(probs.cpu().numpy())

    return np.concatenate(probs_list, axis=0)


@dataclass
class UncertaintyDecomposition:
    """Decomposed predictive uncertainty from an ensemble.

    Args:
        total: Total predictive uncertainty (entropy of the mean prediction).
        aleatoric: Expected entropy within individual members.
        epistemic: Mutual information between predictions and model index.
        predictive_variance: Per-class variance across members, shape ``(N, C)``.
    """

    total: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    predictive_variance: np.ndarray


def ensemble_uncertainty(
    ensemble: ModelEnsemble,
    input_ids: np.ndarray,
    batch_size: int = 64,
    device: str = "auto",
) -> UncertaintyDecomposition:
    """Decompose predictive variance into aleatoric and epistemic components.

    Total uncertainty  H[y|x] = −Σ_c p̄_c log p̄_c  (entropy of mean probs).
    Aleatoric         = (1/K) Σ_k H[y|x, θ_k].
    Epistemic         = Total − Aleatoric  (mutual information).

    Args:
        ensemble: A ``ModelEnsemble`` instance.
        input_ids: Token IDs of shape ``(N, seq_len)``.
        batch_size: Inference batch size.
        device: Target device string.

    Returns:
        ``UncertaintyDecomposition`` with per-sample uncertainty arrays.
    """
    dev = get_device(device)
    ensemble.to(dev).eval()
    x = torch.tensor(input_ids, dtype=torch.long)

    all_member_probs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(dev)
            logits = ensemble.member_logits(batch)
            probs = torch.softmax(logits, dim=-1)
            all_member_probs.append(probs.cpu().numpy())

    member_probs = np.concatenate(all_member_probs, axis=1)

    mean_probs = member_probs.mean(axis=0)
    eps = 1e-10
    total_entropy = -np.sum(
        mean_probs * np.log(mean_probs + eps), axis=-1
    )

    per_member_entropy = -np.sum(
        member_probs * np.log(member_probs + eps), axis=-1
    )
    aleatoric_entropy = per_member_entropy.mean(axis=0)

    epistemic_entropy = total_entropy - aleatoric_entropy

    predictive_variance = member_probs.var(axis=0)

    logger.debug(
        "Uncertainty: total=%.4f, aleatoric=%.4f, epistemic=%.4f (mean)",
        total_entropy.mean(),
        aleatoric_entropy.mean(),
        epistemic_entropy.mean(),
    )

    return UncertaintyDecomposition(
        total=total_entropy,
        aleatoric=aleatoric_entropy,
        epistemic=epistemic_entropy,
        predictive_variance=predictive_variance,
    )
