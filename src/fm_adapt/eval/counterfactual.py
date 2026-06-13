"""Counterfactual consistency evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fm_adapt.eval.metrics import compute_metrics


def counterfactual_consistency(
    model: nn.Module,
    counterfactual_ids: np.ndarray,
    oracle_labels: np.ndarray,
    observed_ids: np.ndarray | None = None,
    observed_labels: np.ndarray | None = None,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Measure whether predictions follow causal (not spurious) features.

    Returns accuracy on counterfactual test set and optional shortcut gap
    (observed acc - counterfactual acc); large gap implies spurious reliance.
    """
    from fm_adapt.adaptation.trainer import predict_proba
    from fm_adapt.utils.device import get_device

    device = device or get_device()
    model.eval()

    cf_probs = predict_proba(model, counterfactual_ids, batch_size, device)
    cf_preds = cf_probs.argmax(axis=1)
    metrics = compute_metrics(oracle_labels, cf_preds, cf_probs)
    metrics["counterfactual_accuracy"] = metrics.pop("accuracy")

    if observed_ids is not None and observed_labels is not None:
        obs_probs = predict_proba(model, observed_ids, batch_size, device)
        obs_preds = obs_probs.argmax(axis=1)
        obs_acc = float((obs_preds == observed_labels).mean())
        metrics["observed_accuracy"] = obs_acc
        metrics["shortcut_gap"] = obs_acc - metrics["counterfactual_accuracy"]

    return metrics
