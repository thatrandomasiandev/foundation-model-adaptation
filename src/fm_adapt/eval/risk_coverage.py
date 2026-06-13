"""Risk-coverage curves for selective prediction."""

from __future__ import annotations

import numpy as np


def risk_coverage_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 20,
) -> dict[str, np.ndarray]:
    """
    Compute risk (error rate) vs coverage as we retain highest-confidence predictions.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    order = np.argsort(-confidences)

    coverages: list[float] = []
    risks: list[float] = []
    n = len(y_true)
    thresholds = np.linspace(0, 1, n_points)

    for thresh in thresholds:
        mask = confidences >= thresh
        if mask.sum() == 0:
            continue
        coverages.append(mask.mean())
        risks.append(1.0 - correct[mask].mean())

    return {
        "coverage": np.array(coverages),
        "risk": np.array(risks),
        "aurc": float(np.trapz(risks, coverages)) if len(coverages) > 1 else float("nan"),
    }
