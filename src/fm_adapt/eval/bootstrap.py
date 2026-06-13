"""Bootstrap confidence intervals for evaluation metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI for a scalar metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores_arr = np.array(scores)
    lo = float(np.percentile(scores_arr, 100 * alpha / 2))
    hi = float(np.percentile(scores_arr, 100 * (1 - alpha / 2)))
    return {
        "point": metric_fn(y_true, y_pred),
        "ci_low": lo,
        "ci_high": hi,
        "std": float(scores_arr.std()),
    }
