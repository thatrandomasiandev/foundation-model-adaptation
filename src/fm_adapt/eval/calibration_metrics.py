"""Extended calibration metrics: ECE, MCE, and reliability-diagram data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error with equal-width confidence bins.

    ECE = Σ_m (|B_m| / N) |acc(B_m) − conf(B_m)| where M = *n_bins* bins
    partition the confidence interval [0, 1].

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_prob: Predicted probabilities of shape ``(N, C)``.
        n_bins: Number of equal-width bins (default 15).

    Returns:
        ECE value in [0, 1] (lower is better).
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    n = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        bin_size = mask.sum()
        if bin_size == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error — worst-case bin deviation.

    MCE = max_m |acc(B_m) − conf(B_m)| over all non-empty bins.

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_prob: Predicted probabilities of shape ``(N, C)``.
        n_bins: Number of equal-width bins (default 15).

    Returns:
        MCE value in [0, 1] (lower is better).
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))
    return float(mce)


@dataclass
class ReliabilityBin:
    """Data for a single bin in a reliability diagram.

    Args:
        bin_lower: Lower edge of the confidence bin.
        bin_upper: Upper edge of the confidence bin.
        bin_mid: Midpoint of the bin.
        accuracy: Mean accuracy of samples falling in this bin.
        confidence: Mean confidence of samples in this bin.
        count: Number of samples in the bin.
        gap: Signed calibration gap (accuracy − confidence).
    """

    bin_lower: float
    bin_upper: float
    bin_mid: float
    accuracy: float
    confidence: float
    count: int
    gap: float


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> list[ReliabilityBin]:
    """Return per-bin data for plotting a reliability diagram.

    Each bin spans an equal-width interval of the predicted confidence.
    Empty bins are included with zero counts.

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_prob: Predicted probabilities of shape ``(N, C)``.
        n_bins: Number of equal-width bins (default 15).

    Returns:
        List of ``ReliabilityBin`` objects, one per bin.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []

    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                ReliabilityBin(
                    bin_lower=lo,
                    bin_upper=hi,
                    bin_mid=(lo + hi) / 2,
                    accuracy=0.0,
                    confidence=0.0,
                    count=0,
                    gap=0.0,
                )
            )
        else:
            bin_acc = float(accuracies[mask].mean())
            bin_conf = float(confidences[mask].mean())
            bins.append(
                ReliabilityBin(
                    bin_lower=lo,
                    bin_upper=hi,
                    bin_mid=(lo + hi) / 2,
                    accuracy=bin_acc,
                    confidence=bin_conf,
                    count=count,
                    gap=bin_acc - bin_conf,
                )
            )

    logger.debug("Reliability diagram: %d bins, %d non-empty", n_bins, sum(1 for b in bins if b.count > 0))
    return bins
