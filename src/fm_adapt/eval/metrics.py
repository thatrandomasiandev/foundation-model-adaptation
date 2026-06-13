"""Classification and domain-shift evaluation metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy = (1/N) Σ 𝟙[ŷ_i = y_i].

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_pred: Predicted labels of shape ``(N,)``.

    Returns:
        Accuracy in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 = (1/C) Σ_c F1_c.

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_pred: Predicted labels of shape ``(N,)``.

    Returns:
        Macro F1 in [0, 1].
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute area under the ROC curve (one-vs-rest for multi-class).

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_prob: Predicted probabilities, shape ``(N,)`` or ``(N, C)``.

    Returns:
        AUROC in [0, 1].
    """
    if y_prob.ndim == 1:
        return float(roc_auc_score(y_true, y_prob))
    if y_prob.shape[1] == 2:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score = (1/N) Σ ||p_i − e_{y_i}||².

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_prob: Predicted probabilities, ``(N,)`` for binary or ``(N, C)``.

    Returns:
        Brier score (lower is better).
    """
    if y_prob.ndim == 1:
        return float(np.mean((y_prob - y_true) ** 2))
    one_hot = np.eye(y_prob.shape[1])[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def domain_gap(source_acc: float, target_acc: float) -> float:
    """Absolute accuracy drop from source to target: Δ = acc_src − acc_tgt.

    Args:
        source_acc: Source-domain accuracy.
        target_acc: Target-domain accuracy.

    Returns:
        Non-negative gap (positive means performance degrades on target).
    """
    return float(source_acc - target_acc)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a standard suite of classification metrics.

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_pred: Predicted labels of shape ``(N,)``.
        y_prob: Optional predicted probabilities for probabilistic metrics.

    Returns:
        Dict mapping metric name to value.
    """
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }
    if y_prob is not None:
        out["brier"] = brier_score(y_true, y_prob)
        try:
            out["auroc"] = auroc(y_true, y_prob)
        except ValueError:
            out["auroc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Selective prediction & adaptation efficiency
# ---------------------------------------------------------------------------


@dataclass
class SelectivePredictionMetrics:
    """Metrics for selective (reject-option) classification.

    At a given coverage level, only the most confident predictions are
    retained and the rest are abstained on.

    Args:
        coverage: Fraction of samples on which a prediction is made.
        selective_accuracy: Accuracy on the non-abstained subset.
        n_predicted: Number of samples actually predicted.
        n_total: Total number of samples.
    """

    coverage: float
    selective_accuracy: float
    n_predicted: int
    n_total: int


def selective_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    coverage: float = 0.8,
) -> SelectivePredictionMetrics:
    """Accuracy on the top-*coverage* fraction of most-confident predictions.

    Given confidence c_i = max_k p(k|x_i), retain the top-*coverage* fraction
    of samples and report accuracy only on those.

    Args:
        y_true: Ground-truth labels of shape ``(N,)``.
        y_prob: Predicted probabilities of shape ``(N, C)``.
        coverage: Fraction of samples to retain, in (0, 1].

    Returns:
        ``SelectivePredictionMetrics`` with the selective accuracy and counts.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    n_total = len(y_true)
    n_keep = max(1, int(np.ceil(coverage * n_total)))

    top_indices = np.argsort(-confidences)[:n_keep]
    sel_acc = float(np.mean(predictions[top_indices] == y_true[top_indices]))
    actual_coverage = n_keep / n_total

    logger.debug(
        "Selective accuracy: %.4f at coverage %.2f (%d / %d)",
        sel_acc,
        actual_coverage,
        n_keep,
        n_total,
    )
    return SelectivePredictionMetrics(
        coverage=actual_coverage,
        selective_accuracy=sel_acc,
        n_predicted=n_keep,
        n_total=n_total,
    )


def adaptation_efficiency(
    task_performance: float,
    trainable_params: int,
) -> float:
    """Task performance per trainable parameter: efficiency = perf / params.

    A higher value means the method achieved more with fewer parameters.

    Args:
        task_performance: A scalar performance metric (e.g. accuracy).
        trainable_params: Number of trainable parameters used.

    Returns:
        Efficiency ratio (performance per parameter).
    """
    if trainable_params <= 0:
        return float("inf") if task_performance > 0 else 0.0
    return task_performance / trainable_params
