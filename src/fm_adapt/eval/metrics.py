"""Classification and domain-shift evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_prob.ndim == 1:
        return float(roc_auc_score(y_true, y_prob))
    if y_prob.shape[1] == 2:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_prob.ndim == 1:
        return float(np.mean((y_prob - y_true) ** 2))
    one_hot = np.eye(y_prob.shape[1])[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def domain_gap(source_acc: float, target_acc: float) -> float:
    """Absolute accuracy drop from source to target."""
    return float(source_acc - target_acc)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
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
