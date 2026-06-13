"""Tests for calibration metrics."""

from __future__ import annotations

import numpy as np

from fm_adapt.eval.calibration import expected_calibration_error, fit_temperature


def test_ece_perfect_calibration() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece < 0.15


def test_temperature_fitting_runs() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 2))
    labels = rng.integers(0, 2, size=50)
    t = fit_temperature(logits, labels, max_iter=10)
    assert t > 0
