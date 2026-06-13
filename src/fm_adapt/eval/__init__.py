"""Evaluation metrics and protocols."""

from fm_adapt.eval.bootstrap import bootstrap_ci
from fm_adapt.eval.calibration import calibrate_probs, expected_calibration_error, fit_temperature
from fm_adapt.eval.counterfactual import counterfactual_consistency
from fm_adapt.eval.metrics import compute_metrics, domain_gap
from fm_adapt.eval.risk_coverage import risk_coverage_curve

__all__ = [
    "bootstrap_ci",
    "calibrate_probs",
    "compute_metrics",
    "counterfactual_consistency",
    "domain_gap",
    "expected_calibration_error",
    "fit_temperature",
    "risk_coverage_curve",
]
