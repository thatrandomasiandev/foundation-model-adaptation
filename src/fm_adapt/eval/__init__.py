"""Evaluation metrics, calibration, and selective prediction."""

from fm_adapt.eval.bootstrap import bootstrap_ci
from fm_adapt.eval.calibration import calibrate_probs, expected_calibration_error, fit_temperature
from fm_adapt.eval.calibration_metrics import (
    ReliabilityBin,
    expected_calibration_error as ece_15_bins,
    maximum_calibration_error,
    reliability_diagram_data,
)
from fm_adapt.eval.counterfactual import counterfactual_consistency
from fm_adapt.eval.metrics import (
    SelectivePredictionMetrics,
    adaptation_efficiency,
    compute_metrics,
    domain_gap,
    selective_accuracy,
)
from fm_adapt.eval.risk_coverage import risk_coverage_curve

__all__ = [
    "ReliabilityBin",
    "SelectivePredictionMetrics",
    "adaptation_efficiency",
    "bootstrap_ci",
    "calibrate_probs",
    "compute_metrics",
    "counterfactual_consistency",
    "domain_gap",
    "ece_15_bins",
    "expected_calibration_error",
    "fit_temperature",
    "maximum_calibration_error",
    "reliability_diagram_data",
    "risk_coverage_curve",
    "selective_accuracy",
]
