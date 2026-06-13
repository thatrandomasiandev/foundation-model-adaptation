"""Benchmark runner for adaptation, PEFT, and evaluation modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fm_adapt.adaptation.methods import run_adaptation
from fm_adapt.adaptation.trainer import TrainConfig, evaluate_accuracy, predict_proba, train_classifier
from fm_adapt.attribution.influence import trac_in_scores
from fm_adapt.data.counterfactual_dgp import CounterfactualDGPConfig, generate_counterfactual_data
from fm_adapt.data.domain_shift_dgp import DomainShiftDGPConfig, generate_domain_shift_data
from fm_adapt.eval.bootstrap import bootstrap_ci
from fm_adapt.eval.calibration import expected_calibration_error, fit_temperature
from fm_adapt.eval.counterfactual import counterfactual_consistency
from fm_adapt.eval.metrics import accuracy, compute_metrics, domain_gap
from fm_adapt.eval.risk_coverage import risk_coverage_curve
from fm_adapt.models.transformer import TransformerEncoder
from fm_adapt.utils.seed import config_hash, set_torch_seed


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _aggregate(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.mean([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float)) and not np.isnan(results[0][k])
    }


def _aggregate_std(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.std([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float)) and not np.isnan(results[0][k])
    }


def _predict_logits(model: torch.nn.Module, input_ids: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    probs = predict_proba(model, input_ids, batch_size)
    eps = 1e-8
    return np.log(np.clip(probs, eps, 1.0))


def run_adaptation_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Sweep adaptation methods across shift strengths."""
    seeds = config.get("seeds", [42])
    methods = config.get("methods", ["linear_probe", "lora", "full", "dann"])
    shift_levels = config.get("shift_levels", [0.5, 1.0])
    pretrain_epochs = config.get("pretrain_epochs", 3)
    adapt_epochs = config.get("adapt_epochs", 5)

    all_results = []
    for shift in shift_levels:
        for method in methods:
            seed_results = []
            for seed in seeds:
                bundle = generate_domain_shift_data(
                    DomainShiftDGPConfig(
                        shift_strength=shift,
                        spurious_strength=config.get("spurious_strength", 0.8),
                        seed=seed,
                    )
                )
                result = run_adaptation(
                    bundle,
                    method=method,
                    pretrain_epochs=pretrain_epochs,
                    adapt_epochs=adapt_epochs,
                    seed=seed,
                    lora_rank=config.get("lora_rank", 8),
                )
                test_acc = evaluate_accuracy(
                    result.model,
                    bundle.target_test.input_ids,
                    bundle.target_test.labels,
                )
                oracle = bundle.ground_truth["oracle_labels_target"]
                oracle_acc = float((bundle.target_test.labels == oracle).mean())

                seed_results.append(
                    {
                        "source_val_acc": result.source_val_acc,
                        "target_val_acc": result.target_val_acc,
                        "target_test_acc": test_acc,
                        "domain_gap": domain_gap(result.source_val_acc, result.target_val_acc),
                        "trainable_params": result.trainable_params,
                        "oracle_acc": oracle_acc,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "method": method,
                    "shift_strength": shift,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "adaptation", "results": all_results}


def run_eval_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Rigorous eval: calibration, bootstrap CIs, risk-coverage, counterfactual."""
    seeds = config.get("seeds", [42])
    all_results = []

    for seed in seeds:
        cf_bundle = generate_counterfactual_data(
            CounterfactualDGPConfig(spurious_strength=0.9, seed=seed)
        )
        set_torch_seed(seed)
        model = TransformerEncoder(
            vocab_size=cf_bundle.vocab_size,
            n_classes=2,
        )
        train_result = train_classifier(
            model,
            cf_bundle.train.input_ids,
            cf_bundle.train.labels,
            cf_bundle.val.input_ids,
            cf_bundle.val.labels,
            TrainConfig(epochs=config.get("epochs", 5)),
        )

        test_probs = predict_proba(model, cf_bundle.test.input_ids)
        test_preds = test_probs.argmax(axis=1)
        test_metrics = compute_metrics(cf_bundle.test.labels, test_preds, test_probs)

        logits = _predict_logits(model, cf_bundle.val.input_ids)
        temperature = fit_temperature(logits, cf_bundle.val.labels)
        cal_probs = predict_proba(model, cf_bundle.test.input_ids)
        ece = expected_calibration_error(cf_bundle.test.labels, cal_probs)

        boot = bootstrap_ci(cf_bundle.test.labels, test_preds, accuracy, seed=seed)

        rc = risk_coverage_curve(cf_bundle.test.labels, test_probs)

        cf_metrics = counterfactual_consistency(
            model,
            cf_bundle.counterfactual_test.input_ids,
            cf_bundle.ground_truth["oracle_counterfactual_labels"],
            cf_bundle.test.input_ids,
            cf_bundle.test.labels,
        )

        all_results.append(
            {
                "seed": seed,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "ece": ece,
                "temperature": temperature,
                "accuracy_ci_low": boot["ci_low"],
                "accuracy_ci_high": boot["ci_high"],
                "aurc": rc["aurc"],
                "counterfactual_accuracy": cf_metrics["counterfactual_accuracy"],
                "shortcut_gap": cf_metrics.get("shortcut_gap", float("nan")),
                "trainable_params": train_result.trainable_params,
            }
        )

    mean = _aggregate(all_results)
    std = _aggregate_std(all_results)
    summary = {f"{k}_mean": v for k, v in mean.items()} | {f"{k}_std": v for k, v in std.items()}
    return {"module": "eval", "results": [summary], "per_seed": all_results}


def run_attribution_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Data attribution on a small counterfactual dataset."""
    seed = config.get("seed", 42)
    cf_bundle = generate_counterfactual_data(CounterfactualDGPConfig(n_train=200, seed=seed))
    set_torch_seed(seed)
    model = TransformerEncoder(vocab_size=cf_bundle.vocab_size, n_classes=2)

    n_query = min(5, len(cf_bundle.counterfactual_test.labels))
    attr = trac_in_scores(
        model,
        cf_bundle.train.input_ids,
        cf_bundle.train.labels,
        cf_bundle.counterfactual_test.input_ids[:n_query],
        cf_bundle.counterfactual_test.labels[:n_query],
        epochs=config.get("epochs", 2),
        checkpoints=config.get("checkpoints", 2),
        seed=seed,
    )

    return {
        "module": "attribution",
        "results": [
            {
                "method": attr.method,
                "n_train": len(cf_bundle.train.labels),
                "n_query": n_query,
                "top_score": float(np.max(np.abs(attr.sample_scores))),
                "mean_abs_score": float(np.mean(np.abs(attr.sample_scores))),
            }
        ],
    }


def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    """Run benchmark(s) and write results."""
    config = load_config(config_path)
    default_path = Path(config_path).parent / "default.yaml"
    merged = {**load_config(default_path), **config} if default_path.exists() else config

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("adaptation", "all"):
        results["modules"]["adaptation"] = run_adaptation_benchmark(merged)
    if module in ("eval", "all"):
        results["modules"]["eval"] = run_eval_benchmark(merged)
    if module in ("attribution", "all"):
        results["modules"]["attribution"] = run_attribution_benchmark(merged)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    from fm_adapt.evaluation.report import write_report

    write_report(results, run_dir / "summary.md")
    return run_dir
