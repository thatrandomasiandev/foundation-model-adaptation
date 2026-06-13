"""Tests for counterfactual evaluation."""

from __future__ import annotations

from fm_adapt.adaptation.trainer import TrainConfig, train_classifier
from fm_adapt.data.counterfactual_dgp import CounterfactualDGPConfig, generate_counterfactual_data
from fm_adapt.eval.counterfactual import counterfactual_consistency
from fm_adapt.models.transformer import TransformerEncoder
from fm_adapt.utils.seed import set_torch_seed


def test_counterfactual_eval_returns_metrics() -> None:
    bundle = generate_counterfactual_data(
        CounterfactualDGPConfig(n_train=200, n_test=100, n_counterfactual=50, seed=0)
    )
    set_torch_seed(0)
    model = TransformerEncoder(vocab_size=bundle.vocab_size, n_classes=2, d_model=32, num_layers=1)
    train_classifier(
        model,
        bundle.train.input_ids,
        bundle.train.labels,
        config=TrainConfig(epochs=2, batch_size=32),
    )
    metrics = counterfactual_consistency(
        model,
        bundle.counterfactual_test.input_ids,
        bundle.ground_truth["oracle_counterfactual_labels"],
        bundle.test.input_ids,
        bundle.test.labels,
    )
    assert "counterfactual_accuracy" in metrics
    assert 0.0 <= metrics["counterfactual_accuracy"] <= 1.0
