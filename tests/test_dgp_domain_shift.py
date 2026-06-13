"""Tests for domain shift DGP invariants."""

from __future__ import annotations

import numpy as np

from fm_adapt.data.domain_shift_dgp import DomainShiftDGPConfig, generate_domain_shift_data
from fm_adapt.data.tokenizer import TokenVocab


def test_domain_shift_generates_valid_shapes() -> None:
    bundle = generate_domain_shift_data(DomainShiftDGPConfig(seed=0, n_target_test=100))
    assert bundle.source_train.n_samples > 0
    assert bundle.target_test.input_ids.shape[0] == 100
    assert bundle.target_test.input_ids.shape[1] == bundle.source_train.seq_len


def test_oracle_labels_match_causal_rule() -> None:
    bundle = generate_domain_shift_data(DomainShiftDGPConfig(seed=1))
    vocab = TokenVocab.default(bundle.vocab_size)
    causal = bundle.target_test.ground_truth["causal_tokens"]
    oracle = bundle.ground_truth["oracle_labels_target"]
    expected = np.array([(t - vocab.causal_start) % 2 for t in causal])
    np.testing.assert_array_equal(oracle, expected)


def test_target_has_weaker_spurious_signal() -> None:
    bundle = generate_domain_shift_data(
        DomainShiftDGPConfig(spurious_strength=0.9, seed=2)
    )
    source_sp = bundle.source_train.ground_truth["spurious_tokens"]
    target_sp = bundle.target_test.ground_truth["spurious_tokens"]
    assert (source_sp >= 0).sum() > (target_sp >= 0).sum()
