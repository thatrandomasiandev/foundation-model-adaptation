"""Counterfactual evaluation data with causal vs spurious features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fm_adapt.data.base import AdaptationDataset, CounterfactualBundle
from fm_adapt.data.tokenizer import TokenVocab, pad_batch, sample_sequence
from fm_adapt.utils.seed import set_seed


@dataclass
class CounterfactualDGPConfig:
    n_train: int = 1000
    n_val: int = 200
    n_test: int = 500
    n_counterfactual: int = 500
    seq_len: int = 16
    vocab_size: int = 128
    spurious_strength: float = 0.9
    noise_rate: float = 0.02
    n_classes: int = 2
    seed: int = 42


def _label_from_causal(causal_token: int, vocab: TokenVocab, n_classes: int) -> int:
    causal_idx = causal_token - vocab.causal_start
    return int(causal_idx % n_classes)


def _generate_observed(
    rng: np.random.Generator,
    vocab: TokenVocab,
    n_samples: int,
    config: CounterfactualDGPConfig,
) -> AdaptationDataset:
    causal_tokens = vocab.causal_tokens()
    spurious_tokens = vocab.spurious_tokens()
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    causal_used: list[int] = []
    spurious_used: list[int] = []

    for _ in range(n_samples):
        causal_token = int(rng.choice(causal_tokens))
        label = _label_from_causal(causal_token, vocab, config.n_classes)
        spurious_token = int(rng.choice(spurious_tokens))

        if rng.random() < config.spurious_strength:
            label = int(spurious_token % config.n_classes)
        if rng.random() < config.noise_rate:
            label = int(rng.integers(0, config.n_classes))

        seq = sample_sequence(rng, vocab, config.seq_len, causal_token, spurious_token, "source")
        sequences.append(seq)
        labels.append(label)
        causal_used.append(causal_token)
        spurious_used.append(spurious_token)

    return AdaptationDataset(
        input_ids=pad_batch(sequences, pad_id=vocab.pad_id),
        labels=np.array(labels, dtype=np.int64),
        domain="source",
        ground_truth={
            "causal_tokens": np.array(causal_used),
            "spurious_tokens": np.array(spurious_used),
        },
    )


def _generate_counterfactual(
    rng: np.random.Generator,
    vocab: TokenVocab,
    n_samples: int,
    config: CounterfactualDGPConfig,
) -> AdaptationDataset:
    """Swap spurious tokens while holding causal tokens fixed; labels follow causal rule."""
    causal_tokens = vocab.causal_tokens()
    spurious_tokens = vocab.spurious_tokens()
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    causal_used: list[int] = []
    flipped_spurious: list[int] = []

    for _ in range(n_samples):
        causal_token = int(rng.choice(causal_tokens))
        label = _label_from_causal(causal_token, vocab, config.n_classes)
        spurious_token = int(rng.choice(spurious_tokens))
        flipped = int(rng.choice([t for t in spurious_tokens if t != spurious_token]))

        seq = sample_sequence(rng, vocab, config.seq_len, causal_token, flipped, "target")
        sequences.append(seq)
        labels.append(label)
        causal_used.append(causal_token)
        flipped_spurious.append(flipped)

    return AdaptationDataset(
        input_ids=pad_batch(sequences, pad_id=vocab.pad_id),
        labels=np.array(labels, dtype=np.int64),
        domain="counterfactual",
        ground_truth={
            "causal_tokens": np.array(causal_used),
            "spurious_tokens": np.array(flipped_spurious),
            "intervention": "spurious_token_swap",
        },
    )


def generate_counterfactual_data(
    config: CounterfactualDGPConfig | None = None,
) -> CounterfactualBundle:
    """Generate train/val/test plus counterfactual test set with known causal labels."""
    config = config or CounterfactualDGPConfig()
    rng = set_seed(config.seed)
    vocab = TokenVocab.default(config.vocab_size)

    train = _generate_observed(rng, vocab, config.n_train, config)
    val = _generate_observed(rng, vocab, config.n_val, config)
    test = _generate_observed(rng, vocab, config.n_test, config)
    cf_test = _generate_counterfactual(rng, vocab, config.n_counterfactual, config)

    oracle = np.array(
        [_label_from_causal(int(t), vocab, config.n_classes) for t in cf_test.ground_truth["causal_tokens"]]
    )

    return CounterfactualBundle(
        train=train,
        val=val,
        test=test,
        counterfactual_test=cf_test,
        vocab_size=vocab.vocab_size,
        ground_truth={
            "oracle_counterfactual_labels": oracle,
            "causal_rule": "label = causal_token mod n_classes",
            "spurious_strength_train": config.spurious_strength,
        },
    )
