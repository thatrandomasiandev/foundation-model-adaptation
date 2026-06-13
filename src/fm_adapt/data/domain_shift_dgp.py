"""Domain shift data generator with known label function."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fm_adapt.data.base import AdaptationDataset, DomainShiftBundle
from fm_adapt.data.tokenizer import TokenVocab, pad_batch, sample_sequence
from fm_adapt.utils.seed import set_seed


@dataclass
class DomainShiftDGPConfig:
    n_source_train: int = 800
    n_source_val: int = 200
    n_target_train: int = 200
    n_target_val: int = 100
    n_target_test: int = 500
    seq_len: int = 16
    vocab_size: int = 128
    shift_strength: float = 1.0
    spurious_strength: float = 0.8
    noise_rate: float = 0.05
    n_classes: int = 2
    seed: int = 42


def _label_from_causal(causal_token: int, vocab: TokenVocab, n_classes: int) -> int:
    causal_idx = causal_token - vocab.causal_start
    return int(causal_idx % n_classes)


def _generate_split(
    rng: np.random.Generator,
    vocab: TokenVocab,
    n_samples: int,
    domain: str,
    config: DomainShiftDGPConfig,
    use_spurious: bool,
) -> AdaptationDataset:
    causal_tokens = vocab.causal_tokens()
    spurious_tokens = vocab.spurious_tokens()
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    causal_used: list[int] = []
    spurious_used: list[int | None] = []

    for _ in range(n_samples):
        causal_token = int(rng.choice(causal_tokens))
        label = _label_from_causal(causal_token, vocab, config.n_classes)

        spurious_token: int | None = None
        if use_spurious and rng.random() < config.spurious_strength:
            spurious_token = int(rng.choice(spurious_tokens))
            if domain == "source":
                spurious_label = int(spurious_token % config.n_classes)
                if rng.random() < 0.7:
                    label = spurious_label

        if rng.random() < config.noise_rate:
            label = int(rng.integers(0, config.n_classes))

        seq = sample_sequence(rng, vocab, config.seq_len, causal_token, spurious_token, domain)
        sequences.append(seq)
        labels.append(label)
        causal_used.append(causal_token)
        spurious_used.append(spurious_token)

    input_ids = pad_batch(sequences, pad_id=vocab.pad_id)
    labels_arr = np.array(labels, dtype=np.int64)
    return AdaptationDataset(
        input_ids=input_ids,
        labels=labels_arr,
        domain=domain,
        metadata={"shift_strength": config.shift_strength},
        ground_truth={
            "causal_tokens": np.array(causal_used),
            "spurious_tokens": np.array([s if s is not None else -1 for s in spurious_used]),
            "label_function": "causal_token mod n_classes",
        },
    )


def generate_domain_shift_data(config: DomainShiftDGPConfig | None = None) -> DomainShiftBundle:
    """Generate source/target splits with covariate and spurious correlation shift."""
    config = config or DomainShiftDGPConfig()
    rng = set_seed(config.seed)
    vocab = TokenVocab.default(config.vocab_size)

    source_train = _generate_split(rng, vocab, config.n_source_train, "source", config, True)
    source_val = _generate_split(rng, vocab, config.n_source_val, "source", config, True)
    target_train = _generate_split(rng, vocab, config.n_target_train, "target", config, False)
    target_val = _generate_split(rng, vocab, config.n_target_val, "target", config, False)
    target_test = _generate_split(rng, vocab, config.n_target_test, "target", config, False)

    return DomainShiftBundle(
        source_train=source_train,
        source_val=source_val,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        vocab_size=vocab.vocab_size,
        metadata={
            "shift_strength": config.shift_strength,
            "spurious_strength": config.spurious_strength,
            "noise_rate": config.noise_rate,
        },
        ground_truth={
            "oracle_labels_source": _oracle_labels(source_train, vocab, config.n_classes),
            "oracle_labels_target": _oracle_labels(target_test, vocab, config.n_classes),
            "label_function": "causal_token mod n_classes (no spurious shortcut)",
        },
    )


def _oracle_labels(data: AdaptationDataset, vocab: TokenVocab, n_classes: int) -> np.ndarray:
    causal = data.ground_truth["causal_tokens"]
    return np.array([_label_from_causal(int(t), vocab, n_classes) for t in causal])
