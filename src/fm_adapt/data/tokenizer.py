"""Synthetic vocabulary and tokenization for controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenVocab:
    """Fixed vocabulary with semantic token groups."""

    vocab_size: int
    causal_start: int
    causal_end: int
    spurious_start: int
    spurious_end: int
    domain_source_start: int
    domain_source_end: int
    domain_target_start: int
    domain_target_end: int
    pad_id: int = 0

    @classmethod
    def default(cls, vocab_size: int = 128) -> TokenVocab:
        """Build a vocab with disjoint causal, spurious, and domain marker ranges."""
        pad_id = 0
        causal_start = 1
        causal_count = max(8, vocab_size // 8)
        spurious_count = max(8, vocab_size // 8)
        domain_count = max(8, vocab_size // 8)
        causal_end = causal_start + causal_count
        spurious_start = causal_end
        spurious_end = spurious_start + spurious_count
        domain_source_start = spurious_end
        domain_source_end = domain_source_start + domain_count // 2
        domain_target_start = domain_source_end
        domain_target_end = domain_target_start + domain_count // 2
        actual_size = max(vocab_size, domain_target_end)
        return cls(
            vocab_size=actual_size,
            causal_start=causal_start,
            causal_end=causal_end,
            spurious_start=spurious_start,
            spurious_end=spurious_end,
            domain_source_start=domain_source_start,
            domain_source_end=domain_source_end,
            domain_target_start=domain_target_start,
            domain_target_end=domain_target_end,
            pad_id=pad_id,
        )

    def causal_tokens(self) -> np.ndarray:
        return np.arange(self.causal_start, self.causal_end)

    def spurious_tokens(self) -> np.ndarray:
        return np.arange(self.spurious_start, self.spurious_end)

    def source_markers(self) -> np.ndarray:
        return np.arange(self.domain_source_start, self.domain_source_end)

    def target_markers(self) -> np.ndarray:
        return np.arange(self.domain_target_start, self.domain_target_end)


def sample_sequence(
    rng: np.random.Generator,
    vocab: TokenVocab,
    seq_len: int,
    causal_token: int,
    spurious_token: int | None,
    domain: str,
) -> np.ndarray:
    """Sample a token sequence with one causal and optional spurious marker."""
    seq = np.full(seq_len, vocab.pad_id, dtype=np.int64)
    positions = rng.choice(seq_len, size=min(4, seq_len), replace=False)
    seq[positions[0]] = causal_token
    if spurious_token is not None and len(positions) > 1:
        seq[positions[1]] = spurious_token

    if domain == "source":
        markers = vocab.source_markers()
    else:
        markers = vocab.target_markers()
    filler = rng.choice(markers, size=seq_len // 3, replace=True)
    fill_pos = rng.choice(seq_len, size=len(filler), replace=False)
    seq[fill_pos] = filler
    return seq


def pad_batch(sequences: list[np.ndarray], pad_id: int = 0) -> np.ndarray:
    """Pad sequences to equal length."""
    max_len = max(len(s) for s in sequences)
    batch = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = seq
    return batch
