"""Dataset protocol for adaptation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AdaptationDataset:
    """Token sequences with labels and optional ground-truth metadata."""

    input_ids: np.ndarray
    labels: np.ndarray
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1] if self.input_ids.ndim > 1 else 0


@dataclass
class DomainShiftBundle:
    """Source/target splits for domain adaptation experiments."""

    source_train: AdaptationDataset
    source_val: AdaptationDataset
    target_train: AdaptationDataset
    target_val: AdaptationDataset
    target_test: AdaptationDataset
    vocab_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualBundle:
    """Dataset with known causal vs spurious feature structure."""

    train: AdaptationDataset
    val: AdaptationDataset
    test: AdaptationDataset
    counterfactual_test: AdaptationDataset
    vocab_size: int
    ground_truth: dict[str, Any] = field(default_factory=dict)
