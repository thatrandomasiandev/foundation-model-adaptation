"""Device selection helpers."""

from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available compute device."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS lacks several transformer ops; CPU is the reliable default on Apple Silicon.
        return torch.device("cpu")
    return torch.device("cpu")
