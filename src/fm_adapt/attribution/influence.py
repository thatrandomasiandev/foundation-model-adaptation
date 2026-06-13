"""Training data attribution via gradient-based influence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fm_adapt.utils.device import get_device


@dataclass
class AttributionResult:
    sample_scores: np.ndarray
    top_indices: np.ndarray
    method: str


def trac_in_scores(
    model: nn.Module,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    query_ids: np.ndarray,
    query_labels: np.ndarray,
    checkpoints: int = 3,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> AttributionResult:
    """
    Approximate TracIn influence: sum of gradient dot-products at saved checkpoints.

    Higher score => training example more influential for the query prediction.
    """
    torch.manual_seed(seed)
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_x = torch.tensor(train_ids, dtype=torch.long)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    saved_grads: list[list[torch.Tensor]] = []
    steps_per_ckpt = max(1, (epochs * len(loader)) // checkpoints)
    step = 0

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            saved_grads.append([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()])
            optimizer.step()
            step += 1
            if len(saved_grads) >= checkpoints:
                break
        if len(saved_grads) >= checkpoints:
            break

    model.eval()
    query_x = torch.tensor(query_ids, dtype=torch.long).to(device)
    query_y = torch.tensor(query_labels, dtype=torch.long).to(device)

    query_grads: list[torch.Tensor] = []
    for qx, qy in zip(query_x, query_y, strict=True):
        model.zero_grad()
        loss = criterion(model(qx.unsqueeze(0)), qy.unsqueeze(0))
        loss.backward()
        query_grads.append([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()])

    scores = np.zeros(len(train_ids))
    for i in range(len(train_ids)):
        tx = train_x[i : i + 1].to(device)
        ty = train_y[i : i + 1].to(device)
        for q_idx, q_grad in enumerate(query_grads):
            model.zero_grad()
            loss = criterion(model(tx), ty)
            loss.backward()
            train_grad = [p.grad.detach() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
            influence = 0.0
            for ckpt_grad, t_grad in zip(saved_grads, train_grad, strict=True):
                for cg, tg in zip(ckpt_grad, t_grad, strict=True):
                    influence += float((cg.flatten() @ tg.flatten()).item())
            scores[i] += influence / max(len(saved_grads), 1)

    top_indices = np.argsort(-np.abs(scores))[: min(10, len(scores))]
    return AttributionResult(sample_scores=scores, top_indices=top_indices, method="trac_in")
