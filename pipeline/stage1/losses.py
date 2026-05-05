#!/usr/bin/env python3
"""Loss utilities for Stage-1 ICD visit forecasting."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def causal_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Standard causal LM CE with one-token shift.

    logits: [B, L, V]
    labels: [B, L], with ignore_index where no supervision.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if (shift_labels != ignore_index).sum() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def candidate_bce_loss(
    logits: torch.Tensor,
    target_codes: torch.Tensor,
    neg_per_pos: int = 8,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Candidate-based BCE over ICD class logits.

    logits: [B, C] over ICD classes
    target_codes: [B, K] with class indices in [0, C-1], -1 as padding
    """
    bsz, n_cls = logits.shape
    dev = logits.device
    losses = []
    for i in range(bsz):
        pos = target_codes[i]
        pos = pos[pos >= 0]
        if pos.numel() == 0:
            continue
        pos = torch.unique(pos)
        pos_set = set(int(x) for x in pos.tolist())

        n_neg = max(int(neg_per_pos) * int(pos.numel()), int(pos.numel()))
        # Uniform negative sampling from classes not in positives.
        # (Simple + stable baseline. Can be replaced with hard negative mining later.)
        neg = []
        while len(neg) < n_neg:
            cand = int(torch.randint(0, n_cls, (1,), device=dev).item())
            if cand not in pos_set:
                neg.append(cand)
        neg = torch.tensor(neg, device=dev, dtype=torch.long)

        cand_idx = torch.cat([pos.to(device=dev, dtype=torch.long), neg], dim=0)
        y = torch.cat(
            [
                torch.ones((pos.numel(),), device=dev, dtype=logits.dtype),
                torch.zeros((neg.numel(),), device=dev, dtype=logits.dtype),
            ],
            dim=0,
        )
        z = logits[i, cand_idx]
        # Balanced BCE: amplify positives slightly.
        w = torch.ones_like(y)
        w[: pos.numel()] = float(pos_weight)
        l = F.binary_cross_entropy_with_logits(z, y, weight=w, reduction="mean")
        losses.append(l)

    if not losses:
        return torch.zeros((), device=dev, dtype=logits.dtype)
    return torch.stack(losses).mean()


def temporal_contrastive_loss(
    z_prev: torch.Tensor,
    z_tgt: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Batch InfoNCE: pair i is (z_prev[i], z_tgt[i]).
    """
    zp = F.normalize(z_prev, dim=-1, eps=eps)
    zt = F.normalize(z_tgt, dim=-1, eps=eps)
    sim = torch.matmul(zp, zt.t()) / float(temperature)
    labels = torch.arange(sim.size(0), device=sim.device, dtype=torch.long)
    return F.cross_entropy(sim, labels)


def perplexity_from_loss(loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(loss, min=0.0, max=20.0))

