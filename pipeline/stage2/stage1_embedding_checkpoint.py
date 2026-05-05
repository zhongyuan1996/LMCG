#!/usr/bin/env python3
"""Load Stage-1 clinical token rows from pipeline/stage1 checkpoints into Show-o2 embeddings."""
from __future__ import annotations

from typing import Any, Dict

import torch


def load_stage1_embedding_rows(model: Any, stage1_ckpt: str) -> Dict[str, int]:
    """
    Restore Stage-1 learned clinical token embedding rows (multicode: DIAG/DRUG/PROC/LAB).

    Expected keys from ``pipeline/stage1/train.py`` checkpoints:
      - ``trainable_token_ids``
      - ``embedding_rows``
      - ``tokenizer_len`` (optional, recorded in manifest only)
    """
    ckpt = torch.load(stage1_ckpt, map_location="cpu")
    if "trainable_token_ids" not in ckpt or "embedding_rows" not in ckpt:
        raise KeyError(
            f"Stage-1 checkpoint missing required keys in {stage1_ckpt}: "
            "need 'trainable_token_ids' and 'embedding_rows'"
        )
    token_ids = [int(x) for x in ckpt["trainable_token_ids"]]
    rows = ckpt["embedding_rows"]
    if not isinstance(rows, torch.Tensor):
        rows = torch.tensor(rows)
    if rows.ndim != 2:
        raise ValueError(f"Invalid embedding_rows shape: {tuple(rows.shape)}")
    if rows.shape[0] != len(token_ids):
        raise ValueError(
            f"Row count mismatch in stage1 ckpt: rows={rows.shape[0]} ids={len(token_ids)}"
        )

    emb = model.showo.get_input_embeddings().weight
    vocab_size, hidden = int(emb.shape[0]), int(emb.shape[1])
    if int(rows.shape[1]) != hidden:
        raise ValueError(
            f"Hidden size mismatch: stage1={int(rows.shape[1])}, model={hidden}"
        )
    max_id = max(token_ids) if token_ids else -1
    min_id = min(token_ids) if token_ids else 0
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"Token id out of range when loading Stage-1 rows: "
            f"min_id={min_id}, max_id={max_id}, vocab_size={vocab_size}"
        )

    with torch.no_grad():
        idx = torch.tensor(token_ids, dtype=torch.long, device=emb.device)
        rows_cast = rows.to(device=emb.device, dtype=emb.dtype)
        emb[idx] = rows_cast
        diff = (emb[idx] - rows_cast).abs().max().item()
    return {
        "loaded_rows": int(len(token_ids)),
        "min_token_id": int(min_id),
        "max_token_id": int(max_id),
        "stage1_tokenizer_len": int(ckpt.get("tokenizer_len", -1)),
        "runtime_tokenizer_len": int(vocab_size),
        "max_abs_diff_after_copy": float(diff),
    }
