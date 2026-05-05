#!/usr/bin/env python3
"""
Run H text-mask helpers: legacy ``<ICD>...</ICD>`` span detection for separating
report-token CE from ICD-token CE.
"""
from __future__ import annotations

import torch


def mask_all_code_spans(labels_row: torch.Tensor, ids_1d: torch.Tensor, tokenizer) -> torch.Tensor:
    """Set labels to -100 for every legacy ``<ICD>...</ICD>`` span (inclusive of tags)."""
    out = labels_row.clone()
    ids = ids_1d.tolist()
    start_id = int(tokenizer.convert_tokens_to_ids("<ICD>"))
    end_id = int(tokenizer.convert_tokens_to_ids("</ICD>"))
    s = None
    for i, tok in enumerate(ids):
        if tok == start_id:
            s = i
        elif tok == end_id and s is not None:
            out[s : i + 1] = -100
            s = None
    return out


def mask_report_span(labels_row: torch.Tensor, ids_1d: torch.Tensor, tokenizer) -> torch.Tensor:
    """Set labels to -100 inside ``<REPORT>...</REPORT>`` (inclusive of tags)."""
    out = labels_row.clone()
    ids = ids_1d.tolist()
    rs = int(tokenizer.convert_tokens_to_ids("<REPORT>"))
    re = int(tokenizer.convert_tokens_to_ids("</REPORT>"))
    s = None
    for i, tok in enumerate(ids):
        if tok == rs:
            s = i
        elif tok == re and s is not None:
            out[s : i + 1] = -100
            s = None
    return out


def collect_supervised_code_mask(
    labels_1d: torch.Tensor, ids_1d: torch.Tensor, tokenizer
) -> torch.Tensor:
    """Boolean mask: supervised positions inside the legacy ``<ICD>...</ICD>`` span."""
    mask = torch.zeros_like(labels_1d, dtype=torch.bool)
    ids = ids_1d.tolist()
    start_id = int(tokenizer.convert_tokens_to_ids("<ICD>"))
    end_id = int(tokenizer.convert_tokens_to_ids("</ICD>"))
    s = None
    for i, tok in enumerate(ids):
        if tok == start_id:
            s = i
        elif tok == end_id and s is not None:
            for j in range(s + 1, i + 1):
                if labels_1d[j] != -100:
                    mask[j] = True
            s = None
    return mask


def collect_supervised_report_mask(
    labels_1d: torch.Tensor, ids_1d: torch.Tensor, tokenizer
) -> torch.Tensor:
    """Boolean mask: supervised positions inside the ``<REPORT>...</REPORT>`` span."""
    mask = torch.zeros_like(labels_1d, dtype=torch.bool)
    ids = ids_1d.tolist()
    rs = int(tokenizer.convert_tokens_to_ids("<REPORT>"))
    re = int(tokenizer.convert_tokens_to_ids("</REPORT>"))
    s = None
    for i, tok in enumerate(ids):
        if tok == rs:
            s = i
        elif tok == re and s is not None:
            for j in range(s + 1, i + 1):
                if labels_1d[j] != -100:
                    mask[j] = True
            s = None
    return mask
