#!/usr/bin/env python3
"""Stage-1 cross-visit 4-modality code forecasting data pipeline."""

from __future__ import annotations

import logging
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# Stage 1 trains ICD-9 diagnosis-code embeddings only. The original framework
# was multicode-capable (diag / drug / proc / lab); for this release the
# trainable modality is restricted to diag.
MODALITY_ORDER = ("diag",)
MODALITY_PREFIX = {"diag": "DIAG"}
MODALITY_CAPS   = {"diag": 16}
NO_RECORD_TOKEN = "<NO_RECORD>"

_LOG = logging.getLogger(__name__)
_OOV_WARN_CAP = 48


def _convert_code_token_to_id(tokenizer, tok: str, unk_id: int):
    """
    Map a code string (e.g. '<DIAG_410>') to an int id.

    Qwen2 fast tokenizers often return None for OOV when unk_token_id is None;
    calling int(None) crashes — handle explicitly.
    """
    raw = tokenizer.convert_tokens_to_ids(tok)
    if raw is None:
        return None
    tid = int(raw)
    if unk_id >= 0 and tid == unk_id:
        return None
    return tid


@dataclass(frozen=True)
class Visit:
    subject_id: int
    hadm_id: int
    admittime: str
    diag_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class WindowSample:
    subject_id: int
    timeline_idx: int
    target_idx: int
    ctx_start_idx: int


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _make_tokens(values: Sequence[object], prefix: str) -> Tuple[str, ...]:
    uniq = sorted({str(v).strip() for v in values if str(v).strip()})
    return tuple(f"<{prefix}_{code}>" for code in uniq)


def _payload_to_timelines(payload: Mapping[int, Mapping[str, object]]) -> Dict[int, List[Visit]]:
    timelines: Dict[int, List[Visit]] = {}
    for sid_raw, patient in payload.items():
        sid = int(sid_raw)
        raw_visits = list(patient.get("visits", []))
        visits: List[Visit] = []
        for raw_visit in sorted(raw_visits, key=lambda v: (str(v.get("admittime") or ""), int(v.get("hadm_id") or 0))):
            visits.append(
                Visit(
                    subject_id=sid,
                    hadm_id=int(raw_visit.get("hadm_id") or 0),
                    admittime=str(raw_visit.get("admittime") or ""),
                    diag_tokens=_make_tokens(raw_visit.get("diag_codes", []), MODALITY_PREFIX["diag"]),
                )
            )
        if len(visits) >= 2:
            timelines[sid] = visits
    return timelines


def load_patient_timelines_from_shared(shared_dir: Path, split_name: str) -> Dict[int, List[Visit]]:
    split_path = Path(shared_dir) / f"{split_name}_subjects.pkl"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing shared split pickle: {split_path}")
    payload = _load_pickle(split_path)
    return _payload_to_timelines(payload)


def load_subject_splits_from_shared(shared_dir: Path) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for split_name in ("train", "val", "test"):
        payload = _load_pickle(Path(shared_dir) / f"{split_name}_subjects.pkl")
        out[split_name] = sorted(int(sid) for sid in payload.keys())
    return out


def build_code_token_inventory(train_timelines: Dict[int, List[Visit]]) -> Dict[str, List[str]]:
    vocab = {mod: set() for mod in MODALITY_ORDER}
    for visits in train_timelines.values():
        for visit in visits:
            for mod in MODALITY_ORDER:
                vocab[mod].update(getattr(visit, f"{mod}_tokens"))
    return {mod: sorted(vocab[mod]) for mod in MODALITY_ORDER}


class MultiCodeVisitWindowDataset(Dataset):
    def __init__(
        self,
        timelines: Dict[int, List[Visit]],
        tokenizer,
        split_name: str,
        k_max: int = 4,
        max_seq_len: int = 512,
        modality_max_codes: Mapping[str, int] | None = None,
        shuffle_within_visit: bool = True,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.split_name = str(split_name)
        self.k_max = int(k_max)
        self.max_seq_len = int(max_seq_len)
        self.modality_max_codes = {mod: int((modality_max_codes or MODALITY_CAPS).get(mod, MODALITY_CAPS[mod])) for mod in MODALITY_ORDER}
        self.shuffle_within_visit = bool(shuffle_within_visit)
        self.rng = random.Random(int(seed))

        self.timeline_subject_ids = sorted(int(sid) for sid in timelines.keys())
        self.timelines = [timelines[sid] for sid in self.timeline_subject_ids]

        def _require_struct_id(s: str) -> int:
            raw = tokenizer.convert_tokens_to_ids(s)
            if raw is None:
                raise ValueError(f"Tokenizer missing required structural token {s!r} (convert_tokens_to_ids returned None)")
            return int(raw)

        self.visit_start_id = _require_struct_id("<VISIT_START>")
        self.visit_end_id = _require_struct_id("<VISIT_END>")
        self.no_record_id = _require_struct_id(NO_RECORD_TOKEN)
        pad_tok = tokenizer.pad_token_id
        if pad_tok is None:
            pad_tok = getattr(tokenizer, "eos_token_id", None)
        if pad_tok is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; set tokenizer.pad_token.")
        self.pad_id = int(pad_tok)
        unk_tok = getattr(tokenizer, "unk_token_id", None)
        self.unk_id = -1 if unk_tok is None else int(unk_tok)
        self.modality_start_ids = {mod: _require_struct_id(f"<{MODALITY_PREFIX[mod]}>") for mod in MODALITY_ORDER}
        self.modality_end_ids = {mod: _require_struct_id(f"</{MODALITY_PREFIX[mod]}>") for mod in MODALITY_ORDER}

        self._oov_warns = 0

        self.windows: List[WindowSample] = []
        self.windows_per_patient: Dict[int, int] = defaultdict(int)
        for tidx, visits in enumerate(self.timelines):
            sid = self.timeline_subject_ids[tidx]
            for t in range(1, len(visits)):
                ctx_len = min(self.k_max, t)
                self.windows.append(
                    WindowSample(
                        subject_id=sid,
                        timeline_idx=tidx,
                        target_idx=t,
                        ctx_start_idx=t - ctx_len,
                    )
                )
                self.windows_per_patient[sid] += 1

    def __len__(self) -> int:
        return len(self.windows)

    def _encode_modality(
        self,
        visit: Visit,
        modality: str,
        *,
        is_target: bool,
        shuffle_codes: bool,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        tokens = list(getattr(visit, f"{modality}_tokens")[: self.modality_max_codes[modality]])
        if shuffle_codes and len(tokens) > 1:
            self.rng.shuffle(tokens)

        code_ids: List[int] = []
        for tok in tokens:
            tid = _convert_code_token_to_id(self.tokenizer, tok, self.unk_id)
            if tid is None:
                if self._oov_warns < _OOV_WARN_CAP:
                    _LOG.warning(
                        "Skipping OOV / unmapped code token %r (modality=%s subject_id=%s hadm_id=%s)",
                        tok,
                        modality,
                        visit.subject_id,
                        visit.hadm_id,
                    )
                    self._oov_warns += 1
                continue
            code_ids.append(tid)

        gold_code_ids = sorted(set(code_ids))
        if not code_ids:
            code_ids = [self.no_record_id]

        ids = [self.modality_start_ids[modality]] + code_ids + [self.modality_end_ids[modality]]
        labels_full = [-100] * len(ids)
        labels_mod = [-100] * len(ids)
        if is_target:
            for i in range(1, len(ids)):
                labels_full[i] = ids[i]
                labels_mod[i] = ids[i]
        return ids, labels_full, labels_mod, gold_code_ids

    def _serialize(self, visits: Sequence[Visit], target_pos: int, do_shuffle: bool):
        input_ids: List[int] = []
        labels_full: List[int] = []
        labels_by_modality = {mod: [] for mod in MODALITY_ORDER}
        target_code_ids = {mod: [] for mod in MODALITY_ORDER}

        for i, visit in enumerate(visits):
            is_target = i == target_pos
            visit_ids = [self.visit_start_id]
            visit_labels_full = [-100]
            visit_labels_by_modality = {mod: [-100] for mod in MODALITY_ORDER}

            for mod in MODALITY_ORDER:
                ids_m, full_m, labels_m, gold_ids = self._encode_modality(
                    visit,
                    mod,
                    is_target=is_target,
                    shuffle_codes=do_shuffle,
                )
                visit_ids.extend(ids_m)
                visit_labels_full.extend(full_m)
                for mod_key in MODALITY_ORDER:
                    if mod_key == mod:
                        visit_labels_by_modality[mod_key].extend(labels_m)
                    else:
                        visit_labels_by_modality[mod_key].extend([-100] * len(ids_m))
                if is_target:
                    target_code_ids[mod] = gold_ids

            visit_ids.append(self.visit_end_id)
            visit_labels_full.append(-100)
            for mod in MODALITY_ORDER:
                visit_labels_by_modality[mod].append(-100)

            input_ids.extend(visit_ids)
            labels_full.extend(visit_labels_full)
            for mod in MODALITY_ORDER:
                labels_by_modality[mod].extend(visit_labels_by_modality[mod])

        return input_ids, labels_full, labels_by_modality, target_code_ids

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        timeline = self.timelines[w.timeline_idx]
        indices = list(range(w.ctx_start_idx, w.target_idx + 1))
        do_shuffle = self.shuffle_within_visit and (self.split_name == "train")

        while True:
            local_target_pos = len(indices) - 1
            visits = [timeline[i] for i in indices]
            input_ids, labels_full, labels_by_modality, target_code_ids = self._serialize(
                visits,
                target_pos=local_target_pos,
                do_shuffle=do_shuffle,
            )
            if len(input_ids) <= self.max_seq_len:
                break
            if len(indices) > 2:
                indices = indices[1:]
                continue
            cut = len(input_ids) - self.max_seq_len
            input_ids = input_ids[cut:]
            labels_full = labels_full[cut:]
            for mod in MODALITY_ORDER:
                labels_by_modality[mod] = labels_by_modality[mod][cut:]
            break

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones((len(input_ids),), dtype=torch.long),
            "labels_full": torch.tensor(labels_full, dtype=torch.long),
            "subject_id": torch.tensor(w.subject_id, dtype=torch.long),
        }
        for mod in MODALITY_ORDER:
            out[f"labels_{mod}"] = torch.tensor(labels_by_modality[mod], dtype=torch.long)
            out[f"target_code_ids_{mod}"] = torch.tensor(sorted(set(target_code_ids[mod])), dtype=torch.long)
        return out

    def sample_weights(self, alpha: float = 0.5) -> List[float]:
        a = float(alpha)
        out: List[float] = []
        for w in self.windows:
            n_w = max(1, int(self.windows_per_patient[w.subject_id]))
            out.append(float(n_w ** (a - 1.0)))
        return out


def collate_windows(batch: List[Dict[str, torch.Tensor]], pad_token_id: int):
    max_len = max(int(x["input_ids"].numel()) for x in batch)
    bsz = len(batch)

    out = {
        "input_ids": torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long),
        "attention_mask": torch.zeros((bsz, max_len), dtype=torch.long),
        "labels_full": torch.full((bsz, max_len), -100, dtype=torch.long),
        "subject_id": torch.zeros((bsz,), dtype=torch.long),
    }
    for mod in MODALITY_ORDER:
        out[f"labels_{mod}"] = torch.full((bsz, max_len), -100, dtype=torch.long)
        max_target_codes = max(int(x[f"target_code_ids_{mod}"].numel()) for x in batch)
        out[f"target_code_ids_{mod}"] = torch.full((bsz, max_target_codes), -1, dtype=torch.long)

    for i, ex in enumerate(batch):
        n = int(ex["input_ids"].numel())
        out["input_ids"][i, :n] = ex["input_ids"]
        out["attention_mask"][i, :n] = ex["attention_mask"]
        out["labels_full"][i, :n] = ex["labels_full"]
        out["subject_id"][i] = ex["subject_id"]
        for mod in MODALITY_ORDER:
            out[f"labels_{mod}"][i, :n] = ex[f"labels_{mod}"]
            m = int(ex[f"target_code_ids_{mod}"].numel())
            if m > 0:
                out[f"target_code_ids_{mod}"][i, :m] = ex[f"target_code_ids_{mod}"]

    return out

