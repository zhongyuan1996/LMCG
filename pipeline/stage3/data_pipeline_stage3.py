#!/usr/bin/env python3
"""
Stage 3 (Run H) data pipeline — sequential clinical chain generation.

Target chain: [image?] -> [report?] -> [legacy <ICD>...</ICD> diagnosis codes?]

Only the legacy diagnosis-only ICD9 tokenizer path is supported in Run H.
Context visits use the same legacy serialization (no per-modality blocks).
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage2"))
from data_pipeline import (  # noqa: E402
    VisitRecord,
    _visit_has_any_clinical_code,
)


@dataclass(frozen=True)
class Stage3Window:
    subject_id: int
    timeline_idx: int
    target_idx: int
    ctx_start_idx: int
    has_image: bool
    has_report: bool
    has_icd: bool


class Stage3ChainWindowDataset(Dataset):
    """
    One sample per (patient, target-visit) with at least one of
    {PA image, report, any clinical code}.

    Context visits and the target visit serialize their diagnosis codes inside a
    single legacy ``<ICD>...</ICD>`` block (Run H uses the diagnosis-only ICD9
    tokenizer; ``<DIAG_*>`` tokens from the data pipeline are remapped to
    ``<ICD9_*>``).
    """

    def __init__(
        self,
        timelines: Dict[int, List[VisitRecord]],
        tokenizer,
        split_name: str,
        k_max: int = 4,
        max_seq_len: int = 2560,
        keep_last_n_ctx_images: int = 1,
        report_max_tokens: int = 192,
        num_image_tokens: int = 1024,
        add_time_embeds: bool = True,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.split_name = str(split_name)
        self.k_max = int(k_max)
        self.max_seq_len = int(max_seq_len)
        self.keep_last_n_ctx_images = int(max(0, keep_last_n_ctx_images))
        self.report_max_tokens = int(report_max_tokens)
        self.add_time_embeds = bool(add_time_embeds)
        self.base_num_image_tokens = int(num_image_tokens)
        self.num_image_tokens = int(num_image_tokens) + (1 if self.add_time_embeds else 0)
        self.rng = random.Random(int(seed))

        self.visit_start_tok = "<VISIT_START>"
        self.visit_end_tok = "<VISIT_END>"
        self.report_start_tok = "<REPORT>"
        self.report_end_tok = "</REPORT>"
        self.icd_start_tok = "<ICD>"
        self.icd_end_tok = "</ICD>"

        unk_tok = getattr(tokenizer, "unk_token_id", None)
        self.unk_id = -1 if unk_tok is None else int(unk_tok)

        self.timeline_subject_ids = sorted(int(sid) for sid in timelines.keys())
        self.timelines = [timelines[sid] for sid in self.timeline_subject_ids]

        self.pad_id = int(tokenizer.pad_token_id)
        self.boi_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        self.eoi_id = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        self.img_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))

        self.windows: List[Stage3Window] = []
        self.windows_per_patient: Dict[int, int] = defaultdict(int)
        self.modality_counts = Counter()

        for tidx, visits in enumerate(self.timelines):
            sid = self.timeline_subject_ids[tidx]
            for t in range(1, len(visits)):
                ctx_len = min(self.k_max, t)
                v_t = visits[t]
                has_img = bool(v_t.pa_image_path)
                has_report = bool(v_t.report_text)
                has_codes = _visit_has_any_clinical_code(v_t)
                ctx_lo = t - ctx_len
                if not (has_img or has_report or has_codes):
                    continue

                w = Stage3Window(
                    subject_id=sid,
                    timeline_idx=tidx,
                    target_idx=t,
                    ctx_start_idx=ctx_lo,
                    has_image=has_img,
                    has_report=has_report,
                    has_icd=has_codes,
                )
                self.windows.append(w)
                self.windows_per_patient[sid] += 1
                if has_img:
                    self.modality_counts["image"] += 1
                if has_report:
                    self.modality_counts["report"] += 1
                if has_codes:
                    self.modality_counts["icd"] += 1

    def __len__(self) -> int:
        return len(self.windows)

    def window_counts(self) -> Dict[str, int]:
        return {
            "windows": len(self.windows),
            **{k: int(v) for k, v in self.modality_counts.items()},
        }

    def sample_weights(
        self,
        patient_balance_alpha: float = 0.5,
        modality_image_weight: float = 5.0,
        report_only: bool = False,
    ) -> List[float]:
        alpha = float(patient_balance_alpha)
        img_w = float(modality_image_weight)
        weights: List[float] = []
        for w in self.windows:
            if report_only and not w.has_report:
                weights.append(0.0)
                continue
            n_w = max(1, int(self.windows_per_patient[w.subject_id]))
            base = float(n_w ** (alpha - 1.0))
            if w.has_image:
                base *= img_w
            weights.append(base)
        return weights

    def _tokenize_report(self, txt: str) -> List[int]:
        if not txt:
            return []
        ids = self.tokenizer.encode(
            txt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.report_max_tokens,
        )
        return [int(x) for x in ids]

    def _legacy_flat_code_tokens(self, v: VisitRecord) -> List[str]:
        """Diagnosis-only ICD9 token stream.

        The data pipeline stores diagnosis codes as ``<DIAG_XXX>``; the legacy
        ICD9 tokenizer expects ``<ICD9_XXX>``. Drug/proc/lab codes have no
        ICD9 equivalent and are omitted (Run H is diagnosis-only).
        """
        out: List[str] = []
        for t in v.diag_tokens:
            if t.startswith("<DIAG_"):
                out.append("<ICD9_" + t[6:])
            else:
                out.append(t)
        return out

    def _encode_legacy_icd_block(self, v: VisitRecord) -> List[int]:
        toks = self._legacy_flat_code_tokens(v)
        if not toks:
            return []
        out = [int(self.tokenizer.convert_tokens_to_ids(self.icd_start_tok))]
        for tok in toks:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is None:
                continue
            out.append(int(tid))
        out.append(int(self.tokenizer.convert_tokens_to_ids(self.icd_end_tok)))
        return out

    def _encode_report_block(self, report_text: str) -> List[int]:
        out = [int(self.tokenizer.convert_tokens_to_ids(self.report_start_tok))]
        out.extend(self._tokenize_report(report_text))
        out.append(int(self.tokenizer.convert_tokens_to_ids(self.report_end_tok)))
        return out

    def _encode_image_block(self) -> Tuple[List[int], Tuple[int, int]]:
        ids = [self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id]
        return ids, (1, self.num_image_tokens)

    def _ctx_image_keep_indices(self, visits_ctx: Sequence[VisitRecord]) -> set:
        if self.keep_last_n_ctx_images <= 0:
            return set()
        idx_has_img = [i for i, v in enumerate(visits_ctx) if bool(v.pa_image_path)]
        return set(idx_has_img[-self.keep_last_n_ctx_images :])

    def _serialize(self, window: Stage3Window) -> Dict:
        timeline = self.timelines[window.timeline_idx]
        idxs = list(range(window.ctx_start_idx, window.target_idx + 1))
        visits = [timeline[i] for i in idxs]
        target_local = len(visits) - 1
        ctx_visits = visits[:-1]
        tgt = visits[-1]

        keep_ctx_img = self._ctx_image_keep_indices(ctx_visits)

        visit_start_id = int(self.tokenizer.convert_tokens_to_ids(self.visit_start_tok))
        visit_end_id = int(self.tokenizer.convert_tokens_to_ids(self.visit_end_tok))

        input_ids: List[int] = []
        labels: List[int] = []
        modality_positions: List[Tuple[int, int]] = []
        image_paths_for_slots: List[str] = []
        image_supervise_mask: List[int] = []

        # Target-visit clinical-code span in input_ids (inclusive indices of the
        # opening tag through the closing tag), used by eval to locate the ICD block.
        target_code_seq_start: int = -1
        target_code_seq_end: int = -1

        for i, v in enumerate(visits):
            is_target = i == target_local
            input_ids.append(visit_start_id)
            labels.append(-100)

            if not is_target:
                b = self._encode_legacy_icd_block(v)
                if b:
                    input_ids.extend(b)
                    labels.extend([-100] * len(b))

                if bool(v.report_text):
                    b = self._encode_report_block(v.report_text)
                    input_ids.extend(b)
                    labels.extend([-100] * len(b))

                if (i in keep_ctx_img) and bool(v.pa_image_path):
                    b, (img_off, img_len) = self._encode_image_block()
                    st = len(input_ids)
                    input_ids.extend(b)
                    labels.extend([-100] * len(b))
                    modality_positions.append((st + img_off, img_len))
                    image_paths_for_slots.append(v.pa_image_path)
                    image_supervise_mask.append(0)

            else:
                if window.has_image:
                    b, (img_off, img_len) = self._encode_image_block()
                    st = len(input_ids)
                    input_ids.extend(b)
                    labels.extend([-100] * len(b))
                    modality_positions.append((st + img_off, img_len))
                    image_paths_for_slots.append(tgt.pa_image_path)
                    image_supervise_mask.append(1)

                if window.has_report:
                    b = self._encode_report_block(tgt.report_text)
                    st = len(input_ids)
                    input_ids.extend(b)
                    labels.extend([-100] * len(b))
                    for p in range(st + 1, st + len(b)):
                        labels[p] = input_ids[p]

                if window.has_icd:
                    b = self._encode_legacy_icd_block(tgt)
                    if b:
                        st = len(input_ids)
                        target_code_seq_start = st
                        input_ids.extend(b)
                        labels.extend([-100] * len(b))
                        for p in range(st + 1, st + len(b)):
                            labels[p] = input_ids[p]
                        target_code_seq_end = len(input_ids) - 1

            input_ids.append(visit_end_id)
            labels.append(-100)

        assert len(input_ids) == len(labels)

        if len(input_ids) > self.max_seq_len:
            cut = len(input_ids) - self.max_seq_len
            input_ids = input_ids[cut:]
            labels = labels[cut:]
            new_mp, new_paths, new_sup = [], [], []
            for (off, ln), pth, sup in zip(modality_positions, image_paths_for_slots, image_supervise_mask):
                off2 = off - cut
                if off2 < 0 or (off2 + ln) > len(input_ids):
                    continue
                new_mp.append((off2, ln))
                new_paths.append(pth)
                new_sup.append(sup)
            modality_positions = new_mp
            image_paths_for_slots = new_paths
            image_supervise_mask = new_sup

            if target_code_seq_start >= 0:
                lo = target_code_seq_start - cut
                hi = target_code_seq_end - cut
                nlen = len(input_ids)
                if lo < 0 or hi < 0 or lo > hi or hi >= nlen:
                    target_code_seq_start = -1
                    target_code_seq_end = -1
                else:
                    target_code_seq_start = lo
                    target_code_seq_end = hi

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "modality_positions": modality_positions,
            "image_paths_for_slots": image_paths_for_slots,
            "image_supervise_mask": torch.tensor(image_supervise_mask, dtype=torch.long),
            "has_target_image": int(window.has_image),
            "has_target_report": int(window.has_report),
            "has_target_icd": int(window.has_icd),
            "target_code_seq_start": int(target_code_seq_start),
            "target_code_seq_end": int(target_code_seq_end),
            "subject_id": int(window.subject_id),
        }

    def __getitem__(self, idx: int):
        return self._serialize(self.windows[idx])


def collate_stage3_windows(batch: List[Dict], pad_token_id: int) -> Dict:
    max_len = max(int(x["input_ids"].numel()) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    subject_ids = torch.zeros(bsz, dtype=torch.long)
    has_image = torch.zeros(bsz, dtype=torch.long)
    has_report = torch.zeros(bsz, dtype=torch.long)
    has_icd = torch.zeros(bsz, dtype=torch.long)
    target_code_seq_start = torch.full((bsz,), -1, dtype=torch.long)
    target_code_seq_end = torch.full((bsz,), -1, dtype=torch.long)

    modality_positions: List[list] = []
    image_paths_for_slots: List[list] = []
    image_supervise_row_tensors: List[torch.Tensor] = []

    max_img_slots = 0
    for ex in batch:
        max_img_slots = max(max_img_slots, len(ex["modality_positions"]))

    for i, ex in enumerate(batch):
        n = int(ex["input_ids"].numel())
        input_ids[i, :n] = ex["input_ids"]
        attention_mask[i, :n] = ex["attention_mask"]
        labels[i, :n] = ex["labels"]
        subject_ids[i] = int(ex["subject_id"])
        has_image[i] = int(ex["has_target_image"])
        has_report[i] = int(ex["has_target_report"])
        has_icd[i] = int(ex["has_target_icd"])
        target_code_seq_start[i] = int(ex.get("target_code_seq_start", -1))
        target_code_seq_end[i] = int(ex.get("target_code_seq_end", -1))
        mp = list(ex["modality_positions"])
        pp = list(ex["image_paths_for_slots"])
        sm = ex["image_supervise_mask"]
        sm_list = sm.tolist() if isinstance(sm, torch.Tensor) else list(sm)
        while len(mp) < max_img_slots:
            mp.append((0, 0))
            pp.append("")
            sm_list.append(0)
        modality_positions.append(mp)
        image_paths_for_slots.append(pp)
        image_supervise_row_tensors.append(torch.tensor(sm_list, dtype=torch.long))

    image_supervise_masks_stacked = (
        torch.stack(image_supervise_row_tensors, dim=0)
        if image_supervise_row_tensors
        else torch.zeros((bsz, 0), dtype=torch.long)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "modality_positions": modality_positions,
        "image_paths_for_slots": image_paths_for_slots,
        "image_supervise_masks": image_supervise_masks_stacked,
        "has_target_image": has_image,
        "has_target_report": has_report,
        "has_target_icd": has_icd,
        "target_code_seq_start": target_code_seq_start,
        "target_code_seq_end": target_code_seq_end,
        "subject_id": subject_ids,
    }
