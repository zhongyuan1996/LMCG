#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


_PHI_RE = re.compile(r"\[\*\*[\s\S]*?\*\*\]")
_UNDER_RE = re.compile(r"_{2,}")
_SECTION_HDR_RE = re.compile(r"^\s*([A-Z][A-Z /_-]{2,}?)\s*:\s*(.*)$")
_ICD9_NUMERIC_RE = re.compile(r"^(\d{3})")

# Keep in sync with pipeline/stage1/data_pipeline.py (Stage-1 multicode EHR)
MODALITY_ORDER: Tuple[str, ...] = ("diag", "drug", "proc", "lab")
MODALITY_PREFIX = {
    "diag": "DIAG",
    "drug": "DRUG",
    "proc": "PROC",
    "lab": "LAB",
}
MODALITY_CAPS = {
    "diag": 16,
    "drug": 32,
    "proc": 8,
    "lab": 64,
}
NO_RECORD_TOKEN = "<NO_RECORD>"


def tokenizer_has_stage1_multicode_vocab(tokenizer) -> bool:
    """True if tokenizer includes Stage-1 multicode tags (e.g. <DIAG>, <DRUG>, ...)."""
    tid = tokenizer.convert_tokens_to_ids("<DIAG>")
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is None:
        return isinstance(tid, int) and int(tid) >= 0
    return isinstance(tid, int) and int(tid) >= 0 and int(tid) != int(unk)


def add_stage2_special_tokens(tokenizer) -> Dict[str, int]:
    """
    Add tokens needed for Stage 2 (report + vision markers).
    When using the Stage-1 multicode tokenizer, <VISIT_*>, <DIAG>, code tokens, and
    <NO_RECORD> are already present — do not add legacy <ICD> wrappers.
    """
    add = ["<REPORT>", "</REPORT>", "<image>", "<|vid_start|>", "<|vid_end|>"]
    legacy = ["<VISIT_START>", "<VISIT_END>", "<ICD>", "</ICD>"]
    existing = list(tokenizer.additional_special_tokens or [])
    merged = list(existing)
    for t in add:
        if t not in merged:
            merged.append(t)
    if not tokenizer_has_stage1_multicode_vocab(tokenizer):
        for t in legacy:
            if t not in merged:
                merged.append(t)
    tokenizer.add_special_tokens({"additional_special_tokens": merged})
    for tok in ("<|image_pad|>", "<|video_pad|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or int(tid) < 0:
            tokenizer.add_tokens(tok)
    return {t: int(tokenizer.convert_tokens_to_ids(t)) for t in add}


def _codes_from_raw_list(raw: object, prefix: str) -> Tuple[str, ...]:
    """Build <PREFIX_code> strings; supports list[str] or list[dict] from enriched matching pickle."""
    if not raw:
        return tuple()
    codes: set[str] = set()
    if isinstance(raw, (list, tuple)):
        for x in raw:
            if isinstance(x, str) and x.strip():
                codes.add(str(x).strip().replace(" ", "_")[:128])
            elif isinstance(x, dict):
                for key in (
                    "code",
                    "ndc",
                    "drug_code",
                    "icd_code",
                    "procedure_code",
                    "itemid",
                    "lab_itemid",
                ):
                    v = x.get(key)
                    if v is not None and str(v).strip():
                        codes.add(str(v).strip().replace(" ", "_")[:128])
                        break
    return tuple(sorted(f"<{prefix}_{c}>" for c in codes))


@dataclass(frozen=True)
class VisitRecord:
    subject_id: int
    hadm_id: int
    admittime: str
    diag_tokens: Tuple[str, ...]
    drug_tokens: Tuple[str, ...]
    proc_tokens: Tuple[str, ...]
    lab_tokens: Tuple[str, ...]
    report_text: str
    pa_image_path: str


@dataclass(frozen=True)
class Stage2Window:
    subject_id: int
    timeline_idx: int
    target_idx: int
    ctx_start_idx: int
    task: str  # icd | report | image


def _parse_dt(admittime: str) -> datetime:
    s = str(admittime or "").strip()
    if not s:
        return datetime.max
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return datetime.max


def _normalize_icd9_code(raw_code: str) -> Optional[str]:
    code = str(raw_code or "").strip().upper()
    if (not code) or code.startswith("E") or code.startswith("V"):
        return None
    m = _ICD9_NUMERIC_RE.match(code)
    if m is None:
        return None
    return m.group(1)


def _clean_report_text(raw: str) -> str:
    s = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _PHI_RE.sub(" ", s)
    s = _UNDER_RE.sub(" ", s)
    s = "\n".join(" ".join(line.strip().split()) for line in s.splitlines())
    s = "\n".join([ln for ln in s.splitlines() if ln.strip()])
    return s.strip()


def _extract_findings_impression(cleaned: str) -> str:
    if not cleaned:
        return ""

    lines = cleaned.splitlines()
    sections: Dict[str, List[str]] = {"_PRE": []}
    current = "_PRE"
    for ln in lines:
        m = _SECTION_HDR_RE.match(ln)
        if m:
            hdr = m.group(1).strip().upper()
            rest = (m.group(2) or "").strip()
            if "FINDING" in hdr:
                current = "FINDINGS"
            elif "IMPRESSION" in hdr or "CONCLUSION" in hdr:
                current = "IMPRESSION"
            elif (
                "INDICATION" in hdr
                or "HISTORY" in hdr
                or "CLINICAL" in hdr
                or "TECHNIQUE" in hdr
                or "COMPARISON" in hdr
            ):
                current = "DROP"
            else:
                current = hdr
            sections.setdefault(current, [])
            if rest:
                sections[current].append(rest)
            continue
        sections.setdefault(current, []).append(ln)

    keep: List[str] = []
    if sections.get("FINDINGS"):
        keep.append("FINDINGS:")
        keep.extend([x for x in sections["FINDINGS"] if x.strip()])
    if sections.get("IMPRESSION"):
        keep.append("IMPRESSION:")
        keep.extend([x for x in sections["IMPRESSION"] if x.strip()])

    out = "\n".join(keep).strip()
    return out if out else cleaned


def trim_report_text(raw: str) -> str:
    return _extract_findings_impression(_clean_report_text(raw))


def _pick_one_pa_study(studies: Sequence[dict]) -> Tuple[int, str, str]:
    """
    Return (report_text, image_path) from earliest PA study in this visit.
    Empty strings if unavailable.
    """
    candidates: List[Tuple[int, str, str]] = []
    for st in studies:
        if not isinstance(st, dict):
            continue
        study_id = st.get("study_id")
        if not isinstance(study_id, int):
            continue
        report_text = st.get("report_text")
        if not isinstance(report_text, str) or (not report_text.strip()):
            continue
        images = st.get("images") or []
        pa_dicom = None
        for im in images:
            if not isinstance(im, dict):
                continue
            view = str(im.get("view_position") or "").upper().strip()
            dicom = str(im.get("dicom_id") or "").strip()
            if view == "PA" and dicom:
                pa_dicom = dicom
                break
        if pa_dicom is not None:
            # We keep dicom id string here. The actual absolute path is not always
            # guaranteed to be resolvable from the pickle itself.
            candidates.append((study_id, report_text, pa_dicom))

    if not candidates:
        return -1, "", ""
    candidates.sort(key=lambda x: x[0])
    study_id, report_text, pa_dicom = candidates[0]
    return int(study_id), str(report_text), str(pa_dicom)


def _pgroup(subject_id: int) -> str:
    return f"p{int(subject_id)//1_000_000:d}"


def _cxr_jpg_path(jpg_root: Path, subject_id: int, study_id: int, dicom_id: str) -> str:
    p = jpg_root / _pgroup(subject_id) / f"p{int(subject_id):d}" / f"s{int(study_id):d}" / f"{dicom_id}.jpg"
    return str(p)


def load_patient_timelines_from_matching_pkl(
    matching_pkl: Path,
    jpg_root: Optional[Path] = None,
) -> Dict[int, List[VisitRecord]]:
    with open(matching_pkl, "rb") as f:
        rows = pickle.load(f)
    if not isinstance(rows, list):
        raise TypeError(f"Expected list in {matching_pkl}, got {type(rows)}")

    by_subject: Dict[int, List[VisitRecord]] = defaultdict(list)
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        if not bool(rec.get("has_ehr", False)):
            continue
        sid = rec.get("subject_id")
        hadm = rec.get("hadm_id")
        admittime = str(rec.get("admittime") or "").strip()
        if not isinstance(sid, int) or not isinstance(hadm, int):
            continue

        diag_codes = set()
        for d in rec.get("diagnoses") or []:
            if not isinstance(d, dict):
                continue
            if int(d.get("icd_version", 0) or 0) != 9:
                continue
            code3 = _normalize_icd9_code(d.get("icd_code", ""))
            if code3 is not None:
                diag_codes.add(code3)
        diag_tokens = tuple(sorted(f"<{MODALITY_PREFIX['diag']}_{c}>" for c in diag_codes))

        # Drug / procedure / lab tokens only appear if the pickle row includes them.
        # The standard ``matching_results.pkl`` produced by ``data_preprocess`` often has *only*
        # ``diagnoses`` + CXR fields — no ``drug_codes``, ``proc_codes``, or ``lab_codes``.
        # In that case every visit gets empty tuples here (diag-only multicode training).
        # To use all four modalities, extend preprocessing to attach list fields per visit, e.g.:
        #   drug_codes | prescriptions, proc_codes | procedures, lab_codes | labevents
        # each item shaped like diagnoses entries or strings parseable by _codes_from_raw_list.
        drug_tokens = _codes_from_raw_list(
            rec.get("drug_codes") or rec.get("prescriptions") or [],
            MODALITY_PREFIX["drug"],
        )
        proc_tokens = _codes_from_raw_list(
            rec.get("proc_codes") or rec.get("procedures") or [],
            MODALITY_PREFIX["proc"],
        )
        lab_tokens = _codes_from_raw_list(
            rec.get("lab_codes") or rec.get("labevents") or [],
            MODALITY_PREFIX["lab"],
        )

        pa_study_id, report_raw, pa_dicom_id = _pick_one_pa_study(rec.get("cxr_studies") or [])
        report_trimmed = trim_report_text(report_raw) if report_raw else ""
        pa_image_path = ""
        if pa_study_id >= 0 and pa_dicom_id:
            if jpg_root is None:
                pa_image_path = pa_dicom_id
            else:
                pa_image_path = _cxr_jpg_path(jpg_root=jpg_root, subject_id=int(sid), study_id=int(pa_study_id), dicom_id=pa_dicom_id)

        by_subject[int(sid)].append(
            VisitRecord(
                subject_id=int(sid),
                hadm_id=int(hadm),
                admittime=admittime,
                diag_tokens=diag_tokens,
                drug_tokens=drug_tokens,
                proc_tokens=proc_tokens,
                lab_tokens=lab_tokens,
                report_text=report_trimmed,
                pa_image_path=pa_image_path,
            )
        )

    # Keep temporal ordering and only subjects with >=2 visits.
    out: Dict[int, List[VisitRecord]] = {}
    for sid, visits in by_subject.items():
        visits_sorted = sorted(visits, key=lambda x: (_parse_dt(x.admittime), x.hadm_id))
        if len(visits_sorted) >= 2:
            out[sid] = visits_sorted
    return out


def split_subjects(
    subject_ids: Sequence[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    ids = list(int(x) for x in subject_ids)
    rnd = random.Random(int(seed))
    rnd.shuffle(ids)
    n = len(ids)
    n_train = int(n * float(train_ratio))
    n_val = int(n * float(val_ratio))
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }


def _visit_has_any_clinical_code(v: VisitRecord) -> bool:
    return bool(v.diag_tokens or v.drug_tokens or v.proc_tokens or v.lab_tokens)


class Stage2MultimodalWindowDataset(Dataset):
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
        task_sampling: str = "fixed",
        task_weights: Optional[Dict[str, float]] = None,
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
        self.modality_max_codes = {k: int(MODALITY_CAPS[k]) for k in MODALITY_ORDER}
        unk_tok = getattr(tokenizer, "unk_token_id", None)
        self.unk_id = -1 if unk_tok is None else int(unk_tok)
        self.no_record_id = int(tokenizer.convert_tokens_to_ids(NO_RECORD_TOKEN))

        self.timeline_subject_ids = sorted(int(sid) for sid in timelines.keys())
        self.timelines = [timelines[sid] for sid in self.timeline_subject_ids]

        # Show-o/Qwen multimodal special ids.
        self.pad_id = int(tokenizer.pad_token_id)
        self.boi_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        self.eoi_id = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        self.img_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))

        self.windows: List[Stage2Window] = []
        self.windows_per_patient: Dict[int, int] = defaultdict(int)
        self.task_counts = Counter()
        self._task_indices: Dict[str, List[int]] = {"icd": [], "report": [], "image": []}

        for tidx, visits in enumerate(self.timelines):
            sid = self.timeline_subject_ids[tidx]
            for t in range(1, len(visits)):
                ctx_len = min(self.k_max, t)
                w_base = dict(subject_id=sid, timeline_idx=tidx, target_idx=t, ctx_start_idx=t - ctx_len)
                v_t = visits[t]
                if _visit_has_any_clinical_code(v_t):
                    self.windows.append(Stage2Window(task="icd", **w_base))
                    self.task_counts["icd"] += 1
                    self.windows_per_patient[sid] += 1
                    self._task_indices["icd"].append(len(self.windows) - 1)
                if bool(v_t.report_text):
                    self.windows.append(Stage2Window(task="report", **w_base))
                    self.task_counts["report"] += 1
                    self.windows_per_patient[sid] += 1
                    self._task_indices["report"].append(len(self.windows) - 1)
                if bool(v_t.pa_image_path):
                    self.windows.append(Stage2Window(task="image", **w_base))
                    self.task_counts["image"] += 1
                    self.windows_per_patient[sid] += 1
                    self._task_indices["image"].append(len(self.windows) - 1)

        self.task_sampling = str(task_sampling)
        self.task_weights = dict(task_weights or {"icd": 4.0, "report": 2.0, "image": 1.0})

    def __len__(self) -> int:
        return len(self.windows)

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

    def _encode_modality_block(self, tokens: Sequence[str], mod: str) -> List[int]:
        """Match Stage-1 layout: <MOD>...</MOD> with code ids or <NO_RECORD>."""
        start_id = int(self.tokenizer.convert_tokens_to_ids(f"<{MODALITY_PREFIX[mod]}>"))
        end_id = int(self.tokenizer.convert_tokens_to_ids(f"</{MODALITY_PREFIX[mod]}>"))
        cap = int(self.modality_max_codes[mod])
        code_ids: List[int] = []
        for tok in list(tokens)[:cap]:
            tid = int(self.tokenizer.convert_tokens_to_ids(tok))
            if self.unk_id >= 0 and tid == self.unk_id:
                continue
            code_ids.append(tid)
        out: List[int] = [start_id]
        if not code_ids:
            out.append(self.no_record_id)
        else:
            out.extend(code_ids)
        out.append(end_id)
        return out

    def _encode_report_block(self, report_text: str) -> List[int]:
        out = [int(self.tokenizer.convert_tokens_to_ids(self.report_start_tok))]
        out.extend(self._tokenize_report(report_text))
        out.append(int(self.tokenizer.convert_tokens_to_ids(self.report_end_tok)))
        return out

    def _encode_image_block(self) -> Tuple[List[int], Tuple[int, int]]:
        # Include boi/eoi in text stream; modality_positions points to img_pad region.
        ids = [self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id]
        # offset relative to this block (skip boi), length is image slots count.
        return ids, (1, self.num_image_tokens)

    def _ctx_image_keep_indices(self, visits_ctx: Sequence[VisitRecord]) -> set:
        if self.keep_last_n_ctx_images <= 0:
            return set()
        idx_has_img = [i for i, v in enumerate(visits_ctx) if bool(v.pa_image_path)]
        return set(idx_has_img[-self.keep_last_n_ctx_images :])

    def _serialize(self, window: Stage2Window) -> Dict[str, torch.Tensor | List[str] | str]:
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

        for i, v in enumerate(visits):
            is_target = i == target_local
            input_ids.append(visit_start_id)
            labels.append(-100)

            # Context visits: always include Stage-1-style code blocks (all 4 modalities).
            # Target visit: include only target modality block(s) to prevent leakage.
            include_clinical = (not is_target) or (is_target and window.task == "icd")
            include_report = (not is_target and bool(v.report_text)) or (is_target and window.task == "report")
            include_img = (
                (not is_target and (i in keep_ctx_img) and bool(v.pa_image_path))
                or (is_target and window.task == "image" and bool(v.pa_image_path))
            )

            if include_clinical:
                for mod in MODALITY_ORDER:
                    toks = getattr(v, f"{mod}_tokens")
                    b = self._encode_modality_block(toks, mod)
                    start = len(input_ids)
                    input_ids.extend(b)
                    labels.extend([-100] * len(b))
                    if is_target and window.task == "icd":
                        for p in range(start + 1, start + len(b)):
                            labels[p] = input_ids[p]

            if include_report:
                b = self._encode_report_block(v.report_text)
                start = len(input_ids)
                input_ids.extend(b)
                labels.extend([-100] * len(b))
                if is_target and window.task == "report":
                    # supervise report tokens and </REPORT>; do not supervise <REPORT>
                    for p in range(start + 1, start + len(b)):
                        labels[p] = input_ids[p]

            if include_img:
                b, (img_off, img_len) = self._encode_image_block()
                start = len(input_ids)
                input_ids.extend(b)
                labels.extend([-100] * len(b))
                modality_positions.append((start + img_off, img_len))
                image_paths_for_slots.append(v.pa_image_path)
                image_supervise_mask.append(1 if (is_target and window.task == "image") else 0)

            input_ids.append(visit_end_id)
            labels.append(-100)

        # Left trim while keeping newest tokens (target-end anchored).
        if len(input_ids) > self.max_seq_len:
            cut = len(input_ids) - self.max_seq_len
            input_ids = input_ids[cut:]
            labels = labels[cut:]
            new_modality_positions = []
            new_paths = []
            new_sup = []
            for (off, ln), pth, sup in zip(modality_positions, image_paths_for_slots, image_supervise_mask):
                off2 = off - cut
                if off2 < 0 or (off2 + ln) > len(input_ids):
                    continue
                new_modality_positions.append((off2, ln))
                new_paths.append(pth)
                new_sup.append(sup)
            modality_positions = new_modality_positions
            image_paths_for_slots = new_paths
            image_supervise_mask = new_sup

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones((len(input_ids),), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task": window.task,
            "subject_id": int(window.subject_id),
            "modality_positions": modality_positions,
            "image_paths_for_slots": image_paths_for_slots,
            "image_supervise_mask": torch.tensor(image_supervise_mask, dtype=torch.long),
        }

    def __getitem__(self, idx: int):
        return self._serialize(self.windows[idx])

    def task_window_counts(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self.task_counts.items()}

    def task_indices(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self._task_indices.items()}

    def sample_weights(self, patient_balance_alpha: float = 0.5) -> List[float]:
        alpha = float(patient_balance_alpha)
        weights: List[float] = []
        for w in self.windows:
            n_w = max(1, int(self.windows_per_patient[w.subject_id]))
            base = float(n_w ** (alpha - 1.0))
            t_w = float(self.task_weights.get(w.task, 1.0))
            weights.append(base * t_w)
        return weights


def collate_stage2_windows(batch: List[Dict], pad_token_id: int):
    max_len = max(int(x["input_ids"].numel()) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    task = []
    subject_ids = torch.zeros((bsz,), dtype=torch.long)

    # modality_positions is ragged; keep python list of list[(offset, len)]
    modality_positions = []
    image_paths_for_slots = []
    image_supervise_masks = []
    max_img_slots = 0
    for ex in batch:
        max_img_slots = max(max_img_slots, len(ex["modality_positions"]))
    for i, ex in enumerate(batch):
        n = int(ex["input_ids"].numel())
        input_ids[i, :n] = ex["input_ids"]
        attention_mask[i, :n] = ex["attention_mask"]
        labels[i, :n] = ex["labels"]
        task.append(str(ex["task"]))
        subject_ids[i] = int(ex["subject_id"])
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
        image_supervise_masks.append(torch.tensor(sm_list, dtype=torch.long))

    image_supervise_masks_stacked = (
        torch.stack(image_supervise_masks, dim=0) if image_supervise_masks else torch.zeros((bsz, 0), dtype=torch.long)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "task": task,
        "subject_id": subject_ids,
        "modality_positions": modality_positions,
        "image_paths_for_slots": image_paths_for_slots,
        "image_supervise_masks": image_supervise_masks_stacked,
    }


def summarize_timelines(timelines: Dict[int, List[VisitRecord]]) -> Dict[str, int]:
    visits = [v for arr in timelines.values() for v in arr]
    n_vis = len(visits)
    n_any_code = sum(1 for v in visits if _visit_has_any_clinical_code(v))
    n_diag = sum(1 for v in visits if bool(v.diag_tokens))
    n_drug = sum(1 for v in visits if bool(v.drug_tokens))
    n_proc = sum(1 for v in visits if bool(v.proc_tokens))
    n_lab = sum(1 for v in visits if bool(v.lab_tokens))
    n_report = sum(1 for v in visits if bool(v.report_text))
    n_pa = sum(1 for v in visits if bool(v.pa_image_path))
    return {
        "subjects": int(len(timelines)),
        "visits": int(n_vis),
        "visits_with_any_clinical_code": int(n_any_code),
        "visits_with_diag_codes": int(n_diag),
        "visits_with_drug_codes": int(n_drug),
        "visits_with_proc_codes": int(n_proc),
        "visits_with_lab_codes": int(n_lab),
        "visits_with_report": int(n_report),
        "visits_with_pa_image": int(n_pa),
    }


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

