#!/usr/bin/env python3
"""
Build a jsonl manifest of MIMIC-CXR-JPG images paired with report txt files,
restricted to the cohort already matched to have BOTH CXR and EHR.

Source cohort: matching_results.pkl (output of the prior modality matching).

This manifest is intended for the Phase A/B grounding generation using a VL LLM:
  - Phase A: image-only grounding draft
  - Phase B: reconcile with report text

By default we DO NOT embed report text in the manifest (to keep file size reasonable).
We write paths + ids so downstream code can read report text on demand.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


def _pgroup(subject_id: int) -> str:
    # MIMIC-CXR directory convention: p10/p10000032/... where "10" is subject_id // 1_000_000.
    # (e.g., 10000032 -> 10, 10002013 -> 10)
    return f"p{int(subject_id)//1_000_000:d}"


def cxr_jpg_path(*, jpg_root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    return jpg_root / _pgroup(subject_id) / f"p{int(subject_id):d}" / f"s{int(study_id):d}" / f"{dicom_id}.jpg"


def cxr_report_txt_path(*, report_root: Path, subject_id: int, study_id: int) -> Path:
    return report_root / _pgroup(subject_id) / f"p{int(subject_id):d}" / f"s{int(study_id):d}.txt"


def iter_rows(
    *,
    matching: list,
    jpg_root: Path,
    report_root: Path,
    require_ehr: bool,
    require_cxr: bool,
    include_report_text_from_pkl: bool,
) -> Iterator[Dict[str, Any]]:
    for rec in matching:
        if not isinstance(rec, dict):
            continue
        if require_ehr and not bool(rec.get("has_ehr", False)):
            continue
        if require_cxr and not bool(rec.get("has_cxr", False)):
            continue
        subject_id = rec.get("subject_id", None)
        hadm_id = rec.get("hadm_id", None)
        cxr_studies = rec.get("cxr_studies") or []
        if not isinstance(subject_id, int):
            continue
        if not cxr_studies:
            continue

        for st in cxr_studies:
            if not isinstance(st, dict):
                continue
            study_id = st.get("study_id", None)
            if not isinstance(study_id, int):
                continue
            report_text = st.get("report_text", None) if include_report_text_from_pkl else None
            images = st.get("images") or []
            if not isinstance(images, list) or not images:
                continue

            # One row per dicom image.
            for im in images:
                if not isinstance(im, dict):
                    continue
                dicom_id = im.get("dicom_id", None)
                if not isinstance(dicom_id, str) or not dicom_id:
                    continue
                view_position = im.get("view_position", None)
                row: Dict[str, Any] = {
                    "subject_id": int(subject_id),
                    "hadm_id": int(hadm_id) if isinstance(hadm_id, int) else hadm_id,
                    "study_id": int(study_id),
                    "dicom_id": dicom_id,
                    "view_position": view_position,
                    "image_path": str(cxr_jpg_path(jpg_root=jpg_root, subject_id=int(subject_id), study_id=int(study_id), dicom_id=dicom_id)),
                    "report_path": str(cxr_report_txt_path(report_root=report_root, subject_id=int(subject_id), study_id=int(study_id))),
                }
                if include_report_text_from_pkl and isinstance(report_text, str) and report_text.strip():
                    row["report_text"] = report_text
                yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matching-pkl",
        type=str,
        default="${MATCHING_PKL}",
        help="Path to matching_results.pkl",
    )
    ap.add_argument(
        "--jpg-root",
        type=str,
        default="${MIMIC_CXR_JPG_ROOT}",
        help="Root directory containing pXX/pSUBJECT/sSTUDY/DICOM.jpg",
    )
    ap.add_argument(
        "--report-root",
        type=str,
        default="${MIMIC_CXR_ROOT}",
        help="Root directory containing pXX/pSUBJECT/sSTUDY.txt",
    )
    ap.add_argument(
        "--out-jsonl",
        type=str,
        default="${REPO_ROOT}/outputs/cxr_ehr_manifest.jsonl",
        help="Output jsonl path",
    )
    ap.add_argument("--require-ehr", action="store_true", default=True, help="Keep only records with has_ehr=True (default).")
    ap.add_argument("--require-cxr", action="store_true", default=True, help="Keep only records with has_cxr=True (default).")
    ap.add_argument("--max-rows", type=int, default=0, help="If >0, cap the number of output rows.")
    ap.add_argument(
        "--include-report-text-from-pkl",
        action="store_true",
        help="If set, also write report_text from matching_results.pkl into each row (file will be large).",
    )
    args = ap.parse_args()

    matching_pkl = Path(args.matching_pkl)
    jpg_root = Path(args.jpg_root)
    report_root = Path(args.report_root)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {matching_pkl} ... (this is large; may take a while)")
    with open(matching_pkl, "rb") as f:
        matching = pickle.load(f)
    if not isinstance(matching, list):
        raise TypeError(f"Expected list in matching_results.pkl, got {type(matching)}")
    print(f"[load] records={len(matching):,}")

    n = 0
    n_missing_img = 0
    n_missing_rpt = 0
    with open(out_jsonl, "w", encoding="utf-8") as w:
        for row in iter_rows(
            matching=matching,
            jpg_root=jpg_root,
            report_root=report_root,
            require_ehr=bool(args.require_ehr),
            require_cxr=bool(args.require_cxr),
            include_report_text_from_pkl=bool(args.include_report_text_from_pkl),
        ):
            # Light validation; do not fail hard (some files may be missing).
            if not Path(row["image_path"]).exists():
                n_missing_img += 1
            if not Path(row["report_path"]).exists():
                n_missing_rpt += 1
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.max_rows and n >= int(args.max_rows):
                break
            if n % 10000 == 0:
                print(f"[write] rows={n:,} missing_img={n_missing_img:,} missing_rpt={n_missing_rpt:,}")

    print(f"[done] wrote={n:,} -> {out_jsonl}")
    print(f"[done] missing_img={n_missing_img:,} missing_rpt={n_missing_rpt:,}")


if __name__ == "__main__":
    main()

