#!/usr/bin/env python3
"""
Build ICD-9/ICD-10 vocabulary index from MIMIC-IV diagnoses_icd.csv.

- ICD-9 codes are aggregated to 3-character stems (common practice).
- ICD-10 codes are aggregated to 5-character stems (common practice).
- Writes a JSON index so we don't recompute on each run.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Set, Tuple


def normalize_code(code: str) -> str:
    """Uppercase, strip spaces/dots."""
    return code.strip().upper().replace(".", "").replace(" ", "")


def stem_code(code: str, length: int) -> str:
    """Return prefix up to length (or shorter if code shorter)."""
    return code[:length] if code else code


def build_vocab(csv_path: Path, icd9_stem_len: int = 3, icd10_stem_len: int = 3) -> Tuple[Set[str], Set[str], int]:
    icd9_raw: Set[str] = set()
    icd10_raw: Set[str] = set()
    total = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            code = normalize_code(row.get("icd_code", ""))
            if not code:
                continue
            version = str(row.get("icd_version", "")).strip()
            if version == "9":
                icd9_raw.add(code)
            elif version == "10":
                icd10_raw.add(code)
    icd9_stems = {stem_code(c, icd9_stem_len) for c in icd9_raw if c}
    icd10_stems = {stem_code(c, icd10_stem_len) for c in icd10_raw if c}
    return icd9_stems, icd10_stems, total


def main():
    parser = argparse.ArgumentParser(description="Build ICD vocab index from diagnoses_icd.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("${MIMIC_HOSP_ROOT}/diagnoses_icd.csv"),
        help="Path to diagnoses_icd.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "output" / "dx_vocab_mimiciv_v3_1.json",
        help="Output JSON path for vocab index",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    icd9_stems, icd10_stems, total_rows = build_vocab(args.csv)

    payload: Dict = {
        "source": str(args.csv),
        "total_rows": total_rows,
        "icd9_codes_stem_len": 3,
        "icd10_codes_stem_len": 3,
        "num_icd9_codes": len(icd9_stems),
        "num_icd10_codes": len(icd10_stems),
        "icd9_codes": sorted(icd9_stems),
        "icd10_codes": sorted(icd10_stems),
        "note": "Codes uppercased, dots/spaces removed; ICD-9 stems to 3 chars; ICD-10 stems to 5 chars.",
    }

    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved vocab to {args.output}")
    print(f"ICD-9 codes: {len(icd9_stems)}, ICD-10 codes: {len(icd10_stems)}, rows read: {total_rows}")


if __name__ == "__main__":
    main()

