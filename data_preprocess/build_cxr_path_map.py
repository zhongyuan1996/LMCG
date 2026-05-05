#!/usr/bin/env python3
"""
Build a CXR path map (hadm_id -> jpg_path) from cxr_vae_train_dataset.json.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Build hadm_id -> CXR jpg_path map")
    ap.add_argument("--dataset", type=Path, default=Path("outputs/preprocess/cxr_vae_train_dataset.json"))
    ap.add_argument("--output", type=Path, default=Path("outputs/preprocess/cxr_path_map.json"))
    ap.add_argument("--sample", type=int, default=None, help="Optional limit for testing")
    args = ap.parse_args()

    with args.dataset.open() as f:
        js = json.load(f)
    images = js.get("images", js if isinstance(js, list) else [])
    if args.sample:
        images = images[: args.sample]

    path_map = {}
    for rec in images:
        hadm = rec.get("hadm_id")
        jpg = rec.get("jpg_path")
        if hadm is None or not jpg:
            continue
        path_map[str(hadm)] = jpg

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(path_map, f, indent=2)
    print(f"Saved {len(path_map)} entries to {args.output}")


if __name__ == "__main__":
    main()

