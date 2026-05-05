# Data Preprocessing

This pipeline turns the raw **MIMIC-IV-Hosp** and **MIMIC-CXR-JPG** PhysioNet
releases into the `matching_results.pkl` file consumed by the three training
stages, plus the per-split subject pickles consumed by the Stage 1 trainer.

All paths come from `env.sh` (`MIMIC_HOSP_ROOT`, `MIMIC_CXR_JPG_ROOT`).
None of the public datasets or vocabulary files are shipped in this repo —
each must be downloaded separately from its source under the appropriate
license.


## External inputs (not redistributed here)

You must obtain each of these yourself. Filenames are quoted as written by the
upstream releases so that the build scripts can find them by name.

### MIMIC-IV-Hosp 2.2 (PhysioNet credentialed)

Download from PhysioNet (https://physionet.org/content/mimiciv/2.2/). The
preprocessing pipeline reads only:

| File | Purpose |
|---|---|
| `hosp/diagnoses_icd.csv`            | Per-admission ICD-9 / ICD-10 diagnosis rows |
| `hosp/d_icd_diagnoses.csv.gz`       | ICD long-title dictionary, used by Stage 1 to initialize ICD-code embeddings from textual descriptions |
| `hosp/admissions.csv.gz`            | Hospital admission timestamps |
| `hosp/patients.csv.gz`              | Patient-level metadata |

Place these files (uncompressed if needed by your script invocation) under
`${MIMIC_HOSP_ROOT}/`.

### MIMIC-CXR-JPG 2.0.0 (PhysioNet credentialed)

Download from PhysioNet (https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
The pipeline reads:

| File | Purpose |
|---|---|
| `files/p10/.../*.jpg`                  | CXR JPEG images |
| `mimic-cxr-2.0.0-metadata.csv.gz`      | Per-study metadata (subject_id, study_id, view position, etc.) |
| `mimic-cxr-2.0.0-chexpert.csv.gz`      | Per-study CheXpert labels (used by downstream eval, not training) |
| `mimic-cxr-2.0.0-split.csv.gz`         | Official train/val/test split (we override with our own subject-level split) |

Place these under `${MIMIC_CXR_JPG_ROOT}/`. The free-text reports come from
the matched `mimic-cxr-reports` distribution; either download them and join,
or use the per-study free-text fields if your snapshot includes them.

### Public diagnosis-vocabulary files (free, not credentialed)

These are public files. Reproduce the paper by downloading them and pointing
the build scripts at the correct paths. None are redistributed in this repo.

| Filename | Source | Used by |
|---|---|---|
| `DXCCSR_v2025-1.csv`        | HCUP CCSR for ICD-10-CM Diagnoses release (https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp) | Optional CCSR-grouped diagnosis vocabulary (not used by Run H, which keeps raw 3-digit ICD-9). |
| `AppendixASingleDX.txt`     | HCUP single-level CCS appendix (https://hcup-us.ahrq.gov/toolssoftware/ccs/AppendixASingleDX.txt) | Optional single-level CCS grouping. |

### Per-code integer mapping (build it yourself)

The Stage 1 trainer expects a CSV mapping each 3-digit ICD-9 code to an integer
index, located at `${ICD_MAPPING_DIR}/diagnosis_to_int_mapping_mimic4.csv`.
Build it from `diagnoses_icd.csv` once at preprocessing time:

```python
# Pseudo-code; adapt to your script of choice.
import pandas as pd
df = pd.read_csv(f"{MIMIC_HOSP_ROOT}/diagnoses_icd.csv", usecols=["icd_code", "icd_version"])
df_icd9 = df[df.icd_version == 9].copy()
df_icd9["stem3"] = df_icd9.icd_code.str.replace(".", "", regex=False).str[:3]
codes = sorted(df_icd9.stem3.unique())
pd.DataFrame({"code": codes, "idx": range(len(codes))}).to_csv(
    f"{ICD_MAPPING_DIR}/diagnosis_to_int_mapping_mimic4.csv", index=False)
```

Run H uses **3-digit ICD-9** stems (859 unique codes in our train split).


## Outputs

| File | Notes |
|---|---|
| `${MATCHING_PKL}` | Per-patient longitudinal record. Keyed by `subject_id`; each value is a list of visits with (`hadm_id`, admit / discharge times, ICD-9 diagnosis codes, matched CXR study_id, image path, free-text report). |
| `${SHARED_DIR}/{train,val,test}_subjects.pkl` | Patient-level subject splits used by Stage 1. We use a fixed 80 / 10 / 10 split seeded with 42 in the paper. |
| `${REPO_ROOT}/outputs/cxr_ehr_manifest.jsonl` | Optional flat manifest produced by `build_cxr_ehr_manifest_from_matching.py`; useful for sanity-checking the join. |


## Scripts (provided in this repo)

| Script | Purpose |
|---|---|
| `filter_multimodal_patients.py`           | Filter MIMIC-IV-Hosp patients down to those with at least one CXR study and one ICD admission. |
| `build_cxr_path_map.py`                   | Build a `study_id -> jpg path` map from MIMIC-CXR-JPG metadata. |
| `build_cxr_ehr_manifest_from_matching.py` | Per-row sanity manifest of (subject_id, hadm_id, study_id, image path, report path). |
| `build_dx_vocab.py`                       | Build the diagnosis vocabulary from `diagnoses_icd.csv`. |


## Suggested pipeline order

```bash
source env.sh

# 1. Filter to multimodal patients.
python data_preprocess/filter_multimodal_patients.py

# 2. Build the CXR study_id -> JPG path lookup.
python data_preprocess/build_cxr_path_map.py

# 3. Build the diagnosis vocabulary.
python data_preprocess/build_dx_vocab.py \
    --csv "${MIMIC_HOSP_ROOT}/diagnoses_icd.csv"

# 4. Sanity-check by exporting a flat manifest.
python data_preprocess/build_cxr_ehr_manifest_from_matching.py
```

The scripts above produce intermediate JSON / pickle artifacts; the final
`${MATCHING_PKL}` is assembled by combining them. Adapt the exact join logic
(reports ↔ studies ↔ admissions) to your local file layout.

Run H uses **only** the ICD-9 diagnosis modality. Drug / lab / procedure
codes are out of scope for this release; the trainers tolerate their absence
on a per-visit basis.
