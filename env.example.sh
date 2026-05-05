#!/usr/bin/env bash
# Copy this file to env.sh and fill in the values for your environment, then
#   source env.sh
# before launching any training or evaluation script.  All paths must be
# absolute.

# ---------------------------------------------------------------------------
# Repository root (auto-detected)
# ---------------------------------------------------------------------------
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make the repository's source tree importable without an editable install.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Pretrained model artifacts
# ---------------------------------------------------------------------------
# WAN 2.1 VAE checkpoint (https://huggingface.co/showlab — Wan2.1_VAE_1.3B.pth)
export VAE_PTH=""

# ---------------------------------------------------------------------------
# Training data (PhysioNet credentialed access required)
# ---------------------------------------------------------------------------
# Output of the data_preprocess pipeline; see data_preprocess/README.md
export MATCHING_PKL=""
# Path to the MIMIC-CXR-JPG 2.0.0 "files" directory (one subdir per p<tens>)
export MIMIC_CXR_JPG_ROOT=""
# Optional companion: original MIMIC-CXR 2.0.0 dicom-derived report files
export MIMIC_CXR_ROOT=""
# Path to the MIMIC-IV-Hosp 2.2 (or 3.1) hosp/ directory.  Used for the
# d_icd_diagnoses.csv.gz and diagnoses_icd.csv files referenced by
# data_preprocess and Stage 1 ICD-code embedding initialization.
export MIMIC_HOSP_ROOT=""

# ICD-9 integer mapping CSV (built from MIMIC-IV-Hosp; see
# data_preprocess/README.md for the one-shot build recipe).
export ICD_MAPPING_DIR="${REPO_ROOT}/data_preprocess/icd_mappings"

# Stage 1 shared subjects directory (subjects pickled into
# {train,val,test}_subjects.pkl by the shared MIMIC-IV-Hosp ICD-9
# preprocessing pipeline).  See data_preprocess/README.md for how to build
# this from MATCHING_PKL.
export SHARED_DIR=""

# ---------------------------------------------------------------------------
# Stage outputs (consumed by later stages)
# ---------------------------------------------------------------------------
export STAGE1_OUT="${REPO_ROOT}/outputs/stage1"
export STAGE2_OUT="${REPO_ROOT}/outputs/stage2"
export STAGE3_OUT="${REPO_ROOT}/outputs/stage3_runh"

# ---------------------------------------------------------------------------
# Optional
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
