#!/usr/bin/env bash
# Stage 1 — ICD code embedding pre-training.
#
# Trains the ICD9 code embedding rows on top of the Show-o2 / Qwen2.5-1.5B
# tokenizer.  Writes the trained embedding rows (and the extended tokenizer)
# to ${STAGE1_OUT}.  Stages 2 and 3 consume those.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/env.sh"

: "${SHARED_DIR:?must be set in env.sh — see data_preprocess/README.md}"
: "${MIMIC_HOSP_ROOT:?must be set in env.sh}"

OUT="${STAGE1_OUT}"
mkdir -p "$OUT"
LOG="${OUT}/train.log"

# 1 GPU, batch size 24, 2 epochs, max_seq_len 162 — Run H Stage 1 recipe.
PER_GPU_BATCH="${PER_GPU_BATCH:-24}"
EPOCHS="${EPOCHS:-2}"

# Stage 1 trains ICD-9 diagnosis-code embedding rows only.
torchrun --standalone --nproc_per_node=1 \
    "${REPO_ROOT}/pipeline/stage1/train.py" \
    --shared-dir              "${SHARED_DIR}" \
    --mimic-d-icd-gz          "${MIMIC_HOSP_ROOT}/d_icd_diagnoses.csv.gz" \
    --output-dir              "${OUT}" \
    --epochs                  "${EPOCHS}" \
    --batch-size              "${PER_GPU_BATCH}" \
    --num-workers             0 \
    --use-distributed \
    --log-vram-every          2000 \
    "$@" \
    2>&1 | tee "${LOG}"

echo "[stage1] done — embedding rows written under ${OUT}"
