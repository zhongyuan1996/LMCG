#!/usr/bin/env bash
# Stage 3 evaluation — Run H sequential chain generation.
#
# Decodes:
#   1. CXR image  (50-step ODE from prior visit context, no oracle)
#   2. Report     (GT image as oracle prefix; "FINDINGS:\n" seed)
#   3. ICD codes  (GT image + GT report as oracle prefix; constrained greedy)
#
# Pass --fair-compare to run the LongCXR/HerGEN-matched protocol used in the
# paper:  --report-max-new-tokens 384, --mask-icd-for-report.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/env.sh"

: "${MATCHING_PKL:?must be set in env.sh}"
: "${MIMIC_CXR_JPG_ROOT:?must be set in env.sh}"
: "${VAE_PTH:?must be set in env.sh}"
: "${STAGE1_OUT:?must be set in env.sh}"
: "${STAGE3_OUT:?must be set in env.sh}"

# Pick the latest Run H checkpoint by default.
if [[ -n "${RUNH_CKPT:-}" ]]; then
    CKPT="${RUNH_CKPT}"
else
    CKPT="$(ls "${STAGE3_OUT}"/checkpoint_step_*.pt 2>/dev/null | sort | tail -1)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "[eval] Cannot locate a Stage 3 checkpoint under ${STAGE3_OUT}." >&2
    echo "       Set RUNH_CKPT to override, or finish Stage 3 first." >&2
    exit 1
fi

EVAL_OUT="${EVAL_OUT:-${STAGE3_OUT}/eval}"
mkdir -p "${EVAL_OUT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python "${REPO_ROOT}/pipeline/stage3/eval_stage3.py" \
    --pretrained             showlab/show-o2-1.5B-HQ \
    --tokenizer-model        Qwen/Qwen2.5-1.5B-Instruct \
    --stage1-tokenizer-dir   "${STAGE1_OUT}/tokenizer" \
    --matching-pkl           "${MATCHING_PKL}" \
    --jpg-root               "${MIMIC_CXR_JPG_ROOT}" \
    --vae-pth                "${VAE_PTH}" \
    --state-dict-path        "${CKPT}" \
    --seed                   42 \
    --train-ratio            0.8 \
    --val-ratio              0.1 \
    --k-max                  4 \
    --keep-last-n-ctx-images 1 \
    --num-image-tokens       1024 \
    --latent-h               32 \
    --latent-w               32 \
    --image-resolution       512 \
    --max-seq-len            3072 \
    --report-max-tokens      384 \
    --lora-r                 64 \
    --lora-alpha             128 \
    --lora-diff-r            16 \
    --lora-diff-alpha        32 \
    --eval-split             val \
    --max-eval-samples       512 \
    --preview-samples        8 \
    --output-dir             "${EVAL_OUT}" \
    "$@" \
    2>&1 | tee -a "${EVAL_OUT}/eval.log"

echo "[eval] done — metrics under ${EVAL_OUT}"
