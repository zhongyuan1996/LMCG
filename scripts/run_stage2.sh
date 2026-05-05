#!/usr/bin/env bash
# Stage 2 — image-only longitudinal CXR generation warm-up.
#
# Trains a diffusion-head LoRA on top of the frozen Show-o2 LLM, with image
# task only (fixed-ratio 0:0:1).  The resulting checkpoint carries the
# diffusion LoRA weights into Stage 3.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/env.sh"

: "${MATCHING_PKL:?must be set in env.sh}"
: "${MIMIC_CXR_JPG_ROOT:?must be set in env.sh}"
: "${VAE_PTH:?must be set in env.sh}"
: "${STAGE1_OUT:?must be set in env.sh}"

# Locate the Stage 1 embedding bundle that will seed Stage 2.
if [[ -n "${STAGE1_EMB:-}" ]]; then
    S1_EMB="${STAGE1_EMB}"
elif [[ -f "${STAGE1_OUT}/best_stage1_icd_embedding_rows.pt" ]]; then
    S1_EMB="${STAGE1_OUT}/best_stage1_icd_embedding_rows.pt"
elif [[ -f "${STAGE1_OUT}/best_stage1_code_embedding_rows.pt" ]]; then
    S1_EMB="${STAGE1_OUT}/best_stage1_code_embedding_rows.pt"
elif [[ -f "${STAGE1_OUT}/latest_train_state.pt" ]]; then
    S1_EMB="${STAGE1_OUT}/latest_train_state.pt"
else
    echo "[stage2] No Stage-1 embedding file under ${STAGE1_OUT}." >&2
    echo "         Set STAGE1_EMB to override, or finish Stage 1 first." >&2
    exit 1
fi

OUT="${STAGE2_OUT}"
mkdir -p "$OUT"

BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:-20000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT:-29508}" \
    "${REPO_ROOT}/pipeline/stage2/train_stage2_baseline.py" \
    --pretrained             showlab/show-o2-1.5B-HQ \
    --tokenizer-model        Qwen/Qwen2.5-1.5B-Instruct \
    --stage1-tokenizer-dir   "${STAGE1_OUT}/tokenizer" \
    --stage1-icd-ckpt        "${S1_EMB}" \
    --matching-pkl           "${MATCHING_PKL}" \
    --jpg-root               "${MIMIC_CXR_JPG_ROOT}" \
    --vae-pth                "${VAE_PTH}" \
    --output-dir             "${OUT}" \
    --seed                   42 \
    --train-ratio            0.8 \
    --val-ratio              0.1 \
    --k-max                  4 \
    --max-seq-len            2560 \
    --keep-last-n-ctx-images 1 \
    --report-max-tokens      192 \
    --num-image-tokens       1024 \
    --latent-h               32 \
    --latent-w               32 \
    --image-resolution       512 \
    --batch-size             "${BATCH_SIZE}" \
    --num-workers            0 \
    --patient-balance-alpha  0.5 \
    --sampling-policy        fixed \
    --fixed-ratio            0:0:1 \
    --lr-text                5e-5 \
    --lr-diffusion           5e-5 \
    --llm-arm                frozen \
    --arm-a-mode             strict \
    --lora-diffusion-head \
    --lora-diff-r            16 \
    --lora-diff-alpha        32 \
    --lora-dropout           0.0 \
    --warmup-ratio           0.0 \
    --max-steps              "${MAX_STEPS}" \
    --save-every             5000 \
    --keep-last              3 \
    --save-final \
    --gradient-checkpointing \
    --sharding               zero1 \
    --mixed-precision        bf16 \
    --distributed \
    "$@" \
    2>&1 | tee -a "${OUT}/train.log"

echo "[stage2] done — checkpoint under ${OUT}"
