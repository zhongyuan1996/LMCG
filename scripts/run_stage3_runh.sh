#!/usr/bin/env bash
# Stage 3 — Run H, sequential clinical chain generation training.
#
# Reproduces the Run H recipe:
#   - LLM LoRA (fresh)        : r=64, alpha=128, lr=2e-4
#   - Diffusion LoRA (continued from Stage 2): r=16, alpha=32, lr=5e-6
#   - 10,000 steps, 2,000-step report-only warm-up, then balanced 8k steps.
#   - Loss weights: report x 10, ICD x 1, image upsampling x 20.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/env.sh"

: "${MATCHING_PKL:?must be set in env.sh}"
: "${MIMIC_CXR_JPG_ROOT:?must be set in env.sh}"
: "${VAE_PTH:?must be set in env.sh}"
: "${STAGE1_OUT:?must be set in env.sh}"
: "${STAGE2_OUT:?must be set in env.sh}"

# Stage 1 embedding bundle (baked into the embedding table of Run H).
if [[ -n "${STAGE1_EMB:-}" ]]; then
    S1_EMB="${STAGE1_EMB}"
elif [[ -f "${STAGE1_OUT}/best_stage1_icd_embedding_rows.pt" ]]; then
    S1_EMB="${STAGE1_OUT}/best_stage1_icd_embedding_rows.pt"
elif [[ -f "${STAGE1_OUT}/best_stage1_code_embedding_rows.pt" ]]; then
    S1_EMB="${STAGE1_OUT}/best_stage1_code_embedding_rows.pt"
else
    echo "[stage3] No Stage-1 embedding file under ${STAGE1_OUT}." >&2
    exit 1
fi

# Stage 2 checkpoint that carries the trained diffusion LoRA.
if [[ -n "${STAGE2_CKPT:-}" ]]; then
    S2_CKPT="${STAGE2_CKPT}"
elif [[ -f "${STAGE2_OUT}/checkpoint_step_00020000.pt" ]]; then
    S2_CKPT="${STAGE2_OUT}/checkpoint_step_00020000.pt"
else
    S2_CKPT="$(ls "${STAGE2_OUT}"/checkpoint_step_*.pt 2>/dev/null | sort | tail -1)"
fi
if [[ -z "${S2_CKPT}" || ! -f "${S2_CKPT}" ]]; then
    echo "[stage3] Cannot locate Stage 2 checkpoint under ${STAGE2_OUT}." >&2
    echo "         Set STAGE2_CKPT to override, or finish Stage 2 first." >&2
    exit 1
fi

OUT="${STAGE3_OUT}"
mkdir -p "${OUT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python "${REPO_ROOT}/pipeline/stage3/train_stage3.py" \
    --pretrained             showlab/show-o2-1.5B-HQ \
    --tokenizer-model        Qwen/Qwen2.5-1.5B-Instruct \
    --stage1-tokenizer-dir   "${STAGE1_OUT}/tokenizer" \
    --stage1-icd-ckpt        "${S1_EMB}" \
    --stage2-ckpt            "${S2_CKPT}" \
    --matching-pkl           "${MATCHING_PKL}" \
    --jpg-root               "${MIMIC_CXR_JPG_ROOT}" \
    --vae-pth                "${VAE_PTH}" \
    --output-dir             "${OUT}" \
    --seed                   42 \
    --train-ratio            0.8 \
    --val-ratio              0.1 \
    --k-max                  4 \
    --keep-last-n-ctx-images 1 \
    --num-image-tokens       1024 \
    --latent-h               32 \
    --latent-w               32 \
    --image-resolution       512 \
    --batch-size             1 \
    --num-workers            0 \
    --patient-balance-alpha  0.5 \
    --warmup-ratio           0.05 \
    --max-steps              10000 \
    --phase1-steps           2000 \
    --icd-ramp-steps         0 \
    --save-every             5000 \
    --save-final \
    --gradient-checkpointing \
    --mixed-precision        bf16 \
    --max-seq-len            3072 \
    --report-max-tokens      384 \
    --report-loss-weight     10.0 \
    --icd-loss-weight        1.0 \
    --modality-image-weight  20.0 \
    --lora-r                 64 \
    --lora-alpha             128 \
    --lora-diff-r            16 \
    --lora-diff-alpha        32 \
    --lr-llm-lora            2e-4 \
    --lr-diff-lora           5e-6 \
    "$@" \
    2>&1 | tee -a "${OUT}/train.log"

echo "[stage3] done — checkpoint under ${OUT}"
