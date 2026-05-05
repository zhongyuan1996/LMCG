# LMCG — Longitudinal Multimodal Clinical Generation

Reference implementation for a sequential multimodal clinical chain generation
model: given a longitudinal patient context (prior CXR images, radiology
reports, and ICD diagnosis codes), the model generates the next visit's

1. CXR image  (flow-matching diffusion head conditioned on the LLM context),
2. radiology report  (next-token prediction from the LLM),
3. ICD-9 diagnosis codes  (constrained next-token prediction).

The released model is built on the Show-o2 1.5B-HQ backbone (Qwen2.5-1.5B-
Instruct LLM + WAN 2.1 VAE diffusion head) and trained in three stages.


## Repository layout

```
LMCG/
  env.example.sh            -> copy to env.sh; edit; source
  requirements.txt
  LICENSE
  README.md
  src/
    models/                 Show-o2 backbone (LLM + diffusion head)
    transport/              Flow-matching transport (Linear / GVP / VP paths)
    utils/                  LoRA, schedulers, logging, config helpers
  pipeline/
    stage1/                 ICD code embedding pre-training
    stage2/                 Image-only diffusion warm-up
    stage3/                 Run H -- sequential chain training and evaluation
  data_preprocess/          Scripts to build matching_results.pkl from the
                            raw MIMIC-IV-Hosp + MIMIC-CXR-JPG releases
  scripts/                  run_stage1.sh / run_stage2.sh / run_stage3_runh.sh /
                            run_eval.sh
```


## Prerequisites

- Linux, CUDA 12.x, NVIDIA driver supporting bf16 (Ampere or newer).
- Python 3.10+ with the dependencies listed in `requirements.txt`.
  (Any conda or venv environment works; the scripts just call `python`.)
- One credentialed copy each of:
    - **MIMIC-IV-Hosp 2.2** (or 3.1) — for the `diagnoses_icd.csv` file
      and the `d_icd_diagnoses.csv.gz` ICD-9 long-title dictionary used to
      initialize the ICD-9 code embeddings.
    - **MIMIC-CXR-JPG 2.0.0** — for the CXR images and the `mimic-cxr-2.0.0-*.csv`
      manifests.
- The **WAN 2.1 1.3B VAE** checkpoint (`Wan2.1_VAE_1.3B.pth`) — public, see
  the original Show-o2 release on Hugging Face for the exact file.
- The base LLM and tokenizer pull automatically from Hugging Face Hub on first
  run:
    - `showlab/show-o2-1.5B-HQ`
    - `Qwen/Qwen2.5-1.5B-Instruct`


## Setup

```bash
git clone <THIS_REPO>
cd <repo-root>

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp env.example.sh env.sh
$EDITOR env.sh        # fill in VAE_PTH, MATCHING_PKL, MIMIC_CXR_JPG_ROOT,
                      # MIMIC_HOSP_ROOT, SHARED_DIR, STAGE1_OUT, ...
source env.sh
```

`env.sh` is the single source of truth for filesystem paths.  None of the
training or evaluation scripts hard-code paths.


## Data preparation

See `data_preprocess/README.md` for the full pipeline.  Briefly:

1. Filter MIMIC-IV-Hosp patients down to the multimodal cohort that also has
   at least one MIMIC-CXR-JPG study.
2. Build a per-patient longitudinal record (`matching_results.pkl`) keyed by
   subject_id, containing visit-level reports, ICD-9 diagnosis codes, and
   matched CXR study paths.
3. Snapshot the patient subject splits into `${SHARED_DIR}` for the Stage 1
   trainer.

`data_preprocess/README.md` lists the public files (PhysioNet MIMIC-IV-Hosp,
PhysioNet MIMIC-CXR-JPG, HCUP CCSR vocabularies) that you must download
yourself; none of them are redistributed in this repo. It also gives a short
recipe for building the per-code integer-index CSV at
`${ICD_MAPPING_DIR}/diagnosis_to_int_mapping_mimic4.csv` that the loaders read.


## Training

The three stages must be run in order.  All scripts expect `env.sh` to be
sourced.

```bash
bash scripts/run_stage1.sh                # ~1 GPU * a few hours
bash scripts/run_stage2.sh                # ~2 GPUs * ~1 day
bash scripts/run_stage3_runh.sh           # 1 GPU * ~1-2 days for 10 000 steps
```

Each script logs to `${STAGE_OUT}/train.log` and writes checkpoints to
`${STAGE_OUT}`.  Stage 2 reads the Stage 1 embedding bundle automatically;
Stage 3 reads both the Stage 1 bundle and the Stage 2 diffusion-LoRA
checkpoint.  Override with `STAGE1_EMB=...`, `STAGE2_CKPT=...` if you want
specific checkpoints.

### Run H hyper-parameters (Stage 3)

| flag | value |
|------|-------|
| `--max-steps`             | 10 000 |
| `--phase1-steps`          | 2 000  (report-only warm-up; ICD loss off) |
| `--icd-ramp-steps`        | 0       (instant transition to balanced loss) |
| `--batch-size`            | 1      (with gradient checkpointing + bf16) |
| `--max-seq-len`           | 3 072  |
| `--report-max-tokens`     | 384    |
| `--num-image-tokens`      | 1 024 (32 x 32 latent grid at resolution 512) |
| `--report-loss-weight`    | 10.0   |
| `--icd-loss-weight`       | 1.0    |
| `--modality-image-weight` | 20.0   |
| `--lora-r / --lora-alpha`               | 64 / 128  (LLM LoRA, fresh) |
| `--lora-diff-r / --lora-diff-alpha`     | 16 / 32   (diffusion LoRA, continued) |
| `--lr-llm-lora`           | 2e-4   |
| `--lr-diff-lora`          | 5e-6   |
| `--patient-balance-alpha` | 0.5    |
| `--k-max`                 | 4      (max prior visits) |
| `--keep-last-n-ctx-images`| 1      |
| `--seed`                  | 42     |


## Evaluation

```bash
bash scripts/run_eval.sh --fair-compare
```

`--fair-compare` enables the LongCXR / HerGEN-matched protocol used in the
paper (oracle GT image + GT report prefix for ICD; report capped at 384
tokens; ICD tokens masked when generating the report so the report cannot
peek at the ground-truth codes).

The output JSON contains:

- `report.bleu_1`, `report.bleu_4`, `report.rouge_l`  (NLTK + rouge-score)
- `icd.diag.f1_mean`  (macro-F1 over the diagnosis vocabulary)
- `image.ssim_mean`, `image.psnr_mean`


## Known issues

- The codebase is a research snapshot.  A handful of optional code paths
  (cross-attention adapters, relational memory, expert-feature distillation,
  and several ablations) appear behind feature flags that the Run H
  configuration does not toggle.  They may have rough edges; the Run H launch
  shells exercise only the supported subset.
- The `cxr_vae_2d.py` module includes an opt-in latent-statistics file
  loaded only when running the auxiliary 2D VAE; it is **not** used by
  Run H.  The default path resolves to `${REPO_ROOT}/outputs/cxr_latent_stats.pth`
  but is gated by an `if path.exists()` check.
- Stage 1 expects an external `${SHARED_DIR}` containing
  `{train,val,test}_subjects.pkl`.  The data_preprocess pipeline produces
  these from `matching_results.pkl`; see `data_preprocess/README.md`.


## Citation

Citation withheld for double-blind review.  Will be added on acceptance.


## License

Apache 2.0 — see `LICENSE`.

Portions of the Show-o2 backbone (`src/models/`, `src/transport/`) are
derived from the public Show-o2 release; their original copyright and
license terms are preserved in the file headers where present.
