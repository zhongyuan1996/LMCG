#!/usr/bin/env python3
"""
Stage 3 (Run H) — sequential clinical chain generation training.

Input : longitudinal multimodal visit context (same as Stage 2)
Output: [CXR image] -> [radiology report] -> [legacy ICD diagnosis codes]

LoRA setup
----------
  - Diffusion-head LoRA (r=16, alpha=32): carried from the Stage 2 checkpoint,
    continued at low LR (5e-6).
  - LLM LoRA (r=64, alpha=128, single adapter): injected fresh on the upper 8
    Qwen-2.5 decoder layers (q/k/v/o/gate/up/down -> 56 modules), trained at
    LR 2e-4.

Loss
----
  Per sample, active losses depend on which modalities have ground truth:
    - flow-matching loss (target visit has a PA image)
    - NTP loss on report tokens (target visit has a report)
    - NTP loss on ICD tokens     (target visit has diagnosis codes)
  Total loss = report_loss_weight * loss_report
             + icd_loss_weight    * loss_icd  (after curriculum ramp)
             + loss_flow

Curriculum
----------
  Two-phase: ``--phase1-steps N`` of report-only sampling (ICD CE forced to 0),
  followed by balanced sampling. ``--icd-ramp-steps`` controls the ramp from 0
  to ``--icd-loss-weight`` after phase 1; Run H uses 0 for a hard switch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image, ImageOps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Stage 2's data_pipeline module is required for the Stage 2 -> Stage 3 helpers
# (timeline loader, special tokens, splits). Insert ahead of stage1 in case both
# directories are on PYTHONPATH so the correct module wins.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage2"))

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

from data_pipeline import (  # noqa: E402
    add_stage2_special_tokens,
    load_patient_timelines_from_matching_pkl,
    save_json,
    split_subjects,
)
from data_pipeline_stage3 import (  # noqa: E402
    Stage3ChainWindowDataset,
    collate_stage3_windows,
)
from stage3_text_masks import (  # noqa: E402
    mask_all_code_spans,
    mask_report_span,
)
from models import Showo2Qwen2_5, WanVAE  # noqa: E402
from transport import create_transport  # noqa: E402
from utils.lora import LoraSpec, LoRALinear  # noqa: E402


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_distributed(enabled: bool) -> Tuple[bool, int, int, int]:
    if not enabled:
        return False, 0, 1, 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def _is_main(rank: int) -> bool:
    return int(rank) == 0


# ---------------------------------------------------------------------------
# Stage 1 embedding load
# ---------------------------------------------------------------------------

def _load_stage1_embedding_rows(model: Showo2Qwen2_5, stage1_ckpt: str) -> Dict:
    ckpt = torch.load(stage1_ckpt, map_location="cpu")
    if "trainable_token_ids" not in ckpt or "embedding_rows" not in ckpt:
        raise KeyError(f"Stage-1 checkpoint missing keys in {stage1_ckpt}")
    token_ids = [int(x) for x in ckpt["trainable_token_ids"]]
    rows = ckpt["embedding_rows"]
    if not isinstance(rows, torch.Tensor):
        rows = torch.tensor(rows)
    emb = model.showo.get_input_embeddings().weight
    with torch.no_grad():
        idx = torch.tensor(token_ids, dtype=torch.long, device=emb.device)
        rows_cast = rows.to(device=emb.device, dtype=emb.dtype)
        emb[idx] = rows_cast
        diff = (emb[idx] - rows_cast).abs().max().item()
    return {
        "loaded_rows": len(token_ids),
        "max_abs_diff_after_copy": float(diff),
    }


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

_LLM_LORA_TARGETS = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}


def _apply_lora_upper8(model: Showo2Qwen2_5, *, r: int, alpha: int, dropout: float) -> Dict:
    """Single LoRA adapter on the upper-8 LLM decoder layers (Run H)."""
    spec = LoraSpec(r=int(r), alpha=int(alpha), dropout=float(dropout))
    layers = getattr(model.showo.model, "layers", None)
    if layers is None:
        raise RuntimeError("Could not find showo.model.layers")
    n_layers = len(layers)
    wrapped = 0
    for li in range(max(0, n_layers - 8), n_layers):
        layer = layers[li]
        for name, child in list(layer.named_modules()):
            if name not in _LLM_LORA_TARGETS or not isinstance(child, nn.Linear):
                continue
            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(child, spec))
            wrapped += 1
    lora_params = sum(
        p.numel() for n, p in model.named_parameters()
        if (".lora_A." in n or ".lora_B." in n) and "diffusion_head_a" not in n
    )
    return {"modules": wrapped, "params": lora_params, "r": int(r), "alpha": int(alpha)}


def _apply_lora_diffusion_head(model: Showo2Qwen2_5, *, r: int, alpha: int, dropout: float) -> Dict:
    spec = LoraSpec(r=int(r), alpha=int(alpha), dropout=float(dropout))
    wrapped = 0
    for layer in model.diffusion_head_a:
        for name, child in list(layer.named_modules()):
            if name not in _LLM_LORA_TARGETS or not isinstance(child, nn.Linear):
                continue
            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(child, spec))
            wrapped += 1
    diff_lora_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "diffusion_head_a" in n and (".lora_A." in n or ".lora_B." in n)
    )
    return {"modules": wrapped, "params": diff_lora_params, "r": int(r), "alpha": int(alpha)}


# ---------------------------------------------------------------------------
# Attention masks
# ---------------------------------------------------------------------------

def _build_causal_padding_mask(attn_1d: torch.Tensor) -> torch.Tensor:
    bsz, seqlen = attn_1d.shape
    dev   = attn_1d.device
    valid = attn_1d.bool()
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=dev))
    keep  = causal.unsqueeze(0) & valid.unsqueeze(1)
    mask  = torch.zeros((bsz, 1, seqlen, seqlen), dtype=torch.float32, device=dev)
    return mask.masked_fill(~keep.unsqueeze(1), -1e4)


def _build_omni_padding_mask(attn_1d: torch.Tensor, modality_positions: torch.Tensor) -> torch.Tensor:
    bsz, seqlen = attn_1d.shape
    dev   = attn_1d.device
    valid = attn_1d.bool()
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=dev))
    keep  = causal.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)
    if modality_positions is not None and modality_positions.numel() > 0:
        mp = modality_positions.to(device=dev)
        for b in range(bsz):
            for off, ln in mp[b]:
                s = max(0, int(off.item()))
                e = min(seqlen, s + int(ln.item()))
                if e > s:
                    keep[b, s:e, s:e] = True
    mask = torch.zeros((bsz, 1, seqlen, seqlen), dtype=torch.float32, device=dev)
    return mask.masked_fill(~keep.unsqueeze(1), -1e4)


# ---------------------------------------------------------------------------
# Image preprocessing / image-slot collection
# ---------------------------------------------------------------------------

def _preprocess_image(path: str, resolution: int) -> torch.Tensor:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("L").convert("RGB")
        w, h = img.size
        scale = float(resolution) / float(min(w, h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = img.resize((nw, nh), resample=Image.BICUBIC)
        left = max(0, (nw - resolution) // 2)
        top  = max(0, (nh - resolution) // 2)
        img  = img.crop((left, top, left + resolution, top + resolution))
        arr  = np.asarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return x * 2.0 - 1.0
    except Exception:
        return torch.zeros((3, resolution, resolution), dtype=torch.float32)


def _collect_image_slots(
    batch: dict,
    *,
    device: torch.device,
    resolution: int,
    img_pad_id: int,
    pad_id: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    bsz, _seqlen = input_ids.shape
    modality_positions = batch["modality_positions"]
    paths = batch["image_paths_for_slots"]
    if not modality_positions or all(len(row) == 0 for row in modality_positions):
        text_masks = ((input_ids != img_pad_id) & (input_ids != pad_id)).long()
        image_masks = (input_ids == img_pad_id).long()
        return None, None, text_masks, image_masks

    m_slots = len(modality_positions[0])
    if not all(len(row) == m_slots for row in modality_positions):
        raise ValueError("collate_stage3_windows must pad modality_positions to a fixed M per batch")
    if not all(len(row) == m_slots for row in paths):
        raise ValueError("collate_stage3_windows must pad image_paths_for_slots to match modality slots")

    pos_t = torch.zeros(bsz, m_slots, 2, dtype=torch.long, device=device)
    for i in range(bsz):
        for j in range(m_slots):
            off, ln = modality_positions[i][j]
            pos_t[i, j, 0] = int(off)
            pos_t[i, j, 1] = int(ln)

    rows: List[torch.Tensor] = []
    for i in range(bsz):
        for j in range(m_slots):
            rows.append(_preprocess_image(str(paths[i][j]), resolution=resolution))
    pixel_values = torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)

    text_masks = ((input_ids != img_pad_id) & (input_ids != pad_id)).long()
    image_masks = (input_ids == img_pad_id).long()
    return pixel_values, pos_t, text_masks, image_masks


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def _save_checkpoint(
    *, out_dir: Path, step: int, model, optimizer, scheduler, stats: Dict,
    args, rank: int, ddp: bool, use_zero1: bool, keep_last: int, is_final: bool = False,
    hardlink_retain: Optional[Path] = None,
) -> None:
    if ddp and use_zero1 and is_final:
        try:
            optimizer.consolidate_state_dict(to=0)
        except Exception as e:
            if rank == 0:
                print(f"[ckpt][warn] ZeRO-1 consolidation failed: {e}")
        dist.barrier()
    elif ddp:
        dist.barrier()

    if not _is_main(rank):
        return

    model_to_save = model.module if hasattr(model, "module") else model
    payload = {
        "step": int(step),
        "model": model_to_save.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stats": stats,
        "args": vars(args),
        "is_final": bool(is_final),
    }
    try:
        payload["optimizer"] = optimizer.state_dict()
    except Exception as e:
        payload["optimizer_save_error"] = str(e)

    ckpt_path = out_dir / f"checkpoint_step_{int(step):08d}.pt"
    tmp_path  = out_dir / f".checkpoint_step_{int(step):08d}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)

    if hardlink_retain is not None:
        hr = Path(hardlink_retain)
        hr.parent.mkdir(parents=True, exist_ok=True)
        if hr.exists():
            hr.unlink()
        try:
            os.link(ckpt_path, hr)
        except OSError:
            shutil.copy2(ckpt_path, hr)

    if int(keep_last) > 0:
        ckpts = sorted(out_dir.glob("checkpoint_step_*.pt"))
        for p in ckpts[:-int(keep_last)]:
            try:
                p.unlink()
            except Exception:
                pass


def _load_stage2_checkpoint(model: Showo2Qwen2_5, path: str, rank: int) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if _is_main(rank):
        print(f"[stage2-ckpt] loaded {path}  missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  first missing keys: {missing[:5]}")


# ---------------------------------------------------------------------------
# Loss helpers / LR scheduler
# ---------------------------------------------------------------------------

def _ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if (shift_labels != ignore_index).sum() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def _build_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def f(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0
    return LambdaLR(optimizer, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained",           type=str, default="showlab/show-o2-1.5B-HQ")
    ap.add_argument("--tokenizer-model",      type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--stage1-tokenizer-dir", type=str, default="")
    ap.add_argument("--stage1-icd-ckpt",      type=str, default="")
    ap.add_argument("--stage2-ckpt",          type=str, required=True,
                    help="Path to Stage 2 final checkpoint (carries diffusion LoRA weights).")
    ap.add_argument("--matching-pkl",    type=str, default="${MATCHING_PKL}")
    ap.add_argument("--jpg-root",        type=str, default="${MIMIC_CXR_JPG_ROOT}")
    ap.add_argument("--vae-pth",         type=str, required=True)
    ap.add_argument("--output-dir",      type=str, default=str(Path(__file__).resolve().parent / "outputs_stage3"))
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--train-ratio",     type=float, default=0.8)
    ap.add_argument("--val-ratio",       type=float, default=0.1)
    ap.add_argument("--k-max",           type=int,   default=4)
    ap.add_argument("--max-seq-len",     type=int,   default=3072)
    ap.add_argument("--keep-last-n-ctx-images", type=int, default=1)
    ap.add_argument("--report-max-tokens",      type=int, default=192)
    ap.add_argument("--num-image-tokens",       type=int, default=1024)
    ap.add_argument("--latent-h",        type=int, default=32)
    ap.add_argument("--latent-w",        type=int, default=32)
    ap.add_argument("--image-resolution",type=int, default=512)
    ap.add_argument("--batch-size",      type=int, default=1)
    ap.add_argument("--num-workers",     type=int, default=0)
    ap.add_argument("--patient-balance-alpha",  type=float, default=0.5)
    ap.add_argument("--modality-image-weight",  type=float, default=5.0,
                    help="Upsample windows with target CXR image by this factor.")
    ap.add_argument("--report-loss-weight",     type=float, default=5.0)
    ap.add_argument("--icd-loss-weight",        type=float, default=0.5)
    ap.add_argument("--icd-ramp-steps",         type=int,   default=0,
                    help="If >0, linearly ramp ICD loss weight from 0 to --icd-loss-weight "
                         "over this many steps after the report-only warm-up. Run H uses 0.")
    ap.add_argument("--phase1-steps",           type=int,   default=0,
                    help="Report-only warm-up phase. Only report-containing windows are sampled "
                         "and ICD loss weight is forced to 0. Set 0 to disable.")
    # LLM LoRA config
    ap.add_argument("--lora-r",        type=int,   default=64)
    ap.add_argument("--lora-alpha",    type=int,   default=128)
    ap.add_argument("--lora-dropout",  type=float, default=0.0)
    # Diffusion LoRA config (must match Stage 2)
    ap.add_argument("--lora-diff-r",   type=int,   default=16)
    ap.add_argument("--lora-diff-alpha", type=int, default=32)
    ap.add_argument("--lr-llm-lora",   type=float, default=2e-4)
    ap.add_argument("--lr-diff-lora",  type=float, default=5e-6)
    # Training schedule
    ap.add_argument("--warmup-ratio",  type=float, default=0.05)
    ap.add_argument("--max-steps",     type=int,   default=10000)
    ap.add_argument("--save-every",    type=int,   default=5000)
    ap.add_argument("--keep-last",     type=int,   default=3)
    ap.add_argument("--save-final",    action="store_true", default=True)
    ap.add_argument("--resume-from",   type=str,   default="")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=False)
    ap.add_argument("--sharding",      type=str,   default="none", choices=["none", "zero1"])
    ap.add_argument("--mixed-precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--distributed",   action="store_true", default=False)
    args = ap.parse_args()

    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be >= 1.")

    _seed_all(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddp, rank, world_size, local_rank = _init_distributed(bool(args.distributed))
    use_zero1 = bool(args.sharding == "zero1")
    if use_zero1 and not ddp:
        raise ValueError("--sharding zero1 requires --distributed.")

    device = torch.device(f"cuda:{local_rank}") if ddp else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.mixed_precision]
    if _is_main(rank):
        print(f"[device] {device} mixed_precision={args.mixed_precision} ddp={ddp} world_size={world_size}")

    # ---- Tokenizer ----
    tok_source = args.stage1_tokenizer_dir.strip() if args.stage1_tokenizer_dir else ""
    tokenizer = AutoTokenizer.from_pretrained(tok_source or args.tokenizer_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tok_meta = add_stage2_special_tokens(tokenizer)
    if _is_main(rank):
        save_json(out_dir / "tokenizer_meta.json", tok_meta)
        tokenizer.save_pretrained(out_dir / "tokenizer")

    # ---- Data ----
    if _is_main(rank):
        print("[data] loading timelines ...")
    timelines = load_patient_timelines_from_matching_pkl(
        Path(args.matching_pkl), jpg_root=Path(args.jpg_root),
    )
    splits = split_subjects(
        subject_ids=sorted(timelines.keys()),
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
    )
    if _is_main(rank):
        save_json(out_dir / "subject_splits.json", splits)
    tr = {sid: timelines[sid] for sid in splits["train"]}

    ds = Stage3ChainWindowDataset(
        timelines=tr,
        tokenizer=tokenizer,
        split_name="train",
        k_max=args.k_max,
        max_seq_len=args.max_seq_len,
        keep_last_n_ctx_images=args.keep_last_n_ctx_images,
        report_max_tokens=args.report_max_tokens,
        num_image_tokens=args.num_image_tokens,
        add_time_embeds=True,
        seed=args.seed,
    )
    window_counts = ds.window_counts()
    if _is_main(rank):
        print("[dataset] window_counts:", window_counts)
        save_json(out_dir / "window_counts.json", window_counts)
        n_img = window_counts.get("image", 0)
        n_tot = window_counts["windows"]
        n_other = n_tot - n_img
        eff_img  = n_img * args.modality_image_weight
        eff_rest = n_other * 1.0
        eff_total = max(1.0, eff_img + eff_rest)
        print(f"[sampling] image fraction (unweighted): {n_img/max(1,n_tot):.1%}  "
              f"-> effective after {args.modality_image_weight}x upsampling: {eff_img/eff_total:.1%}")

    def _build_loader(report_only: bool, seed_offset: int = 0):
        w = ds.sample_weights(
            patient_balance_alpha=args.patient_balance_alpha,
            modality_image_weight=args.modality_image_weight,
            report_only=report_only,
        )
        n_pos = sum(1 for x in w if float(x) > 0.0)
        if _is_main(rank):
            print(f"[sampling] report_only={bool(report_only)} positive_windows={n_pos}")
        gen = torch.Generator()
        gen.manual_seed(int(args.seed) + int(rank) * 1000 + seed_offset)
        samp = WeightedRandomSampler(
            weights=torch.tensor(w, dtype=torch.double),
            num_samples=max(1, len(ds)),
            replacement=True,
            generator=gen,
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=samp,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_stage3_windows(b, pad_token_id=int(tokenizer.pad_token_id)),
            pin_memory=torch.cuda.is_available(),
        )

    def _infinite(dl: DataLoader) -> Iterator[dict]:
        while True:
            for batch in dl:
                yield batch

    # ---- Model ----
    if _is_main(rank):
        print("[model] loading Showo2 ...")
    model = Showo2Qwen2_5.from_pretrained(args.pretrained, use_safetensors=False).to(device)
    model.showo.resize_token_embeddings(len(tokenizer))
    model.showo.tie_weights()
    model.config.llm_vocab_size = int(len(tokenizer))
    model.config.image_latent_height = int(args.latent_h)
    model.config.image_latent_width  = int(args.latent_w)
    model.image_position_ids = torch.arange(
        int(args.latent_h) * int(args.latent_w), device=device
    ).expand((1, -1))

    if args.stage1_icd_ckpt:
        s1_meta = _load_stage1_embedding_rows(model, args.stage1_icd_ckpt)
        if _is_main(rank):
            print("[stage1]", json.dumps(s1_meta, indent=2))

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    # Inject diffusion LoRA (architecture must match before loading S2 weights).
    diff_lora_meta = _apply_lora_diffusion_head(
        model, r=int(args.lora_diff_r), alpha=int(args.lora_diff_alpha), dropout=float(args.lora_dropout),
    )

    # Load Stage 2 checkpoint -- carries the trained diffusion LoRA weights.
    _load_stage2_checkpoint(model, args.stage2_ckpt, rank)

    # Inject LLM LoRA on top (fresh -- not in S2 checkpoint).
    llm_lora_meta = _apply_lora_upper8(
        model, r=int(args.lora_r), alpha=int(args.lora_alpha), dropout=float(args.lora_dropout),
    )

    # Mark trainable: diffusion LoRA (low LR), diffusion adaLN, LLM LoRA.
    for n, p in model.named_parameters():
        if "diffusion_head_a" in n and (".lora_A." in n or ".lora_B." in n):
            p.requires_grad = True
        elif "diffusion_head_a" in n and "adaLN_modulation" in n:
            p.requires_grad = True
        elif "diffusion_head_a" not in n and (".lora_A." in n or ".lora_B." in n):
            p.requires_grad = True

    # Keep non-LoRA diffusion projections trainable (time_embed etc).
    for name in ("diffusion_head_b", "time_embed", "time_embed_proj", "diff_proj", "fusion_proj"):
        mod = getattr(model, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if _is_main(rank):
        print(f"[lora-llm]  {json.dumps(llm_lora_meta, indent=2)}")
        print(f"[lora-diff] {json.dumps(diff_lora_meta, indent=2)}")
        print(f"[freeze] total={total_params:,}  trainable={trainable_params:,}")
        save_json(out_dir / "lora_meta.json", {
            "llm": llm_lora_meta,
            "diff": diff_lora_meta,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "report_loss_weight": float(args.report_loss_weight),
            "icd_loss_weight": float(args.icd_loss_weight),
            "icd_ramp_steps": int(args.icd_ramp_steps),
            "phase1_steps": int(args.phase1_steps),
        })

    if bool(args.gradient_checkpointing):
        if hasattr(model.showo, "gradient_checkpointing_enable"):
            try:
                model.showo.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.showo.gradient_checkpointing_enable()
        if hasattr(model.showo, "config"):
            model.showo.config.use_cache = False

    # ---- Optimizer param groups ----
    diff_lora_params, llm_lora_params, other_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "diffusion_head_a" in n and (".lora_A." in n or ".lora_B." in n):
            diff_lora_params.append(p)
        elif ".lora_A." in n or ".lora_B." in n:
            llm_lora_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {"params": llm_lora_params,  "lr": float(args.lr_llm_lora),  "weight_decay": 0.0},
        {"params": diff_lora_params, "lr": float(args.lr_diff_lora), "weight_decay": 0.0},
        {"params": other_params,     "lr": float(args.lr_diff_lora), "weight_decay": 0.0},
    ]
    if use_zero1:
        optimizer = ZeroRedundancyOptimizer(param_groups, optimizer_class=AdamW)
    else:
        optimizer = AdamW(param_groups)

    scheduler = _build_lr_scheduler(optimizer, total_steps=max(1, args.max_steps), warmup_ratio=args.warmup_ratio)

    # ---- Resume ----
    resume_step = 0
    stats = {
        "steps": 0,
        "loss_running": {"image": None, "report": None, "icd": None, "total": None},
        "modality_steps": {"image": 0, "report": 0, "icd": 0},
    }
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        raw_state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(raw_state, strict=False)
        if _is_main(rank):
            print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
        if "optimizer" in ckpt:
            saved_opt = ckpt["optimizer"]
            saved_group_sizes = [len(g.get("params", [])) for g in saved_opt.get("param_groups", [])]
            current_group_sizes = [len(g["params"]) for g in param_groups]
            if saved_group_sizes != current_group_sizes:
                if _is_main(rank):
                    print("[resume] skipping optimizer state "
                          f"(param-group mismatch saved={saved_group_sizes} current={current_group_sizes})")
            else:
                try:
                    optimizer.load_state_dict(saved_opt)
                except Exception as e:
                    if _is_main(rank):
                        print(f"[resume][warn] optimizer restore failed: {e}")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                if _is_main(rank):
                    print(f"[resume][warn] scheduler restore failed: {e}")
        stats = ckpt.get("stats", stats)
        stats.setdefault("loss_running", {})
        stats.setdefault("modality_steps", {})
        for key in ("image", "report", "icd", "total"):
            stats["loss_running"].setdefault(key, None)
        for key in ("image", "report", "icd"):
            stats["modality_steps"].setdefault(key, 0)
        resume_step = int(ckpt.get("step", 0))

    # ---- Curriculum tracking ----
    _use_curriculum = int(args.phase1_steps) > 0
    _phase1_end = resume_step + int(args.phase1_steps)
    _icd_ramp = int(args.icd_ramp_steps)
    _phase2_end = _phase1_end + _icd_ramp
    _in_phase1 = _use_curriculum and (resume_step < _phase1_end)

    loader = _build_loader(report_only=_in_phase1, seed_offset=0)
    data_iter = _infinite(loader)

    if _is_main(rank) and _use_curriculum:
        print(f"[curriculum] Phase 1 (report-only): steps {resume_step+1}-{_phase1_end}")
        if _icd_ramp > 0:
            print(f"[curriculum] Phase 2 (ICD ramp):    steps {_phase1_end+1}-{_phase2_end}")
            print(f"[curriculum] Phase 3 (balanced):    steps {_phase2_end+1}-{args.max_steps}")
        else:
            print(f"[curriculum] Phase 2 (balanced):    steps {_phase1_end+1}-{args.max_steps}")

    # ---- VAE + transport ----
    if _is_main(rank):
        print("[vae] loading WanVAE ...")
    vae = WanVAE(vae_pth=str(args.vae_pth), dtype=weight_type, device=device)
    transport = create_transport(
        path_type="Linear", prediction="velocity",
        loss_weight=None, train_eps=None, sample_eps=None,
        snr_type="lognorm", do_shift=True,
        seq_len=16,  # matches x.shape[1] = latent channels at inference
    )

    # ---- DDP wrap ----
    if ddp:
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
    model.train()

    sync_param = next((p for p in model.parameters() if p.requires_grad), None)

    img_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    pad_id     = int(tokenizer.pad_token_id)

    report_loss_weight = float(args.report_loss_weight)
    icd_loss_weight_target = float(args.icd_loss_weight)
    _last_rep_ce: List[float] = [0.0]
    _last_icd_ce: List[float] = [0.0]

    def _get_icd_weight(step: int) -> float:
        """ICD loss weight schedule: 0 during phase 1; linear ramp; full target afterwards."""
        if _use_curriculum and step <= _phase1_end:
            return 0.0
        if _icd_ramp > 0:
            elapsed = max(0, step - _phase1_end)
            return icd_loss_weight_target * min(1.0, elapsed / _icd_ramp)
        return icd_loss_weight_target

    def _split_ce_loss(logits: torch.Tensor, labels: torch.Tensor,
                       input_ids_b: torch.Tensor, cur_icd_w: float,
                       enable_report: bool, enable_icd: bool) -> torch.Tensor:
        """Weighted combined CE: report-only labels and ICD-only labels reduced separately."""

        def _safe_ce(lg, lb):
            shift_lg = lg[..., :-1, :].contiguous()
            shift_lb = lb[..., 1:].contiguous()
            if not (shift_lb.view(-1) != -100).any():
                return torch.zeros((), device=lg.device, dtype=torch.float32)
            return F.cross_entropy(shift_lg.view(-1, shift_lg.size(-1)),
                                   shift_lb.view(-1), ignore_index=-100)

        bsz = int(labels.size(0))
        loss_rep_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
        loss_icd_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
        for b in range(bsz):
            rep_labels = mask_all_code_spans(labels[b], input_ids_b[b], tokenizer).unsqueeze(0)
            icd_labels = mask_report_span(labels[b], input_ids_b[b], tokenizer).unsqueeze(0)
            loss_rep_sum = loss_rep_sum + _safe_ce(logits[b : b + 1], rep_labels)
            loss_icd_sum = loss_icd_sum + _safe_ce(logits[b : b + 1], icd_labels)
        denom = max(1, bsz)
        loss_rep = loss_rep_sum / denom
        loss_icd = loss_icd_sum / denom
        _last_rep_ce[0] = float(loss_rep.detach().cpu())
        _last_icd_ce[0] = float(loss_icd.detach().cpu())
        rep_w = report_loss_weight if enable_report else 0.0
        icd_w = cur_icd_w if enable_icd else 0.0
        return rep_w * loss_rep + icd_w * loss_icd

    # ---- Training loop ----
    pbar = tqdm(range(resume_step + 1, args.max_steps + 1),
                desc="stage3", disable=not _is_main(rank))
    last_step = resume_step

    for step in pbar:
        last_step = step

        # ---- Phase 1 -> Phase 2 transition: rebuild loader without report-only filter ----
        if _in_phase1 and step > _phase1_end:
            _in_phase1 = False
            loader = _build_loader(report_only=False, seed_offset=1)
            data_iter = _infinite(loader)
            if _is_main(rank):
                print(f"\n[curriculum] Phase 1 -> Phase 2 at step {step}.")
            retain_path = out_dir / "checkpoint_after_report_warmup.pt"
            p1_ckpt = out_dir / f"checkpoint_step_{int(_phase1_end):08d}.pt"
            if ddp:
                dist.barrier()
            need_write = False
            if _is_main(rank):
                if p1_ckpt.is_file():
                    if retain_path.exists():
                        retain_path.unlink()
                    try:
                        os.link(p1_ckpt, retain_path)
                    except OSError:
                        shutil.copy2(p1_ckpt, retain_path)
                    print(f"[ckpt] report-warmup snapshot linked -> {retain_path}")
                else:
                    need_write = True
                    print(f"[ckpt] phase-1 step file missing; saving step {int(_phase1_end)} + snapshot")
            if ddp:
                flag = torch.tensor([1 if need_write else 0], device=device, dtype=torch.int32)
                dist.broadcast(flag, src=0)
                need_write = bool(flag.item())
            if need_write:
                _save_checkpoint(
                    out_dir=out_dir, step=int(_phase1_end), model=model, optimizer=optimizer,
                    scheduler=scheduler, stats=stats, args=args, rank=rank,
                    ddp=ddp, use_zero1=use_zero1, keep_last=0, is_final=False,
                    hardlink_retain=retain_path,
                )
            if ddp:
                dist.barrier()

        batch = next(data_iter)

        has_img    = bool(batch["has_target_image"].any().item())
        has_report = bool(batch["has_target_report"].any().item())
        has_icd    = bool(batch["has_target_icd"].any().item())

        input_ids         = batch["input_ids"].to(device)
        labels            = batch["labels"].to(device)
        attention_mask_1d = batch["attention_mask"].to(device)
        bsz = int(input_ids.size(0))

        pixel_values, modality_positions, text_masks, image_masks = _collect_image_slots(
            batch, device=device, resolution=int(args.image_resolution),
            img_pad_id=img_pad_id, pad_id=pad_id,
        )

        if modality_positions is not None:
            attn = _build_omni_padding_mask(attention_mask_1d, modality_positions)
        else:
            attn = _build_causal_padding_mask(attention_mask_1d)

        optimizer.zero_grad(set_to_none=True)

        # ---- Prepare image latents ----
        xt, ut, t_vec = None, None, None
        sup = None
        if pixel_values is not None and modality_positions is not None:
            with torch.no_grad():
                x1 = vae.sample(pixel_values.unsqueeze(2)).squeeze(2)
            sup = batch["image_supervise_masks"].to(device=device, dtype=torch.long)
            xt   = x1.clone()
            ut   = torch.zeros_like(x1)
            t_vec = torch.ones((x1.size(0),), device=device, dtype=torch.float32)

            for i in range(x1.size(0)):
                bi, sj = i // int(sup.size(1)), i % int(sup.size(1))
                if int(sup[bi, sj].item()) != 1:
                    continue  # context slot -- keep clean
                t_i, x0, x1_i = transport.sample(x1[i][None])
                t_i, xt_i, ut_i = transport.path_sampler.plan(t_i, x0, x1_i)
                t_vec[i] = t_i.squeeze(0)
                xt[i]    = xt_i.squeeze(0)
                ut[i]    = ut_i.squeeze(0)

            xt    = xt.to(dtype=weight_type)
            ut    = ut.to(dtype=weight_type)
            t_vec = t_vec.to(device=device, dtype=weight_type)

            for bi in range(bsz):
                for j in range(int(sup.size(1))):
                    off, ln = int(modality_positions[bi, j, 0].item()), int(modality_positions[bi, j, 1].item())
                    if ln <= 0:
                        continue
                    if int(sup[bi, j].item()) == 0:
                        image_masks[bi, off : off + ln] = 0

        need_report_loss = has_report
        need_icd_loss = has_icd
        need_text_loss = need_report_loss or need_icd_loss
        need_image_loss = has_img and (xt is not None)

        text_labels_arg  = labels if need_text_loss  else None
        image_labels_arg = ut     if need_image_loss else None

        loss_img  = torch.zeros((), device=device, dtype=torch.float32)
        loss_text = torch.zeros((), device=device, dtype=torch.float32)
        cur_icd_w = _get_icd_weight(step)

        if xt is not None:
            forward_out = model(
                text_tokens=input_ids, image_latents=xt, t=t_vec,
                attention_mask=attn, text_masks=text_masks, image_masks=image_masks,
                text_labels=text_labels_arg, image_labels=image_labels_arg,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=input_ids.size(1), device=device,
            )
            if need_image_loss and need_text_loss:
                _logits, _combined_text_loss, loss_img = forward_out
            elif need_image_loss:
                _logits, loss_img = forward_out
            elif need_text_loss:
                _logits, _combined_text_loss = forward_out
            if need_text_loss:
                loss_text = _split_ce_loss(
                    _logits, labels, input_ids, cur_icd_w,
                    enable_report=need_report_loss, enable_icd=need_icd_loss,
                )
        else:
            out = model(
                text_tokens=input_ids, image_latents=None,
                attention_mask=attn,
                output_hidden_states=False,
                max_seq_len=input_ids.size(1), device=device,
            )
            raw_logits = out.logits if hasattr(out, "logits") else out["logits"]
            if need_text_loss:
                loss_text = _split_ce_loss(
                    raw_logits, labels, input_ids, cur_icd_w,
                    enable_report=need_report_loss, enable_icd=need_icd_loss,
                )

        if not need_text_loss and not need_image_loss:
            loss_text = sync_param.sum() * 0.0 if sync_param is not None else \
                        torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

        loss = loss_text + loss_img

        if not torch.isfinite(loss):
            if _is_main(rank):
                print(f"[warn] non-finite loss={float(loss.detach().cpu()):.4f} step={step} -- skipping")
            if sync_param is not None:
                noop = sync_param.sum() * 0.0
                noop.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            continue

        if not bool(loss.requires_grad):
            loss = sync_param.sum() * 0.0 if sync_param is not None else \
                   torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        # ---- Stats ----
        total_v = float(loss.detach().cpu())
        img_v   = float(loss_img.detach().cpu())
        rep_v_stat = float(_last_rep_ce[0])
        icd_v_stat = float(_last_icd_ce[0])

        def _ema(prev, cur, alpha=0.98):
            return cur if prev is None else (alpha * prev + (1 - alpha) * cur)

        stats["steps"] = step
        stats["loss_running"]["total"]  = _ema(stats["loss_running"]["total"],  total_v)
        if need_image_loss:
            stats["loss_running"]["image"]  = _ema(stats["loss_running"]["image"],  img_v)
            stats["modality_steps"]["image"] += int(batch["has_target_image"].sum().item())
        if need_report_loss:
            stats["loss_running"]["report"] = _ema(stats["loss_running"]["report"], rep_v_stat)
            stats["modality_steps"]["report"] += int(batch["has_target_report"].sum().item())
        if need_icd_loss:
            stats["loss_running"]["icd"]    = _ema(stats["loss_running"]["icd"],    icd_v_stat)
            stats["modality_steps"]["icd"] += int(batch["has_target_icd"].sum().item())

        if _is_main(rank):
            postfix = {
                "total": f"{(stats['loss_running']['total'] or 0):.3f}",
                "img":   f"{(stats['loss_running']['image']  or 0):.3f}",
                "rep":   f"{(stats['loss_running']['report'] or 0):.3f}",
                "icd":   f"{(stats['loss_running']['icd']    or 0):.3f}",
                "img_n": stats["modality_steps"]["image"],
            }
            if _use_curriculum or _icd_ramp > 0:
                postfix["icd_w"] = f"{cur_icd_w:.3f}"
            if _use_curriculum:
                if step <= _phase1_end:
                    postfix["phase"] = "1:warmup"
                elif step <= _phase2_end:
                    postfix["phase"] = "2:ramp" if _icd_ramp > 0 else "2:balanced"
                else:
                    postfix["phase"] = "3:balanced"
            pbar.set_postfix(postfix)

        if int(args.save_every) > 0 and step % int(args.save_every) == 0:
            _save_checkpoint(
                out_dir=out_dir, step=step, model=model, optimizer=optimizer,
                scheduler=scheduler, stats=stats, args=args, rank=rank,
                ddp=ddp, use_zero1=use_zero1, keep_last=int(args.keep_last), is_final=False,
            )

    if bool(args.save_final):
        _save_checkpoint(
            out_dir=out_dir, step=last_step, model=model, optimizer=optimizer,
            scheduler=scheduler, stats=stats, args=args, rank=rank,
            ddp=ddp, use_zero1=use_zero1, keep_last=int(args.keep_last), is_final=True,
        )

    if ddp:
        dist.barrier()
    if _is_main(rank):
        save_json(out_dir / "train_stats.json", stats)
        print(json.dumps(stats, indent=2))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
