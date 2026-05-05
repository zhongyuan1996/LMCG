#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import random
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from PIL import Image, ImageOps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
# wandb optional — not used in the released code path

from data_pipeline import (  # noqa: E402
    Stage2MultimodalWindowDataset,
    add_stage2_special_tokens,
    collate_stage2_windows,
    load_patient_timelines_from_matching_pkl,
    save_json,
    split_subjects,
    tokenizer_has_stage1_multicode_vocab,
)
from models import Showo2Qwen2_5, WanVAE  # noqa: E402
from models.omni_attention import omni_attn_mask_naive  # noqa: E402
from stage1_embedding_checkpoint import load_stage1_embedding_rows  # noqa: E402
from transport import create_transport  # noqa: E402
from utils.lora import LoraSpec, LoRALinear  # noqa: E402


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
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def _is_main(rank: int) -> bool:
    return int(rank) == 0


def _save_checkpoint(
    *,
    out_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    stats: Dict,
    args,
    rank: int,
    ddp: bool,
    use_zero1: bool,
    keep_last: int,
    is_final: bool = False,
) -> None:
    # ZeRO-1 consolidation gathers all optimizer shards onto rank-0 before saving.
    # This is a multi-minute all-gather for large models and can spike memory enough
    # to trigger the OOM killer. Only do it on the final save where full resume
    # capability is needed. Periodic saves skip consolidation and save the local
    # rank-0 shard only (sufficient for loss monitoring / model-only resumption).
    if ddp and use_zero1 and is_final:
        try:
            optimizer.consolidate_state_dict(to=0)
        except Exception as e:
            if rank == 0:
                print(f"[ckpt][warn] ZeRO-1 consolidation failed: {e}. Saving without optimizer state.")
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
    tmp_path = out_dir / f".checkpoint_step_{int(step):08d}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)

    if int(keep_last) > 0:
        ckpts = sorted(out_dir.glob("checkpoint_step_*.pt"))
        stale = ckpts[:-int(keep_last)]
        for p in stale:
            try:
                p.unlink()
            except Exception:
                pass


def _build_causal_padding_mask(attn_1d: torch.Tensor) -> torch.Tensor:
    bsz, seqlen = attn_1d.shape
    dev = attn_1d.device
    valid = attn_1d.bool()
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=dev))
    keep = causal.unsqueeze(0) & valid.unsqueeze(1)
    mask = torch.zeros((bsz, 1, seqlen, seqlen), dtype=torch.float32, device=dev)
    mask = mask.masked_fill(~keep.unsqueeze(1), -1e4)
    return mask


def _build_omni_padding_mask(attn_1d: torch.Tensor, modality_positions: torch.Tensor) -> torch.Tensor:
    """
    Omni attention with padding for Stage-2 variable-length sequences:
      - causal by default
      - full attention inside each image span (offset:length)
      - padding positions masked out
    """
    bsz, seqlen = attn_1d.shape
    dev = attn_1d.device
    valid = attn_1d.bool()
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=dev))
    keep = causal.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)  # [B,L,L]

    if modality_positions is not None and modality_positions.numel() > 0:
        mp = modality_positions.to(device=dev)
        for b in range(bsz):
            for off, ln in mp[b]:
                off_i = int(off.item())
                ln_i = int(ln.item())
                if ln_i <= 0:
                    continue
                s = max(0, off_i)
                e = min(seqlen, off_i + ln_i)
                if e <= s:
                    continue
                keep[b, s:e, s:e] = True

    mask = torch.zeros((bsz, 1, seqlen, seqlen), dtype=torch.float32, device=dev)
    mask = mask.masked_fill(~keep.unsqueeze(1), -1e4)
    return mask


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


def _parse_ratio(ratio: str) -> List[str]:
    # e.g. "4:2:1" -> [icd,icd,icd,icd,report,report,image]
    a, b, c = [max(0, int(x)) for x in ratio.split(":")]
    sched = (["icd"] * a) + (["report"] * b) + (["image"] * c)
    if not sched:
        raise ValueError(f"invalid empty ratio: {ratio}")
    return sched


def _infinite(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


def _preprocess_image(path: str, resolution: int) -> torch.Tensor:
    if not str(path).strip():
        return torch.zeros((3, resolution, resolution), dtype=torch.float32)
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("L").convert("RGB")
        w, h = img.size
        scale = float(resolution) / float(min(w, h))
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        img = img.resize((nw, nh), resample=Image.BICUBIC)
        left = max(0, (nw - resolution) // 2)
        top = max(0, (nh - resolution) // 2)
        img = img.crop((left, top, left + resolution, top + resolution))
        arr = np.asarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        x = x * 2.0 - 1.0
        return x
    except Exception:
        return torch.zeros((3, resolution, resolution), dtype=torch.float32)


def _collect_image_slots(
    batch: dict,
    *,
    device: torch.device,
    resolution: int,
    img_pad_id: int,
    pad_id: int,
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    Returns:
      pixel_values_slots: [B*M,3,H,W] or None
      modality_positions: [B, M, 2] long on device, or None
      text_masks: [B, L]
      image_masks: [B, L] (context slots can be zeroed later)
    """
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
        raise ValueError("collate_stage2_windows must pad modality_positions to a fixed M per batch")
    if not all(len(row) == m_slots for row in paths):
        raise ValueError("collate_stage2_windows must pad image_paths_for_slots to match modality slots")

    pos_t = torch.zeros(bsz, m_slots, 2, dtype=torch.long, device=device)
    for i in range(bsz):
        for j in range(m_slots):
            off, ln = modality_positions[i][j]
            pos_t[i, j, 0] = int(off)
            pos_t[i, j, 1] = int(ln)

    rows: List[torch.Tensor] = []
    for i in range(bsz):
        for j in range(m_slots):
            rows.append(_preprocess_image(paths[i][j], resolution=resolution))
    pixel_values = torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)

    text_masks = ((input_ids != img_pad_id) & (input_ids != pad_id)).long()
    image_masks = (input_ids == img_pad_id).long()
    return pixel_values, pos_t, text_masks, image_masks


def _set_trainability(model, *, train_text_embeddings: bool) -> Dict[str, Dict[str, int]]:
    for p in model.parameters():
        p.requires_grad = False

    # Always train diffusion-side heads/projections in Stage-2 baseline.
    for name in ("diffusion_head_a", "diffusion_head_b", "time_embed", "time_embed_proj", "diff_proj", "fusion_proj"):
        mod = getattr(model, name, None)
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad = True

    # Keep text-side learnability minimal in frozen-LLM arm.
    if train_text_embeddings:
        emb = model.showo.get_input_embeddings()
        emb.weight.requires_grad = True

    return _collect_trainability_manifest(model)


def _collect_trainability_manifest(model) -> Dict[str, Dict[str, int]]:
    groups_local = defaultdict(lambda: {"params": 0, "trainable": 0})

    def _count_local(group: str, module: torch.nn.Module):
        n = sum(p.numel() for p in module.parameters())
        t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        groups_local[group]["params"] += int(n)
        groups_local[group]["trainable"] += int(t)

    _count_local("showo", model.showo)
    _count_local("image_embedder_und", model.image_embedder_und)
    _count_local("image_embedder_gen", model.image_embedder_gen)
    _count_local("und_trans", model.und_trans)
    if hasattr(model, "position_embedding"):
        _count_local("position_embedding", model.position_embedding)
    for g in ("diffusion_head_a", "diffusion_head_b", "time_embed", "time_embed_proj", "diff_proj", "fusion_proj"):
        mod = getattr(model, g, None)
        if mod is not None:
            _count_local(g, mod)
    total_local = sum(p.numel() for p in model.parameters())
    trainable_local = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "summary": {"total_params": int(total_local), "trainable_params": int(trainable_local)},
        "groups": dict(groups_local),
    }


def _apply_lora_upper8(model: Showo2Qwen2_5, *, r: int, alpha: int, dropout: float) -> Dict[str, int]:
    """
    Inject LoRA into upper 8 Qwen decoder layers for linear projections.
    Target modules: self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj
    """
    spec = LoraSpec(r=int(r), alpha=int(alpha), dropout=float(dropout))
    wrapped = 0
    target_suffix = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }

    layers = getattr(model.showo.model, "layers", None)
    if layers is None:
        raise RuntimeError("Could not find showo.model.layers for LoRA injection")
    n_layers = len(layers)
    start = max(0, n_layers - 8)
    for li in range(start, n_layers):
        layer = layers[li]
        for name, child in list(layer.named_modules()):
            if name not in target_suffix:
                continue
            if not isinstance(child, nn.Linear):
                continue
            # replace module by traversing parent path
            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(child, spec))
            wrapped += 1

    total_lora = sum(p.numel() for n, p in model.named_parameters() if (".lora_A." in n or ".lora_B." in n))
    trainable_lora = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and (".lora_A." in n or ".lora_B." in n)
    )
    return {"wrapped_linear_modules": int(wrapped), "lora_params": int(total_lora), "lora_trainable": int(trainable_lora)}


def _apply_lora_diffusion_head(model: Showo2Qwen2_5, *, r: int, alpha: int, dropout: float) -> Dict[str, int]:
    """
    Inject LoRA into every ModulatedAttentionBlock inside diffusion_head_a.

    Targeted linear suffixes (same structure as LLM blocks):
        self_attn.{q,k,v,o}_proj  and  mlp.{gate,up,down}_proj

    adaLN_modulation is intentionally left as a plain nn.Linear so it stays
    fully trainable — it is zero-initialised and must move freely to learn
    time-step conditioning.  The base weights of the wrapped linears are
    frozen automatically by LoRALinear.__init__.
    """
    spec = LoraSpec(r=int(r), alpha=int(alpha), dropout=float(dropout))
    wrapped = 0
    target_suffix = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }

    for layer in model.diffusion_head_a:
        for name, child in list(layer.named_modules()):
            if name not in target_suffix:
                continue
            if not isinstance(child, nn.Linear):
                continue
            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(child, spec))
            wrapped += 1

    total_lora = sum(
        p.numel() for n, p in model.named_parameters()
        if "diffusion_head_a" in n and (".lora_A." in n or ".lora_B." in n)
    )
    trainable_lora = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "diffusion_head_a" in n and (".lora_A." in n or ".lora_B." in n)
    )
    return {"wrapped_linear_modules": int(wrapped), "lora_diff_params": int(total_lora), "lora_diff_trainable": int(trainable_lora)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, default="showlab/show-o2-1.5B-HQ")
    ap.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument(
        "--stage1-tokenizer-dir",
        type=str,
        default="",
        help="Directory with tokenizer/ from Stage 1 (multicode). Strongly recommended before training.",
    )
    ap.add_argument(
        "--stage1-icd-ckpt",
        type=str,
        default="",
        help="Stage-1 embedding bundle (*_stage1_code_embedding_rows.pt, *_icd_*, or latest_train_state.pt).",
    )
    ap.add_argument("--matching-pkl", type=str, default="${MATCHING_PKL}")
    ap.add_argument("--jpg-root", type=str, default="${MIMIC_CXR_JPG_ROOT}")
    ap.add_argument("--vae-pth", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs_stage2_baseline"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--keep-last-n-ctx-images", type=int, default=1)
    ap.add_argument("--report-max-tokens", type=int, default=192)
    ap.add_argument("--num-image-tokens", type=int, default=1024)
    ap.add_argument("--latent-h", type=int, default=32)
    ap.add_argument("--latent-w", type=int, default=32)
    ap.add_argument("--image-resolution", type=int, default=512)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-rank microbatch size. Increase (e.g. 2–4) if VRAM allows; collator pads image slots to max M in batch.",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--patient-balance-alpha", type=float, default=0.5)
    ap.add_argument("--sampling-policy", type=str, default="fixed", choices=["fixed", "capped_inverse"])
    ap.add_argument("--fixed-ratio", type=str, default="4:2:1")
    ap.add_argument("--task-weight-icd", type=float, default=4.0)
    ap.add_argument("--task-weight-report", type=float, default=2.0)
    ap.add_argument("--task-weight-image", type=float, default=1.0)
    ap.add_argument("--lr-text", type=float, default=5e-5)
    ap.add_argument("--lr-diffusion", type=float, default=5e-5)
    ap.add_argument("--llm-arm", type=str, default="frozen", choices=["frozen", "lora_upper8"])
    ap.add_argument("--arm-a-mode", type=str, default="lite", choices=["strict", "lite"])
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-diffusion-head", action="store_true", default=False,
                    help="Arm C: inject LoRA into diffusion_head_a to prevent catastrophic forgetting.")
    ap.add_argument("--lora-diff-r", type=int, default=16,
                    help="LoRA rank for diffusion_head_a (default 16, higher than LLM r=8).")
    ap.add_argument("--lora-diff-alpha", type=int, default=32,
                    help="LoRA alpha for diffusion_head_a (default 32).")
    ap.add_argument("--warmup-ratio", type=float, default=0.08)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N steps; 0 disables periodic saves.")
    ap.add_argument("--keep-last", type=int, default=3, help="Keep only latest K periodic checkpoints.")
    _sf = ap.add_mutually_exclusive_group()
    _sf.add_argument("--save-final", dest="save_final", action="store_true", help="Save a final checkpoint when training ends (default).")
    _sf.add_argument(
        "--no-save-final",
        dest="save_final",
        action="store_false",
        help="Skip final checkpoint (faster smoke tests).",
    )
    ap.set_defaults(save_final=True)
    ap.add_argument("--resume-from", type=str, default="", help="Resume from checkpoint path.")
    ap.add_argument("--train-text-embeddings", action="store_true", default=False)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=False)
    ap.add_argument("--sharding", type=str, default="none", choices=["none", "fsdp", "zero1"])
    ap.add_argument("--mixed-precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--distributed", action="store_true", default=False)
    args = ap.parse_args()

    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be >= 1")

    _seed_all(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddp, rank, world_size, local_rank = _init_distributed(bool(args.distributed))
    use_fsdp = bool(args.sharding == "fsdp")
    use_zero1 = bool(args.sharding == "zero1")
    if use_fsdp and (not ddp):
        raise ValueError("--sharding fsdp requires distributed launch (torchrun + --distributed).")
    if use_zero1 and (not ddp):
        raise ValueError("--sharding zero1 requires distributed launch (torchrun + --distributed).")

    if ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mixed_precision == "bf16":
        weight_type = torch.bfloat16
    elif args.mixed_precision == "fp16":
        weight_type = torch.float16
    else:
        weight_type = torch.float32
    if _is_main(rank):
        print(f"[device] {device} mixed_precision={args.mixed_precision} ddp={ddp} world_size={world_size}")

    tok_source = args.stage1_tokenizer_dir.strip() if args.stage1_tokenizer_dir else ""
    if tok_source:
        tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tok_source and (not tokenizer_has_stage1_multicode_vocab(tokenizer)) and _is_main(rank):
        print(
            "[warn] --stage1-tokenizer-dir is set but <DIAG> is missing/unk; "
            "use the Stage-1 multicode tokenizer output."
        )
    tok_meta = add_stage2_special_tokens(tokenizer)
    if _is_main(rank):
        save_json(out_dir / "tokenizer_meta.json", tok_meta)
        tokenizer.save_pretrained(out_dir / "tokenizer")

    if _is_main(rank):
        print("[data] loading timelines ...")
    timelines = load_patient_timelines_from_matching_pkl(
        Path(args.matching_pkl),
        jpg_root=Path(args.jpg_root),
    )
    splits = split_subjects(
        subject_ids=sorted(timelines.keys()),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if _is_main(rank):
        save_json(out_dir / "subject_splits.json", splits)
    tr = {sid: timelines[sid] for sid in splits["train"]}

    task_weights = {"icd": args.task_weight_icd, "report": args.task_weight_report, "image": args.task_weight_image}
    ds = Stage2MultimodalWindowDataset(
        timelines=tr,
        tokenizer=tokenizer,
        split_name="train",
        k_max=args.k_max,
        max_seq_len=args.max_seq_len,
        keep_last_n_ctx_images=args.keep_last_n_ctx_images,
        report_max_tokens=args.report_max_tokens,
        num_image_tokens=args.num_image_tokens,
        add_time_embeds=True,
        task_sampling=args.sampling_policy,
        task_weights=task_weights,
        seed=args.seed,
    )
    task_counts = ds.task_window_counts()
    if _is_main(rank):
        print("[dataset] task_counts:", task_counts)
        save_json(out_dir / "task_counts.json", task_counts)

    base_weights = ds.sample_weights(patient_balance_alpha=args.patient_balance_alpha)
    task_idx = ds.task_indices()
    task_loaders = {}
    for task in ("icd", "report", "image"):
        idxs = task_idx[task]
        sub = Subset(ds, idxs)
        sub_w = [base_weights[i] for i in idxs]
        g = torch.Generator()
        g.manual_seed(int(args.seed) + int(rank) * 1000 + (0 if task == "icd" else (1 if task == "report" else 2)))
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sub_w, dtype=torch.double),
            num_samples=max(1, len(idxs)),
            replacement=True,
            generator=g,
        )
        task_loaders[task] = DataLoader(
            sub,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_stage2_windows(b, pad_token_id=int(tokenizer.pad_token_id)),
            pin_memory=torch.cuda.is_available(),
        )
    task_iters = {k: _infinite(v) for k, v in task_loaders.items()}

    if args.sampling_policy == "fixed":
        schedule = _parse_ratio(args.fixed_ratio)
    else:
        # capped inverse-availability
        cmin = max(1, min(task_counts.values()))
        inv = {k: min(4.0, float(cmin) / float(max(1, task_counts[k]))) for k in task_counts}
        # Keep some floor for ICD and report.
        inv["icd"] = max(inv["icd"], 1.0)
        inv["report"] = max(inv["report"], 1.0)
        inv["image"] = max(inv["image"], 1.0)
        schedule = []
        for k in ("icd", "report", "image"):
            schedule.extend([k] * max(1, int(round(inv[k] * 2))))
    if _is_main(rank):
        save_json(out_dir / "sampling_schedule.json", {"policy": args.sampling_policy, "schedule": schedule})
        print("[sampling] schedule cycle:", schedule)

    if _is_main(rank):
        print("[model] loading Showo2 ...")
    model = Showo2Qwen2_5.from_pretrained(args.pretrained, use_safetensors=False).to(device)
    model.showo.resize_token_embeddings(len(tokenizer))
    model.showo.tie_weights()
    model.config.llm_vocab_size = int(len(tokenizer))
    stage1_manifest = None
    if args.stage1_icd_ckpt:
        stage1_manifest = load_stage1_embedding_rows(model, args.stage1_icd_ckpt)
        if _is_main(rank):
            print("[stage1]", json.dumps(stage1_manifest, indent=2))
            save_json(out_dir / "stage1_embedding_load_manifest.json", stage1_manifest)
    # Align runtime latent grid with current training resolution.
    model.config.image_latent_height = int(args.latent_h)
    model.config.image_latent_width = int(args.latent_w)
    model.image_position_ids = torch.arange(
        int(args.latent_h) * int(args.latent_w),
        device=device,
    ).expand((1, -1))

    train_text_embeddings = bool(args.train_text_embeddings)
    if args.llm_arm == "frozen":
        train_text_embeddings = (args.arm_a_mode == "lite")
    freeze_manifest = _set_trainability(model, train_text_embeddings=train_text_embeddings)
    lora_manifest = None
    if args.llm_arm == "lora_upper8":
        lora_manifest = _apply_lora_upper8(
            model,
            r=int(args.lora_r),
            alpha=int(args.lora_alpha),
            dropout=float(args.lora_dropout),
        )
        # LoRA modules are inside showo; keep LoRA trainable on top of baseline trainable sets.
        for n, p in model.named_parameters():
            if ".lora_A." in n or ".lora_B." in n:
                p.requires_grad = True
        freeze_manifest = _collect_trainability_manifest(model)

    # Arm C: LoRA on diffusion_head_a to prevent catastrophic forgetting of the
    # pretrained velocity field.  adaLN_modulation stays fully trainable (zero-init,
    # must move freely).  Base weights of wrapped linears are frozen by LoRALinear.
    diff_lora_manifest = None
    if args.lora_diffusion_head:
        diff_lora_manifest = _apply_lora_diffusion_head(
            model,
            r=int(args.lora_diff_r),
            alpha=int(args.lora_diff_alpha),
            dropout=float(args.lora_dropout),
        )
        for n, p in model.named_parameters():
            if "diffusion_head_a" in n and (".lora_A." in n or ".lora_B." in n):
                p.requires_grad = True
        freeze_manifest = _collect_trainability_manifest(model)

    if _is_main(rank):
        save_json(out_dir / "freeze_manifest.json", freeze_manifest)
        if lora_manifest is not None:
            save_json(out_dir / "lora_manifest.json", lora_manifest)
            print("[lora]", json.dumps(lora_manifest, indent=2))
        if diff_lora_manifest is not None:
            save_json(out_dir / "lora_diff_manifest.json", diff_lora_manifest)
            print("[lora-diff]", json.dumps(diff_lora_manifest, indent=2))
        print("[freeze]", json.dumps(freeze_manifest["summary"], indent=2))
        print(
            f"[memory] sharding={args.sharding} gradient_checkpointing={bool(args.gradient_checkpointing)} "
            f"arm_a_mode={args.arm_a_mode} train_text_embeddings={train_text_embeddings} "
            f"lora_diffusion_head={bool(args.lora_diffusion_head)}"
        )

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

    # Build optimizer groups by name.
    text_params = []
    diff_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("showo.model.embed_tokens"):
            text_params.append(p)
        else:
            diff_params.append(p)
    param_groups = [
        {"params": text_params, "lr": float(args.lr_text), "weight_decay": 0.0},
        {"params": diff_params, "lr": float(args.lr_diffusion), "weight_decay": 0.0},
    ]
    if use_zero1:
        optimizer = ZeroRedundancyOptimizer(
            param_groups,
            optimizer_class=AdamW,
        )
    else:
        optimizer = AdamW(param_groups)
    scheduler = _build_lr_scheduler(optimizer, total_steps=max(1, int(args.max_steps)), warmup_ratio=float(args.warmup_ratio))

    stats = {
        "task_steps": {"icd": 0, "report": 0, "image": 0},
        "loss_running": {"icd": None, "report": None, "image": None},
    }
    resume_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if _is_main(rank):
            print(
                f"[resume] loaded model from {args.resume_from} "
                f"(missing={len(missing)} unexpected={len(unexpected)})"
            )
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                if _is_main(rank):
                    print("[resume] optimizer state restored")
            except Exception as e:
                if _is_main(rank):
                    print(f"[resume][warn] optimizer state restore failed: {e}")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                if _is_main(rank):
                    print("[resume] scheduler state restored")
            except Exception as e:
                if _is_main(rank):
                    print(f"[resume][warn] scheduler restore failed: {e}")
        stats = ckpt.get("stats", stats)
        resume_step = int(ckpt.get("step", 0))
        if _is_main(rank):
            print(f"[resume] resume_step={resume_step}")

    if _is_main(rank):
        print("[vae] loading WanVAE ...")
    vae = WanVAE(vae_pth=str(args.vae_pth), dtype=weight_type, device=device)
    # seq_len must match x.shape[1] that the inference ODE solver sees at run-time.
    # The ODE integrates over latents of shape [N, C, H, W], so x.shape[1] = C = 16
    # (latent channels), NOT the number of patch tokens (1024).  Using 1024 here
    # produced mu≈0.630 vs the inference mu≈0.459, causing a train/inference
    # time-schedule mismatch.  seq_len=16 gives mu≈0.459 on both sides.
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        snr_type="lognorm",
        do_shift=True,
        seq_len=16,
    )

    if use_fsdp:
        mp_policy = None
        if weight_type in (torch.bfloat16, torch.float16):
            mp_policy = MixedPrecision(
                param_dtype=weight_type,
                reduce_dtype=weight_type,
                buffer_dtype=weight_type,
            )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            limit_all_gathers=True,
        )
    elif ddp:
        strict_arm_a_ddp = (args.llm_arm == "frozen") and (args.arm_a_mode == "strict")
        # Mixed-task training: on ICD/report steps diffusion heads receive no gradients;
        # on image steps LoRA/embedding params may not receive gradients depending on
        # the forward path. find_unused_parameters=True is required for all arms except
        # strict Arm A, where text tasks are fully short-circuited (no backward at all)
        # so only image steps run backward and every trainable param participates.
        find_unused = not strict_arm_a_ddp
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
    model.train()
    sync_param = None
    for p in model.parameters():
        if p.requires_grad:
            sync_param = p
            break
    start_step = int(resume_step) + 1
    iterator = range(start_step, int(args.max_steps) + 1)
    pbar = tqdm(iterator, desc="stage2-baseline", disable=not _is_main(rank))
    last_step = int(resume_step)
    for step in pbar:
        last_step = int(step)
        task = schedule[(step - 1) % len(schedule)]
        batch = next(task_iters[task])
        _tasks = batch["task"]
        if len(set(_tasks)) > 1:
            raise RuntimeError(f"Mixed tasks in one batch (bug): {_tasks}")

        strict_arm_a = (args.llm_arm == "frozen") and (args.arm_a_mode == "strict")
        if strict_arm_a and task in ("icd", "report"):
            stats["task_steps"][task] += 1
            prev = stats["loss_running"][task]
            cur = 0.0
            stats["loss_running"][task] = cur if prev is None else (0.9 * prev + 0.1 * cur)
            if _is_main(rank):
                pbar.set_postfix(
                    {
                        "task": task,
                        "icd_n": stats["task_steps"]["icd"],
                        "rep_n": stats["task_steps"]["report"],
                        "img_n": stats["task_steps"]["image"],
                        "icd_l": f"{(stats['loss_running']['icd'] or 0):.3f}",
                        "rep_l": f"{(stats['loss_running']['report'] or 0):.3f}",
                        "img_l": f"{(stats['loss_running']['image'] or 0):.3f}",
                    }
                )
            continue

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask_1d = batch["attention_mask"].to(device)

        pixel_values, modality_positions, text_masks, image_masks = _collect_image_slots(
            batch,
            device=device,
            resolution=int(args.image_resolution),
            img_pad_id=int(tokenizer.convert_tokens_to_ids("<|image_pad|>")),
            pad_id=int(tokenizer.pad_token_id),
        )
        if pixel_values is None or modality_positions is None:
            attn = _build_causal_padding_mask(attention_mask_1d)
        else:
            attn = _build_omni_padding_mask(attention_mask_1d, modality_positions)

        optimizer.zero_grad(set_to_none=True)

        if task in ("icd", "report"):
            if pixel_values is None:
                out = model(
                    text_tokens=input_ids,
                    image_latents=None,
                    attention_mask=attn,
                    output_hidden_states=False,
                    max_seq_len=input_ids.size(1),
                    device=device,
                )
                logits = out.logits if hasattr(out, "logits") else out["logits"]
                loss = _ce_loss(logits, labels)
            else:
                # Context images are observed inputs: keep them CLEAN for NTP tasks.
                # Passing noisy x_t here creates a train/infer mismatch (at inference
                # we have clean context images), and also injects a time embedding
                # that isn't meaningful for pure autoregressive ICD/report.
                with torch.no_grad():
                    x1 = vae.sample(pixel_values.unsqueeze(2)).squeeze(2)
                xt = x1.to(dtype=weight_type)
                t = torch.ones((x1.size(0),), device=device, dtype=weight_type)

                _logits, loss_ntp = model(
                    text_tokens=input_ids,
                    image_latents=xt,
                    t=t,
                    attention_mask=attn,
                    text_masks=text_masks,
                    image_masks=image_masks,
                    text_labels=labels,
                    image_labels=None,
                    modality_positions=modality_positions,
                    output_hidden_states=True,
                    max_seq_len=input_ids.size(1),
                    device=device,
                )
                loss = loss_ntp

        else:
            # image task: flow-only supervision on target slot.
            if pixel_values is None or modality_positions is None:
                if sync_param is not None:
                    loss = sync_param.sum() * 0.0
                else:
                    loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                stats["task_steps"]["image"] += 1
                prev = stats["loss_running"]["image"]
                cur = 0.0
                stats["loss_running"]["image"] = cur if prev is None else (0.9 * prev + 0.1 * cur)
                if _is_main(rank):
                    pbar.set_postfix(
                        {
                            "task": task,
                            "icd_n": stats["task_steps"]["icd"],
                            "rep_n": stats["task_steps"]["report"],
                            "img_n": stats["task_steps"]["image"],
                            "icd_l": f"{(stats['loss_running']['icd'] or 0):.3f}",
                            "rep_l": f"{(stats['loss_running']['report'] or 0):.3f}",
                            "img_l": f"{(stats['loss_running']['image'] or 0):.3f}",
                        }
                    )
                continue
            with torch.no_grad():
                x1 = vae.sample(pixel_values.unsqueeze(2)).squeeze(2)
            # Keep context slots CLEAN, and only noise/supervise target slots.
            sup = batch["image_supervise_masks"].to(device=device, dtype=torch.long)
            xt = x1.clone()
            ut = torch.zeros_like(x1)
            t = torch.ones((x1.size(0),), device=device, dtype=torch.float32)

            bsz_i, m_slots = int(sup.shape[0]), int(sup.shape[1])
            for idx in range(x1.size(0)):
                bi = idx // m_slots
                sj = idx % m_slots
                if int(sup[bi, sj].item()) != 1:
                    continue
                t_i, x0, x1_i = transport.sample(x1[idx][None])
                t_i, xt_i, ut_i = transport.path_sampler.plan(t_i, x0, x1_i)
                t[idx] = t_i.squeeze(0)
                xt[idx] = xt_i.squeeze(0)
                ut[idx] = ut_i.squeeze(0)

            # Cast for model
            xt = xt.to(dtype=weight_type)
            ut = ut.to(dtype=weight_type)
            t = t.to(device=device, dtype=weight_type)

            # Zero out context slots in image mask; keep only supervised target slots.
            pos = modality_positions
            for bi in range(bsz_i):
                for j in range(m_slots):
                    if int(sup[bi, j].item()) != 0:
                        continue
                    off = int(pos[bi, j, 0].item())
                    ln = int(pos[bi, j, 1].item())
                    if ln > 0:
                        image_masks[bi, off : off + ln] = 0

            _logits, loss_flow = model(
                text_tokens=input_ids,
                image_latents=xt,
                t=t,
                attention_mask=attn,
                text_masks=text_masks,
                image_masks=image_masks,
                text_labels=None,
                image_labels=ut,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=input_ids.size(1),
                device=device,
            )
            loss = loss_flow

        if not torch.isfinite(loss):
            if _is_main(rank):
                print(
                    f"[warn] non-finite loss={float(loss.detach().cpu()):.4f} "
                    f"task={task} step={step} — skipping backward"
                )
            stats["task_steps"][task] += 1
            if sync_param is not None:
                noop = sync_param.sum() * 0.0
                noop.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            continue
        if not bool(loss.requires_grad):
            if sync_param is not None:
                # Strict Arm-A can produce text-task losses detached from trainable params.
                # Use a synchronized no-op gradient so distributed steps stay aligned.
                loss = sync_param.sum() * 0.0
            else:
                loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        loss.backward()
        if use_fsdp:
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        v = float(loss.detach().cpu())
        stats["task_steps"][task] += 1
        cur = stats["loss_running"][task]
        stats["loss_running"][task] = v if cur is None else (0.98 * cur + 0.02 * v)
        if _is_main(rank):
            pbar.set_postfix(
                {
                    "task": task,
                    "icd_n": stats["task_steps"]["icd"],
                    "rep_n": stats["task_steps"]["report"],
                    "img_n": stats["task_steps"]["image"],
                    "icd_l": f"{(stats['loss_running']['icd'] or 0):.3f}",
                    "rep_l": f"{(stats['loss_running']['report'] or 0):.3f}",
                    "img_l": f"{(stats['loss_running']['image'] or 0):.3f}",
                }
            )

        if int(args.save_every) > 0 and (int(step) % int(args.save_every) == 0):
            _save_checkpoint(
                out_dir=out_dir,
                step=int(step),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stats=stats,
                args=args,
                rank=rank,
                ddp=ddp,
                use_zero1=use_zero1,
                keep_last=int(args.keep_last),
                is_final=False,
            )

    if bool(args.save_final):
        _save_checkpoint(
            out_dir=out_dir,
            step=int(last_step),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            stats=stats,
            args=args,
            rank=rank,
            ddp=ddp,
            use_zero1=use_zero1,
            keep_last=max(int(args.keep_last), 1),
            is_final=True,
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

