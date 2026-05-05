#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import importlib.machinery
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
# wandb optional — not used in the released code path

from data_pipeline import (  # noqa: E402
    MODALITY_CAPS,
    MODALITY_ORDER,
    MODALITY_PREFIX,
    NO_RECORD_TOKEN,
    MultiCodeVisitWindowDataset,
    build_code_token_inventory,
    collate_windows,
    load_patient_timelines_from_shared,
    load_subject_splits_from_shared,
)
from losses import causal_ce_loss, perplexity_from_loss  # noqa: E402
from models import Showo2Qwen2_5  # noqa: E402
from build_icd_descriptions import load_icd_descriptions  # noqa: E402
from icd_embedding_init import initialize_icd_embeddings  # noqa: E402
from utils.misc import get_text_tokenizer  # noqa: E402


def _unwrap_showo(showo):
    if isinstance(showo, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return showo.module
    return showo


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_diag_code(code: str) -> str:
    code = str(code or "").strip().upper()
    if code.startswith("E") or code.startswith("V"):
        return code[:4] if len(code) > 4 else code
    return code[:3] if len(code) > 3 else code


def _add_tokens(tokenizer, code_inventory: Dict[str, List[str]]) -> Dict[str, object]:
    struct_tokens = ["<VISIT_START>", "<VISIT_END>", NO_RECORD_TOKEN]
    modality_struct_tokens = {
        mod: [f"<{MODALITY_PREFIX[mod]}>", f"</{MODALITY_PREFIX[mod]}>"]
        for mod in MODALITY_ORDER
    }
    all_code_tokens: List[str] = []
    for mod in MODALITY_ORDER:
        all_code_tokens.extend(code_inventory[mod])

    new_tokens = list(struct_tokens)
    for mod in MODALITY_ORDER:
        new_tokens.extend(modality_struct_tokens[mod])
    new_tokens.extend(all_code_tokens)

    existing = list(tokenizer.additional_special_tokens or [])
    merged = existing + [tok for tok in new_tokens if tok not in existing]
    tokenizer.add_special_tokens({"additional_special_tokens": merged})

    unk_tok = getattr(tokenizer, "unk_token_id", None)
    unk = -1 if unk_tok is None else int(unk_tok)
    modality_token_ids = {}
    for mod in MODALITY_ORDER:
        ids = [int(tokenizer.convert_tokens_to_ids(tok)) for tok in code_inventory[mod]]
        modality_token_ids[mod] = [tid for tid in ids if tid != unk]

    trainable_token_ids = [int(tokenizer.convert_tokens_to_ids(tok)) for tok in new_tokens]
    trainable_token_ids = [tid for tid in trainable_token_ids if tid != unk]
    return {
        "struct_tokens": struct_tokens,
        "modality_struct_tokens": modality_struct_tokens,
        "code_tokens": code_inventory,
        "all_code_tokens": all_code_tokens,
        "modality_token_ids": modality_token_ids,
        "trainable_token_ids": sorted(set(trainable_token_ids)),
    }


def _build_desc_mapping_json(
    out_json: Path,
    tok_meta: Dict[str, object],
    d_icd_gz: Path | None,
) -> None:
    """Build the per-token description JSON used by initialize_icd_embeddings.

    Run H is diagnosis-only, so this only emits descriptions for ICD-9 diag
    code tokens and the structural tokens. Other multicode modalities, if
    present in the tokenizer, get a generic placeholder description that the
    embedding initializer can ignore.
    """
    diag_desc = {}
    if d_icd_gz is not None and d_icd_gz.exists():
        all_desc = load_icd_descriptions(str(d_icd_gz))
        for (code, version), desc in all_desc.items():
            if int(version) != 9:
                continue
            diag_desc.setdefault(_normalize_diag_code(code), desc)

    mappings = {}
    struct_desc = {
        "<VISIT_START>": "visit start marker",
        "<VISIT_END>": "visit end marker",
        "<NO_RECORD>": "no record sentinel for missing modality data",
        "<DIAG>": "diagnosis code block start",
        "</DIAG>": "diagnosis code block end",
    }
    for tok in tok_meta["struct_tokens"]:
        mappings[tok] = {"code": tok, "version": 9,
                          "description": struct_desc.get(tok, ""), "method": "direct"}
    # Only the diag modality is exercised in this release.
    if "diag" in tok_meta.get("modality_struct_tokens", {}):
        for tok in tok_meta["modality_struct_tokens"]["diag"]:
            mappings[tok] = {"code": tok, "version": 9,
                              "description": struct_desc.get(tok, ""), "method": "direct"}
    if "diag" in tok_meta.get("code_tokens", {}):
        for tok in tok_meta["code_tokens"]["diag"]:
            code = tok.replace(f"<{MODALITY_PREFIX['diag']}_", "").replace(">", "")
            desc = diag_desc.get(code, f"diagnosis code {code}")
            mappings[tok] = {"code": code, "version": 9, "description": desc, "method": "direct"}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)


def _set_trainable_embedding_rows(embed_weight: torch.nn.Parameter, trainable_token_ids: List[int]) -> None:
    embed_weight.requires_grad = True
    keep_ids = torch.tensor(sorted(set(int(x) for x in trainable_token_ids)), dtype=torch.long)

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        mask = torch.zeros((grad.size(0),), dtype=grad.dtype, device=grad.device)
        mask[keep_ids.to(device=grad.device)] = 1.0
        return grad * mask.unsqueeze(1)

    embed_weight.register_hook(_mask_grad)


def _build_token_id_to_name(tokenizer, tok_meta: Dict[str, object]) -> Dict[int, str]:
    token_id_to_name = {}
    for tok in tok_meta["struct_tokens"]:
        token_id_to_name[int(tokenizer.convert_tokens_to_ids(tok))] = tok
    for mod in MODALITY_ORDER:
        for tok in tok_meta["modality_struct_tokens"][mod]:
            token_id_to_name[int(tokenizer.convert_tokens_to_ids(tok))] = tok
    for tok in tok_meta["all_code_tokens"]:
        token_id_to_name[int(tokenizer.convert_tokens_to_ids(tok))] = tok
    return token_id_to_name


def _build_embedding_checkpoint(model, tokenizer, tok_meta: Dict[str, object], args, *, epoch: int, step: int) -> Dict[str, object]:
    token_id_to_name = _build_token_id_to_name(tokenizer, tok_meta)
    trainable_ids = sorted(int(x) for x in tok_meta["trainable_token_ids"])
    emb_weight = _unwrap_showo(model.showo).get_input_embeddings().weight.detach().cpu()
    return {
        "step": int(step),
        "epoch": int(epoch),
        "tokenizer_len": len(tokenizer),
        "trainable_token_ids": trainable_ids,
        "trainable_token_names": [token_id_to_name[tid] for tid in trainable_ids],
        "modality_token_ids": tok_meta["modality_token_ids"],
        "embedding_rows": emb_weight[trainable_ids].clone(),
        "args": vars(args),
    }


def _atomic_torch_save(obj, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _save_embedding_checkpoint(out_dir: Path, stem: str, ckpt: Dict[str, object]) -> None:
    _atomic_torch_save(ckpt, out_dir / f"{stem}_stage1_code_embedding_rows.pt")
    _atomic_torch_save(ckpt, out_dir / f"{stem}_stage1_icd_embedding_rows.pt")


class _IndexSampler(Sampler[int]):
    def __init__(self, indices: List[int]):
        self.indices = [int(x) for x in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _ddp_rank_indices_from_epoch_stream(
    stream: List[int],
    *,
    batch_size: int,
    world_size: int,
    rank: int,
) -> List[int]:
    """Interleave global batches of size (batch_size * world_size) across ranks."""
    M = int(batch_size) * int(world_size)
    assert len(stream) % M == 0, "stream length must be divisible by batch_size * world_size"
    rank_ids: List[int] = []
    num_global = len(stream) // M
    for g in range(num_global):
        off = g * M + int(rank) * int(batch_size)
        rank_ids.extend(stream[off : off + int(batch_size)])
    return rank_ids


def _sample_epoch_indices(
    sample_weights: torch.Tensor,
    *,
    num_samples: int,
    seed: int,
) -> List[int]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.multinomial(
        sample_weights,
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    ).tolist()


def _build_train_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    collate_fn,
    pin_memory: bool,
    sample_weights: torch.Tensor,
    epoch: int,
    base_seed: int,
    resume_step_in_epoch: int = 0,
    ddp_world_size: int = 1,
    ddp_rank: int = 0,
    ddp_warn_truncation: bool = True,
):
    epoch_indices = _sample_epoch_indices(
        sample_weights,
        num_samples=len(dataset),
        seed=int(base_seed) + int(epoch),
    )
    if int(ddp_world_size) > 1:
        M = int(batch_size) * int(ddp_world_size)
        start_idx = max(0, int(resume_step_in_epoch) * M)
        stream = epoch_indices[start_idx:]
        n = len(stream)
        n_complete = (n // M) * M
        if n_complete < n and ddp_warn_truncation and int(ddp_rank) == 0:
            print(
                f"[warn] DDP: dropping {n - n_complete} sampled indices so stream length "
                f"is divisible by batch_size*world_size={M}"
            )
        stream = stream[:n_complete]
        rank_indices = _ddp_rank_indices_from_epoch_stream(
            stream,
            batch_size=int(batch_size),
            world_size=int(ddp_world_size),
            rank=int(ddp_rank),
        )
    else:
        start_idx = max(0, int(resume_step_in_epoch) * int(batch_size))
        rank_indices = epoch_indices[start_idx:]
    sampler = _IndexSampler(rank_indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def _build_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * float(warmup_ratio)))

    def _lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return LambdaLR(optimizer, _lr_lambda)


def _build_causal_padding_mask(attn_1d: torch.Tensor) -> torch.Tensor:
    bsz, seqlen = attn_1d.shape
    dev = attn_1d.device
    valid = attn_1d.bool()
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=dev))
    keep = causal.unsqueeze(0) & valid.unsqueeze(1)
    mask = torch.zeros((bsz, 1, seqlen, seqlen), dtype=torch.float32, device=dev)
    return mask.masked_fill(~keep.unsqueeze(1), -1e4)


def _validate_id_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    vocab_size: int,
    step: int,
    ignore_index: int | None = None,
    max_reports: int = 8,
) -> bool:
    if tensor.numel() == 0:
        return True
    if ignore_index is None:
        bad = (tensor < 0) | (tensor >= vocab_size)
    else:
        bad = (tensor != ignore_index) & ((tensor < 0) | (tensor >= vocab_size))
    if not bool(bad.any().item()):
        return True
    bad_pos = bad.nonzero(as_tuple=False)[:max_reports]
    bad_vals = tensor[bad][:max_reports].detach().cpu().tolist()
    _rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if _rank0:
        print(
            f"[warn] invalid {name} ids at step={step}: "
            f"vocab_size={vocab_size} bad_vals={bad_vals} bad_pos={bad_pos.detach().cpu().tolist()}",
            flush=True,
        )
    return False


def _cuda_mem_snapshot_lines() -> List[str]:
    if not torch.cuda.is_available():
        return []
    lines = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserv = torch.cuda.memory_reserved(i) / (1024**3)
        peak = torch.cuda.max_memory_allocated(i) / (1024**3)
        lines.append(f"cuda:{i} alloc={alloc:.2f}GiB reserved={reserv:.2f}GiB peak={peak:.2f}GiB")
    return lines


def _log_cuda_memory(tag: str) -> None:
    snap = _cuda_mem_snapshot_lines()
    if snap:
        print(f"{tag} " + " | ".join(snap))


def _format_duration(sec: float) -> str:
    if sec != sec or sec < 0:
        return "?"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m}m{s}s"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


def _parse_modality_loss_weights(arg: str) -> Optional[torch.Tensor]:
    """Comma-separated weights in MODALITY_ORDER (diag,). Empty = equal weights (i.e., diag-only here)."""
    s = (arg or "").strip()
    if not s:
        return None
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != len(MODALITY_ORDER):
        raise ValueError(
            f"--modality-loss-weights expects {len(MODALITY_ORDER)} values "
            f"({','.join(MODALITY_ORDER)}), got {len(parts)}"
        )
    if any(w < 0 for w in parts):
        raise ValueError("--modality-loss-weights must be non-negative")
    return torch.tensor(parts, dtype=torch.float32)


def _balanced_modality_losses(
    logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    device,
    modality_weights: Optional[torch.Tensor] = None,
    *,
    compute_loss_full: bool = True,
) -> Dict[str, float | torch.Tensor]:
    loss_by_mod = {}
    loss_terms = []
    for mod in MODALITY_ORDER:
        labels = batch[f"labels_{mod}"].to(device)
        loss_mod = causal_ce_loss(logits, labels)
        loss_by_mod[mod] = loss_mod
        loss_terms.append(loss_mod)
    if not loss_terms:
        balanced = torch.zeros((), device=device)
    elif modality_weights is None:
        balanced = torch.stack(loss_terms).mean()
    else:
        w = modality_weights.to(device=device, dtype=loss_terms[0].dtype)
        stacked = torch.stack(loss_terms)
        balanced = (stacked * w).sum() / w.sum().clamp_min(1e-8)
    # Extra full-sequence CE is not used in the training objective (we optimize loss_balanced only).
    # Skipping it saves a large graph chunk and VRAM; eval still computes loss_full for metrics.
    if compute_loss_full:
        loss_full = causal_ce_loss(logits, batch["labels_full"].to(device))
    else:
        loss_full = torch.zeros((), device=logits.device, dtype=logits.dtype)
    return {
        "loss_full": loss_full,
        "loss_balanced": balanced,
        **{f"loss_{mod}": loss_by_mod[mod] for mod in MODALITY_ORDER},
    }


def _evaluate(
    model,
    loader,
    device,
    modality_weights: Optional[torch.Tensor] = None,
    *,
    tqdm_disable: bool = False,
):
    model.eval()
    sums = {"loss_full": 0.0, "loss_balanced": 0.0, **{f"loss_{mod}": 0.0 for mod in MODALITY_ORDER}}
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False, disable=tqdm_disable):
            input_ids = batch["input_ids"].to(device)
            attention_mask = _build_causal_padding_mask(batch["attention_mask"].to(device))
            out = model.showo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            losses = _balanced_modality_losses(
                out.logits,
                batch,
                out.logits.device,
                modality_weights=modality_weights,
                compute_loss_full=True,
            )
            for key, value in losses.items():
                sums[key] += float(value.detach().cpu())
            n_batches += 1
    if n_batches == 0:
        return {key: float("nan") for key in sums}
    metrics = {key: sums[key] / n_batches for key in sums}
    metrics["ppl_target_full"] = float(perplexity_from_loss(torch.tensor(metrics["loss_full"])).item())
    for mod in MODALITY_ORDER:
        metrics[f"ppl_{mod}"] = float(perplexity_from_loss(torch.tensor(metrics[f"loss_{mod}"])).item())
    return metrics


def _build_resume_state(
    model,
    tokenizer,
    tok_meta: Dict[str, object],
    args,
    optimizer,
    scheduler,
    *,
    epoch: int,
    step: int,
    step_in_epoch: int,
    best_val: float,
    skipped_nonfinite: int,
    skipped_invalid: int,
) -> Dict[str, object]:
    state = _build_embedding_checkpoint(
        model,
        tokenizer,
        tok_meta,
        args,
        epoch=epoch,
        step=step,
    )
    state.update(
        {
            "step_in_epoch": int(step_in_epoch),
            "best_val": float(best_val),
            "skipped_nonfinite": int(skipped_nonfinite),
            "skipped_invalid": int(skipped_invalid),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
    )
    return state


def _save_resume_state(path: Path, state: Dict[str, object]) -> None:
    _atomic_torch_save(state, path)


def _load_resume_state(path: Path, model, optimizer, scheduler, tok_meta: Dict[str, object], device) -> Dict[str, object]:
    state = torch.load(path, map_location="cpu")
    saved_ids = [int(x) for x in state["trainable_token_ids"]]
    expected_ids = sorted(int(x) for x in tok_meta["trainable_token_ids"])
    if saved_ids != expected_ids:
        raise RuntimeError(f"resume token ids mismatch for {path}")

    embed_weight = _unwrap_showo(model.showo).get_input_embeddings().weight.data
    saved_rows = state["embedding_rows"].to(device=embed_weight.device, dtype=embed_weight.dtype)
    embed_weight[expected_ids] = saved_rows

    optimizer.load_state_dict(state["optimizer_state_dict"])
    for group in optimizer.state.values():
        for key, value in group.items():
            if torch.is_tensor(value):
                group[key] = value.to(device)
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, default="showlab/show-o2-1.5B-HQ")
    ap.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument(
        "--shared-dir",
        type=str,
        default="${SHARED_DIR}",
        help="Directory containing {train,val,test}_subjects.pkl produced by the "
             "shared MIMIC-IV-Hosp ICD-9 preprocessing pipeline. Supply via env "
             "variable SHARED_DIR.",
    )
    ap.add_argument("--mimic-d-icd-gz", type=str, default="${MIMIC_HOSP_ROOT}/d_icd_diagnoses.csv.gz",
                    help="MIMIC d_icd_diagnoses dictionary used to text-initialize ICD-9 diag code embeddings.")
    ap.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs_stage1"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--max-diag-per-visit", type=int, default=MODALITY_CAPS["diag"])
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-process batch size. With --use-distributed, global batch ≈ batch_size × WORLD_SIZE.",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps (0 = run all epochs). Saves resume state and exits.",
    )
    ap.add_argument(
        "--modality-loss-weights",
        type=str,
        default="",
        help=f"Comma-separated weights for {','.join(MODALITY_ORDER)}; empty = equal average.",
    )
    ap.add_argument("--use-data-parallel", action="store_true", default=False)
    ap.add_argument(
        "--use-distributed",
        action="store_true",
        default=False,
        help="Multi-GPU via torchrun + DistributedDataParallel (better VRAM balance than DataParallel). "
        "batch_size is per GPU; global batch = batch_size × WORLD_SIZE. "
        "Example: torchrun --standalone --nproc_per_node=2 path/to/train.py ... --use-distributed",
    )
    ap.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        default=False,
        help="Set find_unused_parameters=True on DDP (slower). Use if backward errors about unused params.",
    )
    ap.add_argument(
        "--dp-gather-on-first-gpu",
        action="store_true",
        default=False,
        help="DataParallel: gather logits on cuda:0 (PyTorch default). "
        "Default off: gather on last visible GPU so large logits+loss are not stacked on cuda:0.",
    )
    ap.add_argument(
        "--log-vram-every",
        type=int,
        default=0,
        help="Print per-GPU alloc/reserved/peak every N optimizer steps (0 = off). "
        "Also prints once after the first successful step.",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patient-balance-alpha", type=float, default=0.5)
    ap.add_argument("--save-every-steps", type=int, default=1000)
    ap.add_argument("--resume-state", type=str, default="")
    ap.add_argument("--no-auto-resume", action="store_true", default=False)
    args = ap.parse_args()
    modality_loss_w = _parse_modality_loss_weights(args.modality_loss_weights)

    use_ddp = bool(args.use_distributed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.use_data_parallel and use_ddp:
        raise SystemExit("Use either --use-data-parallel or --use-distributed, not both.")
    if use_ddp:
        if not torch.cuda.is_available():
            raise SystemExit("--use-distributed requires CUDA")
        if world_size < 2:
            raise SystemExit(
                "--use-distributed needs WORLD_SIZE>=2. Launch with e.g.\n"
                "  torchrun --standalone --nproc_per_node=2 .../train.py ... --use-distributed"
            )
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        def _destroy_dist_if_needed() -> None:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

        atexit.register(_destroy_dist_if_needed)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (not use_ddp) or (local_rank == 0)

    def log_info(msg: str) -> None:
        if is_main:
            print(msg, flush=True)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()
    _seed_all(args.seed + local_rank if use_ddp else args.seed)

    log_info(f"[device] {device}" + (f" (DDP rank {local_rank}/{world_size})" if use_ddp else ""))

    shared_dir = Path(args.shared_dir)
    log_info(f"[data] loading shared timelines from {shared_dir}")
    timelines_train = load_patient_timelines_from_shared(shared_dir, "train")
    timelines_val = load_patient_timelines_from_shared(shared_dir, "val")
    splits = load_subject_splits_from_shared(shared_dir)
    if is_main:
        with open(out_dir / "subject_splits.json", "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
    if use_ddp:
        dist.barrier()

    code_inventory = build_code_token_inventory(timelines_train)
    for mod in MODALITY_ORDER:
        log_info(f"[vocab] {mod}_tokens={len(code_inventory[mod]):,}")

    tokenizer = get_text_tokenizer(
        args.tokenizer_model,
        add_showo_tokens=True,
        return_showo_token_ids=False,
        llm_name="qwen2_5",
    )
    tok_meta = _add_tokens(tokenizer, code_inventory)

    modality_caps = {
        "diag": int(args.max_diag_per_visit),
    }
    if is_main:
        with open(out_dir / "token_inventory.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "struct_tokens": tok_meta["struct_tokens"],
                    "modality_struct_tokens": tok_meta["modality_struct_tokens"],
                    "code_tokens": tok_meta["code_tokens"],
                    "all_code_tokens": tok_meta["all_code_tokens"],
                    "icd_tokens": tok_meta["all_code_tokens"],
                    "modality_token_counts": {mod: len(code_inventory[mod]) for mod in MODALITY_ORDER},
                    "modality_caps": modality_caps,
                    "num_icd_tokens": len(tok_meta["all_code_tokens"]),
                },
                f,
                indent=2,
            )
            # Save tokenizer immediately so eval / resume works even if training dies mid-epoch
            # (tokenizer was previously only written after the first full epoch).
            tok_dir = out_dir / "tokenizer"
            tokenizer.save_pretrained(tok_dir)
            log_info(f"[ckpt] saved tokenizer (early) -> {tok_dir}")
    if use_ddp:
        dist.barrier()

    log_info("[model] loading Showo2 backbone ...")
    model = Showo2Qwen2_5.from_pretrained(args.pretrained, use_safetensors=False).to(device)
    model.showo.resize_token_embeddings(len(tokenizer))
    model.showo.tie_weights()

    desc_json = out_dir / "icd_category_descriptions_stage1.json"
    if is_main:
        # Diagnosis-only embedding initialization (Run H configuration).
        _build_desc_mapping_json(
            out_json=desc_json,
            tok_meta=tok_meta,
            d_icd_gz=Path(args.mimic_d_icd_gz) if args.mimic_d_icd_gz else None,
        )
    if use_ddp:
        dist.barrier()
    wrap = type("_Wrap", (), {"backbone": model.showo})()
    initialize_icd_embeddings(
        wrap,
        tokenizer,
        str(desc_json),
        device=str(model.showo.get_input_embeddings().weight.device),
        verbose=is_main,
        use_cache=True,
    )

    for p in model.parameters():
        p.requires_grad = False
    embed = _unwrap_showo(model.showo).get_input_embeddings()
    _set_trainable_embedding_rows(embed.weight, tok_meta["trainable_token_ids"])

    if use_ddp:
        model.showo = torch.nn.parallel.DistributedDataParallel(
            model.showo,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )
        log_info(
            f"[model] DistributedDataParallel rank {local_rank}/{world_size}: "
            f"per-GPU batch_size={args.batch_size}, "
            f"global batch ≈ {int(args.batch_size) * world_size} (even split, no gather logits on one GPU)"
        )
    elif args.use_data_parallel and (device.type == "cuda") and torch.cuda.device_count() > 1:
        n_gpu = torch.cuda.device_count()
        dev_ids = list(range(n_gpu))
        out_dev = dev_ids[0] if args.dp_gather_on_first_gpu else dev_ids[-1]
        model.showo = torch.nn.DataParallel(
            model.showo,
            device_ids=dev_ids,
            output_device=out_dev,
        )
        log_info(
            f"[model] DataParallel: {n_gpu} GPUs device_ids={dev_ids} output_device=cuda:{out_dev} "
            f"(loss computed on logits device; avoids piling logits on cuda:0)"
        )
        log_info(
            "[model] Note: each GPU still holds a full model replica; only embedding rows get grads. "
            "Prefer --use-distributed + torchrun for more even VRAM and a larger effective batch."
        )

    ds_train = MultiCodeVisitWindowDataset(
        timelines=timelines_train,
        tokenizer=tokenizer,
        split_name="train",
        k_max=args.k_max,
        max_seq_len=args.max_seq_len,
        modality_max_codes=modality_caps,
        shuffle_within_visit=True,
        seed=args.seed,
    )
    ds_val = MultiCodeVisitWindowDataset(
        timelines=timelines_val,
        tokenizer=tokenizer,
        split_name="val",
        k_max=args.k_max,
        max_seq_len=args.max_seq_len,
        modality_max_codes=modality_caps,
        shuffle_within_visit=False,
        seed=args.seed,
    )
    if int(args.max_train_samples) > 0:
        ds_train.windows = ds_train.windows[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        ds_val.windows = ds_val.windows[: int(args.max_val_samples)]

    sample_weights = torch.tensor(ds_train.sample_weights(alpha=args.patient_balance_alpha), dtype=torch.double)
    collate = lambda batch: collate_windows(batch, pad_token_id=int(tokenizer.pad_token_id))
    # Val loader on every rank when using DDP: rank 0-only eval + broadcast_object_list leaves other
    # ranks blocked in NCCL for longer than the default watchdog (~600s) and triggers DistBackendError.
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW([embed.weight], lr=args.lr, weight_decay=args.weight_decay)
    if use_ddp:
        M = int(args.batch_size) * world_size
        if len(ds_train) < M:
            log_info(
                f"[warn] train samples {len(ds_train)} < global batch {M}; "
                "this DDP epoch may run 0 optimizer steps (raise batch_size or use 1 GPU)."
            )
        steps_per_epoch = max(1, len(ds_train) // M)
    else:
        steps_per_epoch = max(1, math.ceil(len(ds_train) / int(args.batch_size)))
    total_steps = max(1, int(args.epochs) * steps_per_epoch)
    if int(args.max_train_steps) > 0:
        total_steps = min(total_steps, int(args.max_train_steps))
    scheduler = _build_lr_scheduler(optimizer, total_steps, args.warmup_ratio)
    if modality_loss_w is not None:
        log_info(f"[loss] modality weights ({','.join(MODALITY_ORDER)}): {modality_loss_w.tolist()}")
    if int(args.max_train_steps) > 0:
        log_info(f"[train] max_train_steps={int(args.max_train_steps)} (LR schedule total_steps={total_steps})")

    log_info(
        f"[train] samples train={len(ds_train):,} val={len(ds_val):,} "
        f"k_max={args.k_max} max_seq_len={args.max_seq_len} caps={modality_caps}"
        + (
            f" | DDP global_batch≈{int(args.batch_size) * world_size} (per_gpu_batch={args.batch_size})"
            if use_ddp
            else ""
        )
    )

    global_step = 0
    skipped_nonfinite = 0
    skipped_invalid = 0
    best_val = float("inf")
    resume_state_path = Path(args.resume_state) if args.resume_state else (out_dir / "latest_train_state.pt")
    start_epoch = 1
    resume_step_in_epoch = 0
    if (not args.no_auto_resume) and resume_state_path.exists():
        resume_state = _load_resume_state(
            resume_state_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tok_meta=tok_meta,
            device=device,
        )
        start_epoch = int(resume_state.get("epoch", 1))
        resume_step_in_epoch = int(resume_state.get("step_in_epoch", 0))
        global_step = int(resume_state.get("step", 0))
        best_val = float(resume_state.get("best_val", best_val))
        skipped_nonfinite = int(resume_state.get("skipped_nonfinite", skipped_nonfinite))
        skipped_invalid = int(resume_state.get("skipped_invalid", skipped_invalid))
        if resume_step_in_epoch >= steps_per_epoch:
            start_epoch += 1
            resume_step_in_epoch = 0
        log_info(
            f"[resume] loaded {resume_state_path} "
            f"epoch={start_epoch} global_step={global_step} step_in_epoch={resume_step_in_epoch}"
        )

    stop_requested = {"flag": False, "signum": None}

    def _request_stop(signum, _frame):
        if not stop_requested["flag"]:
            stop_requested["flag"] = True
            stop_requested["signum"] = int(signum)
            try:
                signame = signal.Signals(signum).name
            except Exception:
                signame = str(signum)
            if is_main:
                print(f"\n[signal] received {signame}; saving resume state after current batch", flush=True)

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            signal.signal(sig, _request_stop)

    full_run_optimizer_steps = int(args.epochs) * steps_per_epoch
    if int(args.max_train_steps) > 0:
        target_global_step = min(full_run_optimizer_steps, int(args.max_train_steps))
    else:
        target_global_step = full_run_optimizer_steps
    wall_start = time.perf_counter()
    step_baseline = int(global_step)
    log_info(
        f"[plan] optimizer steps: ~{full_run_optimizer_steps:,} this run "
        f"(epochs={args.epochs} × steps/epoch≈{steps_per_epoch:,}); "
        f"target_global_step={target_global_step:,} (for ETA); "
        f"starting from global_step={global_step:,}"
    )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        epoch_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" and is_main else None
        epoch_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" and is_main else None
        if epoch_start is not None:
            epoch_start.record()
        model.train()
        dl_train = _build_train_loader(
            ds_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            sample_weights=sample_weights,
            epoch=epoch,
            base_seed=args.seed,
            resume_step_in_epoch=resume_step_in_epoch if epoch == start_epoch else 0,
            ddp_world_size=world_size if use_ddp else 1,
            ddp_rank=local_rank if use_ddp else 0,
        )
        train_iter = dl_train
        if is_main:
            train_iter = tqdm(dl_train, desc=f"epoch {epoch}/{args.epochs}")
        step_in_epoch = resume_step_in_epoch if epoch == start_epoch else 0
        try:
            n_batches_epoch = len(dl_train)
        except TypeError:
            n_batches_epoch = -1
        rem_opt = max(0, target_global_step - global_step)
        log_info(
            f"[plan] epoch {epoch}/{args.epochs}: batches_this_epoch={n_batches_epoch} "
            f"remaining_optimizer_steps≈{rem_opt:,} (for ETA to target_step={target_global_step:,})"
        )
        for batch in train_iter:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = _build_causal_padding_mask(batch["attention_mask"].to(device))
            vocab_size = int(_unwrap_showo(model.showo).get_input_embeddings().num_embeddings)
            step_id = global_step + 1

            ids_ok = _validate_id_tensor(input_ids, name="input_ids", vocab_size=vocab_size, step=step_id)
            label_keys = ["labels_full"] + [f"labels_{mod}" for mod in MODALITY_ORDER]
            labels_ok = all(
                _validate_id_tensor(
                    batch[key].to(device),
                    name=key,
                    vocab_size=vocab_size,
                    step=step_id,
                    ignore_index=-100,
                )
                for key in label_keys
            )
            if not (ids_ok and labels_ok):
                skipped_invalid += 1
                continue

            out = model.showo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            loss_dev = out.logits.device
            losses = _balanced_modality_losses(
                out.logits,
                batch,
                loss_dev,
                modality_weights=modality_loss_w,
                compute_loss_full=False,
            )
            loss = losses["loss_balanced"]
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                if is_main and skipped_nonfinite <= 10:
                    print(f"[warn] non-finite loss at step={step_id}; skipping batch", flush=True)
                continue

            loss.backward()
            if embed.weight.grad is not None and not torch.isfinite(embed.weight.grad).all():
                skipped_nonfinite += 1
                if is_main and skipped_nonfinite <= 10:
                    print(f"[warn] non-finite gradient at step={step_id}; skipping optimizer step", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([embed.weight], max_norm=float(args.grad_clip))
            optimizer.step()
            scheduler.step()
            global_step += 1
            step_in_epoch += 1

            if device.type == "cuda" and is_main:
                if global_step == 1:
                    _log_cuda_memory(f"[vram] after first optimizer step (global_step={global_step})")
                elif int(args.log_vram_every) > 0 and (global_step % int(args.log_vram_every) == 0):
                    _log_cuda_memory(f"[vram] step={global_step}")

            if int(args.save_every_steps) > 0 and (global_step % int(args.save_every_steps) == 0):
                resume_state = _build_resume_state(
                    model,
                    tokenizer,
                    tok_meta,
                    args,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    step=global_step,
                    step_in_epoch=step_in_epoch,
                    best_val=best_val,
                    skipped_nonfinite=skipped_nonfinite,
                    skipped_invalid=skipped_invalid,
                )
                if is_main:
                    _save_resume_state(resume_state_path, resume_state)
                    log_info(f"\n[resume] saved train state at epoch={epoch} step={global_step} step_in_epoch={step_in_epoch}")
                if use_ddp:
                    dist.barrier()

            if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                resume_state = _build_resume_state(
                    model,
                    tokenizer,
                    tok_meta,
                    args,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    step=global_step,
                    step_in_epoch=step_in_epoch,
                    best_val=best_val,
                    skipped_nonfinite=skipped_nonfinite,
                    skipped_invalid=skipped_invalid,
                )
                if is_main:
                    _save_resume_state(resume_state_path, resume_state)
                    log_info(f"[train] reached --max-train-steps={int(args.max_train_steps)}; saved resume and exiting")
                if use_ddp:
                    dist.barrier()
                return

            if global_step % 20 == 0:
                postfix = {
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "balanced": f"{float(losses['loss_balanced'].detach().cpu()):.4f}",
                }
                for mod in MODALITY_ORDER:
                    postfix[mod] = f"{float(losses[f'loss_{mod}'].detach().cpu()):.4f}"
                elapsed = time.perf_counter() - wall_start
                done = global_step - step_baseline
                rem = max(0, target_global_step - global_step)
                if done >= 10 and elapsed > 0 and rem > 0:
                    sps = done / elapsed
                    eta_sec = rem / sps if sps > 0 else float("nan")
                    finish = datetime.now() + timedelta(seconds=eta_sec) if eta_sec == eta_sec else None
                    postfix["sps"] = f"{sps:.2f}"
                    postfix["eta_left"] = _format_duration(eta_sec)
                    if finish is not None:
                        postfix["eta_clock"] = finish.strftime("%m-%d %H:%M")
                    if is_main and global_step % 200 == 0:
                        prog = {
                            "updated": datetime.now().isoformat(timespec="seconds"),
                            "epoch": epoch,
                            "epochs": int(args.epochs),
                            "global_step": global_step,
                            "target_global_step": target_global_step,
                            "steps_per_epoch": steps_per_epoch,
                            "optimizer_steps_per_sec": round(sps, 4),
                            "eta_seconds": round(eta_sec, 1) if eta_sec == eta_sec else None,
                            "eta_finish_local": finish.isoformat(timespec="seconds") if finish is not None else None,
                        }
                        try:
                            with open(out_dir / "training_progress.json", "w", encoding="utf-8") as pf:
                                json.dump(prog, pf, indent=2)
                        except OSError:
                            pass
                if is_main:
                    train_iter.set_postfix(postfix)

            if stop_requested["flag"]:
                resume_state = _build_resume_state(
                    model,
                    tokenizer,
                    tok_meta,
                    args,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    step=global_step,
                    step_in_epoch=step_in_epoch,
                    best_val=best_val,
                    skipped_nonfinite=skipped_nonfinite,
                    skipped_invalid=skipped_invalid,
                )
                if is_main:
                    _save_resume_state(resume_state_path, resume_state)
                    log_info(f"[resume] stop requested; saved train state at epoch={epoch} step={global_step}")
                if use_ddp:
                    dist.barrier()
                return

        if use_ddp:
            dist.barrier()
        assert dl_val is not None
        # Each rank runs full val (redundant compute; keeps NCCL from timing out during long rank-0-only work).
        val_metrics = _evaluate(
            model,
            dl_val,
            device=device,
            modality_weights=modality_loss_w,
            tqdm_disable=not is_main,
        )
        if epoch_end is not None:
            epoch_end.record()
            torch.cuda.synchronize()
            epoch_sec = epoch_start.elapsed_time(epoch_end) / 1000.0
        else:
            epoch_sec = float("nan")

        epochs_left = int(args.epochs) - epoch
        eta_by_epoch = ""
        if epochs_left > 0 and epoch_sec == epoch_sec and not math.isnan(epoch_sec):
            approx_left = epochs_left * epoch_sec
            fin = datetime.now() + timedelta(seconds=approx_left)
            eta_by_epoch = (
                f" | est_remaining_by_epoch_time≈{_format_duration(approx_left)} "
                f"(~{epochs_left}×train_epoch + val, rough) done~{fin.strftime('%m-%d %H:%M')}"
            )

        log_info(
            f"[val] epoch={epoch} "
            f"loss_balanced={val_metrics['loss_balanced']:.4f} "
            f"diag={val_metrics['loss_diag']:.4f} "
            f"epoch_sec={epoch_sec:.2f}"
            f"{eta_by_epoch}"
        )
        if is_main:
            with open(out_dir / f"val_epoch_{epoch}.json", "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)

            latest_ckpt = _build_embedding_checkpoint(
                model,
                tokenizer,
                tok_meta,
                args,
                epoch=epoch,
                step=global_step,
            )
            _save_embedding_checkpoint(out_dir, "latest", latest_ckpt)
            tokenizer.save_pretrained(out_dir / "tokenizer")
            log_info(f"[ckpt] saved latest at epoch={epoch}")

            if val_metrics["loss_balanced"] < best_val:
                best_val = val_metrics["loss_balanced"]
                _save_embedding_checkpoint(out_dir, "best", latest_ckpt)
                log_info(f"[ckpt] saved best at epoch={epoch}")
            next_state = _build_resume_state(
                model,
                tokenizer,
                tok_meta,
                args,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                step=global_step,
                step_in_epoch=0,
                best_val=best_val,
                skipped_nonfinite=skipped_nonfinite,
                skipped_invalid=skipped_invalid,
            )
            _save_resume_state(resume_state_path, next_state)
        if use_ddp:
            bt = torch.tensor([best_val], dtype=torch.float64, device=device)
            dist.broadcast(bt, src=0)
            best_val = float(bt.item())
            dist.barrier()
        resume_step_in_epoch = 0

    if is_main:
        final_ckpt = _build_embedding_checkpoint(
            model,
            tokenizer,
            tok_meta,
            args,
            epoch=int(args.epochs),
            step=global_step,
        )
        _save_embedding_checkpoint(out_dir, "final", final_ckpt)
        tokenizer.save_pretrained(out_dir / "tokenizer")
        log_info("[ckpt] saved final embedding checkpoint")
        if resume_state_path.exists():
            resume_state_path.unlink()

        log_info("[done] training finished.")
        if skipped_nonfinite > 0:
            log_info(f"[done] skipped_nonfinite_batches={skipped_nonfinite}")
        if skipped_invalid > 0:
            log_info(f"[done] skipped_invalid_id_batches={skipped_invalid}")


if __name__ == "__main__":
    main()

