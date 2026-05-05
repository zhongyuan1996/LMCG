#!/usr/bin/env python3
"""
Stage 3 (Run H) evaluation -- sequential clinical chain generation.

Decodes:
  1. CXR image  (50-step Euler ODE from prior visit context).
  2. Report     (GT image as oracle prefix; "FINDINGS:\\n" seed for `--fair-compare`).
  3. ICD codes  (GT image + GT report as oracle prefix; greedy with `</ICD>` stop).

Metrics: SSIM/PSNR/FID for image; BLEU-1/4 + ROUGE-1/2/L (NLTK + rouge_score)
for report; per-window precision/recall/F1 over the legacy `<ICD9_*>` token set
and teacher-forced CE/PPL.

`--fair-compare` enables the LongCXR / HerGEN-matched protocol:
beam-width=3, report-max-new-tokens=128 (or whatever was set), pycocoevalcap
BLEU+ROUGE, ICD tokens masked from the report-generation prefix.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as _ssim_fn
from skimage.metrics import peak_signal_noise_ratio as _psnr_fn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from torchmetrics.image.fid import FrechetInceptionDistance as _FID
    _FID_AVAILABLE = True
except ImportError:
    _FID_AVAILABLE = False

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as _rouge_scorer_mod

# pycocoevalcap -- used when --fair-compare is set, matching LongCXR/HERGen protocol
try:
    from pycocoevalcap.bleu.bleu import Bleu as _CocoBleu
    from pycocoevalcap.rouge.rouge import Rouge as _CocoRouge
    _HAS_COCO_EVAL = True
except ImportError:
    _HAS_COCO_EVAL = False

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
    summarize_timelines,
)
from data_pipeline_stage3 import (  # noqa: E402
    Stage3ChainWindowDataset,
    collate_stage3_windows,
)
from stage3_text_masks import (  # noqa: E402
    collect_supervised_code_mask,
    collect_supervised_report_mask,
    mask_all_code_spans,
    mask_report_span,
)
from models import Showo2Qwen2_5, WanVAE  # noqa: E402
from models.misc import interpolate_pos_encoding  # noqa: E402
from transport import Sampler, create_transport  # noqa: E402
from utils.lora import LoraSpec, LoRALinear  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers shared with Stage 2 eval
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_causal_padding_mask(attn_1d: torch.Tensor) -> torch.Tensor:
    bsz, seqlen = attn_1d.shape
    dev = attn_1d.device
    valid = attn_1d.bool()
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=dev))
    keep = causal.unsqueeze(0) & valid.unsqueeze(1)
    mask = torch.zeros((bsz, 1, seqlen, seqlen), dtype=torch.bfloat16, device=dev)
    return mask.masked_fill(~keep.unsqueeze(1), -1e4)


def _build_omni_padding_mask(attn_1d: torch.Tensor, modality_positions: torch.Tensor) -> torch.Tensor:
    bsz, seqlen = attn_1d.shape
    dev = attn_1d.device
    valid = attn_1d.bool()
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=dev))
    keep = causal.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)
    if modality_positions is not None and modality_positions.numel() > 0:
        mp = modality_positions.to(device=dev)
        for b in range(bsz):
            for off, ln in mp[b]:
                s = max(0, int(off.item()))
                e = min(seqlen, s + int(ln.item()))
                if e > s:
                    keep[b, s:e, s:e] = True
    mask = torch.zeros((bsz, 1, seqlen, seqlen), dtype=torch.bfloat16, device=dev)
    return mask.masked_fill(~keep.unsqueeze(1), -1e4)


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


def _ce_sum_and_count(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    valid = flat_labels != ignore_index
    if not bool(valid.any()):
        z = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return z, torch.zeros((), device=logits.device, dtype=torch.long)
    ce = F.cross_entropy(
        flat_logits[valid],
        flat_labels[valid],
        reduction="sum",
    )
    return ce, valid.sum()


def _preprocess_image(path: str, resolution: int = 512) -> torch.Tensor:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("L").convert("RGB")
    w, h = img.size
    scale = float(resolution) / float(min(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), resample=Image.BICUBIC)
    left = (nw - resolution) // 2
    top = (nh - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _collect_image_slots(
    batch: Dict,
    *,
    device: torch.device,
    resolution: int,
    img_pad_id: int,
    pad_id: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    if int(input_ids.size(0)) != 1:
        raise ValueError("eval_stage3 assumes batch_size=1")

    positions = batch["modality_positions"][0]
    paths = batch["image_paths_for_slots"][0]

    text_masks = ((input_ids != img_pad_id) & (input_ids != pad_id)).long()
    image_masks = (input_ids == img_pad_id).long()

    if len(positions) == 0:
        return None, None, text_masks, image_masks

    imgs: List[torch.Tensor] = []
    valid_positions: List[Tuple[int, int]] = []
    for (off, ln), pth in zip(positions, paths):
        if not pth:
            continue
        p = Path(pth)
        if not p.exists():
            continue
        imgs.append(_preprocess_image(str(p), resolution=resolution))
        valid_positions.append((int(off), int(ln)))

    if len(imgs) == 0:
        return None, None, text_masks, image_masks

    pixel_values = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
    modality_positions = torch.tensor([valid_positions], dtype=torch.long, device=device)
    return pixel_values, modality_positions, text_masks, image_masks


def _find_block_span(
    input_ids_1d: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
) -> Optional[Tuple[int, int]]:
    """Locate the LAST [start_token ... end_token] block in input_ids."""
    ids = input_ids_1d.tolist()
    s = None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == start_token_id:
            s = i
            break
    if s is None:
        return None
    try:
        e = ids.index(end_token_id, s + 1)
        return s, e
    except ValueError:
        return None


def _decode_icd_token_set(tokenizer, token_ids: Sequence[int]) -> List[str]:
    """Legacy `<ICD9_*>` diagnosis tokens contained in `token_ids`."""
    out: List[str] = []
    for x in token_ids:
        t = tokenizer.convert_ids_to_tokens(int(x))
        if isinstance(t, str) and t.startswith("<ICD9_"):
            out.append(t)
    return sorted(set(out))


def _pr_f1_from_sets(gt_s: set, pr_s: set) -> Tuple[float, float, float]:
    tp = len(gt_s & pr_s)
    p = float(tp) / float(max(1, len(pr_s)))
    r = float(tp) / float(max(1, len(gt_s)))
    f1 = 0.0 if (p + r) == 0.0 else (2.0 * p * r) / (p + r)
    return p, r, f1


# ---------------------------------------------------------------------------
# LoRA injection (must match training setup exactly)
# ---------------------------------------------------------------------------

_LLM_LORA_TARGETS = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}


def _apply_lora_upper8(model: Showo2Qwen2_5, *, r: int, alpha: int, dropout: float) -> None:
    spec = LoraSpec(r=int(r), alpha=int(alpha), dropout=float(dropout))
    layers = getattr(model.showo.model, "layers", None)
    n_layers = len(layers)
    for li in range(max(0, n_layers - 8), n_layers):
        layer = layers[li]
        for name, child in list(layer.named_modules()):
            if name not in _LLM_LORA_TARGETS or not isinstance(child, torch.nn.Linear):
                continue
            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(child, spec))


def _apply_lora_diffusion_head(model: Showo2Qwen2_5, *, r: int, alpha: int, dropout: float) -> None:
    spec = LoraSpec(r=int(r), alpha=int(alpha), dropout=float(dropout))
    for layer in model.diffusion_head_a:
        for name, child in list(layer.named_modules()):
            if name not in _LLM_LORA_TARGETS or not isinstance(child, torch.nn.Linear):
                continue
            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(child, spec))


def _load_state_dict(model: torch.nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_state] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:5]}")


# ---------------------------------------------------------------------------
# Prefix-embed builder (identical to Stage 2 eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_prefix_embeds(
    model: Showo2Qwen2_5,
    vae: WanVAE,
    prefix_ids: Sequence[int],
    batch: Dict,
    *,
    device: torch.device,
    resolution: int,
    img_pad_id: int,
    pad_id: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    prefix_tensor = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
    input_embeds = model.showo.model.embed_tokens(prefix_tensor)
    dtype = input_embeds.dtype
    prefix_len = prefix_tensor.shape[1]

    pixel_values, modality_positions, _, _ = _collect_image_slots(
        batch, device=device, resolution=resolution,
        img_pad_id=img_pad_id, pad_id=pad_id,
    )
    if pixel_values is None or modality_positions is None:
        return input_embeds, None

    valid_idx, valid_mp = [], []
    for j, (off, ln) in enumerate(modality_positions[0]):
        if int(off.item()) + int(ln.item()) <= prefix_len:
            valid_idx.append(j)
            valid_mp.append((int(off.item()), int(ln.item())))
    if not valid_idx:
        return input_embeds, None

    px = pixel_values[valid_idx]
    x1 = vae.sample(px.unsqueeze(2)).squeeze(2).to(dtype=dtype)
    t = torch.ones(x1.size(0), device=device, dtype=dtype)

    p = model.config.patch_size
    h, w = x1.shape[2], x1.shape[3]
    h_, w_ = h // p, w // p

    img_und = model.image_embedder_und(x1)
    img_gen = model.image_embedder_gen(x1)

    if model.position_embedding.weight.shape[0] == model.image_position_ids.shape[-1]:
        img_und = img_und + model.position_embedding(model.image_position_ids)
    else:
        img_und = img_und + interpolate_pos_encoding(
            model.config.clip_latent_dim, model.position_embedding, h_, w_, 1,
        )
    img_und = model.und_trans(img_und)["last_hidden_state"]
    img_embs = model.fusion_proj(torch.cat([img_und, img_gen], dim=-1))

    time_embs = model.time_embed(t, dtype)
    te_proj = model.time_embed_proj(time_embs) if hasattr(model, "time_embed_proj") else time_embs

    for k, (off, ln) in enumerate(valid_mp):
        if model.config.add_time_embeds:
            input_embeds[0, off] = te_proj[k]
            input_embeds[0, off + 1:off + 1 + ln - 1] = img_embs[k, :max(ln - 1, 0)]
        else:
            input_embeds[0, off:off + ln] = img_embs[k, :ln]

    attn_1d   = torch.ones((1, prefix_len), dtype=torch.long, device=device)
    mp_tensor = torch.tensor([valid_mp], dtype=torch.long, device=device)
    omni_mask = _build_omni_padding_mask(attn_1d, mp_tensor).to(dtype=dtype)

    return input_embeds, omni_mask


def _mask_icd_spans_in_prefix(
    prefix_ids: List[int],
    tokenizer,
    pad_id: int,
) -> List[int]:
    """Replace legacy `<ICD>...</ICD>` span tokens with pad_id.

    Used to keep the model from peeking at GT diagnosis codes when the eval
    protocol asks for an oracle CXR + report (LongCXR/HerGEN-matched compare).
    """
    open_id  = int(tokenizer.convert_tokens_to_ids("<ICD>"))
    close_id = int(tokenizer.convert_tokens_to_ids("</ICD>"))

    result  = list(prefix_ids)
    in_span = False
    for i, tid in enumerate(result):
        if tid == open_id:
            in_span    = True
            result[i]  = pad_id
        elif tid == close_id and in_span:
            result[i]  = pad_id
            in_span    = False
        elif in_span:
            result[i]  = pad_id
    return result


@torch.no_grad()
def _greedy_generate_ids(
    model: Showo2Qwen2_5,
    vae: WanVAE,
    batch: Dict,
    *,
    prefix_ids: Sequence[int],
    stop_token_id: Optional[int],
    max_new_tokens: int,
    device: torch.device,
    resolution: int,
    img_pad_id: int,
    pad_id: int,
    blocked_token_ids: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
    repetition_penalty: float = 1.0,
    repetition_penalty_window: int = 32,
    stop_boost_start: float = 0.0,
    stop_boost_max: float = 0.0,
    gt_length_hint: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
    forced_prefix_ids: Optional[Sequence[int]] = None,
    ngram_hard_stop_n: int = 0,
    ngram_hard_stop_count: int = 3,
) -> List[int]:
    prefix_embeds, omni_mask = _build_prefix_embeds(
        model, vae, prefix_ids, batch,
        device=device, resolution=resolution,
        img_pad_id=img_pad_id, pad_id=pad_id,
    )
    out = model.showo(
        inputs_embeds=prefix_embeds,
        attention_mask=omni_mask,
        use_cache=True,
        return_dict=True,
    )
    past_kv = out.past_key_values
    logits = out.logits[:, -1, :]

    _boost_active = (stop_boost_max > 0.0 and gt_length_hint > 0)
    _boost_onset  = int(gt_length_hint * stop_boost_start) if _boost_active else 0
    _boost_ramp   = max(1, gt_length_hint - _boost_onset) if _boost_active else 1
    _use_sampling = (top_p > 0.0)

    generated: List[int] = []
    _forced = list(forced_prefix_ids) if forced_prefix_ids else []

    for step_i in range(int(max_new_tokens)):
        if step_i < len(_forced):
            nxt = int(_forced[step_i])
            generated.append(nxt)
            nxt_t = torch.tensor([[nxt]], dtype=torch.long, device=device)
            out = model.showo(input_ids=nxt_t, past_key_values=past_kv,
                              use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            continue

        if blocked_token_ids is not None:
            logits[:, blocked_token_ids] = -float("inf")

        if repetition_penalty != 1.0 and generated:
            recent = generated[-int(repetition_penalty_window):]
            for tok_id in set(recent):
                if logits[0, tok_id] > 0:
                    logits[0, tok_id] /= repetition_penalty
                else:
                    logits[0, tok_id] *= repetition_penalty

        if _boost_active and step_i >= _boost_onset and stop_token_id is not None:
            progress = min(1.0, (step_i - _boost_onset) / _boost_ramp)
            logits[0, int(stop_token_id)] += stop_boost_max * progress
            if eos_token_id is not None:
                logits[0, int(eos_token_id)] += stop_boost_max * progress

        if _use_sampling:
            scaled = logits[0] / max(temperature, 1e-8)
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = -float("inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            chosen = torch.multinomial(probs, num_samples=1).item()
            nxt = int(sorted_idx[chosen].item())
        else:
            nxt = int(torch.argmax(logits, dim=-1).item())

        generated.append(nxt)

        if ngram_hard_stop_n > 0 and len(generated) >= ngram_hard_stop_n * ngram_hard_stop_count:
            ng = tuple(generated[-ngram_hard_stop_n:])
            cnt = sum(
                1 for i in range(len(generated) - ngram_hard_stop_n)
                if tuple(generated[i:i + ngram_hard_stop_n]) == ng
            )
            if cnt >= ngram_hard_stop_count:
                break

        if stop_token_id is not None and nxt == int(stop_token_id):
            break
        if eos_token_id is not None and nxt == int(eos_token_id):
            break

        nxt_t = torch.tensor([[nxt]], dtype=torch.long, device=device)
        out = model.showo(input_ids=nxt_t, past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]

    return generated


@torch.no_grad()
def _beam_search_generate_ids(
    model: Showo2Qwen2_5,
    vae: WanVAE,
    batch: Dict,
    *,
    prefix_ids: Sequence[int],
    stop_token_id: int,
    max_new_tokens: int,
    device: torch.device,
    resolution: int,
    img_pad_id: int,
    pad_id: int,
    beam_width: int = 3,
    blocked_token_ids: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
    repetition_penalty: float = 1.0,
    repetition_penalty_window: int = 32,
    length_penalty: float = 1.0,
    forced_prefix_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    prefix_embeds, omni_mask = _build_prefix_embeds(
        model, vae, prefix_ids, batch,
        device=device, resolution=resolution,
        img_pad_id=img_pad_id, pad_id=pad_id,
    )
    out = model.showo(
        inputs_embeds=prefix_embeds,
        attention_mask=omni_mask,
        use_cache=True,
        return_dict=True,
    )
    past_kv = out.past_key_values
    logits = out.logits[:, -1, :]

    _forced = list(forced_prefix_ids) if forced_prefix_ids else []
    if _forced:
        for tok in _forced:
            nxt_t = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
            out = model.showo(input_ids=nxt_t, past_key_values=past_kv,
                              use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]

    beams: List[Tuple[float, List[int], object, torch.Tensor]] = [
        (0.0, list(_forced), past_kv, logits)
    ]
    finished: List[Tuple[float, List[int]]] = []

    for _step in range(int(max_new_tokens)):
        new_beams: List[Tuple[float, List[int], object, torch.Tensor]] = []
        for lp, toks, kv, lg in beams:
            cur_logits = lg.clone()
            if blocked_token_ids is not None:
                cur_logits[:, blocked_token_ids] = -float("inf")
            if repetition_penalty != 1.0 and toks:
                recent = toks[-int(repetition_penalty_window):]
                for tok_id in set(recent):
                    if cur_logits[0, tok_id] > 0:
                        cur_logits[0, tok_id] /= repetition_penalty
                    else:
                        cur_logits[0, tok_id] *= repetition_penalty
            log_probs = torch.log_softmax(cur_logits[0], dim=-1)
            top_lp, top_idx = torch.topk(log_probs, k=beam_width, dim=-1)
            for k in range(beam_width):
                tok = int(top_idx[k].item())
                new_lp = lp + float(top_lp[k].item())
                new_toks = toks + [tok]
                if tok == int(stop_token_id) or (eos_token_id is not None and tok == int(eos_token_id)):
                    score = new_lp / (max(1, len(new_toks)) ** length_penalty)
                    finished.append((score, new_toks))
                    continue
                nxt_t = torch.tensor([[tok]], dtype=torch.long, device=device)
                out = model.showo(input_ids=nxt_t, past_key_values=kv, use_cache=True, return_dict=True)
                new_beams.append((new_lp, new_toks, out.past_key_values, out.logits[:, -1, :]))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
        if len(finished) >= beam_width:
            break

    if finished:
        finished.sort(key=lambda x: x[0], reverse=True)
        return finished[0][1]
    beams_scored = [(lp / (max(1, len(t)) ** length_penalty), t) for lp, t, _, _ in beams]
    beams_scored.sort(key=lambda x: x[0], reverse=True)
    return beams_scored[0][1]


# ---------------------------------------------------------------------------
# Preview / per-modality generators
# ---------------------------------------------------------------------------

def _build_report_block_ids(tokenizer, device: torch.device) -> torch.Tensor:
    """Token IDs to suppress during report decoding (clinical codes + layout)."""
    blocked = []
    for tok, idx in tokenizer.get_added_vocab().items():
        if tok.startswith("<ICD9_") or tok in ("<ICD>", "</ICD>",
                                                "<VISIT_START>", "<VISIT_END>",
                                                "<|image_pad|>"):
            blocked.append(idx)
    return torch.tensor(blocked, dtype=torch.long, device=device)


def _report_generation_diagnostics(pred_text: str) -> Dict[str, float]:
    words = pred_text.lower().split()
    n_w = len(words)
    if n_w == 0:
        return {
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "max_repeat_line_fraction": 0.0,
            "line_distinct_1": 0.0,
        }
    distinct_1 = float(len(set(words))) / float(n_w)
    if n_w >= 2:
        bi = list(zip(words, words[1:]))
        distinct_2 = float(len(set(bi))) / float(len(bi))
    else:
        distinct_2 = 1.0
    lines = [ln.strip() for ln in pred_text.split("\n") if ln.strip()]
    if not lines:
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "max_repeat_line_fraction": 0.0,
            "line_distinct_1": 0.0,
        }
    c = Counter(lines)
    max_c = max(c.values())
    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "max_repeat_line_fraction": float(max_c) / float(len(lines)),
        "line_distinct_1": float(len(set(lines))) / float(len(lines)),
    }


@torch.no_grad()
def _generate_text_preview(
    model: Showo2Qwen2_5,
    vae: WanVAE,
    tokenizer,
    input_ids_1d: torch.Tensor,
    batch: Dict,
    *,
    task: str,
    resolution: int,
    img_pad_id: int,
    pad_id: int,
    repetition_penalty: float = 1.0,
    repetition_penalty_window: int = 32,
    stop_boost_start: float = 0.5,
    stop_boost_max: float = 0.0,
    top_p: float = 0.0,
    temperature: float = 1.0,
    report_prefix: str = "",
    beam_width: int = 1,
    length_penalty: float = 1.0,
    report_max_new_tokens: Optional[int] = None,
    mask_icd_for_report: bool = False,
    ngram_hard_stop_n: int = 0,
    ngram_hard_stop_count: int = 3,
) -> Dict[str, object]:
    if task == "icd":
        lo = int(batch["target_code_seq_start"][0].item())
        hi = int(batch["target_code_seq_end"][0].item())
        if lo >= 0 and hi >= lo:
            prefix = [int(x) for x in input_ids_1d[: lo + 1].tolist()]
            target_ids = [int(x) for x in input_ids_1d[lo + 1 : hi + 1].tolist()]
            stop_tok_id = int(tokenizer.convert_tokens_to_ids("</ICD>"))
        else:
            start_tok_id = int(tokenizer.convert_tokens_to_ids("<ICD>"))
            stop_tok_id = int(tokenizer.convert_tokens_to_ids("</ICD>"))
            span = _find_block_span(input_ids_1d, start_tok_id, stop_tok_id)
            if span is None:
                return {"task": task, "error": "block_not_found_in_input_ids"}
            s, e = span
            prefix = [int(x) for x in input_ids_1d[: s + 1].tolist()]
            target_ids = [int(x) for x in input_ids_1d[s + 1 : e + 1].tolist()]
        block_span = [int(lo), int(hi)] if (lo >= 0 and hi >= lo) else [int(s), int(e)]
    else:
        start_tok_id = int(tokenizer.convert_tokens_to_ids("<REPORT>"))
        stop_tok_id = int(tokenizer.convert_tokens_to_ids("</REPORT>"))
        span = _find_block_span(input_ids_1d, start_tok_id, stop_tok_id)
        if span is None:
            return {"task": task, "error": "block_not_found_in_input_ids"}
        s, e = span
        prefix = [int(x) for x in input_ids_1d[: s + 1].tolist()]
        target_ids = [int(x) for x in input_ids_1d[s + 1 : e + 1].tolist()]
        if mask_icd_for_report:
            prefix = _mask_icd_spans_in_prefix(prefix, tokenizer, pad_id)
        block_span = [int(s), int(e)]

    if task == "report":
        if report_max_new_tokens is not None and int(report_max_new_tokens) > 0:
            max_new = max(1, int(report_max_new_tokens))
        else:
            max_new = max(256, len(target_ids) * 2)
    else:
        max_new = max(1, len(target_ids))

    rep_pen = repetition_penalty if task == "report" else 1.0
    rep_win = repetition_penalty_window
    eos_id  = getattr(tokenizer, "eos_token_id", None)

    block_ids = (
        _build_report_block_ids(tokenizer, input_ids_1d.device)
        if task == "report" else None
    )

    boost_max = stop_boost_max if task == "report" else 0.0
    gt_len    = len(target_ids)

    forced = None
    if task == "report" and report_prefix:
        forced = tokenizer.encode(report_prefix, add_special_tokens=False)

    use_beam = (beam_width > 1 and task == "report")

    if use_beam:
        pred_ids = _beam_search_generate_ids(
            model=model, vae=vae, batch=batch,
            prefix_ids=prefix, stop_token_id=stop_tok_id,
            max_new_tokens=max_new,
            device=input_ids_1d.device, resolution=resolution,
            img_pad_id=img_pad_id, pad_id=pad_id,
            beam_width=beam_width,
            blocked_token_ids=block_ids,
            eos_token_id=eos_id,
            repetition_penalty=rep_pen,
            repetition_penalty_window=rep_win,
            length_penalty=length_penalty,
            forced_prefix_ids=forced,
        )
    else:
        gen_top_p = top_p if task == "report" else 0.0
        gen_temp  = temperature if task == "report" else 1.0
        pred_ids = _greedy_generate_ids(
            model=model, vae=vae, batch=batch,
            prefix_ids=prefix, stop_token_id=stop_tok_id,
            max_new_tokens=max_new,
            device=input_ids_1d.device, resolution=resolution,
            img_pad_id=img_pad_id, pad_id=pad_id,
            blocked_token_ids=block_ids,
            eos_token_id=eos_id,
            repetition_penalty=rep_pen,
            repetition_penalty_window=rep_win,
            stop_boost_start=stop_boost_start,
            stop_boost_max=boost_max,
            gt_length_hint=gt_len,
            top_p=gen_top_p,
            temperature=gen_temp,
            forced_prefix_ids=forced,
            ngram_hard_stop_n=ngram_hard_stop_n if task == "report" else 0,
            ngram_hard_stop_count=ngram_hard_stop_count,
        )

    if task == "icd":
        gt_set = _decode_icd_token_set(tokenizer, target_ids)
        pr_set = _decode_icd_token_set(tokenizer, pred_ids)
        gt_s, pr_s = set(gt_set), set(pr_set)
        p, r, f1 = _pr_f1_from_sets(gt_s, pr_s)
        ended_eos = (
            eos_id is not None
            and len(pred_ids) > 0
            and int(pred_ids[-1]) == int(eos_id)
        )
        return {
            "task": task,
            "block_span": block_span,
            "gt_icd_tokens": gt_set,
            "pred_icd_tokens": pr_set,
            "precision": p,
            "recall": r,
            "f1": f1,
            "gen_debug": {
                "pred_raw_len": int(len(pred_ids)),
                "gt_span_len": int(len(target_ids)),
                "max_new_cap": int(max_new),
                "ended_with_eos": bool(ended_eos),
            },
        }

    gt_txt = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
    pr_txt = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

    ended_by_stop  = len(pred_ids) > 0 and pred_ids[-1] == stop_tok_id
    ended_by_eos   = (eos_id is not None and len(pred_ids) > 0
                      and pred_ids[-1] == int(eos_id))
    stop_reason    = ("stop_token" if ended_by_stop
                      else "eos" if ended_by_eos
                      else "max_tokens")

    dec = _report_generation_diagnostics(pr_txt)
    return {
        "task": task,
        "block_span": [s, e],
        "gt_report_text": gt_txt,
        "pred_report_text": pr_txt,
        "gt_tokens": len(target_ids),
        "pred_tokens": len(pred_ids),
        "max_new_tokens": max_new,
        "stop_reason": stop_reason,
        "decode_stats": dec,
    }


@torch.no_grad()
def _generate_image_preview(
    model: Showo2Qwen2_5,
    vae: WanVAE,
    sampler: Sampler,
    batch: Dict,
    *,
    device: torch.device,
    resolution: int,
    max_seq_len: int,
    img_pad_id: int,
    pad_id: int,
    out_png: Optional[Path] = None,
) -> Dict[str, object]:
    input_ids = batch["input_ids"].to(device)
    pixel_values, modality_positions, _text_masks, _image_masks = _collect_image_slots(
        batch, device=device, resolution=resolution,
        img_pad_id=img_pad_id, pad_id=pad_id,
    )
    if pixel_values is None or modality_positions is None:
        return {"task": "image", "error": "no_image_slots_available"}

    attn = _build_omni_padding_mask(batch["attention_mask"].to(device), modality_positions)
    x1 = vae.sample(pixel_values.unsqueeze(2)).squeeze(2).float()

    z = x1.clone()
    z[-1] = torch.randn_like(x1[-1])

    sample_fn = sampler.sample_ode(
        sampling_method="euler",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        do_shift=True,
        time_shifting_factor=3.0,
    )
    model_kwargs = dict(
        text_tokens=input_ids,
        attention_mask=attn,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=int(max_seq_len),
        guidance_scale=0.0,
        only_denoise_last_image=True,
    )
    samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
    gen = vae.batch_decode(samples.unsqueeze(2)).squeeze(2)
    gen = ((gen.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    pred_np = np.transpose(gen[-1], (1, 2, 0))

    gt_decoded = vae.batch_decode(x1[-1:].unsqueeze(2)).squeeze(2)
    gt_np = ((gt_decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    gt_np = np.transpose(gt_np[0], (1, 2, 0))

    try:
        min_dim = min(pred_np.shape[0], pred_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        ssim_val = float(_ssim_fn(gt_np, pred_np, channel_axis=2,
                                   data_range=255, win_size=win_size))
        psnr_val = float(_psnr_fn(gt_np, pred_np, data_range=255))
    except Exception as e:
        ssim_val, psnr_val = None, None
        print(f"[warn] image metrics failed: {e}")

    result = {
        "task": "image",
        "ssim": ssim_val,
        "psnr": psnr_val,
        "_pred_np": pred_np,
        "_gt_np": gt_np,
    }
    if out_png is not None:
        im = Image.fromarray(pred_np)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        im.save(str(out_png))
        result["generated_image"] = str(out_png)
    return result


def _extract_sample(batch: Dict, idx: int) -> Dict:
    out: Dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx:idx + 1]
        elif isinstance(v, list):
            out[k] = [v[idx]]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained",           type=str, default="showlab/show-o2-1.5B-HQ")
    ap.add_argument("--tokenizer-model",      type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--stage1-tokenizer-dir", type=str, default="")
    ap.add_argument("--matching-pkl",  type=str, default="${MATCHING_PKL}")
    ap.add_argument("--jpg-root",      type=str, default="${MIMIC_CXR_JPG_ROOT}")
    ap.add_argument("--vae-pth",       type=str, required=True)
    ap.add_argument("--state-dict-path", type=str, required=True,
                    help="Path to Stage 3 checkpoint (.pt)")
    # LoRA config -- must match training
    ap.add_argument("--lora-r",          type=int,   default=64)
    ap.add_argument("--lora-alpha",      type=int,   default=128)
    ap.add_argument("--lora-dropout",    type=float, default=0.0)
    ap.add_argument("--lora-diff-r",     type=int,   default=16)
    ap.add_argument("--lora-diff-alpha", type=int,   default=32)
    # Decoding hyperparameters
    ap.add_argument("--rep-penalty",        type=float, default=1.3)
    ap.add_argument("--rep-penalty-window", type=int,   default=32)
    ap.add_argument("--stop-boost-start",  type=float, default=0.5)
    ap.add_argument("--stop-boost-max",    type=float, default=5.0)
    ap.add_argument("--ngram-hard-stop-n", type=int, default=0)
    ap.add_argument("--ngram-hard-stop-count", type=int, default=3)
    ap.add_argument("--top-p",             type=float, default=0.0)
    ap.add_argument("--temperature",       type=float, default=0.8)
    ap.add_argument("--report-prefix",     type=str,   default="",
                    help="Force this string as the start of every generated report.")
    ap.add_argument("--beam-width",        type=int,   default=1)
    ap.add_argument("--length-penalty",    type=float, default=1.0)
    # Data
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--train-ratio",     type=float, default=0.8)
    ap.add_argument("--val-ratio",       type=float, default=0.1)
    ap.add_argument("--k-max",           type=int,   default=4)
    ap.add_argument("--max-seq-len",     type=int,   default=3072)
    ap.add_argument("--keep-last-n-ctx-images", type=int, default=1)
    ap.add_argument("--report-max-tokens",      type=int, default=192)
    ap.add_argument("--num-image-tokens",       type=int, default=1024)
    ap.add_argument("--latent-h",        type=int,   default=32)
    ap.add_argument("--latent-w",        type=int,   default=32)
    ap.add_argument("--image-resolution",type=int,   default=512)
    ap.add_argument("--max-eval-samples", type=int,  default=512)
    ap.add_argument("--eval-indices-json", type=str, default="",
                    help="If set: load val window indices from this JSON array; "
                         "if the file does not exist, sample --max-eval-samples and save it.")
    ap.add_argument("--report-max-new-tokens", type=int, default=0,
                    help="Hard cap on generated report tokens; 0 = use max(256, 2 * |GT|).")
    ap.add_argument("--preview-samples", type=int,   default=8)
    ap.add_argument("--fair-compare", action="store_true", default=False,
                    help="Use pycocoevalcap (Bleu+Rouge) for report metrics. "
                         "Also sets beam-width=3 and report-max-new-tokens=128 if not "
                         "already overridden.")
    ap.add_argument("--mask-icd-for-report", action="store_true", default=False,
                    help="Replace ICD span tokens in the prefix with pad_id before report "
                         "generation so the model only attends to image + prior report context.")
    ap.add_argument("--output-dir",      type=str,
                    default=str(Path(__file__).resolve().parent / "outputs_eval_stage3"))
    ap.add_argument("--eval-split", type=str, default="val", choices=("train", "val", "test"))
    ap.add_argument("--rank",       type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--save-images-dir", type=str, default=None)
    ap.add_argument("--save-reports-json", type=str, default=None)
    args = ap.parse_args()

    if bool(args.fair_compare):
        if not _HAS_COCO_EVAL:
            raise RuntimeError("--fair-compare requires pycocoevalcap. "
                               "Install with: pip install pycocoevalcap")
        if int(args.beam_width) == 1:
            args.beam_width = 3
        if int(args.report_max_new_tokens) == 0:
            args.report_max_new_tokens = 128
        print(f"[fair-compare] beam_width={args.beam_width}  "
              f"report_max_new_tokens={args.report_max_new_tokens}  "
              f"metric_lib=pycocoevalcap")

    _seed_all(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    save_images_dir = Path(args.save_images_dir) if args.save_images_dir else None
    if save_images_dir is not None:
        save_images_dir.mkdir(parents=True, exist_ok=True)
    save_reports_json = Path(args.save_reports_json) if args.save_reports_json else None
    if save_reports_json is not None:
        save_reports_json.parent.mkdir(parents=True, exist_ok=True)
    _saved_reports: Dict[str, str] = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    # ---- Tokenizer ----
    tok_source = args.stage1_tokenizer_dir.strip() if args.stage1_tokenizer_dir else ""
    tokenizer = AutoTokenizer.from_pretrained(tok_source or args.tokenizer_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    add_stage2_special_tokens(tokenizer)

    img_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    pad_id = int(tokenizer.pad_token_id)

    # ---- Data ----
    timelines = load_patient_timelines_from_matching_pkl(
        Path(args.matching_pkl), jpg_root=Path(args.jpg_root),
    )
    splits = split_subjects(
        subject_ids=sorted(timelines.keys()),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    eval_split = str(args.eval_split)
    val_tl = {sid: timelines[sid] for sid in splits[eval_split]}
    cov = summarize_timelines(val_tl)
    print(f"[eval] {eval_split}_timeline_code_coverage:", cov)

    ds = Stage3ChainWindowDataset(
        timelines=val_tl,
        tokenizer=tokenizer,
        split_name=eval_split,
        k_max=args.k_max,
        max_seq_len=args.max_seq_len,
        keep_last_n_ctx_images=args.keep_last_n_ctx_images,
        report_max_tokens=args.report_max_tokens,
        num_image_tokens=args.num_image_tokens,
        add_time_embeds=True,
        seed=args.seed,
    )
    print(f"[eval] {eval_split}_window_counts:", ds.window_counts())

    n_total = len(ds)
    rng = random.Random(args.seed)
    indices_path = str(args.eval_indices_json).strip()
    if indices_path:
        p_idx = Path(indices_path)
        if p_idx.exists():
            raw = json.loads(p_idx.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("--eval-indices-json must contain a JSON list of integers")
            eval_indices = sorted(int(x) for x in raw)
            for x in eval_indices:
                if x < 0 or x >= n_total:
                    raise ValueError(f"eval index {x} out of range for val set size {n_total}")
            print(f"[eval] loaded {len(eval_indices)} indices from {p_idx}")
        else:
            n_eval = min(int(args.max_eval_samples), n_total)
            eval_indices = sorted(rng.sample(range(n_total), n_eval))
            p_idx.parent.mkdir(parents=True, exist_ok=True)
            p_idx.write_text(json.dumps(eval_indices), encoding="utf-8")
            print(f"[eval] sampled {len(eval_indices)} indices and saved to {p_idx}")
    else:
        n_eval = min(int(args.max_eval_samples), n_total)
        eval_indices = sorted(rng.sample(range(n_total), n_eval))

    rank       = int(args.rank)
    world_size = int(args.world_size)
    if world_size > 1:
        eval_indices = eval_indices[rank::world_size]
        print(f"[eval] rank={rank}/{world_size}: {len(eval_indices)} windows assigned")

    sub = Subset(ds, eval_indices)
    dl = DataLoader(
        sub,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_stage3_windows(b, pad_token_id=pad_id),
    )

    # ---- Model ----
    model = Showo2Qwen2_5.from_pretrained(
        args.pretrained, use_safetensors=False, torch_dtype=torch.bfloat16
    ).to(device)
    model.showo.resize_token_embeddings(len(tokenizer))
    model.showo.tie_weights()
    model.config.llm_vocab_size = int(len(tokenizer))
    model.config.image_latent_height = int(args.latent_h)
    model.config.image_latent_width = int(args.latent_w)
    model.image_position_ids = torch.arange(
        int(args.latent_h) * int(args.latent_w), device=device
    ).expand((1, -1))

    _apply_lora_diffusion_head(
        model, r=int(args.lora_diff_r), alpha=int(args.lora_diff_alpha), dropout=float(args.lora_dropout),
    )
    _apply_lora_upper8(
        model, r=int(args.lora_r), alpha=int(args.lora_alpha), dropout=float(args.lora_dropout),
    )

    _load_state_dict(model, args.state_dict_path)
    model.eval()

    # ---- VAE + transport ----
    vae = WanVAE(vae_pth=str(args.vae_pth), dtype=torch.bfloat16, device=device)
    transport = create_transport(
        path_type="Linear", prediction="velocity",
        loss_weight=None, train_eps=None, sample_eps=None,
        snr_type="lognorm", do_shift=True, seq_len=16,
    )
    sampler_obj = Sampler(transport)

    # ---- Eval loop ----
    stats: Dict[str, Dict] = {
        "image":  {"sum": 0.0, "n": 0},
        "report": {"sum": 0.0, "n": 0},
        "icd":    {"sum": 0.0, "n": 0},
        "total":  {"sum": 0.0, "n": 0},
    }
    previews_done = 0
    all_previews: List[Dict] = []
    all_image_results: List[Dict] = []
    all_report_results: List[Dict] = []
    all_icd_results: List[Dict] = []

    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in tqdm(dl, desc="eval-stage3"):
            actual_bsz = int(batch["input_ids"].shape[0])

            for b_idx in range(actual_bsz):
                sb = _extract_sample(batch, b_idx)

                has_img    = bool(sb["has_target_image"][0].item())
                has_report = bool(sb["has_target_report"][0].item())
                has_icd    = bool(sb["has_target_icd"][0].item())

                input_ids         = sb["input_ids"].to(device)
                labels            = sb["labels"].to(device)
                attention_mask_1d = sb["attention_mask"].to(device)

                pixel_values, modality_positions, text_masks, image_masks = _collect_image_slots(
                    sb, device=device, resolution=int(args.image_resolution),
                    img_pad_id=img_pad_id, pad_id=pad_id,
                )

                if modality_positions is not None:
                    attn = _build_omni_padding_mask(attention_mask_1d, modality_positions)
                else:
                    attn = _build_causal_padding_mask(attention_mask_1d)

                xt, ut, t_vec = None, None, None
                if pixel_values is not None and modality_positions is not None:
                    x1 = vae.sample(pixel_values.unsqueeze(2)).squeeze(2)
                    sup = sb["image_supervise_masks"][0].to(device=device, dtype=torch.long)
                    xt = x1.clone()
                    ut = torch.zeros_like(x1)
                    t_vec = torch.ones((x1.size(0),), device=device, dtype=torch.float32)
                    for i in range(x1.size(0)):
                        if int(sup[i].item()) != 1:
                            continue
                        t_i, x0, x1_i = transport.sample(x1[i][None])
                        t_i, xt_i, ut_i = transport.path_sampler.plan(t_i, x0, x1_i)
                        t_vec[i] = t_i.squeeze(0)
                        xt[i] = xt_i.squeeze(0)
                        ut[i] = ut_i.squeeze(0)
                    xt = xt.to(dtype=torch.bfloat16)
                    ut = ut.to(dtype=torch.bfloat16)
                    t_vec = t_vec.to(device=device, dtype=torch.bfloat16)

                    pos = modality_positions[0]
                    for j, (off, ln) in enumerate(pos):
                        if int(sup[j].item()) == 0:
                            image_masks[0, int(off.item()):int(off.item() + ln.item())] = 0

                need_text_loss  = has_report or has_icd
                need_image_loss = has_img and (xt is not None)

                text_labels_arg  = labels if need_text_loss  else None
                image_labels_arg = ut     if need_image_loss else None

                loss_img  = torch.zeros((), device=device)
                _logits   = None

                if xt is not None:
                    forward_out = model(
                        text_tokens=input_ids,
                        image_latents=xt,
                        t=t_vec,
                        attention_mask=attn,
                        text_masks=text_masks,
                        image_masks=image_masks,
                        text_labels=text_labels_arg,
                        image_labels=image_labels_arg,
                        modality_positions=modality_positions,
                        output_hidden_states=True,
                        max_seq_len=input_ids.size(1),
                        device=device,
                    )
                    if need_image_loss and need_text_loss:
                        _logits, _, loss_img = forward_out
                    elif need_image_loss:
                        _logits, loss_img = forward_out
                    elif need_text_loss:
                        _logits, _ = forward_out
                else:
                    out = model(
                        text_tokens=input_ids,
                        image_latents=None,
                        attention_mask=attn,
                        output_hidden_states=False,
                        max_seq_len=input_ids.size(1),
                        device=device,
                    )
                    _logits = out.logits if hasattr(out, "logits") else out["logits"]

                loss_rep = torch.zeros((), device=device)
                loss_icd = torch.zeros((), device=device)
                if need_text_loss and _logits is not None:
                    if has_report:
                        rep_labels = mask_all_code_spans(labels[0], input_ids[0], tokenizer).unsqueeze(0)
                        loss_rep = _ce_loss(_logits, rep_labels)
                    if has_icd:
                        icd_labels_row = mask_report_span(labels[0], input_ids[0], tokenizer)
                        icd_labels = icd_labels_row.unsqueeze(0)
                        loss_icd = _ce_loss(_logits, icd_labels)

                img_v = float(loss_img.detach().cpu())
                rep_v = float(loss_rep.detach().cpu())
                icd_v = float(loss_icd.detach().cpu())
                total_v = rep_v + icd_v + img_v

                if need_image_loss:
                    stats["image"]["sum"] += img_v
                    stats["image"]["n"]   += 1
                if has_report:
                    stats["report"]["sum"] += rep_v
                    stats["report"]["n"]   += 1
                if has_icd:
                    stats["icd"]["sum"] += icd_v
                    stats["icd"]["n"]   += 1
                stats["total"]["sum"] += total_v
                stats["total"]["n"]   += 1

                # ---- Generation metrics + saved previews ----
                save_preview = previews_done < int(args.preview_samples)
                image_result = None
                report_result = None
                icd_result = None

                if has_img:
                    png_path = preview_dir / f"image_preview_{previews_done:03d}.png" if save_preview else None
                    if save_images_dir is not None:
                        import re as _re
                        _img_paths = sb.get("image_paths_for_slots", [[]])[0]
                        _target_path = str(_img_paths[-1]) if _img_paths else ""
                        _m = _re.search(r'/s(\d+)/', _target_path)
                        _sid = _m.group(1) if _m else Path(_target_path).stem
                        png_path = save_images_dir / f"{_sid}.png"
                    image_result = _generate_image_preview(
                        model=model, vae=vae, sampler=sampler_obj, batch=sb,
                        device=device, resolution=int(args.image_resolution),
                        max_seq_len=int(args.max_seq_len),
                        img_pad_id=img_pad_id, pad_id=pad_id,
                        out_png=png_path,
                    )
                    if image_result.get("ssim") is not None:
                        all_image_results.append(image_result)

                if has_report:
                    report_result = _generate_text_preview(
                        model=model, vae=vae, tokenizer=tokenizer,
                        input_ids_1d=input_ids[0],
                        batch=sb, task="report",
                        resolution=int(args.image_resolution),
                        img_pad_id=img_pad_id, pad_id=pad_id,
                        repetition_penalty=float(args.rep_penalty),
                        repetition_penalty_window=int(args.rep_penalty_window),
                        stop_boost_start=float(args.stop_boost_start),
                        stop_boost_max=float(args.stop_boost_max),
                        top_p=float(args.top_p),
                        temperature=float(args.temperature),
                        report_prefix=str(args.report_prefix),
                        beam_width=int(args.beam_width),
                        length_penalty=float(args.length_penalty),
                        report_max_new_tokens=(
                            int(args.report_max_new_tokens)
                            if int(args.report_max_new_tokens) > 0
                            else None
                        ),
                        mask_icd_for_report=bool(args.mask_icd_for_report),
                        ngram_hard_stop_n=int(args.ngram_hard_stop_n),
                        ngram_hard_stop_count=int(args.ngram_hard_stop_count),
                    )
                    if "gt_report_text" in report_result:
                        all_report_results.append(report_result)
                        if save_reports_json is not None:
                            import re as _re2
                            _img_paths = sb.get("image_paths_for_slots", [[]])[0]
                            _target_path = str(_img_paths[-1]) if _img_paths else ""
                            _m2 = _re2.search(r'/s(\d+)/', _target_path)
                            _rsid = _m2.group(1) if _m2 else Path(_target_path).stem
                            _saved_reports[_rsid] = report_result["pred_report_text"]

                if has_icd:
                    icd_result = _generate_text_preview(
                        model=model, vae=vae, tokenizer=tokenizer,
                        input_ids_1d=input_ids[0],
                        batch=sb, task="icd",
                        resolution=int(args.image_resolution),
                        img_pad_id=img_pad_id, pad_id=pad_id,
                        repetition_penalty=float(args.rep_penalty),
                        repetition_penalty_window=int(args.rep_penalty_window),
                        stop_boost_start=float(args.stop_boost_start),
                        stop_boost_max=float(args.stop_boost_max),
                    )
                    if "f1" in icd_result:
                        if "subject_id" in sb:
                            icd_result["subject_id"] = int(sb["subject_id"][0].item())
                        all_icd_results.append(icd_result)

                if save_preview:
                    preview_entry: Dict = {"sample_idx": previews_done}
                    if image_result is not None:
                        preview_entry["image"] = image_result
                    if report_result is not None:
                        preview_entry["report"] = report_result
                    if icd_result is not None:
                        preview_entry["icd"] = icd_result
                    all_previews.append(preview_entry)
                    previews_done += 1

    # ---- Aggregate loss metrics ----
    metrics = {
        k: {
            "avg_loss": v["sum"] / float(max(1, v["n"])),
            "num_samples": v["n"],
        }
        for k, v in stats.items()
    }

    # ---- Report NLG metrics ----
    if all_report_results:
        avg_rep_loss = metrics["report"]["avg_loss"]
        metrics["report"]["ppl"] = float(math.exp(min(avg_rep_loss, 20.0)))

        if bool(args.fair_compare) and _HAS_COCO_EVAL:
            gts = {i: [r["gt_report_text"]]   for i, r in enumerate(all_report_results)}
            res = {i: [r["pred_report_text"]]  for i, r in enumerate(all_report_results)}
            _bleu_scorer = _CocoBleu(4)
            try:
                _bleu_scores, _ = _bleu_scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                _bleu_scores, _ = _bleu_scorer.compute_score(gts, res)
            metrics["report"]["bleu_1"] = float(_bleu_scores[0])
            metrics["report"]["bleu_2"] = float(_bleu_scores[1])
            metrics["report"]["bleu_3"] = float(_bleu_scores[2])
            metrics["report"]["bleu_4"] = float(_bleu_scores[3])
            _rouge_scorer2 = _CocoRouge()
            try:
                _rouge_score, _ = _rouge_scorer2.compute_score(gts, res, verbose=0)
            except TypeError:
                _rouge_score, _ = _rouge_scorer2.compute_score(gts, res)
            metrics["report"]["rouge_l"] = float(_rouge_score)
            metrics["report"]["metric_lib"] = "pycocoevalcap"
        else:
            refs_bleu = [[r["gt_report_text"].split()] for r in all_report_results]
            hyps_bleu = [r["pred_report_text"].split() for r in all_report_results]
            smooth = SmoothingFunction().method1
            metrics["report"]["bleu_1"] = float(corpus_bleu(
                refs_bleu, hyps_bleu, weights=(1.0, 0, 0, 0), smoothing_function=smooth))
            metrics["report"]["bleu_4"] = float(corpus_bleu(
                refs_bleu, hyps_bleu, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
            scorer = _rouge_scorer_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            rouge_1s, rouge_2s, rouge_ls = [], [], []
            for r in all_report_results:
                sc = scorer.score(r["gt_report_text"], r["pred_report_text"])
                rouge_1s.append(sc["rouge1"].fmeasure)
                rouge_2s.append(sc["rouge2"].fmeasure)
                rouge_ls.append(sc["rougeL"].fmeasure)
            metrics["report"]["rouge_1"] = float(np.mean(rouge_1s))
            metrics["report"]["rouge_2"] = float(np.mean(rouge_2s))
            metrics["report"]["rouge_l"] = float(np.mean(rouge_ls))
            metrics["report"]["metric_lib"] = "nltk+rouge_score"
        metrics["report"]["num_generated"] = len(all_report_results)

        stop_counts = Counter(r.get("stop_reason", "unknown") for r in all_report_results)
        metrics["report"]["stop_reasons"] = dict(stop_counts)

        d1, d2, mx_rep, ln_d1 = [], [], [], []
        for r in all_report_results:
            st = r.get("decode_stats") or {}
            d1.append(float(st.get("distinct_1", 0.0)))
            d2.append(float(st.get("distinct_2", 0.0)))
            mx_rep.append(float(st.get("max_repeat_line_fraction", 0.0)))
            ln_d1.append(float(st.get("line_distinct_1", 0.0)))
        metrics["report"]["distinct_1_mean"] = float(np.mean(d1))
        metrics["report"]["distinct_2_mean"] = float(np.mean(d2))
        metrics["report"]["max_repeat_line_fraction_mean"] = float(np.mean(mx_rep))
        metrics["report"]["line_distinct_1_mean"] = float(np.mean(ln_d1))

    # ---- Image metrics ----
    if all_image_results:
        ssims = [p["ssim"] for p in all_image_results if p.get("ssim") is not None]
        psnrs = [p["psnr"] for p in all_image_results if p.get("psnr") is not None]
        if ssims:
            metrics["image"]["ssim_mean"] = float(np.mean(ssims))
            metrics["image"]["ssim_std"]  = float(np.std(ssims))
        if psnrs:
            metrics["image"]["psnr_mean"] = float(np.mean(psnrs))
            metrics["image"]["psnr_std"]  = float(np.std(psnrs))
        metrics["image"]["num_generated"] = len(all_image_results)

        if _FID_AVAILABLE and len(all_image_results) >= 2:
            try:
                fid_metric = _FID(feature=2048, normalize=True).to(device)
                for r in all_image_results:
                    pred_np_img = r.get("_pred_np")
                    gt_np_img = r.get("_gt_np")
                    if pred_np_img is None or gt_np_img is None:
                        continue
                    pred_t = torch.from_numpy(pred_np_img).permute(2, 0, 1).unsqueeze(0)
                    gt_t   = torch.from_numpy(gt_np_img).permute(2, 0, 1).unsqueeze(0)
                    fid_metric.update(pred_t.to(device), real=False)
                    fid_metric.update(gt_t.to(device),   real=True)
                fid_val = float(fid_metric.compute().item())
                metrics["image"]["fid"] = fid_val
                print(f"[eval] FID = {fid_val:.4f}  (n={len(all_image_results)})")
            except Exception as e:
                print(f"[warn] FID computation failed: {e}")
        else:
            if not _FID_AVAILABLE:
                print("[warn] torchmetrics not available -- FID skipped")

        for r in all_image_results:
            r.pop("_pred_np", None)
            r.pop("_gt_np", None)

    # ---- ICD metrics: macro + micro P/R/F1, PPL ----
    if all_icd_results:
        precs   = [p["precision"] for p in all_icd_results]
        recalls = [p["recall"]    for p in all_icd_results]
        f1s     = [p["f1"]        for p in all_icd_results]
        metrics["icd"]["precision_mean"] = float(np.mean(precs))
        metrics["icd"]["recall_mean"]    = float(np.mean(recalls))
        metrics["icd"]["f1_mean"]        = float(np.mean(f1s))
        metrics["icd"]["f1_std"]         = float(np.std(f1s))
        metrics["icd"]["num_generated"]  = len(all_icd_results)
        avg_icd_loss = metrics["icd"]["avg_loss"]
        metrics["icd"]["ppl"] = float(math.exp(min(avg_icd_loss, 20.0)))

        sum_tp = sum(
            len(set(p["gt_icd_tokens"]) & set(p["pred_icd_tokens"]))
            for p in all_icd_results
        )
        sum_pred = sum(len(p["pred_icd_tokens"]) for p in all_icd_results)
        sum_gt = sum(len(p["gt_icd_tokens"]) for p in all_icd_results)
        mic_p = float(sum_tp) / float(max(1, sum_pred))
        mic_r = float(sum_tp) / float(max(1, sum_gt))
        mic_f1 = 0.0 if (mic_p + mic_r) == 0.0 else (2.0 * mic_p * mic_r) / (mic_p + mic_r)
        metrics["icd"]["precision_micro"] = mic_p
        metrics["icd"]["recall_micro"] = mic_r
        metrics["icd"]["f1_micro"] = mic_f1
        n_ex = sum(
            1 for p in all_icd_results
            if set(p["gt_icd_tokens"]) == set(p["pred_icd_tokens"])
        )
        metrics["icd"]["exact_set_match_rate"] = float(n_ex) / float(len(all_icd_results))

        # Run H is diagnosis-only; surface as `diag` for downstream consistency.
        metrics["icd"]["diag"] = {
            "precision_mean": metrics["icd"]["precision_mean"],
            "recall_mean":    metrics["icd"]["recall_mean"],
            "f1_mean":        metrics["icd"]["f1_mean"],
            "f1_std":         metrics["icd"]["f1_std"],
            "precision_micro": mic_p,
            "recall_micro":   mic_r,
            "f1_micro":       mic_f1,
            "num_samples":    len(all_icd_results),
        }

    metrics["eval_protocol"] = {
        "seed": int(args.seed),
        "eval_split": eval_split,
        "max_eval_samples_arg": int(args.max_eval_samples),
        "num_eval_windows": len(eval_indices),
        "eval_indices_json": str(indices_path) if indices_path else None,
        "rep_penalty": float(args.rep_penalty),
        "rep_penalty_window": int(args.rep_penalty_window),
        "report_max_new_tokens_arg": int(args.report_max_new_tokens),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "state_dict_path": str(args.state_dict_path),
    }

    if world_size > 1:
        shard_suffix = f"_shard{rank:02d}"
    else:
        shard_suffix = ""
    metrics_path  = out_dir / f"eval_metrics{shard_suffix}.json"
    previews_path = out_dir / f"eval_previews{shard_suffix}.json"

    raw_results = {
        "image":  all_image_results,
        "report": all_report_results,
        "icd":    all_icd_results,
        "rank":   rank,
        "world_size": world_size,
        "eval_indices": eval_indices,
    }
    save_json(out_dir / f"eval_raw{shard_suffix}.json", raw_results)

    save_json(metrics_path,  metrics)
    save_json(previews_path, all_previews)
    print(json.dumps(metrics, indent=2))
    print(f"[done] wrote: {metrics_path}")
    print(f"[done] wrote: {previews_path}")

    if save_reports_json is not None and _saved_reports:
        save_json(save_reports_json, _saved_reports)
        print(f"[done] wrote {len(_saved_reports)} reports -> {save_reports_json}")


if __name__ == "__main__":
    main()
