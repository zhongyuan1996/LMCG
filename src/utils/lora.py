"""
Minimal LoRA utilities for Showo2 fine-tuning.

Goal: allow LoRA adaptation for *non-LLM* (and non-VAE) components without touching
the LLM backbone weights.

We implement lightweight LoRA wrappers for nn.Linear and nn.Conv2d and provide a
recursive module-replacement helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoraSpec:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.r) if self.r > 0 else 1.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, spec: LoraSpec):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be nn.Linear, got {type(base)}")
        if spec.r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Linear(base.in_features, spec.r, bias=False)
        self.lora_B = nn.Linear(spec.r, base.out_features, bias=False)
        self.dropout = nn.Dropout(spec.dropout) if spec.dropout and spec.dropout > 0 else nn.Identity()
        self.scale = spec.scale

        # Init: A ~ N(0, 0.01), B = 0 so start as identity (no-op)
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

        # Match base module device/dtype (important for inference-time injection).
        # If base is still on meta, defer; typical training uses accelerator.prepare which will move params later.
        if getattr(self.base.weight, "device", None) is not None and self.base.weight.device.type != "meta":
            self.lora_A.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
            self.lora_B.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.lora_B(self.lora_A(self.dropout(x))) * self.scale)


class LoRAConv2d(nn.Module):
    """
    LoRA for Conv2d using:
      delta = B( A(x) )
    where A uses the same spatial kernel/stride/padding as the base conv to match shapes,
    and B is a 1x1 conv to project back to out_channels.
    """

    def __init__(self, base: nn.Conv2d, spec: LoraSpec):
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError(f"base must be nn.Conv2d, got {type(base)}")
        if spec.r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        if base.groups != 1:
            # Keep it simple: showo2 convs we care about are not grouped.
            raise ValueError(f"LoRAConv2d does not support groups={base.groups}")

        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=spec.r,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            in_channels=spec.r,
            out_channels=base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(spec.dropout) if spec.dropout and spec.dropout > 0 else nn.Identity()
        self.scale = spec.scale

        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

        # Match base module device/dtype (important for inference-time injection).
        if getattr(self.base.weight, "device", None) is not None and self.base.weight.device.type != "meta":
            self.lora_A.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
            self.lora_B.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dropout for convs: apply on activations (same as Linear path)
        x_d = self.dropout(x)
        return self.base(x) + (self.lora_B(self.lora_A(x_d)) * self.scale)


def _freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _is_excluded(full_name: str, exclude_prefixes: Sequence[str]) -> bool:
    for pfx in exclude_prefixes:
        if full_name == pfx or full_name.startswith(pfx + "."):
            return True
    return False


def inject_lora(
    model: nn.Module,
    *,
    spec: LoraSpec,
    exclude_prefixes: Sequence[str] = ("showo",),
    enable_linear: bool = True,
    enable_conv2d: bool = True,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Recursively replace eligible submodules with LoRA-wrapped versions.

    Returns:
        (num_linear_wrapped, num_conv2d_wrapped)
    """

    n_linear = 0
    n_conv2d = 0

    def _rec(parent: nn.Module, prefix: str) -> None:
        nonlocal n_linear, n_conv2d
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name

            if _is_excluded(full, exclude_prefixes):
                continue

            if enable_linear and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, spec))
                n_linear += 1
                continue

            if enable_conv2d and isinstance(child, nn.Conv2d):
                setattr(parent, child_name, LoRAConv2d(child, spec))
                n_conv2d += 1
                continue

            _rec(child, full)

    _rec(model, "")

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"[lora] wrapped linear={n_linear} conv2d={n_conv2d} | "
            f"trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M | "
            f"exclude_prefixes={list(exclude_prefixes)}"
        )

    return n_linear, n_conv2d


def maybe_apply_showo2_lora(model: nn.Module, config) -> nn.Module:
    """
    Apply LoRA to Showo2 model components *excluding* the LLM backbone (model.showo.*).

    Config keys (CLI or YAML):
      model.lora.enabled: bool (default False)
      model.lora.r: int (default 8)
      model.lora.alpha: int (default 16)
      model.lora.dropout: float (default 0.0)
      model.lora.exclude_prefixes: list[str] (default ["showo"])
      model.lora.enable_conv2d: bool (default True)
      model.lora.enable_linear: bool (default True)
    """

    lora_cfg = None
    try:
        lora_cfg = config.model.get("lora", None)
    except Exception:
        lora_cfg = getattr(getattr(config, "model", None), "lora", None)

    if not lora_cfg:
        return model

    enabled = bool(getattr(lora_cfg, "enabled", False) if hasattr(lora_cfg, "enabled") else lora_cfg.get("enabled", False))
    if not enabled:
        return model

    # Freeze ALL weights; only LoRA params will be trainable.
    _freeze_all_params(model)

    spec = LoraSpec(
        r=int(lora_cfg.get("r", 8) if hasattr(lora_cfg, "get") else getattr(lora_cfg, "r", 8)),
        alpha=int(lora_cfg.get("alpha", 16) if hasattr(lora_cfg, "get") else getattr(lora_cfg, "alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.0) if hasattr(lora_cfg, "get") else getattr(lora_cfg, "dropout", 0.0)),
    )

    exclude_prefixes = lora_cfg.get("exclude_prefixes", ["showo"]) if hasattr(lora_cfg, "get") else getattr(lora_cfg, "exclude_prefixes", ["showo"])
    if isinstance(exclude_prefixes, str):
        exclude_prefixes = [exclude_prefixes]
    exclude_prefixes = tuple(str(x) for x in exclude_prefixes)

    enable_conv2d = bool(lora_cfg.get("enable_conv2d", True) if hasattr(lora_cfg, "get") else getattr(lora_cfg, "enable_conv2d", True))
    enable_linear = bool(lora_cfg.get("enable_linear", True) if hasattr(lora_cfg, "get") else getattr(lora_cfg, "enable_linear", True))

    inject_lora(
        model,
        spec=spec,
        exclude_prefixes=exclude_prefixes,
        enable_linear=enable_linear,
        enable_conv2d=enable_conv2d,
        verbose=True,
    )
    return model

