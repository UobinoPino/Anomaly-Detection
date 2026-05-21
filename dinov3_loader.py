"""Centralised DINOv3 / DINOv2-with-registers loader.

# Why this module exists

All Spacepresso baselines (PatchCore, FastFlow, CFA, UniAD, EfficientAD,
Reverse Distillation, CutPaste-ViT) need access to the same modern ViT
backbones with the SAME loading code path. This module is the single
source of truth.

# Supported backbone families

  ── DINOv3 (Meta, Aug 2025, patch=16, 4 register tokens) ─────────────
      dinov3_vits16          21M    embed=384
      dinov3_vits16plus      29M    embed=384  (SwiGLU FFN)
      dinov3_vitb16          86M    embed=768
      dinov3_vitl16         300M    embed=1024
      dinov3_vith16plus     840M    embed=1280
      dinov3_vit7b16        6700M   embed=4096   (don't try this on L4)
      dinov3_convnext_tiny  small CNN backbone, multi-scale
      dinov3_convnext_small
      dinov3_convnext_base
      dinov3_convnext_large

  ── DINOv2 with registers (Meta, 2024 update; patch=14, 4 reg tokens)
      dinov2_vits14_reg      21M    embed=384
      dinov2_vitb14_reg      86M    embed=768
      dinov2_vitl14_reg     300M    embed=1024
      dinov2_vitg14_reg    1100M    embed=1536

  ── Plain DINOv2 (your existing exp7 backbone; patch=14, no registers)
      dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14

# Patch-size sanity

  DINOv2 family: patch=14 → input_size must be multiple of 14
                 (you've been using 392 = 14*28 and 518 = 14*37)
  DINOv3 family: patch=16 → input_size must be multiple of 16
                 (recommended: 224, 384, 512, 768; or 32-aligned for OCBE
                 in reverse_distillation_baseline.py).

# Loading mechanics

  DINOv2 (plain + reg):
      torch.hub.load('facebookresearch/dinov2', NAME)
      Public, anonymous, just works.

  DINOv3:
      DINOv3 weights are gated. After you accept the Meta license,
      they email you per-checkpoint URLs. You can either:
        (a) Download with wget into $DINOV3_WEIGHTS (a directory) and
            this module reads from disk, OR
        (b) Pass the URL directly via $DINOV3_<NAME>_URL.

      The reference repo (https://github.com/facebookresearch/dinov3)
      must also be available locally because the model definitions
      themselves aren't on torch.hub. Clone it once:
          git clone https://github.com/facebookresearch/dinov3.git \
                    $HOME/dinov3_repo
      and export DINOV3_REPO=$HOME/dinov3_repo (or pass repo_dir=...).

      As a fallback (if dinov3 repo is not present), this loader will
      try the HuggingFace `transformers` route:
          from transformers import AutoModel
          AutoModel.from_pretrained("facebook/dinov3-vitX16-pretrain-lvd1689m")
      which auto-downloads weights via HF Hub (still requires accepting
      the license on the HF model card first).

# What this loader returns

  A tuple (model, info) where info is a dict:
      info["family"]      : "dinov3" | "dinov2" | "dinov2_reg"
      info["patch_size"]  : 14 or 16
      info["embed_dim"]   : feature dim
      info["n_blocks"]    : transformer depth (for layer-index selection)
      info["n_register"]  : number of register tokens to strip
      info["mean"], info["std"] : ImageNet stats — same for all DINO variants
      info["forward_patches"]   : callable(model, x, layers) -> dict
                                  layer_idx -> (B, C, H/P, W/P) tensor
        - hides the wire-format difference between DINOv2 (no registers)
          and DINOv3 (4 register tokens after the CLS token).

  The forward_patches callable is what the rest of the codebase calls;
  it's the only knob FeatureExtractor needs to know about.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Static spec tables ──────────────────────────────────────────────────────

DINOV2_SPECS = {
    "dinov2_vits14":      {"embed": 384,  "n_blocks": 12, "patch": 14, "reg": 0},
    "dinov2_vitb14":      {"embed": 768,  "n_blocks": 12, "patch": 14, "reg": 0},
    "dinov2_vitl14":      {"embed": 1024, "n_blocks": 24, "patch": 14, "reg": 0},
    "dinov2_vitg14":      {"embed": 1536, "n_blocks": 40, "patch": 14, "reg": 0},
    "dinov2_vits14_reg":  {"embed": 384,  "n_blocks": 12, "patch": 14, "reg": 4},
    "dinov2_vitb14_reg":  {"embed": 768,  "n_blocks": 12, "patch": 14, "reg": 4},
    "dinov2_vitl14_reg":  {"embed": 1024, "n_blocks": 24, "patch": 14, "reg": 4},
    "dinov2_vitg14_reg":  {"embed": 1536, "n_blocks": 40, "patch": 14, "reg": 4},
}

DINOV3_VIT_SPECS = {
    "dinov3_vits16":       {"embed": 384,  "n_blocks": 12, "patch": 16, "reg": 4},
    "dinov3_vits16plus":   {"embed": 384,  "n_blocks": 12, "patch": 16, "reg": 4},
    "dinov3_vitb16":       {"embed": 768,  "n_blocks": 12, "patch": 16, "reg": 4},
    "dinov3_vitl16":       {"embed": 1024, "n_blocks": 24, "patch": 16, "reg": 4},
    "dinov3_vith16plus":   {"embed": 1280, "n_blocks": 32, "patch": 16, "reg": 4},
    "dinov3_vit7b16":      {"embed": 4096, "n_blocks": 40, "patch": 16, "reg": 4},
}

# ConvNeXt variants: very different forward path — handled separately if needed
DINOV3_CONVNEXT_SPECS = {
    "dinov3_convnext_tiny":  {"channels": [96,  192, 384, 768],  "patch_eff": 4},
    "dinov3_convnext_small": {"channels": [96,  192, 384, 768],  "patch_eff": 4},
    "dinov3_convnext_base":  {"channels": [128, 256, 512, 1024], "patch_eff": 4},
    "dinov3_convnext_large": {"channels": [192, 384, 768, 1536], "patch_eff": 4},
}

ALL_DINO_BACKBONES = sorted(
    list(DINOV2_SPECS) + list(DINOV3_VIT_SPECS) + list(DINOV3_CONVNEXT_SPECS))

# Both DINOv2 and DINOv3 use ImageNet normalisation in inference code.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass
class DinoInfo:
    name: str
    family: str          # "dinov2" | "dinov2_reg" | "dinov3" | "dinov3_convnext"
    embed_dim: int
    n_blocks: int        # for ConvNeXt: number of stages (4)
    patch_size: int      # effective stride to first feature map
    n_register: int
    mean: tuple = IMAGENET_MEAN
    std: tuple  = IMAGENET_STD


def is_dino_backbone(name: str) -> bool:
    return name in DINOV2_SPECS or name in DINOV3_VIT_SPECS or name in DINOV3_CONVNEXT_SPECS


# ─── DINOv2 ──────────────────────────────────────────────────────────────────

def _load_dinov2(name: str) -> tuple[nn.Module, DinoInfo]:
    """Plain DINOv2 or DINOv2-with-registers, from torch.hub."""
    spec = DINOV2_SPECS[name]
    model = torch.hub.load("facebookresearch/dinov2", name,
                            trust_repo=True, source="github")
    family = "dinov2_reg" if spec["reg"] > 0 else "dinov2"
    info = DinoInfo(name=name, family=family, embed_dim=spec["embed"],
                     n_blocks=spec["n_blocks"], patch_size=spec["patch"],
                     n_register=spec["reg"])
    return model, info


# ─── DINOv3 (local repo + downloaded weights) ────────────────────────────────

def _dinov3_repo_dir() -> Path | None:
    p = os.environ.get("DINOV3_REPO")
    if p and Path(p).exists():
        return Path(p)
    # Common fallbacks
    for cand in (Path.home() / "dinov3_repo",
                 Path.home() / "dinov3",
                 Path("/work/u10813429/dinov3_repo"),
                 Path("/workspace/dinov3_repo")):
        if cand.exists() and (cand / "hubconf.py").exists():
            return cand
    return None


def _dinov3_weights_for(name: str) -> str | None:
    """Resolve weights for a given DINOv3 model name. Tries:
      1. $DINOV3_<NAME_UPPER>_URL (explicit URL)
      2. $DINOV3_WEIGHTS/<name>_*.pth (downloaded file)
    Returns a URL or filepath; or None if not found.
    """
    env_key = f"DINOV3_{name.upper()}_URL"
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]

    weights_dir = os.environ.get("DINOV3_WEIGHTS")
    if weights_dir and Path(weights_dir).exists():
        # Look for files like dinov3_vitb16_pretrain_lvd1689m-*.pth
        for p in sorted(Path(weights_dir).glob(f"{name}*.pth")):
            return str(p)
        for p in sorted(Path(weights_dir).glob(f"{name}*.pt")):
            return str(p)
    return None


def _load_dinov3_via_torchhub(name: str) -> tuple[nn.Module, DinoInfo] | None:
    repo = _dinov3_repo_dir()
    if repo is None:
        return None
    wpath = _dinov3_weights_for(name)
    if wpath is None:
        return None
    try:
        # The dinov3 repo's hubconf.py registers entrypoints like
        # `dinov3_vitb16`, `dinov3_vitl16`, etc.
        model = torch.hub.load(str(repo), name, source="local", weights=wpath)
    except Exception as e:
        print(f"  [warn] DINOv3 torch.hub load failed: {e}")
        return None
    spec = (DINOV3_VIT_SPECS if name in DINOV3_VIT_SPECS
             else DINOV3_CONVNEXT_SPECS)[name]
    if name in DINOV3_VIT_SPECS:
        info = DinoInfo(name=name, family="dinov3",
                         embed_dim=spec["embed"], n_blocks=spec["n_blocks"],
                         patch_size=spec["patch"], n_register=spec["reg"])
    else:
        info = DinoInfo(name=name, family="dinov3_convnext",
                         embed_dim=0, n_blocks=4,
                         patch_size=spec["patch_eff"], n_register=0)
    return model, info


def _load_dinov3_via_huggingface(name: str) -> tuple[nn.Module, DinoInfo] | None:
    """Fallback path. Requires `transformers` >= 4.45. Returns a thin
    wrapper exposing get_intermediate_layers(x, n, reshape, norm) so the
    same downstream code works."""
    try:
        from transformers import AutoModel
    except ImportError:
        print("  [warn] transformers not installed; cannot load DINOv3 from HF")
        return None

    # Map our internal name → HF model id.
    hf_map = {
        "dinov3_vits16":       "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dinov3_vits16plus":   "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "dinov3_vitb16":       "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_vitl16":       "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dinov3_vith16plus":   "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "dinov3_vit7b16":      "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "dinov3_convnext_tiny":  "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "dinov3_convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        "dinov3_convnext_base":  "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        "dinov3_convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    }
    if name not in hf_map:
        return None
    try:
        backbone = AutoModel.from_pretrained(hf_map[name])
    except Exception as e:
        print(f"  [warn] HF load of {hf_map[name]} failed: {e}")
        return None
    backbone.eval()

    spec = (DINOV3_VIT_SPECS if name in DINOV3_VIT_SPECS
             else DINOV3_CONVNEXT_SPECS)[name]
    if name in DINOV3_VIT_SPECS:
        info = DinoInfo(name=name, family="dinov3",
                         embed_dim=spec["embed"], n_blocks=spec["n_blocks"],
                         patch_size=spec["patch"], n_register=spec["reg"])
        wrapped = _HFDinov3Wrapper(backbone, info)
    else:
        info = DinoInfo(name=name, family="dinov3_convnext",
                         embed_dim=0, n_blocks=4,
                         patch_size=spec["patch_eff"], n_register=0)
        wrapped = _HFDinov3ConvNeXtWrapper(backbone, info)
    return wrapped, info


class _HFDinov3Wrapper(nn.Module):
    """Make a HuggingFace DINOv3 model look enough like the torch.hub
    DINOv2/v3 model that `get_intermediate_layers` works the same way."""
    def __init__(self, backbone: nn.Module, info: DinoInfo):
        super().__init__()
        self.backbone = backbone
        self._info = info

    def get_intermediate_layers(self, x: torch.Tensor, n,
                                  reshape: bool = True, norm: bool = True,
                                  return_class_token: bool = False):
        # n can be int (last n layers) or list of indices.
        if isinstance(n, int):
            block_indices = list(range(self._info.n_blocks - n,
                                          self._info.n_blocks))
        else:
            block_indices = list(n)

        B, _, H, W = x.shape
        P = self._info.patch_size
        Hp, Wp = H // P, W // P

        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple, includes input embed
        # hidden_states[0] = embeddings; hidden_states[k+1] = output of block k
        outs = []
        for idx in block_indices:
            h = hidden_states[idx + 1]  # (B, 1 + Reg + Hp*Wp, D)
            # Strip CLS + register tokens.
            n_skip = 1 + self._info.n_register
            patch_tokens = h[:, n_skip:, :]  # (B, Hp*Wp, D)
            if norm:
                # Norm layer is at self.backbone.layernorm (HF naming)
                ln = getattr(self.backbone, "layernorm", None)
                if ln is not None and idx == block_indices[-1]:
                    patch_tokens = ln(patch_tokens)
            if reshape:
                patch_tokens = patch_tokens.reshape(B, Hp, Wp, -1).permute(
                    0, 3, 1, 2).contiguous()
            outs.append(patch_tokens)
        return outs


class _HFDinov3ConvNeXtWrapper(nn.Module):
    """ConvNeXt variant. Returns the 4 stage feature maps directly."""
    def __init__(self, backbone: nn.Module, info: DinoInfo):
        super().__init__()
        self.backbone = backbone
        self._info = info

    def get_intermediate_layers(self, x: torch.Tensor, n,
                                  reshape: bool = True, norm: bool = True,
                                  return_class_token: bool = False):
        outputs = self.backbone(x, output_hidden_states=True)
        # ConvNeXt hidden states are already 4D feature maps in HF.
        hs = outputs.hidden_states
        if isinstance(n, int):
            indices = list(range(len(hs) - n, len(hs)))
        else:
            indices = list(n)
        return [hs[i] for i in indices]


# ─── Public entry point ──────────────────────────────────────────────────────

def load_dino_backbone(name: str) -> tuple[nn.Module, DinoInfo]:
    """Load any DINOv2 or DINOv3 backbone and return (model, info).

    Resolution order:
      DINOv2/DINOv2-reg : torch.hub from facebookresearch/dinov2
      DINOv3            : (a) local clone + downloaded weights via torch.hub,
                          else (b) HuggingFace transformers
    """
    if name in DINOV2_SPECS:
        return _load_dinov2(name)
    if name in DINOV3_VIT_SPECS or name in DINOV3_CONVNEXT_SPECS:
        out = _load_dinov3_via_torchhub(name)
        if out is not None:
            return out
        out = _load_dinov3_via_huggingface(name)
        if out is not None:
            return out
        raise RuntimeError(
            f"Could not load {name}. To use DINOv3:\n"
            f"  Option A — clone the repo + download weights:\n"
            f"      git clone https://github.com/facebookresearch/dinov3 $HOME/dinov3_repo\n"
            f"      export DINOV3_REPO=$HOME/dinov3_repo\n"
            f"      # request access at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/\n"
            f"      # wget the {name}_*.pth file emailed to you into $DINOV3_WEIGHTS/\n"
            f"      export DINOV3_WEIGHTS=$HOME/dinov3_weights\n"
            f"  Option B — install transformers and accept the license on HF:\n"
            f"      uv pip install 'transformers>=4.45'\n"
            f"      # accept license at https://huggingface.co/facebook/dinov3-vitX16-pretrain-lvd1689m\n"
            f"      huggingface-cli login")
    raise ValueError(f"unknown backbone: {name}")


# ─── Helper: unified patch-token extraction at arbitrary block depths ────────

def get_patch_tokens_at_layers(model: nn.Module, info: DinoInfo,
                                  x: torch.Tensor, layers) -> dict:
    """Returns {layer_idx: (B, C, H/P, W/P)} for ViT, or
    {stage_idx: (B, C_stage, H/2^k, W/2^k)} for ConvNeXt.

    Handles register-token stripping internally so callers never see
    them — works the same way for DINOv2-plain, DINOv2-reg, DINOv3-ViT,
    and (with caveat about stage indexing) DINOv3-ConvNeXt.
    """
    layers = sorted(set(int(l) for l in layers))
    if info.family == "dinov3_convnext":
        # ConvNeXt has 4 stages. Allow caller to ask 0..3.
        n_stages = 4
        for l in layers:
            if not (0 <= l < n_stages):
                raise ValueError(f"ConvNeXt stage {l} out of [0, {n_stages-1}]")
        outs = model.get_intermediate_layers(x, n=list(layers),
                                                reshape=True, norm=True)
        return {l: o for l, o in zip(layers, outs)}

    # ViT path (DINOv2 plain/reg, DINOv3 ViT) — torch.hub model exposes
    # get_intermediate_layers with proper handling. We pass layers
    # explicitly and let the model handle (or we handle) registers.
    B, _, H, W = x.shape
    P = info.patch_size
    if H % P or W % P:
        raise ValueError(
            f"input ({H}, {W}) not divisible by patch size {P} for {info.name}")

    for l in layers:
        if not (0 <= l < info.n_blocks):
            raise ValueError(
                f"layer {l} out of range [0, {info.n_blocks - 1}] for {info.name}")

    outs = model.get_intermediate_layers(x, n=list(layers),
                                            reshape=True, norm=True)
    # On the official DINOv2/v3 torch.hub model, `reshape=True` already
    # strips CLS and register tokens. The HF wrapper above also strips
    # them. So outs[i] is already (B, C, Hp, Wp).
    return {l: o for l, o in zip(layers, outs)}


def validate_input_size(input_size: int, info: DinoInfo) -> None:
    if input_size % info.patch_size != 0:
        raise SystemExit(
            f"[FATAL] --input-size {input_size} must be a multiple of "
            f"{info.patch_size} for {info.name} (DINOv2 uses 14, DINOv3 "
            f"uses 16).")