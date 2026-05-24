"""Spacepresso UniAD baseline (DINOv2 teacher, L2-normalised features).

This is a drop-in replacement for the EfficientNet-B4 version. Same
training/inference/multi-view scaffolding; only the teacher and the
loss target normalisation change.

# What was wrong with the EfficientNet-B4 version

  1. Resolution too coarse. EfficientNet-B4 features fused at H/16 give
     16x16 = 256 tokens at input 256. Defects spanning 30-100 px occupy
     1-4 tokens — well below what a transformer reconstructor can
     localise.

  2. Loss dominated by high-magnitude channels. Concatenating 4 EffNet
     stages (32+56+160+272 = 520 ch) produced raw activations spanning
     ~3 orders of magnitude across channels. MSE on those is dominated
     by the 5-10 strongest channels; the model fits those, ignores the
     rest, and the rest are where defects show up. Loss ended at ~75 on
     train_good — that number is unitless because the targets are
     unitless, but it never approached the "low" regime UniAD assumes.

# What this version does

  TEACHER (frozen, single layer):
    DINOv2 ViT-S/14 (proven on this dataset by PatchCore exp7). One
    intermediate block (default: block 9) reshaped to (B, 384, H/14,
    W/14). Multi-layer fusion is intentionally NOT done — UniAD wants
    a single well-defined reconstruction target, and PatchCore already
    occupies the multi-layer-fusion niche in the stacker.

    Features are L2-normalised along the channel dimension before they
    become the reconstruction target. Each spatial token is now a unit
    vector. MSE between a free reconstruction and a unit-vector target
    is bounded:
        ||r - t||^2 = ||r||^2 - 2<r,t> + 1
    The model converges to ||r||=1 with <r,t>=1, so loss converges
    toward 0. Anomalous pixels produce reconstruction misses → high
    MSE → clean anomaly signal.

  TOKEN GRID:
    Input 392 → 28x28 = 784 tokens (default).
    Input 518 → 37x37 = 1369 tokens (matches PatchCore exp7).
    Input 280 → 20x20 = 400 tokens (fastest, less spatial resolution).

  ENCODER / DECODER:
    Same neighbour-masked transformer with layer-wise query injection
    as the original UniAD. Architecture unchanged.

  ANOMALY SCORE:
    per-pixel MSE between reconstruction and (L2-normalised) teacher.
    Higher = more anomalous. Same as original UniAD up to the target
    normalisation.

  MULTI-VIEW (--multiview sibling-bank):
    Same scaffolding as before — group views of a sample, compute
    per-pixel cross-view feature inconsistency, blend in by alpha.
    NOTE: as we discovered with PatchCore consensus-v3, this works
    badly when the 5 views are different camera angles (which they
    are for Spacepresso). Expected behaviour: --multiview none beats
    --multiview sibling-bank on most classes. Run both A/B-style; do
    NOT default to MV.

# Memory / speed (L4 24 GB)

  Per class @ input 392, batch 8, 5000 iters:
    Training:        ~7-9 min (transformer + DINOv2 forward each iter)
    Standard score:  ~25-40 s eval + 25-40 s test
    Multi-view:      ~40-60 s eval + 40-60 s test
  Full 8 classes:    ~75-110 min standard, ~100-140 min multi-view.

# Dependencies

  patchcore_baseline_v2.py and local_preds_saver.py in same directory.
  DINOv2 weights are pulled via torch.hub on first run; they cache to
  ~/.cache/torch/hub/. Pre-cache on a node with internet if compute
  nodes are offline:
      python -c "import torch; \\
          torch.hub.load('facebookresearch/dinov2', \\
                         'dinov2_vits14', trust_repo=True, source='github')"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patchcore_baseline_v2 import (
    ImageRecord, scan_dataset,
    pixel_average_precision, gaussian_smooth,
    calibrate_to_unit, float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
    IMAGENET_MEAN, IMAGENET_STD,
)
from local_preds_saver import LocalPredSaver


PROJECT_ROOT = Path("/work/u10813429/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"

# DINOv2 model registry — keep in sync with patchcore_baseline_v2.py.
DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14": 1024,
}
DINOV2_NBLOCKS = {
    "dinov2_vits14": 12,
    "dinov2_vitb14": 12,
    "dinov2_vitb14_reg": 12,
    "dinov2_vitl14": 24,
}
PATCH_SIZE = 14


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


@contextmanager
def tee_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    old = sys.stdout
    sys.stdout = Tee(old, f)
    try: yield
    finally:
        sys.stdout = old; f.close()


def hr(t, c="="): print(f"\n{c * 78}\n  {t}\n{c * 78}")
def sub(t): print(f"\n--- {t} ---")
def now_hms(): return time.strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────
class TrainGoodDataset(Dataset):
    def __init__(self, records: list[ImageRecord], input_size: int):
        self.records = records
        self.tx = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        with Image.open(self.records[i].path) as im:
            return self.tx(im.convert("RGB"))


class InferenceDataset(Dataset):
    def __init__(self, records: list[ImageRecord], input_size: int,
                 load_masks: bool):
        self.records = records
        self.input_size = input_size
        self.load_masks = load_masks
        self.tx = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r.path) as im:
            x = self.tx(im.convert("RGB"))
        if self.load_masks and r.mask_path is not None:
            with Image.open(r.mask_path) as mm:
                mm = mm.convert("L").resize(
                    (self.input_size, self.input_size), Image.NEAREST)
                m = (np.asarray(mm) > 127).astype(np.float32)
        else:
            m = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        return x, torch.from_numpy(m), i


def worker_init_fn(_worker_id):
    base = torch.initial_seed() % 2 ** 32
    np.random.seed(base); random.seed(base)


# ─────────────────────────────────────────────────────────────────────────────
# Teacher: DINOv2 ViT, single block, L2-normalised
# ─────────────────────────────────────────────────────────────────────────────
class DINOv2Teacher(nn.Module):
    """Frozen DINOv2 backbone. Returns features from a SINGLE intermediate
    block reshaped to (B, C, h, w) where h = w = input_size / 14.

    Features are L2-normalised along the channel dimension so the
    reconstruction MSE loss is bounded and channel-balanced — high-
    magnitude channels can no longer dominate the gradient.
    """

    def __init__(self, model_name: str = "dinov2_vits14",
                  block_idx: int = 9):
        super().__init__()
        if model_name not in DINOV2_DIMS:
            raise ValueError(f"unknown DINOv2 model: {model_name}. "
                              f"Pick from {list(DINOV2_DIMS)}")
        n_blocks = DINOV2_NBLOCKS[model_name]
        if not (0 <= block_idx < n_blocks):
            raise ValueError(
                f"block_idx={block_idx} out of range [0, {n_blocks - 1}] "
                f"for {model_name}")

        self.model_name = model_name
        self.block_idx = block_idx
        self.patch_size = PATCH_SIZE
        self.out_channels = DINOV2_DIMS[model_name]
        self.target_stride = PATCH_SIZE

        self.dinov2 = torch.hub.load(
            "facebookresearch/dinov2", model_name,
            trust_repo=True, source="github")
        self.dinov2.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"DINOv2 needs H, W divisible by {self.patch_size}; "
                f"got ({H}, {W}).")
        outs = self.dinov2.get_intermediate_layers(
            x, n=[self.block_idx], reshape=True, norm=True)
        feat = outs[0]                                   # (B, C, h, w)
        # L2-normalise along channels — every spatial token becomes a
        # unit vector. This is the key fix relative to EffNet-B4 UniAD.
        feat = F.normalize(feat, p=2, dim=1)
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# Neighbour-masked attention mask
# ─────────────────────────────────────────────────────────────────────────────
def build_neighbour_mask(h: int, w: int, neighbour_radius: int,
                          device: torch.device | None = None) -> torch.Tensor:
    """Returns (L, L) bool mask with True where attention is FORBIDDEN.
    Token at (r1, c1) is forbidden to attend to (r2, c2) if their
    Chebyshev (L_inf) distance is <= neighbour_radius. With radius=0,
    only the diagonal (self) is masked — enough to break the identity
    shortcut. Larger radii mask a square neighbourhood; UniAD paper used
    radius=7 on 14x14 grids (effectively all-far attention)."""
    rs = torch.arange(h).repeat_interleave(w)
    cs = torch.arange(w).repeat(h)
    dr = (rs.unsqueeze(0) - rs.unsqueeze(1)).abs()
    dc = (cs.unsqueeze(0) - cs.unsqueeze(1)).abs()
    cheby = torch.maximum(dr, dc)
    mask = cheby <= neighbour_radius
    if device is not None:
        mask = mask.to(device)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Transformer blocks
# ─────────────────────────────────────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """Pre-LN encoder block with neighbour-masked self-attention."""
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                  dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Pre-LN decoder: cross-attn to encoder, then masked self-attn, then FFN."""
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                  dropout: float = 0.1):
        super().__init__()
        self.ln_cross_q = nn.LayerNorm(dim)
        self.ln_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, encoder_out, self_attn_mask=None):
        q = self.ln_cross_q(x)
        kv = self.ln_cross_kv(encoder_out)
        h, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + h
        h = self.ln_self(x)
        h, _ = self.self_attn(h, h, h, attn_mask=self_attn_mask,
                                need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln_mlp(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# UniAD model — unchanged from EffNet version
# ─────────────────────────────────────────────────────────────────────────────
class UniADModel(nn.Module):
    def __init__(self, teacher_dim: int, feature_hw: tuple[int, int],
                  model_dim: int = 256, n_heads: int = 8,
                  n_enc_layers: int = 4, n_dec_layers: int = 4,
                  mlp_ratio: float = 4.0, dropout: float = 0.1,
                  neighbour_radius: int = 7):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.model_dim = model_dim
        h, w = feature_hw
        self.feature_hw = (h, w)
        self.n_tokens = h * w
        self.neighbour_radius = neighbour_radius

        self.input_proj = nn.Linear(teacher_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.query_embed = nn.Parameter(
            torch.zeros(1, self.n_tokens, model_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(model_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_enc_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(model_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_ln_out = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, teacher_dim)

        nm = build_neighbour_mask(h, w, neighbour_radius)
        self.register_buffer("attn_forbid_mask", nm, persistent=False)

    def forward(self, t_feat: torch.Tensor,
                jitter_sigma: float = 0.0) -> torch.Tensor:
        B, C, h, w = t_feat.shape
        assert (h, w) == self.feature_hw, (
            f"feature shape mismatch: got ({h},{w}), expected "
            f"{self.feature_hw}")
        x = t_feat.permute(0, 2, 3, 1).reshape(B, h * w, C)
        if jitter_sigma > 0 and self.training:
            std = x.detach().std(dim=(0, 1), keepdim=True) + 1e-6
            x = x + torch.randn_like(x) * (jitter_sigma * std)
        x = self.input_proj(x) + self.pos_embed
        attn_mask = self.attn_forbid_mask
        for blk in self.encoder:
            x = blk(x, attn_mask=attn_mask)
        dec_in = self.query_embed.expand(B, -1, -1).clone()
        for blk in self.decoder:
            dec_in = dec_in + self.query_embed
            dec_in = blk(dec_in, encoder_out=x, self_attn_mask=attn_mask)
        dec_in = self.dec_ln_out(dec_in)
        recon = self.output_proj(dec_in)
        return recon.reshape(B, h, w, C).permute(0, 3, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_uniad(teacher, model, records, cfg, device) -> None:
    ds = TrainGoodDataset(records, input_size=cfg.input_size)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=worker_init_fn, drop_last=True)
    iters_per_epoch = max(len(loader), 1)
    if cfg.total_iters and cfg.total_iters > 0:
        n_epochs = max(1, math.ceil(cfg.total_iters / iters_per_epoch))
        print(f"    [auto-epoch] total_iters={cfg.total_iters} / "
              f"{iters_per_epoch} iters/epoch -> {n_epochs} epochs "
              f"(--epochs={cfg.epochs} overridden)")
    else:
        n_epochs = cfg.epochs
    total_iters = iters_per_epoch * n_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))
    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"    [{now_hms()}] training: {n_epochs} epochs x "
          f"{iters_per_epoch} iters ({total_iters} total)  "
          f"bs={cfg.batch_size}  amp={use_amp}  "
          f"jitter_sigma={cfg.jitter_sigma}")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    teacher.eval()
    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0.0; n = 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    t_feat = teacher(x)
                recon = model(t_feat, jitter_sigma=cfg.jitter_sigma)
                loss = F.mse_loss(recon, t_feat)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            B = x.shape[0]
            loss_sum += loss.item() * B
            n += B
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={loss_sum/max(n,1):.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    model.eval()
    print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Norm stats over train_good
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def compute_norm_stats(teacher, model, records, cfg, device,
                        n_max: int = 64) -> dict:
    if len(records) > n_max:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(records), size=n_max, replace=False)
        records = [records[i] for i in idx]
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=False)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    use_amp = (device.type == "cuda" and cfg.amp)
    vals = []
    teacher.eval(); model.eval()
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            t_feat = teacher(x)
            recon = model(t_feat, jitter_sigma=0.0)
        diff = ((recon.float() - t_feat.float()) ** 2).mean(dim=1)
        vals.append(diff.cpu().numpy().reshape(-1))
    arr = np.concatenate(vals)
    stats = {"recon_mean": float(arr.mean()),
              "recon_std":  float(arr.std() + 1e-9)}
    print(f"    norm stats over {len(records)} train_good: "
          f"recon={stats['recon_mean']:.4f}±{stats['recon_std']:.4f}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Inference primitives
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _compute_maps_one(teacher, model, x, stats, cfg):
    use_amp = (cfg.device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        t_feat = teacher(x)
        recon = model(t_feat, jitter_sigma=0.0)
    diff = ((recon.float() - t_feat.float()) ** 2).mean(dim=1)
    diff_n = (diff - stats["recon_mean"]) / stats["recon_std"]
    return diff_n, t_feat.float()


@torch.inference_mode()
def _maps_with_tta(teacher, model, x, stats, cfg, tta):
    accum_s = None; accum_t = None; n_acc = 0
    def _add(s, t):
        nonlocal accum_s, accum_t, n_acc
        if accum_s is None:
            accum_s = s.clone(); accum_t = t.clone()
        else:
            accum_s += s; accum_t += t
        n_acc += 1
    s, t = _compute_maps_one(teacher, model, x, stats, cfg)
    _add(s, t)
    if tta in ("hflip", "hvflip"):
        s2, t2 = _compute_maps_one(
            teacher, model, torch.flip(x, dims=[-1]), stats, cfg)
        _add(torch.flip(s2, dims=[-1]), torch.flip(t2, dims=[-1]))
    if tta in ("vflip", "hvflip"):
        s2, t2 = _compute_maps_one(
            teacher, model, torch.flip(x, dims=[-2]), stats, cfg)
        _add(torch.flip(s2, dims=[-2]), torch.flip(t2, dims=[-2]))
    return accum_s / n_acc, accum_t / n_acc


@torch.inference_mode()
def score_batch_standard(teacher, model, x, stats, cfg):
    x = x.to(cfg.device, non_blocking=True)
    diff_n, _ = _maps_with_tta(teacher, model, x, stats, cfg, cfg.tta)
    up = F.interpolate(diff_n.unsqueeze(1),
                       size=(cfg.input_size, cfg.input_size),
                       mode="bilinear", align_corners=False).squeeze(1)
    return up.cpu()


@torch.inference_mode()
def score_sample_with_siblings(teacher, model, x_sample, stats, cfg):
    x_sample = x_sample.to(cfg.device, non_blocking=True)
    diff_n, t_feat = _maps_with_tta(teacher, model, x_sample, stats, cfg,
                                       cfg.tta)
    V, C, H_, W_ = t_feat.shape
    if V < 2:
        up = F.interpolate(diff_n.unsqueeze(1),
                           size=(cfg.input_size, cfg.input_size),
                           mode="bilinear", align_corners=False).squeeze(1)
        return up.cpu()
    t_flat = t_feat.permute(0, 2, 3, 1).reshape(V, -1, C)
    t_norm = F.normalize(t_flat, p=2, dim=-1)
    P = t_norm.shape[1]
    mv_dist = torch.empty(V, P, device=t_feat.device, dtype=torch.float32)
    for i in range(V):
        siblings = torch.cat(
            [t_norm[j] for j in range(V) if j != i], dim=0)
        sim = t_norm[i].float() @ siblings.float().T
        max_sim = sim.max(dim=1).values
        mv_dist[i] = 1.0 - max_sim
    mv_dist = mv_dist.reshape(V, H_, W_)
    sd_min = mv_dist.min(); sd_max = mv_dist.max()
    if (sd_max - sd_min) > 1e-9:
        mv_norm = (mv_dist - sd_min) / (sd_max - sd_min)
    else:
        mv_norm = torch.zeros_like(mv_dist)
    alpha = cfg.mv_alpha
    boost = (1.0 - alpha) + alpha * mv_norm
    boosted = diff_n * boost
    up = F.interpolate(boosted.unsqueeze(1),
                       size=(cfg.input_size, cfg.input_size),
                       mode="bilinear", align_corners=False).squeeze(1)
    return up.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_one(r: ImageRecord, transform, load_masks: bool, input_size: int):
    with Image.open(r.path) as im:
        x = transform(im.convert("RGB"))
    if load_masks and r.mask_path is not None:
        with Image.open(r.mask_path) as mm:
            mm = mm.convert("L").resize(
                (input_size, input_size), Image.NEAREST)
            m = (np.asarray(mm) > 127).astype(np.float32)
    else:
        m = np.zeros((input_size, input_size), dtype=np.float32)
    return x, m


def _score_records_standard(teacher, model, records, stats, cfg, load_masks):
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores, gts = {}, {}
    n_done = 0; last_log = 0
    for x, masks, idxs in loader:
        sm = score_batch_standard(teacher, model, x, stats, cfg).numpy()
        m_np = masks.numpy()
        for b in range(sm.shape[0]):
            scores[int(idxs[b])] = sm[b]
            gts[int(idxs[b])] = m_np[b]
        n_done += sm.shape[0]
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


def _score_records_by_sample(teacher, model, records, stats, cfg, load_masks):
    by_sample = defaultdict(list)
    for idx, r in enumerate(records):
        sid = r.sample_id or r.path.stem
        by_sample[sid].append((idx, r))
    print(f"      grouped {len(records)} images into {len(by_sample)} samples")
    transform = _build_transform(cfg.input_size)
    scores, gts = {}, {}
    n_done = 0; last_log = 0
    for sid, items in by_sample.items():
        imgs, masks_np = [], []
        for _idx, r in items:
            x, m = _load_one(r, transform, load_masks, cfg.input_size)
            imgs.append(x); masks_np.append(m)
        x_batch = torch.stack(imgs)
        sm = score_sample_with_siblings(teacher, model, x_batch, stats,
                                          cfg).numpy()
        for k, (idx, _r) in enumerate(items):
            scores[idx] = sm[k]
            gts[idx] = masks_np[k]
        n_done += len(items)
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _infer_feature_hw(teacher, cfg, device) -> tuple[int, int]:
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, cfg.input_size, cfg.input_size,
                              device=device)
        f = teacher(dummy)
    return tuple(f.shape[-2:])


def run_one_class(cls, records_all, teacher, cfg, run_dir, device,
                   feature_hw, local_saver=None) -> dict:
    hr(f"CLASS {cls}", "─")
    t_start = time.time()
    train_good = [r for r in records_all
                   if r.cls == cls and r.split == "train_good"]
    train_anom = [r for r in records_all
                   if r.cls == cls and r.split == "train_anomaly"]
    test       = [r for r in records_all
                   if r.cls == cls and r.split == "test"]
    print(f"  train_good={len(train_good)}  "
          f"train_anomaly={len(train_anom)}  test={len(test)}")
    if not train_good:
        return {"class": cls, "class_mean_ap": float("nan"),
                "eval_rows": [], "test_results": [], "elapsed_min": 0.0}

    model = UniADModel(
        teacher_dim=teacher.out_channels,
        feature_hw=feature_hw,
        model_dim=cfg.model_dim,
        n_heads=cfg.n_heads,
        n_enc_layers=cfg.n_enc_layers,
        n_dec_layers=cfg.n_dec_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        neighbour_radius=cfg.neighbour_radius,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UniAD: model_dim={cfg.model_dim} heads={cfg.n_heads} "
          f"enc={cfg.n_enc_layers} dec={cfg.n_dec_layers} "
          f"tokens={feature_hw[0]*feature_hw[1]} params={n_params/1e6:.2f}M")

    train_uniad(teacher, model, train_good, cfg, device)
    print(f"    [{now_hms()}] computing normalisation stats...")
    stats = compute_norm_stats(teacher, model, train_good, cfg, device)

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_uniad.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "stats": stats}, ck)
        print(f"    saved checkpoint -> {ck}")

    score_fn = (_score_records_by_sample
                if cfg.multiview == "sibling-bank"
                else _score_records_standard)

    eval_rows = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  multiview={cfg.multiview}  "
            f"alpha={cfg.mv_alpha}  tta={cfg.tta}")
        scores, gts = score_fn(teacher, model, train_anom, stats, cfg,
                                 load_masks=True)
        by_anom = defaultdict(list)
        for r_idx, sm in scores.items():
            r = train_anom[r_idx]
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            ap = pixel_average_precision(sm_smooth, gts[r_idx])
            by_anom[r.anomaly_type or "?"].append(ap)
            if local_saver is not None:
                local_saver.add(
                    cls=cls,
                    anomaly_type=r.anomaly_type or "unknown",
                    view_idx=int(r_idx),
                    score_map=sm_smooth,
                    gt_mask=gts[r_idx],
                    image_path=r.path)
        print(f"    {'anomaly_type':<14} {'n_views':>8} "
              f"{'pixel-AP (mean ± std)':>26}")
        per_type_means = []
        for a_type in sorted(by_anom):
            arr = np.asarray(by_anom[a_type])
            per_type_means.append(float(arr.mean()))
            print(f"    {a_type:<14} {len(arr):>8} "
                  f"{arr.mean():>15.4f} ± {arr.std():.4f}")
            eval_rows.append({"class": cls, "anomaly_type": a_type,
                               "n_views": int(len(arr)),
                               "ap_mean": float(arr.mean()),
                               "ap_std": float(arr.std()),
                               "ap_min": float(arr.min()),
                               "ap_max": float(arr.max())})
        class_mean_ap = (float(np.mean(per_type_means))
                          if per_type_means else 0.0)
        print(f"    >>> class {cls} mean pixel-AP: {class_mean_ap:.4f}")

    test_results = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images  "
            f"multiview={cfg.multiview}  tta={cfg.tta}")
        scores, _ = score_fn(teacher, model, test, stats, cfg,
                              load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            sm_final = maybe_resize_to_submission(sm_smooth)
            test_results.append((test[r_idx], sm_final))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"class": cls, "class_mean_ap": class_mean_ap,
            "eval_rows": eval_rows, "test_results": test_results,
            "elapsed_min": elapsed_min}


def write_submission(all_test_results, run_dir: Path,
                      zip_it: bool = True) -> Path:
    sub("calibrating scores and writing submission.csv")
    scores = [sm for _, sm in all_test_results]
    if not scores: raise RuntimeError("no test scores")
    lo, hi = calibrate_to_unit(scores)
    print(f"    global score calibration  lo={lo:.4f}  hi={hi:.4f}")
    csv_path = run_dir / "submission.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        for r, sm in all_test_results:
            normed = np.clip((sm - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            w.writerow([r.path.stem, float_matrix_to_q8rle(normed)])
            n += 1
    print(f"    wrote {n} rows -> {csv_path}")
    if zip_it:
        zip_path = csv_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w",
                              compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
        print(f"    zipped         -> {zip_path}")
        return zip_path
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Run config and CLI
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    input_size: int = 392
    # Teacher
    backbone: str = "dinov2_vits14"
    block_idx: int = 9
    # UniAD architecture
    model_dim: int = 256
    n_heads: int = 8
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    neighbour_radius: int = 7
    jitter_sigma: float = 0.0
    # Training
    epochs: int = 200
    total_iters: int | None = 5000
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    amp: bool = True
    num_workers: int = 8
    # Inference
    score_batch_size: int = 16
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    multiview: str = "none"
    mv_alpha: float = 0.5
    # Bookkeeping
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_checkpoints: bool = False
    zip_submission: bool = True
    run_tag: str = ""
    device: torch.device | None = None


def _backbone_short(name: str) -> str:
    return {
        "dinov2_vits14": "dnv2s14",
        "dinov2_vitb14": "dnv2b14",
        "dinov2_vitb14_reg": "dnv2b14reg",
        "dinov2_vitl14": "dnv2l14",
    }.get(name, name)


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "uniad",
        "backbone": cfg.backbone,
        "block_idx": cfg.block_idx,
        "input_size": cfg.input_size,
        "model_dim": cfg.model_dim,
        "n_heads": cfg.n_heads,
        "n_enc_layers": cfg.n_enc_layers,
        "n_dec_layers": cfg.n_dec_layers,
        "neighbour_radius": cfg.neighbour_radius,
        "jitter_sigma": cfg.jitter_sigma,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "multiview": cfg.multiview,
        "mv_alpha": cfg.mv_alpha,
        "tta": cfg.tta,
        "smooth_sigma": cfg.smooth_sigma,
        "seed": cfg.seed,
        "v": 2,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bb = _backbone_short(cfg.backbone)
    budget = (f"it{cfg.total_iters}"
              if (cfg.total_iters and cfg.total_iters > 0)
              else f"e{cfg.epochs}")
    mv_tag = "noMV" if cfg.multiview == "none" else f"MV-a{cfg.mv_alpha:.2f}"
    bits = (f"{stamp}_uniad_{bb}_b{cfg.block_idx}_in{cfg.input_size}"
            f"_d{cfg.model_dim}_e{cfg.n_enc_layers}d{cfg.n_dec_layers}"
            f"_nr{cfg.neighbour_radius}_{budget}_bs{cfg.batch_size}_{mv_tag}")
    if cfg.tta != "none":
        bits += f"_tta-{cfg.tta}"
    if cfg.run_tag:
        bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--input-size", type=int, default=392,
                    help="Must be multiple of 14 (DINOv2 patch size). "
                         "Common choices: 280 (20x20 tokens), 392 (28x28), "
                         "518 (37x37). Larger = finer spatial resolution "
                         "for small defects, but quadratic attention cost.")
    # Teacher
    ap.add_argument("--backbone", default="dinov2_vits14",
                    choices=sorted(DINOV2_DIMS.keys()))
    ap.add_argument("--block-idx", type=int, default=9,
                    help="Which DINOv2 intermediate block to use as the "
                         "reconstruction target. For vits14/vitb14 (12 "
                         "blocks), 9 = late-mid (semantic but not yet "
                         "task-specific). For vitl14 (24 blocks), try "
                         "around 18.")
    # Architecture
    ap.add_argument("--model-dim", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-enc-layers", type=int, default=4)
    ap.add_argument("--n-dec-layers", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--neighbour-radius", type=int, default=7,
                    help="Chebyshev radius around each token that is "
                         "FORBIDDEN to attend (incl. self). Paper default "
                         "is 7 on a 14x14 grid (effectively all-far "
                         "attention). On 28x28 (input 392), 7 keeps the "
                         "anti-shortcut absolute distance the same but "
                         "permits more long-range learning.")
    ap.add_argument("--jitter-sigma", type=float, default=0.0,
                    help="Std of Gaussian noise added to teacher features "
                         "at train time (scaled by per-channel std). "
                         "Disabled by default; try 0.2-0.5 if overfitting.")
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=5000,
                    help="Overrides --epochs. Doubled vs the EffNet "
                         "version because L2-normalised features need "
                         "more steps to converge to small loss.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Smaller default than EffNet version because "
                         "DINOv2 + larger token grid costs more memory.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    # Inference
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip"])
    ap.add_argument("--multiview", default="none",
                    choices=["none", "sibling-bank"],
                    help="`sibling-bank`: batch all 5 views of a sample "
                         "and boost each view's scores by cross-view "
                         "feature inconsistency. WARNING: known to "
                         "underperform `none` on this dataset because "
                         "the 5 views are different camera angles, not "
                         "perturbations of the same scene.")
    ap.add_argument("--mv-alpha", type=float, default=0.5,
                    help="Multi-view blend weight in [0, 1].")
    # Bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    if args.input_size % PATCH_SIZE != 0:
        raise SystemExit(f"[FATAL] --input-size must be multiple of "
                          f"{PATCH_SIZE} (DINOv2 patch size); "
                          f"got {args.input_size}. Try 280, 392, or 518.")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        input_size=args.input_size,
        backbone=args.backbone, block_idx=args.block_idx,
        model_dim=args.model_dim, n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        neighbour_radius=args.neighbour_radius,
        jitter_sigma=args.jitter_sigma,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        amp=not args.no_amp, num_workers=args.num_workers,
        score_batch_size=args.score_batch_size,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
        multiview=args.multiview, mv_alpha=args.mv_alpha,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        save_checkpoints=args.save_checkpoints,
        zip_submission=not args.no_zip,
        run_tag=args.run_tag,
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"UNIAD-DINOV2 — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  teacher          : frozen {cfg.backbone}, "
              f"block {cfg.block_idx}, L2-normed -> "
              f"{DINOV2_DIMS[cfg.backbone]} ch")
        print(f"  input_size       : {cfg.input_size}  "
              f"(tokens: {cfg.input_size // PATCH_SIZE}x"
              f"{cfg.input_size // PATCH_SIZE})")
        print(f"  model_dim        : {cfg.model_dim}  "
              f"n_heads={cfg.n_heads}")
        print(f"  enc/dec layers   : {cfg.n_enc_layers} / {cfg.n_dec_layers}")
        print(f"  neighbour_radius : {cfg.neighbour_radius}")
        print(f"  jitter_sigma     : {cfg.jitter_sigma}")
        if cfg.total_iters and cfg.total_iters > 0:
            print(f"  total_iters      : {cfg.total_iters}  "
                  f"(overrides --epochs={cfg.epochs})")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / wd          : {cfg.lr} / {cfg.weight_decay}")
        print(f"  amp              : {cfg.amp}    "
              f"num_workers: {cfg.num_workers}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  multiview        : {cfg.multiview}  alpha={cfg.mv_alpha}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            cfg_dump = {k: (str(v) if isinstance(v, (Path, torch.device))
                              else v)
                          for k, v in asdict(cfg).items()}
            json.dump(cfg_dump, f, indent=2, default=str)

        teacher = DINOv2Teacher(model_name=cfg.backbone,
                                  block_idx=cfg.block_idx).to(device).eval()
        feature_hw = _infer_feature_hw(teacher, cfg, device)
        print(f"\n  teacher feature grid: {feature_hw} "
              f"({feature_hw[0]*feature_hw[1]} tokens)")

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("\n[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): "
              f"{', '.join(classes)}")

        local_saver: LocalPredSaver | None = None
        if not cfg.skip_eval and not args.no_save_local_preds:
            local_saver = LocalPredSaver()

        all_test_results, all_eval_rows = [], []
        class_aps, class_elapsed = {}, {}
        for cls in classes:
            res = run_one_class(cls, records, teacher, cfg, run_dir,
                                  device, feature_hw,
                                  local_saver=local_saver)
            all_test_results.extend(res["test_results"])
            all_eval_rows.extend(res["eval_rows"])
            class_aps[cls] = res["class_mean_ap"]
            class_elapsed[cls] = res["elapsed_min"]

        if local_saver is not None and len(local_saver) > 0:
            local_saver.save(run_dir / "local_predictions.npz")

        hr("LOCAL VALIDATION SUMMARY", "=")
        print(f"  {'class':<10} {'mean pixel-AP':>15} {'time (min)':>12}")
        for cls in classes:
            print(f"  {cls:<10} {class_aps.get(cls, float('nan')):>15.4f} "
                  f"{class_elapsed.get(cls, 0):>12.1f}")
        valid_aps = [v for v in class_aps.values() if not math.isnan(v)]
        overall_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")
        if valid_aps:
            print(f"  {'OVERALL':<10} {overall_ap:>15.4f}")

        if all_eval_rows:
            tab_path = run_dir / "local_eval.csv"
            with open(tab_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f,
                                     fieldnames=list(all_eval_rows[0].keys()))
                w.writeheader()
                for row in all_eval_rows: w.writerow(row)
            print(f"  saved per-(class, anomaly_type) AP -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload: {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        row = {
            "run_id": run_id, "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": f"UNIAD_{_backbone_short(cfg.backbone).upper()}",
            "feature_layers": f"block_{cfg.block_idx}",
            "target_layer": "",
            "input_size": cfg.input_size,
            "smooth_sigma": cfg.smooth_sigma, "tta": cfg.tta,
            "batch_size": cfg.batch_size,
            "score_batch_size": cfg.score_batch_size,
            "seed": cfg.seed, "n_classes": len(classes),
            **{f"AP_{c}": f"{class_aps.get(c, float('nan')):.4f}"
                for c in sorted(class_aps)},
            "AP_overall": f"{overall_ap:.4f}",
            "runtime_min": f"{(time.time() - t_total) / 60:.1f}",
            "submission_path": str(run_dir / "submission.zip")
                                if not cfg.skip_submission else "",
            "notes": (f"uniad-{_backbone_short(cfg.backbone)}-b{cfg.block_idx} "
                      f"L2norm multiview={cfg.multiview} "
                      f"mv_alpha={cfg.mv_alpha} d{cfg.model_dim} "
                      f"e{cfg.n_enc_layers}d{cfg.n_dec_layers} "
                      f"nr{cfg.neighbour_radius} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()