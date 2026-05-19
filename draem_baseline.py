"""Spacepresso DRAEM baseline — v1.1.

Implements "DRAEM: A discriminatively trained reconstruction embedding
for surface anomaly detection" (Zavrtanik et al., ICCV 2021).

# v1.1 changelog (bugs fixed for stacker integration)

  - Grad-clipping ordering: clip is now between scaler.unscale_() and
    scaler.step(), not after the step (which was a no-op at best and a
    hard error on recent PyTorch).
  - Synthetic-anomaly diversity: per-worker advancing Generator instead
    of a deterministic (worker_seed, idx) hash. Old behaviour produced
    only ~|train_good| unique Perlin shapes per worker for the entire
    run; new behaviour reseeds-and-advances per __getitem__ call while
    staying reproducible from the worker seed.
  - torch.cuda.amp.autocast(...) → torch.amp.autocast("cuda", ...).
  - AMP no longer gated on unet_base >= 32.

# Stacker integration contract (xgboost_stacker_v5.py)

  This baseline writes the SAME three artefacts every other baseline
  writes, in the same run_dir layout. The shell stacker just needs:

      $DRAEM_RUN/submission.csv          ← test scores (q8rle)
      $DRAEM_RUN/local_predictions.npz   ← per-pixel val scores + GTs

  Critically, each call to LocalPredSaver.add() passes image_path=r.path
  (full path). The saver stores it; xgboost_stacker_v5 reads it via
  Path(p).name and parses _viewN to group siblings. This is what
  makes the v4 cross-view aggregates (xv_max/xv_mean/xv_std/xv_lonely),
  CVD, and per-(class, view) rank-norm light up for DRAEM. Without
  image_path the stacker silently falls back to per-class rank-norm
  and zero cross-view features.

  The run_dir name starts with "draem_" so detect_model_family in the
  stacker can recognise it if you add ("draem", "draem") to
  MODEL_FAMILY_PATTERNS. Without that addition the family resolves to
  "unknown", which in v4/v5 defaults to small_cc=0 — exactly what we
  want for DRAEM anyway (no CC suppression needed; softmax-over-mask
  scores are already well-localised).

# Why this is an interesting fusion partner for the stacker

  - All your other bases (PatchCore, CutPaste, RD, EfficientAD, UniAD,
    FastFlow, CFA) score by FEATURE DISTANCE in some pretrained
    backbone's latent space. They share a failure mode: defects that
    look in-distribution in feature space but are obvious to a
    pixel-space segmentation head get systematically missed.
  - DRAEM operates on RGB end-to-end with a SEGMENTATION head trained
    on synthetic defects. It learns "what a defect looks like in
    pixel space", which is fundamentally orthogonal to "is this
    feature far from normal-data feature distribution". Different
    error modes → stacker gain.

# Architecture (unchanged from v1)

  RECONSTRUCTIVE U-NET (R):
    Input  : 3xHxW (RGB in [0, 1], no ImageNet norm)
    Encoder: 6 stride-2 conv blocks (base → 32*base channels)
    Decoder: symmetric up-convs with skip connections
    Output : 3xHxW reconstruction
    Trained to denoise images that have synthetic anomalies pasted on
    them, back to the clean original.

  DISCRIMINATIVE U-NET (D):
    Input  : concat([anom_img, R(anom_img)], dim=1)  → 6xHxW
    Output : 2xHxW logits (class 0 = normal, class 1 = anomaly)

  TOTAL PARAMS: ~63M for base=32 (R: ~31M, D: ~31M).

# Synthetic anomaly generation (Perlin-noise + texture source)

  1. Sample fractal Perlin noise at the image resolution; threshold at
     a percentile sampled in [60, 85] to produce an irregular mask M
     with reasonable coverage. Sometimes rotated 90/180/270°.
  2. Sample a "texture" image T:
       * 50%: random colour gradient (horizontal or vertical)
       * 50%: flat random colour
     (We don't ship DTD; the gradient + solid scheme works for
     Spacepresso and avoids a 1 GB download.)
  3. Beta-blend the clean image with T weighted by M:
        x_anom = (1 - β·M)·x_clean + (β·M)·T,   β ~ U(0.15, 1.0)
  4. Reconstruction target = x_clean (R learns to undo corruption).
  5. Segmentation target = M (binary, gives D the per-pixel label).

# Losses

    L_R = L_l2(R(x_anom), x_clean) + L_ssim(R(x_anom), x_clean)
    L_D = focal_loss(D([x_anom, R(x_anom)]), M)
    L   = L_R + L_D

# Multi-view note (unchanged)

  Sibling-bank multi-view (the one used in EfficientAD / UniAD)
  compares TEACHER FEATURES across the 5 views of a sample. DRAEM has
  no teacher backbone — the reconstructive network reads RGB directly
  and the discriminative head outputs per-pixel logits. There is no
  shared feature space across views to compare against. We therefore
  do TTA-only inference here. Score-level cross-view aggregation
  happens inside the stacker (v4 xv_aggregates / CVD features) which
  is exactly the right layer for it.

# Memory and speed (L4 24 GB)

  Per class at input 256, batch 8, 2500 iters:
    Training: ~5 min  (R + D forward + backward, AMP)
    Eval:     ~10 s
    Test:     ~25 s
  Full 8-class run: ~50 min.

# Dependencies

  patchcore_baseline_v2.py and local_preds_saver.py in same directory.
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
)
from local_preds_saver import LocalPredSaver


PROJECT_ROOT = Path("/work/u10813429/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"


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
# Perlin noise (numpy, deterministic given a Generator)
# ─────────────────────────────────────────────────────────────────────────────
def _smoothstep(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _generate_perlin_2d(shape, res, rng: np.random.Generator):
    """Classic 2D Perlin noise. shape = (H, W); res = (Hres, Wres) such
    that H, W are multiples of res. Returns values in roughly [-1, 1]."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]] \
            .transpose(1, 2, 0) % 1.0
    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:,   :-d[1]]
    g01 = gradients[:-d[0],  d[1]:]
    g11 = gradients[d[0]:,   d[1]:]
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum((grid - np.array([1, 0])) * g10, 2)
    n01 = np.sum((grid - np.array([0, 1])) * g01, 2)
    n11 = np.sum((grid - np.array([1, 1])) * g11, 2)
    t = _smoothstep(grid)
    n0 = n00 * (1 - t[..., 0]) + t[..., 0] * n10
    n1 = n01 * (1 - t[..., 0]) + t[..., 0] * n11
    return np.sqrt(2) * ((1 - t[..., 1]) * n0 + t[..., 1] * n1)


def _generate_fractal_noise(shape, res, octaves=4,
                              persistence=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * _generate_perlin_2d(
            shape, (frequency * res[0], frequency * res[1]), rng)
        frequency *= 2
        amplitude *= persistence
    return noise


def sample_anomaly_mask(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Binary (H, W) mask via Perlin noise + percentile threshold.

    Roughly 15–40% positive coverage (depends on the sampled threshold
    percentile in [60, 85]). Two scales mixed for shape variety."""
    scale_a = rng.choice([2, 4, 8])
    scale_b = rng.choice([8, 16])
    base_a = ((H + scale_a * 8 - 1) // (scale_a * 8)) * (scale_a * 8)
    noise_a = _generate_fractal_noise(
        (base_a, base_a), (scale_a, scale_a),
        octaves=3, persistence=0.5, rng=rng)[:H, :W]
    base_b = ((H + scale_b * 8 - 1) // (scale_b * 8)) * (scale_b * 8)
    noise_b = _generate_fractal_noise(
        (base_b, base_b), (scale_b, scale_b),
        octaves=3, persistence=0.5, rng=rng)[:H, :W]
    noise = 0.5 * noise_a + 0.5 * noise_b
    pct = rng.uniform(60, 85)
    th = np.percentile(noise, pct)
    mask = (noise > th).astype(np.float32)
    if rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        mask = np.rot90(mask, k=k).copy()
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic anomaly generation (Perlin mask + texture)
# ─────────────────────────────────────────────────────────────────────────────
def random_texture(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap texture: either a 2-colour gradient or a flat colour, plus
    light pixel noise. (H, W, 3) float32 in [0, 1]."""
    if rng.random() < 0.5:
        c1 = rng.random(3, dtype=np.float64)
        c2 = rng.random(3, dtype=np.float64)
        if rng.random() < 0.5:
            ramp = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
        else:
            ramp = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        c1f = c1.astype(np.float32)[None, None, :]
        c2f = c2.astype(np.float32)[None, None, :]
        tex = c1f * (1.0 - ramp) + c2f * ramp
        tex = np.broadcast_to(tex, (H, W, 3)).copy()
    else:
        c = rng.random(3, dtype=np.float64).astype(np.float32)
        tex = np.broadcast_to(c[None, None, :], (H, W, 3)).copy()
    noise = rng.normal(0.0, 0.03, size=(H, W, 3)).astype(np.float32)
    tex = np.clip(tex + noise, 0.0, 1.0)
    return tex


def synthesise_anomaly(img01: np.ndarray, rng: np.random.Generator
                        ) -> tuple[np.ndarray, np.ndarray]:
    """img01: (H, W, 3) float32 in [0, 1].
    Returns (corrupted_img01, binary_mask)."""
    H, W = img01.shape[:2]
    mask = sample_anomaly_mask(H, W, rng)        # (H, W) in {0, 1}
    tex  = random_texture(H, W, rng)              # (H, W, 3) in [0, 1]
    beta = float(rng.uniform(0.15, 1.0))
    m3 = mask[..., None]
    corrupted = (1.0 - beta * m3) * img01 + (beta * m3) * tex
    corrupted = np.clip(corrupted, 0.0, 1.0).astype(np.float32)
    return corrupted, mask.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
#
# IMPORTANT — augmentation diversity (v1.1 fix):
# Each DataLoader worker keeps a single advancing Generator in module
# scope (set by worker_init_fn). __getitem__ pulls from it so every
# call produces a different Perlin mask + texture, while the worker
# seed alone determines the full sequence. Reproducible per seed,
# diverse across the run.
# ─────────────────────────────────────────────────────────────────────────────
_WORKER_RNG: np.random.Generator | None = None


def worker_init_fn(_worker_id):
    """Seed numpy/random per worker AND build the advancing Generator
    used by DraemTrainDataset.__getitem__."""
    global _WORKER_RNG
    base = torch.initial_seed() % 2 ** 32
    np.random.seed(base)
    random.seed(base)
    _WORKER_RNG = np.random.default_rng(base)


class DraemTrainDataset(Dataset):
    """train/good images. Each __getitem__ returns:
        clean : (3, H, W) float32 in [0, 1]
        anom  : (3, H, W) float32 in [0, 1] (clean + synthetic defect)
        mask  : (H, W)    float32 in {0, 1}
    Half the time we return clean as-is with a zero mask, so the
    discriminator also sees normal pairs (prevents D from collapsing
    to always-predicting-anomaly)."""
    def __init__(self, records: list[ImageRecord], input_size: int,
                 anomaly_prob: float = 0.5, seed: int = 0):
        self.records = records
        self.input_size = input_size
        self.anomaly_prob = anomaly_prob
        self.seed = seed
        # DRAEM normalises to [0, 1], NOT ImageNet stats.
        self.resize = transforms.Resize((input_size, input_size))

    def __len__(self): return len(self.records)

    def _load_rgb01(self, path: Path) -> np.ndarray:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = self.resize(im)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr

    def _get_rng(self, item_idx: int) -> np.random.Generator:
        """Returns the worker-local advancing Generator. Falls back to
        a deterministic one when running with num_workers=0 (no worker
        init has fired) — that path is only used for ad-hoc debugging."""
        global _WORKER_RNG
        if _WORKER_RNG is None:
            seed = ((self.seed * 1_000_003 + item_idx * 9973)
                    & 0xFFFFFFFF)
            _WORKER_RNG = np.random.default_rng(seed)
        return _WORKER_RNG

    def __getitem__(self, i):
        r = self.records[i]
        rng = self._get_rng(i)
        img = self._load_rgb01(r.path)
        H, W = img.shape[:2]
        if rng.random() < self.anomaly_prob:
            anom, mask = synthesise_anomaly(img, rng)
        else:
            anom = img.copy()
            mask = np.zeros((H, W), dtype=np.float32)
        clean_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        anom_t  = torch.from_numpy(anom).permute(2, 0, 1).contiguous()
        mask_t  = torch.from_numpy(mask).contiguous()
        return clean_t, anom_t, mask_t


class DraemInferenceDataset(Dataset):
    """test/train_anomaly images. Returns (img01, mask_or_zeros, index)."""
    def __init__(self, records: list[ImageRecord], input_size: int,
                 load_masks: bool):
        self.records = records
        self.input_size = input_size
        self.load_masks = load_masks
        self.resize = transforms.Resize((input_size, input_size))

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r.path) as im:
            im = im.convert("RGB")
            im = self.resize(im)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.load_masks and r.mask_path is not None:
            with Image.open(r.mask_path) as mm:
                mm = mm.convert("L").resize(
                    (self.input_size, self.input_size), Image.NEAREST)
                m = (np.asarray(mm) > 127).astype(np.float32)
        else:
            m = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        return x, torch.from_numpy(m), i


# ─────────────────────────────────────────────────────────────────────────────
# Network blocks
# ─────────────────────────────────────────────────────────────────────────────
def _conv_relu(c_in, c_out, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=p, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=c_out),
        nn.ReLU(inplace=True),
    )


class _DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            _conv_relu(c_in, c_out),
            _conv_relu(c_out, c_out),
        )
    def forward(self, x): return self.net(x)


class _Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(c_in, c_out)
    def forward(self, x): return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        # c_in = channels BEFORE skip concat; after upsample we
        # concat with skip (c_out channels), so DoubleConv input is
        # c_in + c_out.
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False)
        self.conv = _DoubleConv(c_in + c_out, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DraemUNet(nn.Module):
    """6-level U-Net used for both R (in=3, out=3) and D (in=6, out=2)."""
    def __init__(self, in_channels: int, out_channels: int,
                 base: int = 32):
        super().__init__()
        c = base
        self.inc   = _DoubleConv(in_channels, c)
        self.down1 = _Down(c,        c * 2)
        self.down2 = _Down(c * 2,    c * 4)
        self.down3 = _Down(c * 4,    c * 8)
        self.down4 = _Down(c * 8,    c * 16)
        self.down5 = _Down(c * 16,   c * 32)
        self.up1   = _Up(c * 32, c * 16)
        self.up2   = _Up(c * 16, c * 8)
        self.up3   = _Up(c * 8,  c * 4)
        self.up4   = _Up(c * 4,  c * 2)
        self.up5   = _Up(c * 2,  c)
        self.outc  = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        u = self.up1(x5, x4)
        u = self.up2(u,  x3)
        u = self.up3(u,  x2)
        u = self.up4(u,  x1)
        u = self.up5(u,  x0)
        return self.outc(u)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────
def ssim_loss(x: torch.Tensor, y: torch.Tensor,
               window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM loss with small numerical constants. Always called in fp32
    (via a nested autocast disable in the training loop) to keep the
    1e-4 / 9e-4 constants representable."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1.0, neginf=0.0)

    C = x.shape[1]
    half = window // 2
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - half
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / (g.sum() + 1e-8)

    k2 = (g[:, None] * g[None, :]).clamp_min(1e-8)
    kernel = k2.expand(C, 1, window, window).contiguous()
    pad = half

    mu_x = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=C)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    sigma_x2 = sigma_x2.clamp(min=1e-8)
    sigma_y2 = sigma_y2.clamp(min=1e-8)
    sigma_xy = sigma_xy.clamp(min=-1e6, max=1e6)

    c1 = 1e-4
    c2 = 9e-4

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / den.clamp_min(1e-8)
    return 1.0 - torch.nan_to_num(ssim_map.mean(), nan=1.0)


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
                gamma: float = 2.0, alpha: float = 0.5
                ) -> torch.Tensor:
    """Multi-class focal loss for the per-pixel segmentation head.
    logits: (B, 2, H, W); target: (B, H, W) long with class indices 0/1.
    `alpha` is the weight on the positive (anomaly) class."""
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    target = target.long()
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
    w = torch.where(target == 1, torch.full_like(p_t, alpha),
                                  torch.full_like(p_t, 1.0 - alpha))
    loss = -w * ((1.0 - p_t) ** gamma) * logp_t
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_draem(net_r: DraemUNet, net_d: DraemUNet,
                  records: list[ImageRecord],
                  cfg: "RunConfig", device: torch.device) -> None:
    ds = DraemTrainDataset(records, input_size=cfg.input_size,
                              anomaly_prob=cfg.anomaly_prob, seed=cfg.seed)
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

    params = list(net_r.parameters()) + list(net_d.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))

    # AMP (no longer gated on unet_base; the SSIM block disables
    # autocast internally for the constants that need fp32).
    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"    [{now_hms()}] training: {n_epochs} epochs x "
          f"{iters_per_epoch} iters ({total_iters} total)  "
          f"bs={cfg.batch_size}  amp={use_amp}  "
          f"anomaly_prob={cfg.anomaly_prob}")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    for epoch in range(n_epochs):
        net_r.train(); net_d.train()
        loss_sum = lr_sum = ld_sum = 0.0
        n = 0
        for clean, anom, mask in loader:
            clean = clean.to(device, non_blocking=True)
            anom  = anom.to(device,  non_blocking=True)
            mask  = mask.to(device,  non_blocking=True)

            clean = torch.nan_to_num(clean, nan=0.0, posinf=1.0, neginf=0.0)
            anom  = torch.nan_to_num(anom,  nan=0.0, posinf=1.0, neginf=0.0)
            mask  = torch.nan_to_num(mask,  nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                recon = net_r(anom)
                d_in = torch.cat([anom, recon], dim=1)         # (B, 6, H, W)
                logits = net_d(d_in)                            # (B, 2, H, W)
                L_l2 = F.mse_loss(recon.float(), clean.float())
                # SSIM uses tiny constants — run it in fp32 even under AMP.
                with torch.amp.autocast("cuda", enabled=False):
                    L_ssim = ssim_loss(recon, clean)
                L_R = L_l2 + L_ssim
                L_D = focal_loss(logits, mask, gamma=cfg.focal_gamma,
                                   alpha=cfg.focal_alpha)
                loss = L_R + L_D

            # ── Correct AMP grad-clip ordering (v1.1 fix) ───────────
            # backward → unscale_ → clip_grad_norm_ → step → update.
            # The v1 ordering ran clip AFTER step, which is a no-op
            # at best and raises on recent PyTorch.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            B = clean.shape[0]
            loss_sum += loss.item() * B
            lr_sum   += L_R.item()  * B
            ld_sum   += L_D.item()  * B
            n += B
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={loss_sum/max(n,1):.4f}  "
                  f"L_R={lr_sum/max(n,1):.4f}  "
                  f"L_D={ld_sum/max(n,1):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    net_r.eval(); net_d.eval()
    print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Inference primitives
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _score_one_pass(net_r, net_d, x: torch.Tensor,
                     cfg: "RunConfig") -> torch.Tensor:
    """Per-pixel anomaly probability at the input resolution.
    Shape (B, H, W) on the configured device."""
    use_amp = (cfg.device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        recon = net_r(x)
        d_in = torch.cat([x, recon], dim=1)
        logits = net_d(d_in)
    prob = F.softmax(logits.float(), dim=1)[:, 1]    # (B, H, W)
    return prob


@torch.inference_mode()
def score_batch(net_r, net_d, x: torch.Tensor, cfg: "RunConfig"
                  ) -> torch.Tensor:
    """TTA-averaged anomaly probability on CPU."""
    x = x.to(cfg.device, non_blocking=True)
    acc = None; n = 0

    def _add(s):
        nonlocal acc, n
        if acc is None: acc = s.clone()
        else: acc += s
        n += 1

    _add(_score_one_pass(net_r, net_d, x, cfg))
    if cfg.tta in ("hflip", "hvflip"):
        s = _score_one_pass(net_r, net_d, torch.flip(x, dims=[-1]), cfg)
        _add(torch.flip(s, dims=[-1]))
    if cfg.tta in ("vflip", "hvflip"):
        s = _score_one_pass(net_r, net_d, torch.flip(x, dims=[-2]), cfg)
        _add(torch.flip(s, dims=[-2]))
    return (acc / max(n, 1)).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _score_records(net_r, net_d, records, cfg, load_masks):
    ds = DraemInferenceDataset(records, input_size=cfg.input_size,
                                  load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores, gts = {}, {}
    n_done = 0; last_log = 0
    for x, masks, idxs in loader:
        sm = score_batch(net_r, net_d, x, cfg).numpy()
        m_np = masks.numpy()
        for b in range(sm.shape[0]):
            scores[int(idxs[b])] = sm[b]
            gts[int(idxs[b])] = m_np[b]
        n_done += sm.shape[0]
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


def run_one_class(cls, records_all, cfg, run_dir, device,
                   local_saver=None) -> dict:
    """If `local_saver` is given, every local-val (score_map, gt_mask)
    pair is appended to it along with image_path=r.path so the stacker
    can train on the raw pixel predictions AND parse views for the
    cross-view feature family."""
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

    net_r = DraemUNet(in_channels=3, out_channels=3,
                        base=cfg.unet_base).to(device)
    net_d = DraemUNet(in_channels=6, out_channels=2,
                        base=cfg.unet_base).to(device)
    n_params = (sum(p.numel() for p in net_r.parameters())
                + sum(p.numel() for p in net_d.parameters()))
    print(f"  DRAEM: unet_base={cfg.unet_base}  "
          f"params={n_params/1e6:.2f}M")

    train_draem(net_r, net_d, train_good, cfg, device)

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_draem.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net_r": net_r.state_dict(),
                    "net_d": net_d.state_dict()}, ck)
        print(f"    saved checkpoint -> {ck}")

    eval_rows = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  tta={cfg.tta}")
        scores, gts = _score_records(net_r, net_d, train_anom, cfg,
                                        load_masks=True)
        by_anom = defaultdict(list)
        for r_idx, sm in scores.items():
            r = train_anom[r_idx]
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            ap = pixel_average_precision(sm_smooth, gts[r_idx])
            by_anom[r.anomaly_type or "?"].append(ap)
            if local_saver is not None:
                # image_path is REQUIRED for the v4 stacker's view
                # parsing (PATH_VIEW_RE on Path(p).name → sample_id,
                # view_idx). Cross-view features depend on this.
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
        sub(f"scoring {len(test)} test images  tta={cfg.tta}")
        scores, _ = _score_records(net_r, net_d, test, cfg,
                                      load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            sm_final = maybe_resize_to_submission(sm_smooth)
            test_results.append((test[r_idx], sm_final))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del net_r, net_d
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
    input_size: int = 256
    unet_base: int = 32          # base channels; ~63M total params at 32
    # Training
    epochs: int = 200
    total_iters: int | None = 2500
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    anomaly_prob: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5
    amp: bool = True
    num_workers: int = 8
    # Inference
    score_batch_size: int = 16
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    # Bookkeeping
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_checkpoints: bool = False
    zip_submission: bool = True
    run_tag: str = ""
    device: torch.device | None = None


def make_run_id(cfg: RunConfig) -> str:
    """Run-dir name starts with `<timestamp>_draem_...`. The
    xgboost_stacker's detect_model_family can be extended to map
    "draem" → "draem" via one line in MODEL_FAMILY_PATTERNS; without
    that change, the family is reported as "unknown" which still
    works (small_cc default 0 in v4/v5)."""
    fp = json.dumps({
        "method": "draem",
        "input_size": cfg.input_size,
        "unet_base": cfg.unet_base,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "anomaly_prob": cfg.anomaly_prob,
        "focal_gamma": cfg.focal_gamma,
        "focal_alpha": cfg.focal_alpha,
        "tta": cfg.tta,
        "smooth_sigma": cfg.smooth_sigma,
        "seed": cfg.seed,
        "v": "1.1",
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    budget = (f"it{cfg.total_iters}"
              if (cfg.total_iters and cfg.total_iters > 0)
              else f"e{cfg.epochs}")
    bits = (f"{stamp}_draem_in{cfg.input_size}_b{cfg.unet_base}"
            f"_{budget}_bs{cfg.batch_size}")
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
    ap.add_argument("--input-size", type=int, default=256,
                    help="Must be multiple of 64 (six 2x downsamples in U-Net).")
    ap.add_argument("--unet-base", type=int, default=32,
                    help="Base channel count for both U-Nets. 32 → ~63M "
                         "total params. 24 → ~35M (cheaper). 16 → ~16M.")
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=2500,
                    help="Overrides --epochs (matches budget of other "
                         "baselines).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--anomaly-prob", type=float, default=0.5,
                    help="Fraction of training samples where a synthetic "
                         "defect is added. 0.5 sees both clean and "
                         "corrupted batches — prevents D from collapsing "
                         "to always-predict-anomaly.")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.5,
                    help="Weight on the positive (anomaly) class in the "
                         "focal loss. 0.5 = balanced.")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    # Inference
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip"])
    # Bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true",
                    help="Disable saving local_predictions.npz "
                         "(default: save). Disabling this breaks "
                         "downstream stacker training.")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    if args.input_size % 64 != 0:
        raise SystemExit(f"[FATAL] --input-size must be multiple of 64 "
                          f"(U-Net has 6 downsamples). Got {args.input_size}.")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        input_size=args.input_size, unet_base=args.unet_base,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        anomaly_prob=args.anomaly_prob,
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
        amp=not args.no_amp, num_workers=args.num_workers,
        score_batch_size=args.score_batch_size,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
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
        hr(f"DRAEM v1.1 — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  unet_base        : {cfg.unet_base}")
        if cfg.total_iters and cfg.total_iters > 0:
            print(f"  total_iters      : {cfg.total_iters}  "
                  f"(overrides --epochs={cfg.epochs})")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / wd          : {cfg.lr} / {cfg.weight_decay}")
        print(f"  anomaly_prob     : {cfg.anomaly_prob}")
        print(f"  focal gamma/alpha: {cfg.focal_gamma} / {cfg.focal_alpha}")
        print(f"  amp              : {cfg.amp}    "
              f"num_workers: {cfg.num_workers}")
        print(f"  score_batch_size : {cfg.score_batch_size}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            cfg_dump = {k: (str(v) if isinstance(v, (Path, torch.device))
                              else v)
                          for k, v in asdict(cfg).items()}
            json.dump(cfg_dump, f, indent=2, default=str)

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
            res = run_one_class(cls, records, cfg, run_dir, device,
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
            "backbone": "DRAEM",
            "feature_layers": "",
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
            "notes": (f"draem v1.1 base{cfg.unet_base} "
                      f"anomaly_prob={cfg.anomaly_prob} "
                      f"focal_alpha={cfg.focal_alpha} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()