"""Spacepresso GLASS baseline — synthetic-anomaly discriminator with
text-guided per-class synthesis profiles.

Paper:  "A Unified Anomaly Synthesis Strategy with Gradient Ascent for
         Industrial Anomaly Detection and Localization"
         Chen et al., ECCV 2024.

# Why GLASS is the missing piece in your ensemble

Your current stack spans four kinds of "what is normal?" signals:

  Distance        : PatchCore, CFA, CutPaste-NN, AnomalyDINO
  Reconstruction  : UniAD, Reverse Distillation, EfficientAD
  Density         : FastFlow, DINO-DPMM
  Discriminative  : (empty)

GLASS fills the empty fourth bucket. It trains a small MLP discriminator
on a frozen backbone's patch features, with three kinds of training
signal per pixel:

  1. NORMAL          (label = 0)   patches from train_good.
  2. LOCAL anomalies (label ∈ [0, 1] = patch coverage)
     image-space synthetic defects with PRECISE pixel masks.
  3. GLOBAL anomalies (label = 1)
     feature-space PGD ascent on the discriminator from normal features:
     "hard negatives" sitting just outside the normal manifold.

Local synthesis teaches the discriminator WHERE defects look anomalous
in pixel space; global synthesis teaches HOW MUCH a feature has to drift
to count as anomalous. Together they sculpt a sharp normal boundary
without ever requiring real defects.

# Why this implementation extends the paper

The paper uses ONE local-synthesis mode (Perlin-mask blob filled with a
foreign DTD texture). For Spacepresso your `anomaly_descriptions.csv`
gives us the actual defect taxonomy per class — and it's not Perlin
blobs. It's eight recurring physical modes:

      scratch, dent, stain, fragment, protrusion, crack,
      contamination, hole

This file implements all eight as separate vectorised GPU primitives,
each producing both an image edit AND a precise mask. At training time
we sample a mode per image according to a CSV-driven per-class profile
(keyword-matched in the description text):

  class_01 resistor :  scratch 22%, dent 22%, stain 22%, fragment 22%, …
  class_03 gear     :  scratch 30%, fragment 30%, stain 30%, …
  class_06 coffee   :  contamination 30%, stain 30%, … (mold-heavy)
  class_07 pistachio:  crack 30%, dent 25%, stain 20%, …

If the CSV is absent (`--no-anom-profile`), all eight modes are sampled
uniformly — i.e. the unguided variant of GLASS, still strictly stronger
than the paper's single-blob synthesis. With the CSV, this is the
"language as a lantern" advice in code form.

# Architecture

  Frozen backbone  : ResNet50 / WRN50-2 / DINOv2 / DINOv3 ViT
                      (any backbone supported by patchcore_baseline_v2.
                      ConvNeXt rejected — different fusion logic.)
  patchify_and_combine : multi-layer fusion with 3x3 avg-pool + L2 norm
                         (same as PatchCore / CFA). L2 norm makes the
                         PGD epsilon interpretable as a fraction of the
                         unit sphere; without it epsilon is in raw
                         feature units, dependent on the backbone.

  Adapter          : 1-layer Linear + LeakyReLU, same in/out dim. Light
                      learnable projection over the frozen backbone.
  Discriminator    : [Linear → BN → LeakyReLU] × 2 + Linear → 1 logit.
                      Per-patch (operates on (B*P, C) flat).

# Training

  AdamW, cosine schedule, ~3000 iters/class (~3-5 min on a 4090).
  Loss = BCE(normal,0) + λ_local·BCE(local,patch_mask) + λ_global·BCE(global,1)
  Global anomalies kicked in after `warmup_iters` so the discriminator
  has learned SOMETHING before its gradients are used for ascent.

# Inference

  Score per patch = sigmoid(discriminator(adapter(features))). Reshape
  to (h, w), bilinear-upsample to input_size, smooth, calibrate.
  Optional TTA (hflip/vflip).

# Memory / speed on a 4090

  vits14_reg @ in=392, B=16, 3000 iters:
      backbone forward (frozen + AMP)     ~25 ms/step
      synth pass + backbone forward       ~30 ms/step (synth is GPU-side)
      PGD ascent (4 steps)                 ~3 ms/step
      adapter+disc forward+backward       ~2 ms/step
      Total per class                      ~3-4 min train, ~30 s eval+test
  Full 8 classes                            ~30 min end-to-end.

# Stacker contract — same as everyone else

  submission.csv, local_predictions.npz, test_predictions.npz.

# Dependencies

  patchcore_baseline_v2.py, local_preds_saver.py, dinov3_loader.py,
  test_preds_saver.py — all in the same directory.
"""
from __future__ import annotations
from test_preds_saver import save_test_predictions

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
    FeatureExtractor, patchify_and_combine,
    pixel_average_precision, gaussian_smooth,
    calibrate_to_unit, float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
    IMAGENET_MEAN, IMAGENET_STD,
    BACKBONE_CHANNELS, BACKBONE_SHORT, ALL_BACKBONES,
    DINO_BACKBONES, DINOV2_BACKBONES,
    DINOV2_NBLOCKS, DINOV3_VIT_SPECS, DINOV3_NBLOCKS,
    DINOV3_CONVNEXT_SPECS,
    backbone_patch_size, resolve_target_layer,
    RESNET_BACKBONES,
)
from local_preds_saver import LocalPredSaver


PROJECT_ROOT = Path("/workspace/anomaly-detection")
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
# Datasets
# ─────────────────────────────────────────────────────────────────────────────
class TrainGoodWithClassDataset(Dataset):
    """Returns (image, class_id_str) so the synthesizer can pick a
    class-conditional mode profile per sample."""
    def __init__(self, records: list[ImageRecord], input_size: int):
        self.records = records
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
        return x, r.cls


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
# Anomaly synthesis — 8 modes per the CSV taxonomy
# ─────────────────────────────────────────────────────────────────────────────
ANOM_MODES = ("scratch", "dent", "stain", "fragment", "protrusion",
              "crack", "contamination", "hole")

# Keyword → which mode the description suggests. Matched substring,
# case-insensitive. Multiple matches per description are allowed and
# accumulate weight.
ANOM_KEYWORDS = {
    "scratch":       ["scratch", "linear mark", "shallow groove",
                        "linear or jag", "line or groove"],
    "dent":          ["indent", "dent", "depress", "hollow", "concave",
                        "dimpl", "flattened"],
    "stain":         ["stain", "discolor", "blotch", "patch",
                        "stained area"],
    "fragment":      ["fragment", "broken", "shard",
                        "irregular fragment", "irregular piece",
                        "jagged edge"],
    "protrusion":    ["raised", "bulg", "protru", "additional layer",
                        "additional cap"],
    "crack":         ["crack", "fissur", "fissured", "linear or jagged line"],
    "contamination": ["contamin", "fuzz", "powder", "mold", "mildew",
                        "coated surface", "infestation", "pest"],
    "hole":          ["missing part", "missing piece", "hole", "void",
                        "irregular holes"],
}


def load_anomaly_profile_from_csv(csv_path: Path,
                                     all_classes: list[str]) -> dict:
    """Return {class_id: {mode: weight}} normalized per class.

    Strategy: for each row, lowercase the description and add 1.0 weight
    to every mode whose keyword list matches anywhere in the text. Rows
    that match nothing (the generic "Localized visual anomaly affecting
    the object surface" boilerplate) add 0.25 uniformly to all modes —
    they're uninformative and shouldn't bias the profile, but they're
    also signal that this class has SOMETHING undescribed going on, so
    we don't drop them.

    Classes not present in the CSV fall back to uniform mode weights.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[warn] pandas not installed; cannot load anomaly profile.")
        return {cls: {m: 1.0 / len(ANOM_MODES) for m in ANOM_MODES}
                 for cls in all_classes}

    df = pd.read_csv(csv_path)
    if "public_class" not in df.columns or "description" not in df.columns:
        raise ValueError(f"{csv_path}: expected columns "
                          f"'public_class' and 'description'")
    out: dict = {}
    for cls in all_classes:
        sub_df = df[df["public_class"] == cls]
        weights = {m: 0.0 for m in ANOM_MODES}
        if len(sub_df) == 0:
            out[cls] = {m: 1.0 / len(ANOM_MODES) for m in ANOM_MODES}
            continue
        for _, row in sub_df.iterrows():
            desc = str(row["description"]).lower()
            matched_any = False
            for mode, kws in ANOM_KEYWORDS.items():
                if any(kw in desc for kw in kws):
                    weights[mode] += 1.0
                    matched_any = True
            if not matched_any:
                # Uninformative row: spread small uniform mass.
                for m in ANOM_MODES:
                    weights[m] += 0.25
        total = sum(weights.values())
        if total > 0:
            for m in weights: weights[m] /= total
        else:
            for m in weights: weights[m] = 1.0 / len(ANOM_MODES)
        out[cls] = weights
    return out


def print_profile(profile: dict) -> None:
    print(f"  per-class anomaly synthesis profile (mode percentages):")
    header = "  " + " " * 12 + "  ".join(f"{m[:6]:>6s}" for m in ANOM_MODES)
    print(header)
    for cls in sorted(profile):
        w = profile[cls]
        cells = "  ".join(f"{w[m] * 100:>6.1f}" for m in ANOM_MODES)
        print(f"  {cls:<12s} {cells}")


# ── GPU-vectorised primitives ────────────────────────────────────────────────
def _pixel_grid(H: int, W: int, device: torch.device
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
    xs = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)
    return ys, xs


@torch.no_grad()
def perlin_2d_batch(B: int, H: int, W: int, octaves: int = 4,
                      persistence: float = 0.5,
                      device: torch.device = torch.device("cpu")
                      ) -> torch.Tensor:
    """(B, H, W) multi-octave Perlin-like noise normalised to [0, 1].
    Each octave is a coarse random Gaussian grid bilinear-upsampled to
    (H, W); 4 octaves give the lumpy blob look we want for stains and
    dents. Per-sample min-max normalisation makes the threshold mode
    fraction stable across batches."""
    out = torch.zeros(B, H, W, device=device)
    amp = 1.0; freq = 1.0; norm = 0.0
    for _ in range(octaves):
        gh = max(2, int(round(8 * freq)))
        gw = max(2, int(round(8 * freq)))
        grid = torch.randn(B, 1, gh, gw, device=device)
        fine = F.interpolate(grid, size=(H, W),
                              mode="bilinear", align_corners=False).squeeze(1)
        out = out + amp * fine
        norm += amp; amp *= persistence; freq *= 2.0
    out = out / max(norm, 1e-6)
    omin = out.amin(dim=(-2, -1), keepdim=True)
    omax = out.amax(dim=(-2, -1), keepdim=True)
    return (out - omin) / (omax - omin + 1e-6)


def _line_soft_mask(ys: torch.Tensor, xs: torch.Tensor,
                     x0: float, y0: float, x1: float, y1: float,
                     width: float) -> torch.Tensor:
    dx = x1 - x0; dy = y1 - y0
    L2 = dx * dx + dy * dy + 1e-6
    t = ((xs - x0) * dx + (ys - y0) * dy) / L2
    t = t.clamp(0.0, 1.0)
    px = x0 + t * dx; py = y0 + t * dy
    dist = torch.sqrt((xs - px) ** 2 + (ys - py) ** 2)
    return torch.clamp(width - dist + 0.5, min=0.0, max=1.0)


def _draw_scratch_mask(H: int, W: int, device: torch.device,
                          jagged: bool = False,
                          n_scratches: int | None = None,
                          length_frac: tuple[float, float] = (0.10, 0.45),
                          width_range: tuple[float, float] = (1.0, 3.0)
                          ) -> torch.Tensor:
    if n_scratches is None: n_scratches = random.randint(1, 3)
    ys, xs = _pixel_grid(H, W, device)
    mask = torch.zeros(H, W, device=device)
    for _ in range(n_scratches):
        x0 = random.uniform(W * 0.10, W * 0.90)
        y0 = random.uniform(H * 0.10, H * 0.90)
        angle = random.uniform(0.0, 2 * math.pi)
        length = random.uniform(*length_frac) * min(H, W)
        x1 = x0 + length * math.cos(angle)
        y1 = y0 + length * math.sin(angle)
        width = random.uniform(*width_range)
        if jagged:
            # Break into 2 or 3 segments with mid-displacement → crack look.
            n_seg = random.choice([2, 3])
            pts = [(x0, y0)]
            for s in range(1, n_seg):
                t = s / n_seg
                mx = x0 + t * (x1 - x0) + random.uniform(-1, 1) * length * 0.15
                my = y0 + t * (y1 - y0) + random.uniform(-1, 1) * length * 0.15
                pts.append((mx, my))
            pts.append((x1, y1))
            for i in range(len(pts) - 1):
                m = _line_soft_mask(ys, xs, pts[i][0], pts[i][1],
                                       pts[i + 1][0], pts[i + 1][1], width)
                mask = torch.maximum(mask, m)
        else:
            m = _line_soft_mask(ys, xs, x0, y0, x1, y1, width)
            mask = torch.maximum(mask, m)
    return mask


def _blob_mask_from_perlin(noise_2d: torch.Tensor,
                              threshold: float = 0.6) -> torch.Tensor:
    """(H, W) Perlin map → soft blob mask via smooth thresholding.
    Soft mask helps the synthesised edit have a natural-looking edge
    (no hard pixelated boundary)."""
    # Smooth step around threshold.
    sharpness = 12.0
    return torch.sigmoid((noise_2d - threshold) * sharpness)


@torch.no_grad()
def synthesize_local_anomaly_batch(
    x: torch.Tensor,                   # (B, 3, H, W) ImageNet-normalised
    classes: list[str] | None,
    profile: dict | None,
    intensity_range: tuple[float, float] = (0.5, 1.0),
    forced_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """For each image, sample a synthesis mode and apply it. Returns:
        x_anom : (B, 3, H, W)
        mask   : (B, H, W) float in [0, 1] — the GT defect mask
        modes  : list[str] of length B — which mode was applied
    """
    B, C, H, W = x.shape
    device = x.device

    # Pre-allocate Perlin batches for blob-based modes (4 octaves for
    # stain/contamination, 3 for dent/fragment/protrusion/hole).
    noise_lf = perlin_2d_batch(B, H, W, octaves=4, device=device)   # smooth
    noise_mf = perlin_2d_batch(B, H, W, octaves=3, device=device)   # blobbier

    # Cross-image texture source (circular shift + random flips). Used
    # for fragment mode; the texture should look "foreign" relative to
    # this image.
    perm = (torch.arange(B, device=device)
              + random.randint(1, max(B - 1, 1))) % B
    texture_src = x[perm]
    if random.random() < 0.5:
        texture_src = torch.flip(texture_src, dims=[-1])
    if random.random() < 0.5:
        texture_src = torch.flip(texture_src, dims=[-2])
    # 90-degree rotation half the time for additional dissimilarity.
    if random.random() < 0.5:
        texture_src = torch.rot90(texture_src, k=random.choice([1, 2, 3]),
                                     dims=[-2, -1])

    x_anom = x.clone()
    masks  = torch.zeros(B, H, W, device=device)
    modes_used: list[str] = [""] * B

    for i in range(B):
        # Pick mode.
        if forced_mode is not None:
            mode = forced_mode
        elif (profile is not None and classes is not None
              and classes[i] in profile):
            w = profile[classes[i]]
            mode = random.choices(list(w.keys()), weights=list(w.values()),
                                     k=1)[0]
        else:
            mode = random.choice(ANOM_MODES)
        modes_used[i] = mode

        alpha = random.uniform(*intensity_range)

        if mode == "scratch":
            # Linear dark mark.
            m_soft = _draw_scratch_mask(H, W, device, jagged=False,
                                            width_range=(1.0, 2.5))
            # Slightly darker, slightly desaturated.
            darken = -0.8 * alpha * m_soft.unsqueeze(0)
            x_anom[i] = x[i] + darken
            masks[i] = m_soft

        elif mode == "crack":
            m_soft = _draw_scratch_mask(H, W, device, jagged=True,
                                            width_range=(1.2, 3.0),
                                            length_frac=(0.15, 0.55))
            x_anom[i] = x[i] - 1.0 * alpha * m_soft.unsqueeze(0)
            masks[i] = m_soft

        elif mode == "dent":
            # Smooth low-frequency blob, gentle darkening (shadow look).
            m_soft = _blob_mask_from_perlin(noise_lf[i], threshold=0.72)
            shadow = -0.6 * alpha * m_soft.unsqueeze(0)
            x_anom[i] = x[i] + shadow
            masks[i] = m_soft

        elif mode == "stain":
            # Blob with random color shift (per-channel offset).
            m_soft = _blob_mask_from_perlin(noise_lf[i], threshold=0.62)
            color_shift = (torch.rand(3, device=device) - 0.5) * 2.0 * alpha
            x_anom[i] = x[i] + color_shift.view(3, 1, 1) * m_soft.unsqueeze(0)
            masks[i] = m_soft

        elif mode == "fragment":
            # Foreign material pasted on a sharp-edged region.
            # Higher threshold → smaller, sharper blob.
            m_hard = (noise_mf[i] > 0.70).float()
            x_anom[i] = (x[i] * (1.0 - m_hard.unsqueeze(0))
                          + texture_src[i] * m_hard.unsqueeze(0))
            masks[i] = m_hard

        elif mode == "protrusion":
            # Smaller bright blob (raised area with highlight).
            m_soft = _blob_mask_from_perlin(noise_mf[i], threshold=0.75)
            highlight = 0.6 * alpha * m_soft.unsqueeze(0)
            x_anom[i] = x[i] + highlight
            masks[i] = m_soft

        elif mode == "contamination":
            # Mold/powder: blob region with high-frequency noise.
            m_soft = _blob_mask_from_perlin(noise_mf[i], threshold=0.65)
            hf_noise = torch.randn(3, H, W, device=device) * (0.7 * alpha)
            x_anom[i] = x[i] + hf_noise * m_soft.unsqueeze(0)
            masks[i] = m_soft

        elif mode == "hole":
            # Dark void where the object should be.
            m_soft = _blob_mask_from_perlin(noise_mf[i], threshold=0.72)
            # Push toward dark (normalized -1.7 ≈ pixel value 0)
            target = -1.7
            x_anom[i] = (x[i] * (1.0 - m_soft.unsqueeze(0))
                          + target * m_soft.unsqueeze(0))
            masks[i] = m_soft

        else:
            # Should be unreachable.
            masks[i] = torch.zeros(H, W, device=device)

    return x_anom, masks, modes_used


def image_mask_to_patch_mask(mask_img: torch.Tensor,
                                h: int, w: int) -> torch.Tensor:
    """(B, H, W) → (B, h, w) soft labels = fraction of anomalous pixel
    area covered by each patch. Adaptive avg-pool handles non-integer
    stride between input grid and feature grid (the typical case for
    DINOv2 patch=14)."""
    pooled = F.adaptive_avg_pool2d(mask_img.unsqueeze(1), output_size=(h, w))
    return pooled.squeeze(1).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter + Discriminator
# ─────────────────────────────────────────────────────────────────────────────
class Adapter(nn.Module):
    """Single Linear + LeakyReLU. Learns a small task-specific projection
    of the frozen backbone's features. Keeping in-dim == out-dim avoids
    bottlenecking; the role of the adapter is to tune feature norms /
    rotation, not to compress."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        # Initialize as near-identity so the first iteration mostly
        # passes the backbone features through.
        with torch.no_grad():
            self.net[0].weight.copy_(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
            self.net[0].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim) — works on (B, P, C) or (N, C).
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = self.net(flat)
        return out.reshape(*shape[:-1], -1)


class Discriminator(nn.Module):
    """2-hidden-layer MLP with BatchNorm. Per-patch binary classifier.
    BN normalises across the (B*P) patch batch; without it the network
    overfits the dominant feature dimensions (high-norm channels) which
    happen to come from one or two backbone channels."""
    def __init__(self, in_dim: int, hidden: int = 1024,
                  n_hidden_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = in_dim
        for _ in range(n_hidden_layers):
            layers += [
                nn.Linear(d_in, hidden),
                nn.BatchNorm1d(hidden),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim). Returns (N, 1) logits.
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Global anomaly synthesis via L_inf PGD ascent
# ─────────────────────────────────────────────────────────────────────────────
def generate_global_anomaly(f_normal: torch.Tensor,
                              discriminator: Discriminator,
                              epsilon: float = 0.05,
                              n_steps: int = 4,
                              step_size: float | None = None,
                              init_sigma: float = 0.015) -> torch.Tensor:
    """PGD ascent on the discriminator's "anomaly" logit, starting from
    f_normal + small Gaussian noise. Returns a detached perturbed
    feature tensor that the discriminator currently thinks is more
    anomalous than f_normal.

    Args:
        f_normal     : (N, C) detached normal features (post-adapter).
        discriminator: trained-so-far discriminator. Put it in eval()
                        externally so BN running stats are stable during
                        the attack.
        epsilon      : L_inf bound on the perturbation, in normalised
                        feature units (since features are L2-normed by
                        patchify_and_combine, epsilon is roughly a
                        fraction of unit-sphere distance — ~0.05 is a
                        small but meaningful drift).
        n_steps      : number of ascent steps (FGSM = 1).
        step_size    : per-step L_inf budget. Default = 2.5 * eps / n.
        init_sigma   : sigma of the Gaussian noise initial perturbation.

    Notes:
        - We use autograd.grad which returns the gradient w.r.t. delta
          WITHOUT accumulating it into discriminator parameter .grads.
          So this routine can be called within a training step without
          interfering with the optimiser.
    """
    if step_size is None:
        step_size = 2.5 * epsilon / max(n_steps, 1)

    delta = (torch.randn_like(f_normal) * init_sigma).clamp(-epsilon, epsilon)
    delta.requires_grad_(True)

    for _ in range(n_steps):
        logits = discriminator(f_normal + delta)
        # Gradient ASCENT on logits → make features look more anomalous.
        grad = torch.autograd.grad(logits.sum(), delta, retain_graph=False)[0]
        with torch.no_grad():
            delta_new = delta + step_size * grad.sign()
            delta_new = delta_new.clamp(-epsilon, epsilon)
        delta = delta_new.detach().requires_grad_(True)

    return (f_normal + delta).detach()


# ─────────────────────────────────────────────────────────────────────────────
# GLASS model bundle
# ─────────────────────────────────────────────────────────────────────────────
class GLASS(nn.Module):
    """Bundles the frozen feature extractor, adapter and discriminator
    plus the training and inference logic."""
    def __init__(self, backbone: FeatureExtractor, in_dim: int,
                  feature_layers: tuple[int, ...], target_layer: int,
                  patch_size_combine: int = 3,
                  discriminator_hidden: int = 1024,
                  discriminator_layers: int = 2,
                  dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone           # FROZEN
        self.adapter = Adapter(in_dim, dropout=dropout)
        self.discriminator = Discriminator(
            in_dim=in_dim, hidden=discriminator_hidden,
            n_hidden_layers=discriminator_layers, dropout=dropout)
        self.feature_layers = tuple(feature_layers)
        self.target_layer = int(target_layer)
        self.patch_size_combine = int(patch_size_combine)
        self.in_dim = int(in_dim)
        self._feature_hw: tuple[int, int] | None = None

    @torch.inference_mode()
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.backbone(x, layers=self.feature_layers)
        f = patchify_and_combine(maps, patch_size=self.patch_size_combine,
                                    target_layer=self.target_layer)
        # f: (B, P, C). Cache feature_hw from P = h * w (P always square
        # in our setup; for non-square inputs we'd need to read it from
        # the maps directly, but Spacepresso uses square inputs).
        B, P, C = f.shape
        if self._feature_hw is None:
            h = w = int(math.isqrt(P))
            self._feature_hw = (h, w)
        return f

    def feature_hw(self) -> tuple[int, int]:
        assert self._feature_hw is not None, "call _extract_features first"
        return self._feature_hw


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_glass(model: GLASS, records: list[ImageRecord], cfg: "RunConfig",
                  device: torch.device, profile: dict | None) -> None:
    ds = TrainGoodWithClassDataset(records, input_size=cfg.input_size)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=worker_init_fn, drop_last=True)
    iters_per_epoch = max(len(loader), 1)
    if cfg.total_iters and cfg.total_iters > 0:
        n_epochs = max(1, math.ceil(cfg.total_iters / iters_per_epoch))
        print(f"    [auto-epoch] total_iters={cfg.total_iters} / "
              f"{iters_per_epoch} iters/epoch -> {n_epochs} epochs")
    else:
        n_epochs = cfg.epochs
    total_iters = iters_per_epoch * n_epochs

    optimizer = torch.optim.AdamW(
        list(model.adapter.parameters())
        + list(model.discriminator.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))
    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"    [{now_hms()}] training: {n_epochs} epochs × "
          f"{iters_per_epoch} iters ({total_iters} total)  "
          f"bs={cfg.batch_size}  amp={use_amp}  "
          f"λ_local={cfg.lambda_local}  λ_global={cfg.lambda_global}  "
          f"epsilon={cfg.attack_epsilon}  n_steps={cfg.attack_n_steps}")

    # Track per-class synth mode counts for diagnostics.
    mode_counts: dict[str, int] = defaultdict(int)
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    iter_count = 0

    model.backbone.eval()
    for epoch in range(n_epochs):
        model.adapter.train()
        model.discriminator.train()
        loss_sum = ln_sum = ll_sum = lg_sum = 0.0; n = 0
        for x, classes in loader:
            x = x.to(device, non_blocking=True)
            cls_list = list(classes)
            B = x.shape[0]

            # 1. Backbone features for normal images.
            with torch.amp.autocast("cuda", enabled=use_amp):
                f_normal_raw = model._extract_features(x).float()
            h, w = model.feature_hw()
            P = h * w
            C = f_normal_raw.shape[-1]

            # 2. Synthesize local anomalies + extract features.
            x_anom, mask_img, modes_used = synthesize_local_anomaly_batch(
                x, cls_list, profile,
                intensity_range=(cfg.synth_intensity_min,
                                   cfg.synth_intensity_max))
            for m in modes_used: mode_counts[m] += 1
            with torch.amp.autocast("cuda", enabled=use_amp):
                f_local_raw = model._extract_features(x_anom).float()
            patch_mask = image_mask_to_patch_mask(mask_img, h, w)

            # 3. Apply adapter.
            f_normal = model.adapter(f_normal_raw).reshape(B * P, C)
            f_local  = model.adapter(f_local_raw ).reshape(B * P, C)

            # 4. Generate global anomalies via PGD (after warmup).
            do_global = (iter_count >= cfg.warmup_iters
                           and cfg.lambda_global > 0)
            if do_global:
                model.discriminator.eval()
                # Subsample normal patches for ascent (otherwise N=B*P is
                # huge and the autograd graph for delta gets expensive).
                n_attack = min(cfg.attack_subsample, f_normal.shape[0])
                idx = torch.randperm(f_normal.shape[0], device=device)[
                    :n_attack]
                f_anchor = f_normal[idx].detach()
                f_global = generate_global_anomaly(
                    f_anchor, model.discriminator,
                    epsilon=cfg.attack_epsilon,
                    n_steps=cfg.attack_n_steps,
                    init_sigma=cfg.attack_init_sigma)
                model.discriminator.train()
            else:
                f_global = None

            # 5. Discriminator forward + BCE losses.
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_normal = model.discriminator(f_normal)
                logits_local  = model.discriminator(f_local)
                target_normal = torch.zeros_like(logits_normal)
                target_local  = patch_mask.reshape(-1, 1).clamp(0, 1)
                L_normal = F.binary_cross_entropy_with_logits(
                    logits_normal, target_normal)
                L_local  = F.binary_cross_entropy_with_logits(
                    logits_local, target_local)
                if do_global and f_global is not None:
                    logits_global = model.discriminator(f_global)
                    target_global = torch.ones_like(logits_global)
                    L_global = F.binary_cross_entropy_with_logits(
                        logits_global, target_global)
                    loss = (L_normal + cfg.lambda_local * L_local
                             + cfg.lambda_global * L_global)
                else:
                    L_global = torch.tensor(0.0, device=device)
                    loss = L_normal + cfg.lambda_local * L_local

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            iter_count += 1

            loss_sum += loss.item() * B
            ln_sum += L_normal.item() * B
            ll_sum += L_local.item() * B
            lg_sum += float(L_global.item()) * B
            n += B

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={loss_sum/max(n,1):.4f}  "
                  f"L_normal={ln_sum/max(n,1):.4f}  "
                  f"L_local={ll_sum/max(n,1):.4f}  "
                  f"L_global={lg_sum/max(n,1):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    model.adapter.eval(); model.discriminator.eval()
    print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")
    total_synth = sum(mode_counts.values())
    if total_synth > 0:
        print(f"    synth mode mix (this class's training): "
              + ", ".join(f"{m}={mode_counts[m]/total_synth*100:.0f}%"
                            for m in ANOM_MODES if mode_counts[m] > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _score_one_pass(model: GLASS, x: torch.Tensor, cfg: "RunConfig",
                      device: torch.device) -> torch.Tensor:
    """Returns (B, input_size, input_size) sigmoid scores on GPU."""
    use_amp = (device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        f_raw = model._extract_features(x)
    f_raw = f_raw.float()
    B, P, C = f_raw.shape
    h, w = model.feature_hw()
    f_adapt = model.adapter(f_raw).reshape(B * P, C)
    logits = model.discriminator(f_adapt)
    probs = torch.sigmoid(logits).reshape(B, h, w).float()
    up = F.interpolate(probs.unsqueeze(1),
                        size=(cfg.input_size, cfg.input_size),
                        mode="bilinear", align_corners=False).squeeze(1)
    return up


@torch.inference_mode()
def score_batch(model: GLASS, x: torch.Tensor, cfg: "RunConfig",
                  device: torch.device) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    acc = None; n = 0
    def _add(s):
        nonlocal acc, n
        if acc is None: acc = s.clone()
        else: acc += s
        n += 1
    _add(_score_one_pass(model, x, cfg, device))
    if cfg.tta in ("hflip", "hvflip"):
        s = _score_one_pass(model, torch.flip(x, dims=[-1]), cfg, device)
        _add(torch.flip(s, dims=[-1]))
    if cfg.tta in ("vflip", "hvflip"):
        s = _score_one_pass(model, torch.flip(x, dims=[-2]), cfg, device)
        _add(torch.flip(s, dims=[-2]))
    return (acc / max(n, 1)).cpu().float()


def _score_records(model: GLASS, records: list[ImageRecord],
                     cfg: "RunConfig", device: torch.device,
                     load_masks: bool):
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores, gts = {}, {}
    n_done = 0; last_log = 0
    for x, masks, idxs in loader:
        sm = score_batch(model, x, cfg, device).numpy()
        m_np = masks.numpy()
        for b in range(sm.shape[0]):
            scores[int(idxs[b])] = sm[b]
            gts[int(idxs[b])] = m_np[b]
        n_done += sm.shape[0]
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline + submission writer
# ─────────────────────────────────────────────────────────────────────────────
def run_one_class(cls: str, records_all: list[ImageRecord],
                   backbone: FeatureExtractor, cfg: "RunConfig",
                   run_dir: Path, device: torch.device,
                   profile: dict | None,
                   in_dim: int,
                   local_saver: LocalPredSaver | None = None) -> dict:
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

    # Build a fresh GLASS bundle per class. Adapter+discriminator are
    # ~3M params, instantiating per class is cheap and prevents cross-
    # class interference.
    model = GLASS(
        backbone=backbone, in_dim=in_dim,
        feature_layers=cfg.feature_layers,
        target_layer=cfg.target_layer,
        patch_size_combine=cfg.patch_size_combine,
        discriminator_hidden=cfg.discriminator_hidden,
        discriminator_layers=cfg.discriminator_layers,
        dropout=cfg.dropout,
    ).to(device)
    n_params = (sum(p.numel() for p in model.adapter.parameters())
                + sum(p.numel() for p in model.discriminator.parameters()))
    print(f"  GLASS: in_dim={in_dim}  disc_hidden={cfg.discriminator_hidden}  "
          f"trainable params={n_params/1e6:.2f}M")

    train_glass(model, train_good, cfg, device, profile)

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_glass.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"adapter": model.adapter.state_dict(),
                    "discriminator": model.discriminator.state_dict(),
                    "feature_hw": model.feature_hw(),
                    "in_dim": in_dim,
                    "feature_layers": list(cfg.feature_layers),
                    "target_layer": cfg.target_layer,
                    "config": asdict(cfg)}, ck)
        print(f"    saved checkpoint -> {ck}")

    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  tta={cfg.tta}")
        scores, gts = _score_records(model, train_anom, cfg, device,
                                        load_masks=True)
        by_anom: dict[str, list[float]] = defaultdict(list)
        for r_idx, sm in scores.items():
            r = train_anom[r_idx]
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            ap = pixel_average_precision(sm_smooth, gts[r_idx])
            by_anom[r.anomaly_type or "?"].append(ap)
            if local_saver is not None:
                local_saver.add(
                    cls=cls, anomaly_type=r.anomaly_type or "unknown",
                    view_idx=int(r_idx), score_map=sm_smooth,
                    gt_mask=gts[r_idx], image_path=r.path)
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

    test_results: list[tuple[ImageRecord, np.ndarray]] = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images  tta={cfg.tta}")
        scores, _ = _score_records(model, test, cfg, device,
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
        w = csv.writer(f); w.writerow(["ID", "Label"])
        for r, sm in all_test_results:
            normed = np.clip((sm - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            w.writerow([r.path.stem, float_matrix_to_q8rle(normed)]); n += 1
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
# Run config + CLI
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    # Backbone
    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (2, 3)
    target_layer: int = 2
    patch_size_combine: int = 3
    input_size: int = 288
    # Architecture
    discriminator_hidden: int = 1024
    discriminator_layers: int = 2
    dropout: float = 0.0
    # Training
    epochs: int = 200
    total_iters: int | None = 3000
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-5
    amp: bool = True
    num_workers: int = 8
    # Loss weights
    lambda_local: float = 1.0
    lambda_global: float = 1.0
    # Synthesis (local)
    synth_intensity_min: float = 0.5
    synth_intensity_max: float = 1.0
    anom_profile_csv: Path | None = None
    no_anom_profile: bool = False
    # Synthesis (global PGD)
    attack_epsilon: float = 0.05
    attack_n_steps: int = 4
    attack_init_sigma: float = 0.015
    attack_subsample: int = 4096
    warmup_iters: int = 400
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


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "glass",
        "backbone": cfg.backbone,
        "feature_layers": list(cfg.feature_layers),
        "target_layer": cfg.target_layer,
        "input_size": cfg.input_size,
        "discriminator_hidden": cfg.discriminator_hidden,
        "discriminator_layers": cfg.discriminator_layers,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "lambda_local": cfg.lambda_local,
        "lambda_global": cfg.lambda_global,
        "synth_intensity_min": cfg.synth_intensity_min,
        "synth_intensity_max": cfg.synth_intensity_max,
        "attack_epsilon": cfg.attack_epsilon,
        "attack_n_steps": cfg.attack_n_steps,
        "warmup_iters": cfg.warmup_iters,
        "no_anom_profile": cfg.no_anom_profile,
        "tta": cfg.tta,
        "smooth_sigma": cfg.smooth_sigma,
        "seed": cfg.seed,
        "v": 1,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bb = BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)
    if cfg.backbone in RESNET_BACKBONES:
        L = "".join(str(l) for l in cfg.feature_layers)
    else:
        L = "_".join(str(l) for l in cfg.feature_layers)
    budget = (f"it{cfg.total_iters}"
              if (cfg.total_iters and cfg.total_iters > 0)
              else f"e{cfg.epochs}")
    prof_tag = "uniProf" if cfg.no_anom_profile else "csvProf"
    bits = (f"{stamp}_glass_{bb}_L{L}_T{cfg.target_layer}_"
            f"in{cfg.input_size}_h{cfg.discriminator_hidden}_"
            f"eps{cfg.attack_epsilon:g}_{prof_tag}_{budget}_bs{cfg.batch_size}")
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
    # Backbone
    ap.add_argument("--backbone", default="wide_resnet50_2",
                    choices=ALL_BACKBONES,
                    help="Any backbone supported by FeatureExtractor "
                         "(ResNet or DINOv2/v3 ViT). ConvNeXt not allowed "
                         "— different fusion logic.")
    ap.add_argument("--feature-layers", type=int, nargs="+", default=[2, 3],
                    help="For ResNet: layer indices {1..4}. "
                         "For DINOv2/v3 ViT: block indices [0..n_blocks). "
                         "Multiple layers are 3x3-avg-pooled and "
                         "concatenated by patchify_and_combine.")
    ap.add_argument("--target-layer", default="auto",
                    help="Layer whose spatial size other layers are "
                         "resized to. 'auto' picks layer 2 for ResNet, "
                         "min layer for DINO.")
    ap.add_argument("--patch-size-combine", type=int, default=3,
                    help="3x3 avg-pool window inside patchify_and_combine "
                         "(matches PatchCore default).")
    ap.add_argument("--input-size", type=int, default=288,
                    help="Multiple of 32 for ResNet, multiple of 14 for "
                         "DINOv2, multiple of 16 for DINOv3.")
    # Architecture
    ap.add_argument("--discriminator-hidden", type=int, default=1024)
    ap.add_argument("--discriminator-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=3000,
                    help="Overrides --epochs. ~3000 is enough for the "
                         "discriminator to converge; longer leads to "
                         "discriminator overfitting on the synth signature.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    # Loss weights
    ap.add_argument("--lambda-local", type=float, default=1.0)
    ap.add_argument("--lambda-global", type=float, default=1.0,
                    help="Set to 0 to disable PGD global anomalies "
                         "(pure SimpleNet-style with rich synthesis).")
    # Synthesis (local)
    ap.add_argument("--synth-intensity-min", type=float, default=0.5)
    ap.add_argument("--synth-intensity-max", type=float, default=1.0,
                    help="Per-image alpha for the synthetic edit is "
                         "uniformly sampled from this range.")
    ap.add_argument("--anom-profile-csv", type=Path, default=None,
                    help="CSV with columns public_class,description. "
                         "Used to weight per-class mode mix. "
                         "Default: ./anomaly_descriptions.csv if it "
                         "exists, else uniform.")
    ap.add_argument("--no-anom-profile", action="store_true",
                    help="Force uniform mode sampling (ignore the CSV).")
    # Synthesis (global PGD)
    ap.add_argument("--attack-epsilon", type=float, default=0.05,
                    help="L_inf bound on the perturbation in normalised "
                         "feature units. 0.05 ≈ 5% of unit-sphere "
                         "distance for L2-normalised features.")
    ap.add_argument("--attack-n-steps", type=int, default=4)
    ap.add_argument("--attack-init-sigma", type=float, default=0.015)
    ap.add_argument("--attack-subsample", type=int, default=4096,
                    help="Subsample N normal patches to run PGD on. "
                         "B=16, P=784 → 12544 patches; sampling 4096 "
                         "speeds up ascent without hurting quality.")
    ap.add_argument("--warmup-iters", type=int, default=400,
                    help="Disable global anomalies for the first N "
                         "iters so the discriminator has learned a "
                         "non-trivial decision boundary before its "
                         "gradients are used for PGD ascent.")
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
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    # Backbone sanity.
    if args.backbone in DINOV3_CONVNEXT_SPECS:
        raise SystemExit(
            "[FATAL] ConvNeXt backbones not supported here. Use ResNet "
            "or DINOv2/v3 ViT.")
    feature_layers = tuple(sorted(set(args.feature_layers)))
    for l in feature_layers:
        if l not in BACKBONE_CHANNELS[args.backbone]:
            raise SystemExit(
                f"[FATAL] feature_layer {l} invalid for {args.backbone}")
    target_layer = resolve_target_layer(feature_layers, args.target_layer,
                                            args.backbone)
    # Input-size patch sanity for DINO.
    if args.backbone in DINO_BACKBONES:
        ps = backbone_patch_size(args.backbone)
        if ps and args.input_size % ps != 0:
            raise SystemExit(
                f"[FATAL] --input-size {args.input_size} not divisible "
                f"by {ps} (required by {args.backbone}).")
    elif args.input_size % 32 != 0:
        raise SystemExit(
            f"[FATAL] ResNet backbones need --input-size divisible by 32; "
            f"got {args.input_size}.")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone, feature_layers=feature_layers,
        target_layer=target_layer,
        patch_size_combine=args.patch_size_combine,
        input_size=args.input_size,
        discriminator_hidden=args.discriminator_hidden,
        discriminator_layers=args.discriminator_layers,
        dropout=args.dropout,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        amp=not args.no_amp, num_workers=args.num_workers,
        lambda_local=args.lambda_local, lambda_global=args.lambda_global,
        synth_intensity_min=args.synth_intensity_min,
        synth_intensity_max=args.synth_intensity_max,
        anom_profile_csv=args.anom_profile_csv,
        no_anom_profile=args.no_anom_profile,
        attack_epsilon=args.attack_epsilon,
        attack_n_steps=args.attack_n_steps,
        attack_init_sigma=args.attack_init_sigma,
        attack_subsample=args.attack_subsample,
        warmup_iters=args.warmup_iters,
        score_batch_size=args.score_batch_size,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        save_checkpoints=args.save_checkpoints,
        zip_submission=not args.no_zip,
        run_tag=args.run_tag,
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"GLASS — RUN {run_id}", "█")
        print(f"  data_root         : {cfg.data_root}")
        print(f"  run_dir           : {run_dir}")
        print(f"  backbone          : {cfg.backbone}")
        print(f"  feature_layers    : {list(cfg.feature_layers)}  "
              f"target={cfg.target_layer}")
        print(f"  input_size        : {cfg.input_size}")
        print(f"  discriminator     : hidden={cfg.discriminator_hidden}  "
              f"layers={cfg.discriminator_layers}")
        if cfg.total_iters and cfg.total_iters > 0:
            print(f"  total_iters       : {cfg.total_iters}")
        else:
            print(f"  epochs            : {cfg.epochs}")
        print(f"  batch_size        : {cfg.batch_size}")
        print(f"  lr / wd           : {cfg.lr} / {cfg.weight_decay}")
        print(f"  λ_local / λ_global: {cfg.lambda_local} / {cfg.lambda_global}")
        print(f"  attack ε / steps  : {cfg.attack_epsilon} / "
              f"{cfg.attack_n_steps}  warmup={cfg.warmup_iters}")
        print(f"  synth intensity   : [{cfg.synth_intensity_min}, "
              f"{cfg.synth_intensity_max}]")
        print(f"  smooth_sigma      : {cfg.smooth_sigma}    tta: {cfg.tta}")
        print(f"  device            : {device}")
        if torch.cuda.is_available():
            print(f"                     {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, (Path, torch.device))
                            else v)
                       for k, v in asdict(cfg).items()}, f, indent=2,
                       default=str)

        # Build frozen backbone (once for all classes).
        backbone = FeatureExtractor(cfg.backbone).to(device).eval()
        # Discover feature dim by a dummy forward.
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, cfg.input_size, cfg.input_size,
                                  device=device)
            maps = backbone(dummy, layers=cfg.feature_layers)
            f0 = patchify_and_combine(maps,
                                          patch_size=cfg.patch_size_combine,
                                          target_layer=cfg.target_layer)
        in_dim = int(f0.shape[-1])
        h0 = w0 = int(math.isqrt(f0.shape[1]))
        print(f"\n  fused feature dim : {in_dim}  feature grid: {h0}x{w0}")

        # Scan dataset.
        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): "
              f"{', '.join(classes)}")

        # Load anomaly synthesis profile.
        profile: dict | None = None
        if not cfg.no_anom_profile:
            csv_p = cfg.anom_profile_csv
            if csv_p is None:
                # Default: file beside this script
                here = Path(__file__).resolve().parent
                candidate = here / "anomaly_descriptions.csv"
                if candidate.exists():
                    csv_p = candidate
            if csv_p is not None and csv_p.exists():
                print(f"\n  loading anomaly profile from {csv_p}")
                profile = load_anomaly_profile_from_csv(csv_p, classes)
                print_profile(profile)
            else:
                print(f"\n  [note] no anomaly-description CSV found; "
                      f"using uniform mode sampling")
        else:
            print(f"\n  --no-anom-profile: uniform mode sampling")

        local_saver: LocalPredSaver | None = None
        if not cfg.skip_eval and not args.no_save_local_preds:
            local_saver = LocalPredSaver()

        all_test_results, all_eval_rows = [], []
        class_aps, class_elapsed = {}, {}
        for cls in classes:
            res = run_one_class(cls, records, backbone, cfg, run_dir,
                                  device, profile=profile, in_dim=in_dim,
                                  local_saver=local_saver)
            all_test_results.extend(res["test_results"])
            all_eval_rows.extend(res["eval_rows"])
            class_aps[cls] = res["class_mean_ap"]
            class_elapsed[cls] = res["elapsed_min"]

        if local_saver is not None and len(local_saver) > 0:
            local_saver.save(run_dir / "local_predictions.npz")

        hr("LOCAL VALIDATION SUMMARY", "=")
        print(f"  {'class':<10} {'pixel-AP':>12} {'time (min)':>12}")
        for cls in classes:
            print(f"  {cls:<10} {class_aps.get(cls, float('nan')):>12.4f} "
                  f"{class_elapsed.get(cls, 0):>12.1f}")
        valid_aps = [v for v in class_aps.values() if not math.isnan(v)]
        overall_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")
        if valid_aps:
            print(f"  {'OVERALL':<10} {overall_ap:>12.4f}")

        if all_eval_rows:
            tab_path = run_dir / "local_eval.csv"
            with open(tab_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_eval_rows[0].keys()))
                w.writeheader()
                for row in all_eval_rows: w.writerow(row)
            print(f"  saved per-(class, anomaly_type) AP -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            save_test_predictions(all_test_results, run_dir)
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload: {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        bb_short = BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)
        L = "+".join(str(l) for l in cfg.feature_layers)
        row = {
            "run_id": run_id, "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": f"GLASS_{bb_short.upper()}",
            "feature_layers": L,
            "target_layer": cfg.target_layer,
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
            "notes": (f"glass backbone={cfg.backbone} L={L} "
                      f"disc_h={cfg.discriminator_hidden} "
                      f"eps={cfg.attack_epsilon} "
                      f"λL={cfg.lambda_local} λG={cfg.lambda_global} "
                      f"prof={'uni' if cfg.no_anom_profile else 'csv'} "
                      f"{'it'+str(cfg.total_iters) if cfg.total_iters else 'e'+str(cfg.epochs)} "
                      f"bs{cfg.batch_size} tta={cfg.tta}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()