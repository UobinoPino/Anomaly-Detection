"""Spacepresso EfficientAD baseline.

Adds an independent track to the stacker with a fundamentally different
anomaly-detection mechanism:

  - PatchCore variants (exp2-7, 8c/d): nearest-neighbour to coreset
  - CutPaste (exp8c/d):                self-supervised classifier features
  - Reverse Distillation (exp10):      WRN50 teacher, OCBE+decoder student
  - --- EfficientAD (this file) ---:   PDN student + autoencoder,
        hard-mined teacher distillation + reconstruction. The AE branch
        catches "logical" anomalies the local student misses.

# Architecture

  TEACHER (frozen, shared across classes):
    ResNet-18 (ImageNet). Output of layer2 (128 ch @ H/8), bilinearly
    upsampled to H/4. Different backbone from RD4AD's WRN50 -> diversity
    for the stacker.

  STUDENT (PDN-S style, per class):
    4-block CNN reading the raw normalised image. Output: 256 channels
    at H/4. First 128 = match teacher, last 128 = match AE.

  AUTOENCODER (per class):
    Encoder (4 stride-2 convs) -> H/16. Decoder (2 upsample+conv) -> H/4.
    Output: 128 channels at H/4 resolution.

# Loss (trained only on train_good)

    L_st  = mean of top-q% of per-pixel ||s_t - teacher||^2     (hard mining)
    L_ae  = ||AE(x) - teacher||^2                                (AE -> teacher)
    L_stae = ||s_a - AE(x).detach()||^2                          (s_a -> AE)
    total = L_st + L_ae + L_stae

  Hard mining (default q=10%) forces the student to focus on the patches
  where it's currently worst -- crucial when defects are small and dense
  background regions dominate the loss.

# Normalization

  Post-training, the two branches' raw maps have different magnitudes.
  We compute mean and std of map_st and map_ae over a sample of
  train_good, then at test time:
      map_st_norm = (map_st - st_mean) / st_std
      map_ae_norm = (map_ae - ae_mean) / ae_std
      combined    = (map_st_norm + map_ae_norm) / 2

# Multi-view inference (--multiview sibling-bank)

  For each test sample, all 5 views are batched and scored together.
  For each view V_i:
    1. Standard combined score (above)
    2. Sibling bank: teacher features of the other 4 views, flattened
    3. For each pixel in V_i: 1 - max cosine sim to sibling bank
       -> "view-inconsistency score" in roughly [0, 2]
    4. Normalised to [0, 1] across all 5 views of the sample
    5. Boost: final = combined * ((1 - alpha) + alpha * mv_inc_norm)

  Rationale: a real defect appears in 1-2 views. Its features differ
  from any patch in the other 4 views -> mv_inc is high. Spurious
  bright regions (lighting reflections, complex but consistent texture)
  recur across views -> mv_inc is low -> downweighted.

  alpha=0.5 by default. Set higher (0.7) if the standard scoring is
  noisy on textured classes (coffee, pistachio). Lower (0.3) if the
  standard scoring is already strong (uniform backgrounds).

# Memory / speed (L4 24 GB)

  Per class @ input 256, batch 16, 2500 iters:
    Training:        ~4 min
    Standard score:  ~30 s + 30 s eval
    Multi-view:      ~50 s + 50 s eval
  Full 8 classes:    ~50 min standard, ~70 min multi-view.

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
from torchvision.models import resnet18, ResNet18_Weights

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

TEACHER_CHANNELS = 128  # ResNet-18 layer2 width


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
    """Standard per-image inference dataset (no sample-id grouping)."""
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
# Networks
# ─────────────────────────────────────────────────────────────────────────────
class TeacherNet(nn.Module):
    """Frozen ResNet-18: layer2 features bilinearly upsampled to H/4.

    Distinct from RD4AD's WRN50 teacher (different backbone, single layer
    only). Smaller, faster, and pushes the stacker toward a different
    feature subspace.
    """
    out_channels = TEACHER_CHANNELS

    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1   # H/4, 64 ch
        self.layer2 = m.layer2   # H/8, 128 ch
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        x = self.stem(x)        # H/4
        x = self.layer1(x)      # H/4
        x = self.layer2(x)      # H/8, 128 ch
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        return x                # (B, 128, H/4, W/4)


def _conv_block(c_in, c_out, k=3, stride=1, pad=None):
    if pad is None: pad = k // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class StudentPDN(nn.Module):
    """PDN-S style small CNN. Reads raw image, outputs 2*C channels
    at H/4 resolution. First C channels are trained to match the teacher;
    last C channels are trained to match the autoencoder."""

    def __init__(self, out_channels: int = 2 * TEACHER_CHANNELS):
        super().__init__()
        self.block1 = nn.Sequential(
            _conv_block(3,    64),
            nn.AvgPool2d(2, stride=2),       # H/2
        )
        self.block2 = nn.Sequential(
            _conv_block(64,  128),
            nn.AvgPool2d(2, stride=2),       # H/4
        )
        self.block3 = nn.Sequential(
            _conv_block(128, 256),
            _conv_block(256, 256),
        )
        self.head = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)                  # (B, 2C, H/4, W/4)


class Autoencoder(nn.Module):
    """Encoder H -> H/16, decoder H/16 -> H/4. Output at H/4 matches
    teacher channels so the student-AE loss is well defined."""

    def __init__(self, out_channels: int = TEACHER_CHANNELS,
                 base: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            # H -> H/2
            nn.Conv2d(3, base, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            # H/2 -> H/4
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            # H/4 -> H/8
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            # H/8 -> H/16
            nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 8), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            # H/16 -> H/8
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_block(base * 8, base * 4),
            # H/8 -> H/4
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_block(base * 4, base * 2),
        )
        self.head = nn.Conv2d(base * 2, out_channels, 1)

    def forward(self, x):
        z = self.enc(x)
        h = self.dec(z)
        return self.head(h)                  # (B, C, H/4, W/4)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_efficientad(teacher: TeacherNet,
                       student: StudentPDN,
                       ae: Autoencoder,
                       records: list[ImageRecord],
                       cfg: "RunConfig", device: torch.device) -> None:
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

    params = list(student.parameters()) + list(ae.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))
    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    C_T = teacher.out_channels
    print(f"    [{now_hms()}] training: {n_epochs} epochs x "
          f"{iters_per_epoch} iters ({total_iters} total)  "
          f"bs={cfg.batch_size}  amp={use_amp}  "
          f"hard_mining_pct={cfg.hard_mining_pct}")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()

    teacher.eval()
    for epoch in range(n_epochs):
        student.train(); ae.train()
        loss_sum = ls_sum = la_sum = lst_sum = 0.0
        n = 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    t_feat = teacher(x)         # (B, C, H/4, W/4)
                s_out = student(x)              # (B, 2C, H/4, W/4)
                ae_out = ae(x)                  # (B, C, H/4, W/4)
                s_t = s_out[:, :C_T]
                s_a = s_out[:, C_T:]

                # Hard-mined student-teacher loss
                diff_st = (s_t - t_feat) ** 2   # (B, C, H, W)
                diff_st_pix = diff_st.mean(dim=1)  # (B, H, W)
                k = max(int(cfg.hard_mining_pct * diff_st_pix.numel()), 1)
                topk = torch.topk(diff_st_pix.reshape(-1), k,
                                    largest=True).values
                L_st = topk.mean()

                # AE-teacher
                L_ae = ((ae_out - t_feat) ** 2).mean()
                # Student-AE branch (AE detached so this only trains s_a)
                L_stae = ((s_a - ae_out.detach()) ** 2).mean()

                loss = L_st + L_ae + L_stae

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            B = x.shape[0]
            loss_sum += loss.item() * B
            ls_sum   += L_st.item() * B
            la_sum   += L_ae.item() * B
            lst_sum  += L_stae.item() * B
            n += B

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={loss_sum/max(n,1):.4f}  "
                  f"L_st={ls_sum/max(n,1):.4f}  "
                  f"L_ae={la_sum/max(n,1):.4f}  "
                  f"L_stae={lst_sum/max(n,1):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    student.eval(); ae.eval()
    print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Map normalisation stats from a sample of train_good
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def compute_norm_stats(teacher, student, ae,
                        records: list[ImageRecord],
                        cfg: "RunConfig", device,
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
    C_T = teacher.out_channels
    use_amp = (device.type == "cuda" and cfg.amp)
    st_vals: list[np.ndarray] = []
    ae_vals: list[np.ndarray] = []
    teacher.eval(); student.eval(); ae.eval()
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            t_feat = teacher(x)
            s_out = student(x)
            ae_out = ae(x)
        s_t = s_out[:, :C_T].float()
        s_a = s_out[:, C_T:].float()
        t_feat = t_feat.float()
        ae_out_f = ae_out.float()
        map_st = ((s_t - t_feat) ** 2).mean(dim=1)
        map_ae = ((s_a - ae_out_f) ** 2).mean(dim=1)
        st_vals.append(map_st.cpu().numpy().reshape(-1))
        ae_vals.append(map_ae.cpu().numpy().reshape(-1))
    st_arr = np.concatenate(st_vals)
    ae_arr = np.concatenate(ae_vals)
    stats = {
        "st_mean": float(st_arr.mean()),
        "st_std":  float(st_arr.std() + 1e-9),
        "ae_mean": float(ae_arr.mean()),
        "ae_std":  float(ae_arr.std() + 1e-9),
    }
    print(f"    norm stats over {len(records)} train_good: "
          f"st={stats['st_mean']:.4f}±{stats['st_std']:.4f}  "
          f"ae={stats['ae_mean']:.4f}±{stats['ae_std']:.4f}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Inference primitives
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _compute_maps_one(teacher, student, ae, x: torch.Tensor, stats: dict,
                       cfg: "RunConfig") -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (combined_score_lr, t_feat) for one forward pass.
    combined_score_lr: (B, H/4, W/4)
    t_feat:            (B, C, H/4, W/4)  [needed for multi-view]"""
    use_amp = (cfg.device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        t_feat = teacher(x)
        s_out = student(x)
        ae_out = ae(x)
    C_T = teacher.out_channels
    s_t = s_out[:, :C_T].float()
    s_a = s_out[:, C_T:].float()
    t_feat_f = t_feat.float()
    ae_out_f = ae_out.float()
    map_st = ((s_t - t_feat_f) ** 2).mean(dim=1)
    map_ae = ((s_a - ae_out_f) ** 2).mean(dim=1)
    map_st_n = (map_st - stats["st_mean"]) / stats["st_std"]
    map_ae_n = (map_ae - stats["ae_mean"]) / stats["ae_std"]
    combined = (map_st_n + map_ae_n) * 0.5
    return combined, t_feat_f


@torch.inference_mode()
def _maps_with_tta(teacher, student, ae, x: torch.Tensor, stats: dict,
                    cfg: "RunConfig", tta: str
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """TTA-averaged (combined_score_lr, t_feat). Flips are inverted before
    averaging so spatial alignment is preserved."""
    accum_s = None; accum_t = None; n_acc = 0

    def _add(s, t):
        nonlocal accum_s, accum_t, n_acc
        if accum_s is None:
            accum_s = s.clone(); accum_t = t.clone()
        else:
            accum_s += s; accum_t += t
        n_acc += 1

    s, t = _compute_maps_one(teacher, student, ae, x, stats, cfg)
    _add(s, t)
    if tta in ("hflip", "hvflip"):
        s2, t2 = _compute_maps_one(teacher, student, ae,
                                     torch.flip(x, dims=[-1]), stats, cfg)
        _add(torch.flip(s2, dims=[-1]), torch.flip(t2, dims=[-1]))
    if tta in ("vflip", "hvflip"):
        s2, t2 = _compute_maps_one(teacher, student, ae,
                                     torch.flip(x, dims=[-2]), stats, cfg)
        _add(torch.flip(s2, dims=[-2]), torch.flip(t2, dims=[-2]))
    return accum_s / n_acc, accum_t / n_acc


@torch.inference_mode()
def score_batch_standard(teacher, student, ae, x: torch.Tensor,
                          stats: dict, cfg: "RunConfig") -> torch.Tensor:
    """Per-image scoring with optional TTA. Returns (B, input_size,
    input_size) on CPU."""
    x = x.to(cfg.device, non_blocking=True)
    combined, _ = _maps_with_tta(teacher, student, ae, x, stats, cfg, cfg.tta)
    up = F.interpolate(combined.unsqueeze(1),
                       size=(cfg.input_size, cfg.input_size),
                       mode="bilinear", align_corners=False).squeeze(1)
    return up.cpu()


@torch.inference_mode()
def score_sample_with_siblings(teacher, student, ae,
                                 x_sample: torch.Tensor,
                                 stats: dict,
                                 cfg: "RunConfig") -> torch.Tensor:
    """All views of one sample batched together. For each view, the
    OTHER views' teacher features form a small memory bank; pixel-level
    inconsistency with that bank boosts standard scores.

    x_sample: (V, 3, H, W). Returns (V, input_size, input_size) on CPU."""
    x_sample = x_sample.to(cfg.device, non_blocking=True)
    combined, t_feat = _maps_with_tta(teacher, student, ae, x_sample,
                                        stats, cfg, cfg.tta)
    V, C, H_, W_ = t_feat.shape

    if V < 2:
        # No siblings; standard upsample.
        up = F.interpolate(combined.unsqueeze(1),
                           size=(cfg.input_size, cfg.input_size),
                           mode="bilinear", align_corners=False).squeeze(1)
        return up.cpu()

    # Flatten patch features per view and L2-normalise so the matmul
    # below is cosine similarity.
    t_flat = t_feat.permute(0, 2, 3, 1).reshape(V, -1, C)
    t_norm = F.normalize(t_flat, p=2, dim=-1)            # (V, P, C)
    P = t_norm.shape[1]

    mv_dist = torch.empty(V, P, device=t_feat.device, dtype=torch.float32)
    for i in range(V):
        siblings = torch.cat(
            [t_norm[j] for j in range(V) if j != i], dim=0)  # ((V-1)*P, C)
        # Cosine sim of view-i patches against all sibling patches.
        sim = t_norm[i].float() @ siblings.float().T          # (P, (V-1)P)
        max_sim = sim.max(dim=1).values                       # (P,)
        mv_dist[i] = 1.0 - max_sim

    mv_dist = mv_dist.reshape(V, H_, W_)
    sd_min = mv_dist.min(); sd_max = mv_dist.max()
    if (sd_max - sd_min) > 1e-9:
        mv_norm = (mv_dist - sd_min) / (sd_max - sd_min)
    else:
        mv_norm = torch.zeros_like(mv_dist)

    alpha = cfg.mv_alpha
    boost = (1.0 - alpha) + alpha * mv_norm                   # (V, H/4, W/4)
    boosted = combined * boost
    up = F.interpolate(boosted.unsqueeze(1),
                       size=(cfg.input_size, cfg.input_size),
                       mode="bilinear", align_corners=False).squeeze(1)
    return up.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _build_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_one(r: ImageRecord, transform, load_masks: bool, input_size: int
              ) -> tuple[torch.Tensor, np.ndarray]:
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


def _score_records_standard(teacher, student, ae,
                             records: list[ImageRecord],
                             stats: dict, cfg: "RunConfig",
                             load_masks: bool
                             ) -> tuple[dict, dict]:
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores: dict[int, np.ndarray] = {}
    gts: dict[int, np.ndarray] = {}
    n_done = 0; last_log = 0
    for x, masks, idxs in loader:
        sm = score_batch_standard(teacher, student, ae, x, stats, cfg).numpy()
        m_np = masks.numpy()
        for b in range(sm.shape[0]):
            scores[int(idxs[b])] = sm[b]
            gts[int(idxs[b])] = m_np[b]
        n_done += sm.shape[0]
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


def _score_records_by_sample(teacher, student, ae,
                              records: list[ImageRecord],
                              stats: dict, cfg: "RunConfig",
                              load_masks: bool
                              ) -> tuple[dict, dict]:
    """Group by sample_id, score all views of each sample together so the
    sibling-bank can be built per sample."""
    by_sample: dict[str, list[tuple[int, ImageRecord]]] = defaultdict(list)
    for idx, r in enumerate(records):
        sid = r.sample_id or r.path.stem
        by_sample[sid].append((idx, r))
    print(f"      grouped {len(records)} images into {len(by_sample)} samples")

    transform = _build_transform(cfg.input_size)
    scores: dict[int, np.ndarray] = {}
    gts: dict[int, np.ndarray] = {}
    n_done = 0; last_log = 0
    for sid, items in by_sample.items():
        imgs: list[torch.Tensor] = []
        masks_np: list[np.ndarray] = []
        for _idx, r in items:
            x, m = _load_one(r, transform, load_masks, cfg.input_size)
            imgs.append(x); masks_np.append(m)
        x_batch = torch.stack(imgs)
        sm = score_sample_with_siblings(
            teacher, student, ae, x_batch, stats, cfg).numpy()
        for k, (idx, _r) in enumerate(items):
            scores[idx] = sm[k]
            gts[idx] = masks_np[k]
        n_done += len(items)
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


def run_one_class(cls: str, records_all: list[ImageRecord],
                   teacher: TeacherNet, cfg: "RunConfig",
                   run_dir: Path, device: torch.device,
                   local_saver: "LocalPredSaver | None" = None) -> dict:
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

    # Fresh student + AE per class.
    student = StudentPDN(out_channels=2 * teacher.out_channels).to(device)
    ae = Autoencoder(out_channels=teacher.out_channels).to(device)
    train_efficientad(teacher, student, ae, train_good, cfg, device)

    print(f"    [{now_hms()}] computing normalisation stats...")
    stats = compute_norm_stats(teacher, student, ae, train_good, cfg, device)

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_efficientad.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"student": student.state_dict(),
                    "ae": ae.state_dict(),
                    "stats": stats}, ck)
        print(f"    saved checkpoint -> {ck}")

    score_fn = (_score_records_by_sample
                if cfg.multiview == "sibling-bank"
                else _score_records_standard)

    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  multiview={cfg.multiview}  "
            f"alpha={cfg.mv_alpha}  tta={cfg.tta}")
        scores, gts = score_fn(teacher, student, ae, train_anom, stats,
                                 cfg, load_masks=True)
        by_anom: dict[str, list[float]] = defaultdict(list)
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
        per_type_means: list[float] = []
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
        sub(f"scoring {len(test)} test images  "
            f"multiview={cfg.multiview}  tta={cfg.tta}")
        scores, _ = score_fn(teacher, student, ae, test, stats, cfg,
                              load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            sm_final = maybe_resize_to_submission(sm_smooth)
            test_results.append((test[r_idx], sm_final))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del student, ae
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"class": cls, "class_mean_ap": class_mean_ap,
            "eval_rows": eval_rows, "test_results": test_results,
            "elapsed_min": elapsed_min}


# ─────────────────────────────────────────────────────────────────────────────
# Submission writer
# ─────────────────────────────────────────────────────────────────────────────
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
    # Training
    epochs: int = 200
    total_iters: int | None = 2500
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    hard_mining_pct: float = 0.10
    amp: bool = True
    num_workers: int = 8
    # Inference
    score_batch_size: int = 32
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    multiview: str = "none"      # "none" | "sibling-bank"
    mv_alpha: float = 0.5
    # Bookkeeping
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_checkpoints: bool = False
    zip_submission: bool = True
    run_tag: str = ""
    # Runtime
    device: torch.device | None = None


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "efficientad",
        "input_size": cfg.input_size,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "hard_mining_pct": cfg.hard_mining_pct,
        "multiview": cfg.multiview,
        "mv_alpha": cfg.mv_alpha,
        "tta": cfg.tta,
        "smooth_sigma": cfg.smooth_sigma,
        "seed": cfg.seed,
        "v": 1,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    budget = (f"it{cfg.total_iters}"
              if (cfg.total_iters and cfg.total_iters > 0)
              else f"e{cfg.epochs}")
    mv_tag = "noMV" if cfg.multiview == "none" else f"MV-a{cfg.mv_alpha:.2f}"
    bits = (f"{stamp}_effad_rn18_in{cfg.input_size}_{budget}"
            f"_bs{cfg.batch_size}_{mv_tag}")
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
    ap.add_argument("--input-size", type=int, default=256)
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=2500,
                    help="Overrides --epochs (matches the budget used by "
                         "CutPaste / RD).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hard-mining-pct", type=float, default=0.10,
                    help="Top fraction of per-pixel student-teacher losses "
                         "kept for L_st. 0.10 = top 10%%.")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    # Inference
    ap.add_argument("--score-batch-size", type=int, default=32)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip"])
    ap.add_argument("--multiview", default="none",
                    choices=["none", "sibling-bank"],
                    help="`sibling-bank`: at test time, batch all 5 views "
                         "of a sample and use the OTHER 4 views' teacher "
                         "features as a small memory bank. Boost each "
                         "view's pixel scores by their view-inconsistency.")
    ap.add_argument("--mv-alpha", type=float, default=0.5,
                    help="Multi-view blend weight in [0, 1]. 0 disables "
                         "the boost; 1 fully replaces the standard score "
                         "with the inconsistency signal.")
    # Bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true",
                    help="Disable local_predictions.npz output.")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        input_size=args.input_size,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        hard_mining_pct=args.hard_mining_pct,
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
        hr(f"EFFICIENTAD — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  teacher          : frozen ResNet-18, layer2 @ H/4")
        print(f"  input_size       : {cfg.input_size}")
        if cfg.total_iters and cfg.total_iters > 0:
            print(f"  total_iters      : {cfg.total_iters}  "
                  f"(overrides --epochs={cfg.epochs})")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / wd          : {cfg.lr} / {cfg.weight_decay}")
        print(f"  hard_mining_pct  : {cfg.hard_mining_pct}")
        print(f"  amp              : {cfg.amp}    num_workers: {cfg.num_workers}")
        print(f"  score_batch_size : {cfg.score_batch_size}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  multiview        : {cfg.multiview}  alpha={cfg.mv_alpha}")
        print(f"  save_local_preds : {not args.no_save_local_preds}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            cfg_dump = {k: (str(v) if isinstance(v, (Path, torch.device))
                              else v)
                          for k, v in asdict(cfg).items()}
            json.dump(cfg_dump, f, indent=2, default=str)

        teacher = TeacherNet().to(device).eval()

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

        all_test_results: list[tuple[ImageRecord, np.ndarray]] = []
        all_eval_rows: list[dict] = []
        class_aps: dict[str, float] = {}
        class_elapsed: dict[str, float] = {}
        for cls in classes:
            res = run_one_class(cls, records, teacher, cfg, run_dir, device,
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
            "backbone": "EFFICIENTAD_RN18",
            "feature_layers": "rn18_layer2",
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
            "notes": (f"efficientad multiview={cfg.multiview} "
                      f"mv_alpha={cfg.mv_alpha} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size} hm{cfg.hard_mining_pct}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()