"""Spacepresso CutPaste — self-supervised pixel-level anomaly detection.

Adds a third independent track alongside PatchCore (WRN50) and PatchCore
(DINOv2). Designed to be fused with them at step 9 of the roadmap.

# What it does

  - Backbone: ImageNet-pretrained ResNet-18 (or resnet50).
  - Training: per-class 3-way self-supervised classification
        class 0: normal image (train/good)
        class 1: CutPaste — random rectangle copy-paste
        class 2: CutPaste-Scar — long, thin, rotated scar
  - Loss: cross-entropy.
  - After training: PaDiM-style per-position Gaussian over the learned
    backbone's multi-scale features. Random dimension selection
    (100 of 448) keeps covariance inversion tractable.
  - Scoring: Mahalanobis distance per position → bilinear upsample →
    Gaussian smooth → q8rle (matches PatchCore's submission format).

# v2 speed knobs (vs initial release)

The first release used --epochs 256 (CutPaste paper default). That recipe
assumes MVTec-AD-sized train/good (~250 imgs per class). On Spacepresso,
train/good is 10-50x larger (2000-2600 imgs), so 256 epochs = ~20k
iterations per class — wildly more than the paper's ~2500 iter budget,
and training accuracy plateaus by epoch 32-64 anyway.

v2 adds:

  --total-iters INT     Overrides --epochs to give approximately this
                        many total optimisation steps. Auto-scales the
                        epoch count per class so the iteration budget
                        stays constant across classes with varying
                        dataset sizes. Set to 2500 to match the paper's
                        effective compute budget.

  Defaults bumped:
      --batch-size       32 -> 64    (effective 192; fits on L4 + AMP)
      --num-workers       2 -> 8     (PIL CutPaste/Scar is CPU-bound)

Combined: ~8-10x speedup, ~40-60 min total walltime for 8 classes.

# CutPaste augmentations (Li et al., CVPR 2021)

  CutPaste: rectangle with area ∈ [2%, 15%] of image area, aspect ratio
  ∈ [0.3, 3.3]. Copied from a random source location, optional color
  jitter applied to the patch, pasted at a random target location.

  CutPaste-Scar: long thin patch, width ∈ [10, 25] px, height ∈ [2, 16]
  px. Color jitter, random rotation ∈ [-45°, 45°], pasted at a random
  target location.

# PaDiM scoring (Defard et al. 2021)

  For each class, after training:
    1. Extract features from layers 1+2+3 on train/good, upsampled to
       the layer-2 spatial grid and concatenated (448 channels for RN18).
    2. Random-select 100 of the 448 channel indices (per-class, seeded).
    3. Per spatial position (h, w): fit Gaussian μ_{hw}, Σ_{hw} + ε·I.
    4. Score(test; h, w) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)).
    5. Bilinear upsample to input_size, Gaussian smooth, resize to 224,
       q8rle.

# Output

  Identical layout to PatchCore so score_fusion.py works as-is:
        baseline_out/runs/<run_id>/
            submission.csv, submission.zip
            local_eval.csv, config.json, run_log.txt
  + one row appended to baseline_out/ablation_master.csv.
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
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/work/u10813429/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
VIEW_RE = re.compile(r"^(?P<base>.+?)_view(?P<v>\d+)\.[A-Za-z]+$")

LAYER_DIMS = {
    "resnet18": {1: 64,  2: 128, 3: 256,  4: 512},
    "resnet50": {1: 256, 2: 512, 3: 1024, 4: 2048},
}
BACKBONE_SHORT = {"resnet18": "rn18", "resnet50": "rn50"}

SUBMISSION_H = SUBMISSION_W = 224


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
# Dataset scanning (mirrors PatchCore v5)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ImageRecord:
    path: Path
    cls: str
    split: str
    anomaly_type: str | None = None
    sample_id: str | None = None
    view: int | None = None
    mask_path: Path | None = None


def parse_view(filename: str) -> tuple[str, int | None]:
    m = VIEW_RE.match(filename)
    if m:
        return m.group("base"), int(m.group("v"))
    return Path(filename).stem, None


def scan_dataset(data_root: Path) -> list[ImageRecord]:
    out: list[ImageRecord] = []
    if not data_root.exists():
        print(f"  [FATAL] {data_root} not found"); return out
    classes = sorted(d.name for d in data_root.iterdir()
                     if d.is_dir() and d.name.startswith("class_"))
    for cls in classes:
        cdir = data_root / cls
        gd = cdir / "train" / "good"
        if gd.exists():
            for p in sorted(gd.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    out.append(ImageRecord(p, cls, "train_good",
                                           sample_id=sid, view=v))
        td = cdir / "train"
        if td.exists():
            for sd in sorted(td.iterdir()):
                if (not sd.is_dir() or sd.name == "good"
                        or not sd.name.startswith("anomaly_")):
                    continue
                a_type = sd.name
                gtd = cdir / "ground_truth_train" / a_type
                for p in sorted(sd.iterdir()):
                    if p.suffix.lower() not in IMG_EXTS: continue
                    sid, v = parse_view(p.name)
                    mp = None
                    if gtd.exists():
                        cand = gtd / p.name
                        if cand.exists():
                            mp = cand
                        else:
                            for q in gtd.iterdir():
                                if (q.stem == p.stem
                                        and q.suffix.lower() in IMG_EXTS):
                                    mp = q; break
                    out.append(ImageRecord(p, cls, "train_anomaly",
                                           anomaly_type=a_type,
                                           sample_id=sid, view=v,
                                           mask_path=mp))
        ted = cdir / "test"
        if ted.exists():
            for p in sorted(ted.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    out.append(ImageRecord(p, cls, "test",
                                           sample_id=sid, view=v))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CutPaste augmentations (operate on PIL.Image)
# ─────────────────────────────────────────────────────────────────────────────
def _apply_color_jitter_pil(patch: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return patch
    cj = transforms.ColorJitter(brightness=strength, contrast=strength,
                                 saturation=strength, hue=min(strength, 0.5))
    return cj(patch)


def apply_cutpaste(img: Image.Image,
                    area_ratio: tuple[float, float] = (0.02, 0.15),
                    aspect_ratio: tuple[float, float] = (0.3, 3.3),
                    color_jitter: float = 0.1,
                    rng: random.Random | None = None) -> Image.Image:
    r = rng or random
    W, H = img.size
    total_area = W * H

    patch_w, patch_h = 0, 0
    for _ in range(10):
        target_area = r.uniform(*area_ratio) * total_area
        log_lo, log_hi = math.log(aspect_ratio[0]), math.log(aspect_ratio[1])
        aspect = math.exp(r.uniform(log_lo, log_hi))
        patch_w = max(1, int(round(math.sqrt(target_area * aspect))))
        patch_h = max(1, int(round(math.sqrt(target_area / aspect))))
        if patch_w < W and patch_h < H:
            break
    else:
        return img.copy()

    src_x = r.randint(0, W - patch_w)
    src_y = r.randint(0, H - patch_h)
    patch = img.crop((src_x, src_y, src_x + patch_w, src_y + patch_h))
    patch = _apply_color_jitter_pil(patch, color_jitter)

    tgt_x = r.randint(0, W - patch_w)
    tgt_y = r.randint(0, H - patch_h)
    out = img.copy()
    out.paste(patch, (tgt_x, tgt_y))
    return out


def apply_cutpaste_scar(img: Image.Image,
                         width: tuple[int, int] = (10, 25),
                         height: tuple[int, int] = (2, 16),
                         rotation_deg: float = 45.0,
                         color_jitter: float = 0.1,
                         rng: random.Random | None = None) -> Image.Image:
    r = rng or random
    W, H = img.size
    pw = r.randint(*width)
    ph = r.randint(*height)
    if pw >= W or ph >= H:
        return img.copy()

    src_x = r.randint(0, W - pw)
    src_y = r.randint(0, H - ph)
    patch = img.crop((src_x, src_y, src_x + pw, src_y + ph))
    patch = _apply_color_jitter_pil(patch, color_jitter)

    angle = r.uniform(-rotation_deg, rotation_deg)
    if patch.mode != "RGBA":
        patch = patch.convert("RGBA")
    patch = patch.rotate(angle, resample=Image.BILINEAR, expand=True)

    rw, rh = patch.size
    if rw >= W or rh >= H:
        return img.copy()
    tgt_x = r.randint(0, W - rw)
    tgt_y = r.randint(0, H - rh)
    out = img.copy().convert("RGB")
    out.paste(patch, (tgt_x, tgt_y), mask=patch.split()[-1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────
class CutPasteTrainDataset(Dataset):
    """Yields (normal, cutpaste, scar) triplets per sample."""

    def __init__(self, records: list[ImageRecord], input_size: int,
                 area_ratio=(0.02, 0.15), aspect_ratio=(0.3, 3.3),
                 color_jitter=0.1, scar_width=(10, 25),
                 scar_height=(2, 16), scar_rot=45.0):
        self.records = records
        self.input_size = input_size
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.color_jitter = color_jitter
        self.scar_width = scar_width
        self.scar_height = scar_height
        self.scar_rot = scar_rot
        self.resize = transforms.Resize((input_size, input_size))
        self.to_tensor = transforms.ToTensor()
        self.normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.records)

    def _pil_to_tensor(self, im: Image.Image) -> torch.Tensor:
        return self.normalise(self.to_tensor(im))

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r.path) as im:
            im = im.convert("RGB")
            base = self.resize(im).copy()
        cp = apply_cutpaste(base, area_ratio=self.area_ratio,
                              aspect_ratio=self.aspect_ratio,
                              color_jitter=self.color_jitter)
        sc = apply_cutpaste_scar(base, width=self.scar_width,
                                   height=self.scar_height,
                                   rotation_deg=self.scar_rot,
                                   color_jitter=self.color_jitter)
        return self._pil_to_tensor(base), self._pil_to_tensor(cp), self._pil_to_tensor(sc)


class InferenceDataset(Dataset):
    """Standard image+mask dataset for feature extraction / scoring."""

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

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r.path) as im:
            im = im.convert("RGB")
            x = self.tx(im)
        if self.load_masks and r.mask_path is not None:
            with Image.open(r.mask_path) as mm:
                mm = mm.convert("L")
                mm = mm.resize((self.input_size, self.input_size),
                               Image.NEAREST)
                m = (np.asarray(mm) > 127).astype(np.float32)
        else:
            m = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        return x, torch.from_numpy(m), i


def worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % 2 ** 32
    np.random.seed(base)
    random.seed(base)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class CutPasteNet(nn.Module):
    """ResNet trunk + small MLP head for 3-way classification."""

    def __init__(self, backbone: str = "resnet18", num_classes: int = 3):
        super().__init__()
        if backbone == "resnet18":
            self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == "resnet50":
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.backbone_name = backbone
        self.feat_dim = feat_dim
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)

    def extract_features(self, x: torch.Tensor,
                          layers: tuple[int, ...] = (1, 2, 3),
                          ) -> dict[int, torch.Tensor]:
        e = self.encoder
        x = e.conv1(x); x = e.bn1(x); x = e.relu(x); x = e.maxpool(x)
        out: dict[int, torch.Tensor] = {}
        x = e.layer1(x)
        if 1 in layers: out[1] = x
        x = e.layer2(x)
        if 2 in layers: out[2] = x
        x = e.layer3(x)
        if 3 in layers: out[3] = x
        if 4 in layers:
            x = e.layer4(x)
            out[4] = x
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_cutpaste(model: CutPasteNet, records: list[ImageRecord],
                    cfg: "RunConfig", device: torch.device) -> None:
    """Fine-tune the model in-place with 3-way CutPaste classification."""
    ds = CutPasteTrainDataset(
        records, input_size=cfg.input_size,
        area_ratio=tuple(cfg.area_ratio),
        aspect_ratio=tuple(cfg.aspect_ratio),
        color_jitter=cfg.color_jitter,
        scar_width=tuple(cfg.scar_width),
        scar_height=tuple(cfg.scar_height),
        scar_rot=cfg.scar_rot,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0),
                         worker_init_fn=worker_init_fn,
                         drop_last=False)

    # Determine effective epochs. --total-iters overrides --epochs to
    # keep the optimisation budget constant across classes with varying
    # dataset sizes (Spacepresso train_good ranges 2135-2640 vs the
    # paper's ~250 per class). Default total_iters=2500 matches the
    # paper's 256 epochs × ~10 iters ≈ 2500 iter budget.
    iters_per_epoch = len(loader)
    if cfg.total_iters is not None and cfg.total_iters > 0:
        n_epochs = max(1, math.ceil(cfg.total_iters / max(iters_per_epoch, 1)))
        print(f"    [auto-epoch] total_iters={cfg.total_iters} / "
              f"{iters_per_epoch} iters/epoch = {n_epochs} epochs "
              f"(--epochs={cfg.epochs} overridden)")
    else:
        n_epochs = cfg.epochs

    optim = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    n_iters_total = max(iters_per_epoch * n_epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=n_iters_total)
    criterion = nn.CrossEntropyLoss()

    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"    [{now_hms()}] training: {n_epochs} epochs, "
          f"{iters_per_epoch} iters/epoch "
          f"({iters_per_epoch * n_epochs} total), "
          f"{len(records)} train_good images, "
          f"batch={cfg.batch_size} (effective {3 * cfg.batch_size}), "
          f"amp={use_amp}")

    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0.0; correct = 0; total = 0
        for normal, cp, sc in loader:
            B = normal.shape[0]
            x = torch.cat([normal, cp, sc], dim=0).to(device, non_blocking=True)
            y = torch.cat([
                torch.zeros(B, dtype=torch.long),
                torch.ones(B,  dtype=torch.long),
                torch.full((B,), 2, dtype=torch.long),
            ]).to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * x.shape[0]
            correct  += (logits.argmax(dim=-1) == y).sum().item()
            total    += x.shape[0]

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch + 1:>4}/{n_epochs}  "
                  f"loss={loss_sum / max(total, 1):.4f}  "
                  f"acc={correct / max(total, 1):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    model.eval()
    print(f"    [{now_hms()}] training done ({time.time() - t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# PaDiM fitting and scoring
# ─────────────────────────────────────────────────────────────────────────────
def _multi_scale_concat(maps: dict[int, torch.Tensor],
                         target_layer: int) -> torch.Tensor:
    H, W = maps[target_layer].shape[-2:]
    parts = []
    for k in sorted(maps.keys()):
        m = maps[k]
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode="bilinear",
                              align_corners=False)
        parts.append(m)
    return torch.cat(parts, dim=1)


def fit_padim(model: CutPasteNet, records: list[ImageRecord],
              cfg: "RunConfig", device: torch.device,
              feature_layers: tuple[int, ...]) -> dict:
    full_dim = sum(LAYER_DIMS[model.backbone_name][l] for l in feature_layers)
    target_layer = cfg.target_layer

    proj_dim = min(cfg.projection_dim, full_dim)
    rng = np.random.default_rng(cfg.seed)
    dim_idx = np.sort(rng.choice(full_dim, size=proj_dim, replace=False))
    dim_idx = torch.from_numpy(dim_idx).long().to(device)

    ds = InferenceDataset(records, input_size=cfg.input_size,
                           load_masks=False)
    loader = DataLoader(ds, batch_size=cfg.fit_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))

    feats_buf: list[torch.Tensor] = []
    H = W = None
    print(f"    [{now_hms()}] extracting train/good features for PaDiM "
          f"({len(records)} images, target_layer={target_layer}, "
          f"projection {full_dim}->{proj_dim})...")
    model.eval()
    with torch.inference_mode():
        for x, _, _ in loader:
            x = x.to(device, non_blocking=True)
            maps = model.extract_features(x, layers=feature_layers)
            f = _multi_scale_concat(maps, target_layer=target_layer)
            B, C, h, w = f.shape
            H, W = h, w
            f = f.permute(0, 2, 3, 1).reshape(B, h * w, C)
            f = f.index_select(dim=2, index=dim_idx)
            feats_buf.append(f.detach().to(torch.float32).cpu())
    feats_all = torch.cat(feats_buf, dim=0)
    N = feats_all.shape[0]

    feats_gpu = feats_all.to(device)
    feats_gpu = feats_gpu.permute(1, 0, 2).contiguous()
    mean = feats_gpu.mean(dim=1)
    centered = feats_gpu - mean.unsqueeze(1)
    cov = torch.einsum("pni,pnj->pij", centered, centered) / max(N - 1, 1)
    eye = torch.eye(proj_dim, device=device, dtype=cov.dtype).unsqueeze(0)
    cov = cov + cfg.padim_eps * eye
    inv_cov = torch.linalg.inv(cov)

    print(f"    [{now_hms()}] PaDiM fit done  N={N}  H×W={H}×{W}  D={proj_dim}  "
          f"GPU={inv_cov.element_size() * inv_cov.numel() / 1e6:.1f} MB")
    del feats_gpu, centered, cov, feats_buf, feats_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "mean": mean,
        "inv_cov": inv_cov,
        "dim_idx": dim_idx,
        "H": H, "W": W,
        "target_layer": target_layer,
        "feature_layers": feature_layers,
        "projection_dim": proj_dim,
    }


@torch.inference_mode()
def _padim_score_one(model: CutPasteNet, x: torch.Tensor,
                      padim: dict, input_size: int) -> torch.Tensor:
    maps = model.extract_features(x, layers=padim["feature_layers"])
    f = _multi_scale_concat(maps, target_layer=padim["target_layer"])
    B, C, H, W = f.shape
    f = f.permute(0, 2, 3, 1).reshape(B, H * W, C)
    f = f.index_select(dim=2, index=padim["dim_idx"])
    delta = f - padim["mean"].unsqueeze(0)
    tmp = torch.einsum("bpi,pij->bpj", delta, padim["inv_cov"])
    m_dist_sq = (tmp * delta).sum(dim=-1)
    m_dist = m_dist_sq.clamp_min(0).sqrt()
    score_lr = m_dist.reshape(B, H, W)
    score = F.interpolate(score_lr.unsqueeze(1),
                          size=(input_size, input_size),
                          mode="bilinear", align_corners=False).squeeze(1)
    return score.cpu()


@torch.inference_mode()
def padim_score_batch(model: CutPasteNet, x: torch.Tensor, padim: dict,
                       input_size: int, tta: str = "none",
                       device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    accumulator: torch.Tensor | None = None
    n_added = 0

    def _add(score_cpu: torch.Tensor) -> None:
        nonlocal accumulator, n_added
        if accumulator is None:
            accumulator = score_cpu.clone()
        else:
            accumulator += score_cpu
        n_added += 1

    _add(_padim_score_one(model, x, padim, input_size))
    if tta in ("hflip", "hvflip", "d4"):
        s = _padim_score_one(model, torch.flip(x, dims=[-1]), padim, input_size)
        _add(torch.flip(s, dims=[-1]))
    if tta in ("vflip", "hvflip", "d4"):
        s = _padim_score_one(model, torch.flip(x, dims=[-2]), padim, input_size)
        _add(torch.flip(s, dims=[-2]))
    if tta == "d4":
        for k in (1, 2, 3):
            s = _padim_score_one(model, torch.rot90(x, k=k, dims=[-2, -1]),
                                  padim, input_size)
            _add(torch.rot90(s, k=-k, dims=[-2, -1]))
    return accumulator / max(n_added, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + smoothing + q8rle (identical to PatchCore for fusion compatibility)
# ─────────────────────────────────────────────────────────────────────────────
def pixel_average_precision(score: np.ndarray, gt: np.ndarray) -> float:
    s = score.astype(np.float32).ravel()
    y = gt.astype(np.int32).ravel()
    if y.sum() == 0:
        return 0.0
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y, s))
    except Exception:
        order = np.argsort(-s, kind="stable")
        y = y[order]
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / max(int(y.sum()), 1)
        recall = np.concatenate([[0.0], recall])
        precision = np.concatenate([[1.0], precision])
        return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def gaussian_smooth(score: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    if sigma <= 0:
        return score
    r = max(1, int(round(3 * sigma)))
    k = _gaussian_kernel_1d(sigma, r)
    sx = np.pad(score, ((r, r), (0, 0)), mode="reflect")
    sx = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, sx)
    sx = np.pad(sx, ((0, 0), (r, r)), mode="reflect")
    sx = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, sx)
    return sx


def calibrate_to_unit(scores: list[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate([s.ravel() for s in scores])
    lo = float(np.percentile(flat, 1.0))
    hi = float(np.percentile(flat, 99.5))
    if hi <= lo: hi = lo + 1e-6
    return lo, hi


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255),
                0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)
    if flat.size == 0:
        return f"q8rle {h} {w}"
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]
    parts = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)


def maybe_resize_to_submission(score: np.ndarray) -> np.ndarray:
    if score.shape == (SUBMISSION_H, SUBMISSION_W):
        return score
    t = torch.from_numpy(score).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(SUBMISSION_H, SUBMISSION_W),
                      mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tracking
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    backbone: str = "resnet18"
    input_size: int = 256
    feature_layers: tuple[int, ...] = (1, 2, 3)
    target_layer: int = 2
    # Training
    epochs: int = 256
    total_iters: int | None = None     # NEW v2: overrides epochs
    batch_size: int = 64               # v2: was 32
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 3e-5
    amp: bool = True
    num_workers: int = 8               # v2: was 2
    # Augmentation
    area_ratio: tuple[float, float] = (0.02, 0.15)
    aspect_ratio: tuple[float, float] = (0.3, 3.3)
    color_jitter: float = 0.1
    scar_width: tuple[int, int] = (10, 25)
    scar_height: tuple[int, int] = (2, 16)
    scar_rot: float = 45.0
    # PaDiM
    projection_dim: int = 100
    padim_eps: float = 0.01
    fit_batch_size: int = 32
    # Scoring
    score_batch_size: int = 16
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    # Bookkeeping
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_checkpoints: bool = False
    save_padim_banks: bool = False
    zip_submission: bool = True
    run_tag: str = ""


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "cutpaste",
        "backbone": cfg.backbone,
        "input_size": cfg.input_size,
        "feature_layers": list(cfg.feature_layers),
        "target_layer": cfg.target_layer,
        "epochs": cfg.epochs,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "projection_dim": cfg.projection_dim,
        "padim_eps": cfg.padim_eps,
        "smooth_sigma": cfg.smooth_sigma,
        "tta": cfg.tta,
        "seed": cfg.seed,
        "v": 2,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bb = BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)
    if cfg.total_iters is not None and cfg.total_iters > 0:
        budget = f"it{cfg.total_iters}"
    else:
        budget = f"e{cfg.epochs}"
    bits = (f"{stamp}_cutpaste_{bb}_in{cfg.input_size}"
            f"_{budget}_bs{cfg.batch_size}_pd{cfg.projection_dim}")
    if cfg.tta != "none":
        bits += f"_tta-{cfg.tta}"
    if cfg.run_tag:
        bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


def append_to_ablation_master(master_csv: Path, row: dict) -> None:
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    fieldnames: list[str] = []
    if master_csv.exists():
        with open(master_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
    for k in row.keys():
        if k not in fieldnames:
            fieldnames.append(k)
    existing_rows.append(row)
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_one_class(cls: str, records_all: list[ImageRecord],
                   cfg: RunConfig, run_dir: Path,
                   device: torch.device) -> dict:
    hr(f"CLASS {cls}", "─")
    t_start = time.time()

    train_good = [r for r in records_all if r.cls == cls and r.split == "train_good"]
    train_anom = [r for r in records_all if r.cls == cls and r.split == "train_anomaly"]
    test       = [r for r in records_all if r.cls == cls and r.split == "test"]
    print(f"  train_good={len(train_good)}  "
          f"train_anomaly={len(train_anom)}  test={len(test)}")

    if not train_good:
        print(f"  [skip] no train_good for {cls}")
        return {"class": cls, "class_mean_ap": float("nan"),
                "eval_rows": [], "test_results": [], "elapsed_min": 0.0}

    model = CutPasteNet(backbone=cfg.backbone, num_classes=3).to(device)
    train_cutpaste(model, train_good, cfg, device)

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_model.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ck)
        print(f"    saved checkpoint -> {ck}")

    padim = fit_padim(model, train_good, cfg, device,
                       feature_layers=cfg.feature_layers)
    if cfg.save_padim_banks:
        bp = run_dir / "banks" / f"{cls}_padim.pt"
        bp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() if torch.is_tensor(v) else v
                    for k, v in padim.items()}, bp)
        print(f"    saved PaDiM bank -> {bp}")

    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation — per-anomaly-type pixel-AP "
            f"(tta={cfg.tta}, score_bs={cfg.score_batch_size})")
        ds_v = InferenceDataset(train_anom, input_size=cfg.input_size,
                                  load_masks=True)
        loader_v = DataLoader(ds_v, batch_size=cfg.score_batch_size,
                               shuffle=False, num_workers=cfg.num_workers,
                               pin_memory=True,
                               persistent_workers=(cfg.num_workers > 0))
        scores_by_idx: dict[int, np.ndarray] = {}
        gt_by_idx: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for x, masks, idxs in loader_v:
                sm = padim_score_batch(model, x, padim,
                                          input_size=cfg.input_size,
                                          tta=cfg.tta, device=device)
                sm = sm.numpy()
                m_np = masks.numpy()
                for b in range(sm.shape[0]):
                    s = gaussian_smooth(sm[b], cfg.smooth_sigma)
                    scores_by_idx[int(idxs[b])] = s
                    gt_by_idx[int(idxs[b])] = m_np[b]
        by_anom: dict[str, list[float]] = defaultdict(list)
        for ridx, s in scores_by_idx.items():
            r = train_anom[ridx]
            ap = pixel_average_precision(s, gt_by_idx[ridx])
            by_anom[r.anomaly_type or "?"].append(ap)
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
        class_mean_ap = float(np.mean(per_type_means)) if per_type_means else 0.0
        print(f"    >>> class {cls} mean pixel-AP "
              f"(avg over types): {class_mean_ap:.4f}")

    test_results: list[tuple[ImageRecord, np.ndarray]] = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images "
            f"(tta={cfg.tta}, score_bs={cfg.score_batch_size})")
        ds_t = InferenceDataset(test, input_size=cfg.input_size,
                                  load_masks=False)
        loader_t = DataLoader(ds_t, batch_size=cfg.score_batch_size,
                               shuffle=False, num_workers=cfg.num_workers,
                               pin_memory=True,
                               persistent_workers=(cfg.num_workers > 0))
        n_done = 0; last_log = 0
        with torch.inference_mode():
            for x, _, idxs in loader_t:
                sm = padim_score_batch(model, x, padim,
                                          input_size=cfg.input_size,
                                          tta=cfg.tta, device=device).numpy()
                for b in range(sm.shape[0]):
                    s = gaussian_smooth(sm[b], cfg.smooth_sigma)
                    s = maybe_resize_to_submission(s)
                    test_results.append((test[int(idxs[b])], s))
                n_done += sm.shape[0]
                if n_done - last_log >= 200:
                    last_log = n_done
                    print(f"      scored {n_done}/{len(test)}", flush=True)

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del model, padim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "class": cls,
        "class_mean_ap": class_mean_ap,
        "eval_rows": eval_rows,
        "test_results": test_results,
        "elapsed_min": elapsed_min,
    }


def write_submission(all_test_results, run_dir: Path,
                     zip_it: bool = True) -> Path:
    sub("calibrating scores and writing submission.csv")
    scores = [sm for _, sm in all_test_results]
    if not scores:
        raise RuntimeError("no test scores")
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50"])
    ap.add_argument("--input-size", type=int, default=256,
                    choices=[224, 256, 320, 384, 448, 512])
    ap.add_argument("--feature-layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--target-layer", type=int, default=2)
    # Training
    ap.add_argument("--epochs", type=int, default=256,
                    help="Number of epochs. IGNORED if --total-iters is set.")
    ap.add_argument("--total-iters", type=int, default=None,
                    help="If set, overrides --epochs to give approximately "
                         "this many total optimization steps. Auto-scales "
                         "the epoch count per class so the iteration budget "
                         "stays constant across classes with varying dataset "
                         "sizes. Recommended: 2500 (matches CutPaste paper).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    # Augmentation
    ap.add_argument("--area-ratio", type=float, nargs=2, default=[0.02, 0.15])
    ap.add_argument("--aspect-ratio", type=float, nargs=2, default=[0.3, 3.3])
    ap.add_argument("--color-jitter", type=float, default=0.1)
    ap.add_argument("--scar-width", type=int, nargs=2, default=[10, 25])
    ap.add_argument("--scar-height", type=int, nargs=2, default=[2, 16])
    ap.add_argument("--scar-rot", type=float, default=45.0)
    # PaDiM
    ap.add_argument("--projection-dim", type=int, default=100)
    ap.add_argument("--padim-eps", type=float, default=0.01)
    ap.add_argument("--fit-batch-size", type=int, default=32)
    # Scoring
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip", "d4"])
    # Bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--save-padim-banks", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    feature_layers = tuple(sorted(set(args.feature_layers)))
    for l in feature_layers:
        if l not in LAYER_DIMS[args.backbone]:
            raise SystemExit(
                f"[FATAL] feature_layer {l} not valid for {args.backbone}; "
                f"valid: {sorted(LAYER_DIMS[args.backbone])}")
    if args.target_layer not in feature_layers:
        raise SystemExit(
            f"[FATAL] --target-layer {args.target_layer} must be in "
            f"--feature-layers {list(feature_layers)}.")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone, input_size=args.input_size,
        feature_layers=feature_layers, target_layer=args.target_layer,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
        amp=not args.no_amp, num_workers=args.num_workers,
        area_ratio=tuple(args.area_ratio),
        aspect_ratio=tuple(args.aspect_ratio),
        color_jitter=args.color_jitter,
        scar_width=tuple(args.scar_width),
        scar_height=tuple(args.scar_height),
        scar_rot=args.scar_rot,
        projection_dim=args.projection_dim, padim_eps=args.padim_eps,
        fit_batch_size=args.fit_batch_size,
        score_batch_size=args.score_batch_size,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        save_checkpoints=args.save_checkpoints,
        save_padim_banks=args.save_padim_banks,
        zip_submission=not args.no_zip,
        run_tag=args.run_tag,
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tee_to(run_dir / "run_log.txt"):
        hr(f"CUTPASTE v2 — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  report_dir       : {cfg.report_dir}")
        print(f"  run_dir          : {run_dir}")
        print(f"  backbone         : {cfg.backbone}")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  feature_layers   : {list(cfg.feature_layers)}")
        print(f"  target_layer     : {cfg.target_layer}")
        if cfg.total_iters is not None:
            print(f"  total_iters      : {cfg.total_iters}  "
                  f"(overrides --epochs={cfg.epochs})")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}  (effective: {3 * cfg.batch_size})")
        print(f"  lr / wd / mom    : {cfg.lr} / {cfg.weight_decay} / {cfg.momentum}")
        print(f"  amp              : {cfg.amp}")
        print(f"  num_workers      : {cfg.num_workers}")
        print(f"  cutpaste cfg     : area={cfg.area_ratio} aspect={cfg.aspect_ratio} "
              f"cj={cfg.color_jitter}")
        print(f"  scar     cfg     : w={cfg.scar_width} h={cfg.scar_height} "
              f"rot=±{cfg.scar_rot}°")
        print(f"  projection_dim   : {cfg.projection_dim}   "
              f"padim_eps={cfg.padim_eps}")
        print(f"  score_batch_size : {cfg.score_batch_size}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        full_dim = sum(LAYER_DIMS[cfg.backbone][l] for l in cfg.feature_layers)
        print(f"  fused feature dim: {full_dim} (projecting to {cfg.projection_dim})")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(cfg).items()}, f, indent=2)

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("\n[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): {', '.join(classes)}")

        all_test_results = []
        all_eval_rows: list[dict] = []
        class_aps: dict[str, float] = {}
        class_elapsed: dict[str, float] = {}
        for cls in classes:
            res = run_one_class(cls, records, cfg, run_dir, device)
            all_test_results.extend(res["test_results"])
            all_eval_rows.extend(res["eval_rows"])
            class_aps[cls] = res["class_mean_ap"]
            class_elapsed[cls] = res["elapsed_min"]

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
                w = csv.DictWriter(f, fieldnames=list(all_eval_rows[0].keys()))
                w.writeheader()
                for row in all_eval_rows: w.writerow(row)
            print(f"  saved per-(class, anomaly_type) AP table -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload to Kaggle:\n"
                  f"    {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        row = {
            "run_id": run_id,
            "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": cfg.backbone,
            "feature_layers": "+".join(str(l) for l in cfg.feature_layers),
            "target_layer": cfg.target_layer,
            "input_size": cfg.input_size,
            "patch_size": 1,
            "knn_k": "",
            "smooth_sigma": cfg.smooth_sigma,
            "tta": cfg.tta,
            "batch_size": cfg.batch_size,
            "score_batch_size": cfg.score_batch_size,
            "seed": cfg.seed,
            "n_classes": len(classes),
            **{f"AP_{c}": f"{class_aps.get(c, float('nan')):.4f}"
                for c in sorted(class_aps)},
            "AP_overall": f"{overall_ap:.4f}",
            "runtime_min": f"{(time.time() - t_total) / 60:.1f}",
            "submission_path": str(run_dir / "submission.zip")
                                if not cfg.skip_submission else "",
            "notes": (f"cutpaste v2 "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size} pd{cfg.projection_dim} "
                      f"target_L{cfg.target_layer}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")

        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()