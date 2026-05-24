"""CutPaste — v5 adds frozen DINOv2/v3 ViT backbone mode.

# Why v5 (key design decision)

The v3/v4 CutPaste trains a ResNet end-to-end with SGD lr=0.03. Applying
that recipe to a DINOv2/v3 ViT would catastrophically degrade the
pretrained SSL features — DINO's training was done with momentum ~0.99,
adaptive optimisers, and orders-of-magnitude smaller learning rates.

v5 introduces a `--encoder-mode frozen` option (default for ViT backbones)
in which:

  * The DINOv2/v3 backbone is FROZEN.
  * A small projection head (Conv2d 1×1 stack) is trained on top of
    patch tokens.
  * A classifier head sits on top of (global-avg-pool of projected
    patch tokens) to do the 3-class CutPaste task (normal / cutpaste / scar).
  * PatchCore-NN scoring uses the PROJECTED patch tokens — so the
    learned task-adapted features are the bank.

For ResNet backbones the behaviour is unchanged (full end-to-end SGD
training).

# What this gives the stacker

A new track whose features are:
  - Pretrained on web-scale SSL (DINOv2/v3): strong general descriptors.
  - Adapted by a tiny projection head to discriminate CutPaste anomalies:
    task-relevant.
  - Independent error mode from plain PatchCore-on-DINOv2 (exp7) because
    the projection is trained, not raw.

# Stacker contract — unchanged

  $RUN/submission.csv, $RUN/local_predictions.npz with image_paths.

# Dependencies

  Same as v4, plus dinov3_loader.py in the same directory.
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
from torchvision.models import (resnet18, ResNet18_Weights,
                                   resnet50, ResNet50_Weights)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patchcore_baseline_v2 import (
    patchify_and_combine, greedy_coreset,
    ImageRecord, scan_dataset,
    pixel_average_precision, gaussian_smooth,
    calibrate_to_unit, float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
    IMAGENET_MEAN, IMAGENET_STD,
    BACKBONE_SHORT, RESNET_BACKBONES, DINO_BACKBONES,
)
from local_preds_saver import LocalPredSaver
from dinov3_loader import (load_dino_backbone, get_patch_tokens_at_layers,
                              is_dino_backbone, DINOV3_CONVNEXT_SPECS)

PROJECT_ROOT = Path("/work/u10813429/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIEW_RE = re.compile(r"^(?P<base>.+?)_view(?P<v>\d+)\.[A-Za-z]+$")
SUBMISSION_H = SUBMISSION_W = 224


# ─── Tee logger ──────────────────────────────────────────────────────────────
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
    old = sys.stdout; sys.stdout = Tee(old, f)
    try: yield
    finally: sys.stdout = old; f.close()


def hr(t, c="="): print(f"\n{c * 78}\n  {t}\n{c * 78}")
def sub(t): print(f"\n--- {t} ---")
def now_hms(): return time.strftime("%H:%M:%S")


# ─── CutPaste augs (PIL in, PIL out) — unchanged from v4 ─────────────────────
def _color_jitter(patch, strength):
    if strength <= 0: return patch
    cj = transforms.ColorJitter(brightness=strength, contrast=strength,
                                 saturation=strength, hue=min(strength, 0.5))
    return cj(patch)


def apply_cutpaste(img, area_ratio=(0.02, 0.15), aspect_ratio=(0.3, 3.3),
                    color_jitter=0.1, rng=None):
    r = rng or random
    W, H = img.size; total = W * H
    pw = ph = 0
    for _ in range(10):
        target_area = r.uniform(*area_ratio) * total
        lo, hi = math.log(aspect_ratio[0]), math.log(aspect_ratio[1])
        aspect = math.exp(r.uniform(lo, hi))
        pw = max(1, int(round(math.sqrt(target_area * aspect))))
        ph = max(1, int(round(math.sqrt(target_area / aspect))))
        if pw < W and ph < H: break
    else:
        return img.copy()
    sx = r.randint(0, W - pw); sy = r.randint(0, H - ph)
    patch = _color_jitter(img.crop((sx, sy, sx + pw, sy + ph)), color_jitter)
    tx = r.randint(0, W - pw); ty = r.randint(0, H - ph)
    out = img.copy(); out.paste(patch, (tx, ty))
    return out


def apply_cutpaste_scar(img, width=(10, 25), height=(2, 16),
                         rotation_deg=45.0, color_jitter=0.1, rng=None):
    r = rng or random
    W, H = img.size
    pw = r.randint(*width); ph = r.randint(*height)
    if pw >= W or ph >= H: return img.copy()
    sx = r.randint(0, W - pw); sy = r.randint(0, H - ph)
    patch = _color_jitter(img.crop((sx, sy, sx + pw, sy + ph)), color_jitter)
    angle = r.uniform(-rotation_deg, rotation_deg)
    if patch.mode != "RGBA": patch = patch.convert("RGBA")
    patch = patch.rotate(angle, resample=Image.BILINEAR, expand=True)
    rw, rh = patch.size
    if rw >= W or rh >= H: return img.copy()
    tx = r.randint(0, W - rw); ty = r.randint(0, H - rh)
    out = img.copy().convert("RGB")
    out.paste(patch, (tx, ty), mask=patch.split()[-1])
    return out


# ─── Datasets ────────────────────────────────────────────────────────────────
class CutPasteTrainDataset(Dataset):
    def __init__(self, records, input_size, area_ratio=(0.02, 0.15),
                 aspect_ratio=(0.3, 3.3), color_jitter=0.1,
                 scar_width=(10, 25), scar_height=(2, 16), scar_rot=45.0):
        self.records = records; self.input_size = input_size
        self.area_ratio = area_ratio; self.aspect_ratio = aspect_ratio
        self.color_jitter = color_jitter
        self.scar_width = scar_width; self.scar_height = scar_height
        self.scar_rot = scar_rot
        self.resize = transforms.Resize((input_size, input_size))
        self.to_tensor = transforms.ToTensor()
        self.normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self): return len(self.records)

    def _to_tensor(self, im): return self.normalise(self.to_tensor(im))

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r.path) as im:
            im = im.convert("RGB"); base = self.resize(im).copy()
        cp = apply_cutpaste(base, area_ratio=self.area_ratio,
                              aspect_ratio=self.aspect_ratio,
                              color_jitter=self.color_jitter)
        sc = apply_cutpaste_scar(base, width=self.scar_width,
                                   height=self.scar_height,
                                   rotation_deg=self.scar_rot,
                                   color_jitter=self.color_jitter)
        return self._to_tensor(base), self._to_tensor(cp), self._to_tensor(sc)


class InferenceDataset(Dataset):
    def __init__(self, records, input_size, load_masks):
        self.records = records; self.input_size = input_size
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
                mm = mm.convert("L").resize((self.input_size, self.input_size),
                                              Image.NEAREST)
                m = (np.asarray(mm) > 127).astype(np.float32)
        else:
            m = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        return x, torch.from_numpy(m), i


def worker_init_fn(_worker_id):
    base = torch.initial_seed() % 2 ** 32
    np.random.seed(base); random.seed(base)


# ─── ResNet CutPasteNet — unchanged from v4 ──────────────────────────────────
class CutPasteNetResNet(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=3):
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
        self.kind = "resnet"

    def forward(self, x):
        return self.head(self.encoder(x))

    def extract_features(self, x, layers=(1, 2, 3)):
        e = self.encoder
        x = e.conv1(x); x = e.bn1(x); x = e.relu(x); x = e.maxpool(x)
        out = {}
        x = e.layer1(x)
        if 1 in layers: out[1] = x
        x = e.layer2(x)
        if 2 in layers: out[2] = x
        x = e.layer3(x)
        if 3 in layers: out[3] = x
        if 4 in layers:
            x = e.layer4(x); out[4] = x
        return out


# ─── ViT CutPasteNet — NEW in v5 ─────────────────────────────────────────────
class CutPasteNetViT(nn.Module):
    """Frozen DINOv2/v3 backbone + per-layer 1x1 projection head + CLS classifier.

    The projection head is a stack of 1x1 Conv2d → BN → ReLU → 1x1 Conv2d
    applied on each requested layer's patch tokens. Its output replaces
    the raw backbone patch tokens in PatchCore-NN scoring.

    Why per-layer rather than one shared projection: each DINOv2/v3 block
    has its own activation statistics, so a per-layer linear projection
    is the safest minimal adapter that doesn't disturb cross-layer
    relative scaling.
    """
    def __init__(self, backbone_name: str, feature_layers: tuple[int, ...],
                 num_classes: int = 3, proj_dim: int = 256):
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_layers = tuple(sorted(set(feature_layers)))
        self.kind = "vit"

        backbone, info = load_dino_backbone(backbone_name)
        self.backbone = backbone
        self.info = info
        self.embed_dim = info.embed_dim
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        # Per-layer 1x1 projection head (Conv2d so it works directly on
        # the (B, C, H, W) patch-token grids returned by the loader).
        self.proj_dim = proj_dim
        self.projections = nn.ModuleDict({
            str(l): nn.Sequential(
                nn.Conv2d(self.embed_dim, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(proj_dim, proj_dim, 1),
            ) for l in self.feature_layers
        })

        # Classifier head consumes mean-pooled projection of the LAST
        # requested layer — keeps the head small and stable.
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, x):
        """Returns the classifier logits."""
        with torch.no_grad():
            feats = get_patch_tokens_at_layers(self.backbone, self.info,
                                                  x, self.feature_layers)
        last_layer = self.feature_layers[-1]
        z = self.projections[str(last_layer)](feats[last_layer])  # (B, P, h, w)
        pooled = z.flatten(2).mean(dim=2)                          # (B, P)
        return self.head(pooled)

    def extract_features(self, x, layers=None):
        """Returns {layer_idx: projected patch tokens (B, C, h, w)} for
        the requested layers; defaults to self.feature_layers."""
        layers = tuple(sorted(layers)) if layers else self.feature_layers
        with torch.no_grad():
            feats = get_patch_tokens_at_layers(self.backbone, self.info,
                                                  x, layers)
        out = {}
        for l in layers:
            out[l] = self.projections[str(l)](feats[l])
        return out


def build_cutpaste_net(cfg, num_classes=3):
    if cfg.backbone in RESNET_BACKBONES:
        return CutPasteNetResNet(backbone=cfg.backbone, num_classes=num_classes)
    if is_dino_backbone(cfg.backbone):
        return CutPasteNetViT(backbone_name=cfg.backbone,
                                 feature_layers=cfg.feature_layers,
                                 num_classes=num_classes,
                                 proj_dim=cfg.vit_proj_dim)
    raise SystemExit(f"[FATAL] unknown --backbone: {cfg.backbone}")


# ─── Training ────────────────────────────────────────────────────────────────
def train_cutpaste(model, records, cfg, device):
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
                         worker_init_fn=worker_init_fn, drop_last=False)
    iters_per_epoch = len(loader)
    if cfg.total_iters and cfg.total_iters > 0:
        n_epochs = max(1, math.ceil(cfg.total_iters / max(iters_per_epoch, 1)))
        print(f"    [auto-epoch] total_iters={cfg.total_iters} -> {n_epochs} epochs")
    else:
        n_epochs = cfg.epochs

    # Trainable params depend on the encoder mode.
    if model.kind == "vit":
        trainable = list(model.projections.parameters()) + list(model.head.parameters())
        print(f"    [{now_hms()}] ViT mode: backbone FROZEN, "
              f"head + projection trainable ({sum(p.numel() for p in trainable)/1e6:.2f}M)")
        # Use Adam for the small head: SGD lr=0.03 would be totally wrong here.
        optim = torch.optim.AdamW(trainable, lr=cfg.lr_head,
                                     weight_decay=cfg.weight_decay)
    else:
        trainable = list(model.parameters())
        print(f"    [{now_hms()}] ResNet mode: full end-to-end SGD "
              f"({sum(p.numel() for p in trainable)/1e6:.2f}M)")
        optim = torch.optim.SGD(trainable, lr=cfg.lr,
                                   momentum=cfg.momentum,
                                   weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(iters_per_epoch * n_epochs, 1))
    criterion = nn.CrossEntropyLoss()
    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        # Frozen backbone stays in eval mode (BN/dropout off).
        if model.kind == "vit":
            model.backbone.eval()
        loss_sum = correct = total = 0
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
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); scheduler.step()
            loss_sum += loss.item() * x.shape[0]
            correct  += (logits.argmax(dim=-1) == y).sum().item()
            total    += x.shape[0]
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch+1:>4}/{n_epochs}  "
                  f"loss={loss_sum/max(total,1):.4f}  "
                  f"acc={correct/max(total,1):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    model.eval()
    print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")


# ─── PatchCore-NN scorer (works for both ResNet and ViT models) ──────────────
def fit_memory_bank(model, records, cfg, device):
    ds = InferenceDataset(records, input_size=cfg.input_size, load_masks=False)
    loader = DataLoader(ds, batch_size=cfg.fit_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    feats_list = []; feature_hw = None; feature_dim = None
    print(f"    [{now_hms()}] extracting train/good patch features "
          f"({len(records)} images)...")
    model.eval()
    with torch.inference_mode():
        for x, _, _ in loader:
            x = x.to(device, non_blocking=True)
            maps = model.extract_features(x, layers=cfg.feature_layers)
            pf = patchify_and_combine(maps, patch_size=cfg.patch_size,
                                        target_layer=cfg.target_layer)
            if feature_hw is None:
                P = pf.shape[1]
                H = W = int(math.isqrt(P))
                feature_hw = (H, W); feature_dim = pf.shape[2]
            pf = pf.reshape(-1, pf.shape[-1]).detach().cpu()
            feats_list.append(pf)
    all_feats = torch.cat(feats_list, dim=0)
    if cfg.coreset_fp16: all_feats = all_feats.half()
    print(f"    -> {all_feats.shape[0]} patch features  D={feature_dim}  "
          f"{all_feats.element_size() * all_feats.numel() / 1e9:.2f} GB CPU")
    n_select = max(int(cfg.coreset_frac * all_feats.shape[0]), 1)
    print(f"    [{now_hms()}] coreset ({cfg.coreset_algo}, "
          f"batch={cfg.coreset_batch}): {n_select} of {all_feats.shape[0]}")
    idx_cpu = greedy_coreset(all_feats, n_select, device,
                              seed=cfg.seed,
                              project_chunk=cfg.project_chunk,
                              algo=cfg.coreset_algo,
                              batch_size=cfg.coreset_batch)
    selected = all_feats[idx_cpu]
    del all_feats
    memory = F.normalize(selected.to(device).float(), p=2, dim=-1)
    memory_dtype = torch.float16 if cfg.memory_dtype == "fp16" else torch.float32
    memory = memory.to(memory_dtype).contiguous()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"    [{now_hms()}] memory bank: {tuple(memory.shape)} {memory.dtype}")
    return {"memory": memory, "feature_hw": feature_hw,
            "feature_dim": feature_dim,
            "feature_layers": cfg.feature_layers,
            "target_layer": cfg.target_layer,
            "patch_size": cfg.patch_size}


@torch.inference_mode()
def _nn_compute_score_and_pf(model, x, bank, cfg):
    maps = model.extract_features(x, layers=bank["feature_layers"])
    pf = patchify_and_combine(maps, patch_size=bank["patch_size"],
                                target_layer=bank["target_layer"])
    B, P, C = pf.shape
    H = W = int(math.isqrt(P))
    flat = pf.reshape(-1, C)
    N_q = flat.shape[0]
    memory = bank["memory"]
    memory_dtype = memory.dtype; M_total = memory.shape[0]
    dist_min = torch.empty(N_q, device=memory.device, dtype=torch.float32)
    for s in range(0, N_q, cfg.score_chunk):
        e = min(N_q, s + cfg.score_chunk)
        q = flat[s:e].to(memory_dtype)
        max_sim = torch.full((q.shape[0],), -2.0, device=memory.device,
                              dtype=memory_dtype)
        for ms in range(0, M_total, cfg.memory_chunk):
            me = min(M_total, ms + cfg.memory_chunk)
            sim = q @ memory[ms:me].T
            chunk_max = sim.max(dim=1).values
            torch.maximum(max_sim, chunk_max, out=max_sim)
        dist_min[s:e] = (1.0 - max_sim.float())
    score_lr = dist_min.reshape(B, H, W)
    return score_lr, pf


@torch.inference_mode()
def _nn_score_one(model, x, bank, cfg):
    score_lr, _ = _nn_compute_score_and_pf(model, x, bank, cfg)
    score = F.interpolate(score_lr.unsqueeze(1),
                          size=(cfg.input_size, cfg.input_size),
                          mode="bilinear", align_corners=False).squeeze(1)
    return score.cpu()


@torch.inference_mode()
def nn_score_batch(model, x, bank, cfg, tta="none", device=None):
    if device is None: device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)
    acc = None; n = 0
    def _add(s):
        nonlocal acc, n
        if acc is None: acc = s.clone()
        else: acc += s
        n += 1
    _add(_nn_score_one(model, x, bank, cfg))
    if tta in ("hflip", "hvflip", "d4"):
        s = _nn_score_one(model, torch.flip(x, dims=[-1]), bank, cfg)
        _add(torch.flip(s, dims=[-1]))
    if tta in ("vflip", "hvflip", "d4"):
        s = _nn_score_one(model, torch.flip(x, dims=[-2]), bank, cfg)
        _add(torch.flip(s, dims=[-2]))
    if tta == "d4":
        for k in (1, 2, 3):
            s = _nn_score_one(model, torch.rot90(x, k=k, dims=[-2, -1]),
                                bank, cfg)
            _add(torch.rot90(s, k=-k, dims=[-2, -1]))
    return acc / max(n, 1)


# ─── Per-class pipeline ──────────────────────────────────────────────────────
def _score_records_standard(model, records, bank, cfg, load_masks):
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores, gts = {}, {}
    for x, masks, idxs in loader:
        sm = nn_score_batch(model, x, bank, cfg, tta=cfg.tta).numpy()
        m_np = masks.numpy()
        for b in range(sm.shape[0]):
            scores[int(idxs[b])] = sm[b]
            gts[int(idxs[b])] = m_np[b]
    return scores, gts


def run_one_class(cls, records_all, cfg, run_dir, device, local_saver=None):
    hr(f"CLASS {cls}", "─")
    t_start = time.time()
    train_good = [r for r in records_all if r.cls == cls and r.split == "train_good"]
    train_anom = [r for r in records_all if r.cls == cls and r.split == "train_anomaly"]
    test       = [r for r in records_all if r.cls == cls and r.split == "test"]
    print(f"  train_good={len(train_good)}  train_anomaly={len(train_anom)}  test={len(test)}")
    if not train_good:
        return {"class": cls, "class_mean_ap": float("nan"),
                "eval_rows": [], "test_results": [], "elapsed_min": 0.0}

    model = build_cutpaste_net(cfg, num_classes=3).to(device)
    train_cutpaste(model, train_good, cfg, device)
    bank = fit_memory_bank(model, train_good, cfg, device)

    eval_rows = []; class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation (tta={cfg.tta})")
        scores_raw, gts = _score_records_standard(model, train_anom, bank, cfg,
                                                       load_masks=True)
        scores_by_idx, gt_by_idx = {}, {}
        for r_idx, sm_raw in scores_raw.items():
            scores_by_idx[r_idx] = gaussian_smooth(sm_raw, cfg.smooth_sigma)
            gt_by_idx[r_idx] = gts[r_idx]
        by_anom = defaultdict(list)
        for ridx, s in scores_by_idx.items():
            r = train_anom[ridx]
            ap = pixel_average_precision(s, gt_by_idx[ridx])
            by_anom[r.anomaly_type or "?"].append(ap)
            if local_saver is not None:
                local_saver.add(cls=cls, anomaly_type=r.anomaly_type or "unknown",
                                 view_idx=int(ridx), score_map=s,
                                 gt_mask=gt_by_idx[ridx], image_path=r.path)
        print(f"    {'anomaly_type':<14} {'n_views':>8} {'pixel-AP (mean ± std)':>26}")
        per_type_means = []
        for a_type in sorted(by_anom):
            arr = np.asarray(by_anom[a_type])
            per_type_means.append(float(arr.mean()))
            print(f"    {a_type:<14} {len(arr):>8} {arr.mean():>15.4f} ± {arr.std():.4f}")
            eval_rows.append({"class": cls, "anomaly_type": a_type,
                               "n_views": int(len(arr)),
                               "ap_mean": float(arr.mean()),
                               "ap_std":  float(arr.std()),
                               "ap_min":  float(arr.min()),
                               "ap_max":  float(arr.max())})
        class_mean_ap = float(np.mean(per_type_means)) if per_type_means else 0.0
        print(f"    >>> class {cls} mean pixel-AP: {class_mean_ap:.4f}")

    test_results = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images")
        scores_raw, _ = _score_records_standard(model, test, bank, cfg,
                                                     load_masks=False)
        for r_idx, sm_raw in scores_raw.items():
            sm = gaussian_smooth(sm_raw, cfg.smooth_sigma)
            sm = maybe_resize_to_submission(sm)
            test_results.append((test[r_idx], sm))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del model, bank
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"class": cls, "class_mean_ap": class_mean_ap,
            "eval_rows": eval_rows, "test_results": test_results,
            "elapsed_min": elapsed_min}


def write_submission(all_test_results, run_dir, zip_it=True):
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


# ─── Config + main ───────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    backbone: str = "resnet18"
    input_size: int = 256
    feature_layers: tuple[int, ...] = (1, 2, 3)
    target_layer: int = 2
    # ViT mode extras
    vit_proj_dim: int = 256
    # Training
    epochs: int = 256
    total_iters: int | None = 2500
    batch_size: int = 64
    lr: float = 0.03           # SGD lr for ResNet mode
    lr_head: float = 1e-4      # AdamW lr for ViT mode
    momentum: float = 0.9
    weight_decay: float = 3e-5
    amp: bool = True
    num_workers: int = 8
    # Augmentation
    area_ratio: tuple[float, float] = (0.02, 0.15)
    aspect_ratio: tuple[float, float] = (0.3, 3.3)
    color_jitter: float = 0.1
    scar_width: tuple[int, int] = (10, 25)
    scar_height: tuple[int, int] = (2, 16)
    scar_rot: float = 45.0
    # PatchCore-NN
    coreset_frac: float = 0.05
    coreset_algo: str = "minibatch"
    coreset_batch: int = 128
    coreset_fp16: bool = True
    patch_size: int = 3
    score_chunk: int = 4096
    memory_chunk: int = 16384
    memory_dtype: str = "fp16"
    project_chunk: int = 65536
    fit_batch_size: int = 32
    score_batch_size: int = 16
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    # Bookkeeping
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    zip_submission: bool = True
    run_tag: str = ""


def make_run_id(cfg):
    fp = json.dumps({
        "method": "cutpaste", "v": 5,
        "backbone": cfg.backbone, "input_size": cfg.input_size,
        "feature_layers": list(cfg.feature_layers),
        "target_layer": cfg.target_layer,
        "total_iters": cfg.total_iters, "batch_size": cfg.batch_size,
        "lr": cfg.lr, "lr_head": cfg.lr_head,
        "vit_proj_dim": cfg.vit_proj_dim,
        "coreset_frac": cfg.coreset_frac,
        "smooth_sigma": cfg.smooth_sigma, "tta": cfg.tta, "seed": cfg.seed,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bb = BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)
    budget = (f"it{cfg.total_iters}" if cfg.total_iters else f"e{cfg.epochs}")
    bits = (f"{stamp}_cutpaste-pcnn_{bb}_in{cfg.input_size}"
            f"_{budget}_bs{cfg.batch_size}_pc_cs{int(cfg.coreset_frac*100):02d}"
            f"mb{cfg.coreset_batch}")
    if cfg.tta != "none": bits += f"_tta-{cfg.tta}"
    if cfg.run_tag: bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--backbone", default="resnet18",
                    help="ResNet (resnet18/50) or DINOv2/v3 name.")
    ap.add_argument("--input-size", type=int, default=256,
                    help="Multiple of patch-size for ViT backbones.")
    ap.add_argument("--feature-layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--target-layer", type=int, default=2)
    ap.add_argument("--vit-proj-dim", type=int, default=256,
                    help="ViT mode: width of the per-layer 1x1 projection head.")
    ap.add_argument("--epochs", type=int, default=256)
    ap.add_argument("--total-iters", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.03,
                    help="ResNet mode: SGD lr. ViT mode ignores this.")
    ap.add_argument("--lr-head", type=float, default=1e-4,
                    help="ViT mode: AdamW lr for the trainable head.")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--area-ratio", type=float, nargs=2, default=[0.02, 0.15])
    ap.add_argument("--aspect-ratio", type=float, nargs=2, default=[0.3, 3.3])
    ap.add_argument("--color-jitter", type=float, default=0.1)
    ap.add_argument("--scar-width", type=int, nargs=2, default=[10, 25])
    ap.add_argument("--scar-height", type=int, nargs=2, default=[2, 16])
    ap.add_argument("--scar-rot", type=float, default=45.0)
    ap.add_argument("--coreset-frac", type=float, default=0.05)
    ap.add_argument("--coreset-algo", default="minibatch",
                    choices=["minibatch", "exact"])
    ap.add_argument("--coreset-batch", type=int, default=128)
    ap.add_argument("--no-coreset-fp16", action="store_true")
    ap.add_argument("--patch-size", type=int, default=3)
    ap.add_argument("--score-chunk", type=int, default=4096)
    ap.add_argument("--memory-chunk", type=int, default=16384)
    ap.add_argument("--memory-dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--project-chunk", type=int, default=65536)
    ap.add_argument("--fit-batch-size", type=int, default=32)
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip", "d4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    feature_layers = tuple(sorted(set(args.feature_layers)))
    if args.target_layer not in feature_layers:
        raise SystemExit(f"[FATAL] --target-layer {args.target_layer} not in "
                          f"--feature-layers {list(feature_layers)}")
    # Patch-size sanity for ViT
    if is_dino_backbone(args.backbone):
        from dinov3_loader import (DINOV2_SPECS as _V2,
                                       DINOV3_VIT_SPECS as _V3V,
                                       DINOV3_CONVNEXT_SPECS as _V3C)
        if args.backbone in _V2: patch = _V2[args.backbone]["patch"]
        elif args.backbone in _V3V: patch = _V3V[args.backbone]["patch"]
        else: patch = _V3C[args.backbone]["patch_eff"]
        if args.input_size % patch != 0:
            raise SystemExit(f"[FATAL] --input-size {args.input_size} not "
                              f"multiple of {patch}")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone, input_size=args.input_size,
        feature_layers=feature_layers, target_layer=args.target_layer,
        vit_proj_dim=args.vit_proj_dim,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr, lr_head=args.lr_head,
        momentum=args.momentum, weight_decay=args.weight_decay,
        amp=not args.no_amp, num_workers=args.num_workers,
        area_ratio=tuple(args.area_ratio),
        aspect_ratio=tuple(args.aspect_ratio),
        color_jitter=args.color_jitter,
        scar_width=tuple(args.scar_width),
        scar_height=tuple(args.scar_height),
        scar_rot=args.scar_rot,
        coreset_frac=args.coreset_frac,
        coreset_algo=args.coreset_algo,
        coreset_batch=args.coreset_batch,
        coreset_fp16=not args.no_coreset_fp16,
        patch_size=args.patch_size,
        score_chunk=args.score_chunk,
        memory_chunk=args.memory_chunk,
        memory_dtype=args.memory_dtype,
        project_chunk=args.project_chunk,
        fit_batch_size=args.fit_batch_size,
        score_batch_size=args.score_batch_size,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        zip_submission=not args.no_zip, run_tag=args.run_tag,
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tee_to(run_dir / "run_log.txt"):
        hr(f"CUTPASTE v5 (backbone={cfg.backbone}) — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  backbone         : {cfg.backbone}")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  feature_layers   : {list(cfg.feature_layers)}")
        print(f"  target_layer     : {cfg.target_layer}")
        print(f"  encoder_mode     : "
              f"{'frozen (ViT)' if is_dino_backbone(cfg.backbone) else 'e2e (ResNet)'}")
        if is_dino_backbone(cfg.backbone):
            print(f"  vit_proj_dim     : {cfg.vit_proj_dim}")
            print(f"  lr_head          : {cfg.lr_head}")
        else:
            print(f"  lr               : {cfg.lr} (SGD)")
        print(f"  total_iters      : {cfg.total_iters}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  device           : {device}")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(cfg).items()}, f, indent=2)

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"  running on {len(classes)} class(es): {', '.join(classes)}")

        local_saver = (LocalPredSaver()
                        if (not cfg.skip_eval and not args.no_save_local_preds)
                        else None)
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
            print(f"  saved per-(class, anomaly_type) AP -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            save_test_predictions(all_test_results, run_dir)
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload: {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        bb_short = BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)
        row = {
            "run_id": run_id, "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": bb_short,
            "feature_layers": "+".join(str(l) for l in cfg.feature_layers),
            "target_layer": cfg.target_layer, "input_size": cfg.input_size,
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
            "notes": (f"cutpaste v5 backbone={cfg.backbone} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()