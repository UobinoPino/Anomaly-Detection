"""Spacepresso EfficientAD — v2 adds DINOv2/v3 ViT teacher option.

# v2 changes (additive over v1)

  * NEW arg: `--teacher-backbone` (default: resnet18).
        resnet18 (original)            — local features, fast, 128-ch teacher
        dinov2_vits14 / vitb14 / vitl14 / reg variants
        dinov3_vits16 / vitb16 / vitl16 / vith16plus
        dinov3_convnext_{tiny,small,base,large}
  * NEW arg: `--teacher-layer` for ViT teachers (default: last block).
        Picking a mid-block (e.g. block 9 of vits14) often gives better
        spatial localisation; the last block is more semantic. Tune on val.
  * Patch-size sanity: ViT teachers force `--input-size % patch == 0`.
  * TEACHER_CHANNELS now resolved at runtime from the chosen backbone.
    Student head + AE output dim adapt automatically.
  * Bilinear-upsample of teacher features to a target stride of 4
    is unchanged; this keeps the existing student PDN-S architecture
    (which expects teacher features at H/4) intact for all backbones.

# Why this is worth doing

The original EfficientAD-RN18 teacher gives student/AE targets that are
ImageNet-classification-tuned. DINOv2/v3 features are SSL-tuned for
dense prediction with substantially sharper object boundaries (see
Dinomaly results: DINOv3-L pixel-AUROC 98.78% on MVTec). The student
still does the cheap forward; only the teacher swaps.

# Stacker contract — unchanged

  $RUN/submission.csv, $RUN/local_predictions.npz with image_paths.

# Dependencies

  Same as v1, plus dinov3_loader.py in the same directory.
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
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patchcore_baseline_v2 import (
    ImageRecord, scan_dataset,
    pixel_average_precision, gaussian_smooth,
    calibrate_to_unit, float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
    IMAGENET_MEAN, IMAGENET_STD,
    BACKBONE_CHANNELS, DINO_BACKBONES, RESNET_BACKBONES,
    BACKBONE_SHORT,
)
from local_preds_saver import LocalPredSaver
from dinov3_loader import (
    load_dino_backbone, get_patch_tokens_at_layers,
    validate_input_size, is_dino_backbone,
    DINOV3_CONVNEXT_SPECS,
)


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
# Datasets — unchanged from v1
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
# Teacher networks — v2 adds DINOv2/v3 path
# ─────────────────────────────────────────────────────────────────────────────
class _TeacherRN18(nn.Module):
    """Original EfficientAD teacher: ResNet-18 layer2 @ H/8, upsampled to H/4."""
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        for p in self.parameters():
            p.requires_grad_(False)
        self.out_channels = 128
        self.target_stride = 4
        self.eval()

    @torch.inference_mode()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)                # H/8, 128 ch
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        return x                          # (B, 128, H/4, W/4)


class _TeacherDinoViT(nn.Module):
    """DINOv2 or DINOv3 ViT teacher.

    Picks features from a single block (mid by default), then bilinearly
    upsamples to the target stride (4) so the student PDN-S can keep
    its original architecture.
    """
    def __init__(self, backbone_name: str, block_idx: int | None,
                 target_stride: int = 4):
        super().__init__()
        model, info = load_dino_backbone(backbone_name)
        self.dino = model
        self.info = info
        # Default: a mid-block. For 12-block ViT-S that's block 9
        # (matches your successful UniAD config); for 24-block ViT-L
        # take block 18.
        if block_idx is None:
            block_idx = max(info.n_blocks - 3, 0)
        self.block_idx = int(block_idx)
        self.out_channels = info.embed_dim
        self.target_stride = target_stride
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x):
        # x is (B, 3, H, W). DINOv2/v3 produces (B, C, H/P, W/P).
        feats = get_patch_tokens_at_layers(self.dino, self.info,
                                              x, [self.block_idx])
        f = feats[self.block_idx]
        # Upsample to H/target_stride. For ViT-S/14 at input 392:
        # f is H/14 = 28, target H/4 = 98 -> upsample x3.5.
        B, _, H, W = x.shape
        th, tw = H // self.target_stride, W // self.target_stride
        if f.shape[-2:] != (th, tw):
            f = F.interpolate(f, size=(th, tw), mode="bilinear",
                              align_corners=False)
        return f


class _TeacherDinoConvNeXt(nn.Module):
    """DINOv3-ConvNeXt teacher. ConvNeXt has 4 stages with strides
    4, 8, 16, 32. We pick stage 0 (stride 4) so no upsampling is
    needed — it natively lands at the target stride."""
    def __init__(self, backbone_name: str, stage_idx: int = 0,
                 target_stride: int = 4):
        super().__init__()
        model, info = load_dino_backbone(backbone_name)
        self.dino = model
        self.info = info
        self.stage_idx = int(stage_idx)
        channels = DINOV3_CONVNEXT_SPECS[backbone_name]["channels"]
        self.out_channels = channels[self.stage_idx]
        self.target_stride = target_stride
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x):
        feats = get_patch_tokens_at_layers(self.dino, self.info,
                                              x, [self.stage_idx])
        f = feats[self.stage_idx]
        B, _, H, W = x.shape
        th, tw = H // self.target_stride, W // self.target_stride
        if f.shape[-2:] != (th, tw):
            f = F.interpolate(f, size=(th, tw), mode="bilinear",
                              align_corners=False)
        return f


def build_teacher(backbone_name: str,
                    teacher_layer: int | None) -> nn.Module:
    if backbone_name == "resnet18":
        return _TeacherRN18()
    if backbone_name in DINOV3_CONVNEXT_SPECS:
        return _TeacherDinoConvNeXt(backbone_name,
                                       stage_idx=teacher_layer or 0)
    if is_dino_backbone(backbone_name):
        return _TeacherDinoViT(backbone_name, block_idx=teacher_layer)
    raise SystemExit(f"[FATAL] unknown --teacher-backbone: {backbone_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Student PDN + Autoencoder — same architecture, channels adapt to teacher
# ─────────────────────────────────────────────────────────────────────────────
def _conv_block(c_in, c_out, k=3, stride=1, pad=None):
    if pad is None: pad = k // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class StudentPDN(nn.Module):
    """PDN-S style small CNN; outputs 2*C channels at H/4."""
    def __init__(self, out_channels: int):
        super().__init__()
        self.block1 = nn.Sequential(_conv_block(3,    64),
                                       nn.AvgPool2d(2, stride=2))
        self.block2 = nn.Sequential(_conv_block(64,  128),
                                       nn.AvgPool2d(2, stride=2))
        self.block3 = nn.Sequential(_conv_block(128, 256),
                                       _conv_block(256, 256))
        self.head = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.head(x)


class Autoencoder(nn.Module):
    def __init__(self, out_channels: int, base: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, base, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 8), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_block(base * 8, base * 4),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_block(base * 4, base * 2),
        )
        self.head = nn.Conv2d(base * 2, out_channels, 1)

    def forward(self, x):
        return self.head(self.dec(self.enc(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Spatial alignment helper
# ─────────────────────────────────────────────────────────────────────────────
def _align_to(ref: torch.Tensor, *tensors: torch.Tensor):
    """Bilinearly resize each tensor to match ref's spatial size.

    EfficientAD's autoencoder uses strided conv-4-s2-p1 followed by ×2
    upsamples. For inputs whose H is a multiple of 16 (256, 384, …) the
    output matches the teacher's H/4 exactly. For ViT-friendly H values
    (e.g. 392 = 14·28) the AE rounds to 96 while the ViT teacher gives
    98 — so we resample to a common reference (the student/teacher's
    H/4) before any per-pixel arithmetic. The student PDN's two
    AvgPool2d already produce true H/4, so we use *any* of the three as
    reference (we pick the first non-AE tensor below).
    """
    H, W = ref.shape[-2], ref.shape[-1]
    out = []
    for t in tensors:
        if t.shape[-2:] != (H, W):
            t = F.interpolate(t, size=(H, W), mode="bilinear",
                              align_corners=False)
        out.append(t)
    return out if len(out) > 1 else out[0]


# ─────────────────────────────────────────────────────────────────────────────
# Training (unchanged logic from v1, just generalised teacher type)
# ─────────────────────────────────────────────────────────────────────────────
def train_efficientad(teacher, student, ae, records, cfg, device):
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
              f"{iters_per_epoch} → {n_epochs} epochs")
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
    print(f"    [{now_hms()}] training: {n_epochs} epochs × "
          f"{iters_per_epoch} iters ({total_iters} total)  "
          f"bs={cfg.batch_size}  amp={use_amp}  "
          f"hard_mining_pct={cfg.hard_mining_pct}  "
          f"teacher_channels={C_T}")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    teacher.eval()
    for epoch in range(n_epochs):
        student.train(); ae.train()
        loss_sum = ls_sum = la_sum = lst_sum = 0.0; n = 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    t_feat = teacher(x)          # (B, C, H/4, W/4)
                s_out  = student(x)               # (B, 2C, H/4, W/4)
                ae_out = ae(x)                    # (B, C, H/4-ish, …)
                # Align AE (and teacher, if needed) to student's grid.
                # student PDN is exact H/4; AE may round when H not /16.
                t_feat, ae_out = _align_to(s_out, t_feat, ae_out)
                s_t = s_out[:, :C_T]
                s_a = s_out[:, C_T:]
                diff_st = (s_t - t_feat) ** 2
                diff_st_pix = diff_st.mean(dim=1)
                k = max(int(cfg.hard_mining_pct * diff_st_pix.numel()), 1)
                topk = torch.topk(diff_st_pix.reshape(-1), k,
                                    largest=True).values
                L_st = topk.mean()
                L_ae = ((ae_out - t_feat) ** 2).mean()
                L_stae = ((s_a - ae_out.detach()) ** 2).mean()
                loss = L_st + L_ae + L_stae
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
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
# Norm stats + scoring (logic unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def compute_norm_stats(teacher, student, ae, records, cfg, device,
                        n_max=64):
    if len(records) > n_max:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(records), size=n_max, replace=False)
        records = [records[i] for i in idx]
    ds = InferenceDataset(records, input_size=cfg.input_size, load_masks=False)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    C_T = teacher.out_channels
    use_amp = (device.type == "cuda" and cfg.amp)
    st_vals, ae_vals = [], []
    teacher.eval(); student.eval(); ae.eval()
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            t_feat = teacher(x); s_out = student(x); ae_out = ae(x)
        # Align all three onto student's spatial grid.
        t_feat, ae_out = _align_to(s_out, t_feat, ae_out)
        s_t = s_out[:, :C_T].float()
        s_a = s_out[:, C_T:].float()
        t_feat   = t_feat.float()
        ae_out_f = ae_out.float()
        map_st = ((s_t - t_feat)   ** 2).mean(dim=1)
        map_ae = ((s_a - ae_out_f) ** 2).mean(dim=1)
        st_vals.append(map_st.cpu().numpy().reshape(-1))
        ae_vals.append(map_ae.cpu().numpy().reshape(-1))
    st_arr = np.concatenate(st_vals); ae_arr = np.concatenate(ae_vals)
    return {"st_mean": float(st_arr.mean()),
            "st_std":  float(st_arr.std() + 1e-9),
            "ae_mean": float(ae_arr.mean()),
            "ae_std":  float(ae_arr.std() + 1e-9)}


@torch.inference_mode()
def _compute_maps_one(teacher, student, ae, x, stats, cfg):
    use_amp = (cfg.device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        t_feat = teacher(x); s_out = student(x); ae_out = ae(x)
    # Align all three onto student's spatial grid.
    t_feat, ae_out = _align_to(s_out, t_feat, ae_out)
    C_T = teacher.out_channels
    s_t = s_out[:, :C_T].float(); s_a = s_out[:, C_T:].float()
    t_feat_f = t_feat.float(); ae_out_f = ae_out.float()
    map_st = ((s_t - t_feat_f) ** 2).mean(dim=1)
    map_ae = ((s_a - ae_out_f) ** 2).mean(dim=1)
    map_st_n = (map_st - stats["st_mean"]) / stats["st_std"]
    map_ae_n = (map_ae - stats["ae_mean"]) / stats["ae_std"]
    return (map_st_n + map_ae_n) * 0.5, t_feat_f


@torch.inference_mode()
def _maps_with_tta(teacher, student, ae, x, stats, cfg, tta):
    acc_s = acc_t = None; n = 0
    def _add(s, t):
        nonlocal acc_s, acc_t, n
        if acc_s is None: acc_s = s.clone(); acc_t = t.clone()
        else: acc_s += s; acc_t += t
        n += 1
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
    return acc_s / n, acc_t / n


@torch.inference_mode()
def score_batch_standard(teacher, student, ae, x, stats, cfg):
    x = x.to(cfg.device, non_blocking=True)
    combined, _ = _maps_with_tta(teacher, student, ae, x, stats, cfg, cfg.tta)
    up = F.interpolate(combined.unsqueeze(1),
                       size=(cfg.input_size, cfg.input_size),
                       mode="bilinear", align_corners=False).squeeze(1)
    return up.cpu()


@torch.inference_mode()
def score_sample_with_siblings(teacher, student, ae, x_sample, stats, cfg):
    x_sample = x_sample.to(cfg.device, non_blocking=True)
    combined, t_feat = _maps_with_tta(teacher, student, ae, x_sample,
                                        stats, cfg, cfg.tta)
    V, C, H_, W_ = t_feat.shape
    if V < 2:
        up = F.interpolate(combined.unsqueeze(1),
                           size=(cfg.input_size, cfg.input_size),
                           mode="bilinear", align_corners=False).squeeze(1)
        return up.cpu()
    t_flat = t_feat.permute(0, 2, 3, 1).reshape(V, -1, C)
    t_norm = F.normalize(t_flat, p=2, dim=-1)
    P = t_norm.shape[1]
    mv_dist = torch.empty(V, P, device=t_feat.device, dtype=torch.float32)
    for i in range(V):
        siblings = torch.cat([t_norm[j] for j in range(V) if j != i], dim=0)
        sim = t_norm[i].float() @ siblings.float().T
        mv_dist[i] = 1.0 - sim.max(dim=1).values
    mv_dist = mv_dist.reshape(V, H_, W_)
    sd_min = mv_dist.min(); sd_max = mv_dist.max()
    mv_norm = ((mv_dist - sd_min) / (sd_max - sd_min)
                 if (sd_max - sd_min) > 1e-9 else torch.zeros_like(mv_dist))
    boost = (1.0 - cfg.mv_alpha) + cfg.mv_alpha * mv_norm
    boosted = combined * boost
    up = F.interpolate(boosted.unsqueeze(1),
                       size=(cfg.input_size, cfg.input_size),
                       mode="bilinear", align_corners=False).squeeze(1)
    return up.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline + submission writer + main — kept compact
# ─────────────────────────────────────────────────────────────────────────────
def _build_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_one(r, transform, load_masks, input_size):
    with Image.open(r.path) as im:
        x = transform(im.convert("RGB"))
    if load_masks and r.mask_path is not None:
        with Image.open(r.mask_path) as mm:
            mm = mm.convert("L").resize((input_size, input_size), Image.NEAREST)
            m = (np.asarray(mm) > 127).astype(np.float32)
    else:
        m = np.zeros((input_size, input_size), dtype=np.float32)
    return x, m


def _score_records_standard(teacher, student, ae, records, stats, cfg,
                                load_masks):
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores, gts = {}, {}
    for x, masks, idxs in loader:
        sm = score_batch_standard(teacher, student, ae, x, stats, cfg).numpy()
        m_np = masks.numpy()
        for b in range(sm.shape[0]):
            scores[int(idxs[b])] = sm[b]
            gts[int(idxs[b])] = m_np[b]
    return scores, gts


def _score_records_by_sample(teacher, student, ae, records, stats, cfg,
                                  load_masks):
    by_sample = defaultdict(list)
    for idx, r in enumerate(records):
        sid = r.sample_id or r.path.stem
        by_sample[sid].append((idx, r))
    transform = _build_transform(cfg.input_size)
    scores, gts = {}, {}
    for sid, items in by_sample.items():
        imgs, masks_np = [], []
        for _idx, r in items:
            x, m = _load_one(r, transform, load_masks, cfg.input_size)
            imgs.append(x); masks_np.append(m)
        x_batch = torch.stack(imgs)
        sm = score_sample_with_siblings(teacher, student, ae, x_batch,
                                          stats, cfg).numpy()
        for k, (idx, _r) in enumerate(items):
            scores[idx] = sm[k]; gts[idx] = masks_np[k]
    return scores, gts


def run_one_class(cls, records_all, teacher, cfg, run_dir, device,
                   local_saver=None):
    hr(f"CLASS {cls}", "─")
    t_start = time.time()
    train_good = [r for r in records_all if r.cls == cls and r.split == "train_good"]
    train_anom = [r for r in records_all if r.cls == cls and r.split == "train_anomaly"]
    test       = [r for r in records_all if r.cls == cls and r.split == "test"]
    print(f"  train_good={len(train_good)}  train_anomaly={len(train_anom)}  test={len(test)}")
    if not train_good:
        return {"class": cls, "class_mean_ap": float("nan"),
                "eval_rows": [], "test_results": [], "elapsed_min": 0.0}

    C_T = teacher.out_channels
    student = StudentPDN(out_channels=2 * C_T).to(device)
    ae      = Autoencoder(out_channels=C_T).to(device)
    train_efficientad(teacher, student, ae, train_good, cfg, device)
    print(f"    [{now_hms()}] computing norm stats ...")
    stats = compute_norm_stats(teacher, student, ae, train_good, cfg, device)

    score_fn = (_score_records_by_sample if cfg.multiview == "sibling-bank"
                else _score_records_standard)

    eval_rows = []; class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation multiview={cfg.multiview} tta={cfg.tta}")
        scores, gts = score_fn(teacher, student, ae, train_anom, stats,
                                 cfg, load_masks=True)
        by_anom = defaultdict(list)
        for r_idx, sm in scores.items():
            r = train_anom[r_idx]
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            ap = pixel_average_precision(sm_smooth, gts[r_idx])
            by_anom[r.anomaly_type or "?"].append(ap)
            if local_saver is not None:
                local_saver.add(cls=cls,
                                 anomaly_type=r.anomaly_type or "unknown",
                                 view_idx=int(r_idx),
                                 score_map=sm_smooth, gt_mask=gts[r_idx],
                                 image_path=r.path)
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
        sub(f"scoring {len(test)} test images multiview={cfg.multiview} tta={cfg.tta}")
        scores, _ = score_fn(teacher, student, ae, test, stats, cfg,
                                load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            test_results.append((test[r_idx], maybe_resize_to_submission(sm_smooth)))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del student, ae
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


@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    teacher_backbone: str = "resnet18"
    teacher_layer: int | None = None
    input_size: int = 256
    epochs: int = 200
    total_iters: int | None = 2500
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    hard_mining_pct: float = 0.10
    amp: bool = True
    num_workers: int = 8
    score_batch_size: int = 32
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    multiview: str = "none"
    mv_alpha: float = 0.5
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_checkpoints: bool = False
    zip_submission: bool = True
    run_tag: str = ""
    device: torch.device | None = None


def make_run_id(cfg):
    fp = json.dumps({
        "method": "efficientad", "v": 2,
        "teacher": cfg.teacher_backbone, "teacher_layer": cfg.teacher_layer,
        "input_size": cfg.input_size, "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size, "lr": cfg.lr,
        "hard_mining_pct": cfg.hard_mining_pct,
        "multiview": cfg.multiview, "mv_alpha": cfg.mv_alpha,
        "tta": cfg.tta, "smooth_sigma": cfg.smooth_sigma, "seed": cfg.seed,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    budget = (f"it{cfg.total_iters}" if cfg.total_iters else f"e{cfg.epochs}")
    mv_tag = "noMV" if cfg.multiview == "none" else f"MV-a{cfg.mv_alpha:.2f}"
    tb = BACKBONE_SHORT.get(cfg.teacher_backbone, cfg.teacher_backbone)
    bits = (f"{stamp}_effad_{tb}"
            + (f"_L{cfg.teacher_layer}" if cfg.teacher_layer is not None else "")
            + f"_in{cfg.input_size}_{budget}_bs{cfg.batch_size}_{mv_tag}")
    if cfg.tta != "none": bits += f"_tta-{cfg.tta}"
    if cfg.run_tag: bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--teacher-backbone", default="resnet18",
                    help="resnet18 (original); any DINOv2/v3 ViT or ConvNeXt name.")
    ap.add_argument("--teacher-layer", type=int, default=None,
                    help="ViT block / ConvNeXt stage index for the teacher. "
                         "Default: ViT → 3rd-from-last block, ConvNeXt → stage 0.")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hard-mining-pct", type=float, default=0.10)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--score-batch-size", type=int, default=32)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip"])
    ap.add_argument("--multiview", default="none",
                    choices=["none", "sibling-bank"])
    ap.add_argument("--mv-alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    # Patch-size sanity for ViT teachers
    if is_dino_backbone(args.teacher_backbone):
        from dinov3_loader import (DINOV2_SPECS as _V2,
                                       DINOV3_VIT_SPECS as _V3V,
                                       DINOV3_CONVNEXT_SPECS as _V3C)
        if args.teacher_backbone in _V2:
            patch = _V2[args.teacher_backbone]["patch"]
        elif args.teacher_backbone in _V3V:
            patch = _V3V[args.teacher_backbone]["patch"]
        else:
            patch = _V3C[args.teacher_backbone]["patch_eff"]
        if args.input_size % patch != 0:
            raise SystemExit(
                f"[FATAL] --input-size {args.input_size} not multiple of "
                f"{patch} (required by teacher {args.teacher_backbone}).")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        teacher_backbone=args.teacher_backbone,
        teacher_layer=args.teacher_layer,
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
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"EFFICIENTAD v2 — RUN {run_id}", "█")
        print(f"  teacher_backbone : {cfg.teacher_backbone}")
        print(f"  teacher_layer    : {cfg.teacher_layer}")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  total_iters      : {cfg.total_iters}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / wd          : {cfg.lr} / {cfg.weight_decay}")
        print(f"  hard_mining_pct  : {cfg.hard_mining_pct}")
        print(f"  multiview        : {cfg.multiview}  alpha={cfg.mv_alpha}")
        print(f"  device           : {device}")
        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (str(v) if isinstance(v, (Path, torch.device)) else v)
                       for k, v in asdict(cfg).items()}, f, indent=2,
                       default=str)

        teacher = build_teacher(cfg.teacher_backbone, cfg.teacher_layer).to(device)
        teacher.eval()
        C_T = teacher.out_channels
        print(f"  teacher built: out_channels={C_T}, "
              f"target_stride={teacher.target_stride}")

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
            res = run_one_class(cls, records, teacher, cfg, run_dir, device,
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
        tb_short = BACKBONE_SHORT.get(cfg.teacher_backbone, cfg.teacher_backbone)
        row = {
            "run_id": run_id, "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": f"EFFICIENTAD_{tb_short.upper()}",
            "feature_layers": f"{tb_short}_{cfg.teacher_layer}"
                               if cfg.teacher_layer is not None
                               else tb_short,
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
            "notes": (f"efficientad v2 teacher={cfg.teacher_backbone}"
                      f"(L{cfg.teacher_layer}) multiview={cfg.multiview} "
                      f"mv_alpha={cfg.mv_alpha} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size} hm{cfg.hard_mining_pct}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()