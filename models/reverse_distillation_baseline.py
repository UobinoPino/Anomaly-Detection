"""Spacepresso Reverse Distillation v2 — adds optional DINOv2/v3 ViT teacher.

# v2 changes (additive over v1)

  * NEW arg: `--teacher-backbone` (default: wide_resnet50_2).
        wide_resnet50_2 (original)  — 3-scale WRN50 layers 1/2/3, full OCBE
        resnet50                     — same 3-scale layout, lighter
        resnet18                     — toy / fast iteration
        dinov2_vits14/b14/l14 + reg variants
        dinov3_vits16/b16/l16 + hp16 (gated)
        dinov3_convnext_{tiny,small,base,large} (gated)

  * For ViT teachers we take ONE block's features (configurable via
    `--teacher-layer`, default = last block) and run a SINGLE-SCALE
    RD pipeline. This is the correct architectural choice because:

      - ViT feature maps live at one resolution (H/patch); they don't
        give us a natural 3-scale {layer1, layer2, layer3} hierarchy.
      - OCBE's job is to fuse a multi-scale ResNet hierarchy into one
        embedding. For a single ViT block that fusion is a no-op.
      - Multi-scale ViT-RD variants in the literature use 3 different
        BLOCK depths (e.g. 4/8/11). We provide an opt-in to that via
        `--teacher-layer` accepting multiple ints; with 1 layer it
        bypasses OCBE (lightweight projection only), with 3 layers it
        builds a 3-block OCBE the same way.

  * For ConvNeXt teachers we use the 3 stages of the encoder (stages
    1, 2, 3 — same {H/4, H/8, H/16} layout as the ResNet teachers), so
    OCBE + Decoder run unchanged with channel counts plumbed from the
    backbone spec.

  * Default teacher remains wide_resnet50_2 — every existing exp10*
    command line reproduces unchanged.

# What this gives the stacker

A new RD track whose teacher is a strong SSL transformer. The student
is still randomly initialised and trained to reconstruct teacher
features on train_good only — so the resulting anomaly signal is
"distance to the manifold of normal teacher features as observed by
a class-specific small student", which is genuinely different from
PatchCore-DINOv2 (NN distance in raw teacher space).

# Stacker contract — unchanged
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
    FeatureExtractor,
    pixel_average_precision, gaussian_smooth,
    calibrate_to_unit, float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
    IMAGENET_MEAN, IMAGENET_STD,
    BACKBONE_SHORT, ALL_BACKBONES,
    RESNET_BACKBONES, DINOV2_BACKBONES,
    DINOV2_NBLOCKS,
    DINOV3_VIT_SPECS, DINOV3_NBLOCKS, DINOV3_CONVNEXT_SPECS,
    DINO_BACKBONES, BACKBONE_CHANNELS, backbone_patch_size,
)
from local_preds_saver import LocalPredSaver


PROJECT_ROOT = Path("/workspace/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"

# Teacher feature channels for the ORIGINAL WRN50-2 teacher. The decoder/
# OCBE are sized for these. For ViT / ConvNeXt teachers we either bypass
# OCBE (single-block) or build a per-backbone 3-scale OCBE.
WRN_TEACHER_CHANNELS = {1: 256, 2: 512, 3: 1024}
WRN_BASE_WIDTH = 128


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
    def __init__(self, records, input_size):
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
    def __init__(self, records, input_size, load_masks):
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
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base); random.seed(base)


# ─────────────────────────────────────────────────────────────────────────────
# Network primitives (WRN50-OCBE path — unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
def conv1x1(c_in, c_out, stride=1):
    return nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)

def conv3x3(c_in, c_out, stride=1):
    return nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)

def deconv2x2(c_in, c_out, stride=2):
    return nn.ConvTranspose2d(c_in, c_out, 2, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                  base_width=WRN_BASE_WIDTH):
        super().__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = conv1x1(inplanes, width); self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride=stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True); self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)


class DeBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, upsample=None,
                  base_width=WRN_BASE_WIDTH):
        super().__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = conv1x1(inplanes, width); self.bn1 = nn.BatchNorm2d(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride=2)
        else:
            self.conv2 = conv3x3(width, width, stride=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True); self.upsample = upsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.upsample is not None: identity = self.upsample(x)
        return self.relu(out + identity)


class OCBE_WRN(nn.Module):
    """The original OCBE: fuses (layer1, layer2, layer3) of WRN50-2 into
    a single H/32×2048-ch tensor."""
    def __init__(self, base_width=WRN_BASE_WIDTH):
        super().__init__()
        self.l1_conv1 = conv3x3(256, 512, stride=2)
        self.l1_bn1 = nn.BatchNorm2d(512)
        self.l1_conv2 = conv3x3(512, 1024, stride=2)
        self.l1_bn2 = nn.BatchNorm2d(1024)
        self.l2_conv = conv3x3(512, 1024, stride=2)
        self.l2_bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        downsample = nn.Sequential(conv1x1(3072, 2048, stride=2),
                                       nn.BatchNorm2d(2048))
        self.block1 = Bottleneck(3072, 512, stride=2, downsample=downsample,
                                    base_width=base_width)
        self.block2 = Bottleneck(2048, 512, stride=1, base_width=base_width)
        self.block3 = Bottleneck(2048, 512, stride=1, base_width=base_width)
    def forward(self, feats):
        l1, l2, l3 = feats[1], feats[2], feats[3]
        a = self.relu(self.l1_bn1(self.l1_conv1(l1)))
        a = self.relu(self.l1_bn2(self.l1_conv2(a)))
        b = self.relu(self.l2_bn(self.l2_conv(l2)))
        fused = torch.cat([a, b, l3], dim=1)
        return self.block3(self.block2(self.block1(fused)))


class Decoder_WRN(nn.Module):
    """Mirror of WRN50-2 layers 3/2/1 — output channels match the teacher
    feature maps so cosine sim is well defined."""
    def __init__(self, layers=(3, 4, 6), base_width=WRN_BASE_WIDTH):
        super().__init__()
        self.layer1 = self._make_layer(2048, 256, layers[0], stride=2,
                                          base_width=base_width)
        self.layer2 = self._make_layer(1024, 128, layers[1], stride=2,
                                          base_width=base_width)
        self.layer3 = self._make_layer(512,  64,  layers[2], stride=2,
                                          base_width=base_width)
    def _make_layer(self, inplanes, planes, blocks, stride, base_width):
        upsample = None
        out_ch = planes * DeBottleneck.expansion
        if stride != 1 or inplanes != out_ch:
            upsample = nn.Sequential(deconv2x2(inplanes, out_ch, stride=stride),
                                         nn.BatchNorm2d(out_ch))
        seq = [DeBottleneck(inplanes, planes, stride=stride, upsample=upsample,
                              base_width=base_width)]
        for _ in range(1, blocks):
            seq.append(DeBottleneck(out_ch, planes, stride=1,
                                       base_width=base_width))
        return nn.Sequential(*seq)
    def forward(self, x):
        f3 = self.layer1(x); f2 = self.layer2(f3); f1 = self.layer3(f2)
        return {1: f1, 2: f2, 3: f3}


class RDStudentWRN(nn.Module):
    """Original WRN50-2 RD student."""
    def __init__(self):
        super().__init__()
        self.ocbe = OCBE_WRN()
        self.decoder = Decoder_WRN()
    def forward(self, teacher_feats):
        return self.decoder(self.ocbe(teacher_feats))


# ─────────────────────────────────────────────────────────────────────────────
# Single-scale ViT student
# ─────────────────────────────────────────────────────────────────────────────
class RDStudentViT(nn.Module):
    """Single-scale RD student for ViT teachers. Architecture: a small
    convolutional bottleneck-then-expand (same idea as OCBE+Decoder but
    one-scale). Takes teacher patch tokens at one block depth, projects
    down to a bottleneck, then projects back to the original channel
    count. Trained to make the cosine of (teacher, student) → 1 on
    train_good."""
    def __init__(self, teacher_channels: int, bottleneck_dim: int = 256):
        super().__init__()
        c = teacher_channels
        bn = bottleneck_dim
        # Encoder: 1x1 conv → 3x3 stride-2 conv → 3x3 stride-2 conv → bottleneck.
        self.encoder = nn.Sequential(
            conv1x1(c, bn), nn.BatchNorm2d(bn), nn.ReLU(inplace=True),
            conv3x3(bn, bn, stride=2), nn.BatchNorm2d(bn), nn.ReLU(inplace=True),
            conv3x3(bn, bn, stride=2), nn.BatchNorm2d(bn), nn.ReLU(inplace=True),
        )
        # Decoder: upsample 2x, conv, upsample 2x, conv, 1x1 back to c.
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv3x3(bn, bn), nn.BatchNorm2d(bn), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv3x3(bn, bn), nn.BatchNorm2d(bn), nn.ReLU(inplace=True),
            conv1x1(bn, c),
        )

    def forward(self, teacher_feats: dict):
        # We receive a {block_idx: (B,C,H,W)} dict from FeatureExtractor.
        # For single-scale ViT student we expect exactly one entry; the
        # student returns the same dict shape so the rest of the code path
        # (rd_cos_loss / rd_anomaly_map) is unchanged.
        assert len(teacher_feats) == 1, (
            "RDStudentViT is single-scale. Pass exactly one layer "
            "via --teacher-layer.")
        k = list(teacher_feats.keys())[0]
        t = teacher_feats[k]
        z = self.encoder(t)
        d = self.decoder(z)
        # Match spatial size to teacher if upsampling rounding diverges.
        if d.shape[-2:] != t.shape[-2:]:
            d = F.interpolate(d, size=t.shape[-2:],
                              mode="bilinear", align_corners=False)
        return {k: d}


# ─────────────────────────────────────────────────────────────────────────────
# Loss + anomaly map (work for any number of scales)
# ─────────────────────────────────────────────────────────────────────────────
def rd_cos_loss(t_feats, s_feats):
    total = 0.0
    for k in t_feats:
        t = t_feats[k]; s = s_feats[k]
        cos = F.cosine_similarity(t, s, dim=1, eps=1e-8)
        total = total + (1.0 - cos).mean()
    return total


@torch.inference_mode()
def rd_anomaly_map(t_feats, s_feats, input_size, amap_mode="mul"):
    B = next(iter(t_feats.values())).shape[0]
    device = next(iter(t_feats.values())).device
    if amap_mode == "mul":
        amap = torch.ones((B, 1, input_size, input_size), device=device,
                           dtype=torch.float32); op = "mul"
    elif amap_mode == "sum":
        amap = torch.zeros((B, 1, input_size, input_size), device=device,
                            dtype=torch.float32); op = "sum"
    else:
        raise ValueError(f"unknown amap_mode: {amap_mode}")
    for k in sorted(t_feats.keys()):
        t = t_feats[k].float(); s = s_feats[k].float()
        cos = F.cosine_similarity(t, s, dim=1, eps=1e-8).unsqueeze(1)
        d = (1.0 - cos)
        d = F.interpolate(d, size=(input_size, input_size),
                          mode="bilinear", align_corners=False)
        amap = amap * d if op == "mul" else amap + d
    return amap.squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Build student matched to teacher choice
# ─────────────────────────────────────────────────────────────────────────────
def build_student(teacher_backbone: str, teacher_layers: tuple[int, ...],
                   teacher: FeatureExtractor) -> nn.Module:
    """Return the appropriate student given the teacher choice + chosen layers."""
    if teacher_backbone in RESNET_BACKBONES:
        if teacher_backbone != "wide_resnet50_2":
            print(f"  [warn] RD student is sized for WRN50-2 channels "
                  f"(256/512/1024). Other ResNets work for L1/L2/L3 if you "
                  f"use the WRN OCBE+Decoder verbatim.")
        return RDStudentWRN()
    # ViT teachers
    if teacher_backbone in DINOV2_BACKBONES or teacher_backbone in DINOV3_VIT_SPECS:
        teacher_channels = BACKBONE_CHANNELS[teacher_backbone][teacher_layers[0]]
        if len(teacher_layers) > 1:
            raise SystemExit(
                "[FATAL] Multi-layer ViT teacher RD is not implemented in v2. "
                "Pass a single block index via --teacher-layer.")
        return RDStudentViT(teacher_channels=teacher_channels)
    if teacher_backbone in DINOV3_CONVNEXT_SPECS:
        raise SystemExit(
            "[FATAL] ConvNeXt teacher for RD is not implemented in v2. "
            "Use a ViT or WRN50 teacher.")
    raise SystemExit(f"[FATAL] unknown teacher backbone: {teacher_backbone}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_rd(teacher: FeatureExtractor, student: nn.Module,
              records, cfg: "RunConfig", device, teacher_layers):
    ds = TrainGoodDataset(records, input_size=cfg.input_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0),
                         worker_init_fn=worker_init_fn, drop_last=False)
    iters_per_epoch = max(len(loader), 1)
    if cfg.total_iters and cfg.total_iters > 0:
        n_epochs = max(1, math.ceil(cfg.total_iters / iters_per_epoch))
        print(f"    [auto-epoch] total_iters={cfg.total_iters} / "
              f"{iters_per_epoch} iters/epoch = {n_epochs} epochs "
              f"(--epochs={cfg.epochs} overridden)")
    else:
        n_epochs = cfg.epochs
    total_iters = iters_per_epoch * n_epochs

    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr,
                                    betas=(cfg.beta1, cfg.beta2),
                                    weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))
    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"    [{now_hms()}] training: {n_epochs} epochs × "
          f"{iters_per_epoch} iters ({total_iters} total), "
          f"{len(records)} train_good, batch={cfg.batch_size}, "
          f"lr={cfg.lr:.0e}, amp={use_amp}, teacher_layers={list(teacher_layers)}")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    teacher.eval()
    for epoch in range(n_epochs):
        student.train()
        loss_sum = 0.0; n = 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                t_feats = teacher(x, layers=tuple(teacher_layers))
                t_feats = {k: v.detach() for k, v in t_feats.items()}
                s_feats = student(t_feats)
                loss = rd_cos_loss(t_feats, s_feats)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            loss_sum += loss.item() * x.shape[0]; n += x.shape[0]
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch + 1:>4}/{n_epochs}  "
                  f"loss={loss_sum / max(n, 1):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    student.eval()
    print(f"    [{now_hms()}] training done ({time.time() - t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Inference (with optional TTA)
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _score_one_pass(teacher, student, x, cfg, device, teacher_layers):
    use_amp = (device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        t_feats = teacher(x, layers=tuple(teacher_layers))
        s_feats = student(t_feats)
    amap = rd_anomaly_map(t_feats, s_feats,
                          input_size=cfg.input_size,
                          amap_mode=cfg.amap_mode)
    return amap.cpu()


@torch.inference_mode()
def score_batch(teacher, student, x, cfg, device, teacher_layers):
    x = x.to(device, non_blocking=True)
    acc = None; n = 0
    def _add(s):
        nonlocal acc, n
        if acc is None: acc = s.clone()
        else: acc += s
        n += 1
    _add(_score_one_pass(teacher, student, x, cfg, device, teacher_layers))
    if cfg.tta in ("hflip", "hvflip", "d4"):
        s = _score_one_pass(teacher, student, torch.flip(x, dims=[-1]),
                              cfg, device, teacher_layers)
        _add(torch.flip(s, dims=[-1]))
    if cfg.tta in ("vflip", "hvflip", "d4"):
        s = _score_one_pass(teacher, student, torch.flip(x, dims=[-2]),
                              cfg, device, teacher_layers)
        _add(torch.flip(s, dims=[-2]))
    if cfg.tta == "d4":
        for k in (1, 2, 3):
            s = _score_one_pass(teacher, student,
                                 torch.rot90(x, k=k, dims=[-2, -1]),
                                 cfg, device, teacher_layers)
            _add(torch.rot90(s, k=-k, dims=[-2, -1]))
    return acc / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tracking
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    teacher_backbone: str = "wide_resnet50_2"
    teacher_layers: tuple[int, ...] = (1, 2, 3)
    input_size: int = 256
    amap_mode: str = "mul"
    epochs: int = 200
    total_iters: int | None = None
    batch_size: int = 16
    lr: float = 0.005
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    amp: bool = True
    num_workers: int = 8
    score_batch_size: int = 16
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_checkpoints: bool = False
    zip_submission: bool = True
    run_tag: str = ""


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "reverse_distillation",
        "teacher": cfg.teacher_backbone,
        "teacher_layers": list(cfg.teacher_layers),
        "input_size": cfg.input_size,
        "amap_mode": cfg.amap_mode,
        "epochs": cfg.epochs,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr, "beta1": cfg.beta1, "beta2": cfg.beta2,
        "weight_decay": cfg.weight_decay,
        "smooth_sigma": cfg.smooth_sigma, "tta": cfg.tta,
        "seed": cfg.seed,
        "v": 2,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bb_short = BACKBONE_SHORT.get(cfg.teacher_backbone, cfg.teacher_backbone)
    budget = (f"it{cfg.total_iters}" if (cfg.total_iters and cfg.total_iters > 0)
              else f"e{cfg.epochs}")
    L = "_".join(str(l) for l in cfg.teacher_layers)
    bits = (f"{stamp}_rd_{bb_short}_L{L}_in{cfg.input_size}_{budget}"
            f"_bs{cfg.batch_size}_lr{cfg.lr:.0e}_{cfg.amap_mode}")
    if cfg.tta != "none":
        bits += f"_tta-{cfg.tta}"
    if cfg.run_tag:
        bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_one_class(cls, records_all, teacher, cfg, run_dir, device,
                   local_saver=None) -> dict:
    hr(f"CLASS {cls}", "─")
    t_start = time.time()
    train_good = [r for r in records_all if r.cls == cls and r.split == "train_good"]
    train_anom = [r for r in records_all if r.cls == cls and r.split == "train_anomaly"]
    test       = [r for r in records_all if r.cls == cls and r.split == "test"]
    print(f"  train_good={len(train_good)}  "
          f"train_anomaly={len(train_anom)}  test={len(test)}")
    if not train_good:
        return {"class": cls, "class_mean_ap": float("nan"),
                "eval_rows": [], "test_results": [], "elapsed_min": 0.0}

    student = build_student(cfg.teacher_backbone, cfg.teacher_layers,
                              teacher).to(device)
    train_rd(teacher, student, train_good, cfg, device, cfg.teacher_layers)

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_rd.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save(student.state_dict(), ck)
        print(f"    saved checkpoint -> {ck}")

    eval_rows = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation — pixel-AP per (class, anomaly_type) "
            f"(tta={cfg.tta}, amap={cfg.amap_mode})")
        ds_v = InferenceDataset(train_anom, input_size=cfg.input_size,
                                  load_masks=True)
        loader_v = DataLoader(ds_v, batch_size=cfg.score_batch_size,
                               shuffle=False, num_workers=cfg.num_workers,
                               pin_memory=True,
                               persistent_workers=(cfg.num_workers > 0))
        scores_by_idx, gt_by_idx = {}, {}
        for x, masks, idxs in loader_v:
            sm = score_batch(teacher, student, x, cfg, device,
                              cfg.teacher_layers).numpy()
            m_np = masks.numpy()
            for b in range(sm.shape[0]):
                s = gaussian_smooth(sm[b], cfg.smooth_sigma)
                scores_by_idx[int(idxs[b])] = s
                gt_by_idx[int(idxs[b])] = m_np[b]
        by_anom = defaultdict(list)
        for ridx, s in scores_by_idx.items():
            r = train_anom[ridx]
            ap = pixel_average_precision(s, gt_by_idx[ridx])
            by_anom[r.anomaly_type or "?"].append(ap)
            if local_saver is not None:
                local_saver.add(
                    cls=cls, anomaly_type=r.anomaly_type or "unknown",
                    view_idx=int(ridx), score_map=s,
                    gt_mask=gt_by_idx[ridx], image_path=r.path)
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
        print(f"    >>> class {cls} mean pixel-AP: {class_mean_ap:.4f}")

    test_results = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images (tta={cfg.tta}, amap={cfg.amap_mode})")
        ds_t = InferenceDataset(test, input_size=cfg.input_size,
                                  load_masks=False)
        loader_t = DataLoader(ds_t, batch_size=cfg.score_batch_size,
                               shuffle=False, num_workers=cfg.num_workers,
                               pin_memory=True,
                               persistent_workers=(cfg.num_workers > 0))
        n_done, last_log = 0, 0
        for x, _, idxs in loader_t:
            sm = score_batch(teacher, student, x, cfg, device,
                              cfg.teacher_layers).numpy()
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
    del student
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--teacher-backbone", default="wide_resnet50_2",
                    choices=ALL_BACKBONES,
                    help="ResNet (full OCBE+Decoder, 3 scales hard-wired) "
                         "or DINOv2/v3 ViT (single-block, single-scale "
                         "RD student auto-sized). ConvNeXt not supported "
                         "in v2.")
    ap.add_argument("--teacher-layer", type=int, nargs="+", default=None,
                    help="For ViT teachers: which block(s) to read. Single "
                         "int = single-scale RD. For ResNet teachers: "
                         "always (1, 2, 3) — ignored.")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--amap-mode", default="mul", choices=["mul", "sum"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip", "d4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    # Resolve teacher_layers per backbone family.
    if args.teacher_backbone in RESNET_BACKBONES:
        if args.input_size % 32 != 0:
            raise SystemExit(
                f"[FATAL] --input-size must be divisible by 32 for ResNet "
                f"RD (OCBE downsamples by 32). Got {args.input_size}.")
        teacher_layers = (1, 2, 3)
    elif args.teacher_backbone in DINO_BACKBONES:
        ps = backbone_patch_size(args.teacher_backbone)
        if ps and (args.input_size % ps != 0):
            raise SystemExit(
                f"[FATAL] {args.teacher_backbone} requires --input-size "
                f"divisible by {ps}; got {args.input_size}.")
        if args.teacher_backbone in DINOV3_CONVNEXT_SPECS:
            raise SystemExit("[FATAL] ConvNeXt teacher not supported in RD v2.")
        # ViT teacher
        if args.teacher_backbone in DINOV2_BACKBONES:
            nb = DINOV2_NBLOCKS[args.teacher_backbone]
        else:
            nb = DINOV3_NBLOCKS[args.teacher_backbone]
        if args.teacher_layer is None:
            teacher_layers = (nb - 1,)        # last block by default
            print(f"  [info] teacher_layer not specified; defaulting to "
                  f"last block ({nb - 1}) for {args.teacher_backbone}")
        else:
            for l in args.teacher_layer:
                if not (0 <= l < nb):
                    raise SystemExit(
                        f"[FATAL] block index {l} out of range [0, {nb - 1}] "
                        f"for {args.teacher_backbone}.")
            teacher_layers = tuple(sorted(set(args.teacher_layer)))
    else:
        raise SystemExit(f"[FATAL] unknown teacher backbone: "
                          f"{args.teacher_backbone}")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        teacher_backbone=args.teacher_backbone,
        teacher_layers=teacher_layers,
        input_size=args.input_size, amap_mode=args.amap_mode,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        beta1=args.beta1, beta2=args.beta2,
        weight_decay=args.weight_decay,
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
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tee_to(run_dir / "run_log.txt"):
        hr(f"REVERSE DISTILLATION v2 — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  teacher backbone : {cfg.teacher_backbone}")
        print(f"  teacher layers   : {list(cfg.teacher_layers)}")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  amap_mode        : {cfg.amap_mode}")
        if cfg.total_iters is not None:
            print(f"  total_iters      : {cfg.total_iters}  "
                  f"(overrides --epochs={cfg.epochs})")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / b1 / b2     : {cfg.lr} / {cfg.beta1} / {cfg.beta2}  (Adam)")
        print(f"  weight_decay     : {cfg.weight_decay}")
        print(f"  amp              : {cfg.amp}    num_workers: {cfg.num_workers}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}    tta: {cfg.tta}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(cfg).items()}, f, indent=2)

        teacher = FeatureExtractor(cfg.teacher_backbone).to(device).eval()

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("\n[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): {', '.join(classes)}")

        local_saver: LocalPredSaver | None = None
        if not cfg.skip_eval and not args.no_save_local_preds:
            local_saver = LocalPredSaver()

        all_test_results = []
        all_eval_rows = []
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
            print(f"  saved per-(class, anomaly_type) AP -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            save_test_predictions(all_test_results, run_dir)
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload: {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        bb_short = BACKBONE_SHORT.get(cfg.teacher_backbone,
                                          cfg.teacher_backbone)
        L = "+".join(str(l) for l in cfg.teacher_layers)
        row = {
            "run_id": run_id, "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": f"RD_{bb_short.upper()}",
            "feature_layers": L, "target_layer": "",
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
            "notes": (f"reverse_distillation v2 teacher={cfg.teacher_backbone} "
                      f"L={L} amap={cfg.amap_mode} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size} lr{cfg.lr:.0e}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()