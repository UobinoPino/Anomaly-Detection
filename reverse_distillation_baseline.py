"""Spacepresso Reverse Distillation baseline — step 10 of the roadmap.

Implements "Anomaly Detection via Reverse Distillation from One-Class
Embedding" (Deng & Li, CVPR 2022) adapted to Spacepresso with the
same logging / submission / ablation_master / local_eval conventions
as patchcore_baseline_v2.py and cutpaste_baseline.py.

Two teacher backbones are supported, with a separate OCBE/Decoder pair
each (they share everything else — training loop, scoring, TTA, RLE):

# CNN teacher  (--backbone wide_resnet50_2  or  resnet50)

  Feature pyramid (hierarchical, fixed at layers [1, 2, 3]):
        layer1 [B, 256,  H/4,  W/4]
        layer2 [B, 512,  H/8,  W/8]
        layer3 [B, 1024, H/16, W/16]

  CNN_OCBE:
        layer1 --(stride-2 3x3 conv)x2--> [B, 1024, H/16, W/16]
        layer2 --(stride-2 3x3 conv)--->  [B, 1024, H/16, W/16]
        concat with layer3                [B, 3072, H/16, W/16]
        --(3 Bottleneck blocks, first stride 2)--> [B, 2048, H/32, W/32]

  CNN_Decoder: 3 stages of inverted Bottleneck blocks (ConvTranspose2d
  for upsampling), output channel/spatial shapes mirror the teacher's
  layer3/2/1 exactly so per-pixel cosine similarity is well defined.

# ViT teacher  (--backbone dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14)

  DINOv2 has NO spatial hierarchy — every transformer block outputs the
  same spatial grid S × S = (input_size / 14)^2 with the same embed_dim.
  So "multi-scale" comes from sampling K different *depths*, not
  resolutions. Default block indices [3, 6, 9, 11] match exp7 (PatchCore
  + DINOv2), so RD with the same blocks gives a directly comparable
  fusion partner with a fundamentally different scoring paradigm.

  Per-block teacher map: [B, D, S, S]   where D ∈ {384, 768, 1024} for
                                              S/B/L respectively.

  ViT_OCBE:
        concat K maps along channels:    [B, K*D,  S,   S]
        proj 1x1 + BN + ReLU:            [B, 1024, S,   S]
        Bottleneck block stride 2:       [B, 2048, S/2, S/2]
        2x Bottleneck refine blocks:     [B, 2048, S/2, S/2]   (= the bn)

  ViT_Decoder:
        2x Bottleneck refine blocks:     [B, 2048, S/2, S/2]
        1x1 reduce + BN + ReLU:          [B, 1024, S/2, S/2]
        bilinear upsample to S:          [B, 1024, S,   S]
        2x Bottleneck refine blocks:     [B, 1024, S,   S]
        final 1x1 conv + BN  (no ReLU,   [B, K*D,  S,   S]
            since ViT features can be negative)
        chunk along channels into K maps each [B, D, S, S].

  S is allowed to be odd (e.g. S=37 at input 518): stride-2 3x3 convs
  with padding 1 map odd S to (S+1)//2, and bilinear F.interpolate
  handles arbitrary target sizes on the way back up.

  - Train loss (only on train/good):
        L = sum_k mean_over_BHW( 1 - cos_sim(t_k, s_k, dim=channels) )

  - Anomaly map (test):
        amap_k = 1 - cos_sim(t_k, s_k)
        amap   = combine_k( F.interpolate(amap_k, input_size, bilinear) )
    where combine is 'mul' (paper) or 'sum'. Result is then Gaussian
    smoothed and global-percentile-calibrated to [0, 1] for q8rle, the
    same way the other baselines do it.

# Memory/speed on an L4 24 GB

  Input 256, batch 16, fp16-AMP per-class budget at 2500 iters:
    activations  ~120 MB, weights+optimizer ~700 MB, gradient ~400 MB
    -> 1.2 GB total. Comfortably fits with room for inference batches.
  At input 384 the activation map ~2.25x but still <3 GB.

  Speed tricks:
    - Teacher is in inference_mode + .eval(), no grad / no BN updates
    - AMP/autocast for training and inference (fp16 forward, fp32 master)
    - DataLoader with persistent workers, pinned memory
    - No memory bank, no coreset -> inference is a single streaming pass
    - TTA (hflip/vflip/hvflip/d4) is fully optional
    - empty_cache() between classes to keep VRAM tidy

# Dependencies on the rest of the project

  Reuses (via import) from patchcore_baseline_v2.py:
    - ImageRecord, scan_dataset                  (dataset scanning)
    - FeatureExtractor                           (frozen WRN50-2 teacher)
    - pixel_average_precision, gaussian_smooth,
      calibrate_to_unit, float_matrix_to_q8rle,
      maybe_resize_to_submission                 (metrics + submission)
    - append_to_ablation_master                  (CSV bookkeeping)
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

# Pull shared utilities from the PatchCore module — must be in same dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patchcore_baseline_v2 import (
    ImageRecord,
    scan_dataset,
    FeatureExtractor,
    pixel_average_precision,
    gaussian_smooth,
    calibrate_to_unit,
    float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/work/u10813429/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"

# Teacher feature channels at layers 1, 2, 3 for WRN50-2 / ResNet50. Both
# share the same Bottleneck block expansion (4) and therefore the same
# output widths, so the CNN OCBE+Decoder pair handles both.
CNN_TEACHER_CHANNELS = {1: 256, 2: 512, 3: 1024}
CNN_BACKBONES = ("wide_resnet50_2", "resnet50")

# DINOv2 ViT teachers. embed_dim and number of transformer blocks differ
# per variant; the ViT OCBE+Decoder pair is parameterised by these.
DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}
DINOV2_NBLOCKS = {
    "dinov2_vits14": 12,
    "dinov2_vitb14": 12,
    "dinov2_vitl14": 24,
}
VIT_BACKBONES = tuple(DINOV2_DIMS.keys())
ALL_BACKBONES = CNN_BACKBONES + VIT_BACKBONES

BACKBONE_SHORT = {
    "wide_resnet50_2": "wrn50",
    "resnet50":        "rn50",
    "dinov2_vits14":   "dnv2s14",
    "dinov2_vitb14":   "dnv2b14",
    "dinov2_vitl14":   "dnv2l14",
}

# Bottleneck "width per planes" multiplier. WRN50-2 uses 128, every other
# supported backbone uses the standard 64. This only affects the *student*
# blocks (OCBE/Decoder): we let the CNN student inherit WRN50's width
# (it has to reconstruct WRN50 features, so wider is appropriate), and
# keep the ViT student at the standard width (lighter, ~30 M params total).
WRN_BASE_WIDTH = 128
STD_BASE_WIDTH = 64


def teacher_kind(backbone: str) -> str:
    if backbone in CNN_BACKBONES:  return "cnn"
    if backbone in VIT_BACKBONES:  return "vit"
    raise ValueError(f"unsupported RD backbone: {backbone!r} "
                     f"(allowed: {ALL_BACKBONES})")


def default_feature_layers(backbone: str) -> tuple[int, ...]:
    """Sensible per-backbone defaults if --feature-layers is unset."""
    if backbone in CNN_BACKBONES:
        return (1, 2, 3)
    if backbone == "dinov2_vits14" or backbone == "dinov2_vitb14":
        return (3, 6, 9, 11)         # matches exp7's PatchCore+DINOv2 config
    if backbone == "dinov2_vitl14":
        return (5, 11, 17, 23)       # spaced through 24 blocks
    raise ValueError(f"no default feature_layers for backbone {backbone!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger (same idiom as the other baselines)
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
    """train_good images only, resized + ImageNet-normalised. No synthetic
    anomalies — RD trains the student to reconstruct teacher features on
    normal images only."""
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
            im = im.convert("RGB")
            return self.tx(im)


class InferenceDataset(Dataset):
    """Same idiom as the other baselines — returns (image, mask, index).
    `mask` is zeros when `load_masks=False` or the record has no mask."""
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


def worker_init_fn(_worker_id):
    base = torch.initial_seed() % 2 ** 32
    np.random.seed(base); random.seed(base)


# ─────────────────────────────────────────────────────────────────────────────
# Network primitives
# ─────────────────────────────────────────────────────────────────────────────
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1,
                     bias=False)


def deconv2x2(in_planes, out_planes, stride=2):
    """2x2 transposed convolution used for the decoder's upsamples."""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2,
                              stride=stride, bias=False)


class Bottleneck(nn.Module):
    """Standard ResNet bottleneck block used inside the OCBE.

    For WRN50-2 we use base_width=128 so the inner 'width' per planes
    is planes * 2 (matching torchvision's wide_resnet50_2)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=WRN_BASE_WIDTH):
        super().__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride=stride)
        self.bn2   = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class DeBottleneck(nn.Module):
    """Inverted bottleneck: the middle conv becomes ConvTranspose2d when
    we need to upsample. Same expansion (4) and base_width as the
    encoder counterpart so the channel arithmetic mirrors WRN50-2."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None,
                 base_width=WRN_BASE_WIDTH):
        super().__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1   = nn.BatchNorm2d(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride=2)
        else:
            self.conv2 = conv3x3(width, width, stride=1)
        self.bn2   = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.upsample is not None:
            identity = self.upsample(x)
        out = out + identity
        return self.relu(out)


# ─────────────────────────────────────────────────────────────────────────────
# CNN_OCBE — One-Class Bottleneck Embedding for WRN50-2 / ResNet50 teacher
# ─────────────────────────────────────────────────────────────────────────────
class CNN_OCBE(nn.Module):
    """Fuses (layer1, layer2, layer3) of a CNN teacher into a single
    H/32 × 2048-ch tensor. Designed for the WRN50-2 / ResNet50 layer
    channel counts (256 / 512 / 1024).

    Branches:
        layer1 [256, H/4]   -> (3x3 s2) -> [512,  H/8]
                            -> (3x3 s2) -> [1024, H/16]
        layer2 [512, H/8]   -> (3x3 s2) -> [1024, H/16]
        layer3 [1024, H/16] -> identity

    Concat along channels -> [3072, H/16], then 3 Bottleneck blocks
    (first block has stride 2 + 1x1 channel projection downsample) ->
    [2048, H/32].
    """
    def __init__(self, base_width: int = WRN_BASE_WIDTH):
        super().__init__()
        # Branch a: layer1 → 1024 ch @ H/16 (two stride-2 3x3 convs).
        self.l1_conv1 = conv3x3(256, 512, stride=2)
        self.l1_bn1   = nn.BatchNorm2d(512)
        self.l1_conv2 = conv3x3(512, 1024, stride=2)
        self.l1_bn2   = nn.BatchNorm2d(1024)
        # Branch b: layer2 → 1024 ch @ H/16 (one stride-2 3x3 conv).
        self.l2_conv  = conv3x3(512, 1024, stride=2)
        self.l2_bn    = nn.BatchNorm2d(1024)
        self.relu     = nn.ReLU(inplace=True)
        # Final bottleneck: 3072 H/16 → 2048 H/32 via 3 Bottleneck blocks.
        downsample = nn.Sequential(
            conv1x1(3072, 2048, stride=2),
            nn.BatchNorm2d(2048),
        )
        self.block1 = Bottleneck(3072, 512, stride=2,
                                  downsample=downsample, base_width=base_width)
        self.block2 = Bottleneck(2048, 512, stride=1, base_width=base_width)
        self.block3 = Bottleneck(2048, 512, stride=1, base_width=base_width)

    def forward(self, feats: dict[int, torch.Tensor]) -> torch.Tensor:
        l1, l2, l3 = feats[1], feats[2], feats[3]
        a = self.relu(self.l1_bn1(self.l1_conv1(l1)))
        a = self.relu(self.l1_bn2(self.l1_conv2(a)))
        b = self.relu(self.l2_bn(self.l2_conv(l2)))
        fused = torch.cat([a, b, l3], dim=1)
        out = self.block3(self.block2(self.block1(fused)))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CNN_Decoder — inverted ResNet that mirrors layers 1, 2, 3 of the teacher
# ─────────────────────────────────────────────────────────────────────────────
class CNN_Decoder(nn.Module):
    """Input: [B, 2048, H/32, W/32]; outputs three feature maps with the
    same channel/spatial shapes as teacher layer3/2/1 respectively.

    Stage block counts follow WRN50-2 ([3, 4, 6, 3]); the decoder uses
    the first three (3, 4, 6) reversed in spatial direction.
    """
    def __init__(self, layers=(3, 4, 6), base_width=WRN_BASE_WIDTH):
        super().__init__()
        self.layer1 = self._make_layer(2048, 256, layers[0], stride=2,
                                         base_width=base_width)
        self.layer2 = self._make_layer(1024, 128, layers[1], stride=2,
                                         base_width=base_width)
        self.layer3 = self._make_layer(512,  64,  layers[2], stride=2,
                                         base_width=base_width)

    def _make_layer(self, inplanes, planes, blocks, stride,
                     base_width=WRN_BASE_WIDTH):
        upsample = None
        out_ch = planes * DeBottleneck.expansion
        if stride != 1 or inplanes != out_ch:
            upsample = nn.Sequential(
                deconv2x2(inplanes, out_ch, stride=stride),
                nn.BatchNorm2d(out_ch),
            )
        seq = [DeBottleneck(inplanes, planes, stride=stride,
                             upsample=upsample, base_width=base_width)]
        for _ in range(1, blocks):
            seq.append(DeBottleneck(out_ch, planes, stride=1,
                                      base_width=base_width))
        return nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        f3 = self.layer1(x)    # ~ teacher layer3:  1024 ch, H/16
        f2 = self.layer2(f3)   # ~ teacher layer2:   512 ch, H/8
        f1 = self.layer3(f2)   # ~ teacher layer1:   256 ch, H/4
        return {1: f1, 2: f2, 3: f3}


# ─────────────────────────────────────────────────────────────────────────────
# ViT_OCBE — One-Class Bottleneck Embedding for DINOv2 teachers
# ─────────────────────────────────────────────────────────────────────────────
class ViT_OCBE(nn.Module):
    """Fuses K transformer-block features [B, D, S, S] into a single
    H/(14*2) × 2048-ch bottleneck. Parameterised by n_feats (= K) and
    embed_dim (= D, e.g. 384 for ViT-S/14).

    Pipeline:
        cat K maps   [B, K*D,  S,   S]
        proj 1x1     [B, 1024, S,   S]   (1x1 conv + BN + ReLU)
        stride-2 Bottleneck   [B, 2048, S/2, S/2]
        2x Bottleneck refine  [B, 2048, S/2, S/2]
    """
    def __init__(self, n_feats: int, embed_dim: int,
                 base_width: int = STD_BASE_WIDTH):
        super().__init__()
        in_ch = n_feats * embed_dim
        # Project the concat down to 1024 channels at full spatial res.
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        # Spatial bottleneck: 1024 -> 2048 channels, S -> S/2.
        downsample = nn.Sequential(
            conv1x1(1024, 2048, stride=2),
            nn.BatchNorm2d(2048),
        )
        self.down    = Bottleneck(1024, 512, stride=2,
                                    downsample=downsample,
                                    base_width=base_width)
        self.refine1 = Bottleneck(2048, 512, stride=1, base_width=base_width)
        self.refine2 = Bottleneck(2048, 512, stride=1, base_width=base_width)

    def forward(self, feats: dict[int, torch.Tensor]) -> torch.Tensor:
        # Preserve the dict order — chunks are reassembled in this order
        # by ViT_Decoder.forward.
        x = torch.cat(list(feats.values()), dim=1)
        x = self.proj(x)
        x = self.down(x)
        x = self.refine1(x)
        x = self.refine2(x)
        return x                            # [B, 2048, S/2, S/2]


# ─────────────────────────────────────────────────────────────────────────────
# ViT_Decoder — mirrors ViT_OCBE back to K maps at the teacher's grid
# ─────────────────────────────────────────────────────────────────────────────
class ViT_Decoder(nn.Module):
    """Input: [B, 2048, S/2, S/2]. Output: K maps each [B, D, S, S],
    where S = target spatial size passed at forward time (= input/14).

    No ConvTranspose2d — bilinear F.interpolate handles odd S (e.g.
    S=37 at input 518). Final 1x1 conv has BN but no ReLU because
    DINOv2 features (post-LayerNorm) can be negative.
    """
    def __init__(self, n_feats: int, embed_dim: int,
                 base_width: int = STD_BASE_WIDTH):
        super().__init__()
        self.n_feats = n_feats
        self.embed_dim = embed_dim
        # Refine at S/2 before upsampling.
        self.refine_pre1 = Bottleneck(2048, 512, stride=1, base_width=base_width)
        self.refine_pre2 = Bottleneck(2048, 512, stride=1, base_width=base_width)
        # Channel reduce 2048 -> 1024 prior to spatial upsample.
        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        # Refine at full S after upsampling.
        self.refine_post1 = Bottleneck(1024, 256, stride=1, base_width=base_width)
        self.refine_post2 = Bottleneck(1024, 256, stride=1, base_width=base_width)
        # Final projection to K*D channels (BN, no ReLU).
        self.final = nn.Sequential(
            nn.Conv2d(1024, n_feats * embed_dim, 1, bias=False),
            nn.BatchNorm2d(n_feats * embed_dim),
        )

    def forward(self, x: torch.Tensor, target_size: int) -> list[torch.Tensor]:
        x = self.refine_pre2(self.refine_pre1(x))   # [B, 2048, S/2, S/2]
        x = self.reduce(x)                          # [B, 1024, S/2, S/2]
        x = F.interpolate(x, size=target_size,
                          mode="bilinear", align_corners=False)
        x = self.refine_post2(self.refine_post1(x)) # [B, 1024, S, S]
        x = self.final(x)                           # [B, K*D, S, S]
        return list(torch.chunk(x, self.n_feats, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# RD student wrapper — pairs OCBE+Decoder for clean optimizer/state handling.
# Dispatches between the CNN and ViT variants based on the teacher's kind.
# ─────────────────────────────────────────────────────────────────────────────
class RDStudent(nn.Module):
    def __init__(self, backbone: str, feature_layers: tuple[int, ...]):
        super().__init__()
        self.backbone = backbone
        self.kind = teacher_kind(backbone)
        self.feature_layers = tuple(feature_layers)
        if self.kind == "cnn":
            # The CNN OCBE/Decoder are hard-wired to layers (1, 2, 3) with
            # WRN50-2/ResNet50 channel counts. Enforce that here.
            if self.feature_layers != (1, 2, 3):
                raise ValueError(
                    f"CNN RD requires --feature-layers 1 2 3 (got "
                    f"{list(self.feature_layers)}). The CNN OCBE/Decoder "
                    f"are architecturally tied to those three stages.")
            self.ocbe    = CNN_OCBE(base_width=WRN_BASE_WIDTH)
            self.decoder = CNN_Decoder(base_width=WRN_BASE_WIDTH)
            self.embed_dim = None
        else:  # vit
            n_blocks = DINOV2_NBLOCKS[backbone]
            bad = [l for l in self.feature_layers if not (0 <= l < n_blocks)]
            if bad:
                raise ValueError(
                    f"DINOv2 block indices out of range [0, {n_blocks - 1}]: "
                    f"{bad}")
            self.embed_dim = DINOV2_DIMS[backbone]
            self.ocbe    = ViT_OCBE(n_feats=len(self.feature_layers),
                                      embed_dim=self.embed_dim,
                                      base_width=STD_BASE_WIDTH)
            self.decoder = ViT_Decoder(n_feats=len(self.feature_layers),
                                         embed_dim=self.embed_dim,
                                         base_width=STD_BASE_WIDTH)

    def forward(self, teacher_feats: dict[int, torch.Tensor]
                 ) -> dict[int, torch.Tensor]:
        if self.kind == "cnn":
            return self.decoder(self.ocbe(teacher_feats))
        # ViT: spatial size of teacher features = input/14; pass it
        # to the decoder so F.interpolate hits the exact S, even when odd.
        bn = self.ocbe(teacher_feats)
        target_size = next(iter(teacher_feats.values())).shape[-1]
        chunks = self.decoder(bn, target_size=target_size)
        # Reassemble {block_idx: chunk} in the same order as teacher_feats.
        return {k: c for k, c in zip(teacher_feats.keys(), chunks)}


# ─────────────────────────────────────────────────────────────────────────────
# Loss + anomaly map
# ─────────────────────────────────────────────────────────────────────────────
def rd_cos_loss(t_feats: dict[int, torch.Tensor],
                 s_feats: dict[int, torch.Tensor]) -> torch.Tensor:
    """Sum over scales of mean(1 - cos_sim) computed per-position."""
    total = 0.0
    for k in t_feats:
        t = t_feats[k]
        s = s_feats[k]
        # Cosine sim along channel axis -> [B, H, W]
        cos = F.cosine_similarity(t, s, dim=1, eps=1e-8)
        total = total + (1.0 - cos).mean()
    return total


@torch.inference_mode()
def rd_anomaly_map(t_feats: dict[int, torch.Tensor],
                    s_feats: dict[int, torch.Tensor],
                    input_size: int,
                    amap_mode: str = "mul") -> torch.Tensor:
    """Per-scale (1 - cos_sim) maps, upsampled to (input_size, input_size)
    then combined. 'mul' is the paper's default; 'sum' is more stable
    when one of the scales saturates to 0."""
    B = next(iter(t_feats.values())).shape[0]
    device = next(iter(t_feats.values())).device
    if amap_mode == "mul":
        amap = torch.ones((B, 1, input_size, input_size), device=device,
                           dtype=torch.float32)
        op = "mul"
    elif amap_mode == "sum":
        amap = torch.zeros((B, 1, input_size, input_size), device=device,
                            dtype=torch.float32)
        op = "sum"
    else:
        raise ValueError(f"unknown amap_mode: {amap_mode}")
    for k in sorted(t_feats.keys()):
        t = t_feats[k].float()
        s = s_feats[k].float()
        cos = F.cosine_similarity(t, s, dim=1, eps=1e-8).unsqueeze(1)
        d = (1.0 - cos)
        d = F.interpolate(d, size=(input_size, input_size),
                          mode="bilinear", align_corners=False)
        if op == "mul":
            amap = amap * d
        else:
            amap = amap + d
    return amap.squeeze(1)  # [B, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_rd(teacher: FeatureExtractor,
             student: RDStudent,
             records: list[ImageRecord],
             cfg: "RunConfig",
             device: torch.device) -> None:
    ds = TrainGoodDataset(records, input_size=cfg.input_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0),
                         worker_init_fn=worker_init_fn, drop_last=False)
    iters_per_epoch = max(len(loader), 1)
    if cfg.total_iters is not None and cfg.total_iters > 0:
        n_epochs = max(1, math.ceil(cfg.total_iters / iters_per_epoch))
        print(f"    [auto-epoch] total_iters={cfg.total_iters} / "
              f"{iters_per_epoch} iters/epoch = {n_epochs} epochs "
              f"(--epochs={cfg.epochs} overridden)")
    else:
        n_epochs = cfg.epochs

    # RD4AD paper: Adam with betas=(0.5, 0.999), lr=5e-3 (single class
    # MVTec). We expose this on the CLI for tuning.
    params = list(student.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr,
                                   betas=(cfg.beta1, cfg.beta2),
                                   weight_decay=cfg.weight_decay)
    total_iters = iters_per_epoch * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))
    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"    [{now_hms()}] training: {n_epochs} epochs, "
          f"{iters_per_epoch} iters/epoch ({total_iters} total), "
          f"{len(records)} train_good, batch={cfg.batch_size}, "
          f"lr={cfg.lr:.0e}, amp={use_amp}")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    teacher.eval()                          # frozen, eval BN stats
    for epoch in range(n_epochs):
        student.train()
        loss_sum = 0.0; n = 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Teacher under no-grad regardless of AMP — its weights
                # don't move, and inference_mode is set in FeatureExtractor.
                t_feats = teacher(x, layers=cfg.feature_layers)
                # Detach defensively — FeatureExtractor uses inference_mode
                # so this is essentially a no-op, but it makes intent clear.
                t_feats = {k: v.detach() for k, v in t_feats.items()}
                s_feats = student(t_feats)
                loss = rd_cos_loss(t_feats, s_feats)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
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
def _score_one_pass(teacher: FeatureExtractor, student: RDStudent,
                     x: torch.Tensor, cfg: "RunConfig",
                     device: torch.device) -> torch.Tensor:
    use_amp = (device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        t_feats = teacher(x, layers=cfg.feature_layers)
        s_feats = student(t_feats)
    amap = rd_anomaly_map(t_feats, s_feats,
                          input_size=cfg.input_size,
                          amap_mode=cfg.amap_mode)
    return amap.cpu()                       # [B, H, W] float32


@torch.inference_mode()
def score_batch(teacher: FeatureExtractor, student: RDStudent,
                 x: torch.Tensor, cfg: "RunConfig",
                 device: torch.device) -> torch.Tensor:
    """Returns the (B, H, W) anomaly map averaged over the TTA views.
    Each TTA view is un-flipped/un-rotated back to the original frame
    before averaging — same convention as PatchCore."""
    x = x.to(device, non_blocking=True)
    acc = None; n = 0

    def _add(s: torch.Tensor):
        nonlocal acc, n
        if acc is None: acc = s.clone()
        else: acc += s
        n += 1

    _add(_score_one_pass(teacher, student, x, cfg, device))
    if cfg.tta in ("hflip", "hvflip", "d4"):
        s = _score_one_pass(teacher, student,
                             torch.flip(x, dims=[-1]), cfg, device)
        _add(torch.flip(s, dims=[-1]))
    if cfg.tta in ("vflip", "hvflip", "d4"):
        s = _score_one_pass(teacher, student,
                             torch.flip(x, dims=[-2]), cfg, device)
        _add(torch.flip(s, dims=[-2]))
    if cfg.tta == "d4":
        for k in (1, 2, 3):
            s = _score_one_pass(teacher, student,
                                 torch.rot90(x, k=k, dims=[-2, -1]),
                                 cfg, device)
            _add(torch.rot90(s, k=-k, dims=[-2, -1]))
    return acc / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tracking
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    # Teacher
    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (1, 2, 3)
    input_size: int = 256
    amap_mode: str = "mul"           # mul | sum
    # Training
    epochs: int = 200
    total_iters: int | None = None   # if set, overrides --epochs
    batch_size: int = 16
    lr: float = 0.005
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
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


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "reverse_distillation",
        "backbone": cfg.backbone,
        "feature_layers": list(cfg.feature_layers),
        "input_size": cfg.input_size,
        "amap_mode": cfg.amap_mode,
        "epochs": cfg.epochs,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "beta1": cfg.beta1,
        "beta2": cfg.beta2,
        "weight_decay": cfg.weight_decay,
        "smooth_sigma": cfg.smooth_sigma,
        "tta": cfg.tta,
        "seed": cfg.seed,
        "v": 2,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    budget = (f"it{cfg.total_iters}" if (cfg.total_iters and cfg.total_iters > 0)
              else f"e{cfg.epochs}")
    bb = BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)
    # CNN: layer indices are 1-digit and contiguous, so a packed form is
    # readable ("L123"). ViT: block indices may be 2-digit, use "_"-sep.
    if cfg.backbone in CNN_BACKBONES:
        L = "".join(str(l) for l in cfg.feature_layers)
    else:
        L = "_".join(str(l) for l in cfg.feature_layers)
    bits = (f"{stamp}_rd-{bb}_L{L}_in{cfg.input_size}_{budget}"
            f"_bs{cfg.batch_size}_lr{cfg.lr:.0e}_{cfg.amap_mode}")
    if cfg.tta != "none":
        bits += f"_tta-{cfg.tta}"
    if cfg.run_tag:
        bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_one_class(cls: str, records_all: list[ImageRecord],
                   teacher: FeatureExtractor,
                   cfg: RunConfig, run_dir: Path,
                   device: torch.device) -> dict:
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

    # Build a fresh student per class — one-class embeddings are
    # class-specific. The teacher is shared (frozen) across classes.
    student = RDStudent(backbone=cfg.backbone,
                         feature_layers=cfg.feature_layers).to(device)

    train_rd(teacher, student, train_good, cfg, device)
    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_rd.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save(student.state_dict(), ck)
        print(f"    saved checkpoint -> {ck}")

    eval_rows: list[dict] = []
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
            sm = score_batch(teacher, student, x, cfg, device).numpy()
            m_np = masks.numpy()
            for b in range(sm.shape[0]):
                s = gaussian_smooth(sm[b], cfg.smooth_sigma)
                scores_by_idx[int(idxs[b])] = s
                gt_by_idx[int(idxs[b])] = m_np[b]
        by_anom = defaultdict(list)
        for ridx, s in scores_by_idx.items():
            r = train_anom[ridx]
            by_anom[r.anomaly_type or "?"].append(
                pixel_average_precision(s, gt_by_idx[ridx]))
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

    test_results: list[tuple[ImageRecord, np.ndarray]] = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images "
            f"(tta={cfg.tta}, amap={cfg.amap_mode})")
        ds_t = InferenceDataset(test, input_size=cfg.input_size,
                                  load_masks=False)
        loader_t = DataLoader(ds_t, batch_size=cfg.score_batch_size,
                               shuffle=False, num_workers=cfg.num_workers,
                               pin_memory=True,
                               persistent_workers=(cfg.num_workers > 0))
        n_done, last_log = 0, 0
        for x, _, idxs in loader_t:
            sm = score_batch(teacher, student, x, cfg, device).numpy()
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--backbone", default="wide_resnet50_2",
                    choices=ALL_BACKBONES,
                    help="Teacher backbone. CNN options use the hierarchical "
                         "OCBE+Decoder (layers fixed at [1,2,3]); DINOv2 "
                         "options use a single-resolution OCBE+Decoder over "
                         "the chosen transformer blocks.")
    ap.add_argument("--feature-layers", type=int, nargs="+", default=None,
                    help="CNN: must be `1 2 3` (architecturally fixed). "
                         "DINOv2: transformer block indices 0..n_blocks-1. "
                         "If omitted, defaults to [1,2,3] for CNN, "
                         "[3,6,9,11] for DINOv2-S/B (12 blocks), "
                         "[5,11,17,23] for DINOv2-L (24 blocks).")
    ap.add_argument("--input-size", type=int, default=256,
                    help="CNN: must be divisible by 32 (OCBE downsamples "
                         "by 32). Recommended: 256, 384. "
                         "DINOv2: must be divisible by 14 (patch size). "
                         "Recommended: 392 (28x28 tokens), 518 (37x37, "
                         "DINOv2 native resolution).")
    ap.add_argument("--amap-mode", default="mul", choices=["mul", "sum"],
                    help="How to combine per-scale anomaly maps. 'mul' is "
                         "the RD4AD paper default; 'sum' tends to be more "
                         "stable when one scale saturates close to 0.")
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=None,
                    help="overrides --epochs (recommended: 2500 for "
                         "Spacepresso's 2000+ train_good per class)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.005,
                    help="RD4AD paper uses 5e-3 with Adam(0.5, 0.999).")
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    # Inference
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
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    # Resolve --feature-layers default based on backbone.
    if args.feature_layers is None:
        args.feature_layers = list(default_feature_layers(args.backbone))
    feature_layers = tuple(sorted(set(args.feature_layers)))

    # Per-backbone input validation.
    kind = teacher_kind(args.backbone)
    if kind == "cnn":
        if args.input_size % 32 != 0:
            raise SystemExit(
                f"[FATAL] CNN RD requires --input-size divisible by 32 "
                f"(OCBE downsamples by 32). Got {args.input_size}; try "
                f"256, 320, 384.")
        if feature_layers != (1, 2, 3):
            raise SystemExit(
                f"[FATAL] CNN RD requires --feature-layers 1 2 3 (got "
                f"{list(feature_layers)}). The CNN OCBE/Decoder are "
                f"architecturally tied to those stages.")
    else:  # vit
        if args.input_size % 14 != 0:
            raise SystemExit(
                f"[FATAL] DINOv2 requires --input-size divisible by 14; "
                f"got {args.input_size}. Try 392 (28x28 tokens) or 518 "
                f"(37x37, DINOv2 native resolution).")
        nb = DINOV2_NBLOCKS[args.backbone]
        bad = [l for l in feature_layers if not (0 <= l < nb)]
        if bad:
            raise SystemExit(
                f"[FATAL] DINOv2 block indices out of range "
                f"[0, {nb - 1}]: {bad}")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone,
        feature_layers=feature_layers,
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
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tee_to(run_dir / "run_log.txt"):
        hr(f"REVERSE DISTILLATION — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  teacher backbone : {cfg.backbone}  "
              f"(kind={teacher_kind(cfg.backbone)}, frozen)")
        print(f"  feature_layers   : {list(cfg.feature_layers)}")
        print(f"  input_size       : {cfg.input_size}"
              + (f"  (tokens: {cfg.input_size // 14}x{cfg.input_size // 14})"
                 if cfg.backbone in VIT_BACKBONES else ""))
        print(f"  amap_mode        : {cfg.amap_mode}")
        if cfg.total_iters is not None:
            print(f"  total_iters      : {cfg.total_iters}  "
                  f"(overrides --epochs={cfg.epochs})")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / b1 / b2     : {cfg.lr} / {cfg.beta1} / {cfg.beta2}  "
              f"(Adam)")
        print(f"  weight_decay     : {cfg.weight_decay}")
        print(f"  amp              : {cfg.amp}    num_workers: {cfg.num_workers}")
        print(f"  score_batch_size : {cfg.score_batch_size}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(cfg).items()}, f, indent=2)

        # Build teacher ONCE (shared across classes — it's frozen).
        teacher = FeatureExtractor(cfg.backbone).to(device).eval()

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("\n[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): {', '.join(classes)}")

        all_test_results = []
        all_eval_rows = []
        class_aps, class_elapsed = {}, {}
        for cls in classes:
            res = run_one_class(cls, records, teacher, cfg, run_dir, device)
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
            print(f"  saved per-(class, anomaly_type) AP -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload: {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        row = {
            "run_id": run_id,
            "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": cfg.backbone,
            "feature_layers": "+".join(str(l) for l in cfg.feature_layers),
            "target_layer": "",
            "input_size": cfg.input_size,
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
            "notes": (f"reverse_distillation {cfg.backbone} "
                      f"L={'+'.join(str(l) for l in cfg.feature_layers)} "
                      f"amap={cfg.amap_mode} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size} lr{cfg.lr:.0e}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()