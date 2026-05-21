"""Spacepresso FastFlow baseline.

Implements "FastFlow: Unsupervised Anomaly Detection and Localization
via 2D Normalizing Flows" (Yu et al., 2021), adapted to Spacepresso.

# Why FastFlow is a useful stacker partner

  Every other track in your stack scores anomalies by some form of
  FEATURE DISTANCE:
    - PatchCore (exp5, exp7): cosine distance to a coreset.
    - CutPaste-NN (exp8c/d): NN over self-supervised classifier features.
    - RD4AD (exp10b): cosine distance between teacher and student.
    - CFA (exp14): NN over a learned contrastive embedding.
    - EfficientAD (effad): teacher-student MSE + AE-student MSE.
    - UniAD (exp12): masked transformer reconstruction MSE.

  FastFlow takes a fundamentally different approach: it learns the
  generative density p(x) of train_good features via a normalizing
  flow. Anomaly score = -log p(x) per pixel. Density estimation has a
  different inductive bias from feature distance:
    - Distance-based methods penalise "you don't look like any normal".
      Density-based methods penalise "you live in a low-probability
      region of the normal manifold."
    - For multi-modal normal data (Spacepresso has 5 views per sample
      with very different appearances), density estimation handles
      multimodality natively: the flow learns a multimodal Gaussian
      mixture by construction, while NN over a coreset implicitly
      handles it by having many centroids.

  → distinct error mode from every other track; the stacker should
  pick it up especially on classes where defects are subtle texture
  perturbations (coffee, pistachio) rather than geometric anomalies.

# Architecture

  BACKBONE (frozen, shared across classes):
    Default: wide_resnet50_2. Extract features at layers 1, 2, 3
    SEPARATELY (no fusion, unlike PatchCore/CFA). FastFlow trains one
    independent flow per scale because the conditional distribution
    over feature maps is scale-specific: layer-1 features encode
    texture, layer-3 features encode semantics. A single flow with
    fused multi-scale input would have to model both simultaneously
    and would degrade.

  FLOW (per scale, per class):
    Stack of N coupling blocks. Each block:
      1. Permute channels (fixed reverse, no learnable permutation).
      2. Affine coupling:
           x1, x2 = split(x, dim=channels)
           s, t = subnet(x1)
           x2 = x2 * exp(clamp(s)) + t
           output = concat(x1, x2)
      3. Subnet: Conv3x3 -> ReLU -> Conv1x1 -> ReLU -> Conv3x3,
         with the final conv zero-initialised so the block starts as
         identity. This keeps gradients sane in the first 100 iters
         when the flow has random output.

    The "2D" in FastFlow: the subnet preserves the spatial structure
    (uses CONV, not MLP). This lets the coupling layer condition each
    pixel's transformation on its neighbours, which is what makes the
    flow capable of modelling spatial correlations in feature maps.

  LOSS (only on train_good):
    For each scale s, per pixel:
      z = flow(features)
      log_det = Σ log|det J|  (summed across coupling blocks)
      log p(z) = -0.5 ||z||² - 0.5 C log(2π)
      NLL_per_pixel = 0.5 ||z||² - log_det
    Total loss: mean of NLL over (batch × pixel × scale).

# Inference

    Per scale, per pixel:
      score_per_scale = 0.5 ||z||² - log_det      (≈ -log p, up to const)
    Per pixel, multi-scale fusion:
      score = mean_over_scales( upsample(score_per_scale to input_size) )
    Smoothed by Gaussian σ=1.5 and percentile-calibrated to [0, 1] for
    q8rle submission encoding, same as every other baseline.

# Memory / speed (NVIDIA L4 24 GB)

  Per class @ input 384, batch 32, 5 epochs (~1000 iters), 3 scales:
    Backbone forward (fp16 AMP):  ~80 ms/batch
    Flow forward + backward:       ~200 ms/batch
    Training:                       ~3-4 min
    Eval + test:                   ~25 + 70 s
  Full 8 classes:                   ~50-60 min

# Hyperparameters that actually matter

  --n-flow-blocks    8-20. 8 is the FastFlow paper default. More blocks
                      gives more model capacity but blows up training
                      time linearly. 12 is a good compromise.
  --hidden-ratio     1.0-2.0. Multiplier on subnet hidden width
                      relative to channels. 1.0 = subnet hidden_channels
                      == feature channels. 2.0 fits ~4x parameters.
  --clamp            1.5-3.0. Clamp on tanh(s) inside the coupling.
                      Lower = more stable but less expressive. 2.0
                      is the FastFlow default and works well.
  --flow-lr          1e-3 default; usually fine across all classes.

# Dependencies

  patchcore_baseline_v2.py and local_preds_saver.py in same directory.
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
    BACKBONE_CHANNELS,
    BACKBONE_SHORT,
    RESNET_BACKBONES,
    DINOV2_BACKBONES,
    DINOV2_NBLOCKS,
    ALL_BACKBONES,
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
# Normalizing flow building blocks
# ─────────────────────────────────────────────────────────────────────────────
class _Subnet(nn.Module):
    """Small CNN that predicts (s, t) for a coupling layer. 3x3 → 1x1 → 3x3
    keeps the receptive field reasonable and matches FastFlow's design.
    Final conv is zero-initialised so the coupling starts as identity."""
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


class AffineCoupling(nn.Module):
    """Real-NVP style affine coupling. Splits channels in half, predicts
    (s, t) from one half to transform the other. clamp(tanh) keeps the
    log-determinant bounded."""
    def __init__(self, channels: int, hidden_ratio: float = 1.0,
                 clamp: float = 2.0):
        super().__init__()
        self.channels = channels
        self.split = channels // 2
        self.rest = channels - self.split
        hidden = max(int(channels * hidden_ratio), 16)
        # subnet outputs 2 * rest channels: half for s, half for t
        self.subnet = _Subnet(self.split, 2 * self.rest, hidden)
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, H, W).
        Returns (out, log_det) where log_det has shape (B, H, W).
        """
        x1, x2 = x[:, :self.split], x[:, self.split:]
        st = self.subnet(x1)
        s, t = st.chunk(2, dim=1)
        s = self.clamp * torch.tanh(s / self.clamp)
        x2 = x2 * torch.exp(s) + t
        out = torch.cat([x1, x2], dim=1)
        # log|det J| per pixel = sum of s over channels (only the
        # transformed half contributes; the identity half contributes 0).
        log_det = s.sum(dim=1)
        return out, log_det


class FastFlowSingle(nn.Module):
    """A stack of coupling blocks for ONE feature scale. Permutation
    between blocks is a fixed channel reversal — cheap and effective in
    practice; learnable 1x1 permutations don't help on Spacepresso's
    relatively small feature dims and add training instability."""

    def __init__(self, channels: int, n_blocks: int = 8,
                 hidden_ratio: float = 1.0, clamp: float = 2.0):
        super().__init__()
        self.channels = channels
        self.blocks = nn.ModuleList([
            AffineCoupling(channels, hidden_ratio, clamp)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        log_det_total = torch.zeros(B, H, W, device=x.device,
                                      dtype=x.dtype)
        for block in self.blocks:
            x, ld = block(x)
            log_det_total = log_det_total + ld
            # Fixed reverse permutation: cheap channel shuffle.
            x = torch.flip(x, dims=[1])
        return x, log_det_total


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scale wrapper — one flow per backbone layer
# ─────────────────────────────────────────────────────────────────────────────
class FastFlowMulti(nn.Module):
    def __init__(self, channels_per_scale: dict[int, int],
                 n_blocks: int = 8, hidden_ratio: float = 1.0,
                 clamp: float = 2.0):
        super().__init__()
        self.scales = sorted(channels_per_scale.keys())
        self.flows = nn.ModuleDict({
            str(s): FastFlowSingle(channels_per_scale[s],
                                     n_blocks=n_blocks,
                                     hidden_ratio=hidden_ratio,
                                     clamp=clamp)
            for s in self.scales
        })

    def forward(self, feats: dict[int, torch.Tensor]
                 ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        zs: dict[int, torch.Tensor] = {}
        lds: dict[int, torch.Tensor] = {}
        for s in self.scales:
            z, ld = self.flows[str(s)](feats[s])
            zs[s] = z
            lds[s] = ld
        return zs, lds


def flow_nll_per_pixel(z: torch.Tensor, log_det: torch.Tensor
                        ) -> torch.Tensor:
    """z: (B, C, H, W).  log_det: (B, H, W).
    Returns NLL per pixel, shape (B, H, W).
    Under N(0, I): log p(z) = -0.5 ||z||² - 0.5 C log(2π)
    NLL = -log p(z) - log_det = 0.5 ||z||² + 0.5 C log(2π) - log_det
    Constant term dropped (irrelevant for ranking / for the average loss
    gradient w.r.t. parameters).
    """
    return 0.5 * (z ** 2).sum(dim=1) - log_det


# ─────────────────────────────────────────────────────────────────────────────
# Backbone feature extraction at the per-scale level (no fusion)
# ─────────────────────────────────────────────────────────────────────────────
def _per_scale_features(extractor: FeatureExtractor, x: torch.Tensor,
                         feature_layers: tuple[int, ...]
                         ) -> dict[int, torch.Tensor]:
    """Returns {layer_idx: (B, C, H, W)} — NO upsampling, NO concat.

    NOTE 1 — context: this function does NOT enforce its own no_grad /
    inference_mode. The caller is responsible:
      - Training:   wrap in `torch.no_grad()` so the backbone forward
                    is untracked, but the returned tensors stay normal-
                    mode and can flow into the trainable flow.
      - Validation/test: callers (`compute_norm_stats`, `_score_one_pass`)
                    are decorated `@torch.inference_mode()` themselves;
                    this function inherits that context.

    NOTE 2 — the .clone() at the end: `FeatureExtractor.forward` in
    patchcore_baseline_v2.py is decorated with @torch.inference_mode(),
    so `maps[l]` are *inference tensors*. .float() propagates the
    inference flag. If we returned those tensors as-is, the flow's
    forward would raise:
        RuntimeError: Inference tensors cannot be saved for backward.
    The .clone() runs *outside* the extractor's inference_mode context
    (it ended when extractor.forward returned), so it produces a normal
    tensor that the flow can save for backward.
    """
    maps = extractor(x, layers=feature_layers)
    return {l: maps[l].float().clone() for l in feature_layers}


def _channels_per_scale(extractor: FeatureExtractor,
                         feature_layers: tuple[int, ...],
                         cfg: "RunConfig",
                         device: torch.device) -> dict[int, int]:
    """Run a dummy forward to read off channel counts per scale (handles
    both ResNet and DINOv2 cases without hardcoding)."""
    with torch.inference_mode():
        x = torch.zeros(1, 3, cfg.input_size, cfg.input_size, device=device)
        maps = extractor(x, layers=feature_layers)
    return {l: maps[l].shape[1] for l in feature_layers}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_fastflow(extractor: FeatureExtractor,
                    flow: FastFlowMulti,
                    records: list[ImageRecord],
                    cfg: "RunConfig",
                    device: torch.device) -> None:
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
              f"{iters_per_epoch} iters/epoch -> {n_epochs} epochs")
    else:
        n_epochs = cfg.epochs
    total_iters = iters_per_epoch * n_epochs

    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))
    use_amp = (device.type == "cuda" and cfg.amp)
    # GradScaler can be unstable with normalizing flows because the
    # backward through exp/log is delicate. Default off here; user can
    # re-enable with --amp-flow. AMP still applies to the backbone
    # forward via _per_scale_features (the backbone is frozen so its
    # numerics don't affect flow training).
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp_flow)

    print(f"    [{now_hms()}] training: {n_epochs} epochs x "
          f"{iters_per_epoch} iters ({total_iters} total)  "
          f"bs={cfg.batch_size}  scales={flow.scales}  "
          f"n_blocks={cfg.n_flow_blocks}  hidden_ratio={cfg.hidden_ratio}  "
          f"clamp={cfg.clamp}")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    extractor.eval()
    for epoch in range(n_epochs):
        flow.train()
        loss_sum = 0.0; n = 0
        per_scale_loss = defaultdict(float)
        for x in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    feats = _per_scale_features(extractor, x,
                                                  cfg.feature_layers)
            # Flow forward (full precision unless --amp-flow)
            with torch.amp.autocast("cuda", enabled=cfg.amp_flow):
                zs, lds = flow(feats)
                loss_terms = []
                for s in flow.scales:
                    nll = flow_nll_per_pixel(zs[s], lds[s])
                    Ls = nll.mean()
                    per_scale_loss[s] += Ls.item() * x.shape[0]
                    loss_terms.append(Ls)
                loss = sum(loss_terms) / len(loss_terms)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            B = x.shape[0]
            loss_sum += loss.item() * B
            n += B

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            per_str = "  ".join(
                f"L{s}={per_scale_loss[s]/max(n,1):.2f}" for s in flow.scales)
            print(f"      epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={loss_sum/max(n,1):.4f}  {per_str}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    flow.eval()
    print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation stats per scale (for cross-scale fusion)
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def compute_norm_stats(extractor, flow, records, cfg, device,
                        n_max: int = 64) -> dict[int, dict]:
    """Run a sample of train_good through the flow, compute per-scale
    mean and std of the per-pixel NLL. Used to z-score each scale before
    averaging — without this, one scale's NLL magnitude can dominate.
    """
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
    vals: dict[int, list[np.ndarray]] = defaultdict(list)
    flow.eval(); extractor.eval()
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            feats = _per_scale_features(extractor, x, cfg.feature_layers)
        zs, lds = flow(feats)
        for s in flow.scales:
            nll = flow_nll_per_pixel(zs[s], lds[s]).float()
            vals[s].append(nll.cpu().numpy().reshape(-1))
    stats: dict[int, dict] = {}
    for s in flow.scales:
        arr = np.concatenate(vals[s])
        stats[s] = {"mean": float(arr.mean()),
                     "std":  float(arr.std() + 1e-9)}
        print(f"    norm stats scale {s}: "
              f"NLL={stats[s]['mean']:.3f} ± {stats[s]['std']:.3f}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Inference primitives
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _score_one_pass(extractor, flow, x: torch.Tensor, stats: dict,
                     cfg: "RunConfig") -> torch.Tensor:
    """Returns (B, input_size, input_size) score map on GPU."""
    use_amp = (cfg.device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        feats = _per_scale_features(extractor, x, cfg.feature_layers)
    zs, lds = flow(feats)
    accum = None
    for s in flow.scales:
        nll = flow_nll_per_pixel(zs[s], lds[s]).float()       # (B, H_s, W_s)
        # z-score within scale so cross-scale magnitudes are comparable.
        nll = (nll - stats[s]["mean"]) / stats[s]["std"]
        up = F.interpolate(nll.unsqueeze(1),
                           size=(cfg.input_size, cfg.input_size),
                           mode="bilinear", align_corners=False).squeeze(1)
        if accum is None: accum = up
        else: accum = accum + up
    return accum / len(flow.scales)


@torch.inference_mode()
def score_batch(extractor, flow, x: torch.Tensor, stats: dict,
                  cfg: "RunConfig") -> torch.Tensor:
    """TTA-aware. Returns (B, input_size, input_size) on CPU."""
    x = x.to(cfg.device, non_blocking=True)
    acc = None; n = 0
    def _add(s):
        nonlocal acc, n
        if acc is None: acc = s.clone()
        else: acc += s
        n += 1
    _add(_score_one_pass(extractor, flow, x, stats, cfg))
    if cfg.tta in ("hflip", "hvflip"):
        s = _score_one_pass(extractor, flow,
                              torch.flip(x, dims=[-1]), stats, cfg)
        _add(torch.flip(s, dims=[-1]))
    if cfg.tta in ("vflip", "hvflip"):
        s = _score_one_pass(extractor, flow,
                              torch.flip(x, dims=[-2]), stats, cfg)
        _add(torch.flip(s, dims=[-2]))
    return (acc / max(n, 1)).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _score_records(extractor, flow, stats, records, cfg, load_masks):
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores, gts = {}, {}
    n_done = 0; last_log = 0
    for x, masks, idxs in loader:
        sm = score_batch(extractor, flow, x, stats, cfg).numpy()
        m_np = masks.numpy()
        for b in range(sm.shape[0]):
            scores[int(idxs[b])] = sm[b]
            gts[int(idxs[b])] = m_np[b]
        n_done += sm.shape[0]
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


def run_one_class(cls: str, records_all: list[ImageRecord],
                   extractor: FeatureExtractor,
                   channels_per_scale: dict[int, int],
                   cfg: "RunConfig", run_dir: Path, device: torch.device,
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

    flow = FastFlowMulti(channels_per_scale,
                          n_blocks=cfg.n_flow_blocks,
                          hidden_ratio=cfg.hidden_ratio,
                          clamp=cfg.clamp).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"  FastFlow: scales={list(channels_per_scale.keys())}  "
          f"channels={list(channels_per_scale.values())}  "
          f"n_blocks={cfg.n_flow_blocks}  "
          f"params={n_params/1e6:.2f}M")

    train_fastflow(extractor, flow, train_good, cfg, device)

    print(f"    [{now_hms()}] computing per-scale NLL stats...")
    stats = compute_norm_stats(extractor, flow, train_good, cfg, device)

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_fastflow.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"flow": flow.state_dict(), "stats": stats,
                    "channels_per_scale": channels_per_scale}, ck)
        print(f"    saved checkpoint -> {ck}")

    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  tta={cfg.tta}")
        scores, gts = _score_records(extractor, flow, stats, train_anom,
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
        sub(f"scoring {len(test)} test images  tta={cfg.tta}")
        scores, _ = _score_records(extractor, flow, stats, test, cfg,
                                      load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            sm_final = maybe_resize_to_submission(sm_smooth)
            test_results.append((test[r_idx], sm_final))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del flow
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
# Run config + CLI
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    # Backbone
    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (1, 2, 3)
    input_size: int = 384
    # Flow architecture
    n_flow_blocks: int = 8
    hidden_ratio: float = 1.0
    clamp: float = 2.0
    # Training
    epochs: int = 200
    total_iters: int | None = 2500
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    amp: bool = True             # backbone forward
    amp_flow: bool = False       # flow forward; default off (stability)
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
    # Runtime
    device: torch.device | None = None


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "fastflow",
        "backbone": cfg.backbone,
        "feature_layers": list(cfg.feature_layers),
        "input_size": cfg.input_size,
        "n_flow_blocks": cfg.n_flow_blocks,
        "hidden_ratio": cfg.hidden_ratio,
        "clamp": cfg.clamp,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "smooth_sigma": cfg.smooth_sigma,
        "tta": cfg.tta,
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
    bits = (f"{stamp}_fastflow_{bb}_L{L}_in{cfg.input_size}"
            f"_nb{cfg.n_flow_blocks}_hr{cfg.hidden_ratio:g}"
            f"_c{cfg.clamp:g}_{budget}_bs{cfg.batch_size}")
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
                    choices=ALL_BACKBONES)
    ap.add_argument("--feature-layers", type=int, nargs="+", default=[1, 2, 3],
                    help="For ResNet: layer indices in {1,2,3,4}. "
                         "For DINOv2: block indices in [0, n_blocks).")
    ap.add_argument("--input-size", type=int, default=384,
                    help="Multiple of 32 for ResNet, multiple of 14 for "
                         "DINOv2.")
    # Flow
    ap.add_argument("--n-flow-blocks", type=int, default=8,
                    help="Number of coupling blocks per scale. 8 = paper "
                         "default; 12-20 = more expressive but slower.")
    ap.add_argument("--hidden-ratio", type=float, default=1.0,
                    help="Subnet hidden_channels / feature_channels. "
                         "1.0 = same as feature dim; 2.0 = wider subnet.")
    ap.add_argument("--clamp", type=float, default=2.0,
                    help="Clamp on tanh(s) inside the affine coupling.")
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=2500,
                    help="Overrides --epochs.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable AMP entirely (backbone too).")
    ap.add_argument("--amp-flow", action="store_true",
                    help="Use AMP for the flow forward as well. Default "
                         "is OFF — fp16 through exp/log in the coupling "
                         "can produce NaNs early in training. Enable "
                         "only if you've verified stability with this "
                         "config.")
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
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    feature_layers = tuple(sorted(set(args.feature_layers)))
    for l in feature_layers:
        if l not in BACKBONE_CHANNELS[args.backbone]:
            raise SystemExit(
                f"[FATAL] feature_layer {l} invalid for {args.backbone}")
    if args.backbone in DINOV2_BACKBONES:
        if args.input_size % 14 != 0:
            raise SystemExit(
                f"[FATAL] DINOv2 needs --input-size divisible by 14")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone, feature_layers=feature_layers,
        input_size=args.input_size,
        n_flow_blocks=args.n_flow_blocks,
        hidden_ratio=args.hidden_ratio,
        clamp=args.clamp,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        amp=not args.no_amp, amp_flow=args.amp_flow,
        num_workers=args.num_workers,
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
        hr(f"FASTFLOW — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  backbone         : {cfg.backbone}")
        print(f"  feature_layers   : {list(cfg.feature_layers)}")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  flow blocks      : {cfg.n_flow_blocks}  "
              f"hidden_ratio={cfg.hidden_ratio}  clamp={cfg.clamp}")
        if cfg.total_iters and cfg.total_iters > 0:
            print(f"  total_iters      : {cfg.total_iters}")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / wd          : {cfg.lr} / {cfg.weight_decay}")
        print(f"  amp (backbone)   : {cfg.amp}")
        print(f"  amp (flow)       : {cfg.amp_flow}  (off by default; "
              f"flow numerics are delicate)")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, (Path, torch.device))
                            else v)
                       for k, v in asdict(cfg).items()}, f, indent=2,
                        default=str)

        extractor = FeatureExtractor(cfg.backbone).to(device).eval()
        channels_per_scale = _channels_per_scale(
            extractor, cfg.feature_layers, cfg, device)
        print(f"\n  channels per scale: {channels_per_scale}")

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
            res = run_one_class(cls, records, extractor, channels_per_scale,
                                  cfg, run_dir, device,
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
            save_test_predictions(all_test_results, run_dir)
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload: {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        row = {
            "run_id": run_id, "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": f"FASTFLOW_{BACKBONE_SHORT.get(cfg.backbone, cfg.backbone).upper()}",
            "feature_layers": "+".join(str(l) for l in cfg.feature_layers),
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
            "notes": (f"fastflow {BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)} "
                      f"nb{cfg.n_flow_blocks} hr{cfg.hidden_ratio} "
                      f"c{cfg.clamp} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()