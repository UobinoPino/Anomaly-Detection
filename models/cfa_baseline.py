"""Spacepresso CFA baseline — Coupled-hypersphere-based Feature Adaptation.

Implements "CFA: Coupled-hypersphere-based Feature Adaptation for
Target-Oriented Anomaly Localization" (Ahn et al., CVPR 2022),
adapted to Spacepresso to match the rest of the codebase (run_id,
local_predictions.npz, ablation_master row, etc).

# Why CFA is a useful stacker partner

  - PatchCore (exp5, exp7): raw frozen-backbone features → coreset NN.
    The features are NEVER adapted to the dataset; they are whatever
    ImageNet pretraining gave you.
  - RD4AD (exp10): one-class reconstruction in the WRN50 teacher's
    feature space. Adapted target, but the adaptation is generative
    (reconstruct features), not discriminative.
  - CutPaste-NN (exp8c/d): backbone is fine-tuned on a 3-way
    classification task (normal/cutpaste/scar). Adaptation is
    *image-level* and synthetic-defect-driven.
  - --- CFA (this file) ---: backbone is FROZEN, but a small MLP
    descriptor on top of it is trained to PULL train_good patch
    features toward a memory bank of normal centroids and PUSH them
    away from non-target neighbours. The adaptation is per-pixel,
    contrastive, and uses no synthetic defects.

  → distinct error mode from all four existing tracks.

# Architecture

  BACKBONE (frozen, shared across classes):
    Default: wide_resnet50_2 (matches exp5/exp10). Multi-scale features
    fused at the target layer's grid via the same `patchify_and_combine`
    used by PatchCore. With --patch-size 1 (default) no local pooling
    is applied (raw patches), as in the CFA paper. Set --patch-size 3
    for PatchCore-style 3x3 local aggregation if you want similarity
    to exp5 in the hash space.

  DESCRIPTOR (per class, trainable):
    Small 2-layer MLP with a residual connection. Maps the fused
    multi-scale feature dim → same dim. Initialized so the descriptor
    starts as the identity function, then drifts to a target-specific
    embedding during training. Residual prevents catastrophic drift
    early in training when the contrastive gradient is noisy.

  MEMORY BANK (per class, fixed after init):
    Greedy k-center coreset over train_good patch features, stored on
    GPU as raw backbone features. Same `greedy_coreset` PatchCore uses,
    so memory budget and selection time are familiar.

# Training (only on train_good)

  Per iteration:
    1. Extract backbone features for a batch of train_good images.
    2. Patchify + combine → patch feature tensor of shape (B, P, C).
    3. Apply descriptor φ to each patch → adapted features of (B, P, C).
    4. Apply the SAME descriptor (under no_grad) to the memory bank.
    5. Pairwise distance matrix between adapted patches and adapted
       memory. Sort each row ascending.
    6. L_att = mean(relu(d[:, :K_att]² − r²))      # K_att nearest
       L_rep = mean(relu(r² + α − d[:, K_att:K_att+K_rep]²))  # next K_rep
       L = L_att + L_rep

  Notes on stability:
    - Memory features go through the descriptor WITHOUT gradient. They
      act as moving targets that stay tied to the descriptor's current
      state without contributing to the gradient. This is what
      prevents collapse to a single point: the K_rep term forces some
      distance to non-target neighbours, but if memory updated WITH
      gradient, both sides could collapse together.
    - Descriptor is initialized so the residual branch dominates (final
      Linear weights/bias zero), making the first iteration's gradient
      well-conditioned.

# Inference

    Per patch:
      adapted_p = φ(patch)
      adapted_C = φ(memory)
      d_min = sorted(||adapted_p - adapted_c_j||²)[:K_test].mean()
    → upsampled bilinearly to input_size and Gaussian smoothed.

# Memory/speed (NVIDIA L4 24 GB, default config)

    Per class @ input 384, batch 16, 2000 iters, 5% coreset:
      Backbone forward (fp16 AMP):  ~100 ms/batch
      Coreset init (one-shot):       ~30 s
      Training loop:                ~2-3 min
      Eval + test:                   ~30 s + 90 s
    Full 8 classes:                  ~50-65 min

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
    ImageRecord,
    scan_dataset,
    FeatureExtractor,
    patchify_and_combine,
    greedy_coreset,
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
# CFA descriptor — small residual MLP applied per patch
# ─────────────────────────────────────────────────────────────────────────────
class CFADescriptor(nn.Module):
    """Per-patch MLP with residual connection, zero-initialised at the
    output so it starts as the identity. The residual lets us run
    gradient through a stable starting point even when the contrastive
    loss is noisy in the first hundred iters."""

    def __init__(self, in_dim: int, hidden_dim: int | None = None,
                 use_bn: bool = True):
        super().__init__()
        h = hidden_dim or in_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, h)]
        if use_bn:
            layers.append(nn.BatchNorm1d(h))
        layers += [nn.LeakyReLU(0.1, inplace=True), nn.Linear(h, in_dim)]
        self.net = nn.Sequential(*layers)
        # Zero-init the final Linear so descriptor starts as identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D). Output: same shape."""
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        out = self.net(flat) + flat
        return out.reshape(orig_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Memory bank — coreset over raw backbone features (one-shot)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: this is @torch.no_grad(), not @torch.inference_mode(). Reason:
# the returned memory tensor is later passed to `descriptor(memory)`
# during training (inside `train_cfa`'s no_grad block), and the result
# is fed into `cfa_loss` where it forms one side of a cdist against the
# grad-tracked adapted patch features. PyTorch refuses to save *inference
# tensors* for backward, even when they are on the constant side of an
# op. no_grad gives the same "no gradient through backbone" behaviour
# without making the outputs into inference tensors.
@torch.no_grad()
def init_memory_bank(extractor: FeatureExtractor,
                      records: list[ImageRecord],
                      cfg: "RunConfig",
                      device: torch.device) -> tuple[torch.Tensor, tuple[int, int], int]:
    """Extract train_good patch features, run coreset, return memory bank
    on GPU plus (H, W) of the feature grid and D = feature dim."""
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=False)
    loader = DataLoader(ds, batch_size=cfg.fit_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    feats_list: list[torch.Tensor] = []
    feature_hw: tuple[int, int] | None = None
    feature_dim: int | None = None
    print(f"    [{now_hms()}] extracting train/good patch features "
          f"({len(records)} images, layers={list(cfg.feature_layers)}, "
          f"target_layer={cfg.target_layer}, patch_size={cfg.patch_size})...")
    extractor.eval()
    use_amp = (device.type == "cuda" and cfg.amp)
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            maps = extractor(x, layers=cfg.feature_layers)
        pf = patchify_and_combine(maps, patch_size=cfg.patch_size,
                                    target_layer=cfg.target_layer)
        if feature_hw is None:
            P = pf.shape[1]
            H = W = int(math.isqrt(P))
            feature_hw = (H, W)
            feature_dim = pf.shape[2]
        pf = pf.reshape(-1, pf.shape[-1]).detach().cpu()
        feats_list.append(pf)
        del x, maps, pf
    all_feats = torch.cat(feats_list, dim=0)
    del feats_list

    if cfg.coreset_fp16:
        all_feats = all_feats.half()
    print(f"    -> {all_feats.shape[0]} patch features (D={feature_dim}, "
          f"{all_feats.element_size() * all_feats.numel() / 1e9:.2f} GB CPU)")

    n_select = max(int(cfg.coreset_frac * all_feats.shape[0]), 1)
    print(f"    [{now_hms()}] greedy coreset ({cfg.coreset_algo}, "
          f"batch={cfg.coreset_batch}): selecting {n_select} of "
          f"{all_feats.shape[0]} ({cfg.coreset_frac:.1%})")
    idx_cpu = greedy_coreset(all_feats, n_select, device,
                              seed=cfg.seed,
                              project_chunk=cfg.project_chunk,
                              algo=cfg.coreset_algo,
                              batch_size=cfg.coreset_batch)
    selected = all_feats[idx_cpu].to(device, non_blocking=True).float()
    del all_feats
    # Store memory in fp32 so descriptor BN statistics are stable.
    memory = selected.contiguous()
    del selected
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"    [{now_hms()}] memory bank  shape={tuple(memory.shape)}  "
          f"dtype={memory.dtype}  "
          f"({memory.element_size() * memory.numel() / 1e6:.1f} MB on GPU)")
    return memory, feature_hw, feature_dim


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def cfa_loss(adapted_p: torch.Tensor,
              mem_t: torch.Tensor,
              mem_norm_sq: torch.Tensor,
              k_att: int, k_rep: int,
              r_sq: float, alpha: float,
              chunk_size: int = 8192
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-bounded, mixed-precision contrastive loss against the
    (adapted) memory bank.

    Inputs
      adapted_p   : (N, D) fp32  -- adapted patch features for this batch
                    (grad-tracked, side we backprop through).
      mem_t       : (D, M)       -- adapted memory bank, ALREADY
                    transposed and contiguous, in the dtype the matmul
                    should run in (bf16 → tensor-core matmul; fp32 →
                    cuBLAS fp32 matmul). Detached / no grad expected.
                    Computed once per refresh interval by train_cfa and
                    reused across iters.
      mem_norm_sq : (M,) fp32    -- precomputed ‖m‖² for the matmul-form
                    distance. fp32 for precision regardless of mem_t
                    dtype.
      k_att, k_rep, r_sq, alpha : CFA hyperparameters.
      chunk_size  : split N over chunks of this size when forming the
                    pairwise distance matrix.

    Why precomputed and split:

      The previous version called `(adapted_mem ** 2).sum(dim=1)` and
      `adapted_mem.t().contiguous()` INSIDE the loss every iteration,
      which is the same M=178K × 1536 work each call (~5-10 ms) on top
      of the per-iter descriptor(memory) forward (~100 ms). Caller now
      owns refresh, so the loss does zero memory-side work.

      The previous version also ran the heavy matmul in fp32, which on
      an L4 (~7.5 TFLOPS fp32 vs ~120 TFLOPS bf16 tensor cores) was
      ~16× slower than necessary. Distance computation is the only
      step in this loss that hits a serious FLOP count, so casting
      just that operation is enough.

    Numerics (bf16 path):

      bf16 has the same exponent range as fp32, only the mantissa
      narrows (7 bits vs 23). For descriptor outputs with norm ~10-30,
      pairwise inner products are ~10²-10³, and the per-element bf16
      error is around 1 part in 256 — well below the gradient noise we
      already tolerate from random shuffles. Norms (‖p‖², ‖m‖²) are
      kept in fp32 because their cancellation with 2 p·m is the
      precision-sensitive step; doing those in bf16 would lose the
      smallest distances which is exactly what topk is selecting.

    Returns
      (loss, L_att, L_rep) — all fp32 scalars.
    """
    n_p, D = adapted_p.shape
    if n_p == 0:
        z = torch.zeros((), device=adapted_p.device, dtype=torch.float32)
        return z, z, z
    k_total = k_att + k_rep
    use_bf16_matmul = (mem_t.dtype == torch.bfloat16)

    L_att_sum = torch.zeros((), device=adapted_p.device, dtype=torch.float32)
    L_rep_sum = torch.zeros((), device=adapted_p.device, dtype=torch.float32)
    n_att_count = 0
    n_rep_count = 0

    for s in range(0, n_p, chunk_size):
        e = min(s + chunk_size, n_p)
        p_chunk = adapted_p[s:e]                                  # (cs, D) fp32
        p_norm_sq = (p_chunk * p_chunk).sum(dim=1, keepdim=True)  # (cs, 1) fp32

        # The one heavy kernel. Tensor cores on Ada/Ampere give a
        # ~16× speedup over fp32 cuBLAS at no cost in our loss.
        if use_bf16_matmul:
            pm = (p_chunk.bfloat16() @ mem_t).float()             # (cs, M)
        else:
            pm = p_chunk @ mem_t                                  # (cs, M) fp32

        # ‖p - m‖² = ‖p‖² + ‖m‖² - 2 p·m. Accumulation in fp32.
        d2 = p_norm_sq + mem_norm_sq.unsqueeze(0) - 2.0 * pm
        del pm
        # Floating-point cancellation can yield tiny negatives near zero;
        # clamp before topk. Gradient is flat through the clamp because
        # the loss is relu(d²-r²) / relu(r²+α-d²), and the clamp only
        # triggers where d² ≈ 0 < r² (inside the attraction zone where
        # the relu(d²-r²) gradient is zero anyway).
        d2 = d2.clamp_min_(0.0)
        topk_d2, _ = torch.topk(d2, k_total, dim=1, largest=False)
        d_att = topk_d2[:, :k_att]
        d_rep = topk_d2[:, k_att:k_att + k_rep]
        L_att_sum = L_att_sum + F.relu(d_att - r_sq).sum()
        L_rep_sum = L_rep_sum + F.relu(r_sq + alpha - d_rep).sum()
        n_att_count += d_att.numel()
        n_rep_count += d_rep.numel()
        del d2, topk_d2, p_norm_sq, p_chunk

    L_att = L_att_sum / max(n_att_count, 1)
    L_rep = L_rep_sum / max(n_rep_count, 1)
    return L_att + L_rep, L_att, L_rep


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_cfa(extractor: FeatureExtractor,
               descriptor: CFADescriptor,
               memory: torch.Tensor,
               records: list[ImageRecord],
               cfg: "RunConfig",
               device: torch.device) -> None:
    """Train the CFA descriptor.

    Performance design (vs. the naive per-iter rebuild):

      Each training iter needs an "adapted memory" tensor — descriptor
      applied to every memory bank entry, optionally transposed and
      cast to bf16 for the loss-time matmul. Recomputing all of that
      every iter costs ~100 ms per iter on M = 1.8e5 for DINOv2 (the
      MLP forward + the bf16 cast + the transpose + the ‖m‖² sum).

      The descriptor barely moves between consecutive iters at our
      learning rate, so we cache (mem_t, mem_norm_sq) and refresh only
      every `cfg.mem_refresh_every` iters. This is the same trick
      momentum encoders (MoCo, BYOL) use for the target network — it
      adds a slight target-staleness regularisation that empirically
      helps rather than hurts.

      Combined with bf16 matmul in cfa_loss (the heavy kernel) this
      collapses the per-iter cost from ~2.0 s to ~150-250 ms on an L4
      for the DINOv2 sweep, ≈10× faster.
    """
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

    optimizer = torch.optim.AdamW(descriptor.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))
    use_amp = (device.type == "cuda" and cfg.amp)
    # GradScaler is a no-op when the trainable forward is fp32/bf16
    # (the descriptor doesn't go through fp16). We keep it enabled
    # under cfg.amp only to stay close to the prior behaviour; with the
    # new bf16 matmul in cfa_loss the scale stays at 1 and the calls
    # have no effect on numerics.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"    [{now_hms()}] training: {n_epochs} epochs x "
          f"{iters_per_epoch} iters ({total_iters} total)  "
          f"bs={cfg.batch_size}  amp_bb={use_amp}  amp_loss={cfg.amp_loss}  "
          f"mem_refresh_every={cfg.mem_refresh_every}  "
          f"loss_chunk={cfg.loss_chunk}  "
          f"k_att={cfg.k_att} k_rep={cfg.k_rep} r²={cfg.radius_sq} "
          f"α={cfg.alpha}")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    extractor.eval()

    # ── Adapted-memory cache (refreshed every cfg.mem_refresh_every iters)
    # Owned by this function; freed when the per-class run returns.
    mem_t: torch.Tensor | None = None        # (D, M) bf16 or fp32
    mem_norm_sq: torch.Tensor | None = None  # (M,) fp32
    last_refresh_iter = -1
    refresh_dtype = (torch.bfloat16 if cfg.amp_loss else torch.float32)

    @torch.no_grad()
    def refresh_adapted_mem():
        """Recompute adapted memory bank + precomputed tensors for cfa_loss.
        Called every cfg.mem_refresh_every iters (and once at start)."""
        nonlocal mem_t, mem_norm_sq
        descriptor.eval()
        am_fp32 = descriptor(memory).detach()           # (M, D) fp32
        descriptor.train()
        # Precompute ‖m‖² in fp32 BEFORE casting — precision-sensitive.
        new_norm_sq = (am_fp32 * am_fp32).sum(dim=1)    # (M,) fp32
        # Pre-transpose and (optionally) cast: this avoids re-doing
        # both inside cfa_loss every iter.
        if cfg.amp_loss:
            new_mem_t = am_fp32.bfloat16().t().contiguous()  # (D, M) bf16
        else:
            new_mem_t = am_fp32.t().contiguous()             # (D, M) fp32
        del am_fp32
        # Free old buffers BEFORE assigning new ones (don't peak 2× memory).
        nonlocal_close = (mem_t, mem_norm_sq)
        mem_t = new_mem_t
        mem_norm_sq = new_norm_sq
        del nonlocal_close
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    refresh_adapted_mem()
    print(f"    [{now_hms()}] adapted memory cached: "
          f"mem_t {tuple(mem_t.shape)} {mem_t.dtype}  "
          f"({mem_t.element_size() * mem_t.numel() / 1e6:.1f} MB)")

    iter_idx = 0
    for epoch in range(n_epochs):
        descriptor.train()
        loss_sum = att_sum = rep_sum = 0.0
        n = 0
        for x in loader:
            if (iter_idx - last_refresh_iter) >= cfg.mem_refresh_every:
                refresh_adapted_mem()
                last_refresh_iter = iter_idx

            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                # Backbone forward — no grad through teacher.
                with torch.amp.autocast("cuda", enabled=use_amp):
                    maps = extractor(x, layers=cfg.feature_layers)
                pf = patchify_and_combine(maps,
                                            patch_size=cfg.patch_size,
                                            target_layer=cfg.target_layer)
                pf = pf.reshape(-1, pf.shape[-1]).float()      # (B*P, D)

            # Descriptor: grad through this side only.
            adapted_p = descriptor(pf)

            loss, L_att, L_rep = cfa_loss(
                adapted_p,
                mem_t=mem_t, mem_norm_sq=mem_norm_sq,
                k_att=cfg.k_att, k_rep=cfg.k_rep,
                r_sq=cfg.radius_sq, alpha=cfg.alpha,
                chunk_size=cfg.loss_chunk)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(descriptor.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            B = x.shape[0]
            loss_sum += loss.item() * B
            att_sum += L_att.item() * B
            rep_sum += L_rep.item() * B
            n += B
            iter_idx += 1

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={loss_sum/max(n,1):.4f}  "
                  f"L_att={att_sum/max(n,1):.4f}  "
                  f"L_rep={rep_sum/max(n,1):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    descriptor.eval()
    # Final refresh so memory_adapted (precomputed below in run_one_class)
    # is in sync with the trained descriptor.
    refresh_adapted_mem()
    print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")
    # Free the cached buffers — the precomputed adapted_mem for inference
    # is rebuilt by run_one_class right after training returns.
    del mem_t, mem_norm_sq
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Inference primitives
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _score_one_pass(extractor, descriptor, x, memory_adapted,
                     cfg, device) -> torch.Tensor:
    """One forward pass. Returns (B, H, W) per-pixel score on GPU."""
    use_amp = (device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        maps = extractor(x, layers=cfg.feature_layers)
    pf = patchify_and_combine(maps, patch_size=cfg.patch_size,
                                target_layer=cfg.target_layer).float()
    B, P, C = pf.shape
    H = W = int(math.isqrt(P))
    pf_flat = pf.reshape(-1, C)
    adapted_p = descriptor(pf_flat)             # (B*P, D)
    # Chunked cdist to keep memory bounded on bigger memory banks.
    n_q = adapted_p.shape[0]
    out = torch.empty(n_q, device=device, dtype=torch.float32)
    chunk = cfg.score_chunk
    M = memory_adapted.shape[0]
    for s in range(0, n_q, chunk):
        e = min(n_q, s + chunk)
        d2 = torch.cdist(adapted_p[s:e], memory_adapted) ** 2
        sd, _ = torch.sort(d2, dim=1)
        k = min(cfg.k_test, M)
        out[s:e] = sd[:, :k].mean(dim=1)
        del d2, sd
    score_lr = out.reshape(B, H, W)
    return score_lr


@torch.inference_mode()
def score_batch(extractor, descriptor, x, memory_adapted,
                  cfg, device) -> torch.Tensor:
    """TTA-aware. Returns (B, input_size, input_size) on CPU."""
    x = x.to(device, non_blocking=True)
    acc = None; n = 0
    def _add(s):
        nonlocal acc, n
        if acc is None: acc = s.clone()
        else: acc += s
        n += 1

    s = _score_one_pass(extractor, descriptor, x, memory_adapted, cfg, device)
    _add(s)
    if cfg.tta in ("hflip", "hvflip", "d4"):
        s2 = _score_one_pass(extractor, descriptor,
                              torch.flip(x, dims=[-1]),
                              memory_adapted, cfg, device)
        _add(torch.flip(s2, dims=[-1]))
    if cfg.tta in ("vflip", "hvflip", "d4"):
        s2 = _score_one_pass(extractor, descriptor,
                              torch.flip(x, dims=[-2]),
                              memory_adapted, cfg, device)
        _add(torch.flip(s2, dims=[-2]))
    if cfg.tta == "d4":
        for k in (1, 2, 3):
            s2 = _score_one_pass(extractor, descriptor,
                                   torch.rot90(x, k=k, dims=[-2, -1]),
                                   memory_adapted, cfg, device)
            _add(torch.rot90(s2, k=-k, dims=[-2, -1]))

    avg = acc / max(n, 1)
    up = F.interpolate(avg.unsqueeze(1),
                       size=(cfg.input_size, cfg.input_size),
                       mode="bilinear", align_corners=False).squeeze(1)
    return up.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _score_records(extractor, descriptor, memory_adapted,
                    records, cfg, device, load_masks):
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores, gts = {}, {}
    n_done = 0; last_log = 0
    for x, masks, idxs in loader:
        sm = score_batch(extractor, descriptor, x,
                           memory_adapted, cfg, device).numpy()
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

    # ── Memory bank
    memory, feature_hw, feature_dim = init_memory_bank(
        extractor, train_good, cfg, device)

    # ── Descriptor
    descriptor = CFADescriptor(
        in_dim=feature_dim,
        hidden_dim=cfg.hidden_dim if cfg.hidden_dim > 0 else None,
        use_bn=not cfg.no_bn,
    ).to(device)
    n_params = sum(p.numel() for p in descriptor.parameters())
    print(f"  CFA: descriptor dim={feature_dim}  "
          f"hidden={descriptor.net[0].out_features}  "
          f"params={n_params/1e6:.2f}M")

    # ── Train
    train_cfa(extractor, descriptor, memory, train_good, cfg, device)

    # ── Precompute adapted memory (constant at inference time)
    with torch.inference_mode():
        descriptor.eval()
        memory_adapted = descriptor(memory).contiguous()
    print(f"    [{now_hms()}] precomputed adapted memory "
          f"shape={tuple(memory_adapted.shape)}")

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_cfa.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"descriptor": descriptor.state_dict(),
                    "memory": memory.cpu(),
                    "memory_adapted": memory_adapted.cpu()}, ck)
        print(f"    saved checkpoint -> {ck}")

    # ── Local eval
    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  tta={cfg.tta}  "
            f"k_test={cfg.k_test}  patch={cfg.patch_size}")
        scores, gts = _score_records(extractor, descriptor, memory_adapted,
                                        train_anom, cfg, device,
                                        load_masks=True)
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

    # ── Test scoring
    test_results: list[tuple[ImageRecord, np.ndarray]] = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images  tta={cfg.tta}")
        scores, _ = _score_records(extractor, descriptor, memory_adapted,
                                      test, cfg, device, load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            sm_final = maybe_resize_to_submission(sm_smooth)
            test_results.append((test[r_idx], sm_final))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del descriptor, memory, memory_adapted
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
    target_layer: int = 2
    input_size: int = 384
    patch_size: int = 1
    # Memory bank
    coreset_frac: float = 0.05
    coreset_algo: str = "minibatch"
    coreset_batch: int = 128
    coreset_fp16: bool = True
    project_chunk: int = 65536
    fit_batch_size: int = 16
    # Descriptor
    hidden_dim: int = 0          # 0 → same as feature dim
    no_bn: bool = False
    # Training
    epochs: int = 200
    total_iters: int | None = 2500
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    k_att: int = 3
    k_rep: int = 3
    radius_sq: float = 0.5
    alpha: float = 0.5
    amp: bool = True
    amp_loss: bool = True        # bf16 matmul inside cfa_loss (Ada/Ampere tensor cores)
    num_workers: int = 8
    # Loss-time memory control
    loss_chunk: int = 8192       # query patches per chunk of the d² matmul
    mem_refresh_every: int = 10  # refresh cached adapted_mem every N iters
    # Inference
    score_batch_size: int = 16
    score_chunk: int = 4096
    k_test: int = 1
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
        "method": "cfa",
        "backbone": cfg.backbone,
        "feature_layers": list(cfg.feature_layers),
        "target_layer": cfg.target_layer,
        "input_size": cfg.input_size,
        "patch_size": cfg.patch_size,
        "coreset_frac": cfg.coreset_frac,
        "hidden_dim": cfg.hidden_dim,
        "k_att": cfg.k_att,
        "k_rep": cfg.k_rep,
        "radius_sq": cfg.radius_sq,
        "alpha": cfg.alpha,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "k_test": cfg.k_test,
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
    bits = (f"{stamp}_cfa_{bb}_L{L}_T{cfg.target_layer}_in{cfg.input_size}"
            f"_p{cfg.patch_size}_cs{int(cfg.coreset_frac*100):02d}"
            f"_{budget}_bs{cfg.batch_size}"
            f"_ka{cfg.k_att}kr{cfg.k_rep}_r{cfg.radius_sq:g}a{cfg.alpha:g}"
            f"_kt{cfg.k_test}")
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
    ap.add_argument("--feature-layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--target-layer", type=int, default=2)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--patch-size", type=int, default=1,
                    help="1 = raw patches (CFA paper). 3 = PatchCore-style "
                         "3x3 local aggregation.")
    # Memory bank
    ap.add_argument("--coreset-frac", type=float, default=0.05)
    ap.add_argument("--coreset-algo", default="minibatch",
                    choices=["exact", "minibatch"])
    ap.add_argument("--coreset-batch", type=int, default=128)
    ap.add_argument("--no-coreset-fp16", action="store_true")
    ap.add_argument("--project-chunk", type=int, default=65536)
    ap.add_argument("--fit-batch-size", type=int, default=16)
    # Descriptor
    ap.add_argument("--hidden-dim", type=int, default=0,
                    help="0 = same as feature dim. Set >0 to override.")
    ap.add_argument("--no-bn", action="store_true",
                    help="Drop BatchNorm in the descriptor MLP.")
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=2500,
                    help="Overrides --epochs (matches other baselines).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--k-att", type=int, default=3,
                    help="Number of nearest memory neighbours treated "
                         "as attractive in the contrastive loss.")
    ap.add_argument("--k-rep", type=int, default=3,
                    help="Number of next-nearest memory neighbours "
                         "treated as repulsive.")
    ap.add_argument("--radius-sq", type=float, default=0.5,
                    help="Squared hypersphere radius r². Attractive "
                         "neighbours should have squared distance < r²; "
                         "repulsive ones should have > r² + alpha.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Repulsion margin.")
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable AMP for the backbone forward.")
    ap.add_argument("--no-amp-loss", action="store_true",
                    help="Disable bf16 tensor-core matmul inside "
                         "cfa_loss. The bf16 path is the single biggest "
                         "speedup on L4/A100 (~10× on the cdist-style "
                         "matmul); disable only if you suspect numeric "
                         "issues for your config.")
    ap.add_argument("--mem-refresh-every", type=int, default=10,
                    help="Refresh the cached descriptor(memory) every "
                         "N training iters. Default 10 -> ~10× fewer "
                         "M=178K MLP forwards on the memory bank. "
                         "Set 1 to recover the per-iter behaviour; "
                         "set 20-50 for further speedup with a slight "
                         "target-staleness regulariser effect.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--loss-chunk", type=int, default=8192,
                    help="Number of query patches processed per chunk "
                         "when computing the (N, M) d² matrix in "
                         "cfa_loss. Peak GPU memory of the distance "
                         "buffer is ~2*chunk*M bytes (bf16 path) or "
                         "~4*chunk*M (fp32 path). Lower this if you "
                         "OOM (e.g. 4096 or 2048); raise it for a "
                         "small speedup if you have headroom.")
    # Inference
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--score-chunk", type=int, default=4096)
    ap.add_argument("--k-test", type=int, default=1,
                    help="Mean over the K nearest memory distances at "
                         "inference. k=1 matches PatchCore-NN; k=3-9 "
                         "smooths the score.")
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
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    feature_layers = tuple(sorted(set(args.feature_layers)))
    if args.target_layer not in feature_layers:
        raise SystemExit(
            f"[FATAL] --target-layer {args.target_layer} must be in "
            f"--feature-layers {list(feature_layers)}")
    for l in feature_layers:
        if l not in BACKBONE_CHANNELS[args.backbone]:
            raise SystemExit(
                f"[FATAL] feature_layer {l} invalid for {args.backbone}")
    if args.backbone in DINOV2_BACKBONES:
        if args.input_size % 14 != 0:
            raise SystemExit(
                f"[FATAL] DINOv2 needs --input-size divisible by 14; "
                f"got {args.input_size}.")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone,
        feature_layers=feature_layers,
        target_layer=args.target_layer,
        input_size=args.input_size,
        patch_size=args.patch_size,
        coreset_frac=args.coreset_frac,
        coreset_algo=args.coreset_algo,
        coreset_batch=args.coreset_batch,
        coreset_fp16=not args.no_coreset_fp16,
        project_chunk=args.project_chunk,
        fit_batch_size=args.fit_batch_size,
        hidden_dim=args.hidden_dim, no_bn=args.no_bn,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        k_att=args.k_att, k_rep=args.k_rep,
        radius_sq=args.radius_sq, alpha=args.alpha,
        amp=not args.no_amp, amp_loss=not args.no_amp_loss,
        num_workers=args.num_workers,
        loss_chunk=args.loss_chunk,
        mem_refresh_every=args.mem_refresh_every,
        score_batch_size=args.score_batch_size,
        score_chunk=args.score_chunk,
        k_test=args.k_test,
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

    # Enable TF32 tensor cores for any fp32 matmul that slips through
    # (e.g. with --no-amp-loss). On Ada/Ampere this gives ~5-10× over
    # full-precision fp32 cuBLAS at essentially zero accuracy cost for
    # this kind of distance computation.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    with tee_to(run_dir / "run_log.txt"):
        hr(f"CFA — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  backbone         : {cfg.backbone}")
        print(f"  feature_layers   : {list(cfg.feature_layers)}")
        print(f"  target_layer     : {cfg.target_layer}")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  patch_size       : {cfg.patch_size}")
        print(f"  coreset          : frac={cfg.coreset_frac:.1%}  "
              f"algo={cfg.coreset_algo}  batch={cfg.coreset_batch}")
        print(f"  descriptor       : hidden_dim={cfg.hidden_dim or 'auto'}  "
              f"BN={not cfg.no_bn}")
        print(f"  loss             : k_att={cfg.k_att}  k_rep={cfg.k_rep}  "
              f"r²={cfg.radius_sq}  α={cfg.alpha}")
        if cfg.total_iters and cfg.total_iters > 0:
            print(f"  total_iters      : {cfg.total_iters}")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / wd          : {cfg.lr} / {cfg.weight_decay}")
        print(f"  amp (backbone)   : {cfg.amp}")
        print(f"  amp_loss (bf16)  : {cfg.amp_loss}  "
              f"(tensor-core matmul in cfa_loss)")
        print(f"  mem_refresh_every: {cfg.mem_refresh_every}  "
              f"(iters between descriptor(memory) refreshes)")
        print(f"  loss_chunk       : {cfg.loss_chunk} "
              f"(peak d² buffer ≈ {2 if cfg.amp_loss else 4} * chunk * M bytes)")
        print(f"  k_test (NN)      : {cfg.k_test}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(cfg).items()}, f, indent=2)

        extractor = FeatureExtractor(cfg.backbone).to(device).eval()

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
            res = run_one_class(cls, records, extractor, cfg, run_dir,
                                  device, local_saver=local_saver)
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
            "backbone": f"CFA_{BACKBONE_SHORT.get(cfg.backbone, cfg.backbone).upper()}",
            "feature_layers": "+".join(str(l) for l in cfg.feature_layers),
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
            "notes": (f"cfa {BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)} "
                      f"p{cfg.patch_size} cs{cfg.coreset_frac:.2f} "
                      f"ka{cfg.k_att}kr{cfg.k_rep} "
                      f"r{cfg.radius_sq}a{cfg.alpha} "
                      f"k_test{cfg.k_test} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()