"""Spacepresso AnomalyDINO baseline.

Implements "AnomalyDINO: Boosting Patch-based Few-Shot Anomaly Detection
with DINOv2" (Damm et al., WACV 2025), adapted to Spacepresso.

# Why this is worth adding to the ensemble

  Your ensemble currently has TWO DINOv2-based memory-bank methods:
  PatchCore (exp7) and CFA. Both fuse multi-scale features and either
  coreset-reduce (PatchCore) or learn a discriminative projection (CFA).

  AnomalyDINO operates on a fundamentally different recipe:

    1. SINGLE LATE BLOCK (default: last block of vits14).
       Late DINOv2 blocks encode semantic identity ("this is coffee bean
       n. 27 from this angle") more than texture. PatchCore-DINOv2 uses
       mid+late blocks fused and loses some of this semantic signal
       through the AvgPool/concat.

    2. NO CORESET REDUCTION — keep ALL train_good patches.
       PatchCore's coreset selects ~5% of patches in feature space, which
       can drop "rare but legitimate" normal patches (legitimate edge
       cases). AnomalyDINO keeps everything; the bank is small enough
       per class (30 imgs × 784 patches × 384 dim ≈ 18 MB fp16) that
       brute-force NN search is fast on a 4090.

    3. FOREGROUND MASKING via first PCA component.
       DINOv2 features cluster into "background-like" and "object-like"
       in the first PC direction; thresholding at a percentile filters
       background patches from the bank AND from per-pixel queries. This
       single trick gives ~2-4% AP on classes with heterogeneous
       backgrounds (analysis §4 lists class_03 / class_07 as candidates).
       Made optional via `--no-foreground-mask`.

    4. k-NN MEAN scoring (default k=1 → identical to PatchCore-style
       min). With k=3-5, the mean of top-k distances is more robust to
       single bad bank neighbours (over-fit train_good patches).

  Net: AnomalyDINO is a complementary track, not a competitor — its
  failure modes (semantic-only block, no cross-scale info, k-NN robust
  averaging) differ from PatchCore's. Stacker should weight it
  separately per class.

# Memory / speed on a 4090

  vits14 @ input 392, ~30 train_good imgs per class:
    Feature extraction (incl. AMP):  ~5 s per class
    Foreground mask fit:              <1 s per class
    NN scoring (test, fp16, chunked): ~15 s per class
  Full 8 classes:                     ~3 min end-to-end. No training.

# Output (stacker contract — same as every other baseline)

  $RUN/submission.csv, $RUN/local_predictions.npz, $RUN/test_predictions.npz

# Dependencies

  patchcore_baseline_v2.py + local_preds_saver.py + dinov3_loader.py +
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
    pixel_average_precision, gaussian_smooth,
    calibrate_to_unit, float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
    IMAGENET_MEAN, IMAGENET_STD,
    BACKBONE_SHORT,
    DINO_BACKBONES, DINOV2_BACKBONES,
    DINOV2_NBLOCKS, DINOV3_VIT_SPECS, DINOV3_NBLOCKS,
    DINOV3_CONVNEXT_SPECS,
    backbone_patch_size,
)
from local_preds_saver import LocalPredSaver
from dinov3_loader import (
    load_dino_backbone, get_patch_tokens_at_layers, is_dino_backbone,
)


PROJECT_ROOT = Path("/workspace/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger (same idiom as the rest of the codebase)
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
# DINO feature extractor (single block, L2-normalised patches)
# ─────────────────────────────────────────────────────────────────────────────
class DinoSingleBlockExtractor(nn.Module):
    """Wraps the unified DINOv2/v3 loader and returns patch features from
    ONE block reshaped to (B, P, C) with optional L2 norm.

    AnomalyDINO's distance metric assumes L2-normalised features so that
    Euclidean ≡ angular distance: ||a - b||^2 = 2(1 - cos(a, b))."""

    def __init__(self, backbone: str, block_idx: int | None,
                 l2_normalise: bool = True):
        super().__init__()
        if not is_dino_backbone(backbone):
            raise ValueError(
                f"AnomalyDINO requires a DINO backbone, got {backbone}.")
        if backbone in DINOV3_CONVNEXT_SPECS:
            raise ValueError(
                f"ConvNeXt backbones not supported here — use a ViT "
                f"(e.g. dinov2_vits14 / dinov3_vitb16).")
        model, info = load_dino_backbone(backbone)
        model.eval()
        if block_idx is None:
            block_idx = info.n_blocks - 1   # last block = most semantic
        if not (0 <= block_idx < info.n_blocks):
            raise ValueError(
                f"block_idx={block_idx} out of range [0, {info.n_blocks-1}] "
                f"for {backbone}")
        self.model = model
        self.info = info
        self.block_idx = int(block_idx)
        self.l2_normalise = l2_normalise
        self.embed_dim = info.embed_dim
        self.patch_size = info.patch_size
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, P, C) patch features where P = (H/patch) * (W/patch)."""
        feats = get_patch_tokens_at_layers(
            self.model, self.info, x, [self.block_idx])
        f = feats[self.block_idx]                 # (B, C, h, w)
        B, C, h, w = f.shape
        out = f.permute(0, 2, 3, 1).reshape(B, h * w, C).contiguous()
        if self.l2_normalise:
            out = F.normalize(out, p=2, dim=-1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Foreground mask via first-PC sign on patch features
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def fit_foreground_pca(bank_patches_gpu: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean and first principal component of the patch-feature
    cloud. Used to score (mean-centred) projection of any future patch;
    a percentile threshold then separates foreground from background.

    Args:
        bank_patches_gpu : (N_patches, C) L2-normed features.
    Returns:
        mean : (C,)
        pc1  : (C,)  first right-singular vector (unit norm)
    """
    mean = bank_patches_gpu.mean(dim=0)
    centred = bank_patches_gpu - mean
    # Subsample for SVD on huge banks — full SVD is O(N C^2), and
    # for N > ~50k it's pointless precision; 20k samples gives a stable
    # principal direction.
    n = centred.shape[0]
    if n > 20000:
        idx = torch.randperm(n, device=centred.device)[:20000]
        sub = centred[idx]
    else:
        sub = centred
    # torch.linalg.svd: V is (C, C); rows of V are right-singular vectors.
    # Use lowrank for speed and memory.
    _, _, V = torch.svd_lowrank(sub, q=1, niter=4)
    pc1 = V[:, 0]
    pc1 = pc1 / (pc1.norm() + 1e-9)
    return mean, pc1


@torch.inference_mode()
def project_onto_pc1(feats: torch.Tensor, mean: torch.Tensor,
                       pc1: torch.Tensor) -> torch.Tensor:
    """feats: (..., C); returns (..., ) scalar projection."""
    return ((feats - mean) * pc1).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyDINO core: build bank + score
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyDINO:
    """Per-class memory bank of L2-normalised foreground patches plus a
    k-NN scoring head. Training-free; `fit()` is feature extraction."""

    def __init__(self, extractor: DinoSingleBlockExtractor,
                  device: torch.device,
                  knn_k: int = 1,
                  use_foreground_mask: bool = True,
                  foreground_keep_pct: float = 75.0,
                  bank_dtype: torch.dtype = torch.float16,
                  chunk_queries: int = 4096,
                  chunk_bank: int = 32768):
        self.extractor = extractor
        self.device = device
        self.knn_k = max(1, int(knn_k))
        self.use_foreground_mask = use_foreground_mask
        self.foreground_keep_pct = float(foreground_keep_pct)
        self.bank_dtype = bank_dtype
        self.chunk_queries = chunk_queries
        self.chunk_bank = chunk_bank

        self.bank: torch.Tensor | None = None     # (N_kept, C), bank_dtype
        self.fg_mean: torch.Tensor | None = None
        self.fg_pc1: torch.Tensor | None = None
        self.fg_threshold: float | None = None
        self.feature_hw: tuple[int, int] | None = None

    # ── fit ─────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def fit(self, train_good: list[ImageRecord], cfg: "RunConfig") -> None:
        ds = TrainGoodDataset(train_good, input_size=cfg.input_size)
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
            worker_init_fn=worker_init_fn)
        use_amp = (self.device.type == "cuda" and cfg.amp)
        all_patches: list[torch.Tensor] = []
        n_imgs = 0
        t0 = time.time()
        for x in loader:
            x = x.to(self.device, non_blocking=True)
            B, _, H, W = x.shape
            with torch.amp.autocast("cuda", enabled=use_amp):
                f = self.extractor(x)             # (B, P, C)
            f = f.float()                          # collapse AMP for safety
            P = f.shape[1]
            h = w = int(math.isqrt(P))
            self.feature_hw = (h, w)
            all_patches.append(f.reshape(-1, f.shape[-1]).cpu())
            n_imgs += B
        bank_cpu = torch.cat(all_patches, dim=0)
        del all_patches
        bank_gpu = bank_cpu.to(self.device, dtype=torch.float32,
                                 non_blocking=True)
        del bank_cpu
        print(f"    [{now_hms()}] extracted {bank_gpu.shape[0]} patches "
              f"from {n_imgs} train_good imgs "
              f"({time.time() - t0:.1f}s)  "
              f"feature_hw={self.feature_hw}  C={bank_gpu.shape[1]}")

        # Foreground masking fit.
        if self.use_foreground_mask:
            self.fg_mean, self.fg_pc1 = fit_foreground_pca(bank_gpu)
            proj = project_onto_pc1(bank_gpu, self.fg_mean, self.fg_pc1)
            # Take the side of the projection axis that has the smaller
            # cluster as "foreground"; the larger one is more likely a
            # uniform background. We pick the absolute-value tail above
            # the (100 - keep_pct) percentile of |proj|. This is sign-
            # invariant — no need to guess which side is the object.
            abs_proj = proj.abs()
            q = float(np.percentile(abs_proj.cpu().numpy(),
                                       100.0 - self.foreground_keep_pct))
            self.fg_threshold = q
            keep = (abs_proj >= q)
            n_keep = int(keep.sum().item())
            print(f"    foreground mask: keep_pct={self.foreground_keep_pct:.1f}%"
                  f"  abs(PC1) threshold={q:.4f}  "
                  f"keeping {n_keep}/{bank_gpu.shape[0]} patches")
            bank_kept = bank_gpu[keep]
        else:
            bank_kept = bank_gpu
            print(f"    foreground mask DISABLED — bank size "
                  f"{bank_kept.shape[0]}")

        # Final L2 norm + dtype cast.
        bank_kept = F.normalize(bank_kept, p=2, dim=-1)
        self.bank = bank_kept.to(self.bank_dtype).contiguous()
        del bank_gpu, bank_kept
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        mb = self.bank.element_size() * self.bank.numel() / 1e6
        print(f"    bank ready  shape={tuple(self.bank.shape)}  "
              f"dtype={self.bank.dtype}  ({mb:.1f} MB on GPU)")

    # ── single-pass score ───────────────────────────────────────────────────
    @torch.inference_mode()
    def _score_one_pass(self, x: torch.Tensor, cfg: "RunConfig"
                          ) -> torch.Tensor:
        """Returns (B, P) anomaly score = mean of top-k distances per patch.

        x is already on self.device."""
        assert self.bank is not None
        use_amp = (self.device.type == "cuda" and cfg.amp)
        with torch.amp.autocast("cuda", enabled=use_amp):
            q = self.extractor(x)                # (B, P, C)
        q = q.float()
        B, P, C = q.shape
        flat = q.reshape(-1, C)
        N_q = flat.shape[0]
        bank = self.bank                          # (M, C) bank_dtype
        M = bank.shape[0]
        k = min(self.knn_k, M)

        out_dist = torch.empty(N_q, k, device=self.device,
                                dtype=torch.float32)
        for s in range(0, N_q, self.chunk_queries):
            e = min(N_q, s + self.chunk_queries)
            q_chunk = flat[s:e].to(self.bank_dtype)
            # Track current top-k SIMILARITY (highest cos sim = nearest).
            best_sim = torch.full((e - s, k), -2.0, device=self.device,
                                    dtype=self.bank_dtype)
            for ms in range(0, M, self.chunk_bank):
                me = min(M, ms + self.chunk_bank)
                sim_chunk = q_chunk @ bank[ms:me].T      # (e-s, me-ms)
                # Combine with existing best, take top-k again.
                cat = torch.cat([best_sim, sim_chunk], dim=1)
                best_sim = cat.topk(k, dim=1, largest=True).values
                del sim_chunk, cat
            dist = 1.0 - best_sim.float()          # Euclidean^2 / 2 ≅ cos-dist
            out_dist[s:e] = dist
            del q_chunk, best_sim

        # Score = mean of top-k distances (k=1 → identical to PatchCore-style).
        score_flat = out_dist.mean(dim=1)
        return score_flat.reshape(B, P)

    # ── TTA-aware score with optional upsample ──────────────────────────────
    @torch.inference_mode()
    def score_batch(self, x: torch.Tensor, cfg: "RunConfig") -> torch.Tensor:
        """Returns (B, input_size, input_size) on CPU, float32."""
        x = x.to(self.device, non_blocking=True)
        acc = None; n = 0
        def _add(s):
            nonlocal acc, n
            if acc is None: acc = s.clone()
            else: acc += s
            n += 1

        s0 = self._score_one_pass(x, cfg)
        _add(s0)
        if cfg.tta in ("hflip", "hvflip"):
            s = self._score_one_pass(torch.flip(x, dims=[-1]), cfg)
            # Patch grid is row-major, P = h*w. To "unflip" horizontally
            # we reshape to (B, h, w), flip width, flatten.
            B, P = s.shape
            h, w = self.feature_hw  # type: ignore[misc]
            _add(torch.flip(s.reshape(B, h, w), dims=[-1]).reshape(B, P))
        if cfg.tta in ("vflip", "hvflip"):
            s = self._score_one_pass(torch.flip(x, dims=[-2]), cfg)
            B, P = s.shape
            h, w = self.feature_hw  # type: ignore[misc]
            _add(torch.flip(s.reshape(B, h, w), dims=[-2]).reshape(B, P))

        score_flat = acc / max(n, 1)                # (B, P)
        B, P = score_flat.shape
        h, w = self.feature_hw  # type: ignore[misc]
        lr = score_flat.reshape(B, h, w)
        up = F.interpolate(lr.unsqueeze(1),
                            size=(cfg.input_size, cfg.input_size),
                            mode="bilinear", align_corners=False).squeeze(1)
        return up.cpu().float()


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline + submission writer
# ─────────────────────────────────────────────────────────────────────────────
def _score_records(model: AnomalyDINO, records: list[ImageRecord],
                    cfg: "RunConfig", load_masks: bool
                    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    scores, gts = {}, {}
    n_done = 0; last_log = 0
    for x, masks, idxs in loader:
        sm = model.score_batch(x, cfg).numpy()
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
                   extractor: DinoSingleBlockExtractor,
                   cfg: "RunConfig", run_dir: Path, device: torch.device,
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

    model = AnomalyDINO(
        extractor=extractor, device=device,
        knn_k=cfg.knn_k,
        use_foreground_mask=cfg.use_foreground_mask,
        foreground_keep_pct=cfg.foreground_keep_pct,
        bank_dtype=(torch.float16 if cfg.bank_dtype == "fp16" else torch.float32),
        chunk_queries=cfg.score_chunk,
        chunk_bank=cfg.memory_chunk,
    )
    model.fit(train_good, cfg)

    if cfg.save_banks:
        bp = run_dir / "banks" / f"{cls}_bank.pt"
        bp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "bank": model.bank.cpu(),
            "feature_hw": model.feature_hw,
            "block_idx": cfg.block_idx,
            "fg_mean": (model.fg_mean.cpu() if model.fg_mean is not None
                         else None),
            "fg_pc1": (model.fg_pc1.cpu() if model.fg_pc1 is not None
                        else None),
            "fg_threshold": model.fg_threshold,
        }, bp)
        print(f"    saved bank -> {bp}")

    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  tta={cfg.tta}  k={cfg.knn_k}  "
            f"fg_mask={cfg.use_foreground_mask}")
        scores, gts = _score_records(model, train_anom, cfg, load_masks=True)
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
        scores, _ = _score_records(model, test, cfg, load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            sm_final = maybe_resize_to_submission(sm_smooth)
            test_results.append((test[r_idx], sm_final))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    # Free bank before next class.
    model.bank = None
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
    backbone: str = "dinov2_vits14"
    block_idx: int | None = None         # default: last block
    input_size: int = 392
    knn_k: int = 1
    use_foreground_mask: bool = True
    foreground_keep_pct: float = 75.0    # keep top 75% by |PC1|
    bank_dtype: str = "fp16"
    score_chunk: int = 4096
    memory_chunk: int = 32768
    batch_size: int = 8
    score_batch_size: int = 8
    num_workers: int = 8
    amp: bool = True
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_banks: bool = False
    zip_submission: bool = True
    run_tag: str = ""


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "anomalydino",
        "backbone": cfg.backbone,
        "block_idx": cfg.block_idx,
        "input_size": cfg.input_size,
        "knn_k": cfg.knn_k,
        "use_foreground_mask": cfg.use_foreground_mask,
        "foreground_keep_pct": cfg.foreground_keep_pct,
        "bank_dtype": cfg.bank_dtype,
        "tta": cfg.tta,
        "smooth_sigma": cfg.smooth_sigma,
        "seed": cfg.seed,
        "v": 1,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bb = BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)
    bits = (f"{stamp}_adino_{bb}"
            + (f"_b{cfg.block_idx}" if cfg.block_idx is not None else "_blast")
            + f"_in{cfg.input_size}_k{cfg.knn_k}"
            + (f"_fg{int(cfg.foreground_keep_pct)}"
                if cfg.use_foreground_mask else "_noFG"))
    if cfg.tta != "none": bits += f"_tta-{cfg.tta}"
    if cfg.run_tag:
        bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--backbone", default="dinov2_vits14",
                    help="Any DINOv2/v3 ViT backbone "
                         "(dinov2_vits14, dinov2_vitb14_reg, "
                         "dinov3_vitb16, ...). ConvNeXt not supported.")
    ap.add_argument("--block-idx", type=int, default=None,
                    help="Which transformer block to extract from. "
                         "Default: last block. AnomalyDINO paper picks "
                         "block 11 of vits14 (the last); for vitl16 the "
                         "last is block 23.")
    ap.add_argument("--input-size", type=int, default=392,
                    help="Multiple of 14 (DINOv2) or 16 (DINOv3). "
                         "392 -> 28x28 = 784 tokens; 518 -> 37x37 = 1369.")
    ap.add_argument("--knn-k", type=int, default=1,
                    help="Number of nearest neighbours to AVERAGE over for "
                         "the per-patch anomaly score. k=1 is identical to "
                         "PatchCore-style min-distance; k=3-5 is robust to "
                         "individual bad neighbours.")
    ap.add_argument("--no-foreground-mask", action="store_true",
                    help="Disable the first-PC foreground filter. Use this "
                         "for classes with very heterogeneous foreground "
                         "(e.g. the background already IS the object).")
    ap.add_argument("--foreground-keep-pct", type=float, default=75.0,
                    help="When --foreground-mask is on, keep this %% of "
                         "patches (those with the largest |PC1| projection). "
                         "Lower = more aggressive background removal.")
    ap.add_argument("--bank-dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--score-chunk", type=int, default=4096)
    ap.add_argument("--memory-chunk", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--score-batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--save-banks", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    # Patch-size sanity.
    if not is_dino_backbone(args.backbone):
        raise SystemExit(f"[FATAL] {args.backbone} is not a DINO backbone.")
    if args.backbone in DINOV3_CONVNEXT_SPECS:
        raise SystemExit("[FATAL] ConvNeXt backbones not supported here.")
    ps = backbone_patch_size(args.backbone)
    if ps and args.input_size % ps != 0:
        raise SystemExit(
            f"[FATAL] --input-size {args.input_size} not divisible by "
            f"{ps} (required by {args.backbone}).")
    if args.backbone in DINOV2_BACKBONES:
        nb = DINOV2_NBLOCKS[args.backbone]
    else:
        nb = DINOV3_NBLOCKS[args.backbone]
    if args.block_idx is not None and not (0 <= args.block_idx < nb):
        raise SystemExit(
            f"[FATAL] block_idx {args.block_idx} out of range "
            f"[0, {nb-1}] for {args.backbone}.")
    if not (0 < args.foreground_keep_pct <= 100):
        raise SystemExit("[FATAL] --foreground-keep-pct must be in (0, 100].")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone, block_idx=args.block_idx,
        input_size=args.input_size,
        knn_k=args.knn_k,
        use_foreground_mask=not args.no_foreground_mask,
        foreground_keep_pct=args.foreground_keep_pct,
        bank_dtype=args.bank_dtype,
        score_chunk=args.score_chunk, memory_chunk=args.memory_chunk,
        batch_size=args.batch_size,
        score_batch_size=args.score_batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        save_banks=args.save_banks,
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
        hr(f"ANOMALYDINO — RUN {run_id}", "█")
        print(f"  backbone         : {cfg.backbone}")
        print(f"  block_idx        : {cfg.block_idx} "
              f"({'last' if cfg.block_idx is None else 'explicit'})")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  knn_k            : {cfg.knn_k}")
        print(f"  foreground_mask  : {cfg.use_foreground_mask}  "
              f"(keep_pct={cfg.foreground_keep_pct})")
        print(f"  bank_dtype       : {cfg.bank_dtype}")
        print(f"  tta              : {cfg.tta}    "
              f"smooth_sigma={cfg.smooth_sigma}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (str(v) if isinstance(v, (Path, torch.device))
                            else v)
                       for k, v in asdict(cfg).items()}, f, indent=2,
                       default=str)

        extractor = DinoSingleBlockExtractor(
            backbone=cfg.backbone, block_idx=cfg.block_idx,
            l2_normalise=True).to(device).eval()
        cfg_block = (extractor.block_idx if cfg.block_idx is None
                      else cfg.block_idx)
        print(f"\n  resolved block_idx={cfg_block}  embed_dim={extractor.embed_dim}")

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): {', '.join(classes)}")

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
            "backbone": f"ADINO_{bb_short.upper()}",
            "feature_layers": f"block_{cfg_block}",
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
            "notes": (f"anomalydino backbone={cfg.backbone} "
                      f"block={cfg_block} k={cfg.knn_k} "
                      f"fg_mask={cfg.use_foreground_mask}"
                      f"({cfg.foreground_keep_pct:.0f}%) "
                      f"tta={cfg.tta}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()