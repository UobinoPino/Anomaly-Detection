"""Spacepresso DINO-DPMM baseline.

Dirichlet Process (truncated) Gaussian Mixture Model fitted on
DINOv2/v3 patch features. Per-pixel anomaly score = -log p(x) under
the fitted DPMM. Implemented with sklearn's BayesianGaussianMixture
under a stick-breaking weight concentration prior, which is exactly
the standard variational truncated-DP-GMM.

# Why DPMM on DINO features is a worthwhile ensemble member

  Your stack already has several "what does normal look like?" signals:

    - PatchCore / AnomalyDINO  : k-NN distance in raw DINO feature
                                  space. Discrete, exemplar-based.
    - FastFlow                 : exact normalizing-flow density on
                                  ResNet/DINO features. Bijective,
                                  high capacity, but a SINGLE flow.
    - UniAD                    : transformer reconstruction MSE. Not
                                  a density at all.
    - Reverse Distillation     : teacher-student cosine; distance, not
                                  density.
    - EfficientAD              : student-teacher MSE + AE MSE; distance.

  DPMM contributes a fundamentally different inductive bias:

    * Non-parametric Bayesian mixture: the model learns its own number
      of "normal modes" via the stick-breaking prior. Spacepresso has
      5 camera views per sample, so each class has at least 5 distinct
      modes of "normal" texture — that's exactly what DPMM models well.
    * Closed-form likelihood: -log p(x) has bounded, well-behaved
      gradients; no exp/log instabilities like FastFlow's coupling.
    * Cheap to fit: variational inference, ~30s per class on CPU.
      Inference is just one matrix-vector op per pixel.

  Expected behaviour vs FastFlow: DPMM gives cleaner localisation
  (sharper boundary at the edge of normal-mode support) at the cost of
  less expressive capacity. Different error modes → useful at stacking.

# Pipeline

  Per class:
    1. Extract DINOv2/v3 patch features for train_good (single or
       fused multi-block).
    2. Fit PCA to dim=`pca_dim` on the patch cloud (default 64).
       PCA reduces the GMM's per-component covariance from O(D^2) ~
       140K parameters per component to ~64. Without PCA the DPMM
       overfits the small train_good pool.
    3. Fit a BayesianGaussianMixture with n_components=`max_components`
       (truncation level) under `weight_concentration_prior_type =
       'dirichlet_process'`. The stick-breaking prior shrinks unused
       components toward zero weight; the EFFECTIVE number of modes is
       usually much smaller than `max_components`.
    4. Inference: project test patches through PCA → log_prob under
       DPMM → anomaly score = -log_prob (per pixel).

  Multi-block fusion is optional and OFF by default — keeping it
  single-block makes DPMM's diversity vs PatchCore (which fuses) more
  meaningful.

# Memory / speed on a 4090

  vits14 @ input 392, ~30 train_good imgs per class, pca_dim=64,
  max_components=30:
    Feature extraction:    ~5 s per class
    PCA fit:               <1 s
    DPMM fit (CPU):       ~20 s
    Test scoring (GPU):   ~12 s
  Full 8 classes:         ~5-6 min end-to-end. Mostly CPU-bound.

# Output

  Standard stacker contract: submission.csv, local_predictions.npz,
  test_predictions.npz.

# Dependencies

  scikit-learn>=1.1 (BayesianGaussianMixture w/ DP prior).
  patchcore_baseline_v2 + local_preds_saver + dinov3_loader +
  test_preds_saver — same directory.
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
import warnings
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
    base = torch.initial_seed() % 2 ** 32
    np.random.seed(base); random.seed(base)


# ─────────────────────────────────────────────────────────────────────────────
# DINO feature extractor — single block OR multi-block fused
# ─────────────────────────────────────────────────────────────────────────────
class DinoFeatureExtractor(nn.Module):
    """Extracts patch features from one or more DINO blocks. When >1
    block is requested, features are concatenated along channels and
    L2-normalised. Returns (B, P, C_total)."""

    def __init__(self, backbone: str, block_indices: tuple[int, ...],
                  l2_normalise: bool = True):
        super().__init__()
        if not is_dino_backbone(backbone):
            raise ValueError(f"need DINO backbone; got {backbone}")
        if backbone in DINOV3_CONVNEXT_SPECS:
            raise ValueError("ConvNeXt backbones not supported here.")
        model, info = load_dino_backbone(backbone)
        model.eval()
        for b in block_indices:
            if not (0 <= b < info.n_blocks):
                raise ValueError(
                    f"block {b} out of range [0, {info.n_blocks-1}] for {backbone}")
        self.model = model
        self.info = info
        self.block_indices = tuple(sorted(set(block_indices)))
        self.l2_normalise = l2_normalise
        self.embed_dim_per_block = info.embed_dim
        self.embed_dim_total = info.embed_dim * len(self.block_indices)
        self.patch_size = info.patch_size
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = get_patch_tokens_at_layers(
            self.model, self.info, x, list(self.block_indices))
        per_block = []
        for b in self.block_indices:
            f = feats[b]                         # (B, C, h, w)
            B, C, h, w = f.shape
            per_block.append(f.permute(0, 2, 3, 1).reshape(B, h * w, C))
        out = (per_block[0] if len(per_block) == 1
                else torch.cat(per_block, dim=-1))
        if self.l2_normalise:
            out = F.normalize(out, p=2, dim=-1)
        return out.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# PCA (torch-based, fitted on training patches, kept on GPU at inference)
# ─────────────────────────────────────────────────────────────────────────────
class TorchPCA:
    """Lightweight PCA. Fits via centered SVD on a subsample of patches.

    Stored on GPU after fit: mean (C,), components (pca_dim, C).
    """
    def __init__(self, pca_dim: int, fit_subsample: int = 50000,
                  seed: int = 0):
        self.pca_dim = pca_dim
        self.fit_subsample = fit_subsample
        self.seed = seed
        self.mean: torch.Tensor | None = None
        self.components: torch.Tensor | None = None
        self.explained_var_ratio: np.ndarray | None = None

    @torch.inference_mode()
    def fit(self, X: torch.Tensor) -> None:
        """X: (N, C) on any device."""
        device = X.device
        N, C = X.shape
        gen = torch.Generator(device=device).manual_seed(self.seed)
        if N > self.fit_subsample:
            idx = torch.randperm(N, generator=gen, device=device)[
                :self.fit_subsample]
            Xs = X[idx]
        else:
            Xs = X
        self.mean = Xs.mean(dim=0)
        centred = Xs - self.mean
        # svd_lowrank gives an approximate top-k SVD — much faster than
        # full SVD when q ≪ min(N, C).
        U, S, V = torch.svd_lowrank(centred,
                                       q=min(self.pca_dim + 8, centred.shape[1]),
                                       niter=6)
        # Rows of V are the right-singular vectors; first pca_dim of them.
        self.components = V[:, :self.pca_dim].T.contiguous()    # (pca_dim, C)
        # Variance explained = S^2 / (N - 1)
        var = (S ** 2) / max(centred.shape[0] - 1, 1)
        total = float((centred ** 2).sum().item() / max(centred.shape[0] - 1, 1))
        if total > 0:
            self.explained_var_ratio = (var[:self.pca_dim].cpu().numpy() / total)

    @torch.inference_mode()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """X: (..., C) → (..., pca_dim) on the same device as the
        components."""
        assert self.mean is not None and self.components is not None
        X = X.to(self.components.device)
        return (X - self.mean) @ self.components.T


# ─────────────────────────────────────────────────────────────────────────────
# DPMM wrapper
# ─────────────────────────────────────────────────────────────────────────────
class DPMM:
    """Wraps sklearn's BayesianGaussianMixture with a stick-breaking
    weight concentration prior. After fit, log-likelihood evaluation
    is moved to torch for fast batch inference on the GPU."""

    def __init__(self, max_components: int = 30,
                  covariance_type: str = "diag",
                  weight_conc_prior: float = 0.01,
                  max_iter: int = 200,
                  reg_covar: float = 1e-6,
                  seed: int = 0):
        from sklearn.mixture import BayesianGaussianMixture
        self.max_components = max_components
        self.covariance_type = covariance_type
        # weight_conc_prior < 1 favours sparser solutions (fewer effective
        # components); the default in sklearn is 1.0.
        self.bgm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type=covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=weight_conc_prior,
            max_iter=max_iter,
            reg_covar=reg_covar,
            init_params="kmeans",
            random_state=seed,
        )
        # GPU-side cached parameters for fast log_prob.
        self._log_weights: torch.Tensor | None = None  # (K,)
        self._means: torch.Tensor | None = None         # (K, D)
        self._precisions_chol: torch.Tensor | None = None  # (K, D) diag
                                                                # or (K, D, D)
        self._log_det_chol: torch.Tensor | None = None     # (K,)
        self._D: int | None = None
        self._device: torch.device | None = None

    def fit(self, X: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bgm.fit(X)

    def n_effective_components(self, threshold: float = 1e-3) -> int:
        return int((self.bgm.weights_ > threshold).sum())

    def to(self, device: torch.device) -> None:
        """Convert fitted parameters to torch tensors on `device` for
        fast batched log_prob inference."""
        self._device = device
        self._D = self.bgm.means_.shape[1]
        # Filter out near-zero-weight components for speed.
        w = self.bgm.weights_
        keep = w > 1e-4
        w = w[keep]
        means = self.bgm.means_[keep]
        if self.covariance_type == "diag":
            # precisions_cholesky_ is (K, D), the diag of L where L L^T = Λ.
            pchol = self.bgm.precisions_cholesky_[keep]
            self._precisions_chol = torch.from_numpy(pchol).to(device).float()
            # log|Λ|^{1/2} = sum(log(diag(L)))
            log_det = np.log(pchol).sum(axis=1)
        elif self.covariance_type == "full":
            pchol = self.bgm.precisions_cholesky_[keep]      # (K, D, D)
            self._precisions_chol = torch.from_numpy(pchol).to(device).float()
            # log|L| = sum(log(diag(L)))
            diag_idx = np.arange(self._D)
            log_det = np.log(pchol[:, diag_idx, diag_idx]).sum(axis=1)
        else:
            raise ValueError(f"covariance_type {self.covariance_type} "
                              f"not supported on GPU; use diag or full.")
        self._means = torch.from_numpy(means).to(device).float()
        self._log_weights = torch.from_numpy(np.log(np.maximum(w, 1e-12))) \
                               .to(device).float()
        self._log_det_chol = torch.from_numpy(log_det).to(device).float()

    @torch.inference_mode()
    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, D) on the same device as the cached parameters.
        Returns (N,) log p(x) under the Gaussian mixture."""
        assert self._means is not None and self._precisions_chol is not None
        assert self._log_weights is not None and self._log_det_chol is not None
        D = self._D
        N = X.shape[0]
        K = self._means.shape[0]
        # log N(x | μ_k, Σ_k) = -0.5 * (D log(2π) + ||L_k^T (x - μ_k)||^2)
        #                       + log|L_k|     (since |Λ|^{1/2} = |L|)
        const = -0.5 * D * math.log(2.0 * math.pi)
        if self.covariance_type == "diag":
            # x_centred: (N, K, D)
            x_centred = X.unsqueeze(1) - self._means.unsqueeze(0)
            # Maha (per k): sum_d ( L_k_d * x_centred_d )^2  (L is diag here)
            scaled = x_centred * self._precisions_chol.unsqueeze(0)
            maha = (scaled ** 2).sum(dim=-1)              # (N, K)
        else:
            # full covariance — slower path
            # x_centred: (N, K, D); L^T x : (N, K, D)
            x_centred = X.unsqueeze(1) - self._means.unsqueeze(0)
            # Apply L_k^T to each row: result has shape (N, K, D)
            scaled = torch.einsum("kij,nkj->nki",
                                     self._precisions_chol, x_centred)
            maha = (scaled ** 2).sum(dim=-1)
        log_comp = (const + self._log_det_chol.unsqueeze(0)
                      - 0.5 * maha)                          # (N, K)
        log_unnorm = log_comp + self._log_weights.unsqueeze(0)
        return torch.logsumexp(log_unnorm, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def extract_patches_to_cpu(extractor: DinoFeatureExtractor,
                              records: list[ImageRecord],
                              cfg: "RunConfig",
                              device: torch.device) -> tuple[torch.Tensor,
                                                                  tuple[int, int]]:
    """Returns (N_patches, C) CPU tensor of L2-normed patch features,
    plus the (h, w) feature-grid shape."""
    ds = TrainGoodDataset(records, input_size=cfg.input_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0),
                         worker_init_fn=worker_init_fn)
    use_amp = (device.type == "cuda" and cfg.amp)
    pieces: list[torch.Tensor] = []
    h = w = None
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            f = extractor(x)                       # (B, P, C)
        f = f.float()
        if h is None:
            P = f.shape[1]
            h = w = int(math.isqrt(P))
        pieces.append(f.reshape(-1, f.shape[-1]).cpu())
    feats = torch.cat(pieces, dim=0)
    return feats, (h, w)


@torch.inference_mode()
def score_records(extractor: DinoFeatureExtractor,
                    pca: TorchPCA, dpmm: DPMM,
                    records: list[ImageRecord], cfg: "RunConfig",
                    device: torch.device, feature_hw: tuple[int, int],
                    load_masks: bool
                    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    ds = InferenceDataset(records, input_size=cfg.input_size,
                            load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    use_amp = (device.type == "cuda" and cfg.amp)
    h, w = feature_hw
    scores, gts = {}, {}
    n_done = 0; last_log = 0

    def _one_pass(x: torch.Tensor) -> torch.Tensor:
        """Returns (B, h, w) anomaly score (NLL)."""
        with torch.amp.autocast("cuda", enabled=use_amp):
            f = extractor(x)                       # (B, P, C)
        f = f.float()
        B, P, _C = f.shape
        z = pca.transform(f.reshape(-1, f.shape[-1]))   # (B*P, pca_dim)
        lp = dpmm.log_prob(z)                      # (B*P,)
        nll = -lp                                   # higher = more anomalous
        return nll.reshape(B, h, w)

    for x, masks, idxs in loader:
        x = x.to(device, non_blocking=True)
        m_np = masks.numpy()

        # TTA aggregation: average NLL across flips (anomaly is symmetric
        # under flip).
        acc = _one_pass(x); n_acc = 1
        if cfg.tta in ("hflip", "hvflip"):
            s = _one_pass(torch.flip(x, dims=[-1]))
            acc = acc + torch.flip(s, dims=[-1]); n_acc += 1
        if cfg.tta in ("vflip", "hvflip"):
            s = _one_pass(torch.flip(x, dims=[-2]))
            acc = acc + torch.flip(s, dims=[-2]); n_acc += 1
        nll = acc / n_acc                          # (B, h, w)
        up = F.interpolate(nll.unsqueeze(1),
                            size=(cfg.input_size, cfg.input_size),
                            mode="bilinear", align_corners=False).squeeze(1)
        up_np = up.cpu().numpy()

        for b in range(up_np.shape[0]):
            scores[int(idxs[b])] = up_np[b]
            gts[int(idxs[b])] = m_np[b]
        n_done += up_np.shape[0]
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


def run_one_class(cls: str, records_all: list[ImageRecord],
                   extractor: DinoFeatureExtractor,
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

    # 1. Extract features.
    print(f"    [{now_hms()}] extracting train_good patch features...")
    feats_cpu, feature_hw = extract_patches_to_cpu(extractor, train_good,
                                                         cfg, device)
    print(f"    -> {feats_cpu.shape[0]} patches × {feats_cpu.shape[1]} dim  "
          f"feature_hw={feature_hw}")

    # 2. PCA on GPU, fitted on patches.
    print(f"    [{now_hms()}] fitting PCA to {cfg.pca_dim} dims...")
    feats_gpu = feats_cpu.to(device, non_blocking=True)
    del feats_cpu
    pca = TorchPCA(pca_dim=cfg.pca_dim,
                     fit_subsample=cfg.pca_fit_subsample,
                     seed=cfg.seed)
    pca.fit(feats_gpu)
    if pca.explained_var_ratio is not None:
        cum = float(pca.explained_var_ratio.cumsum()[-1])
        print(f"      PCA cumulative explained variance "
              f"({cfg.pca_dim} dims): {cum:.3f}")

    # 3. DPMM fit on the reduced features (CPU — sklearn).
    Z_gpu = pca.transform(feats_gpu)
    Z_cpu = Z_gpu.cpu().numpy()
    del feats_gpu, Z_gpu
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if Z_cpu.shape[0] > cfg.dpmm_fit_subsample:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(Z_cpu.shape[0], cfg.dpmm_fit_subsample, replace=False)
        Z_fit = Z_cpu[idx]
        print(f"      DPMM fit on {Z_fit.shape[0]} subsampled patches "
              f"(of {Z_cpu.shape[0]})")
    else:
        Z_fit = Z_cpu
        print(f"      DPMM fit on all {Z_fit.shape[0]} patches")
    print(f"    [{now_hms()}] fitting DPMM "
          f"(max_components={cfg.max_components}, "
          f"cov={cfg.covariance_type}, prior={cfg.weight_conc_prior})...")
    t0 = time.time()
    dpmm = DPMM(max_components=cfg.max_components,
                   covariance_type=cfg.covariance_type,
                   weight_conc_prior=cfg.weight_conc_prior,
                   max_iter=cfg.max_iter, seed=cfg.seed)
    dpmm.fit(Z_fit)
    n_eff = dpmm.n_effective_components()
    print(f"    DPMM fit done in {time.time()-t0:.1f}s; "
          f"effective components (w>1e-3): {n_eff}/{cfg.max_components}  "
          f"(converged={dpmm.bgm.converged_})")
    dpmm.to(device)

    if cfg.save_models:
        mp = run_dir / "ckpt" / f"{cls}_dpmm.pkl"
        mp.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(mp, "wb") as f:
            pickle.dump({"pca_mean": pca.mean.cpu(),
                          "pca_components": pca.components.cpu(),
                          "bgm": dpmm.bgm,
                          "feature_hw": feature_hw,
                          "block_indices": extractor.block_indices,
                          "config": asdict(cfg)}, f)
        print(f"    saved model -> {mp}")

    # 4. Score.
    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  tta={cfg.tta}  "
            f"K_eff={n_eff}/{cfg.max_components}")
        scores, gts = score_records(extractor, pca, dpmm, train_anom, cfg,
                                       device, feature_hw, load_masks=True)
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
        scores, _ = score_records(extractor, pca, dpmm, test, cfg,
                                     device, feature_hw, load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            sm_final = maybe_resize_to_submission(sm_smooth)
            test_results.append((test[r_idx], sm_final))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del dpmm, pca
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"class": cls, "class_mean_ap": class_mean_ap,
            "eval_rows": eval_rows, "test_results": test_results,
            "elapsed_min": elapsed_min, "n_eff_components": n_eff}


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
    block_indices: tuple[int, ...] = (9,)
    input_size: int = 392
    pca_dim: int = 64
    pca_fit_subsample: int = 50000
    dpmm_fit_subsample: int = 30000
    max_components: int = 30
    covariance_type: str = "diag"
    weight_conc_prior: float = 0.01
    max_iter: int = 200
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
    save_models: bool = False
    zip_submission: bool = True
    run_tag: str = ""


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "dino_dpmm",
        "backbone": cfg.backbone,
        "block_indices": list(cfg.block_indices),
        "input_size": cfg.input_size,
        "pca_dim": cfg.pca_dim,
        "max_components": cfg.max_components,
        "covariance_type": cfg.covariance_type,
        "weight_conc_prior": cfg.weight_conc_prior,
        "max_iter": cfg.max_iter,
        "tta": cfg.tta, "smooth_sigma": cfg.smooth_sigma,
        "seed": cfg.seed,
        "v": 1,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bb = BACKBONE_SHORT.get(cfg.backbone, cfg.backbone)
    L = "_".join(str(b) for b in cfg.block_indices)
    bits = (f"{stamp}_dpmm_{bb}_b{L}_in{cfg.input_size}"
            f"_pca{cfg.pca_dim}_K{cfg.max_components}_"
            f"{cfg.covariance_type}_p{cfg.weight_conc_prior:g}")
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
                    help="Any DINOv2/v3 ViT backbone.")
    ap.add_argument("--block-indices", type=int, nargs="+", default=[9],
                    help="One or more ViT block indices. If multiple, "
                         "features are concatenated.")
    ap.add_argument("--input-size", type=int, default=392)
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="PCA target dim. Lower -> less overfitting on small "
                         "train_good but loses subtle modes. 32-128 is a "
                         "reasonable sweep.")
    ap.add_argument("--pca-fit-subsample", type=int, default=50000)
    ap.add_argument("--dpmm-fit-subsample", type=int, default=30000,
                    help="Cap DPMM training set; ~30k is plenty for "
                         "vits14 features and keeps fit time <1 min/class.")
    ap.add_argument("--max-components", type=int, default=30,
                    help="Truncation level of the Dirichlet Process; "
                         "DPMM will only USE a smaller effective number "
                         "(reported as 'K_eff' in the log).")
    ap.add_argument("--covariance-type", default="diag",
                    choices=["diag", "full"],
                    help="diag is ~10x faster to fit; full captures more "
                         "but needs more train_good. Try full with "
                         "pca-dim <= 32.")
    ap.add_argument("--weight-conc-prior", type=float, default=0.01,
                    help="alpha for the DP stick-breaking prior. Lower "
                         "-> fewer effective components.")
    ap.add_argument("--max-iter", type=int, default=200,
                    help="Variational max iterations. 200 usually "
                         "converges; bump to 400 if 'converged=False' "
                         "in the log.")
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
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    # Sanity.
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
    for b in args.block_indices:
        if not (0 <= b < nb):
            raise SystemExit(f"[FATAL] block {b} out of range [0, {nb-1}].")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone,
        block_indices=tuple(sorted(set(args.block_indices))),
        input_size=args.input_size,
        pca_dim=args.pca_dim,
        pca_fit_subsample=args.pca_fit_subsample,
        dpmm_fit_subsample=args.dpmm_fit_subsample,
        max_components=args.max_components,
        covariance_type=args.covariance_type,
        weight_conc_prior=args.weight_conc_prior,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        score_batch_size=args.score_batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        save_models=args.save_models,
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
        hr(f"DINO-DPMM — RUN {run_id}", "█")
        print(f"  backbone          : {cfg.backbone}")
        print(f"  block_indices     : {list(cfg.block_indices)}")
        print(f"  input_size        : {cfg.input_size}")
        print(f"  pca_dim           : {cfg.pca_dim}")
        print(f"  dpmm K_max        : {cfg.max_components}  "
              f"cov={cfg.covariance_type}  prior={cfg.weight_conc_prior}")
        print(f"  max_iter          : {cfg.max_iter}")
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

        extractor = DinoFeatureExtractor(
            backbone=cfg.backbone, block_indices=cfg.block_indices,
            l2_normalise=True).to(device).eval()
        print(f"\n  features per patch: {extractor.embed_dim_total} "
              f"({len(cfg.block_indices)} block(s) × "
              f"{extractor.embed_dim_per_block})")

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
        class_aps, class_elapsed, class_keff = {}, {}, {}
        for cls in classes:
            res = run_one_class(cls, records, extractor, cfg, run_dir,
                                  device, local_saver=local_saver)
            all_test_results.extend(res["test_results"])
            all_eval_rows.extend(res["eval_rows"])
            class_aps[cls] = res["class_mean_ap"]
            class_elapsed[cls] = res["elapsed_min"]
            class_keff[cls] = res.get("n_eff_components", 0)

        if local_saver is not None and len(local_saver) > 0:
            local_saver.save(run_dir / "local_predictions.npz")

        hr("LOCAL VALIDATION SUMMARY", "=")
        print(f"  {'class':<10} {'pixel-AP':>12} {'K_eff':>8} {'time (min)':>12}")
        for cls in classes:
            print(f"  {cls:<10} {class_aps.get(cls, float('nan')):>12.4f} "
                  f"{class_keff.get(cls, 0):>8d} "
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
        L = "+".join(str(b) for b in cfg.block_indices)
        row = {
            "run_id": run_id, "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": f"DPMM_{bb_short.upper()}",
            "feature_layers": f"blocks_{L}",
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
            "notes": (f"dino-dpmm backbone={cfg.backbone} "
                      f"L={L} pca={cfg.pca_dim} K={cfg.max_components} "
                      f"cov={cfg.covariance_type} "
                      f"prior={cfg.weight_conc_prior} tta={cfg.tta}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()