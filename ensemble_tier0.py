#!/usr/bin/env python3
"""ensemble_tier0.py — PDF-native ensemble for Spacepresso anomaly detection.

Fully unsupervised stacker over base-model score maps. Five density families
compete in the 14-d rank-vector space per class; LOAO over anomaly types
crowns the winning family per class; multi-view consensus + post-processing
shape the final map. NO XGBoost, NO logistic regression, NO supervised
classifier sees an anomaly label in any loss.

The five families (all from the PDF):
  (a) PADIM         — closed-form multivariate Gaussian.            §4.5.2
  (b) GMM           — K-component Gaussian mixture (EM).            §4.4.4
  (c) PatchCore     — memory bank of normal rank vectors + k-NN.    §4.5.3
  (d) Deep SVDD     — tiny MLP collapse to a center (no bias).      §4.8
  (e) Student-Teach — cross-method discrepancy + uncertainty.       §4.6
                      [optional, --enable-student-teacher]

Tier 0 contract
---------------
No retraining of base models, no fresh train_good predictions required.
Normality in score space is approximated by the mask==0 pixels of the
train_anomaly val images (typically 98–99.5% of those images' pixels).
Each base model's score map already encodes the model's verdict on what's
normal in input space — we just fit a density on top of those verdicts.

Pipeline
--------
  Stage 1   per-(class, view, method) ECDF rank-norm           Advice 15/05
  Stage 2-5 five density families on the 14-d rank vector     PDF §4.4-4.8
  Stage 3   LOAO over anomaly types crowns winner per class    Advice 14/05
  Stage 4   multi-view consensus over the 5 views/sample       Advice 12/05
  Stage 5   Gaussian smooth, drop tiny CCs, spatial prior      Advice 08/05
  Stage 6   within-image rank transform before q8 encode       Advice 07/05

Reuses the v6 stacker's IO, caching, and post-processing primitives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import time
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Reuse the v6 stacker's IO / cache / decode / post-process primitives.
# Required symbols (must exist in xgboost_stacker_v6.py):
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xgboost_stacker_v6 import (
    Tee, tee_to, hr, sub,
    float_matrix_to_q8rle, q8rle_to_uint8_matrix, to_f32,
    Cache, file_meta_hash,
    load_submission, load_local_preds, build_class_map_from_data,
    parse_sample_id,
    load_spatial_priors, _resize_prior_to,
    detect_model_family,
    align_local_preds, save_aligned_val, load_aligned_val,
    group_test_ids_by_sample,
    decode_submissions_to_uint8, cache_decoded_test, load_decoded_test,
    append_to_master,
    parse_small_cc_spec, DEFAULT_SMALL_CC_PER_FAMILY,
)

# Cosmetic helpers — provided by v6 in some builds, fallback otherwise.
try:
    from xgboost_stacker_v6 import mem_print as _v6_mem_print  # type: ignore
    def mem_print(tag: str = "") -> None:
        _v6_mem_print(tag)
except ImportError:
    def mem_print(tag: str = "") -> None:
        try:
            import psutil  # noqa
            rss = psutil.Process().memory_info().rss / (1024 ** 3)
            print(f"    [mem] {tag}: {rss:.2f} GB RSS")
        except Exception:
            pass  # silent if psutil missing

try:
    from xgboost_stacker_v6 import fmt_bytes as _v6_fmt_bytes  # type: ignore
    def fmt_bytes(n: int) -> str:
        return _v6_fmt_bytes(n)
except ImportError:
    def fmt_bytes(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-AP
# ─────────────────────────────────────────────────────────────────────────────
def pixel_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pixel-level Average Precision. `scores`/`labels` are flat arrays."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int32).ravel()
    if y.sum() == 0 or y.sum() == y.size:
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — ECDF rank-normalization
# ─────────────────────────────────────────────────────────────────────────────
def fit_ecdf(values: np.ndarray) -> np.ndarray:
    """Returns a sorted copy of `values` usable as an ECDF LUT for searchsorted."""
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    return np.sort(values.astype(np.float32, copy=False))


def apply_ecdf(sorted_normals: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Maps `query` to ranks in [0, 1] using the sorted normal values.
    Out-of-distribution queries above the highest normal get rank 1.0,
    below the lowest get rank 0.0.
    """
    n = sorted_normals.size
    if n == 0:
        return np.zeros_like(query, dtype=np.float32)
    ranks = np.searchsorted(sorted_normals, query.astype(np.float32),
                            side="right")
    return (ranks.astype(np.float32) / float(n))


# def build_ecdfs_from_val(
#     val: dict,
#     methods: list[str],
#     class_map: dict[str, str],
#     max_normals_per_cell: int = 200_000,
#     seed: int = 0,
# ) -> dict:
#     """Per-(class, view, method) ECDF fit on val mask==0 pixels.
#
#     Returns a nested dict { (class, view): { method: sorted_normals } }.
#     """
#     rng = np.random.default_rng(seed)
#     ids = val["ids"]
#     scores = val["scores"]              # (N, H, W, M)
#     masks = val["masks"]                # (N, H, W)
#     N = scores.shape[0]
#     M = scores.shape[3]
#     assert M == len(methods), f"val has {M} methods, expected {len(methods)}"
#
#     # Bucket image indices by (class, view).
#     bucket_idx: dict[tuple[str, int], list[int]] = defaultdict(list)
#     for i, vid in enumerate(ids):
#         cls = class_map.get(str(vid)) or class_map.get(str(vid).split("/")[0])
#         if cls is None:
#             continue
#         _, view = parse_sample_id(str(vid))
#         if view is None:
#             continue
#         bucket_idx[(cls, view)].append(i)
#
#     ecdfs: dict[tuple[str, int], dict[str, np.ndarray]] = {}
#     for (cls, view), idxs in sorted(bucket_idx.items()):
#         # Gather mask==0 pixels for this (class, view).
#         per_method: dict[str, list[np.ndarray]] = {m: [] for m in methods}
#         for i in idxs:
#             mm = masks[i] == 0
#             if not mm.any():
#                 continue
#             for j, m in enumerate(methods):
#                 per_method[m].append(scores[i, :, :, j][mm])
#         cell: dict[str, np.ndarray] = {}
#         for j, m in enumerate(methods):
#             arrs = per_method[m]
#             if not arrs:
#                 cell[m] = np.empty(0, dtype=np.float32)
#                 continue
#             vals = np.concatenate(arrs).astype(np.float32)
#             if vals.size > max_normals_per_cell:
#                 pick = rng.choice(vals.size, max_normals_per_cell,
#                                   replace=False)
#                 vals = vals[pick]
#             cell[m] = fit_ecdf(vals)
#         ecdfs[(cls, view)] = cell
#         print(f"    fit ECDFs  class={cls} view={view}  "
#               f"normals(per method)={[cell[m].size for m in methods]}  "
#               f"images={len(idxs)}", flush=True)
#     return ecdfs
#
#
# def rank_norm_val_inplace(val: dict, methods: list[str],
#                           class_map: dict[str, str], ecdfs: dict) -> None:
#     """Replaces val['scores'][:,:,:,j] with rank-normed values per
#     (class, view, method).
#     """
#     ids = val["ids"]
#     scores = val["scores"]              # (N, H, W, M), float32
#     for i, vid in enumerate(ids):
#         cls = class_map.get(str(vid)) or class_map.get(str(vid).split("/")[0])
#         if cls is None:
#             continue
#         _, view = parse_sample_id(str(vid))
#         if view is None:
#             continue
#         cell = ecdfs.get((cls, view))
#         if cell is None:
#             continue
#         for j, m in enumerate(methods):
#             sn = cell.get(m)
#             if sn is None or sn.size == 0:
#                 # Degenerate: leave as zeros; the family will get a
#                 # constant column and downweight it via Σ.
#                 scores[i, :, :, j] = 0.0
#                 continue
#             scores[i, :, :, j] = apply_ecdf(sn, scores[i, :, :, j])

def build_ecdfs_from_val(
    val: dict,
    methods: list[str],
    class_map: dict[str, str],
    max_normals_per_cell: int = 200_000,
    seed: int = 0,
) -> dict:
    """Per-(class, view, method) ECDF fit on val mask==0 pixels.

    View comes from val['views'] — set by align_local_preds() by parsing
    the image_path filename. The val IDs themselves encode r_idx, not the
    real view, so parse_sample_id(vid) would give bucket keys 0..24 that
    don't match the test side's view 1..5.

    Returns a nested dict { (class, view): { method: sorted_normals } }.
    """
    rng = np.random.default_rng(seed)
    ids = val["ids"]
    scores = val["scores"]
    masks = val["masks"]
    classes_arr = val.get("classes")
    views_arr = val.get("views")
    if views_arr is None:
        raise RuntimeError(
            "val['views'] is missing — align_local_preds must populate it "
            "from image_paths. Re-build the val cache with --rebuild-cache.")
    N = scores.shape[0]
    M = scores.shape[3]
    assert M == len(methods), f"val has {M} methods, expected {len(methods)}"

    bucket_idx: dict[tuple[str, int], list[int]] = defaultdict(list)
    n_dropped_no_view = 0
    for i in range(N):
        cls = None
        if classes_arr is not None and classes_arr[i]:
            cls = str(classes_arr[i])
        if not cls:
            vid = str(ids[i])
            cls = class_map.get(vid) or class_map.get(vid.split("/")[0])
        if not cls:
            continue
        view = int(views_arr[i])
        if view < 0:
            n_dropped_no_view += 1
            continue
        bucket_idx[(cls, view)].append(i)
    if n_dropped_no_view:
        print(f"    [warn] {n_dropped_no_view}/{N} val images had view=-1 "
              f"(missing image_paths in local_predictions.npz?)")

    ecdfs: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for (cls, view), idxs in sorted(bucket_idx.items()):
        per_method: dict[str, list[np.ndarray]] = {m: [] for m in methods}
        for i in idxs:
            mm = masks[i] == 0
            if not mm.any():
                continue
            for j, m in enumerate(methods):
                per_method[m].append(scores[i, :, :, j][mm])
        cell: dict[str, np.ndarray] = {}
        for j, m in enumerate(methods):
            arrs = per_method[m]
            if not arrs:
                cell[m] = np.empty(0, dtype=np.float32)
                continue
            vals = np.concatenate(arrs).astype(np.float32)
            if vals.size > max_normals_per_cell:
                pick = rng.choice(vals.size, max_normals_per_cell, replace=False)
                vals = vals[pick]
            cell[m] = fit_ecdf(vals)
        ecdfs[(cls, view)] = cell
        print(f"    fit ECDFs  class={cls} view={view}  "
              f"normals(per method)={[cell[m].size for m in methods]}  "
              f"images={len(idxs)}", flush=True)
    return ecdfs


def rank_norm_val_inplace(val: dict, methods: list[str],
                          class_map: dict[str, str], ecdfs: dict) -> None:
    """Replaces val['scores'][:,:,:,j] with rank-normed values per
    (class, view, method). View comes from val['views'], NOT parse_sample_id.
    """
    ids = val["ids"]
    scores = val["scores"]
    classes_arr = val.get("classes")
    views_arr = val.get("views")
    if views_arr is None:
        raise RuntimeError("val['views'] missing — see build_ecdfs_from_val.")
    N = scores.shape[0]
    for i in range(N):
        cls = None
        if classes_arr is not None and classes_arr[i]:
            cls = str(classes_arr[i])
        if not cls:
            vid = str(ids[i])
            cls = class_map.get(vid) or class_map.get(vid.split("/")[0])
        if not cls:
            continue
        view = int(views_arr[i])
        if view < 0:
            continue
        cell = ecdfs.get((cls, view))
        if cell is None:
            continue
        for j, m in enumerate(methods):
            sn = cell.get(m)
            if sn is None or sn.size == 0:
                scores[i, :, :, j] = 0.0
                continue
            scores[i, :, :, j] = apply_ecdf(sn, scores[i, :, :, j])


def rank_norm_test_per_view_inplace(
    decoded: list[dict],
    methods: list[str],
    class_map: dict[str, str],
    ecdfs: dict,
    out_dtype: np.dtype = np.float32,
) -> None:
    """Each `decoded[j]` is a dict {test_id: uint8 (H,W)} for method j.
    Replaces the uint8 maps with float32 rank-normed maps in [0,1].
    """
    new_decoded: list[dict] = []
    for j, m in enumerate(methods):
        d = decoded[j]
        nd: dict[str, np.ndarray] = {}
        for tid, raw in d.items():
            cls = class_map.get(tid) or class_map.get(tid.split("/")[0])
            if cls is None:
                nd[tid] = np.zeros_like(raw, dtype=out_dtype)
                continue
            _, view = parse_sample_id(tid)
            cell = ecdfs.get((cls, view)) if view is not None else None
            if cell is None or m not in cell or cell[m].size == 0:
                nd[tid] = np.zeros_like(raw, dtype=out_dtype)
                continue
            # uint8 → float32 (preserve scale, ECDF will re-rank anyway).
            q = raw.astype(np.float32) / 255.0
            nd[tid] = apply_ecdf(cell[m], q).astype(out_dtype)
        new_decoded.append(nd)
    # Replace contents in place.
    for j in range(len(decoded)):
        decoded[j].clear()
        decoded[j].update(new_decoded[j])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2/5 — density families
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Family:
    name: str
    fit: Callable[..., dict]
    score: Callable[[dict, np.ndarray], np.ndarray]
    needs_torch: bool = False


def _subsample(R: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if R.shape[0] <= max_rows:
        return R
    idx = rng.choice(R.shape[0], max_rows, replace=False)
    return R[idx]


# --- (a) PADIM ---------------------------------------------------------------
def fit_padim(R_normal: np.ndarray, ridge: float = 1e-3,
              seed: int = 0, **kw) -> dict:
    """Closed-form multivariate Gaussian on rank vectors.

    R_normal: (n, M) float in [0, 1].
    """
    R = R_normal.astype(np.float64, copy=False)
    n, M = R.shape
    if n < 2:
        return {"mu": np.zeros(M, dtype=np.float32),
                "inv_cov": np.eye(M, dtype=np.float32)}
    mu = R.mean(axis=0)
    centered = R - mu
    cov = (centered.T @ centered) / max(n - 1, 1)
    cov += ridge * np.eye(M)
    inv_cov = np.linalg.pinv(cov)
    return {"mu": mu.astype(np.float32),
            "inv_cov": inv_cov.astype(np.float32)}


def score_padim(params: dict, R: np.ndarray) -> np.ndarray:
    d = R.astype(np.float32) - params["mu"]
    a = np.einsum("ij,jk,ik->i", d, params["inv_cov"], d)
    return a.astype(np.float32)


# --- (b) GMM (DAGMM-style, K-component) --------------------------------------
def fit_gmm(R_normal: np.ndarray, n_components: int = 3,
            max_rows: int = 80_000, seed: int = 0, **kw) -> dict:
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    R = _subsample(R_normal.astype(np.float32), max_rows, rng)
    if R.shape[0] < n_components * 5:
        n_components = max(1, R.shape[0] // 5)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-3,
        max_iter=100,
        n_init=1,
        random_state=seed,
    )
    gmm.fit(R)
    return {"gmm": gmm}


def score_gmm(params: dict, R: np.ndarray) -> np.ndarray:
    # Negative log-likelihood; rare points score high.
    ll = params["gmm"].score_samples(R.astype(np.float32))
    return (-ll).astype(np.float32)


# --- (c) PatchCore on rank vectors (memory bank + k-NN) ----------------------
def fit_patchcore_ranks(R_normal: np.ndarray, max_bank: int = 6000,
                        seed: int = 0, **kw) -> dict:
    """Random subsample as the memory bank. Core-set selection in 14-d
    is overkill — random samples already cover the manifold well.
    """
    rng = np.random.default_rng(seed)
    bank = _subsample(R_normal.astype(np.float32), max_bank, rng)
    return {"bank": bank}


def score_patchcore_ranks(params: dict, R: np.ndarray,
                          k: int = 5, chunk: int = 8192) -> np.ndarray:
    bank = params["bank"]                # (B, M)
    B = bank.shape[0]
    n = R.shape[0]
    if B == 0:
        return np.zeros(n, dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    R32 = R.astype(np.float32, copy=False)
    bank_sq = (bank ** 2).sum(axis=1)    # (B,)
    k_eff = int(min(k, B))
    for i in range(0, n, chunk):
        e = min(n, i + chunk)
        chunk_R = R32[i:e]                       # (c, M)
        # Squared L2: ||r-b||^2 = ||r||^2 + ||b||^2 - 2 r.b
        d2 = (chunk_R ** 2).sum(axis=1, keepdims=True) \
             + bank_sq[None, :] \
             - 2.0 * (chunk_R @ bank.T)
        np.maximum(d2, 0.0, out=d2)
        if k_eff < B:
            part = np.partition(d2, k_eff - 1, axis=1)[:, :k_eff]
        else:
            part = d2
        out[i:e] = np.sqrt(part).mean(axis=1)
    return out


# --- (d) Deep SVDD on rank vectors -------------------------------------------
def fit_deep_svdd(R_normal: np.ndarray, n_epochs: int = 50,
                  batch_size: int = 4096, hidden: int = 32,
                  out_dim: int = 16, lr: float = 1e-3,
                  weight_decay: float = 1e-6, max_rows: int = 100_000,
                  device: str = "cuda", seed: int = 0, **kw) -> dict:
    """Tiny MLP, no bias, LeakyReLU (unbounded). Trained to collapse
    train_good rank vectors to a fixed center.
    """
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    R = _subsample(R_normal.astype(np.float32), max_rows, rng)
    n, M = R.shape
    if n < 32:
        return {"net_state": None, "center": None, "in_dim": M,
                "hidden": hidden, "out_dim": out_dim, "device": device}

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    class SVDDNet(nn.Module):
        def __init__(self, m, h, o):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(m, h, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(h, h, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(h, o, bias=False),
            )
        def forward(self, x): return self.net(x)

    net = SVDDNet(M, hidden, out_dim).to(dev)
    # Center c: mean of initial forward outputs. Per PDF §4.8, must not
    # equal psi_0(0). With no bias and LeakyReLU(0.1), psi(0)=0, so we
    # avoid zero-init by ensuring |c_i| >= 0.1 (sign-preserving floor).
    with torch.no_grad():
        x0 = torch.from_numpy(R[: min(8192, n)]).to(dev)
        c = net(x0).mean(dim=0)
        sign = torch.sign(c)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        c = torch.where(c.abs() < 0.1, sign * 0.1, c)
        c = c.detach()

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    R_t = torch.from_numpy(R).to(dev)
    n_iter = max(1, n // batch_size)
    rng_t = np.random.default_rng(seed + 1)
    for epoch in range(n_epochs):
        perm = rng_t.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = R_t[idx]
            y = net(x)
            loss = ((y - c) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Return state as CPU tensors (small, ~kB) for pickling.
    return {"net_state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "center": c.detach().cpu(),
            "in_dim": M, "hidden": hidden, "out_dim": out_dim,
            "device": device}


def score_deep_svdd(params: dict, R: np.ndarray, chunk: int = 16384) -> np.ndarray:
    import torch
    import torch.nn as nn
    if params.get("net_state") is None:
        return np.zeros(R.shape[0], dtype=np.float32)
    dev = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
    M = params["in_dim"]; h = params["hidden"]; o = params["out_dim"]

    class SVDDNet(nn.Module):
        def __init__(self, m, h, o):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(m, h, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(h, h, bias=False),
                nn.LeakyReLU(0.1),
                nn.Linear(h, o, bias=False),
            )
        def forward(self, x): return self.net(x)

    net = SVDDNet(M, h, o).to(dev)
    net.load_state_dict(params["net_state"])
    net.eval()
    c = params["center"].to(dev)

    out = np.empty(R.shape[0], dtype=np.float32)
    R32 = R.astype(np.float32, copy=False)
    with torch.no_grad():
        for i in range(0, R32.shape[0], chunk):
            e = min(R32.shape[0], i + chunk)
            x = torch.from_numpy(R32[i:e]).to(dev)
            y = net(x)
            d = ((y - c) ** 2).sum(dim=1)
            out[i:e] = d.cpu().numpy()
    return out


# --- (e) Cross-method student-teacher ----------------------------------------
def fit_student_teacher(R_normal: np.ndarray, teacher_idx: int = 0,
                        n_students: int = 3, n_epochs: int = 25,
                        batch_size: int = 4096, hidden: int = 32,
                        lr: float = 1e-3, max_rows: int = 100_000,
                        device: str = "cuda", seed: int = 0, **kw) -> dict:
    """Train `n_students` MLPs to predict the teacher method's rank from
    the other methods' ranks. At test, anomaly = discrepancy + uncertainty.
    """
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    R = _subsample(R_normal.astype(np.float32), max_rows, rng)
    n, M = R.shape
    other = [i for i in range(M) if i != teacher_idx]
    if n < 64 or len(other) < 1:
        return {"students": [], "teacher_idx": teacher_idx, "other": other,
                "in_dim": len(other), "hidden": hidden, "device": device}

    X_np = R[:, other]
    y_np = R[:, teacher_idx]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    class StuNet(nn.Module):
        def __init__(self, m, h):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(m, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    students = []
    X_t = torch.from_numpy(X_np).to(dev)
    y_t = torch.from_numpy(y_np).to(dev)
    for s in range(n_students):
        torch.manual_seed(seed + 10 * s + 1)
        net = StuNet(len(other), hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        rng_t = np.random.default_rng(seed + 100 * s + 7)
        for epoch in range(n_epochs):
            perm = rng_t.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                p = net(X_t[idx])
                loss = ((p - y_t[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        students.append({k: v.detach().cpu() for k, v in net.state_dict().items()})

    return {"students": students, "teacher_idx": teacher_idx, "other": other,
            "in_dim": len(other), "hidden": hidden, "device": device}


def score_student_teacher(params: dict, R: np.ndarray,
                          chunk: int = 16384) -> np.ndarray:
    import torch
    import torch.nn as nn
    if not params.get("students"):
        return np.zeros(R.shape[0], dtype=np.float32)
    dev = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
    M_in = params["in_dim"]; h = params["hidden"]
    teacher_idx = params["teacher_idx"]
    other = params["other"]

    class StuNet(nn.Module):
        def __init__(self, m, h):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(m, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    nets = []
    for sd in params["students"]:
        net = StuNet(M_in, h).to(dev)
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)

    out = np.empty(R.shape[0], dtype=np.float32)
    R32 = R.astype(np.float32, copy=False)
    with torch.no_grad():
        for i in range(0, R32.shape[0], chunk):
            e = min(R32.shape[0], i + chunk)
            x_other = torch.from_numpy(R32[i:e][:, other]).to(dev)
            y_teach = R32[i:e, teacher_idx]
            preds = torch.stack([net(x_other) for net in nets], dim=0)
            mean = preds.mean(dim=0).cpu().numpy()
            std = preds.std(dim=0, unbiased=False).cpu().numpy()
            disc = np.abs(y_teach - mean)
            out[i:e] = (disc + std).astype(np.float32)
    return out


# Teacher-selection heuristic (no leak — uses only val mask==0 statistics):
# pick the method whose rank distribution on negatives is closest to
# Uniform[0, 1] (i.e., the best-calibrated method, most "centered").
def pick_teacher_idx(R_normal: np.ndarray) -> int:
    if R_normal.size == 0:
        return 0
    # mean rank closest to 0.5 = best-centered ECDF
    means = R_normal.mean(axis=0)
    return int(np.argmin(np.abs(means - 0.5)))


FAMILIES: dict[str, Family] = {
    "padim":     Family("padim",     fit_padim,            score_padim,            False),
    "gmm":       Family("gmm",       fit_gmm,              score_gmm,              False),
    "patchcore": Family("patchcore", fit_patchcore_ranks,  score_patchcore_ranks,  False),
    "svdd":      Family("svdd",      fit_deep_svdd,        score_deep_svdd,        True),
    "student_teacher": Family("student_teacher", fit_student_teacher,
                              score_student_teacher, True),
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-class index of rank vectors (for fast LOAO)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClassIndex:
    """Holds rank-vector splits for one class:
       - per_type_normal[atype] : (n_neg, M) float32 — pooled normals
       - per_type_anom  [atype] : (n_pos, M) float32 — pooled anomalies
       - per_image: list of dicts with .ranks (H, W, M), .mask (H, W),
                    .anomaly_type, .image_idx — for per-image AP eval.
       Per-type pixel pools are produced by flattening across all 5 views.
    """
    cls: str
    per_type_normal: dict[str, np.ndarray] = field(default_factory=dict)
    per_type_anom:   dict[str, np.ndarray] = field(default_factory=dict)
    per_image: list[dict] = field(default_factory=list)


def index_val_by_class(val: dict, class_map: dict[str, str],
                       methods: list[str]) -> dict[str, ClassIndex]:
    ids = val["ids"]; scores = val["scores"]; masks = val["masks"]
    types = val["anomaly_types"]
    classes = val["classes"]
    idx_per_class: dict[str, ClassIndex] = {}
    for i, vid in enumerate(ids):
        cls = (str(classes[i]) if i < len(classes) and classes[i] else
               class_map.get(str(vid)) or
               class_map.get(str(vid).split("/")[0]))
        if not cls:
            continue
        atype = (str(types[i]) if i < len(types) and types[i] else "unknown")
        ci = idx_per_class.setdefault(cls, ClassIndex(cls=cls))

        s = scores[i]                     # (H, W, M) float32
        m = masks[i]                      # (H, W)
        H, W, M = s.shape
        flat = s.reshape(-1, M)
        mflat = m.reshape(-1) > 0

        ci.per_type_normal.setdefault(atype, []).append(flat[~mflat])
        ci.per_type_anom.setdefault(atype, []).append(flat[mflat])
        ci.per_image.append(dict(ranks=s, mask=m, anomaly_type=atype,
                                  image_idx=i))

    # Concatenate per-type lists.
    for cls, ci in idx_per_class.items():
        for d in (ci.per_type_normal, ci.per_type_anom):
            for atype, lst in list(d.items()):
                if isinstance(lst, list):
                    if lst:
                        d[atype] = np.concatenate(lst, axis=0)
                    else:
                        d[atype] = np.empty((0, len(methods)),
                                            dtype=np.float32)
    return idx_per_class


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — LOAO winner selection
# ─────────────────────────────────────────────────────────────────────────────
def loao_one_class(
    ci: ClassIndex,
    family_names: list[str],
    *,
    seed: int,
    device: str,
    gmm_K: int,
    patchcore_k: int,
    patchcore_bank: int,
    svdd_epochs: int,
    st_epochs: int,
) -> dict:
    """For each family, runs LOAO over anomaly_types in this class and
    returns {family_name: {anomaly_type: ap, "mean": mean_ap}}.
    """
    atypes = sorted(ci.per_type_normal.keys())
    results: dict[str, dict[str, float]] = {f: {} for f in family_names}
    if len(atypes) < 2:
        # Single type: no fold-out possible; report 0 across the board.
        for f in family_names:
            results[f] = {"mean": 0.0}
        return results

    # Group images per type for per-fold per-image AP eval.
    imgs_by_type: dict[str, list[dict]] = defaultdict(list)
    for r in ci.per_image:
        imgs_by_type[r["anomaly_type"]].append(r)

    for ho in atypes:
        # Train on negatives of all OTHER types.
        train_neg_chunks = [ci.per_type_normal[t] for t in atypes if t != ho
                            and ci.per_type_normal[t].size > 0]
        if not train_neg_chunks:
            continue
        train_neg = np.concatenate(train_neg_chunks, axis=0)
        # Held-out images for evaluation.
        ho_imgs = imgs_by_type[ho]
        if not ho_imgs:
            continue

        for fname in family_names:
            fam = FAMILIES[fname]
            t0 = time.time()
            kw = dict(seed=seed, device=device)
            if fname == "gmm":          kw["n_components"] = gmm_K
            if fname == "patchcore":    kw["max_bank"]    = patchcore_bank
            if fname == "svdd":         kw["n_epochs"]    = svdd_epochs
            if fname == "student_teacher":
                kw["n_epochs"]    = st_epochs
                kw["teacher_idx"] = pick_teacher_idx(train_neg)
            try:
                params = fam.fit(train_neg, **kw)
            except Exception as e:
                print(f"    [WARN] fit failed for {fname} class={ci.cls} "
                      f"ho={ho}: {e}", flush=True)
                continue
            aps_this_fold = []
            for r in ho_imgs:
                H, W, M = r["ranks"].shape
                R_flat = r["ranks"].reshape(-1, M)
                A_flat = fam.score(params, R_flat)
                ap = pixel_ap(A_flat, r["mask"].reshape(-1))
                aps_this_fold.append(ap)
            fold_ap = float(np.mean(aps_this_fold)) if aps_this_fold else 0.0
            results[fname][ho] = fold_ap
            sec = time.time() - t0
            n_keys = len(results[fname])
            print(f"    LOAO  class={ci.cls}  family={fname:<16}  "
                  f"held-out={ho:<14}  AP={fold_ap:.4f}  "
                  f"({sec:.1f}s, fold {n_keys}/{len(atypes)})",
                  flush=True)

            if fname == "patchcore":
                _ = patchcore_k  # k controlled at score-time; bank size at fit-time

    # Per-family mean across folds.
    for fname in family_names:
        vals = [v for k, v in results[fname].items() if k != "mean"]
        results[fname]["mean"] = float(np.mean(vals)) if vals else 0.0
    return results


def select_winners(loao_results: dict[str, dict[str, dict[str, float]]]
                    ) -> dict[str, str]:
    """For each class, pick the family with the highest mean LOAO AP."""
    winners: dict[str, str] = {}
    for cls, fam_results in loao_results.items():
        best = max(fam_results.items(),
                   key=lambda kv: kv[1].get("mean", 0.0))
        winners[cls] = best[0]
    return winners


def fit_final_per_class(
    ci_by_class: dict[str, ClassIndex],
    winners: dict[str, str],
    *,
    seed: int,
    device: str,
    gmm_K: int,
    patchcore_bank: int,
    svdd_epochs: int,
    st_epochs: int,
) -> dict[str, dict]:
    """Refit the winning family per class on ALL of that class's normals."""
    out: dict[str, dict] = {}
    for cls, ci in ci_by_class.items():
        fname = winners.get(cls, "padim")
        all_neg_chunks = [v for v in ci.per_type_normal.values() if v.size > 0]
        if not all_neg_chunks:
            print(f"    [WARN] no negatives for class {cls}; skipping fit")
            out[cls] = {"family": fname, "params": None}
            continue
        all_neg = np.concatenate(all_neg_chunks, axis=0)
        fam = FAMILIES[fname]
        kw = dict(seed=seed, device=device)
        if fname == "gmm":          kw["n_components"] = gmm_K
        if fname == "patchcore":    kw["max_bank"]    = patchcore_bank
        if fname == "svdd":         kw["n_epochs"]    = svdd_epochs
        if fname == "student_teacher":
            kw["n_epochs"]    = st_epochs
            kw["teacher_idx"] = pick_teacher_idx(all_neg)
        t0 = time.time()
        params = fam.fit(all_neg, **kw)
        print(f"    [{cls}] final fit  family={fname}  "
              f"n_negatives={all_neg.shape[0]}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        out[cls] = {"family": fname, "params": params}
    return out



# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Multi-view consensus over the 5 views of one sample
# ─────────────────────────────────────────────────────────────────────────────
def multi_view_consensus(per_view: np.ndarray,
                          mode: str = "max",
                          beta: float = 0.3) -> np.ndarray:
    """Combine V views' anomaly maps into V refined maps.

    `per_view` : (V, H, W) float32 — one anomaly score map per view.
    `mode`     : "max"            -> A_v(x) ← max_v A_v(x)  (broadcast)
                 "max_minus_std"  -> A_v(x) ← max_v A - beta * std_v A
                 "agreement_boost" -> A_v(x) ← A_v * (1 + beta * agreement)
                 "passthrough"    -> A_v(x) ← A_v(x)  (no consensus)

    Returns: (V, H, W) float32. We deliberately return one map *per view*
    rather than a single shared map, so the submission still has one row
    per (sample, view).
    """
    V, H, W = per_view.shape
    if V < 2 or mode == "passthrough":
        return per_view.astype(np.float32, copy=False)

    pv = per_view.astype(np.float32, copy=False)
    if mode == "max":
        m = pv.max(axis=0, keepdims=True)        # (1, H, W)
        return np.broadcast_to(m, (V, H, W)).copy()
    if mode == "max_minus_std":
        m = pv.max(axis=0)                       # (H, W)
        s = pv.std(axis=0)                       # (H, W)
        combined = m - beta * s                  # (H, W)
        return np.broadcast_to(combined[None], (V, H, W)).copy()
    if mode == "agreement_boost":
        # Per-pixel "agreement" = high when all views report similar
        # (non-zero) scores; low when one view disagrees.
        mean_v = pv.mean(axis=0)                 # (H, W)
        std_v  = pv.std(axis=0)                  # (H, W)
        denom = mean_v + std_v + 1e-6
        agreement = mean_v / denom               # in [0, 1], high = agree
        boost = 1.0 + beta * agreement           # (H, W)
        return (pv * boost[None, :, :]).astype(np.float32)
    raise ValueError(f"unknown multi-view consensus mode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Post-processing
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_smooth_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    return ndi.gaussian_filter(arr.astype(np.float32), sigma=sigma,
                                mode="reflect").astype(np.float32)


def drop_small_components(arr: np.ndarray, min_area: int,
                          quantile: float = 0.95) -> np.ndarray:
    """Threshold at `quantile`, find connected components, zero-out the
    pixels whose component is smaller than `min_area`. Conservative:
    only suppresses the smallest CC speckle, leaves the rest alone.
    """
    if min_area <= 0:
        return arr
    a = arr.astype(np.float32, copy=False)
    thr = float(np.quantile(a, quantile))
    if not np.isfinite(thr):
        return a
    mask = a > thr
    if not mask.any():
        return a
    labels, n = ndi.label(mask)
    if n == 0:
        return a
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1)).astype(np.int64)
    too_small_labels = np.where(sizes < min_area)[0] + 1
    if too_small_labels.size == 0:
        return a
    suppress = np.isin(labels, too_small_labels)
    out = a.copy()
    out[suppress] = a[~mask].max() if (~mask).any() else 0.0
    return out


def apply_spatial_prior(arr: np.ndarray, prior: np.ndarray,
                        gamma: float = 0.5) -> np.ndarray:
    """Multiplicative spatial prior: A(x) ← A(x) · (1 + gamma · prior(x)).
    `prior` is expected to be in [0, 1]; resized to arr's H, W on the fly.
    """
    if prior is None or gamma <= 0:
        return arr
    H, W = arr.shape[-2:]
   # p = _resize_prior_to(prior, (H, W))
    p = _resize_prior_to(prior, H, W)
    p = np.clip(p, 0.0, 1.0).astype(np.float32)
    return (arr.astype(np.float32) * (1.0 + gamma * p)).astype(np.float32)


def within_image_rank(arr: np.ndarray) -> np.ndarray:
    """Replace each pixel by its within-image rank, normalized to [0, 1].
    Stable for the q8 quantization step.
    """
    a = arr.astype(np.float32).ravel()
    n = a.size
    if n == 0:
        return arr.astype(np.float32)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    return ranks.reshape(arr.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Test scoring (streaming per sample group)
# ─────────────────────────────────────────────────────────────────────────────
def score_test_streaming(
    decoded: list[dict],
    methods: list[str],
    class_map: dict[str, str],
    final_fits: dict[str, dict],
    *,
    multiview_mode: str,
    multiview_beta: float,
    smooth_sigma: float,
    small_cc_per_family: dict[str, int],
    family_per_run: list[str],
    priors: dict[str, np.ndarray] | None,
    prior_gamma: float,
    use_within_image_rank: bool,
    patchcore_k: int,
) -> dict[str, np.ndarray]:
    """Walks every test sample group (class, sample_id), computes 14-d
    rank vectors per view, applies the per-class winning family scorer,
    multi-view consensus, post-process, returns {test_id: float32 (H, W)
    in [0, 1]} for downstream q8 encoding.
    """
    # Index test IDs by (class, sample_id).
    all_ids: list[str] = []
    for d in decoded:
        all_ids.extend(d.keys())
    all_ids = sorted(set(all_ids))

    _default_cls = next(iter(set(class_map.values())), "_unknown_")
  #  sample_groups = group_test_ids_by_sample(all_ids, class_map, _default_cls)
    sample_groups_flat = group_test_ids_by_sample(all_ids, class_map, _default_cls)
    # Re-attach views: (cls, sid) -> list[(tid, view)]
    sample_groups: dict[tuple[str, str], list[tuple[str, int | None]]] = {}
    for (cls, sid), tids in sample_groups_flat.items():
        pairs = []
        for tid in tids:
            _, v = parse_sample_id(tid)
            pairs.append((tid, v))
        sample_groups[(cls, sid)] = pairs
    results: dict[str, np.ndarray] = {}
    n_groups = len(sample_groups)
    last_log = 0
    t0 = time.time()
    for grp_i, ((cls, sid), id_view_pairs) in enumerate(
            sorted(sample_groups.items())):
        fit = final_fits.get(cls)
        if fit is None or fit["params"] is None:
            # Fall back to per-pixel mean rank.
            for tid, _ in id_view_pairs:
                shape = None
                for d in decoded:
                    if tid in d:
                        shape = d[tid].shape
                        break
                if shape is None:
                    continue
                results[tid] = np.zeros(shape, dtype=np.float32)
            continue
        fname = fit["family"]
        params = fit["params"]
        fam = FAMILIES[fname]

        # Assemble (V, H, W, M) rank tensor for this sample.
        id_view_pairs = sorted(id_view_pairs, key=lambda p: (p[1] or 0))
        per_view_ranks: list[np.ndarray] = []
        view_tids: list[str] = []
        for tid, view in id_view_pairs:
            slices: list[np.ndarray] = []
            H = W = None
            for j, m in enumerate(methods):
                if tid not in decoded[j]:
                    if H is None:
                        # Find a peer view that has this method.
                        for d_other in decoded:
                            if tid in d_other:
                                H, W = d_other[tid].shape
                                break
                    if H is None:
                        H = W = 256
                    slices.append(np.zeros((H, W), dtype=np.float32))
                else:
                    arr = decoded[j][tid]
                    H, W = arr.shape
                    slices.append(arr.astype(np.float32))
            R = np.stack(slices, axis=-1)        # (H, W, M)
            per_view_ranks.append(R)
            view_tids.append(tid)

        V = len(per_view_ranks)
        if V == 0:
            continue
        H, W, M = per_view_ranks[0].shape

        # Score each view → (V, H, W)
        per_view_score = np.empty((V, H, W), dtype=np.float32)
        for v, R in enumerate(per_view_ranks):
            R_flat = R.reshape(-1, M)
            if fname == "patchcore":
                A = fam.score(params, R_flat, k=patchcore_k)
            else:
                A = fam.score(params, R_flat)
            per_view_score[v] = A.reshape(H, W)

        # Stage 4 — multi-view consensus.
        per_view_refined = multi_view_consensus(
            per_view_score, mode=multiview_mode, beta=multiview_beta)

        # Stage 5 — post-process each view independently and store.
        for v, tid in enumerate(view_tids):
            a = per_view_refined[v]
            # 5a — Gaussian smoothing (sparse oracle / 08/05)
            a = gaussian_smooth_2d(a, smooth_sigma)
            # 5b — small-CC drop (per-family threshold lookup, using a
            # default since we no longer have a single source family).
            min_area = max(small_cc_per_family.values(), default=0) // 4 \
                       if small_cc_per_family else 0
            if min_area > 0:
                a = drop_small_components(a, min_area=min_area, quantile=0.97)
            # 5c — spatial prior
            if priors and prior_gamma > 0:
                p = priors.get(cls)
                if p is not None:
                    a = apply_spatial_prior(a, p, gamma=prior_gamma)
            # 6 — within-image rank transform
            if use_within_image_rank:
                a = within_image_rank(a)
            results[tid] = a.astype(np.float32)

        if grp_i - last_log >= max(1, n_groups // 20):
            last_log = grp_i
            print(f"    test scoring  {grp_i + 1}/{n_groups} sample groups  "
                  f"({(time.time() - t0):.1f}s elapsed)", flush=True)
    print(f"    test scoring complete: {len(results)} maps "
          f"({(time.time() - t0):.1f}s)", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Submission writer (global percentile calibration + q8rle)
# ─────────────────────────────────────────────────────────────────────────────
def write_submission(
    results: dict[str, np.ndarray],
    run_dir: Path,
    *,
    submission_h: int = 224,
    submission_w: int = 224,
    calibrate_lo_pct: float = 0.1,
    calibrate_hi_pct: float = 99.9,
    zip_it: bool = True,
) -> Path:
    """Resize each map to (submission_h, submission_w), apply global
    percentile clip+rescale to [0, 1], encode as q8rle, write CSV+ZIP.
    """
    sub("writing submission")
    if not results:
        raise RuntimeError("no results to write")

    # Global percentile calibration over the pooled population of test
    # pixels — critical when the leaderboard is pooled-AP rather than
    # per-image AP averaged. For per-image AP this is invariant, so it
    # costs us nothing to apply.
    flat_parts: list[np.ndarray] = []
    # Take a subsample for percentile estimation; keeps it fast.
    keys = list(results.keys())
    pick_n = min(len(keys), 1024)
    rng = np.random.default_rng(0)
    pick_idx = rng.choice(len(keys), pick_n, replace=False)
    for i in pick_idx:
        flat_parts.append(results[keys[i]].ravel()[::4])
    sample = np.concatenate(flat_parts).astype(np.float32) if flat_parts \
              else np.array([0.0, 1.0], dtype=np.float32)
    lo = float(np.percentile(sample, calibrate_lo_pct))
    hi = float(np.percentile(sample, calibrate_hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    print(f"    global calibration: lo={lo:.4f} (p{calibrate_lo_pct})  "
          f"hi={hi:.4f} (p{calibrate_hi_pct})", flush=True)

    def _resize(arr: np.ndarray) -> np.ndarray:
        if arr.shape == (submission_h, submission_w):
            return arr
        # Pure numpy bilinear resize via scipy.zoom — avoids torch dep here.
        zy = submission_h / arr.shape[0]
        zx = submission_w / arr.shape[1]
        return ndi.zoom(arr.astype(np.float32), (zy, zx), order=1,
                         mode="reflect", grid_mode=False)

    csv_path = run_dir / "submission.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    t0 = time.time()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        for tid in sorted(results.keys()):
            arr = _resize(results[tid])
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            row_id = tid.split("/")[-1] if "/" in tid else tid
            w.writerow([row_id, float_matrix_to_q8rle(arr)])
            n_rows += 1
    print(f"    wrote {n_rows} rows -> {csv_path} ({time.time() - t0:.1f}s)",
          flush=True)
    if zip_it:
        zip_path = csv_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w",
                              compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
        print(f"    zipped         -> {zip_path}", flush=True)
        return zip_path
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EnsembleConfig:
    runs: list[Path] = field(default_factory=list)
    data_root: Path = Path("/workspace/anomaly-detection/data")
    report_dir: Path = Path("/workspace/anomaly-detection/baseline_out")
    cache_dir: Path = Path("/workspace/anomaly-detection/baseline_out/stacker_cache")
    families: list[str] = field(default_factory=lambda: [
        "padim", "gmm", "patchcore", "svdd"])
    multiview_mode: str = "max_minus_std"      # or "max" / "agreement_boost" / "passthrough"
    multiview_beta: float = 0.3
    smooth_sigma: float = 1.0
    prior_gamma: float = 0.5
    priors_dir: Path | None = None
    small_cc_spec: str = ""
    use_within_image_rank: bool = True
    calibrate_lo_pct: float = 0.1
    calibrate_hi_pct: float = 99.9
    submission_h: int = 224
    submission_w: int = 224
    # family hyperparameters
    gmm_K: int = 3
    patchcore_k: int = 5
    patchcore_bank: int = 6000
    svdd_epochs: int = 50
    st_epochs: int = 25
    enable_student_teacher: bool = False
    # housekeeping
    seed: int = 0
    device: str = "cuda"
    only_classes: list[str] = field(default_factory=list)
    skip_loao: bool = False
    force_family: str = ""                     # if set, override LOAO selection
    run_tag: str = ""
    rebuild_cache: bool = False


def make_run_id(cfg: EnsembleConfig, methods: list[str]) -> str:
    fp = {
        "tier": 0,
        "families": list(cfg.families),
        "multiview_mode": cfg.multiview_mode,
        "multiview_beta": cfg.multiview_beta,
        "smooth_sigma": cfg.smooth_sigma,
        "prior_gamma": cfg.prior_gamma,
        "small_cc_spec": cfg.small_cc_spec,
        "use_within_image_rank": cfg.use_within_image_rank,
        "calibrate_lo_pct": cfg.calibrate_lo_pct,
        "calibrate_hi_pct": cfg.calibrate_hi_pct,
        "gmm_K": cfg.gmm_K,
        "patchcore_k": cfg.patchcore_k,
        "patchcore_bank": cfg.patchcore_bank,
        "svdd_epochs": cfg.svdd_epochs,
        "st_epochs": cfg.st_epochs,
        "enable_student_teacher": cfg.enable_student_teacher,
        "seed": cfg.seed,
        "force_family": cfg.force_family,
        "methods": methods,
    }
    digest = hashlib.sha1(json.dumps(fp, sort_keys=True).encode("utf-8")) \
                     .hexdigest()[:8]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bits = f"{stamp}_tier0_mv-{cfg.multiview_mode}_b{cfg.multiview_beta:.2f}" \
           f"_sig{cfg.smooth_sigma:.1f}_pr{cfg.prior_gamma:.2f}"
    if cfg.force_family:
        bits += f"_force-{cfg.force_family}"
    if cfg.run_tag:
        bits += f"_{cfg.run_tag}"
    return f"{bits}_{digest}"


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
# def main():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         description=__doc__)
#     p.add_argument("--runs", type=Path, nargs="+", required=True,
#                    help="One run directory per base model (14 expected).")
#     p.add_argument("--data-root", type=Path,
#                    default=Path("/workspace/anomaly-detection/data"))
#     p.add_argument("--report-dir", type=Path,
#                    default=Path("/workspace/anomaly-detection/baseline_out"))
#     p.add_argument("--cache-dir", type=Path,
#                    default=Path("/workspace/anomaly-detection/baseline_out/stacker_cache"))
#     p.add_argument("--priors-dir", type=Path, default=None,
#                    help="Directory with class-conditional spatial-prior .npy "
#                         "files (e.g. analysis_out/tables/06_heat_*.npy).")
#     p.add_argument("--families", nargs="+",
#                    default=["padim", "gmm", "patchcore", "svdd"],
#                    choices=list(FAMILIES.keys()),
#                    help="Which families compete in LOAO.")
#     p.add_argument("--enable-student-teacher", action="store_true",
#                    help="Add student-teacher to the family pool. Off by "
#                         "default because it is the costliest fit.")
#     p.add_argument("--force-family", default="",
#                    choices=["", *list(FAMILIES.keys())],
#                    help="Skip LOAO and use this family for every class.")
#     p.add_argument("--multiview-mode", default="max_minus_std",
#                    choices=["passthrough", "max", "max_minus_std",
#                             "agreement_boost"])
#     p.add_argument("--multiview-beta", type=float, default=0.3)
#     p.add_argument("--smooth-sigma", type=float, default=1.0)
#     p.add_argument("--prior-gamma", type=float, default=0.5)
#     p.add_argument("--small-cc-spec", default="",
#                    help="Per-family small-CC suppression spec; passed to "
#                         "parse_small_cc_spec.")
#     p.add_argument("--no-within-image-rank", action="store_true",
#                    help="Disable Stage 6 within-image rank transform.")
#     p.add_argument("--calibrate-lo-pct", type=float, default=0.1)
#     p.add_argument("--calibrate-hi-pct", type=float, default=99.9)
#     p.add_argument("--submission-h", type=int, default=224)
#     p.add_argument("--submission-w", type=int, default=224)
#     p.add_argument("--gmm-K", type=int, default=3)
#     p.add_argument("--patchcore-k", type=int, default=5)
#     p.add_argument("--patchcore-bank", type=int, default=6000)
#     p.add_argument("--svdd-epochs", type=int, default=50)
#     p.add_argument("--st-epochs", type=int, default=25)
#     p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--device", default="cuda",
#                    choices=["cuda", "cpu"])
#     p.add_argument("--only-classes", nargs="*", default=[])
#     p.add_argument("--skip-loao", action="store_true",
#                    help="With --force-family, skip LOAO entirely (faster).")
#     p.add_argument("--rebuild-cache", action="store_true",
#                    help="Invalidate val-alignment and decoded-test caches.")
#     p.add_argument("--run-tag", default="")
#     args = p.parse_args()
#
#     if args.enable_student_teacher and \
#        "student_teacher" not in args.families:
#         args.families = list(args.families) + ["student_teacher"]
#     if args.force_family and args.force_family not in args.families:
#         args.families = [args.force_family]
#
#     cfg = EnsembleConfig(
#         runs=[Path(r) for r in args.runs],
#         data_root=Path(args.data_root),
#         report_dir=Path(args.report_dir),
#         cache_dir=Path(args.cache_dir),
#         families=list(args.families),
#         multiview_mode=args.multiview_mode,
#         multiview_beta=args.multiview_beta,
#         smooth_sigma=args.smooth_sigma,
#         prior_gamma=args.prior_gamma,
#         priors_dir=Path(args.priors_dir) if args.priors_dir else None,
#         small_cc_spec=args.small_cc_spec,
#         use_within_image_rank=(not args.no_within_image_rank),
#         calibrate_lo_pct=args.calibrate_lo_pct,
#         calibrate_hi_pct=args.calibrate_hi_pct,
#         submission_h=args.submission_h,
#         submission_w=args.submission_w,
#         gmm_K=args.gmm_K,
#         patchcore_k=args.patchcore_k,
#         patchcore_bank=args.patchcore_bank,
#         svdd_epochs=args.svdd_epochs,
#         st_epochs=args.st_epochs,
#         enable_student_teacher=args.enable_student_teacher,
#         seed=args.seed,
#         device=args.device,
#         only_classes=list(args.only_classes),
#         skip_loao=args.skip_loao,
#         force_family=args.force_family,
#         run_tag=args.run_tag,
#         rebuild_cache=args.rebuild_cache,
#     )
#
#     # Method identification = the run-directory basename per run.
#     methods = [r.name for r in cfg.runs]
#     run_id = make_run_id(cfg, methods)
#     out_dir = cfg.report_dir / "tier0_ensembles" / run_id
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     with tee_to(out_dir / "run_log.txt"):
#         hr(f"TIER-0 ENSEMBLE — {run_id}", "█")
#         print(f"  n_methods        : {len(methods)}")
#         print(f"  families         : {cfg.families}")
#         print(f"  multiview        : {cfg.multiview_mode}  "
#               f"beta={cfg.multiview_beta}")
#         print(f"  smooth_sigma     : {cfg.smooth_sigma}")
#         print(f"  prior_gamma      : {cfg.prior_gamma}")
#         print(f"  small_cc_spec    : {cfg.small_cc_spec!r}")
#         print(f"  within_img_rank  : {cfg.use_within_image_rank}")
#         print(f"  calibrate_pct    : ({cfg.calibrate_lo_pct}, "
#               f"{cfg.calibrate_hi_pct})")
#         print(f"  force_family     : {cfg.force_family!r}")
#         print(f"  device           : {cfg.device}")
#         print(f"  out_dir          : {out_dir}")
#
#         # Persist the config as JSON for repro.
#         with open(out_dir / "ensemble_config.json", "w") as f:
#             json.dump({
#                 "run_id": run_id,
#                 "methods": methods,
#                 **{k: (str(v) if isinstance(v, Path)
#                        else [str(x) if isinstance(x, Path) else x for x in v]
#                        if isinstance(v, list) else v)
#                    for k, v in asdict(cfg).items()},
#             }, f, indent=2)
#
#         # ── Load + align validation predictions (reuse v6 cache) ──────────
#         hr("Stage 0 — load + align base-model predictions")
#         cache = Cache(cfg.cache_dir)
#
#         # Per-run local_predictions.npz files.
#         npz_paths = [r / "local_predictions.npz" for r in cfg.runs]
#         for q in npz_paths:
#             if not q.exists():
#                 raise SystemExit(f"[FATAL] missing: {q}")
#         # Test submission CSVs.
#         sub_paths = [r / "submission.csv" for r in cfg.runs]
#         for q in sub_paths:
#             if not q.exists():
#                 # accept .zip too
#                 qz = q.with_suffix(".zip")
#                 if not qz.exists():
#                     raise SystemExit(f"[FATAL] missing: {q} (and .zip)")
#
#         meta = file_meta_hash(npz_paths) + ":" + file_meta_hash(sub_paths)
#         val_cache_key = "tier0_val_aligned_" + hashlib.sha1(meta.encode()).hexdigest()[:12]
#         val_path = cfg.cache_dir / f"{val_cache_key}.npz"
#         if val_path.exists() and not cfg.rebuild_cache:
#             # print(f"  loading cached aligned val -> {val_path}")
#             # val = load_aligned_val(val_path)
#             print(f"  loading cached aligned val -> {val_path}")
#             d = np.load(val_path, allow_pickle=True)
#             paths = d["image_paths"]
#             val = {
#                 "ids": d["ids"].astype(str),
#                 "classes": d["classes"].astype(str),
#                 "anomaly_types": d["anomaly_types"].astype(str),
#                 "views": d["views"].astype(np.int32),
#                 "scores": d["scores"].astype(np.float32),
#                 "masks": d["masks"].astype(np.uint8),
#                 "image_paths": (None if paths.size == 0 else paths.astype(str)),
#             }
#         else:
#             # print("  loading + aligning local_predictions.npz from each run...")
#             # per_run_val = []
#             # for run_dir, npz in zip(cfg.runs, npz_paths):
#             #     v = load_local_preds(npz)
#             #     per_run_val.append((run_dir.name, v))
#             # val = align_local_preds(per_run_val)
#             # save_aligned_val(val, val_path)
#             print("  loading + aligning local_predictions.npz from each run...")
#             per_run_val = []
#             run_names: list[str] = []
#             for run_dir, npz in zip(cfg.runs, npz_paths):
#                 per_run_val.append(load_local_preds(npz))
#                 run_names.append(run_dir.name)
#             val = align_local_preds(per_run_val, run_names)
#             save_aligned_val(val, val_path)
#
#         N_val = val["scores"].shape[0]
#         H_val, W_val = val["scores"].shape[1:3]
#         M = val["scores"].shape[3]
#         print(f"  aligned val: N={N_val}  H,W=({H_val},{W_val})  M={M}")
#         print(f"  memory      : {fmt_bytes(val['scores'].nbytes)} scores + "
#               f"{fmt_bytes(val['masks'].nbytes)} masks")
#
#         # Build class map.
#         class_map = build_class_map_from_data(cfg.data_root)
#         if cfg.only_classes:
#             wanted = set(cfg.only_classes)
#             class_map = {k: v for k, v in class_map.items() if v in wanted}
#
#         # ── Stage 1 — ECDFs ───────────────────────────────────────────────
#         hr("Stage 1 — per-(class, view, method) ECDF rank-norm")
#         ecdfs = build_ecdfs_from_val(val, methods, class_map,
#                                        seed=cfg.seed)
#         print("  rank-normalizing val in place...")
#         rank_norm_val_inplace(val, methods, class_map, ecdfs)
#
#         # Save the ECDFs (small) for downstream inspection.
#         ecdfs_path = out_dir / "ecdfs.pkl"
#         with open(ecdfs_path, "wb") as f:
#             pickle.dump(ecdfs, f)
#         print(f"  saved ECDFs -> {ecdfs_path}")
#
#         # Index val by class (per-type normal/anomaly pixel pools).
#         ci_by_class = index_val_by_class(val, class_map, methods)
#         for cls, ci in sorted(ci_by_class.items()):
#             tot_neg = sum(v.shape[0] for v in ci.per_type_normal.values())
#             tot_pos = sum(v.shape[0] for v in ci.per_type_anom.values())
#             n_types = len(ci.per_type_normal)
#             print(f"    class {cls}: {n_types} anomaly types  "
#                   f"normals={tot_neg:,}  anomalies={tot_pos:,}")
#
#         # ── Stage 3 — LOAO winner selection ───────────────────────────────
#         if cfg.force_family and cfg.skip_loao:
#             hr(f"Stage 3 — forced family={cfg.force_family} (LOAO skipped)")
#             winners = {cls: cfg.force_family for cls in ci_by_class}
#             loao_results: dict[str, dict[str, dict[str, float]]] = {
#                 cls: {cfg.force_family: {"mean": float("nan")}}
#                 for cls in ci_by_class}
#         else:
#             hr("Stage 3 — LOAO over anomaly types per class")
#             loao_results = {}
#             for cls in sorted(ci_by_class.keys()):
#                 ci = ci_by_class[cls]
#                 print(f"  ── class {cls} ──", flush=True)
#                 res = loao_one_class(
#                     ci, cfg.families,
#                     seed=cfg.seed, device=cfg.device,
#                     gmm_K=cfg.gmm_K, patchcore_k=cfg.patchcore_k,
#                     patchcore_bank=cfg.patchcore_bank,
#                     svdd_epochs=cfg.svdd_epochs,
#                     st_epochs=cfg.st_epochs,
#                 )
#                 loao_results[cls] = res
#                 # Summary line.
#                 summary = "  ".join(f"{f}={res[f]['mean']:.4f}"
#                                      for f in cfg.families
#                                      if f in res)
#                 print(f"    SUMMARY class={cls}  {summary}", flush=True)
#             winners = select_winners(loao_results)
#             if cfg.force_family:
#                 # Override.
#                 winners = {cls: cfg.force_family for cls in winners}
#         print("\n  per-class winners:")
#         for cls in sorted(winners):
#             mean_ap = loao_results.get(cls, {}).get(winners[cls], {}).get("mean", float("nan"))
#             print(f"    {cls:<12} -> {winners[cls]:<18}  "
#                   f"mean LOAO AP={mean_ap:.4f}")
#
#         with open(out_dir / "loao_results.json", "w") as f:
#             json.dump(loao_results, f, indent=2)
#         with open(out_dir / "winners.json", "w") as f:
#             json.dump(winners, f, indent=2)
#
#         # ── Final per-class fit on ALL negatives ──────────────────────────
#         hr("Stage 3b — final fit per class on all val negatives")
#         final_fits = fit_final_per_class(
#             ci_by_class, winners,
#             seed=cfg.seed, device=cfg.device,
#             gmm_K=cfg.gmm_K, patchcore_bank=cfg.patchcore_bank,
#             svdd_epochs=cfg.svdd_epochs, st_epochs=cfg.st_epochs,
#         )
#
#         # Free val score tensor before going to test (large).
#         del val
#         mem_print("after dropping val scores")
#
#         # ── Stage 0b — decode + rank-norm test ────────────────────────────
#         hr("Stage 0b — decode test submissions and rank-norm in place")
#         decoded_cache_key = "tier0_decoded_raw_" + \
#             hashlib.sha1(file_meta_hash(sub_paths).encode()).hexdigest()[:12]
#         decoded_path = cfg.cache_dir / f"{decoded_cache_key}.pkl"
#         if decoded_path.exists() and not cfg.rebuild_cache:
#             print(f"  loading cached decoded test -> {decoded_path}")
#             decoded = load_decoded_test(decoded_path)
#         else:
#             print("  decoding test submissions from q8rle...")
#             decoded = decode_submissions_to_uint8(sub_paths)
#             cache_decoded_test(decoded, decoded_path)
#         n_per_method = [len(d) for d in decoded]
#         print(f"  decoded test counts per method: {n_per_method}")
#
#         print("  rank-normalizing test per (class, view, method)...")
#         rank_norm_test_per_view_inplace(decoded, methods, class_map, ecdfs)
#         mem_print("after rank-norm test")
#
#         # ── Load priors ───────────────────────────────────────────────────
#         priors: dict[str, np.ndarray] | None = None
#         if cfg.priors_dir and cfg.prior_gamma > 0:
#             print(f"  loading spatial priors from {cfg.priors_dir}")
#             try:
#                 priors = load_spatial_priors(cfg.priors_dir)
#                 print(f"  loaded priors for: {sorted(priors.keys())}")
#             except Exception as e:
#                 print(f"  [WARN] failed to load priors: {e}")
#                 priors = None
#
#         # Small-CC spec.
#         small_cc_per_family: dict[str, int] = parse_small_cc_spec(
#             cfg.small_cc_spec) if cfg.small_cc_spec \
#             else dict(DEFAULT_SMALL_CC_PER_FAMILY)
#
#         # ── Stage 4-6 — test scoring + post-process ───────────────────────
#         hr("Stage 4-6 — test scoring with per-class winning family")
#         # Family per run: irrelevant in tier-0 (we score by winning family,
#         # not per source method), but kept for the small-CC lookup API.
#         family_per_run = [detect_model_family(r) for r in cfg.runs]
#         results = score_test_streaming(
#             decoded, methods, class_map, final_fits,
#             multiview_mode=cfg.multiview_mode,
#             multiview_beta=cfg.multiview_beta,
#             smooth_sigma=cfg.smooth_sigma,
#             small_cc_per_family=small_cc_per_family,
#             family_per_run=family_per_run,
#             priors=priors,
#             prior_gamma=cfg.prior_gamma,
#             use_within_image_rank=cfg.use_within_image_rank,
#             patchcore_k=cfg.patchcore_k,
#         )
#
#         # ── Submission ────────────────────────────────────────────────────
#         hr("Submission")
#         sub_path = write_submission(
#             results, out_dir,
#             submission_h=cfg.submission_h,
#             submission_w=cfg.submission_w,
#             calibrate_lo_pct=cfg.calibrate_lo_pct,
#             calibrate_hi_pct=cfg.calibrate_hi_pct,
#             zip_it=True,
#         )
#
#         # ── Append a master ablation row ──────────────────────────────────
#         master_csv = cfg.report_dir / "tier0_ablation_master.csv"
#         row = {
#             "run_id": run_id,
#             "run_tag": cfg.run_tag,
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "n_methods": len(methods),
#             "families": "+".join(cfg.families),
#             "force_family": cfg.force_family,
#             "multiview_mode": cfg.multiview_mode,
#             "multiview_beta": cfg.multiview_beta,
#             "smooth_sigma": cfg.smooth_sigma,
#             "prior_gamma": cfg.prior_gamma,
#             "calibrate_lo_pct": cfg.calibrate_lo_pct,
#             "calibrate_hi_pct": cfg.calibrate_hi_pct,
#             "use_within_image_rank": int(cfg.use_within_image_rank),
#             "gmm_K": cfg.gmm_K,
#             "patchcore_k": cfg.patchcore_k,
#             "patchcore_bank": cfg.patchcore_bank,
#             "svdd_epochs": cfg.svdd_epochs,
#             "st_epochs": cfg.st_epochs,
#             "seed": cfg.seed,
#             **{f"winner_{cls}": winners.get(cls, "")
#                 for cls in sorted(ci_by_class.keys())},
#             **{f"loao_mean_AP_{cls}":
#                 f"{loao_results.get(cls, {}).get(winners.get(cls, ''), {}).get('mean', 0.0):.4f}"
#                 for cls in sorted(ci_by_class.keys())},
#             "submission_path": str(sub_path),
#         }
#         try:
#             append_to_master(master_csv, row)
#             print(f"  ablation row appended -> {master_csv}")
#         except Exception as e:
#             print(f"  [WARN] failed to append ablation row: {e}")
#
#         hr(f"DONE — run_id={run_id}", "█")

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True,
                   help="One run directory per base model (14 expected).")
    p.add_argument("--data-root", type=Path,
                   default=Path("/workspace/anomaly-detection/data"))
    p.add_argument("--report-dir", type=Path,
                   default=Path("/workspace/anomaly-detection/baseline_out"))
    p.add_argument("--cache-dir", type=Path,
                   default=Path("/workspace/anomaly-detection/baseline_out/stacker_cache"))
    p.add_argument("--priors-dir", type=Path, default=None,
                   help="Directory with class-conditional spatial-prior .npy "
                        "files (e.g. analysis_out/tables/06_heat_*.npy).")
    p.add_argument("--families", nargs="+",
                   default=["padim", "gmm", "patchcore", "svdd"],
                   choices=list(FAMILIES.keys()),
                   help="Which families compete in LOAO.")
    p.add_argument("--enable-student-teacher", action="store_true",
                   help="Add student-teacher to the family pool. Off by "
                        "default because it is the costliest fit.")
    p.add_argument("--force-family", default="",
                   choices=["", *list(FAMILIES.keys())],
                   help="Skip LOAO and use this family for every class.")
    p.add_argument("--multiview-mode", default="max_minus_std",
                   choices=["passthrough", "max", "max_minus_std",
                            "agreement_boost"])
    p.add_argument("--multiview-beta", type=float, default=0.3)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--prior-gamma", type=float, default=0.5)
    p.add_argument("--small-cc-spec", default="",
                   help="Per-family small-CC suppression spec (key=val,key=val).")
    p.add_argument("--no-within-image-rank", action="store_true",
                   help="Disable Stage 6 within-image rank transform.")
    p.add_argument("--calibrate-lo-pct", type=float, default=0.1)
    p.add_argument("--calibrate-hi-pct", type=float, default=99.9)
    p.add_argument("--submission-h", type=int, default=224)
    p.add_argument("--submission-w", type=int, default=224)
    p.add_argument("--gmm-K", type=int, default=3)
    p.add_argument("--patchcore-k", type=int, default=5)
    p.add_argument("--patchcore-bank", type=int, default=6000)
    p.add_argument("--svdd-epochs", type=int, default=50)
    p.add_argument("--st-epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--only-classes", nargs="*", default=[])
    p.add_argument("--skip-loao", action="store_true",
                   help="With --force-family, skip LOAO entirely (faster).")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Invalidate val-alignment and decoded-test caches.")
    p.add_argument("--run-tag", default="")
    args = p.parse_args()

    if args.enable_student_teacher and \
       "student_teacher" not in args.families:
        args.families = list(args.families) + ["student_teacher"]
    if args.force_family and args.force_family not in args.families:
        args.families = [args.force_family]

    cfg = EnsembleConfig(
        runs=[Path(r) for r in args.runs],
        data_root=Path(args.data_root),
        report_dir=Path(args.report_dir),
        cache_dir=Path(args.cache_dir),
        families=list(args.families),
        multiview_mode=args.multiview_mode,
        multiview_beta=args.multiview_beta,
        smooth_sigma=args.smooth_sigma,
        prior_gamma=args.prior_gamma,
        priors_dir=Path(args.priors_dir) if args.priors_dir else None,
        small_cc_spec=args.small_cc_spec,
        use_within_image_rank=(not args.no_within_image_rank),
        calibrate_lo_pct=args.calibrate_lo_pct,
        calibrate_hi_pct=args.calibrate_hi_pct,
        submission_h=args.submission_h,
        submission_w=args.submission_w,
        gmm_K=args.gmm_K,
        patchcore_k=args.patchcore_k,
        patchcore_bank=args.patchcore_bank,
        svdd_epochs=args.svdd_epochs,
        st_epochs=args.st_epochs,
        enable_student_teacher=args.enable_student_teacher,
        seed=args.seed,
        device=args.device,
        only_classes=list(args.only_classes),
        skip_loao=args.skip_loao,
        force_family=args.force_family,
        run_tag=args.run_tag,
        rebuild_cache=args.rebuild_cache,
    )

    # Method identification = the run-directory basename per run.
    methods = [r.name for r in cfg.runs]
    run_id = make_run_id(cfg, methods)
    out_dir = cfg.report_dir / "tier0_ensembles" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with tee_to(out_dir / "run_log.txt"):
        hr(f"TIER-0 ENSEMBLE — {run_id}", "█")
        print(f"  n_methods        : {len(methods)}")
        print(f"  families         : {cfg.families}")
        print(f"  multiview        : {cfg.multiview_mode}  "
              f"beta={cfg.multiview_beta}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  prior_gamma      : {cfg.prior_gamma}")
        print(f"  small_cc_spec    : {cfg.small_cc_spec!r}")
        print(f"  within_img_rank  : {cfg.use_within_image_rank}")
        print(f"  calibrate_pct    : ({cfg.calibrate_lo_pct}, "
              f"{cfg.calibrate_hi_pct})")
        print(f"  force_family     : {cfg.force_family!r}")
        print(f"  device           : {cfg.device}")
        print(f"  out_dir          : {out_dir}")

        # Persist the config as JSON for repro.
        with open(out_dir / "ensemble_config.json", "w") as f:
            json.dump({
                "run_id": run_id,
                "methods": methods,
                **{k: (str(v) if isinstance(v, Path)
                       else [str(x) if isinstance(x, Path) else x for x in v]
                       if isinstance(v, list) else v)
                   for k, v in asdict(cfg).items()},
            }, f, indent=2)

        # ── Load + align validation predictions ───────────────────────────
        hr("Stage 0 — load + align base-model predictions")
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)

        npz_paths = [r / "local_predictions.npz" for r in cfg.runs]
        for q in npz_paths:
            if not q.exists():
                raise SystemExit(f"[FATAL] missing: {q}")
        sub_paths = [r / "submission.csv" for r in cfg.runs]
        for q in sub_paths:
            if not q.exists():
                qz = q.with_suffix(".zip")
                if not qz.exists():
                    raise SystemExit(f"[FATAL] missing: {q} (and .zip)")

        meta = file_meta_hash(npz_paths) + ":" + file_meta_hash(sub_paths)
        val_cache_key = "tier0_val_aligned_" + hashlib.sha1(meta.encode()).hexdigest()[:12]
        val_path = cfg.cache_dir / f"{val_cache_key}.npz"

        if val_path.exists() and not cfg.rebuild_cache:
            print(f"  loading cached aligned val -> {val_path}")
            d = np.load(val_path, allow_pickle=True)
            paths = d["image_paths"]
            val = {
                "ids":           d["ids"].astype(str),
                "classes":       d["classes"].astype(str),
                "anomaly_types": d["anomaly_types"].astype(str),
                "views":         d["views"].astype(np.int32),
                "scores":        d["scores"].astype(np.float32),
                "masks":         d["masks"].astype(np.uint8),
                "image_paths":   (None if paths.size == 0 else paths.astype(str)),
            }
        else:
            print("  loading + aligning local_predictions.npz from each run...")
            per_run_val: list[dict] = []
            run_names: list[str] = []
            for run_dir, npz in zip(cfg.runs, npz_paths):
                per_run_val.append(load_local_preds(npz))
                run_names.append(run_dir.name)
            val = align_local_preds(per_run_val, run_names)
            paths = val.get("image_paths")
            np.savez_compressed(
                val_path,
                ids=val["ids"],
                classes=val["classes"],
                anomaly_types=val["anomaly_types"],
                views=val["views"],
                scores=val["scores"],
                masks=val["masks"],
                image_paths=(np.array([], dtype=object)
                             if paths is None else paths),
            )
            print(f"  saved aligned val -> {val_path}")
            del per_run_val

        N_val = val["scores"].shape[0]
        H_val, W_val = val["scores"].shape[1:3]
        M = val["scores"].shape[3]
        print(f"  aligned val: N={N_val}  H,W=({H_val},{W_val})  M={M}")
        print(f"  memory      : {fmt_bytes(val['scores'].nbytes)} scores + "
              f"{fmt_bytes(val['masks'].nbytes)} masks")

        # Build class map.
        class_map = build_class_map_from_data(cfg.data_root)
        if cfg.only_classes:
            wanted = set(cfg.only_classes)
            class_map = {k: v for k, v in class_map.items() if v in wanted}

        # ── Stage 1 — ECDFs ───────────────────────────────────────────────
        hr("Stage 1 — per-(class, view, method) ECDF rank-norm")
        ecdfs = build_ecdfs_from_val(val, methods, class_map, seed=cfg.seed)
        print("  rank-normalizing val in place...")
        rank_norm_val_inplace(val, methods, class_map, ecdfs)

        ecdfs_path = out_dir / "ecdfs.pkl"
        with open(ecdfs_path, "wb") as f:
            pickle.dump(ecdfs, f)
        print(f"  saved ECDFs -> {ecdfs_path}")

        # Index val by class (per-type normal/anomaly pixel pools).
        ci_by_class = index_val_by_class(val, class_map, methods)
        for cls, ci in sorted(ci_by_class.items()):
            tot_neg = sum(v.shape[0] for v in ci.per_type_normal.values())
            tot_pos = sum(v.shape[0] for v in ci.per_type_anom.values())
            n_types = len(ci.per_type_normal)
            print(f"    class {cls}: {n_types} anomaly types  "
                  f"normals={tot_neg:,}  anomalies={tot_pos:,}")

        classes_list = sorted(ci_by_class.keys())
        default_class = classes_list[0] if classes_list else "_unknown_"

        # ── Stage 3 — LOAO winner selection ───────────────────────────────
        if cfg.force_family and cfg.skip_loao:
            hr(f"Stage 3 — forced family={cfg.force_family} (LOAO skipped)")
            winners = {cls: cfg.force_family for cls in ci_by_class}
            loao_results: dict[str, dict[str, dict[str, float]]] = {
                cls: {cfg.force_family: {"mean": float("nan")}}
                for cls in ci_by_class}
        else:
            hr("Stage 3 — LOAO over anomaly types per class")
            loao_results = {}
            for cls in sorted(ci_by_class.keys()):
                ci = ci_by_class[cls]
                print(f"  ── class {cls} ──", flush=True)
                res = loao_one_class(
                    ci, cfg.families,
                    seed=cfg.seed, device=cfg.device,
                    gmm_K=cfg.gmm_K, patchcore_k=cfg.patchcore_k,
                    patchcore_bank=cfg.patchcore_bank,
                    svdd_epochs=cfg.svdd_epochs,
                    st_epochs=cfg.st_epochs,
                )
                loao_results[cls] = res
                summary = "  ".join(f"{f}={res[f]['mean']:.4f}"
                                     for f in cfg.families
                                     if f in res)
                print(f"    SUMMARY class={cls}  {summary}", flush=True)
            winners = select_winners(loao_results)
            if cfg.force_family:
                winners = {cls: cfg.force_family for cls in winners}
        print("\n  per-class winners:")
        for cls in sorted(winners):
            mean_ap = loao_results.get(cls, {}).get(winners[cls], {}).get("mean", float("nan"))
            print(f"    {cls:<12} -> {winners[cls]:<18}  "
                  f"mean LOAO AP={mean_ap:.4f}")

        with open(out_dir / "loao_results.json", "w") as f:
            json.dump(loao_results, f, indent=2)
        with open(out_dir / "winners.json", "w") as f:
            json.dump(winners, f, indent=2)

        # ── Final per-class fit on ALL negatives ──────────────────────────
        hr("Stage 3b — final fit per class on all val negatives")
        final_fits = fit_final_per_class(
            ci_by_class, winners,
            seed=cfg.seed, device=cfg.device,
            gmm_K=cfg.gmm_K, patchcore_bank=cfg.patchcore_bank,
            svdd_epochs=cfg.svdd_epochs, st_epochs=cfg.st_epochs,
        )

        # Free val score tensor before going to test (large).
        del val
        mem_print("after dropping val scores")

        # ── Stage 0b — decode + rank-norm test ────────────────────────────
        hr("Stage 0b — decode test submissions and rank-norm in place")
        decoded_cache_key = "tier0_decoded_raw_" + \
            hashlib.sha1(file_meta_hash(sub_paths).encode()).hexdigest()[:12]
        decoded_path = cfg.cache_dir / f"{decoded_cache_key}.pkl"
        if decoded_path.exists() and not cfg.rebuild_cache:
            print(f"  loading cached decoded test -> {decoded_path}")
            with open(decoded_path, "rb") as f:
                decoded = pickle.load(f)
        else:
            print("  loading test submissions...")
            submissions = [load_submission(p) for p in sub_paths]
            common = set.intersection(*[set(s.keys()) for s in submissions])
            if not common:
                raise SystemExit("[FATAL] no test IDs in common across submissions")
            all_ids = sorted(common)
            print(f"  decoding {len(submissions)} submissions × "
                  f"{len(all_ids)} ids each from q8rle...")
            decoded = decode_submissions_to_uint8(submissions, all_ids)
            with open(decoded_path, "wb") as f:
                pickle.dump(decoded, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  saved decoded test -> {decoded_path}")
            del submissions
        n_per_method = [len(d) for d in decoded]
        print(f"  decoded test counts per method: {n_per_method}")

        print("  rank-normalizing test per (class, view, method)...")
        rank_norm_test_per_view_inplace(decoded, methods, class_map, ecdfs)
        mem_print("after rank-norm test")

        # ── Load priors ───────────────────────────────────────────────────
        priors: dict[str, np.ndarray] | None = None
        if cfg.priors_dir and cfg.prior_gamma > 0:
            print(f"  loading spatial priors from {cfg.priors_dir}")
            try:
                priors = load_spatial_priors(cfg.priors_dir, classes_list)
                print(f"  loaded priors for: {sorted(priors.keys())}")
            except Exception as e:
                print(f"  [WARN] failed to load priors: {e}")
                priors = None

        # Small-CC spec (inline parser; v6's parse_small_cc_spec returns a
        # list[int] keyed on methods, which doesn't match our family API).
        small_cc_per_family: dict[str, int] = dict(DEFAULT_SMALL_CC_PER_FAMILY)
        if cfg.small_cc_spec:
            for kv in cfg.small_cc_spec.replace(";", ",").split(","):
                kv = kv.strip()
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        small_cc_per_family[k.strip()] = int(v)
                    except ValueError:
                        print(f"  [WARN] bad small-cc spec entry: {kv!r}")

        # ── Stage 4-6 — test scoring + post-process ───────────────────────
        hr("Stage 4-6 — test scoring with per-class winning family")
        family_per_run = [detect_model_family(r.name) for r in cfg.runs]
        results = score_test_streaming(
            decoded, methods, class_map, final_fits,
            multiview_mode=cfg.multiview_mode,
            multiview_beta=cfg.multiview_beta,
            smooth_sigma=cfg.smooth_sigma,
            small_cc_per_family=small_cc_per_family,
            family_per_run=family_per_run,
            priors=priors,
            prior_gamma=cfg.prior_gamma,
            use_within_image_rank=cfg.use_within_image_rank,
            patchcore_k=cfg.patchcore_k,
        )

        # ── Submission ────────────────────────────────────────────────────
        hr("Submission")
        sub_path = write_submission(
            results, out_dir,
            submission_h=cfg.submission_h,
            submission_w=cfg.submission_w,
            calibrate_lo_pct=cfg.calibrate_lo_pct,
            calibrate_hi_pct=cfg.calibrate_hi_pct,
            zip_it=True,
        )

        # ── Append a master ablation row ──────────────────────────────────
        master_csv = cfg.report_dir / "tier0_ablation_master.csv"
        row = {
            "run_id": run_id,
            "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_methods": len(methods),
            "families": "+".join(cfg.families),
            "force_family": cfg.force_family,
            "multiview_mode": cfg.multiview_mode,
            "multiview_beta": cfg.multiview_beta,
            "smooth_sigma": cfg.smooth_sigma,
            "prior_gamma": cfg.prior_gamma,
            "calibrate_lo_pct": cfg.calibrate_lo_pct,
            "calibrate_hi_pct": cfg.calibrate_hi_pct,
            "use_within_image_rank": int(cfg.use_within_image_rank),
            "gmm_K": cfg.gmm_K,
            "patchcore_k": cfg.patchcore_k,
            "patchcore_bank": cfg.patchcore_bank,
            "svdd_epochs": cfg.svdd_epochs,
            "st_epochs": cfg.st_epochs,
            "seed": cfg.seed,
            **{f"winner_{cls}": winners.get(cls, "")
                for cls in sorted(ci_by_class.keys())},
            **{f"loao_mean_AP_{cls}":
                f"{loao_results.get(cls, {}).get(winners.get(cls, ''), {}).get('mean', 0.0):.4f}"
                for cls in sorted(ci_by_class.keys())},
            "submission_path": str(sub_path),
        }
        try:
            append_to_master(master_csv, row)
            print(f"  ablation row appended -> {master_csv}")
        except Exception as e:
            print(f"  [WARN] failed to append ablation row: {e}")

        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()