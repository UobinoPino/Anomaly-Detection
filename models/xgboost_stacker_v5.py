"""XGBoost stacker v4 — analysis-driven feature additions over v3.

Direct successor to xgboost_stacker_v3.py. Same I/O contract as v3:
N submission.csv + N local_predictions.npz → fused submission +
stacker_config.json + oof_predictions.npz + run_log.txt + ablation row.

Every v3 feature is preserved verbatim and stays enabled by default. The
v4 additions below are wired ON by default because they're all
"information-only" features: XGBoost can split on them if useful and
ignore them if not. Each one is justified by a section of the latest
analyze_spacepresso_dataset.py / analyze_predictions.py report.

# v4 NEW: per-class SPATIAL PRIOR heatmap feature
  Dataset §12 measured the standalone pixel-AP of the per-class mean
  defect heatmap (computed in dataset §6). Some classes get serious AP
  out of nothing but the prior:
    class_06 = 0.60   class_08 = 0.39   class_07 = 0.31
  We load tables/06_heat_<class>.npy from --prior-heatmaps-dir,
  resize to each image's (H, W), and feed it as a single per-pixel
  feature `s_prior`. XGBoost can mix it with the model scores. Disable
  with --no-spatial-prior.

# v4 NEW: CLASS one-hot
  Predictions §5 shows the best-per-class model varies; per-class XGB
  already handles this for tuning, but the SHARED fallback model loses
  the class signal. With class one-hot it can still split on class
  inside the SHARED tree ensemble. 8 features (one per detected class).

# v4 NEW: VIEW one-hot (+ "unknown" bucket)
  Dataset §9: I(view; loc_quadrant | class) = 0.131 nats, I(view;
  area_quartile | class) = 0.153. View is a real conditioning variable
  for where defects appear inside an object. View one-hot (5 dims
  + 1 unknown) lets the trees split on view inside any model.

# v4 NEW: --rank-norm per-class-view  (third granularity)
  Predictions §3 flagged many (model, class, view) cells with
  Δp95/p95_mean > 0.30 (uniad_12mv class_01 Δp95=0.71, effad_nomv
  class_06 Δp95=1.36). v3 only rank-normed within (class). v4 adds a
  third bucketing: within (class, view) — exactly the granularity the
  diagnostic recommended. Falls back to per-class for ids where the
  view can't be parsed.

# v4 NEW: CROSS-VIEW aggregates per method (max / mean / std / lonely)
  Dataset §10: P(≥2 views show defect) = 0.83 across this dataset.
  But predictions §7 showed PIXEL-WISE cross-view Pearson is *lower*
  on positives than negatives (0.20 vs 0.44 for patchcore_exp7), which
  means the same pixel doesn't carry the same defect across views
  (different camera angles). So pixel-wise mean is misleading; the
  *max* across views is what captures "some view saw a defect here".
  We expose four sample-level aggregates per method:
    m{mi}_xv_max   : max  over sibling views (per pixel)
    m{mi}_xv_mean  : mean over sibling views
    m{mi}_xv_std   : std  over sibling views (= uncertainty)
    m{mi}_xv_lonely: this_view_score − max(sibling_views)
                       large positive = this view says hot AND
                       siblings disagree → potentially a false
                       positive OR a defect only this view can see
  Single-view samples get 0 for all four; XGB won't split.

# v4 CHANGED: small-CC default = 0 for every family
  Predictions §6 in the latest run shows CC filtering at p98 + any
  min_area NEVER beats area≥0 on any of the seven models we use. v3
  used per-family defaults derived from an older diagnostic. v4
  defaults to 0 for everything. Override with --small-cc as before.

# Compatibility
  All v3 features remain accessible and default ON: per-class rank-
  norm, cross-method consensus, CVD, per-image rank, etc. Set the
  new --no-* flags to bit-match v3 behaviour for direct A/B.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
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
        sys.stdout = old
        f.close()


def hr(t, c="="): print(f"\n{c * 78}\n  {t}\n{c * 78}")
def sub(t): print(f"\n--- {t} ---")


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
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


def q8rle_to_float_matrix(s: str) -> np.ndarray:
    parts = s.split()
    h, w = int(parts[1]), int(parts[2])
    if len(parts) <= 3:
        return np.zeros((h, w), dtype=np.float32)
    body = np.array(parts[3:], dtype=np.int64)
    vals = body[0::2].astype(np.uint8)
    lens = body[1::2]
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Loaders (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
def load_submission(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["ID", "Label"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for row in reader:
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def load_local_preds(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Re-run the baseline with the "
            f"local_preds_saver hook.")
    data = np.load(path, allow_pickle=True)
    out = {
        "ids":           data["ids"].astype(str),
        "classes":       data["classes"].astype(str),
        "anomaly_types": data["anomaly_types"].astype(str),
        "scores":        data["scores"].astype(np.float32),
        "masks":         data["masks"].astype(np.uint8),
    }
    if "image_paths" in data.files:
        out["image_paths"] = data["image_paths"].astype(str)
    else:
        out["image_paths"] = None
    return out


def build_class_map_from_data(data_root: Path) -> dict[str, str]:
    if not data_root.exists():
        return {}
    out: dict[str, str] = {}
    for cdir in sorted(data_root.iterdir()):
        if not cdir.is_dir() or not cdir.name.startswith("class_"):
            continue
        test_dir = cdir / "test"
        if not test_dir.exists():
            continue
        for p in test_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
                out[p.stem] = cdir.name
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sample-id / view parsing (unchanged from v3 — reused for v4 view-index)
# ─────────────────────────────────────────────────────────────────────────────
PATH_VIEW_RE = re.compile(r"^(?P<sid>.+?)_view(?P<v>\d+)(?:\.[A-Za-z]+)?$")


def parse_sample_id(name: str) -> tuple[str, int | None]:
    m = PATH_VIEW_RE.match(name)
    if m:
        return m.group("sid"), int(m.group("v"))
    return name, None


# ─────────────────────────────────────────────────────────────────────────────
# v4 NEW: load per-class spatial prior heatmaps from disk
# (these are produced by analyze_spacepresso_dataset.py §6 →
#  tables/06_heat_<class>.npy at side 128). We resize them to each
# image's (H, W) lazily inside featurize_image.
# ─────────────────────────────────────────────────────────────────────────────
def load_spatial_priors(prior_dir: Path,
                          classes: list[str]) -> dict[str, np.ndarray]:
    """Returns {class_name: heatmap_2d}. Missing classes get an
    all-zeros tile so featurize_image's shape contract is preserved."""
    if not prior_dir.exists():
        print(f"  [warn] --prior-heatmaps-dir {prior_dir} does not exist; "
              f"spatial-prior feature will be all-zeros.")
        return {cls: np.zeros((128, 128), dtype=np.float32) for cls in classes}
    out: dict[str, np.ndarray] = {}
    for cls in classes:
        p = prior_dir / f"06_heat_{cls}.npy"
        if p.exists():
            try:
                h = np.load(p).astype(np.float32)
                # Normalise to [0, 1] for stability (heatmap is an average
                # of binary masks → already in [0, 1] but be safe).
                h = np.clip(h, 0.0, 1.0)
                out[cls] = h
                print(f"    loaded spatial prior for {cls}: shape {h.shape}  "
                      f"max={h.max():.4f}")
            except Exception as e:
                print(f"    [warn] could not load {p}: {e}; using zeros")
                out[cls] = np.zeros((128, 128), dtype=np.float32)
        else:
            print(f"    [warn] no spatial prior file for {cls} "
                  f"(looked for {p.name}); using zeros")
            out[cls] = np.zeros((128, 128), dtype=np.float32)
    return out


def _resize_prior_to(prior: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinear-ish resize using scipy.ndimage.zoom. Cached at the
    featurize-call site via spatial_cache to avoid recomputing for
    every image of a given (class, H, W).
    """
    if prior.shape == (H, W):
        return prior.astype(np.float32, copy=False)
    zy = H / prior.shape[0]
    zx = W / prior.shape[1]
    out = ndi.zoom(prior, zoom=(zy, zx), order=1, mode="nearest")
    # zoom can drift shape by 1 px on some inputs; trim/pad if so.
    if out.shape != (H, W):
        h2 = min(out.shape[0], H); w2 = min(out.shape[1], W)
        clean = np.zeros((H, W), dtype=np.float32)
        clean[:h2, :w2] = out[:h2, :w2]
        out = clean
    return out.astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model-name detection (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("cutpaste", "cutpaste"),
    ("dnv2",     "patchcore_dnv2"),
    ("wrn50",    "patchcore_wrn50"),
    ("uniad",    "uniad"),
    ("cfa_",     "cfa"),
    ("fastflow", "fastflow"),
    ("effad",    "effad"),
]


def detect_model_family(run_dir_name: str) -> str:
    s = run_dir_name.lower()
    for pat, fam in MODEL_FAMILY_PATTERNS:
        if pat in s:
            return fam
    return "unknown"


# v4: latest predictions §6 showed CC filter hurts every model → 0
# everywhere. Override with --small-cc if you re-test and find gains.
DEFAULT_SMALL_CC_PER_FAMILY = {
    "patchcore_dnv2":  0,
    "patchcore_wrn50": 0,
    "cutpaste":        0,
    "uniad":           0,
    "cfa":             0,
    "fastflow":        0,
    "effad":           0,
    "unknown":         0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Small-CC suppression (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
def suppress_small_ccs_image(score: np.ndarray, min_cc: int,
                               t_pct: float = 99.0) -> np.ndarray:
    if min_cc <= 0:
        return score
    s = np.asarray(score, dtype=np.float32)
    t = float(np.percentile(s, t_pct))
    hot = s >= t
    if not hot.any():
        return s
    labels, n = ndi.label(hot)
    if n == 0:
        return s
    sizes = np.bincount(labels.ravel())
    if sizes.size <= 1:
        return s
    small_labels = np.where(sizes <= min_cc)[0]
    small_labels = small_labels[small_labels > 0]
    if small_labels.size == 0:
        return s
    drop = np.isin(labels, small_labels)
    out = s.copy()
    out[drop] = 0.0
    return out


def suppress_small_ccs_volume(scores: np.ndarray, min_cc: int,
                                 t_pct: float = 99.0) -> np.ndarray:
    if min_cc <= 0:
        return scores
    out = np.empty_like(scores)
    for i in range(scores.shape[0]):
        out[i] = suppress_small_ccs_image(scores[i], min_cc, t_pct)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rank-norm: global / per-class (v3) / per-class-view (v4 NEW)
# ─────────────────────────────────────────────────────────────────────────────
def _rank_replace(col: np.ndarray) -> np.ndarray:
    """Replace values with linearly-spaced ranks in [0, 1]."""
    n = col.size
    order = np.argsort(col, kind="stable")
    ranks = np.empty_like(col, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return ranks


def rank_normalise_global_inplace_val(scores: np.ndarray) -> None:
    M = scores.shape[-1]
    flat = scores.reshape(-1, M)
    for mi in range(M):
        flat[:, mi] = _rank_replace(flat[:, mi])


def rank_normalise_per_class_inplace_val(scores: np.ndarray,
                                            classes: np.ndarray) -> None:
    M = scores.shape[-1]
    for cls in sorted(set(classes.tolist())):
        idx = np.flatnonzero(classes == cls)
        if idx.size == 0:
            continue
        block = scores[idx]
        flat = block.reshape(-1, M)
        for mi in range(M):
            flat[:, mi] = _rank_replace(flat[:, mi])
        scores[idx] = flat.reshape(block.shape)


def rank_normalise_per_class_view_inplace_val(scores: np.ndarray,
                                                  classes: np.ndarray,
                                                  views: np.ndarray) -> None:
    """v4 NEW: rank within {class × view}. Falls back to per-class for
    val records that don't have a parseable view (view == -1)."""
    M = scores.shape[-1]
    # First, handle (class, view) buckets where view is known.
    have_view = views >= 0
    for cls in sorted(set(classes.tolist())):
        cls_mask = (classes == cls)
        # Per (class, view) for those with view
        for v in sorted({int(vv) for vv in views[cls_mask] if vv >= 0}):
            idx = np.flatnonzero(cls_mask & have_view & (views == v))
            if idx.size == 0:
                continue
            block = scores[idx]
            flat = block.reshape(-1, M)
            for mi in range(M):
                flat[:, mi] = _rank_replace(flat[:, mi])
            scores[idx] = flat.reshape(block.shape)
        # Fallback bucket for (class, unknown-view)
        idx_unk = np.flatnonzero(cls_mask & (~have_view))
        if idx_unk.size > 0:
            block = scores[idx_unk]
            flat = block.reshape(-1, M)
            for mi in range(M):
                flat[:, mi] = _rank_replace(flat[:, mi])
            scores[idx_unk] = flat.reshape(block.shape)


def rank_normalise_test_per_class_inplace(
    decoded_per_method: list[dict[str, np.ndarray]],
    all_ids: list[str],
    class_map: dict[str, str] | None,
    default_class: str,
) -> None:
    buckets: dict[str, list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        buckets[cls].append(sid)
    _bucket_rank(decoded_per_method, buckets, label="per-class")


def rank_normalise_test_per_class_view_inplace(
    decoded_per_method: list[dict[str, np.ndarray]],
    all_ids: list[str],
    class_map: dict[str, str] | None,
    default_class: str,
) -> None:
    """v4 NEW: rank within {class × view} at test time."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        _, v = parse_sample_id(sid)
        key = f"{cls}|view{v}" if v is not None else f"{cls}|view?"
        buckets[key].append(sid)
    _bucket_rank(decoded_per_method, buckets, label="per-class-view")


def _bucket_rank(decoded_per_method: list[dict[str, np.ndarray]],
                  buckets: dict[str, list[str]], label: str) -> None:
    """Generic bucketed rank-norm at test time."""
    for mi, d in enumerate(decoded_per_method):
        t0 = time.time()
        for bk, ids_in_b in buckets.items():
            shapes = {sid: d[sid].shape for sid in ids_in_b}
            sizes  = {sid: int(np.prod(shapes[sid])) for sid in ids_in_b}
            total  = sum(sizes.values())
            if total == 0:
                continue
            flat = np.empty(total, dtype=np.float32)
            idx = 0
            for sid in ids_in_b:
                n = sizes[sid]
                flat[idx:idx + n] = d[sid].ravel()
                idx += n
            ranks = _rank_replace(flat)
            del flat
            idx = 0
            for sid in ids_in_b:
                n = sizes[sid]
                d[sid] = ranks[idx:idx + n].reshape(shapes[sid]).astype(np.float32)
                idx += n
            del ranks
        print(f"    method {mi + 1}/{len(decoded_per_method)}: "
              f"{label} rank-norm done ({time.time() - t0:.1f}s)")


def rank_normalise_test_global_inplace(
    decoded_per_method: list[dict[str, np.ndarray]],
    all_ids: list[str]) -> None:
    _bucket_rank(decoded_per_method, {"_GLOBAL_": list(all_ids)},
                  label="global")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-view siblings (v3) + v4 NEW aggregates (max/mean/std/lonely)
# ─────────────────────────────────────────────────────────────────────────────
def group_val_by_sample(val: dict) -> dict[tuple[str, str], list[int]]:
    image_paths = val.get("image_paths")
    out: dict[tuple[str, str], list[int]] = defaultdict(list)
    if image_paths is not None:
        for i, (cls, p) in enumerate(zip(val["classes"], image_paths)):
            sid, _v = parse_sample_id(Path(str(p)).name)
            out[(str(cls), sid)].append(i)
    else:
        for i, (cls, id_) in enumerate(zip(val["classes"], val["ids"])):
            sid = str(id_).rsplit("/", 1)[-1]
            out[(str(cls), sid)].append(i)
    return out


def group_test_ids_by_sample(all_ids: list[str],
                                class_map: dict[str, str] | None,
                                default_class: str
                                ) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        sample_id, _v = parse_sample_id(sid)
        out[(str(cls), sample_id)].append(sid)
    return out


def compute_cross_view_disagreement_val(val_scores: np.ndarray,
                                           classes: np.ndarray,
                                           image_paths: np.ndarray | None
                                           ) -> tuple[np.ndarray, np.ndarray]:
    """v3 CVD (this_view - sibling_mean)."""
    N, H, W, M = val_scores.shape
    val_dict = {"scores": val_scores, "classes": classes,
                "image_paths": image_paths,
                "ids": np.array([f"placeholder/{i}" for i in range(N)])}
    groups = group_val_by_sample(val_dict)
    cvd = np.zeros_like(val_scores)
    is_multi = np.zeros(N, dtype=np.uint8)
    n_multi_samples = 0
    for (_cls, _sid), idxs in groups.items():
        if len(idxs) < 2:
            continue
        n_multi_samples += 1
        for vi in idxs:
            is_multi[vi] = 1
        for mi in range(M):
            stack = val_scores[idxs, :, :, mi]
            total = stack.sum(axis=0)
            v = len(idxs)
            for k, vi in enumerate(idxs):
                sibling_mean = (total - stack[k]) / max(v - 1, 1)
                cvd[vi, :, :, mi] = stack[k] - sibling_mean
    print(f"  cross-view: {n_multi_samples} multi-view samples grouped "
          f"({int(is_multi.sum())}/{N} val images get a non-zero CVD map)")
    return cvd, is_multi


def compute_xv_aggregates_val(val_scores: np.ndarray,
                                 classes: np.ndarray,
                                 image_paths: np.ndarray | None
                                 ) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray,
                                              np.ndarray]:
    """v4 NEW: per-pixel cross-view aggregates for the val set.

    Returns five (N, H, W, M) arrays:
      xv_max    : max  over SIBLING views (other views of same sample)
      xv_mean   : mean over sibling views
      xv_std    : std  over sibling views
      xv_lonely : this_view - max(sibling views)
                  large positive = this view alone is hot
      is_multi  : (N,) uint8 flag (1 = sample has ≥2 views in val set)

    Single-view samples get 0 for all four aggregates; the trees
    won't split on a constant.
    """
    N, H, W, M = val_scores.shape
    val_dict = {"scores": val_scores, "classes": classes,
                "image_paths": image_paths,
                "ids": np.array([f"placeholder/{i}" for i in range(N)])}
    groups = group_val_by_sample(val_dict)

    xv_max    = np.zeros_like(val_scores)
    xv_mean   = np.zeros_like(val_scores)
    xv_std    = np.zeros_like(val_scores)
    xv_lonely = np.zeros_like(val_scores)
    is_multi  = np.zeros(N, dtype=np.uint8)
    n_multi_samples = 0

    for (_cls, _sid), idxs in groups.items():
        if len(idxs) < 2:
            continue
        n_multi_samples += 1
        for vi in idxs:
            is_multi[vi] = 1
        for mi in range(M):
            stack = val_scores[idxs, :, :, mi]   # (V, H, W)
            V = stack.shape[0]
            for k, vi in enumerate(idxs):
                # Siblings = all-but-k
                if V == 1:
                    continue
                siblings = np.delete(stack, k, axis=0)   # (V-1, H, W)
                sib_max  = siblings.max(axis=0)
                sib_mean = siblings.mean(axis=0)
                sib_std  = siblings.std(axis=0) if V > 2 else \
                              np.zeros_like(sib_mean)
                xv_max[vi, :, :, mi]    = sib_max
                xv_mean[vi, :, :, mi]   = sib_mean
                xv_std[vi, :, :, mi]    = sib_std
                xv_lonely[vi, :, :, mi] = stack[k] - sib_max
    print(f"  xv-aggregates: {n_multi_samples} multi-view samples "
          f"({int(is_multi.sum())}/{N} val images get non-zero aggregates)")
    return xv_max, xv_mean, xv_std, xv_lonely, is_multi


# ─────────────────────────────────────────────────────────────────────────────
# Local-val alignment (extended to track views from image_paths)
# ─────────────────────────────────────────────────────────────────────────────
def _nn_resize_2d(arr: np.ndarray, target_shape, dtype) -> np.ndarray:
    th, tw = target_shape
    h, w = arr.shape
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)


def align_local_preds(preds_per_method: list[dict],
                       method_names: list[str]) -> dict:
    id_sets = [set(p["ids"].tolist()) for p in preds_per_method]
    common = sorted(set.intersection(*id_sets))
    if not common:
        raise RuntimeError("no local-val IDs in common across the methods")
    H0, W0 = preds_per_method[0]["scores"].shape[1:3]
    print(f"  reference shape (method 0): {H0}x{W0}")
    indexed = []
    for p in preds_per_method:
        idx_of = {id_: i for i, id_ in enumerate(p["ids"])}
        indexed.append((p, idx_of))
    N = len(common); M = len(preds_per_method)
    scores = np.empty((N, H0, W0, M), dtype=np.float32)
    masks = np.empty((N, H0, W0), dtype=np.uint8)
    classes = np.empty(N, dtype=object)
    anomaly_types = np.empty(N, dtype=object)
    views = np.full(N, -1, dtype=np.int32)            # v4: track views
    image_paths_m0 = preds_per_method[0].get("image_paths")
    image_paths_aligned: np.ndarray | None = None
    if image_paths_m0 is not None:
        image_paths_aligned = np.empty(N, dtype=object)
    for i, id_ in enumerate(common):
        for mi, (p, idx_of) in enumerate(indexed):
            j = idx_of[id_]
            s = p["scores"][j]
            if s.shape != (H0, W0):
                s = _nn_resize_2d(s, (H0, W0), np.float32)
            scores[i, :, :, mi] = s
            if mi == 0:
                classes[i] = str(p["classes"][j])
                anomaly_types[i] = str(p["anomaly_types"][j])
                m = p["masks"][j]
                if m.shape != (H0, W0):
                    m = _nn_resize_2d(m, (H0, W0), np.uint8)
                masks[i] = m
                if image_paths_aligned is not None and image_paths_m0 is not None:
                    pth = str(image_paths_m0[j])
                    image_paths_aligned[i] = pth
                    _, v = parse_sample_id(Path(pth).name)
                    if v is not None:
                        views[i] = int(v)
    n_with_view = int((views >= 0).sum())
    print(f"  aligned {N} val images × {M} methods @ {H0}x{W0}"
          + (f" (with image_paths, {n_with_view}/{N} views parsed)"
             if image_paths_aligned is not None else " (no image_paths)"))
    return {"ids": np.asarray(common),
            "classes": classes.astype(str),
            "anomaly_types": anomaly_types.astype(str),
            "views": views,
            "scores": scores,
            "masks": masks,
            "image_paths": image_paths_aligned}


# ─────────────────────────────────────────────────────────────────────────────
# Feature configuration (v3 + v4 additions)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    use_raw_score: bool = True
    use_per_image_rank: bool = True
    gauss_sigmas: tuple[float, ...] = (1.0, 3.0, 7.0)
    window_sizes: tuple[int, ...] = (3, 7, 15)
    use_window_mean: bool = True
    use_window_max: bool = True
    use_window_std: bool = True
    use_gradient: bool = True
    use_laplacian: bool = True
    use_dist_to_hot: bool = True
    hot_percentile: float = 99.0
    use_image_aggregates: bool = True
    use_cross_stats: bool = True
    use_spatial: bool = True
    # v3
    use_cross_method_consensus: bool = True
    use_cross_view_disagreement: bool = False
    # v4 NEW
    use_spatial_prior: bool = True
    use_class_onehot: bool = True
    use_view_onehot: bool = True
    use_xv_aggregates: bool = True       # max/mean/std/lonely per method


# ─────────────────────────────────────────────────────────────────────────────
# Featurization primitives (v3 — unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _per_image_rank(s: np.ndarray) -> np.ndarray:
    flat = s.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty_like(flat, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks.reshape(s.shape)


def _window_mean(s, size):
    return ndi.uniform_filter(s, size=size, mode="reflect").astype(np.float32)


def _window_max(s, size):
    return ndi.maximum_filter(s, size=size, mode="reflect").astype(np.float32)


def _window_std(s, size):
    mean = ndi.uniform_filter(s, size=size, mode="reflect")
    sq = ndi.uniform_filter(s * s, size=size, mode="reflect")
    var = np.clip(sq - mean * mean, 0.0, None)
    return np.sqrt(var).astype(np.float32)


def _gauss(s, sigma):
    return ndi.gaussian_filter(s, sigma=sigma, mode="reflect").astype(np.float32)


def _grad_mag(s):
    sx = ndi.sobel(s, axis=0, mode="reflect")
    sy = ndi.sobel(s, axis=1, mode="reflect")
    return np.sqrt(sx * sx + sy * sy).astype(np.float32)


def _laplacian(s):
    return ndi.laplace(s, mode="reflect").astype(np.float32)


def _dist_to_hot(s, pct):
    thresh = float(np.percentile(s, pct))
    hot = s >= thresh
    if not hot.any():
        H, W = s.shape
        return np.full(s.shape, float(np.hypot(H, W)), dtype=np.float32)
    return ndi.distance_transform_edt(~hot).astype(np.float32)


def _spatial_cache(H: int, W: int) -> dict:
    ys, xs = np.indices((H, W)).astype(np.float32)
    xs_n = xs / max(W - 1, 1)
    ys_n = ys / max(H - 1, 1)
    de = np.minimum(np.minimum(xs_n, ys_n),
                     np.minimum(1.0 - xs_n, 1.0 - ys_n)).astype(np.float32)
    dc = np.sqrt((xs_n - 0.5) ** 2 + (ys_n - 0.5) ** 2).astype(np.float32)
    return {"x": xs_n, "y": ys_n, "dist_edge": de, "dist_center": dc,
            "_shape": (H, W),
            # v4: prior cache (filled lazily per-class inside featurize)
            "_prior_resized": {}}


# v4 NEW: one-hot utility
def _onehot_const_planes(value: int, n: int, H: int, W: int
                            ) -> list[np.ndarray]:
    """Return n constant (H, W) planes; the `value`-th one is filled
    with 1, the rest with 0. value out of range → all zeros (unknown)."""
    out = [np.zeros((H, W), dtype=np.float32) for _ in range(n)]
    if 0 <= value < n:
        out[value][:] = 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# featurize_image — v4: accepts class/view/prior/xv_aggregates
# ─────────────────────────────────────────────────────────────────────────────
def featurize_image(
    scores_per_method: list[np.ndarray],
    cfg: FeatureConfig,
    spatial_cache: dict | None = None,
    feature_names: list[str] | None = None,
    cross_view_disagreement_per_method: list[np.ndarray] | None = None,
    is_multiview_sample: bool = True,
    # v4 NEW
    class_id: str | None = None,
    all_classes: list[str] | None = None,
    view: int | None = None,
    n_views_onehot: int = 5,
    spatial_prior: np.ndarray | None = None,
    xv_max_per_method: list[np.ndarray] | None = None,
    xv_mean_per_method: list[np.ndarray] | None = None,
    xv_std_per_method: list[np.ndarray] | None = None,
    xv_lonely_per_method: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the per-pixel feature stack for ONE image.

    v4 additions to v3:
      - spatial_prior: (H, W) heatmap value loaded from dataset §6 npy
      - class_id + all_classes: one-hot class encoding
      - view (1..n_views): one-hot view + 1 'unknown' bucket
      - xv_*_per_method: per-method sibling-view aggregates
    Everything new is optional and gated by cfg flags.
    """
    assert scores_per_method
    H, W = scores_per_method[0].shape
    M = len(scores_per_method)
    layers: list[np.ndarray] = []
    names: list[str] = []

    # ── v3 per-method features (raw / rank / smoothed / windowed / etc.)
    for mi, s in enumerate(scores_per_method):
        if s.shape != (H, W):
            raise ValueError(f"method {mi} shape {s.shape} != ({H}, {W})")
        s = np.asarray(s, dtype=np.float32)
        if cfg.use_raw_score:
            layers.append(s); names.append(f"m{mi}_raw")
        if cfg.use_per_image_rank:
            layers.append(_per_image_rank(s)); names.append(f"m{mi}_rank")
        for sigma in cfg.gauss_sigmas:
            layers.append(_gauss(s, sigma)); names.append(f"m{mi}_g{sigma:g}")
        for ws in cfg.window_sizes:
            if cfg.use_window_mean:
                layers.append(_window_mean(s, ws)); names.append(f"m{mi}_mean{ws}")
            if cfg.use_window_max:
                layers.append(_window_max(s, ws));  names.append(f"m{mi}_max{ws}")
            if cfg.use_window_std:
                layers.append(_window_std(s, ws));  names.append(f"m{mi}_std{ws}")
        if cfg.use_gradient:
            layers.append(_grad_mag(s));  names.append(f"m{mi}_grad")
        if cfg.use_laplacian:
            layers.append(_laplacian(s)); names.append(f"m{mi}_lap")
        if cfg.use_dist_to_hot:
            layers.append(_dist_to_hot(s, cfg.hot_percentile))
            names.append(f"m{mi}_d2hot")
        if cfg.use_image_aggregates:
            for stat_name, val in (("imgmax", float(s.max())),
                                     ("imgp99", float(np.percentile(s, 99))),
                                     ("imgmean", float(s.mean())),
                                     ("imgstd", float(s.std()))):
                layers.append(np.full((H, W), val, dtype=np.float32))
                names.append(f"m{mi}_{stat_name}")

    # ── v2 cross-method statistics
    if cfg.use_cross_stats and M >= 2:
        stack = np.stack(scores_per_method, axis=0).astype(np.float32)
        cmean = stack.mean(axis=0); layers.append(cmean); names.append("x_mean")
        cmax  = stack.max(axis=0);  layers.append(cmax);  names.append("x_max")
        cmin  = stack.min(axis=0);  layers.append(cmin);  names.append("x_min")
        cstd  = stack.std(axis=0);  layers.append(cstd);  names.append("x_std")
        layers.append((cmax - cmin).astype(np.float32))
        names.append("x_range")

    # ── v3 cross-method CONSENSUS features
    if cfg.use_cross_method_consensus and M >= 2:
        stack = np.stack(scores_per_method, axis=0).astype(np.float32)
        top1pct = (stack >= 0.99).sum(axis=0).astype(np.float32)
        top5pct = (stack >= 0.95).sum(axis=0).astype(np.float32)
        mean_rank = stack.mean(axis=0).astype(np.float32)
        layers.append(top1pct); names.append("xc_top1pct_count")
        layers.append(top5pct); names.append("xc_top5pct_count")
        layers.append(mean_rank); names.append("xc_mean_rank")

    # ── v3 cross-view DISAGREEMENT features (one per method)
    if (cfg.use_cross_view_disagreement
        and cross_view_disagreement_per_method is not None):
        for mi, cvd in enumerate(cross_view_disagreement_per_method):
            if not is_multiview_sample:
                layers.append(np.zeros((H, W), dtype=np.float32))
            else:
                cvd = np.asarray(cvd, dtype=np.float32)
                if cvd.shape != (H, W):
                    raise ValueError(
                        f"CVD method {mi} shape {cvd.shape} != ({H}, {W})")
                layers.append(cvd)
            names.append(f"m{mi}_cvd")

    # ── v4 NEW: per-method cross-view aggregates (max/mean/std/lonely)
    if cfg.use_xv_aggregates and xv_max_per_method is not None:
        for mi in range(M):
            xm  = xv_max_per_method[mi]    if xv_max_per_method    else None
            xmn = xv_mean_per_method[mi]   if xv_mean_per_method   else None
            xs_ = xv_std_per_method[mi]    if xv_std_per_method    else None
            xl  = xv_lonely_per_method[mi] if xv_lonely_per_method else None
            for arr, suffix in [(xm,  "xv_max"),
                                  (xmn, "xv_mean"),
                                  (xs_, "xv_std"),
                                  (xl,  "xv_lonely")]:
                if arr is None or not is_multiview_sample:
                    layers.append(np.zeros((H, W), dtype=np.float32))
                else:
                    a = np.asarray(arr, dtype=np.float32)
                    if a.shape != (H, W):
                        raise ValueError(
                            f"xv {suffix} method {mi}: shape {a.shape} != "
                            f"({H}, {W})")
                    layers.append(a)
                names.append(f"m{mi}_{suffix}")

    # ── v3 spatial features
    if cfg.use_spatial:
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        for k in ("x", "y", "dist_edge", "dist_center"):
            layers.append(spatial_cache[k]); names.append(f"s_{k}")

    # ── v4 NEW: per-class spatial prior heatmap (resized to H, W)
    if cfg.use_spatial_prior and spatial_prior is not None:
        # cache resized prior by (class_id, H, W) inside spatial_cache
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        key = (class_id, H, W)
        cache = spatial_cache.setdefault("_prior_resized", {})
        if key not in cache:
            cache[key] = _resize_prior_to(spatial_prior, H, W)
        layers.append(cache[key])
        names.append("s_prior")

    # ── v4 NEW: class one-hot
    if cfg.use_class_onehot and class_id is not None and all_classes:
        try:
            idx = all_classes.index(class_id)
        except ValueError:
            idx = -1
        planes = _onehot_const_planes(idx, len(all_classes), H, W)
        for cls_name, plane in zip(all_classes, planes):
            layers.append(plane); names.append(f"c_{cls_name}")

    # ── v4 NEW: view one-hot (1..n_views_onehot + 'unknown')
    if cfg.use_view_onehot:
        # Encoded values: -1 → 'unknown' slot (last); 1..n → slots 0..n-1.
        if view is not None and 1 <= int(view) <= n_views_onehot:
            v_idx = int(view) - 1
        else:
            v_idx = n_views_onehot  # 'unknown' bucket
        n_slots = n_views_onehot + 1
        planes = _onehot_const_planes(v_idx, n_slots, H, W)
        for k, plane in enumerate(planes):
            tag = (f"view_{k + 1}" if k < n_views_onehot
                   else "view_unknown")
            layers.append(plane); names.append(f"v_{tag}")

    feats = np.stack(layers, axis=-1).astype(np.float32, copy=False)
    if feature_names is not None and names != feature_names:
        missing = [n for n in feature_names if n not in names]
        extra   = [n for n in names if n not in feature_names]
        raise RuntimeError(
            f"feature drift between fit and predict: "
            f"got {len(names)} cols, expected {len(feature_names)}; "
            f"missing={missing[:5]}; extra={extra[:5]}")
    return feats, names


# ─────────────────────────────────────────────────────────────────────────────
# Build per-class training matrices (v4: also plumbs aggregates + prior)
# ─────────────────────────────────────────────────────────────────────────────
def build_training_data(val: dict, classes: list[str], cfg: FeatureConfig,
                         *, neg_per_pos: int, seed: int,
                         cvd: np.ndarray | None = None,
                         is_multi: np.ndarray | None = None,
                         xv_max: np.ndarray | None = None,
                         xv_mean: np.ndarray | None = None,
                         xv_std: np.ndarray | None = None,
                         xv_lonely: np.ndarray | None = None,
                         spatial_priors: dict[str, np.ndarray] | None = None,
                         all_classes_onehot: list[str] | None = None,
                         ) -> dict:
    rng = np.random.default_rng(seed)
    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]; val_anom_types = val["anomaly_types"]
    val_views = val.get("views", np.full(val_scores.shape[0], -1, dtype=np.int32))
    N, H, W, M = val_scores.shape
    spatial = _spatial_cache(H, W)
    out: dict = {}
    feature_names_global: list[str] | None = None

    for cls in classes:
        cls_idx = np.flatnonzero(val_classes == cls)
        if cls_idx.size == 0:
            print(f"  [warn] class {cls}: no val images; skipping")
            continue
        per_img_X, per_img_y, per_img_atypes = [], [], []
        per_img_pixrange: list[tuple[int, int]] = []
        cursor = 0
        prior_cls = (spatial_priors.get(cls) if spatial_priors else None)
        for i in cls_idx:
            mat_list = [val_scores[i, :, :, mi] for mi in range(M)]
            cvd_list = (None if cvd is None
                          else [cvd[i, :, :, mi] for mi in range(M)])
            is_mv = bool(is_multi[i]) if is_multi is not None else True
            xv_max_list    = (None if xv_max is None
                                else [xv_max[i, :, :, mi]    for mi in range(M)])
            xv_mean_list   = (None if xv_mean is None
                                else [xv_mean[i, :, :, mi]   for mi in range(M)])
            xv_std_list    = (None if xv_std is None
                                else [xv_std[i, :, :, mi]    for mi in range(M)])
            xv_lonely_list = (None if xv_lonely is None
                                else [xv_lonely[i, :, :, mi] for mi in range(M)])
            v_int = int(val_views[i]) if val_views[i] >= 0 else None
            feats, names = featurize_image(
                mat_list, cfg, spatial_cache=spatial,
                cross_view_disagreement_per_method=cvd_list,
                is_multiview_sample=is_mv,
                class_id=cls,
                all_classes=all_classes_onehot,
                view=v_int,
                spatial_prior=prior_cls,
                xv_max_per_method=xv_max_list,
                xv_mean_per_method=xv_mean_list,
                xv_std_per_method=xv_std_list,
                xv_lonely_per_method=xv_lonely_list,
            )
            if feature_names_global is None:
                feature_names_global = names
            F = feats.shape[-1]
            flat_x = feats.reshape(-1, F)
            flat_y = val_masks[i].ravel().astype(np.int32)
            per_img_X.append(flat_x); per_img_y.append(flat_y)
            per_img_atypes.append(str(val_anom_types[i]))
            per_img_pixrange.append((cursor, cursor + flat_x.shape[0]))
            cursor += flat_x.shape[0]
            del feats
        X_full = np.concatenate(per_img_X, axis=0).astype(np.float32)
        y_full = np.concatenate(per_img_y, axis=0).astype(np.int32)
        del per_img_X, per_img_y
        n_pos = int((y_full == 1).sum())
        n_neg = int((y_full == 0).sum())
        if n_pos == 0:
            print(f"  [warn] class {cls}: zero positive pixels in val")
            out[cls] = {"_fallback_to_shared": True,
                        "n_pos": 0, "n_neg_sampled": 0,
                        "X_full": X_full, "y_full": y_full,
                        "anomaly_types": per_img_atypes,
                        "img_pixranges": per_img_pixrange}
            continue
        target_neg = min(n_neg, n_pos * neg_per_pos)
        neg_idx = np.flatnonzero(y_full == 0)
        sample_neg = (rng.choice(neg_idx, size=target_neg, replace=False)
                       if target_neg < n_neg else neg_idx)
        pos_idx = np.flatnonzero(y_full == 1)
        keep = np.concatenate([pos_idx, sample_neg])
        out[cls] = {
            "X_train": X_full[keep], "y_train": y_full[keep],
            "X_full":  X_full,       "y_full":  y_full,
            "anomaly_types": per_img_atypes,
            "img_pixranges": per_img_pixrange,
            "n_pos": n_pos, "n_neg_sampled": int(len(sample_neg)),
        }
        print(f"  class {cls}: train {len(keep):>8d} rows "
              f"(pos={n_pos:>6d}, neg={len(sample_neg):>8d}, "
              f"feats={X_full.shape[1]:>3d})")
    out["_feature_names"] = feature_names_global or []
    return out


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost params (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators":     300, #300
    "max_depth":        4, #6
    "learning_rate":    0.03, #0.07
    "subsample":        0.6, #0.8
    "colsample_bytree": 0.6, #0.8
    "reg_alpha":        0.5, #0.0
    "reg_lambda":       1.0, #1.0
    "min_child_weight": 20.0, #1.0
    "tree_method":      "hist",
    "max_bin":          256,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss", #logloss
    "n_jobs":           -1,
    "verbosity":        0,
}


def _make_xgb(params: dict, seed: int) -> "xgb.XGBClassifier":
    if not HAS_XGB:
        raise SystemExit("xgboost not installed. uv pip install xgboost")
    p = dict(DEFAULT_XGB_PARAMS)
    p.update(params or {})
    return xgb.XGBClassifier(random_state=seed, **p)


def get_params_for_class(cls: str, params_global: dict,
                          params_per_class: dict | None) -> dict:
    if params_per_class and cls in params_per_class:
        return params_per_class[cls]
    return params_global


def _bucketize_one_class(td: dict, mode: str) -> dict[object, list[int]]:
    atypes = td["anomaly_types"]
    n_imgs = len(td["img_pixranges"])
    if mode == "loao":
        key = lambda i: atypes[i]
    elif mode == "loio":
        key = lambda i: i
    else:
        raise ValueError(f"unknown cv_mode: {mode}")
    buckets: dict = defaultdict(list)
    for i in range(n_imgs):
        buckets[key(i)].append(i)
    return buckets


# ─────────────────────────────────────────────────────────────────────────────
# Pooled CV (v3, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _pixel_ap_pooled(preds: np.ndarray, labels: np.ndarray) -> float:
    if int(labels.sum()) == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, preds))
    except Exception:
        order = np.argsort(-preds, kind="stable")
        y = labels[order]
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / max(int(y.sum()), 1)
        rec = np.concatenate([[0.0], rec])
        prec = np.concatenate([[1.0], prec])
        return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def pooled_cv_score_one_class(td: dict, params: dict, seed: int,
                                mode: str = "loao",
                                neg_per_pos: int = 30) -> float:
    rng = np.random.default_rng(seed)
    X_full = td["X_full"]; y_full = td["y_full"]
    ranges = td["img_pixranges"]
    buckets = _bucketize_one_class(td, mode)
    fold_aps: list[float] = []
    for held_imgs in buckets.values():
        held_set = set(held_imgs)
        train_imgs = [i for i in range(len(ranges)) if i not in held_set]
        Xs, ys = [], []
        for ti in train_imgs:
            s, e = ranges[ti]
            yp = y_full[s:e]
            pos = np.flatnonzero(yp == 1)
            if pos.size == 0:
                continue
            neg = np.flatnonzero(yp == 0)
            n_keep = min(neg.size, pos.size * neg_per_pos)
            sneg = (rng.choice(neg, n_keep, replace=False)
                    if n_keep < neg.size else neg)
            keep = np.concatenate([pos, sneg])
            Xs.append(X_full[s:e][keep]); ys.append(yp[keep])
        if not Xs:
            continue
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0),
                np.concatenate(ys, axis=0))
        held_preds = []; held_labels = []
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            held_preds.append(p)
            held_labels.append(y_full[s:e].astype(np.int32))
        held_preds_cat = np.concatenate(held_preds)
        held_labels_cat = np.concatenate(held_labels)
        if int(held_labels_cat.sum()) > 0:
            fold_aps.append(_pixel_ap_pooled(held_preds_cat, held_labels_cat))
    valid = [a for a in fold_aps if not math.isnan(a)]
    return float(np.mean(valid)) if valid else 0.0


def loao_oof_one_class(td: dict, params: dict, seed: int,
                        cv_mode: str = "loao",
                        neg_per_pos: int = 30
                        ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_full = td["X_full"]; y_full = td["y_full"]
    ranges = td["img_pixranges"]
    buckets = _bucketize_one_class(td, cv_mode)
    oof_preds = np.full(y_full.shape, np.nan, dtype=np.float32)
    for held_imgs in buckets.values():
        held_set = set(held_imgs)
        train_imgs = [i for i in range(len(ranges)) if i not in held_set]
        Xs, ys = [], []
        for ti in train_imgs:
            s, e = ranges[ti]
            yp = y_full[s:e]
            pos = np.flatnonzero(yp == 1)
            if pos.size == 0:
                continue
            neg = np.flatnonzero(yp == 0)
            n_keep = min(neg.size, pos.size * neg_per_pos)
            sneg = (rng.choice(neg, n_keep, replace=False)
                    if n_keep < neg.size else neg)
            keep = np.concatenate([pos, sneg])
            Xs.append(X_full[s:e][keep]); ys.append(yp[keep])
        if not Xs:
            continue
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0),
                np.concatenate(ys, axis=0))
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            oof_preds[s:e] = p
    mask = ~np.isnan(oof_preds)
    return oof_preds[mask].astype(np.float32), y_full[mask].astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning (v3, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _optuna_suggest(trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
    }


def tune_global(training_data: dict, *, n_trials: int, seed: int,
                cv_mode: str = "loao",
                timeout: float | None = None,
                neg_per_pos: int = 30) -> dict:
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed. uv pip install optuna")
    classes = [c for c in training_data if not c.startswith("_")
                and not training_data[c].get("_fallback_to_shared")]

    def objective(trial):
        params = _optuna_suggest(trial)
        aps = [pooled_cv_score_one_class(training_data[cls], params, seed,
                                            mode=cv_mode,
                                            neg_per_pos=neg_per_pos)
                for cls in classes]
        return float(np.mean(aps)) if aps else 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize",
                                  sampler=sampler, pruner=pruner)
    print(f"\n>>> Global Optuna tuning: {n_trials} trials, CV={cv_mode}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f">>> best mean pooled-AP: {study.best_value:.4f}")
    print(f">>> best params: {json.dumps(study.best_params, indent=2)}")
    return study.best_params


def tune_per_class(training_data: dict, *, n_trials: int, seed: int,
                    cv_mode: str = "loao",
                    timeout_per_class: float | None = None,
                    neg_per_pos: int = 30) -> dict:
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed. uv pip install optuna")
    out: dict = {}
    for cls, td in training_data.items():
        if cls.startswith("_"):
            continue
        if td.get("_fallback_to_shared"):
            print(f"\n  class {cls}: fallback-to-shared, no per-class tuning")
            continue
        print(f"\n>>> Per-class tuning: {cls}  (CV={cv_mode}, "
              f"n_trials={n_trials}, timeout_per_class={timeout_per_class}s)")

        def objective(trial, _td=td):
            return pooled_cv_score_one_class(_td, _optuna_suggest(trial),
                                                seed, mode=cv_mode,
                                                neg_per_pos=neg_per_pos)

        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize",
                                      sampler=sampler, pruner=pruner,
                                      study_name=f"xgb_{cls}")
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials,
                        timeout=timeout_per_class)
        elapsed = time.time() - t0
        print(f"  {cls}: best CV pooled-AP = {study.best_value:.4f}  "
              f"({elapsed:.1f}s, {len(study.trials)} trials)")
        print(f"  {cls}: best params = {study.best_params}")
        oof_preds, oof_labels = loao_oof_one_class(
            td, study.best_params, seed, cv_mode=cv_mode,
            neg_per_pos=neg_per_pos)
        out[cls] = {
            "best_params": study.best_params,
            "best_cv_ap": float(study.best_value),
            "n_trials": len(study.trials),
            "oof_preds": oof_preds,
            "oof_labels": oof_labels,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Calibration (v3, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def fit_calibrator(method: str, oof_preds: np.ndarray,
                    oof_labels: np.ndarray) -> dict:
    if method == "none":
        return {"method": "none"}
    if oof_preds.size == 0 or int(oof_labels.sum()) == 0:
        print(f"    [warn] no positives in OOF; falling back to no-op")
        return {"method": "none"}
    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(solver="lbfgs", max_iter=500)
        clf.fit(oof_preds.reshape(-1, 1), oof_labels)
        return {"method": "platt",
                "coef": float(clf.coef_[0][0]),
                "intercept": float(clf.intercept_[0])}
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        n_max = 1_000_000
        if oof_preds.size > n_max:
            pos_idx = np.flatnonzero(oof_labels == 1)
            neg_idx = np.flatnonzero(oof_labels == 0)
            target_neg = max(n_max - pos_idx.size, 0)
            rng = np.random.default_rng(0)
            sneg = (rng.choice(neg_idx, size=target_neg, replace=False)
                    if target_neg < neg_idx.size else neg_idx)
            sel = np.concatenate([pos_idx, sneg])
            oof_preds = oof_preds[sel]
            oof_labels = oof_labels[sel]
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(oof_preds, oof_labels.astype(np.float32))
        return {"method": "isotonic",
                "X_thresholds": [float(v) for v in ir.X_thresholds_],
                "y_thresholds": [float(v) for v in ir.y_thresholds_]}
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(cal: dict, scores: np.ndarray) -> np.ndarray:
    m = cal.get("method", "none")
    if m == "none":
        return scores.astype(np.float32)
    if m == "platt":
        z = cal["coef"] * scores + cal["intercept"]
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    if m == "isotonic":
        xs = np.asarray(cal["X_thresholds"], dtype=np.float32)
        ys = np.asarray(cal["y_thresholds"], dtype=np.float32)
        return np.interp(scores, xs, ys).astype(np.float32)
    raise ValueError(m)


# ─────────────────────────────────────────────────────────────────────────────
# Fit per-class production models (v3, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def fit_per_class(training_data: dict, params_global: dict,
                   params_per_class: dict | None, seed: int,
                   min_pos_for_per_class: int = 200) -> dict:
    from sklearn.metrics import log_loss, average_precision_score
    feature_names = training_data.get("_feature_names", [])
    out: dict = {"_feature_names": feature_names}
    shared_X, shared_y = [], []
    for cls, td in training_data.items():
        if cls.startswith("_"):
            continue
        if td.get("_fallback_to_shared"):
            out[cls] = {"_fallback_to_shared": True,
                        "n_pos": td.get("n_pos", 0),
                        "n_neg_sampled": td.get("n_neg_sampled", 0)}
            continue
        X = td["X_train"]; y = td["y_train"]
        shared_X.append(X); shared_y.append(y)
        if td["n_pos"] < min_pos_for_per_class:
            print(f"  class {cls}: only {td['n_pos']} positives "
                  f"(< {min_pos_for_per_class}); using SHARED model")
            out[cls] = {"_fallback_to_shared": True,
                        "n_pos": td["n_pos"],
                        "n_neg_sampled": td["n_neg_sampled"]}
            continue
        params = get_params_for_class(cls, params_global, params_per_class)
        clf = _make_xgb(params, seed)
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        try: ll = float(log_loss(y, p, labels=[0, 1]))
        except Exception: ll = float("nan")
        try: ap = float(average_precision_score(y, p))
        except Exception: ap = float("nan")
        out[cls] = {"model": clf,
                    "n_pos": td["n_pos"],
                    "n_neg_sampled": td["n_neg_sampled"],
                    "logloss": ll, "train_ap": ap,
                    "params": params,
                    "feature_importance": clf.feature_importances_.tolist()}
        print(f"  class {cls}: n_pos={td['n_pos']:>6d}  "
              f"n_neg={td['n_neg_sampled']:>8d}  "
              f"train_logloss={ll:.4f}  train_ap={ap:.3f}")
    if shared_X:
        X_all = np.concatenate(shared_X, axis=0)
        y_all = np.concatenate(shared_y, axis=0)
        clf_sh = _make_xgb(params_global, seed)
        clf_sh.fit(X_all, y_all)
        p_sh = clf_sh.predict_proba(X_all)[:, 1]
        try: ll_sh = float(log_loss(y_all, p_sh, labels=[0, 1]))
        except Exception: ll_sh = float("nan")
        try: ap_sh = float(average_precision_score(y_all, p_sh))
        except Exception: ap_sh = float("nan")
        out["_SHARED_"] = {"model": clf_sh,
                            "n_pos": int(y_all.sum()),
                            "n_neg_sampled": int(len(y_all) - y_all.sum()),
                            "logloss": ll_sh, "train_ap": ap_sh,
                            "params": params_global,
                            "feature_importance":
                                clf_sh.feature_importances_.tolist()}
        print(f"  SHARED:    n_pos={int(y_all.sum()):>6d}  "
              f"n_neg={int(len(y_all) - y_all.sum()):>8d}  "
              f"train_logloss={ll_sh:.4f}  train_ap={ap_sh:.3f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Verification (v3, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def verify_stacker_vs_singles(
    val: dict,
    method_names: list[str],
    oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    pretty_method_names: list[str] | None = None,
) -> dict:
    from sklearn.metrics import average_precision_score
    pretty = pretty_method_names or method_names
    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]
    N, H, W, M = val_scores.shape

    print(f"\n  Computing per-class pooled pixel-AP for {M} single models...")
    single_ap: dict[str, dict[str, float]] = {}
    for mi, mname in enumerate(pretty):
        single_ap[mname] = {}
        for cls in sorted(set(val_classes.tolist())):
            idx = np.flatnonzero(val_classes == cls)
            if idx.size == 0: continue
            s = val_scores[idx, :, :, mi].ravel()
            y = val_masks[idx].ravel()
            if int(y.sum()) == 0:
                ap = float("nan")
            else:
                try: ap = float(average_precision_score(y, s))
                except Exception: ap = float("nan")
            single_ap[mname][cls] = ap

    print(f"  Computing per-class pooled pixel-AP for stacker OOF...")
    stacker_ap: dict[str, float] = {}
    for cls, (op, ol) in oof_per_class.items():
        if int(ol.sum()) == 0:
            stacker_ap[cls] = float("nan")
            continue
        try: stacker_ap[cls] = float(average_precision_score(ol, op))
        except Exception: stacker_ap[cls] = float("nan")

    all_classes = sorted(set(val_classes.tolist()))
    hdr = f"  {'class':<10}"
    for n in pretty:
        hdr += f" {n[:12]:>13}"
    hdr += f" {'STACKER':>13} {'best_single':>13} {'win?':>7}"
    hr("VERIFICATION — per-class POOLED pixel-AP (stacker vs singles)", "=")
    print(hdr)
    n_class_wins = 0; n_class_total = 0; deltas: list[float] = []
    for cls in all_classes:
        line = f"  {cls:<10}"
        bests = []
        for n in pretty:
            ap = single_ap[n].get(cls, float("nan"))
            line += f" {ap:>13.4f}"
            if not math.isnan(ap):
                bests.append((ap, n))
        st = stacker_ap.get(cls, float("nan"))
        best_ap, best_n = (max(bests) if bests else (float("nan"), "?"))
        line += f" {st:>13.4f}"; line += f" {best_ap:>13.4f}"
        if not (math.isnan(st) or math.isnan(best_ap)):
            n_class_total += 1
            delta = st - best_ap
            deltas.append(delta)
            line += f" {'WIN' if delta > 0 else 'lose':>7}"
            if delta > 0: n_class_wins += 1
        else:
            line += f" {'?':>7}"
        print(line)

    def _mean_skipnan(d):
        vals = [v for v in d.values() if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")
    overall_singles = {n: _mean_skipnan(single_ap[n]) for n in pretty}
    overall_stacker = _mean_skipnan(stacker_ap)
    best_single_name = max(overall_singles, key=lambda n: overall_singles[n])
    best_single_ap   = overall_singles[best_single_name]
    print()
    print(f"  mean pooled-AP across classes (per single model):")
    for n in sorted(pretty, key=lambda nn: -overall_singles[nn]):
        marker = "  <- best single" if n == best_single_name else ""
        print(f"    {n:<20} {overall_singles[n]:.4f}{marker}")
    print(f"  STACKER mean pooled-AP:  {overall_stacker:.4f}")
    delta_overall = overall_stacker - best_single_ap
    if not math.isnan(delta_overall):
        if delta_overall > 0:
            print(f"  >>> STACKER BEATS best single ({best_single_name}) "
                  f"by +{delta_overall:.4f} mean pooled-AP.")
        else:
            print(f"  >>> STACKER LOSES to best single ({best_single_name}) "
                  f"by {delta_overall:+.4f} mean pooled-AP.")
    if n_class_total > 0:
        print(f"  >>> per-class: stacker wins {n_class_wins}/{n_class_total} "
              f"classes  (mean delta = {np.mean(deltas):+.4f})")
    return {"per_class_single": single_ap,
            "per_class_stacker": stacker_ap,
            "overall_singles": overall_singles,
            "overall_stacker": overall_stacker,
            "best_single_name": best_single_name,
            "best_single_overall_ap": best_single_ap,
            "delta_overall": delta_overall,
            "class_wins": n_class_wins,
            "class_total": n_class_total}


# ─────────────────────────────────────────────────────────────────────────────
# Test-time inference — v4: builds xv-aggregates + spatial prior + view
# ─────────────────────────────────────────────────────────────────────────────
def compute_test_cvd_per_method(
    decoded_per_method: list[dict[str, np.ndarray]],
    all_ids: list[str],
    class_map: dict[str, str] | None,
    default_class: str,
) -> tuple[dict[str, list[np.ndarray]], dict[str, bool]]:
    M = len(decoded_per_method)
    groups = group_test_ids_by_sample(all_ids, class_map, default_class)
    cvd_per_id: dict[str, list[np.ndarray]] = {}
    is_multi: dict[str, bool] = {}
    n_multi = 0
    for (_cls, _sid), ids_in_sample in groups.items():
        V = len(ids_in_sample)
        if V < 2:
            for id_ in ids_in_sample:
                H, W = decoded_per_method[0][id_].shape
                cvd_per_id[id_] = [np.zeros((H, W), dtype=np.float32)
                                    for _ in range(M)]
                is_multi[id_] = False
            continue
        n_multi += 1
        for id_ in ids_in_sample:
            is_multi[id_] = True
        for mi in range(M):
            stack = np.stack([decoded_per_method[mi][id_]
                                for id_ in ids_in_sample], axis=0)
            total = stack.sum(axis=0)
            for k, id_ in enumerate(ids_in_sample):
                sibling_mean = (total - stack[k]) / max(V - 1, 1)
                cvd_map = (stack[k] - sibling_mean).astype(np.float32)
                cvd_per_id.setdefault(id_, [None] * M)
                cvd_per_id[id_][mi] = cvd_map
    print(f"    cross-view (test): {n_multi} multi-view samples "
          f"({sum(1 for v in is_multi.values() if v)}/{len(is_multi)} "
          f"test images get non-zero CVD maps)")
    return cvd_per_id, is_multi


def compute_test_xv_aggregates(
    decoded_per_method: list[dict[str, np.ndarray]],
    all_ids: list[str],
    class_map: dict[str, str] | None,
    default_class: str,
) -> tuple[dict[str, dict[str, list[np.ndarray]]], dict[str, bool]]:
    """v4 NEW: per-test-id, per-method cross-view aggregates.

    Returns:
      xv_per_id[id] = {
        'max':    [map_method_0, map_method_1, ...],
        'mean':   [...],
        'std':    [...],
        'lonely': [...],
      }
      is_multi[id] = bool
    Single-view samples get all-zero maps so the feature column is 0.
    """
    M = len(decoded_per_method)
    groups = group_test_ids_by_sample(all_ids, class_map, default_class)
    xv_per_id: dict[str, dict[str, list[np.ndarray]]] = {}
    is_multi: dict[str, bool] = {}
    n_multi = 0
    for (_cls, _sid), ids_in_sample in groups.items():
        V = len(ids_in_sample)
        if V < 2:
            for id_ in ids_in_sample:
                H, W = decoded_per_method[0][id_].shape
                zero = np.zeros((H, W), dtype=np.float32)
                xv_per_id[id_] = {
                    "max":    [zero.copy() for _ in range(M)],
                    "mean":   [zero.copy() for _ in range(M)],
                    "std":    [zero.copy() for _ in range(M)],
                    "lonely": [zero.copy() for _ in range(M)],
                }
                is_multi[id_] = False
            continue
        n_multi += 1
        for id_ in ids_in_sample:
            is_multi[id_] = True
            xv_per_id[id_] = {
                "max":    [None] * M, "mean":   [None] * M,
                "std":    [None] * M, "lonely": [None] * M,
            }
        for mi in range(M):
            stack = np.stack([decoded_per_method[mi][id_]
                                for id_ in ids_in_sample], axis=0)
            for k, id_ in enumerate(ids_in_sample):
                if V == 1:
                    continue
                siblings = np.delete(stack, k, axis=0)
                sib_max  = siblings.max(axis=0).astype(np.float32)
                sib_mean = siblings.mean(axis=0).astype(np.float32)
                sib_std  = (siblings.std(axis=0).astype(np.float32)
                              if V > 2
                              else np.zeros_like(sib_mean))
                lonely   = (stack[k] - sib_max).astype(np.float32)
                xv_per_id[id_]["max"][mi]    = sib_max
                xv_per_id[id_]["mean"][mi]   = sib_mean
                xv_per_id[id_]["std"][mi]    = sib_std
                xv_per_id[id_]["lonely"][mi] = lonely
    print(f"    xv-aggregates (test): {n_multi} multi-view samples "
          f"({sum(1 for v in is_multi.values() if v)}/{len(is_multi)} "
          f"test images get non-zero aggregates)")
    return xv_per_id, is_multi


def fuse_test(submissions: list[dict[str, str]],
               models: dict,
               class_map: dict[str, str] | None,
               cfg: FeatureConfig,
               rank_norm_mode: str,
               default_class: str,
               calibrators_per_class: dict | None,
               small_cc_per_method: list[int],
               include_cvd: bool,
               include_xv: bool,
               spatial_priors: dict[str, np.ndarray] | None,
               all_classes_onehot: list[str] | None) -> dict[str, str]:
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common:
        raise RuntimeError("no test IDs in common across submissions")
    all_ids = sorted(common)
    M = len(submissions)
    feature_names = models.get("_feature_names", None)

    print(f"\nDecoding {M} submissions × {len(all_ids)} IDs each...")
    decoded_per_method: list[dict[str, np.ndarray]] = []
    t0 = time.time()
    for mi, sub in enumerate(submissions):
        d: dict[str, np.ndarray] = {}
        for j, sid in enumerate(all_ids):
            d[sid] = q8rle_to_float_matrix(sub[sid])
            if (j + 1) % 1000 == 0:
                print(f"    method {mi + 1}/{M}: {j + 1}/{len(all_ids)} "
                      f"({time.time() - t0:.1f}s)", flush=True)
        decoded_per_method.append(d)
        print(f"    method {mi + 1}/{M} decoded ({time.time() - t0:.1f}s)")

    if any(c > 0 for c in small_cc_per_method):
        print(f"\nApplying per-method small-CC suppression "
              f"(min_cc = {small_cc_per_method})...")
        for mi, min_cc in enumerate(small_cc_per_method):
            if min_cc <= 0:
                continue
            t1 = time.time()
            for sid in all_ids:
                decoded_per_method[mi][sid] = suppress_small_ccs_image(
                    decoded_per_method[mi][sid], min_cc=min_cc)
            print(f"    method {mi + 1}/{M}: suppressed CCs ≤ {min_cc} px "
                  f"({time.time() - t1:.1f}s)")

    if rank_norm_mode == "per-class-view":
        print(f"\nPer-(class, view) per-method rank-normalisation (test)...")
        rank_normalise_test_per_class_view_inplace(
            decoded_per_method, all_ids, class_map, default_class)
    elif rank_norm_mode == "per-class":
        print(f"\nPer-class per-method rank-normalisation (test)...")
        rank_normalise_test_per_class_inplace(
            decoded_per_method, all_ids, class_map, default_class)
    elif rank_norm_mode == "global":
        print(f"\nGlobal rank-normalisation (test) — v2/v3 compatibility...")
        rank_normalise_test_global_inplace(decoded_per_method, all_ids)
    elif rank_norm_mode == "none":
        print(f"\nSkipping rank-norm (test): --rank-norm none")
    else:
        raise ValueError(f"unknown rank_norm_mode: {rank_norm_mode}")

    cvd_per_id = None; is_multi_per_id_cvd = None
    if include_cvd:
        print(f"\nComputing cross-view disagreement maps for test...")
        cvd_per_id, is_multi_per_id_cvd = compute_test_cvd_per_method(
            decoded_per_method, all_ids, class_map, default_class)

    xv_per_id = None; is_multi_per_id_xv = None
    if include_xv:
        print(f"\nComputing cross-view aggregates (max/mean/std/lonely) for test...")
        xv_per_id, is_multi_per_id_xv = compute_test_xv_aggregates(
            decoded_per_method, all_ids, class_map, default_class)

    print(f"\nFusing {len(all_ids)} images with per-class XGBoost"
          f"{' + calibration' if calibrators_per_class else ''}...")
    fused: dict[str, str] = {}
    shared_entry = models.get("_SHARED_")
    t1 = time.time()
    n_uniform = 0
    spatial_cache: dict | None = None

    for i, sid in enumerate(all_ids):
        cls = (class_map.get(sid) if class_map else None) or default_class
        _, v = parse_sample_id(sid)
        v_int = int(v) if v is not None else None
        entry = models.get(cls)
        if entry is None or entry.get("_fallback_to_shared"):
            entry = shared_entry
        mats = [d[sid] for d in decoded_per_method]
        H, W = mats[0].shape
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        if entry is None or "model" not in entry:
            n_uniform += 1
            fused_mat = np.mean(np.stack(mats, axis=0), axis=0)
        else:
            cvd_list = None; is_mv_cvd = True
            if cvd_per_id is not None:
                cvd_list = cvd_per_id[sid]
                is_mv_cvd = is_multi_per_id_cvd.get(sid, False)
            xv_max_list = xv_mean_list = xv_std_list = xv_lonely_list = None
            is_mv_xv = True
            if xv_per_id is not None:
                xv_max_list    = xv_per_id[sid]["max"]
                xv_mean_list   = xv_per_id[sid]["mean"]
                xv_std_list    = xv_per_id[sid]["std"]
                xv_lonely_list = xv_per_id[sid]["lonely"]
                is_mv_xv = is_multi_per_id_xv.get(sid, False)
            is_mv_combined = (is_mv_cvd or is_mv_xv)
            prior_cls = (spatial_priors.get(cls) if spatial_priors else None)
            feats, _ = featurize_image(
                mats, cfg, spatial_cache=spatial_cache,
                feature_names=feature_names,
                cross_view_disagreement_per_method=cvd_list,
                is_multiview_sample=is_mv_combined,
                class_id=cls,
                all_classes=all_classes_onehot,
                view=v_int,
                spatial_prior=prior_cls,
                xv_max_per_method=xv_max_list,
                xv_mean_per_method=xv_mean_list,
                xv_std_per_method=xv_std_list,
                xv_lonely_per_method=xv_lonely_list,
            )
            X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
            p_raw = entry["model"].predict_proba(X)[:, 1].astype(np.float32)
            cal = (calibrators_per_class.get(cls)
                   if calibrators_per_class else None)
            p_out = apply_calibrator(cal, p_raw) if cal is not None else p_raw
            fused_mat = p_out.reshape(H, W)
            del feats, X, p_raw
        fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
        fused[sid] = float_matrix_to_q8rle(fused_mat)
        if (i + 1) % 500 == 0:
            print(f"    fused {i + 1}/{len(all_ids)}  "
                  f"({time.time() - t1:.1f}s)", flush=True)
    if n_uniform:
        print(f"  [warn] {n_uniform} images had no model — uniform avg")
    print(f"  fused all {len(all_ids)} in {time.time() - t1:.1f}s")
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# ablation_master append (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def append_to_master(master_csv: Path, row: dict) -> None:
    existing: list[dict] = []
    fieldnames: list[str] = []
    if master_csv.exists():
        with open(master_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)
    for k in row.keys():
        if k not in fieldnames:
            fieldnames.append(k)
    existing.append(row)
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def parse_small_cc_spec(spec: list[str] | None,
                         method_names: list[str],
                         default_min_cc: int) -> list[int]:
    cli_pairs: dict[str, int] = {}
    if spec:
        for s in spec:
            if "=" not in s:
                raise SystemExit(
                    f"[FATAL] --small-cc entries must be key=value; got {s!r}")
            k, v = s.split("=", 1)
            cli_pairs[k.strip().lower()] = int(v)
    out: list[int] = []
    for i, name in enumerate(method_names):
        fam = detect_model_family(name)
        n_lower = name.lower()
        val: int | None = None
        for k, v in cli_pairs.items():
            if k.isdigit() and int(k) == i:
                val = v; break
            if not k.isdigit() and k in n_lower:
                val = v; break
        if val is None and fam in cli_pairs:
            val = cli_pairs[fam]
        if val is None:
            val = DEFAULT_SMALL_CC_PER_FAMILY.get(fam, default_min_cc)
        out.append(int(val))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--local-preds", nargs="+", required=True, type=Path)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/data"))
    ap.add_argument("--class-map", type=Path)
    # v4: rank-norm with 3 modes (added per-class-view)
    ap.add_argument("--rank-norm", default="per-class-view",
                    choices=["per-class-view", "per-class", "global", "none"],
                    help="per-class-view (v4 default): rank within "
                         "{class × view}. per-class: v3 default. "
                         "global: v2 default. none: leave raw scores.")
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="stacker-xgb-v4")
    ap.add_argument("--no-zip", action="store_true")
    # small-CC
    ap.add_argument("--small-cc", nargs="*", default=None)
    ap.add_argument("--small-cc-default", type=int, default=0)
    ap.add_argument("--no-small-cc", action="store_true")
    # v3 toggles
    ap.add_argument("--no-cross-method-consensus", action="store_true")
    ap.add_argument("--cross-view-disagreement", action="store_true",
                    default=True)
    ap.add_argument("--no-cross-view-disagreement", dest="cross_view_disagreement",
                    action="store_false")
    # v4 toggles
    ap.add_argument("--prior-heatmaps-dir", type=Path,
                    default=Path("analysis_out/tables"),
                    help="Directory of per-class heatmaps "
                         "(06_heat_<class>.npy) saved by "
                         "analyze_spacepresso_dataset.py §6.")
    ap.add_argument("--no-spatial-prior", action="store_true",
                    help="v4: disable per-class spatial-prior feature.")
    ap.add_argument("--no-class-onehot", action="store_true",
                    help="v4: disable class one-hot feature.")
    ap.add_argument("--no-view-onehot", action="store_true",
                    help="v4: disable view one-hot feature.")
    ap.add_argument("--no-xv-aggregates", action="store_true",
                    help="v4: disable per-method cross-view "
                         "max/mean/std/lonely features.")
    # Feature toggles (v2/v3 compat)
    ap.add_argument("--no-window-stats", action="store_true")
    ap.add_argument("--no-gradient", action="store_true")
    ap.add_argument("--no-laplacian", action="store_true")
    ap.add_argument("--no-dist-to-hot", action="store_true")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-cross-stats", action="store_true")
    ap.add_argument("--no-image-aggregates", action="store_true")
    # XGB params (unchanged)
    ap.add_argument("--n-estimators", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--subsample", type=float)
    ap.add_argument("--colsample-bytree", type=float)
    ap.add_argument("--reg-alpha", type=float)
    ap.add_argument("--reg-lambda", type=float)
    ap.add_argument("--min-child-weight", type=float)
    # Tuning
    ap.add_argument("--tune-mode", default="none",
                    choices=["none", "global", "per-class"])
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--tune-cv", default="loao", choices=["loao", "loio"])
    ap.add_argument("--tune-timeout-min", type=float, default=None)
    # Calibration
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "platt", "isotonic"])
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if not HAS_XGB:
        raise SystemExit("[FATAL] xgboost not installed.")
    if len(args.runs) < 2:
        raise SystemExit("need ≥ 2 methods to stack")
    if len(args.local_preds) != len(args.runs):
        raise SystemExit("--local-preds count must match --runs count")
    if args.tune_mode != "none" and not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")

    method_names = [p.parent.name for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.out.parent

    with tee_to(run_dir / "run_log.txt"):
        hr(f"XGBOOST STACKER v4 — {len(args.runs)} methods", "=")
        print(f"  rank_norm    : {args.rank_norm}")
        print(f"  tune_mode    : {args.tune_mode}")
        print(f"  calibrate    : {args.calibrate}")
        print(f"  prior dir    : {args.prior_heatmaps_dir}")
        for i, (r, lp) in enumerate(zip(args.runs, args.local_preds)):
            fam = detect_model_family(method_names[i])
            print(f"  method {i}: {method_names[i]}  (family={fam})")

        # Small-CC resolve
        if args.no_small_cc:
            small_cc_per_method = [0] * len(method_names)
        else:
            small_cc_per_method = parse_small_cc_spec(
                args.small_cc, method_names, args.small_cc_default)
        print(f"\n  small-CC suppression per method: {small_cc_per_method}")

        # Load submissions
        print("\nLoading test submissions...")
        subs = [load_submission(p) for p in args.runs]
        for p, s in zip(args.runs, subs):
            print(f"  {p.parent.name}/{p.name}: {len(s)} rows")

        # Load local-val
        print("\nLoading local-val predictions...")
        preds_per_method = []
        any_has_paths = False
        for p in args.local_preds:
            d = load_local_preds(p)
            if d.get("image_paths") is not None:
                any_has_paths = True
            print(f"  {p.parent.name}/{p.name}: {len(d['ids'])} val images, "
                  f"{float(d['masks'].mean()) * 100:.3f}% positive pixels"
                  f"{' (with image_paths)' if d.get('image_paths') is not None else ''}")
            preds_per_method.append(d)

        # Small-CC on val
        if any(c > 0 for c in small_cc_per_method):
            print(f"\nApplying per-method small-CC suppression to local-val...")
            for mi, min_cc in enumerate(small_cc_per_method):
                if min_cc <= 0:
                    continue
                t1 = time.time()
                preds_per_method[mi]["scores"] = suppress_small_ccs_volume(
                    preds_per_method[mi]["scores"], min_cc=min_cc)
                print(f"    method {mi + 1}: min_cc={min_cc} ({time.time() - t1:.1f}s)")

        # Align (now also tracks views)
        print("\nAligning local-val predictions across methods...")
        val = align_local_preds(preds_per_method, method_names)

        # Rank-norm on val
        if args.rank_norm == "per-class-view":
            if (val["views"] >= 0).any():
                print(f"\nPer-(class, view) per-method rank-normalisation (val)...")
                rank_normalise_per_class_view_inplace_val(
                    val["scores"], val["classes"], val["views"])
            else:
                print(f"\n[warn] --rank-norm per-class-view requested but no "
                      f"views were parsed; falling back to per-class.")
                args.rank_norm = "per-class"
                rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
        elif args.rank_norm == "per-class":
            print(f"\nPer-class per-method rank-normalisation (val)...")
            rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
        elif args.rank_norm == "global":
            print(f"\nGlobal rank-normalisation (val)...")
            rank_normalise_global_inplace_val(val["scores"])
        else:
            print(f"\nSkipping rank-norm (val): --rank-norm none")

        # Cross-view aggregates + CVD on val
        cvd_val = is_multi_cvd_val = None
        use_cvd = args.cross_view_disagreement and any_has_paths
        if args.cross_view_disagreement and not any_has_paths:
            print(f"\n[warn] CVD disabled (no image_paths in any npz).")
        if use_cvd:
            print(f"\nBuilding cross-view DISAGREEMENT maps (val)...")
            cvd_val, is_multi_cvd_val = compute_cross_view_disagreement_val(
                val["scores"], val["classes"], val.get("image_paths"))

        xv_max_val = xv_mean_val = xv_std_val = xv_lonely_val = None
        is_multi_xv_val = None
        use_xv = (not args.no_xv_aggregates) and any_has_paths
        if (not args.no_xv_aggregates) and not any_has_paths:
            print(f"\n[warn] xv-aggregates disabled (no image_paths).")
        if use_xv:
            print(f"\nBuilding cross-view AGGREGATES (max/mean/std/lonely) (val)...")
            (xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val,
             is_multi_xv_val) = compute_xv_aggregates_val(
                val["scores"], val["classes"], val.get("image_paths"))

        # Spatial priors
        classes = sorted(set(val["classes"].tolist()))
        spatial_priors: dict[str, np.ndarray] | None = None
        if not args.no_spatial_prior:
            print(f"\nLoading per-class spatial priors from "
                  f"{args.prior_heatmaps_dir}...")
            spatial_priors = load_spatial_priors(args.prior_heatmaps_dir, classes)
        else:
            print(f"\nSkipping spatial-prior feature (--no-spatial-prior).")

        # Feature config
        cfg = FeatureConfig(
            use_window_mean=not args.no_window_stats,
            use_window_max=not args.no_window_stats,
            use_window_std=not args.no_window_stats,
            use_gradient=not args.no_gradient,
            use_laplacian=not args.no_laplacian,
            use_dist_to_hot=not args.no_dist_to_hot,
            use_spatial=not args.no_spatial,
            use_cross_stats=not args.no_cross_stats,
            use_image_aggregates=not args.no_image_aggregates,
            use_cross_method_consensus=not args.no_cross_method_consensus,
            use_cross_view_disagreement=use_cvd,
            use_spatial_prior=(spatial_priors is not None),
            use_class_onehot=not args.no_class_onehot,
            use_view_onehot=not args.no_view_onehot,
            use_xv_aggregates=use_xv,
        )
        print(f"\nFeature config:")
        for k, v in asdict(cfg).items():
            print(f"  {k:<32} = {v}")

        all_classes_onehot = classes if cfg.use_class_onehot else None

        # Build training matrices
        print(f"\nBuilding training matrices (featurize + sample negatives)...")
        print(f"  classes present in val: {classes}")
        training_data = build_training_data(
            val, classes, cfg,
            neg_per_pos=args.neg_per_pos, seed=args.seed,
            cvd=cvd_val, is_multi=is_multi_cvd_val,
            xv_max=xv_max_val, xv_mean=xv_mean_val,
            xv_std=xv_std_val, xv_lonely=xv_lonely_val,
            spatial_priors=spatial_priors,
            all_classes_onehot=all_classes_onehot,
        )
        feature_names = training_data.get("_feature_names", [])
        print(f"  total features per pixel: {len(feature_names)}")

        # XGB params
        cli_overrides = {k: v for k, v in {
            "n_estimators":     args.n_estimators,
            "max_depth":        args.max_depth,
            "learning_rate":    args.learning_rate,
            "subsample":        args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha":        args.reg_alpha,
            "reg_lambda":       args.reg_lambda,
            "min_child_weight": args.min_child_weight,
        }.items() if v is not None}
        params_global = {**DEFAULT_XGB_PARAMS, **cli_overrides}
        params_per_class: dict | None = None
        tune_per_class_results: dict | None = None
        timeout_sec = (args.tune_timeout_min * 60.0
                       if args.tune_timeout_min else None)

        if args.tune_mode == "global":
            tuned = tune_global(training_data, n_trials=args.n_trials,
                                  seed=args.seed, cv_mode=args.tune_cv,
                                  timeout=timeout_sec,
                                  neg_per_pos=args.neg_per_pos)
            params_global = {**params_global, **tuned}
        elif args.tune_mode == "per-class":
            tune_per_class_results = tune_per_class(
                training_data, n_trials=args.n_trials,
                seed=args.seed, cv_mode=args.tune_cv,
                timeout_per_class=timeout_sec,
                neg_per_pos=args.neg_per_pos)
            params_per_class = {cls: r["best_params"]
                                for cls, r in tune_per_class_results.items()}

        # OOF
        oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if tune_per_class_results is not None:
            for cls, r in tune_per_class_results.items():
                oof_per_class[cls] = (r["oof_preds"], r["oof_labels"])
        else:
            print(f"\nCollecting OOF preds per class...")
            for cls, td in training_data.items():
                if cls.startswith("_") or td.get("_fallback_to_shared"):
                    continue
                params = get_params_for_class(cls, params_global,
                                                params_per_class)
                t0 = time.time()
                op, ol = loao_oof_one_class(
                    td, params, args.seed, cv_mode=args.tune_cv,
                    neg_per_pos=args.neg_per_pos)
                print(f"  {cls}: {len(op):>9d} OOF preds "
                      f"({time.time() - t0:.1f}s)")
                oof_per_class[cls] = (op, ol)

        # Calibration
        calibrators_per_class: dict | None = None
        if args.calibrate != "none":
            print(f"\nFitting per-class {args.calibrate} calibrators...")
            calibrators_per_class = {}
            from sklearn.metrics import average_precision_score
            for cls, (op, ol) in oof_per_class.items():
                cal = fit_calibrator(args.calibrate, op, ol)
                calibrators_per_class[cls] = cal
                try: ap_pre = float(average_precision_score(ol, op))
                except Exception: ap_pre = float("nan")
                p_cal = apply_calibrator(cal, op)
                try: ap_post = float(average_precision_score(ol, p_cal))
                except Exception: ap_post = float("nan")
                print(f"  {cls}: OOF pooled-AP pre={ap_pre:.4f} "
                      f"post={ap_post:.4f}")

        # Verify
        verification: dict = {}
        if not args.no_verify and oof_per_class:
            verification = verify_stacker_vs_singles(
                val, method_names, oof_per_class)

        # Fit production
        print(f"\nFitting final per-class XGBoost models...")
        models = fit_per_class(training_data, params_global,
                                params_per_class, args.seed)

        # Class map for test
        print("\nBuilding ID → class map for test set...")
        if args.class_map and args.class_map.exists():
            class_map: dict[str, str] = {}
            with open(args.class_map, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "ID" in row and "class" in row:
                        class_map[row["ID"]] = row["class"]
            print(f"  loaded {len(class_map)} entries from {args.class_map}")
        else:
            class_map = build_class_map_from_data(args.data_root)
            print(f"  built from {args.data_root}: {len(class_map)} entries")
        default_class = classes[0] if classes else "_default_"
        if not class_map:
            print(f"  [warn] no class map — every test image uses SHARED model")
            class_map = None

        # Inference
        fused = fuse_test(
            subs, models, class_map, cfg,
            rank_norm_mode=args.rank_norm,
            default_class=default_class,
            calibrators_per_class=calibrators_per_class,
            small_cc_per_method=small_cc_per_method,
            include_cvd=use_cvd,
            include_xv=use_xv,
            spatial_priors=spatial_priors,
            all_classes_onehot=all_classes_onehot)

        # Submission
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ID", "Label"])
            for sid in sorted(fused):
                w.writerow([sid, fused[sid]])
        print(f"\nWrote {len(fused)} rows -> {args.out}")
        if not args.no_zip:
            zip_path = args.out.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w",
                                 compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(args.out, arcname=args.out.name)
            print(f"Zipped -> {zip_path}")

        # OOF dump
        if oof_per_class:
            oof_path = run_dir / "oof_predictions.npz"
            to_save = {"classes": np.array(list(oof_per_class.keys()),
                                            dtype=object)}
            for cls, (op, ol) in oof_per_class.items():
                to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
                to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
            np.savez_compressed(oof_path, **to_save)
            print(f"Saved OOF preds -> {oof_path}")

        # Config dump
        model_dump = {
            "version": 4,
            "methods": method_names,
            "rank_norm": args.rank_norm,
            "small_cc_per_method": small_cc_per_method,
            "use_cross_method_consensus": cfg.use_cross_method_consensus,
            "use_cross_view_disagreement": cfg.use_cross_view_disagreement,
            "use_xv_aggregates": cfg.use_xv_aggregates,
            "use_spatial_prior": cfg.use_spatial_prior,
            "use_class_onehot": cfg.use_class_onehot,
            "use_view_onehot": cfg.use_view_onehot,
            "prior_heatmaps_dir": str(args.prior_heatmaps_dir),
            "neg_per_pos": args.neg_per_pos,
            "seed": args.seed,
            "feature_config": asdict(cfg),
            "feature_names": feature_names,
            "xgb_params_global": params_global,
            "xgb_params_per_class": params_per_class,
            "tune_mode": args.tune_mode,
            "tune_cv": args.tune_cv,
            "tune_n_trials": args.n_trials if args.tune_mode != "none" else 0,
            "tune_timeout_min": args.tune_timeout_min,
            "tune_per_class_best_ap": (
                {cls: r["best_cv_ap"]
                 for cls, r in (tune_per_class_results or {}).items()}
                if tune_per_class_results else None),
            "calibration_method": args.calibrate,
            "calibrators_per_class": calibrators_per_class,
            "verification": verification,
            "per_class_models": {},
        }
        for k, v in models.items():
            if k.startswith("_") and k != "_SHARED_":
                continue
            if isinstance(v, dict) and "model" in v:
                model_dump["per_class_models"][k] = {
                    "n_pos": v["n_pos"],
                    "n_neg_sampled": v["n_neg_sampled"],
                    "train_logloss": v.get("logloss"),
                    "train_ap": v.get("train_ap"),
                    "params": v.get("params"),
                    "feature_importance": v.get("feature_importance"),
                }
            else:
                model_dump["per_class_models"][k] = {
                    "fallback_to_shared": True,
                    "n_pos": v.get("n_pos"),
                    "n_neg_sampled": v.get("n_neg_sampled"),
                }
        cfg_path = run_dir / "stacker_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(model_dump, f, indent=2)
        print(f"\nWrote stacker config -> {cfg_path}")

        # Top features
        if "_SHARED_" in models and "feature_importance" in models["_SHARED_"]:
            fi = np.asarray(models["_SHARED_"]["feature_importance"])
            if feature_names and len(feature_names) == len(fi):
                top = np.argsort(-fi)[:25]
                print(f"\nTop-25 feature importances (SHARED model):")
                for r, j in enumerate(top, 1):
                    print(f"  {r:>2d}. {feature_names[j]:<26s}  {fi[j]:.4f}")
                # v4 specific: group importances by family
                groups_fi: dict[str, float] = defaultdict(float)
                for name, imp in zip(feature_names, fi):
                    if name.startswith("m") and "_xv_" in name:
                        groups_fi["xv_aggregates"] += float(imp)
                    elif name.startswith("m") and "_cvd" in name:
                        groups_fi["cvd"] += float(imp)
                    elif name == "s_prior":
                        groups_fi["spatial_prior"] += float(imp)
                    elif name.startswith("c_"):
                        groups_fi["class_onehot"] += float(imp)
                    elif name.startswith("v_"):
                        groups_fi["view_onehot"] += float(imp)
                    elif name.startswith("xc_"):
                        groups_fi["cross_method_consensus"] += float(imp)
                    elif name.startswith("x_"):
                        groups_fi["cross_method_stats"] += float(imp)
                    elif name.startswith("s_"):
                        groups_fi["spatial"] += float(imp)
                    else:
                        groups_fi["per_method_v3"] += float(imp)
                print(f"\nFeature-family importance shares (SHARED):")
                tot = sum(groups_fi.values()) + 1e-12
                for fam in sorted(groups_fi, key=lambda k: -groups_fi[k]):
                    print(f"  {fam:<28s} {groups_fi[fam] / tot * 100:>6.2f}%")

        # Ablation row
        run_id = "stacker_xgb_v4_" + hashlib.sha1(
            "|".join(str(p) for p in args.runs).encode("utf-8")
        ).hexdigest()[:6]
        overall_stacker = verification.get("overall_stacker", float("nan"))
        notes = (f"xgb v4 | M={len(subs)} | F={len(feature_names)} | "
                  f"rank_norm={args.rank_norm} | "
                  f"small_cc=[{','.join(str(c) for c in small_cc_per_method)}] | "
                  f"consensus={int(cfg.use_cross_method_consensus)} | "
                  f"cvd={int(cfg.use_cross_view_disagreement)} | "
                  f"xv={int(cfg.use_xv_aggregates)} | "
                  f"prior={int(cfg.use_spatial_prior)} | "
                  f"cls1h={int(cfg.use_class_onehot)} | "
                  f"v1h={int(cfg.use_view_onehot)} | "
                  f"tune={args.tune_mode} | calibrate={args.calibrate}")
        row = {
            "run_id": run_id, "run_tag": args.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "STACKER_XGB_V4",
            "feature_layers": "", "input_size": "",
            "n_classes": len(classes),
            "AP_overall": (f"{overall_stacker:.4f}"
                             if not math.isnan(overall_stacker) else ""),
            "runtime_min": "",
            "submission_path": str(args.out.with_suffix(".zip")),
            "notes": notes,
        }
        append_to_master(args.master_csv, row)
        print(f"\nAppended row to {args.master_csv}")
        hr(f"DONE — run_id={run_id}", "=")


if __name__ == "__main__":
    main()