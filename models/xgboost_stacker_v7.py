"""XGBoost stacker v7 — generalization-first redesign over v6.

Same I/O contract as v6: N submission.csv + N local_predictions.npz
→ fused submission + stacker_config.json + oof_predictions.npz + run_log.txt
+ ablation row. Inputs only — no model checkpoints, no memory banks needed.

==============================================================================
# v7 design philosophy: every feature must travel from val to test.
==============================================================================

Top teams reach ~0.90 on the LB; we sit at ~0.82. The gap is not tuning —
it's representation. v7 attacks four pathologies in v6:

  P1. Rank-norm operates on *test-set* pools, not on stable references.
      A pixel's rank depends on which other test images happen to be in
      its bucket. Class_08's symKL=0.12 (analysis §4) means its global
      rank distribution drifts substantially between train and test;
      pooled rank-norm bakes that drift into the feature.

  P2. Pixel-aligned cross-view aggregates assume "same defect → same
      pixel across views". Analysis §10 shows pairwise mask IoU p50=0.35
      with class_03 at IoU=0.0 — the same defect, when seen by another
      camera, sits at a *different* pixel. xv_max/mean/std/lonely on a
      pixel basis washes signal on low-IoU classes.

  P3. Post-processing is off (small_cc = 0 everywhere). The advisor
      brief is explicit: "drop what is too small, too weak, or too
      lonely." Analysis §7 measured p25(area)=695px and p95(compactness)
      =1.10. v7 turns CC structure into *features* (not hard filters)
      so XGBoost decides where to suppress.

  P4. Too many features → small val (~40 anomalies/class) overfits.
      v6 ships 339-391 features; v7 ships ~120 with denser per-feature
      signal.

==============================================================================
# v7 feature catalogue (∼120 features, see FEATURE_FAMILIES below)
==============================================================================

## Stage-A: per-image rank        (M features: m{mi}_rank)
   Standard, unchanged from v6. Image-local, always travels.

## Stage-B: train_good-anchored normalization (3*M features)
   For each (method, class, view), fit p50/p95/p99 on val's TRAIN_GOOD
   pixels (not anomalous pixels — keeps it test-legal and stable).
   At inference per pixel:
     m{mi}_normed     = (s - p50) / (p99 - p50 + ε)
     m{mi}_tail_excess = max(0, s - p99) / (p99 - p95 + ε)
     m{mi}_normed_clip = clip(m{mi}_normed, -3, 10)   (XGB-friendly)
   This replaces v6's per-(class, view) rank-norm. Robust to test-side
   distribution drift because the anchors are independent of test.

## Connected-component features  (≤7 * K features; K = n_top_methods)
   Threshold each top-method's map at per-(method, class) p99 of
   train_good. Label CCs. Per pixel inside its CC:
     cc_logarea_top{k}     log1p(area)
     cc_compactness_top{k} 4π·area / perim²
     cc_max_score_top{k}   max raw score in this CC
     cc_mean_score_top{k}  mean raw score in this CC
     cc_n_in_image_top{k}  total CCs above threshold in this image
     cc_size_rank_top{k}   rank by size (1=largest)
     cc_centroid_dist_top{k}  L2 distance to CC centroid (px)
   "Lonely / weak / small" all turn into XGB splits.

## Image-level multi-view agreement  (4*M features, broadcast)
   For each sample group (class, sample_id), compute per method:
     mv_self_p99_{mi}       this view's p99
     mv_sib_max_p99_{mi}    max p99 across sibling views
     mv_sib_mean_p99_{mi}   mean p99 across sibling views
     mv_lonely_image_{mi}   self_p99 - sib_max_p99
   Broadcast as constant planes. Captures "the same defect lights up
   somewhere in at least one sibling view," which §10 says happens
   with P=0.83 — but at a different *pixel*. Image-level beats pixel.

## Patch-level sibling consistency  (M features)
   For each pixel in view V, get the max score within a 7×7 window in
   each sibling view, then take the max over siblings:
     m{mi}_sib_window_max
   Tolerates spatial misalignment (multi-view IoU low) but still
   catches "corroborated by some nearby region in another view."

## Cross-method consensus  (4 features)
   xc_n_above_p95           count of methods above their per-(class,view)
                            train_good p95 at this pixel
   xc_mean_normed           mean of normed scores
   xc_max_normed            max  of normed scores
   xc_min_top3_normed       min of top-3 normed (consensus-of-strongest)

## Cross-method dispersion  (2 features)
   xc_cv_normed             std/(mean+ε) of normed scores
   xc_range_normed          max - min of normed scores
   Low cv = methods agree → trust; high cv = methods disagree → wary.

## Spatial priors  (3 features)
   s_prior_class            per-class mean defect heatmap from §6
   s_prior_anomtype_max     max over per-(class, type) heatmaps (§6)
   s_prior_global           cross-class mean heatmap
   All from analysis_out/tables/ pre-computed npys.

## Spatial geometry  (4 features)
   s_x, s_y, s_dist_edge, s_dist_center
   Unchanged from v6.

## Class one-hot                 (n_classes features, ~8)
## View one-hot                   (n_views + 1 = 6 features)
## Best-model-per-class rank      (M features, one-hot of class's
                                    expert ranking by val pooled AP)

==============================================================================
# Removed from v6
==============================================================================
- gradient + laplacian          (boundary noise on smoothed scores)
- distance-to-hot               (subsumed by CC features)
- pixel-aligned xv_max/mean/std/lonely (replaced by image-level + window)
- xc_top1pct_count, xc_top5pct_count   (per-(class,view) tail features
                                          do this job better)
- per-image image_aggregates (imgmax/p99/mean/std) for ALL methods
  (kept only for image_p99 z-score on top-K methods)
- mahalanobis on normed scores  (kept as optional, default OFF — was
                                  marginal in v5 and adds RAM overhead)

==============================================================================
# Test-set legality
==============================================================================
All percentile/heatmap statistics fit on VAL TRAIN_GOOD pixels (or VAL
TRAIN_GOOD bucketed maps). At test time we only:
  - decode q8rle predictions
  - look up stats indexed by (method, class, view)
  - run featurize_image
  - apply XGBoost
No test pixels, masks, or aggregate statistics ever feed back into
training or fitting. Inspection-only.

==============================================================================
# Memory layout
==============================================================================
Same uint8/uint16 compact storage as v6. fuse_test streams by sample
group; per-group memory is O(V × M × H × W) float32, transient. The
expensive things (train_good stats, heatmaps, top-K method picks) are
small dicts that live for the whole run.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
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
# Tee logger
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


def fmt_bytes(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0: return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} PB"


def mem_rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return 0.0


def mem_print(tag: str) -> None:
    g = mem_rss_gb()
    if g > 0.0:
        print(f"  [mem] {tag:<48s} RSS = {g:6.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec
# ─────────────────────────────────────────────────────────────────────────────
def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255),
                0, 255).astype(np.uint8)
    return _uint8_matrix_to_q8rle(q)


def _uint8_matrix_to_q8rle(q: np.ndarray) -> str:
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


def q8rle_to_uint8_matrix(s: str) -> np.ndarray:
    parts = s.split()
    h, w = int(parts[1]), int(parts[2])
    if len(parts) <= 3:
        return np.zeros((h, w), dtype=np.uint8)
    body = np.array(parts[3:], dtype=np.int64)
    vals = body[0::2].astype(np.uint8)
    lens = body[1::2]
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat


def to_f32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (1.0 / 255.0)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) * (1.0 / 65535.0)
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# File-based cache
# ─────────────────────────────────────────────────────────────────────────────
def file_meta_hash(paths: list[Path], extra: str = "") -> str:
    h = hashlib.sha1(); h.update(extra.encode())
    for p in sorted(paths, key=str):
        try:
            st = p.stat()
            h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}|".encode())
        except FileNotFoundError:
            h.update(f"{p.name}|missing|".encode())
    return h.hexdigest()[:16]


class Cache:
    """Trivial file-based cache; disabled when cache_dir is None."""
    def __init__(self, cache_dir: Path | None):
        self.dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _p(self, name: str) -> Path:
        return self.dir / name  # type: ignore[union-attr]

    def has(self, name: str) -> bool:
        return self.enabled and self._p(name).exists()

    def load_npz(self, name: str, allow_pickle: bool = True):
        if not self.enabled: return None
        p = self._p(name)
        if not p.exists(): return None
        try:
            d = np.load(p, allow_pickle=allow_pickle)
            print(f"  [cache] HIT  {name}  ({fmt_bytes(p.stat().st_size)})")
            return d
        except Exception as e:
            print(f"  [cache] FAIL {name}: {e}  (recomputing)")
            return None

    def save_npz(self, name: str, **kwargs) -> None:
        if not self.enabled: return
        p = self._p(name); tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **kwargs)
        tmp.replace(p)
        print(f"  [cache] SAVE {name}  ({fmt_bytes(p.stat().st_size)})")

    def load_pkl(self, name: str):
        if not self.enabled: return None
        p = self._p(name)
        if not p.exists(): return None
        try:
            with open(p, "rb") as f: obj = pickle.load(f)
            print(f"  [cache] HIT  {name}  ({fmt_bytes(p.stat().st_size)})")
            return obj
        except Exception as e:
            print(f"  [cache] FAIL {name}: {e}  (recomputing)")
            return None

    def save_pkl(self, name: str, obj) -> None:
        if not self.enabled: return
        p = self._p(name); tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)
        print(f"  [cache] SAVE {name}  ({fmt_bytes(p.stat().st_size)})")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders (preserved verbatim from v6)
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
    out["image_paths"] = (data["image_paths"].astype(str)
                            if "image_paths" in data.files else None)
    return out


def build_class_map_from_data(data_root: Path) -> dict[str, str]:
    if not data_root.exists(): return {}
    out: dict[str, str] = {}
    for cdir in sorted(data_root.iterdir()):
        if not cdir.is_dir() or not cdir.name.startswith("class_"): continue
        test_dir = cdir / "test"
        if not test_dir.exists(): continue
        for p in test_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
                out[p.stem] = cdir.name
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sample id / view parsing
# ─────────────────────────────────────────────────────────────────────────────
PATH_VIEW_RE = re.compile(r"^(?P<sid>.+?)_view(?P<v>\d+)(?:\.[A-Za-z]+)?$")


def parse_sample_id(name: str) -> tuple[str, int | None]:
    m = PATH_VIEW_RE.match(name)
    if m: return m.group("sid"), int(m.group("v"))
    return name, None


# ═════════════════════════════════════════════════════════════════════════════
# v7 CORE — train_good-anchored statistics
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class NormStats:
    """Per-(method, class, view) percentile anchors fit on train_good
    pixels only. Used to normalise both val and test consistently.
    Anchors are robust quantiles (not mean/std) so they survive class_08-
    style domain drift better."""
    p50: float
    p95: float
    p99: float
    n_pixels: int      # how many train_good pixels backed this fit

    def normed(self, s: np.ndarray) -> np.ndarray:
        denom = (self.p99 - self.p50)
        if denom < 1e-6: denom = 1e-6
        return ((s - self.p50) / denom).astype(np.float32)

    def tail_excess(self, s: np.ndarray) -> np.ndarray:
        denom = (self.p99 - self.p95)
        if denom < 1e-6: denom = 1e-6
        return np.maximum(0.0, s - self.p99).astype(np.float32) / denom


# Anchor lookup: dict[(mi, cls, view_or_None)] -> NormStats
AnchorTable = dict[tuple[int, str, "int | None"], NormStats]


def fit_train_good_anchors(val: dict, n_methods: int,
                              min_pixels_per_cell: int = 2000) -> AnchorTable:
    """For each (method, class, view) cell, fit (p50, p95, p99) on the
    pixels of val images that are NON-anomalous (mask==0). This is the
    closest test-legal proxy we have to a "what does this method say
    about a clean image of this view of this class" reference.

    Why non-anomalous PIXELS of all val images, not just train_good
    images? Because local_predictions.npz only contains anomalous val
    images (each anomalous image still has ~95-99% clean pixels). This
    keeps anchors stable.

    Fallback chain when a (method, class, view) cell has too few pixels:
       (mi, cls, view) → (mi, cls, None) → (mi, None, None)
    """
    val_scores = val["scores"]      # (N, H, W, M)
    val_masks  = val["masks"]       # (N, H, W) uint8
    val_classes = val["classes"]
    val_views   = val.get("views", np.full(len(val_classes), -1, dtype=np.int32))
    N, H, W, M = val_scores.shape

    print(f"\nFitting train_good anchors over (method × class × view)...")
    print(f"  val pool: {N} images @ {H}x{W}, neg-pixel rate "
          f"= {1.0 - float(val_masks.mean()):.4f}")

    # Collect clean-pixel scores per cell.
    cell_buf: dict[tuple[int, str, "int | None"], list[np.ndarray]] = defaultdict(list)
    for i in range(N):
        cls = str(val_classes[i])
        v_raw = int(val_views[i])
        v: int | None = v_raw if v_raw >= 0 else None
        neg = (val_masks[i] == 0).ravel()
        if neg.sum() == 0: continue
        for mi in range(M):
            s = val_scores[i, :, :, mi].ravel()[neg]
            cell_buf[(mi, cls, v)].append(s)

    # Class-wide fallback (view=None).
    class_buf: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    method_buf: dict[int, list[np.ndarray]] = defaultdict(list)
    for (mi, cls, _v), arrs in cell_buf.items():
        class_buf[(mi, cls)].extend(arrs)
        method_buf[mi].extend(arrs)

    out: AnchorTable = {}
    n_full = n_class_fb = n_method_fb = 0
    cells_seen = set(cell_buf.keys())

    # Add explicit (mi, cls, None) and (mi, None, None) keys too so
    # fallback at inference is a pure dict lookup.
    for (mi, cls, v), arrs in cell_buf.items():
        flat = np.concatenate(arrs)
        if flat.size < min_pixels_per_cell:
            # Promote: try (mi, cls, None).
            cls_flat = np.concatenate(class_buf[(mi, cls)])
            if cls_flat.size >= min_pixels_per_cell:
                p50, p95, p99 = np.percentile(cls_flat, [50, 95, 99])
                out[(mi, cls, v)] = NormStats(float(p50), float(p95),
                                                  float(p99), int(cls_flat.size))
                n_class_fb += 1
                continue
            # Promote: try (mi, None, None).
            method_flat = np.concatenate(method_buf[mi])
            p50, p95, p99 = np.percentile(method_flat, [50, 95, 99])
            out[(mi, cls, v)] = NormStats(float(p50), float(p95),
                                              float(p99), int(method_flat.size))
            n_method_fb += 1
            continue
        p50, p95, p99 = np.percentile(flat, [50, 95, 99])
        out[(mi, cls, v)] = NormStats(float(p50), float(p95),
                                          float(p99), int(flat.size))
        n_full += 1

    # Pre-populate (mi, cls, None) entries for unseen views at test time.
    for (mi, cls), arrs in class_buf.items():
        if (mi, cls, None) in out: continue
        flat = np.concatenate(arrs)
        if flat.size >= min_pixels_per_cell:
            p50, p95, p99 = np.percentile(flat, [50, 95, 99])
            out[(mi, cls, None)] = NormStats(float(p50), float(p95),
                                              float(p99), int(flat.size))

    # Pre-populate (mi, None, None) method-wide fallbacks.
    for mi, arrs in method_buf.items():
        if (mi, None, None) in out: continue  # type: ignore
        flat = np.concatenate(arrs)
        if flat.size > 0:
            p50, p95, p99 = np.percentile(flat, [50, 95, 99])
            out[(mi, None, None)] = NormStats(float(p50), float(p95),  # type: ignore
                                                float(p99), int(flat.size))

    print(f"  fitted {n_full} cells at full granularity, "
          f"{n_class_fb} via class fallback, {n_method_fb} via method fallback")
    # Pretty summary for a few cells
    sample_keys = sorted(cells_seen)[:5]
    print(f"  sample anchor values (mi, cls, view) → p50/p95/p99 (n):")
    for k in sample_keys:
        a = out[k]
        print(f"    {k}: {a.p50:.4f} / {a.p95:.4f} / {a.p99:.4f}  "
              f"(n={a.n_pixels:>8d})")
    return out


def get_anchor(anchors: AnchorTable, mi: int, cls: str,
                view: int | None) -> NormStats:
    """Look up with fallback. Always returns something (final fallback is
    a benign identity-ish anchor)."""
    k1 = (mi, cls, view)
    if k1 in anchors: return anchors[k1]
    k2 = (mi, cls, None)
    if k2 in anchors: return anchors[k2]
    k3: tuple = (mi, None, None)  # type: ignore[assignment]
    if k3 in anchors: return anchors[k3]
    # Final safety net.
    return NormStats(p50=0.5, p95=0.9, p99=0.95, n_pixels=0)


def apply_normalisation(s_f32: np.ndarray, anchor: NormStats,
                          clip_lo: float = -3.0,
                          clip_hi: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (normed_clipped, tail_excess)."""
    normed = anchor.normed(s_f32)
    normed = np.clip(normed, clip_lo, clip_hi).astype(np.float32)
    tail = anchor.tail_excess(s_f32)
    return normed, tail


# ═════════════════════════════════════════════════════════════════════════════
# v7 CORE — connected-component features
# ═════════════════════════════════════════════════════════════════════════════
def cc_features_for_method(score_f32: np.ndarray,
                             threshold: float,
                             min_perim: float = 4.0
                             ) -> dict[str, np.ndarray]:
    """All CC features for one method's score map.

    Threshold at `threshold` (typically the (method, class) p99 from
    train_good anchors). For each labeled CC, compute area / max /
    mean / compactness / size_rank / centroid distance. Broadcast each
    statistic into a (H, W) map: every pixel inside CC k gets CC k's
    statistic; off-CC pixels get 0 (the trees treat them as 'no CC
    context' which is fine).

    Returns dict of named (H, W) float32 maps so the caller can plug
    feature_names directly.
    """
    H, W = score_f32.shape
    out = {
        "cc_logarea":     np.zeros((H, W), dtype=np.float32),
        "cc_compactness": np.zeros((H, W), dtype=np.float32),
        "cc_max_score":   np.zeros((H, W), dtype=np.float32),
        "cc_mean_score":  np.zeros((H, W), dtype=np.float32),
        "cc_n_in_image":  np.zeros((H, W), dtype=np.float32),
        "cc_size_rank":   np.zeros((H, W), dtype=np.float32),
        "cc_centroid_dist": np.zeros((H, W), dtype=np.float32),
    }
    hot = score_f32 >= threshold
    if not hot.any():
        return out
    labels, n = ndi.label(hot)
    if n == 0:
        return out

    sizes = np.bincount(labels.ravel())  # sizes[0] is background

    # Per-CC stats
    cc_ids = np.arange(1, n + 1)
    max_per = ndi.maximum(score_f32, labels=labels, index=cc_ids).astype(np.float32)
    mean_per = ndi.mean(score_f32, labels=labels, index=cc_ids).astype(np.float32)

    # Perimeter via eroded subtraction
    eroded = ndi.binary_erosion(hot, border_value=0)
    perim_hot = hot & (~eroded)
    perim_labels = labels.copy()
    perim_labels[~perim_hot] = 0
    perim_per = np.bincount(perim_labels.ravel(), minlength=n + 1)[1:].astype(np.float32)
    perim_per = np.maximum(perim_per, min_perim)

    areas = sizes[1:].astype(np.float32)
    compact_per = (4.0 * math.pi * areas) / (perim_per ** 2)

    # Size rank (1 = largest)
    size_order = np.argsort(-areas, kind="stable")  # indices into cc_ids
    rank_for_cc = np.empty(n, dtype=np.int32)
    for r_idx, cc_idx in enumerate(size_order):
        rank_for_cc[cc_idx] = r_idx + 1

    # Centroid distance via per-CC mass centre
    centroids = ndi.center_of_mass(hot, labels=labels, index=cc_ids)
    cy_per = np.array([c[0] for c in centroids], dtype=np.float32)
    cx_per = np.array([c[1] for c in centroids], dtype=np.float32)

    # Build per-pixel lookups via labels array
    log_area_lut = np.zeros(n + 1, dtype=np.float32)
    log_area_lut[1:] = np.log1p(areas)
    out["cc_logarea"] = log_area_lut[labels]

    max_lut = np.zeros(n + 1, dtype=np.float32)
    max_lut[1:] = max_per
    out["cc_max_score"] = max_lut[labels]

    mean_lut = np.zeros(n + 1, dtype=np.float32)
    mean_lut[1:] = mean_per
    out["cc_mean_score"] = mean_lut[labels]

    compact_lut = np.zeros(n + 1, dtype=np.float32)
    compact_lut[1:] = compact_per
    out["cc_compactness"] = compact_lut[labels]

    rank_lut = np.zeros(n + 1, dtype=np.float32)
    rank_lut[1:] = rank_for_cc.astype(np.float32)
    out["cc_size_rank"] = rank_lut[labels]

    out["cc_n_in_image"] = np.full((H, W), float(n), dtype=np.float32)
    out["cc_n_in_image"][labels == 0] = 0.0  # off-CC pixels: 0

    # Centroid distance per pixel (vectorised per CC)
    cy_lut = np.zeros(n + 1, dtype=np.float32)
    cx_lut = np.zeros(n + 1, dtype=np.float32)
    cy_lut[1:] = cy_per; cx_lut[1:] = cx_per
    cy_at = cy_lut[labels]; cx_at = cx_lut[labels]
    ys, xs = np.indices((H, W)).astype(np.float32)
    dist = np.sqrt((ys - cy_at) ** 2 + (xs - cx_at) ** 2).astype(np.float32)
    dist[labels == 0] = 0.0
    out["cc_centroid_dist"] = dist
    return out


# ═════════════════════════════════════════════════════════════════════════════
# v7 CORE — best-model-per-class identification (val pooled AP)
# ═════════════════════════════════════════════════════════════════════════════
def rank_methods_per_class(val: dict, n_methods: int) -> dict[str, list[int]]:
    """For each class, rank methods by pooled pixel AP on val. Used to:
       (a) pick top-K methods for CC features
       (b) emit a per-class best-model one-hot as a feature
    """
    from sklearn.metrics import average_precision_score
    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]
    N, H, W, M = val_scores.shape
    out: dict[str, list[int]] = {}
    print(f"\nRanking methods per class by val pooled pixel-AP...")
    for cls in sorted(set(val_classes.tolist())):
        idx = np.flatnonzero(val_classes == cls)
        if idx.size == 0:
            out[cls] = list(range(M))
            continue
        y = val_masks[idx].ravel()
        if int(y.sum()) == 0:
            out[cls] = list(range(M))
            continue
        aps: list[tuple[float, int]] = []
        for mi in range(M):
            s = val_scores[idx, :, :, mi].ravel()
            try:
                aps.append((float(average_precision_score(y, s)), mi))
            except Exception:
                aps.append((0.0, mi))
        aps.sort(key=lambda t: -t[0])
        out[cls] = [mi for _, mi in aps]
        top3 = ", ".join(f"m{mi}({ap:.3f})" for ap, mi in aps[:3])
        print(f"  {cls}: top-3 = {top3}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# v7 CORE — spatial priors (per-class + per-(class,type) + global)
# ═════════════════════════════════════════════════════════════════════════════
def load_spatial_priors_v7(prior_dir: Path,
                              classes: list[str]) -> dict:
    """Loads three families of priors from analysis_out/tables/:
       06_heat_global.npy             — cross-class mean heatmap
       06_heat_<class>.npy            — per-class mean heatmap
       06_heat_<class>_<anom>.npy     — per-(class, anomaly_type) heatmap
                                          (we MAX over types per class so
                                          inference doesn't depend on a
                                          unknown-at-test anomaly_type)

    Returns:
        {
          "global":  (Hp, Wp) or None,
          "per_class": {cls: (Hp, Wp)},
          "per_class_atype_max": {cls: (Hp, Wp)},  # max over types
        }
    """
    out: dict = {"global": None, "per_class": {}, "per_class_atype_max": {}}
    if not prior_dir.exists():
        print(f"  [warn] --prior-heatmaps-dir {prior_dir} not found; "
              f"spatial-prior features will be all-zeros.")
        zero = np.zeros((128, 128), dtype=np.float32)
        for cls in classes:
            out["per_class"][cls] = zero.copy()
            out["per_class_atype_max"][cls] = zero.copy()
        out["global"] = zero.copy()
        return out

    g_path = prior_dir / "06_heat_global.npy"
    if g_path.exists():
        out["global"] = np.clip(np.load(g_path).astype(np.float32),
                                  0.0, 1.0)
        print(f"    loaded global prior: shape {out['global'].shape}  "
              f"max={out['global'].max():.4f}")
    else:
        out["global"] = np.zeros((128, 128), dtype=np.float32)
        print(f"    [warn] no global prior file found at {g_path}")

    for cls in classes:
        p = prior_dir / f"06_heat_{cls}.npy"
        if p.exists():
            h = np.clip(np.load(p).astype(np.float32), 0.0, 1.0)
            out["per_class"][cls] = h
        else:
            print(f"    [warn] no per-class prior for {cls}")
            out["per_class"][cls] = np.zeros((128, 128), dtype=np.float32)

        # Find all anomaly-type heatmaps for this class.
        type_maps: list[np.ndarray] = []
        for q in prior_dir.glob(f"06_heat_{cls}_anomaly_*.npy"):
            try:
                m = np.clip(np.load(q).astype(np.float32), 0.0, 1.0)
                type_maps.append(m)
            except Exception:
                pass
        if type_maps:
            stack = np.stack(type_maps, axis=0)
            out["per_class_atype_max"][cls] = stack.max(axis=0).astype(np.float32)
            print(f"    {cls}: per-class prior loaded, "
                  f"max-over-{len(type_maps)}-types prior built")
        else:
            out["per_class_atype_max"][cls] = out["per_class"][cls].copy()
            print(f"    {cls}: no per-type priors, falling back to "
                  f"per-class for atype_max")

    return out


def _resize_prior_to(prior: np.ndarray, H: int, W: int) -> np.ndarray:
    if prior.shape == (H, W): return prior.astype(np.float32, copy=False)
    zy = H / prior.shape[0]; zx = W / prior.shape[1]
    out = ndi.zoom(prior, zoom=(zy, zx), order=1, mode="nearest")
    if out.shape != (H, W):
        h2 = min(out.shape[0], H); w2 = min(out.shape[1], W)
        clean = np.zeros((H, W), dtype=np.float32)
        clean[:h2, :w2] = out[:h2, :w2]; out = clean
    return out.astype(np.float32, copy=False)


# ═════════════════════════════════════════════════════════════════════════════
# Local-val alignment (reused from v6 with minor cleanup)
# ═════════════════════════════════════════════════════════════════════════════
def _nn_resize_2d(arr: np.ndarray, target_shape, dtype) -> np.ndarray:
    th, tw = target_shape; h, w = arr.shape
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)


def align_local_preds(preds_per_method, method_names):
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
    views = np.full(N, -1, dtype=np.int32)
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
                    if v is not None: views[i] = int(v)
    n_with_view = int((views >= 0).sum())
    print(f"  aligned {N} val images × {M} methods @ {H0}x{W0}"
          + (f" (with image_paths, {n_with_view}/{N} views parsed)"
             if image_paths_aligned is not None else " (no image_paths)"))
    return {"ids": np.asarray(common),
            "classes": classes.astype(str),
            "anomaly_types": anomaly_types.astype(str),
            "views": views, "scores": scores, "masks": masks,
            "image_paths": image_paths_aligned}


def save_aligned_val(cache: Cache, name: str, val: dict) -> None:
    if not cache.enabled: return
    paths = val.get("image_paths")
    cache.save_npz(
        name, ids=val["ids"], classes=val["classes"],
        anomaly_types=val["anomaly_types"], views=val["views"],
        scores=val["scores"], masks=val["masks"],
        image_paths=(np.array([], dtype=object) if paths is None else paths))


def load_aligned_val(cache: Cache, name: str) -> dict | None:
    d = cache.load_npz(name)
    if d is None: return None
    paths = d["image_paths"]
    if paths.size == 0: paths = None
    return {"ids": d["ids"].astype(str),
            "classes": d["classes"].astype(str),
            "anomaly_types": d["anomaly_types"].astype(str),
            "views": d["views"].astype(np.int32),
            "scores": d["scores"].astype(np.float32),
            "masks": d["masks"].astype(np.uint8),
            "image_paths": (None if paths is None else paths.astype(str))}


# ═════════════════════════════════════════════════════════════════════════════
# Multi-view grouping (val + test)
# ═════════════════════════════════════════════════════════════════════════════
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


def group_test_ids_by_sample(all_ids, class_map, default_class
                                ) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        sample_id, _v = parse_sample_id(sid)
        out[(str(cls), sample_id)].append(sid)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# v7 CORE — multi-view features (image-level + window-based)
# ═════════════════════════════════════════════════════════════════════════════
def compute_mv_image_level_for_group(
    group_scores_f32: np.ndarray,  # (V, M, H, W)
    anchors: AnchorTable, cls: str, views: list[int | None]
) -> dict[str, np.ndarray]:
    """Image-level multi-view features. Returns dict of (V, M) arrays
    where row v, col mi is the scalar feature for that (view, method).
    Caller broadcasts to (H, W) planes downstream.

    For each (view v, method mi):
      self_p99           p99 of this view's score map
      sib_max_p99        max p99 over OTHER views
      sib_mean_p99       mean p99 over OTHER views
      lonely_image       self_p99 - sib_max_p99
                            (large positive = this view is uniquely hot)
      sib_above_anchor   number of sibling views whose p99 exceeds the
                           per-(method, class, view) p99 anchor from
                           train_good (count: 0..V-1)
    """
    V, M, H, W = group_scores_f32.shape
    # Precompute per-(view, method) p99
    p99_vm = np.empty((V, M), dtype=np.float32)
    for k in range(V):
        for mi in range(M):
            p99_vm[k, mi] = float(np.percentile(group_scores_f32[k, mi], 99))

    self_p99       = p99_vm.copy()
    sib_max_p99    = np.zeros_like(p99_vm)
    sib_mean_p99   = np.zeros_like(p99_vm)
    lonely_image   = np.zeros_like(p99_vm)
    sib_above      = np.zeros_like(p99_vm)

    # Per-(view, method) anchor.p99 lookup (sibling threshold).
    anchor_p99_vm = np.empty((V, M), dtype=np.float32)
    for k in range(V):
        v_here = views[k]
        for mi in range(M):
            anchor_p99_vm[k, mi] = get_anchor(anchors, mi, cls, v_here).p99

    if V < 2:
        return {"self_p99": self_p99, "sib_max_p99": sib_max_p99,
                "sib_mean_p99": sib_mean_p99,
                "lonely_image": lonely_image,
                "sib_above_anchor": sib_above}

    for k in range(V):
        sib_idx = [j for j in range(V) if j != k]
        sib_block = p99_vm[sib_idx]                    # (V-1, M)
        sib_max_p99[k]  = sib_block.max(axis=0)
        sib_mean_p99[k] = sib_block.mean(axis=0)
        lonely_image[k] = self_p99[k] - sib_max_p99[k]
        # Count sibling views where p99 > anchor.p99 of SIBLING view's anchor
        sib_anchor_block = anchor_p99_vm[sib_idx]      # (V-1, M)
        sib_above[k] = (sib_block > sib_anchor_block).sum(axis=0).astype(np.float32)

    return {"self_p99": self_p99, "sib_max_p99": sib_max_p99,
            "sib_mean_p99": sib_mean_p99,
            "lonely_image": lonely_image,
            "sib_above_anchor": sib_above}


def compute_mv_window_max_for_group(
    group_scores_f32: np.ndarray,  # (V, M, H, W)
    window: int = 7
) -> np.ndarray:
    """Per-pixel: max sibling-view score within a window×window box
    around the same (y, x). This tolerates the low pixel-IoU between
    views (analysis §10 p50=0.35) — we don't need pixel-perfect
    alignment, just "somewhere nearby in another view also lit up."

    Returns (V, M, H, W) float32. View v's map is max over OTHER views'
    windowed maxima.
    """
    V, M, H, W = group_scores_f32.shape
    out = np.zeros_like(group_scores_f32, dtype=np.float32)
    if V < 2: return out
    # Pre-compute windowed-max for each (view, method).
    win_max = np.empty_like(group_scores_f32, dtype=np.float32)
    for k in range(V):
        for mi in range(M):
            win_max[k, mi] = ndi.maximum_filter(
                group_scores_f32[k, mi], size=window, mode="reflect"
            ).astype(np.float32)
    # For view v, the sibling window-max = max over OTHER views.
    for k in range(V):
        sib_idx = [j for j in range(V) if j != k]
        out[k] = win_max[sib_idx].max(axis=0)
    return out


# Reused on val side to build the same image-level mv features for
# every val image. Slightly different signature: we collect groups
# from val_dict and build (N, M)-shaped per-image scalars.
def compute_mv_image_level_val(
    val: dict, anchors: AnchorTable
) -> dict[str, np.ndarray]:
    """Returns 5 (N, M) arrays: self_p99, sib_max_p99, sib_mean_p99,
    lonely_image, sib_above_anchor. Broadcast as planes downstream."""
    val_scores = val["scores"]; val_classes = val["classes"]
    val_views  = val.get("views", np.full(len(val_classes), -1, dtype=np.int32))
    N, H, W, M = val_scores.shape
    out: dict[str, np.ndarray] = {
        "self_p99":         np.zeros((N, M), dtype=np.float32),
        "sib_max_p99":      np.zeros((N, M), dtype=np.float32),
        "sib_mean_p99":     np.zeros((N, M), dtype=np.float32),
        "lonely_image":     np.zeros((N, M), dtype=np.float32),
        "sib_above_anchor": np.zeros((N, M), dtype=np.float32),
    }
    groups = group_val_by_sample(val)
    n_multi = 0
    for (cls, _sid), idxs in groups.items():
        V = len(idxs)
        # Build the (V, M, H, W) block.
        block = np.stack(
            [val_scores[i].transpose(2, 0, 1) for i in idxs], axis=0
        ).astype(np.float32)
        views_in_grp = [int(val_views[i]) if val_views[i] >= 0 else None
                          for i in idxs]
        feats = compute_mv_image_level_for_group(block, anchors, str(cls),
                                                       views_in_grp)
        if V >= 2: n_multi += 1
        for k, i in enumerate(idxs):
            for key in out:
                out[key][i] = feats[key][k]
        del block
    print(f"  image-level MV: {n_multi} multi-view groups out of "
          f"{len(groups)}")
    return out


def compute_mv_window_max_val(val: dict, window: int = 7) -> np.ndarray:
    """Returns (N, H, W, M) float32 — view v's sibling-window-max for
    each pixel. Caller selects per-image, per-method slices for
    featurise."""
    val_scores = val["scores"]
    val_classes = val["classes"]
    N, H, W, M = val_scores.shape
    out = np.zeros((N, H, W, M), dtype=np.float32)
    groups = group_val_by_sample(val)
    for (_cls, _sid), idxs in groups.items():
        V = len(idxs)
        if V < 2: continue
        block = np.stack(
            [val_scores[i].transpose(2, 0, 1) for i in idxs], axis=0
        ).astype(np.float32)
        winmax = compute_mv_window_max_for_group(block, window=window)
        # winmax: (V, M, H, W); store back per-image as (H, W, M)
        for k, i in enumerate(idxs):
            out[i] = winmax[k].transpose(1, 2, 0)
        del block, winmax
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Feature configuration
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FeatureConfig:
    # Per-method features
    use_raw_score:        bool = True   # m{mi}_raw            (after norm-clip)
    use_per_image_rank:   bool = True   # m{mi}_rank
    use_normed:           bool = True   # m{mi}_normed (train_good anchored)
    use_tail_excess:      bool = True   # m{mi}_tail_excess
    use_gauss_smooth:     bool = True   # m{mi}_gsmooth (sigma=3)
    gauss_sigma:          float = 3.0

    # Cross-method
    use_cross_consensus:  bool = True   # n_above_p95, mean/max/min-top-3
    use_cross_dispersion: bool = True   # CV + range

    # Multi-view (image-level)
    use_mv_image_level:   bool = True   # 5*M scalar features broadcast

    # Multi-view (patch-window)
    use_mv_window_max:    bool = True   # M maps (sibling window-max)
    mv_window:            int  = 7

    # CC features (top-K methods per class)
    use_cc_features:      bool = True
    n_top_methods_cc:     int  = 5      # K
    cc_threshold_pct:     float = 99.0  # which anchor percentile = threshold

    # Priors
    use_spatial_prior:    bool = True   # s_prior_class, atype_max, global
    use_spatial_geom:     bool = True   # x, y, dist_edge, dist_center

    # Categorical
    use_class_onehot:     bool = True
    use_view_onehot:      bool = True
    n_views_onehot:       int  = 5      # views indexed 1..5

    # Class-wise best-method indicator
    use_best_method_rank: bool = True   # for each method mi, its rank
                                            # within this class's ordering


FEATURE_FAMILIES = [
    "per_method", "cross_consensus", "cross_dispersion",
    "mv_image_level", "mv_window_max", "cc_features",
    "spatial_prior", "spatial_geom",
    "class_onehot", "view_onehot", "best_method_rank",
]


# ═════════════════════════════════════════════════════════════════════════════
# Featurise (CORE)
# ═════════════════════════════════════════════════════════════════════════════
def _per_image_rank(s: np.ndarray) -> np.ndarray:
    flat = s.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty_like(flat, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks.reshape(s.shape)


def _gauss(s, sigma):
    return ndi.gaussian_filter(s, sigma=sigma, mode="reflect").astype(np.float32)


def _spatial_cache(H: int, W: int) -> dict:
    ys, xs = np.indices((H, W)).astype(np.float32)
    xs_n = xs / max(W - 1, 1); ys_n = ys / max(H - 1, 1)
    de = np.minimum(np.minimum(xs_n, ys_n),
                     np.minimum(1.0 - xs_n, 1.0 - ys_n)).astype(np.float32)
    dc = np.sqrt((xs_n - 0.5) ** 2 + (ys_n - 0.5) ** 2).astype(np.float32)
    return {"x": xs_n, "y": ys_n, "dist_edge": de, "dist_center": dc,
            "_shape": (H, W), "_prior_resized": {}}


def _onehot_const_planes(value: int, n: int, H: int, W: int):
    out = [np.zeros((H, W), dtype=np.float32) for _ in range(n)]
    if 0 <= value < n: out[value][:] = 1.0
    return out


def featurize_image_v7(
    scores_per_method: list[np.ndarray],   # any storage dtype, length M
    cfg: FeatureConfig,
    *,
    class_id: str,
    all_classes: list[str],
    view: int | None,
    anchors: AnchorTable,
    spatial_cache: dict | None = None,
    feature_names: list[str] | None = None,
    # Multi-view inputs (None → multiview features become 0)
    mv_image_level: dict[str, np.ndarray] | None = None,  # {key: (M,)}
    mv_window_max:  list[np.ndarray] | None = None,        # list of (H, W) per method
    # Priors and rankings
    prior_class: np.ndarray | None = None,
    prior_atype_max: np.ndarray | None = None,
    prior_global: np.ndarray | None = None,
    best_method_rank: list[int] | None = None,  # ordered list of method indices
    # CC top-K methods for THIS class (in order best-to-worst)
    top_methods_for_cls: list[int] | None = None,
):
    """Build the per-pixel feature stack for ONE image.

    All scores normalised via the per-(method, class, view) anchor table;
    everything downstream operates on normalised values. CC features
    also use anchored thresholds.
    """
    assert scores_per_method, "need at least one method"
    scores_f32 = [to_f32(s) for s in scores_per_method]
    H, W = scores_f32[0].shape
    M = len(scores_f32)
    layers: list[np.ndarray] = []
    names: list[str] = []

    # Pre-compute normed + tail_excess once per method (used many times).
    normed_per_m: list[np.ndarray] = []
    tail_per_m:   list[np.ndarray] = []
    anchors_per_m: list[NormStats] = []
    for mi, s in enumerate(scores_f32):
        anchor = get_anchor(anchors, mi, class_id, view)
        anchors_per_m.append(anchor)
        n, t = apply_normalisation(s, anchor)
        normed_per_m.append(n); tail_per_m.append(t)

    # ── Per-method features ────────────────────────────────────────────────
    for mi, s in enumerate(scores_f32):
        if s.shape != (H, W):
            raise ValueError(f"method {mi} shape {s.shape} != ({H}, {W})")
        if cfg.use_raw_score:
            layers.append(s); names.append(f"m{mi}_raw")
        if cfg.use_per_image_rank:
            layers.append(_per_image_rank(s)); names.append(f"m{mi}_rank")
        if cfg.use_normed:
            layers.append(normed_per_m[mi]); names.append(f"m{mi}_normed")
        if cfg.use_tail_excess:
            layers.append(tail_per_m[mi]); names.append(f"m{mi}_tail")
        if cfg.use_gauss_smooth:
            g = _gauss(normed_per_m[mi], cfg.gauss_sigma)
            layers.append(g); names.append(f"m{mi}_gnormed")

    # ── Cross-method consensus (operates on normed scores) ─────────────────
    if cfg.use_cross_consensus and M >= 2:
        normed_stack = np.stack(normed_per_m, axis=0).astype(np.float32)
        # Count of methods exceeding their p95 threshold; p95 normed
        # value = (p95 - p50)/(p99 - p50). Anchor stores p50/p95/p99,
        # so the threshold in normed space is (p95 - p50)/(p99 - p50).
        # We compute per-method thresholds and broadcast-compare.
        thr_p95_normed = np.array(
            [(a.p95 - a.p50) / max(a.p99 - a.p50, 1e-6) for a in anchors_per_m],
            dtype=np.float32
        ).reshape(M, 1, 1)
        n_above = (normed_stack > thr_p95_normed).sum(axis=0).astype(np.float32)
        layers.append(n_above); names.append("xc_n_above_p95")
        layers.append(normed_stack.mean(axis=0).astype(np.float32))
        names.append("xc_mean_normed")
        layers.append(normed_stack.max(axis=0).astype(np.float32))
        names.append("xc_max_normed")
        # min-top-3 normed
        k_top = min(3, M)
        # -partition gives top-k descending; min over those = floor of agreement
        top_k = -np.partition(-normed_stack, k_top - 1, axis=0)[:k_top]
        layers.append(top_k.min(axis=0).astype(np.float32))
        names.append("xc_min_top3_normed")

    # ── Cross-method dispersion ────────────────────────────────────────────
    if cfg.use_cross_dispersion and M >= 2:
        normed_stack = np.stack(normed_per_m, axis=0).astype(np.float32)
        mean_ = normed_stack.mean(axis=0)
        std_  = normed_stack.std(axis=0)
        cv = std_ / (np.abs(mean_) + 1e-6)
        layers.append(cv.astype(np.float32)); names.append("xc_cv_normed")
        layers.append((normed_stack.max(axis=0) - normed_stack.min(axis=0))
                        .astype(np.float32))
        names.append("xc_range_normed")

    # ── Multi-view (image-level, broadcast scalars to H×W) ─────────────────
    if cfg.use_mv_image_level and mv_image_level is not None:
        for key in ("self_p99", "sib_max_p99", "sib_mean_p99",
                      "lonely_image", "sib_above_anchor"):
            arr_M = mv_image_level[key]   # shape (M,)
            for mi in range(M):
                v_val = float(arr_M[mi])
                plane = np.full((H, W), v_val, dtype=np.float32)
                layers.append(plane)
                names.append(f"m{mi}_mv_{key}")
    elif cfg.use_mv_image_level:
        # No MV info available (single-view sample): emit zero planes
        for key in ("self_p99", "sib_max_p99", "sib_mean_p99",
                      "lonely_image", "sib_above_anchor"):
            for mi in range(M):
                layers.append(np.zeros((H, W), dtype=np.float32))
                names.append(f"m{mi}_mv_{key}")

    # ── Multi-view window-max (per method) ─────────────────────────────────
    if cfg.use_mv_window_max:
        if mv_window_max is not None:
            for mi in range(M):
                a = np.asarray(mv_window_max[mi], dtype=np.float32)
                if a.shape != (H, W):
                    raise ValueError(
                        f"mv_window_max[{mi}].shape={a.shape} != ({H}, {W})")
                layers.append(a); names.append(f"m{mi}_mv_winmax")
        else:
            for mi in range(M):
                layers.append(np.zeros((H, W), dtype=np.float32))
                names.append(f"m{mi}_mv_winmax")

    # ── CC features for top-K methods of this class ────────────────────────
    if cfg.use_cc_features:
        K = cfg.n_top_methods_cc
        top_list = (top_methods_for_cls or list(range(M)))[:K]
        for rank_idx in range(K):
            if rank_idx >= len(top_list):
                # No method available — emit zero planes
                for suf in ("logarea", "compactness", "max_score",
                              "mean_score", "n_in_image", "size_rank",
                              "centroid_dist"):
                    layers.append(np.zeros((H, W), dtype=np.float32))
                    names.append(f"top{rank_idx}_cc_{suf}")
                continue
            mi = top_list[rank_idx]
            anchor = anchors_per_m[mi]
            thr = anchor.p99   # threshold at p99 of train_good
            cc = cc_features_for_method(scores_f32[mi], threshold=thr)
            for suf in ("logarea", "compactness", "max_score",
                          "mean_score", "n_in_image", "size_rank",
                          "centroid_dist"):
                layers.append(cc[f"cc_{suf}"])
                names.append(f"top{rank_idx}_cc_{suf}")

    # ── Spatial geometry ────────────────────────────────────────────────────
    if cfg.use_spatial_geom:
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        for k in ("x", "y", "dist_edge", "dist_center"):
            layers.append(spatial_cache[k]); names.append(f"s_{k}")

    # ── Spatial priors ──────────────────────────────────────────────────────
    if cfg.use_spatial_prior:
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        cache_p = spatial_cache.setdefault("_prior_resized", {})
        # per-class
        if prior_class is not None:
            key = ("class", class_id, H, W)
            if key not in cache_p:
                cache_p[key] = _resize_prior_to(prior_class, H, W)
            layers.append(cache_p[key]); names.append("s_prior_class")
        else:
            layers.append(np.zeros((H, W), dtype=np.float32))
            names.append("s_prior_class")
        # max-over-anomaly-types
        if prior_atype_max is not None:
            key = ("atype", class_id, H, W)
            if key not in cache_p:
                cache_p[key] = _resize_prior_to(prior_atype_max, H, W)
            layers.append(cache_p[key]); names.append("s_prior_atype_max")
        else:
            layers.append(np.zeros((H, W), dtype=np.float32))
            names.append("s_prior_atype_max")
        # global
        if prior_global is not None:
            key = ("global", H, W)
            if key not in cache_p:
                cache_p[key] = _resize_prior_to(prior_global, H, W)
            layers.append(cache_p[key]); names.append("s_prior_global")
        else:
            layers.append(np.zeros((H, W), dtype=np.float32))
            names.append("s_prior_global")

    # ── Class one-hot ──────────────────────────────────────────────────────
    if cfg.use_class_onehot and all_classes:
        try: idx_cls = all_classes.index(class_id)
        except ValueError: idx_cls = -1
        planes = _onehot_const_planes(idx_cls, len(all_classes), H, W)
        for cls_name, plane in zip(all_classes, planes):
            layers.append(plane); names.append(f"c_{cls_name}")

    # ── View one-hot ───────────────────────────────────────────────────────
    if cfg.use_view_onehot:
        n_v = cfg.n_views_onehot
        if view is not None and 1 <= int(view) <= n_v:
            v_idx = int(view) - 1
        else:
            v_idx = n_v   # unknown bucket
        n_slots = n_v + 1
        planes = _onehot_const_planes(v_idx, n_slots, H, W)
        for k, plane in enumerate(planes):
            tag = (f"view_{k + 1}" if k < n_v else "view_unknown")
            layers.append(plane); names.append(f"v_{tag}")

    # ── Best-method-rank: for each method mi, its rank in this class.
    # Lower rank = better. Encoded as a constant scalar.
    if cfg.use_best_method_rank and best_method_rank is not None:
        # rank_lut[mi] = position (0-indexed) of mi in best_method_rank
        rank_lut = {mi: r for r, mi in enumerate(best_method_rank)}
        for mi in range(M):
            r = rank_lut.get(mi, M - 1)
            layers.append(np.full((H, W), float(r), dtype=np.float32))
            names.append(f"m{mi}_class_rank")

    feats = np.stack(layers, axis=-1).astype(np.float32, copy=False)
    if feature_names is not None and names != feature_names:
        missing = [n for n in feature_names if n not in names]
        extra   = [n for n in names if n not in feature_names]
        raise RuntimeError(
            f"feature drift between fit and predict: "
            f"got {len(names)} cols, expected {len(feature_names)}; "
            f"missing={missing[:5]}; extra={extra[:5]}")
    return feats, names


# ═════════════════════════════════════════════════════════════════════════════
# Build per-class training matrices
# ═════════════════════════════════════════════════════════════════════════════
def build_training_data(
    val: dict, classes: list[str], cfg: FeatureConfig,
    *, neg_per_pos: int, seed: int,
    anchors: AnchorTable,
    mv_image_level_val: dict[str, np.ndarray] | None,
    mv_window_max_val:  np.ndarray | None,
    priors: dict | None,
    top_methods_per_class: dict[str, list[int]] | None,
    all_classes_onehot: list[str] | None,
) -> dict:
    """Featurize every val image and prepare per-class training matrices.

    Per-class layout:
        td["X_train"], td["y_train"]   — class-balanced sample (neg_per_pos)
        td["X_full"],  td["y_full"]    — every pixel of every val image
        td["img_pixranges"]            — [(start, end), ...] in X_full
        td["anomaly_types"]            — per-image anomaly type strings
    """
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
            print(f"  [warn] class {cls}: no val images; skipping"); continue

        prior_cls       = priors["per_class"].get(cls) if priors else None
        prior_atype_max = priors["per_class_atype_max"].get(cls) if priors else None
        prior_global    = priors.get("global") if priors else None
        top_methods_cls = (top_methods_per_class.get(cls)
                              if top_methods_per_class else None)
        best_rank_cls   = top_methods_cls  # same ordering

        per_img_X, per_img_y, per_img_atypes = [], [], []
        per_img_pixrange: list[tuple[int, int]] = []
        cursor = 0
        for i in cls_idx:
            mats = [val_scores[i, :, :, mi] for mi in range(M)]
            v_int: int | None = (int(val_views[i])
                                    if val_views[i] >= 0 else None)
            # Image-level MV slice for this image: dict[key] -> (M,)
            if mv_image_level_val is not None:
                mvil_one = {key: mv_image_level_val[key][i]
                              for key in mv_image_level_val}
            else:
                mvil_one = None
            # Window-max slice for this image: list-of-(H, W) per method
            if mv_window_max_val is not None:
                mvwm_one = [mv_window_max_val[i, :, :, mi] for mi in range(M)]
            else:
                mvwm_one = None
            feats, names = featurize_image_v7(
                mats, cfg, class_id=cls, all_classes=all_classes_onehot or [],
                view=v_int, anchors=anchors,
                spatial_cache=spatial,
                mv_image_level=mvil_one,
                mv_window_max=mvwm_one,
                prior_class=prior_cls,
                prior_atype_max=prior_atype_max,
                prior_global=prior_global,
                best_method_rank=best_rank_cls,
                top_methods_for_cls=top_methods_cls)
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
        mem_print(f"  built training data for {cls}")
    out["_feature_names"] = feature_names_global or []
    return out


def free_class_training_data(training_data: dict, cls: str,
                                 keep_xtrain: bool = False) -> None:
    if cls not in training_data: return
    td = training_data[cls]
    if isinstance(td, dict):
        td.pop("X_full", None); td.pop("y_full", None)
        td.pop("img_pixranges", None); td.pop("anomaly_types", None)
        if not keep_xtrain:
            td.pop("X_train", None); td.pop("y_train", None)


# ═════════════════════════════════════════════════════════════════════════════
# XGB params + CV
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.03,
    "subsample":        0.6,
    "colsample_bytree": 0.6,
    "reg_alpha":        0.5,
    "reg_lambda":       1.0,
    "min_child_weight": 20.0,
    "gamma":            0.5,
    "tree_method":      "hist",
    "max_bin":          256,
    "eval_metric":      "aucpr",
    "n_jobs":           -1,
    "verbosity":        0,
}


def _make_xgb(params: dict, seed: int):
    if not HAS_XGB:
        raise SystemExit("xgboost not installed. uv pip install xgboost")
    p = dict(DEFAULT_XGB_PARAMS); p.update(params or {})
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
    for i in range(n_imgs): buckets[key(i)].append(i)
    return buckets


def _pixel_ap_pooled(preds: np.ndarray, labels: np.ndarray) -> float:
    if int(labels.sum()) == 0: return float("nan")
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
            s, e = ranges[ti]; yp = y_full[s:e]
            pos = np.flatnonzero(yp == 1)
            if pos.size == 0: continue
            neg = np.flatnonzero(yp == 0)
            n_keep = min(neg.size, pos.size * neg_per_pos)
            sneg = (rng.choice(neg, n_keep, replace=False)
                    if n_keep < neg.size else neg)
            keep = np.concatenate([pos, sneg])
            Xs.append(X_full[s:e][keep]); ys.append(yp[keep])
        if not Xs: continue
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
        held_preds = []; held_labels = []
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            held_preds.append(p); held_labels.append(y_full[s:e].astype(np.int32))
        held_preds_cat = np.concatenate(held_preds)
        held_labels_cat = np.concatenate(held_labels)
        if int(held_labels_cat.sum()) > 0:
            fold_aps.append(_pixel_ap_pooled(held_preds_cat, held_labels_cat))
    valid = [a for a in fold_aps if not math.isnan(a)]
    return float(np.mean(valid)) if valid else 0.0


def loao_oof_one_class(td: dict, params: dict, seed: int,
                        cv_mode: str = "loao",
                        neg_per_pos: int = 30):
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
            s, e = ranges[ti]; yp = y_full[s:e]
            pos = np.flatnonzero(yp == 1)
            if pos.size == 0: continue
            neg = np.flatnonzero(yp == 0)
            n_keep = min(neg.size, pos.size * neg_per_pos)
            sneg = (rng.choice(neg, n_keep, replace=False)
                    if n_keep < neg.size else neg)
            keep = np.concatenate([pos, sneg])
            Xs.append(X_full[s:e][keep]); ys.append(yp[keep])
        if not Xs: continue
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            oof_preds[s:e] = p
    mask = ~np.isnan(oof_preds)
    return oof_preds[mask].astype(np.float32), y_full[mask].astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# Optuna tuning
# ═════════════════════════════════════════════════════════════════════════════
def _optuna_suggest(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 600, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 5),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "reg_alpha":        trial.suggest_float("reg_alpha",  1e-3, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 100.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
    }


def tune_global(training_data, *, n_trials, seed, cv_mode="loao",
                timeout=None, neg_per_pos=30):
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")
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


def tune_per_class(training_data, *, n_trials, seed, cv_mode="loao",
                    timeout_per_class=None, neg_per_pos=30):
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")
    out: dict = {}
    for cls, td in training_data.items():
        if cls.startswith("_"): continue
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
        out[cls] = {"best_params": study.best_params,
                    "best_cv_ap": float(study.best_value),
                    "n_trials": len(study.trials),
                    "oof_preds": oof_preds, "oof_labels": oof_labels}
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Calibration (Platt / isotonic)
# ═════════════════════════════════════════════════════════════════════════════
def fit_calibrator(method, oof_preds, oof_labels) -> dict:
    if method == "none": return {"method": "none"}
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
            oof_preds = oof_preds[sel]; oof_labels = oof_labels[sel]
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(oof_preds, oof_labels.astype(np.float32))
        return {"method": "isotonic",
                "X_thresholds": [float(v) for v in ir.X_thresholds_],
                "y_thresholds": [float(v) for v in ir.y_thresholds_]}
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(cal: dict, scores: np.ndarray) -> np.ndarray:
    m = cal.get("method", "none")
    if m == "none": return scores.astype(np.float32)
    if m == "platt":
        z = cal["coef"] * scores + cal["intercept"]
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    if m == "isotonic":
        xs = np.asarray(cal["X_thresholds"], dtype=np.float32)
        ys = np.asarray(cal["y_thresholds"], dtype=np.float32)
        return np.interp(scores, xs, ys).astype(np.float32)
    raise ValueError(m)


# ═════════════════════════════════════════════════════════════════════════════
# Fit per-class production models
# ═════════════════════════════════════════════════════════════════════════════
def fit_per_class(training_data, params_global, params_per_class, seed,
                   min_pos_for_per_class=200) -> dict:
    from sklearn.metrics import log_loss, average_precision_score
    feature_names = training_data.get("_feature_names", [])
    out: dict = {"_feature_names": feature_names}
    shared_X, shared_y = [], []
    for cls, td in training_data.items():
        if cls.startswith("_"): continue
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
                    "logloss": ll, "train_ap": ap, "params": params,
                    "feature_importance": clf.feature_importances_.tolist()}
        print(f"  class {cls}: n_pos={td['n_pos']:>6d}  "
              f"n_neg={td['n_neg_sampled']:>8d}  "
              f"train_logloss={ll:.4f}  train_ap={ap:.3f}")
    if shared_X:
        X_all = np.concatenate(shared_X, axis=0)
        y_all = np.concatenate(shared_y, axis=0)
        del shared_X, shared_y; gc.collect()
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
        del X_all, y_all; gc.collect()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Verification (per-class + pooled LB-proxy)
# ═════════════════════════════════════════════════════════════════════════════
def compute_pooled_lb_proxy_ap(val, method_names, oof_per_class):
    from sklearn.metrics import average_precision_score
    val_scores = val["scores"]; val_masks = val["masks"]
    N, H, W, M = val_scores.shape
    all_p = np.concatenate([op for (op, _) in oof_per_class.values()])
    all_l = np.concatenate([ol for (_, ol) in oof_per_class.values()])
    if int(all_l.sum()) == 0:
        stacker_pooled = float("nan")
    else:
        stacker_pooled = float(average_precision_score(all_l, all_p))
    singles_pooled: list[float] = []
    y_all_pix = val_masks.ravel()
    for mi in range(M):
        s_all = val_scores[:, :, :, mi].ravel()
        if int(y_all_pix.sum()) == 0:
            singles_pooled.append(float("nan"))
        else:
            try:
                singles_pooled.append(
                    float(average_precision_score(y_all_pix, s_all)))
            except Exception:
                singles_pooled.append(float("nan"))
    best_idx = int(np.nanargmax(singles_pooled)) if any(
        not math.isnan(x) for x in singles_pooled) else 0
    best_single_pooled = (method_names[best_idx], singles_pooled[best_idx])
    delta_pooled = (stacker_pooled - best_single_pooled[1]
                     if not (math.isnan(stacker_pooled)
                             or math.isnan(best_single_pooled[1]))
                     else float("nan"))
    return {"stacker_pooled_ap": stacker_pooled,
            "singles_pooled_ap": singles_pooled,
            "best_single_pooled": best_single_pooled,
            "delta_pooled": delta_pooled}


def verify_stacker_vs_singles(val, method_names, oof_per_class):
    from sklearn.metrics import average_precision_score
    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]
    N, H, W, M = val_scores.shape
    print(f"\n  Computing per-class pooled pixel-AP for {M} single models...")
    single_ap: dict[str, dict[str, float]] = {}
    for mi, mname in enumerate(method_names):
        single_ap[mname] = {}
        for cls in sorted(set(val_classes.tolist())):
            idx = np.flatnonzero(val_classes == cls)
            if idx.size == 0: continue
            s = val_scores[idx, :, :, mi].ravel()
            y = val_masks[idx].ravel()
            if int(y.sum()) == 0: ap = float("nan")
            else:
                try: ap = float(average_precision_score(y, s))
                except Exception: ap = float("nan")
            single_ap[mname][cls] = ap
    stacker_ap: dict[str, float] = {}
    for cls, (op, ol) in oof_per_class.items():
        if int(ol.sum()) == 0:
            stacker_ap[cls] = float("nan"); continue
        try: stacker_ap[cls] = float(average_precision_score(ol, op))
        except Exception: stacker_ap[cls] = float("nan")
    all_classes_sorted = sorted(set(val_classes.tolist()))
    hr("VERIFICATION — per-class POOLED pixel-AP (stacker vs singles)", "=")
    hdr = f"  {'class':<10}"
    for n in method_names: hdr += f" {n[:12]:>13}"
    hdr += f" {'STACKER':>13} {'best_single':>13} {'win?':>7}"
    print(hdr)
    n_class_wins = 0; n_class_total = 0; deltas: list[float] = []
    for cls in all_classes_sorted:
        line = f"  {cls:<10}"
        bests = []
        for n in method_names:
            ap = single_ap[n].get(cls, float("nan"))
            line += f" {ap:>13.4f}"
            if not math.isnan(ap): bests.append((ap, n))
        st = stacker_ap.get(cls, float("nan"))
        best_ap, _best_n = (max(bests) if bests else (float("nan"), "?"))
        line += f" {st:>13.4f}"; line += f" {best_ap:>13.4f}"
        if not (math.isnan(st) or math.isnan(best_ap)):
            n_class_total += 1; delta = st - best_ap; deltas.append(delta)
            line += f" {'WIN' if delta > 0 else 'lose':>7}"
            if delta > 0: n_class_wins += 1
        else:
            line += f" {'?':>7}"
        print(line)

    def _mean_skipnan(d):
        vals = [v for v in d.values() if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")
    overall_singles = {n: _mean_skipnan(single_ap[n]) for n in method_names}
    overall_stacker = _mean_skipnan(stacker_ap)
    best_single_name = max(overall_singles, key=lambda n: overall_singles[n])
    best_single_ap   = overall_singles[best_single_name]
    print()
    print(f"  mean pooled-AP across classes (per single model):")
    for n in sorted(method_names, key=lambda nn: -overall_singles[nn]):
        marker = "  <- best single" if n == best_single_name else ""
        print(f"    {n:<24} {overall_singles[n]:.4f}{marker}")
    print(f"  STACKER mean pooled-AP:  {overall_stacker:.4f}")
    delta_overall = overall_stacker - best_single_ap
    if not math.isnan(delta_overall):
        prefix = "BEATS" if delta_overall > 0 else "LOSES to"
        print(f"  >>> STACKER {prefix} best single ({best_single_name}) "
              f"by {delta_overall:+.4f} mean pooled-AP.")
    if n_class_total > 0:
        print(f"  >>> per-class: stacker wins {n_class_wins}/{n_class_total} "
              f"classes  (mean delta = {np.mean(deltas):+.4f})")

    hr("LB-PROXY POOLED PIXEL-AP (mirrors public leaderboard exactly)", "=")
    pooled = compute_pooled_lb_proxy_ap(val, method_names, oof_per_class)
    print(f"  Single-model POOLED pixel-AP (all pixels concatenated):")
    sorted_singles = sorted(
        zip(method_names, pooled["singles_pooled_ap"]),
        key=lambda kv: -kv[1] if not math.isnan(kv[1]) else 1)
    for n, ap in sorted_singles:
        marker = "  <- best single (pooled)" if n == pooled["best_single_pooled"][0] else ""
        print(f"    {n:<24} {ap:.4f}{marker}")
    print(f"")
    print(f"  STACKER pooled pixel-AP:     {pooled['stacker_pooled_ap']:.4f}")
    print(f"  Best-single pooled pixel-AP: {pooled['best_single_pooled'][1]:.4f}  "
          f"({pooled['best_single_pooled'][0]})")
    if not math.isnan(pooled["delta_pooled"]):
        prefix = "BEATS" if pooled["delta_pooled"] > 0 else "LOSES to"
        print(f"  >>> STACKER {prefix} best single (pooled) by "
              f"{pooled['delta_pooled']:+.4f}.  ← this tracks the LB.")
    print(f"\n  ⚠ Use the POOLED number above as your offline LB proxy.")
    return {"per_class_single": single_ap, "per_class_stacker": stacker_ap,
            "overall_singles": overall_singles,
            "overall_stacker": overall_stacker,
            "best_single_name": best_single_name,
            "best_single_overall_ap": best_single_ap,
            "delta_overall": delta_overall,
            "class_wins": n_class_wins, "class_total": n_class_total,
            "pooled_stacker_ap": pooled["stacker_pooled_ap"],
            "pooled_singles_ap": dict(zip(method_names, pooled["singles_pooled_ap"])),
            "pooled_best_single": pooled["best_single_pooled"],
            "pooled_delta": pooled["delta_pooled"]}


# ═════════════════════════════════════════════════════════════════════════════
# Test-time inference (streaming, on-the-fly multi-view)
# ═════════════════════════════════════════════════════════════════════════════
def decode_submissions_to_uint8(submissions, all_ids):
    M = len(submissions); N = len(all_ids)
    decoded: list[dict[str, np.ndarray]] = []
    t0 = time.time()
    for mi, sub in enumerate(submissions):
        d: dict[str, np.ndarray] = {}
        for j, sid in enumerate(all_ids):
            d[sid] = q8rle_to_uint8_matrix(sub[sid])
            if (j + 1) % 1000 == 0:
                print(f"    method {mi + 1}/{M}: {j + 1}/{N} "
                      f"({time.time() - t0:.1f}s)", flush=True)
        decoded.append(d)
        print(f"    method {mi + 1}/{M} decoded ({time.time() - t0:.1f}s)")
    return decoded


def cache_decoded_test(cache: Cache, name: str, decoded_per_method) -> None:
    if not cache.enabled: return
    cache.save_pkl(name, {"_n_methods": len(decoded_per_method),
                            "_per_method": decoded_per_method})


def load_decoded_test(cache: Cache, name: str):
    obj = cache.load_pkl(name)
    if obj is None: return None
    return obj["_per_method"]


def fuse_test(submissions, models, class_map, cfg: FeatureConfig,
               *, default_class: str,
               calibrators_per_class, anchors: AnchorTable,
               priors: dict | None,
               top_methods_per_class: dict[str, list[int]],
               all_classes_onehot: list[str] | None,
               cache: Cache | None = None, cache_key_base: str = "",
               drop_decoded_during_fusion: bool = True) -> dict[str, str]:
    """Streaming, group-by-sample fusion.

    For each sample group (class, sample_id):
      1. Build (V, M, H, W) float32 block from decoded uint8 maps.
      2. Compute on-the-fly:
           - image-level MV features  (V, M) scalars × 5 keys
           - window-max MV features   (V, M, H, W)
      3. For each (view k, sample) call featurize_image_v7 and predict.
      4. Discard the group's decoded entries (if drop_decoded_during_fusion).

    No precomputed N_test-sized arrays anywhere.
    """
    cache = cache or Cache(None)
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common:
        raise RuntimeError("no test IDs in common across submissions")
    all_ids = sorted(common)
    M = len(submissions)
    feature_names = models.get("_feature_names", None)
    mem_print("fuse_test: enter")

    # ── 1. Decode (uint8) ────────────────────────────────────────────────────
    decoded_per_method: list[dict[str, np.ndarray]] | None = None
    raw_key = f"decoded_raw_{cache_key_base}.pkl"
    if cache.has(raw_key):
        cached = load_decoded_test(cache, raw_key)
        if (cached is not None and len(cached) == M
                and set(cached[0].keys()) >= set(all_ids)):
            decoded_per_method = cached

    if decoded_per_method is None:
        print(f"\nDecoding {M} submissions × {len(all_ids)} IDs each (uint8)...")
        decoded_per_method = decode_submissions_to_uint8(submissions, all_ids)
        cache_decoded_test(cache, raw_key, decoded_per_method)
    mem_print("fuse_test: after decode (uint8)")
    gc.collect()

    # ── 2. Group by (class, sample_id) and stream-fuse ──────────────────────
    groups = group_test_ids_by_sample(all_ids, class_map, default_class)
    n_groups = len(groups)
    n_multi = sum(1 for ids in groups.values() if len(ids) >= 2)
    print(f"\nFusing {len(all_ids)} images in {n_groups} sample groups "
          f"({n_multi} multi-view)"
          f"{' + calibration' if calibrators_per_class else ''}...")

    fused: dict[str, str] = {}
    shared_entry = models.get("_SHARED_")
    t1 = time.time()
    n_uniform = 0
    spatial_cache: dict | None = None
    n_done = 0

    for (cls, sample_id), ids_in_sample in sorted(groups.items(),
                                                       key=lambda kv: kv[0]):
        V = len(ids_in_sample)
        first_sid = ids_in_sample[0]
        H, W = decoded_per_method[0][first_sid].shape
        # Build (V, M, H, W) float32 block from compact storage.
        group_scores_f32 = np.empty((V, M, H, W), dtype=np.float32)
        for k, sid in enumerate(ids_in_sample):
            for mi in range(M):
                a = decoded_per_method[mi][sid]
                if a.shape != (H, W):
                    raise ValueError(
                        f"shape mismatch in group ({cls}, {sample_id}): "
                        f"{sid} method {mi} is {a.shape} != ({H}, {W})")
                group_scores_f32[k, mi] = to_f32(a)

        # Parse views for this group.
        views_in_grp: list[int | None] = []
        for sid in ids_in_sample:
            _, v = parse_sample_id(sid)
            views_in_grp.append(int(v) if v is not None else None)

        # Image-level MV features (V, M) per key.
        if cfg.use_mv_image_level and V >= 2:
            mv_il_group = compute_mv_image_level_for_group(
                group_scores_f32, anchors, cls, views_in_grp)
        else:
            # Single-view (or disabled): all-zero scalars.
            mv_il_group = {
                "self_p99":         np.zeros((V, M), dtype=np.float32),
                "sib_max_p99":      np.zeros((V, M), dtype=np.float32),
                "sib_mean_p99":     np.zeros((V, M), dtype=np.float32),
                "lonely_image":     np.zeros((V, M), dtype=np.float32),
                "sib_above_anchor": np.zeros((V, M), dtype=np.float32),
            }
            if cfg.use_mv_image_level and V == 1:
                # Still set self_p99 (no sibling info).
                for mi in range(M):
                    mv_il_group["self_p99"][0, mi] = float(
                        np.percentile(group_scores_f32[0, mi], 99))

        # Window-max MV features (V, M, H, W).
        if cfg.use_mv_window_max and V >= 2:
            mv_wm_group = compute_mv_window_max_for_group(
                group_scores_f32, window=cfg.mv_window)
        else:
            mv_wm_group = np.zeros((V, M, H, W), dtype=np.float32)

        # Model entry for this class.
        entry = models.get(cls)
        if entry is None or entry.get("_fallback_to_shared"):
            entry = shared_entry
        prior_cls       = priors["per_class"].get(cls) if priors else None
        prior_atype_max = priors["per_class_atype_max"].get(cls) if priors else None
        prior_global    = priors.get("global") if priors else None
        top_methods_cls = top_methods_per_class.get(cls)
        best_rank_cls   = top_methods_cls

        for k, sid in enumerate(ids_in_sample):
            v_int = views_in_grp[k]
            if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
                spatial_cache = _spatial_cache(H, W)
            scores_this = [group_scores_f32[k, mi] for mi in range(M)]

            if entry is None or "model" not in entry:
                n_uniform += 1
                fused_mat = np.mean(np.stack(scores_this, axis=0), axis=0)
            else:
                # Slice MV inputs for this view k.
                mvil_one = {key: mv_il_group[key][k] for key in mv_il_group}
                mvwm_one = [mv_wm_group[k, mi] for mi in range(M)]

                feats, _ = featurize_image_v7(
                    scores_this, cfg, class_id=cls,
                    all_classes=all_classes_onehot or [],
                    view=v_int, anchors=anchors,
                    spatial_cache=spatial_cache,
                    feature_names=feature_names,
                    mv_image_level=mvil_one,
                    mv_window_max=mvwm_one,
                    prior_class=prior_cls,
                    prior_atype_max=prior_atype_max,
                    prior_global=prior_global,
                    best_method_rank=best_rank_cls,
                    top_methods_for_cls=top_methods_cls)
                X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
                p_raw = entry["model"].predict_proba(X)[:, 1].astype(np.float32)
                cal = (calibrators_per_class.get(cls)
                       if calibrators_per_class else None)
                p_out = (apply_calibrator(cal, p_raw)
                          if cal is not None else p_raw)
                fused_mat = p_out.reshape(H, W)
                del feats, X, p_raw

            fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
            fused[sid] = float_matrix_to_q8rle(fused_mat)
            n_done += 1
            if n_done % 500 == 0:
                print(f"    fused {n_done}/{len(all_ids)}  "
                      f"({time.time() - t1:.1f}s)", flush=True)

        # Free this group's decoded arrays
        if drop_decoded_during_fusion:
            for sid in ids_in_sample:
                for mi in range(M):
                    decoded_per_method[mi].pop(sid, None)

        del group_scores_f32, mv_il_group, mv_wm_group

    if n_uniform:
        print(f"  [warn] {n_uniform} images had no model — uniform avg")
    print(f"  fused all {len(all_ids)} in {time.time() - t1:.1f}s")
    mem_print("fuse_test: exit")
    return fused


# ═════════════════════════════════════════════════════════════════════════════
# Ablation master CSV append + small-CC stub (kept for parity)
# ═════════════════════════════════════════════════════════════════════════════
def append_to_master(master_csv: Path, row: dict) -> None:
    existing: list[dict] = []
    fieldnames: list[str] = []
    if master_csv.exists():
        with open(master_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)
    for k in row.keys():
        if k not in fieldnames: fieldnames.append(k)
    existing.append(row)
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing: w.writerow({k: r.get(k, "") for k in fieldnames})


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--local-preds", nargs="+", required=True, type=Path)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/data"))
    ap.add_argument("--class-map", type=Path)
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="stacker-xgb-v7")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=None)
    # Feature toggles
    ap.add_argument("--no-raw-score", action="store_true")
    ap.add_argument("--no-per-image-rank", action="store_true")
    ap.add_argument("--no-normed", action="store_true")
    ap.add_argument("--no-tail-excess", action="store_true")
    ap.add_argument("--no-gauss-smooth", action="store_true")
    ap.add_argument("--gauss-sigma", type=float, default=3.0)
    ap.add_argument("--no-cross-consensus", action="store_true")
    ap.add_argument("--no-cross-dispersion", action="store_true")
    ap.add_argument("--no-mv-image-level", action="store_true")
    ap.add_argument("--no-mv-window-max", action="store_true")
    ap.add_argument("--mv-window", type=int, default=7)
    ap.add_argument("--no-cc-features", action="store_true")
    ap.add_argument("--n-top-methods-cc", type=int, default=3)
    ap.add_argument("--no-spatial-prior", action="store_true")
    ap.add_argument("--no-spatial-geom", action="store_true")
    ap.add_argument("--no-class-onehot", action="store_true")
    ap.add_argument("--no-view-onehot", action="store_true")
    ap.add_argument("--no-best-method-rank", action="store_true")
    ap.add_argument("--prior-heatmaps-dir", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "analysis_out/tables"))
    # XGB overrides
    ap.add_argument("--n-estimators", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--subsample", type=float)
    ap.add_argument("--colsample-bytree", type=float)
    ap.add_argument("--reg-alpha", type=float)
    ap.add_argument("--reg-lambda", type=float)
    ap.add_argument("--min-child-weight", type=float)
    ap.add_argument("--gamma", type=float)
    # Tuning + calibration
    ap.add_argument("--tune-mode", default="none",
                    choices=["none", "global", "per-class"])
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--tune-cv", default="loao", choices=["loao", "loio"])
    ap.add_argument("--tune-timeout-min", type=float, default=None)
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "platt", "isotonic"])
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--min-pixels-per-cell", type=int, default=2000)
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
    cache = Cache(args.cache_dir)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"XGBOOST STACKER v7 — {len(args.runs)} methods", "=")
        print(f"  tune_mode        : {args.tune_mode}")
        print(f"  calibrate        : {args.calibrate}")
        print(f"  prior dir        : {args.prior_heatmaps_dir}")
        print(f"  cache_dir        : {args.cache_dir}")
        for i, p in enumerate(args.runs):
            print(f"  method {i}: {method_names[i]}")
        mem_print("start")

        # Cache keys
        local_preds_meta = file_meta_hash(args.local_preds)
        runs_meta = file_meta_hash(args.runs)
        cache_key_val   = f"val_aligned_{local_preds_meta}.npz"
        cache_key_test  = f"{runs_meta}__{local_preds_meta}"

        # ── Load + align val (cached) ────────────────────────────────────────
        val = load_aligned_val(cache, cache_key_val)
        loaded_subs = False
        subs = None
        if val is None:
            print("\nLoading test submissions...")
            subs = [load_submission(p) for p in args.runs]
            loaded_subs = True
            for p, s in zip(args.runs, subs):
                print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
            print("\nLoading local-val predictions...")
            preds_per_method = []
            for p in args.local_preds:
                d = load_local_preds(p)
                print(f"  {p.parent.name}/{p.name}: {len(d['ids'])} val images, "
                      f"{float(d['masks'].mean()) * 100:.3f}% positive pixels")
                preds_per_method.append(d)
            print("\nAligning local-val predictions across methods...")
            val = align_local_preds(preds_per_method, method_names)
            del preds_per_method; gc.collect()
            save_aligned_val(cache, cache_key_val, val)
        if not loaded_subs:
            print("\nLoading test submissions...")
            subs = [load_submission(p) for p in args.runs]
            for p, s in zip(args.runs, subs):
                print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
        mem_print("after val align")

        any_has_paths = val.get("image_paths") is not None
        N_val, H_val, W_val, M = val["scores"].shape
        classes = sorted(set(val["classes"].tolist()))
        print(f"\nClasses: {classes}")
        print(f"Val shape: N={N_val} H={H_val} W={W_val} M={M}")

        # ── 1. Train_good-anchored normalisation table ─────────────────────
        anchors = fit_train_good_anchors(val, M,
                                            min_pixels_per_cell=args.min_pixels_per_cell)
        mem_print("after anchors")

        # ── 2. Rank methods per class by val pooled AP ─────────────────────
        top_methods_per_class = rank_methods_per_class(val, M)
        # Trim to top-K for CC use.
        top_K = args.n_top_methods_cc
        top_methods_for_cc = {cls: order[:top_K]
                                 for cls, order in top_methods_per_class.items()}
        mem_print("after method ranking")

        # ── 3. Spatial priors ──────────────────────────────────────────────
        priors: dict | None = None
        if not args.no_spatial_prior:
            print(f"\nLoading per-class spatial priors from "
                  f"{args.prior_heatmaps_dir}...")
            priors = load_spatial_priors_v7(args.prior_heatmaps_dir, classes)

        # ── 4. Multi-view features (val) ───────────────────────────────────
        mv_image_level_val = None
        mv_window_max_val  = None
        if not args.no_mv_image_level:
            if any_has_paths:
                print(f"\nBuilding image-level MV features (val)...")
                mv_image_level_val = compute_mv_image_level_val(val, anchors)
            else:
                print(f"\n[warn] image-level MV features disabled "
                      f"(no image_paths in val).")
        if not args.no_mv_window_max:
            if any_has_paths:
                print(f"\nBuilding window-max MV features (val, window={args.mv_window})...")
                mv_window_max_val = compute_mv_window_max_val(val,
                                                                 window=args.mv_window)
                mem_print("after window-max MV (val)")
            else:
                print(f"\n[warn] window-max MV features disabled "
                      f"(no image_paths in val).")

        # ── Build feature config ───────────────────────────────────────────
        cfg = FeatureConfig(
            use_raw_score=not args.no_raw_score,
            use_per_image_rank=not args.no_per_image_rank,
            use_normed=not args.no_normed,
            use_tail_excess=not args.no_tail_excess,
            use_gauss_smooth=not args.no_gauss_smooth,
            gauss_sigma=args.gauss_sigma,
            use_cross_consensus=not args.no_cross_consensus,
            use_cross_dispersion=not args.no_cross_dispersion,
            use_mv_image_level=(not args.no_mv_image_level)
                                 and (mv_image_level_val is not None or not any_has_paths),
            use_mv_window_max=(not args.no_mv_window_max)
                                and (mv_window_max_val is not None or not any_has_paths),
            mv_window=args.mv_window,
            use_cc_features=not args.no_cc_features,
            n_top_methods_cc=args.n_top_methods_cc,
            use_spatial_prior=(not args.no_spatial_prior) and (priors is not None),
            use_spatial_geom=not args.no_spatial_geom,
            use_class_onehot=not args.no_class_onehot,
            use_view_onehot=not args.no_view_onehot,
            use_best_method_rank=not args.no_best_method_rank,
        )
        print(f"\nFeature config:")
        for k, v in asdict(cfg).items():
            print(f"  {k:<32} = {v}")

        all_classes_onehot = classes if cfg.use_class_onehot else None

        # ── Build training matrices ────────────────────────────────────────
        print(f"\nBuilding training matrices (featurize + sample negatives)...")
        training_data = build_training_data(
            val, classes, cfg,
            neg_per_pos=args.neg_per_pos, seed=args.seed,
            anchors=anchors,
            mv_image_level_val=mv_image_level_val,
            mv_window_max_val=mv_window_max_val,
            priors=priors,
            top_methods_per_class=top_methods_for_cc,
            all_classes_onehot=all_classes_onehot)
        feature_names = training_data.get("_feature_names", [])
        print(f"  total features per pixel: {len(feature_names)}")
        mem_print("after build_training_data")

        # Free MV-val arrays (already encoded into training matrices).
        if mv_window_max_val is not None:
            del mv_window_max_val
        if mv_image_level_val is not None:
            del mv_image_level_val
        gc.collect()
        mem_print("after dropping val MV arrays")

        # ── XGB params ─────────────────────────────────────────────────────
        cli_overrides = {k: v for k, v in {
            "n_estimators":     args.n_estimators,
            "max_depth":        args.max_depth,
            "learning_rate":    args.learning_rate,
            "subsample":        args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha":        args.reg_alpha,
            "reg_lambda":       args.reg_lambda,
            "min_child_weight": args.min_child_weight,
            "gamma":            args.gamma,
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

        # ── OOF preds ──────────────────────────────────────────────────────
        oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if tune_per_class_results is not None:
            for cls, r in tune_per_class_results.items():
                oof_per_class[cls] = (r["oof_preds"], r["oof_labels"])
        else:
            print(f"\nCollecting OOF preds per class...")
            for cls in [c for c in training_data
                         if not c.startswith("_")
                         and not training_data[c].get("_fallback_to_shared")]:
                td = training_data[cls]
                params = get_params_for_class(cls, params_global,
                                                params_per_class)
                t0 = time.time()
                op, ol = loao_oof_one_class(
                    td, params, args.seed, cv_mode=args.tune_cv,
                    neg_per_pos=args.neg_per_pos)
                print(f"  {cls}: {len(op):>9d} OOF preds "
                      f"({time.time() - t0:.1f}s)")
                oof_per_class[cls] = (op, ol)

        # ── Calibration ────────────────────────────────────────────────────
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

        # ── Verify ─────────────────────────────────────────────────────────
        verification: dict = {}
        if not args.no_verify and oof_per_class:
            verification = verify_stacker_vs_singles(
                val, method_names, oof_per_class)

        # Free val now.
        del val; gc.collect()
        mem_print("after val freed")

        # ── Fit production models ──────────────────────────────────────────
        print(f"\nFitting final per-class XGBoost models...")
        models = fit_per_class(training_data, params_global,
                                params_per_class, args.seed)
        # Free per-class arrays
        for cls in list(training_data.keys()):
            if cls.startswith("_"): continue
            free_class_training_data(training_data, cls)
        gc.collect()
        mem_print("after free_class_training_data")

        # ── Class map for test ─────────────────────────────────────────────
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

        # ── Inference ──────────────────────────────────────────────────────
        fused = fuse_test(
            subs, models, class_map, cfg,
            default_class=default_class,
            calibrators_per_class=calibrators_per_class,
            anchors=anchors, priors=priors,
            top_methods_per_class=top_methods_for_cc,
            all_classes_onehot=all_classes_onehot,
            cache=cache, cache_key_base=cache_key_test)

        del subs; gc.collect()
        mem_print("after fuse_test return + free subs")

        # ── Submission ─────────────────────────────────────────────────────
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

        # ── OOF dump ───────────────────────────────────────────────────────
        if oof_per_class:
            oof_path = run_dir / "oof_predictions.npz"
            to_save = {"classes": np.array(list(oof_per_class.keys()),
                                            dtype=object)}
            for cls, (op, ol) in oof_per_class.items():
                to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
                to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
            np.savez_compressed(oof_path, **to_save)
            print(f"Saved OOF preds -> {oof_path}")

        # ── Config dump ────────────────────────────────────────────────────
        model_dump = {
            "version": 7,
            "methods": method_names,
            "neg_per_pos": args.neg_per_pos,
            "seed": args.seed,
            "feature_config": asdict(cfg),
            "feature_names": feature_names,
            "top_methods_per_class": top_methods_per_class,
            "anchors_summary": {
                f"{mi}|{cls}|{view}": {
                    "p50": a.p50, "p95": a.p95, "p99": a.p99,
                    "n_pixels": a.n_pixels,
                }
                for (mi, cls, view), a in list(anchors.items())[:200]
            },
            "xgb_params_global": params_global,
            "xgb_params_per_class": params_per_class,
            "tune_mode": args.tune_mode,
            "tune_cv": args.tune_cv,
            "calibration_method": args.calibrate,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "verification": {
                k: v for k, v in (verification or {}).items()
                if k != "per_class_single"
            },
            "per_class_models": {},
        }
        for k, v in models.items():
            if k.startswith("_") and k != "_SHARED_": continue
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
            json.dump(model_dump, f, indent=2, default=str)
        print(f"\nWrote stacker config -> {cfg_path}")

        # ── Top features ───────────────────────────────────────────────────
        if "_SHARED_" in models and "feature_importance" in models["_SHARED_"]:
            fi = np.asarray(models["_SHARED_"]["feature_importance"])
            if feature_names and len(feature_names) == len(fi):
                top = np.argsort(-fi)[:30]
                print(f"\nTop-30 feature importances (SHARED model):")
                for r, j in enumerate(top, 1):
                    print(f"  {r:>2d}. {feature_names[j]:<32s}  {fi[j]:.4f}")
                # Family rollup
                groups_fi: dict[str, float] = defaultdict(float)
                for name, imp in zip(feature_names, fi):
                    if name.startswith("m") and "_mv_" in name:
                        groups_fi["mv_features"] += float(imp)
                    elif name.startswith("top") and "_cc_" in name:
                        groups_fi["cc_features"] += float(imp)
                    elif name.startswith("xc_"):
                        groups_fi["cross_method"] += float(imp)
                    elif name.startswith("s_prior"):
                        groups_fi["spatial_prior"] += float(imp)
                    elif name.startswith("s_"):
                        groups_fi["spatial_geom"] += float(imp)
                    elif name.startswith("c_"):
                        groups_fi["class_onehot"] += float(imp)
                    elif name.startswith("v_"):
                        groups_fi["view_onehot"] += float(imp)
                    elif name.endswith("_class_rank"):
                        groups_fi["best_method_rank"] += float(imp)
                    elif "_normed" in name or name.endswith("_tail") \
                            or name.endswith("_gnormed"):
                        groups_fi["normed_per_method"] += float(imp)
                    elif name.endswith("_raw") or name.endswith("_rank"):
                        groups_fi["raw_per_method"] += float(imp)
                    else:
                        groups_fi["other"] += float(imp)
                print(f"\nFeature-family importance shares (SHARED):")
                tot = sum(groups_fi.values()) + 1e-12
                for fam in sorted(groups_fi, key=lambda k: -groups_fi[k]):
                    print(f"  {fam:<28s} {groups_fi[fam] / tot * 100:>6.2f}%")

        # ── Ablation row ───────────────────────────────────────────────────
        run_id = "stacker_xgb_v7_" + hashlib.sha1(
            "|".join(str(p) for p in args.runs).encode("utf-8")
        ).hexdigest()[:6]
        overall_stacker = verification.get("overall_stacker", float("nan"))
        pooled_stacker = verification.get("pooled_stacker_ap", float("nan"))
        notes = (f"xgb v7 | M={len(method_names)} | F={len(feature_names)} | "
                  f"normed={int(cfg.use_normed)} | "
                  f"mv_image={int(cfg.use_mv_image_level)} | "
                  f"mv_window={int(cfg.use_mv_window_max)} | "
                  f"cc={int(cfg.use_cc_features)}/top{cfg.n_top_methods_cc} | "
                  f"prior={int(cfg.use_spatial_prior)} | "
                  f"tune={args.tune_mode} | calibrate={args.calibrate} | "
                  f"pooled_AP={pooled_stacker:.4f}")
        row = {
            "run_id": run_id, "run_tag": args.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "STACKER_XGB_V7",
            "feature_layers": "", "input_size": "",
            "n_classes": len(classes),
            "AP_overall": (f"{overall_stacker:.4f}"
                             if not math.isnan(overall_stacker) else ""),
            "AP_pooled": (f"{pooled_stacker:.4f}"
                            if not math.isnan(pooled_stacker) else ""),
            "runtime_min": "",
            "submission_path": str(args.out.with_suffix(".zip")),
            "notes": notes,
        }
        append_to_master(args.master_csv, row)
        print(f"\nAppended row to {args.master_csv}")
        mem_print("end")
        hr(f"DONE — run_id={run_id}", "=")


if __name__ == "__main__":
    main()