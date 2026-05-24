"""XGBoost stacker v3 — analysis-driven improvements over v2.

Direct successor to xgboost_stacker.py (v2). Same I/O contract: takes N
submission.csv + N local_predictions.npz, emits one fused submission +
stacker_config.json + oof_predictions.npz + run_log.txt + ablation row.

Each change below is tied to a finding in analyze_predictions.py's report.

# CHANGE 1 — Pooled pixel-AP in CV (CORRECTNESS, free)
  v2 averaged per-image APs inside each LOAO fold. The leaderboard
  metric is pooled pixel-AP across the entire class. Section 7 of the
  analysis quantifies how badly these diverge: `pooled − per_image`
  drift ranges from −0.23 (cutpaste_8d class_04) to +0.11
  (cutpaste_8c class_01). Optuna under the wrong metric was picking the
  wrong hyperparams. The new `pooled_cv_score_one_class` pools labels +
  preds across the entire held set within each fold, computes one AP.

# CHANGE 2 — Per-class per-method rank normalization (default on)
  Section 2 of the analysis showed wildly different score scales across
  models: patchcore_exp7 in [0.01, 0.31], uniad_12mv pinned at [0, 1]
  with p95 = p99 = p99.9 = 1.0 for most classes (saturated). Global
  rank-norm (v2) puts every method on the same monotone curve over the
  ENTIRE val set, which silently flattens patchcore's good calibration
  to match the noisy methods inside each class. Per-class per-method
  ranks (within `{class × method}`) restore each method's
  discriminative resolution where it matters: inside the class XGB will
  evaluate. Falls back to v2's global rank-norm via
  --rank-norm {per-class, global, none}.

# CHANGE 3 — Per-method small-CC suppression preprocessing
  Section 9 measured exactly how much AP each method gains from
  suppressing small connected components in its score map at the
  per-image 99% threshold:
    patchcore_exp7  : no gain (smooth)
    cutpaste_8c/8d  : ~0 gain (smooth)
    uniad_12mv      : +0.0024 @ min_cc=200
    cfa_14c         : +0.0035 @ min_cc=200
    fastflow_15c    : +0.0009 @ min_cc=200
    effad_nomv      : +0.0121 @ min_cc=200    ← biggest
  These add up across 6 noisy methods. Suppression runs ONCE up front
  on every score map (val and test) before featurization. Defaults
  derived from path substring matching; --small-cc and
  --small-cc-default override.

# CHANGE 4 — Cross-method consensus features (default on)
  Section 8b showed that on positive pixels alone, some method pairs
  have very low Spearman (patchcore vs cutpaste_8d = 0.086 on positives,
  0.533 on all pixels). That's textbook stacking diversity: methods
  agree on background but disagree on where defects are. We turn this
  into 3 explicit per-pixel features the XGB can split on:
    x_top1pct_count : # methods that rank this pixel in their top 1%
    x_top5pct_count : # methods that rank this pixel in their top 5%
    x_mean_perclass_rank : mean of per-class per-method ranks at the
                            pixel
  These use the per-class-rank-normed scores so they're scale-free.

# CHANGE 5 — Cross-view disagreement features (opt-in via
  --cross-view-disagreement)
  Section 6b: 17% of samples have ≥80% of their positive pixels in a
  single view (max_view_share ≥ 0.80). For these, per-pixel
  `view_score − sibling_mean` lights up only the concentrated view and
  is ~0 elsewhere. Cheap, scale-free under per-class rank-norm, and
  hurts nothing on the 83% where positives are spread evenly across
  views. One feature per method. Requires image_paths in
  local_predictions.npz (the v0+ saver always writes them).

# CHANGE 6 — Verification: stacker OOF vs every single model (per class)
  After fitting + (optional) calibration, prints a per-class pixel-AP
  table comparing the stacker's OOF predictions (pooled) against every
  input method's OOF on the SAME val pixels. Flags wins/losses and
  prints the headline: did the stacker beat the best single model
  overall, and per-class. Lets you decide from one run whether stacking
  is paying off.

ALL changes can be turned OFF via CLI flags so v3 is bit-compatible
with v2 if you want a like-for-like comparison. Defaults reflect the
analysis: changes 1-4 on, 5 on iff image_paths is present, 6 always on.
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


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec — copied from v2, unchanged
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
# Loaders
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
    # v3: image_paths is needed for sample_id parsing → cross-view features.
    if "image_paths" in data.files:
        out["image_paths"] = data["image_paths"].astype(str)
    else:
        # Older saver versions didn't store paths; just leave it None.
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
# Sample-id / view parsing
# ─────────────────────────────────────────────────────────────────────────────
# local_preds_saver stores image_paths like:
#   .../class_03/train/anomaly_01/img_5cb26b5ff84a2385612c_view1.png
# Submission stems look like:
#   img_5cb26b5ff84a2385612c_view1
# The sample-id is everything before "_viewN". We use the same regex
# as analyze_predictions.py for consistency.
PATH_VIEW_RE = re.compile(r"^(?P<sid>.+?)_view(?P<v>\d+)(?:\.[A-Za-z]+)?$")


def parse_sample_id(name: str) -> tuple[str, int | None]:
    """Returns (sample_id, view). name can be a filename or a stem."""
    m = PATH_VIEW_RE.match(name)
    if m:
        return m.group("sid"), int(m.group("v"))
    return name, None


# ─────────────────────────────────────────────────────────────────────────────
# Model-name detection (for per-method small-CC defaults)
# ─────────────────────────────────────────────────────────────────────────────
# Substring patterns matched against the run directory name. Ordered most
# specific first. Returns the family key used to look up per-family
# defaults; "unknown" is the fallback.
MODEL_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("cutpaste", "cutpaste"),
    ("dnv2",     "patchcore_dnv2"),   # patchcore w/ DINOv2 backbone
    ("wrn50",    "patchcore_wrn50"),  # patchcore w/ WRN50 backbone
    ("uniad",    "uniad"),
    ("cfa_",     "cfa"),
    ("fastflow", "fastflow"),
    ("effad",    "effad"),
]


def detect_model_family(run_dir_name: str) -> str:
    """Return the family key for a method given its run-dir name.
    Used only to pick a sensible small-CC default; user can override
    via --small-cc explicitly."""
    s = run_dir_name.lower()
    for pat, fam in MODEL_FAMILY_PATTERNS:
        if pat in s:
            return fam
    return "unknown"


# Per-family defaults for small-CC min size at the per-image p99
# threshold. Numbers come straight from Section 9 of the diagnostic.
DEFAULT_SMALL_CC_PER_FAMILY = {
    "patchcore_dnv2":  0,     # smooth, no gain (Section 9)
    "patchcore_wrn50": 0,     # same family, similar smoothness
    "cutpaste":        0,     # essentially neutral
    "uniad":           200,   # +0.0024
    "cfa":             200,   # +0.0035
    "fastflow":        200,   # +0.0009
    "effad":           200,   # +0.0121 ← biggest gain
    "unknown":         0,     # conservative default
}


# ─────────────────────────────────────────────────────────────────────────────
# Small-CC suppression preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def suppress_small_ccs_image(score: np.ndarray, min_cc: int,
                               t_pct: float = 99.0) -> np.ndarray:
    """Zero pixels of small connected components in the per-image
    p_{t_pct}+ hot region. `score` is (H, W) float32 in [0, 1].
    Returns a NEW array; original is not mutated.

    Why this is safe to apply unconditionally:
      Section 9 of the diagnostic tested this exact transform across
      thresholds × min_cc grids per method. PatchCore/CutPaste get
      delta ≈ 0 (their hot regions don't have tiny lonely ghosts to
      remove). Noisy methods get small but consistently positive deltas.
      The transform NEVER hurts at min_cc ≤ 200 across the tested grid.
    """
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
    # sizes[0] is background; never zero that
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
    """Apply per-image suppression to a (N, H, W) stack. Returns NEW."""
    if min_cc <= 0:
        return scores
    out = np.empty_like(scores)
    for i in range(scores.shape[0]):
        out[i] = suppress_small_ccs_image(scores[i], min_cc, t_pct)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rank-norm: global, per-class, or none
# ─────────────────────────────────────────────────────────────────────────────
def rank_normalise_global_inplace_val(scores: np.ndarray) -> None:
    """v2 behaviour. Per-method, pool all val pixels, replace by rank.
    scores: (N, H, W, M)."""
    M = scores.shape[-1]
    flat = scores.reshape(-1, M)
    n = flat.shape[0]
    linsp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    for mi in range(M):
        col = flat[:, mi]
        order = np.argsort(col, kind="stable")
        ranks = np.empty_like(col)
        ranks[order] = linsp
        flat[:, mi] = ranks


def rank_normalise_per_class_inplace_val(scores: np.ndarray,
                                            classes: np.ndarray) -> None:
    """v3 default. Per (class, method), pool all val pixels of that
    class, replace by rank. scores: (N, H, W, M); classes: (N,).
    Restores per-class resolution that gets crushed under global
    rank-norm when one of the M methods is saturated (e.g. fastflow
    pinned at 1.0 across half the val set)."""
    M = scores.shape[-1]
    for cls in sorted(set(classes.tolist())):
        idx = np.flatnonzero(classes == cls)
        if idx.size == 0:
            continue
        # View into the class block (N_cls, H, W, M), then flatten.
        block = scores[idx]
        n = block.size // M
        flat = block.reshape(-1, M)
        linsp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        for mi in range(M):
            col = flat[:, mi]
            order = np.argsort(col, kind="stable")
            ranks = np.empty_like(col)
            ranks[order] = linsp
            flat[:, mi] = ranks
        # Write back. .reshape returns a view above so flat already
        # mutates the original; but be defensive.
        scores[idx] = flat.reshape(block.shape)


def rank_normalise_test_per_class_inplace(
    decoded_per_method: list[dict[str, np.ndarray]],
    all_ids: list[str],
    class_map: dict[str, str] | None,
    default_class: str,
) -> None:
    """Per-class per-method rank norm at test time. Falls back to global
    rank-norm for ids with no class mapping."""
    # Bucket ids by class.
    buckets: dict[str, list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        buckets[cls].append(sid)
    for mi, d in enumerate(decoded_per_method):
        t0 = time.time()
        for cls, ids_in_cls in buckets.items():
            shapes = {sid: d[sid].shape for sid in ids_in_cls}
            sizes  = {sid: int(np.prod(shapes[sid])) for sid in ids_in_cls}
            total  = sum(sizes.values())
            if total == 0:
                continue
            flat = np.empty(total, dtype=np.float32)
            idx = 0
            for sid in ids_in_cls:
                n = sizes[sid]
                flat[idx:idx + n] = d[sid].ravel()
                idx += n
            order = np.argsort(flat, kind="stable")
            ranks = np.empty_like(flat)
            ranks[order] = np.linspace(0.0, 1.0, total, dtype=np.float32)
            del order, flat
            idx = 0
            for sid in ids_in_cls:
                n = sizes[sid]
                d[sid] = ranks[idx:idx + n].reshape(shapes[sid]).astype(np.float32)
                idx += n
            del ranks
        print(f"    method {mi + 1}/{len(decoded_per_method)} per-class "
              f"rank-norm done ({time.time() - t0:.1f}s)")


def rank_normalise_test_global_inplace(
    decoded_per_method: list[dict[str, np.ndarray]],
    all_ids: list[str]) -> None:
    """v2 global rank-norm at test time. Kept for --rank-norm global."""
    for mi, d in enumerate(decoded_per_method):
        t0 = time.time()
        shapes = {sid: d[sid].shape for sid in all_ids}
        sizes  = {sid: int(np.prod(shapes[sid])) for sid in all_ids}
        total  = sum(sizes.values())
        flat = np.empty(total, dtype=np.float32)
        idx = 0
        for sid in all_ids:
            n = sizes[sid]
            flat[idx:idx + n] = d[sid].ravel()
            idx += n
        order = np.argsort(flat, kind="stable")
        ranks = np.empty_like(flat)
        ranks[order] = np.linspace(0.0, 1.0, total, dtype=np.float32)
        del order, flat
        idx = 0
        for sid in all_ids:
            n = sizes[sid]
            d[sid] = ranks[idx:idx + n].reshape(shapes[sid]).astype(np.float32)
            idx += n
        del ranks
        print(f"    method {mi + 1}/{len(decoded_per_method)}: global "
              f"rank-norm done ({time.time() - t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-view siblings: group val records and test ids by sample_id within
# the same class.
# ─────────────────────────────────────────────────────────────────────────────
def group_val_by_sample(val: dict) -> dict[tuple[str, str], list[int]]:
    """Return {(cls, sample_id) -> [val_idx, ...]}. sample_id is parsed
    from val['ids'] (which the saver builds as
    f'{cls}/{anomaly_type}/view_{enum_idx:02d}'), or from image_paths
    if those carry the real view structure. We prefer image_paths
    because the local_preds_saver enum_idx is NOT the real view
    number; analyze_predictions.py makes the same correction."""
    image_paths = val.get("image_paths")
    out: dict[tuple[str, str], list[int]] = defaultdict(list)
    if image_paths is not None:
        for i, (cls, p) in enumerate(zip(val["classes"], image_paths)):
            sid, _v = parse_sample_id(Path(str(p)).name)
            out[(str(cls), sid)].append(i)
    else:
        # Fallback: use ids field. Same-sample views may not group
        # correctly because the saver's enum_idx is per-(cls, atype).
        # Better than nothing.
        for i, (cls, id_) in enumerate(zip(val["classes"], val["ids"])):
            sid = str(id_).rsplit("/", 1)[-1]
            out[(str(cls), sid)].append(i)
    return out


def group_test_ids_by_sample(all_ids: list[str],
                                class_map: dict[str, str] | None,
                                default_class: str
                                ) -> dict[tuple[str, str], list[str]]:
    """Mirror group_val_by_sample for test ids."""
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        sample_id, _v = parse_sample_id(sid)
        out[(str(cls), sample_id)].append(sid)
    return out


def compute_sibling_mean_val(val_scores: np.ndarray,
                                groups: dict[tuple[str, str], list[int]],
                                method_idx: int) -> np.ndarray:
    """For each val image i in a multi-view sample, compute the mean
    score map of its sibling views (i.e. SAME sample, OTHER views) for
    the given method. Returns array shape (N, H, W). Image whose
    sample is single-view gets the zero map (so 'view - sibling_mean'
    is just the score itself, but then cross-view feature value = 0
    after subtracting itself — same effect)."""
    N, H, W, _M = val_scores.shape
    out = np.zeros((N, H, W), dtype=np.float32)
    for (_cls, _sid), idxs in groups.items():
        if len(idxs) < 2:
            # No siblings → leave zeros; downstream `score - 0 = score`
            # but the cross-view feature is meant to be ~0 here; we'll
            # multiply by an `is_multiview` indicator instead. See
            # compute_cross_view_disagreement below.
            continue
        stack = val_scores[idxs, :, :, method_idx]   # (V, H, W)
        total = stack.sum(axis=0, keepdims=False)    # (H, W)
        v = len(idxs)
        for k, vi in enumerate(idxs):
            # sibling mean = (total - this view) / (V - 1)
            out[vi] = (total - stack[k]) / max(v - 1, 1)
    return out


def compute_cross_view_disagreement_val(val_scores: np.ndarray,
                                           classes: np.ndarray,
                                           image_paths: np.ndarray | None
                                           ) -> tuple[np.ndarray, np.ndarray]:
    """v3 cross-view feature builder for the val set.

    Returns:
      cvd      : (N, H, W, M) — per-pixel (view - sibling_mean) per method.
      is_multi : (N,) uint8   — 1 if the val image's sample has ≥ 2 views
                  present in val, else 0. cvd[i] is forced to 0 when
                  is_multi[i] == 0; this is what makes the feature
                  harmless for single-view samples.

    Implementation note: the saver's val ids are NOT the original view
    numbers (they're enum_idx within the (cls, anomaly_type) list).
    We get the true sample_id from image_paths. If image_paths is None
    we still try with the saver's ids but the grouping will be lossy.
    """
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
            stack = val_scores[idxs, :, :, mi]       # (V, H, W)
            total = stack.sum(axis=0)                # (H, W)
            v = len(idxs)
            for k, vi in enumerate(idxs):
                sibling_mean = (total - stack[k]) / max(v - 1, 1)
                cvd[vi, :, :, mi] = stack[k] - sibling_mean
    print(f"  cross-view: {n_multi_samples} multi-view samples grouped "
          f"({int(is_multi.sum())}/{N} val images get a non-zero CVD map)")
    return cvd, is_multi


# ─────────────────────────────────────────────────────────────────────────────
# Local-val alignment (extended: also returns image_paths)
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
    # Take image_paths from method 0 — they're shared across methods
    # because every model evaluated the same val images.
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
                    image_paths_aligned[i] = str(image_paths_m0[j])
    print(f"  aligned {N} val images × {M} methods @ {H0}x{W0}"
          + (" (with image_paths)" if image_paths_aligned is not None else ""))
    return {"ids": np.asarray(common),
            "classes": classes.astype(str),
            "anomaly_types": anomaly_types.astype(str),
            "scores": scores,
            "masks": masks,
            "image_paths": image_paths_aligned}


# ─────────────────────────────────────────────────────────────────────────────
# Feature configuration
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
    # v3 additions
    use_cross_method_consensus: bool = True
    use_cross_view_disagreement: bool = False   # auto-set if image_paths present


# ─────────────────────────────────────────────────────────────────────────────
# Featurization primitives (mostly v2)
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
            "_shape": (H, W)}


def featurize_image(
    scores_per_method: list[np.ndarray],
    cfg: FeatureConfig,
    spatial_cache: dict | None = None,
    feature_names: list[str] | None = None,
    cross_view_disagreement_per_method: list[np.ndarray] | None = None,
    is_multiview_sample: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build the per-pixel feature stack for ONE image.

    v3 additions to v2's featurize:
      - cross_view_disagreement_per_method: optional list of (H, W)
        arrays, one per method, each = (this_view - sibling_mean) for
        that method on this image. If None, no CVD features are added.
        Forced to all-zero when is_multiview_sample=False so the
        feature is exactly 0 for single-view samples (XGB won't split
        on a constant).
      - cross_method_consensus features: top-pct counts and mean ranks
        across methods, computed from the per-method rank-normed scores
        (which are the SAME values we already have in scores_per_method
        when --rank-norm per-class).
    """
    assert scores_per_method
    H, W = scores_per_method[0].shape
    M = len(scores_per_method)
    layers: list[np.ndarray] = []
    names: list[str] = []

    # ── Per-method features (raw / rank / smoothed / windowed / etc.)
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

    # ── Cross-method statistics (v2)
    if cfg.use_cross_stats and M >= 2:
        stack = np.stack(scores_per_method, axis=0).astype(np.float32)
        cmean = stack.mean(axis=0); layers.append(cmean); names.append("x_mean")
        cmax  = stack.max(axis=0);  layers.append(cmax);  names.append("x_max")
        cmin  = stack.min(axis=0);  layers.append(cmin);  names.append("x_min")
        cstd  = stack.std(axis=0);  layers.append(cstd);  names.append("x_std")
        layers.append((cmax - cmin).astype(np.float32))
        names.append("x_range")

    # ── v3: cross-method CONSENSUS features
    # These reuse the per-method rank-normalised values (which is what
    # scores_per_method contains when --rank-norm per-class). The
    # interpretation is intentionally scale-free: "how many methods
    # rank this pixel near the top".
    if cfg.use_cross_method_consensus and M >= 2:
        stack = np.stack(scores_per_method, axis=0).astype(np.float32)
        top1pct = (stack >= 0.99).sum(axis=0).astype(np.float32)
        top5pct = (stack >= 0.95).sum(axis=0).astype(np.float32)
        mean_rank = stack.mean(axis=0).astype(np.float32)
        layers.append(top1pct); names.append("xc_top1pct_count")
        layers.append(top5pct); names.append("xc_top5pct_count")
        layers.append(mean_rank); names.append("xc_mean_rank")

    # ── v3: cross-view DISAGREEMENT features (one per method)
    # Section 6b: spikes only on the 17% of samples with view-concentrated
    # positives; ~0 elsewhere. Forced to 0 on single-view samples.
    if (cfg.use_cross_view_disagreement
        and cross_view_disagreement_per_method is not None):
        for mi, cvd in enumerate(cross_view_disagreement_per_method):
            if not is_multiview_sample:
                # zero map — XGB won't split on a constant
                layers.append(np.zeros((H, W), dtype=np.float32))
            else:
                cvd = np.asarray(cvd, dtype=np.float32)
                if cvd.shape != (H, W):
                    raise ValueError(
                        f"CVD method {mi} shape {cvd.shape} != ({H}, {W})")
                layers.append(cvd)
            names.append(f"m{mi}_cvd")

    # ── Spatial features (v2)
    if cfg.use_spatial:
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        for k in ("x", "y", "dist_edge", "dist_center"):
            layers.append(spatial_cache[k]); names.append(f"s_{k}")

    feats = np.stack(layers, axis=-1).astype(np.float32, copy=False)
    if feature_names is not None and names != feature_names:
        # Diagnostic; help debug rare drift between fit and predict.
        missing = [n for n in feature_names if n not in names]
        extra   = [n for n in names if n not in feature_names]
        raise RuntimeError(
            f"feature drift between fit and predict: "
            f"got {len(names)} cols, expected {len(feature_names)}; "
            f"missing={missing[:5]}; extra={extra[:5]}")
    return feats, names


# ─────────────────────────────────────────────────────────────────────────────
# Build per-class training matrices (v3: optional CVD)
# ─────────────────────────────────────────────────────────────────────────────
def build_training_data(val: dict, classes: list[str], cfg: FeatureConfig,
                         *, neg_per_pos: int, seed: int,
                         cvd: np.ndarray | None = None,
                         is_multi: np.ndarray | None = None) -> dict:
    """val: aligned dict from align_local_preds.
    cvd:  (N, H, W, M) cross-view disagreement, or None.
    is_multi: (N,) uint8 indicator of multi-view samples, or None.
    """
    rng = np.random.default_rng(seed)
    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]; val_anom_types = val["anomaly_types"]
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
        for i in cls_idx:
            mat_list = [val_scores[i, :, :, mi] for mi in range(M)]
            cvd_list = None
            is_mv = True
            if cvd is not None:
                cvd_list = [cvd[i, :, :, mi] for mi in range(M)]
                is_mv = bool(is_multi[i]) if is_multi is not None else True
            feats, names = featurize_image(
                mat_list, cfg, spatial_cache=spatial,
                cross_view_disagreement_per_method=cvd_list,
                is_multiview_sample=is_mv)
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
# XGBoost params
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.07,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.0,
    "reg_lambda":       1.0,
    "min_child_weight": 1.0,
    "tree_method":      "hist",
    "max_bin":          256,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-class LOAO/LOIO buckets
# ─────────────────────────────────────────────────────────────────────────────
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
# CHANGE 1 — POOLED CV pixel-AP (replaces v2's per-image-mean)
# ─────────────────────────────────────────────────────────────────────────────
def _pixel_ap_pooled(preds: np.ndarray, labels: np.ndarray) -> float:
    """Pool then AP, matching the leaderboard."""
    if int(labels.sum()) == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, preds))
    except Exception:
        order = np.argsort(-preds, kind="stable")
        y = labels[order]
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / max(int(y.sum()), 1)
        rec = np.concatenate([[0.0], rec])
        prec = np.concatenate([[1.0], prec])
        return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def pooled_cv_score_one_class(td: dict, params: dict, seed: int,
                                mode: str = "loao",
                                neg_per_pos: int = 30) -> float:
    """v3 CV objective: pool labels+preds across the held set of each
    fold, compute ONE AP per fold, then mean across folds. Matches
    leaderboard pixel-AP. The v2 alternative (per-image-AP averaging)
    was insensitive to cross-image score drift, which Section 7 of the
    diagnostic showed can be ±0.20 of AP."""
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
        # Pool predictions + labels across ALL held images in this fold.
        held_preds = []
        held_labels = []
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
    """LOAO with `params`; return OOF (preds, labels) over ALL pixels
    of held images. Used for calibration AND verification."""
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
# Optuna tuning (uses pooled CV)
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
    print(f"\n>>> Global Optuna tuning: {n_trials} trials, CV={cv_mode} "
          f"(POOLED pixel-AP)")
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
        print(f"\n>>> Per-class tuning: {cls}  (CV={cv_mode} POOLED, "
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
# Calibration (unchanged from v2)
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
# Fit per-class production models
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
# CHANGE 6 — Verification: stacker OOF vs every single, per class
# ─────────────────────────────────────────────────────────────────────────────
def verify_stacker_vs_singles(
    val: dict,
    method_names: list[str],
    oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    pretty_method_names: list[str] | None = None,
) -> dict:
    """Compute per-class POOLED pixel-AP for each single model AND the
    stacker OOF on the same val pixels (same class subset). Print a
    comparison table and the headline answer.

    Returns a summary dict suitable for stacker_config.json.
    """
    from sklearn.metrics import average_precision_score

    pretty = pretty_method_names or method_names
    val_scores = val["scores"]
    val_masks = val["masks"]
    val_classes = val["classes"]
    N, H, W, M = val_scores.shape

    # ── Single-model per-class pooled AP. Same data we trained on, no
    # smoothing applied here (we use the rank-normed + suppressed scores
    # that went into the feature matrix). This is the right comparison
    # because the stacker's OOF was produced from those same scores.
    print(f"\n  Computing per-class pooled pixel-AP for {M} single models...")
    single_ap: dict[str, dict[str, float]] = {}
    for mi, mname in enumerate(pretty):
        single_ap[mname] = {}
        for cls in sorted(set(val_classes.tolist())):
            idx = np.flatnonzero(val_classes == cls)
            if idx.size == 0:
                continue
            s = val_scores[idx, :, :, mi].ravel()
            y = val_masks[idx].ravel()
            if int(y.sum()) == 0:
                ap = float("nan")
            else:
                try:
                    ap = float(average_precision_score(y, s))
                except Exception:
                    ap = float("nan")
            single_ap[mname][cls] = ap

    # ── Stacker OOF per-class pooled AP.
    print(f"  Computing per-class pooled pixel-AP for stacker OOF...")
    stacker_ap: dict[str, float] = {}
    for cls, (op, ol) in oof_per_class.items():
        if int(ol.sum()) == 0:
            stacker_ap[cls] = float("nan")
            continue
        try:
            stacker_ap[cls] = float(average_precision_score(ol, op))
        except Exception:
            stacker_ap[cls] = float("nan")

    # ── Pretty table
    all_classes = sorted(set(val_classes.tolist()))
    name_w = max(14, max(len(n) for n in pretty + ["STACKER"]) + 1)
    hdr = f"  {'class':<10}"
    for n in pretty:
        hdr += f" {n[:12]:>13}"
    hdr += f" {'STACKER':>13} {'best_single':>13} {'win?':>7}"
    hr("VERIFICATION — per-class POOLED pixel-AP (stacker vs singles)", "=")
    print(hdr)
    n_class_wins = 0
    n_class_total = 0
    deltas: list[float] = []
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
        line += f" {st:>13.4f}"
        line += f" {best_ap:>13.4f}"
        if not (math.isnan(st) or math.isnan(best_ap)):
            n_class_total += 1
            delta = st - best_ap
            deltas.append(delta)
            line += f" {'WIN' if delta > 0 else 'lose':>7}"
            if delta > 0:
                n_class_wins += 1
        else:
            line += f" {'?':>7}"
        print(line)

    # ── Overall (mean across classes)
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
    print()
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

    return {
        "per_class_single": single_ap,
        "per_class_stacker": stacker_ap,
        "overall_singles": overall_singles,
        "overall_stacker": overall_stacker,
        "best_single_name": best_single_name,
        "best_single_overall_ap": best_single_ap,
        "delta_overall": delta_overall,
        "class_wins": n_class_wins,
        "class_total": n_class_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test-time inference
# ─────────────────────────────────────────────────────────────────────────────
def compute_test_cvd_per_method(
    decoded_per_method: list[dict[str, np.ndarray]],
    all_ids: list[str],
    class_map: dict[str, str] | None,
    default_class: str,
) -> tuple[dict[str, list[np.ndarray]], dict[str, bool]]:
    """For each test id and each method, compute the CVD map
    (this_view - sibling_mean) using same-sample views from
    decoded_per_method. Returns:
      cvd_per_id[id] = [cvd_map_method0, cvd_map_method1, ...]
      is_multi[id]   = True if sample has >= 2 views in test set.
    """
    M = len(decoded_per_method)
    groups = group_test_ids_by_sample(all_ids, class_map, default_class)
    cvd_per_id: dict[str, list[np.ndarray]] = {}
    is_multi: dict[str, bool] = {}
    n_multi = 0
    for (_cls, _sid), ids_in_sample in groups.items():
        V = len(ids_in_sample)
        if V < 2:
            # Single view: CVD = 0 for every method.
            for id_ in ids_in_sample:
                # We don't know the shape yet — defer to 0-map per method.
                # Use the first method's score shape for this id.
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


def fuse_test(submissions: list[dict[str, str]],
               models: dict,
               class_map: dict[str, str] | None,
               cfg: FeatureConfig,
               rank_norm_mode: str,
               default_class: str,
               calibrators_per_class: dict | None,
               small_cc_per_method: list[int],
               include_cvd: bool) -> dict[str, str]:
    """v3 inference loop. Adds:
      - Per-method small-CC suppression (after decode, before rank-norm)
      - Per-class per-method rank-norm at test time
      - Per-sample CVD maps for the cross-view feature
    """
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

    # ── v3: small-CC suppression per method
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

    # ── v3: per-class per-method rank-norm (or global, or skip)
    if rank_norm_mode == "per-class":
        print(f"\nPer-class per-method rank-normalisation (test)...")
        rank_normalise_test_per_class_inplace(
            decoded_per_method, all_ids, class_map, default_class)
    elif rank_norm_mode == "global":
        print(f"\nGlobal rank-normalisation (test) — v2 compatibility mode...")
        rank_normalise_test_global_inplace(decoded_per_method, all_ids)
    elif rank_norm_mode == "none":
        print(f"\nSkipping rank-norm (test): --rank-norm none")
    else:
        raise ValueError(f"unknown rank_norm_mode: {rank_norm_mode}")

    # ── v3: cross-view disagreement (test) — needs sample grouping
    cvd_per_id: dict[str, list[np.ndarray]] | None = None
    is_multi_per_id: dict[str, bool] | None = None
    if include_cvd:
        print(f"\nComputing cross-view disagreement maps for test...")
        cvd_per_id, is_multi_per_id = compute_test_cvd_per_method(
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
            cvd_list = None
            is_mv = True
            if cvd_per_id is not None:
                cvd_list = cvd_per_id[sid]
                is_mv = is_multi_per_id.get(sid, False)
            feats, _ = featurize_image(
                mats, cfg, spatial_cache=spatial_cache,
                feature_names=feature_names,
                cross_view_disagreement_per_method=cvd_list,
                is_multiview_sample=is_mv)
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
# ablation_master append
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


# ─────────────────────────────────────────────────────────────────────────────
# Parse --small-cc spec
# ─────────────────────────────────────────────────────────────────────────────
def parse_small_cc_spec(
    spec: list[str] | None,
    method_names: list[str],
    default_min_cc: int,
) -> list[int]:
    """Parse --small-cc 'family=N' or 'index=N' pairs into a per-method
    list aligned with method_names.

    Resolution order:
      1. CLI override matching the method name exactly (index or
         substring of method_names[i])
      2. CLI override matching the family detected from method_names[i]
      3. Per-family default from DEFAULT_SMALL_CC_PER_FAMILY
      4. --small-cc-default
    """
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
        # 1: exact method-name substring
        for k, v in cli_pairs.items():
            if k.isdigit() and int(k) == i:
                val = v; break
            if not k.isdigit() and k in n_lower:
                val = v; break
        # 2: family override from CLI
        if val is None and fam in cli_pairs:
            val = cli_pairs[fam]
        # 3: per-family default
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
    # CHANGE 2: rank-norm granularity
    ap.add_argument("--rank-norm", default="per-class",
                    choices=["per-class", "global", "none"],
                    help="per-class: rank within {class × method} "
                         "(v3 default; restores per-class resolution "
                         "lost by global rank-norm when one method is "
                         "saturated). global: v2 behavior. none: leave "
                         "raw scores.")
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="stacker-xgb-v3")
    ap.add_argument("--no-zip", action="store_true")
    # CHANGE 3: small-CC suppression
    ap.add_argument("--small-cc", nargs="*", default=None,
                    help="Per-method small-CC suppression overrides. "
                         "Format: 'family=N' or 'index=N' or "
                         "'substring=N'. Examples: --small-cc "
                         "uniad=200 effad=200 patchcore_dnv2=0. "
                         "Defaults to per-family values derived from "
                         "Section 9 of the diagnostic.")
    ap.add_argument("--small-cc-default", type=int, default=0,
                    help="Default min_cc for methods whose family "
                         "isn't recognised. 0 = no suppression.")
    ap.add_argument("--no-small-cc", action="store_true",
                    help="Disable small-CC suppression entirely "
                         "(v2 compat). Overrides --small-cc.")
    # CHANGE 4: cross-method consensus
    ap.add_argument("--no-cross-method-consensus", action="store_true",
                    help="Disable v3 cross-method consensus features "
                         "(top-pct counts + mean rank).")
    # CHANGE 5: cross-view disagreement
    ap.add_argument("--cross-view-disagreement", action="store_true",
                    default=True,
                    help="Add per-method (view - sibling_mean) features. "
                         "Auto-disabled if image_paths missing. Section 6b "
                         "shows this helps the 17% of samples with "
                         "view-concentrated positives.")
    ap.add_argument("--no-cross-view-disagreement", dest="cross_view_disagreement",
                    action="store_false")
    # Feature toggles (v2 compat)
    ap.add_argument("--no-window-stats", action="store_true")
    ap.add_argument("--no-gradient", action="store_true")
    ap.add_argument("--no-laplacian", action="store_true")
    ap.add_argument("--no-dist-to-hot", action="store_true")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-cross-stats", action="store_true")
    ap.add_argument("--no-image-aggregates", action="store_true")
    # XGB param overrides
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
    ap.add_argument("--params-per-class-json", type=Path, default=None,
                    help="Load per-class XGB params from JSON; skips tuning.")
    # Calibration
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "platt", "isotonic"])
    # Skip the verification step (it's cheap, but exposed for flexibility)
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the final stacker-vs-singles per-class "
                         "AP comparison.")
    args = ap.parse_args()

    if not HAS_XGB:
        raise SystemExit("[FATAL] xgboost not installed. uv pip install xgboost")
    if len(args.runs) < 2:
        raise SystemExit("need ≥ 2 methods to stack")
    if len(args.local_preds) != len(args.runs):
        raise SystemExit("--local-preds count must match --runs count")
    if args.tune_mode != "none" and not HAS_OPTUNA:
        raise SystemExit("optuna not installed. uv pip install optuna")

    method_names = [p.parent.name for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.out.parent

    with tee_to(run_dir / "run_log.txt"):
        hr(f"XGBOOST STACKER v3 — {len(args.runs)} methods", "=")
        print(f"  rank_norm    : {args.rank_norm}")
        print(f"  tune_mode    : {args.tune_mode}")
        print(f"  calibrate    : {args.calibrate}")
        for i, (r, lp) in enumerate(zip(args.runs, args.local_preds)):
            fam = detect_model_family(method_names[i])
            print(f"  method {i}: {method_names[i]}")
            print(f"            family={fam}")
            print(f"            submission : {r}")
            print(f"            local_preds: {lp}")

        # ── Resolve small-CC config ────────────────────────────────────────
        if args.no_small_cc:
            small_cc_per_method = [0] * len(method_names)
        else:
            small_cc_per_method = parse_small_cc_spec(
                args.small_cc, method_names, args.small_cc_default)
        print(f"\n  small-CC suppression per method:")
        for i, (n, m) in enumerate(zip(method_names, small_cc_per_method)):
            tag = "off" if m == 0 else f"min_cc≤{m} @ p99"
            print(f"    [{i}] {n[:48]:<48}  fam={detect_model_family(n):<16}  {tag}")

        # ── Load test submissions ──────────────────────────────────────────
        print("\nLoading test submissions...")
        subs = [load_submission(p) for p in args.runs]
        for p, s in zip(args.runs, subs):
            print(f"  {p.parent.name}/{p.name}: {len(s)} rows")

        # ── Load local-val predictions ─────────────────────────────────────
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

        # ── Apply small-CC suppression to LOCAL-VAL maps (per method) ─────
        if any(c > 0 for c in small_cc_per_method):
            print(f"\nApplying per-method small-CC suppression to local-val...")
            for mi, min_cc in enumerate(small_cc_per_method):
                if min_cc <= 0:
                    continue
                t1 = time.time()
                preds_per_method[mi]["scores"] = suppress_small_ccs_volume(
                    preds_per_method[mi]["scores"], min_cc=min_cc)
                print(f"    method {mi + 1}/{len(small_cc_per_method)}: "
                      f"suppressed CCs ≤ {min_cc} px "
                      f"({time.time() - t1:.1f}s)")

        # ── Align ──────────────────────────────────────────────────────────
        print("\nAligning local-val predictions across methods...")
        val = align_local_preds(preds_per_method, method_names)

        # ── Rank-norm ─────────────────────────────────────────────────────
        if args.rank_norm == "per-class":
            print(f"\nPer-class per-method rank-normalisation (val)...")
            rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
        elif args.rank_norm == "global":
            print(f"\nGlobal rank-normalisation (val) — v2 compat mode...")
            rank_normalise_global_inplace_val(val["scores"])
        else:
            print(f"\nSkipping rank-norm (val): --rank-norm none")

        # ── Cross-view disagreement (val) ──────────────────────────────────
        cvd_val: np.ndarray | None = None
        is_multi_val: np.ndarray | None = None
        use_cvd = args.cross_view_disagreement and any_has_paths
        if args.cross_view_disagreement and not any_has_paths:
            print(f"\n[warn] --cross-view-disagreement requested but no method "
                  f"has image_paths in its local_predictions.npz — disabling. "
                  f"Re-export local_predictions.npz with the current saver "
                  f"to enable.")
        if use_cvd:
            print(f"\nBuilding cross-view disagreement maps (val)...")
            cvd_val, is_multi_val = compute_cross_view_disagreement_val(
                val["scores"], val["classes"], val.get("image_paths"))

        # ── Build feature config ───────────────────────────────────────────
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
        )
        print(f"\nFeature config: {asdict(cfg)}")

        classes = sorted(set(val["classes"].tolist()))
        print(f"\nBuilding training matrices "
              f"(featurize + sample negatives)...")
        print(f"  classes present in val: {classes}")
        training_data = build_training_data(
            val, classes, cfg,
            neg_per_pos=args.neg_per_pos, seed=args.seed,
            cvd=cvd_val, is_multi=is_multi_val)
        feature_names = training_data.get("_feature_names", [])
        print(f"  total features per pixel: {len(feature_names)}")

        # ── Build XGB params (global base + optional Optuna) ───────────────
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

        if args.params_per_class_json is not None:
            with open(args.params_per_class_json, "r", encoding="utf-8") as f:
                params_per_class = json.load(f)
            print(f"\nLoaded per-class params from {args.params_per_class_json}")
            print(f"  classes: {sorted(params_per_class.keys())}")
            args.tune_mode = "none"

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

        print(f"\nFinal params (global): " + json.dumps(
            {k: v for k, v in params_global.items()
              if k not in ('tree_method', 'max_bin', 'objective',
                           'eval_metric', 'n_jobs', 'verbosity')},
            indent=2))
        if params_per_class:
            print(f"\nPer-class params present for: "
                  f"{sorted(params_per_class.keys())}")

        # ── Calibration: collect OOF preds (always, even if no calibrator,
        #    so the verification step has data) ─────────────────────────────
        oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if tune_per_class_results is not None:
            for cls, r in tune_per_class_results.items():
                oof_per_class[cls] = (r["oof_preds"], r["oof_labels"])
            print(f"\nReusing OOF preds from per-class tuning")
        else:
            # print(f"\nCollecting OOF preds per class (for verification"
            #       f"{' + calibration' if args.calibrate != 'none' else ''})...")
            # for cls, td in training_data.items():
            #     if cls.startswith("_") or td.get("_fallback_to_shared"):
            #         continue
            #     params = get_params_for_class(cls, params_global,
            #                                     params_per_class)
            #     t0 = time.time()
            #     op, ol = loao_oof_one_class(
            #         td, params, args.seed, cv_mode=args.tune_cv,
            #         neg_per_pos=args.neg_per_pos)
            #     print(f"  {cls}: {len(op):>9d} OOF preds "
            #           f"({time.time() - t0:.1f}s)")
            #     oof_per_class[cls] = (op, ol)
            need_oof = (args.calibrate != "none") or (not args.no_verify)
            if need_oof:
                print(f"\nCollecting OOF preds per class (for verification"
                      f"{' + calibration' if args.calibrate != 'none' else ''})...")
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
            else:
                print(f"\nSkipping OOF collection (--no-verify + --calibrate none)")

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
                      f"post={ap_post:.4f}  "
                      f"(monotonic; benefit is in cross-class scale alignment)")

        # ── CHANGE 6 — VERIFICATION ────────────────────────────────────────
        verification: dict = {}
        if not args.no_verify and oof_per_class:
            verification = verify_stacker_vs_singles(
                val, method_names, oof_per_class)

        # ── Fit final per-class production models ──────────────────────────
        print(f"\nFitting final per-class XGBoost models...")
        models = fit_per_class(training_data, params_global,
                                params_per_class, args.seed)

        # ── Build class map for test ───────────────────────────────────────
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
            rank_norm_mode=args.rank_norm,
            default_class=default_class,
            calibrators_per_class=calibrators_per_class,
            small_cc_per_method=small_cc_per_method,
            include_cvd=use_cvd)

        # ── Write submission CSV (+ ZIP) ───────────────────────────────────
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

        # ── Save OOF preds ─────────────────────────────────────────────────
        if oof_per_class:
            oof_path = run_dir / "oof_predictions.npz"
            to_save = {"classes": np.array(list(oof_per_class.keys()),
                                            dtype=object)}
            for cls, (op, ol) in oof_per_class.items():
                to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
                to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
            np.savez_compressed(oof_path, **to_save)
            print(f"Saved OOF preds -> {oof_path}")

        # ── Dump stacker_config.json ───────────────────────────────────────
        model_dump = {
            "version": 3,
            "methods": method_names,
            "rank_norm": args.rank_norm,
            "small_cc_per_method": small_cc_per_method,
            "use_cross_method_consensus": cfg.use_cross_method_consensus,
            "use_cross_view_disagreement": cfg.use_cross_view_disagreement,
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

        # ── Top feature importances (SHARED) ───────────────────────────────
        if "_SHARED_" in models and "feature_importance" in models["_SHARED_"]:
            fi = np.asarray(models["_SHARED_"]["feature_importance"])
            if feature_names and len(feature_names) == len(fi):
                top = np.argsort(-fi)[:15]
                print(f"\nTop-15 feature importances (SHARED model):")
                for r, j in enumerate(top, 1):
                    print(f"  {r:>2d}. {feature_names[j]:<26s}  {fi[j]:.4f}")

        # ── ablation_master row ────────────────────────────────────────────
        run_id = "stacker_xgb_v3_" + hashlib.sha1(
            "|".join(str(p) for p in args.runs).encode("utf-8")
        ).hexdigest()[:6]
        overall_stacker = verification.get("overall_stacker", float("nan"))
        notes = (f"xgb v3 | M={len(subs)} | F={len(feature_names)} | "
                  f"rank_norm={args.rank_norm} | "
                  f"small_cc=[{','.join(str(c) for c in small_cc_per_method)}] | "
                  f"consensus={int(cfg.use_cross_method_consensus)} | "
                  f"cvd={int(cfg.use_cross_view_disagreement)} | "
                  f"tune={args.tune_mode} | calibrate={args.calibrate}")
        row = {
            "run_id": run_id, "run_tag": args.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "STACKER_XGB_V3",
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