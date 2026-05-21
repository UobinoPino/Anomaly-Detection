"""Ensemble multiple anomaly-detection runs into a single submission.

Pipeline (each step maps to an advice line)
-------------------------------------------
  1.  Per-(method, class) ECDF anchored on normal pixels in
      local_predictions.npz                                      [15/05]
  2.  Per-(method, class) pixel-AP on local data → weights
      w = AP^gamma, with optional AP floor + top-K-per-class
  3.  Method-family pre-averaging within-family (mitigates
      correlated methods dominating the vote)
  4.  Streaming weighted geometric/arithmetic mean across
      families, per-pixel, one method at a time                  [14/05]
  5.  Light Gaussian smoothing
  6.  Multiview sample-gate                                      [12/05]
  7.  Per-class background floor                                 [08/05]
  8.  Small-blob removal (optional, scipy required)              [10/05]
  9.  Final per-class ECDF                                       [15/05]
 10.  q8rle quantise + write submission.csv                      [07/05]

Diagnostics written every run
-----------------------------
  per_method_per_class_ap.csv           full AP table
  per_method_per_class_weights.csv      final weights used
  method_correlation.csv                pairwise pearson r on local
  family_assignment.csv                 which run went to which family
  loo_recommendations.txt               leave-one-out improvement table

Inputs
------
Each --runs <dir> must contain:
    local_predictions.npz   — train_anomaly raw scores + GT masks
    test_predictions.npz    — test raw scores (add via test_preds_saver)

Family auto-detection
---------------------
By default, a run is assigned to a family based on substring matching
on its directory name (the run_tag). The default rules cover the
spacepresso retrain layout:

    exp2 / exp3 / exp4 / exp5 / exp6 / exp7    → "patchcore"
    exp8c / exp8d                              → "cutpaste"
    effad                                      → "efficientad"
    rd-vit                                     → "rd"
    exp14c                                     → "cfa"
    exp15c                                     → "fastflow"
    exp12mv                                    → "uniad"
    draem                                      → "draem"
    textad                                     → "textad"

Anything that matches no rule becomes its own family (the run name).
Override the mapping with `--family-rule SUBSTRING=FAMILY` flags,
which are matched in order before the defaults.

Members of the same family are averaged together (arithmetic mean of
calibrated scores) before the cross-family ensemble step. This is the
robust correction for correlated voters dominating the vote.

Usage
-----
    python submit_ensemble.py \\
        --runs /path/to/run1 /path/to/run2 ... \\
        --output-dir /path/to/ensembles/v1

Knobs (defaults reflect the reasoning above):
    --ensemble {geomean,mean}            default geomean
    --weight-power FLOAT                 default 2.0; 0 = equal weights
    --min-ap-keep FLOAT                  default 0.0 (off). Methods with
                                         per-(class) AP below this are
                                         dropped FROM THAT CLASS only.
    --top-k-per-class INT                default 0 (off). If >0, keep
                                         only the top-K families by AP
                                         per class.
    --no-family-averaging                make every run its own family
    --family-rule SUB=FAM                custom rule, repeatable
    --smooth-sigma FLOAT                 default 1.0
    --multiview {none,sample-gate}       default sample-gate
    --mv-gate-floor FLOAT                default 0.3
    --mv-tail-threshold FLOAT            default 0.99
    --mv-tail-norm-rate FLOAT            default 0.01
    --background-quantile FLOAT          default 0.5
    --min-blob-size INT                  default 0 (off; needs scipy)
    --blob-threshold FLOAT               default 0.5
    --blob-attenuation FLOAT             default 0.3
    --no-final-calibration               skip final per-class ECDF
    --no-zip                             skip submission.zip
    --skip-loo                           skip leave-one-out diagnostic
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Sample-id parsing (mirrors VIEW_RE in patchcore_baseline_v2)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_ID_RE = re.compile(r"^(?P<base>.+?)_view(?P<v>\d+)$")


def parse_sample_id(id_stem: str) -> str:
    m = SAMPLE_ID_RE.match(id_stem)
    return m.group("base") if m else id_stem


# ─────────────────────────────────────────────────────────────────────────────
# Family detection (substring → family name; first hit wins)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FAMILY_RULES: list[tuple[str, str]] = [
    # PatchCore variants — order matters: exp8 family before exp matches
    ("exp8c",    "cutpaste"),
    ("exp8d",    "cutpaste"),
    ("exp12mv",  "uniad"),
    ("exp14c",   "cfa"),
    ("exp15c",   "fastflow"),
    ("exp2",     "patchcore"),
    ("exp3",     "patchcore"),
    ("exp4",     "patchcore"),
    ("exp5",     "patchcore"),
    ("exp6",     "patchcore"),
    ("exp7",     "patchcore"),
    # Other named tracks
    ("cutpaste", "cutpaste"),
    ("effad",    "efficientad"),
    ("rd-vit",   "rd"),
    ("rd_vit",   "rd"),
    ("draem",    "draem"),
    ("textad",   "textad"),
]


def detect_family(run_name: str, rules: list[tuple[str, str]]) -> str:
    """Return the family name matched by the first substring rule that
    fires on run_name; if nothing matches, fall back to the run name."""
    name = run_name.lower()
    for sub, fam in rules:
        if sub.lower() in name:
            return fam
    return run_name


# ─────────────────────────────────────────────────────────────────────────────
# ECDF helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_ecdf(values: np.ndarray, n_bins: int = 4096, seed: int = 0):
    v = np.asarray(values, dtype=np.float32).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32))
    if v.size > 2_000_000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(v.size, 2_000_000, replace=False)
        v = v[idx]
    v_sorted = np.sort(v)
    cdf = np.linspace(0.0, 1.0, len(v_sorted), dtype=np.float32)
    if len(v_sorted) > n_bins:
        idx = np.linspace(0, len(v_sorted) - 1, n_bins).astype(np.int64)
        v_sorted = v_sorted[idx]
        cdf = cdf[idx]
    return v_sorted.astype(np.float32), cdf


def apply_ecdf(scores: np.ndarray, ecdf) -> np.ndarray:
    v_sorted, cdf = ecdf
    flat = scores.astype(np.float32).ravel()
    out = np.interp(flat, v_sorted, cdf, left=0.0, right=1.0)
    return out.reshape(scores.shape).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Bilinear resize without torch / scipy (CPU-only fallback)
# ─────────────────────────────────────────────────────────────────────────────
def bilinear_resize(arr: np.ndarray, target_h: int,
                     target_w: int) -> np.ndarray:
    """Bilinear-resize a 2-D or stacked 3-D array to (target_h, target_w).
    Pure-numpy so we don't require torch or scipy."""
    if arr.ndim == 3:
        return np.stack([bilinear_resize(arr[i], target_h, target_w)
                         for i in range(arr.shape[0])], axis=0)
    h, w = arr.shape
    if (h, w) == (target_h, target_w):
        return arr.astype(np.float32, copy=False)
    # Sample positions in source coordinates.
    yy = np.linspace(0, h - 1, target_h, dtype=np.float32)
    xx = np.linspace(0, w - 1, target_w, dtype=np.float32)
    y0 = np.floor(yy).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xx).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yy - y0).reshape(-1, 1)
    wx = (xx - x0).reshape(1, -1)
    src = arr.astype(np.float32, copy=False)
    Ia = src[np.ix_(y0, x0)]
    Ib = src[np.ix_(y0, x1)]
    Ic = src[np.ix_(y1, x0)]
    Id = src[np.ix_(y1, x1)]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def maybe_resize_stack(scores: np.ndarray, target_hw: tuple[int, int]
                        ) -> np.ndarray:
    """Resize (N, H, W) to (N, target_h, target_w) if shapes differ."""
    if scores.ndim != 3:
        raise ValueError(f"expected (N, H, W), got {scores.shape}")
    if scores.shape[1:] == target_hw:
        return scores.astype(np.float32, copy=False)
    return bilinear_resize(scores, target_hw[0], target_hw[1])


# ─────────────────────────────────────────────────────────────────────────────
# Multiview sample-gate
# ─────────────────────────────────────────────────────────────────────────────
def sample_evidence_from_tail(views: np.ndarray, threshold: float,
                                norm_rate: float) -> float:
    """Tail-mass excess normalised so norm_rate → 0 and 5×norm_rate → 1."""
    all_pix = views.ravel()
    frac_above = float((all_pix > threshold).sum() / all_pix.size)
    excess = max(frac_above - norm_rate, 0.0)
    denom = max(4.0 * norm_rate, 1e-8)
    return float(min(excess / denom, 1.0))


def apply_multiview_gate(ens: np.ndarray, ids: np.ndarray,
                          gate_floor: float, tail_threshold: float,
                          tail_norm_rate: float,
                          log_fn=None, cls_label: str = "") -> np.ndarray:
    out = ens.copy()
    sample_groups: dict[str, list[int]] = defaultdict(list)
    for k, id_ in enumerate(ids):
        sample_groups[parse_sample_id(str(id_))].append(k)
    n_multiview = sum(1 for v in sample_groups.values() if len(v) > 1)
    n_single = len(sample_groups) - n_multiview
    if log_fn:
        log_fn(f"  {cls_label}: {len(sample_groups)} samples "
               f"({n_multiview} multi-view, {n_single} single-view)")
    n_attenuated = 0
    for sid, indices in sample_groups.items():
        if len(indices) <= 1:
            continue
        views = ens[indices]
        ev = sample_evidence_from_tail(views, tail_threshold, tail_norm_rate)
        gate = gate_floor + (1.0 - gate_floor) * ev
        if gate < 0.95:
            n_attenuated += 1
        for i in indices:
            out[i] = ens[i] * gate
    if log_fn:
        log_fn(f"  {cls_label}: attenuated {n_attenuated}/{n_multiview} "
               f"multi-view groups")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Small-blob removal (optional, requires scipy)
# ─────────────────────────────────────────────────────────────────────────────
def remove_small_blobs(score: np.ndarray, threshold: float,
                        min_size: int, attenuation: float) -> np.ndarray:
    try:
        from scipy.ndimage import label, sum_labels
    except ImportError:
        return score
    bin_mask = (score > threshold).astype(np.uint8)
    labeled, num = label(bin_mask)
    if num == 0:
        return score
    sizes = sum_labels(bin_mask, labeled, index=np.arange(1, num + 1))
    small_label_ids = np.where(sizes < min_size)[0] + 1
    if len(small_label_ids) == 0:
        return score
    small_pixel_mask = np.isin(labeled, small_label_ids)
    out = score.copy()
    out[small_pixel_mask] *= attenuation
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing & q8rle (byte-equivalent to patchcore_baseline_v2)
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_smooth_2d(score: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return score
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2)).astype(np.float32)
    k = k / k.sum()
    sx = np.pad(score, ((r, r), (0, 0)), mode="reflect")
    sx = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, sx)
    sx = np.pad(sx, ((0, 0), (r, r)), mode="reflect")
    sx = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, sx)
    return sx.astype(np.float32)


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


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-AP
# ─────────────────────────────────────────────────────────────────────────────
def _pixel_ap_one(s: np.ndarray, m: np.ndarray) -> float:
    if m.sum() == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(m.ravel(), s.ravel()))
    except ImportError:
        s_flat = s.ravel(); m_flat = m.ravel()
        order = np.argsort(-s_flat, kind="stable")
        m_flat = m_flat[order]
        tp = np.cumsum(m_flat); fp = np.cumsum(1 - m_flat)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / max(int(m_flat.sum()), 1)
        recall = np.concatenate([[0.0], recall])
        precision = np.concatenate([[1.0], precision])
        return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def mean_pixel_ap(scores: np.ndarray, masks: np.ndarray) -> float:
    aps = []
    for i in range(scores.shape[0]):
        ap = _pixel_ap_one(scores[i], masks[i])
        if np.isfinite(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pearson correlation between two flattened arrays (chunked to avoid OOM)
# ─────────────────────────────────────────────────────────────────────────────
def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    am = af.mean(); bm = bf.mean()
    af = af - am; bf = bf - bm
    denom = float(np.sqrt((af ** 2).sum()) * np.sqrt((bf ** 2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((af * bf).sum() / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.write_text("")

    def __call__(self, msg: str = ""):
        print(msg)
        with open(self.path, "a") as f:
            f.write(msg + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ensemble", default="geomean",
                    choices=["geomean", "mean"])
    ap.add_argument("--weight-power", type=float, default=2.0)
    ap.add_argument("--min-ap-keep", type=float, default=0.0,
                    help="Drop method from a class if its local AP is "
                         "below this. 0 = keep all.")
    ap.add_argument("--top-k-per-class", type=int, default=0,
                    help="Keep only top-K families by AP per class. "
                         "0 = keep all.")
    ap.add_argument("--no-family-averaging", action="store_true")
    ap.add_argument("--family-rule", action="append", default=[],
                    metavar="SUB=FAM",
                    help="Custom family-detection rule. Repeatable.")
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    ap.add_argument("--multiview", default="sample-gate",
                    choices=["none", "sample-gate"])
    ap.add_argument("--mv-gate-floor", type=float, default=0.3)
    ap.add_argument("--mv-tail-threshold", type=float, default=0.99)
    ap.add_argument("--mv-tail-norm-rate", type=float, default=0.01)
    ap.add_argument("--background-quantile", type=float, default=0.5)
    ap.add_argument("--min-blob-size", type=int, default=0)
    ap.add_argument("--blob-threshold", type=float, default=0.5)
    ap.add_argument("--blob-attenuation", type=float, default=0.3)
    ap.add_argument("--no-final-calibration", action="store_true")
    ap.add_argument("--final-lo-pct", type=float, default=90.0,
                    help="Percentile of per-class scores treated as 0 "
                         "in the final calibration. Higher = sparser "
                         "submission. Default 90 means the bottom 90%% "
                         "of pixels get clipped to 0, leaving the top "
                         "10%% to carry the anomaly signal.")
    ap.add_argument("--final-hi-pct", type=float, default=99.9,
                    help="Percentile of per-class scores treated as 1 "
                         "in the final calibration.")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--skip-loo", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(args.output_dir / "ensemble_log.txt")
    t0 = time.time()

    # Build family rules: user rules first, then defaults.
    custom_rules: list[tuple[str, str]] = []
    for r in args.family_rule:
        if "=" not in r:
            log(f"WARN: ignoring malformed --family-rule {r!r}")
            continue
        sub, fam = r.split("=", 1)
        custom_rules.append((sub.strip(), fam.strip()))
    family_rules = custom_rules + DEFAULT_FAMILY_RULES

    log(f"submit_ensemble.py  output={args.output_dir}")
    log(f"  ensemble = {args.ensemble}  weight_power = {args.weight_power}")
    log(f"  family_averaging = {not args.no_family_averaging}")
    if custom_rules:
        log(f"  custom_family_rules = {custom_rules}")
    log(f"  min_ap_keep = {args.min_ap_keep}  "
        f"top_k_per_class = {args.top_k_per_class}")
    log(f"  smooth_sigma = {args.smooth_sigma}")
    log(f"  multiview = {args.multiview} "
        f"(floor={args.mv_gate_floor}, "
        f"tail={args.mv_tail_threshold}->{args.mv_tail_norm_rate})")
    log(f"  background_quantile = {args.background_quantile}")
    log(f"  min_blob_size = {args.min_blob_size} "
        f"(thr={args.blob_threshold}, att={args.blob_attenuation})")
    log()

    # ── Step 1: Load LOCAL predictions + remember test path ─────────────────
    log(f"[1/10] Loading {len(args.runs)} model runs (local only)...")
    models = []
    for rd in args.runs:
        local_p = rd / "local_predictions.npz"
        test_p  = rd / "test_predictions.npz"
        if not local_p.exists():
            log(f"  SKIP  {rd.name}: missing local_predictions.npz")
            continue
        if not test_p.exists():
            log(f"  SKIP  {rd.name}: missing test_predictions.npz — patch "
                f"with test_preds_saver and rerun submission step.")
            continue
        local = np.load(local_p, allow_pickle=True)
        family = (rd.name if args.no_family_averaging
                  else detect_family(rd.name, family_rules))
        m = {
            "name": rd.name,
            "family": family,
            "test_npz_path": test_p,
            "local_scores":  local["scores"].astype(np.float32),
            "local_classes": np.asarray(local["classes"]),
            "local_masks":   local["masks"].astype(np.uint8),
        }
        models.append(m)
        log(f"  OK    {rd.name[:55]:<55}  family={family:<12}  "
            f"local_n={len(m['local_classes'])}")

    if not models:
        log("\nNo usable runs. Aborting.")
        return 1
    M = len(models)
    log(f"\nUsing {M} runs grouped into "
        f"{len(set(m['family'] for m in models))} families.")

    # Write family assignment
    fam_csv = args.output_dir / "family_assignment.csv"
    with open(fam_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "family"])
        for m in models: w.writerow([m["name"], m["family"]])
    log(f"  saved family assignment -> {fam_csv}")

    all_classes = sorted(set().union(
        *[set(np.unique(m["local_classes"]).tolist()) for m in models]))
    log(f"Classes detected: {all_classes}")

    # ── Step 2: Per-(method, class) ECDF from local normal pixels ───────────
    log(f"\n[2/10] Building per-(method, class) ECDF anchors...")
    for m in models:
        m["ecdf"] = {}
        for cls in all_classes:
            cls_mask = m["local_classes"] == cls
            if not cls_mask.any():
                continue
            sc = m["local_scores"][cls_mask]
            mk = m["local_masks"][cls_mask]
            normal_pix = sc[mk == 0]
            if normal_pix.size < 1000:
                normal_pix = sc.ravel()
            m["ecdf"][cls] = build_ecdf(normal_pix)

    # ── Step 3: Per-(method, class) pixel-AP on local data ──────────────────
    log(f"\n[3/10] Computing per-(method, class) pixel-AP on local...")
    method_class_ap: dict[tuple[int, str], float] = {}
    # Cache calibrated local scores per (method, class) for downstream
    # correlation analysis and LOO diagnostic. All methods for the same
    # class are resized to a common reference shape (the largest H,W
    # encountered for that class) so that family means can be stacked.
    local_calibrated: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    # Pick a reference shape per class: the largest (H, W) any method
    # reports. Up-resizing low-res maps preserves AP rank ordering and
    # avoids destroying high-res spatial detail from FastFlow / hi-res
    # PatchCore runs.
    cls_ref_shape: dict[str, tuple[int, int]] = {}
    for cls in all_classes:
        best_hw = (0, 0)
        for m in models:
            cls_mask = m["local_classes"] == cls
            if not cls_mask.any():
                continue
            sh = m["local_scores"][cls_mask].shape[1:]
            if sh[0] * sh[1] > best_hw[0] * best_hw[1]:
                best_hw = (int(sh[0]), int(sh[1]))
        if best_hw != (0, 0):
            cls_ref_shape[cls] = best_hw
    log(f"  per-class reference shapes (largest): {cls_ref_shape}")

    for i, m in enumerate(models):
        for cls in all_classes:
            cls_mask = m["local_classes"] == cls
            if not cls_mask.any() or cls not in m["ecdf"]:
                method_class_ap[(i, cls)] = 0.0
                continue
            sc = m["local_scores"][cls_mask]
            mk = m["local_masks"][cls_mask]
            sc_cal = apply_ecdf(sc, m["ecdf"][cls])
            # AP is rank-based and shape-invariant — compute on native
            # resolution to avoid any artefacts from resizing.
            method_class_ap[(i, cls)] = mean_pixel_ap(sc_cal, mk)
            # For downstream stacking/LOO we need a common shape.
            ref_hw = cls_ref_shape.get(cls)
            if ref_hw is not None:
                sc_cal = maybe_resize_stack(sc_cal, ref_hw)
                if mk.shape[1:] != ref_hw:
                    # Resize masks with nearest-neighbour (preserve binary).
                    mk = (maybe_resize_stack(mk.astype(np.float32),
                                              ref_hw) > 0.5).astype(np.uint8)
            local_calibrated[(i, cls)] = (sc_cal, mk)
        # release raw to save RAM (we no longer need it after calibration)
        m["local_scores"] = None

    # AP table
    name_w = max(20, min(60, max(len(m["name"]) for m in models) + 2))
    fam_w = max(len(m["family"]) for m in models) + 2
    log("")
    header = (f"  {'method':<{name_w}}  {'family':<{fam_w}}  "
              + "  ".join(f"{c:>8}" for c in all_classes)
              + f"  {'mean':>8}")
    log(header)
    log("  " + "-" * (name_w + fam_w + 10 * len(all_classes) + 12))
    for i, m in enumerate(models):
        row = [method_class_ap[(i, c)] for c in all_classes]
        log(f"  {m['name'][:name_w]:<{name_w}}  {m['family']:<{fam_w}}  "
            + "  ".join(f"{a:>8.4f}" for a in row)
            + f"  {np.mean(row):>8.4f}")
    ap_csv = args.output_dir / "per_method_per_class_ap.csv"
    with open(ap_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "family"] + list(all_classes) + ["mean"])
        for i, m in enumerate(models):
            row = [method_class_ap[(i, c)] for c in all_classes]
            w.writerow([m["name"], m["family"]] + [f"{a:.4f}" for a in row]
                       + [f"{np.mean(row):.4f}"])
    log(f"\n  saved AP table -> {ap_csv}")

    # ── Step 4: Pairwise correlation matrix on local-calibrated scores ─────
    log(f"\n[4/10] Pairwise method correlation on local-calibrated scores...")
    corr_csv = args.output_dir / "method_correlation.csv"
    # Concatenate per-method calibrated local scores across all classes
    # into one long vector per method, in a deterministic order.
    cls_order = list(all_classes)
    method_vecs: list[np.ndarray] = []
    for i, m in enumerate(models):
        parts = []
        for cls in cls_order:
            if (i, cls) in local_calibrated:
                sc_cal, _mk = local_calibrated[(i, cls)]
                parts.append(sc_cal.ravel())
        method_vecs.append(np.concatenate(parts) if parts
                            else np.array([], dtype=np.float32))
    # Pearson r matrix
    R = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        R[i, i] = 1.0
        for j in range(i + 1, M):
            # length mismatch guard
            li = len(method_vecs[i]); lj = len(method_vecs[j])
            L = min(li, lj)
            if L < 100:
                R[i, j] = R[j, i] = 0.0
                continue
            r = pearson_r(method_vecs[i][:L], method_vecs[j][:L])
            R[i, j] = R[j, i] = r
    with open(corr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [m["name"] for m in models])
        for i, m in enumerate(models):
            w.writerow([m["name"]] + [f"{R[i, j]:.3f}" for j in range(M)])
    log(f"  saved {M}x{M} correlation matrix -> {corr_csv}")
    # Print top-3 most correlated pairs
    pairs = [(R[i, j], i, j) for i in range(M) for j in range(i + 1, M)]
    pairs.sort(reverse=True)
    log(f"  most-correlated pairs (top 5):")
    for r, i, j in pairs[:5]:
        log(f"    r={r:.3f}  {models[i]['name'][:55]}  <->  "
            f"{models[j]['name'][:55]}  "
            f"({models[i]['family']}/{models[j]['family']})")
    del method_vecs  # release

    # ── Step 5: Per-class weights with AP-floor and top-K-per-class ────────
    log(f"\n[5/10] Building per-class weights "
        f"(gamma={args.weight_power}, min_ap={args.min_ap_keep}, "
        f"top_k={args.top_k_per_class})...")

    # Per-class participation mask
    participates: dict[tuple[int, str], bool] = {}
    n_dropped_floor = 0
    n_dropped_topk = 0
    for cls in all_classes:
        # AP floor first
        candidate_indices = []
        for i in range(M):
            ap_ic = method_class_ap[(i, cls)]
            if ap_ic >= args.min_ap_keep:
                candidate_indices.append(i)
            else:
                participates[(i, cls)] = False
                n_dropped_floor += 1
        # Top-K per class — operates on FAMILIES, not individual runs.
        if args.top_k_per_class > 0 and not args.no_family_averaging:
            # Best-method-per-family table
            family_best_ap: dict[str, tuple[float, int]] = {}
            for i in candidate_indices:
                fam = models[i]["family"]
                ap_ic = method_class_ap[(i, cls)]
                if fam not in family_best_ap or ap_ic > family_best_ap[fam][0]:
                    family_best_ap[fam] = (ap_ic, i)
            top_families = sorted(family_best_ap.items(),
                                   key=lambda kv: kv[1][0],
                                   reverse=True)[:args.top_k_per_class]
            keep_fams = set(f for f, _ in top_families)
            for i in candidate_indices:
                if models[i]["family"] in keep_fams:
                    participates[(i, cls)] = True
                else:
                    participates[(i, cls)] = False
                    n_dropped_topk += 1
        elif args.top_k_per_class > 0:
            # No family averaging — top-K on runs directly
            ranked = sorted(candidate_indices,
                             key=lambda ii: method_class_ap[(ii, cls)],
                             reverse=True)[:args.top_k_per_class]
            keep = set(ranked)
            for i in candidate_indices:
                participates[(i, cls)] = (i in keep)
                if i not in keep:
                    n_dropped_topk += 1
        else:
            for i in candidate_indices:
                participates[(i, cls)] = True

    if args.min_ap_keep > 0:
        log(f"  AP-floor drops: {n_dropped_floor} (method, class) pairs")
    if args.top_k_per_class > 0:
        log(f"  top-K drops:    {n_dropped_topk} (method, class) pairs")

    # Family-level weights: w(family, class) = (mean AP among participating
    # members in that family) ^ gamma. Family contribution is mean of
    # member scores within the family.
    family_weights: dict[tuple[str, str], float] = {}  # (family, cls) -> w
    family_members: dict[tuple[str, str], list[int]] = defaultdict(list)
    families = sorted(set(m["family"] for m in models))
    for cls in all_classes:
        per_family_ap: dict[str, list[float]] = defaultdict(list)
        for i in range(M):
            if not participates.get((i, cls), False):
                continue
            family_members[(models[i]["family"], cls)].append(i)
            per_family_ap[models[i]["family"]].append(
                method_class_ap[(i, cls)])
        raw_ws: dict[str, float] = {}
        for fam in families:
            if fam not in per_family_ap:
                raw_ws[fam] = 0.0
                continue
            mean_ap = float(np.mean(per_family_ap[fam]))
            if args.weight_power == 0.0:
                raw_ws[fam] = 1.0
            else:
                raw_ws[fam] = max(mean_ap, 1e-6) ** args.weight_power
        total = sum(raw_ws.values())
        for fam in families:
            family_weights[(fam, cls)] = (raw_ws[fam] / total
                                            if total > 0 else 0.0)
    # Per-run weight inside its family (uniform across members)
    run_weight: dict[tuple[int, str], float] = {}
    for cls in all_classes:
        for fam in families:
            members = family_members.get((fam, cls), [])
            if not members:
                continue
            fw = family_weights[(fam, cls)]
            per_member = fw / len(members)
            for i in members:
                run_weight[(i, cls)] = per_member

    # Save weight tables (both family and effective per-run)
    w_csv = args.output_dir / "per_method_per_class_weights.csv"
    with open(w_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "family"] + list(all_classes))
        for i, m in enumerate(models):
            row = [run_weight.get((i, c), 0.0) for c in all_classes]
            w.writerow([m["name"], m["family"]]
                       + [f"{x:.4f}" for x in row])
    log(f"  saved per-run weights -> {w_csv}")

    fam_w_csv = args.output_dir / "per_family_per_class_weights.csv"
    with open(fam_w_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family"] + list(all_classes))
        for fam in families:
            row = [family_weights.get((fam, c), 0.0) for c in all_classes]
            w.writerow([fam] + [f"{x:.4f}" for x in row])
    log(f"  saved per-family weights -> {fam_w_csv}")

    # ── Step 6: Leave-one-out (LOO) recommendation diagnostic ──────────────
    if not args.skip_loo:
        log(f"\n[6/10] Leave-one-(family)-out on local data...")
        # Baseline ensemble AP per class (using current weights)
        def _baseline_ap_on_local() -> dict[str, float]:
            out_ap = {}
            for cls in all_classes:
                # Stack all participating local-calibrated scores per (idx, cls)
                participating = [i for i in range(M)
                                  if participates.get((i, cls), False)]
                if not participating:
                    out_ap[cls] = float("nan")
                    continue
                # Build per-family arithmetic mean, then geomean across families
                fam_to_idx: dict[str, list[int]] = defaultdict(list)
                for i in participating:
                    fam_to_idx[models[i]["family"]].append(i)
                # Pick canonical sample order from FIRST participating idx
                sc_first, mk_ref = local_calibrated[(participating[0], cls)]
                n_imgs = sc_first.shape[0]
                fam_means = []
                fam_names = []
                for fam, idxs in fam_to_idx.items():
                    fam_stack = np.stack([local_calibrated[(i, cls)][0]
                                           for i in idxs], axis=0)
                    fam_means.append(fam_stack.mean(axis=0))
                    fam_names.append(fam)
                fam_means = np.stack(fam_means, axis=0)  # (F, N, H, W)
                # Geomean across families weighted by w(fam, cls)
                ws = np.array([family_weights[(fam, cls)] for fam in fam_names],
                               dtype=np.float32)
                ws = ws / max(ws.sum(), 1e-12)
                if args.ensemble == "geomean":
                    ens = np.exp((ws[:, None, None, None]
                                  * np.log(fam_means + 1e-6)).sum(axis=0))
                else:
                    ens = (ws[:, None, None, None] * fam_means).sum(axis=0)
                out_ap[cls] = mean_pixel_ap(ens, mk_ref)
            return out_ap

        baseline_ap = _baseline_ap_on_local()
        baseline_mean = float(np.mean([v for v in baseline_ap.values()
                                         if np.isfinite(v)]))
        log(f"  baseline ensemble mean local AP = {baseline_mean:.4f}")

        # Build LOO over families (not individual runs) — that's what matters
        loo_rows = []
        for fam_to_drop in families:
            cls_aps = {}
            for cls in all_classes:
                participating = [i for i in range(M)
                                  if participates.get((i, cls), False)
                                  and models[i]["family"] != fam_to_drop]
                if not participating:
                    cls_aps[cls] = float("nan")
                    continue
                fam_to_idx: dict[str, list[int]] = defaultdict(list)
                for i in participating:
                    fam_to_idx[models[i]["family"]].append(i)
                mk_ref = local_calibrated[(participating[0], cls)][1]
                fam_means = []
                fam_names = []
                for fam, idxs in fam_to_idx.items():
                    fam_stack = np.stack([local_calibrated[(i, cls)][0]
                                           for i in idxs], axis=0)
                    fam_means.append(fam_stack.mean(axis=0))
                    fam_names.append(fam)
                fam_means = np.stack(fam_means, axis=0)
                # Recompute family weights from per-family-mean APs
                # excluding fam_to_drop
                raw_ws = {}
                for fam in fam_names:
                    aps_in_fam = [method_class_ap[(i, cls)]
                                   for i in fam_to_idx[fam]]
                    mean_ap = float(np.mean(aps_in_fam))
                    raw_ws[fam] = (1.0 if args.weight_power == 0.0
                                    else max(mean_ap, 1e-6)
                                            ** args.weight_power)
                total = sum(raw_ws.values())
                ws = np.array([raw_ws[fam] / total if total > 0 else 0.0
                                for fam in fam_names], dtype=np.float32)
                if args.ensemble == "geomean":
                    ens = np.exp((ws[:, None, None, None]
                                  * np.log(fam_means + 1e-6)).sum(axis=0))
                else:
                    ens = (ws[:, None, None, None] * fam_means).sum(axis=0)
                cls_aps[cls] = mean_pixel_ap(ens, mk_ref)
            loo_mean = float(np.mean([v for v in cls_aps.values()
                                       if np.isfinite(v)]))
            delta = loo_mean - baseline_mean
            loo_rows.append((fam_to_drop, loo_mean, delta, cls_aps))

        # Sort by delta DESC — biggest improvement from dropping at top
        loo_rows.sort(key=lambda r: r[2], reverse=True)

        loo_txt = args.output_dir / "loo_recommendations.txt"
        with open(loo_txt, "w") as f:
            f.write(f"Baseline (all families) mean local AP: "
                    f"{baseline_mean:.4f}\n\n")
            f.write(f"{'family':<14} {'mean AP':>10} {'delta':>10}  "
                    f"{'recommendation':<20}\n")
            f.write("-" * 60 + "\n")
            log(f"\n  LOO results (sorted by delta, biggest "
                f"improvement from dropping at top):")
            log(f"  {'family':<14} {'mean AP':>10} {'delta':>10}  "
                f"{'recommendation':<20}")
            for fam, mean_ap, delta, _ in loo_rows:
                if delta > 0.005:
                    rec = "CONSIDER DROPPING"
                elif delta > 0.001:
                    rec = "borderline"
                elif delta < -0.005:
                    rec = "keep (helps a lot)"
                else:
                    rec = "neutral"
                line = (f"  {fam:<14} {mean_ap:>10.4f} {delta:>+10.4f}  "
                        f"{rec:<20}")
                log(line)
                f.write(line.lstrip() + "\n")
            log(f"\n  saved LOO recommendations -> {loo_txt}")
    else:
        log(f"\n[6/10] SKIPPED leave-one-out diagnostic.")

    # We don't need local_calibrated for the rest of the pipeline
    del local_calibrated

    # ── Step 7: Streaming ensemble over TEST data ───────────────────────────
    log(f"\n[7/10] Streaming {args.ensemble} ensemble over test data...")
    # Strategy: family-by-family, compute family's arithmetic mean over its
    # member runs first (streamed), then accumulate into the cross-family
    # geomean / arithmetic mean.
    canon = models[0]
    canon_test = np.load(canon["test_npz_path"], allow_pickle=True)
    canon_ids     = np.asarray(canon_test["ids"])
    canon_classes = np.asarray(canon_test["classes"])
    log(f"  canonical id ordering from: {canon['name']}  "
        f"test_n={len(canon_ids)}")

    # Probe every method to find the largest test (H, W) per class.
    # That becomes the accumulator shape so we preserve full detail
    # from high-resolution methods (FastFlow / 518-input PatchCore) and
    # upsample low-resolution ones.
    cls_test_ref_shape: dict[str, tuple[int, int]] = {}
    for m_ in models:
        test = np.load(m_["test_npz_path"], allow_pickle=True)
        t_classes = np.asarray(test["classes"])
        t_scores = test["scores"]
        for cls in all_classes:
            cls_mask = t_classes == cls
            if not cls_mask.any():
                continue
            sh = t_scores[cls_mask].shape[1:]
            cur = cls_test_ref_shape.get(cls, (0, 0))
            if sh[0] * sh[1] > cur[0] * cur[1]:
                cls_test_ref_shape[cls] = (int(sh[0]), int(sh[1]))
        del test, t_scores, t_classes
    log(f"  per-class test reference shapes: {cls_test_ref_shape}")
    del canon_test

    # Per-class id ordering (canonical)
    class_idx_map: dict[str, np.ndarray] = {}
    class_ids_map: dict[str, np.ndarray] = {}
    for cls in all_classes:
        m_ = canon_classes == cls
        if m_.any():
            class_idx_map[cls] = np.where(m_)[0]
            class_ids_map[cls] = canon_ids[m_]

    # Initialise per-class accumulators at the reference shape per class
    accs: dict[str, dict] = {}
    for cls, cls_ids in class_ids_map.items():
        ref = cls_test_ref_shape.get(cls)
        if ref is None:
            continue
        H, W = ref
        accs[cls] = {
            "ids": cls_ids,
            "log_acc": np.zeros((len(cls_ids), H, W), dtype=np.float32),
            "lin_acc": np.zeros((len(cls_ids), H, W), dtype=np.float32),
            "weight_sum": 0.0,
        }

    # For each family, build the per-class mean of its members, then
    # weighted-merge into the accumulator.
    family_to_runs: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(models):
        family_to_runs[m["family"]].append(i)

    for fam_name, run_indices in family_to_runs.items():
        log(f"  family {fam_name} ({len(run_indices)} run(s))")
        # Per-class arithmetic mean of family members.
        for cls in all_classes:
            members = [i for i in run_indices
                        if participates.get((i, cls), False)]
            fam_w = family_weights.get((fam_name, cls), 0.0)
            if not members or fam_w <= 0 or cls not in accs:
                continue
            payload = accs[cls]
            cls_ids = payload["ids"]
            ref_hw = cls_test_ref_shape[cls]
            fam_mean = np.zeros_like(payload["lin_acc"])
            n_loaded = 0
            for i in members:
                m_ = models[i]
                test = np.load(m_["test_npz_path"], allow_pickle=True)
                t_scores  = test["scores"].astype(np.float32)
                t_classes = np.asarray(test["classes"])
                t_ids     = np.asarray(test["ids"])
                cls_mask = t_classes == cls
                if not cls_mask.any():
                    continue
                m_ids = t_ids[cls_mask]
                m_scores = t_scores[cls_mask]
                m_cal = apply_ecdf(m_scores, m_["ecdf"][cls])
                # Resize to reference shape if this method outputs a
                # different resolution from the accumulator.
                if m_cal.shape[1:] != ref_hw:
                    m_cal = maybe_resize_stack(m_cal, ref_hw)
                # Align by id
                id_to_idx = {str(s): j for j, s in enumerate(m_ids)}
                aligned = np.empty_like(payload["lin_acc"])
                n_missing = 0
                for k, cid in enumerate(cls_ids):
                    j = id_to_idx.get(str(cid))
                    if j is None:
                        aligned[k] = 0.5
                        n_missing += 1
                    else:
                        aligned[k] = m_cal[j]
                if n_missing:
                    log(f"    WARN: {m_['name'][:50]} missing "
                        f"{n_missing}/{len(cls_ids)} ids for {cls}")
                fam_mean += aligned
                n_loaded += 1
                del test, t_scores, t_classes, t_ids, m_ids, m_scores, m_cal
            if n_loaded == 0:
                continue
            fam_mean /= n_loaded
            # Merge into accumulators (we keep both forms; choose at finalisation)
            payload["log_acc"] += fam_w * np.log(fam_mean + 1e-6)
            payload["lin_acc"] += fam_w * fam_mean
            payload["weight_sum"] += fam_w

    # Finalise
    final: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cls, payload in accs.items():
        ws = payload["weight_sum"]
        if ws <= 0:
            ens = np.zeros_like(payload["lin_acc"])
        elif args.ensemble == "geomean":
            ens = np.exp(payload["log_acc"] / ws).astype(np.float32)
        else:
            ens = (payload["lin_acc"] / ws).astype(np.float32)
        final[cls] = (payload["ids"], ens)
        del payload["log_acc"], payload["lin_acc"]
    log(f"  ensemble finalised for {len(final)} classes")

    # ── Step 8: Smooth ──────────────────────────────────────────────────────
    if args.smooth_sigma > 0:
        log(f"\n[8a/10] Smoothing (sigma={args.smooth_sigma})...")
        for cls in list(final):
            ids, ens = final[cls]
            for n in range(ens.shape[0]):
                ens[n] = gaussian_smooth_2d(ens[n], args.smooth_sigma)
            final[cls] = (ids, ens)

    # ── Step 9: Multiview sample-gate ───────────────────────────────────────
    if args.multiview != "none":
        log(f"\n[8b/10] Multiview sample-gate "
            f"(floor={args.mv_gate_floor})...")
        for cls in list(final):
            ids, ens = final[cls]
            ens = apply_multiview_gate(
                ens, ids,
                gate_floor=args.mv_gate_floor,
                tail_threshold=args.mv_tail_threshold,
                tail_norm_rate=args.mv_tail_norm_rate,
                log_fn=log, cls_label=cls)
            final[cls] = (ids, ens)
    else:
        log(f"\n[8b/10] Multiview gate disabled.")

    # ── Step 10a: Per-class background floor + small-blob removal ──────────
    if args.background_quantile > 0:
        log(f"\n[9a/10] Per-class background floor "
            f"(q={args.background_quantile})...")
        for cls in list(final):
            ids, ens = final[cls]
            floor = float(np.quantile(ens, args.background_quantile))
            ens = np.maximum(ens - floor, 0.0)
            final[cls] = (ids, ens)
            log(f"  {cls}: floor {floor:.4f}")
    if args.min_blob_size > 0:
        try:
            import scipy.ndimage  # noqa: F401
            scipy_ok = True
        except ImportError:
            scipy_ok = False
            log(f"\n[9b/10] min_blob_size={args.min_blob_size} requested but "
                f"scipy missing; skipping.")
        if scipy_ok:
            log(f"\n[9b/10] Small-blob removal "
                f"(min_size={args.min_blob_size})...")
            for cls in list(final):
                ids, ens = final[cls]
                n_changed = 0
                for n in range(ens.shape[0]):
                    before = ens[n].copy()
                    ens[n] = remove_small_blobs(
                        ens[n], args.blob_threshold,
                        args.min_blob_size, args.blob_attenuation)
                    if not np.array_equal(before, ens[n]):
                        n_changed += 1
                final[cls] = (ids, ens)
                log(f"  {cls}: blobs suppressed in {n_changed}/{len(ids)}")

    # ── Step 10b: Final per-class score calibration ────────────────────────
    # CRITICAL: the final calibration MUST preserve sparsity (most pixels
    # at exactly 0). q8rle compression depends on long runs of identical
    # values; a calibration that leaves 50% of pixels at non-zero
    # produces 20k+ RLE pairs per image and a 700 MB submission.
    #
    # We use percentile clipping with a HIGH lower bound (default p90)
    # so >90% of pixels collapse to 0, leaving only the actual anomaly
    # tail to carry meaningful values. Anomaly detection is a sparse-tail
    # problem: most pixels are background, only a few percent are
    # anomalous. The calibration must match that prior.
    if not args.no_final_calibration:
        log(f"\n[10a/10] Final per-class sparse-tail calibration "
            f"(lo=p{args.final_lo_pct}, hi=p{args.final_hi_pct})...")
        for cls in list(final):
            ids, ens = final[cls]
            lo = float(np.percentile(ens, args.final_lo_pct))
            hi = float(np.percentile(ens, args.final_hi_pct))
            if hi <= lo:
                hi = lo + 1e-6
            ens_cal = np.clip((ens - lo) / (hi - lo), 0.0, 1.0
                              ).astype(np.float32)
            final[cls] = (ids, ens_cal)
            n_active_pix = int((ens_cal > 0).sum())
            n_total_pix = int(ens_cal.size)
            log(f"  {cls}: lo={lo:.4f} hi={hi:.4f}  "
                f"active_pix={n_active_pix / n_total_pix:.2%}")
    else:
        log(f"\n[10a/10] SKIPPED final calibration.")
        for cls in list(final):
            ids, ens = final[cls]
            final[cls] = (ids, np.clip(ens, 0.0, 1.0))

    # Resize to submission resolution (224x224) — matches the per-method
    # submission.csv format used by the existing single-model runs.
    SUBMISSION_HW = (224, 224)
    log(f"\n[10b/10] Resizing to submission resolution {SUBMISSION_HW}...")
    for cls in list(final):
        ids, ens = final[cls]
        if ens.shape[1:] != SUBMISSION_HW:
            ens = maybe_resize_stack(ens, SUBMISSION_HW)
        final[cls] = (ids, ens)

    # ── Step 10c: Write submission.csv ──────────────────────────────────────
    # Pre-flight sanity check: estimate RLE pair count + density before
    # writing. If the maps are not sparse, the submission will be huge
    # AND the leaderboard score will collapse (uniform-rank submissions
    # destroy the global ranking the metric depends on).
    log(f"\n[10b/10] Pre-submission sanity check...")
    total_runs = 0
    n_imgs = 0
    sum_q = 0
    sum_pix = 0
    per_cls_mean_q = {}
    for cls in sorted(final):
        ids, ens = final[cls]
        cls_sum = 0
        cls_pix = 0
        for k in range(ens.shape[0]):
            arr = ens[k]
            q = np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8)
            cls_sum += int(q.sum())
            cls_pix += int(q.size)
            sum_q += int(q.sum())
            sum_pix += int(q.size)
            n_imgs += 1
            flat = q.T.reshape(-1)
            if flat.size > 0:
                # number of value-change positions == number of runs - 1
                total_runs += 1 + int((flat[1:] != flat[:-1]).sum())
        per_cls_mean_q[cls] = cls_sum / max(cls_pix, 1)

    median_pairs = total_runs / max(n_imgs, 1)
    overall_mean_q = sum_q / max(sum_pix, 1)
    est_csv_mb = (total_runs * 6) / 1e6
    log(f"  avg pairs/image  : {median_pairs:.1f}")
    log(f"  estimated CSV MB : {est_csv_mb:.1f}  (zip ~ 1/3 of that)")
    log(f"  overall mean q   : {overall_mean_q:.2f} / 255  "
        f"(healthy: 5-40; bad: >80)")
    log(f"  per-class mean q :")
    cls_q_values = list(per_cls_mean_q.values())
    cv_q = (float(np.std(cls_q_values) / max(np.mean(cls_q_values), 1e-6))
            if cls_q_values else 0)
    for cls in sorted(per_cls_mean_q):
        log(f"    {cls}  mean_q={per_cls_mean_q[cls]:.2f}")
    log(f"  cross-class CV   : {cv_q:.3f}  (healthy: <0.3)")

    bad_signals = []
    if median_pairs > 5000:
        bad_signals.append(f"median pairs/image {median_pairs:.0f} >> 5000")
    if overall_mean_q > 80:
        bad_signals.append(f"overall mean q {overall_mean_q:.0f} >> 80")
    if cv_q > 0.5:
        bad_signals.append(f"cross-class CV {cv_q:.2f} >> 0.5")
    if bad_signals:
        log(f"")
        log(f"  ⚠️  PRE-SUBMISSION WARNINGS:")
        for s in bad_signals:
            log(f"      • {s}")
        log(f"  These are leading indicators of a broken submission. "
            f"Run diagnose_submission.py on the output before uploading.")
    else:
        log(f"  ✓ pre-submission sanity check passed")

    log(f"\n  Writing submission.csv...")
    csv_path = args.output_dir / "submission.csv"
    n_rows = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        for cls in sorted(final):
            ids, ens = final[cls]
            for k in range(len(ids)):
                w.writerow([str(ids[k]), float_matrix_to_q8rle(ens[k])])
                n_rows += 1
    csv_mb = csv_path.stat().st_size / 1e6
    log(f"  wrote {n_rows} rows -> {csv_path}  ({csv_mb:.1f} MB)")
    if not args.no_zip:
        zip_path = csv_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w",
                              compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
        zip_mb = zip_path.stat().st_size / 1e6
        log(f"  zipped         -> {zip_path}  ({zip_mb:.1f} MB)")

    log(f"\nDone in {(time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())