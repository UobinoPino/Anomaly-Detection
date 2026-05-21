"""Ensemble multiple anomaly-detection runs into a single submission.

Pipeline (each step maps to an advice line)
-------------------------------------------
  1. Per-(method, class) ECDF anchored on normal pixels in
     local_predictions.npz                                  [15/05]
  2. Per-(method, class) pixel-AP on local data → weights w = AP^gamma
  3. Streaming weighted geometric mean (or arithmetic mean)
     across methods, per-pixel, one method at a time         [14/05]
  4. Light Gaussian smoothing
  5. **Multiview sample-gate**: per-sample (5-views) evidence
     gates pixels of clearly-normal samples down toward
     gate_floor without penalising views that legitimately
     cannot see the defect                                   [12/05]
  6. Per-class background floor (silence below class quantile) [08/05]
  7. Small-blob removal (optional, scipy required)           [10/05]
  8. Final per-class ECDF (cross-class fairness)             [15/05]
  9. q8rle quantise + write submission.csv                   [07/05]

Inputs
------
Each --runs <dir> argument must point to a run directory containing:
    local_predictions.npz   — train_anomaly raw scores + GT masks
    test_predictions.npz    — test raw scores (add via test_preds_saver)

Usage
-----
    python submit_ensemble.py \\
        --runs /work/.../runs/20260521-... \\
               /work/.../runs/20260521-... \\
               ... \\
        --output-dir /work/.../baseline_out/ensembles/v1

Knobs (all optional, defaults reflect the advice above):
    --ensemble {geomean,mean}            default geomean
    --weight-power FLOAT                 default 2.0; 0 = equal weights
    --smooth-sigma FLOAT                 default 1.0
    --multiview {none,sample-gate}       default sample-gate
    --mv-gate-floor FLOAT                default 0.3 (attenuation floor)
    --mv-tail-threshold FLOAT            default 0.99 (tail definition)
    --mv-tail-norm-rate FLOAT            default 0.01 (expected normal rate)
    --background-quantile FLOAT          default 0.5 (0 disables)
    --min-blob-size INT                  default 0 (off; needs scipy)
    --blob-threshold FLOAT               default 0.5
    --blob-attenuation FLOAT             default 0.3
    --no-final-calibration               skip final per-class ECDF
    --no-zip                             skip submission.zip
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
    """Strip "_viewNN" suffix to get the sample id. Returns the input
    unchanged if it doesn't look like a multi-view stem."""
    m = SAMPLE_ID_RE.match(id_stem)
    return m.group("base") if m else id_stem


# ─────────────────────────────────────────────────────────────────────────────
# ECDF helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_ecdf(values: np.ndarray, n_bins: int = 4096, seed: int = 0):
    """Return (sorted_values, cdf) for fast np.interp evaluation."""
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
# Multiview sample-gate
# ─────────────────────────────────────────────────────────────────────────────
def sample_evidence_from_tail(views: np.ndarray, threshold: float,
                                norm_rate: float) -> float:
    """How anomalous is this sample? In [0, 1].

    After per-class ECDF on NORMAL pixels, the expected fraction of
    pixels above `threshold` for a purely normal sample is ~ (1 -
    threshold) = `norm_rate`. Anomalous samples have substantially more
    mass in that tail. We measure the excess and normalise so that
    `norm_rate` (purely normal) → 0 and 5 × norm_rate (clearly
    anomalous) → 1.

    Args:
        views    : (V, H, W) ensemble scores in roughly [0, 1].
        threshold: pixels above this count as tail.
        norm_rate: expected tail fraction under the normal hypothesis.

    Returns:
        evidence in [0, 1].
    """
    all_pix = views.ravel()
    frac_above = float((all_pix > threshold).sum() / all_pix.size)
    excess = max(frac_above - norm_rate, 0.0)
    denom = max(4.0 * norm_rate, 1e-8)        # 5× → 1 (4× excess)
    return float(min(excess / denom, 1.0))


def apply_multiview_gate(ens: np.ndarray, ids: np.ndarray,
                          gate_floor: float, tail_threshold: float,
                          tail_norm_rate: float,
                          log_fn=None, cls_label: str = "") -> np.ndarray:
    """Apply a per-sample multiplicative gate.

    For each sample (group of views sharing a sample id):
      gate = gate_floor + (1 - gate_floor) * sample_evidence
      out[view] = ens[view] * gate

    A clearly-normal sample's views are attenuated toward gate_floor.
    A clearly-anomalous sample's views pass through unchanged. A view
    that legitimately can't see the defect (e.g. defect on the
    opposite side) still receives the same sample-level multiplier as
    its siblings, so we never punish "legitimate silence".
    """
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
    """Suppress small isolated high-score blobs.

    Threshold the score map, find connected components, multiply
    pixels belonging to blobs smaller than min_size by `attenuation`
    (default 0.3). This implements "drop what is too small, too weak,
    or too lonely" without binary thresholding the output.
    """
    try:
        from scipy.ndimage import label, sum_labels
    except ImportError:
        return score  # silently skip
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
# Smoothing & q8rle (matches patchcore_baseline_v2 byte-for-byte)
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
    ap.add_argument("--runs", nargs="+", required=True, type=Path,
                    help="Run directories (each must contain "
                         "local_predictions.npz AND test_predictions.npz)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ensemble", default="geomean",
                    choices=["geomean", "mean"])
    ap.add_argument("--weight-power", type=float, default=2.0,
                    help="Exponent on per-class AP weights. 0 = equal.")
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    # Multiview
    ap.add_argument("--multiview", default="sample-gate",
                    choices=["none", "sample-gate"])
    ap.add_argument("--mv-gate-floor", type=float, default=0.3,
                    help="Attenuation floor for clearly-normal samples. "
                         "0.3 = attenuate normal samples to 30%. "
                         "Set 1.0 to disable gating entirely.")
    ap.add_argument("--mv-tail-threshold", type=float, default=0.99,
                    help="Pixel score threshold defining the tail.")
    ap.add_argument("--mv-tail-norm-rate", type=float, default=0.01,
                    help="Expected tail fraction under the normal "
                         "hypothesis (should equal 1 - mv_tail_threshold).")
    # Background floor + blob removal
    ap.add_argument("--background-quantile", type=float, default=0.5,
                    help="Subtract per-class quantile to silence "
                         "background; 0 disables.")
    ap.add_argument("--min-blob-size", type=int, default=0,
                    help="Suppress connected components smaller than this. "
                         "0 = disable. Requires scipy.")
    ap.add_argument("--blob-threshold", type=float, default=0.5)
    ap.add_argument("--blob-attenuation", type=float, default=0.3)
    # Misc
    ap.add_argument("--no-final-calibration", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(args.output_dir / "ensemble_log.txt")
    t0 = time.time()

    log(f"submit_ensemble.py  output={args.output_dir}")
    log(f"  ensemble = {args.ensemble}  weight_power = {args.weight_power}")
    log(f"  smooth_sigma = {args.smooth_sigma}")
    log(f"  multiview = {args.multiview} "
        f"(gate_floor={args.mv_gate_floor}, "
        f"tail_threshold={args.mv_tail_threshold}, "
        f"tail_norm_rate={args.mv_tail_norm_rate})")
    log(f"  background_quantile = {args.background_quantile}")
    log(f"  min_blob_size = {args.min_blob_size} "
        f"(threshold={args.blob_threshold}, "
        f"attenuation={args.blob_attenuation})")
    log()

    # ── Step 1: Load LOCAL predictions + remember test path ─────────────────
    log(f"[1/9] Loading {len(args.runs)} model runs (local only)...")
    models = []
    for rd in args.runs:
        local_p = rd / "local_predictions.npz"
        test_p  = rd / "test_predictions.npz"
        if not local_p.exists():
            log(f"  SKIP  {rd.name}: missing local_predictions.npz")
            continue
        if not test_p.exists():
            log(f"  SKIP  {rd.name}: missing test_predictions.npz — patch "
                f"this baseline with `from test_preds_saver import "
                f"save_test_predictions; save_test_predictions(...)` and "
                f"re-run the submission step.")
            continue
        local = np.load(local_p, allow_pickle=True)
        m = {
            "name": rd.name,
            "test_npz_path": test_p,
            "local_scores":  local["scores"].astype(np.float32),
            "local_classes": np.asarray(local["classes"]),
            "local_masks":   local["masks"].astype(np.uint8),
        }
        models.append(m)
        log(f"  OK    {rd.name}: local_n={len(m['local_classes'])}  "
            f"range=[{float(m['local_scores'].min()):.3g}, "
            f"{float(m['local_scores'].max()):.3g}]")

    if not models:
        log("\nNo usable runs. Aborting.")
        return 1
    M = len(models)
    log(f"\nUsing {M} models.")

    all_classes = sorted(set().union(
        *[set(np.unique(m["local_classes"]).tolist()) for m in models]))
    log(f"Classes detected: {all_classes}")

    # ── Step 2: Per-(method, class) ECDF from local normal pixels ───────────
    log(f"\n[2/9] Building per-(method, class) ECDF anchors...")
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
        log(f"  {m['name'][:70]:<70}  ECDFs={len(m['ecdf'])}")

    # ── Step 3: Per-(method, class) pixel-AP → weights ──────────────────────
    log(f"\n[3/9] Computing per-(method, class) pixel-AP on local...")
    method_class_ap = {}
    for i, m in enumerate(models):
        for cls in all_classes:
            cls_mask = m["local_classes"] == cls
            if not cls_mask.any() or cls not in m["ecdf"]:
                method_class_ap[(i, cls)] = 0.0
                continue
            sc = m["local_scores"][cls_mask]
            mk = m["local_masks"][cls_mask]
            sc_cal = apply_ecdf(sc, m["ecdf"][cls])
            method_class_ap[(i, cls)] = mean_pixel_ap(sc_cal, mk)

    name_w = max(20, min(60, max(len(m["name"]) for m in models) + 2))
    header = f"  {'method':<{name_w}}  " + "  ".join(
        f"{c:>10}" for c in all_classes) + f"  {'mean':>10}"
    log("")
    log(header)
    log("  " + "-" * (name_w + 12 * len(all_classes) + 14))
    for i, m in enumerate(models):
        row = [method_class_ap[(i, c)] for c in all_classes]
        log(f"  {m['name'][:name_w]:<{name_w}}  " +
            "  ".join(f"{a:>10.4f}" for a in row) +
            f"  {np.mean(row):>10.4f}")
    ap_csv = args.output_dir / "per_method_per_class_ap.csv"
    with open(ap_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + list(all_classes) + ["mean"])
        for i, m in enumerate(models):
            row = [method_class_ap[(i, c)] for c in all_classes]
            w.writerow([m["name"]] + [f"{a:.4f}" for a in row]
                       + [f"{np.mean(row):.4f}"])
    log(f"\n  saved AP table -> {ap_csv}")

    # ── Step 4: Per-class weights ───────────────────────────────────────────
    log(f"\n[4/9] Building per-class weights (gamma={args.weight_power})...")
    weights = {}
    for cls in all_classes:
        aps = np.array([method_class_ap[(i, cls)] for i in range(M)],
                       dtype=np.float64)
        if args.weight_power == 0.0:
            ws = np.ones_like(aps)
        else:
            ws = np.maximum(aps, 1e-6) ** args.weight_power
        s = ws.sum()
        ws = ws / s if s > 0 else np.ones_like(ws) / M
        for i in range(M):
            weights[(i, cls)] = float(ws[i])

    w_csv = args.output_dir / "per_method_per_class_weights.csv"
    with open(w_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + list(all_classes))
        for i, m in enumerate(models):
            row = [weights[(i, c)] for c in all_classes]
            w.writerow([m["name"]] + [f"{w_:.4f}" for w_ in row])
    log(f"  saved weights table -> {w_csv}")

    # ── Step 5: Streaming ensemble ──────────────────────────────────────────
    log(f"\n[5/9] Streaming {args.ensemble} ensemble (one model at a time)...")
    canon = models[0]
    canon_test = np.load(canon["test_npz_path"], allow_pickle=True)
    canon_ids     = np.asarray(canon_test["ids"])
    canon_classes = np.asarray(canon_test["classes"])
    canon_scores  = canon_test["scores"].astype(np.float32)
    log(f"  canonical: {canon['name']}  "
        f"test_n={len(canon_ids)}  HxW={canon_scores.shape[1:]}")

    accs = {}
    for cls in all_classes:
        cls_mask = canon_classes == cls
        if not cls_mask.any() or cls not in canon["ecdf"]:
            continue
        cls_ids = canon_ids[cls_mask]
        cls_scores = canon_scores[cls_mask]
        cls_cal = apply_ecdf(cls_scores, canon["ecdf"][cls])
        w = weights[(0, cls)]
        if args.ensemble == "geomean":
            acc = (w * np.log(cls_cal + 1e-6)).astype(np.float32)
        else:
            acc = (w * cls_cal).astype(np.float32)
        accs[cls] = {"ids": cls_ids, "acc": acc, "weight_sum": w}
    del canon_test, canon_scores, canon_classes

    for i in range(1, M):
        m = models[i]
        log(f"  [{i+1}/{M}] loading {m['name'][:60]} ...")
        test = np.load(m["test_npz_path"], allow_pickle=True)
        t_scores  = test["scores"].astype(np.float32)
        t_classes = np.asarray(test["classes"])
        t_ids     = np.asarray(test["ids"])

        for cls in all_classes:
            if cls not in accs or cls not in m["ecdf"]:
                continue
            payload = accs[cls]
            cls_mask = t_classes == cls
            if not cls_mask.any():
                continue
            m_ids = t_ids[cls_mask]
            m_scores = t_scores[cls_mask]
            m_cal = apply_ecdf(m_scores, m["ecdf"][cls])

            id_to_idx = {str(s): j for j, s in enumerate(m_ids)}
            n_missing = 0
            aligned = np.empty_like(payload["acc"])
            for k, cid in enumerate(payload["ids"]):
                j = id_to_idx.get(str(cid))
                if j is None:
                    aligned[k] = 0.5
                    n_missing += 1
                else:
                    aligned[k] = m_cal[j]
            if n_missing:
                log(f"    WARN: {m['name'][:50]} missing "
                    f"{n_missing}/{len(payload['ids'])} ids for {cls}")
            w = weights[(i, cls)]
            if args.ensemble == "geomean":
                payload["acc"] += w * np.log(aligned + 1e-6)
            else:
                payload["acc"] += w * aligned
            payload["weight_sum"] += w
        del test, t_scores, t_classes, t_ids

    # Finalise per-class ensemble
    final = {}
    for cls, payload in accs.items():
        ws = payload["weight_sum"]
        if ws <= 0:
            ens = np.zeros_like(payload["acc"])
        else:
            if args.ensemble == "geomean":
                ens = np.exp(payload["acc"] / ws).astype(np.float32)
            else:
                ens = (payload["acc"] / ws).astype(np.float32)
        final[cls] = (payload["ids"], ens)
        del payload["acc"]
    log(f"  ensemble finalised for {len(final)} classes")

    # ── Step 6: Smooth ──────────────────────────────────────────────────────
    if args.smooth_sigma > 0:
        log(f"\n[6/9] Smoothing (sigma={args.smooth_sigma})...")
        for cls in list(final):
            ids, ens = final[cls]
            for n in range(ens.shape[0]):
                ens[n] = gaussian_smooth_2d(ens[n], args.smooth_sigma)
            final[cls] = (ids, ens)

    # ── Step 7: Multiview sample-gate ───────────────────────────────────────
    if args.multiview != "none":
        log(f"\n[7/9] Multiview sample-gate "
            f"(floor={args.mv_gate_floor}, "
            f"tail={args.mv_tail_threshold} -> rate={args.mv_tail_norm_rate})...")
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
        log(f"\n[7/9] Multiview gate disabled.")

    # ── Step 8: Per-class background floor + small-blob removal ─────────────
    if args.background_quantile > 0:
        log(f"\n[8a/9] Per-class background floor "
            f"(q={args.background_quantile})...")
        for cls in list(final):
            ids, ens = final[cls]
            floor = float(np.quantile(ens, args.background_quantile))
            ens = np.maximum(ens - floor, 0.0)
            final[cls] = (ids, ens)
            log(f"  {cls}: floor {floor:.4f}")
    else:
        log(f"\n[8a/9] Background floor disabled.")

    if args.min_blob_size > 0:
        try:
            import scipy.ndimage  # noqa: F401
            scipy_ok = True
        except ImportError:
            scipy_ok = False
            log(f"\n[8b/9] min_blob_size={args.min_blob_size} requested but "
                f"scipy not installed; skipping blob removal.")
        if scipy_ok:
            log(f"\n[8b/9] Small-blob removal "
                f"(min_size={args.min_blob_size}, "
                f"threshold={args.blob_threshold}, "
                f"attenuation={args.blob_attenuation})...")
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
                log(f"  {cls}: small blobs suppressed in "
                    f"{n_changed}/{len(ids)} images")
    else:
        log(f"\n[8b/9] Small-blob removal disabled.")

    # ── Step 9a: Final per-class ECDF ───────────────────────────────────────
    if not args.no_final_calibration:
        log(f"\n[9a/9] Final per-class ECDF...")
        for cls in list(final):
            ids, ens = final[cls]
            ecdf = build_ecdf(ens.ravel())
            ens_cal = apply_ecdf(ens, ecdf)
            final[cls] = (ids, ens_cal)
            log(f"  {cls}: ECDF applied, range [0, 1]")
    else:
        log(f"\n[9a/9] SKIPPED final calibration.")
        for cls in list(final):
            ids, ens = final[cls]
            final[cls] = (ids, np.clip(ens, 0.0, 1.0))

    # ── Step 9b: Write submission.csv ───────────────────────────────────────
    log(f"\n[9b/9] Writing submission.csv...")
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
    log(f"  wrote {n_rows} rows -> {csv_path}")

    if not args.no_zip:
        zip_path = csv_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w",
                              compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
        log(f"  zipped         -> {zip_path}")

    log(f"\nDone in {(time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())