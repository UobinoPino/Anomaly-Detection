"""multiview_consensus.py — v4: per-view affine calibration (CORRECTED).

Drop-in replacement for v3. Public API names are unchanged so no edits
to patchcore_baseline_v2.py are required.

What was wrong with v3
======================
Two design errors specific to this dataset killed AP:

  (1) The "per-view rank-normalisation" path used np.searchsorted to map
      raw scores to a sketched train_good CDF, with values above the LUT
      max implicitly clipped to rank 1.0. Anomalies — which by
      DEFINITION score above the train_good distribution — all saturated
      to 1.0, as did any normal pixel whose feature happened to exceed
      train_good's empirical maximum. Within that tied group, sklearn's
      average_precision_score cannot rank true positives above false
      positives, so per-image pixel-AP collapsed even though the
      transform was "monotonic in principle".

      Observed: class_01 baseline AP=0.5768 → consensus-v3 AP=0.0653.

  (2) The "good-filtered sibling bank" assumed each sample's 5 views are
      slight perturbations of the same scene (e.g. rotation + lighting).
      They are not. Each view is a DIFFERENT CAMERA ANGLE of the object
      (top, side, bottom, ...). After L2-normalising backbone features,
      patches from different angles are near-orthogonal regardless of
      anomaly status, so mv_inc ≈ 1 everywhere, dominated by view
      geometry rather than defect signal. Adding alpha * mv_inc to
      rank_np pushed everything to saturation, and the multiplicative
      agreement boost in (A) then clipped the rest.

What this version does
======================
A single defensible operation: per-view AFFINE calibration using
train_good percentiles, applied WITHOUT clipping at the output.

  For each view v, fit lo_v = p1, hi_v = p99.5 of PatchCore scores on
  train_good of that view. At inference:

      s'(x, y) = (s(x, y) - lo_v) / (hi_v - lo_v)

  No clipping. The transform is strictly monotonic within each image,
  so PER-IMAGE PIXEL-AP IS INVARIANT to it (modulo float noise). Across
  views, the scales are now comparable — a score s'=0.8 in view 1
  means the same "rarity wrt train_good" as in view 5. The downstream
  global submission quantisation (calibrate_to_unit) then pools
  comparable values across views.

How this honours the teacher's advice
=====================================
  • Advice 15/05 ("calibration across views"): handled. Per-view
    percentiles from train_good are exactly the "validation statistics"
    the advice asks for. The calibration step that matters — global
    quantisation at submission time — now sees scores on a common scale.

  • Advice 12/05 ("five windows"): partially handled. Per-view stats are
    computed FROM the five-view structure of train_good. Full sample-
    level cross-view aggregation belongs in the stacker, which sees
    per-image predictions for all 5 views of every sample and can learn
    when agreement should boost confidence. Per-pixel cross-view
    feature comparison is fundamentally inappropriate when views are
    different camera angles with no spatial correspondence — that is
    why v3's sibling-bank failed.

Backward compatibility
======================
The public API names are unchanged:

    fit_per_view_norm()                 returns {view: (lo, hi)} now
    apply_per_view_norm()               applies the affine
    score_records_by_sample_consensus() unchanged signature
    save_view_luts() / load_view_luts() unchanged signatures

The CLI flags --mv-alpha, --mv-beta, --good-keep-frac, --agreement-pct,
--agreement-thresh are NO-OPS in this version. They are accepted by
argparse only for backward compatibility.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Per-view affine calibration
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def fit_per_view_norm(pc,
                       train_good_records: list,
                       batch_size: int = 16,
                       num_workers: int = 2,
                       max_imgs_per_view: int = 200,
                       max_pixels_per_view: int = 1_500_000,
                       lo_pct: float = 1.0,
                       hi_pct: float = 99.5,
                       seed: int = 0,
                       **_unused) -> dict[int, tuple[float, float]]:
    """Compute per-view (lo, hi) PatchCore-score percentiles on train_good.

    Returns {view_index: (lo, hi)} where lo = p1, hi = p99.5 of upsampled
    PatchCore distance maps over a subset of train_good images for that
    view. These are the "validation statistics" used by
    apply_per_view_norm to map raw inference scores to a per-view-
    calibrated scale before downstream quantisation.

    `_unused` swallows legacy kwargs like `n_quantiles=` from v3 callers.
    """
    # Lazy import to avoid circular dependency.
    from patchcore_baseline_v2 import make_loader

    by_view: dict[int, list] = defaultdict(list)
    for r in train_good_records:
        by_view[r.view if r.view is not None else 0].append(r)

    rng = np.random.default_rng(seed)
    view_stats: dict[int, tuple[float, float]] = {}
    for v in sorted(by_view):
        recs = by_view[v]
        if not recs:
            continue
        if len(recs) > max_imgs_per_view:
            idx = rng.choice(len(recs), size=max_imgs_per_view, replace=False)
            recs = [recs[i] for i in idx]
        loader = make_loader(recs,
                              batch_size=batch_size,
                              input_size=pc.cfg.input_size,
                              num_workers=num_workers,
                              load_masks=False, shuffle=False)
        chunks: list[np.ndarray] = []
        for x, _, _ in loader:
            sm = pc._score_one_pass(x).numpy().astype(np.float32)
            chunks.append(sm.reshape(-1))
        flat = (np.concatenate(chunks) if chunks
                else np.zeros(1, dtype=np.float32))
        if flat.size > max_pixels_per_view:
            idx = rng.choice(flat.size, size=max_pixels_per_view,
                              replace=False)
            flat = flat[idx]
        lo = float(np.percentile(flat, lo_pct))
        hi = float(np.percentile(flat, hi_pct))
        if hi <= lo:
            hi = lo + 1e-6
        view_stats[v] = (lo, hi)
        print(f"      [per-view-affine] view={v}  n_imgs={len(recs)}  "
              f"n_pixels={flat.size:,}  p{lo_pct:g}={lo:.5f}  "
              f"p{hi_pct:g}={hi:.5f}")
    return view_stats


def apply_per_view_norm(score_map: np.ndarray,
                         view,
                         view_stats: dict) -> np.ndarray:
    """Affine-calibrate a single score map using per-view (lo, hi).

    KEY PROPERTY: no clipping. The output is a strictly monotonic
    function of the input within the image, so per-image pixel-AP is
    invariant. Values can fall below 0 (very-clean pixels) or above 1
    (anomalies); downstream global calibrate_to_unit handles the [0, 1]
    clip-and-quantise step.

    Falls back to per-image min-max scaling when the view is unknown
    (still monotonic, but loses cross-view comparability).
    """
    s = score_map.astype(np.float32)
    if view is None or view not in view_stats:
        lo, hi = float(s.min()), float(s.max())
        return ((s - lo) / max(hi - lo, 1e-9)).astype(np.float32)
    lo, hi = view_stats[view]
    return ((s - lo) / max(hi - lo, 1e-9)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring routing — drop-in for the v3 entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def score_records_by_sample_consensus(pc,
                                       records,
                                       cfg,
                                       load_masks: bool,
                                       view_stats: dict
                                       ) -> tuple[dict, dict]:
    """Standard per-image PatchCore scoring (with TTA), followed by
    per-view affine calibration. No sample grouping, no sibling-bank,
    no agreement boost — those were the v3 failure modes.

    Returns (idx -> calibrated score map, idx -> gt mask). Per-image
    pixel-AP on the returned maps reproduces the baseline-multiview-
    none AP (within float noise) because the calibration is monotonic
    per image. The leaderboard value comes from the cross-view scale
    matching after the global submission quantisation, not from any
    change to local AP.
    """
    from patchcore_baseline_v2 import make_loader

    loader = make_loader(records,
                          batch_size=cfg.score_batch_size,
                          input_size=cfg.input_size,
                          num_workers=cfg.num_workers,
                          load_masks=load_masks)
    scores: dict[int, np.ndarray] = {}
    gts: dict[int, np.ndarray] = {}
    n_done = 0
    last_log = 0
    with torch.inference_mode():
        for x, masks, idxs in loader:
            sm = pc.score_batch(x, tta=cfg.tta).numpy()
            m_np = masks.numpy()
            for b in range(sm.shape[0]):
                idx = int(idxs[b])
                r = records[idx]
                scores[idx] = apply_per_view_norm(sm[b], r.view, view_stats)
                gts[idx] = m_np[b]
            n_done += sm.shape[0]
            if n_done - last_log >= 200:
                last_log = n_done
                print(f"      scored {n_done}/{len(records)}", flush=True)
            if getattr(cfg, "aggressive_cleanup", False) \
                    and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return scores, gts


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation — names unchanged for backward compat
# ─────────────────────────────────────────────────────────────────────────────
def save_view_luts(view_stats: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    views = sorted(view_stats.keys())
    los = np.asarray([view_stats[v][0] for v in views], dtype=np.float32)
    his = np.asarray([view_stats[v][1] for v in views], dtype=np.float32)
    np.savez_compressed(path,
                         views=np.asarray(views, dtype=np.int32),
                         los=los, his=his)
    print(f"      [per-view-affine] saved -> {path}")


def load_view_luts(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    views = [int(v) for v in z["views"].tolist()]
    los = z["los"].tolist()
    his = z["his"].tolist()
    return {v: (float(lo), float(hi))
            for v, lo, hi in zip(views, los, his)}