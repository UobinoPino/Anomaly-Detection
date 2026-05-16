"""multiview_consensus.py — A + B + D drop-in for patchcore_baseline_v2.

This module fixes the self-suppression flaw of the v6 `sibling-bank` path
(when an anomaly is visible in ≥2 views, anomalous patches find matches in
sibling-view anomalous patches → mv_inc collapses → boost ≈ 1-α →
the anomaly is SUPPRESSED exactly when most reliable). It replaces that
path with three composable fixes:

  (B) Per-view rank normalisation
      Each view position has its own empirical CDF of PatchCore scores
      built from train_good. Inference scores are mapped through that
      CDF so per-view scale drift (lighting/perspective/exposure) can
      no longer dominate cross-view aggregation. Output: scores in
      [0, 1] with a calibrated "rank in train_good" meaning.

  (D) Good-filtered sibling bank
      For view i, the sibling bank is built from sibling views' patches
      FILTERED to keep only those that look normal under M_good (the
      bottom `good_keep_frac` by raw PatchCore distance). Anomalous
      regions of sibling views are excluded, so an anomaly visible in
      3-5 views NO LONGER self-suppresses. mv_inc is added (not gated)
      on top of the rank-normalised score.

  (A) Agreement-only boost (never suppresses)
      Per-view "intensity" = high-percentile of its refined score map.
      `agreement` = fraction of views whose intensity > threshold.
      Final = refined * (1 + β * agreement). Boost ∈ [1, 1+β], so a
      view that uniquely sees a defect is preserved at its raw value.

Composition order:
    raw_i  →  rank-norm (B)  →  + α·mv_inc (D)  →  × (1 + β·agreement) (A)

All three are independent: mv_alpha=0 disables D, mv_beta=0 disables A,
and view_luts={} disables B (falls back to per-view min-max).

# Integration with patchcore_baseline_v2.py

Three edits — full diff at bottom of the docstring:

  1. In `RunConfig` add:
         mv_beta: float = 0.4
         good_keep_frac: float = 0.7
         agreement_pct: float = 95.0
         agreement_thresh: float = 0.90

  2. In `main()` argparse add:
         --multiview now accepts "consensus-v3"
         --mv-beta, --good-keep-frac, --agreement-pct, --agreement-thresh

  3. In `run_one_class()` after `pc.fit(train_good)`:
         view_luts = {}
         if cfg.multiview == "consensus-v3":
             from multiview_consensus import fit_per_view_norm
             view_luts = fit_per_view_norm(
                 pc, train_good,
                 batch_size=cfg.score_batch_size,
                 num_workers=cfg.num_workers)

     Then choose score_fn:
         if cfg.multiview == "consensus-v3":
             from multiview_consensus import score_records_by_sample_consensus
             def score_fn(_pc, records, _cfg, load_masks):
                 return score_records_by_sample_consensus(
                     _pc, records, _cfg, load_masks, view_luts)
         elif cfg.multiview == "sibling-bank":
             score_fn = _score_records_by_sample_pc
         else:
             score_fn = _score_records_standard_pc

# Cost

  Fit overhead: scoring train_good once per class (~1-3 min/class).
  Inference overhead vs sibling-bank path: ~5-10% (one extra topk +
  one extra matmul against a smaller bank).
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# (B) Per-view rank normalisation
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def fit_per_view_norm(pc,
                      train_good_records: list,
                      batch_size: int = 16,
                      num_workers: int = 2,
                      n_quantiles: int = 1024,
                      max_imgs_per_view: int = 200,
                      max_pixels_per_view: int = 1_500_000,
                      seed: int = 0,
                      ) -> dict[int, np.ndarray]:
    """Build per-view empirical-CDF LUTs from train_good PatchCore scores.

    Returns a dict {view_index: np.ndarray of length n_quantiles} where
    the LUT is a sorted monotonic sketch. Percentile rank of a score x:

        rank = searchsorted(lut, x, side='right') / len(lut)

    A score below the LUT minimum gets rank 0; above the max, rank 1.
    Both behaviours are what we want: clean pixels rank near 0 (≈
    indistinguishable from train_good), anomalous pixels saturate at 1.

    Subsampling:
      - At most `max_imgs_per_view` train_good images per view are scored
        (sufficient for percentile estimation, keeps fit time bounded).
      - At most `max_pixels_per_view` pixels feed the quantile sketch.

    Falls back gracefully: if all train_good records share view=None
    (e.g. single-view classes), one global LUT keyed at 0 is produced.
    """
    # Lazy import to avoid a circular dependency at module load time.
    from patchcore_baseline_v2 import make_loader

    by_view: dict[int, list] = defaultdict(list)
    for r in train_good_records:
        by_view[r.view if r.view is not None else 0].append(r)
 

    rng = np.random.default_rng(seed)
    view_luts: dict[int, np.ndarray] = {}
    for v in sorted(by_view):
        recs = by_view[v]
        if not recs:
            continue
        # Subsample images per view (200 is plenty for a CDF sketch).
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
            sm = pc._score_one_pass(x).numpy().astype(np.float32)  # (B,H,W) cpu
            chunks.append(sm.reshape(-1))
        flat = (np.concatenate(chunks).astype(np.float32)
                if chunks else np.zeros(1, dtype=np.float32))
        if flat.size > max_pixels_per_view:
            idx = rng.choice(flat.size, size=max_pixels_per_view, replace=False)
            flat = flat[idx]
        qs = np.linspace(0.0, 1.0, n_quantiles, endpoint=True)
        lut = np.quantile(flat, qs).astype(np.float32)
        # Defensive: ensure strict monotonicity for searchsorted stability.
        lut = np.maximum.accumulate(lut)
        view_luts[v] = lut
        print(f"      [per-view-norm] view={v}  n_imgs={len(recs)}  "
              f"n_pixels={flat.size:,}  "
              f"lut_range=[{lut[0]:.4f}, {lut[-1]:.4f}]")
    return view_luts


def apply_per_view_norm(score_map: np.ndarray,
                        view: Optional[int],
                        view_luts: dict[int, np.ndarray],
                        ) -> np.ndarray:
    """Map raw score_map → percentile rank in [0, 1] using the LUT for
    `view`. Falls back to per-image min-max if the view is unknown."""
    if view is None or view not in view_luts:
        s = score_map.astype(np.float32)
        lo, hi = float(s.min()), float(s.max())
        return ((s - lo) / max(hi - lo, 1e-9)).astype(np.float32)
    lut = view_luts[view]
    flat = score_map.ravel().astype(np.float32)
    ranks = np.searchsorted(lut, flat, side="right") / float(len(lut))
    return np.clip(ranks, 0.0, 1.0).reshape(score_map.shape).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# (D) good-filtered sibling-bank + (A) agreement boost
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def score_sample_consensus_v3(pc,
                              x_sample: torch.Tensor,
                              views: list[Optional[int]],
                              view_luts: dict[int, np.ndarray],
                              tta: str = "none",
                              mv_alpha: float = 0.5,
                              mv_beta: float = 0.4,
                              good_keep_frac: float = 0.7,
                              agreement_pct: float = 95.0,
                              agreement_thresh: float = 0.90,
                              ) -> np.ndarray:
    """Score all V views of one sample. Returns (V, H, H) np.float32 on CPU.

    Args:
        x_sample        : (V, 3, H, W) float tensor.
        views           : list of int|None, view index per row of x_sample.
        view_luts       : output of fit_per_view_norm.
        tta             : 'none' | 'hflip' | 'vflip' | 'hvflip' | 'd4'.
        mv_alpha        : (D) weight of filtered-sibling additive refinement.
                          0 disables D.
        mv_beta         : (A) max agreement boost; 0 disables A.
        good_keep_frac  : (D) fraction of lowest-score sibling patches kept
                          in the good-filtered sibling bank (per view).
        agreement_pct   : (A) percentile of each refined map used as that
                          view's anomaly intensity (default p95 — robust).
        agreement_thresh: (A) rank-norm threshold above which a view is
                          said to agree the sample is anomalous.

    Never multiplies by a value < 1, so unique-to-one-view defects are
    preserved at their per-view rank-normalised score.
    """
    device = pc.device
    x_sample = x_sample.to(device, non_blocking=True)
    H_in = pc.cfg.input_size

    # ── TTA-averaged per-view raw score + patch features ───────────────────
    acc_score = None
    acc_pf = None
    n_acc = 0

    def _add(s, pf):
        nonlocal acc_score, acc_pf, n_acc
        if acc_score is None:
            acc_score = s.clone(); acc_pf = pf.clone()
        else:
            acc_score = acc_score + s
            acc_pf = acc_pf + pf
        n_acc += 1

    def _inv_flip(pf2, dim_in_grid):
        Bf, Pf, Cf = pf2.shape
        Hf = Wf = int(math.isqrt(Pf))
        return torch.flip(pf2.reshape(Bf, Hf, Wf, Cf),
                          dims=[dim_in_grid]).reshape(Bf, Pf, Cf)

    def _inv_rot(pf2, k):
        Bf, Pf, Cf = pf2.shape
        Hf = Wf = int(math.isqrt(Pf))
        return torch.rot90(pf2.reshape(Bf, Hf, Wf, Cf),
                           k=-k, dims=[-3, -2]).reshape(Bf, Pf, Cf)

    s, pf = pc._compute_score_and_pf(x_sample)
    _add(s, pf)
    if tta in ("hflip", "hvflip", "d4"):
        s2, pf2 = pc._compute_score_and_pf(torch.flip(x_sample, dims=[-1]))
        _add(torch.flip(s2, dims=[-1]), _inv_flip(pf2, -2))
    if tta in ("vflip", "hvflip", "d4"):
        s2, pf2 = pc._compute_score_and_pf(torch.flip(x_sample, dims=[-2]))
        _add(torch.flip(s2, dims=[-2]), _inv_flip(pf2, -3))
    if tta == "d4":
        for k in (1, 2, 3):
            s2, pf2 = pc._compute_score_and_pf(
                torch.rot90(x_sample, k=k, dims=[-2, -1]))
            _add(torch.rot90(s2, k=-k, dims=[-2, -1]), _inv_rot(pf2, k))

    score_lr = acc_score / n_acc                   # (V, h, w) GPU
    pf = F.normalize(acc_pf / n_acc, p=2, dim=-1)  # (V, P, C) GPU L2-normed
    V, P, C = pf.shape
    h = w = int(math.isqrt(P))

    # Upsample raw score to input resolution.
    score_up = F.interpolate(score_lr.unsqueeze(1),
                             size=(H_in, H_in),
                             mode="bilinear", align_corners=False).squeeze(1)
    score_np = score_up.detach().cpu().numpy().astype(np.float32)

    # ── (B) per-view rank normalisation ───────────────────────────────────
    rank_np = np.empty_like(score_np)
    for i in range(V):
        rank_np[i] = apply_per_view_norm(score_np[i], views[i], view_luts)

    # ── (D) good-filtered sibling-bank additive refinement ────────────────
    if V >= 2 and mv_alpha > 0.0:
        score_lr_flat = score_lr.reshape(V, P)
        k_keep = max(int(round(good_keep_frac * P)), 1)
        # Per view: indices of bottom-k patches (lowest raw scores = most "good").
        _, idx_good = torch.topk(score_lr_flat, k_keep,
                                 dim=1, largest=False)
        mv_inc = torch.zeros(V, P, device=device, dtype=torch.float32)
        for i in range(V):
            sib_chunks = []
            for j in range(V):
                if j == i:
                    continue
                sib_chunks.append(pf[j].index_select(0, idx_good[j]))
            sib_bank = torch.cat(sib_chunks, dim=0)         # (M, C)
            sim = pf[i].float() @ sib_bank.float().T        # (P, M)
            max_sim = sim.max(dim=1).values                 # (P,)
            mv_inc[i] = 1.0 - max_sim
            del sib_chunks, sib_bank, sim, max_sim

        mv_inc_2d = mv_inc.reshape(V, h, w)
        mv_inc_up = F.interpolate(
            mv_inc_2d.unsqueeze(1), size=(H_in, H_in),
            mode="bilinear", align_corners=False
        ).squeeze(1).detach().cpu().numpy().astype(np.float32)
        # Absolute mv_inc: ~0 for normal patches, ~0.3-0.5 at anomalies.
        # Additive on top of the rank-normalised score, clipped to [0, 1].
        refined = np.clip(rank_np + mv_alpha * mv_inc_up, 0.0, 1.0)
        del mv_inc, mv_inc_2d
    else:
        refined = rank_np

    # ── (A) agreement-only boost ──────────────────────────────────────────
    if V >= 2 and mv_beta > 0.0:
        per_view_intensity = np.percentile(
            refined.reshape(V, -1), agreement_pct, axis=1)  # (V,)
        n_agree = int((per_view_intensity > agreement_thresh).sum())
        agreement = float(n_agree) / float(V)
        boost = 1.0 + mv_beta * agreement
        final = np.clip(refined * boost, 0.0, 1.0).astype(np.float32)
    else:
        final = refined

    del score_up, score_lr, pf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Wiring helper — drop-in replacement for _score_records_by_sample_pc
# ─────────────────────────────────────────────────────────────────────────────
def score_records_by_sample_consensus(pc,
                                      records: list,
                                      cfg,
                                      load_masks: bool,
                                      view_luts: dict[int, np.ndarray]
                                      ) -> tuple[dict, dict]:
    """Group records by sample_id, score all views together with the
    consensus-v3 path. Drop-in compatible with the existing dispatch in
    `run_one_class`. Returns (idx → score_map, idx → gt_mask).
    """
    from patchcore_baseline_v2 import _build_transform_pc, _load_one_pc

    by_sample: dict[str, list[tuple[int, "ImageRecord"]]] = defaultdict(list)
    for idx, r in enumerate(records):
        sid = r.sample_id or r.path.stem
        by_sample[sid].append((idx, r))
    print(f"      grouped {len(records)} images into {len(by_sample)} samples")

    transform = _build_transform_pc(cfg.input_size)
    scores: dict[int, np.ndarray] = {}
    gts: dict[int, np.ndarray] = {}
    n_done = 0
    last_log = 0

    mv_beta         = float(getattr(cfg, "mv_beta",          0.4))
    good_keep_frac  = float(getattr(cfg, "good_keep_frac",   0.7))
    agreement_pct   = float(getattr(cfg, "agreement_pct",    95.0))
    agreement_thresh= float(getattr(cfg, "agreement_thresh", 0.90))

    for sid, items in by_sample.items():
        # Deterministic view ordering (helps stable A% aggregation).
        items_sorted = sorted(items, key=lambda kv: (kv[1].view or 0))
        imgs = []; masks_np = []; views = []
        for _idx, r in items_sorted:
            x, m = _load_one_pc(r, transform, load_masks, cfg.input_size)
            imgs.append(x); masks_np.append(m); views.append(r.view)
        x_batch = torch.stack(imgs)

        sm = score_sample_consensus_v3(
            pc, x_batch, views=views, view_luts=view_luts,
            tta=cfg.tta,
            mv_alpha=cfg.mv_alpha,
            mv_beta=mv_beta,
            good_keep_frac=good_keep_frac,
            agreement_pct=agreement_pct,
            agreement_thresh=agreement_thresh,
        )
        for k, (idx, _r) in enumerate(items_sorted):
            scores[idx] = sm[k]
            gts[idx] = masks_np[k]
        n_done += len(items_sorted)
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
        if getattr(cfg, "aggressive_cleanup", False) and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores, gts


# ─────────────────────────────────────────────────────────────────────────────
# Optional: persist / restore view LUTs across runs
# ─────────────────────────────────────────────────────────────────────────────
def save_view_luts(view_luts: dict[int, np.ndarray], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        views=np.asarray(sorted(view_luts.keys())),
                        **{f"lut_{v}": view_luts[v]
                           for v in view_luts})
    print(f"      [per-view-norm] saved -> {path}")


def load_view_luts(path: Path) -> dict[int, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    views = [int(v) for v in z["views"].tolist()]
    return {v: z[f"lut_{v}"].astype(np.float32) for v in views}