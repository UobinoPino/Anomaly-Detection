"""Compute per-class spatial priors from training data.

Writes one `<class>_priors.npz` file per class to `--out`, plus a
`_summary.json`. Each .npz contains:

  foreground       : (H, W) float32 in [0, 1]
                     1 = strong foreground, 0 = background.
  centre_heat      : (H, W) float32 in [0, 1] (max-normalised)
                     Smoothed empirical anomaly-occurrence prior.
  centre_bias_frac : float32 scalar
                     Fraction of anomaly-mask mass in the central
                     50% x 50% box. (Same metric as Section 7 of
                     analyze_spacepresso_dataset.py.)
  fg_coverage      : float32 scalar
                     Mean of the foreground mask. Cheap sanity:
                     coverage near 1.0 means FG masking has no effect
                     for that class.
  n_masks          : int32 scalar
                     Number of GT masks used for the centre heat.
                     If 0, centre_heat == all-ones (no-op multiplier).
  side             : int32 scalar
                     Spatial size of the saved priors. Must match the
                     downstream submission size (224 by default).

# Foreground mask construction

  1. Sample up to `--max-good-imgs` train/good images per class. Resize
     each to (side, side), convert to grayscale [0, 1].
  2. Per-pixel mean and std across the sample (streaming Welford to keep
     memory flat regardless of N).
  3. Background brightness estimate: median over the four corner crops
     (each 10% of side) of the per-pixel MEAN. This is robust to a few
     anomalous corners.
  4. Saliency = 0.5 * norm(std) + 0.5 * norm(|mean - bg_mean|).
        Captures pixels that either vary a lot across views (std signal)
        OR consistently deviate from background brightness (mean signal).
     Both terms are normalised to [0, 1] before averaging so neither
     dominates.
  5. Otsu-threshold the saliency to a binary FG mask.
  6. Light dilation (2 iters) + Gaussian smooth (sigma=3) -> soft mask
     in [0, 1]. Soft so the postprocess can interpolate between
     "no masking" and "full masking" via a single strength knob.

# Centre heat construction

  1. Aggregate all available train_anomaly GT masks for the class.
  2. Compute centre_bias_frac on the raw aggregate (BEFORE smoothing,
     so the metric matches the EDA report).
  3. Smooth with Gaussian sigma = side / 16. Result is suggestive, not
     surgical -- it can boost the central region by up to 2x but won't
     suppress peripheral defects.
  4. Renormalise so max = 1.0.
     Classes with no masks get centre_heat = all-ones; in postprocess,
     this means the centre-prior multiplier is exactly 1 regardless of
     centre-strength, so the prior is a no-op for those classes.

# CLI

    python compute_priors.py \\
        --data-root /work/.../data \\
        --out       /work/.../baseline_out/priors \\
        --side 224

Re-run only when training data changes -- the priors are independent of
which inference method produced your submission.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# Reuse the dataset scanner from the main codebase.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patchcore_baseline_v2 import scan_dataset, ImageRecord


# ─────────────────────────────────────────────────────────────────────────────
# Otsu (pure numpy — no skimage dependency)
# ─────────────────────────────────────────────────────────────────────────────
def otsu_threshold(x: np.ndarray) -> float:
    flat = np.asarray(x, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.5
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(flat, bins=256, range=(lo, hi + 1e-9))
    total = hist.sum()
    if total == 0:
        return float(flat.mean())
    hist = hist.astype(np.float64)
    bin_mids = (edges[:-1] + edges[1:]) * 0.5
    w1 = np.cumsum(hist) / total
    w2 = 1.0 - w1
    cum_mu = np.cumsum(hist * bin_mids) / total
    mu_total = (hist * bin_mids).sum() / total
    safe_w1 = np.maximum(w1, 1e-12)
    safe_w2 = np.maximum(w2, 1e-12)
    mu1 = cum_mu / safe_w1
    mu2 = (mu_total - cum_mu) / safe_w2
    sigma_b2 = w1 * w2 * (mu1 - mu2) ** 2
    sigma_b2[~np.isfinite(sigma_b2)] = -np.inf
    return float(bin_mids[int(np.argmax(sigma_b2))])


# ─────────────────────────────────────────────────────────────────────────────
# Foreground mask
# ─────────────────────────────────────────────────────────────────────────────
def compute_foreground(records_good: list[ImageRecord],
                        side: int = 224,
                        max_imgs: int = 400,
                        seed: int = 0) -> np.ndarray:
    """Returns (side, side) float32 in [0, 1]. 1 = strong foreground."""
    if not records_good:
        # Conservative default: no info -> treat everything as foreground.
        return np.ones((side, side), dtype=np.float32)

    rng = np.random.default_rng(seed)
    if len(records_good) > max_imgs:
        idx = rng.choice(len(records_good), size=max_imgs, replace=False)
        records_good = [records_good[i] for i in idx]

    n = 0
    mean = np.zeros((side, side), dtype=np.float64)
    m2   = np.zeros((side, side), dtype=np.float64)
    for r in records_good:
        try:
            with Image.open(r.path) as im:
                im_g = im.convert("L")
                if im_g.size != (side, side):
                    im_g = im_g.resize((side, side), Image.BILINEAR)
                g = np.asarray(im_g, dtype=np.float64) / 255.0
        except Exception as e:
            print(f"    [warn] could not read {r.path}: {e}")
            continue
        n += 1
        delta = g - mean
        mean += delta / n
        delta2 = g - mean
        m2 += delta * delta2

    if n == 0:
        return np.ones((side, side), dtype=np.float32)
    var = m2 / max(n, 1)
    std = np.sqrt(var)

    # Background brightness estimate from corner crops.
    s = max(int(side * 0.1), 4)
    corners = np.concatenate([
        mean[:s, :s].ravel(),
        mean[:s, -s:].ravel(),
        mean[-s:, :s].ravel(),
        mean[-s:, -s:].ravel(),
    ])
    bg_mean = float(np.median(corners))

    # Saliency: combine variance and deviation from background brightness.
    dev   = np.abs(mean - bg_mean)
    std_n = std / max(float(std.max()), 1e-9)
    dev_n = dev / max(float(dev.max()), 1e-9)
    saliency = (0.5 * std_n + 0.5 * dev_n).astype(np.float32)

    thresh = otsu_threshold(saliency)
    binary = (saliency > thresh).astype(np.float32)

    # Soften: dilate so we don't carve into objects, then Gaussian-smooth.
    binary_d = ndi.binary_dilation(binary, iterations=2).astype(np.float32)
    soft = ndi.gaussian_filter(binary_d, sigma=3.0)
    if soft.max() > 0:
        soft = soft / soft.max()
    return soft.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Centre heat
# ─────────────────────────────────────────────────────────────────────────────
def compute_centre_heat(records_anom: list[ImageRecord],
                         side: int = 224
                         ) -> tuple[np.ndarray, float, int]:
    """Returns (centre_heat in [0, 1], centre_bias_frac, n_masks)."""
    accum = np.zeros((side, side), dtype=np.float64)
    n = 0
    for r in records_anom:
        if r.mask_path is None:
            continue
        try:
            with Image.open(r.mask_path) as im:
                im_g = im.convert("L")
                if im_g.size != (side, side):
                    im_g = im_g.resize((side, side), Image.NEAREST)
                m = (np.asarray(im_g) > 127).astype(np.float64)
        except Exception:
            continue
        accum += m
        n += 1

    if n == 0:
        # No masks for this class -> all-ones means "no-op multiplier"
        # regardless of centre-strength at postprocess time.
        return np.ones((side, side), dtype=np.float32), 0.0, 0

    heat = accum / n
    total_mass = float(heat.sum())
    H = W = side
    central = float(heat[H // 4: 3 * H // 4, W // 4: 3 * W // 4].sum())
    centre_bias_frac = central / max(total_mass, 1e-12)

    smoothed = ndi.gaussian_filter(heat, sigma=side / 16.0)
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()
    else:
        smoothed = np.ones_like(smoothed)
    return smoothed.astype(np.float32), float(centre_bias_frac), int(n)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for <class>_priors.npz files.")
    ap.add_argument("--side", type=int, default=224,
                    help="Spatial size of the priors (must match the "
                         "submission size, which is 224 by default).")
    ap.add_argument("--max-good-imgs", type=int, default=400,
                    help="Cap on train/good images sampled per class.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"=" * 78)
    print(f"COMPUTE PRIORS")
    print(f"=" * 78)
    print(f"  data_root : {args.data_root}")
    print(f"  out       : {args.out}")
    print(f"  side      : {args.side}")
    print(f"  max_good  : {args.max_good_imgs}")

    print(f"\nScanning dataset...")
    records = scan_dataset(args.data_root)
    if not records:
        raise SystemExit(f"[FATAL] no records under {args.data_root}")
    classes = sorted({r.cls for r in records})
    print(f"  found {len(records)} records across {len(classes)} classes")

    summary: dict[str, dict] = {}
    t_total = time.time()
    for cls in classes:
        good = [r for r in records if r.cls == cls and r.split == "train_good"]
        anom = [r for r in records if r.cls == cls and r.split == "train_anomaly"]
        print(f"\n--- {cls}: train_good={len(good)}  train_anomaly={len(anom)} ---")

        t0 = time.time()
        print(f"  computing foreground mask...")
        fg = compute_foreground(good, side=args.side,
                                  max_imgs=args.max_good_imgs,
                                  seed=args.seed)
        fg_cov = float(fg.mean())
        print(f"    coverage (mean of mask): {fg_cov:.4f}  "
              f"({time.time() - t0:.1f}s)")

        t0 = time.time()
        print(f"  computing centre heat...")
        ch, cbf, n_masks = compute_centre_heat(anom, side=args.side)
        print(f"    n_masks={n_masks}  centre_bias_frac={cbf:.4f}  "
              f"({time.time() - t0:.1f}s)")

        out_path = args.out / f"{cls}_priors.npz"
        np.savez_compressed(
            out_path,
            foreground=fg,
            centre_heat=ch,
            centre_bias_frac=np.float32(cbf),
            fg_coverage=np.float32(fg_cov),
            n_masks=np.int32(n_masks),
            side=np.int32(args.side),
        )
        print(f"    saved -> {out_path}")
        summary[cls] = {
            "fg_coverage": fg_cov,
            "centre_bias_frac": cbf,
            "n_masks": n_masks,
        }

    summary_path = args.out / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")
    print(f"\nTotal: {time.time() - t_total:.1f}s")
    print(f"\nUsage:")
    print(f"  python postprocess_submission.py \\")
    print(f"      --in   <submission.csv> \\")
    print(f"      --out  <submission_pp.csv> \\")
    print(f"      --priors-dir {args.out} \\")
    print(f"      --data-root  {args.data_root}")


if __name__ == "__main__":
    main()