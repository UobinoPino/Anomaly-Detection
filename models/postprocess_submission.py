"""Post-process a Spacepresso submission.csv with the four priors of
Steps 1-4 of the improvement plan:

  Step 1. Multi-view score aggregation by `sample_id`.
          Group all (~5) test images sharing the same sample_id and
          compute a per-sample anomaly score (max of per-view top-k
          pooling). Boost / suppress each view's per-pixel scores by
          a multiplier in [floor, 1.0] depending on the sample's
          normalised anomaly score. Per-class normalisation by default
          so sparse-defect and dense-defect classes are treated fairly.

  Step 2. Foreground masking from train/good.
          Multiply each per-pixel score by `(1 - α) + α * fg_mask`,
          where `fg_mask in [0, 1]` is the precomputed foreground prior
          for this image's class. α is `--fg-strength`.

  Step 3. Connected-component (CC) filtering + morphological opening.
          Threshold the image at its `cc-threshold-pct` percentile (so
          this happens AFTER FG / centre / multi-view rescaling, on
          the actual ranking of scores). Open the binary mask with a
          square structuring element of size `--opening-size`. Drop
          CCs smaller than `--min-cc-area`. For pixels that WERE above
          threshold but DIDN'T survive opening + min-area filtering,
          multiply their score by `--dust-weight`. Conservative by
          design: pixels that were already below the local threshold
          are untouched.

  Step 4. Centre-bias prior.
          Multiply each per-pixel score by `(1 - β) + β * centre_heat`,
          where `centre_heat in [0, 1]` is the smoothed per-class anomaly
          occurrence prior. β is `--centre-strength`. Classes with no
          train_anomaly GT masks have centre_heat=ones (no-op) so the
          multiplier is exactly 1 for them regardless of β.

# Order of operations

The four operations don't commute. Default order:

    score -> FG mask -> centre prior -> CC filter -> multi-view

Rationale:
  - FG and centre priors are per-pixel multiplicative; doing them first
    means they reshape the score distribution before CC thresholding.
  - CC filter looks at the rescaled scores, so it suppresses background
    blobs that survived FG masking (e.g. on classes with poor FG masks).
  - Multi-view runs LAST: it pools per-view anomaly evidence after all
    the per-pixel cleanups, so the "is this sample anomalous overall"
    signal is computed from a clean per-pixel signal.

# I/O

Input  : `<run>/submission.csv` in the standard q8rle format.
Output : `<out_dir>/submission.csv` (+ `.zip` if not `--no-zip`) plus
         `<out_dir>/postprocess_config.json` recording every knob.

# What to skip

Any step can be made a no-op:
    --fg-strength 0
    --centre-strength 0
    --opening-size 0 --min-cc-area 0    (disables Step 3)
    --multiview-floor 1.0               (disables Step 1)

# Dependencies

  pip install scipy numpy pillow
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec — bit-identical to the rest of the codebase
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
# Submission I/O
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


def write_submission(rows: dict[str, str], out_path: Path,
                     zip_it: bool = True) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        for sid in sorted(rows):
            w.writerow([sid, rows[sid]])
    if zip_it:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w",
                             compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        return zip_path
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Class map (filename stem -> class_XX)
# ─────────────────────────────────────────────────────────────────────────────
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


def load_class_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"--class-map {path} not found")
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "ID" in row and "class" in row:
                out[row["ID"]] = row["class"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# View parsing — submission IDs are filename stems like "img_xxxx_view1"
# ─────────────────────────────────────────────────────────────────────────────
VIEW_STEM_RE = re.compile(r"^(?P<base>.+?)_view(?P<v>\d+)$")


def parse_view_stem(stem: str) -> tuple[str, int | None]:
    """Returns (sample_id, view_idx_or_None). Falls back to (stem, None)
    when no _viewN suffix is present, which makes the multi-view step a
    no-op for those entries (each is its own sample)."""
    m = VIEW_STEM_RE.match(stem)
    if m:
        return m.group("base"), int(m.group("v"))
    return stem, None


# ─────────────────────────────────────────────────────────────────────────────
# Priors loading
# ─────────────────────────────────────────────────────────────────────────────
def load_priors(priors_dir: Path | None,
                 classes: set[str]) -> dict[str, dict]:
    """Returns {class: {foreground, centre_heat, centre_bias_frac,
    fg_coverage, side}}. Missing classes are skipped silently."""
    if priors_dir is None or not priors_dir.exists():
        return {}
    out: dict[str, dict] = {}
    for cls in sorted(classes):
        path = priors_dir / f"{cls}_priors.npz"
        if not path.exists():
            print(f"  [warn] no priors file for {cls} ({path}); "
                  f"FG and centre priors will be no-ops for this class.")
            continue
        with np.load(path) as data:
            out[cls] = {
                "foreground":      np.asarray(data["foreground"], dtype=np.float32),
                "centre_heat":     np.asarray(data["centre_heat"], dtype=np.float32),
                "centre_bias_frac": float(data["centre_bias_frac"]),
                "fg_coverage":     float(data["fg_coverage"]),
                "n_masks":         int(data["n_masks"]),
                "side":            int(data["side"]),
            }
    return out


def _resize_2d(arr: np.ndarray, target_shape) -> np.ndarray:
    """Nearest-neighbour resize. Priors are smooth so NN is fine and
    avoids pulling in scipy.ndimage.zoom for a single call."""
    th, tw = target_shape
    h, w = arr.shape
    if (h, w) == (th, tw):
        return arr
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(arr.dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image post-processing (Steps 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_one_image(score: np.ndarray,
                            foreground: np.ndarray | None,
                            centre_heat: np.ndarray | None,
                            fg_strength: float,
                            centre_strength: float,
                            opening_size: int,
                            min_cc_area: int,
                            cc_threshold_pct: float,
                            dust_weight: float) -> np.ndarray:
    """Apply Steps 2 (foreground), 4 (centre), 3 (CC + opening) to one
    score map. Multi-view (Step 1) is applied globally afterwards so
    each sample's image-level signal is computed from cleaned maps."""
    H, W = score.shape
    out = score.astype(np.float32, copy=True)

    # Step 2 -- Foreground mask
    if foreground is not None and fg_strength > 0:
        fg = _resize_2d(foreground, (H, W))
        fg_mult = ((1.0 - fg_strength) + fg_strength * fg).astype(np.float32)
        out = out * fg_mult

    # Step 4 -- Centre prior
    if centre_heat is not None and centre_strength > 0:
        ch = _resize_2d(centre_heat, (H, W))
        ch_mult = ((1.0 - centre_strength) + centre_strength * ch).astype(np.float32)
        out = out * ch_mult

    # Step 3 -- CC filter + morphological opening (down-weight only the
    # pixels that were above the local threshold but didn't survive
    # opening + min-area filtering).
    do_cc = (opening_size > 0 or min_cc_area > 0) and dust_weight < 1.0
    if do_cc:
        thresh = float(np.percentile(out, cc_threshold_pct))
        if thresh > 0:
            binary_init = (out >= thresh).astype(np.uint8)
            binary_kept = binary_init.copy()
            if opening_size > 0:
                se = np.ones((opening_size, opening_size), dtype=bool)
                binary_kept = ndi.binary_opening(
                    binary_kept, structure=se).astype(np.uint8)
            if min_cc_area > 0:
                labels, n_cc = ndi.label(binary_kept)
                if n_cc > 0:
                    sizes = ndi.sum(binary_kept, labels,
                                     index=np.arange(1, n_cc + 1))
                    keep = np.where(sizes >= min_cc_area)[0] + 1
                    binary_kept = np.isin(labels, keep).astype(np.uint8)
            # Pixels that were ABOVE local threshold but got filtered.
            filtered_mask = (binary_init == 1) & (binary_kept == 0)
            if filtered_mask.any():
                out = np.where(filtered_mask,
                                out * dust_weight, out).astype(np.float32)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Multi-view aggregation
# ─────────────────────────────────────────────────────────────────────────────
def pool_view_score(sm: np.ndarray, mode: str) -> float:
    """Single scalar summary of one view's anomaly content."""
    if mode == "max":
        return float(sm.max())
    if mode == "p99":
        return float(np.percentile(sm, 99))
    if mode == "top0p1":
        k = max(int(sm.size * 0.001), 1)
        # partition is O(n); equivalent to sort + last-k mean for k pixels.
        return float(np.partition(sm.ravel(), -k)[-k:].mean())
    if mode == "top1":
        k = max(int(sm.size * 0.01), 1)
        return float(np.partition(sm.ravel(), -k)[-k:].mean())
    raise ValueError(f"unknown multiview pool: {mode!r}")


def multiview_aggregate(decoded: dict[str, np.ndarray],
                          *,
                          multiview_floor: float,
                          multiview_pool: str,
                          per_class_normalise: bool,
                          class_map: dict[str, str] | None
                          ) -> tuple[dict[str, np.ndarray], dict]:
    """Per-sample anomaly score (max over views) drives a multiplicative
    boost on each view's pixel scores. Per-class normalisation is the
    default because Spacepresso has very different anomaly prevalence
    across classes; without it the global max would come from one class
    and squash the others.

    Returns (rescaled_dict, stats).
    """
    if multiview_floor >= 1.0:
        return decoded, {"applied": False, "reason": "multiview_floor >= 1.0"}

    # Group by sample_id (parsed from the filename stem).
    by_sid: dict[str, list[str]] = defaultdict(list)
    for sid_full in decoded:
        sample_id, _v = parse_view_stem(sid_full)
        by_sid[sample_id].append(sid_full)

    # Pool every view to a scalar.
    view_pooled = {sid_full: pool_view_score(decoded[sid_full], multiview_pool)
                    for ids in by_sid.values() for sid_full in ids}

    # Per-sample summary = max across views.
    sample_max = {sample_id: max(view_pooled[i] for i in ids)
                   for sample_id, ids in by_sid.items()}

    # Normalise sample_max to [0, 1]. Per-class is more correct because
    # absolute score scales aren't comparable across classes.
    if per_class_normalise and class_map:
        sample_class: dict[str, str] = {}
        for sample_id, ids in by_sid.items():
            for i in ids:
                if i in class_map:
                    sample_class[sample_id] = class_map[i]
                    break
        by_class: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for sample_id, sc in sample_max.items():
            cls = sample_class.get(sample_id, "_default_")
            by_class[cls].append((sample_id, sc))
        sample_norm: dict[str, float] = {}
        per_class_stats: dict[str, dict] = {}
        for cls, items in by_class.items():
            vals = np.array([x[1] for x in items], dtype=np.float32)
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax <= vmin:
                normed = np.zeros_like(vals)
            else:
                normed = (vals - vmin) / (vmax - vmin)
            per_class_stats[cls] = {
                "n_samples": int(len(items)),
                "min": vmin, "max": vmax, "p50": float(np.percentile(vals, 50)),
            }
            for (sample_id, _), n in zip(items, normed):
                sample_norm[sample_id] = float(n)
        scope = "per-class"
    else:
        vals = np.array(list(sample_max.values()), dtype=np.float32)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            sample_norm = {s: 0.0 for s in sample_max}
        else:
            sample_norm = {s: float((v - vmin) / (vmax - vmin))
                            for s, v in sample_max.items()}
        per_class_stats = {"_global_": {
            "n_samples": int(len(sample_max)),
            "min": vmin, "max": vmax, "p50": float(np.percentile(vals, 50))}}
        scope = "global"

    # Apply: multiplier in [floor, 1.0] linear in sample_norm.
    rescaled: dict[str, np.ndarray] = {}
    factor_dist: list[float] = []
    for sample_id, ids in by_sid.items():
        f = multiview_floor + (1.0 - multiview_floor) * sample_norm[sample_id]
        factor_dist.append(f)
        for sid_full in ids:
            rescaled[sid_full] = np.clip(decoded[sid_full] * f, 0.0, 1.0)

    stats = {
        "applied": True,
        "scope": scope,
        "multiview_floor": multiview_floor,
        "multiview_pool": multiview_pool,
        "n_samples": len(by_sid),
        "factor_min": float(np.min(factor_dist)),
        "factor_p25": float(np.percentile(factor_dist, 25)),
        "factor_p50": float(np.percentile(factor_dist, 50)),
        "factor_p75": float(np.percentile(factor_dist, 75)),
        "factor_max": float(np.max(factor_dist)),
        "per_class_pool_stats": per_class_stats,
    }
    return rescaled, stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PostprocessConfig:
    in_path: Path
    out_path: Path
    priors_dir: Path | None
    data_root: Path
    class_map_csv: Path | None
    fg_strength: float
    centre_strength: float
    opening_size: int
    min_cc_area: int
    cc_threshold_pct: float
    dust_weight: float
    multiview_floor: float
    multiview_pool: str
    per_class_normalise: bool
    no_zip: bool


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in",  dest="in_path", type=Path, required=True,
                    help="Input submission.csv to post-process.")
    ap.add_argument("--out", dest="out_path", type=Path, required=True,
                    help="Output submission.csv (and .zip beside it).")
    ap.add_argument("--priors-dir", type=Path, default=None,
                    help="Directory of <class>_priors.npz files from "
                         "compute_priors.py. If unset, FG and centre "
                         "priors are skipped (no-op).")
    ap.add_argument("--data-root", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/data"),
                    help="Used to build the ID->class map by walking "
                         "class_XX/test/. Override with --class-map "
                         "for a precomputed mapping.")
    ap.add_argument("--class-map", type=Path, default=None,
                    help="Optional precomputed ID,class CSV.")
    # Step 2: foreground
    ap.add_argument("--fg-strength", type=float, default=0.5,
                    help="0 = disable. 1 = full multiplication by FG mask.")
    # Step 4: centre prior
    ap.add_argument("--centre-strength", type=float, default=0.3,
                    help="0 = disable. 1 = full multiplication by centre heat.")
    # Step 3: CC + opening
    ap.add_argument("--opening-size", type=int, default=3,
                    help="Morphological opening structuring-element size "
                         "(pixels). 0 disables opening.")
    ap.add_argument("--min-cc-area", type=int, default=8,
                    help="Min CC area (pixels) to retain at full score. "
                         "0 disables CC filtering.")
    ap.add_argument("--cc-threshold-pct", type=float, default=95.0,
                    help="Per-image percentile threshold for binarising "
                         "before CC analysis.")
    ap.add_argument("--dust-weight", type=float, default=0.2,
                    help="Multiplier applied to pixels that WERE above "
                         "the local threshold but got filtered out by "
                         "opening + min-area. 1.0 = no down-weighting.")
    # Step 1: multi-view
    ap.add_argument("--multiview-floor", type=float, default=0.5,
                    help="Per-view multiplier floor. Clean samples get "
                         "their pixels multiplied by this floor; the "
                         "most anomalous sample gets 1.0. 1.0 disables "
                         "the multi-view step.")
    ap.add_argument("--multiview-pool", default="top0p1",
                    choices=["max", "p99", "top0p1", "top1"],
                    help="How to summarise a view's anomaly content "
                         "into one scalar. top0p1 = mean of top 0.1%% "
                         "of pixels (robust to single bright outliers).")
    ap.add_argument("--no-per-class-normalise", action="store_true",
                    help="Use one global min/max for multi-view "
                         "normalisation. Default: per-class.")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    cfg = PostprocessConfig(
        in_path=args.in_path, out_path=args.out_path,
        priors_dir=args.priors_dir, data_root=args.data_root,
        class_map_csv=args.class_map,
        fg_strength=args.fg_strength,
        centre_strength=args.centre_strength,
        opening_size=args.opening_size,
        min_cc_area=args.min_cc_area,
        cc_threshold_pct=args.cc_threshold_pct,
        dust_weight=args.dust_weight,
        multiview_floor=args.multiview_floor,
        multiview_pool=args.multiview_pool,
        per_class_normalise=not args.no_per_class_normalise,
        no_zip=args.no_zip,
    )

    print("=" * 78)
    print("POSTPROCESS SUBMISSION")
    print("=" * 78)
    print(f"  in            : {cfg.in_path}")
    print(f"  out           : {cfg.out_path}")
    print(f"  priors_dir    : {cfg.priors_dir}")
    print(f"  fg_strength   : {cfg.fg_strength}")
    print(f"  centre_strength : {cfg.centre_strength}")
    print(f"  opening_size  : {cfg.opening_size}")
    print(f"  min_cc_area   : {cfg.min_cc_area}")
    print(f"  cc_threshold_pct : {cfg.cc_threshold_pct}")
    print(f"  dust_weight   : {cfg.dust_weight}")
    print(f"  multiview_floor : {cfg.multiview_floor}")
    print(f"  multiview_pool  : {cfg.multiview_pool}")
    print(f"  per_class_norm  : {cfg.per_class_normalise}")

    # ── Load class map
    print(f"\n[1/5] Building ID -> class map...")
    if cfg.class_map_csv and cfg.class_map_csv.exists():
        class_map = load_class_map(cfg.class_map_csv)
        print(f"      loaded {len(class_map)} from {cfg.class_map_csv}")
    else:
        class_map = build_class_map_from_data(cfg.data_root)
        print(f"      built {len(class_map)} from {cfg.data_root}")

    # ── Load priors
    print(f"\n[2/5] Loading priors...")
    classes_seen = set(class_map.values()) if class_map else set()
    priors = load_priors(cfg.priors_dir, classes_seen)
    if priors:
        print(f"      loaded priors for: {sorted(priors.keys())}")
        for cls in sorted(priors):
            p = priors[cls]
            print(f"        {cls}: fg_cov={p['fg_coverage']:.3f}  "
                  f"centre_bias={p['centre_bias_frac']:.3f}  "
                  f"n_masks={p['n_masks']}  side={p['side']}")
    else:
        print(f"      no priors loaded -- FG / centre steps are no-ops")

    # ── Load submission and decode
    print(f"\n[3/5] Decoding {cfg.in_path}...")
    t0 = time.time()
    sub = load_submission(cfg.in_path)
    print(f"      {len(sub)} rows")
    decoded: dict[str, np.ndarray] = {}
    for j, (sid, q) in enumerate(sub.items()):
        decoded[sid] = q8rle_to_float_matrix(q)
        if (j + 1) % 2000 == 0:
            print(f"        decoded {j + 1}/{len(sub)}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
    print(f"      decoded all in {time.time() - t0:.1f}s")

    # ── Per-image post-processing (Steps 2, 4, 3)
    print(f"\n[4/5] Per-image post-process (FG -> centre -> CC)...")
    t1 = time.time()
    n_missing_class = 0
    n_missing_prior = 0
    for j, (sid, sm) in enumerate(decoded.items()):
        cls = class_map.get(sid)
        fg = ch = None
        if cls is None:
            n_missing_class += 1
        else:
            p = priors.get(cls)
            if p is None:
                n_missing_prior += 1
            else:
                fg = p["foreground"]
                ch = p["centre_heat"]
        decoded[sid] = postprocess_one_image(
            sm,
            foreground=fg,
            centre_heat=ch,
            fg_strength=cfg.fg_strength,
            centre_strength=cfg.centre_strength,
            opening_size=cfg.opening_size,
            min_cc_area=cfg.min_cc_area,
            cc_threshold_pct=cfg.cc_threshold_pct,
            dust_weight=cfg.dust_weight,
        )
        if (j + 1) % 2000 == 0:
            print(f"        processed {j + 1}/{len(decoded)}  "
                  f"({time.time() - t1:.1f}s)", flush=True)
    print(f"      processed all in {time.time() - t1:.1f}s")
    if n_missing_class:
        print(f"      [warn] {n_missing_class} IDs had no class in the map "
              f"(post-process applied without FG/centre)")
    if n_missing_prior:
        print(f"      [warn] {n_missing_prior} IDs had a class but no "
              f"priors file (FG/centre skipped)")

    # ── Step 1: multi-view aggregation
    print(f"\n[5/5] Multi-view aggregation (per-class={cfg.per_class_normalise})...")
    t2 = time.time()
    decoded, mv_stats = multiview_aggregate(
        decoded,
        multiview_floor=cfg.multiview_floor,
        multiview_pool=cfg.multiview_pool,
        per_class_normalise=cfg.per_class_normalise,
        class_map=class_map,
    )
    print(f"      done in {time.time() - t2:.1f}s")
    if mv_stats["applied"]:
        print(f"      n_samples={mv_stats['n_samples']}  scope={mv_stats['scope']}")
        print(f"      multiplier distribution: "
              f"min={mv_stats['factor_min']:.3f}  "
              f"p25={mv_stats['factor_p25']:.3f}  "
              f"p50={mv_stats['factor_p50']:.3f}  "
              f"p75={mv_stats['factor_p75']:.3f}  "
              f"max={mv_stats['factor_max']:.3f}")
    else:
        print(f"      skipped: {mv_stats.get('reason')}")

    # ── Encode and write
    print(f"\nEncoding and writing...")
    t3 = time.time()
    rows = {sid: float_matrix_to_q8rle(sm) for sid, sm in decoded.items()}
    written = write_submission(rows, cfg.out_path, zip_it=not cfg.no_zip)
    print(f"      encoded + wrote in {time.time() - t3:.1f}s")
    print(f"      submission: {cfg.out_path}")
    if not cfg.no_zip:
        print(f"      zip       : {written}")

    # ── Persist exact config and multi-view stats for reproducibility
    cfg_json = {
        **{k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(cfg).items()},
        "multiview_stats": mv_stats,
        "n_rows": len(rows),
        "input_sha1": hashlib.sha1(str(cfg.in_path).encode()).hexdigest()[:8],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg_path = cfg.out_path.parent / "postprocess_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg_json, f, indent=2)
    print(f"      config    : {cfg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()