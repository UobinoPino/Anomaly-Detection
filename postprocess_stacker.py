#!/usr/bin/env python3
"""postprocess_stacker.py — learn-then-apply per-class postprocess
for an XGBoost stacker submission.

# What this does (project advice 08/05 + 10/05)

Reads three things:
  1. {stacker_dir}/oof_predictions.npz   — per-class out-of-fold preds
                                            + GT labels (1-D flat).
                                            Produced by xgboost_stacker.py
                                            when --tune-mode per-class
                                            OR --calibrate != none.
  2. {stacker_dir}/submission.csv        — test predictions from the
                                            same stacker run.
  3. one of the per-method local_predictions.npz files used by the
     stacker — only to read the spatial (H, W) shape of OOF preds
     (the stacker doesn't save it explicitly).

For each class, it grid-searches three postprocess knobs on the
local-val OOF predictions:

  - Gaussian smooth sigma        — silences pixel-level noise
                                   ("smooth the dust")
  - CC + opening + min_cc_area   — silences tiny isolated blobs
                                   ("silence the small ghosts")
  - dust_weight                  — multiplicative downweight applied
                                   to pixels that were above the local
                                   threshold but got filtered out by
                                   opening + min-area

The pixel-AP-maximising configuration per class is then applied to
the test submission and written to {out_dir}/submission.csv (+.zip).

# Why a *learned* postprocess

Pixel-AP is rank-based. Per-image AP is invariant to any strictly
monotonic operation on the score map; the benefit of postprocess
comes from cross-image rank realignment AND from killing isolated
high-value noise that competes with true positives in defective
images. Wrong hyperparameters can hurt (over-smoothing erases small
defects). Per-class tuning lets sparse-defect classes keep small
CCs while textured classes downweight them aggressively.

# Validation logic

For each class:
  Stage 1 (sigma sweep, no CC filter):
    AP = pool all OOF preds + labels across images, single sklearn
         average_precision_score call. The right local proxy for
         the leaderboard's global pixel-AP.
  Stage 2 (CC sweep at the Stage-1-best sigma):
    iterate (dust_weight, min_cc_area, opening_size, cc_pct);
    accept ONLY if AP strictly improves.

If no setting improves the no-op AP, the class gets no-op config
(sigma=0, no CC filter) — a strict improvement-only safety net so
postprocessing can never lower local AP, only the leaderboard AP can
disagree (and only by overfitting to local val, which we try to
limit by using few enough hyperparameters).

# CLI

    python postprocess_stacker.py \\
        --stacker-dir /work/.../runs/<stacker_run> \\
        --local-preds-ref /work/.../runs/<any_method>/local_predictions.npz \\
        --data-root /work/.../data \\
        --out-dir /work/.../runs/<stacker_run>_pp
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec — bit-identical to xgboost_stacker.py / patchcore_baseline_v2.py
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
# I/O helpers
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
# Resize helpers
# ─────────────────────────────────────────────────────────────────────────────
def nn_resize_2d(arr: np.ndarray, target_shape, dtype) -> np.ndarray:
    """Nearest-neighbour resize. Used for GT masks (preserve 0/1)."""
    th, tw = target_shape
    h, w = arr.shape
    if (h, w) == (th, tw):
        return arr.astype(dtype, copy=False)
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)


def bilinear_resize_2d(arr: np.ndarray, target_shape) -> np.ndarray:
    """Bilinear-ish resize for score maps via scipy.ndimage.zoom."""
    th, tw = target_shape
    h, w = arr.shape
    if (h, w) == (th, tw):
        return arr.astype(np.float32, copy=False)
    zh = th / h
    zw = tw / w
    return ndi.zoom(arr.astype(np.float32), (zh, zw),
                    order=1, mode="reflect").astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# The postprocess itself — one image
# ─────────────────────────────────────────────────────────────────────────────
def apply_postprocess(score: np.ndarray, *,
                       sigma: float,
                       opening_size: int,
                       min_cc_area: int,
                       cc_threshold_pct: float,
                       dust_weight: float) -> np.ndarray:
    """Apply: Gaussian smooth → CC + opening filter (multiplicative
    dust downweight on filtered-out pixels). Output is clipped to
    [0, 1] for q8rle encoding."""
    out = score.astype(np.float32, copy=True)

    # ── Gaussian smooth (handles 08/05: "smooth the dust")
    if sigma > 0.0:
        out = ndi.gaussian_filter(out, sigma=sigma,
                                    mode="reflect").astype(np.float32)

    # ── CC + opening (handles 10/05: "silence the small ghosts")
    do_cc = (opening_size > 0 or min_cc_area > 0) and (dust_weight < 1.0)
    if do_cc and cc_threshold_pct < 100.0:
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
            # Pixels that were above the local threshold but got filtered:
            filtered = (binary_init == 1) & (binary_kept == 0)
            if filtered.any():
                out = np.where(filtered, out * dust_weight, out
                                ).astype(np.float32)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class grid search
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_config_on_class(oof_preds: np.ndarray,
                               oof_labels: np.ndarray,
                               sigma: float,
                               opening_size: int,
                               min_cc_area: int,
                               cc_threshold_pct: float,
                               dust_weight: float) -> float:
    """Pool all images of the class into one pixel-AP call. This
    matches how the leaderboard computes pixel-AP (global pooling)
    better than a per-image average."""
    from sklearn.metrics import average_precision_score
    if oof_labels.sum() == 0:
        return 0.0
    if sigma == 0.0 and dust_weight >= 1.0:
        flat_p = oof_preds.ravel()
    else:
        processed = np.empty_like(oof_preds)
        for i in range(oof_preds.shape[0]):
            processed[i] = apply_postprocess(
                oof_preds[i],
                sigma=sigma, opening_size=opening_size,
                min_cc_area=min_cc_area,
                cc_threshold_pct=cc_threshold_pct,
                dust_weight=dust_weight)
        flat_p = processed.ravel()
    return float(average_precision_score(oof_labels.ravel(), flat_p))


def grid_search_class(oof_preds, oof_labels, *,
                        sigmas, dust_weights, min_cc_areas,
                        opening_sizes, cc_percentiles):
    """Two-stage coordinate descent. Returns (best_cfg, best_ap,
    base_ap, history)."""
    base_ap = evaluate_config_on_class(
        oof_preds, oof_labels,
        sigma=0.0, opening_size=0, min_cc_area=0,
        cc_threshold_pct=100.0, dust_weight=1.0)

    history = [{"stage": 0, "sigma": 0.0, "opening_size": 0,
                "min_cc_area": 0, "cc_threshold_pct": 100.0,
                "dust_weight": 1.0, "ap": base_ap}]

    # ── Stage 1: sigma sweep with no CC filter
    best_sigma = 0.0
    best_ap = base_ap
    for sigma in sigmas:
        if sigma == 0.0:
            ap = base_ap
        else:
            ap = evaluate_config_on_class(
                oof_preds, oof_labels,
                sigma=sigma, opening_size=0, min_cc_area=0,
                cc_threshold_pct=100.0, dust_weight=1.0)
        history.append({"stage": 1, "sigma": sigma, "opening_size": 0,
                         "min_cc_area": 0, "cc_threshold_pct": 100.0,
                         "dust_weight": 1.0, "ap": ap})
        if ap > best_ap:
            best_ap = ap
            best_sigma = sigma

    # ── Stage 2: CC sweep at the Stage-1-best sigma
    best_cfg = {"sigma": best_sigma, "opening_size": 0,
                 "min_cc_area": 0, "cc_threshold_pct": 100.0,
                 "dust_weight": 1.0}
    for dw in dust_weights:
        if dw >= 1.0:
            continue
        for mca in min_cc_areas:
            for op_size in opening_sizes:
                if mca == 0 and op_size == 0:
                    continue  # no CC filter active
                for pct in cc_percentiles:
                    ap = evaluate_config_on_class(
                        oof_preds, oof_labels,
                        sigma=best_sigma, opening_size=op_size,
                        min_cc_area=mca, cc_threshold_pct=pct,
                        dust_weight=dw)
                    history.append({"stage": 2, "sigma": best_sigma,
                                     "opening_size": op_size,
                                     "min_cc_area": mca,
                                     "cc_threshold_pct": pct,
                                     "dust_weight": dw, "ap": ap})
                    if ap > best_ap:
                        best_ap = ap
                        best_cfg = {"sigma": best_sigma,
                                     "opening_size": op_size,
                                     "min_cc_area": mca,
                                     "cc_threshold_pct": pct,
                                     "dust_weight": dw}
    return best_cfg, best_ap, base_ap, history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stacker-dir", type=Path, required=True,
                    help="Stacker run directory containing "
                         "oof_predictions.npz AND submission.csv.")
    ap.add_argument("--local-preds-ref", type=Path, required=True,
                    help="Any local_predictions.npz file the stacker used "
                         "(we only need its spatial (H, W) shape, which is "
                         "the OOF resolution).")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Data root to build the ID -> class map.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for the postprocessed submission.")
    ap.add_argument("--target-size", type=int, default=None,
                    help="Resolution to tune at. If unset, auto-detected "
                         "from the first row of the test submission "
                         "(typically 224).")
    # Grid
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--dust-weights", type=float, nargs="+",
                    default=[1.0, 0.5, 0.3])
    ap.add_argument("--min-cc-areas", type=int, nargs="+",
                    default=[0, 4, 8, 16])
    ap.add_argument("--opening-sizes", type=int, nargs="+",
                    default=[0, 3])
    ap.add_argument("--cc-percentiles", type=float, nargs="+",
                    default=[95.0, 98.0])
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("POSTPROCESS STACKER  —  learn-then-apply per-class postprocess")
    print("=" * 78)
    print(f"  stacker_dir     : {args.stacker_dir}")
    print(f"  local_preds_ref : {args.local_preds_ref}")
    print(f"  data_root       : {args.data_root}")
    print(f"  out_dir         : {args.out_dir}")

    # ── 1. Load OOF predictions
    oof_path = args.stacker_dir / "oof_predictions.npz"
    if not oof_path.exists():
        raise SystemExit(
            f"[FATAL] {oof_path} not found.\n"
            f"        Re-run xgboost_stacker.py with --tune-mode per-class\n"
            f"        OR --calibrate != none so the OOF predictions get\n"
            f"        saved.")
    sub_path = args.stacker_dir / "submission.csv"
    if not sub_path.exists():
        raise SystemExit(f"[FATAL] {sub_path} not found.")

    print(f"\nLoading OOF predictions from {oof_path}...")
    oof_data = np.load(oof_path, allow_pickle=True)
    classes_in_oof = [str(c) for c in oof_data["classes"]]
    print(f"  found OOF for classes: {classes_in_oof}")

    # ── 2. Load reference shape from local_preds_ref
    print(f"\nReading OOF spatial shape from {args.local_preds_ref}...")
    if not args.local_preds_ref.exists():
        raise SystemExit(f"[FATAL] {args.local_preds_ref} not found.")
    ref = np.load(args.local_preds_ref, allow_pickle=True)
    H_oof, W_oof = int(ref["scores"].shape[1]), int(ref["scores"].shape[2])
    print(f"  OOF resolution: {H_oof}x{W_oof}")

    # ── 3. Detect target tuning resolution (from submission shape)
    print(f"\nProbing submission shape from first row of {sub_path}...")
    with open(sub_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr)
        first_row = next(rdr)
    first_mat = q8rle_to_float_matrix(first_row[1])
    sub_h, sub_w = first_mat.shape
    print(f"  submission resolution: {sub_h}x{sub_w}")
    target_size = args.target_size if args.target_size else sub_h
    if target_size != sub_h:
        print(f"  [info] tuning at {target_size}x{target_size} per --target-size; "
              f"submission is {sub_h}x{sub_w}.")

    # ── 4. Reshape OOF per class and resize to target_size
    print(f"\nReshaping + resizing OOF to {target_size}x{target_size}...")
    oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cls in classes_in_oof:
        preds_flat = oof_data[f"oof_preds_{cls}"]
        labels_flat = oof_data[f"oof_labels_{cls}"]
        if preds_flat.size != labels_flat.size:
            raise SystemExit(
                f"[FATAL] {cls}: OOF preds size {preds_flat.size} != "
                f"labels size {labels_flat.size}")
        per_img = H_oof * W_oof
        if preds_flat.size % per_img != 0:
            raise SystemExit(
                f"[FATAL] {cls}: OOF size {preds_flat.size} not divisible\n"
                f"        by H*W = {per_img}. The --local-preds-ref shape\n"
                f"        {H_oof}x{W_oof} probably doesn't match what the\n"
                f"        stacker used. Try a different reference file\n"
                f"        (it must come from one of the methods listed in\n"
                f"        {args.stacker_dir}/stacker_config.json).")
        n_imgs = preds_flat.size // per_img
        preds_3d = preds_flat.reshape(n_imgs, H_oof, W_oof)
        labels_3d = labels_flat.reshape(n_imgs, H_oof, W_oof)
        if (H_oof, W_oof) != (target_size, target_size):
            preds_resized = np.stack([
                bilinear_resize_2d(p, (target_size, target_size))
                for p in preds_3d])
            labels_resized = np.stack([
                nn_resize_2d(l, (target_size, target_size), np.uint8)
                for l in labels_3d])
            oof_per_class[cls] = (preds_resized, labels_resized)
        else:
            oof_per_class[cls] = (preds_3d.astype(np.float32),
                                    labels_3d.astype(np.uint8))
        pos_frac = float(labels_3d.mean()) * 100
        print(f"  {cls}: {n_imgs:>3d} OOF images,  "
              f"{pos_frac:>6.3f}% positive pixels")

    # ── 5. Per-class grid search
    print(f"\n{'=' * 78}")
    print(f"PER-CLASS GRID SEARCH")
    print(f"{'=' * 78}")
    print(f"  sigmas         : {args.sigmas}")
    print(f"  dust_weights   : {args.dust_weights}")
    print(f"  min_cc_areas   : {args.min_cc_areas}")
    print(f"  opening_sizes  : {args.opening_sizes}")
    print(f"  cc_percentiles : {args.cc_percentiles}")

    best_per_class: dict[str, dict] = {}
    all_history: dict[str, list] = {}
    t0 = time.time()
    for cls, (preds, labels) in oof_per_class.items():
        if labels.sum() == 0:
            print(f"\n--- {cls}: no positives in OOF, using no-op ---")
            best_per_class[cls] = {
                "sigma": 0.0, "opening_size": 0, "min_cc_area": 0,
                "cc_threshold_pct": 100.0, "dust_weight": 1.0,
                "base_ap": 0.0, "best_ap": 0.0, "delta_ap": 0.0,
            }
            continue
        print(f"\n--- {cls}: searching ({preds.shape[0]} OOF images) ---")
        t_cls = time.time()
        cfg, ap, base_ap, history = grid_search_class(
            preds, labels,
            sigmas=args.sigmas, dust_weights=args.dust_weights,
            min_cc_areas=args.min_cc_areas,
            opening_sizes=args.opening_sizes,
            cc_percentiles=args.cc_percentiles)
        cfg_full = dict(cfg)
        cfg_full["base_ap"] = base_ap
        cfg_full["best_ap"] = ap
        cfg_full["delta_ap"] = ap - base_ap
        best_per_class[cls] = cfg_full
        all_history[cls] = history
        print(f"  baseline AP : {base_ap:.4f}")
        print(f"  best AP     : {ap:.4f}   (Δ = {ap - base_ap:+.4f})")
        print(f"  best config : sigma={cfg['sigma']}, "
              f"opening={cfg['opening_size']}, "
              f"min_cc_area={cfg['min_cc_area']}, "
              f"cc_pct={cfg['cc_threshold_pct']}, "
              f"dust_weight={cfg['dust_weight']}")
        print(f"  search time : {time.time() - t_cls:.1f}s")
    print(f"\n  total search time: {time.time() - t0:.1f}s")

    # ── Summary table
    print(f"\n{'=' * 78}")
    print(f"PER-CLASS RESULTS")
    print(f"{'=' * 78}")
    print(f"  {'class':<12} {'baseline':>10} {'best':>10} {'delta':>10}  config")
    total_delta = 0.0
    base_aps = []
    best_aps = []
    for cls in sorted(best_per_class):
        c = best_per_class[cls]
        cfg_str = (f"σ={c['sigma']}, op={c['opening_size']}, "
                    f"mca={c['min_cc_area']}, pct={c['cc_threshold_pct']}, "
                    f"dw={c['dust_weight']}")
        print(f"  {cls:<12} {c['base_ap']:>10.4f} {c['best_ap']:>10.4f} "
              f"{c['delta_ap']:>+10.4f}  {cfg_str}")
        total_delta += c["delta_ap"]
        base_aps.append(c["base_ap"])
        best_aps.append(c["best_ap"])
    if base_aps:
        print(f"  {'MEAN':<12} {np.mean(base_aps):>10.4f} "
              f"{np.mean(best_aps):>10.4f} "
              f"{np.mean(best_aps) - np.mean(base_aps):>+10.4f}")

    # ── 6. Apply to test submission
    print(f"\n{'=' * 78}")
    print(f"APPLY TO TEST SUBMISSION")
    print(f"{'=' * 78}")
    print(f"\nLoading {sub_path}...")
    sub = load_submission(sub_path)
    print(f"  {len(sub)} rows")
    print(f"\nBuilding ID -> class map from {args.data_root}...")
    class_map = build_class_map_from_data(args.data_root)
    print(f"  {len(class_map)} entries")
    n_unmapped = sum(1 for sid in sub if sid not in class_map)
    if n_unmapped:
        print(f"  [warn] {n_unmapped} submission IDs not in class map; "
              f"those will use no-op config.")

    no_op_cfg = {"sigma": 0.0, "opening_size": 0, "min_cc_area": 0,
                  "cc_threshold_pct": 100.0, "dust_weight": 1.0}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "submission.csv"
    print(f"\nWriting {out_csv}...")
    n_done = 0
    n_no_op = 0
    t1 = time.time()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        for sid in sorted(sub):
            cls = class_map.get(sid)
            cfg = best_per_class.get(cls) if cls else None
            if cfg is None:
                cfg = no_op_cfg
                n_no_op += 1
            sm = q8rle_to_float_matrix(sub[sid])
            out = apply_postprocess(
                sm,
                sigma=cfg["sigma"],
                opening_size=cfg["opening_size"],
                min_cc_area=cfg["min_cc_area"],
                cc_threshold_pct=cfg["cc_threshold_pct"],
                dust_weight=cfg["dust_weight"],
            )
            w.writerow([sid, float_matrix_to_q8rle(out)])
            n_done += 1
            if n_done % 1000 == 0:
                print(f"  processed {n_done}/{len(sub)}  "
                      f"({time.time() - t1:.1f}s)", flush=True)
    print(f"  wrote {n_done} rows in {time.time() - t1:.1f}s "
          f"({n_no_op} used no-op fallback)")

    if not args.no_zip:
        zip_path = out_csv.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w",
                              compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_csv, arcname=out_csv.name)
        print(f"  zipped  -> {zip_path}")

    # ── 7. Save report + history
    report = {
        "version": 1,
        "stacker_dir": str(args.stacker_dir),
        "local_preds_ref": str(args.local_preds_ref),
        "out_dir": str(args.out_dir),
        "target_size": int(target_size),
        "oof_resolution": [int(H_oof), int(W_oof)],
        "submission_resolution": [int(sub_h), int(sub_w)],
        "grid": {
            "sigmas": list(args.sigmas),
            "dust_weights": list(args.dust_weights),
            "min_cc_areas": list(args.min_cc_areas),
            "opening_sizes": list(args.opening_sizes),
            "cc_percentiles": list(args.cc_percentiles),
        },
        "per_class_best": best_per_class,
        "n_unmapped_test_ids": int(n_unmapped),
        "n_no_op_applied": int(n_no_op),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = args.out_dir / "postprocess_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  report  -> {report_path}")
    history_path = args.out_dir / "postprocess_history.json"
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"  history -> {history_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()