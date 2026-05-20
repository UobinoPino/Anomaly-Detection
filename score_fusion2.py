"""Score fusion v3 — five strategies for maximising Pixel-Level Average Precision.

Pixel-AP ranks ALL pixels GLOBALLY across all test images, so the metric
rewards anything that improves cross-image / cross-method calibration and
suppresses noisy background activations.

v3 extends v2 (which fixed per-image rank-normalisation bugs) with FIVE
distinct fusion strategies and several post-processing knobs aligned with
the project advices:

  --strategy rank_wmean   Per-class weighted mean of global-rank-normalised
                          scores. The calibrated baseline. (Advice 15/05.)
  --strategy max          Pixel-wise max over methods whose per-class
                          weight is above a small floor. Aggressive: any
                          trusted method's flag is a flag.
  --strategy median       Pixel-wise median. Robust to a few catastrophic
                          methods (e.g. a flow model that hallucinates).
  --strategy agreement    Mean of the top-K scoring methods per pixel
                          (K = ceil(M * --top-k-frac)). Rewards rough
                          multi-method agreement without requiring all.
  --strategy consensus    Weighted GEOMETRIC mean. Rewards STRONG
                          multi-method agreement; near-zero where any
                          trusted method disagrees.

Post-processing (apply via flags; off by default unless the strategy
needs them; advices 08/05 + 10/05 + 12/05):
  --suppress-below PCT     Zero out everything below PCT-percentile per
                           fused image. 0 = off. 75 keeps the top 25%.
                           (Advice 08/05 — The Sparse Oracle.)
  --remove-small-cc PX     Drop connected components smaller than PX
                           pixels. 0 = off. Needs scipy.
                           (Advice 10/05 — Silence the Small Ghosts.)
  --multiview-regex RE     Regex with ONE capture group = object_id; IDs
                           sharing the same object_id are pooled.
                           Empty = off.
  --multiview-pool MODE    How to pool the 5 views of one object:
                             none  : leave each view independent
                             mean  : replace each view's map with the
                                     average across views
                             max   : replace with pixel-wise max
                             soft  : 50% per-view + 50% cross-view mean
                                     (most often best; preserves view
                                     specifics while sharing structure)
                           (Advice 12/05 — The Five Windows.)

v3 internals
------------
  * Decode every submission ONCE to uint8 (q8rle is already 8-bit, so
    nothing is lost). 14 methods × ~5910 images × 256² ≈ 5 GB.
  * Rank-normalisation uses a 256-bin histogram → 256-entry LUT applied
    per pixel. O(N) memory, ~200× faster than argsort on the full pixel
    stream, identical result up to ties.
  * Per-image fusion materialises only one (M, H, W) float32 stack at a
    time; freed before the next image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# q8rle strings can exceed csv's default 128 KB field limit.
csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec
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


def q8rle_to_uint8_matrix(s: str) -> np.ndarray:
    """Decode q8rle string to (H, W) uint8 array. Numpy-vectorised."""
    parts = s.split()
    h, w = int(parts[1]), int(parts[2])
    if len(parts) <= 3:
        return np.zeros((h, w), dtype=np.uint8)
    body = np.array(parts[3:], dtype=np.int64)
    vals = body[0::2].astype(np.uint8)
    lens = body[1::2]
    return np.repeat(vals, lens).reshape(w, h).T


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
            if len(row) < 2:
                continue
            out[row[0]] = row[1]
    return out


def load_local_eval(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ap = float(row["ap_mean"])
            except (KeyError, ValueError):
                continue
            out[row["class"]][row["anomaly_type"]] = ap
    return out


def per_class_ap_summary(eval_table: dict[str, dict[str, float]]) -> dict[str, float]:
    return {cls: float(np.mean(list(types.values())))
            for cls, types in eval_table.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Class mapping
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
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg",
                                                    ".bmp", ".tiff", ".webp"}:
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
# Weights
# ─────────────────────────────────────────────────────────────────────────────
def compute_weights(local_evals: list[dict[str, dict[str, float]]],
                    classes: list[str],
                    mode: str = "local_ap",
                    temperature: float = 1.0,
                    top_k: int | None = None) -> dict[str, list[float]]:
    """Per-class method weights. 'local_ap' uses each method's mean AP for
    that class (averaged over anomaly types). --top-k-methods-per-class
    hard-zeros all but the top-K per class; --weight-temperature applies
    softmax-with-temperature instead of linear normalisation."""
    M = len(local_evals)
    out: dict[str, list[float]] = {}
    if mode == "uniform" or not local_evals or any(not e for e in local_evals):
        for cls in classes:
            out[cls] = [1.0 / M] * M
        return out
    if mode == "local_ap":
        for cls in classes:
            aps = np.array([
                float(np.mean(list(e.get(cls, {}).values()))) if e.get(cls) else 0.0
                for e in local_evals
            ], dtype=np.float64)
            if top_k is not None and top_k < M:
                keep_idx = np.argsort(aps)[-top_k:]
                mask = np.zeros(M)
                mask[keep_idx] = 1.0
                aps = aps * mask
            if temperature != 1.0 and aps.max() > 0:
                z = aps / max(temperature, 1e-6)
                ex = np.exp(z - z.max())
                # zero out methods that were hard-zeroed by top_k
                ex = ex * (aps > 0)
                w = ex / (ex.sum() or 1.0)
            else:
                best = aps.max() if aps.size else 1.0
                floor = best / (M * 4) if best > 0 else 1e-6
                # only floor methods that survived top_k filtering
                aps_floored = np.where(aps > 0, np.maximum(aps, floor), 0.0)
                tot = aps_floored.sum() or 1.0
                w = aps_floored / tot
            out[cls] = w.tolist()
        return out
    raise ValueError(f"unknown weight mode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Global rank normalisation via 256-bin histogram LUT
# ─────────────────────────────────────────────────────────────────────────────
def compute_rank_lut(decoded_uint8: dict[str, np.ndarray]) -> np.ndarray:
    """Return a (256,) float32 LUT mapping uint8 score → global rank
    percentile (computed across all pixels of all test images for this
    method). Apply via `lut[uint8_matrix]`.

    Mathematically equivalent to argsort-based rank-normalisation up to
    ties, which we resolve to the midpoint of the tied range — the
    standard convention for ranking-aware metrics."""
    counts = np.zeros(256, dtype=np.int64)
    for sid, mat in decoded_uint8.items():
        counts += np.bincount(mat.ravel(), minlength=256)
    cum = np.cumsum(counts).astype(np.float64)
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    prev = np.concatenate([[0.0], cum[:-1]])
    mid = (prev + cum) / (2.0 * total)
    return mid.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Multiview pooling
# ─────────────────────────────────────────────────────────────────────────────
def pool_views_in_place(decoded_uint8: dict[str, np.ndarray],
                        view_regex: str,
                        pool_mode: str) -> tuple[int, int]:
    """Group IDs by object_id (regex group 1) and pool across views in
    place. Returns (n_objects_with_multiple_views, n_views_skipped_diffshape)."""
    if pool_mode == "none" or not view_regex:
        return (0, 0)
    rx = re.compile(view_regex)
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in decoded_uint8:
        m = rx.match(sid)
        if not m or not m.groups():
            continue
        obj = m.group(1)
        groups[obj].append(sid)
    n_pooled = 0
    n_skipped_shape = 0
    for obj, sids in groups.items():
        if len(sids) <= 1:
            continue
        mats = [decoded_uint8[sid].astype(np.float32) for sid in sids]
        shapes = {m.shape for m in mats}
        if len(shapes) > 1:
            n_skipped_shape += 1
            continue
        n_pooled += 1
        if pool_mode == "mean":
            pooled = np.mean(mats, axis=0)
            for sid in sids:
                decoded_uint8[sid] = np.clip(np.rint(pooled), 0, 255).astype(np.uint8)
        elif pool_mode == "max":
            pooled = np.maximum.reduce(mats)
            for sid in sids:
                decoded_uint8[sid] = np.clip(np.rint(pooled), 0, 255).astype(np.uint8)
        elif pool_mode == "soft":
            mean = np.mean(mats, axis=0)
            for sid, mat in zip(sids, mats):
                blended = 0.5 * mat + 0.5 * mean
                decoded_uint8[sid] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return (n_pooled, n_skipped_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────
def suppress_below_percentile(mat: np.ndarray, pct: float) -> np.ndarray:
    if pct <= 0:
        return mat
    thresh = np.percentile(mat, pct)
    return np.where(mat < thresh, 0.0, mat).astype(np.float32)


def remove_small_components(mat: np.ndarray, min_px: int,
                            binarise_at: float = 0.1) -> np.ndarray:
    if min_px <= 0:
        return mat
    try:
        from scipy import ndimage
    except ImportError:
        return mat
    binary = mat > binarise_at
    labels, n = ndimage.label(binary)
    if n == 0:
        return mat
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background label
    keep_lab = sizes >= min_px
    keep_mask = keep_lab[labels]
    return np.where(keep_mask, mat, 0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Fusion strategies
# ─────────────────────────────────────────────────────────────────────────────
def fuse_pixelwise(scores: np.ndarray,
                   weights: np.ndarray,
                   strategy: str,
                   top_k_frac: float = 0.5) -> np.ndarray:
    """scores: (M, H, W) float32 in [0, 1].
       weights: (M,) summing to 1.
       Returns (H, W) float32 in [0, 1]."""
    M = scores.shape[0]
    if strategy in ("wmean", "rank_wmean"):
        return np.tensordot(weights.astype(np.float32), scores, axes=1)
    if strategy == "max":
        # Drop low-weight methods (noisy ones) then unweighted max
        floor = 1.0 / (3 * M)
        mask = weights >= floor
        if not mask.any():
            mask = np.ones(M, dtype=bool)
        return scores[mask].max(axis=0)
    if strategy == "median":
        return np.median(scores, axis=0)
    if strategy == "agreement":
        K = max(1, int(np.ceil(M * top_k_frac)))
        if K >= M:
            return scores.mean(axis=0)
        # np.partition is O(MHW) — last K rows are the top-K (unsorted)
        part = np.partition(scores, M - K, axis=0)
        return part[M - K:].mean(axis=0)
    if strategy == "consensus":
        # Weighted geometric mean. Floor at eps prevents log(0) and gives
        # zero-weight methods no influence (log(1) * 0 = 0).
        eps = 1e-3
        log_s = np.log(np.clip(scores, eps, 1.0))
        log_mean = np.tensordot(weights.astype(np.float32), log_s, axes=1)
        return np.exp(log_mean)
    raise ValueError(f"unknown strategy: {strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# Resize fallback for shape mismatches between methods
# ─────────────────────────────────────────────────────────────────────────────
def _nearest_resize(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    th, tw = target_shape
    sh, sw = arr.shape
    yi = (np.arange(th) * sh / th).astype(np.int64)
    xi = (np.arange(tw) * sw / tw).astype(np.int64)
    return arr[yi[:, None], xi[None, :]]


# ─────────────────────────────────────────────────────────────────────────────
# Fusion driver
# ─────────────────────────────────────────────────────────────────────────────
def fuse_submissions(submissions: list[dict[str, str]],
                     weights_per_class: dict[str, list[float]],
                     class_map: dict[str, str] | None,
                     rank_normalise: bool,
                     strategy: str,
                     top_k_frac: float,
                     view_regex: str,
                     multiview_pool: str,
                     suppress_below: float,
                     remove_small_cc: int,
                     default_class: str = "_default_") -> dict[str, str]:
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common:
        raise RuntimeError("no IDs in common across submissions")
    all_ids = sorted(common)
    n_missing = sum(len(s) - len(common) for s in submissions)
    if n_missing > 0:
        print(f"  [warn] {n_missing} ID-slots present in some submissions but "
              f"not all — fusing on intersection ({len(common)} IDs).")

    M = len(submissions)
    uniform_default = [1.0 / M] * M
    if default_class not in weights_per_class:
        weights_per_class[default_class] = uniform_default

    # ── Decode all submissions to uint8 (4× smaller than float32) ──────────
    print(f"\nDecoding {M} submissions × {len(all_ids)} IDs each (uint8)...")
    decoded_per_method: list[dict[str, np.ndarray]] = []
    t0 = time.time()
    for mi, sub in enumerate(submissions):
        d: dict[str, np.ndarray] = {}
        for j, sid in enumerate(all_ids):
            d[sid] = q8rle_to_uint8_matrix(sub[sid])
            if (j + 1) % 1000 == 0:
                print(f"    method {mi + 1}/{M}: decoded {j + 1}/{len(all_ids)} "
                      f"({time.time() - t0:.1f}s)", flush=True)
        decoded_per_method.append(d)
        nbytes = sum(m.nbytes for m in d.values())
        print(f"    method {mi + 1}/{M} decoded "
              f"({nbytes / 1e9:.2f} GB uint8, {time.time() - t0:.1f}s total)")

    # ── Multiview pooling (in uint8, BEFORE rank-norm so the calibration
    # ── reflects the post-pool distribution) ──────────────────────────────
    if multiview_pool != "none" and view_regex:
        print(f"\nMultiview pooling (mode={multiview_pool}, regex={view_regex!r})...")
        for mi, d in enumerate(decoded_per_method):
            n_pool, n_skip = pool_views_in_place(d, view_regex, multiview_pool)
            print(f"    method {mi + 1}/{M}: pooled {n_pool} objects"
                  + (f", skipped {n_skip} (shape mismatch)" if n_skip else ""))
        # Sanity check after first method
        if mi == 0 and n_pool == 0:
            print(f"  [warn] regex matched 0 objects with multiple views — "
                  f"check --multiview-regex (must have one capture group "
                  f"that strips the view suffix).")

    # ── Rank-normalisation LUTs (one per method, 256 entries) ─────────────
    rank_luts: list[np.ndarray | None] = []
    if rank_normalise:
        print(f"\nBuilding rank-normalisation LUTs (per method, global)...")
        for mi, d in enumerate(decoded_per_method):
            t = time.time()
            lut = compute_rank_lut(d)
            rank_luts.append(lut)
            print(f"    method {mi + 1}/{M}: LUT in {time.time() - t:.2f}s "
                  f"(min={lut.min():.4f}, max={lut.max():.4f}, "
                  f"median={lut[127]:.4f})")
    else:
        rank_luts = [None] * M

    # ── Per-image fusion ──────────────────────────────────────────────────
    pp_str = ""
    if suppress_below > 0:
        pp_str += f", suppress<{suppress_below:g}%ile"
    if remove_small_cc > 0:
        pp_str += f", drop-cc<{remove_small_cc}px"
    print(f"\nFusing {len(all_ids)} images: strategy={strategy}"
          f"{' [rank-normed]' if rank_normalise else ''}"
          f"{pp_str}...")

    fused: dict[str, str] = {}
    t1 = time.time()
    for i, sid in enumerate(all_ids):
        cls = (class_map.get(sid) if class_map else None) or default_class
        weights = np.array(
            weights_per_class.get(cls, weights_per_class[default_class]),
            dtype=np.float32)

        # Build (M, H, W) float32 stack — apply LUT if rank-norming
        layers = []
        for m in range(M):
            u8 = decoded_per_method[m][sid]
            if rank_luts[m] is not None:
                layers.append(rank_luts[m][u8])
            else:
                layers.append(u8.astype(np.float32) / 255.0)

        # Shape consistency check (defensive — methods at different input
        # resolutions should already upsample to the dataset's native size)
        shapes = [l.shape for l in layers]
        if len(set(shapes)) > 1:
            target = Counter(shapes).most_common(1)[0][0]
            for k, l in enumerate(layers):
                if l.shape != target:
                    layers[k] = _nearest_resize(l, target).astype(np.float32)

        stack = np.stack(layers, axis=0)
        fused_mat = fuse_pixelwise(stack, weights, strategy,
                                   top_k_frac=top_k_frac)

        if suppress_below > 0:
            fused_mat = suppress_below_percentile(fused_mat, suppress_below)
        if remove_small_cc > 0:
            fused_mat = remove_small_components(fused_mat, remove_small_cc)

        fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
        fused[sid] = float_matrix_to_q8rle(fused_mat)

        if (i + 1) % 1000 == 0:
            print(f"    fused {i + 1}/{len(all_ids)} "
                  f"({time.time() - t1:.1f}s)", flush=True)
    print(f"  fused all {len(all_ids)} in {time.time() - t1:.1f}s")
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path,
                    help="2+ submission.csv paths")
    ap.add_argument("--local-evals", nargs="*", type=Path,
                    help="One local_eval.csv per --runs entry (for local_ap weights)")
    ap.add_argument("--weights", default="local_ap",
                    choices=["uniform", "local_ap"])
    ap.add_argument("--weight-temperature", type=float, default=1.0,
                    help="Softmax temperature on AP (only used if !=1.0). "
                         "<1 sharpens (top method dominates), >1 smooths. "
                         "Default 1.0 = linear normalisation with floor.")
    ap.add_argument("--top-k-methods-per-class", type=int, default=None,
                    help="Per class, only the top-K methods (by local AP) "
                         "get nonzero weight.")
    ap.add_argument("--strategy", default="rank_wmean",
                    choices=["wmean", "rank_wmean", "max", "median",
                             "agreement", "consensus"],
                    help="Fusion strategy. See module docstring.")
    ap.add_argument("--top-k-frac", type=float, default=0.5,
                    help="Fraction of methods to mean over for "
                         "--strategy agreement.")
    ap.add_argument("--rank-normalise", action="store_true",
                    help="Global per-method rank-norm across all pixels of all "
                         "test images. Forced ON for rank_wmean. Strongly "
                         "recommended for all other strategies too.")
    ap.add_argument("--suppress-below", type=float, default=0.0,
                    help="Zero out scores below this percentile per fused "
                         "image. 0 = off. 75 = keep top 25%%.")
    ap.add_argument("--remove-small-cc", type=int, default=0,
                    help="Drop connected components smaller than N pixels "
                         "after suppression. 0 = off. Needs scipy.")
    ap.add_argument("--multiview-regex", type=str, default="",
                    help="Regex with ONE capture group = object_id, used to "
                         "group IDs into objects for per-view pooling. "
                         "Empty disables multiview pooling.")
    ap.add_argument("--multiview-pool",
                    choices=["none", "mean", "max", "soft"], default="none",
                    help="How to pool views of one object. 'soft' = 50%% "
                         "per-view + 50%% across-view mean (usually best).")
    ap.add_argument("--class-map", type=Path,
                    help="CSV with columns ID,class. If absent, derived from "
                         "--data-root.")
    ap.add_argument("--data-root", type=Path,
                    default=Path("/workspace/anomaly-detection/data"))
    ap.add_argument("--out", type=Path, required=True,
                    help="Output submission.csv path")
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/workspace/anomaly-detection/"
                                 "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="fusion")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    # rank_wmean implies rank-normalisation
    if args.strategy == "rank_wmean" and not args.rank_normalise:
        print("[info] strategy=rank_wmean implies --rank-normalise")
        args.rank_normalise = True

    # ── Load submissions ──────────────────────────────────────────────────
    print(f"Loading {len(args.runs)} submission CSVs...")
    subs = []
    for p in args.runs:
        s = load_submission(p)
        print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
        subs.append(s)
    if len(subs) < 2:
        raise SystemExit("need ≥ 2 submissions to fuse")

    # ── Load local evals ──────────────────────────────────────────────────
    local_evals = []
    if args.local_evals:
        if len(args.local_evals) != len(args.runs):
            raise SystemExit("--local-evals count must match --runs count")
        for p in args.local_evals:
            le = load_local_eval(p)
            summary = {c: round(v, 3) for c, v in per_class_ap_summary(le).items()}
            print(f"  {p.parent.name}/{p.name}: per-class AP = {summary}")
            local_evals.append(le)

    # ── Class map ─────────────────────────────────────────────────────────
    print(f"\nBuilding ID → class map...")
    if args.class_map and args.class_map.exists():
        class_map = load_class_map(args.class_map)
        print(f"  loaded {len(class_map)} entries from {args.class_map}")
    else:
        class_map = build_class_map_from_data(args.data_root)
        print(f"  built from {args.data_root}: {len(class_map)} entries")
    if not class_map:
        print(f"  [warn] no class map — uniform weights for all IDs")
        class_map = None

    # ── Weights ───────────────────────────────────────────────────────────
    classes = sorted(set(class_map.values())) if class_map else ["_default_"]
    weights_per_class = compute_weights(
        local_evals, classes, mode=args.weights,
        temperature=args.weight_temperature,
        top_k=args.top_k_methods_per_class)
    method_tags = [p.parent.name[:28] for p in args.runs]
    print(f"\nMethod order:")
    for i, t in enumerate(method_tags):
        print(f"  [{i:2d}] {t}")
    print(f"\nWeights per class (top-3 methods shown):")
    for cls in classes:
        w = weights_per_class[cls]
        top = sorted(enumerate(w), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"[{i}]={v:.2f}" for i, v in top)
        print(f"  {cls:<12}  {top_str}")

    # ── Strategy summary ──────────────────────────────────────────────────
    print(f"\nStrategy: {args.strategy}"
          + (f" (top_k_frac={args.top_k_frac})" if args.strategy == "agreement" else "")
          + (", rank-norm=ON" if args.rank_normalise else ", rank-norm=OFF"))
    if args.multiview_pool != "none" and args.multiview_regex:
        print(f"Multiview pool: {args.multiview_pool}  regex={args.multiview_regex!r}")
    if args.suppress_below > 0:
        print(f"Sparsity: zero pixels below {args.suppress_below:g}-th percentile")
    if args.remove_small_cc > 0:
        print(f"Sparsity: drop connected components < {args.remove_small_cc} px")

    # ── Fuse ──────────────────────────────────────────────────────────────
    fused = fuse_submissions(
        subs, weights_per_class, class_map,
        rank_normalise=args.rank_normalise,
        strategy=args.strategy,
        top_k_frac=args.top_k_frac,
        view_regex=args.multiview_regex,
        multiview_pool=args.multiview_pool,
        suppress_below=args.suppress_below,
        remove_small_cc=args.remove_small_cc,
    )

    # ── Write ─────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
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

    # ── Append to ablation_master ─────────────────────────────────────────
    run_id = "fusion_" + hashlib.sha1(
        ("|".join(str(p) for p in args.runs)
         + f"|{args.strategy}|sb{args.suppress_below}|cc{args.remove_small_cc}"
         + f"|mvp{args.multiview_pool}|tk{args.top_k_methods_per_class}"
        ).encode("utf-8")
    ).hexdigest()[:8]
    row = {
        "run_id": run_id,
        "run_tag": args.run_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": "FUSION",
        "feature_layers": "",
        "input_size": "",
        "n_classes": len(classes),
        "AP_overall": "",
        "runtime_min": "",
        "submission_path": str(args.out.with_suffix(".zip")),
        "notes": (f"fusion-{args.strategy} of {len(subs)} methods "
                  f"(w={args.weights}, rn={int(args.rank_normalise)}, "
                  f"sb={args.suppress_below:g}, cc={args.remove_small_cc}, "
                  f"mvp={args.multiview_pool}, "
                  f"tkc={args.top_k_methods_per_class})"),
    }
    existing: list[dict] = []
    fieldnames: list[str] = []
    if args.master_csv.exists():
        with open(args.master_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)
    for k in row.keys():
        if k not in fieldnames:
            fieldnames.append(k)
    existing.append(row)
    args.master_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Appended fusion row to {args.master_csv}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()