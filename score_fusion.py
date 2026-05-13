# """Score fusion across multiple submission.csv files.
#
# Takes 2+ submission CSVs (each produced by some anomaly-detection method),
# combines them per pixel into a single fused submission. Supports:
#
#   - Uniform averaging (simplest baseline)
#   - Per-(class, method) weights, computed from local_eval.csv files
#   - Rank-normalisation per image (recommended) — undoes any global score-
#     scale differences between methods so weights mean something
#   - Per-(class, method) AP-proportional weighting via --weights local_ap
#     (uses the local_eval.csv from each run as the weighting source)
#
# The standard recipe at this stage of the project:
#
#     python score_fusion.py \\
#         --runs  baseline_out/runs/<best_patchcore>/submission.csv \\
#                 baseline_out/runs/<best_dinov2>/submission.csv \\
#                 baseline_out/runs/<cutpaste>/submission.csv \\
#         --local-evals  baseline_out/runs/<best_patchcore>/local_eval.csv \\
#                        baseline_out/runs/<best_dinov2>/local_eval.csv \\
#                        baseline_out/runs/<cutpaste>/local_eval.csv \\
#         --weights local_ap \\
#         --rank-normalise \\
#         --out baseline_out/runs/fusion_v1/submission.csv
#
# The script writes a new submission.csv + .zip and appends a "fusion" row
# to ablation_master.csv so the ablation table tracks it like any other run.
#
# Class is inferred from the ID prefix matching pattern (we ask the user to
# pass --class-map if the ID alone doesn't tell us the class). For
# spacepresso, each test image filename does NOT contain the class, so we
# fall back to a uniform weight per method unless --class-map is given.
# """
# from __future__ import annotations
#
# import argparse
# import csv
# import hashlib
# import json
# import time
# import zipfile
# from collections import defaultdict
# from pathlib import Path
#
# import numpy as np
#
# import sys
# csv.field_size_limit(sys.maxsize)
#
# # ─────────────────────────────────────────────────────────────────────────────
# # q8rle codec (re-exported for self-containment)
# # ─────────────────────────────────────────────────────────────────────────────
# def float_matrix_to_q8rle(x: np.ndarray) -> str:
#     q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255),
#                 0, 255).astype(np.uint8)
#     h, w = q.shape
#     flat = q.T.reshape(-1)
#     if flat.size == 0:
#         return f"q8rle {h} {w}"
#     cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
#     starts = np.r_[0, cuts]
#     ends = np.r_[cuts, flat.size]
#     parts = ["q8rle", str(h), str(w)]
#     for v, n in zip(flat[starts], ends - starts):
#         parts += [str(int(v)), str(int(n))]
#     return " ".join(parts)
#
#
# def q8rle_to_float_matrix(s: str) -> np.ndarray:
#     t = s.split()
#     h, w = int(t[1]), int(t[2])
#     vals = np.array(list(map(int, t[3::2])), dtype=np.uint8)
#     lens = np.array(list(map(int, t[4::2])), dtype=np.int64)
#     flat = np.repeat(vals, lens).reshape(w, h).T
#     return flat.astype(np.float32) / 255.0
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Loaders
# # ─────────────────────────────────────────────────────────────────────────────
# def load_submission(path: Path) -> dict[str, str]:
#     """Returns {ID: q8rle_string}."""
#     out: dict[str, str] = {}
#     with open(path, "r", encoding="utf-8") as f:
#         reader = csv.reader(f)
#         header = next(reader, None)
#         if header != ["ID", "Label"]:
#             raise ValueError(f"{path}: unexpected header {header}")
#         for row in reader:
#             if len(row) < 2: continue
#             out[row[0]] = row[1]
#     return out
#
#
# def load_local_eval(path: Path) -> dict[str, dict[str, float]]:
#     """Returns {class: {anomaly_type: ap_mean}}."""
#     out: dict[str, dict[str, float]] = defaultdict(dict)
#     with open(path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             try:
#                 ap = float(row["ap_mean"])
#             except (KeyError, ValueError):
#                 continue
#             out[row["class"]][row["anomaly_type"]] = ap
#     return out
#
#
# def per_class_ap_summary(eval_table: dict[str, dict[str, float]]
#                           ) -> dict[str, float]:
#     """Mean AP per class (averaging over anomaly types)."""
#     return {cls: float(np.mean(list(types.values())))
#             for cls, types in eval_table.items()}
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Class mapping (the trickiest piece)
# # ─────────────────────────────────────────────────────────────────────────────
# def load_class_map(path: Path | None,
#                     sample_ids: list[str]) -> dict[str, str] | None:
#     """Load a {ID -> class} map from a CSV with columns 'ID,class'.
#     Returns None if no path provided; warns about missing IDs.
#     """
#     if path is None:
#         return None
#     if not path.exists():
#         raise FileNotFoundError(f"--class-map {path} not found")
#     out: dict[str, str] = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             if "ID" in row and "class" in row:
#                 out[row["ID"]] = row["class"]
#     missing = [i for i in sample_ids if i not in out]
#     if missing:
#         print(f"  [warn] {len(missing)}/{len(sample_ids)} IDs not in class map; "
#               f"falling back to uniform weights for those.")
#     return out
#
#
# def build_class_map_from_data(data_root: Path) -> dict[str, str]:
#     """Walk class_XX/test/ and emit {filename_stem -> class}."""
#     if not data_root.exists():
#         return {}
#     out: dict[str, str] = {}
#     for cdir in sorted(data_root.iterdir()):
#         if not cdir.is_dir() or not cdir.name.startswith("class_"):
#             continue
#         test_dir = cdir / "test"
#         if not test_dir.exists():
#             continue
#         for p in test_dir.rglob("*"):
#             if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg",
#                                                      ".bmp", ".tiff", ".webp"}:
#                 out[p.stem] = cdir.name
#     return out
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Fusion core
# # ─────────────────────────────────────────────────────────────────────────────
# def rank_normalise(score: np.ndarray) -> np.ndarray:
#     """Map score values to their fractional rank in [0, 1]."""
#     flat = score.ravel()
#     order = np.argsort(flat, kind="stable")
#     ranks = np.empty_like(order, dtype=np.float32)
#     ranks[order] = np.linspace(0, 1, len(flat), endpoint=True, dtype=np.float32)
#     return ranks.reshape(score.shape)
#
#
# def compute_weights(local_evals: list[dict[str, dict[str, float]]],
#                      classes: list[str],
#                      mode: str = "uniform") -> dict[str, list[float]]:
#     """Returns {class: [w_method_0, w_method_1, ...]} normalised to sum 1."""
#     out: dict[str, list[float]] = {}
#     M = len(local_evals)
#     if mode == "uniform" or not local_evals or any(not e for e in local_evals):
#         for cls in classes:
#             out[cls] = [1.0 / M] * M
#         return out
#
#     if mode == "local_ap":
#         for cls in classes:
#             aps_per_method = []
#             for e in local_evals:
#                 cls_eval = e.get(cls, {})
#                 aps_per_method.append(
#                     float(np.mean(list(cls_eval.values()))) if cls_eval else 0.0)
#             # Avoid zero-weight collapse: floor at 1/(M*4) of the best per class
#             best = max(aps_per_method) if aps_per_method else 1.0
#             floor = best / (M * 4) if best > 0 else 1e-6
#             aps_per_method = [max(a, floor) for a in aps_per_method]
#             tot = sum(aps_per_method) or 1.0
#             out[cls] = [a / tot for a in aps_per_method]
#         return out
#
#     raise ValueError(f"unknown weight mode: {mode}")
#
#
# def fuse_submissions(submissions: list[dict[str, str]],
#                       weights_per_class: dict[str, list[float]],
#                       class_map: dict[str, str] | None,
#                       rank_normalise_flag: bool,
#                       default_class: str = "_default_") -> dict[str, str]:
#     """Returns fused {ID -> q8rle}."""
#     # Ensure all submissions cover the same IDs
#     common = set.intersection(*[set(s.keys()) for s in submissions])
#     if not common:
#         raise RuntimeError("no IDs in common across submissions")
#     all_ids = sorted(common)
#     n_missing = sum(len(s) - len(common) for s in submissions)
#     if n_missing > 0:
#         print(f"  [warn] {n_missing} IDs were present in some submissions "
#               f"but not all — fusing on the intersection ({len(common)} IDs).")
#
#     # Make sure default weights exist
#     M = len(submissions)
#     uniform_default = [1.0 / M] * M
#     if default_class not in weights_per_class:
#         weights_per_class[default_class] = uniform_default
#
#     fused: dict[str, str] = {}
#     n_processed = 0
#     t0 = time.time()
#     for sid in all_ids:
#         cls = (class_map.get(sid) if class_map else None) or default_class
#         weights = weights_per_class.get(cls,
#                                           weights_per_class[default_class])
#         mats = [q8rle_to_float_matrix(s[sid]) for s in submissions]
#         if rank_normalise_flag:
#             mats = [rank_normalise(m) for m in mats]
#         fused_mat = np.zeros_like(mats[0], dtype=np.float32)
#         for w, m in zip(weights, mats):
#             fused_mat += float(w) * m
#         # If we rank-normalised, the fused output is already in [0,1].
#         # If not, clip to [0,1] before encoding.
#         fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
#         fused[sid] = float_matrix_to_q8rle(fused_mat)
#         n_processed += 1
#         if n_processed % 500 == 0:
#             print(f"    fused {n_processed}/{len(all_ids)} "
#                   f"({time.time() - t0:.1f}s elapsed)", flush=True)
#     return fused
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────
# def main():
#     ap = argparse.ArgumentParser(description=__doc__,
#                                   formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--runs", nargs="+", required=True,
#                     type=Path, help="2+ submission.csv paths")
#     ap.add_argument("--local-evals", nargs="*", type=Path,
#                     help="local_eval.csv paths (one per --runs entry) for "
#                          "AP-proportional weighting")
#     ap.add_argument("--weights", default="uniform",
#                     choices=["uniform", "local_ap"])
#     ap.add_argument("--rank-normalise", action="store_true",
#                     help="rank-normalise each method's score map per image "
#                          "before fusing — recommended when methods have "
#                          "different score scales.")
#     ap.add_argument("--class-map",
#                     type=Path,
#                     help="CSV with columns ID,class — assigns each test "
#                          "image to a class so per-class weights apply.")
#     ap.add_argument("--data-root", type=Path,
#                     default=Path("/work/u10813429/anomaly-detection/data"),
#                     help="If --class-map not given, walk this dir to build "
#                          "the {filename_stem -> class} map.")
#     ap.add_argument("--out",
#                     type=Path,
#                     default=Path("/work/u10813429/anomaly-detection/"
#                                   "baseline_out/runs/fusion/submission.csv"))
#     ap.add_argument("--master-csv",
#                     type=Path,
#                     default=Path("/work/u10813429/anomaly-detection/"
#                                   "baseline_out/ablation_master.csv"))
#     ap.add_argument("--run-tag", default="fusion",
#                     help="row tag for the ablation master CSV.")
#     ap.add_argument("--no-zip", action="store_true")
#     args = ap.parse_args()
#
#     # ── Load all submissions
#     print(f"Loading {len(args.runs)} submission CSVs...")
#     subs = []
#     for p in args.runs:
#         s = load_submission(p)
#         print(f"  {p.name}: {len(s)} rows")
#         subs.append(s)
#     if len(subs) < 2:
#         raise SystemExit("need ≥ 2 submissions to fuse")
#
#     # ── Load local evals (optional, only needed for local_ap weights)
#     local_evals = []
#     if args.local_evals:
#         if len(args.local_evals) != len(args.runs):
#             raise SystemExit("--local-evals count must match --runs count")
#         for p in args.local_evals:
#             le = load_local_eval(p)
#             print(f"  {p.name}: AP summary {{cls: mean_ap}} = "
#                   f"{ {c: round(v, 3) for c, v in per_class_ap_summary(le).items()} }")
#             local_evals.append(le)
#
#     # ── Build class map
#     print(f"\nBuilding ID → class map...")
#     if args.class_map and args.class_map.exists():
#         class_map = load_class_map(args.class_map,
#                                      list(subs[0].keys()))
#     else:
#         class_map = build_class_map_from_data(args.data_root)
#         print(f"  built from {args.data_root}: {len(class_map)} entries")
#     if not class_map:
#         print(f"  [warn] no class map available — falling back to uniform weights")
#         class_map = None
#
#     # ── Compute weights
#     classes = sorted(set(class_map.values())) if class_map else ["_default_"]
#     weights_per_class = compute_weights(local_evals, classes,
#                                           mode=args.weights)
#     print(f"\nWeights per class (method order = order of --runs):")
#     for cls in classes:
#         w = weights_per_class[cls]
#         print(f"  {cls:<12}  {['%.3f' % x for x in w]}")
#
#     # ── Fuse
#     print(f"\nFusing... (rank_normalise={args.rank_normalise})")
#     fused = fuse_submissions(subs, weights_per_class, class_map,
#                               rank_normalise_flag=args.rank_normalise)
#
#     # ── Write
#     args.out.parent.mkdir(parents=True, exist_ok=True)
#     with open(args.out, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["ID", "Label"])
#         for sid in sorted(fused):
#             w.writerow([sid, fused[sid]])
#     print(f"\nWrote {len(fused)} rows -> {args.out}")
#     if not args.no_zip:
#         zip_path = args.out.with_suffix(".zip")
#         with zipfile.ZipFile(zip_path, "w",
#                              compression=zipfile.ZIP_DEFLATED) as zf:
#             zf.write(args.out, arcname=args.out.name)
#         print(f"Zipped -> {zip_path}")
#
#     # ── Append a row to ablation_master so the fusion shows up in the table
#     run_id_bits = "fusion_" + hashlib.sha1(
#         "|".join(str(p) for p in args.runs).encode("utf-8")
#     ).hexdigest()[:6]
#     row = {
#         "run_id": run_id_bits,
#         "run_tag": args.run_tag,
#         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "backbone": "FUSION",
#         "feature_layers": "",
#         "input_size": "",
#         "coreset_frac": "",
#         "knn_k": "",
#         "smooth_sigma": "",
#         "tta": "",
#         "n_classes": len(classes),
#         "AP_overall": "",   # local-eval requires GT masks; not run here
#         "runtime_min": "",
#         "submission_path": str(args.out.with_suffix(".zip")),
#         "notes": (f"fusion of {len(subs)} methods "
#                   f"({args.weights} weights, "
#                   f"rank_norm={int(args.rank_normalise)})"),
#     }
#     # append to CSV
#     existing: list[dict] = []
#     fieldnames: list[str] = []
#     if args.master_csv.exists():
#         with open(args.master_csv, "r", newline="", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             fieldnames = list(reader.fieldnames or [])
#             existing = list(reader)
#     for k in row.keys():
#         if k not in fieldnames:
#             fieldnames.append(k)
#     existing.append(row)
#     with open(args.master_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for r in existing:
#             w.writerow({k: r.get(k, "") for k in fieldnames})
#     print(f"\nAppended fusion row to {args.master_csv}")
#     print(f"\nDone.")
#
#
# if __name__ == "__main__":
#     main()


"""Score fusion across multiple submission.csv files.

Takes 2+ submission CSVs (each produced by some anomaly-detection method),
combines them per pixel into a single fused submission.

# v2 fixes (CRITICAL)

The original --rank-normalise applied PER-IMAGE rank normalisation, which
destroys cross-image score scale. For pixel-AP (which ranks ALL pixels
GLOBALLY across all test images), per-image normalisation makes every
clean image's brightest-background pixel rank ~1.0 — those compete with
true anomalies in defective images, and AP collapses.

v2 changes the default mode to "average" (no rank-norm — submissions are
already calibrated to [0, 1] via percentile clipping in PatchCore /
CutPaste output, so they're already on a comparable scale).

  --rank-normalise   now does GLOBAL per-method rank-normalisation
                     (across all pixels of all test images for that
                     method). Mathematically correct for pixel-AP but
                     usually unnecessary since submissions are already
                     percentile-calibrated.

Per-image rank-normalisation has been REMOVED entirely. It was only
useful for image-level scoring (one score per image), which is not the
Spacepresso evaluation.

# Other v2 changes

  - Decode each submission once at the start, then reuse the numpy
    matrices for fusion. The original re-decoded inside the per-image
    loop (and re-decoded the same submission map every call) — slow
    for 1 GB submission CSVs.
  - Numpy-vectorised q8rle decoding (was Python list-of-lists, slow).
  - Progress logging during decode and fuse.

# Standard recipe

    python score_fusion.py \\
        --runs        baseline_out/runs/<exp5_wrn50>/submission.csv \\
                      baseline_out/runs/<exp7_dinov2>/submission.csv \\
                      baseline_out/runs/<exp8c_cutpaste_nn>/submission.csv \\
        --local-evals baseline_out/runs/<exp5_wrn50>/local_eval.csv \\
                      baseline_out/runs/<exp7_dinov2>/local_eval.csv \\
                      baseline_out/runs/<exp8c_cutpaste_nn>/local_eval.csv \\
        --weights local_ap \\
        --out baseline_out/runs/fusion_v1/submission.csv

NOTE: do NOT pass --rank-normalise unless you have a specific reason.
The default (average of pre-calibrated [0, 1] scores) is what you want.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# q8rle strings can exceed csv's default 128 KB field limit.
csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec — numpy-vectorised decoder
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
    """Decode q8rle string to (H, W) float32 array in [0, 1].
    Numpy-vectorised for speed on large submissions."""
    parts = s.split()
    h, w = int(parts[1]), int(parts[2])
    if len(parts) <= 3:
        return np.zeros((h, w), dtype=np.float32)
    # parts[3::2] = vals, parts[4::2] = lens.
    # Convert via np.fromstring-style trick: join and parse once.
    body = np.array(parts[3:], dtype=np.int64)
    vals = body[0::2].astype(np.uint8)
    lens = body[1::2]
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_submission(path: Path) -> dict[str, str]:
    """Returns {ID: q8rle_string}."""
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
    """Returns {class: {anomaly_type: ap_mean}}."""
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


def per_class_ap_summary(eval_table: dict[str, dict[str, float]]
                          ) -> dict[str, float]:
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
                     mode: str = "uniform") -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    M = len(local_evals)
    if mode == "uniform" or not local_evals or any(not e for e in local_evals):
        for cls in classes:
            out[cls] = [1.0 / M] * M
        return out
    if mode == "local_ap":
        for cls in classes:
            aps_per_method = []
            for e in local_evals:
                cls_eval = e.get(cls, {})
                aps_per_method.append(
                    float(np.mean(list(cls_eval.values()))) if cls_eval else 0.0)
            best = max(aps_per_method) if aps_per_method else 1.0
            floor = best / (M * 4) if best > 0 else 1e-6
            aps_per_method = [max(a, floor) for a in aps_per_method]
            tot = sum(aps_per_method) or 1.0
            out[cls] = [a / tot for a in aps_per_method]
        return out
    raise ValueError(f"unknown weight mode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Global rank normalisation (the correct kind for pixel-AP)
# ─────────────────────────────────────────────────────────────────────────────
def rank_normalise_global_per_method(decoded_per_method: list[dict[str, np.ndarray]],
                                       all_ids: list[str]) -> None:
    """In-place: replace each method's per-image score matrices with
    their GLOBAL rank-normalised versions (rank computed across all
    pixels of all test images, per method).

    This is the only correct rank-normalisation for pixel-AP. The old
    per-image version destroys cross-image score scale and was the cause
    of the catastrophic fusion-AP drop.
    """
    for mi, method_dict in enumerate(decoded_per_method):
        t0 = time.time()
        shapes = {sid: method_dict[sid].shape for sid in all_ids}
        sizes  = {sid: int(np.prod(shapes[sid])) for sid in all_ids}
        total  = sum(sizes.values())
        print(f"    method {mi + 1}/{len(decoded_per_method)}: "
              f"global rank-norm over {total:,} pixels...")
        # Concat
        flat = np.empty(total, dtype=np.float32)
        idx = 0
        for sid in all_ids:
            n = sizes[sid]
            flat[idx:idx + n] = method_dict[sid].ravel()
            idx += n
        # Argsort + linspace assignment
        order = np.argsort(flat, kind="stable")
        ranks = np.empty_like(flat)
        ranks[order] = np.linspace(0.0, 1.0, total, dtype=np.float32)
        del order, flat
        # Scatter back
        idx = 0
        for sid in all_ids:
            n = sizes[sid]
            method_dict[sid] = ranks[idx:idx + n].reshape(shapes[sid]).astype(np.float32)
            idx += n
        del ranks
        print(f"      done in {time.time() - t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Fusion core
# ─────────────────────────────────────────────────────────────────────────────
def fuse_submissions(submissions: list[dict[str, str]],
                      weights_per_class: dict[str, list[float]],
                      class_map: dict[str, str] | None,
                      rank_normalise_global: bool,
                      default_class: str = "_default_") -> dict[str, str]:
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common:
        raise RuntimeError("no IDs in common across submissions")
    all_ids = sorted(common)
    n_missing = sum(len(s) - len(common) for s in submissions)
    if n_missing > 0:
        print(f"  [warn] {n_missing} IDs were present in some submissions "
              f"but not all — fusing on the intersection ({len(common)} IDs).")

    M = len(submissions)
    uniform_default = [1.0 / M] * M
    if default_class not in weights_per_class:
        weights_per_class[default_class] = uniform_default

    # ── Decode every submission once. With ~1 GB CSVs of 5910 images at
    # 224×224, this is ~1.18 GB of float32 per method → ~3.5 GB for 3
    # methods. Comfortably fits on a typical cluster node.
    print(f"\nDecoding {M} submissions × {len(all_ids)} IDs each...")
    decoded_per_method: list[dict[str, np.ndarray]] = []
    t0 = time.time()
    for mi, sub in enumerate(submissions):
        d: dict[str, np.ndarray] = {}
        for j, sid in enumerate(all_ids):
            d[sid] = q8rle_to_float_matrix(sub[sid])
            if (j + 1) % 1000 == 0:
                print(f"    method {mi + 1}/{M}: decoded {j + 1}/{len(all_ids)}  "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)
        decoded_per_method.append(d)
        print(f"    method {mi + 1}/{M} decoded "
              f"({time.time() - t0:.1f}s total elapsed)")

    if rank_normalise_global:
        print(f"\nGlobal rank-normalisation (per method)...")
        rank_normalise_global_per_method(decoded_per_method, all_ids)

    print(f"\nFusing {len(all_ids)} images with weighted average...")
    fused: dict[str, str] = {}
    t1 = time.time()
    for i, sid in enumerate(all_ids):
        cls = (class_map.get(sid) if class_map else None) or default_class
        weights = weights_per_class.get(cls, weights_per_class[default_class])
        mats = [d[sid] for d in decoded_per_method]
        fused_mat = np.zeros_like(mats[0], dtype=np.float32)
        for w, m in zip(weights, mats):
            fused_mat += float(w) * m
        fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
        fused[sid] = float_matrix_to_q8rle(fused_mat)
        if (i + 1) % 1000 == 0:
            print(f"    fused {i + 1}/{len(all_ids)}  "
                  f"({time.time() - t1:.1f}s elapsed)", flush=True)
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
                    help="local_eval.csv paths (one per --runs entry) for "
                         "--weights local_ap")
    ap.add_argument("--weights", default="local_ap",
                    choices=["uniform", "local_ap"])
    ap.add_argument("--rank-normalise", action="store_true",
                    help="v2: GLOBAL per-method rank normalisation across "
                         "all pixels of all test images. The old per-image "
                         "behavior has been removed (it broke pixel-AP). "
                         "Usually unnecessary because PatchCore/CutPaste "
                         "already calibrate to [0, 1] via percentile clipping.")
    ap.add_argument("--class-map", type=Path,
                    help="CSV with columns ID,class. If absent, derived "
                         "from --data-root.")
    ap.add_argument("--data-root", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/data"))
    ap.add_argument("--out", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/runs/fusion/submission.csv"))
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="fusion")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    # ── Load all submissions
    print(f"Loading {len(args.runs)} submission CSVs...")
    subs = []
    for p in args.runs:
        s = load_submission(p)
        print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
        subs.append(s)
    if len(subs) < 2:
        raise SystemExit("need ≥ 2 submissions to fuse")

    # ── Load local evals
    local_evals = []
    if args.local_evals:
        if len(args.local_evals) != len(args.runs):
            raise SystemExit("--local-evals count must match --runs count")
        for p in args.local_evals:
            le = load_local_eval(p)
            summary = {c: round(v, 3) for c, v in per_class_ap_summary(le).items()}
            print(f"  {p.parent.name}/{p.name}: per-class AP = {summary}")
            local_evals.append(le)

    # ── Build class map
    print(f"\nBuilding ID → class map...")
    if args.class_map and args.class_map.exists():
        class_map = load_class_map(args.class_map)
        print(f"  loaded {len(class_map)} entries from {args.class_map}")
    else:
        class_map = build_class_map_from_data(args.data_root)
        print(f"  built from {args.data_root}: {len(class_map)} entries")
    if not class_map:
        print(f"  [warn] no class map — falling back to uniform weights")
        class_map = None

    # ── Compute weights
    classes = sorted(set(class_map.values())) if class_map else ["_default_"]
    weights_per_class = compute_weights(local_evals, classes, mode=args.weights)
    print(f"\nWeights per class (method order = --runs order):")
    for cls in classes:
        w = weights_per_class[cls]
        print(f"  {cls:<12}  {['%.3f' % x for x in w]}")

    # ── Fuse
    if args.rank_normalise:
        print(f"\nMode: weighted average AFTER global rank-normalisation.")
    else:
        print(f"\nMode: weighted average of pre-calibrated submissions "
              f"(submissions already in [0, 1] via percentile clipping).")
    fused = fuse_submissions(subs, weights_per_class, class_map,
                              rank_normalise_global=args.rank_normalise)

    # ── Write
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

    # ── Append to ablation_master
    run_id_bits = "fusion_" + hashlib.sha1(
        "|".join(str(p) for p in args.runs).encode("utf-8")
    ).hexdigest()[:6]
    row = {
        "run_id": run_id_bits,
        "run_tag": args.run_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": "FUSION",
        "feature_layers": "",
        "input_size": "",
        "n_classes": len(classes),
        "AP_overall": "",
        "runtime_min": "",
        "submission_path": str(args.out.with_suffix(".zip")),
        "notes": (f"fusion of {len(subs)} methods "
                  f"({args.weights} weights, "
                  f"global_rank_norm={int(args.rank_normalise)})"),
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
    with open(args.master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nAppended fusion row to {args.master_csv}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()