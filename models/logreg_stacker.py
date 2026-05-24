"""Logistic-regression stacker for Spacepresso anomaly detection.

Replaces the heuristic local_ap weighting in score_fusion.py with a
per-class logistic regression model fitted on the LOCAL VALIDATION
set, where ground-truth masks are available. The fitted weights are
the *optimal* linear combination of per-method scores for predicting
the true anomaly mask, in the sense of binary log-loss.

Pipeline
--------
Inputs (for each of the M methods you want to stack):
  - <run_dir>/submission.csv         (test predictions, q8rle)
  - <run_dir>/local_predictions.npz  (val predictions + GT masks;
                                      produced by local_preds_saver.py)

Steps:
  1. Read all M `local_predictions.npz`. They must share identical
     `ids` arrays (same images, in the same order) so per-pixel
     features can be stacked. If not, intersect on ids and warn.
  2. Optionally rank-normalise each method GLOBALLY (across all val
     pixels, per method).  Strongly recommended — makes the per-method
     score distributions comparable.
  3. For each class:
       a. Build (n_pixels, M) feature matrix X and (n_pixels,) label y.
       b. Subsample negatives so the pos/neg ratio is reasonable for
          fitting (defaults: keep ALL positives, sample R=30 negatives
          per positive).
       c. Fit `sklearn.linear_model.LogisticRegression(C=1.0)`.
       d. Log coefficients + intercept so the user can inspect what
          the model learned.
  4. Decode each method's test submission once.
  5. Apply per-class model to test pixels, write fused submission.csv
     + .zip, append a row to ablation_master.csv.

Why per-class
-------------
Method APs vary wildly across classes (class_03 has AP < 0.2 for all
methods; class_08 > 0.9 for some). The optimal mixing weights are
class-dependent; one global model would underfit. Per-class fitting
uses ~25 local-val images × 224² pixels ≈ 1.25M training rows per
class — plenty of data for a 3-feature logistic regression.

Why rank-normalisation
----------------------
The three methods (PatchCore-DINOv2, CutPaste, RD) output different
score distributions inside [0, 1]. CutPaste's classifier head saturates
near 1.0; PatchCore distance has a heavier left tail. Logistic
regression on RAW scores spends most of its capacity learning to
calibrate methods rather than learning their COMBINATION. After
GLOBAL rank-normalisation each method's pixel scores are uniform in
[0, 1], so the model learns pure combination.

Why subsample negatives
-----------------------
Defect pixels are <1% of all pixels. Without subsampling, the loss is
dominated by easy negatives and the model collapses toward "predict
the prior". Subsampling 30 negatives per positive gives a roughly
balanced binary problem; class_weight='balanced' would also work but
is more wasteful at fit time.

CLI
---
    python logreg_stacker.py \\
        --runs        baseline_out/runs/<exp7>/submission.csv \\
                      baseline_out/runs/<exp8c>/submission.csv \\
                      baseline_out/runs/<exp8d>/submission.csv \\
        --local-preds baseline_out/runs/<exp7>/local_predictions.npz \\
                      baseline_out/runs/<exp8c>/local_predictions.npz \\
                      baseline_out/runs/<exp8d>/local_predictions.npz \\
        --data-root   /work/u10813429/anomaly-detection/data \\
        --out         baseline_out/runs/stacker_logreg_v1/submission.csv \\
        --run-tag     stacker-logreg-exp7-exp8c-exp8d
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# q8rle strings can exceed csv's default 128 KB field limit.
csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec — bit-identical to score_fusion.py
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
    """Returns dict with keys: ids, classes, anomaly_types, scores, masks."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.  Run your baseline with the "
            f"local_preds_saver hook to generate it (see "
            f"local_preds_saver.py docstring).")
    data = np.load(path, allow_pickle=True)
    return {
        "ids":           data["ids"].astype(str),
        "classes":       data["classes"].astype(str),
        "anomaly_types": data["anomaly_types"].astype(str),
        "scores":        data["scores"].astype(np.float32),
        "masks":         data["masks"].astype(np.uint8),
    }


def build_class_map_from_data(data_root: Path) -> dict[str, str]:
    """Walk class_XX/test/ and return {filename_stem -> class}."""
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
# Local-val alignment
# ─────────────────────────────────────────────────────────────────────────────
def align_local_preds(preds_per_method: list[dict],
                        method_names: list[str]) -> dict:
    """Take the per-method local-pred dicts and align them on a common
    set of `ids`. Returns dict with:
        ids:     (N,) str
        classes: (N,) str
        scores:  (N, H, W, M) float32 — per-method stack
        masks:   (N, H, W) uint8 — assumed identical across methods,
                                    verified
    """
    # Common ids across all methods (sorted for determinism).
    id_sets = [set(p["ids"].tolist()) for p in preds_per_method]
    common = sorted(set.intersection(*id_sets))
    if not common:
        raise RuntimeError("no local-val IDs in common across the methods")
    n_total = len(preds_per_method[0]["ids"])
    if any(len(p["ids"]) != n_total for p in preds_per_method):
        print(f"  [warn] methods have different numbers of local-val "
              f"images; intersecting on the {len(common)} shared IDs.")
    # Verify spatial shapes match across methods. If they differ for any
    # method, we resize that method's maps to the first method's shape.
    H0, W0 = preds_per_method[0]["scores"].shape[1:3]
    print(f"  reference shape (method 0): {H0}x{W0}")
    # Index by id per method so we can pick the same order.
    indexed = []
    for mi, p in enumerate(preds_per_method):
        idx_of = {id_: i for i, id_ in enumerate(p["ids"])}
        indexed.append((p, idx_of))
    # Build aligned stacks.
    N = len(common); M = len(preds_per_method)
    scores = np.empty((N, H0, W0, M), dtype=np.float32)
    masks = np.empty((N, H0, W0), dtype=np.uint8)
    classes = np.empty(N, dtype=object)
    masks_set = False
    for i, id_ in enumerate(common):
        for mi, (p, idx_of) in enumerate(indexed):
            j = idx_of[id_]
            s = p["scores"][j]
            if s.shape != (H0, W0):
                s = _nn_resize_2d(s, (H0, W0), np.float32)
            scores[i, :, :, mi] = s
            if mi == 0:
                classes[i] = str(p["classes"][j])
                m = p["masks"][j]
                if m.shape != (H0, W0):
                    m = _nn_resize_2d(m, (H0, W0), np.uint8)
                masks[i] = m
                masks_set = True
            else:
                # Sanity: masks must agree across methods (same GT).
                m_other = p["masks"][j]
                if m_other.shape != (H0, W0):
                    m_other = _nn_resize_2d(m_other, (H0, W0), np.uint8)
                if not np.array_equal(masks[i], m_other):
                    n_diff = int(np.sum(masks[i] != m_other))
                    if mi == 1 and i == 0:
                        print(f"  [warn] GT masks differ between methods "
                              f"for {id_} ({n_diff} pixels). Using "
                              f"method 0's mask. Suppressing further "
                              f"warnings.")
    assert masks_set
    print(f"  aligned {N} val images × {M} methods @ {H0}x{W0}")
    return {"ids": np.asarray(common),
            "classes": classes.astype(str),
            "scores": scores,
            "masks": masks}


def _nn_resize_2d(arr: np.ndarray, target_shape, dtype) -> np.ndarray:
    th, tw = target_shape
    h, w = arr.shape
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Rank normalisation (global, per method)
# ─────────────────────────────────────────────────────────────────────────────
def rank_normalise_in_place(scores: np.ndarray) -> None:
    """In-place global rank normalisation per method.
    scores shape: (..., M) — last dim is methods.
    For each method, replace its values by their fractional rank in
    [0, 1] computed over ALL non-method dims jointly.
    """
    if scores.ndim < 2:
        raise ValueError(f"expected (..., M), got shape {scores.shape}")
    M = scores.shape[-1]
    flat_view = scores.reshape(-1, M)
    n = flat_view.shape[0]
    linsp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    for mi in range(M):
        col = flat_view[:, mi]
        order = np.argsort(col, kind="stable")
        ranks = np.empty_like(col)
        ranks[order] = linsp
        flat_view[:, mi] = ranks


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def fit_per_class(val: dict, classes: list[str],
                  *, neg_per_pos: int, C: float, seed: int):
    """Returns dict {class: (model, n_pos, n_neg_sampled, train_logloss)}.
    Falls back to a SHARED model for classes with too few positives.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, average_precision_score

    rng = np.random.default_rng(seed)

    val_scores = val["scores"]                  # (N, H, W, M)
    val_masks  = val["masks"]                   # (N, H, W)
    val_classes = val["classes"]                # (N,)
    M = val_scores.shape[-1]

    # Pre-flatten per class for fast indexing.
    out: dict[str, dict] = {}
    shared_X, shared_y = [], []
    min_pos_for_per_class = 200    # below this, fall back to shared

    for cls in classes:
        mask_cls = (val_classes == cls)
        if not mask_cls.any():
            print(f"  [warn] class {cls}: no local-val images; skipping")
            continue
        Sc = val_scores[mask_cls].reshape(-1, M)   # (n_pix, M)
        Y  = val_masks [mask_cls].reshape(-1)      # (n_pix,)
        n_pos = int(Y.sum())
        n_neg = int(len(Y) - n_pos)
        if n_pos == 0:
            print(f"  [warn] class {cls}: zero positive pixels in val; "
                  f"this class will get UNIFORM weights at predict time")
            continue

        # Subsample negatives.
        target_neg = min(n_neg, n_pos * neg_per_pos)
        neg_idx = np.flatnonzero(Y == 0)
        sample_neg = rng.choice(neg_idx, size=target_neg, replace=False)
        pos_idx = np.flatnonzero(Y == 1)
        keep = np.concatenate([pos_idx, sample_neg])
        Xc = Sc[keep]
        Yc = Y[keep]

        # Accumulate for shared model too.
        shared_X.append(Xc)
        shared_y.append(Yc)

        if n_pos < min_pos_for_per_class:
            print(f"  class {cls}: only {n_pos} positives (<"
                  f"{min_pos_for_per_class}); will use SHARED model")
            out[cls] = {"_fallback_to_shared": True,
                        "n_pos": n_pos, "n_neg_sampled": target_neg}
            continue

        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=500,
                                  random_state=seed)
        clf.fit(Xc, Yc)
        # Diagnostics
        p = clf.predict_proba(Xc)[:, 1]
        ll = float(log_loss(Yc, p, labels=[0, 1]))
        try:
            ap = float(average_precision_score(Yc, p))
        except Exception:
            ap = float("nan")
        out[cls] = {
            "model": clf,
            "n_pos": n_pos, "n_neg_sampled": target_neg,
            "logloss": ll, "train_ap": ap,
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
        }
        coef_str = " ".join(f"{c:+.3f}" for c in clf.coef_[0])
        print(f"  class {cls}: n_pos={n_pos:6d}  n_neg={target_neg:8d}  "
              f"logloss={ll:.4f}  train_ap={ap:.3f}  "
              f"coef=[{coef_str}]  b={clf.intercept_[0]:+.3f}")

    # Fit shared model (used as fallback).
    if shared_X:
        X_all = np.concatenate(shared_X, axis=0)
        y_all = np.concatenate(shared_y, axis=0)
        clf_sh = LogisticRegression(C=C, solver="lbfgs", max_iter=500,
                                      random_state=seed)
        clf_sh.fit(X_all, y_all)
        p_sh = clf_sh.predict_proba(X_all)[:, 1]
        ll_sh = float(log_loss(y_all, p_sh, labels=[0, 1]))
        try:
            ap_sh = float(average_precision_score(y_all, p_sh))
        except Exception:
            ap_sh = float("nan")
        coef_str = " ".join(f"{c:+.3f}" for c in clf_sh.coef_[0])
        print(f"  SHARED:    n_pos={int(y_all.sum()):6d}  "
              f"n_neg={int(len(y_all) - y_all.sum()):8d}  "
              f"logloss={ll_sh:.4f}  train_ap={ap_sh:.3f}  "
              f"coef=[{coef_str}]  b={clf_sh.intercept_[0]:+.3f}")
        out["_SHARED_"] = {
            "model": clf_sh, "n_pos": int(y_all.sum()),
            "n_neg_sampled": int(len(y_all) - y_all.sum()),
            "logloss": ll_sh, "train_ap": ap_sh,
            "coef": clf_sh.coef_[0].tolist(),
            "intercept": float(clf_sh.intercept_[0]),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def fuse_test(submissions: list[dict[str, str]],
              models_per_class: dict, class_map: dict[str, str] | None,
              rank_norm_test: bool, default_class: str) -> dict[str, str]:
    """Apply per-class logreg to every test image. Returns
    {id -> q8rle_string}.
    """
    # 1. Intersect IDs across methods.
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common:
        raise RuntimeError("no test IDs in common across submissions")
    all_ids = sorted(common)
    if any(len(s) != len(common) for s in submissions):
        diffs = [len(s) - len(common) for s in submissions]
        print(f"  [warn] some methods have extra IDs (counts diff = "
              f"{diffs}); fusing on the {len(common)} shared IDs.")

    M = len(submissions)

    # 2. Decode every submission once. Keep as a list of dicts so each
    #    method's per-image maps can be different shapes (test images
    #    may have heterogeneous sizes per class).
    print(f"\nDecoding {M} submissions × {len(all_ids)} IDs each...")
    decoded_per_method: list[dict[str, np.ndarray]] = []
    t0 = time.time()
    for mi, sub in enumerate(submissions):
        d: dict[str, np.ndarray] = {}
        for j, sid in enumerate(all_ids):
            d[sid] = q8rle_to_float_matrix(sub[sid])
            if (j + 1) % 1000 == 0:
                print(f"    method {mi + 1}/{M}: {j + 1}/{len(all_ids)}  "
                      f"({time.time() - t0:.1f}s)", flush=True)
        decoded_per_method.append(d)
        print(f"    method {mi + 1}/{M} decoded "
              f"({time.time() - t0:.1f}s total)")

    # 3. Global per-method rank-normalisation across the FULL test set.
    if rank_norm_test:
        print(f"\nGlobal rank-normalisation (per method, across test)...")
        for mi, d in enumerate(decoded_per_method):
            t1 = time.time()
            shapes = {sid: d[sid].shape for sid in all_ids}
            sizes  = {sid: int(np.prod(shapes[sid])) for sid in all_ids}
            total  = sum(sizes.values())
            print(f"    method {mi + 1}/{M}: {total:,} pixels...")
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
                d[sid] = ranks[idx:idx + n].reshape(
                    shapes[sid]).astype(np.float32)
                idx += n
            del ranks
            print(f"      done in {time.time() - t1:.1f}s")

    # 4. Per-image prediction with the appropriate per-class model.
    print(f"\nFusing {len(all_ids)} images with per-class logreg...")
    fused: dict[str, str] = {}
    shared_entry = models_per_class.get("_SHARED_")
    t1 = time.time()
    n_uniform = 0
    for i, sid in enumerate(all_ids):
        cls = (class_map.get(sid) if class_map else None) or default_class
        entry = models_per_class.get(cls, None)
        if entry is None or entry.get("_fallback_to_shared"):
            entry = shared_entry
        if entry is None:
            # No model at all: fall back to uniform average.
            n_uniform += 1
            mats = [d[sid] for d in decoded_per_method]
            fused_mat = np.mean(np.stack(mats, axis=0), axis=0)
        else:
            clf = entry["model"]
            mats = [d[sid] for d in decoded_per_method]
            H, W = mats[0].shape
            X = np.stack([m.reshape(-1) for m in mats], axis=1)   # (HW, M)
            # predict_proba is fast (single matmul + sigmoid).
            p = clf.predict_proba(X)[:, 1].astype(np.float32)
            fused_mat = p.reshape(H, W)
        fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
        fused[sid] = float_matrix_to_q8rle(fused_mat)
        if (i + 1) % 1000 == 0:
            print(f"    fused {i + 1}/{len(all_ids)}  "
                  f"({time.time() - t1:.1f}s)", flush=True)
    if n_uniform:
        print(f"  [warn] {n_uniform} images had no per-class model and "
              f"no shared model; fell back to uniform average.")
    print(f"  fused all {len(all_ids)} in {time.time() - t1:.1f}s")
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Master CSV append
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path,
                    help="2+ submission.csv paths (test predictions)")
    ap.add_argument("--local-preds", nargs="+", required=True, type=Path,
                    help="local_predictions.npz paths, one per --runs "
                         "entry, IN THE SAME ORDER")
    ap.add_argument("--data-root", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/data"))
    ap.add_argument("--class-map", type=Path,
                    help="Optional CSV with ID,class. Otherwise built "
                         "from --data-root by scanning class_XX/test/.")
    ap.add_argument("--rank-normalise", action="store_true", default=True,
                    help="GLOBAL per-method rank-norm on BOTH val and "
                         "test pixels before fitting / predicting "
                         "(default: on). Strongly recommended.")
    ap.add_argument("--no-rank-normalise", dest="rank_normalise",
                    action="store_false",
                    help="Disable rank-normalisation (use raw scores).")
    ap.add_argument("--C", type=float, default=1.0,
                    help="LogisticRegression regularisation strength "
                         "(smaller = stronger regularisation).")
    ap.add_argument("--neg-per-pos", type=int, default=30,
                    help="negatives sampled per positive pixel during fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to fused submission.csv")
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="stacker-logreg")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    if len(args.runs) < 2:
        raise SystemExit("need ≥ 2 methods to stack")
    if len(args.local_preds) != len(args.runs):
        raise SystemExit("--local-preds count must match --runs count")

    method_names = [p.parent.name for p in args.runs]

    print("=" * 78)
    print(f"LOGREG STACKER — {len(args.runs)} methods")
    print("=" * 78)
    for i, (r, lp) in enumerate(zip(args.runs, args.local_preds)):
        print(f"  method {i}: {method_names[i]}")
        print(f"            submission : {r}")
        print(f"            local_preds: {lp}")

    # ── Load submissions
    print("\nLoading test submissions...")
    subs = []
    for p in args.runs:
        s = load_submission(p)
        print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
        subs.append(s)

    # ── Load local-val preds
    print("\nLoading local-val predictions...")
    preds_per_method = []
    for p in args.local_preds:
        d = load_local_preds(p)
        print(f"  {p.parent.name}/{p.name}: "
              f"{len(d['ids'])} val images, "
              f"scores {d['scores'].shape}, masks {d['masks'].shape}, "
              f"{float(d['masks'].mean()) * 100:.3f}% positive")
        preds_per_method.append(d)

    # ── Align across methods
    print("\nAligning local-val predictions across methods...")
    val = align_local_preds(preds_per_method, method_names)

    # ── Rank-normalise val side
    if args.rank_normalise:
        print("\nRank-normalising local-val scores (global, per method)...")
        rank_normalise_in_place(val["scores"])

    # ── Fit per-class logistic regression
    classes = sorted(set(val["classes"].tolist()))
    print(f"\nFitting per-class logistic regression "
          f"(C={args.C}, neg_per_pos={args.neg_per_pos})...")
    print(f"  classes present in val: {classes}")
    models = fit_per_class(val, classes,
                            neg_per_pos=args.neg_per_pos,
                            C=args.C, seed=args.seed)

    # ── Build class map for test
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
        print(f"  [warn] no class map — every test image uses "
              f"the SHARED model")
        class_map = None

    # ── Inference
    fused = fuse_test(subs, models, class_map,
                       rank_norm_test=args.rank_normalise,
                       default_class=default_class)

    # ── Write submission CSV + ZIP
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

    # ── Save model config (coefficients are useful to inspect)
    model_dump = {
        "version": 1,
        "methods": method_names,
        "rank_normalise": bool(args.rank_normalise),
        "C": args.C, "neg_per_pos": args.neg_per_pos, "seed": args.seed,
        "per_class": {},
    }
    for k, v in models.items():
        if "model" in v:
            model_dump["per_class"][k] = {
                "n_pos": v["n_pos"],
                "n_neg_sampled": v["n_neg_sampled"],
                "logloss": v.get("logloss"),
                "train_ap": v.get("train_ap"),
                "coef": v.get("coef"),
                "intercept": v.get("intercept"),
            }
        else:
            model_dump["per_class"][k] = {"fallback_to_shared": True,
                                          "n_pos": v.get("n_pos"),
                                          "n_neg_sampled": v.get("n_neg_sampled")}
    cfg_path = args.out.parent / "stacker_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(model_dump, f, indent=2)
    print(f"Wrote model config -> {cfg_path}")

    # ── Append to ablation_master
    run_id_bits = "stacker_" + hashlib.sha1(
        "|".join(str(p) for p in args.runs).encode("utf-8")
    ).hexdigest()[:6]
    row = {
        "run_id": run_id_bits,
        "run_tag": args.run_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": "STACKER_LOGREG",
        "feature_layers": "",
        "input_size": "",
        "n_classes": len(classes),
        "AP_overall": "",
        "runtime_min": "",
        "submission_path": str(args.out.with_suffix(".zip")),
        "notes": (f"logreg stacker on {len(subs)} methods "
                  f"(rank_norm={int(args.rank_normalise)}, C={args.C}, "
                  f"neg_per_pos={args.neg_per_pos})"),
    }
    append_to_master(args.master_csv, row)
    print(f"\nAppended row to {args.master_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()