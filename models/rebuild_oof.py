#!/usr/bin/env python3
"""rebuild_oof.py — recompute the missing oof_predictions.npz file
for a stacker run that was executed with --tune-mode per-class but
WITHOUT --calibrate, so OOF predictions were computed during tuning
but never persisted to disk.

This script bypasses Optuna entirely: it reads the saved per-class
XGBoost params from {stacker-dir}/stacker_config.json, reloads the
same local_predictions.npz files the stacker used, re-runs LOAO once
per class with those frozen params, and writes oof_predictions.npz
to {stacker-dir} in the exact format postprocess_stacker.py expects.

Walltime: ~5-25 min (8 classes × LOAO with ~6 anomaly buckets each
= ~50 XGBoost fits total). No GPU needed.

# Output

  {stacker-dir}/oof_predictions.npz

  Contains:
    classes        : (C,) object array of class names
    oof_preds_<cls>: (N_cls,) float32, one OOF probability per
                     pixel in the val set for that class
    oof_labels_<cls>: (N_cls,) uint8, matching binary labels

# CLI

    python rebuild_oof.py \\
        --stacker-dir /work/.../runs/<stacker_run> \\
        --runs-dir    /work/.../baseline_out/runs

The methods list is read from {stacker-dir}/stacker_config.json.
For each method `m`, the script expects to find
{runs-dir}/<m>/local_predictions.npz.

# After this finishes

    python postprocess_stacker.py \\
        --stacker-dir <stacker_dir> \\
        --local-preds-ref <one of the local_predictions.npz files> \\
        --data-root /work/.../data \\
        --out-dir <stacker_dir>_pp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the stacker's helpers — must be in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xgboost_stacker import (
    load_local_preds,
    align_local_preds,
    rank_normalise_in_place_val,
    build_training_data,
    loao_oof_one_class,
    FeatureConfig,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stacker-dir", type=Path, required=True,
                    help="Directory containing stacker_config.json "
                         "(and the missing oof_predictions.npz will "
                         "be written here too).")
    ap.add_argument("--runs-dir", type=Path, required=True,
                    help="Parent directory of all run directories. "
                         "For each method m listed in stacker_config.json, "
                         "we look for {runs-dir}/<m>/local_predictions.npz.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. Defaults to "
                         "{stacker-dir}/oof_predictions.npz.")
    args = ap.parse_args()

    out_path = args.out or (args.stacker_dir / "oof_predictions.npz")
    cfg_path = args.stacker_dir / "stacker_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"[FATAL] {cfg_path} not found.")
    if out_path.exists():
        print(f"[warn] {out_path} already exists; will be overwritten.")

    print("=" * 78)
    print("REBUILD OOF  —  recompute oof_predictions.npz for a stacker run")
    print("=" * 78)
    print(f"  stacker_dir : {args.stacker_dir}")
    print(f"  runs_dir    : {args.runs_dir}")
    print(f"  output      : {out_path}")

    print(f"\nLoading {cfg_path}...")
    with open(cfg_path) as f:
        sc = json.load(f)
    if sc.get("version") != 2:
        raise SystemExit(
            f"[FATAL] stacker_config.json version "
            f"{sc.get('version')} unsupported (need v2).")
    params_per_class = sc.get("xgb_params_per_class")
    if not params_per_class:
        raise SystemExit(
            f"[FATAL] no xgb_params_per_class in stacker_config.json. "
            f"This script only handles runs done with --tune-mode "
            f"per-class.")
    methods = sc["methods"]
    feat_cfg_dict = dict(sc["feature_config"])
    # FeatureConfig has tuple fields that JSON serialised as lists.
    for k in ("gauss_sigmas", "window_sizes"):
        if k in feat_cfg_dict and isinstance(feat_cfg_dict[k], list):
            feat_cfg_dict[k] = tuple(feat_cfg_dict[k])
    feat_cfg = FeatureConfig(**feat_cfg_dict)
    rank_normalise = bool(sc.get("rank_normalise", True))
    neg_per_pos = int(sc.get("neg_per_pos", 30))
    seed = int(sc.get("seed", 0))
    tune_cv = str(sc.get("tune_cv", "loao"))
    expected_feat_names = sc.get("feature_names", [])
    print(f"  methods        : {len(methods)}")
    print(f"  rank_normalise : {rank_normalise}")
    print(f"  neg_per_pos    : {neg_per_pos}")
    print(f"  tune_cv        : {tune_cv}")
    print(f"  seed           : {seed}")
    print(f"  feature config : F = {len(expected_feat_names)}")

    # ── Load each method's local_predictions.npz
    print(f"\nLoading local_predictions.npz for {len(methods)} methods...")
    preds_per_method = []
    for m in methods:
        lp_path = args.runs_dir / m / "local_predictions.npz"
        if not lp_path.exists():
            raise SystemExit(
                f"[FATAL] {lp_path} not found.\n"
                f"        Check --runs-dir is the parent of all run\n"
                f"        directories listed in stacker_config.json.")
        d = load_local_preds(lp_path)
        print(f"  {m[:78]}")
        print(f"      {len(d['ids'])} val images, "
              f"{float(d['masks'].mean()) * 100:.3f}% positive pixels")
        preds_per_method.append(d)

    # ── Align (mirror what the stacker did)
    print(f"\nAligning local-val predictions across methods...")
    val = align_local_preds(preds_per_method, methods)
    if rank_normalise:
        print(f"Rank-normalising local-val scores (global per method)...")
        rank_normalise_in_place_val(val["scores"])

    # ── Featurize + sample negatives (mirror stacker)
    classes = sorted(set(val["classes"].tolist()))
    print(f"\nBuilding training data (featurize + sample negatives)...")
    print(f"  classes: {classes}")
    training_data = build_training_data(
        val, classes, feat_cfg, neg_per_pos=neg_per_pos, seed=seed)
    actual_feat_names = training_data.get("_feature_names", [])
    if expected_feat_names and actual_feat_names != expected_feat_names:
        raise SystemExit(
            f"[FATAL] feature drift!\n"
            f"  expected {len(expected_feat_names)} features\n"
            f"  computed {len(actual_feat_names)} features\n"
            f"        Was the FeatureConfig modified between the\n"
            f"        original stacker run and this script?")
    print(f"  feature match: {len(actual_feat_names)} features OK")

    # ── LOAO refits with saved params, one per class
    print(f"\n{'=' * 78}")
    print(f"LOAO REFITS — computing OOF per class")
    print(f"{'=' * 78}")
    oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    t_total = time.time()
    for cls in classes:
        td = training_data.get(cls)
        if td is None or td.get("_fallback_to_shared"):
            print(f"\n  {cls}: fallback-to-shared in stacker; skipping")
            continue
        if cls not in params_per_class:
            print(f"\n  [warn] {cls}: no per-class params; skipping")
            continue
        params = params_per_class[cls]
        n_est = params.get("n_estimators")
        max_depth = params.get("max_depth")
        lr = params.get("learning_rate")
        print(f"\n  {cls}: LOAO  (n_est={n_est}, max_depth={max_depth}, "
              f"lr={lr:.4f})")
        t0 = time.time()
        oof_preds, oof_labels = loao_oof_one_class(
            td, params, seed, cv_mode=tune_cv, neg_per_pos=neg_per_pos)
        elapsed = time.time() - t0
        print(f"    {len(oof_preds):>9d} OOF preds  ({elapsed:.1f}s)")
        try:
            from sklearn.metrics import average_precision_score
            ap = float(average_precision_score(oof_labels, oof_preds))
            stacker_ap = (sc.get("tune_per_class_best_ap") or {}).get(cls)
            if stacker_ap is not None:
                print(f"    OOF pixel-AP = {ap:.4f}  "
                      f"(stacker reported {stacker_ap:.4f})")
            else:
                print(f"    OOF pixel-AP = {ap:.4f}")
        except Exception:
            pass
        oof_per_class[cls] = (oof_preds, oof_labels)
    print(f"\n  total LOAO time: {time.time() - t_total:.1f}s")

    # ── Save
    if not oof_per_class:
        raise SystemExit("[FATAL] no OOF computed; nothing to save.")
    print(f"\nSaving {out_path}...")
    to_save: dict = {"classes": np.array(list(oof_per_class.keys()),
                                           dtype=object)}
    for cls, (op, ol) in oof_per_class.items():
        to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
        to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
    np.savez_compressed(out_path, **to_save)
    print(f"  done.")

    print(f"\nNext step:")
    print(f"  uv run python postprocess_stacker.py \\")
    print(f"      --stacker-dir {args.stacker_dir} \\")
    print(f"      --local-preds-ref {args.runs_dir / methods[0]}/local_predictions.npz \\")
    print(f"      --data-root <your_data_root> \\")
    print(f"      --out-dir {args.stacker_dir}_pp")


if __name__ == "__main__":
    main()