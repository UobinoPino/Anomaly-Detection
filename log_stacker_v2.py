#!/usr/bin/env python3
"""logreg_stacker_v2.py — overfitting diagnostic for xgboost_stacker_v6.

PURPOSE
=======
Run a strongly-regularized linear stacker on the SAME 14 base models,
SAME engineered features, SAME LOAO/LOIO folds, SAME negative sampling
as xgboost_stacker_v6.py. The only thing that changes is the model:

    XGBClassifier(max_depth=4, ...)   ─▶   Pipeline(StandardScaler, LR)

Comparing the two stackers' CV pooled-AP and OOF pooled-AP tells you
how much of the XGB gain comes from real nonlinear signal vs CV
overfitting:

  LR pooled-AP within 1–2 pts of XGB         → XGB is mostly overfitting
                                                the CV; LB gain ≪ CV gain.
                                                Prefer LR for submission.
  LR pooled-AP much lower than XGB           → XGB is doing real nonlinear
                                                work. The CV signal is
                                                trustworthy.
  LR pooled-AP higher than XGB               → XGB is underregularized
                                                AND overfitting. Use LR.

DESIGN
======
All feature/IO/CV code is imported verbatim from xgboost_stacker_v6 so
this script CANNOT silently drift from v6's feature pipeline.

Only differences vs v6:
  - StandardScaler before LR (LR is scale-sensitive; XGB is not).
  - Optuna search over (C, l1_ratio) instead of XGB params.
  - "Feature importance" = |standardized coefficient| (interpretable).
  - Default C=0.1, L2 — strong shrinkage; with ~150 features and ~40k
    training rows per class, weaker regularization memorizes fast.

CLI mirrors v6 closely. Reuses v6's stacker_cache directory so the
expensive aligned-val and decoded-test steps don't repeat.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
import warnings
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Silence sklearn 1.8 FutureWarnings that come from the deprecation path
# we've already migrated away from. Any *new* deprecation is still visible
# because the filter is scoped to two specific message regexes.
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
    message=r".*'penalty' was deprecated.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
    message=r".*'n_jobs' has no effect.*")

# All shared machinery comes from v6. If v6's feature pipeline changes,
# this script automatically tracks it (which is the whole point — the
# logreg result has to be apples-to-apples to function as a diagnostic).
from xgboost_stacker_v6 import (
    Tee, tee_to, hr, sub, mem_print, fmt_bytes,
    float_matrix_to_q8rle, q8rle_to_uint8_matrix, to_f32,
    Cache, file_meta_hash,
    load_submission, load_local_preds, build_class_map_from_data,
    parse_sample_id,
    load_spatial_priors,
    detect_model_family, DEFAULT_SMALL_CC_PER_FAMILY, parse_small_cc_spec,
    suppress_small_ccs_uint8, suppress_small_ccs_volume,
    rank_normalise_global_inplace_val,
    rank_normalise_per_class_inplace_val,
    rank_normalise_per_class_view_inplace_val,
    rank_normalise_test_global_inplace,
    rank_normalise_test_per_class_inplace,
    rank_normalise_test_per_class_view_inplace,
    group_test_ids_by_sample, compute_xv_aggregates_val_f16,
    compute_xv_for_group,
    align_local_preds, save_aligned_val, load_aligned_val,
    MahalanobisParams, fit_mahalanobis_per_class,
    ImgP99Stats, fit_imgp99_stats, pick_top_methods_per_class,
    FeatureConfig, featurize_image, build_training_data,
    free_class_training_data, _spatial_cache,
    _bucketize_one_class, _pixel_ap_pooled,
    fit_calibrator, apply_calibrator,
    verify_stacker_vs_singles, compute_pooled_lb_proxy_ap,
    decode_submissions_to_uint8, cache_decoded_test, load_decoded_test,
    append_to_master,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# LR factory + defaults
#
# Why StandardScaler is non-optional here:
#   v6's features have wildly different scales: rank-norm ∈ [0, 1],
#   Mahalanobis distances are O(M)=O(150)+, one-hot ∈ {0, 1}, image
#   p99 z-scores ∈ ℝ. Without standardization, L2 penalizes large-
#   scale features into uselessness and small-scale features blow up.
#   With standardization the penalty acts uniformly on every feature,
#   which is the whole point of using LR as a regularization baseline.
#
# Why C=0.1 by default:
#   ~150 features × ~40k rows/class is well into the regime where
#   unregularized LR fits training pixels to ≥0.99 AP. Empirically
#   C ∈ [0.01, 1.0] tracks XGB's CV closely when there's real signal.
# ─────────────────────────────────────────────────────────────────────────────
# sklearn 1.8+ replaced `penalty` with `l1_ratio` (a single knob: 0=L2, 1=L1,
# anything in between = elastic-net) and made `n_jobs` a no-op. We use the new
# API directly; minimum sklearn version is 1.8.
DEFAULT_LR_PARAMS: dict = {
    "C": 0.1,
    "l1_ratio": 0.0,           # 0=L2, 1=L1, 0<x<1=elastic-net
    "solver": "lbfgs",         # auto-swapped to saga when l1_ratio > 0
    "max_iter": 500,
    "tol": 1e-3,
    "class_weight": None,      # we already balance via neg-per-pos sampling
}


def _resolve_solver(l1_ratio: float, current: str) -> str:
    # lbfgs / newton-cg / newton-cholesky support pure L2 only.
    # saga supports L1 and elastic-net; we route everything except pure L2
    # there. (liblinear could handle L1 too but doesn't support warm_start
    # or elastic-net, so saga is the simpler choice.)
    return "saga" if l1_ratio > 0.0 else current


def _make_logreg(params: dict, seed: int) -> Pipeline:
    if not HAS_SKLEARN:
        raise SystemExit("scikit-learn not installed. uv pip install scikit-learn")
    p = dict(DEFAULT_LR_PARAMS); p.update(params or {})
    # Old-style `penalty` may sneak in via legacy CLI or pickled trial state;
    # translate it transparently to `l1_ratio` so callers don't have to care.
    legacy_penalty = p.pop("penalty", None)
    if legacy_penalty is not None and "l1_ratio" not in (params or {}):
        if   legacy_penalty == "l2":         p["l1_ratio"] = 0.0
        elif legacy_penalty == "l1":         p["l1_ratio"] = 1.0
        elif legacy_penalty == "elasticnet": p["l1_ratio"] = p.get("l1_ratio", 0.5)
    # n_jobs is a no-op in 1.8 and emits FutureWarning. Drop it.
    p.pop("n_jobs", None)
    p["solver"] = _resolve_solver(float(p.get("l1_ratio", 0.0)),
                                    p.get("solver", "lbfgs"))
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(random_state=seed, **p)),
    ])


def get_params_for_class(cls: str, params_global: dict,
                          params_per_class: dict | None) -> dict:
    if params_per_class and cls in params_per_class:
        return params_per_class[cls]
    return params_global


# ─────────────────────────────────────────────────────────────────────────────
# CV / OOF — copies of v6 functions with _make_xgb → _make_logreg.
# Everything else (bucketize, neg-sampling, pooled-AP) is reused.
# ─────────────────────────────────────────────────────────────────────────────
def pooled_cv_score_one_class(td: dict, params: dict, seed: int,
                                mode: str = "loao",
                                neg_per_pos: int = 30) -> float:
    rng = np.random.default_rng(seed)
    X_full = td["X_full"]; y_full = td["y_full"]; ranges = td["img_pixranges"]
    buckets = _bucketize_one_class(td, mode)
    fold_aps: list[float] = []
    for held_imgs in buckets.values():
        held_set = set(held_imgs)
        train_imgs = [i for i in range(len(ranges)) if i not in held_set]
        Xs, ys = [], []
        for ti in train_imgs:
            s, e = ranges[ti]; yp = y_full[s:e]
            pos = np.flatnonzero(yp == 1)
            if pos.size == 0: continue
            neg = np.flatnonzero(yp == 0)
            n_keep = min(neg.size, pos.size * neg_per_pos)
            sneg = (rng.choice(neg, n_keep, replace=False)
                    if n_keep < neg.size else neg)
            keep = np.concatenate([pos, sneg])
            Xs.append(X_full[s:e][keep]); ys.append(yp[keep])
        if not Xs: continue
        clf = _make_logreg(params, seed)
        clf.fit(np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
        held_preds, held_labels = [], []
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            held_preds.append(p); held_labels.append(y_full[s:e].astype(np.int32))
        hpc = np.concatenate(held_preds); hlc = np.concatenate(held_labels)
        if int(hlc.sum()) > 0:
            fold_aps.append(_pixel_ap_pooled(hpc, hlc))
    valid = [a for a in fold_aps if not math.isnan(a)]
    return float(np.mean(valid)) if valid else 0.0


def loao_oof_one_class(td: dict, params: dict, seed: int,
                        cv_mode: str = "loao", neg_per_pos: int = 30):
    rng = np.random.default_rng(seed)
    X_full = td["X_full"]; y_full = td["y_full"]; ranges = td["img_pixranges"]
    buckets = _bucketize_one_class(td, cv_mode)
    oof_preds = np.full(y_full.shape, np.nan, dtype=np.float32)
    for held_imgs in buckets.values():
        held_set = set(held_imgs)
        train_imgs = [i for i in range(len(ranges)) if i not in held_set]
        Xs, ys = [], []
        for ti in train_imgs:
            s, e = ranges[ti]; yp = y_full[s:e]
            pos = np.flatnonzero(yp == 1)
            if pos.size == 0: continue
            neg = np.flatnonzero(yp == 0)
            n_keep = min(neg.size, pos.size * neg_per_pos)
            sneg = (rng.choice(neg, n_keep, replace=False)
                    if n_keep < neg.size else neg)
            keep = np.concatenate([pos, sneg])
            Xs.append(X_full[s:e][keep]); ys.append(yp[keep])
        if not Xs: continue
        clf = _make_logreg(params, seed)
        clf.fit(np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            oof_preds[s:e] = p
    mask = ~np.isnan(oof_preds)
    return oof_preds[mask].astype(np.float32), y_full[mask].astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Optuna — search over LR hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
def _optuna_suggest_lr(trial):
    # l1_ratio is now a continuous knob: 0=L2, 1=L1, in-between=elastic-net.
    # We bias the prior slightly toward L2 (the value 0 still has positive
    # probability under TPE because the lower bound is inclusive) since L2
    # is well-known to be the strongest baseline on dense rank-normed
    # features.
    return {
        "C":        trial.suggest_float("C", 1e-3, 1.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def tune_global(training_data, *, n_trials, seed, cv_mode="loao",
                timeout=None, neg_per_pos=30):
    if not HAS_OPTUNA: raise SystemExit("optuna not installed.")
    classes = [c for c in training_data if not c.startswith("_")
                and not training_data[c].get("_fallback_to_shared")]
    def objective(trial):
        params = _optuna_suggest_lr(trial)
        aps = [pooled_cv_score_one_class(training_data[cls], params, seed,
                                            mode=cv_mode,
                                            neg_per_pos=neg_per_pos)
                for cls in classes]
        return float(np.mean(aps)) if aps else 0.0
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    print(f"\n>>> Global Optuna tuning (LR): {n_trials} trials, CV={cv_mode}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f">>> best mean pooled-AP: {study.best_value:.4f}")
    print(f">>> best params: {json.dumps(study.best_params, indent=2)}")
    return study.best_params


def tune_per_class(training_data, *, n_trials, seed, cv_mode="loao",
                    timeout_per_class=None, neg_per_pos=30):
    if not HAS_OPTUNA: raise SystemExit("optuna not installed.")
    out: dict = {}
    for cls, td in training_data.items():
        if cls.startswith("_"): continue
        if td.get("_fallback_to_shared"):
            print(f"\n  class {cls}: fallback-to-shared, no per-class tuning")
            continue
        print(f"\n>>> Per-class tuning (LR): {cls}  (CV={cv_mode}, "
              f"n_trials={n_trials})")
        def objective(trial, _td=td):
            return pooled_cv_score_one_class(_td, _optuna_suggest_lr(trial),
                                                seed, mode=cv_mode,
                                                neg_per_pos=neg_per_pos)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler,
                                      study_name=f"lr_{cls}")
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials,
                        timeout=timeout_per_class)
        print(f"  {cls}: best CV pooled-AP = {study.best_value:.4f}  "
              f"({time.time() - t0:.1f}s, {len(study.trials)} trials)")
        print(f"  {cls}: best params = {study.best_params}")
        oof_preds, oof_labels = loao_oof_one_class(
            td, study.best_params, seed, cv_mode=cv_mode,
            neg_per_pos=neg_per_pos)
        out[cls] = {"best_params": study.best_params,
                    "best_cv_ap": float(study.best_value),
                    "n_trials": len(study.trials),
                    "oof_preds": oof_preds, "oof_labels": oof_labels}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-class production fit
# ─────────────────────────────────────────────────────────────────────────────
def fit_per_class(training_data, params_global, params_per_class, seed,
                   min_pos_for_per_class=200) -> dict:
    from sklearn.metrics import log_loss, average_precision_score
    feature_names = training_data.get("_feature_names", [])
    out: dict = {"_feature_names": feature_names}
    shared_X, shared_y = [], []
    for cls, td in training_data.items():
        if cls.startswith("_"): continue
        if td.get("_fallback_to_shared"):
            out[cls] = {"_fallback_to_shared": True,
                        "n_pos": td.get("n_pos", 0),
                        "n_neg_sampled": td.get("n_neg_sampled", 0)}
            continue
        X = td["X_train"]; y = td["y_train"]
        shared_X.append(X); shared_y.append(y)
        if td["n_pos"] < min_pos_for_per_class:
            print(f"  class {cls}: only {td['n_pos']} positives "
                  f"(< {min_pos_for_per_class}); using SHARED model")
            out[cls] = {"_fallback_to_shared": True,
                        "n_pos": td["n_pos"],
                        "n_neg_sampled": td["n_neg_sampled"]}
            continue
        params = get_params_for_class(cls, params_global, params_per_class)
        clf = _make_logreg(params, seed)
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        try: ll = float(log_loss(y, p, labels=[0, 1]))
        except Exception: ll = float("nan")
        try: ap = float(average_precision_score(y, p))
        except Exception: ap = float("nan")
        coef = clf.named_steps["clf"].coef_[0].astype(np.float32)
        intercept = float(clf.named_steps["clf"].intercept_[0])
        out[cls] = {"model": clf, "n_pos": td["n_pos"],
                    "n_neg_sampled": td["n_neg_sampled"],
                    "logloss": ll, "train_ap": ap, "params": params,
                    "coef": coef.tolist(), "intercept": intercept,
                    "feature_importance": np.abs(coef).tolist()}
        print(f"  class {cls}: n_pos={td['n_pos']:>6d}  "
              f"n_neg={td['n_neg_sampled']:>8d}  "
              f"train_logloss={ll:.4f}  train_ap={ap:.3f}  "
              f"||w||₂={np.linalg.norm(coef):.3f}  b={intercept:+.3f}")
    if shared_X:
        X_all = np.concatenate(shared_X, axis=0)
        y_all = np.concatenate(shared_y, axis=0)
        del shared_X, shared_y; gc.collect()
        clf_sh = _make_logreg(params_global, seed)
        clf_sh.fit(X_all, y_all)
        p_sh = clf_sh.predict_proba(X_all)[:, 1]
        try: ll_sh = float(log_loss(y_all, p_sh, labels=[0, 1]))
        except Exception: ll_sh = float("nan")
        try: ap_sh = float(average_precision_score(y_all, p_sh))
        except Exception: ap_sh = float("nan")
        coef = clf_sh.named_steps["clf"].coef_[0].astype(np.float32)
        intercept = float(clf_sh.named_steps["clf"].intercept_[0])
        out["_SHARED_"] = {"model": clf_sh,
                            "n_pos": int(y_all.sum()),
                            "n_neg_sampled": int(len(y_all) - y_all.sum()),
                            "logloss": ll_sh, "train_ap": ap_sh,
                            "params": params_global,
                            "coef": coef.tolist(),
                            "intercept": intercept,
                            "feature_importance": np.abs(coef).tolist()}
        print(f"  SHARED:    n_pos={int(y_all.sum()):>6d}  "
              f"n_neg={int(len(y_all) - y_all.sum()):>8d}  "
              f"train_logloss={ll_sh:.4f}  train_ap={ap_sh:.3f}")
        del X_all, y_all; gc.collect()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# fuse_test — same shape as v6 (LR is wrapped in a Pipeline so
# .predict_proba is the same interface as XGB)
# ─────────────────────────────────────────────────────────────────────────────
def fuse_test(submissions, models, class_map, cfg: FeatureConfig,
               rank_norm_mode: str, default_class: str,
               calibrators_per_class, small_cc_per_method,
               include_xv: bool, spatial_priors, all_classes_onehot,
               mahal_per_class, top_methods_per_class, imgp99_stats,
               *, cache: Cache | None = None, cache_key_base: str = "",
               storage_dtype: str = "uint16",
               drop_decoded_during_fusion: bool = True) -> dict[str, str]:
    cache = cache or Cache(None)
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common: raise RuntimeError("no test IDs in common")
    all_ids = sorted(common)
    M = len(submissions)
    feature_names = models.get("_feature_names", None)

    # 1. decode (cached)
    decoded_per_method = None
    raw_key = f"decoded_raw_{cache_key_base}.pkl"
    if cache.has(raw_key):
        cached = load_decoded_test(cache, raw_key)
        if (cached is not None and len(cached) == M
                and set(cached[0].keys()) >= set(all_ids)):
            decoded_per_method = cached
    if decoded_per_method is None:
        print(f"\nDecoding {M} submissions × {len(all_ids)} IDs (uint8)...")
        decoded_per_method = decode_submissions_to_uint8(submissions, all_ids)
        if any(c > 0 for c in small_cc_per_method):
            for mi, min_cc in enumerate(small_cc_per_method):
                if min_cc <= 0: continue
                for sid in all_ids:
                    decoded_per_method[mi][sid] = suppress_small_ccs_uint8(
                        decoded_per_method[mi][sid], min_cc=min_cc)
        cache_decoded_test(cache, raw_key, decoded_per_method)

    # 2. rank-norm (cached)
    norm_key = (f"decoded_norm_{rank_norm_mode}_{storage_dtype}_"
                 f"{cache_key_base}.pkl")
    norm_cached = (load_decoded_test(cache, norm_key)
                    if cache.has(norm_key) else None)
    if (norm_cached is not None and len(norm_cached) == M
            and set(norm_cached[0].keys()) >= set(all_ids)):
        decoded_per_method = norm_cached
    else:
        if rank_norm_mode == "per-class-view":
            rank_normalise_test_per_class_view_inplace(
                decoded_per_method, all_ids, class_map, default_class,
                out_dtype=storage_dtype)
        elif rank_norm_mode == "per-class":
            rank_normalise_test_per_class_inplace(
                decoded_per_method, all_ids, class_map, default_class,
                out_dtype=storage_dtype)
        elif rank_norm_mode == "global":
            rank_normalise_test_global_inplace(
                decoded_per_method, all_ids, out_dtype=storage_dtype)
        cache_decoded_test(cache, norm_key, decoded_per_method)
    gc.collect()

    # 3. stream-fuse per sample group
    groups = group_test_ids_by_sample(all_ids, class_map, default_class)
    print(f"\nFusing {len(all_ids)} images in {len(groups)} groups...")
    fused: dict[str, str] = {}
    shared_entry = models.get("_SHARED_")
    t1 = time.time(); n_uniform = 0
    spatial_cache = None; n_done = 0
    for (cls, sample_id), ids_in_sample in sorted(groups.items(),
                                                     key=lambda kv: kv[0]):
        V = len(ids_in_sample); first_sid = ids_in_sample[0]
        H, W = decoded_per_method[0][first_sid].shape
        group_scores_f32 = np.empty((V, M, H, W), dtype=np.float32)
        for k, sid in enumerate(ids_in_sample):
            for mi in range(M):
                group_scores_f32[k, mi] = to_f32(decoded_per_method[mi][sid])
        if include_xv and V >= 2:
            xv_max_g, xv_mean_g, xv_std_g, xv_lonely_g = compute_xv_for_group(
                group_scores_f32)
        else:
            xv_max_g = xv_mean_g = xv_std_g = xv_lonely_g = None
        entry = models.get(cls)
        if entry is None or entry.get("_fallback_to_shared"):
            entry = shared_entry
        prior_cls = (spatial_priors.get(cls) if spatial_priors else None)
        mahal_cls = (mahal_per_class.get(cls) if mahal_per_class else None)
        top_methods_cls = (top_methods_per_class.get(cls)
                              if top_methods_per_class else None)
        for k, sid in enumerate(ids_in_sample):
            _, v = parse_sample_id(sid)
            v_int = int(v) if v is not None else None
            if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
                spatial_cache = _spatial_cache(H, W)
            scores_this = [group_scores_f32[k, mi] for mi in range(M)]
            if entry is None or "model" not in entry:
                n_uniform += 1
                fused_mat = np.mean(np.stack(scores_this, axis=0), axis=0)
            else:
                if xv_max_g is not None:
                    xv_max_list    = [xv_max_g[k, mi]    for mi in range(M)]
                    xv_mean_list   = [xv_mean_g[k, mi]   for mi in range(M)]
                    xv_std_list    = [xv_std_g[k, mi]    for mi in range(M)]
                    xv_lonely_list = [xv_lonely_g[k, mi] for mi in range(M)]
                    is_mv_xv = True
                else:
                    xv_max_list = xv_mean_list = xv_std_list = xv_lonely_list = None
                    is_mv_xv = False
                feats, _ = featurize_image(
                    scores_this, cfg, spatial_cache=spatial_cache,
                    feature_names=feature_names,
                    is_multiview_sample=is_mv_xv,
                    class_id=cls, all_classes=all_classes_onehot,
                    view=v_int, spatial_prior=prior_cls,
                    xv_max_per_method=xv_max_list,
                    xv_mean_per_method=xv_mean_list,
                    xv_std_per_method=xv_std_list,
                    xv_lonely_per_method=xv_lonely_list,
                    mahal_params=mahal_cls,
                    top_methods_for_cls=top_methods_cls,
                    imgp99_stats=imgp99_stats)
                X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
                p_raw = entry["model"].predict_proba(X)[:, 1].astype(np.float32)
                cal = (calibrators_per_class.get(cls)
                       if calibrators_per_class else None)
                p_out = (apply_calibrator(cal, p_raw)
                          if cal is not None else p_raw)
                fused_mat = p_out.reshape(H, W)
                del feats, X, p_raw
            fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
            fused[sid] = float_matrix_to_q8rle(fused_mat)
            n_done += 1
            if n_done % 500 == 0:
                print(f"    fused {n_done}/{len(all_ids)}  "
                      f"({time.time() - t1:.1f}s)", flush=True)
        if drop_decoded_during_fusion:
            for sid in ids_in_sample:
                for mi in range(M):
                    decoded_per_method[mi].pop(sid, None)
        del group_scores_f32
        if xv_max_g is not None:
            del xv_max_g, xv_mean_g, xv_std_g, xv_lonely_g
    if n_uniform:
        print(f"  [warn] {n_uniform} images had no model — uniform avg")
    print(f"  fused all {len(all_ids)} in {time.time() - t1:.1f}s")
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--local-preds", nargs="+", required=True, type=Path)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/data"))
    ap.add_argument("--class-map", type=Path)
    ap.add_argument("--rank-norm", default="per-class",
                    choices=["per-class-view", "per-class", "global", "none"])
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="stacker-logreg-v2")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--small-cc", nargs="*", default=None)
    ap.add_argument("--small-cc-default", type=int, default=0)
    ap.add_argument("--no-small-cc", action="store_true")
    # Feature toggles — MIRROR v6 defaults for apples-to-apples
    ap.add_argument("--no-cross-method-consensus", action="store_true")
    ap.add_argument("--prior-heatmaps-dir", type=Path,
                    default=Path("analysis_out/tables"))
    ap.add_argument("--no-spatial-prior", action="store_true")
    ap.add_argument("--no-class-onehot", action="store_true")
    ap.add_argument("--no-view-onehot", action="store_true")
    ap.add_argument("--no-xv-aggregates", action="store_true")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-cross-stats", action="store_true")
    ap.add_argument("--no-image-aggregates", action="store_true")
    ap.add_argument("--no-mahalanobis", action="store_true")
    ap.add_argument("--no-min-top-k", action="store_true")
    ap.add_argument("--top-k-for-min", type=int, default=3)
    ap.add_argument("--no-cc-features", action="store_true")
    ap.add_argument("--n-top-methods-for-cc", type=int, default=3)
    ap.add_argument("--cc-top-pct", type=float, default=98.0)
    ap.add_argument("--no-per-method-zrank-top", action="store_true")
    ap.add_argument("--no-cross-method-cv", action="store_true")
    # ★ LR-specific overrides ★
    ap.add_argument("--C", type=float, default=None,
                    help="LR inverse-regularization strength (default 0.1).")
    ap.add_argument("--l1-ratio", type=float, default=None,
                    help="LR penalty mix: 0=L2 (default), 1=L1, "
                         "between=elastic-net. Replaces --penalty in "
                         "sklearn 1.8+.")
    ap.add_argument("--penalty", default=None,
                    choices=["l2", "l1", "elasticnet"],
                    help="[LEGACY] sklearn <1.8 penalty name. Translated "
                         "internally to --l1-ratio. Prefer --l1-ratio.")
    # Tuning + calibration
    ap.add_argument("--tune-mode", default="none",
                    choices=["none", "global", "per-class"])
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--tune-cv", default="loao", choices=["loao", "loio"])
    ap.add_argument("--tune-timeout-min", type=float, default=None)
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "platt", "isotonic"])
    ap.add_argument("--no-verify", action="store_true")
    # Cache + storage (shared with v6 so warm-up is free)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--test-storage-dtype", default="uint16",
                    choices=["uint16", "float16", "float32"])
    ap.add_argument("--no-free-class-data", action="store_true")
    ap.add_argument("--no-drop-decoded-during-fusion", action="store_true")
    # ★ DIAGNOSTIC FLAGS — drop suspect-leaky features ★
    ap.add_argument("--drop-leaky-top-methods", action="store_true",
                    help="Disable pick_top_methods_per_class + the CC and "
                         "zrank features that depend on it. These features "
                         "are selected by validating on the SAME data they "
                         "later feature in, which is a textbook leak.")
    ap.add_argument("--drop-leaky-mahalanobis", action="store_true",
                    help="Disable Mahalanobis (fit on val negative pixels, "
                         "applied to same val for training).")
    args = ap.parse_args()

    if not HAS_SKLEARN:
        raise SystemExit("[FATAL] scikit-learn not installed.")
    if len(args.runs) < 2:
        raise SystemExit("need ≥ 2 methods to stack")
    if len(args.local_preds) != len(args.runs):
        raise SystemExit("--local-preds count must match --runs count")
    if args.tune_mode != "none" and not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")

    method_names = [p.parent.name for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.out.parent
    cache = Cache(args.cache_dir)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"LOGREG STACKER v2 — {len(args.runs)} methods", "=")
        print(f"  rank_norm        : {args.rank_norm}")
        print(f"  tune_mode        : {args.tune_mode}")
        print(f"  calibrate        : {args.calibrate}")
        print(f"  cache_dir        : {args.cache_dir}")
        print(f"  drop-leaky-top   : {args.drop_leaky_top_methods}")
        print(f"  drop-leaky-mahal : {args.drop_leaky_mahalanobis}")
        for i in range(len(args.runs)):
            fam = detect_model_family(method_names[i])
            print(f"  method {i}: {method_names[i]}  (family={fam})")
        mem_print("start")

        # Small-CC
        small_cc_per_method = ([0] * len(method_names) if args.no_small_cc
                                 else parse_small_cc_spec(
                                     args.small_cc, method_names,
                                     args.small_cc_default))
        print(f"\n  small-CC suppression per method: {small_cc_per_method}")

        # Cache keys (compatible with v6's cache; first run warms it)
        local_preds_meta = file_meta_hash(
            args.local_preds, extra=f"smallcc={small_cc_per_method}")
        runs_meta = file_meta_hash(args.runs)
        cache_key_val   = f"val_aligned_{local_preds_meta}.npz"
        cache_key_test  = f"{runs_meta}__{local_preds_meta}"

        # Aligned val
        val = load_aligned_val(cache, cache_key_val)
        if val is None:
            print("\nLoading test submissions + local-preds...")
            subs = [load_submission(p) for p in args.runs]
            preds_per_method = []
            for p in args.local_preds:
                d = load_local_preds(p)
                print(f"  {p.parent.name}/{p.name}: {len(d['ids'])} val images")
                preds_per_method.append(d)
            if any(c > 0 for c in small_cc_per_method):
                for mi, min_cc in enumerate(small_cc_per_method):
                    if min_cc <= 0: continue
                    preds_per_method[mi]["scores"] = suppress_small_ccs_volume(
                        preds_per_method[mi]["scores"], min_cc=min_cc)
            print("\nAligning local-val predictions across methods...")
            val = align_local_preds(preds_per_method, method_names)
            del preds_per_method; gc.collect()
            save_aligned_val(cache, cache_key_val, val)
        else:
            print("\nLoading test submissions...")
            subs = [load_submission(p) for p in args.runs]
        mem_print("after val align")

        # Rank-norm val
        any_has_paths = val.get("image_paths") is not None
        if args.rank_norm == "per-class-view":
            if (val["views"] >= 0).any():
                print(f"\nPer-(class, view) rank-norm (val)...")
                rank_normalise_per_class_view_inplace_val(
                    val["scores"], val["classes"], val["views"])
            else:
                args.rank_norm = "per-class"
                rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
        elif args.rank_norm == "per-class":
            print(f"\nPer-class rank-norm (val)...")
            rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
        elif args.rank_norm == "global":
            print(f"\nGlobal rank-norm (val)...")
            rank_normalise_global_inplace_val(val["scores"])
        mem_print("after val rank-norm")

        # XV aggregates
        use_xv = (not args.no_xv_aggregates) and any_has_paths
        xv_max_val = xv_mean_val = xv_std_val = xv_lonely_val = None
        is_multi_xv_val = None
        if use_xv:
            print(f"\nBuilding cross-view AGGREGATES (val, float16)...")
            (xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val,
             is_multi_xv_val) = compute_xv_aggregates_val_f16(
                val["scores"], val["classes"], val.get("image_paths"))

        # Spatial priors
        classes = sorted(set(val["classes"].tolist()))
        spatial_priors = None
        if not args.no_spatial_prior:
            print(f"\nLoading per-class spatial priors...")
            spatial_priors = load_spatial_priors(args.prior_heatmaps_dir, classes)

        # v6 features that may leak: gated by diagnostic flags
        mahal_per_class = None
        if not (args.no_mahalanobis or args.drop_leaky_mahalanobis):
            mahal_per_class = fit_mahalanobis_per_class(
                val["scores"], val["masks"], val["classes"], seed=args.seed)
        elif args.drop_leaky_mahalanobis:
            print(f"\n[diag] Mahalanobis DISABLED via --drop-leaky-mahalanobis")

        top_methods_per_class = None
        if not (args.no_cc_features and args.no_per_method_zrank_top
                 or args.drop_leaky_top_methods):
            top_methods_per_class = pick_top_methods_per_class(
                val["scores"], val["masks"], val["classes"],
                top_k=args.n_top_methods_for_cc)
        elif args.drop_leaky_top_methods:
            print(f"\n[diag] pick_top_methods + CC + zrank features "
                  f"DISABLED via --drop-leaky-top-methods")

        imgp99_stats = None
        if not args.no_per_method_zrank_top and top_methods_per_class is not None:
            imgp99_stats = fit_imgp99_stats(val["scores"])

        cfg = FeatureConfig(
            use_spatial=not args.no_spatial,
            use_cross_stats=not args.no_cross_stats,
            use_image_aggregates=not args.no_image_aggregates,
            use_cross_method_consensus=not args.no_cross_method_consensus,
            use_spatial_prior=(spatial_priors is not None),
            use_class_onehot=not args.no_class_onehot,
            use_view_onehot=not args.no_view_onehot,
            use_xv_aggregates=use_xv,
            use_mahalanobis=(mahal_per_class is not None),
            use_min_top_k=not args.no_min_top_k,
            top_k_for_min=args.top_k_for_min,
            use_cc_features=(not args.no_cc_features
                              and top_methods_per_class is not None),
            n_top_methods_for_cc=args.n_top_methods_for_cc,
            cc_top_pct=args.cc_top_pct,
            use_per_method_zrank_top=(not args.no_per_method_zrank_top
                                          and top_methods_per_class is not None
                                          and imgp99_stats is not None),
            use_cross_method_cv=not args.no_cross_method_cv,
        )
        print(f"\nFeature config:")
        for k, v in asdict(cfg).items():
            print(f"  {k:<32} = {v}")
        all_classes_onehot = classes if cfg.use_class_onehot else None

        print(f"\nBuilding training matrices...")
        print(f"  classes present in val: {classes}")
        training_data = build_training_data(
            val, classes, cfg,
            neg_per_pos=args.neg_per_pos, seed=args.seed,
            xv_max=xv_max_val, xv_mean=xv_mean_val,
            xv_std=xv_std_val, xv_lonely=xv_lonely_val,
            is_multi=is_multi_xv_val,
            spatial_priors=spatial_priors,
            all_classes_onehot=all_classes_onehot,
            mahal_per_class=mahal_per_class,
            top_methods_per_class=top_methods_per_class,
            imgp99_stats=imgp99_stats)
        feature_names = training_data.get("_feature_names", [])
        print(f"  total features per pixel: {len(feature_names)}")
        if xv_max_val is not None:
            del xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val
            gc.collect()

        # LR params — fold legacy --penalty into --l1-ratio if the user
        # passed it. --l1-ratio takes precedence if both are supplied.
        cli_overrides: dict = {}
        if args.C is not None:        cli_overrides["C"] = args.C
        if args.l1_ratio is not None: cli_overrides["l1_ratio"] = args.l1_ratio
        elif args.penalty is not None:
            cli_overrides["l1_ratio"] = (
                0.0 if args.penalty == "l2"
                else 1.0 if args.penalty == "l1"
                else 0.5)
        params_global = {**DEFAULT_LR_PARAMS, **cli_overrides}
        params_per_class = None
        tune_per_class_results = None
        timeout_sec = (args.tune_timeout_min * 60.0
                       if args.tune_timeout_min else None)

        if args.tune_mode == "global":
            tuned = tune_global(training_data, n_trials=args.n_trials,
                                  seed=args.seed, cv_mode=args.tune_cv,
                                  timeout=timeout_sec,
                                  neg_per_pos=args.neg_per_pos)
            params_global = {**params_global, **tuned}
        elif args.tune_mode == "per-class":
            tune_per_class_results = tune_per_class(
                training_data, n_trials=args.n_trials,
                seed=args.seed, cv_mode=args.tune_cv,
                timeout_per_class=timeout_sec,
                neg_per_pos=args.neg_per_pos)
            params_per_class = {cls: r["best_params"]
                                for cls, r in tune_per_class_results.items()}

        # OOF predictions
        oof_per_class = {}
        if tune_per_class_results is not None:
            for cls, r in tune_per_class_results.items():
                oof_per_class[cls] = (r["oof_preds"], r["oof_labels"])
        else:
            print(f"\nCollecting OOF preds per class (LR)...")
            for cls in [c for c in training_data
                         if not c.startswith("_")
                         and not training_data[c].get("_fallback_to_shared")]:
                td = training_data[cls]
                params = get_params_for_class(cls, params_global,
                                                params_per_class)
                t0 = time.time()
                op, ol = loao_oof_one_class(
                    td, params, args.seed, cv_mode=args.tune_cv,
                    neg_per_pos=args.neg_per_pos)
                print(f"  {cls}: {len(op):>9d} OOF preds "
                      f"({time.time() - t0:.1f}s)")
                oof_per_class[cls] = (op, ol)

        # Calibration
        calibrators_per_class = None
        if args.calibrate != "none":
            print(f"\nFitting per-class {args.calibrate} calibrators...")
            calibrators_per_class = {}
            from sklearn.metrics import average_precision_score
            for cls, (op, ol) in oof_per_class.items():
                cal = fit_calibrator(args.calibrate, op, ol)
                calibrators_per_class[cls] = cal
                try: ap_pre = float(average_precision_score(ol, op))
                except Exception: ap_pre = float("nan")
                p_cal = apply_calibrator(cal, op)
                try: ap_post = float(average_precision_score(ol, p_cal))
                except Exception: ap_post = float("nan")
                print(f"  {cls}: OOF pooled-AP pre={ap_pre:.4f} "
                      f"post={ap_post:.4f}")

        # Verify
        verification = {}
        if not args.no_verify and oof_per_class:
            verification = verify_stacker_vs_singles(
                val, method_names, oof_per_class)

        del val; gc.collect()
        mem_print("after val freed")

        # Production fits
        print(f"\nFitting final per-class LR models...")
        models = fit_per_class(training_data, params_global,
                                params_per_class, args.seed)
        if not args.no_free_class_data:
            for cls in list(training_data.keys()):
                if cls.startswith("_"): continue
                free_class_training_data(training_data, cls)
            gc.collect()

        # Class map for test
        print("\nBuilding ID → class map for test set...")
        if args.class_map and args.class_map.exists():
            class_map = {}
            with open(args.class_map, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "ID" in row and "class" in row:
                        class_map[row["ID"]] = row["class"]
        else:
            class_map = build_class_map_from_data(args.data_root)
        default_class = classes[0] if classes else "_default_"
        if not class_map:
            print(f"  [warn] no class map — every test image uses SHARED model")
            class_map = None

        # Fuse
        fused = fuse_test(
            subs, models, class_map, cfg,
            rank_norm_mode=args.rank_norm,
            default_class=default_class,
            calibrators_per_class=calibrators_per_class,
            small_cc_per_method=small_cc_per_method,
            include_xv=use_xv,
            spatial_priors=spatial_priors,
            all_classes_onehot=all_classes_onehot,
            mahal_per_class=mahal_per_class,
            top_methods_per_class=top_methods_per_class,
            imgp99_stats=imgp99_stats,
            cache=cache, cache_key_base=cache_key_test,
            storage_dtype=args.test_storage_dtype,
            drop_decoded_during_fusion=(not args.no_drop_decoded_during_fusion))
        del subs; gc.collect()

        # Write submission
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ID", "Label"])
            for sid in sorted(fused):
                w.writerow([sid, fused[sid]])
        print(f"\nWrote {len(fused)} rows -> {args.out}")
        if not args.no_zip:
            zip_path = args.out.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w",
                                 compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(args.out, arcname=args.out.name)
            print(f"Zipped -> {zip_path}")

        # OOF dump (for downstream comparison with XGB)
        if oof_per_class:
            oof_path = run_dir / "oof_predictions.npz"
            to_save = {"classes": np.array(list(oof_per_class.keys()),
                                            dtype=object)}
            for cls, (op, ol) in oof_per_class.items():
                to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
                to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
            np.savez_compressed(oof_path, **to_save)
            print(f"Saved OOF preds -> {oof_path}")

        # Config + coefficient dump
        model_dump = {
            "version": "logreg-v2",
            "stacker_family": "LR",
            "methods": method_names,
            "rank_norm": args.rank_norm,
            "small_cc_per_method": small_cc_per_method,
            "neg_per_pos": args.neg_per_pos,
            "seed": args.seed,
            "feature_config": asdict(cfg),
            "feature_names": feature_names,
            "lr_params_global": params_global,
            "lr_params_per_class": params_per_class,
            "tune_mode": args.tune_mode,
            "tune_cv": args.tune_cv,
            "calibration_method": args.calibrate,
            "drop_leaky_top_methods": args.drop_leaky_top_methods,
            "drop_leaky_mahalanobis": args.drop_leaky_mahalanobis,
            "verification": {
                k: v for k, v in (verification or {}).items()
                if k != "per_class_single"
            },
            "per_class_models": {},
        }
        for k, v in models.items():
            if k.startswith("_") and k != "_SHARED_": continue
            if isinstance(v, dict) and "model" in v:
                model_dump["per_class_models"][k] = {
                    "n_pos": v["n_pos"],
                    "n_neg_sampled": v["n_neg_sampled"],
                    "train_logloss": v.get("logloss"),
                    "train_ap": v.get("train_ap"),
                    "params": v.get("params"),
                    "coef": v.get("coef"),
                    "intercept": v.get("intercept"),
                    "feature_importance": v.get("feature_importance"),
                }
            else:
                model_dump["per_class_models"][k] = {
                    "fallback_to_shared": True,
                    "n_pos": v.get("n_pos"),
                    "n_neg_sampled": v.get("n_neg_sampled"),
                }
        cfg_path = run_dir / "stacker_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(model_dump, f, indent=2, default=str)
        print(f"\nWrote LR stacker config -> {cfg_path}")

        # Top coefficients (signed!) — interpretable
        if ("_SHARED_" in models and "coef" in models["_SHARED_"]
                and feature_names):
            coef = np.asarray(models["_SHARED_"]["coef"])
            if len(coef) == len(feature_names):
                order = np.argsort(-np.abs(coef))[:25]
                print(f"\nTop-25 |coef| (SHARED, standardized features):")
                for r, j in enumerate(order, 1):
                    sign = "+" if coef[j] >= 0 else "-"
                    print(f"  {r:>2d}. {feature_names[j]:<30s}  "
                          f"{sign}{abs(coef[j]):.4f}")

        # Ablation row — DIRECTLY comparable to XGB row
        run_id = "stacker_lr_v2_" + hashlib.sha1(
            "|".join(str(p) for p in args.runs).encode("utf-8")
        ).hexdigest()[:6]
        overall_stacker = verification.get("overall_stacker", float("nan"))
        pooled_stacker  = verification.get("pooled_stacker_ap", float("nan"))
        notes = (f"logreg-v2 | M={len(method_names)} | "
                  f"F={len(feature_names)} | rank_norm={args.rank_norm} | "
                  f"C={params_global.get('C')} | "
                  f"l1_ratio={params_global.get('l1_ratio')} | "
                  f"drop_leaky_top={int(args.drop_leaky_top_methods)} | "
                  f"drop_leaky_mahal={int(args.drop_leaky_mahalanobis)} | "
                  f"tune={args.tune_mode} | "
                  f"pooled_AP={pooled_stacker:.4f}")
        row = {
            "run_id": run_id, "run_tag": args.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "STACKER_LR_V2",
            "feature_layers": "", "input_size": "",
            "n_classes": len(classes),
            "AP_overall": (f"{overall_stacker:.4f}"
                             if not math.isnan(overall_stacker) else ""),
            "AP_pooled": (f"{pooled_stacker:.4f}"
                            if not math.isnan(pooled_stacker) else ""),
            "runtime_min": "",
            "submission_path": str(args.out.with_suffix(".zip")),
            "notes": notes,
        }
        append_to_master(args.master_csv, row)
        print(f"\nAppended row to {args.master_csv}")
        mem_print("end")
        hr(f"DONE — run_id={run_id}", "=")


if __name__ == "__main__":
    main()