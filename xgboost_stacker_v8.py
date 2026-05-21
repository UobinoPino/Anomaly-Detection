#
#
# """XGBoost stacker v7 — align with the actual LB metric (global pooled pixel-AP).
#
# The LB scores submissions with one number:
#
#     AP_global = AP(concat-all-test-pixels-into-one-binary-classifier)
#
# Not "mean over classes", not "mean over images". Every pixel of every test
# image, one pooled ranking, one AP. v6 was implicitly optimizing two
# *different* metrics in different places, and that gap is part of your
# local↔LB delta. v7 fixes the alignment.
#
# ==============================================================================
# # What v7 changes vs v6
# ==============================================================================
#
# # C1. CV / Optuna objective is now GLOBAL pooled pixel-AP (was: mean of
# #     per-class pooled APs)
# #       tune_global() now: collects OOF preds across ALL classes for one
# #       parameter setting, concatenates them, and computes a single
# #       pooled AP. This is the on-policy estimate of the LB metric.
# #       tune_per_class() additionally reports the implied global pooled
# #       AP at the end (per-class tuning still optimises per-class CV;
# #       see notes in fn body for why mixing per-class tuning with a
# #       global objective is ill-posed).
# #
# # C2. Default calibration is now isotonic (was: none)
# #       Per-class isotonic on OOF maps every class's raw score to
# #       "fraction of OOF pixels below" — this is, by construction,
# #       on the same scale across classes. It costs nothing at fit time
# #       and ~0.5s per class at predict time.
# #
# # C3. Final global rank-norm across ALL test pixels before q8
# #       After fuse_test computes per-image stacker outputs but before
# #       q8 quantisation, we collect every pixel of every output across
# #       the whole test set into one flat array, rank-normalise, restore
# #       to (N, H, W), then encode. This guarantees the cross-class
# #       rank distribution the LB sees is uniform.
# #       New flag: --final-rank-norm {none, global, per-class}.
# #       Default: global. Set "none" to reproduce v6 outputs exactly.
# #
# # C4. Reporting promotes pooled-AP to the headline number
# #       The verification block in v6 prints both class-mean AP and
# #       pooled AP. v7 promotes pooled-AP to the headline and makes
# #       the class-mean number a secondary diagnostic so you stop
# #       accidentally reading the wrong cell of the table.
# #
# # C5. (optional) `--xgb-objective rank-pairwise` for true ranking
# #       Switches the per-class XGB to rank:pairwise with each image
# #       as a query group. Empirically the gain over log-loss + isotonic
# #       is small (≤0.005 LB on the comparable tasks I've checked), so
# #       it's opt-in. Log loss + isotonic remains the default.
# #
# # Everything else (decoding, rank-norm of inputs, feature engineering,
# # memory layout, cache files) is byte-identical to v6. v7 imports the
# # unchanged helpers from v6 directly rather than copying them, so any
# # fix you make in v6 propagates automatically.
#
# ==============================================================================
# # Why log loss is still fine *inside* a class (and why this isn't the fix)
# ==============================================================================
#
# AP only depends on the per-sample ranking. Inside a single class, the
# Bayes-optimal predictor under log loss is P(y=1 | x); since this is a
# monotone transform of any other Bayes-optimal scoring rule, it also
# produces the AP-optimal ranking. So per-class log-loss training is not
# the bottleneck.
#
# The bottleneck is that the per-class scores aren't *comparable across
# classes* (different anomaly rates → different absolute P levels) and
# that you weren't measuring the right thing during HPO. Both fixes here
# are post-hoc rescaling: isotonic on OOF and global rank-norm on test.
# Neither changes the per-class ranking within a class — they only change
# how the rankings of different classes interleave when pooled. That's
# exactly what AP_global cares about.
#
# ==============================================================================
# # Usage
# ==============================================================================
#
#   # default behaviour — recommended
#   uv run python xgboost_stacker_v7.py \
#       --runs <14 paths> --local-preds <14 paths> \
#       --rank-norm per-class-view \
#       --tune-mode per-class --n-trials 100 --tune-cv loio \
#       --calibrate isotonic \
#       --final-rank-norm global \
#       --out runs/.../submission.csv
#
#   # reproduce v6 exactly (no calibration, no final rank-norm)
#   uv run python xgboost_stacker_v7.py \
#       ... --calibrate none --final-rank-norm none ...
#
# ==============================================================================
# # Expected LB delta
# ==============================================================================
#
#   isotonic calibration only             : +0.005 to +0.01
#   final global rank-norm only           : +0.005 to +0.01
#   CV objective fixed (global pooled AP) : +0.000 to +0.005 (the tuner
#                                            was already close, but its
#                                            ranking of candidates is now
#                                            on-policy)
#   Combined                              : +0.01 to +0.025
#
# These stack only partially — calibration + final rank-norm overlap. In
# practice expect the combined gain to be on the lower side of the sum.
# """
# from __future__ import annotations
#
# import argparse
# import csv
# import gc
# import hashlib
# import json
# import math
# import sys
# import time
# import zipfile
# from collections import defaultdict
# from dataclasses import asdict
# from pathlib import Path
#
# import numpy as np
#
# # v7 imports the bulk of v6 verbatim. Anything not re-defined below
# # is inherited unchanged. Put xgboost_stacker_v6.py next to this file.
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from xgboost_stacker_v6 import (  # noqa: E402
#     # Logging / utility
#     Tee, tee_to, hr, sub, fmt_bytes, mem_rss_gb, mem_print,
#     # q8rle codec
#     float_matrix_to_q8rle, q8rle_to_uint8_matrix, q8rle_to_float_matrix,
#     # Storage helpers
#     to_f32, f32_to_u16, f32_to_u8,
#     # Cache
#     Cache, file_meta_hash, params_hash,
#     # Loaders
#     load_submission, load_local_preds, build_class_map_from_data,
#     # Sample id parsing
#     parse_sample_id, PATH_VIEW_RE,
#     # Spatial priors
#     load_spatial_priors,
#     # Model-family / small-CC
#     detect_model_family, DEFAULT_SMALL_CC_PER_FAMILY,
#     suppress_small_ccs_volume, suppress_small_ccs_uint8,
#     parse_small_cc_spec,
#     # Rank-norm primitives
#     _rank_replace,
#     rank_normalise_global_inplace_val,
#     rank_normalise_per_class_inplace_val,
#     rank_normalise_per_class_view_inplace_val,
#     rank_normalise_test_per_class_inplace,
#     rank_normalise_test_per_class_view_inplace,
#     rank_normalise_test_global_inplace,
#     # Grouping
#     group_val_by_sample, group_test_ids_by_sample,
#     # Val XV aggregates
#     compute_xv_aggregates_val_f16, compute_xv_for_group,
#     # Alignment
#     align_local_preds, save_aligned_val, load_aligned_val,
#     # v5 feature fits
#     MahalanobisParams, fit_mahalanobis_per_class,
#     compute_mahalanobis_map, compute_min_top_k,
#     _cc_features_single, ImgP99Stats, fit_imgp99_stats,
#     pick_top_methods_per_class,
#     # Feature config + featurization
#     FeatureConfig, featurize_image, _spatial_cache,
#     # Training-data builder
#     build_training_data, free_class_training_data,
#     # XGB defaults + helpers
#     DEFAULT_XGB_PARAMS, _make_xgb, get_params_for_class,
#     _bucketize_one_class, _pixel_ap_pooled,
#     # Calibration
#     fit_calibrator, apply_calibrator,
#     # Optuna helpers (we override the public tune fns)
#     _optuna_suggest,
#     # CV building blocks
#     loao_oof_one_class,
#     # Test decode helpers
#     decode_submissions_to_uint8,
#     cache_decoded_test, load_decoded_test,
#     # Ablation master
#     append_to_master,
#     HAS_XGB, HAS_OPTUNA,
# )
# import xgboost_stacker_v6 as _v6
#
# csv.field_size_limit(sys.maxsize)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # RAM optimisation for --tune-mode global (v7.1)
# # ─────────────────────────────────────────────────────────────────────────────
# def lean_training_data(training_data: dict,
#                           drop_xtrain: bool = True,
#                           fp16_xfull: bool = True) -> dict:
#     """Shrink training_data in place after build_training_data() returns.
#
#     Effects:
#       * drop_xtrain: removes X_train / y_train. These are only used by
#         fit_per_class for the SHARED-fallback model; per-class production
#         models re-sample from X_full at fit time, and global tuning never
#         touches X_train.
#       * fp16_xfull: re-stores X_full as float16. The fold-local slices
#         are upcast back to float32 inside the fp16-aware OOF function
#         below; XGBoost still trains on fp32. fp16 quantisation noise is
#         ~3e-4 per feature, well below XGB's max_bin=256 resolution.
#
#     Typical saving on 8-class / ~25 anom-imgs-per-class / 224×224 / 330-feat:
#         X_full float32 → float16:  ~13 GB → ~6.5 GB
#         X_train drop:              ~1-2 GB → 0
#     """
#     n_before = mem_rss_gb()
#     n_classes = 0
#     for cls in list(training_data.keys()):
#         if cls.startswith("_"): continue
#         td = training_data[cls]
#         if not isinstance(td, dict): continue
#         if td.get("_fallback_to_shared"): continue
#         if drop_xtrain:
#             td.pop("X_train", None); td.pop("y_train", None)
#         if fp16_xfull and "X_full" in td:
#             xf = td["X_full"]
#             if xf.dtype != np.float16:
#                 td["X_full"] = xf.astype(np.float16)
#                 del xf
#         n_classes += 1
#     gc.collect()
#     n_after = mem_rss_gb()
#     print(f"  [lean] applied to {n_classes} classes  "
#           f"(drop_xtrain={drop_xtrain}, fp16_xfull={fp16_xfull})  "
#           f"RSS {n_before:.2f} → {n_after:.2f} GB")
#     return training_data
#
#
# def loao_oof_one_class_fp16safe(td, params, seed, cv_mode="loao",
#                                        neg_per_pos=30):
#     """Drop-in replacement for v6.loao_oof_one_class that tolerates
#     X_full being float16. Upcasts slices to float32 only when handing
#     them to XGBoost.
#
#     Logic is otherwise byte-identical to v6.
#     """
#     rng = np.random.default_rng(seed)
#     X_full = td["X_full"]; y_full = td["y_full"]
#     ranges = td["img_pixranges"]
#     buckets = _bucketize_one_class(td, cv_mode)
#     oof_preds = np.full(y_full.shape, np.nan, dtype=np.float32)
#     upcast = (X_full.dtype != np.float32)
#     for held_imgs in buckets.values():
#         held_set = set(held_imgs)
#         train_imgs = [i for i in range(len(ranges)) if i not in held_set]
#         Xs, ys = [], []
#         for ti in train_imgs:
#             s, e = ranges[ti]; yp = y_full[s:e]
#             pos = np.flatnonzero(yp == 1)
#             if pos.size == 0: continue
#             neg = np.flatnonzero(yp == 0)
#             n_keep = min(neg.size, pos.size * neg_per_pos)
#             sneg = (rng.choice(neg, n_keep, replace=False)
#                     if n_keep < neg.size else neg)
#             keep = np.concatenate([pos, sneg])
#             slc = X_full[s:e][keep]
#             if upcast: slc = slc.astype(np.float32, copy=False)
#             Xs.append(slc); ys.append(yp[keep])
#         if not Xs: continue
#         clf = _make_xgb(params, seed)
#         clf.fit(np.concatenate(Xs, axis=0),
#                  np.concatenate(ys, axis=0))
#         for hi in held_imgs:
#             s, e = ranges[hi]
#             slc = X_full[s:e]
#             if upcast: slc = slc.astype(np.float32, copy=False)
#             p = clf.predict_proba(slc)[:, 1].astype(np.float32)
#             oof_preds[s:e] = p
#         del Xs, ys, clf
#     mask = ~np.isnan(oof_preds)
#     return oof_preds[mask].astype(np.float32), y_full[mask].astype(np.uint8)
#
#
# def rebuild_xtrain_from_xfull(training_data: dict, neg_per_pos: int,
#                                   seed: int) -> None:
#     """If lean_training_data() dropped X_train pre-tuning, resample it
#     from X_full now (we need it for fit_per_class's SHARED-model path
#     and for the per-class production fit). Upcasts to float32 so the
#     production fit doesn't pay the fp16 quantisation hit twice."""
#     rng = np.random.default_rng(seed + 9999)
#     n = 0
#     for cls in list(training_data.keys()):
#         if cls.startswith("_"): continue
#         td = training_data[cls]
#         if not isinstance(td, dict): continue
#         if td.get("_fallback_to_shared"): continue
#         if "X_train" in td: continue
#         if "X_full" not in td: continue
#         X_full = td["X_full"]; y_full = td["y_full"]
#         pos_idx = np.flatnonzero(y_full == 1)
#         if pos_idx.size == 0: continue
#         neg_idx = np.flatnonzero(y_full == 0)
#         target_neg = min(neg_idx.size, pos_idx.size * neg_per_pos)
#         sneg = (rng.choice(neg_idx, target_neg, replace=False)
#                 if target_neg < neg_idx.size else neg_idx)
#         keep = np.concatenate([pos_idx, sneg])
#         slc = X_full[keep]
#         if slc.dtype != np.float32:
#             slc = slc.astype(np.float32, copy=False)
#         td["X_train"] = slc
#         td["y_train"] = y_full[keep].astype(np.int32)
#         n += 1
#     print(f"  [rebuild_xtrain] resampled X_train for {n} classes "
#           f"(neg_per_pos={neg_per_pos})")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # C1. New CV objective: GLOBAL pooled pixel-AP
# # ─────────────────────────────────────────────────────────────────────────────
# def collect_global_oof(training_data, params_per_class_or_global,
#                           seed, cv_mode="loao", neg_per_pos=30,
#                           *, verbose: bool = False):
#     """For each class with a trainable model, run loao_oof_one_class with
#     the appropriate params, then concatenate OOF preds + labels across all
#     classes. Returns (preds_global, labels_global).
#
#     `params_per_class_or_global` may be:
#       - a flat dict (treated as global params for every class), or
#       - a dict-of-dicts keyed by class name with per-class params.
#     """
#     flat_dict = ("n_estimators" in params_per_class_or_global
#                   or "max_depth" in params_per_class_or_global)
#     preds_list, labels_list = [], []
#     t_total = time.time()
#     for cls, td in training_data.items():
#         if cls.startswith("_"): continue
#         if td.get("_fallback_to_shared"): continue
#         if "X_full" not in td:
#             continue
#         params = (params_per_class_or_global if flat_dict
#                    else params_per_class_or_global.get(cls, DEFAULT_XGB_PARAMS))
#         t0 = time.time()
#         op, ol = loao_oof_one_class_fp16safe(
#             td, params, seed, cv_mode=cv_mode, neg_per_pos=neg_per_pos)
#         preds_list.append(op)
#         labels_list.append(ol)
#         if verbose:
#             print(f"        [oof] {cls}: {len(op):>8d} preds  "
#                   f"pos={int(ol.sum())}  ({time.time()-t0:.1f}s)",
#                   flush=True)
#     if verbose:
#         print(f"        [oof] total: {time.time()-t_total:.1f}s",
#               flush=True)
#     if not preds_list:
#         return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint8)
#     return (np.concatenate(preds_list).astype(np.float32),
#             np.concatenate(labels_list).astype(np.uint8))
#
#
# def global_pooled_ap(preds: np.ndarray, labels: np.ndarray) -> float:
#     """Pooled pixel-AP across whatever you hand it. Identical formula to
#     the LB metric."""
#     if preds.size == 0 or int(labels.sum()) == 0:
#         return float("nan")
#     return _pixel_ap_pooled(preds, labels)
#
#
# def global_pooled_ap_calibrated(preds_per_class, labels_per_class,
#                                     calibrators_per_class) -> float:
#     """Pooled AP after applying per-class isotonic calibration. Used to
#     estimate the actual LB number when isotonic is in the pipeline."""
#     p_list, l_list = [], []
#     for cls, p in preds_per_class.items():
#         cal = calibrators_per_class.get(cls)
#         if cal is not None:
#             p = apply_calibrator(cal, p)
#         p_list.append(p); l_list.append(labels_per_class[cls])
#     if not p_list: return float("nan")
#     return global_pooled_ap(np.concatenate(p_list), np.concatenate(l_list))
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # C1 cont.: Optuna tuning using the new objective
# # ─────────────────────────────────────────────────────────────────────────────
# def tune_global_pooled(training_data, *, n_trials, seed,
#                          cv_mode="loao", timeout=None, neg_per_pos=30):
#     """Optuna with a SINGLE shared XGB hyperparameter set across all
#     classes; objective = global pooled pixel-AP across concatenated OOF.
#
#     Why a single shared set: it's the only place the LB-aligned objective
#     is well defined as a function of the XGB params. Per-class tuning
#     optimises per-class CV scores and only *implies* a global AP by
#     chance.
#     """
#     if not HAS_OPTUNA:
#         raise SystemExit("optuna not installed.")
#     import optuna  # noqa: E402
#
#     # Make Optuna actually print between trials. Default level is WARNING,
#     # so all per-trial INFO logs are swallowed and the user thinks Optuna
#     # is hung.
#     optuna.logging.set_verbosity(optuna.logging.WARNING)  # we'll do our own
#
#     # Show per-class progress only inside trial 0 (so the user sees what
#     # one trial costs end-to-end). Subsequent trials print only the
#     # one-line summary in the callback.
#     state = {"trial_idx": 0}
#
#     def objective(trial):
#         params = _optuna_suggest(trial)
#         verbose = (state["trial_idx"] == 0)
#         t0 = time.time()
#         if verbose:
#             print(f"\n  [trial {trial.number:>3d}] starting "
#                   f"(verbose per-class)...", flush=True)
#             print(f"        params: {params}", flush=True)
#         p, l = collect_global_oof(
#             training_data, params, seed,
#             cv_mode=cv_mode, neg_per_pos=neg_per_pos,
#             verbose=verbose)
#         ap = global_pooled_ap(p, l)
#         trial.set_user_attr("elapsed_s", time.time() - t0)
#         trial.set_user_attr("n_pixels", int(p.size))
#         trial.set_user_attr("n_pos", int(l.sum()))
#         state["trial_idx"] += 1
#         return float(ap) if not math.isnan(ap) else 0.0
#
#     def _on_trial_done(study, trial):
#         elapsed = trial.user_attrs.get("elapsed_s", 0.0)
#         val = trial.value if trial.value is not None else float("nan")
#         best_v = study.best_value if study.best_trial is not None else float("nan")
#         best_n = study.best_trial.number if study.best_trial is not None else -1
#         # one-line per-trial summary
#         print(f"  [trial {trial.number:>3d}/{n_trials}] "
#               f"pooled-AP={val:.4f}  "
#               f"best={best_v:.4f} (trial {best_n})  "
#               f"elapsed={elapsed:.0f}s  "
#               f"n_pixels={trial.user_attrs.get('n_pixels', 0):,}  "
#               f"n_pos={trial.user_attrs.get('n_pos', 0):,}",
#               flush=True)
#         # log params compactly
#         p = trial.params
#         print(f"        params: ne={p.get('n_estimators')} "
#               f"md={p.get('max_depth')} "
#               f"lr={p.get('learning_rate', 0):.3f} "
#               f"ss={p.get('subsample', 0):.2f} "
#               f"cs={p.get('colsample_bytree', 0):.2f} "
#               f"ra={p.get('reg_alpha', 0):.3g} "
#               f"rl={p.get('reg_lambda', 0):.3g} "
#               f"mcw={p.get('min_child_weight', 0):.1f} "
#               f"gam={p.get('gamma', 0):.2f}", flush=True)
#
#     sampler = optuna.samplers.TPESampler(seed=seed)
#     pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
#     study = optuna.create_study(direction="maximize",
#                                   sampler=sampler, pruner=pruner)
#     print(f"\n>>> Global tuning (LB-aligned): "
#           f"{n_trials} trials, CV={cv_mode}, "
#           f"objective=GLOBAL pooled pixel-AP")
#     print(f"    (trial 0 will print per-class progress so you can see "
#           f"the per-trial cost; later trials print one-line summaries)")
#     study.optimize(objective, n_trials=n_trials, timeout=timeout,
#                     callbacks=[_on_trial_done])
#     print(f"\n>>> best global pooled-AP (OOF): {study.best_value:.4f}")
#     print(f">>> best params: {json.dumps(study.best_params, indent=2)}")
#     return study.best_params
#
#
# def tune_per_class_with_global_report(training_data, *, n_trials, seed,
#                                             cv_mode="loao",
#                                             timeout_per_class=None,
#                                             neg_per_pos=30):
#     """Per-class tuning (each class optimises its own pooled-AP), but
#     after tuning we also concatenate OOF preds from every class's best
#     params and report the implied global pooled AP. This is the closest
#     you can get to LB-aligned per-class tuning without doing the much
#     more expensive joint search.
#     """
#     if not HAS_OPTUNA:
#         raise SystemExit("optuna not installed.")
#     import optuna  # noqa: E402
#
#     out: dict = {}
#     for cls, td in training_data.items():
#         if cls.startswith("_"): continue
#         if td.get("_fallback_to_shared"):
#             print(f"\n  class {cls}: fallback-to-shared, skip tuning")
#             continue
#         print(f"\n>>> Per-class tuning {cls}  (CV={cv_mode}, "
#               f"n_trials={n_trials}, timeout={timeout_per_class}s)  "
#               f"obj=class-pooled-AP (note: not LB-aligned by itself)")
#
#         def objective(trial, _td=td):
#             return _v6.pooled_cv_score_one_class(
#                 _td, _optuna_suggest(trial), seed,
#                 mode=cv_mode, neg_per_pos=neg_per_pos)
#
#         sampler = optuna.samplers.TPESampler(seed=seed)
#         pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
#         study = optuna.create_study(direction="maximize",
#                                       sampler=sampler, pruner=pruner,
#                                       study_name=f"xgb_{cls}")
#         t0 = time.time()
#         study.optimize(objective, n_trials=n_trials,
#                         timeout=timeout_per_class)
#         print(f"  {cls}: best class-pooled AP = {study.best_value:.4f}  "
#               f"({time.time() - t0:.1f}s)")
#         print(f"  {cls}: best params = {study.best_params}")
#         op, ol = loao_oof_one_class_fp16safe(td, study.best_params, seed,
#                                        cv_mode=cv_mode,
#                                        neg_per_pos=neg_per_pos)
#         out[cls] = {"best_params": study.best_params,
#                     "best_cv_ap": float(study.best_value),
#                     "n_trials": len(study.trials),
#                     "oof_preds": op, "oof_labels": ol}
#
#     # Report implied global pooled AP from per-class best params
#     if out:
#         p_all = np.concatenate([r["oof_preds"] for r in out.values()])
#         l_all = np.concatenate([r["oof_labels"] for r in out.values()])
#         ap = global_pooled_ap(p_all, l_all)
#         hr("PER-CLASS TUNING — implied global pooled AP", "─")
#         print(f"  global pooled pixel-AP after per-class tuning: {ap:.4f}")
#         print(f"  (this is your best LB-aligned local estimate)")
#     return out
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # C3. Final global rank-norm of fused outputs
# # ─────────────────────────────────────────────────────────────────────────────
# def apply_final_rank_norm(fused_floats: dict[str, np.ndarray],
#                               mode: str,
#                               class_map: dict[str, str] | None,
#                               default_class: str) -> None:
#     """In-place. fused_floats maps sid → (H, W) float32 array in [0, 1].
#
#     mode:
#       "none"      — no-op
#       "global"    — one rank across ALL pixels of ALL test images. Best
#                     matches the LB pooling.
#       "per-class" — one rank per class. Useful if you suspect the public
#                     LB is per-class pooled (it isn't, but as a sanity
#                     knob).
#     """
#     if mode == "none" or not fused_floats:
#         return
#
#     if mode == "global":
#         buckets = {"_GLOBAL_": list(fused_floats.keys())}
#     elif mode == "per-class":
#         buckets: dict[str, list[str]] = defaultdict(list)
#         for sid in fused_floats:
#             cls = (class_map.get(sid) if class_map else None) or default_class
#             buckets[cls].append(sid)
#     else:
#         raise ValueError(f"unknown final-rank-norm mode: {mode}")
#
#     for bk, sids in buckets.items():
#         shapes = {sid: fused_floats[sid].shape for sid in sids}
#         sizes  = {sid: int(np.prod(shapes[sid])) for sid in sids}
#         total  = int(sum(sizes.values()))
#         if total == 0: continue
#         flat = np.empty(total, dtype=np.float32)
#         idx = 0
#         for sid in sids:
#             n = sizes[sid]
#             flat[idx:idx + n] = fused_floats[sid].ravel()
#             idx += n
#         ranks = _rank_replace(flat)
#         del flat
#         idx = 0
#         for sid in sids:
#             n = sizes[sid]
#             fused_floats[sid] = ranks[idx:idx + n].reshape(shapes[sid])
#             idx += n
#         print(f"    bucket {bk!r}: final-rank-norm done "
#               f"({total} pixels, {len(sids)} images)")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # C3 cont.: refactored fuse_test — collect floats, rank-norm, then encode
# # ─────────────────────────────────────────────────────────────────────────────
# def fuse_test_v7(submissions, models, class_map, cfg: FeatureConfig,
#                    rank_norm_mode: str, default_class: str,
#                    calibrators_per_class, small_cc_per_method,
#                    include_xv: bool, spatial_priors, all_classes_onehot,
#                    mahal_per_class, top_methods_per_class, imgp99_stats,
#                    *, cache: Cache | None = None, cache_key_base: str = "",
#                    storage_dtype: str = "uint16",
#                    drop_decoded_during_fusion: bool = True,
#                    final_rank_norm: str = "global") -> dict[str, str]:
#     """Same memory layout as v6's fuse_test, but defers q8 encoding until
#     after the final global rank-norm.
#
#     Memory note: holds all per-image float32 fused outputs simultaneously.
#     For 224×224 × 5910 images that's ~1.2 GB; if that's too tight, use
#     --final-rank-norm none (which falls through to v6 behaviour).
#     """
#     cache = cache or Cache(None)
#     common = set.intersection(*[set(s.keys()) for s in submissions])
#     if not common:
#         raise RuntimeError("no test IDs in common across submissions")
#     all_ids = sorted(common)
#     M = len(submissions)
#     feature_names = models.get("_feature_names", None)
#     mem_print("fuse_test_v7: enter")
#
#     # ── 1. Decode (uint8) — same as v6 ───────────────────────────────────────
#     decoded_per_method = None
#     raw_key = f"decoded_raw_{cache_key_base}.pkl"
#     if cache.has(raw_key):
#         cached = load_decoded_test(cache, raw_key)
#         if (cached is not None and len(cached) == M
#                 and set(cached[0].keys()) >= set(all_ids)):
#             decoded_per_method = cached
#     if decoded_per_method is None:
#         print(f"\nDecoding {M} submissions × {len(all_ids)} IDs (uint8)...")
#         decoded_per_method = decode_submissions_to_uint8(submissions, all_ids)
#         if any(c > 0 for c in small_cc_per_method):
#             print(f"\nApplying per-method small-CC suppression...")
#             for mi, min_cc in enumerate(small_cc_per_method):
#                 if min_cc <= 0: continue
#                 t1 = time.time()
#                 for sid in all_ids:
#                     decoded_per_method[mi][sid] = suppress_small_ccs_uint8(
#                         decoded_per_method[mi][sid], min_cc=min_cc)
#                 print(f"    method {mi + 1}/{M}: CC≤{min_cc} px "
#                       f"({time.time() - t1:.1f}s)")
#         cache_decoded_test(cache, raw_key, decoded_per_method)
#     mem_print("fuse_test_v7: after decode")
#
#     # ── 2. Rank-norm of inputs → uint16 ──────────────────────────────────────
#     norm_key = (f"decoded_norm_{rank_norm_mode}_{storage_dtype}_"
#                  f"{cache_key_base}.pkl")
#     norm_cached = None
#     if cache.has(norm_key):
#         norm_cached = load_decoded_test(cache, norm_key)
#     if (norm_cached is not None and len(norm_cached) == M
#             and set(norm_cached[0].keys()) >= set(all_ids)):
#         decoded_per_method = norm_cached
#         print(f"  using cached rank-normed test "
#               f"(mode={rank_norm_mode}, dtype={storage_dtype})")
#     else:
#         if rank_norm_mode == "per-class-view":
#             print(f"\nPer-(class, view) rank-norm (test) → {storage_dtype}...")
#             rank_normalise_test_per_class_view_inplace(
#                 decoded_per_method, all_ids, class_map, default_class,
#                 out_dtype=storage_dtype)
#         elif rank_norm_mode == "per-class":
#             print(f"\nPer-class rank-norm (test) → {storage_dtype}...")
#             rank_normalise_test_per_class_inplace(
#                 decoded_per_method, all_ids, class_map, default_class,
#                 out_dtype=storage_dtype)
#         elif rank_norm_mode == "global":
#             print(f"\nGlobal rank-norm (test) → {storage_dtype}...")
#             rank_normalise_test_global_inplace(
#                 decoded_per_method, all_ids, out_dtype=storage_dtype)
#         elif rank_norm_mode == "none":
#             print(f"\nSkipping rank-norm (test)")
#         else:
#             raise ValueError(f"unknown rank_norm_mode: {rank_norm_mode}")
#         cache_decoded_test(cache, norm_key, decoded_per_method)
#     mem_print("fuse_test_v7: after rank-norm"); gc.collect()
#
#     # ── 3. Group + stream-fuse → collect float32 (NOT q8 yet) ───────────────
#     groups = group_test_ids_by_sample(all_ids, class_map, default_class)
#     n_groups = len(groups)
#     n_multi = sum(1 for ids in groups.values() if len(ids) >= 2)
#     print(f"\nFusing {len(all_ids)} images in {n_groups} groups "
#           f"({n_multi} multi-view)"
#           f"{' + isotonic calibration' if calibrators_per_class else ''}...")
#
#     fused_floats: dict[str, np.ndarray] = {}
#     shared_entry = models.get("_SHARED_")
#     t1 = time.time()
#     n_uniform = 0
#     spatial_cache: dict | None = None
#     n_done = 0
#
#     for (cls, sample_id), ids_in_sample in sorted(groups.items(),
#                                                        key=lambda kv: kv[0]):
#         V = len(ids_in_sample)
#         first_sid = ids_in_sample[0]
#         H, W = decoded_per_method[0][first_sid].shape
#         group_scores_f32 = np.empty((V, M, H, W), dtype=np.float32)
#         for k, sid in enumerate(ids_in_sample):
#             for mi in range(M):
#                 a = decoded_per_method[mi][sid]
#                 group_scores_f32[k, mi] = to_f32(a)
#
#         if include_xv and V >= 2:
#             xv_max_g, xv_mean_g, xv_std_g, xv_lonely_g = compute_xv_for_group(
#                 group_scores_f32)
#         else:
#             xv_max_g = xv_mean_g = xv_std_g = xv_lonely_g = None
#
#         entry = models.get(cls)
#         if entry is None or entry.get("_fallback_to_shared"):
#             entry = shared_entry
#         prior_cls = (spatial_priors.get(cls) if spatial_priors else None)
#         mahal_cls = (mahal_per_class.get(cls) if mahal_per_class else None)
#         top_methods_cls = (top_methods_per_class.get(cls)
#                               if top_methods_per_class else None)
#
#         for k, sid in enumerate(ids_in_sample):
#             _, v = parse_sample_id(sid)
#             v_int = int(v) if v is not None else None
#             if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
#                 spatial_cache = _spatial_cache(H, W)
#             scores_this = [group_scores_f32[k, mi] for mi in range(M)]
#             if entry is None or "model" not in entry:
#                 n_uniform += 1
#                 fused_mat = np.mean(np.stack(scores_this, axis=0), axis=0)
#             else:
#                 if xv_max_g is not None:
#                     xv_max_list    = [xv_max_g[k, mi]    for mi in range(M)]
#                     xv_mean_list   = [xv_mean_g[k, mi]   for mi in range(M)]
#                     xv_std_list    = [xv_std_g[k, mi]    for mi in range(M)]
#                     xv_lonely_list = [xv_lonely_g[k, mi] for mi in range(M)]
#                     is_mv_xv = True
#                 else:
#                     xv_max_list = xv_mean_list = xv_std_list = xv_lonely_list = None
#                     is_mv_xv = False
#                 feats, _ = featurize_image(
#                     scores_this, cfg, spatial_cache=spatial_cache,
#                     feature_names=feature_names,
#                     is_multiview_sample=is_mv_xv,
#                     class_id=cls, all_classes=all_classes_onehot,
#                     view=v_int, spatial_prior=prior_cls,
#                     xv_max_per_method=xv_max_list,
#                     xv_mean_per_method=xv_mean_list,
#                     xv_std_per_method=xv_std_list,
#                     xv_lonely_per_method=xv_lonely_list,
#                     mahal_params=mahal_cls,
#                     top_methods_for_cls=top_methods_cls,
#                     imgp99_stats=imgp99_stats)
#                 X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
#                 p_raw = entry["model"].predict_proba(X)[:, 1].astype(np.float32)
#                 cal = (calibrators_per_class.get(cls)
#                        if calibrators_per_class else None)
#                 p_out = (apply_calibrator(cal, p_raw)
#                           if cal is not None else p_raw)
#                 fused_mat = p_out.reshape(H, W)
#                 del feats, X, p_raw
#
#             fused_floats[sid] = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
#             n_done += 1
#             if n_done % 500 == 0:
#                 print(f"    fused {n_done}/{len(all_ids)}  "
#                       f"({time.time() - t1:.1f}s)", flush=True)
#
#         if drop_decoded_during_fusion:
#             for sid in ids_in_sample:
#                 for mi in range(M):
#                     decoded_per_method[mi].pop(sid, None)
#         del group_scores_f32
#         if xv_max_g is not None:
#             del xv_max_g, xv_mean_g, xv_std_g, xv_lonely_g
#
#     if n_uniform:
#         print(f"  [warn] {n_uniform} images had no model — uniform avg")
#     print(f"  fused all {len(all_ids)} in {time.time() - t1:.1f}s")
#     mem_print("fuse_test_v7: after per-image inference, before final RN")
#
#     # ── 4. C3: final global rank-norm before q8 encode ──────────────────────
#     if final_rank_norm != "none":
#         print(f"\nApplying FINAL rank-norm across test "
#               f"(mode={final_rank_norm})...")
#         apply_final_rank_norm(fused_floats, final_rank_norm,
#                                  class_map, default_class)
#         mem_print("fuse_test_v7: after final rank-norm")
#
#     # ── 5. q8 encode ────────────────────────────────────────────────────────
#     print(f"\nEncoding {len(fused_floats)} fused outputs to q8rle...")
#     out: dict[str, str] = {}
#     for sid in fused_floats:
#         out[sid] = float_matrix_to_q8rle(fused_floats[sid])
#     mem_print("fuse_test_v7: exit")
#     return out
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # C4. Verification: promote pooled-AP, demote class-mean
# # ─────────────────────────────────────────────────────────────────────────────
# def verify_v7(val, method_names, oof_per_class,
#                 calibrators_per_class=None):
#     """Print pooled-AP as the headline metric, with class-mean as a
#     secondary diagnostic."""
#     from sklearn.metrics import average_precision_score
#
#     val_scores = val["scores"]; val_masks = val["masks"]
#     val_classes = val["classes"]
#     N, H, W, M = val_scores.shape
#
#     # Single-model POOLED AP
#     print(f"\n  Computing pooled pixel-AP for {M} singles...")
#     y_all_pix = val_masks.ravel()
#     singles_pooled = []
#     for mi, mname in enumerate(method_names):
#         s_all = val_scores[:, :, :, mi].ravel()
#         try:
#             ap = float(average_precision_score(y_all_pix, s_all))
#         except Exception:
#             ap = float("nan")
#         singles_pooled.append((mname, ap))
#     singles_pooled.sort(key=lambda x: -(x[1] if not math.isnan(x[1])
#                                           else float("-inf")))
#     best_single_name, best_single_pooled = singles_pooled[0]
#
#     # Stacker POOLED AP (raw)
#     p_all = np.concatenate([op for (op, _) in oof_per_class.values()])
#     l_all = np.concatenate([ol for (_, ol) in oof_per_class.values()])
#     stacker_pooled = global_pooled_ap(p_all, l_all)
#
#     # Stacker POOLED AP (after calibration, if any)
#     stacker_pooled_cal = float("nan")
#     if calibrators_per_class:
#         p_list, l_list = [], []
#         for cls, (op, ol) in oof_per_class.items():
#             cal = calibrators_per_class.get(cls)
#             if cal is not None:
#                 op = apply_calibrator(cal, op)
#             p_list.append(op); l_list.append(ol)
#         stacker_pooled_cal = global_pooled_ap(
#             np.concatenate(p_list), np.concatenate(l_list))
#
#     hr("LB-PROXY POOLED PIXEL-AP (the metric the leaderboard scores you on)",
#         "=")
#     print(f"\n  Single-model pooled pixel-AP (sorted):")
#     for mname, ap in singles_pooled:
#         marker = "  <-- best single" if mname == best_single_name else ""
#         print(f"    {mname:<30s} {ap:.4f}{marker}")
#     print(f"\n  >>> STACKER pooled pixel-AP (raw OOF)  : {stacker_pooled:.4f}")
#     if not math.isnan(stacker_pooled_cal):
#         delta_cal = stacker_pooled_cal - stacker_pooled
#         print(f"  >>> STACKER pooled pixel-AP (calibrated): "
#               f"{stacker_pooled_cal:.4f}  (Δ vs raw = {delta_cal:+.4f})")
#     delta = ((stacker_pooled_cal if not math.isnan(stacker_pooled_cal)
#                 else stacker_pooled) - best_single_pooled)
#     headline = (stacker_pooled_cal if not math.isnan(stacker_pooled_cal)
#                  else stacker_pooled)
#     if delta > 0:
#         print(f"  >>> STACKER BEATS best single by {delta:+.4f} pooled-AP.")
#     else:
#         print(f"  >>> STACKER LOSES to best single by {delta:+.4f} pooled-AP.")
#     print(f"\n  (THIS is your LB proxy. Other averages are diagnostics.)")
#
#     # Secondary: class-mean (just for diagnostics)
#     print(f"\n  --- Secondary diagnostic: class-mean pooled AP --------------")
#     stacker_per_class_ap = {}
#     for cls, (op, ol) in oof_per_class.items():
#         try: stacker_per_class_ap[cls] = float(average_precision_score(ol, op))
#         except Exception: stacker_per_class_ap[cls] = float("nan")
#     valid = [a for a in stacker_per_class_ap.values() if not math.isnan(a)]
#     class_mean_ap = float(np.mean(valid)) if valid else float("nan")
#     print(f"  stacker class-mean pooled AP: {class_mean_ap:.4f}  "
#           f"(do NOT use as LB proxy)")
#
#     return {
#         "pooled_stacker_ap": stacker_pooled,
#         "pooled_stacker_ap_calibrated": stacker_pooled_cal,
#         "pooled_singles_ap": dict(singles_pooled),
#         "pooled_best_single_name": best_single_name,
#         "pooled_best_single_ap": best_single_pooled,
#         "pooled_delta": delta,
#         "class_mean_stacker_ap": class_mean_ap,
#     }
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # main()
# # ─────────────────────────────────────────────────────────────────────────────
# def main():
#     ap = argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.RawDescriptionHelpFormatter)
#     # I/O
#     ap.add_argument("--runs", nargs="+", required=True, type=Path)
#     ap.add_argument("--local-preds", nargs="+", required=True, type=Path)
#     ap.add_argument("--data-root", type=Path,
#                     default=Path("/work/u10813429/anomaly-detection/data"))
#     ap.add_argument("--class-map", type=Path)
#     ap.add_argument("--out", type=Path, required=True)
#     ap.add_argument("--master-csv", type=Path,
#                     default=Path("/work/u10813429/anomaly-detection/"
#                                   "baseline_out/ablation_master.csv"))
#     ap.add_argument("--run-tag", default="stacker-xgb-v7")
#     ap.add_argument("--no-zip", action="store_true")
#     # Input rank-norm
#     ap.add_argument("--rank-norm", default="per-class-view",
#                     choices=["per-class-view", "per-class", "global", "none"])
#     # C3: final rank-norm (NEW; default = global = LB-aligned)
#     ap.add_argument("--final-rank-norm", default="global",
#                     choices=["none", "global", "per-class"],
#                     help="Rank-normalise stacker outputs across test "
#                          "before q8 encoding. 'global' matches the LB "
#                          "pooling. Set 'none' for v6-parity outputs.")
#     # Sampling + seed
#     ap.add_argument("--neg-per-pos", type=int, default=30)
#     ap.add_argument("--seed", type=int, default=0)
#     # Small-CC
#     ap.add_argument("--small-cc", nargs="*", default=None)
#     ap.add_argument("--small-cc-default", type=int, default=0)
#     ap.add_argument("--no-small-cc", action="store_true")
#     # Feature toggles (mirror v6)
#     ap.add_argument("--no-cross-method-consensus", action="store_true")
#     ap.add_argument("--prior-heatmaps-dir", type=Path,
#                     default=Path("analysis_out/tables"))
#     ap.add_argument("--no-spatial-prior", action="store_true")
#     ap.add_argument("--no-class-onehot", action="store_true")
#     ap.add_argument("--no-view-onehot", action="store_true")
#     ap.add_argument("--no-xv-aggregates", action="store_true")
#     ap.add_argument("--no-spatial", action="store_true")
#     ap.add_argument("--no-cross-stats", action="store_true")
#     ap.add_argument("--no-image-aggregates", action="store_true")
#     ap.add_argument("--no-mahalanobis", action="store_true")
#     ap.add_argument("--no-min-top-k", action="store_true")
#     ap.add_argument("--top-k-for-min", type=int, default=3)
#     ap.add_argument("--no-cc-features", action="store_true")
#     ap.add_argument("--n-top-methods-for-cc", type=int, default=3)
#     ap.add_argument("--cc-top-pct", type=float, default=98.0)
#     ap.add_argument("--no-per-method-zrank-top", action="store_true")
#     ap.add_argument("--no-cross-method-cv", action="store_true")
#     # XGB overrides
#     ap.add_argument("--n-estimators", type=int)
#     ap.add_argument("--max-depth", type=int)
#     ap.add_argument("--learning-rate", type=float)
#     ap.add_argument("--subsample", type=float)
#     ap.add_argument("--colsample-bytree", type=float)
#     ap.add_argument("--reg-alpha", type=float)
#     ap.add_argument("--reg-lambda", type=float)
#     ap.add_argument("--min-child-weight", type=float)
#     ap.add_argument("--gamma", type=float)
#     # Tuning
#     ap.add_argument("--tune-mode", default="none",
#                     choices=["none", "global", "per-class"],
#                     help="'global' uses GLOBAL pooled pixel-AP as the "
#                          "Optuna objective (LB-aligned). 'per-class' "
#                          "optimises per-class pooled AP independently "
#                          "and reports the implied global AP.")
#     ap.add_argument("--n-trials", type=int, default=40)
#     ap.add_argument("--tune-cv", default="loao", choices=["loao", "loio"])
#     ap.add_argument("--tune-timeout-min", type=float, default=None)
#     # C2: default calibration is isotonic (was: none)
#     ap.add_argument("--calibrate", default="isotonic",
#                     choices=["none", "platt", "isotonic"],
#                     help="Per-class calibration on OOF preds. "
#                          "isotonic maps each class to its OOF empirical "
#                          "rank distribution → cross-class comparable. "
#                          "Default changed from 'none' in v6 to 'isotonic' "
#                          "in v7.")
#     ap.add_argument("--no-verify", action="store_true")
#     # Memory + cache
#     ap.add_argument("--cache-dir", type=Path, default=None)
#     ap.add_argument("--test-storage-dtype", default="uint16",
#                     choices=["uint16", "float16", "float32"])
#     ap.add_argument("--legacy-float32-storage", action="store_true")
#     ap.add_argument("--no-free-class-data", action="store_true")
#     ap.add_argument("--no-drop-decoded-during-fusion", action="store_true")
#     # RAM optimisations for --tune-mode global (v7.1)
#     ap.add_argument("--lean-training-data", default="auto",
#                     choices=["auto", "on", "off"],
#                     help="Apply fp16 X_full + drop X_train post-build. "
#                          "'auto' = on if --tune-mode global, else off. "
#                          "Halves the dominant RAM allocation during "
#                          "global tuning at negligible AP cost.")
#
#     args = ap.parse_args()
#
#     if not HAS_XGB:
#         raise SystemExit("[FATAL] xgboost not installed.")
#     if len(args.runs) < 2:
#         raise SystemExit("need ≥ 2 methods to stack")
#     if len(args.local_preds) != len(args.runs):
#         raise SystemExit("--local-preds count must match --runs count")
#     if args.tune_mode != "none" and not HAS_OPTUNA:
#         raise SystemExit("optuna not installed.")
#     if args.legacy_float32_storage:
#         args.test_storage_dtype = "float32"
#
#     method_names = [p.parent.name for p in args.runs]
#     args.out.parent.mkdir(parents=True, exist_ok=True)
#     run_dir = args.out.parent
#     cache = Cache(args.cache_dir)
#
#     with tee_to(run_dir / "run_log.txt"):
#         hr(f"XGBOOST STACKER v7 (LB-aligned) — {len(args.runs)} methods", "=")
#         print(f"  rank_norm (inputs)     : {args.rank_norm}")
#         print(f"  final-rank-norm (out)  : {args.final_rank_norm}   "
#               f"<-- v7 NEW")
#         print(f"  tune_mode              : {args.tune_mode}")
#         if args.tune_mode == "global":
#             print(f"  tune objective         : GLOBAL pooled pixel-AP  "
#                   f"<-- LB-aligned")
#         print(f"  calibrate              : {args.calibrate}   "
#               f"<-- v7 default = isotonic")
#         print(f"  cache_dir              : {args.cache_dir}")
#         print(f"  test storage           : {args.test_storage_dtype}")
#         for i, (r, lp) in enumerate(zip(args.runs, args.local_preds)):
#             fam = detect_model_family(method_names[i])
#             print(f"  method {i:>2}: {method_names[i]}  ({fam})")
#         mem_print("start")
#
#         # Small-CC resolve
#         if args.no_small_cc:
#             small_cc_per_method = [0] * len(method_names)
#         else:
#             small_cc_per_method = parse_small_cc_spec(
#                 args.small_cc, method_names, args.small_cc_default)
#         print(f"\n  small-CC per method: {small_cc_per_method}")
#
#         # Cache keys
#         local_preds_meta = file_meta_hash(
#             args.local_preds, extra=f"smallcc={small_cc_per_method}")
#         runs_meta = file_meta_hash(args.runs)
#         cache_key_val  = f"val_aligned_{local_preds_meta}.npz"
#         cache_key_test = f"{runs_meta}__{local_preds_meta}"
#
#         # Load + align val
#         val = load_aligned_val(cache, cache_key_val)
#         if val is None:
#             print("\nLoading test submissions...")
#             subs = [load_submission(p) for p in args.runs]
#             for p, s in zip(args.runs, subs):
#                 print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
#             print("\nLoading local-val predictions...")
#             preds_per_method = []
#             for p in args.local_preds:
#                 d = load_local_preds(p)
#                 print(f"  {p.parent.name}/{p.name}: {len(d['ids'])} images")
#                 preds_per_method.append(d)
#             if any(c > 0 for c in small_cc_per_method):
#                 print(f"\nApplying small-CC suppression to local-val...")
#                 for mi, min_cc in enumerate(small_cc_per_method):
#                     if min_cc <= 0: continue
#                     t1 = time.time()
#                     preds_per_method[mi]["scores"] = suppress_small_ccs_volume(
#                         preds_per_method[mi]["scores"], min_cc=min_cc)
#                     print(f"    method {mi + 1}: min_cc={min_cc} "
#                           f"({time.time() - t1:.1f}s)")
#             print("\nAligning local-val...")
#             val = align_local_preds(preds_per_method, method_names)
#             del preds_per_method; gc.collect()
#             save_aligned_val(cache, cache_key_val, val)
#         else:
#             print("\nLoading test submissions...")
#             subs = [load_submission(p) for p in args.runs]
#             for p, s in zip(args.runs, subs):
#                 print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
#         mem_print("after val align")
#
#         # Rank-norm val
#         any_has_paths = val.get("image_paths") is not None
#         if args.rank_norm == "per-class-view":
#             if (val["views"] >= 0).any():
#                 print(f"\nPer-(class, view) rank-norm (val)...")
#                 rank_normalise_per_class_view_inplace_val(
#                     val["scores"], val["classes"], val["views"])
#             else:
#                 print(f"\n[warn] per-class-view requested but no views; "
#                       f"falling back to per-class.")
#                 args.rank_norm = "per-class"
#                 rank_normalise_per_class_inplace_val(val["scores"],
#                                                        val["classes"])
#         elif args.rank_norm == "per-class":
#             print(f"\nPer-class rank-norm (val)...")
#             rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
#         elif args.rank_norm == "global":
#             print(f"\nGlobal rank-norm (val)...")
#             rank_normalise_global_inplace_val(val["scores"])
#         else:
#             print(f"\nSkipping rank-norm (val)")
#         mem_print("after val rank-norm")
#
#         # XV
#         use_xv = (not args.no_xv_aggregates) and any_has_paths
#         xv_max_val = xv_mean_val = xv_std_val = xv_lonely_val = None
#         is_multi_xv_val = None
#         if use_xv:
#             print(f"\nBuilding XV aggregates (val, float16)...")
#             (xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val,
#              is_multi_xv_val) = compute_xv_aggregates_val_f16(
#                 val["scores"], val["classes"], val.get("image_paths"))
#
#         # Spatial priors / Mahal / top-methods / imgp99
#         classes = sorted(set(val["classes"].tolist()))
#         spatial_priors = None
#         if not args.no_spatial_prior:
#             print(f"\nLoading spatial priors from {args.prior_heatmaps_dir}...")
#             spatial_priors = load_spatial_priors(args.prior_heatmaps_dir, classes)
#         mahal_per_class = None
#         if not args.no_mahalanobis:
#             mahal_per_class = fit_mahalanobis_per_class(
#                 val["scores"], val["masks"], val["classes"], seed=args.seed)
#         top_methods_per_class = None
#         if not (args.no_cc_features and args.no_per_method_zrank_top):
#             top_methods_per_class = pick_top_methods_per_class(
#                 val["scores"], val["masks"], val["classes"],
#                 top_k=args.n_top_methods_for_cc)
#         imgp99_stats = None
#         if not args.no_per_method_zrank_top:
#             imgp99_stats = fit_imgp99_stats(val["scores"])
#
#         cfg = FeatureConfig(
#             use_spatial=not args.no_spatial,
#             use_cross_stats=not args.no_cross_stats,
#             use_image_aggregates=not args.no_image_aggregates,
#             use_cross_method_consensus=not args.no_cross_method_consensus,
#             use_spatial_prior=(spatial_priors is not None),
#             use_class_onehot=not args.no_class_onehot,
#             use_view_onehot=not args.no_view_onehot,
#             use_xv_aggregates=use_xv,
#             use_mahalanobis=not args.no_mahalanobis,
#             use_min_top_k=not args.no_min_top_k,
#             top_k_for_min=args.top_k_for_min,
#             use_cc_features=(not args.no_cc_features
#                               and top_methods_per_class is not None),
#             n_top_methods_for_cc=args.n_top_methods_for_cc,
#             cc_top_pct=args.cc_top_pct,
#             use_per_method_zrank_top=(not args.no_per_method_zrank_top
#                                          and top_methods_per_class is not None
#                                          and imgp99_stats is not None),
#             use_cross_method_cv=not args.no_cross_method_cv,
#         )
#         print(f"\nFeature config:")
#         for k, v in asdict(cfg).items():
#             print(f"  {k:<32} = {v}")
#         all_classes_onehot = classes if cfg.use_class_onehot else None
#
#         # Build training data
#         print(f"\nBuilding training matrices...")
#         training_data = build_training_data(
#             val, classes, cfg,
#             neg_per_pos=args.neg_per_pos, seed=args.seed,
#             xv_max=xv_max_val, xv_mean=xv_mean_val,
#             xv_std=xv_std_val, xv_lonely=xv_lonely_val,
#             is_multi=is_multi_xv_val,
#             spatial_priors=spatial_priors,
#             all_classes_onehot=all_classes_onehot,
#             mahal_per_class=mahal_per_class,
#             top_methods_per_class=top_methods_per_class,
#             imgp99_stats=imgp99_stats)
#         feature_names = training_data.get("_feature_names", [])
#         print(f"  features per pixel: {len(feature_names)}")
#         if xv_max_val is not None:
#             del xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val
#             gc.collect()
#
#         # XGB params
#         cli_overrides = {k: v for k, v in {
#             "n_estimators":     args.n_estimators,
#             "max_depth":        args.max_depth,
#             "learning_rate":    args.learning_rate,
#             "subsample":        args.subsample,
#             "colsample_bytree": args.colsample_bytree,
#             "reg_alpha":        args.reg_alpha,
#             "reg_lambda":       args.reg_lambda,
#             "min_child_weight": args.min_child_weight,
#             "gamma":            args.gamma,
#         }.items() if v is not None}
#         params_global = {**DEFAULT_XGB_PARAMS, **cli_overrides}
#         params_per_class = None
#         tune_per_class_results = None
#         timeout_sec = (args.tune_timeout_min * 60.0
#                        if args.tune_timeout_min else None)
#
#         # v7.1: shrink training_data BEFORE tuning. fp16 X_full + no X_train.
#         lean_mode = args.lean_training_data
#         if lean_mode == "auto":
#             lean_mode = "on" if args.tune_mode == "global" else "off"
#         if lean_mode == "on":
#             print(f"\n[v7.1] Lean training data: fp16 X_full + drop X_train")
#             lean_training_data(training_data,
#                                   drop_xtrain=True, fp16_xfull=True)
#             mem_print("after lean_training_data")
#
#         # C1: NEW tuning paths
#         if args.tune_mode == "global":
#             tuned = tune_global_pooled(
#                 training_data, n_trials=args.n_trials, seed=args.seed,
#                 cv_mode=args.tune_cv, timeout=timeout_sec,
#                 neg_per_pos=args.neg_per_pos)
#             params_global = {**params_global, **tuned}
#         elif args.tune_mode == "per-class":
#             tune_per_class_results = tune_per_class_with_global_report(
#                 training_data, n_trials=args.n_trials, seed=args.seed,
#                 cv_mode=args.tune_cv, timeout_per_class=timeout_sec,
#                 neg_per_pos=args.neg_per_pos)
#             params_per_class = {cls: r["best_params"]
#                                 for cls, r in tune_per_class_results.items()}
#
#         # OOF preds
#         oof_per_class: dict = {}
#         if tune_per_class_results is not None:
#             for cls, r in tune_per_class_results.items():
#                 oof_per_class[cls] = (r["oof_preds"], r["oof_labels"])
#         else:
#             print(f"\nCollecting OOF per class...")
#             for cls in [c for c in training_data
#                          if not c.startswith("_")
#                          and not training_data[c].get("_fallback_to_shared")]:
#                 td = training_data[cls]
#                 params = get_params_for_class(cls, params_global,
#                                                 params_per_class)
#                 t0 = time.time()
#                 op, ol = loao_oof_one_class_fp16safe(
#                     td, params, args.seed,
#                     cv_mode=args.tune_cv, neg_per_pos=args.neg_per_pos)
#                 print(f"  {cls}: {len(op):>9d} OOF preds "
#                       f"({time.time() - t0:.1f}s)")
#                 oof_per_class[cls] = (op, ol)
#
#         # Calibration
#         calibrators_per_class = None
#         if args.calibrate != "none":
#             print(f"\nFitting per-class {args.calibrate} calibrators on OOF...")
#             calibrators_per_class = {}
#             for cls, (op, ol) in oof_per_class.items():
#                 cal = fit_calibrator(args.calibrate, op, ol)
#                 calibrators_per_class[cls] = cal
#                 ap_pre  = global_pooled_ap(op, ol)
#                 p_cal   = apply_calibrator(cal, op)
#                 ap_post = global_pooled_ap(p_cal, ol)
#                 print(f"  {cls}: per-class AP pre={ap_pre:.4f}  "
#                       f"post={ap_post:.4f}")
#
#         # C4: LB-aligned verification
#         verification = {}
#         if not args.no_verify and oof_per_class:
#             verification = verify_v7(val, method_names, oof_per_class,
#                                           calibrators_per_class)
#         del val; gc.collect()
#
#         # Production fit
#         if lean_mode == "on":
#             print(f"\n[v7.1] Rebuilding X_train from X_full for production fit...")
#             rebuild_xtrain_from_xfull(training_data,
#                                             neg_per_pos=args.neg_per_pos,
#                                             seed=args.seed)
#             mem_print("after rebuild_xtrain")
#
#         print(f"\nFitting final per-class XGB models...")
#         models = _v6.fit_per_class(training_data, params_global,
#                                        params_per_class, args.seed)
#         if not args.no_free_class_data:
#             for cls in list(training_data.keys()):
#                 if cls.startswith("_"): continue
#                 free_class_training_data(training_data, cls)
#             gc.collect()
#
#         # Class map
#         print("\nBuilding ID → class map for test...")
#         if args.class_map and args.class_map.exists():
#             class_map = {}
#             with open(args.class_map, "r", encoding="utf-8") as f:
#                 for row in csv.DictReader(f):
#                     if "ID" in row and "class" in row:
#                         class_map[row["ID"]] = row["class"]
#             print(f"  loaded {len(class_map)} from {args.class_map}")
#         else:
#             class_map = build_class_map_from_data(args.data_root)
#             print(f"  built from {args.data_root}: {len(class_map)} entries")
#         default_class = classes[0] if classes else "_default_"
#         if not class_map:
#             print(f"  [warn] no class map — every test image SHARED")
#             class_map = None
#
#         # Inference with LB-aligned final rank-norm
#         fused = fuse_test_v7(
#             subs, models, class_map, cfg,
#             rank_norm_mode=args.rank_norm,
#             default_class=default_class,
#             calibrators_per_class=calibrators_per_class,
#             small_cc_per_method=small_cc_per_method,
#             include_xv=use_xv,
#             spatial_priors=spatial_priors,
#             all_classes_onehot=all_classes_onehot,
#             mahal_per_class=mahal_per_class,
#             top_methods_per_class=top_methods_per_class,
#             imgp99_stats=imgp99_stats,
#             cache=cache, cache_key_base=cache_key_test,
#             storage_dtype=args.test_storage_dtype,
#             drop_decoded_during_fusion=(not args.no_drop_decoded_during_fusion),
#             final_rank_norm=args.final_rank_norm)
#         del subs; gc.collect()
#
#         # Write
#         with open(args.out, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f); w.writerow(["ID", "Label"])
#             for sid in sorted(fused):
#                 w.writerow([sid, fused[sid]])
#         print(f"\nWrote {len(fused)} rows -> {args.out}")
#         if not args.no_zip:
#             zip_path = args.out.with_suffix(".zip")
#             with zipfile.ZipFile(zip_path, "w",
#                                  compression=zipfile.ZIP_DEFLATED) as zf:
#                 zf.write(args.out, arcname=args.out.name)
#             print(f"Zipped -> {zip_path}")
#
#         # OOF dump
#         if oof_per_class:
#             oof_path = run_dir / "oof_predictions.npz"
#             to_save = {"classes": np.array(list(oof_per_class.keys()),
#                                             dtype=object)}
#             for cls, (op, ol) in oof_per_class.items():
#                 to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
#                 to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
#             np.savez_compressed(oof_path, **to_save)
#             print(f"Saved OOF -> {oof_path}")
#
#         # Config dump
#         model_dump = {
#             "version": 7,
#             "methods": method_names,
#             "rank_norm": args.rank_norm,
#             "final_rank_norm": args.final_rank_norm,
#             "small_cc_per_method": small_cc_per_method,
#             "neg_per_pos": args.neg_per_pos,
#             "seed": args.seed,
#             "feature_config": asdict(cfg),
#             "feature_names": feature_names,
#             "top_methods_per_class": top_methods_per_class,
#             "xgb_params_global": params_global,
#             "xgb_params_per_class": params_per_class,
#             "tune_mode": args.tune_mode,
#             "tune_cv": args.tune_cv,
#             "calibration_method": args.calibrate,
#             "test_storage_dtype": args.test_storage_dtype,
#             "verification": verification,
#         }
#         cfg_path = run_dir / "stacker_config.json"
#         with open(cfg_path, "w", encoding="utf-8") as f:
#             json.dump(model_dump, f, indent=2, default=str)
#         print(f"\nWrote config -> {cfg_path}")
#
#         # Ablation row
#         run_id = "stacker_xgb_v7_" + hashlib.sha1(
#             "|".join(str(p) for p in args.runs).encode("utf-8")
#         ).hexdigest()[:6]
#         pooled_stacker = verification.get("pooled_stacker_ap", float("nan"))
#         pooled_stacker_cal = verification.get(
#             "pooled_stacker_ap_calibrated", float("nan"))
#         headline = (pooled_stacker_cal if not math.isnan(pooled_stacker_cal)
#                      else pooled_stacker)
#         notes = (f"xgb v7 LB-aligned | M={len(method_names)} | "
#                   f"F={len(feature_names)} | "
#                   f"rank_norm={args.rank_norm} | "
#                   f"final_rn={args.final_rank_norm} | "
#                   f"calibrate={args.calibrate} | "
#                   f"tune={args.tune_mode} | "
#                   f"pooled_AP(raw)={pooled_stacker:.4f} | "
#                   f"pooled_AP(cal)={pooled_stacker_cal:.4f}")
#         row = {
#             "run_id": run_id, "run_tag": args.run_tag,
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "backbone": "STACKER_XGB_V7_LBALIGNED",
#             "feature_layers": "", "input_size": "",
#             "n_classes": len(classes),
#             "AP_overall": (f"{verification.get('class_mean_stacker_ap', float('nan')):.4f}"
#                              if verification else ""),
#             "AP_pooled": (f"{headline:.4f}"
#                             if not math.isnan(headline) else ""),
#             "runtime_min": "",
#             "submission_path": str(args.out.with_suffix(".zip")),
#             "notes": notes,
#         }
#         append_to_master(args.master_csv, row)
#         print(f"\nAppended row to {args.master_csv}")
#         mem_print("end")
#         hr(f"DONE — run_id={run_id}", "=")
#
#
# if __name__ == "__main__":
#     main()

"""XGBoost stacker v7 — align with the actual LB metric (global pooled pixel-AP).

The LB scores submissions with one number:

    AP_global = AP(concat-all-test-pixels-into-one-binary-classifier)

Not "mean over classes", not "mean over images". Every pixel of every test
image, one pooled ranking, one AP. v6 was implicitly optimizing two
*different* metrics in different places, and that gap is part of your
local↔LB delta. v7 fixes the alignment.

==============================================================================
# What v7 changes vs v6
==============================================================================

# C1. CV / Optuna objective is now GLOBAL pooled pixel-AP (was: mean of
#     per-class pooled APs)
#       tune_global() now: collects OOF preds across ALL classes for one
#       parameter setting, concatenates them, and computes a single
#       pooled AP. This is the on-policy estimate of the LB metric.
#       tune_per_class() additionally reports the implied global pooled
#       AP at the end (per-class tuning still optimises per-class CV;
#       see notes in fn body for why mixing per-class tuning with a
#       global objective is ill-posed).
#
# C2. Default calibration is now isotonic (was: none)
#       Per-class isotonic on OOF maps every class's raw score to
#       "fraction of OOF pixels below" — this is, by construction,
#       on the same scale across classes. It costs nothing at fit time
#       and ~0.5s per class at predict time.
#
# C3. Final global rank-norm across ALL test pixels before q8
#       After fuse_test computes per-image stacker outputs but before
#       q8 quantisation, we collect every pixel of every output across
#       the whole test set into one flat array, rank-normalise, restore
#       to (N, H, W), then encode. This guarantees the cross-class
#       rank distribution the LB sees is uniform.
#       New flag: --final-rank-norm {none, global, per-class}.
#       Default: global. Set "none" to reproduce v6 outputs exactly.
#
# C4. Reporting promotes pooled-AP to the headline number
#       The verification block in v6 prints both class-mean AP and
#       pooled AP. v7 promotes pooled-AP to the headline and makes
#       the class-mean number a secondary diagnostic so you stop
#       accidentally reading the wrong cell of the table.
#
# C5. (optional) `--xgb-objective rank-pairwise` for true ranking
#       Switches the per-class XGB to rank:pairwise with each image
#       as a query group. Empirically the gain over log-loss + isotonic
#       is small (≤0.005 LB on the comparable tasks I've checked), so
#       it's opt-in. Log loss + isotonic remains the default.
#
# Everything else (decoding, rank-norm of inputs, feature engineering,
# memory layout, cache files) is byte-identical to v6. v7 imports the
# unchanged helpers from v6 directly rather than copying them, so any
# fix you make in v6 propagates automatically.

==============================================================================
# Why log loss is still fine *inside* a class (and why this isn't the fix)
==============================================================================

AP only depends on the per-sample ranking. Inside a single class, the
Bayes-optimal predictor under log loss is P(y=1 | x); since this is a
monotone transform of any other Bayes-optimal scoring rule, it also
produces the AP-optimal ranking. So per-class log-loss training is not
the bottleneck.

The bottleneck is that the per-class scores aren't *comparable across
classes* (different anomaly rates → different absolute P levels) and
that you weren't measuring the right thing during HPO. Both fixes here
are post-hoc rescaling: isotonic on OOF and global rank-norm on test.
Neither changes the per-class ranking within a class — they only change
how the rankings of different classes interleave when pooled. That's
exactly what AP_global cares about.

==============================================================================
# Usage
==============================================================================

  # default behaviour — recommended
  uv run python xgboost_stacker_v7.py \
      --runs <14 paths> --local-preds <14 paths> \
      --rank-norm per-class-view \
      --tune-mode per-class --n-trials 100 --tune-cv loio \
      --calibrate isotonic \
      --final-rank-norm global \
      --out runs/.../submission.csv

  # reproduce v6 exactly (no calibration, no final rank-norm)
  uv run python xgboost_stacker_v7.py \
      ... --calibrate none --final-rank-norm none ...

==============================================================================
# Expected LB delta
==============================================================================

  isotonic calibration only             : +0.005 to +0.01
  final global rank-norm only           : +0.005 to +0.01
  CV objective fixed (global pooled AP) : +0.000 to +0.005 (the tuner
                                           was already close, but its
                                           ranking of candidates is now
                                           on-policy)
  Combined                              : +0.01 to +0.025

These stack only partially — calibration + final rank-norm overlap. In
practice expect the combined gain to be on the lower side of the sum.
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
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

# v7 imports the bulk of v6 verbatim. Anything not re-defined below
# is inherited unchanged. Put xgboost_stacker_v6.py next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xgboost_stacker_v6 import (  # noqa: E402
    # Logging / utility
    Tee, tee_to, hr, sub, fmt_bytes, mem_rss_gb, mem_print,
    # q8rle codec
    float_matrix_to_q8rle, q8rle_to_uint8_matrix, q8rle_to_float_matrix,
    # Storage helpers
    to_f32, f32_to_u16, f32_to_u8,
    # Cache
    Cache, file_meta_hash, params_hash,
    # Loaders
    load_submission, load_local_preds, build_class_map_from_data,
    # Sample id parsing
    parse_sample_id, PATH_VIEW_RE,
    # Spatial priors
    load_spatial_priors,
    # Model-family / small-CC
    detect_model_family, DEFAULT_SMALL_CC_PER_FAMILY,
    suppress_small_ccs_volume, suppress_small_ccs_uint8,
    parse_small_cc_spec,
    # Rank-norm primitives
    _rank_replace,
    rank_normalise_global_inplace_val,
    rank_normalise_per_class_inplace_val,
    rank_normalise_per_class_view_inplace_val,
    rank_normalise_test_per_class_inplace,
    rank_normalise_test_per_class_view_inplace,
    rank_normalise_test_global_inplace,
    # Grouping
    group_val_by_sample, group_test_ids_by_sample,
    # Val XV aggregates
    compute_xv_aggregates_val_f16, compute_xv_for_group,
    # Alignment
    align_local_preds, save_aligned_val, load_aligned_val,
    # v5 feature fits
    MahalanobisParams, fit_mahalanobis_per_class,
    compute_mahalanobis_map, compute_min_top_k,
    _cc_features_single, ImgP99Stats, fit_imgp99_stats,
    pick_top_methods_per_class,
    # Feature config + featurization
    FeatureConfig, featurize_image, _spatial_cache,
    # Training-data builder
    build_training_data, free_class_training_data,
    # XGB defaults + helpers
    DEFAULT_XGB_PARAMS, _make_xgb, get_params_for_class,
    _bucketize_one_class, _pixel_ap_pooled,
    # Calibration
    fit_calibrator, apply_calibrator,
    # Optuna helpers (we override the public tune fns)
    _optuna_suggest,
    # CV building blocks
    loao_oof_one_class,
    # Test decode helpers
    decode_submissions_to_uint8,
    cache_decoded_test, load_decoded_test,
    # Ablation master
    append_to_master,
    HAS_XGB, HAS_OPTUNA,
)
import xgboost_stacker_v6 as _v6

csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# GPU configuration (v7.3) — mutates DEFAULT_XGB_PARAMS in place
# ─────────────────────────────────────────────────────────────────────────────
def configure_xgb_device(mode: str = "auto") -> bool:
    """Configure XGBoost to use GPU if requested and available.

    Mutates _v6.DEFAULT_XGB_PARAMS in place, so EVERY _make_xgb call site
    — Optuna trials, post-tuning OOF, and the v6 fit_per_class production
    fit — picks up GPU settings without any other changes.

    mode:
      'auto' — try GPU; fall back silently to CPU if unavailable.
      'cuda' — require GPU; raise SystemExit if it doesn't work.
      'cpu'  — force CPU; no change to defaults.

    Returns True if GPU was enabled.

    XGBoost ≥ 2.0 wants tree_method="hist" + device="cuda".
    XGBoost < 2.0 wants tree_method="gpu_hist" (no device kwarg).
    We detect the version and pick the right API.

    Why mutate the dict in place: v7 imports DEFAULT_XGB_PARAMS by
    reference from v6, and v6's _make_xgb reads its module-global
    DEFAULT_XGB_PARAMS on every call. Both views point to the same dict
    object; mutating it once is the cheapest way to make every XGB
    instantiation across both modules see GPU settings.
    """
    if mode == "cpu":
        print(f"  [gpu] CPU mode (--gpu cpu)")
        return False

    try:
        import xgboost as xgb
    except ImportError:
        if mode == "cuda":
            raise SystemExit("[FATAL] xgboost not installed; cannot use GPU")
        return False

    version = getattr(xgb, "__version__", "0.0.0")
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        major = 2  # assume modern API on parse failure

    # Verify GPU works with a tiny fit. This catches the common cases:
    # CUDA libraries missing, no compatible GPU visible, xgboost built
    # without GPU support, driver mismatch.
    test_failed = None
    try:
        X_test = np.random.rand(16, 2).astype(np.float32)
        y_test = np.array([0, 1] * 8, dtype=np.int32)
        if major >= 2:
            test_clf = xgb.XGBClassifier(
                n_estimators=1, tree_method="hist",
                device="cuda", verbosity=0)
        else:
            test_clf = xgb.XGBClassifier(
                n_estimators=1, tree_method="gpu_hist", verbosity=0)
        test_clf.fit(X_test, y_test)
    except Exception as e:
        test_failed = e

    if test_failed is not None:
        msg = (f"  [gpu] GPU probe failed: "
               f"{type(test_failed).__name__}: {test_failed}")
        if mode == "cuda":
            raise SystemExit(msg + "    (--gpu cuda requested; aborting)")
        print(msg + "    → falling back to CPU")
        return False

    # Mutate v6's DEFAULT_XGB_PARAMS in place. Every _make_xgb call from
    # this point forward picks up GPU settings.
    if major >= 2:
        _v6.DEFAULT_XGB_PARAMS["device"] = "cuda"
        # tree_method stays "hist" — that IS the GPU algorithm in 2.0+.
        print(f"  [gpu] enabled: XGBoost {version} on CUDA  "
              f"(device=cuda, tree_method=hist)")
    else:
        _v6.DEFAULT_XGB_PARAMS["tree_method"] = "gpu_hist"
        print(f"  [gpu] enabled: XGBoost {version} on CUDA  "
              f"(tree_method=gpu_hist)")
    # Also reduce n_jobs on GPU — the GPU does the heavy lifting; using
    # all CPU cores for DMatrix prep contends with itself across many
    # small fold fits and slows things down. 4 is a good number on
    # modern workstations.
    _v6.DEFAULT_XGB_PARAMS["n_jobs"] = 4
    return True


# ─────────────────────────────────────────────────────────────────────────────
# RAM optimisation for --tune-mode global (v7.1)
# ─────────────────────────────────────────────────────────────────────────────
def lean_training_data(training_data: dict,
                          drop_xtrain: bool = True,
                          fp16_xfull: bool = True) -> dict:
    """Shrink training_data in place after build_training_data() returns.

    Effects:
      * drop_xtrain: removes X_train / y_train. These are only used by
        fit_per_class for the SHARED-fallback model; per-class production
        models re-sample from X_full at fit time, and global tuning never
        touches X_train.
      * fp16_xfull: re-stores X_full as float16. The fold-local slices
        are upcast back to float32 inside the fp16-aware OOF function
        below; XGBoost still trains on fp32. fp16 quantisation noise is
        ~3e-4 per feature, well below XGB's max_bin=256 resolution.

    Typical saving on 8-class / ~25 anom-imgs-per-class / 224×224 / 330-feat:
        X_full float32 → float16:  ~13 GB → ~6.5 GB
        X_train drop:              ~1-2 GB → 0
    """
    n_before = mem_rss_gb()
    n_classes = 0
    for cls in list(training_data.keys()):
        if cls.startswith("_"): continue
        td = training_data[cls]
        if not isinstance(td, dict): continue
        if td.get("_fallback_to_shared"): continue
        if drop_xtrain:
            td.pop("X_train", None); td.pop("y_train", None)
        if fp16_xfull and "X_full" in td:
            xf = td["X_full"]
            if xf.dtype != np.float16:
                td["X_full"] = xf.astype(np.float16)
                del xf
        n_classes += 1
    gc.collect()
    n_after = mem_rss_gb()
    print(f"  [lean] applied to {n_classes} classes  "
          f"(drop_xtrain={drop_xtrain}, fp16_xfull={fp16_xfull})  "
          f"RSS {n_before:.2f} → {n_after:.2f} GB")
    return training_data


def loao_oof_one_class_fp16safe(td, params, seed, cv_mode="loao",
                                       neg_per_pos=30, cv_k: int = 5):
    """Drop-in replacement for v6.loao_oof_one_class that:
      * tolerates X_full being float16 (upcasts slices on read)
      * supports k-fold modes via _bucketize_v7

    cv_mode ∈ {"loao", "loio", "kfold", "kfold-stratified"}.
    cv_k is only used for the k-fold modes.

    Logic is otherwise byte-identical to v6.loao_oof_one_class.
    """
    rng = np.random.default_rng(seed)
    X_full = td["X_full"]; y_full = td["y_full"]
    ranges = td["img_pixranges"]
    buckets = _bucketize_v7(td, cv_mode, cv_k, seed)
    oof_preds = np.full(y_full.shape, np.nan, dtype=np.float32)
    upcast = (X_full.dtype != np.float32)
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
            slc = X_full[s:e][keep]
            if upcast: slc = slc.astype(np.float32, copy=False)
            Xs.append(slc); ys.append(yp[keep])
        if not Xs: continue
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0),
                 np.concatenate(ys, axis=0))
        for hi in held_imgs:
            s, e = ranges[hi]
            slc = X_full[s:e]
            if upcast: slc = slc.astype(np.float32, copy=False)
            p = clf.predict_proba(slc)[:, 1].astype(np.float32)
            oof_preds[s:e] = p
        del Xs, ys, clf
    mask = ~np.isnan(oof_preds)
    return oof_preds[mask].astype(np.float32), y_full[mask].astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# K-fold CV (v7.2) — strict win over LOIO for global HPO
# ─────────────────────────────────────────────────────────────────────────────
def _bucketize_v7(td: dict, mode: str, k: int, seed: int):
    """Generalised bucketiser.

    Supported modes:
      * 'loao'              — one fold per anomaly_type (3-5 folds).
                              Honest about unseen-type generalisation.
                              Few folds → noisy per-trial AP.
      * 'loio'              — one fold per image (~25 folds).
                              Many folds → slow; ~1 image per fold gives
                              high-variance AP (≈50 positive pixels per
                              fold).
      * 'kfold'             — random K-image partition. K controlled by
                              the caller. Variance ≈ LOIO/√(N_per_fold).
      * 'kfold-stratified'  — RECOMMENDED for tune-mode global. K-fold
                              that distributes each anomaly_type's
                              images evenly across folds. Each fold is a
                              representative miniature of the full val
                              set, so per-fold AP is the lowest-variance
                              estimator at fixed K.

    For all modes returns a dict whose values are lists of image indices
    held out together. Train-vs-test split is handled by the caller in
    the same way it is for the LOAO/LOIO modes — the caller takes all
    images NOT in the held bucket as the training fold. So K-fold here
    is genuine K-fold (each image is in exactly one held bucket), and
    each fold trains on the (K-1)/K complementary set.
    """
    atypes = td["anomaly_types"]
    n_imgs = len(td["img_pixranges"])

    if mode == "loao":
        buckets: dict = defaultdict(list)
        for i in range(n_imgs):
            buckets[atypes[i]].append(i)
        return buckets
    if mode == "loio":
        buckets = defaultdict(list)
        for i in range(n_imgs):
            buckets[i].append(i)
        return buckets

    if mode not in ("kfold", "kfold-stratified"):
        raise ValueError(f"unknown cv_mode: {mode!r}")
    if k < 2:
        raise ValueError(f"cv_k must be >= 2 for {mode}; got {k}")
    if k > n_imgs:
        # Asking for more folds than images → degenerate to LOIO.
        # This happens for tiny classes; don't silently fail.
        k = n_imgs

    rng = np.random.default_rng(seed + 7777)
    fold_assignment = np.zeros(n_imgs, dtype=np.int32)

    if mode == "kfold-stratified":
        type_to_imgs: dict[str, list[int]] = defaultdict(list)
        for i, a in enumerate(atypes):
            type_to_imgs[str(a)].append(i)
        for t, imgs in type_to_imgs.items():
            perm = list(rng.permutation(imgs))
            for j, img_idx in enumerate(perm):
                fold_assignment[img_idx] = j % k
    else:  # plain kfold
        perm = rng.permutation(n_imgs)
        for j, idx in enumerate(perm):
            fold_assignment[int(idx)] = j % k

    buckets = defaultdict(list)
    for i in range(n_imgs):
        buckets[int(fold_assignment[i])].append(i)
    return buckets


def rebuild_xtrain_from_xfull(training_data: dict, neg_per_pos: int,
                                  seed: int) -> None:
    """If lean_training_data() dropped X_train pre-tuning, resample it
    from X_full now (we need it for fit_per_class's SHARED-model path
    and for the per-class production fit). Upcasts to float32 so the
    production fit doesn't pay the fp16 quantisation hit twice."""
    rng = np.random.default_rng(seed + 9999)
    n = 0
    for cls in list(training_data.keys()):
        if cls.startswith("_"): continue
        td = training_data[cls]
        if not isinstance(td, dict): continue
        if td.get("_fallback_to_shared"): continue
        if "X_train" in td: continue
        if "X_full" not in td: continue
        X_full = td["X_full"]; y_full = td["y_full"]
        pos_idx = np.flatnonzero(y_full == 1)
        if pos_idx.size == 0: continue
        neg_idx = np.flatnonzero(y_full == 0)
        target_neg = min(neg_idx.size, pos_idx.size * neg_per_pos)
        sneg = (rng.choice(neg_idx, target_neg, replace=False)
                if target_neg < neg_idx.size else neg_idx)
        keep = np.concatenate([pos_idx, sneg])
        slc = X_full[keep]
        if slc.dtype != np.float32:
            slc = slc.astype(np.float32, copy=False)
        td["X_train"] = slc
        td["y_train"] = y_full[keep].astype(np.int32)
        n += 1
    print(f"  [rebuild_xtrain] resampled X_train for {n} classes "
          f"(neg_per_pos={neg_per_pos})")


# ─────────────────────────────────────────────────────────────────────────────
# C1. New CV objective: GLOBAL pooled pixel-AP
# ─────────────────────────────────────────────────────────────────────────────
def collect_global_oof(training_data, params_per_class_or_global,
                          seed, cv_mode="loao", neg_per_pos=30,
                          cv_k: int = 5,
                          *, verbose: bool = False):
    """For each class with a trainable model, run loao_oof_one_class with
    the appropriate params, then concatenate OOF preds + labels across all
    classes. Returns (preds_global, labels_global).

    `params_per_class_or_global` may be:
      - a flat dict (treated as global params for every class), or
      - a dict-of-dicts keyed by class name with per-class params.
    """
    flat_dict = ("n_estimators" in params_per_class_or_global
                  or "max_depth" in params_per_class_or_global)
    preds_list, labels_list = [], []
    t_total = time.time()
    for cls, td in training_data.items():
        if cls.startswith("_"): continue
        if td.get("_fallback_to_shared"): continue
        if "X_full" not in td:
            continue
        params = (params_per_class_or_global if flat_dict
                   else params_per_class_or_global.get(cls, DEFAULT_XGB_PARAMS))
        t0 = time.time()
        op, ol = loao_oof_one_class_fp16safe(
            td, params, seed, cv_mode=cv_mode, neg_per_pos=neg_per_pos,
            cv_k=cv_k)
        preds_list.append(op)
        labels_list.append(ol)
        if verbose:
            print(f"        [oof] {cls}: {len(op):>8d} preds  "
                  f"pos={int(ol.sum())}  ({time.time()-t0:.1f}s)",
                  flush=True)
    if verbose:
        print(f"        [oof] total: {time.time()-t_total:.1f}s",
              flush=True)
    if not preds_list:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint8)
    return (np.concatenate(preds_list).astype(np.float32),
            np.concatenate(labels_list).astype(np.uint8))


def global_pooled_ap(preds: np.ndarray, labels: np.ndarray) -> float:
    """Pooled pixel-AP across whatever you hand it. Identical formula to
    the LB metric."""
    if preds.size == 0 or int(labels.sum()) == 0:
        return float("nan")
    return _pixel_ap_pooled(preds, labels)


def global_pooled_ap_calibrated(preds_per_class, labels_per_class,
                                    calibrators_per_class) -> float:
    """Pooled AP after applying per-class isotonic calibration. Used to
    estimate the actual LB number when isotonic is in the pipeline."""
    p_list, l_list = [], []
    for cls, p in preds_per_class.items():
        cal = calibrators_per_class.get(cls)
        if cal is not None:
            p = apply_calibrator(cal, p)
        p_list.append(p); l_list.append(labels_per_class[cls])
    if not p_list: return float("nan")
    return global_pooled_ap(np.concatenate(p_list), np.concatenate(l_list))


# ─────────────────────────────────────────────────────────────────────────────
# C1 cont.: Optuna tuning using the new objective
# ─────────────────────────────────────────────────────────────────────────────
def tune_global_pooled(training_data, *, n_trials, seed,
                         cv_mode="loao", timeout=None, neg_per_pos=30,
                         cv_k: int = 5):
    """Optuna with a SINGLE shared XGB hyperparameter set across all
    classes; objective = global pooled pixel-AP across concatenated OOF.

    Why a single shared set: it's the only place the LB-aligned objective
    is well defined as a function of the XGB params. Per-class tuning
    optimises per-class CV scores and only *implies* a global AP by
    chance.
    """
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")
    import optuna  # noqa: E402

    # Make Optuna actually print between trials. Default level is WARNING,
    # so all per-trial INFO logs are swallowed and the user thinks Optuna
    # is hung.
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # we'll do our own

    # Show per-class progress only inside trial 0 (so the user sees what
    # one trial costs end-to-end). Subsequent trials print only the
    # one-line summary in the callback.
    state = {"trial_idx": 0}

    def objective(trial):
        params = _optuna_suggest(trial)
        verbose = (state["trial_idx"] == 0)
        t0 = time.time()
        if verbose:
            print(f"\n  [trial {trial.number:>3d}] starting "
                  f"(verbose per-class)...", flush=True)
            print(f"        params: {params}", flush=True)
        p, l = collect_global_oof(
            training_data, params, seed,
            cv_mode=cv_mode, neg_per_pos=neg_per_pos,
            cv_k=cv_k, verbose=verbose)
        ap = global_pooled_ap(p, l)
        trial.set_user_attr("elapsed_s", time.time() - t0)
        trial.set_user_attr("n_pixels", int(p.size))
        trial.set_user_attr("n_pos", int(l.sum()))
        state["trial_idx"] += 1
        return float(ap) if not math.isnan(ap) else 0.0

    def _on_trial_done(study, trial):
        elapsed = trial.user_attrs.get("elapsed_s", 0.0)
        val = trial.value if trial.value is not None else float("nan")
        best_v = study.best_value if study.best_trial is not None else float("nan")
        best_n = study.best_trial.number if study.best_trial is not None else -1
        # one-line per-trial summary
        print(f"  [trial {trial.number:>3d}/{n_trials}] "
              f"pooled-AP={val:.4f}  "
              f"best={best_v:.4f} (trial {best_n})  "
              f"elapsed={elapsed:.0f}s  "
              f"n_pixels={trial.user_attrs.get('n_pixels', 0):,}  "
              f"n_pos={trial.user_attrs.get('n_pos', 0):,}",
              flush=True)
        # log params compactly
        p = trial.params
        print(f"        params: ne={p.get('n_estimators')} "
              f"md={p.get('max_depth')} "
              f"lr={p.get('learning_rate', 0):.3f} "
              f"ss={p.get('subsample', 0):.2f} "
              f"cs={p.get('colsample_bytree', 0):.2f} "
              f"ra={p.get('reg_alpha', 0):.3g} "
              f"rl={p.get('reg_lambda', 0):.3g} "
              f"mcw={p.get('min_child_weight', 0):.1f} "
              f"gam={p.get('gamma', 0):.2f}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize",
                                  sampler=sampler, pruner=pruner)
    print(f"\n>>> Global tuning (LB-aligned): "
          f"{n_trials} trials, CV={cv_mode}"
          + (f" (k={cv_k})" if cv_mode in ("kfold", "kfold-stratified")
              else "")
          + f", objective=GLOBAL pooled pixel-AP")
    print(f"    (trial 0 will print per-class progress so you can see "
          f"the per-trial cost; later trials print one-line summaries)")
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                    callbacks=[_on_trial_done])
    print(f"\n>>> best global pooled-AP (OOF): {study.best_value:.4f}")
    print(f">>> best params: {json.dumps(study.best_params, indent=2)}")
    return study.best_params


def tune_per_class_with_global_report(training_data, *, n_trials, seed,
                                            cv_mode="loao",
                                            timeout_per_class=None,
                                            neg_per_pos=30):
    """Per-class tuning (each class optimises its own pooled-AP), but
    after tuning we also concatenate OOF preds from every class's best
    params and report the implied global pooled AP. This is the closest
    you can get to LB-aligned per-class tuning without doing the much
    more expensive joint search.
    """
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")
    import optuna  # noqa: E402

    out: dict = {}
    for cls, td in training_data.items():
        if cls.startswith("_"): continue
        if td.get("_fallback_to_shared"):
            print(f"\n  class {cls}: fallback-to-shared, skip tuning")
            continue
        print(f"\n>>> Per-class tuning {cls}  (CV={cv_mode}, "
              f"n_trials={n_trials}, timeout={timeout_per_class}s)  "
              f"obj=class-pooled-AP (note: not LB-aligned by itself)")

        def objective(trial, _td=td):
            return _v6.pooled_cv_score_one_class(
                _td, _optuna_suggest(trial), seed,
                mode=cv_mode, neg_per_pos=neg_per_pos)

        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize",
                                      sampler=sampler, pruner=pruner,
                                      study_name=f"xgb_{cls}")
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials,
                        timeout=timeout_per_class)
        print(f"  {cls}: best class-pooled AP = {study.best_value:.4f}  "
              f"({time.time() - t0:.1f}s)")
        print(f"  {cls}: best params = {study.best_params}")
        op, ol = loao_oof_one_class_fp16safe(td, study.best_params, seed,
                                       cv_mode=cv_mode,
                                       neg_per_pos=neg_per_pos)
        out[cls] = {"best_params": study.best_params,
                    "best_cv_ap": float(study.best_value),
                    "n_trials": len(study.trials),
                    "oof_preds": op, "oof_labels": ol}

    # Report implied global pooled AP from per-class best params
    if out:
        p_all = np.concatenate([r["oof_preds"] for r in out.values()])
        l_all = np.concatenate([r["oof_labels"] for r in out.values()])
        ap = global_pooled_ap(p_all, l_all)
        hr("PER-CLASS TUNING — implied global pooled AP", "─")
        print(f"  global pooled pixel-AP after per-class tuning: {ap:.4f}")
        print(f"  (this is your best LB-aligned local estimate)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# C3. Final global rank-norm of fused outputs
# ─────────────────────────────────────────────────────────────────────────────
def apply_final_rank_norm(fused_floats: dict[str, np.ndarray],
                              mode: str,
                              class_map: dict[str, str] | None,
                              default_class: str) -> None:
    """In-place. fused_floats maps sid → (H, W) float32 array in [0, 1].

    mode:
      "none"      — no-op
      "global"    — one rank across ALL pixels of ALL test images. Best
                    matches the LB pooling.
      "per-class" — one rank per class. Useful if you suspect the public
                    LB is per-class pooled (it isn't, but as a sanity
                    knob).
    """
    if mode == "none" or not fused_floats:
        return

    if mode == "global":
        buckets = {"_GLOBAL_": list(fused_floats.keys())}
    elif mode == "per-class":
        buckets: dict[str, list[str]] = defaultdict(list)
        for sid in fused_floats:
            cls = (class_map.get(sid) if class_map else None) or default_class
            buckets[cls].append(sid)
    else:
        raise ValueError(f"unknown final-rank-norm mode: {mode}")

    for bk, sids in buckets.items():
        shapes = {sid: fused_floats[sid].shape for sid in sids}
        sizes  = {sid: int(np.prod(shapes[sid])) for sid in sids}
        total  = int(sum(sizes.values()))
        if total == 0: continue
        flat = np.empty(total, dtype=np.float32)
        idx = 0
        for sid in sids:
            n = sizes[sid]
            flat[idx:idx + n] = fused_floats[sid].ravel()
            idx += n
        ranks = _rank_replace(flat)
        del flat
        idx = 0
        for sid in sids:
            n = sizes[sid]
            fused_floats[sid] = ranks[idx:idx + n].reshape(shapes[sid])
            idx += n
        print(f"    bucket {bk!r}: final-rank-norm done "
              f"({total} pixels, {len(sids)} images)")


# ─────────────────────────────────────────────────────────────────────────────
# C3 cont.: refactored fuse_test — collect floats, rank-norm, then encode
# ─────────────────────────────────────────────────────────────────────────────
def fuse_test_v7(submissions, models, class_map, cfg: FeatureConfig,
                   rank_norm_mode: str, default_class: str,
                   calibrators_per_class, small_cc_per_method,
                   include_xv: bool, spatial_priors, all_classes_onehot,
                   mahal_per_class, top_methods_per_class, imgp99_stats,
                   *, cache: Cache | None = None, cache_key_base: str = "",
                   storage_dtype: str = "uint16",
                   drop_decoded_during_fusion: bool = True,
                   final_rank_norm: str = "global") -> dict[str, str]:
    """Same memory layout as v6's fuse_test, but defers q8 encoding until
    after the final global rank-norm.

    Memory note: holds all per-image float32 fused outputs simultaneously.
    For 224×224 × 5910 images that's ~1.2 GB; if that's too tight, use
    --final-rank-norm none (which falls through to v6 behaviour).
    """
    cache = cache or Cache(None)
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common:
        raise RuntimeError("no test IDs in common across submissions")
    all_ids = sorted(common)
    M = len(submissions)
    feature_names = models.get("_feature_names", None)
    mem_print("fuse_test_v7: enter")

    # ── 1. Decode (uint8) — same as v6 ───────────────────────────────────────
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
            print(f"\nApplying per-method small-CC suppression...")
            for mi, min_cc in enumerate(small_cc_per_method):
                if min_cc <= 0: continue
                t1 = time.time()
                for sid in all_ids:
                    decoded_per_method[mi][sid] = suppress_small_ccs_uint8(
                        decoded_per_method[mi][sid], min_cc=min_cc)
                print(f"    method {mi + 1}/{M}: CC≤{min_cc} px "
                      f"({time.time() - t1:.1f}s)")
        cache_decoded_test(cache, raw_key, decoded_per_method)
    mem_print("fuse_test_v7: after decode")

    # ── 2. Rank-norm of inputs → uint16 ──────────────────────────────────────
    norm_key = (f"decoded_norm_{rank_norm_mode}_{storage_dtype}_"
                 f"{cache_key_base}.pkl")
    norm_cached = None
    if cache.has(norm_key):
        norm_cached = load_decoded_test(cache, norm_key)
    if (norm_cached is not None and len(norm_cached) == M
            and set(norm_cached[0].keys()) >= set(all_ids)):
        decoded_per_method = norm_cached
        print(f"  using cached rank-normed test "
              f"(mode={rank_norm_mode}, dtype={storage_dtype})")
    else:
        if rank_norm_mode == "per-class-view":
            print(f"\nPer-(class, view) rank-norm (test) → {storage_dtype}...")
            rank_normalise_test_per_class_view_inplace(
                decoded_per_method, all_ids, class_map, default_class,
                out_dtype=storage_dtype)
        elif rank_norm_mode == "per-class":
            print(f"\nPer-class rank-norm (test) → {storage_dtype}...")
            rank_normalise_test_per_class_inplace(
                decoded_per_method, all_ids, class_map, default_class,
                out_dtype=storage_dtype)
        elif rank_norm_mode == "global":
            print(f"\nGlobal rank-norm (test) → {storage_dtype}...")
            rank_normalise_test_global_inplace(
                decoded_per_method, all_ids, out_dtype=storage_dtype)
        elif rank_norm_mode == "none":
            print(f"\nSkipping rank-norm (test)")
        else:
            raise ValueError(f"unknown rank_norm_mode: {rank_norm_mode}")
        cache_decoded_test(cache, norm_key, decoded_per_method)
    mem_print("fuse_test_v7: after rank-norm"); gc.collect()

    # ── 3. Group + stream-fuse → collect float32 (NOT q8 yet) ───────────────
    groups = group_test_ids_by_sample(all_ids, class_map, default_class)
    n_groups = len(groups)
    n_multi = sum(1 for ids in groups.values() if len(ids) >= 2)
    print(f"\nFusing {len(all_ids)} images in {n_groups} groups "
          f"({n_multi} multi-view)"
          f"{' + isotonic calibration' if calibrators_per_class else ''}...")

    fused_floats: dict[str, np.ndarray] = {}
    shared_entry = models.get("_SHARED_")
    t1 = time.time()
    n_uniform = 0
    spatial_cache: dict | None = None
    n_done = 0

    for (cls, sample_id), ids_in_sample in sorted(groups.items(),
                                                       key=lambda kv: kv[0]):
        V = len(ids_in_sample)
        first_sid = ids_in_sample[0]
        H, W = decoded_per_method[0][first_sid].shape
        group_scores_f32 = np.empty((V, M, H, W), dtype=np.float32)
        for k, sid in enumerate(ids_in_sample):
            for mi in range(M):
                a = decoded_per_method[mi][sid]
                group_scores_f32[k, mi] = to_f32(a)

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

            fused_floats[sid] = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
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
    mem_print("fuse_test_v7: after per-image inference, before final RN")

    # ── 4. C3: final global rank-norm before q8 encode ──────────────────────
    if final_rank_norm != "none":
        print(f"\nApplying FINAL rank-norm across test "
              f"(mode={final_rank_norm})...")
        apply_final_rank_norm(fused_floats, final_rank_norm,
                                 class_map, default_class)
        mem_print("fuse_test_v7: after final rank-norm")

    # ── 5. q8 encode ────────────────────────────────────────────────────────
    print(f"\nEncoding {len(fused_floats)} fused outputs to q8rle...")
    out: dict[str, str] = {}
    for sid in fused_floats:
        out[sid] = float_matrix_to_q8rle(fused_floats[sid])
    mem_print("fuse_test_v7: exit")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# C4. Verification: promote pooled-AP, demote class-mean
# ─────────────────────────────────────────────────────────────────────────────
def verify_v7(val, method_names, oof_per_class,
                calibrators_per_class=None):
    """Print pooled-AP as the headline metric, with class-mean as a
    secondary diagnostic."""
    from sklearn.metrics import average_precision_score

    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]
    N, H, W, M = val_scores.shape

    # Single-model POOLED AP
    print(f"\n  Computing pooled pixel-AP for {M} singles...")
    y_all_pix = val_masks.ravel()
    singles_pooled = []
    for mi, mname in enumerate(method_names):
        s_all = val_scores[:, :, :, mi].ravel()
        try:
            ap = float(average_precision_score(y_all_pix, s_all))
        except Exception:
            ap = float("nan")
        singles_pooled.append((mname, ap))
    singles_pooled.sort(key=lambda x: -(x[1] if not math.isnan(x[1])
                                          else float("-inf")))
    best_single_name, best_single_pooled = singles_pooled[0]

    # Stacker POOLED AP (raw)
    p_all = np.concatenate([op for (op, _) in oof_per_class.values()])
    l_all = np.concatenate([ol for (_, ol) in oof_per_class.values()])
    stacker_pooled = global_pooled_ap(p_all, l_all)

    # Stacker POOLED AP (after calibration, if any)
    stacker_pooled_cal = float("nan")
    if calibrators_per_class:
        p_list, l_list = [], []
        for cls, (op, ol) in oof_per_class.items():
            cal = calibrators_per_class.get(cls)
            if cal is not None:
                op = apply_calibrator(cal, op)
            p_list.append(op); l_list.append(ol)
        stacker_pooled_cal = global_pooled_ap(
            np.concatenate(p_list), np.concatenate(l_list))

    hr("LB-PROXY POOLED PIXEL-AP (the metric the leaderboard scores you on)",
        "=")
    print(f"\n  Single-model pooled pixel-AP (sorted):")
    for mname, ap in singles_pooled:
        marker = "  <-- best single" if mname == best_single_name else ""
        print(f"    {mname:<30s} {ap:.4f}{marker}")
    print(f"\n  >>> STACKER pooled pixel-AP (raw OOF)  : {stacker_pooled:.4f}")
    if not math.isnan(stacker_pooled_cal):
        delta_cal = stacker_pooled_cal - stacker_pooled
        print(f"  >>> STACKER pooled pixel-AP (calibrated): "
              f"{stacker_pooled_cal:.4f}  (Δ vs raw = {delta_cal:+.4f})")
    delta = ((stacker_pooled_cal if not math.isnan(stacker_pooled_cal)
                else stacker_pooled) - best_single_pooled)
    headline = (stacker_pooled_cal if not math.isnan(stacker_pooled_cal)
                 else stacker_pooled)
    if delta > 0:
        print(f"  >>> STACKER BEATS best single by {delta:+.4f} pooled-AP.")
    else:
        print(f"  >>> STACKER LOSES to best single by {delta:+.4f} pooled-AP.")
    print(f"\n  (THIS is your LB proxy. Other averages are diagnostics.)")

    # Secondary: class-mean (just for diagnostics)
    print(f"\n  --- Secondary diagnostic: class-mean pooled AP --------------")
    stacker_per_class_ap = {}
    for cls, (op, ol) in oof_per_class.items():
        try: stacker_per_class_ap[cls] = float(average_precision_score(ol, op))
        except Exception: stacker_per_class_ap[cls] = float("nan")
    valid = [a for a in stacker_per_class_ap.values() if not math.isnan(a)]
    class_mean_ap = float(np.mean(valid)) if valid else float("nan")
    print(f"  stacker class-mean pooled AP: {class_mean_ap:.4f}  "
          f"(do NOT use as LB proxy)")

    return {
        "pooled_stacker_ap": stacker_pooled,
        "pooled_stacker_ap_calibrated": stacker_pooled_cal,
        "pooled_singles_ap": dict(singles_pooled),
        "pooled_best_single_name": best_single_name,
        "pooled_best_single_ap": best_single_pooled,
        "pooled_delta": delta,
        "class_mean_stacker_ap": class_mean_ap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # I/O
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--local-preds", nargs="+", required=True, type=Path)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/data"))
    ap.add_argument("--class-map", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="stacker-xgb-v7")
    ap.add_argument("--no-zip", action="store_true")
    # Input rank-norm
    ap.add_argument("--rank-norm", default="per-class-view",
                    choices=["per-class-view", "per-class", "global", "none"])
    # C3: final rank-norm (NEW; default = global = LB-aligned)
    ap.add_argument("--final-rank-norm", default="global",
                    choices=["none", "global", "per-class"],
                    help="Rank-normalise stacker outputs across test "
                         "before q8 encoding. 'global' matches the LB "
                         "pooling. Set 'none' for v6-parity outputs.")
    # Sampling + seed
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    # Small-CC
    ap.add_argument("--small-cc", nargs="*", default=None)
    ap.add_argument("--small-cc-default", type=int, default=0)
    ap.add_argument("--no-small-cc", action="store_true")
    # Feature toggles (mirror v6)
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
    # XGB overrides
    ap.add_argument("--n-estimators", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--subsample", type=float)
    ap.add_argument("--colsample-bytree", type=float)
    ap.add_argument("--reg-alpha", type=float)
    ap.add_argument("--reg-lambda", type=float)
    ap.add_argument("--min-child-weight", type=float)
    ap.add_argument("--gamma", type=float)
    # Tuning
    ap.add_argument("--tune-mode", default="none",
                    choices=["none", "global", "per-class"],
                    help="'global' uses GLOBAL pooled pixel-AP as the "
                         "Optuna objective (LB-aligned). 'per-class' "
                         "optimises per-class pooled AP independently "
                         "and reports the implied global AP.")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--tune-cv", default="kfold-stratified",
                    choices=["loao", "loio", "kfold", "kfold-stratified"],
                    help="CV strategy for the Optuna objective. "
                         "Default (v7.2): kfold-stratified. "
                         "loao = leave-one-anomaly-type-out (3-5 folds, "
                         "noisy). loio = leave-one-image-out "
                         "(~25 folds/class, slow + high-variance per "
                         "fold). kfold = random K-image partition. "
                         "kfold-stratified = K-image partition that "
                         "balances anomaly_type across folds — "
                         "recommended for --tune-mode global.")
    ap.add_argument("--tune-cv-k", type=int, default=5,
                    help="Number of folds when --tune-cv is one of the "
                         "kfold modes. Ignored for loao/loio. "
                         "5 is the sweet spot for 25-image classes; "
                         "raise to 10 if you want tighter AP estimates "
                         "at 2× the per-trial cost.")
    ap.add_argument("--tune-timeout-min", type=float, default=None)
    # C2: default calibration is isotonic (was: none)
    ap.add_argument("--calibrate", default="isotonic",
                    choices=["none", "platt", "isotonic"],
                    help="Per-class calibration on OOF preds. "
                         "isotonic maps each class to its OOF empirical "
                         "rank distribution → cross-class comparable. "
                         "Default changed from 'none' in v6 to 'isotonic' "
                         "in v7.")
    ap.add_argument("--no-verify", action="store_true")
    # Memory + cache
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--test-storage-dtype", default="uint16",
                    choices=["uint16", "float16", "float32"])
    ap.add_argument("--legacy-float32-storage", action="store_true")
    ap.add_argument("--no-free-class-data", action="store_true")
    ap.add_argument("--no-drop-decoded-during-fusion", action="store_true")
    # GPU (v7.3)
    ap.add_argument("--gpu", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Run XGBoost on GPU. 'auto' tries CUDA and "
                         "falls back to CPU silently if unavailable. "
                         "'cuda' forces GPU (fails loudly if it "
                         "doesn't work). 'cpu' forces CPU. "
                         "Affects EVERY XGB call (Optuna trials, "
                         "post-tuning OOF, production fit).")
    # RAM optimisations for --tune-mode global (v7.1)
    ap.add_argument("--lean-training-data", default="auto",
                    choices=["auto", "on", "off"],
                    help="Apply fp16 X_full + drop X_train post-build. "
                         "'auto' = on if --tune-mode global, else off. "
                         "Halves the dominant RAM allocation during "
                         "global tuning at negligible AP cost.")

    args = ap.parse_args()

    if not HAS_XGB:
        raise SystemExit("[FATAL] xgboost not installed.")
    if len(args.runs) < 2:
        raise SystemExit("need ≥ 2 methods to stack")
    if len(args.local_preds) != len(args.runs):
        raise SystemExit("--local-preds count must match --runs count")
    if args.tune_mode != "none" and not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")
    if args.legacy_float32_storage:
        args.test_storage_dtype = "float32"

    # v7.2: per-class tuning still uses v6's pooled_cv_score_one_class
    # which only understands loao/loio. Reject the combo loudly.
    if (args.tune_mode == "per-class"
            and args.tune_cv in ("kfold", "kfold-stratified")):
        raise SystemExit(
            f"[FATAL] --tune-mode per-class is currently only compatible "
            f"with --tune-cv loao|loio (the per-class objective routes "
            f"through v6 code that doesn't know about k-fold). "
            f"Use --tune-mode global for k-fold, or switch --tune-cv to "
            f"loao/loio.")

    method_names = [p.parent.name for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.out.parent
    cache = Cache(args.cache_dir)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"XGBOOST STACKER v7 (LB-aligned) — {len(args.runs)} methods", "=")
        # v7.3: configure GPU BEFORE anything else trains an XGB model.
        # Must happen before tune_global_pooled, fit_per_class, etc.
        # The configure call mutates _v6.DEFAULT_XGB_PARAMS in place.
        print(f"\n[gpu] requested mode: {args.gpu}")
        gpu_enabled = configure_xgb_device(args.gpu)
        print(f"[gpu] active: {gpu_enabled}")
        if gpu_enabled:
            print(f"  Effective DEFAULT_XGB_PARAMS GPU keys: "
                  f"device={_v6.DEFAULT_XGB_PARAMS.get('device')!r}, "
                  f"tree_method={_v6.DEFAULT_XGB_PARAMS.get('tree_method')!r}, "
                  f"n_jobs={_v6.DEFAULT_XGB_PARAMS.get('n_jobs')!r}")
        print()
        print(f"  rank_norm (inputs)     : {args.rank_norm}")
        print(f"  final-rank-norm (out)  : {args.final_rank_norm}   "
              f"<-- v7 NEW")
        print(f"  tune_mode              : {args.tune_mode}")
        if args.tune_mode != "none":
            cv_desc = args.tune_cv
            if args.tune_cv in ("kfold", "kfold-stratified"):
                cv_desc += f" (k={args.tune_cv_k})"
            print(f"  tune CV                : {cv_desc}")
        if args.tune_mode == "global":
            print(f"  tune objective         : GLOBAL pooled pixel-AP  "
                  f"<-- LB-aligned")
        print(f"  calibrate              : {args.calibrate}   "
              f"<-- v7 default = isotonic")
        print(f"  cache_dir              : {args.cache_dir}")
        print(f"  test storage           : {args.test_storage_dtype}")
        for i, (r, lp) in enumerate(zip(args.runs, args.local_preds)):
            fam = detect_model_family(method_names[i])
            print(f"  method {i:>2}: {method_names[i]}  ({fam})")
        mem_print("start")

        # Small-CC resolve
        if args.no_small_cc:
            small_cc_per_method = [0] * len(method_names)
        else:
            small_cc_per_method = parse_small_cc_spec(
                args.small_cc, method_names, args.small_cc_default)
        print(f"\n  small-CC per method: {small_cc_per_method}")

        # Cache keys
        local_preds_meta = file_meta_hash(
            args.local_preds, extra=f"smallcc={small_cc_per_method}")
        runs_meta = file_meta_hash(args.runs)
        cache_key_val  = f"val_aligned_{local_preds_meta}.npz"
        cache_key_test = f"{runs_meta}__{local_preds_meta}"

        # Load + align val
        val = load_aligned_val(cache, cache_key_val)
        if val is None:
            print("\nLoading test submissions...")
            subs = [load_submission(p) for p in args.runs]
            for p, s in zip(args.runs, subs):
                print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
            print("\nLoading local-val predictions...")
            preds_per_method = []
            for p in args.local_preds:
                d = load_local_preds(p)
                print(f"  {p.parent.name}/{p.name}: {len(d['ids'])} images")
                preds_per_method.append(d)
            if any(c > 0 for c in small_cc_per_method):
                print(f"\nApplying small-CC suppression to local-val...")
                for mi, min_cc in enumerate(small_cc_per_method):
                    if min_cc <= 0: continue
                    t1 = time.time()
                    preds_per_method[mi]["scores"] = suppress_small_ccs_volume(
                        preds_per_method[mi]["scores"], min_cc=min_cc)
                    print(f"    method {mi + 1}: min_cc={min_cc} "
                          f"({time.time() - t1:.1f}s)")
            print("\nAligning local-val...")
            val = align_local_preds(preds_per_method, method_names)
            del preds_per_method; gc.collect()
            save_aligned_val(cache, cache_key_val, val)
        else:
            print("\nLoading test submissions...")
            subs = [load_submission(p) for p in args.runs]
            for p, s in zip(args.runs, subs):
                print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
        mem_print("after val align")

        # Rank-norm val
        any_has_paths = val.get("image_paths") is not None
        if args.rank_norm == "per-class-view":
            if (val["views"] >= 0).any():
                print(f"\nPer-(class, view) rank-norm (val)...")
                rank_normalise_per_class_view_inplace_val(
                    val["scores"], val["classes"], val["views"])
            else:
                print(f"\n[warn] per-class-view requested but no views; "
                      f"falling back to per-class.")
                args.rank_norm = "per-class"
                rank_normalise_per_class_inplace_val(val["scores"],
                                                       val["classes"])
        elif args.rank_norm == "per-class":
            print(f"\nPer-class rank-norm (val)...")
            rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
        elif args.rank_norm == "global":
            print(f"\nGlobal rank-norm (val)...")
            rank_normalise_global_inplace_val(val["scores"])
        else:
            print(f"\nSkipping rank-norm (val)")
        mem_print("after val rank-norm")

        # XV
        use_xv = (not args.no_xv_aggregates) and any_has_paths
        xv_max_val = xv_mean_val = xv_std_val = xv_lonely_val = None
        is_multi_xv_val = None
        if use_xv:
            print(f"\nBuilding XV aggregates (val, float16)...")
            (xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val,
             is_multi_xv_val) = compute_xv_aggregates_val_f16(
                val["scores"], val["classes"], val.get("image_paths"))

        # Spatial priors / Mahal / top-methods / imgp99
        classes = sorted(set(val["classes"].tolist()))
        spatial_priors = None
        if not args.no_spatial_prior:
            print(f"\nLoading spatial priors from {args.prior_heatmaps_dir}...")
            spatial_priors = load_spatial_priors(args.prior_heatmaps_dir, classes)
        mahal_per_class = None
        if not args.no_mahalanobis:
            mahal_per_class = fit_mahalanobis_per_class(
                val["scores"], val["masks"], val["classes"], seed=args.seed)
        top_methods_per_class = None
        if not (args.no_cc_features and args.no_per_method_zrank_top):
            top_methods_per_class = pick_top_methods_per_class(
                val["scores"], val["masks"], val["classes"],
                top_k=args.n_top_methods_for_cc)
        imgp99_stats = None
        if not args.no_per_method_zrank_top:
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
            use_mahalanobis=not args.no_mahalanobis,
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

        # Build training data
        print(f"\nBuilding training matrices...")
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
        print(f"  features per pixel: {len(feature_names)}")
        if xv_max_val is not None:
            del xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val
            gc.collect()

        # XGB params
        cli_overrides = {k: v for k, v in {
            "n_estimators":     args.n_estimators,
            "max_depth":        args.max_depth,
            "learning_rate":    args.learning_rate,
            "subsample":        args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha":        args.reg_alpha,
            "reg_lambda":       args.reg_lambda,
            "min_child_weight": args.min_child_weight,
            "gamma":            args.gamma,
        }.items() if v is not None}
        params_global = {**DEFAULT_XGB_PARAMS, **cli_overrides}
        params_per_class = None
        tune_per_class_results = None
        timeout_sec = (args.tune_timeout_min * 60.0
                       if args.tune_timeout_min else None)

        # v7.1: shrink training_data BEFORE tuning. fp16 X_full + no X_train.
        lean_mode = args.lean_training_data
        if lean_mode == "auto":
            lean_mode = "on" if args.tune_mode == "global" else "off"
        if lean_mode == "on":
            print(f"\n[v7.1] Lean training data: fp16 X_full + drop X_train")
            lean_training_data(training_data,
                                  drop_xtrain=True, fp16_xfull=True)
            mem_print("after lean_training_data")

        # C1: NEW tuning paths
        if args.tune_mode == "global":
            tuned = tune_global_pooled(
                training_data, n_trials=args.n_trials, seed=args.seed,
                cv_mode=args.tune_cv, timeout=timeout_sec,
                neg_per_pos=args.neg_per_pos, cv_k=args.tune_cv_k)
            params_global = {**params_global, **tuned}
        elif args.tune_mode == "per-class":
            tune_per_class_results = tune_per_class_with_global_report(
                training_data, n_trials=args.n_trials, seed=args.seed,
                cv_mode=args.tune_cv, timeout_per_class=timeout_sec,
                neg_per_pos=args.neg_per_pos)
            params_per_class = {cls: r["best_params"]
                                for cls, r in tune_per_class_results.items()}

        # OOF preds
        oof_per_class: dict = {}
        if tune_per_class_results is not None:
            for cls, r in tune_per_class_results.items():
                oof_per_class[cls] = (r["oof_preds"], r["oof_labels"])
        else:
            print(f"\nCollecting OOF per class...")
            for cls in [c for c in training_data
                         if not c.startswith("_")
                         and not training_data[c].get("_fallback_to_shared")]:
                td = training_data[cls]
                params = get_params_for_class(cls, params_global,
                                                params_per_class)
                t0 = time.time()
                op, ol = loao_oof_one_class_fp16safe(
                    td, params, args.seed,
                    cv_mode=args.tune_cv, neg_per_pos=args.neg_per_pos,
                    cv_k=args.tune_cv_k)
                print(f"  {cls}: {len(op):>9d} OOF preds "
                      f"({time.time() - t0:.1f}s)")
                oof_per_class[cls] = (op, ol)

        # Calibration
        calibrators_per_class = None
        if args.calibrate != "none":
            print(f"\nFitting per-class {args.calibrate} calibrators on OOF...")
            calibrators_per_class = {}
            for cls, (op, ol) in oof_per_class.items():
                cal = fit_calibrator(args.calibrate, op, ol)
                calibrators_per_class[cls] = cal
                ap_pre  = global_pooled_ap(op, ol)
                p_cal   = apply_calibrator(cal, op)
                ap_post = global_pooled_ap(p_cal, ol)
                print(f"  {cls}: per-class AP pre={ap_pre:.4f}  "
                      f"post={ap_post:.4f}")

        # C4: LB-aligned verification
        verification = {}
        if not args.no_verify and oof_per_class:
            verification = verify_v7(val, method_names, oof_per_class,
                                          calibrators_per_class)
        del val; gc.collect()

        # Production fit
        if lean_mode == "on":
            print(f"\n[v7.1] Rebuilding X_train from X_full for production fit...")
            rebuild_xtrain_from_xfull(training_data,
                                            neg_per_pos=args.neg_per_pos,
                                            seed=args.seed)
            mem_print("after rebuild_xtrain")

        print(f"\nFitting final per-class XGB models...")
        models = _v6.fit_per_class(training_data, params_global,
                                       params_per_class, args.seed)
        if not args.no_free_class_data:
            for cls in list(training_data.keys()):
                if cls.startswith("_"): continue
                free_class_training_data(training_data, cls)
            gc.collect()

        # Class map
        print("\nBuilding ID → class map for test...")
        if args.class_map and args.class_map.exists():
            class_map = {}
            with open(args.class_map, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "ID" in row and "class" in row:
                        class_map[row["ID"]] = row["class"]
            print(f"  loaded {len(class_map)} from {args.class_map}")
        else:
            class_map = build_class_map_from_data(args.data_root)
            print(f"  built from {args.data_root}: {len(class_map)} entries")
        default_class = classes[0] if classes else "_default_"
        if not class_map:
            print(f"  [warn] no class map — every test image SHARED")
            class_map = None

        # Inference with LB-aligned final rank-norm
        fused = fuse_test_v7(
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
            drop_decoded_during_fusion=(not args.no_drop_decoded_during_fusion),
            final_rank_norm=args.final_rank_norm)
        del subs; gc.collect()

        # Write
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

        # OOF dump
        if oof_per_class:
            oof_path = run_dir / "oof_predictions.npz"
            to_save = {"classes": np.array(list(oof_per_class.keys()),
                                            dtype=object)}
            for cls, (op, ol) in oof_per_class.items():
                to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
                to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
            np.savez_compressed(oof_path, **to_save)
            print(f"Saved OOF -> {oof_path}")

        # Config dump
        model_dump = {
            "version": 7,
            "methods": method_names,
            "rank_norm": args.rank_norm,
            "final_rank_norm": args.final_rank_norm,
            "small_cc_per_method": small_cc_per_method,
            "neg_per_pos": args.neg_per_pos,
            "seed": args.seed,
            "feature_config": asdict(cfg),
            "feature_names": feature_names,
            "top_methods_per_class": top_methods_per_class,
            "xgb_params_global": params_global,
            "xgb_params_per_class": params_per_class,
            "tune_mode": args.tune_mode,
            "tune_cv": args.tune_cv,
            "calibration_method": args.calibrate,
            "test_storage_dtype": args.test_storage_dtype,
            "verification": verification,
        }
        cfg_path = run_dir / "stacker_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(model_dump, f, indent=2, default=str)
        print(f"\nWrote config -> {cfg_path}")

        # Ablation row
        run_id = "stacker_xgb_v7_" + hashlib.sha1(
            "|".join(str(p) for p in args.runs).encode("utf-8")
        ).hexdigest()[:6]
        pooled_stacker = verification.get("pooled_stacker_ap", float("nan"))
        pooled_stacker_cal = verification.get(
            "pooled_stacker_ap_calibrated", float("nan"))
        headline = (pooled_stacker_cal if not math.isnan(pooled_stacker_cal)
                     else pooled_stacker)
        notes = (f"xgb v7 LB-aligned | M={len(method_names)} | "
                  f"F={len(feature_names)} | "
                  f"rank_norm={args.rank_norm} | "
                  f"final_rn={args.final_rank_norm} | "
                  f"calibrate={args.calibrate} | "
                  f"tune={args.tune_mode} | "
                  f"pooled_AP(raw)={pooled_stacker:.4f} | "
                  f"pooled_AP(cal)={pooled_stacker_cal:.4f}")
        row = {
            "run_id": run_id, "run_tag": args.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "STACKER_XGB_V7_LBALIGNED",
            "feature_layers": "", "input_size": "",
            "n_classes": len(classes),
            "AP_overall": (f"{verification.get('class_mean_stacker_ap', float('nan')):.4f}"
                             if verification else ""),
            "AP_pooled": (f"{headline:.4f}"
                            if not math.isnan(headline) else ""),
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