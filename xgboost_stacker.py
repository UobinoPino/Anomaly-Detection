# """XGBoost stacker for Spacepresso anomaly detection.
#
# Drop-in replacement for logreg_stacker.py. Same I/O contract:
#     INPUT  : N submission.csv  +  N local_predictions.npz
#     OUTPUT : one fused submission.csv (+ .zip) + stacker_config.json
#              + ablation_master.csv row
#
# # Why XGBoost and not just more logreg
#
# Logreg with M raw method scores per pixel has a fundamental ceiling:
# the optimal mix per (class, pixel) is independent of position, texture,
# and the local geometry of the score map. Trees can express:
#   - "use method A when its 7×7-mean is high but its 3×3-std is low"
#   - "boost score where the gradient magnitude says we're on a defect edge"
#   - "downweight pixels near the image border for class_03"
#   - "if methods disagree (high range), trust the max"
# These are all interactions in score-map features that logreg cannot
# represent without manual feature engineering. XGBoost finds them.
#
# # Features per pixel (Category 3: lazy transforms of score_map)
#
# Computed by `featurize_image()` from each method's score map. Nothing
# is saved beyond what local_preds_saver already writes. To experiment
# with new features, just edit FeatureConfig and re-run the stacker —
# no need to re-run the baselines.
#
#   Per method  (F_pm columns; ~17 per method by default):
#     raw                  : rank-normed score (default) or raw score
#     rank                 : per-image percentile rank
#     g{σ}                 : Gaussian-smoothed copies, σ ∈ {1, 3, 7}
#     mean{w}, max{w},
#     std{w}               : neighborhood stats, window ∈ {3, 7, 15}
#     grad                 : Sobel gradient magnitude
#     lap                  : Laplacian (2nd derivative)
#     d2hot                : Euclidean distance to nearest pixel above
#                             the per-image 99-th percentile
#     imgmax, imgp99,
#     imgmean, imgstd      : image-level aggregates broadcast as
#                             constant planes (gives trees a coarse
#                             "is this image anomalous at all" signal)
#
#   Cross-method  (5 columns; only if M >= 2):
#     x_mean, x_max,
#     x_min, x_std         : statistics over methods
#     x_range              : max - min, a disagreement signal
#
#   Spatial  (4 columns; shared across methods, computed once per (H,W)):
#     s_x, s_y             : normalized pixel coords in [0, 1]
#     s_dist_edge          : min distance to image edge
#     s_dist_center        : distance to image center
#
# Default total for M=3 methods: 3×17 + 5 + 4 = 60 features per pixel.
#
# # CLI
#
#     python xgboost_stacker.py \\
#         --runs        runs/<exp7>/submission.csv \\
#                       runs/<exp8c>/submission.csv \\
#                       runs/<exp10b>/submission.csv \\
#         --local-preds runs/<exp7>/local_predictions.npz \\
#                       runs/<exp8c>/local_predictions.npz \\
#                       runs/<exp10b>/local_predictions.npz \\
#         --data-root   /work/.../data \\
#         --out         runs/stacker_xgb_v1/submission.csv \\
#         --run-tag     stacker-xgb-exp7-exp8c-exp10b
#
# With Optuna tuning:
#
#     python xgboost_stacker.py ... --tune --n-trials 60 \\
#         --tune-cv loao   # leave-one-anomaly-type-out
#
# # Dependencies
#
#     uv pip install xgboost scipy scikit-learn
#     uv pip install optuna     # only if you use --tune
# """
# from __future__ import annotations
#
# import argparse
# import csv
# import hashlib
# import json
# import sys
# import time
# import zipfile
# from collections import defaultdict
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
#
# import numpy as np
# from scipy import ndimage as ndi
#
# try:
#     import xgboost as xgb
#     HAS_XGB = True
# except Exception:
#     HAS_XGB = False
#
# try:
#     import optuna
#     HAS_OPTUNA = True
# except Exception:
#     HAS_OPTUNA = False
#
# # q8rle strings can exceed csv's default 128 KB field limit.
# csv.field_size_limit(sys.maxsize)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # q8rle codec — bit-identical to logreg_stacker.py / score_fusion.py
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
#     parts = s.split()
#     h, w = int(parts[1]), int(parts[2])
#     if len(parts) <= 3:
#         return np.zeros((h, w), dtype=np.float32)
#     body = np.array(parts[3:], dtype=np.int64)
#     vals = body[0::2].astype(np.uint8)
#     lens = body[1::2]
#     flat = np.repeat(vals, lens).reshape(w, h).T
#     return flat.astype(np.float32) / 255.0
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Loaders (same as logreg_stacker)
# # ─────────────────────────────────────────────────────────────────────────────
# def load_submission(path: Path) -> dict[str, str]:
#     out: dict[str, str] = {}
#     with open(path, "r", encoding="utf-8") as f:
#         reader = csv.reader(f)
#         header = next(reader, None)
#         if header != ["ID", "Label"]:
#             raise ValueError(f"{path}: unexpected header {header}")
#         for row in reader:
#             if len(row) >= 2:
#                 out[row[0]] = row[1]
#     return out
#
#
# def load_local_preds(path: Path) -> dict:
#     if not path.exists():
#         raise FileNotFoundError(
#             f"{path} not found. Re-run the baseline with the "
#             f"local_preds_saver hook (see local_preds_saver.py).")
#     data = np.load(path, allow_pickle=True)
#     return {
#         "ids":           data["ids"].astype(str),
#         "classes":       data["classes"].astype(str),
#         "anomaly_types": data["anomaly_types"].astype(str),
#         "scores":        data["scores"].astype(np.float32),
#         "masks":         data["masks"].astype(np.uint8),
#     }
#
#
# def build_class_map_from_data(data_root: Path) -> dict[str, str]:
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
#             if p.is_file() and p.suffix.lower() in {
#                     ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
#                 out[p.stem] = cdir.name
#     return out
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Local-val alignment (same as logreg_stacker, slimmed)
# # ─────────────────────────────────────────────────────────────────────────────
# def _nn_resize_2d(arr: np.ndarray, target_shape, dtype) -> np.ndarray:
#     th, tw = target_shape
#     h, w = arr.shape
#     ys = np.linspace(0, h - 1, th).round().astype(np.int64)
#     xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
#     return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)
#
#
# def align_local_preds(preds_per_method: list[dict],
#                        method_names: list[str]) -> dict:
#     id_sets = [set(p["ids"].tolist()) for p in preds_per_method]
#     common = sorted(set.intersection(*id_sets))
#     if not common:
#         raise RuntimeError("no local-val IDs in common across the methods")
#     H0, W0 = preds_per_method[0]["scores"].shape[1:3]
#     print(f"  reference shape (method 0): {H0}x{W0}")
#     indexed = []
#     for p in preds_per_method:
#         idx_of = {id_: i for i, id_ in enumerate(p["ids"])}
#         indexed.append((p, idx_of))
#     N = len(common); M = len(preds_per_method)
#     scores = np.empty((N, H0, W0, M), dtype=np.float32)
#     masks  = np.empty((N, H0, W0), dtype=np.uint8)
#     classes = np.empty(N, dtype=object)
#     anomaly_types = np.empty(N, dtype=object)
#     for i, id_ in enumerate(common):
#         for mi, (p, idx_of) in enumerate(indexed):
#             j = idx_of[id_]
#             s = p["scores"][j]
#             if s.shape != (H0, W0):
#                 s = _nn_resize_2d(s, (H0, W0), np.float32)
#             scores[i, :, :, mi] = s
#             if mi == 0:
#                 classes[i] = str(p["classes"][j])
#                 anomaly_types[i] = str(p["anomaly_types"][j])
#                 m = p["masks"][j]
#                 if m.shape != (H0, W0):
#                     m = _nn_resize_2d(m, (H0, W0), np.uint8)
#                 masks[i] = m
#     print(f"  aligned {N} val images × {M} methods @ {H0}x{W0}")
#     return {"ids": np.asarray(common),
#             "classes": classes.astype(str),
#             "anomaly_types": anomaly_types.astype(str),
#             "scores": scores,
#             "masks": masks}
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Rank normalisation (global, per method) — same idea as logreg_stacker
# # ─────────────────────────────────────────────────────────────────────────────
# def rank_normalise_in_place_val(scores: np.ndarray) -> None:
#     """scores shape: (..., M). Replace each method's values by their
#     fractional rank in [0, 1] over all non-method dims jointly."""
#     M = scores.shape[-1]
#     flat_view = scores.reshape(-1, M)
#     n = flat_view.shape[0]
#     linsp = np.linspace(0.0, 1.0, n, dtype=np.float32)
#     for mi in range(M):
#         col = flat_view[:, mi]
#         order = np.argsort(col, kind="stable")
#         ranks = np.empty_like(col)
#         ranks[order] = linsp
#         flat_view[:, mi] = ranks
#
#
# def rank_normalise_test_in_place(decoded_per_method: list[dict[str, np.ndarray]],
#                                    all_ids: list[str]) -> None:
#     for mi, d in enumerate(decoded_per_method):
#         t1 = time.time()
#         shapes = {sid: d[sid].shape for sid in all_ids}
#         sizes  = {sid: int(np.prod(shapes[sid])) for sid in all_ids}
#         total  = sum(sizes.values())
#         print(f"    method {mi + 1}/{len(decoded_per_method)}: "
#               f"global rank-norm over {total:,} pixels...")
#         flat = np.empty(total, dtype=np.float32)
#         idx = 0
#         for sid in all_ids:
#             n = sizes[sid]
#             flat[idx:idx + n] = d[sid].ravel()
#             idx += n
#         order = np.argsort(flat, kind="stable")
#         ranks = np.empty_like(flat)
#         ranks[order] = np.linspace(0.0, 1.0, total, dtype=np.float32)
#         del order, flat
#         idx = 0
#         for sid in all_ids:
#             n = sizes[sid]
#             d[sid] = ranks[idx:idx + n].reshape(shapes[sid]).astype(np.float32)
#             idx += n
#         del ranks
#         print(f"      done in {time.time() - t1:.1f}s")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Feature configuration + helpers
# # ─────────────────────────────────────────────────────────────────────────────
# @dataclass
# class FeatureConfig:
#     # Per-method, on each method's score map
#     use_raw_score: bool = True
#     use_per_image_rank: bool = True
#     gauss_sigmas: tuple[float, ...] = (1.0, 3.0, 7.0)
#     window_sizes: tuple[int, ...] = (3, 7, 15)
#     use_window_mean: bool = True
#     use_window_max: bool = True
#     use_window_std: bool = True
#     use_gradient: bool = True
#     use_laplacian: bool = True
#     use_dist_to_hot: bool = True
#     hot_percentile: float = 99.0
#     use_image_aggregates: bool = True
#
#     # Cross-method
#     use_cross_stats: bool = True
#
#     # Spatial (image-level, shared across methods)
#     use_spatial: bool = True
#
#
# def _per_image_rank(s: np.ndarray) -> np.ndarray:
#     flat = s.ravel()
#     order = np.argsort(flat, kind="stable")
#     ranks = np.empty_like(flat, dtype=np.float32)
#     ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
#     return ranks.reshape(s.shape)
#
#
# def _window_mean(s: np.ndarray, size: int) -> np.ndarray:
#     return ndi.uniform_filter(s, size=size, mode="reflect").astype(np.float32)
#
#
# def _window_max(s: np.ndarray, size: int) -> np.ndarray:
#     return ndi.maximum_filter(s, size=size, mode="reflect").astype(np.float32)
#
#
# def _window_std(s: np.ndarray, size: int) -> np.ndarray:
#     """sqrt(E[X^2] - E[X]^2) via two box filters."""
#     mean = ndi.uniform_filter(s, size=size, mode="reflect")
#     sq = ndi.uniform_filter(s * s, size=size, mode="reflect")
#     var = np.clip(sq - mean * mean, 0.0, None)
#     return np.sqrt(var).astype(np.float32)
#
#
# def _gauss(s: np.ndarray, sigma: float) -> np.ndarray:
#     return ndi.gaussian_filter(s, sigma=sigma, mode="reflect").astype(np.float32)
#
#
# def _grad_mag(s: np.ndarray) -> np.ndarray:
#     sx = ndi.sobel(s, axis=0, mode="reflect")
#     sy = ndi.sobel(s, axis=1, mode="reflect")
#     return np.sqrt(sx * sx + sy * sy).astype(np.float32)
#
#
# def _laplacian(s: np.ndarray) -> np.ndarray:
#     return ndi.laplace(s, mode="reflect").astype(np.float32)
#
#
# def _dist_to_hot(s: np.ndarray, pct: float) -> np.ndarray:
#     thresh = float(np.percentile(s, pct))
#     hot = s >= thresh
#     if not hot.any():
#         H, W = s.shape
#         return np.full(s.shape, float(np.hypot(H, W)), dtype=np.float32)
#     # distance_transform_edt returns distance to nearest 0; we want to
#     # measure distance to nearest 1 (hot pixel), so invert.
#     return ndi.distance_transform_edt(~hot).astype(np.float32)
#
#
# def _spatial_cache(H: int, W: int) -> dict:
#     ys, xs = np.indices((H, W)).astype(np.float32)
#     xs_n = xs / max(W - 1, 1)
#     ys_n = ys / max(H - 1, 1)
#     de = np.minimum(np.minimum(xs_n, ys_n),
#                      np.minimum(1.0 - xs_n, 1.0 - ys_n)).astype(np.float32)
#     cx, cy = 0.5, 0.5
#     dc = np.sqrt((xs_n - cx) ** 2 + (ys_n - cy) ** 2).astype(np.float32)
#     return {"x": xs_n, "y": ys_n, "dist_edge": de, "dist_center": dc,
#             "_shape": (H, W)}
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Featurize one image
# # ─────────────────────────────────────────────────────────────────────────────
# def featurize_image(
#     scores_per_method: list[np.ndarray],
#     cfg: FeatureConfig,
#     spatial_cache: dict | None = None,
#     feature_names: list[str] | None = None,
# ) -> tuple[np.ndarray, list[str]]:
#     """Compute per-pixel features for ONE image, across M methods.
#
#     Returns:
#         feats: (H, W, F) float32.
#         names: list of F strings.
#
#     If `feature_names` is given, this is treated as an inference call:
#     output is asserted to match. (Catches drift between fit-time and
#     predict-time feature sets.)
#     """
#     assert scores_per_method, "no methods supplied"
#     H, W = scores_per_method[0].shape
#     M = len(scores_per_method)
#
#     layers: list[np.ndarray] = []
#     names: list[str] = []
#
#     # ─── Per-method features
#     for mi, s in enumerate(scores_per_method):
#         if s.shape != (H, W):
#             raise ValueError(f"method {mi} shape {s.shape} != ({H}, {W})")
#         s = np.asarray(s, dtype=np.float32)
#
#         if cfg.use_raw_score:
#             layers.append(s); names.append(f"m{mi}_raw")
#         if cfg.use_per_image_rank:
#             layers.append(_per_image_rank(s)); names.append(f"m{mi}_rank")
#         for sigma in cfg.gauss_sigmas:
#             layers.append(_gauss(s, sigma)); names.append(f"m{mi}_g{sigma:g}")
#         for ws in cfg.window_sizes:
#             if cfg.use_window_mean:
#                 layers.append(_window_mean(s, ws)); names.append(f"m{mi}_mean{ws}")
#             if cfg.use_window_max:
#                 layers.append(_window_max(s, ws));  names.append(f"m{mi}_max{ws}")
#             if cfg.use_window_std:
#                 layers.append(_window_std(s, ws));  names.append(f"m{mi}_std{ws}")
#         if cfg.use_gradient:
#             layers.append(_grad_mag(s));  names.append(f"m{mi}_grad")
#         if cfg.use_laplacian:
#             layers.append(_laplacian(s)); names.append(f"m{mi}_lap")
#         if cfg.use_dist_to_hot:
#             layers.append(_dist_to_hot(s, cfg.hot_percentile))
#             names.append(f"m{mi}_d2hot")
#         if cfg.use_image_aggregates:
#             mx = float(s.max())
#             pp = float(np.percentile(s, 99))
#             mn = float(s.mean())
#             st = float(s.std())
#             layers.append(np.full((H, W), mx, dtype=np.float32))
#             names.append(f"m{mi}_imgmax")
#             layers.append(np.full((H, W), pp, dtype=np.float32))
#             names.append(f"m{mi}_imgp99")
#             layers.append(np.full((H, W), mn, dtype=np.float32))
#             names.append(f"m{mi}_imgmean")
#             layers.append(np.full((H, W), st, dtype=np.float32))
#             names.append(f"m{mi}_imgstd")
#
#     # ─── Cross-method features
#     if cfg.use_cross_stats and M >= 2:
#         stack = np.stack(scores_per_method, axis=0).astype(np.float32)  # (M,H,W)
#         cmean = stack.mean(axis=0); layers.append(cmean); names.append("x_mean")
#         cmax  = stack.max(axis=0);  layers.append(cmax);  names.append("x_max")
#         cmin  = stack.min(axis=0);  layers.append(cmin);  names.append("x_min")
#         cstd  = stack.std(axis=0);  layers.append(cstd);  names.append("x_std")
#         layers.append((cmax - cmin).astype(np.float32))
#         names.append("x_range")
#
#     # ─── Spatial features
#     if cfg.use_spatial:
#         if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
#             spatial_cache = _spatial_cache(H, W)
#         for k in ("x", "y", "dist_edge", "dist_center"):
#             layers.append(spatial_cache[k]); names.append(f"s_{k}")
#
#     feats = np.stack(layers, axis=-1).astype(np.float32, copy=False)
#
#     if feature_names is not None and names != feature_names:
#         raise RuntimeError(
#             "feature drift between fit and predict: "
#             f"expected {len(feature_names)} cols, got {len(names)}.\n"
#             f"  fit:  {feature_names[:5]} ... {feature_names[-5:]}\n"
#             f"  now:  {names[:5]} ... {names[-5:]}")
#
#     return feats, names
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Build training matrices per class (featurize val, sample negatives)
# # ─────────────────────────────────────────────────────────────────────────────
# def build_training_data(
#     val: dict,
#     classes: list[str],
#     cfg: FeatureConfig,
#     *,
#     neg_per_pos: int,
#     seed: int,
# ) -> dict:
#     """For each class: featurize all val images, then subsample negative
#     pixels at the class level (matching logreg_stacker.py)."""
#     rng = np.random.default_rng(seed)
#     val_scores = val["scores"]        # (N, H, W, M)
#     val_masks  = val["masks"]         # (N, H, W)
#     val_classes = val["classes"]      # (N,)
#     val_anom_types = val["anomaly_types"]
#     N, H, W, M = val_scores.shape
#
#     spatial = _spatial_cache(H, W)
#
#     out: dict = {}
#     feature_names_global: list[str] | None = None
#
#     for cls in classes:
#         cls_idx = np.flatnonzero(val_classes == cls)
#         if cls_idx.size == 0:
#             print(f"  [warn] class {cls}: no val images; skipping")
#             continue
#
#         per_img_X: list[np.ndarray] = []
#         per_img_y: list[np.ndarray] = []
#         per_img_groups: list[np.ndarray] = []   # for CV
#         per_img_atypes: list[str] = []           # for CV
#         per_img_pixrange: list[tuple[int, int]] = []  # for AP evaluation
#
#         cursor = 0
#         for img_i, i in enumerate(cls_idx):
#             mat_list = [val_scores[i, :, :, mi] for mi in range(M)]
#             feats, names = featurize_image(mat_list, cfg, spatial_cache=spatial)
#             if feature_names_global is None:
#                 feature_names_global = names
#             F = feats.shape[-1]
#             flat_x = feats.reshape(-1, F)
#             flat_y = val_masks[i].ravel().astype(np.int32)
#             per_img_X.append(flat_x)
#             per_img_y.append(flat_y)
#             per_img_groups.append(np.full(flat_x.shape[0], img_i, dtype=np.int32))
#             per_img_atypes.append(str(val_anom_types[i]))
#             per_img_pixrange.append((cursor, cursor + flat_x.shape[0]))
#             cursor += flat_x.shape[0]
#             del feats
#
#         X_full = np.concatenate(per_img_X, axis=0).astype(np.float32)
#         y_full = np.concatenate(per_img_y, axis=0).astype(np.int32)
#         g_full = np.concatenate(per_img_groups, axis=0)
#         del per_img_X, per_img_y, per_img_groups
#
#         n_pos = int((y_full == 1).sum())
#         n_neg = int((y_full == 0).sum())
#
#         # Class-level negative subsampling (same as logreg_stacker)
#         if n_pos == 0:
#             print(f"  [warn] class {cls}: zero positive pixels in val; "
#                   f"this class will use SHARED model at predict time")
#             out[cls] = {"_fallback_to_shared": True,
#                         "n_pos": 0, "n_neg_sampled": 0,
#                         # Keep CV data so the tuner can still skip cleanly.
#                         "X_full": X_full, "y_full": y_full,
#                         "groups": g_full,
#                         "anomaly_types": per_img_atypes,
#                         "img_pixranges": per_img_pixrange}
#             continue
#
#         target_neg = min(n_neg, n_pos * neg_per_pos)
#         neg_idx = np.flatnonzero(y_full == 0)
#         if target_neg < n_neg:
#             sample_neg = rng.choice(neg_idx, size=target_neg, replace=False)
#         else:
#             sample_neg = neg_idx
#         pos_idx = np.flatnonzero(y_full == 1)
#         keep = np.concatenate([pos_idx, sample_neg])
#         X_train = X_full[keep]
#         y_train = y_full[keep]
#
#         out[cls] = {
#             "X_train": X_train, "y_train": y_train,
#             "X_full":  X_full,  "y_full":  y_full,   # for CV
#             "groups":  g_full,
#             "anomaly_types": per_img_atypes,
#             "img_pixranges": per_img_pixrange,
#             "n_pos": n_pos, "n_neg_sampled": int(len(sample_neg)),
#         }
#         print(f"  class {cls}: train {X_train.shape[0]:>8d} rows  "
#               f"(pos={n_pos:>6d}, neg={len(sample_neg):>8d}, "
#               f"feats={X_train.shape[1]:>3d})")
#
#     out["_feature_names"] = feature_names_global or []
#     return out
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # XGBoost parameters
# # ─────────────────────────────────────────────────────────────────────────────
# DEFAULT_XGB_PARAMS: dict = {
#     "n_estimators":   300,
#     "max_depth":      6,
#     "learning_rate":  0.07,
#     "subsample":      0.8,
#     "colsample_bytree": 0.8,
#     "reg_alpha":      0.0,
#     "reg_lambda":     1.0,
#     "min_child_weight": 1.0,
#     "tree_method":    "hist",
#     "max_bin":        256,
#     "objective":      "binary:logistic",
#     "eval_metric":    "logloss",
#     "n_jobs":         -1,
#     "verbosity":      0,
# }
#
#
# def _make_xgb(params: dict, seed: int) -> "xgb.XGBClassifier":
#     if not HAS_XGB:
#         raise SystemExit("xgboost not installed. uv pip install xgboost")
#     p = dict(DEFAULT_XGB_PARAMS)
#     p.update(params or {})
#     return xgb.XGBClassifier(random_state=seed, **p)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Fit per-class XGBoost models (+ shared fallback)
# # ─────────────────────────────────────────────────────────────────────────────
# def fit_per_class(training_data: dict, params: dict, seed: int,
#                   min_pos_for_per_class: int = 200) -> dict:
#     from sklearn.metrics import log_loss, average_precision_score
#
#     feature_names = training_data.get("_feature_names", [])
#
#     out: dict = {"_feature_names": feature_names}
#     shared_X: list[np.ndarray] = []
#     shared_y: list[np.ndarray] = []
#
#     for cls, td in training_data.items():
#         if cls.startswith("_"):
#             continue
#         if td.get("_fallback_to_shared"):
#             out[cls] = {"_fallback_to_shared": True,
#                         "n_pos": td.get("n_pos", 0),
#                         "n_neg_sampled": td.get("n_neg_sampled", 0)}
#             continue
#
#         X = td["X_train"]; y = td["y_train"]
#         shared_X.append(X); shared_y.append(y)
#
#         if td["n_pos"] < min_pos_for_per_class:
#             print(f"  class {cls}: only {td['n_pos']} positives "
#                   f"(< {min_pos_for_per_class}); will use SHARED model")
#             out[cls] = {"_fallback_to_shared": True,
#                         "n_pos": td["n_pos"],
#                         "n_neg_sampled": td["n_neg_sampled"]}
#             continue
#
#         clf = _make_xgb(params, seed)
#         clf.fit(X, y)
#         p = clf.predict_proba(X)[:, 1]
#         try: ll = float(log_loss(y, p, labels=[0, 1]))
#         except Exception: ll = float("nan")
#         try: ap = float(average_precision_score(y, p))
#         except Exception: ap = float("nan")
#         out[cls] = {"model": clf,
#                     "n_pos": td["n_pos"],
#                     "n_neg_sampled": td["n_neg_sampled"],
#                     "logloss": ll, "train_ap": ap,
#                     "feature_importance": clf.feature_importances_.tolist()}
#         print(f"  class {cls}: n_pos={td['n_pos']:>6d}  "
#               f"n_neg={td['n_neg_sampled']:>8d}  "
#               f"train_logloss={ll:.4f}  train_ap={ap:.3f}")
#
#     # Shared fallback model
#     if shared_X:
#         X_all = np.concatenate(shared_X, axis=0)
#         y_all = np.concatenate(shared_y, axis=0)
#         clf_sh = _make_xgb(params, seed)
#         clf_sh.fit(X_all, y_all)
#         p_sh = clf_sh.predict_proba(X_all)[:, 1]
#         try: ll_sh = float(log_loss(y_all, p_sh, labels=[0, 1]))
#         except Exception: ll_sh = float("nan")
#         try: ap_sh = float(average_precision_score(y_all, p_sh))
#         except Exception: ap_sh = float("nan")
#         out["_SHARED_"] = {"model": clf_sh,
#                             "n_pos": int(y_all.sum()),
#                             "n_neg_sampled": int(len(y_all) - y_all.sum()),
#                             "logloss": ll_sh, "train_ap": ap_sh,
#                             "feature_importance":
#                                 clf_sh.feature_importances_.tolist()}
#         print(f"  SHARED:    n_pos={int(y_all.sum()):>6d}  "
#               f"n_neg={int(len(y_all) - y_all.sum()):>8d}  "
#               f"train_logloss={ll_sh:.4f}  train_ap={ap_sh:.3f}")
#
#     return out
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Cross-validation harness for Optuna
# # ─────────────────────────────────────────────────────────────────────────────
# def cv_score(training_data: dict, params: dict, seed: int,
#              mode: str = "loao", verbose: bool = False) -> float:
#     """Estimate pixel-AP on held-out anomaly buckets using ONE shared
#     XGBoost (so per-class data is pooled — gives stable signal even
#     when individual classes have few positives).
#
#     mode='loao' : leave-one-(class, anomaly_type)-out
#     mode='loio' : leave-one-image-out (more folds, slower)
#     """
#     from sklearn.metrics import average_precision_score
#
#     # Build the master pool: for each (class, image_index) collect
#     # the full pixel set + its anomaly_type and class.
#     classes = [c for c in training_data if not c.startswith("_")]
#     img_records = []  # list of dicts: cls, atype, X_pix (all pixels of img),
#                        #               y_pix, idx_in_class
#     for cls in classes:
#         td = training_data[cls]
#         if td.get("_fallback_to_shared"):
#             continue
#         X_full = td["X_full"]; y_full = td["y_full"]
#         atypes = td["anomaly_types"]
#         ranges = td["img_pixranges"]
#         for img_i, ((s, e), at) in enumerate(zip(ranges, atypes)):
#             img_records.append({
#                 "cls": cls, "atype": at, "img_i": img_i,
#                 "Xp": X_full[s:e], "yp": y_full[s:e],
#             })
#
#     if not img_records:
#         return 0.0
#
#     # Define folds
#     if mode == "loao":
#         bucket_key = lambda r: (r["cls"], r["atype"])
#     elif mode == "loio":
#         bucket_key = lambda r: (r["cls"], r["img_i"])
#     else:
#         raise ValueError(f"unknown mode: {mode}")
#
#     buckets: dict = defaultdict(list)
#     for r in img_records:
#         buckets[bucket_key(r)].append(r)
#
#     rng = np.random.default_rng(seed)
#     neg_per_pos = 30   # match the main training subsample ratio
#
#     per_fold_aps: list[float] = []
#     for fold_key, held in buckets.items():
#         train_records = [r for r in img_records if bucket_key(r) != fold_key]
#         # Build a balanced training set across all train_records.
#         Xs, ys = [], []
#         n_pos_total = 0
#         for r in train_records:
#             yp = r["yp"]
#             pos = np.flatnonzero(yp == 1)
#             neg = np.flatnonzero(yp == 0)
#             if pos.size == 0:
#                 # Skip clean images (no signal) — same convention as
#                 # the main subsample.
#                 continue
#             n_keep_neg = min(neg.size, pos.size * neg_per_pos)
#             sneg = rng.choice(neg, size=n_keep_neg, replace=False) \
#                     if n_keep_neg < neg.size else neg
#             keep = np.concatenate([pos, sneg])
#             Xs.append(r["Xp"][keep])
#             ys.append(yp[keep])
#             n_pos_total += int(pos.size)
#         if not Xs or n_pos_total == 0:
#             continue
#         Xtr = np.concatenate(Xs, axis=0)
#         ytr = np.concatenate(ys, axis=0)
#         clf = _make_xgb(params, seed)
#         clf.fit(Xtr, ytr)
#         # Eval on FULL held-out pixels (no subsample).
#         fold_aps = []
#         for r in held:
#             if int(r["yp"].sum()) == 0:
#                 continue
#             p = clf.predict_proba(r["Xp"])[:, 1]
#             try:
#                 ap = float(average_precision_score(r["yp"], p))
#             except Exception:
#                 continue
#             fold_aps.append(ap)
#         if fold_aps:
#             per_fold_aps.extend(fold_aps)
#         if verbose:
#             print(f"    fold {fold_key}: n_held={len(held)}  "
#                   f"mean_AP={np.mean(fold_aps) if fold_aps else 0:.4f}")
#
#     return float(np.mean(per_fold_aps)) if per_fold_aps else 0.0
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Optuna driver
# # ─────────────────────────────────────────────────────────────────────────────
# def tune_with_optuna(training_data: dict, *,
#                      n_trials: int, seed: int,
#                      cv_mode: str = "loao",
#                      timeout: float | None = None) -> dict:
#     if not HAS_OPTUNA:
#         raise SystemExit("optuna not installed. uv pip install optuna")
#
#     def objective(trial: "optuna.Trial") -> float:
#         params = {
#             "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
#             "max_depth":        trial.suggest_int("max_depth", 3, 8),
#             "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
#             "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#             "reg_alpha":        trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
#             "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
#             "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
#         }
#         ap = cv_score(training_data, params, seed, mode=cv_mode)
#         return ap
#
#     sampler = optuna.samplers.TPESampler(seed=seed)
#     pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
#     study = optuna.create_study(direction="maximize",
#                                   sampler=sampler, pruner=pruner)
#     print(f"\n>>> Optuna tuning: {n_trials} trials, CV={cv_mode}")
#     study.optimize(objective, n_trials=n_trials, timeout=timeout,
#                     show_progress_bar=False)
#     print(f">>> best AP: {study.best_value:.4f}")
#     print(f">>> best params: {json.dumps(study.best_params, indent=2)}")
#     return study.best_params
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Inference on the test set
# # ─────────────────────────────────────────────────────────────────────────────
# def fuse_test(
#     submissions: list[dict[str, str]],
#     models: dict,
#     class_map: dict[str, str] | None,
#     cfg: FeatureConfig,
#     rank_norm_test: bool,
#     default_class: str,
# ) -> dict[str, str]:
#     """Decode every submission, optionally global-rank-normalise, then
#     per image: featurize → predict → encode."""
#     common = set.intersection(*[set(s.keys()) for s in submissions])
#     if not common:
#         raise RuntimeError("no test IDs in common across submissions")
#     all_ids = sorted(common)
#     if any(len(s) != len(common) for s in submissions):
#         diffs = [len(s) - len(common) for s in submissions]
#         print(f"  [warn] some methods have extra IDs (diffs = {diffs}); "
#               f"fusing on the shared {len(common)} IDs.")
#
#     M = len(submissions)
#     feature_names = models.get("_feature_names", None)
#
#     print(f"\nDecoding {M} submissions × {len(all_ids)} IDs each...")
#     decoded_per_method: list[dict[str, np.ndarray]] = []
#     t0 = time.time()
#     for mi, sub in enumerate(submissions):
#         d: dict[str, np.ndarray] = {}
#         for j, sid in enumerate(all_ids):
#             d[sid] = q8rle_to_float_matrix(sub[sid])
#             if (j + 1) % 1000 == 0:
#                 print(f"    method {mi + 1}/{M}: {j + 1}/{len(all_ids)}  "
#                       f"({time.time() - t0:.1f}s)", flush=True)
#         decoded_per_method.append(d)
#         print(f"    method {mi + 1}/{M} decoded "
#               f"({time.time() - t0:.1f}s total)")
#
#     if rank_norm_test:
#         print(f"\nGlobal rank-normalisation (per method)...")
#         rank_normalise_test_in_place(decoded_per_method, all_ids)
#
#     print(f"\nFusing {len(all_ids)} images with per-class XGBoost...")
#     fused: dict[str, str] = {}
#     shared_entry = models.get("_SHARED_")
#     t1 = time.time()
#     n_uniform = 0
#
#     # Cache spatial features per (H, W) — test images may vary in shape
#     # (though Spacepresso normalises to 224x224 in the submission step).
#     spatial_cache: dict | None = None
#
#     for i, sid in enumerate(all_ids):
#         cls = (class_map.get(sid) if class_map else None) or default_class
#         entry = models.get(cls)
#         if entry is None or entry.get("_fallback_to_shared"):
#             entry = shared_entry
#
#         mats = [d[sid] for d in decoded_per_method]
#         H, W = mats[0].shape
#
#         if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
#             spatial_cache = _spatial_cache(H, W)
#
#         if entry is None or "model" not in entry:
#             n_uniform += 1
#             fused_mat = np.mean(np.stack(mats, axis=0), axis=0)
#         else:
#             feats, _ = featurize_image(mats, cfg,
#                                           spatial_cache=spatial_cache,
#                                           feature_names=feature_names)
#             X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
#             p = entry["model"].predict_proba(X)[:, 1].astype(np.float32)
#             fused_mat = p.reshape(H, W)
#             del feats, X, p
#
#         fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
#         fused[sid] = float_matrix_to_q8rle(fused_mat)
#         if (i + 1) % 500 == 0:
#             print(f"    fused {i + 1}/{len(all_ids)}  "
#                   f"({time.time() - t1:.1f}s)", flush=True)
#
#     if n_uniform:
#         print(f"  [warn] {n_uniform} images had no model — used uniform avg.")
#     print(f"  fused all {len(all_ids)} in {time.time() - t1:.1f}s")
#     return fused
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # ablation_master row append (same as logreg_stacker)
# # ─────────────────────────────────────────────────────────────────────────────
# def append_to_master(master_csv: Path, row: dict) -> None:
#     existing: list[dict] = []
#     fieldnames: list[str] = []
#     if master_csv.exists():
#         with open(master_csv, "r", newline="", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             fieldnames = list(reader.fieldnames or [])
#             existing = list(reader)
#     for k in row.keys():
#         if k not in fieldnames:
#             fieldnames.append(k)
#     existing.append(row)
#     with open(master_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for r in existing:
#             w.writerow({k: r.get(k, "") for k in fieldnames})
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────
# def main():
#     ap = argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--runs", nargs="+", required=True, type=Path)
#     ap.add_argument("--local-preds", nargs="+", required=True, type=Path)
#     ap.add_argument("--data-root", type=Path,
#                     default=Path("/work/u10813429/anomaly-detection/data"))
#     ap.add_argument("--class-map", type=Path)
#     ap.add_argument("--rank-normalise", action="store_true", default=True)
#     ap.add_argument("--no-rank-normalise", dest="rank_normalise",
#                     action="store_false")
#     ap.add_argument("--neg-per-pos", type=int, default=30)
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--out", type=Path, required=True)
#     ap.add_argument("--master-csv", type=Path,
#                     default=Path("/work/u10813429/anomaly-detection/"
#                                   "baseline_out/ablation_master.csv"))
#     ap.add_argument("--run-tag", default="stacker-xgb")
#     ap.add_argument("--no-zip", action="store_true")
#     # Feature-config toggles (just the most useful)
#     ap.add_argument("--no-window-stats", action="store_true",
#                     help="Disable mean/max/std neighborhood features.")
#     ap.add_argument("--no-gradient", action="store_true")
#     ap.add_argument("--no-laplacian", action="store_true")
#     ap.add_argument("--no-dist-to-hot", action="store_true")
#     ap.add_argument("--no-spatial", action="store_true")
#     ap.add_argument("--no-cross-stats", action="store_true")
#     ap.add_argument("--no-image-aggregates", action="store_true")
#     # XGBoost params (override defaults)
#     ap.add_argument("--n-estimators", type=int)
#     ap.add_argument("--max-depth", type=int)
#     ap.add_argument("--learning-rate", type=float)
#     ap.add_argument("--subsample", type=float)
#     ap.add_argument("--colsample-bytree", type=float)
#     ap.add_argument("--reg-alpha", type=float)
#     ap.add_argument("--reg-lambda", type=float)
#     ap.add_argument("--min-child-weight", type=float)
#     # Optuna
#     ap.add_argument("--tune", action="store_true",
#                     help="Tune XGBoost hyperparameters with Optuna BEFORE "
#                          "the final per-class fit.")
#     ap.add_argument("--n-trials", type=int, default=50)
#     ap.add_argument("--tune-cv", default="loao",
#                     choices=["loao", "loio"],
#                     help="CV scheme for tuning. 'loao' = leave-one-"
#                          "anomaly_type-out (faster, ~47 folds). "
#                          "'loio' = leave-one-image-out (~200 folds, slower).")
#     ap.add_argument("--tune-timeout-min", type=float, default=None,
#                     help="Optional walltime cap on Optuna in minutes.")
#     args = ap.parse_args()
#
#     if not HAS_XGB:
#         raise SystemExit("[FATAL] xgboost not installed. uv pip install xgboost")
#     if len(args.runs) < 2:
#         raise SystemExit("need ≥ 2 methods to stack")
#     if len(args.local_preds) != len(args.runs):
#         raise SystemExit("--local-preds count must match --runs count")
#
#     method_names = [p.parent.name for p in args.runs]
#
#     print("=" * 78)
#     print(f"XGBOOST STACKER — {len(args.runs)} methods")
#     print("=" * 78)
#     for i, (r, lp) in enumerate(zip(args.runs, args.local_preds)):
#         print(f"  method {i}: {method_names[i]}")
#         print(f"            submission : {r}")
#         print(f"            local_preds: {lp}")
#
#     # ── Feature config from flags
#     cfg = FeatureConfig(
#         use_window_mean = not args.no_window_stats,
#         use_window_max  = not args.no_window_stats,
#         use_window_std  = not args.no_window_stats,
#         use_gradient    = not args.no_gradient,
#         use_laplacian   = not args.no_laplacian,
#         use_dist_to_hot = not args.no_dist_to_hot,
#         use_spatial     = not args.no_spatial,
#         use_cross_stats = not args.no_cross_stats,
#         use_image_aggregates = not args.no_image_aggregates,
#     )
#     print(f"\nFeature config: {asdict(cfg)}")
#
#     # ── Load test submissions
#     print("\nLoading test submissions...")
#     subs = []
#     for p in args.runs:
#         s = load_submission(p)
#         print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
#         subs.append(s)
#
#     # ── Load local-val preds
#     print("\nLoading local-val predictions...")
#     preds_per_method = []
#     for p in args.local_preds:
#         d = load_local_preds(p)
#         print(f"  {p.parent.name}/{p.name}: "
#               f"{len(d['ids'])} val images, "
#               f"scores {d['scores'].shape}, "
#               f"{float(d['masks'].mean()) * 100:.3f}% positive")
#         preds_per_method.append(d)
#
#     # ── Align
#     print("\nAligning local-val predictions across methods...")
#     val = align_local_preds(preds_per_method, method_names)
#
#     # ── Rank-norm val side
#     if args.rank_normalise:
#         print("\nRank-normalising local-val scores (global per method)...")
#         rank_normalise_in_place_val(val["scores"])
#
#     # ── Build training data
#     classes = sorted(set(val["classes"].tolist()))
#     print(f"\nBuilding training matrices (featurize + sample negatives)...")
#     print(f"  classes present in val: {classes}")
#     training_data = build_training_data(
#         val, classes, cfg,
#         neg_per_pos=args.neg_per_pos, seed=args.seed)
#     feature_names = training_data.get("_feature_names", [])
#     print(f"  total features per pixel: {len(feature_names)}")
#
#     # ── Build hyperparameter dict (CLI overrides → optional Optuna search)
#     params: dict = {}
#     cli_overrides = {
#         "n_estimators":   args.n_estimators,
#         "max_depth":      args.max_depth,
#         "learning_rate":  args.learning_rate,
#         "subsample":      args.subsample,
#         "colsample_bytree": args.colsample_bytree,
#         "reg_alpha":      args.reg_alpha,
#         "reg_lambda":     args.reg_lambda,
#         "min_child_weight": args.min_child_weight,
#     }
#     for k, v in cli_overrides.items():
#         if v is not None:
#             params[k] = v
#
#     if args.tune:
#         timeout = (args.tune_timeout_min * 60.0
#                    if args.tune_timeout_min else None)
#         tuned = tune_with_optuna(training_data,
#                                   n_trials=args.n_trials,
#                                   seed=args.seed,
#                                   cv_mode=args.tune_cv,
#                                   timeout=timeout)
#         # Optuna params overwrite CLI overrides for keys it tuned;
#         # any keys NOT tuned but set on the CLI still apply.
#         params = {**params, **tuned}
#
#     final_params = {**DEFAULT_XGB_PARAMS, **params}
#     print(f"\nFinal XGBoost params: "
#           f"{ {k: v for k, v in final_params.items() if k not in ('tree_method','max_bin','objective','eval_metric','n_jobs','verbosity')} }")
#
#     # ── Fit per-class
#     print(f"\nFitting per-class XGBoost models...")
#     models = fit_per_class(training_data, params, args.seed)
#
#     # ── Class map for test
#     print("\nBuilding ID → class map for test set...")
#     if args.class_map and args.class_map.exists():
#         class_map: dict[str, str] = {}
#         with open(args.class_map, "r", encoding="utf-8") as f:
#             for row in csv.DictReader(f):
#                 if "ID" in row and "class" in row:
#                     class_map[row["ID"]] = row["class"]
#         print(f"  loaded {len(class_map)} entries from {args.class_map}")
#     else:
#         class_map = build_class_map_from_data(args.data_root)
#         print(f"  built from {args.data_root}: {len(class_map)} entries")
#     default_class = classes[0] if classes else "_default_"
#     if not class_map:
#         print(f"  [warn] no class map — every test image uses SHARED model")
#         class_map = None
#
#     # ── Inference
#     fused = fuse_test(subs, models, class_map, cfg,
#                        rank_norm_test=args.rank_normalise,
#                        default_class=default_class)
#
#     # ── Write submission CSV (+ ZIP)
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
#     # ── Dump model config + feature importances
#     model_dump = {
#         "version": 1,
#         "methods": method_names,
#         "rank_normalise": bool(args.rank_normalise),
#         "neg_per_pos": args.neg_per_pos,
#         "seed": args.seed,
#         "feature_config": asdict(cfg),
#         "feature_names": feature_names,
#         "xgb_params": final_params,
#         "per_class": {},
#     }
#     for k, v in models.items():
#         if k.startswith("_") and k != "_SHARED_":
#             continue
#         if isinstance(v, dict) and "model" in v:
#             model_dump["per_class"][k] = {
#                 "n_pos": v["n_pos"],
#                 "n_neg_sampled": v["n_neg_sampled"],
#                 "logloss": v.get("logloss"),
#                 "train_ap": v.get("train_ap"),
#                 "feature_importance": v.get("feature_importance"),
#             }
#         else:
#             model_dump["per_class"][k] = {"fallback_to_shared": True,
#                                           "n_pos": v.get("n_pos"),
#                                           "n_neg_sampled": v.get("n_neg_sampled")}
#     cfg_path = args.out.parent / "stacker_config.json"
#     with open(cfg_path, "w", encoding="utf-8") as f:
#         json.dump(model_dump, f, indent=2)
#     print(f"Wrote model config + importances -> {cfg_path}")
#
#     # ── Quick summary of top-10 importances (SHARED model)
#     if "_SHARED_" in models and "feature_importance" in models["_SHARED_"]:
#         fi = np.asarray(models["_SHARED_"]["feature_importance"])
#         if feature_names and len(feature_names) == len(fi):
#             top = np.argsort(-fi)[:10]
#             print(f"\nTop-10 feature importances (SHARED model):")
#             for r, j in enumerate(top, 1):
#                 print(f"  {r:>2d}. {feature_names[j]:<22s}  {fi[j]:.4f}")
#
#     # ── ablation_master
#     run_id = "stacker_xgb_" + hashlib.sha1(
#         "|".join(str(p) for p in args.runs).encode("utf-8")
#     ).hexdigest()[:6]
#     row = {
#         "run_id": run_id,
#         "run_tag": args.run_tag,
#         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "backbone": "STACKER_XGB",
#         "feature_layers": "",
#         "input_size": "",
#         "n_classes": len(classes),
#         "AP_overall": "",
#         "runtime_min": "",
#         "submission_path": str(args.out.with_suffix(".zip")),
#         "notes": (f"xgb stacker on {len(subs)} methods "
#                   f"(F={len(feature_names)}, rank_norm={int(args.rank_normalise)}, "
#                   f"neg_per_pos={args.neg_per_pos}, tuned={int(args.tune)})"),
#     }
#     append_to_master(args.master_csv, row)
#     print(f"\nAppended row to {args.master_csv}")
#     print("\nDone.")
#
#
# if __name__ == "__main__":
#     main()

"""XGBoost stacker for Spacepresso anomaly detection — v2 (steps 11 + 12).

Drop-in replacement for v1. Same I/O contract:
    INPUT  : N submission.csv  +  N local_predictions.npz
    OUTPUT : one fused submission.csv (+ .zip) + stacker_config.json
             + oof_predictions.npz + run_log.txt + ablation_master row

# What's new vs v1

  --tune-mode {none, global, per-class}
       none       (default) train with default or CLI-override params
       global     one tuned config shared across all classes (v1 --tune)
       per-class  Optuna runs INDEPENDENTLY per class           [STEP 11]

  --calibrate {none, platt, isotonic}
       none       (default) raw XGB probs go to q8rle
       platt      per-class 1-D logistic on OOF preds            [STEP 12]
       isotonic   per-class monotonic non-parametric on OOF      [STEP 12]

When --calibrate is given AND --tune-mode != per-class, we still run a
one-shot LOAO pass per class to harvest OOF preds. When --tune-mode is
per-class, the OOF preds drop out of the tuning loop for free.
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
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


@contextmanager
def tee_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    old = sys.stdout
    sys.stdout = Tee(old, f)
    try: yield
    finally:
        sys.stdout = old; f.close()


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
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Re-run the baseline with the "
            f"local_preds_saver hook (see local_preds_saver.py).")
    data = np.load(path, allow_pickle=True)
    return {
        "ids":           data["ids"].astype(str),
        "classes":       data["classes"].astype(str),
        "anomaly_types": data["anomaly_types"].astype(str),
        "scores":        data["scores"].astype(np.float32),
        "masks":         data["masks"].astype(np.uint8),
    }


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
# Local-val alignment
# ─────────────────────────────────────────────────────────────────────────────
def _nn_resize_2d(arr: np.ndarray, target_shape, dtype) -> np.ndarray:
    th, tw = target_shape
    h, w = arr.shape
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)


def align_local_preds(preds_per_method: list[dict],
                       method_names: list[str]) -> dict:
    id_sets = [set(p["ids"].tolist()) for p in preds_per_method]
    common = sorted(set.intersection(*id_sets))
    if not common:
        raise RuntimeError("no local-val IDs in common across the methods")
    H0, W0 = preds_per_method[0]["scores"].shape[1:3]
    print(f"  reference shape (method 0): {H0}x{W0}")
    indexed = []
    for p in preds_per_method:
        idx_of = {id_: i for i, id_ in enumerate(p["ids"])}
        indexed.append((p, idx_of))
    N = len(common); M = len(preds_per_method)
    scores = np.empty((N, H0, W0, M), dtype=np.float32)
    masks  = np.empty((N, H0, W0), dtype=np.uint8)
    classes = np.empty(N, dtype=object)
    anomaly_types = np.empty(N, dtype=object)
    for i, id_ in enumerate(common):
        for mi, (p, idx_of) in enumerate(indexed):
            j = idx_of[id_]
            s = p["scores"][j]
            if s.shape != (H0, W0):
                s = _nn_resize_2d(s, (H0, W0), np.float32)
            scores[i, :, :, mi] = s
            if mi == 0:
                classes[i] = str(p["classes"][j])
                anomaly_types[i] = str(p["anomaly_types"][j])
                m = p["masks"][j]
                if m.shape != (H0, W0):
                    m = _nn_resize_2d(m, (H0, W0), np.uint8)
                masks[i] = m
    print(f"  aligned {N} val images × {M} methods @ {H0}x{W0}")
    return {"ids": np.asarray(common),
            "classes": classes.astype(str),
            "anomaly_types": anomaly_types.astype(str),
            "scores": scores,
            "masks": masks}


# ─────────────────────────────────────────────────────────────────────────────
# Rank normalisation (global, per method)
# ─────────────────────────────────────────────────────────────────────────────
def rank_normalise_in_place_val(scores: np.ndarray) -> None:
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


def rank_normalise_test_in_place(decoded_per_method: list[dict[str, np.ndarray]],
                                   all_ids: list[str]) -> None:
    for mi, d in enumerate(decoded_per_method):
        t1 = time.time()
        shapes = {sid: d[sid].shape for sid in all_ids}
        sizes  = {sid: int(np.prod(shapes[sid])) for sid in all_ids}
        total  = sum(sizes.values())
        print(f"    method {mi + 1}/{len(decoded_per_method)}: "
              f"global rank-norm over {total:,} pixels...")
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
            d[sid] = ranks[idx:idx + n].reshape(shapes[sid]).astype(np.float32)
            idx += n
        del ranks
        print(f"      done in {time.time() - t1:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Feature configuration + helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    use_raw_score: bool = True
    use_per_image_rank: bool = True
    gauss_sigmas: tuple[float, ...] = (1.0, 3.0, 7.0)
    window_sizes: tuple[int, ...] = (3, 7, 15)
    use_window_mean: bool = True
    use_window_max: bool = True
    use_window_std: bool = True
    use_gradient: bool = True
    use_laplacian: bool = True
    use_dist_to_hot: bool = True
    hot_percentile: float = 99.0
    use_image_aggregates: bool = True
    use_cross_stats: bool = True
    use_spatial: bool = True


def _per_image_rank(s: np.ndarray) -> np.ndarray:
    flat = s.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty_like(flat, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks.reshape(s.shape)


def _window_mean(s, size):
    return ndi.uniform_filter(s, size=size, mode="reflect").astype(np.float32)


def _window_max(s, size):
    return ndi.maximum_filter(s, size=size, mode="reflect").astype(np.float32)


def _window_std(s, size):
    mean = ndi.uniform_filter(s, size=size, mode="reflect")
    sq = ndi.uniform_filter(s * s, size=size, mode="reflect")
    var = np.clip(sq - mean * mean, 0.0, None)
    return np.sqrt(var).astype(np.float32)


def _gauss(s, sigma):
    return ndi.gaussian_filter(s, sigma=sigma, mode="reflect").astype(np.float32)


def _grad_mag(s):
    sx = ndi.sobel(s, axis=0, mode="reflect")
    sy = ndi.sobel(s, axis=1, mode="reflect")
    return np.sqrt(sx * sx + sy * sy).astype(np.float32)


def _laplacian(s):
    return ndi.laplace(s, mode="reflect").astype(np.float32)


def _dist_to_hot(s, pct):
    thresh = float(np.percentile(s, pct))
    hot = s >= thresh
    if not hot.any():
        H, W = s.shape
        return np.full(s.shape, float(np.hypot(H, W)), dtype=np.float32)
    return ndi.distance_transform_edt(~hot).astype(np.float32)


def _spatial_cache(H: int, W: int) -> dict:
    ys, xs = np.indices((H, W)).astype(np.float32)
    xs_n = xs / max(W - 1, 1)
    ys_n = ys / max(H - 1, 1)
    de = np.minimum(np.minimum(xs_n, ys_n),
                     np.minimum(1.0 - xs_n, 1.0 - ys_n)).astype(np.float32)
    dc = np.sqrt((xs_n - 0.5) ** 2 + (ys_n - 0.5) ** 2).astype(np.float32)
    return {"x": xs_n, "y": ys_n, "dist_edge": de, "dist_center": dc,
            "_shape": (H, W)}


def featurize_image(
    scores_per_method: list[np.ndarray],
    cfg: FeatureConfig,
    spatial_cache: dict | None = None,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    assert scores_per_method
    H, W = scores_per_method[0].shape
    M = len(scores_per_method)
    layers: list[np.ndarray] = []
    names: list[str] = []
    for mi, s in enumerate(scores_per_method):
        if s.shape != (H, W):
            raise ValueError(f"method {mi} shape {s.shape} != ({H}, {W})")
        s = np.asarray(s, dtype=np.float32)
        if cfg.use_raw_score:
            layers.append(s); names.append(f"m{mi}_raw")
        if cfg.use_per_image_rank:
            layers.append(_per_image_rank(s)); names.append(f"m{mi}_rank")
        for sigma in cfg.gauss_sigmas:
            layers.append(_gauss(s, sigma)); names.append(f"m{mi}_g{sigma:g}")
        for ws in cfg.window_sizes:
            if cfg.use_window_mean:
                layers.append(_window_mean(s, ws)); names.append(f"m{mi}_mean{ws}")
            if cfg.use_window_max:
                layers.append(_window_max(s, ws));  names.append(f"m{mi}_max{ws}")
            if cfg.use_window_std:
                layers.append(_window_std(s, ws));  names.append(f"m{mi}_std{ws}")
        if cfg.use_gradient:
            layers.append(_grad_mag(s));  names.append(f"m{mi}_grad")
        if cfg.use_laplacian:
            layers.append(_laplacian(s)); names.append(f"m{mi}_lap")
        if cfg.use_dist_to_hot:
            layers.append(_dist_to_hot(s, cfg.hot_percentile))
            names.append(f"m{mi}_d2hot")
        if cfg.use_image_aggregates:
            for stat_name, val in (("imgmax", float(s.max())),
                                     ("imgp99", float(np.percentile(s, 99))),
                                     ("imgmean", float(s.mean())),
                                     ("imgstd", float(s.std()))):
                layers.append(np.full((H, W), val, dtype=np.float32))
                names.append(f"m{mi}_{stat_name}")
    if cfg.use_cross_stats and M >= 2:
        stack = np.stack(scores_per_method, axis=0).astype(np.float32)
        cmean = stack.mean(axis=0); layers.append(cmean); names.append("x_mean")
        cmax  = stack.max(axis=0);  layers.append(cmax);  names.append("x_max")
        cmin  = stack.min(axis=0);  layers.append(cmin);  names.append("x_min")
        cstd  = stack.std(axis=0);  layers.append(cstd);  names.append("x_std")
        layers.append((cmax - cmin).astype(np.float32))
        names.append("x_range")
    if cfg.use_spatial:
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        for k in ("x", "y", "dist_edge", "dist_center"):
            layers.append(spatial_cache[k]); names.append(f"s_{k}")
    feats = np.stack(layers, axis=-1).astype(np.float32, copy=False)
    if feature_names is not None and names != feature_names:
        raise RuntimeError(
            f"feature drift between fit and predict: "
            f"expected {len(feature_names)} cols, got {len(names)}")
    return feats, names


# ─────────────────────────────────────────────────────────────────────────────
# Build training matrices per class
# ─────────────────────────────────────────────────────────────────────────────
def build_training_data(val: dict, classes: list[str], cfg: FeatureConfig,
                         *, neg_per_pos: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]; val_anom_types = val["anomaly_types"]
    N, H, W, M = val_scores.shape
    spatial = _spatial_cache(H, W)
    out: dict = {}
    feature_names_global: list[str] | None = None
    for cls in classes:
        cls_idx = np.flatnonzero(val_classes == cls)
        if cls_idx.size == 0:
            print(f"  [warn] class {cls}: no val images; skipping")
            continue
        per_img_X, per_img_y, per_img_atypes = [], [], []
        per_img_pixrange: list[tuple[int, int]] = []
        cursor = 0
        for i in cls_idx:
            mat_list = [val_scores[i, :, :, mi] for mi in range(M)]
            feats, names = featurize_image(mat_list, cfg, spatial_cache=spatial)
            if feature_names_global is None:
                feature_names_global = names
            F = feats.shape[-1]
            flat_x = feats.reshape(-1, F)
            flat_y = val_masks[i].ravel().astype(np.int32)
            per_img_X.append(flat_x); per_img_y.append(flat_y)
            per_img_atypes.append(str(val_anom_types[i]))
            per_img_pixrange.append((cursor, cursor + flat_x.shape[0]))
            cursor += flat_x.shape[0]
            del feats
        X_full = np.concatenate(per_img_X, axis=0).astype(np.float32)
        y_full = np.concatenate(per_img_y, axis=0).astype(np.int32)
        del per_img_X, per_img_y
        n_pos = int((y_full == 1).sum())
        n_neg = int((y_full == 0).sum())
        if n_pos == 0:
            print(f"  [warn] class {cls}: zero positive pixels in val")
            out[cls] = {"_fallback_to_shared": True,
                        "n_pos": 0, "n_neg_sampled": 0,
                        "X_full": X_full, "y_full": y_full,
                        "anomaly_types": per_img_atypes,
                        "img_pixranges": per_img_pixrange}
            continue
        target_neg = min(n_neg, n_pos * neg_per_pos)
        neg_idx = np.flatnonzero(y_full == 0)
        sample_neg = (rng.choice(neg_idx, size=target_neg, replace=False)
                       if target_neg < n_neg else neg_idx)
        pos_idx = np.flatnonzero(y_full == 1)
        keep = np.concatenate([pos_idx, sample_neg])
        out[cls] = {
            "X_train": X_full[keep], "y_train": y_full[keep],
            "X_full":  X_full,       "y_full":  y_full,
            "anomaly_types": per_img_atypes,
            "img_pixranges": per_img_pixrange,
            "n_pos": n_pos, "n_neg_sampled": int(len(sample_neg)),
        }
        print(f"  class {cls}: train {len(keep):>8d} rows "
              f"(pos={n_pos:>6d}, neg={len(sample_neg):>8d}, feats={X_full.shape[1]:>3d})")
    out["_feature_names"] = feature_names_global or []
    return out


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost params
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.07,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.0,
    "reg_lambda":       1.0,
    "min_child_weight": 1.0,
    "tree_method":      "hist",
    "max_bin":          256,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "n_jobs":           -1,
    "verbosity":        0,
}


def _make_xgb(params: dict, seed: int) -> "xgb.XGBClassifier":
    if not HAS_XGB:
        raise SystemExit("xgboost not installed. uv pip install xgboost")
    p = dict(DEFAULT_XGB_PARAMS)
    p.update(params or {})
    return xgb.XGBClassifier(random_state=seed, **p)


def get_params_for_class(cls: str, params_global: dict,
                          params_per_class: dict | None) -> dict:
    if params_per_class and cls in params_per_class:
        return params_per_class[cls]
    return params_global


# ─────────────────────────────────────────────────────────────────────────────
# Per-class LOAO/LOIO helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bucketize_one_class(td: dict, mode: str) -> dict[object, list[int]]:
    atypes = td["anomaly_types"]
    n_imgs = len(td["img_pixranges"])
    if mode == "loao":
        key = lambda i: atypes[i]
    elif mode == "loio":
        key = lambda i: i
    else:
        raise ValueError(f"unknown cv_mode: {mode}")
    buckets: dict = defaultdict(list)
    for i in range(n_imgs):
        buckets[key(i)].append(i)
    return buckets


def cv_score_one_class(td: dict, params: dict, seed: int,
                        mode: str = "loao",
                        neg_per_pos: int = 30) -> float:
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    X_full = td["X_full"]; y_full = td["y_full"]
    ranges = td["img_pixranges"]
    buckets = _bucketize_one_class(td, mode)
    fold_aps: list[float] = []
    for held_imgs in buckets.values():
        held_set = set(held_imgs)
        train_imgs = [i for i in range(len(ranges)) if i not in held_set]
        Xs, ys = [], []
        for ti in train_imgs:
            s, e = ranges[ti]
            yp = y_full[s:e]
            pos = np.flatnonzero(yp == 1)
            if pos.size == 0:
                continue
            neg = np.flatnonzero(yp == 0)
            n_keep = min(neg.size, pos.size * neg_per_pos)
            sneg = (rng.choice(neg, n_keep, replace=False)
                    if n_keep < neg.size else neg)
            keep = np.concatenate([pos, sneg])
            Xs.append(X_full[s:e][keep]); ys.append(yp[keep])
        if not Xs:
            continue
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0),
                np.concatenate(ys, axis=0))
        for hi in held_imgs:
            s, e = ranges[hi]
            yp = y_full[s:e]
            if int(yp.sum()) == 0:
                continue
            p = clf.predict_proba(X_full[s:e])[:, 1]
            try:
                fold_aps.append(float(average_precision_score(yp, p)))
            except Exception:
                pass
    return float(np.mean(fold_aps)) if fold_aps else 0.0


def loao_oof_one_class(td: dict, params: dict, seed: int,
                        cv_mode: str = "loao",
                        neg_per_pos: int = 30
                        ) -> tuple[np.ndarray, np.ndarray]:
    """LOAO/LOIO with `params`; return OOF (preds, labels) over ALL pixels
    of held images (no subsampling on the held side — calibrator must see
    the true class prior)."""
    rng = np.random.default_rng(seed)
    X_full = td["X_full"]; y_full = td["y_full"]
    ranges = td["img_pixranges"]
    buckets = _bucketize_one_class(td, cv_mode)
    oof_preds = np.full(y_full.shape, np.nan, dtype=np.float32)
    for held_imgs in buckets.values():
        held_set = set(held_imgs)
        train_imgs = [i for i in range(len(ranges)) if i not in held_set]
        Xs, ys = [], []
        for ti in train_imgs:
            s, e = ranges[ti]
            yp = y_full[s:e]
            pos = np.flatnonzero(yp == 1)
            if pos.size == 0:
                continue
            neg = np.flatnonzero(yp == 0)
            n_keep = min(neg.size, pos.size * neg_per_pos)
            sneg = (rng.choice(neg, n_keep, replace=False)
                    if n_keep < neg.size else neg)
            keep = np.concatenate([pos, sneg])
            Xs.append(X_full[s:e][keep]); ys.append(yp[keep])
        if not Xs:
            continue
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0),
                np.concatenate(ys, axis=0))
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            oof_preds[s:e] = p
    mask = ~np.isnan(oof_preds)
    return oof_preds[mask].astype(np.float32), y_full[mask].astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Step 11 — global + per-class tuning
# ─────────────────────────────────────────────────────────────────────────────
def _optuna_suggest(trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha",  1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
    }


def tune_global(training_data: dict, *, n_trials: int, seed: int,
                cv_mode: str = "loao",
                timeout: float | None = None) -> dict:
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed. uv pip install optuna")
    classes = [c for c in training_data if not c.startswith("_")
                and not training_data[c].get("_fallback_to_shared")]

    def objective(trial):
        params = _optuna_suggest(trial)
        aps = [cv_score_one_class(training_data[cls], params, seed, mode=cv_mode)
                for cls in classes]
        return float(np.mean(aps)) if aps else 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize",
                                  sampler=sampler, pruner=pruner)
    print(f"\n>>> Global Optuna tuning: {n_trials} trials, CV={cv_mode}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f">>> best mean-AP: {study.best_value:.4f}")
    print(f">>> best params: {json.dumps(study.best_params, indent=2)}")
    return study.best_params


def tune_per_class(training_data: dict, *, n_trials: int, seed: int,
                    cv_mode: str = "loao",
                    timeout_per_class: float | None = None) -> dict:
    """[STEP 11] Independent Optuna study per class. Also returns OOF
    preds + labels (collected with each class's best params) so step 12
    calibration gets them for free."""
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed. uv pip install optuna")
    out: dict = {}
    for cls, td in training_data.items():
        if cls.startswith("_"):
            continue
        if td.get("_fallback_to_shared"):
            print(f"\n  class {cls}: fallback-to-shared, no per-class tuning")
            continue
        print(f"\n>>> Per-class tuning: {cls}  (CV={cv_mode}, "
              f"n_trials={n_trials}, timeout_per_class={timeout_per_class}s)")

        def objective(trial, _td=td):    # bind td via default arg
            return cv_score_one_class(_td, _optuna_suggest(trial),
                                        seed, mode=cv_mode)

        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize",
                                      sampler=sampler, pruner=pruner,
                                      study_name=f"xgb_{cls}")
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials,
                        timeout=timeout_per_class)
        elapsed = time.time() - t0
        print(f"  {cls}: best CV-AP = {study.best_value:.4f}  "
              f"({elapsed:.1f}s, {len(study.trials)} trials)")
        print(f"  {cls}: best params = {study.best_params}")
        oof_preds, oof_labels = loao_oof_one_class(
            td, study.best_params, seed, cv_mode=cv_mode)
        out[cls] = {
            "best_params": study.best_params,
            "best_cv_ap": float(study.best_value),
            "n_trials": len(study.trials),
            "oof_preds": oof_preds,
            "oof_labels": oof_labels,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 12 — per-class calibration
# ─────────────────────────────────────────────────────────────────────────────
def fit_calibrator(method: str, oof_preds: np.ndarray,
                    oof_labels: np.ndarray) -> dict:
    if method == "none":
        return {"method": "none"}
    if oof_preds.size == 0 or int(oof_labels.sum()) == 0:
        print(f"    [warn] no positives in OOF; falling back to no-op calibrator")
        return {"method": "none"}

    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(solver="lbfgs", max_iter=500)
        clf.fit(oof_preds.reshape(-1, 1), oof_labels)
        return {"method": "platt",
                "coef": float(clf.coef_[0][0]),
                "intercept": float(clf.intercept_[0])}

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        # Subsample to 1M while keeping ALL positives.
        n_max = 1_000_000
        if oof_preds.size > n_max:
            pos_idx = np.flatnonzero(oof_labels == 1)
            neg_idx = np.flatnonzero(oof_labels == 0)
            target_neg = max(n_max - pos_idx.size, 0)
            rng = np.random.default_rng(0)
            sneg = (rng.choice(neg_idx, size=target_neg, replace=False)
                    if target_neg < neg_idx.size else neg_idx)
            sel = np.concatenate([pos_idx, sneg])
            oof_preds = oof_preds[sel]; oof_labels = oof_labels[sel]
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(oof_preds, oof_labels.astype(np.float32))
        return {"method": "isotonic",
                "X_thresholds": [float(v) for v in ir.X_thresholds_],
                "y_thresholds": [float(v) for v in ir.y_thresholds_]}

    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(cal: dict, scores: np.ndarray) -> np.ndarray:
    m = cal.get("method", "none")
    if m == "none":
        return scores.astype(np.float32)
    if m == "platt":
        z = cal["coef"] * scores + cal["intercept"]
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    if m == "isotonic":
        xs = np.asarray(cal["X_thresholds"], dtype=np.float32)
        ys = np.asarray(cal["y_thresholds"], dtype=np.float32)
        return np.interp(scores, xs, ys).astype(np.float32)
    raise ValueError(m)


# ─────────────────────────────────────────────────────────────────────────────
# Fit final per-class production models
# ─────────────────────────────────────────────────────────────────────────────
def fit_per_class(training_data: dict, params_global: dict,
                   params_per_class: dict | None, seed: int,
                   min_pos_for_per_class: int = 200) -> dict:
    from sklearn.metrics import log_loss, average_precision_score
    feature_names = training_data.get("_feature_names", [])
    out: dict = {"_feature_names": feature_names}
    shared_X, shared_y = [], []
    for cls, td in training_data.items():
        if cls.startswith("_"):
            continue
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
        clf = _make_xgb(params, seed)
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        try: ll = float(log_loss(y, p, labels=[0, 1]))
        except Exception: ll = float("nan")
        try: ap = float(average_precision_score(y, p))
        except Exception: ap = float("nan")
        out[cls] = {"model": clf,
                    "n_pos": td["n_pos"],
                    "n_neg_sampled": td["n_neg_sampled"],
                    "logloss": ll, "train_ap": ap,
                    "params": params,
                    "feature_importance": clf.feature_importances_.tolist()}
        print(f"  class {cls}: n_pos={td['n_pos']:>6d}  "
              f"n_neg={td['n_neg_sampled']:>8d}  "
              f"train_logloss={ll:.4f}  train_ap={ap:.3f}")
    if shared_X:
        X_all = np.concatenate(shared_X, axis=0)
        y_all = np.concatenate(shared_y, axis=0)
        clf_sh = _make_xgb(params_global, seed)
        clf_sh.fit(X_all, y_all)
        p_sh = clf_sh.predict_proba(X_all)[:, 1]
        try: ll_sh = float(log_loss(y_all, p_sh, labels=[0, 1]))
        except Exception: ll_sh = float("nan")
        try: ap_sh = float(average_precision_score(y_all, p_sh))
        except Exception: ap_sh = float("nan")
        out["_SHARED_"] = {"model": clf_sh,
                            "n_pos": int(y_all.sum()),
                            "n_neg_sampled": int(len(y_all) - y_all.sum()),
                            "logloss": ll_sh, "train_ap": ap_sh,
                            "params": params_global,
                            "feature_importance":
                                clf_sh.feature_importances_.tolist()}
        print(f"  SHARED:    n_pos={int(y_all.sum()):>6d}  "
              f"n_neg={int(len(y_all) - y_all.sum()):>8d}  "
              f"train_logloss={ll_sh:.4f}  train_ap={ap_sh:.3f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inference: decode → featurize → predict → calibrate → q8rle
# ─────────────────────────────────────────────────────────────────────────────
def fuse_test(submissions: list[dict[str, str]],
               models: dict,
               class_map: dict[str, str] | None,
               cfg: FeatureConfig,
               rank_norm_test: bool,
               default_class: str,
               calibrators_per_class: dict | None = None) -> dict[str, str]:
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common:
        raise RuntimeError("no test IDs in common across submissions")
    all_ids = sorted(common)
    M = len(submissions)
    feature_names = models.get("_feature_names", None)

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
        print(f"    method {mi + 1}/{M} decoded ({time.time() - t0:.1f}s)")

    if rank_norm_test:
        print(f"\nGlobal rank-normalisation (per method)...")
        rank_normalise_test_in_place(decoded_per_method, all_ids)

    print(f"\nFusing {len(all_ids)} images with per-class XGBoost"
          f"{' + calibration' if calibrators_per_class else ''}...")
    fused: dict[str, str] = {}
    shared_entry = models.get("_SHARED_")
    t1 = time.time()
    n_uniform = 0
    spatial_cache: dict | None = None

    for i, sid in enumerate(all_ids):
        cls = (class_map.get(sid) if class_map else None) or default_class
        entry = models.get(cls)
        if entry is None or entry.get("_fallback_to_shared"):
            entry = shared_entry
        mats = [d[sid] for d in decoded_per_method]
        H, W = mats[0].shape
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        if entry is None or "model" not in entry:
            n_uniform += 1
            fused_mat = np.mean(np.stack(mats, axis=0), axis=0)
        else:
            feats, _ = featurize_image(mats, cfg, spatial_cache=spatial_cache,
                                         feature_names=feature_names)
            X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
            p_raw = entry["model"].predict_proba(X)[:, 1].astype(np.float32)
            cal = (calibrators_per_class.get(cls)
                   if calibrators_per_class else None)
            p_out = apply_calibrator(cal, p_raw) if cal is not None else p_raw
            fused_mat = p_out.reshape(H, W)
            del feats, X, p_raw
        fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
        fused[sid] = float_matrix_to_q8rle(fused_mat)
        if (i + 1) % 500 == 0:
            print(f"    fused {i + 1}/{len(all_ids)}  "
                  f"({time.time() - t1:.1f}s)", flush=True)
    if n_uniform:
        print(f"  [warn] {n_uniform} images had no model — uniform avg")
    print(f"  fused all {len(all_ids)} in {time.time() - t1:.1f}s")
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# ablation_master append
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
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--local-preds", nargs="+", required=True, type=Path)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/data"))
    ap.add_argument("--class-map", type=Path)
    ap.add_argument("--rank-normalise", action="store_true", default=True)
    ap.add_argument("--no-rank-normalise", dest="rank_normalise",
                    action="store_false")
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="stacker-xgb")
    ap.add_argument("--no-zip", action="store_true")
    # Feature toggles
    ap.add_argument("--no-window-stats", action="store_true")
    ap.add_argument("--no-gradient", action="store_true")
    ap.add_argument("--no-laplacian", action="store_true")
    ap.add_argument("--no-dist-to-hot", action="store_true")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-cross-stats", action="store_true")
    ap.add_argument("--no-image-aggregates", action="store_true")
    # XGB param overrides
    ap.add_argument("--n-estimators", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--subsample", type=float)
    ap.add_argument("--colsample-bytree", type=float)
    ap.add_argument("--reg-alpha", type=float)
    ap.add_argument("--reg-lambda", type=float)
    ap.add_argument("--min-child-weight", type=float)
    # Step 11
    ap.add_argument("--tune-mode", default="none",
                    choices=["none", "global", "per-class"])
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--tune-cv", default="loao", choices=["loao", "loio"])
    ap.add_argument("--tune-timeout-min", type=float, default=None,
                    help="Per-class walltime cap when tune-mode=per-class. "
                         "Total cap when tune-mode=global.")
    # Step 12
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "platt", "isotonic"])
    args = ap.parse_args()

    if not HAS_XGB:
        raise SystemExit("[FATAL] xgboost not installed. uv pip install xgboost")
    if len(args.runs) < 2:
        raise SystemExit("need ≥ 2 methods to stack")
    if len(args.local_preds) != len(args.runs):
        raise SystemExit("--local-preds count must match --runs count")
    if args.tune_mode != "none" and not HAS_OPTUNA:
        raise SystemExit("optuna not installed. uv pip install optuna")

    method_names = [p.parent.name for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.out.parent

    with tee_to(run_dir / "run_log.txt"):
        print("=" * 78)
        print(f"XGBOOST STACKER v2 — {len(args.runs)} methods  "
              f"[tune={args.tune_mode}  calibrate={args.calibrate}]")
        print("=" * 78)
        for i, (r, lp) in enumerate(zip(args.runs, args.local_preds)):
            print(f"  method {i}: {method_names[i]}")
            print(f"            submission : {r}")
            print(f"            local_preds: {lp}")

        cfg = FeatureConfig(
            use_window_mean = not args.no_window_stats,
            use_window_max  = not args.no_window_stats,
            use_window_std  = not args.no_window_stats,
            use_gradient    = not args.no_gradient,
            use_laplacian   = not args.no_laplacian,
            use_dist_to_hot = not args.no_dist_to_hot,
            use_spatial     = not args.no_spatial,
            use_cross_stats = not args.no_cross_stats,
            use_image_aggregates = not args.no_image_aggregates,
        )
        print(f"\nFeature config: {asdict(cfg)}")

        print("\nLoading test submissions...")
        subs = [load_submission(p) for p in args.runs]
        for p, s in zip(args.runs, subs):
            print(f"  {p.parent.name}/{p.name}: {len(s)} rows")

        print("\nLoading local-val predictions...")
        preds_per_method = []
        for p in args.local_preds:
            d = load_local_preds(p)
            print(f"  {p.parent.name}/{p.name}: {len(d['ids'])} val images, "
                  f"{float(d['masks'].mean()) * 100:.3f}% positive pixels")
            preds_per_method.append(d)

        print("\nAligning local-val predictions across methods...")
        val = align_local_preds(preds_per_method, method_names)
        if args.rank_normalise:
            print("\nRank-normalising local-val scores (global per method)...")
            rank_normalise_in_place_val(val["scores"])

        classes = sorted(set(val["classes"].tolist()))
        print(f"\nBuilding training matrices (featurize + sample negatives)...")
        print(f"  classes present in val: {classes}")
        training_data = build_training_data(
            val, classes, cfg,
            neg_per_pos=args.neg_per_pos, seed=args.seed)
        feature_names = training_data.get("_feature_names", [])
        print(f"  total features per pixel: {len(feature_names)}")

        # ── Build params (global base + optional Optuna)
        cli_overrides = {k: v for k, v in {
            "n_estimators":     args.n_estimators,
            "max_depth":        args.max_depth,
            "learning_rate":    args.learning_rate,
            "subsample":        args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha":        args.reg_alpha,
            "reg_lambda":       args.reg_lambda,
            "min_child_weight": args.min_child_weight,
        }.items() if v is not None}
        params_global = {**DEFAULT_XGB_PARAMS, **cli_overrides}
        params_per_class: dict | None = None
        tune_per_class_results: dict | None = None
        timeout_sec = (args.tune_timeout_min * 60.0
                       if args.tune_timeout_min else None)

        if args.tune_mode == "global":
            tuned = tune_global(training_data, n_trials=args.n_trials,
                                  seed=args.seed, cv_mode=args.tune_cv,
                                  timeout=timeout_sec)
            params_global = {**params_global, **tuned}
        elif args.tune_mode == "per-class":
            tune_per_class_results = tune_per_class(
                training_data, n_trials=args.n_trials,
                seed=args.seed, cv_mode=args.tune_cv,
                timeout_per_class=timeout_sec)
            params_per_class = {cls: r["best_params"]
                                for cls, r in tune_per_class_results.items()}

        print(f"\nFinal params (global): "
              f"{ {k: v for k, v in params_global.items() if k not in ('tree_method','max_bin','objective','eval_metric','n_jobs','verbosity')} }")
        if params_per_class:
            print(f"Per-class params present for: {sorted(params_per_class.keys())}")

        # ── Step 12: collect OOF preds + fit calibrators
        calibrators_per_class: dict | None = None
        oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if args.calibrate != "none":
            print(f"\nFitting per-class {args.calibrate} calibrators...")
            from sklearn.metrics import average_precision_score
            if tune_per_class_results is not None:
                for cls, r in tune_per_class_results.items():
                    oof_per_class[cls] = (r["oof_preds"], r["oof_labels"])
                print(f"  reusing OOF preds from per-class tuning (free)")
            else:
                print(f"  running LOAO once per class to collect OOF preds...")
                for cls, td in training_data.items():
                    if cls.startswith("_") or td.get("_fallback_to_shared"):
                        continue
                    params = get_params_for_class(cls, params_global,
                                                    params_per_class)
                    t0 = time.time()
                    oof_preds, oof_labels = loao_oof_one_class(
                        td, params, args.seed, cv_mode=args.tune_cv,
                        neg_per_pos=args.neg_per_pos)
                    print(f"    {cls}: {len(oof_preds):>9d} OOF preds  "
                          f"({time.time() - t0:.1f}s)")
                    oof_per_class[cls] = (oof_preds, oof_labels)

            calibrators_per_class = {}
            for cls, (op, ol) in oof_per_class.items():
                cal = fit_calibrator(args.calibrate, op, ol)
                calibrators_per_class[cls] = cal
                try: ap_pre = float(average_precision_score(ol, op))
                except Exception: ap_pre = float("nan")
                p_cal = apply_calibrator(cal, op)
                try: ap_post = float(average_precision_score(ol, p_cal))
                except Exception: ap_post = float("nan")
                # Monotonic — equal expected; benefit is cross-class scale
                # alignment at the GLOBAL pixel-AP ranking step.
                print(f"  {cls}: OOF AP pre={ap_pre:.4f} post={ap_post:.4f}  "
                      f"(monotonic; benefit appears at test-time global ranking)")

        # ── Fit final per-class production models
        print(f"\nFitting final per-class XGBoost models...")
        models = fit_per_class(training_data, params_global,
                                params_per_class, args.seed)

        # ── Class map for test
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
            print(f"  [warn] no class map — every test image uses SHARED model")
            class_map = None

        # ── Inference
        fused = fuse_test(subs, models, class_map, cfg,
                           rank_norm_test=args.rank_normalise,
                           default_class=default_class,
                           calibrators_per_class=calibrators_per_class)

        # ── Write submission CSV (+ ZIP)
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

        # ── Save OOF preds
        if oof_per_class:
            oof_path = run_dir / "oof_predictions.npz"
            to_save = {"classes": np.array(list(oof_per_class.keys()),
                                            dtype=object)}
            for cls, (op, ol) in oof_per_class.items():
                to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
                to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
            np.savez_compressed(oof_path, **to_save)
            print(f"Saved OOF preds -> {oof_path}")

        # ── Dump stacker_config.json
        model_dump = {
            "version": 2,
            "methods": method_names,
            "rank_normalise": bool(args.rank_normalise),
            "neg_per_pos": args.neg_per_pos,
            "seed": args.seed,
            "feature_config": asdict(cfg),
            "feature_names": feature_names,
            "xgb_params_global": params_global,
            "xgb_params_per_class": params_per_class,
            "tune_mode": args.tune_mode,
            "tune_cv": args.tune_cv,
            "tune_n_trials": args.n_trials if args.tune_mode != "none" else 0,
            "tune_timeout_min": args.tune_timeout_min,
            "tune_per_class_best_ap": (
                {cls: r["best_cv_ap"]
                 for cls, r in (tune_per_class_results or {}).items()}
                if tune_per_class_results else None),
            "calibration_method": args.calibrate,
            "calibrators_per_class": calibrators_per_class,
            "per_class_models": {},
        }
        for k, v in models.items():
            if k.startswith("_") and k != "_SHARED_":
                continue
            if isinstance(v, dict) and "model" in v:
                model_dump["per_class_models"][k] = {
                    "n_pos": v["n_pos"],
                    "n_neg_sampled": v["n_neg_sampled"],
                    "train_logloss": v.get("logloss"),
                    "train_ap": v.get("train_ap"),
                    "params": v.get("params"),
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
            json.dump(model_dump, f, indent=2)
        print(f"Wrote stacker config -> {cfg_path}")

        # ── Top-10 importances (SHARED)
        if "_SHARED_" in models and "feature_importance" in models["_SHARED_"]:
            fi = np.asarray(models["_SHARED_"]["feature_importance"])
            if feature_names and len(feature_names) == len(fi):
                top = np.argsort(-fi)[:10]
                print(f"\nTop-10 feature importances (SHARED model):")
                for r, j in enumerate(top, 1):
                    print(f"  {r:>2d}. {feature_names[j]:<22s}  {fi[j]:.4f}")

        # ── ablation_master row
        run_id = "stacker_xgb_" + hashlib.sha1(
            "|".join(str(p) for p in args.runs).encode("utf-8")
        ).hexdigest()[:6]
        notes = (f"xgb v2 | M={len(subs)} | F={len(feature_names)} | "
                  f"rank_norm={int(args.rank_normalise)} | "
                  f"tune={args.tune_mode} | calibrate={args.calibrate}")
        row = {
            "run_id": run_id, "run_tag": args.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "STACKER_XGB_V2",
            "feature_layers": "", "input_size": "",
            "n_classes": len(classes),
            "AP_overall": "", "runtime_min": "",
            "submission_path": str(args.out.with_suffix(".zip")),
            "notes": notes,
        }
        append_to_master(args.master_csv, row)
        print(f"\nAppended row to {args.master_csv}")
        print("\nDone.")


if __name__ == "__main__":
    main()