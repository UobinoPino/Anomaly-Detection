"""XGBoost stacker v6 — memory-efficient + cached successor to v5.

Same I/O contract as v5: N submission.csv + N local_predictions.npz
→ fused submission + stacker_config.json + oof_predictions.npz + run_log.txt
+ ablation row. Defaults match v5 except for internal memory layout
(no LB impact beyond <1e-4 quantization noise on uint16 storage).

==============================================================================
# What changed vs v5 (memory hotspots, in priority order)
==============================================================================

# M1. ON-THE-FLY TEST CROSS-VIEW AGGREGATES   (was the biggest leak)
   v5 precomputed xv_per_id: 4 stats × M × N_test × H × W float32, all held
   simultaneously. At H=W=256, M=12, N=5910 that's ~74 GB just for this
   dict. v6 deletes the precompute. Inside fuse_test, we group test IDs by
   (class, sample_id) and compute the V×M×4 small XV maps on the fly for
   each group. Memory: O(V × M × H × W) per group instead of O(N × …).

# M2. COMPACT INTEGER STORAGE FOR TEST PREDICTIONS
   v5 stored decoded test scores as float32. q8rle is uint8-precision
   already, so float32 is 4× larger than needed. v6 stores them as uint8
   straight out of the codec, then as uint16 after rank-norm (65 536 levels;
   smaller than float32 but finer than max_bin=256 in XGB-hist). Conversion
   to float32 happens lazily inside featurize_image. Saves ~50% of test-
   predictions RAM vs float32.

# M3. FLOAT16 VAL XV AGGREGATES
   Val xv_max/mean/std/lonely now stored as float16 (halved). Featurize
   upcasts on demand.

# M4. PER-CLASS TRAINING-DATA LIFECYCLE
   After a class's OOF preds are collected AND production model is fit,
   X_full / y_full / X_train / y_train for that class are dropped and gc-
   collected. Peak training-data RAM drops from sum-over-classes to max-
   over-classes.

# M5. FILE-BASED CACHE FOR EXPENSIVE PREPROCESSING (--cache-dir)
   Caches: aligned val (after small-CC), decoded test (uint8 pre-rank-norm),
   rank-normed test (uint16). Keys hashed from file mtimes + params.

# M6. EXPLICIT GC AT PHASE TRANSITIONS
   Manual gc.collect() at end-of-val-build, after fusion, after fit_per_class.

==============================================================================
# Precision notes
==============================================================================
uint16 quantization is 1/65535 ≈ 1.5e-5; XGB-hist with max_bin=256 merges
values finer than 1/256 anyway, so AP is unaffected. Per-image rank features
are recomputed from the uint16→float32 upcast inside featurize_image, at
65k-distinct-level precision per image. If you want bit-for-bit parity with
v5, pass --legacy-float32-storage at the cost of the memory gains.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
import re
import sys
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
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
        for st in self.streams:
            st.write(s); st.flush()
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
        sys.stdout = old
        f.close()


def hr(t, c="="): print(f"\n{c * 78}\n  {t}\n{c * 78}")
def sub(t): print(f"\n--- {t} ---")


def fmt_bytes(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0: return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} PB"


def mem_rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return 0.0


def mem_print(tag: str) -> None:
    g = mem_rss_gb()
    if g > 0.0:
        print(f"  [mem] {tag:<48s} RSS = {g:6.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# q8rle codec — uint8 path avoids the float32 cast
# ─────────────────────────────────────────────────────────────────────────────
def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255),
                0, 255).astype(np.uint8)
    return _uint8_matrix_to_q8rle(q)


def _uint8_matrix_to_q8rle(q: np.ndarray) -> str:
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
    parts = s.split()
    h, w = int(parts[1]), int(parts[2])
    if len(parts) <= 3:
        return np.zeros((h, w), dtype=np.uint8)
    body = np.array(parts[3:], dtype=np.int64)
    vals = body[0::2].astype(np.uint8)
    lens = body[1::2]
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat


def q8rle_to_float_matrix(s: str) -> np.ndarray:
    """Legacy v5 helper (still callable, not used internally)."""
    return q8rle_to_uint8_matrix(s).astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Compact-storage helpers
# ─────────────────────────────────────────────────────────────────────────────
def to_f32(arr: np.ndarray) -> np.ndarray:
    """Promote any compact storage dtype to float32 in [0, 1] (or pass-through
    for float16/float32)."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (1.0 / 255.0)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) * (1.0 / 65535.0)
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def f32_to_u16(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr.astype(np.float32) * 65535.0),
                    0, 65535).astype(np.uint16)


def f32_to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr.astype(np.float32) * 255.0),
                    0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Cache (file-based, content-keyed)
# ─────────────────────────────────────────────────────────────────────────────
def file_meta_hash(paths: list[Path], extra: str = "") -> str:
    h = hashlib.sha1(); h.update(extra.encode())
    for p in sorted(paths, key=str):
        try:
            st = p.stat()
            h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}|".encode())
        except FileNotFoundError:
            h.update(f"{p.name}|missing|".encode())
    return h.hexdigest()[:16]


def params_hash(**kwargs) -> str:
    h = hashlib.sha1()
    h.update(json.dumps(kwargs, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


class Cache:
    """Trivial file-based cache; disabled when cache_dir is None."""
    def __init__(self, cache_dir: Path | None):
        self.dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _p(self, name: str) -> Path:
        return self.dir / name  # type: ignore[union-attr]

    def has(self, name: str) -> bool:
        return self.enabled and self._p(name).exists()

    def load_npz(self, name: str, allow_pickle: bool = True):
        if not self.enabled: return None
        p = self._p(name)
        if not p.exists(): return None
        try:
            d = np.load(p, allow_pickle=allow_pickle)
            print(f"  [cache] HIT  {name}  ({fmt_bytes(p.stat().st_size)})")
            return d
        except Exception as e:
            print(f"  [cache] FAIL {name}: {e}  (recomputing)")
            return None

    def save_npz(self, name: str, **kwargs) -> None:
        if not self.enabled: return
        p = self._p(name); tmp = p.with_suffix(p.suffix + ".tmp")
        # Pass a file handle, not a Path: np.savez_compressed auto-appends
        # ".npz" to string/Path args whose name doesn't already end in .npz,
        # which would write to "<name>.npz.tmp.npz" and break tmp.replace().
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **kwargs)
        tmp.replace(p)
        print(f"  [cache] SAVE {name}  ({fmt_bytes(p.stat().st_size)})")

    def load_pkl(self, name: str):
        if not self.enabled: return None
        p = self._p(name)
        if not p.exists(): return None
        try:
            with open(p, "rb") as f: obj = pickle.load(f)
            print(f"  [cache] HIT  {name}  ({fmt_bytes(p.stat().st_size)})")
            return obj
        except Exception as e:
            print(f"  [cache] FAIL {name}: {e}  (recomputing)")
            return None

    def save_pkl(self, name: str, obj) -> None:
        if not self.enabled: return
        p = self._p(name); tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)
        print(f"  [cache] SAVE {name}  ({fmt_bytes(p.stat().st_size)})")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders (unchanged from v5)
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
            f"local_preds_saver hook.")
    data = np.load(path, allow_pickle=True)
    out = {
        "ids":           data["ids"].astype(str),
        "classes":       data["classes"].astype(str),
        "anomaly_types": data["anomaly_types"].astype(str),
        "scores":        data["scores"].astype(np.float32),
        "masks":         data["masks"].astype(np.uint8),
    }
    out["image_paths"] = (data["image_paths"].astype(str)
                            if "image_paths" in data.files else None)
    return out


def build_class_map_from_data(data_root: Path) -> dict[str, str]:
    if not data_root.exists(): return {}
    out: dict[str, str] = {}
    for cdir in sorted(data_root.iterdir()):
        if not cdir.is_dir() or not cdir.name.startswith("class_"): continue
        test_dir = cdir / "test"
        if not test_dir.exists(): continue
        for p in test_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
                out[p.stem] = cdir.name
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sample id / view parsing
# ─────────────────────────────────────────────────────────────────────────────
PATH_VIEW_RE = re.compile(r"^(?P<sid>.+?)_view(?P<v>\d+)(?:\.[A-Za-z]+)?$")


def parse_sample_id(name: str) -> tuple[str, int | None]:
    m = PATH_VIEW_RE.match(name)
    if m: return m.group("sid"), int(m.group("v"))
    return name, None


# ─────────────────────────────────────────────────────────────────────────────
# Spatial priors
# ─────────────────────────────────────────────────────────────────────────────
def load_spatial_priors(prior_dir: Path,
                          classes: list[str]) -> dict[str, np.ndarray]:
    if not prior_dir.exists():
        print(f"  [warn] --prior-heatmaps-dir {prior_dir} does not exist; "
              f"spatial-prior feature will be all-zeros.")
        return {cls: np.zeros((128, 128), dtype=np.float32) for cls in classes}
    out: dict[str, np.ndarray] = {}
    for cls in classes:
        p = prior_dir / f"06_heat_{cls}.npy"
        if p.exists():
            try:
                h = np.load(p).astype(np.float32)
                h = np.clip(h, 0.0, 1.0); out[cls] = h
                print(f"    loaded spatial prior for {cls}: shape {h.shape}  "
                      f"max={h.max():.4f}")
            except Exception as e:
                print(f"    [warn] could not load {p}: {e}; using zeros")
                out[cls] = np.zeros((128, 128), dtype=np.float32)
        else:
            print(f"    [warn] no spatial prior file for {cls} "
                  f"(looked for {p.name}); using zeros")
            out[cls] = np.zeros((128, 128), dtype=np.float32)
    return out


def _resize_prior_to(prior: np.ndarray, H: int, W: int) -> np.ndarray:
    if prior.shape == (H, W): return prior.astype(np.float32, copy=False)
    zy = H / prior.shape[0]; zx = W / prior.shape[1]
    out = ndi.zoom(prior, zoom=(zy, zx), order=1, mode="nearest")
    if out.shape != (H, W):
        h2 = min(out.shape[0], H); w2 = min(out.shape[1], W)
        clean = np.zeros((H, W), dtype=np.float32)
        clean[:h2, :w2] = out[:h2, :w2]; out = clean
    return out.astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model family detection
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("cutpaste", "cutpaste"), ("dnv2", "patchcore_dnv2"),
    ("wrn50", "patchcore_wrn50"), ("uniad", "uniad"),
    ("cfa_", "cfa"), ("fastflow", "fastflow"), ("effad", "effad"),
]


def detect_model_family(run_dir_name: str) -> str:
    s = run_dir_name.lower()
    for pat, fam in MODEL_FAMILY_PATTERNS:
        if pat in s: return fam
    return "unknown"


DEFAULT_SMALL_CC_PER_FAMILY = {
    "patchcore_dnv2": 0, "patchcore_wrn50": 0, "cutpaste": 0,
    "uniad": 0, "cfa": 0, "fastflow": 0, "effad": 0, "unknown": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Small-CC suppression (float + uint8 wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def suppress_small_ccs_image(score: np.ndarray, min_cc: int,
                               t_pct: float = 99.0) -> np.ndarray:
    if min_cc <= 0: return score
    s = np.asarray(score, dtype=np.float32)
    t = float(np.percentile(s, t_pct))
    hot = s >= t
    if not hot.any(): return s
    labels, n = ndi.label(hot)
    if n == 0: return s
    sizes = np.bincount(labels.ravel())
    if sizes.size <= 1: return s
    small_labels = np.where(sizes <= min_cc)[0]
    small_labels = small_labels[small_labels > 0]
    if small_labels.size == 0: return s
    drop = np.isin(labels, small_labels)
    out = s.copy(); out[drop] = 0.0
    return out


def suppress_small_ccs_volume(scores: np.ndarray, min_cc: int,
                                 t_pct: float = 99.0) -> np.ndarray:
    if min_cc <= 0: return scores
    out = np.empty_like(scores)
    for i in range(scores.shape[0]):
        out[i] = suppress_small_ccs_image(scores[i], min_cc, t_pct)
    return out


def suppress_small_ccs_uint8(arr_u8: np.ndarray, min_cc: int,
                                t_pct: float = 99.0) -> np.ndarray:
    """uint8 in/out; transient float32 stays scoped to this call."""
    if min_cc <= 0: return arr_u8
    s = arr_u8.astype(np.float32) * (1.0 / 255.0)
    s = suppress_small_ccs_image(s, min_cc, t_pct)
    return f32_to_u8(s)


# ─────────────────────────────────────────────────────────────────────────────
# Rank-norm primitives
# ─────────────────────────────────────────────────────────────────────────────
def _rank_replace(col: np.ndarray) -> np.ndarray:
    n = col.size
    order = np.argsort(col, kind="stable")
    ranks = np.empty_like(col, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return ranks


def rank_normalise_global_inplace_val(scores: np.ndarray) -> None:
    M = scores.shape[-1]
    flat = scores.reshape(-1, M)
    for mi in range(M):
        flat[:, mi] = _rank_replace(flat[:, mi])


def rank_normalise_per_class_inplace_val(scores: np.ndarray,
                                            classes: np.ndarray) -> None:
    M = scores.shape[-1]
    for cls in sorted(set(classes.tolist())):
        idx = np.flatnonzero(classes == cls)
        if idx.size == 0: continue
        block = scores[idx]; flat = block.reshape(-1, M)
        for mi in range(M):
            flat[:, mi] = _rank_replace(flat[:, mi])
        scores[idx] = flat.reshape(block.shape)


def rank_normalise_per_class_view_inplace_val(scores: np.ndarray,
                                                  classes: np.ndarray,
                                                  views: np.ndarray) -> None:
    M = scores.shape[-1]
    have_view = views >= 0
    for cls in sorted(set(classes.tolist())):
        cls_mask = (classes == cls)
        for v in sorted({int(vv) for vv in views[cls_mask] if vv >= 0}):
            idx = np.flatnonzero(cls_mask & have_view & (views == v))
            if idx.size == 0: continue
            block = scores[idx]; flat = block.reshape(-1, M)
            for mi in range(M):
                flat[:, mi] = _rank_replace(flat[:, mi])
            scores[idx] = flat.reshape(block.shape)
        idx_unk = np.flatnonzero(cls_mask & (~have_view))
        if idx_unk.size > 0:
            block = scores[idx_unk]; flat = block.reshape(-1, M)
            for mi in range(M):
                flat[:, mi] = _rank_replace(flat[:, mi])
            scores[idx_unk] = flat.reshape(block.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Test-side rank-norm — operates on compact storage. Each bucket peaks at
# O(bucket_size) float32, not O(N_total).
# ─────────────────────────────────────────────────────────────────────────────
def _bucket_rank_compact(decoded_per_method, buckets, label,
                          out_dtype: str = "uint16") -> None:
    for mi, d in enumerate(decoded_per_method):
        t0 = time.time()
        for bk, ids_in_b in buckets.items():
            shapes = {sid: d[sid].shape for sid in ids_in_b}
            sizes  = {sid: int(np.prod(shapes[sid])) for sid in ids_in_b}
            total  = int(sum(sizes.values()))
            if total == 0: continue
            flat = np.empty(total, dtype=np.float32)
            idx = 0
            for sid in ids_in_b:
                a = d[sid]; n = sizes[sid]
                flat[idx:idx + n] = to_f32(a).ravel()
                idx += n
            ranks = _rank_replace(flat)
            del flat
            if out_dtype == "uint16":
                ranks_q = np.clip(np.rint(ranks * 65535.0),
                                    0, 65535).astype(np.uint16)
            elif out_dtype == "float16":
                ranks_q = ranks.astype(np.float16)
            elif out_dtype == "float32":
                ranks_q = ranks
            else:
                raise ValueError(out_dtype)
            del ranks
            idx = 0
            for sid in ids_in_b:
                n = sizes[sid]
                d[sid] = ranks_q[idx:idx + n].reshape(shapes[sid])
                idx += n
            del ranks_q
        print(f"    method {mi + 1}/{len(decoded_per_method)}: "
              f"{label} rank-norm done ({time.time() - t0:.1f}s)")


def rank_normalise_test_per_class_inplace(
    decoded_per_method, all_ids, class_map, default_class,
    out_dtype: str = "uint16") -> None:
    buckets: dict[str, list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        buckets[cls].append(sid)
    _bucket_rank_compact(decoded_per_method, buckets, "per-class", out_dtype)


def rank_normalise_test_per_class_view_inplace(
    decoded_per_method, all_ids, class_map, default_class,
    out_dtype: str = "uint16") -> None:
    buckets: dict[str, list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        _, v = parse_sample_id(sid)
        key = f"{cls}|view{v}" if v is not None else f"{cls}|view?"
        buckets[key].append(sid)
    _bucket_rank_compact(decoded_per_method, buckets, "per-class-view",
                          out_dtype)


def rank_normalise_test_global_inplace(decoded_per_method, all_ids,
                                            out_dtype: str = "uint16") -> None:
    _bucket_rank_compact(decoded_per_method, {"_GLOBAL_": list(all_ids)},
                          "global", out_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-view grouping
# ─────────────────────────────────────────────────────────────────────────────
def group_val_by_sample(val: dict) -> dict[tuple[str, str], list[int]]:
    image_paths = val.get("image_paths")
    out: dict[tuple[str, str], list[int]] = defaultdict(list)
    if image_paths is not None:
        for i, (cls, p) in enumerate(zip(val["classes"], image_paths)):
            sid, _v = parse_sample_id(Path(str(p)).name)
            out[(str(cls), sid)].append(i)
    else:
        for i, (cls, id_) in enumerate(zip(val["classes"], val["ids"])):
            sid = str(id_).rsplit("/", 1)[-1]
            out[(str(cls), sid)].append(i)
    return out


def group_test_ids_by_sample(all_ids, class_map, default_class
                                ) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid in all_ids:
        cls = (class_map.get(sid) if class_map else None) or default_class
        sample_id, _v = parse_sample_id(sid)
        out[(str(cls), sample_id)].append(sid)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Val XV aggregates — float16 storage (halves memory)
# ─────────────────────────────────────────────────────────────────────────────
def compute_xv_aggregates_val_f16(val_scores: np.ndarray,
                                       classes: np.ndarray,
                                       image_paths: np.ndarray | None):
    N, H, W, M = val_scores.shape
    val_dict = {"scores": val_scores, "classes": classes,
                "image_paths": image_paths,
                "ids": np.array([f"placeholder/{i}" for i in range(N)])}
    groups = group_val_by_sample(val_dict)
    xv_max    = np.zeros((N, H, W, M), dtype=np.float16)
    xv_mean   = np.zeros((N, H, W, M), dtype=np.float16)
    xv_std    = np.zeros((N, H, W, M), dtype=np.float16)
    xv_lonely = np.zeros((N, H, W, M), dtype=np.float16)
    is_multi  = np.zeros(N, dtype=np.uint8)
    n_multi_samples = 0
    for (_cls, _sid), idxs in groups.items():
        if len(idxs) < 2: continue
        n_multi_samples += 1
        for vi in idxs: is_multi[vi] = 1
        for mi in range(M):
            stack = val_scores[idxs, :, :, mi]
            V = stack.shape[0]
            for k, vi in enumerate(idxs):
                if V == 1: continue
                siblings = np.delete(stack, k, axis=0)
                sib_max  = siblings.max(axis=0)
                sib_mean = siblings.mean(axis=0)
                xv_max[vi, :, :, mi]    = sib_max.astype(np.float16)
                xv_mean[vi, :, :, mi]   = sib_mean.astype(np.float16)
                if V > 2:
                    xv_std[vi, :, :, mi] = siblings.std(axis=0).astype(np.float16)
                xv_lonely[vi, :, :, mi] = (stack[k] - sib_max).astype(np.float16)
    print(f"  xv-aggregates (val, f16): {n_multi_samples} multi-view samples "
          f"({int(is_multi.sum())}/{N} val images get non-zero aggregates)")
    return xv_max, xv_mean, xv_std, xv_lonely, is_multi


# ─────────────────────────────────────────────────────────────────────────────
# Test XV aggregates — on-the-fly, per sample group (V × M × H × W float32)
# ─────────────────────────────────────────────────────────────────────────────
def compute_xv_for_group(group_scores_f32: np.ndarray):
    """group_scores_f32: (V, M, H, W). Returns four (V, M, H, W) f32 maps."""
    V, M, H, W = group_scores_f32.shape
    if V < 2:
        z = np.zeros((V, M, H, W), dtype=np.float32)
        return z, z.copy(), z.copy(), z.copy()
    xv_max    = np.empty((V, M, H, W), dtype=np.float32)
    xv_mean   = np.empty((V, M, H, W), dtype=np.float32)
    xv_std    = np.zeros((V, M, H, W), dtype=np.float32)
    xv_lonely = np.empty((V, M, H, W), dtype=np.float32)
    for k in range(V):
        idxs = [j for j in range(V) if j != k]
        siblings = group_scores_f32[idxs]
        sib_max  = siblings.max(axis=0)
        sib_mean = siblings.mean(axis=0)
        xv_max[k]    = sib_max
        xv_mean[k]   = sib_mean
        xv_lonely[k] = group_scores_f32[k] - sib_max
        if V > 2:
            xv_std[k] = siblings.std(axis=0)
    return xv_max, xv_mean, xv_std, xv_lonely


# ─────────────────────────────────────────────────────────────────────────────
# Local-val alignment
# ─────────────────────────────────────────────────────────────────────────────
def _nn_resize_2d(arr: np.ndarray, target_shape, dtype) -> np.ndarray:
    th, tw = target_shape; h, w = arr.shape
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)


def align_local_preds(preds_per_method, method_names):
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
    masks = np.empty((N, H0, W0), dtype=np.uint8)
    classes = np.empty(N, dtype=object)
    anomaly_types = np.empty(N, dtype=object)
    views = np.full(N, -1, dtype=np.int32)
    image_paths_m0 = preds_per_method[0].get("image_paths")
    image_paths_aligned: np.ndarray | None = None
    if image_paths_m0 is not None:
        image_paths_aligned = np.empty(N, dtype=object)
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
                if image_paths_aligned is not None and image_paths_m0 is not None:
                    pth = str(image_paths_m0[j])
                    image_paths_aligned[i] = pth
                    _, v = parse_sample_id(Path(pth).name)
                    if v is not None: views[i] = int(v)
    n_with_view = int((views >= 0).sum())
    print(f"  aligned {N} val images × {M} methods @ {H0}x{W0}"
          + (f" (with image_paths, {n_with_view}/{N} views parsed)"
             if image_paths_aligned is not None else " (no image_paths)"))
    return {"ids": np.asarray(common),
            "classes": classes.astype(str),
            "anomaly_types": anomaly_types.astype(str),
            "views": views, "scores": scores, "masks": masks,
            "image_paths": image_paths_aligned}


def save_aligned_val(cache: Cache, name: str, val: dict) -> None:
    if not cache.enabled: return
    paths = val.get("image_paths")
    cache.save_npz(
        name, ids=val["ids"], classes=val["classes"],
        anomaly_types=val["anomaly_types"], views=val["views"],
        scores=val["scores"], masks=val["masks"],
        image_paths=(np.array([], dtype=object) if paths is None else paths))


def load_aligned_val(cache: Cache, name: str) -> dict | None:
    d = cache.load_npz(name)
    if d is None: return None
    paths = d["image_paths"]
    if paths.size == 0: paths = None
    return {"ids": d["ids"].astype(str),
            "classes": d["classes"].astype(str),
            "anomaly_types": d["anomaly_types"].astype(str),
            "views": d["views"].astype(np.int32),
            "scores": d["scores"].astype(np.float32),
            "masks": d["masks"].astype(np.uint8),
            "image_paths": (None if paths is None else paths.astype(str))}


# ═════════════════════════════════════════════════════════════════════════════
# v5 NEW FEATURE COMPUTATIONS (kept — tiny RAM footprint)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class MahalanobisParams:
    mu: np.ndarray
    inv_cov: np.ndarray
    n_fit: int


def fit_mahalanobis_per_class(val_scores, val_masks, val_classes,
                                  max_neg_per_class: int = 200_000,
                                  ridge: float = 1e-4, seed: int = 0):
    N, H, W, M = val_scores.shape
    rng = np.random.default_rng(seed)
    out: dict[str, MahalanobisParams] = {}
    print(f"\nFitting Mahalanobis per class (negative pixels of val)...")
    for cls in sorted(set(val_classes.tolist())):
        idx = np.flatnonzero(val_classes == cls)
        if idx.size == 0: continue
        scores_cls = val_scores[idx]
        masks_cls = val_masks[idx]
        neg_mask = (masks_cls == 0).ravel()
        flat_scores = scores_cls.reshape(-1, M)[neg_mask]
        if flat_scores.shape[0] > max_neg_per_class:
            sel = rng.choice(flat_scores.shape[0], max_neg_per_class,
                              replace=False)
            flat_scores = flat_scores[sel]
        if flat_scores.shape[0] < M + 2:
            print(f"  [warn] class {cls}: only {flat_scores.shape[0]} neg "
                  f"pixels available (< M+2); skipping Mahalanobis")
            continue
        mu = flat_scores.mean(axis=0).astype(np.float64)
        cov = np.cov(flat_scores.T).astype(np.float64)
        cov += ridge * np.eye(M, dtype=np.float64)
        try: inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError: inv_cov = np.linalg.pinv(cov)
        out[cls] = MahalanobisParams(
            mu=mu.astype(np.float32),
            inv_cov=inv_cov.astype(np.float32),
            n_fit=flat_scores.shape[0])
        d = flat_scores - mu
        m2 = float(np.einsum("ij,jk,ik->i", d, inv_cov, d).mean())
        print(f"    class {cls}: μ={mu.mean():.3f} ± {mu.std():.3f}, "
              f"E[d²]≈{m2:.1f} (expected ≈ M={M}), "
              f"n_fit={flat_scores.shape[0]}")
    return out


def compute_mahalanobis_map(scores_per_method_f32, params):
    if params is None:
        H, W = scores_per_method_f32[0].shape
        return np.zeros((H, W), dtype=np.float32)
    H, W = scores_per_method_f32[0].shape
    M = len(scores_per_method_f32)
    stack = np.stack(scores_per_method_f32, axis=-1).reshape(-1, M)
    d = stack - params.mu
    tmp = d @ params.inv_cov
    out = np.einsum("ij,ij->i", tmp, d).astype(np.float32)
    return out.reshape(H, W)


def compute_min_top_k(scores_per_method_f32, k: int = 3):
    if len(scores_per_method_f32) < k:
        return scores_per_method_f32[0].astype(np.float32).copy()
    stack = np.stack(scores_per_method_f32, axis=0).astype(np.float32)
    top_k = -np.partition(-stack, k - 1, axis=0)[:k]
    return top_k.min(axis=0)


def _cc_features_single(s: np.ndarray, top_pct: float = 98.0):
    H, W = s.shape
    log_area = np.zeros((H, W), dtype=np.float32)
    max_in_cc = np.zeros((H, W), dtype=np.float32)
    dist_centroid = np.full((H, W), fill_value=float(np.hypot(H, W)),
                              dtype=np.float32)
    t = float(np.percentile(s, top_pct))
    hot = s >= t
    if not hot.any(): return log_area, max_in_cc, dist_centroid
    labels, n = ndi.label(hot)
    if n == 0: return log_area, max_in_cc, dist_centroid
    sizes = np.bincount(labels.ravel())
    max_per_cc = ndi.maximum(s, labels=labels, index=np.arange(1, n + 1))
    centroids = ndi.center_of_mass(hot, labels=labels,
                                      index=np.arange(1, n + 1))
    log_area_lookup = np.zeros(n + 1, dtype=np.float32)
    log_area_lookup[1:] = np.log1p(sizes[1:]).astype(np.float32)
    log_area = log_area_lookup[labels]
    max_lookup = np.zeros(n + 1, dtype=np.float32)
    max_lookup[1:] = np.asarray(max_per_cc, dtype=np.float32)
    max_in_cc = max_lookup[labels]
    ys, xs = np.indices((H, W))
    for k in range(1, n + 1):
        cy, cx = centroids[k - 1]
        mask = labels == k
        dy = ys[mask] - cy; dx = xs[mask] - cx
        dist_centroid[mask] = np.sqrt(dy * dy + dx * dx).astype(np.float32)
    return log_area, max_in_cc, dist_centroid


@dataclass
class ImgP99Stats:
    mean: float
    std: float


def fit_imgp99_stats(val_scores: np.ndarray) -> list[ImgP99Stats]:
    N, H, W, M = val_scores.shape
    out: list[ImgP99Stats] = []
    for mi in range(M):
        p99s = np.array([float(np.percentile(val_scores[i, :, :, mi], 99))
                          for i in range(N)], dtype=np.float64)
        out.append(ImgP99Stats(mean=float(p99s.mean()),
                                std=float(p99s.std() + 1e-6)))
    print(f"\nImgP99 stats fitted per method:")
    for mi, s in enumerate(out):
        print(f"  method {mi}: imgp99 ~ N({s.mean:.3f}, {s.std:.3f})")
    return out


def pick_top_methods_per_class(val_scores, val_masks, val_classes,
                                  top_k: int = 3):
    from sklearn.metrics import average_precision_score
    N, H, W, M = val_scores.shape
    out: dict[str, list[int]] = {}
    print(f"\nPicking top-{top_k} methods per class by val pooled-AP...")
    for cls in sorted(set(val_classes.tolist())):
        idx = np.flatnonzero(val_classes == cls)
        if idx.size == 0: continue
        aps = []
        y = val_masks[idx].ravel()
        if int(y.sum()) == 0:
            out[cls] = list(range(min(top_k, M))); continue
        for mi in range(M):
            s = val_scores[idx, :, :, mi].ravel()
            try: aps.append((float(average_precision_score(y, s)), mi))
            except Exception: aps.append((0.0, mi))
        aps.sort(reverse=True)
        out[cls] = [mi for _, mi in aps[:top_k]]
        top_aps = ", ".join(f"m{mi}({ap:.3f})" for ap, mi in aps[:top_k])
        print(f"  {cls}: {top_aps}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    use_raw_score: bool = True
    use_per_image_rank: bool = True
    gauss_sigmas: tuple[float, ...] = (3.0,)
    window_sizes: tuple[int, ...] = (7,)
    use_window_mean: bool = True
    use_window_max: bool = False
    use_window_std: bool = False
    use_gradient: bool = False
    use_laplacian: bool = False
    use_dist_to_hot: bool = False
    hot_percentile: float = 99.0
    use_image_aggregates: bool = True
    image_aggregate_names: tuple[str, ...] = ("imgp99", "imgstd")
    use_cross_stats: bool = True
    use_spatial: bool = True
    use_cross_method_consensus: bool = True
    use_cross_view_disagreement: bool = False
    use_spatial_prior: bool = True
    use_class_onehot: bool = True
    use_view_onehot: bool = True
    use_xv_aggregates: bool = True
    use_mahalanobis: bool = True
    use_min_top_k: bool = True
    top_k_for_min: int = 3
    use_cc_features: bool = True
    n_top_methods_for_cc: int = 3
    cc_top_pct: float = 98.0
    use_per_method_zrank_top: bool = True
    use_cross_method_cv: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Featurization primitives
# ─────────────────────────────────────────────────────────────────────────────
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
    xs_n = xs / max(W - 1, 1); ys_n = ys / max(H - 1, 1)
    de = np.minimum(np.minimum(xs_n, ys_n),
                     np.minimum(1.0 - xs_n, 1.0 - ys_n)).astype(np.float32)
    dc = np.sqrt((xs_n - 0.5) ** 2 + (ys_n - 0.5) ** 2).astype(np.float32)
    return {"x": xs_n, "y": ys_n, "dist_edge": de, "dist_center": dc,
            "_shape": (H, W), "_prior_resized": {}}


def _onehot_const_planes(value: int, n: int, H: int, W: int):
    out = [np.zeros((H, W), dtype=np.float32) for _ in range(n)]
    if 0 <= value < n: out[value][:] = 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# featurize_image — accepts any storage dtype, upcasts lazily
# ─────────────────────────────────────────────────────────────────────────────
def featurize_image(
    scores_per_method: list[np.ndarray],
    cfg: FeatureConfig,
    spatial_cache: dict | None = None,
    feature_names: list[str] | None = None,
    is_multiview_sample: bool = True,
    class_id: str | None = None,
    all_classes: list[str] | None = None,
    view: int | None = None,
    n_views_onehot: int = 5,
    spatial_prior: np.ndarray | None = None,
    xv_max_per_method: list[np.ndarray] | None = None,
    xv_mean_per_method: list[np.ndarray] | None = None,
    xv_std_per_method: list[np.ndarray] | None = None,
    xv_lonely_per_method: list[np.ndarray] | None = None,
    mahal_params: MahalanobisParams | None = None,
    top_methods_for_cls: list[int] | None = None,
    imgp99_stats: list[ImgP99Stats] | None = None,
):
    assert scores_per_method
    scores_f32 = [to_f32(s) for s in scores_per_method]
    H, W = scores_f32[0].shape; M = len(scores_f32)
    layers: list[np.ndarray] = []; names: list[str] = []

    # Per-method (PRUNED in v5/v6)
    for mi, s in enumerate(scores_f32):
        if s.shape != (H, W):
            raise ValueError(f"method {mi} shape {s.shape} != ({H}, {W})")
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
            agg_map = {"imgmax": float(s.max()),
                       "imgp99": float(np.percentile(s, 99)),
                       "imgmean": float(s.mean()),
                       "imgstd": float(s.std())}
            for stat_name in cfg.image_aggregate_names:
                val = agg_map[stat_name]
                layers.append(np.full((H, W), val, dtype=np.float32))
                names.append(f"m{mi}_{stat_name}")

    # Cross-method stats
    if cfg.use_cross_stats and M >= 2:
        stack = np.stack(scores_f32, axis=0).astype(np.float32)
        cmean = stack.mean(axis=0); layers.append(cmean); names.append("x_mean")
        cmax  = stack.max(axis=0);  layers.append(cmax);  names.append("x_max")
        cmin  = stack.min(axis=0);  layers.append(cmin);  names.append("x_min")
        cstd  = stack.std(axis=0);  layers.append(cstd);  names.append("x_std")
        layers.append((cmax - cmin).astype(np.float32)); names.append("x_range")

    # Cross-method consensus
    if cfg.use_cross_method_consensus and M >= 2:
        stack = np.stack(scores_f32, axis=0).astype(np.float32)
        top1pct = (stack >= 0.99).sum(axis=0).astype(np.float32)
        top5pct = (stack >= 0.95).sum(axis=0).astype(np.float32)
        mean_rank = stack.mean(axis=0).astype(np.float32)
        layers.append(top1pct); names.append("xc_top1pct_count")
        layers.append(top5pct); names.append("xc_top5pct_count")
        layers.append(mean_rank); names.append("xc_mean_rank")

    # Cross-method CV (v5)
    if cfg.use_cross_method_cv and M >= 2:
        stack = np.stack(scores_f32, axis=0).astype(np.float32)
        cmean_ = stack.mean(axis=0); cstd_ = stack.std(axis=0)
        cv = (cstd_ / (cmean_ + 1e-6)).astype(np.float32)
        layers.append(cv); names.append("xc_cv")

    # Min-of-top-k (v5)
    if cfg.use_min_top_k and M >= cfg.top_k_for_min:
        m_topk = compute_min_top_k(scores_f32, k=cfg.top_k_for_min)
        layers.append(m_topk); names.append(f"xc_min_top{cfg.top_k_for_min}")

    # XV aggregates (CVD dropped)
    if cfg.use_xv_aggregates and xv_max_per_method is not None:
        for mi in range(M):
            xm  = xv_max_per_method[mi]    if xv_max_per_method    else None
            xmn = xv_mean_per_method[mi]   if xv_mean_per_method   else None
            xs_ = xv_std_per_method[mi]    if xv_std_per_method    else None
            xl  = xv_lonely_per_method[mi] if xv_lonely_per_method else None
            for arr, suffix in [(xm,"xv_max"),(xmn,"xv_mean"),
                                  (xs_,"xv_std"),(xl,"xv_lonely")]:
                if arr is None or not is_multiview_sample:
                    layers.append(np.zeros((H, W), dtype=np.float32))
                else:
                    a = to_f32(arr)
                    if a.shape != (H, W):
                        raise ValueError(
                            f"xv {suffix} method {mi}: shape {a.shape} != ({H}, {W})")
                    layers.append(a)
                names.append(f"m{mi}_{suffix}")

    # Spatial
    if cfg.use_spatial:
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        for k in ("x", "y", "dist_edge", "dist_center"):
            layers.append(spatial_cache[k]); names.append(f"s_{k}")

    # Spatial prior
    if cfg.use_spatial_prior and spatial_prior is not None:
        if spatial_cache is None or spatial_cache.get("_shape") != (H, W):
            spatial_cache = _spatial_cache(H, W)
        key = (class_id, H, W)
        cache_p = spatial_cache.setdefault("_prior_resized", {})
        if key not in cache_p:
            cache_p[key] = _resize_prior_to(spatial_prior, H, W)
        layers.append(cache_p[key]); names.append("s_prior")

    # Class one-hot
    if cfg.use_class_onehot and class_id is not None and all_classes:
        try: idx_cls = all_classes.index(class_id)
        except ValueError: idx_cls = -1
        planes = _onehot_const_planes(idx_cls, len(all_classes), H, W)
        for cls_name, plane in zip(all_classes, planes):
            layers.append(plane); names.append(f"c_{cls_name}")

    # View one-hot
    if cfg.use_view_onehot:
        if view is not None and 1 <= int(view) <= n_views_onehot:
            v_idx = int(view) - 1
        else:
            v_idx = n_views_onehot
        n_slots = n_views_onehot + 1
        planes = _onehot_const_planes(v_idx, n_slots, H, W)
        for k, plane in enumerate(planes):
            tag = (f"view_{k + 1}" if k < n_views_onehot else "view_unknown")
            layers.append(plane); names.append(f"v_{tag}")

    # Mahalanobis (v5)
    if cfg.use_mahalanobis:
        m_map = compute_mahalanobis_map(scores_f32, mahal_params)
        layers.append(m_map); names.append("s_mahal")

    # CC features (v5)
    if cfg.use_cc_features and top_methods_for_cls is not None:
        for rank_idx in range(cfg.n_top_methods_for_cc):
            if rank_idx >= len(top_methods_for_cls):
                for suffix in ("cc_logarea", "cc_max", "cc_distcent"):
                    layers.append(np.zeros((H, W), dtype=np.float32))
                    names.append(f"top{rank_idx}_{suffix}")
                continue
            mi = top_methods_for_cls[rank_idx]
            log_a, mx, dc = _cc_features_single(scores_f32[mi],
                                                  top_pct=cfg.cc_top_pct)
            layers.append(log_a);  names.append(f"top{rank_idx}_cc_logarea")
            layers.append(mx);     names.append(f"top{rank_idx}_cc_max")
            layers.append(dc);     names.append(f"top{rank_idx}_cc_distcent")

    # Per-method z-rank (v5)
    if (cfg.use_per_method_zrank_top and top_methods_for_cls is not None
            and imgp99_stats is not None):
        for rank_idx in range(cfg.n_top_methods_for_cc):
            if rank_idx >= len(top_methods_for_cls):
                layers.append(np.zeros((H, W), dtype=np.float32))
                names.append(f"top{rank_idx}_imgzrank"); continue
            mi = top_methods_for_cls[rank_idx]
            this_p99 = float(np.percentile(scores_f32[mi], 99))
            stats = imgp99_stats[mi]
            z = (this_p99 - stats.mean) / stats.std
            layers.append(np.full((H, W), z, dtype=np.float32))
            names.append(f"top{rank_idx}_imgzrank")

    feats = np.stack(layers, axis=-1).astype(np.float32, copy=False)
    if feature_names is not None and names != feature_names:
        missing = [n for n in feature_names if n not in names]
        extra   = [n for n in names if n not in feature_names]
        raise RuntimeError(
            f"feature drift between fit and predict: "
            f"got {len(names)} cols, expected {len(feature_names)}; "
            f"missing={missing[:5]}; extra={extra[:5]}")
    return feats, names


# ─────────────────────────────────────────────────────────────────────────────
# Build per-class training matrices, with free() helper for lifecycle
# ─────────────────────────────────────────────────────────────────────────────
def build_training_data(val: dict, classes: list[str], cfg: FeatureConfig,
                         *, neg_per_pos: int, seed: int,
                         xv_max=None, xv_mean=None,
                         xv_std=None, xv_lonely=None,
                         is_multi=None, spatial_priors=None,
                         all_classes_onehot=None,
                         mahal_per_class=None,
                         top_methods_per_class=None,
                         imgp99_stats=None) -> dict:
    rng = np.random.default_rng(seed)
    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]; val_anom_types = val["anomaly_types"]
    val_views = val.get("views", np.full(val_scores.shape[0], -1, dtype=np.int32))
    N, H, W, M = val_scores.shape
    spatial = _spatial_cache(H, W)
    out: dict = {}
    feature_names_global: list[str] | None = None

    for cls in classes:
        cls_idx = np.flatnonzero(val_classes == cls)
        if cls_idx.size == 0:
            print(f"  [warn] class {cls}: no val images; skipping"); continue
        per_img_X, per_img_y, per_img_atypes = [], [], []
        per_img_pixrange: list[tuple[int, int]] = []
        cursor = 0
        prior_cls = (spatial_priors.get(cls) if spatial_priors else None)
        mahal_cls = (mahal_per_class.get(cls) if mahal_per_class else None)
        top_methods_cls = (top_methods_per_class.get(cls)
                              if top_methods_per_class else None)
        for i in cls_idx:
            mat_list = [val_scores[i, :, :, mi] for mi in range(M)]
            is_mv = bool(is_multi[i]) if is_multi is not None else True
            xv_max_list    = (None if xv_max    is None
                                else [xv_max[i, :, :, mi]    for mi in range(M)])
            xv_mean_list   = (None if xv_mean   is None
                                else [xv_mean[i, :, :, mi]   for mi in range(M)])
            xv_std_list    = (None if xv_std    is None
                                else [xv_std[i, :, :, mi]    for mi in range(M)])
            xv_lonely_list = (None if xv_lonely is None
                                else [xv_lonely[i, :, :, mi] for mi in range(M)])
            v_int = int(val_views[i]) if val_views[i] >= 0 else None
            feats, names = featurize_image(
                mat_list, cfg, spatial_cache=spatial,
                is_multiview_sample=is_mv,
                class_id=cls, all_classes=all_classes_onehot,
                view=v_int, spatial_prior=prior_cls,
                xv_max_per_method=xv_max_list,
                xv_mean_per_method=xv_mean_list,
                xv_std_per_method=xv_std_list,
                xv_lonely_per_method=xv_lonely_list,
                mahal_params=mahal_cls,
                top_methods_for_cls=top_methods_cls,
                imgp99_stats=imgp99_stats)
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
              f"(pos={n_pos:>6d}, neg={len(sample_neg):>8d}, "
              f"feats={X_full.shape[1]:>3d})")
        mem_print(f"  built training data for {cls}")
    out["_feature_names"] = feature_names_global or []
    return out


def free_class_training_data(training_data: dict, cls: str,
                                 keep_xtrain: bool = False) -> None:
    """Drop large per-class arrays after OOF + production model fit."""
    if cls not in training_data: return
    td = training_data[cls]
    if isinstance(td, dict):
        td.pop("X_full", None); td.pop("y_full", None)
        td.pop("img_pixranges", None); td.pop("anomaly_types", None)
        if not keep_xtrain:
            td.pop("X_train", None); td.pop("y_train", None)


# ─────────────────────────────────────────────────────────────────────────────
# XGB params (v5/v6 regularised defaults)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.03,
    "subsample":        0.6,
    "colsample_bytree": 0.6,
    "reg_alpha":        0.5,
    "reg_lambda":       1.0,
    "min_child_weight": 20.0,
    "gamma":            0.5,
    "tree_method":      "hist",
    "max_bin":          256,
    "eval_metric":      "aucpr",
    "n_jobs":           -1,
    "verbosity":        0,
}


def _make_xgb(params: dict, seed: int):
    if not HAS_XGB:
        raise SystemExit("xgboost not installed. uv pip install xgboost")
    p = dict(DEFAULT_XGB_PARAMS); p.update(params or {})
    return xgb.XGBClassifier(random_state=seed, **p)


def get_params_for_class(cls: str, params_global: dict,
                          params_per_class: dict | None) -> dict:
    if params_per_class and cls in params_per_class:
        return params_per_class[cls]
    return params_global


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
    for i in range(n_imgs): buckets[key(i)].append(i)
    return buckets


def _pixel_ap_pooled(preds: np.ndarray, labels: np.ndarray) -> float:
    if int(labels.sum()) == 0: return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, preds))
    except Exception:
        order = np.argsort(-preds, kind="stable")
        y = labels[order]
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / max(int(y.sum()), 1)
        rec = np.concatenate([[0.0], rec])
        prec = np.concatenate([[1.0], prec])
        return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def pooled_cv_score_one_class(td: dict, params: dict, seed: int,
                                mode: str = "loao",
                                neg_per_pos: int = 30) -> float:
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
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
        held_preds = []; held_labels = []
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            held_preds.append(p); held_labels.append(y_full[s:e].astype(np.int32))
        held_preds_cat = np.concatenate(held_preds)
        held_labels_cat = np.concatenate(held_labels)
        if int(held_labels_cat.sum()) > 0:
            fold_aps.append(_pixel_ap_pooled(held_preds_cat, held_labels_cat))
    valid = [a for a in fold_aps if not math.isnan(a)]
    return float(np.mean(valid)) if valid else 0.0


def loao_oof_one_class(td: dict, params: dict, seed: int,
                        cv_mode: str = "loao",
                        neg_per_pos: int = 30):
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
        clf = _make_xgb(params, seed)
        clf.fit(np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
        for hi in held_imgs:
            s, e = ranges[hi]
            p = clf.predict_proba(X_full[s:e])[:, 1].astype(np.float32)
            oof_preds[s:e] = p
    mask = ~np.isnan(oof_preds)
    return oof_preds[mask].astype(np.float32), y_full[mask].astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────
def _optuna_suggest(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 600, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 5),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "reg_alpha":        trial.suggest_float("reg_alpha",  1e-3, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 100.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
    }


def tune_global(training_data, *, n_trials, seed, cv_mode="loao",
                timeout=None, neg_per_pos=30):
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")
    classes = [c for c in training_data if not c.startswith("_")
                and not training_data[c].get("_fallback_to_shared")]
    def objective(trial):
        params = _optuna_suggest(trial)
        aps = [pooled_cv_score_one_class(training_data[cls], params, seed,
                                            mode=cv_mode,
                                            neg_per_pos=neg_per_pos)
                for cls in classes]
        return float(np.mean(aps)) if aps else 0.0
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize",
                                  sampler=sampler, pruner=pruner)
    print(f"\n>>> Global Optuna tuning: {n_trials} trials, CV={cv_mode}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f">>> best mean pooled-AP: {study.best_value:.4f}")
    print(f">>> best params: {json.dumps(study.best_params, indent=2)}")
    return study.best_params


def tune_per_class(training_data, *, n_trials, seed, cv_mode="loao",
                    timeout_per_class=None, neg_per_pos=30):
    if not HAS_OPTUNA:
        raise SystemExit("optuna not installed.")
    out: dict = {}
    for cls, td in training_data.items():
        if cls.startswith("_"): continue
        if td.get("_fallback_to_shared"):
            print(f"\n  class {cls}: fallback-to-shared, no per-class tuning")
            continue
        print(f"\n>>> Per-class tuning: {cls}  (CV={cv_mode}, "
              f"n_trials={n_trials}, timeout_per_class={timeout_per_class}s)")
        def objective(trial, _td=td):
            return pooled_cv_score_one_class(_td, _optuna_suggest(trial),
                                                seed, mode=cv_mode,
                                                neg_per_pos=neg_per_pos)
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize",
                                      sampler=sampler, pruner=pruner,
                                      study_name=f"xgb_{cls}")
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials,
                        timeout=timeout_per_class)
        elapsed = time.time() - t0
        print(f"  {cls}: best CV pooled-AP = {study.best_value:.4f}  "
              f"({elapsed:.1f}s, {len(study.trials)} trials)")
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
# Calibration
# ─────────────────────────────────────────────────────────────────────────────
def fit_calibrator(method, oof_preds, oof_labels) -> dict:
    if method == "none": return {"method": "none"}
    if oof_preds.size == 0 or int(oof_labels.sum()) == 0:
        print(f"    [warn] no positives in OOF; falling back to no-op")
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
    if m == "none": return scores.astype(np.float32)
    if m == "platt":
        z = cal["coef"] * scores + cal["intercept"]
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    if m == "isotonic":
        xs = np.asarray(cal["X_thresholds"], dtype=np.float32)
        ys = np.asarray(cal["y_thresholds"], dtype=np.float32)
        return np.interp(scores, xs, ys).astype(np.float32)
    raise ValueError(m)


# ─────────────────────────────────────────────────────────────────────────────
# Fit per-class production models — drops X_full as it progresses
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
                    "logloss": ll, "train_ap": ap, "params": params,
                    "feature_importance": clf.feature_importances_.tolist()}
        print(f"  class {cls}: n_pos={td['n_pos']:>6d}  "
              f"n_neg={td['n_neg_sampled']:>8d}  "
              f"train_logloss={ll:.4f}  train_ap={ap:.3f}")
    if shared_X:
        X_all = np.concatenate(shared_X, axis=0)
        y_all = np.concatenate(shared_y, axis=0)
        del shared_X, shared_y; gc.collect()
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
        del X_all, y_all; gc.collect()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# LB-proxy pooled AP + verification
# ═════════════════════════════════════════════════════════════════════════════
def compute_pooled_lb_proxy_ap(val, method_names, oof_per_class):
    from sklearn.metrics import average_precision_score
    val_scores = val["scores"]; val_masks = val["masks"]
    N, H, W, M = val_scores.shape
    all_p = np.concatenate([op for (op, _) in oof_per_class.values()])
    all_l = np.concatenate([ol for (_, ol) in oof_per_class.values()])
    if int(all_l.sum()) == 0:
        stacker_pooled = float("nan")
    else:
        stacker_pooled = float(average_precision_score(all_l, all_p))
    singles_pooled: list[float] = []
    y_all_pix = val_masks.ravel()
    for mi in range(M):
        s_all = val_scores[:, :, :, mi].ravel()
        if int(y_all_pix.sum()) == 0:
            singles_pooled.append(float("nan"))
        else:
            try:
                singles_pooled.append(
                    float(average_precision_score(y_all_pix, s_all)))
            except Exception:
                singles_pooled.append(float("nan"))
    best_idx = int(np.nanargmax(singles_pooled)) if any(
        not math.isnan(x) for x in singles_pooled) else 0
    best_single_pooled = (method_names[best_idx], singles_pooled[best_idx])
    delta_pooled = (stacker_pooled - best_single_pooled[1]
                     if not (math.isnan(stacker_pooled)
                             or math.isnan(best_single_pooled[1]))
                     else float("nan"))
    return {"stacker_pooled_ap": stacker_pooled,
            "singles_pooled_ap": singles_pooled,
            "best_single_pooled": best_single_pooled,
            "delta_pooled": delta_pooled}


def verify_stacker_vs_singles(val, method_names, oof_per_class, *,
                                pretty_method_names=None):
    from sklearn.metrics import average_precision_score
    pretty = pretty_method_names or method_names
    val_scores = val["scores"]; val_masks = val["masks"]
    val_classes = val["classes"]
    N, H, W, M = val_scores.shape
    print(f"\n  Computing per-class pooled pixel-AP for {M} single models...")
    single_ap: dict[str, dict[str, float]] = {}
    for mi, mname in enumerate(pretty):
        single_ap[mname] = {}
        for cls in sorted(set(val_classes.tolist())):
            idx = np.flatnonzero(val_classes == cls)
            if idx.size == 0: continue
            s = val_scores[idx, :, :, mi].ravel()
            y = val_masks[idx].ravel()
            if int(y.sum()) == 0: ap = float("nan")
            else:
                try: ap = float(average_precision_score(y, s))
                except Exception: ap = float("nan")
            single_ap[mname][cls] = ap
    print(f"  Computing per-class pooled pixel-AP for stacker OOF...")
    stacker_ap: dict[str, float] = {}
    for cls, (op, ol) in oof_per_class.items():
        if int(ol.sum()) == 0:
            stacker_ap[cls] = float("nan"); continue
        try: stacker_ap[cls] = float(average_precision_score(ol, op))
        except Exception: stacker_ap[cls] = float("nan")
    all_classes = sorted(set(val_classes.tolist()))
    hdr = f"  {'class':<10}"
    for n in pretty: hdr += f" {n[:12]:>13}"
    hdr += f" {'STACKER':>13} {'best_single':>13} {'win?':>7}"
    hr("VERIFICATION — per-class POOLED pixel-AP (stacker vs singles)", "=")
    print(hdr)
    n_class_wins = 0; n_class_total = 0; deltas: list[float] = []
    for cls in all_classes:
        line = f"  {cls:<10}"
        bests = []
        for n in pretty:
            ap = single_ap[n].get(cls, float("nan"))
            line += f" {ap:>13.4f}"
            if not math.isnan(ap): bests.append((ap, n))
        st = stacker_ap.get(cls, float("nan"))
        best_ap, best_n = (max(bests) if bests else (float("nan"), "?"))
        line += f" {st:>13.4f}"; line += f" {best_ap:>13.4f}"
        if not (math.isnan(st) or math.isnan(best_ap)):
            n_class_total += 1; delta = st - best_ap; deltas.append(delta)
            line += f" {'WIN' if delta > 0 else 'lose':>7}"
            if delta > 0: n_class_wins += 1
        else:
            line += f" {'?':>7}"
        print(line)
    def _mean_skipnan(d):
        vals = [v for v in d.values() if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")
    overall_singles = {n: _mean_skipnan(single_ap[n]) for n in pretty}
    overall_stacker = _mean_skipnan(stacker_ap)
    best_single_name = max(overall_singles, key=lambda n: overall_singles[n])
    best_single_ap   = overall_singles[best_single_name]
    print()
    print(f"  mean pooled-AP across classes (per single model):")
    for n in sorted(pretty, key=lambda nn: -overall_singles[nn]):
        marker = "  <- best single" if n == best_single_name else ""
        print(f"    {n:<20} {overall_singles[n]:.4f}{marker}")
    print(f"  STACKER mean pooled-AP:  {overall_stacker:.4f}")
    delta_overall = overall_stacker - best_single_ap
    if not math.isnan(delta_overall):
        if delta_overall > 0:
            print(f"  >>> STACKER BEATS best single ({best_single_name}) "
                  f"by +{delta_overall:.4f} mean pooled-AP.")
        else:
            print(f"  >>> STACKER LOSES to best single ({best_single_name}) "
                  f"by {delta_overall:+.4f} mean pooled-AP.")
    if n_class_total > 0:
        print(f"  >>> per-class: stacker wins {n_class_wins}/{n_class_total} "
              f"classes  (mean delta = {np.mean(deltas):+.4f})")

    hr("LB-PROXY POOLED PIXEL-AP (mirrors public leaderboard exactly)", "=")
    pooled = compute_pooled_lb_proxy_ap(val, pretty, oof_per_class)
    print(f"  Single-model POOLED pixel-AP (all pixels concatenated):")
    sorted_singles = sorted(
        zip(pretty, pooled["singles_pooled_ap"]),
        key=lambda kv: -kv[1] if not math.isnan(kv[1]) else 1)
    for n, ap in sorted_singles:
        marker = "  <- best single (pooled)" if n == pooled["best_single_pooled"][0] else ""
        print(f"    {n:<20} {ap:.4f}{marker}")
    print(f"")
    print(f"  STACKER pooled pixel-AP:     {pooled['stacker_pooled_ap']:.4f}")
    print(f"  Best-single pooled pixel-AP: {pooled['best_single_pooled'][1]:.4f}  "
          f"({pooled['best_single_pooled'][0]})")
    if not math.isnan(pooled["delta_pooled"]):
        if pooled["delta_pooled"] > 0:
            print(f"  >>> STACKER BEATS best single (pooled) by "
                  f"+{pooled['delta_pooled']:.4f}.  ← this tracks the LB.")
        else:
            print(f"  >>> STACKER LOSES to best single (pooled) by "
                  f"{pooled['delta_pooled']:+.4f}.  ← this tracks the LB.")
    print(f"\n  ⚠ Use the POOLED number above as your offline LB proxy.")
    return {"per_class_single": single_ap, "per_class_stacker": stacker_ap,
            "overall_singles": overall_singles,
            "overall_stacker": overall_stacker,
            "best_single_name": best_single_name,
            "best_single_overall_ap": best_single_ap,
            "delta_overall": delta_overall,
            "class_wins": n_class_wins, "class_total": n_class_total,
            "pooled_stacker_ap": pooled["stacker_pooled_ap"],
            "pooled_singles_ap": dict(zip(pretty, pooled["singles_pooled_ap"])),
            "pooled_best_single": pooled["best_single_pooled"],
            "pooled_delta": pooled["delta_pooled"]}


# ═════════════════════════════════════════════════════════════════════════════
# Test decode + cache helpers
# ═════════════════════════════════════════════════════════════════════════════
def decode_submissions_to_uint8(submissions, all_ids):
    M = len(submissions); N = len(all_ids)
    decoded: list[dict[str, np.ndarray]] = []
    t0 = time.time()
    for mi, sub in enumerate(submissions):
        d: dict[str, np.ndarray] = {}
        for j, sid in enumerate(all_ids):
            d[sid] = q8rle_to_uint8_matrix(sub[sid])
            if (j + 1) % 1000 == 0:
                print(f"    method {mi + 1}/{M}: {j + 1}/{N} "
                      f"({time.time() - t0:.1f}s)", flush=True)
        decoded.append(d)
        print(f"    method {mi + 1}/{M} decoded ({time.time() - t0:.1f}s)")
    return decoded


def cache_decoded_test(cache: Cache, name: str, decoded_per_method) -> None:
    if not cache.enabled: return
    cache.save_pkl(name, {"_n_methods": len(decoded_per_method),
                            "_per_method": decoded_per_method})


def load_decoded_test(cache: Cache, name: str):
    obj = cache.load_pkl(name)
    if obj is None: return None
    return obj["_per_method"]


# ─────────────────────────────────────────────────────────────────────────────
# fuse_test — streaming, grouped, on-the-fly XV
# ─────────────────────────────────────────────────────────────────────────────
def fuse_test(submissions, models, class_map, cfg: FeatureConfig,
               rank_norm_mode: str, default_class: str,
               calibrators_per_class, small_cc_per_method,
               include_xv: bool, spatial_priors, all_classes_onehot,
               mahal_per_class, top_methods_per_class, imgp99_stats,
               *, cache: Cache | None = None, cache_key_base: str = "",
               storage_dtype: str = "uint16",
               drop_decoded_during_fusion: bool = True) -> dict[str, str]:
    """Fuse with memory-efficient layout.

    Storage strategy:
      - decoded_per_method[mi]: dict[sid → uint8]   (raw, post-decode)
                                       → dict[sid → uint16]  (post-rank-norm)
        Promoted to float32 inside featurize_image only for the few images
        currently being processed.

    Cross-view aggregates:
      - Computed per (class, sample_id) group inside the loop.  Group memory
        is V × M × H × W float32, transient.

    drop_decoded_during_fusion:
      - When True, removes a group's decoded arrays from decoded_per_method
        after that group is fused.  Decoded memory shrinks monotonically.
    """
    cache = cache or Cache(None)
    common = set.intersection(*[set(s.keys()) for s in submissions])
    if not common:
        raise RuntimeError("no test IDs in common across submissions")
    all_ids = sorted(common)
    M = len(submissions)
    feature_names = models.get("_feature_names", None)
    mem_print("fuse_test: enter")

    # ── 1. Decode (uint8) ────────────────────────────────────────────────────
    decoded_per_method: list[dict[str, np.ndarray]] | None = None
    raw_key = f"decoded_raw_{cache_key_base}.pkl"
    if cache.has(raw_key):
        cached = load_decoded_test(cache, raw_key)
        if (cached is not None and len(cached) == M
                and set(cached[0].keys()) >= set(all_ids)):
            decoded_per_method = cached

    if decoded_per_method is None:
        print(f"\nDecoding {M} submissions × {len(all_ids)} IDs each (uint8)...")
        decoded_per_method = decode_submissions_to_uint8(submissions, all_ids)
        if any(c > 0 for c in small_cc_per_method):
            print(f"\nApplying per-method small-CC suppression "
                  f"(min_cc = {small_cc_per_method})...")
            for mi, min_cc in enumerate(small_cc_per_method):
                if min_cc <= 0: continue
                t1 = time.time()
                for sid in all_ids:
                    decoded_per_method[mi][sid] = suppress_small_ccs_uint8(
                        decoded_per_method[mi][sid], min_cc=min_cc)
                print(f"    method {mi + 1}/{M}: suppressed CCs ≤ {min_cc} px "
                      f"({time.time() - t1:.1f}s)")
        cache_decoded_test(cache, raw_key, decoded_per_method)
    mem_print("fuse_test: after decode (uint8)")

    # ── 2. Rank-norm → uint16 storage ───────────────────────────────────────
    norm_key = (f"decoded_norm_{rank_norm_mode}_{storage_dtype}_"
                 f"{cache_key_base}.pkl")
    norm_cached = None
    if cache.has(norm_key):
        norm_cached = load_decoded_test(cache, norm_key)

    if (norm_cached is not None and len(norm_cached) == M
            and set(norm_cached[0].keys()) >= set(all_ids)):
        decoded_per_method = norm_cached
        print(f"  using cached rank-normed test (mode={rank_norm_mode}, "
              f"dtype={storage_dtype})")
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
            print(f"\nSkipping rank-norm (test) — storage stays uint8")
        else:
            raise ValueError(f"unknown rank_norm_mode: {rank_norm_mode}")
        cache_decoded_test(cache, norm_key, decoded_per_method)
    mem_print("fuse_test: after rank-norm"); gc.collect()

    # ── 3. Group by (class, sample_id) and stream-fuse ──────────────────────
    groups = group_test_ids_by_sample(all_ids, class_map, default_class)
    n_groups = len(groups)
    n_multi = sum(1 for ids in groups.values() if len(ids) >= 2)
    print(f"\nFusing {len(all_ids)} images in {n_groups} sample groups "
          f"({n_multi} multi-view)"
          f"{' + calibration' if calibrators_per_class else ''}...")

    fused: dict[str, str] = {}
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
                if a.shape != (H, W):
                    raise ValueError(
                        f"shape mismatch in group ({cls}, {sample_id}): "
                        f"{sid} method {mi} is {a.shape} != ({H}, {W})")
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

            fused_mat = np.clip(fused_mat, 0.0, 1.0).astype(np.float32)
            fused[sid] = float_matrix_to_q8rle(fused_mat)
            n_done += 1
            if n_done % 500 == 0:
                print(f"    fused {n_done}/{len(all_ids)}  "
                      f"({time.time() - t1:.1f}s)", flush=True)

        # Free this group's decoded arrays
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
    mem_print("fuse_test: exit")
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
        if k not in fieldnames: fieldnames.append(k)
    existing.append(row)
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing: w.writerow({k: r.get(k, "") for k in fieldnames})


def parse_small_cc_spec(spec, method_names, default_min_cc) -> list[int]:
    cli_pairs: dict[str, int] = {}
    if spec:
        for s in spec:
            if "=" not in s:
                raise SystemExit(f"[FATAL] --small-cc must be key=value; got {s!r}")
            k, v = s.split("=", 1)
            cli_pairs[k.strip().lower()] = int(v)
    out: list[int] = []
    for i, name in enumerate(method_names):
        fam = detect_model_family(name)
        n_lower = name.lower()
        val: int | None = None
        for k, v in cli_pairs.items():
            if k.isdigit() and int(k) == i: val = v; break
            if not k.isdigit() and k in n_lower: val = v; break
        if val is None and fam in cli_pairs: val = cli_pairs[fam]
        if val is None: val = DEFAULT_SMALL_CC_PER_FAMILY.get(fam, default_min_cc)
        out.append(int(val))
    return out


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
    ap.add_argument("--rank-norm", default="per-class-view",
                    choices=["per-class-view", "per-class", "global", "none"])
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-csv", type=Path,
                    default=Path("/work/u10813429/anomaly-detection/"
                                  "baseline_out/ablation_master.csv"))
    ap.add_argument("--run-tag", default="stacker-xgb-v6")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--small-cc", nargs="*", default=None)
    ap.add_argument("--small-cc-default", type=int, default=0)
    ap.add_argument("--no-small-cc", action="store_true")
    # Feature toggles
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
    # v5 feature toggles
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
    # Tuning + calibration
    ap.add_argument("--tune-mode", default="none",
                    choices=["none", "global", "per-class"])
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--tune-cv", default="loao", choices=["loao", "loio"])
    ap.add_argument("--tune-timeout-min", type=float, default=None)
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "platt", "isotonic"])
    ap.add_argument("--no-verify", action="store_true")
    # v6 memory + cache
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="If set, cache aligned val + decoded test there.")
    ap.add_argument("--test-storage-dtype", default="uint16",
                    choices=["uint16", "float16", "float32"],
                    help="Storage dtype for rank-normed test predictions.")
    ap.add_argument("--legacy-float32-storage", action="store_true",
                    help="Force float32 for test storage (parity with v5).")
    ap.add_argument("--no-free-class-data", action="store_true",
                    help="Disable per-class lifecycle free (parity with v5).")
    ap.add_argument("--no-drop-decoded-during-fusion", action="store_true",
                    help="Don't evict groups from decoded_per_method as we go.")
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

    method_names = [p.parent.name for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.out.parent
    cache = Cache(args.cache_dir)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"XGBOOST STACKER v6 — {len(args.runs)} methods", "=")
        print(f"  rank_norm        : {args.rank_norm}")
        print(f"  tune_mode        : {args.tune_mode}")
        print(f"  calibrate        : {args.calibrate}")
        print(f"  prior dir        : {args.prior_heatmaps_dir}")
        print(f"  cache_dir        : {args.cache_dir}")
        print(f"  test storage     : {args.test_storage_dtype}")
        print(f"  free class data  : {not args.no_free_class_data}")
        print(f"  drop decoded     : {not args.no_drop_decoded_during_fusion}")
        for i, (r, lp) in enumerate(zip(args.runs, args.local_preds)):
            fam = detect_model_family(method_names[i])
            print(f"  method {i}: {method_names[i]}  (family={fam})")
        mem_print("start")

        # Small-CC resolve
        if args.no_small_cc:
            small_cc_per_method = [0] * len(method_names)
        else:
            small_cc_per_method = parse_small_cc_spec(
                args.small_cc, method_names, args.small_cc_default)
        print(f"\n  small-CC suppression per method: {small_cc_per_method}")

        # Cache keys
        local_preds_meta = file_meta_hash(args.local_preds,
                                              extra=f"smallcc={small_cc_per_method}")
        runs_meta = file_meta_hash(args.runs)
        cache_key_val   = f"val_aligned_{local_preds_meta}.npz"
        cache_key_test  = f"{runs_meta}__{local_preds_meta}"

        # ── Load + align val (cached) ────────────────────────────────────────
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
                print(f"  {p.parent.name}/{p.name}: {len(d['ids'])} val images, "
                      f"{float(d['masks'].mean()) * 100:.3f}% positive pixels")
                preds_per_method.append(d)

            if any(c > 0 for c in small_cc_per_method):
                print(f"\nApplying per-method small-CC suppression to local-val...")
                for mi, min_cc in enumerate(small_cc_per_method):
                    if min_cc <= 0: continue
                    t1 = time.time()
                    preds_per_method[mi]["scores"] = suppress_small_ccs_volume(
                        preds_per_method[mi]["scores"], min_cc=min_cc)
                    print(f"    method {mi + 1}: min_cc={min_cc} "
                          f"({time.time() - t1:.1f}s)")

            print("\nAligning local-val predictions across methods...")
            val = align_local_preds(preds_per_method, method_names)
            del preds_per_method; gc.collect()
            save_aligned_val(cache, cache_key_val, val)
        else:
            # Lazy-load test submissions (we'll need them for fusion)
            print("\nLoading test submissions...")
            subs = [load_submission(p) for p in args.runs]
            for p, s in zip(args.runs, subs):
                print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
        mem_print("after val align")

        # Rank-norm val (in place; cheap, recomputed each run)
        any_has_paths = val.get("image_paths") is not None
        if args.rank_norm == "per-class-view":
            if (val["views"] >= 0).any():
                print(f"\nPer-(class, view) rank-norm (val)...")
                rank_normalise_per_class_view_inplace_val(
                    val["scores"], val["classes"], val["views"])
            else:
                print(f"\n[warn] --rank-norm per-class-view requested but no "
                      f"views were parsed; falling back to per-class.")
                args.rank_norm = "per-class"
                rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
        elif args.rank_norm == "per-class":
            print(f"\nPer-class rank-norm (val)...")
            rank_normalise_per_class_inplace_val(val["scores"], val["classes"])
        elif args.rank_norm == "global":
            print(f"\nGlobal rank-norm (val)...")
            rank_normalise_global_inplace_val(val["scores"])
        else:
            print(f"\nSkipping rank-norm (val): --rank-norm none")
        mem_print("after val rank-norm")

        # XV aggregates on val (float16!)
        use_xv = (not args.no_xv_aggregates) and any_has_paths
        if (not args.no_xv_aggregates) and not any_has_paths:
            print(f"\n[warn] xv-aggregates disabled (no image_paths).")
        xv_max_val = xv_mean_val = xv_std_val = xv_lonely_val = None
        is_multi_xv_val = None
        if use_xv:
            print(f"\nBuilding cross-view AGGREGATES (val, float16)...")
            (xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val,
             is_multi_xv_val) = compute_xv_aggregates_val_f16(
                val["scores"], val["classes"], val.get("image_paths"))
            mem_print("after val xv aggregates (f16)")

        # Spatial priors
        classes = sorted(set(val["classes"].tolist()))
        spatial_priors: dict[str, np.ndarray] | None = None
        if not args.no_spatial_prior:
            print(f"\nLoading per-class spatial priors from "
                  f"{args.prior_heatmaps_dir}...")
            spatial_priors = load_spatial_priors(args.prior_heatmaps_dir, classes)

        # v5 fits
        mahal_per_class: dict[str, MahalanobisParams] | None = None
        if not args.no_mahalanobis:
            mahal_per_class = fit_mahalanobis_per_class(
                val["scores"], val["masks"], val["classes"], seed=args.seed)

        top_methods_per_class: dict[str, list[int]] | None = None
        if not (args.no_cc_features and args.no_per_method_zrank_top):
            top_methods_per_class = pick_top_methods_per_class(
                val["scores"], val["masks"], val["classes"],
                top_k=args.n_top_methods_for_cc)

        imgp99_stats: list[ImgP99Stats] | None = None
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

        # Build training matrices
        print(f"\nBuilding training matrices (featurize + sample negatives)...")
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
        mem_print("after build_training_data")

        # Now that training matrices contain everything we need from val
        # scores+xv arrays, we can free those (val itself stays for verify)
        if xv_max_val is not None:
            del xv_max_val, xv_mean_val, xv_std_val, xv_lonely_val
            gc.collect()
            mem_print("after dropping val xv arrays")

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
        params_per_class: dict | None = None
        tune_per_class_results: dict | None = None
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

        # OOF (per-class lifecycle: drop X_full after OOF + after prod fit)
        oof_per_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if tune_per_class_results is not None:
            for cls, r in tune_per_class_results.items():
                oof_per_class[cls] = (r["oof_preds"], r["oof_labels"])
        else:
            print(f"\nCollecting OOF preds per class...")
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
        calibrators_per_class: dict | None = None
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

        # Verify (uses val + OOF; we still hold val for this)
        verification: dict = {}
        if not args.no_verify and oof_per_class:
            verification = verify_stacker_vs_singles(
                val, method_names, oof_per_class)

        # Free val now that verification is done; keep only what we need
        # for the rest of main (classes list + masks already used).
        del val; gc.collect()
        mem_print("after val freed")

        # Fit production models, then DROP per-class X_full/X_train as we go
        print(f"\nFitting final per-class XGBoost models...")
        models = fit_per_class(training_data, params_global,
                                params_per_class, args.seed)
        # Now we can free training data (production models are already fit)
        if not args.no_free_class_data:
            for cls in list(training_data.keys()):
                if cls.startswith("_"): continue
                free_class_training_data(training_data, cls)
            gc.collect()
            mem_print("after free_class_training_data")
        # training_data itself only retains feature_names + small dicts now

        # Class map for test
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

        # Inference (memory-efficient streaming fuse)
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

        # Free subs after fusion
        del subs; gc.collect()
        mem_print("after fuse_test return + free subs")

        # Submission
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

        # OOF dump
        if oof_per_class:
            oof_path = run_dir / "oof_predictions.npz"
            to_save = {"classes": np.array(list(oof_per_class.keys()),
                                            dtype=object)}
            for cls, (op, ol) in oof_per_class.items():
                to_save[f"oof_preds_{cls}"] = op.astype(np.float32)
                to_save[f"oof_labels_{cls}"] = ol.astype(np.uint8)
            np.savez_compressed(oof_path, **to_save)
            print(f"Saved OOF preds -> {oof_path}")

        # Config dump
        model_dump = {
            "version": 6,
            "methods": method_names,
            "rank_norm": args.rank_norm,
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
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "verification": {
                k: v for k, v in (verification or {}).items()
                if k != "per_class_single"  # too big for JSON
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
        print(f"\nWrote stacker config -> {cfg_path}")

        # Top features
        if "_SHARED_" in models and "feature_importance" in models["_SHARED_"]:
            fi = np.asarray(models["_SHARED_"]["feature_importance"])
            if feature_names and len(feature_names) == len(fi):
                top = np.argsort(-fi)[:25]
                print(f"\nTop-25 feature importances (SHARED model):")
                for r, j in enumerate(top, 1):
                    print(f"  {r:>2d}. {feature_names[j]:<30s}  {fi[j]:.4f}")
                groups_fi: dict[str, float] = defaultdict(float)
                for name, imp in zip(feature_names, fi):
                    if name == "s_mahal":
                        groups_fi["v5_mahalanobis"] += float(imp)
                    elif name.startswith("xc_min_top"):
                        groups_fi["v5_min_top_k"] += float(imp)
                    elif name == "xc_cv":
                        groups_fi["v5_cross_cv"] += float(imp)
                    elif name.startswith("top") and "_cc_" in name:
                        groups_fi["v5_cc_features"] += float(imp)
                    elif name.startswith("top") and "_imgzrank" in name:
                        groups_fi["v5_imgzrank"] += float(imp)
                    elif name.startswith("m") and "_xv_" in name:
                        groups_fi["xv_aggregates"] += float(imp)
                    elif name == "s_prior":
                        groups_fi["spatial_prior"] += float(imp)
                    elif name.startswith("c_"):
                        groups_fi["class_onehot"] += float(imp)
                    elif name.startswith("v_"):
                        groups_fi["view_onehot"] += float(imp)
                    elif name.startswith("xc_"):
                        groups_fi["cross_method_consensus"] += float(imp)
                    elif name.startswith("x_"):
                        groups_fi["cross_method_stats"] += float(imp)
                    elif name.startswith("s_"):
                        groups_fi["spatial"] += float(imp)
                    else:
                        groups_fi["per_method_pruned"] += float(imp)
                print(f"\nFeature-family importance shares (SHARED):")
                tot = sum(groups_fi.values()) + 1e-12
                for fam in sorted(groups_fi, key=lambda k: -groups_fi[k]):
                    print(f"  {fam:<28s} {groups_fi[fam] / tot * 100:>6.2f}%")

        # Ablation row
        run_id = "stacker_xgb_v6_" + hashlib.sha1(
            "|".join(str(p) for p in args.runs).encode("utf-8")
        ).hexdigest()[:6]
        overall_stacker = verification.get("overall_stacker", float("nan"))
        pooled_stacker = verification.get("pooled_stacker_ap", float("nan"))
        notes = (f"xgb v6 | M={len(method_names)} | F={len(feature_names)} | "
                  f"rank_norm={args.rank_norm} | "
                  f"storage={args.test_storage_dtype} | "
                  f"mahal={int(cfg.use_mahalanobis)} | "
                  f"mintopk={int(cfg.use_min_top_k)} | "
                  f"cc={int(cfg.use_cc_features)} | "
                  f"zrank={int(cfg.use_per_method_zrank_top)} | "
                  f"cv={int(cfg.use_cross_method_cv)} | "
                  f"prior={int(cfg.use_spatial_prior)} | "
                  f"xv={int(cfg.use_xv_aggregates)} | "
                  f"tune={args.tune_mode} | calibrate={args.calibrate} | "
                  f"pooled_AP={pooled_stacker:.4f}")
        row = {
            "run_id": run_id, "run_tag": args.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "STACKER_XGB_V6",
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