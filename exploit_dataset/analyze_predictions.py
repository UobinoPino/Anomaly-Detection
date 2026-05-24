"""Spacepresso predictions analysis — v2.

Cross-model audit of `local_predictions.npz` outputs before stacker tuning.
Each section closes with a one-line takeaway. Designed to maximise the
*information* fed into the XGBoost stacker (xgboost_stacker_v3.py) while
exposing where the stacker should be told to NOT trust a model.

Sections (each writes both prose to stdout AND a CSV under tables/):

  Per-model audit
    1. Inventory                — N, shape, classes × types per file
    2. Score distributions      — at-positive vs at-negative percentiles per
                                   (model, class) → quantifies separability
    3. Calibration drift        — per-(model, class, view) percentile diff
                                   → which models speak louder on some views
    4. Sparsity vs prevalence   — frac pixels > threshold vs true positive rate
    5. Per-(model, class, type) AP — strength stratification

  Structural prior (drop dust / keep structure)
    6. CC filter AP gain        — sweep min-CC-area ∈ {0, 25, 50, 100, 200}
                                   and report AP delta per model

  Multi-view audit
    7. Cross-view score agreement — score variance across views of same sample;
                                     at GT-positive pixels vs at GT-negative

  Information-theoretic ladder
    8. MI: score → truth        — per (model, class, view), I(score_bin; truth)
                                   and U(truth | score). The "how much signal
                                   does this model carry, here?" answer.
    9. CMI: model redundancy    — pairwise I(A; truth | B). If small, A is
                                   redundant given B; drop or down-weight.

  Ensemble construction
   10. Cross-model agreement    — pixel-level Pearson + Spearman on POS and NEG
                                   separately. Low pos-agreement = diverse;
                                   high neg-agreement = same noise pattern.
   11. Greedy forward ensemble  — mean-rank fusion, greedy add-one. AP gain at
                                   each step; the curve tells you the best
                                   stacker-input subset.

  Final
   12. Stacker recipe           — recommendations consolidated from above.

Usage:
    python analyze_predictions.py \\
        --predictions \\
            wrn50_e5:/path/to/exp5/local_predictions.npz \\
            dnv2s14_e7:/path/to/exp7/local_predictions.npz \\
            cutpaste_8d:/path/to/exp8d/local_predictions.npz \\
            ... \\
        --output-dir /path/to/analysis_out

`name:path` pairs; keep names ≤ 14 chars for readable tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as ndi
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from sklearn.metrics import average_precision_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from scipy.stats import spearmanr
    HAS_SPEARMAN = True
except Exception:
    HAS_SPEARMAN = False


# ─────────────────────────────────────────────────────────────────────────────
# Image-path parsing — recover real view number from the file path
# (local_preds_saver stores enum_idx in `ids`, not the actual view N)
# ─────────────────────────────────────────────────────────────────────────────
PATH_VIEW_RE = re.compile(r"^(?P<sid>.+?)_view(?P<v>\d+)\.[A-Za-z]+$")


def parse_path(p: str) -> tuple[str, int | None]:
    """Returns (sample_id, view_number) from .../img_<id>_view<N>.png."""
    name = Path(p).name
    m = PATH_VIEW_RE.match(name)
    if m:
        return m.group("sid"), int(m.group("v"))
    return Path(p).stem, None


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers (same shape as the dataset script)
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
    try:
        yield
    finally:
        sys.stdout = old
        f.close()


def hr(t, c="="): print(f"\n{c * 78}\n  {t}\n{c * 78}")
def sub(t):       print(f"\n--- {t} ---")


def pct_one_line(arr, label: str, indent="  "):
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        print(f"{indent}{label}: (empty)"); return
    p = np.percentile(a, [25, 50, 75, 95])
    print(f"{indent}{label}: n={a.size}  μ={a.mean():.4g}  σ={a.std():.4g}  "
          f"p25={p[0]:.4g}  p50={p[1]:.4g}  p75={p[2]:.4g}  p95={p[3]:.4g}")


def save_csv(rows: list[dict], path: Path):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})


# ─────────────────────────────────────────────────────────────────────────────
# Information theory (same shape as analyze_spacepresso_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────
def mi_xy(joint: Counter,
          marg_x: Counter | None = None,
          marg_y: Counter | None = None) -> tuple[float, float, float]:
    N = sum(joint.values())
    if N == 0: return (0.0, 0.0, 0.0)
    if marg_x is None:
        marg_x = Counter()
        for (x, _), c in joint.items(): marg_x[x] += c
    if marg_y is None:
        marg_y = Counter()
        for (_, y), c in joint.items(): marg_y[y] += c
    mi = 0.0
    for (x, y), c in joint.items():
        if c == 0: continue
        p_xy = c / N
        p_x = marg_x[x] / N
        p_y = marg_y[y] / N
        if p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))
    hx = -sum((c / N) * math.log(c / N) for c in marg_x.values() if c > 0)
    hy = -sum((c / N) * math.log(c / N) for c in marg_y.values() if c > 0)
    return (mi, hx, hy)


def cmi_xyz(joint_xyz: Counter) -> tuple[float, float]:
    """Compute I(X; Y | Z) and H(Y | Z)."""
    N = sum(joint_xyz.values())
    if N == 0: return (0.0, 0.0)
    mz, mxz, myz = Counter(), Counter(), Counter()
    for (x, y, z), c in joint_xyz.items():
        mz[z] += c
        mxz[(x, z)] += c
        myz[(y, z)] += c
    cmi = 0.0
    for (x, y, z), c in joint_xyz.items():
        if c == 0: continue
        p_xyz = c / N
        p_z = mz[z] / N
        p_xz = mxz[(x, z)] / N
        p_yz = myz[(y, z)] / N
        if p_z > 0 and p_xz > 0 and p_yz > 0:
            cmi += p_xyz * math.log(p_xyz * p_z / (p_xz * p_yz))
    h_y_z = 0.0
    for (y, z), c_yz in myz.items():
        c_z = mz[z]
        if c_yz > 0 and c_z > 0:
            p_yz = c_yz / N
            p_y_given_z = c_yz / c_z
            h_y_z -= p_yz * math.log(p_y_given_z)
    return (cmi, h_y_z)


def quantile_bin(values: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Equal-frequency quantile binning. Returns int bin ids."""
    if values.size == 0: return values.astype(np.int8)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(values, qs))
    if edges.size == 0: return np.zeros_like(values, dtype=np.int8)
    return np.searchsorted(edges, values, side="right").astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Mean-pool sub-sampler for memory-bounded cross-model work
# ─────────────────────────────────────────────────────────────────────────────
def mean_pool_to(arr: np.ndarray, side: int = 64) -> np.ndarray:
    N, H, W = arr.shape
    bh = max(H // side, 1)
    bw = max(W // side, 1)
    Hc = bh * (H // bh); Wc = bw * (W // bw)
    # If H or W is smaller than side, just return as-is.
    if H < side or W < side:
        return arr.astype(np.float32, copy=False)
    cropped = arr[:, :bh * side, :bw * side].astype(np.float32, copy=False)
    return cropped.reshape(N, side, bh, side, bw).mean(axis=(2, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Connected-component helpers
# ─────────────────────────────────────────────────────────────────────────────
def cc_label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if mask.dtype != bool:
        mask = mask > 0
    if HAS_SCIPY:
        labels, n = ndi.label(mask)
        return labels, int(n)
    labels = np.zeros(mask.shape, dtype=np.int32); n = 0
    H, W = mask.shape
    for y in range(H):
        for x in range(W):
            if mask[y, x] and labels[y, x] == 0:
                n += 1; stack = [(y, x)]
                while stack:
                    yy, xx = stack.pop()
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and labels[yy, xx] == 0:
                        labels[yy, xx] = n
                        stack.extend([(yy+1, xx), (yy-1, xx), (yy, xx+1), (yy, xx-1)])
    return labels, n


def pixel_ap(scores: np.ndarray, gt: np.ndarray) -> float:
    """Pixel-level average precision."""
    y = gt.ravel().astype(np.int32)
    s = scores.ravel().astype(np.float32)
    if y.sum() == 0: return 0.0
    if HAS_SKLEARN:
        return float(average_precision_score(y, s))
    order = np.argsort(-s, kind="stable"); y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    p = tp / (tp + fp + 1e-12)
    r = tp / max(int(y.sum()), 1)
    r = np.concatenate([[0.0], r]); p = np.concatenate([[1.0], p])
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


# ─────────────────────────────────────────────────────────────────────────────
# npz loader
# ─────────────────────────────────────────────────────────────────────────────
def load_npz(path: Path) -> dict:
    """Loads a `local_predictions.npz` into a dict with parsed views."""
    data = np.load(path, allow_pickle=True)
    ids = np.asarray(data["ids"], dtype=object)
    classes = np.asarray(data["classes"], dtype=object)
    atypes = np.asarray(data["anomaly_types"], dtype=object)
    scores = np.asarray(data["scores"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=np.uint8)
    image_paths = (np.asarray(data["image_paths"], dtype=object)
                    if "image_paths" in data.files
                    else np.array(["" for _ in ids], dtype=object))
    sample_ids = []
    views = []
    for p in image_paths:
        sid, v = parse_path(str(p))
        sample_ids.append(sid)
        views.append(int(v) if v is not None else -1)
    return {
        "ids": ids, "classes": classes, "anomaly_types": atypes,
        "scores": scores, "masks": masks, "image_paths": image_paths,
        "sample_ids": np.asarray(sample_ids, dtype=object),
        "views": np.asarray(views, dtype=np.int32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Inventory
# ─────────────────────────────────────────────────────────────────────────────
def section_1_inventory(named_paths: list[tuple[str, Path]], output_dir: Path):
    hr("SECTION 1 — Inventory", "=")
    print(f"  {'model':<14} {'N':>5} {'H':>4} {'W':>4} "
          f"{'#classes':>9} {'#types':>7} {'pos_frac':>9}  file")
    rows = []
    for name, p in named_paths:
        try:
            d = load_npz(p)
        except Exception as e:
            print(f"  [WARN] {name}: failed to load ({e})"); continue
        N, H, W = d["scores"].shape
        n_cls = len(set(d["classes"].tolist()))
        n_type = len({(c, a) for c, a in zip(d["classes"], d["anomaly_types"])})
        pos_frac = float(d["masks"].mean())
        rows.append({"model": name, "N": N, "H": H, "W": W,
                     "n_classes": n_cls, "n_types": n_type,
                     "pos_frac": pos_frac, "path": str(p)})
        print(f"  {name:<14} {N:>5} {H:>4} {W:>4} "
              f"{n_cls:>9} {n_type:>7} {pos_frac:>9.5f}  {p.name}")
    save_csv(rows, output_dir / "tables" / "01_inventory.csv")
    print(f"  → tables/01_inventory.csv")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Score distributions per (model, class), at-pos vs at-neg
# ─────────────────────────────────────────────────────────────────────────────
def section_2_score_distributions(named_paths, output_dir, sample_pixels=2_000_000):
    hr("SECTION 2 — Score distributions per (model, class)", "=")
    rng = np.random.default_rng(0)
    rows = []
    print(f"  {'model':<14} {'class':<10} {'n_imgs':>7} "
          f"{'neg μ':>7} {'neg p95':>8} {'pos μ':>7} {'pos p50':>8} "
          f"{'sep':>7}")
    for name, p in named_paths:
        d = load_npz(p)
        classes_all = d["classes"]
        scores = d["scores"]; masks = d["masks"]
        for cls in sorted(set(classes_all.tolist())):
            sel = np.where(classes_all == cls)[0]
            if sel.size == 0: continue
            s_cls = scores[sel].ravel()
            m_cls = masks[sel].ravel()
            n_pos = int(m_cls.sum()); n_neg = int(m_cls.size - n_pos)
            if n_pos == 0: continue
            # subsample negatives (they dominate memory)
            neg_idx = np.where(m_cls == 0)[0]
            pos_idx = np.where(m_cls > 0)[0]
            if neg_idx.size > sample_pixels:
                neg_idx = rng.choice(neg_idx, sample_pixels, replace=False)
            if pos_idx.size > sample_pixels:
                pos_idx = rng.choice(pos_idx, sample_pixels, replace=False)
            s_neg = s_cls[neg_idx]; s_pos = s_cls[pos_idx]
            neg_mu = float(s_neg.mean()); neg_p95 = float(np.percentile(s_neg, 95))
            pos_mu = float(s_pos.mean()); pos_p50 = float(np.percentile(s_pos, 50))
            # separability: (pos_p50 − neg_p95) / (neg_std + 1e-6); higher is better
            sep = (pos_p50 - neg_p95) / (float(s_neg.std()) + 1e-6)
            rows.append({"model": name, "class": cls, "n_imgs": int(sel.size),
                         "neg_mean": neg_mu, "neg_p95": neg_p95,
                         "pos_mean": pos_mu, "pos_p50": pos_p50,
                         "separability": sep})
            print(f"  {name:<14} {cls:<10} {sel.size:>7} "
                  f"{neg_mu:>7.3f} {neg_p95:>8.3f} "
                  f"{pos_mu:>7.3f} {pos_p50:>8.3f} {sep:>7.3f}")
    save_csv(rows, output_dir / "tables" / "02_score_distributions.csv")
    print(f"  TAKEAWAY: sep ≥ 2 = clean separation; sep ≤ 0.5 = weak; sep < 0 "
          f"= model is harmful for that class. Stacker should down-weight cells "
          f"with sep < 0.5.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Calibration drift per (model, class, view)
# (the "one view always speaks louder" problem)
# ─────────────────────────────────────────────────────────────────────────────
def section_3_calibration_drift(named_paths, output_dir):
    hr("SECTION 3 — Calibration drift per (model, class, view)", "=")
    rows = []
    print(f"  Δ95 = max(class,view p95) − min(class,view p95) per (model, class).")
    print(f"  Large Δ95 ⇒ per-(class, view) rank-normalisation is mandatory.")
    print(f"  {'model':<14} {'class':<10} {'views':>5} "
          f"{'p95 μ':>7} {'Δp95':>7} {'Δp99':>7} {'rec':>3}")
    for name, p in named_paths:
        d = load_npz(p)
        classes_all = d["classes"]
        views = d["views"]
        scores = d["scores"]
        for cls in sorted(set(classes_all.tolist())):
            views_seen = sorted({int(v) for v in views[classes_all == cls] if v >= 0})
            if not views_seen: continue
            p95s = []; p99s = []
            for v in views_seen:
                sel = np.where((classes_all == cls) & (views == v))[0]
                if sel.size == 0: continue
                s = scores[sel].ravel()
                p95s.append(float(np.percentile(s, 95)))
                p99s.append(float(np.percentile(s, 99)))
            if not p95s: continue
            p95_mu = float(np.mean(p95s))
            d95 = float(max(p95s) - min(p95s))
            d99 = float(max(p99s) - min(p99s))
            rec = "‼" if d95 / max(p95_mu, 1e-6) > 0.30 else ("·" if d95 / max(p95_mu, 1e-6) > 0.10 else "ok")
            rows.append({"model": name, "class": cls,
                         "n_views": len(views_seen),
                         "p95_mean": p95_mu, "delta_p95": d95, "delta_p99": d99,
                         "rec": rec})
            print(f"  {name:<14} {cls:<10} {len(views_seen):>5} "
                  f"{p95_mu:>7.3f} {d95:>7.3f} {d99:>7.3f} {rec:>3}")
    save_csv(rows, output_dir / "tables" / "03_calibration_drift.csv")
    print(f"  TAKEAWAY: rows marked ‼ HAVE per-view scale mismatch. The stacker "
          f"MUST rank-norm per (class, view); global rank-norm is not enough. "
          f"This corresponds to the per-view LUT path in xgboost_stacker_v3.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Sparsity vs prevalence
# ─────────────────────────────────────────────────────────────────────────────
def section_4_sparsity(named_paths, output_dir, thresholds=(0.3, 0.5, 0.7, 0.9)):
    hr("SECTION 4 — Sparsity (frac pixels > τ) vs prevalence", "=")
    print(f"  {'model':<14} {'prev':>7}  " +
          "  ".join(f"τ={t:.1f}" for t in thresholds) +
          "  bloat@0.5")
    rows = []
    for name, p in named_paths:
        d = load_npz(p)
        scores = d["scores"]; masks = d["masks"]
        prev = float(masks.mean())
        fracs = [float((scores >= t).mean()) for t in thresholds]
        bloat = fracs[1] / prev if prev > 0 else 0.0
        rows.append({"model": name, "prevalence": prev,
                     **{f"frac_ge_{t:.1f}": f for t, f in zip(thresholds, fracs)},
                     "bloat_at_0.5": bloat})
        print(f"  {name:<14} {prev:>7.5f}  " +
              "  ".join(f"{f:>5.3f}" for f in fracs) +
              f"   {bloat:>6.1f}×")
    save_csv(rows, output_dir / "tables" / "04_sparsity.csv")
    print(f"  TAKEAWAY: bloat > 50× ⇒ model produces a fog of low-confidence "
          f"activations. Strong candidate for CC-area dropping (§6) and/or "
          f"top-percentile thresholding before stacker ingestion.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Per-(model, class, anomaly_type) AP
# ─────────────────────────────────────────────────────────────────────────────
def section_5_per_type_ap(named_paths, output_dir):
    hr("SECTION 5 — Per-(model, class, anomaly_type) AP", "=")
    rows = []
    by_model: dict[str, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, p in named_paths:
        d = load_npz(p)
        classes_all = d["classes"]; atypes = d["anomaly_types"]
        scores = d["scores"]; masks = d["masks"]
        for i in range(scores.shape[0]):
            ap = pixel_ap(scores[i], masks[i])
            by_model[name][(classes_all[i], atypes[i])].append(ap)
    # Class-level + overall
    model_class_ap: dict[tuple[str, str], float] = {}
    model_overall: dict[str, float] = {}
    classes_seen = sorted({k[0] for d in by_model.values() for k in d})
    for name in by_model:
        per_cls_means = []
        for cls in classes_seen:
            type_means = []
            for (c, a), aps in by_model[name].items():
                if c != cls: continue
                if not aps: continue
                m = float(np.mean(aps))
                type_means.append(m)
                rows.append({"model": name, "class": c, "anomaly_type": a,
                             "n_views": len(aps),
                             "ap_mean": m, "ap_std": float(np.std(aps))})
            if type_means:
                cls_mean = float(np.mean(type_means))
                model_class_ap[(name, cls)] = cls_mean
                per_cls_means.append(cls_mean)
        if per_cls_means:
            model_overall[name] = float(np.mean(per_cls_means))
    save_csv(rows, output_dir / "tables" / "05_per_type_ap.csv")

    # Print compact wide table: rows=models, columns=classes, last col=overall
    print(f"  {'model':<14} " +
          " ".join(f"{c:>8}" for c in classes_seen) +
          "  OVERALL")
    for name in sorted(by_model):
        cells = [f"{model_class_ap.get((name, c), float('nan')):>8.4f}" for c in classes_seen]
        ov = model_overall.get(name, float('nan'))
        print(f"  {name:<14} " + " ".join(cells) + f"  {ov:>7.4f}")
    print(f"  TAKEAWAY: best-per-class model varies → stacker should learn "
          f"per-class weights (per-class XGBoost, not one global model).")
    return {"model_class_ap": model_class_ap, "model_overall": model_overall,
            "classes": classes_seen, "by_model": by_model}


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — CC filter AP gain (drop dust / keep structure)
# ─────────────────────────────────────────────────────────────────────────────
def section_6_cc_filter(named_paths, output_dir,
                         min_areas=(0, 25, 50, 100, 200),
                         pct_thresh=98.0, max_imgs_per_model=400):
    hr("SECTION 6 — Connected-component filter AP gain", "=")
    print(f"  For each model: threshold scores at p{pct_thresh}, label CCs, drop")
    print(f"  CCs below min_area, then mean-fill the dropped pixels with the")
    print(f"  global minimum. Report AP per min_area. The min_area that beats")
    print(f"  baseline (col 0) is the drop-dust setting to apply pre-stacker.")
    print(f"  {'model':<14}  " +
          "  ".join(f"area≥{a:>3}" for a in min_areas) +
          "    best")
    rng = np.random.default_rng(0)
    rows = []
    for name, p in named_paths:
        d = load_npz(p)
        scores = d["scores"].astype(np.float32)
        masks = d["masks"].astype(np.uint8)
        # subsample images for speed
        idx = np.arange(scores.shape[0])
        if idx.size > max_imgs_per_model:
            idx = rng.choice(idx, max_imgs_per_model, replace=False)
        scores_sub = scores[idx]; masks_sub = masks[idx]
        # compute thresh per image (p98 of that image's scores)
        aps_by_area = {a: [] for a in min_areas}
        for i in range(scores_sub.shape[0]):
            if masks_sub[i].sum() == 0: continue
            s = scores_sub[i]
            for a in min_areas:
                if a == 0:
                    aps_by_area[a].append(pixel_ap(s, masks_sub[i])); continue
                thr = float(np.percentile(s, pct_thresh))
                mask_high = s >= thr
                labels, ncc = cc_label(mask_high)
                if ncc == 0:
                    aps_by_area[a].append(pixel_ap(s, masks_sub[i])); continue
                # area per CC
                areas = np.bincount(labels.ravel())  # areas[0] = bg
                keep = np.zeros_like(labels, dtype=bool)
                for k in range(1, ncc + 1):
                    if areas[k] >= a:
                        keep |= (labels == k)
                # Build cleaned score: keep original where keep, else mean of low
                s_clean = np.where(keep, s, s.min())
                aps_by_area[a].append(pixel_ap(s_clean, masks_sub[i]))
        means = {a: float(np.mean(v)) if v else float("nan") for a, v in aps_by_area.items()}
        best_a = max(means, key=lambda k: means[k])
        rows.append({"model": name, **{f"ap_area_{a}": means[a] for a in min_areas},
                     "best_min_area": best_a, "best_ap": means[best_a]})
        cells = "  ".join(f"{means[a]:>7.4f}" for a in min_areas)
        print(f"  {name:<14}  {cells}    area≥{best_a:>3}")
    save_csv(rows, output_dir / "tables" / "06_cc_filter.csv")
    print(f"  TAKEAWAY: pre-stacker, apply CC-min-area = best_min_area per model "
          f"to scores. This is structural denoising: 'drop what's too small'.")
    return {"rows": rows}


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Cross-view score agreement (multi-view consistency, predictions)
# ─────────────────────────────────────────────────────────────────────────────
def section_7_crossview_score(named_paths, output_dir, side=64):
    hr("SECTION 7 — Cross-view score agreement (predictions)", "=")
    print(f"  For each sample with ≥2 views, mean-pool to {side}x{side} and")
    print(f"  compute pairwise Pearson between views. Done on POSITIVE pixels")
    print(f"  vs NEGATIVE pixels separately — high pos-agreement = the model")
    print(f"  has coherent multi-view evidence; low neg-agreement = noise is")
    print(f"  decorrelated (good for stacker variance feature).")
    print(f"  {'model':<14}  "
          f"{'pos μ':>7} {'pos p50':>8} {'pos p25':>8}  "
          f"{'neg μ':>7} {'neg p50':>8}  {'pos>0.5':>8}")
    rows = []
    for name, p in named_paths:
        d = load_npz(p)
        sample_ids = d["sample_ids"]; classes_all = d["classes"]
        atypes = d["anomaly_types"]
        scores = d["scores"]; masks = d["masks"]
        # group by sample
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, (cls, atype, sid) in enumerate(zip(classes_all, atypes, sample_ids)):
            groups[(cls, atype, sid)].append(i)
        groups_multi = {k: v for k, v in groups.items() if len(v) >= 2}
        if not groups_multi:
            print(f"  {name:<14}  (no multi-view samples)"); continue

        scores_lo = mean_pool_to(scores, side=side)
        masks_lo  = mean_pool_to(masks.astype(np.float32), side=side)
        masks_bin = (masks_lo > 0.5).astype(np.uint8)

        pos_corrs = []; neg_corrs = []
        for key, idxs in groups_multi.items():
            # Cross-pair on positives and negatives
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    pos_pix = np.where(masks_bin[a] | masks_bin[b])
                    neg_pix = np.where((masks_bin[a] | masks_bin[b]) == 0)
                    if len(pos_pix[0]) > 4:
                        sa = scores_lo[a][pos_pix]; sb = scores_lo[b][pos_pix]
                        if sa.std() > 0 and sb.std() > 0:
                            pos_corrs.append(float(np.corrcoef(sa, sb)[0, 1]))
                    if len(neg_pix[0]) > 64:
                        neg_idx = np.random.default_rng(0).choice(len(neg_pix[0]),
                                                                    min(1024, len(neg_pix[0])),
                                                                    replace=False)
                        sa = scores_lo[a][neg_pix[0][neg_idx], neg_pix[1][neg_idx]]
                        sb = scores_lo[b][neg_pix[0][neg_idx], neg_pix[1][neg_idx]]
                        if sa.std() > 0 and sb.std() > 0:
                            neg_corrs.append(float(np.corrcoef(sa, sb)[0, 1]))
        if not pos_corrs:
            print(f"  {name:<14}  (no positive pixels)"); continue
        pos_arr = np.asarray(pos_corrs); neg_arr = np.asarray(neg_corrs) if neg_corrs else np.array([0.0])
        rows.append({"model": name,
                     "n_pairs_pos": len(pos_corrs), "n_pairs_neg": len(neg_corrs),
                     "pos_mean": float(pos_arr.mean()),
                     "pos_p50": float(np.percentile(pos_arr, 50)),
                     "pos_p25": float(np.percentile(pos_arr, 25)),
                     "neg_mean": float(neg_arr.mean()),
                     "neg_p50": float(np.percentile(neg_arr, 50)),
                     "p_pos_corr_above_05": float((pos_arr > 0.5).mean())})
        print(f"  {name:<14}  "
              f"{pos_arr.mean():>7.3f} {np.percentile(pos_arr, 50):>8.3f} "
              f"{np.percentile(pos_arr, 25):>8.3f}  "
              f"{neg_arr.mean():>7.3f} {np.percentile(neg_arr, 50):>8.3f}  "
              f"{float((pos_arr > 0.5).mean()):>8.3f}")
    save_csv(rows, output_dir / "tables" / "07_crossview_score.csv")
    print(f"  TAKEAWAY: pos μ ≫ neg μ ⇒ multi-view AGREEMENT is a strong signal "
          f"for this model ⇒ add cross-view mean / max / std as stacker features. "
          f"If pos μ ≈ neg μ, the model's view-consistency is noise; use max.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — MI: score → truth, per (model, class, view)
# ─────────────────────────────────────────────────────────────────────────────
def section_8_mi_score_truth(named_paths, output_dir,
                              n_bins=8, sample_pixels=500_000, seed=0):
    hr("SECTION 8 — MI: score → truth per (model, class, view)", "=")
    print(f"  I(score_bin; truth) measures how much the model's score predicts")
    print(f"  the pixel label. U(truth | score) = MI / H(truth) is the fraction")
    print(f"  of label entropy explained — directly comparable across rows.")
    rng = np.random.default_rng(seed)
    rows = []
    # Per (model, class) MI; per (model, class, view) U
    summary_rows = []
    print(f"  {'model':<14} {'class':<10} {'view':>4}  "
          f"{'n':>6}  {'MI':>6} {'H(Y)':>6} {'U(Y|X)':>8}")
    for name, p in named_paths:
        d = load_npz(p)
        classes_all = d["classes"]; views = d["views"]
        scores = d["scores"]; masks = d["masks"]
        cls_pool: dict[str, dict] = defaultdict(lambda: {"u": [], "mi": []})
        for cls in sorted(set(classes_all.tolist())):
            views_seen = sorted({int(v) for v in views[classes_all == cls] if v >= 0})
            if not views_seen:
                views_seen = [-1]  # treat as single bucket
            for v in views_seen:
                if v == -1:
                    sel = np.where(classes_all == cls)[0]
                else:
                    sel = np.where((classes_all == cls) & (views == v))[0]
                if sel.size == 0: continue
                s = scores[sel].ravel()
                y = masks[sel].ravel().astype(np.uint8)
                if y.sum() == 0: continue
                # balance: subsample negatives
                if s.size > sample_pixels:
                    pos_idx = np.where(y > 0)[0]
                    neg_idx = np.where(y == 0)[0]
                    if neg_idx.size > sample_pixels - pos_idx.size:
                        neg_idx = rng.choice(neg_idx,
                                              max(sample_pixels - pos_idx.size, 1),
                                              replace=False)
                    keep = np.concatenate([pos_idx, neg_idx])
                    s = s[keep]; y = y[keep]
                s_bin = quantile_bin(s, n_bins=n_bins)
                joint = Counter(zip(s_bin.tolist(), y.tolist()))
                mi, hx, hy = mi_xy(joint)
                u = mi / hy if hy > 0 else 0.0
                cls_pool[cls]["u"].append(u); cls_pool[cls]["mi"].append(mi)
                summary_rows.append({"model": name, "class": cls, "view": v,
                                     "n_pixels": int(y.size),
                                     "MI": mi, "H_Y": hy, "U": u})
                if v != -1 or len(views_seen) == 1:
                    print(f"  {name:<14} {cls:<10} {v:>4}  {y.size:>6}  "
                          f"{mi:>6.4f} {hy:>6.4f} {u:>8.4f}")
        # Aggregated row per (model, class)
        for cls, vals in cls_pool.items():
            rows.append({"model": name, "class": cls,
                         "U_mean": float(np.mean(vals["u"])),
                         "U_min":  float(np.min(vals["u"])),
                         "U_max":  float(np.max(vals["u"]))})
    save_csv(summary_rows, output_dir / "tables" / "08_mi_score_truth_view.csv")
    save_csv(rows, output_dir / "tables" / "08_mi_score_truth_class.csv")
    print(f"  TAKEAWAY: cells with U < 0.005 carry negligible signal; the stacker")
    print(f"  should not include that model's score for that (class, view) at all.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — CMI: pairwise model redundancy
# I(A; truth | B) per class: does A add information beyond B?
# Heavy section — use mean-pooled scores for tractability.
# ─────────────────────────────────────────────────────────────────────────────
def section_9_redundancy(named_paths, output_dir,
                          side=64, n_bins=4, max_pixels_per_class=2_000_000):
    hr("SECTION 9 — Pairwise CMI: model redundancy", "=")
    if len(named_paths) < 2:
        print("  [SKIP] need ≥ 2 models"); return None
    # Build a registry by (cls, atype, sid, view) → row idx per model.
    # Use mean-pooled scores and masks to keep memory manageable.
    print(f"  Loading and mean-pooling all models to {side}x{side}...")
    pooled: dict[str, dict] = {}
    for name, p in named_paths:
        d = load_npz(p)
        s = mean_pool_to(d["scores"], side=side)
        m = mean_pool_to(d["masks"].astype(np.float32), side=side) > 0.5
        pooled[name] = {
            "scores": s, "masks": m, "classes": d["classes"],
            "atypes": d["anomaly_types"], "sample_ids": d["sample_ids"],
            "views": d["views"],
        }
        print(f"    {name:<14}  scores {s.shape}, ~{s.nbytes/1e6:.1f} MB")

    # Find row alignment across models (need same (cls, atype, sid, view))
    # Build a canonical key list as the intersection.
    def key_set(d):
        return {(d["classes"][i], d["atypes"][i], d["sample_ids"][i], int(d["views"][i]))
                for i in range(d["scores"].shape[0])}
    keys_common = None
    for name in pooled:
        ks = key_set(pooled[name])
        keys_common = ks if keys_common is None else keys_common & ks
    if not keys_common:
        print("  [WARN] no common (cls, atype, sample, view) keys across models")
        return None
    keys_common = sorted(keys_common)
    print(f"  common keys across models : {len(keys_common)}")

    # Per-model row indexer for the common key list
    key_to_idx: dict[str, dict] = {}
    for name in pooled:
        m = {}
        d = pooled[name]
        for i in range(d["scores"].shape[0]):
            k = (d["classes"][i], d["atypes"][i], d["sample_ids"][i], int(d["views"][i]))
            m[k] = i
        key_to_idx[name] = m

    # Compute, per class, per pair of models (A, B): I(A; truth | B)
    classes_seen = sorted({k[0] for k in keys_common})
    model_names = [n for n, _ in named_paths if n in pooled]
    rng = np.random.default_rng(0)
    rows = []
    pairs = [(a, b) for a in model_names for b in model_names if a != b]
    print(f"\n  Per-class CMI table: I(A; truth | B). Read each cell as "
          f"'A adds info beyond B'.")
    for cls in classes_seen:
        cls_keys = [k for k in keys_common if k[0] == cls]
        if not cls_keys: continue
        # Gather flattened scores and masks per model for this class
        per_model = {}
        for name in model_names:
            idxs = [key_to_idx[name][k] for k in cls_keys]
            per_model[name] = {
                "s": pooled[name]["scores"][idxs].ravel(),
                "m": pooled[name]["masks"][idxs].ravel().astype(np.uint8),
            }
        truth = per_model[model_names[0]]["m"]  # same per key
        # Subsample
        if truth.size > max_pixels_per_class:
            pos_idx = np.where(truth > 0)[0]
            neg_idx = np.where(truth == 0)[0]
            neg_keep = max(max_pixels_per_class - pos_idx.size, 1)
            if neg_idx.size > neg_keep:
                neg_idx = rng.choice(neg_idx, neg_keep, replace=False)
            keep = np.concatenate([pos_idx, neg_idx])
            for name in model_names:
                per_model[name]["s"] = per_model[name]["s"][keep]
            truth = truth[keep]
        bins = {name: quantile_bin(per_model[name]["s"], n_bins=n_bins)
                for name in model_names}
        # H(Y | B) baseline
        sub(f"class={cls}  n_pixels={truth.size}")
        header = ("    " + f"{'A \\ B':<14}" +
                  "".join(f"{b:>10}" for b in model_names))
        print(header)
        for a in model_names:
            cells = []
            for b in model_names:
                if a == b:
                    cells.append("—".rjust(10))
                    continue
                # I(A; Y | B) = sum over (a, y, b) of p(a,y,b) log [p(a,y|b)/(p(a|b)p(y|b))]
                joint = Counter()
                for i in range(truth.size):
                    joint[(int(bins[a][i]), int(truth[i]), int(bins[b][i]))] += 1
                cmi, hyz = cmi_xyz(joint)
                # Display in nats, but also store the RELATIVE quantity below.
                cells.append(f"{cmi:>10.4f}")
                rows.append({"class": cls, "A": a, "B": b,
                             "CMI": cmi, "H_Y_given_B": hyz,
                             "rel": cmi / hyz if hyz > 0 else 0.0})
            print(f"    {a:<14}" + "".join(cells))
    save_csv(rows, output_dir / "tables" / "09_redundancy_cmi.csv")
    print(f"  TAKEAWAY: row A all-near-zero ⇒ A is DOMINATED by other models, "
          f"drop or down-weight in stacker. Column B near zero ⇒ B IS the "
          f"workhorse for that class; do not exclude it.")
    return {"rows": rows, "classes": classes_seen, "models": model_names,
            "pooled": pooled, "keys_common": keys_common,
            "key_to_idx": key_to_idx}


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Cross-model agreement (Pearson, Spearman) on pos vs neg
# ─────────────────────────────────────────────────────────────────────────────
def section_10_cross_model_agreement(s9, output_dir, max_pixels=200_000, seed=0):
    hr("SECTION 10 — Cross-model agreement", "=")
    if s9 is None:
        print("  [SKIP] need §9 data"); return
    pooled = s9["pooled"]; keys_common = s9["keys_common"]; key_to_idx = s9["key_to_idx"]
    model_names = s9["models"]
    # Pool scores & truth across the common keys
    rng = np.random.default_rng(seed)
    truth_full = []
    per_model: dict[str, list[np.ndarray]] = {n: [] for n in model_names}
    for k in keys_common:
        i0 = key_to_idx[model_names[0]][k]
        truth_full.append(pooled[model_names[0]]["masks"][i0].ravel())
        for n in model_names:
            per_model[n].append(pooled[n]["scores"][key_to_idx[n][k]].ravel())
    truth = np.concatenate(truth_full).astype(np.uint8)
    per_model_arr = {n: np.concatenate(v).astype(np.float32) for n, v in per_model.items()}

    pos_idx = np.where(truth > 0)[0]; neg_idx = np.where(truth == 0)[0]
    if pos_idx.size > max_pixels:
        pos_idx = rng.choice(pos_idx, max_pixels, replace=False)
    if neg_idx.size > max_pixels:
        neg_idx = rng.choice(neg_idx, max_pixels, replace=False)

    def _corr_matrix(idx, mode):
        n = len(model_names); M = np.zeros((n, n)); S = np.zeros((n, n))
        vals = {name: per_model_arr[name][idx] for name in model_names}
        for i, a in enumerate(model_names):
            for j, b in enumerate(model_names):
                if i == j:
                    M[i, j] = 1.0; S[i, j] = 1.0; continue
                va = vals[a]; vb = vals[b]
                if va.std() > 0 and vb.std() > 0:
                    M[i, j] = float(np.corrcoef(va, vb)[0, 1])
                if HAS_SPEARMAN:
                    try:
                        S[i, j] = float(spearmanr(va, vb).correlation)
                    except Exception:
                        S[i, j] = float("nan")
        return M, S

    print(f"  Pearson ρ on POSITIVE pixels (n={pos_idx.size}):")
    Mp, Sp = _corr_matrix(pos_idx, "pos")
    print(f"    {'':<14}" + " ".join(f"{n:>10}" for n in model_names))
    for i, a in enumerate(model_names):
        print(f"    {a:<14}" + " ".join(f"{Mp[i, j]:>10.3f}" for j in range(len(model_names))))

    print(f"\n  Pearson ρ on NEGATIVE pixels (n={neg_idx.size}):")
    Mn, Sn = _corr_matrix(neg_idx, "neg")
    print(f"    {'':<14}" + " ".join(f"{n:>10}" for n in model_names))
    for i, a in enumerate(model_names):
        print(f"    {a:<14}" + " ".join(f"{Mn[i, j]:>10.3f}" for j in range(len(model_names))))

    rows = []
    for i, a in enumerate(model_names):
        for j, b in enumerate(model_names):
            rows.append({"A": a, "B": b,
                         "pearson_pos": Mp[i, j], "pearson_neg": Mn[i, j],
                         "spearman_pos": Sp[i, j], "spearman_neg": Sn[i, j],
                         "diversity_score": float(Mp[i, j] - Mn[i, j])})
    save_csv(rows, output_dir / "tables" / "10_cross_model_agreement.csv")
    print(f"  TAKEAWAY: high pos-ρ + low neg-ρ ⇒ models AGREE on real defects "
          f"but DECORRELATED on noise — ideal stacker pair. Low pos-ρ ⇒ disagree "
          f"on truth (bad). Keep the small set with maximum diversity_score.")
    return {"M_pos": Mp, "M_neg": Mn, "model_names": model_names}


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 — Greedy forward ensemble (mean-rank fusion)
# ─────────────────────────────────────────────────────────────────────────────
def section_11_greedy_ensemble(s5, named_paths, output_dir,
                                side=64, max_pixels_per_class=1_000_000):
    hr("SECTION 11 — Greedy forward ensemble (mean-rank fusion)", "=")
    if len(named_paths) < 2:
        print("  [SKIP] need ≥ 2 models"); return
    # Strategy: start with the best-overall model, greedily add the one that
    # maximises mean-rank fusion AP. Use per-image rank within each class
    # (so we're not punishing the model with smaller dynamic range).
    print(f"  Building mean-pooled rank stacks at {side}x{side}...")
    pooled: dict[str, dict] = {}
    for name, p in named_paths:
        d = load_npz(p)
        # Per-image rank: rank within each image (so each model gets one vote)
        s = mean_pool_to(d["scores"], side=side)
        ranks = np.empty_like(s, dtype=np.float32)
        for i in range(s.shape[0]):
            flat = s[i].ravel()
            order = np.argsort(np.argsort(flat))  # ranks 0..n-1
            ranks[i] = order.reshape(s[i].shape).astype(np.float32) / max(flat.size - 1, 1)
        m = mean_pool_to(d["masks"].astype(np.float32), side=side) > 0.5
        pooled[name] = {"ranks": ranks, "masks": m,
                         "classes": d["classes"], "atypes": d["anomaly_types"],
                         "sample_ids": d["sample_ids"], "views": d["views"]}

    # Key alignment
    def key_set(d):
        return {(d["classes"][i], d["atypes"][i], d["sample_ids"][i], int(d["views"][i]))
                for i in range(d["ranks"].shape[0])}
    keys_common = None
    for name in pooled:
        ks = key_set(pooled[name])
        keys_common = ks if keys_common is None else keys_common & ks
    keys_common = sorted(keys_common)
    print(f"  common keys: {len(keys_common)}")
    key_to_idx = {n: {} for n in pooled}
    for n in pooled:
        d = pooled[n]
        for i in range(d["ranks"].shape[0]):
            k = (d["classes"][i], d["atypes"][i], d["sample_ids"][i], int(d["views"][i]))
            key_to_idx[n][k] = i

    classes_seen = sorted({k[0] for k in keys_common})
    model_names = [n for n, _ in named_paths if n in pooled]

    def ensemble_ap(active: list[str]) -> dict[str, float]:
        ap_per_cls = {}
        for cls in classes_seen:
            cls_keys = [k for k in keys_common if k[0] == cls]
            if not cls_keys: continue
            # Mean rank across active models, per pixel
            acc = np.zeros((len(cls_keys), side, side), dtype=np.float32)
            for n in active:
                idxs = [key_to_idx[n][k] for k in cls_keys]
                acc += pooled[n]["ranks"][idxs]
            acc /= max(len(active), 1)
            masks_arr = pooled[active[0]]["masks"][[key_to_idx[active[0]][k] for k in cls_keys]]
            aps = []
            for i in range(len(cls_keys)):
                if masks_arr[i].sum() == 0: continue
                aps.append(pixel_ap(acc[i], masks_arr[i]))
            if aps: ap_per_cls[cls] = float(np.mean(aps))
        return ap_per_cls

    # Start with best individual model
    overall = s5["model_overall"]
    seed_model = max((n for n in model_names if n in overall), key=lambda n: overall[n])
    active = [seed_model]
    history = []
    print(f"\n  Greedy forward selection — seed: {seed_model} (AP_overall={overall[seed_model]:.4f})")
    print(f"  {'step':>4} {'added':<14} {'AP_overall':>10}  {'Δ':>7}")
    cur_ap = float(np.mean(list(ensemble_ap(active).values())))
    history.append({"step": 0, "added": seed_model, "AP_overall": cur_ap, "delta": 0.0})
    print(f"  {0:>4} {seed_model:<14} {cur_ap:>10.4f}  {'—':>7}")
    candidates = [n for n in model_names if n != seed_model]
    step = 1
    while candidates:
        best = None; best_delta = -1e9; best_ap = None
        for n in candidates:
            ap = float(np.mean(list(ensemble_ap(active + [n]).values())))
            d = ap - cur_ap
            if d > best_delta:
                best_delta = d; best = n; best_ap = ap
        if best is None: break
        history.append({"step": step, "added": best, "AP_overall": best_ap,
                         "delta": best_delta})
        sign = "+" if best_delta >= 0 else "-"
        print(f"  {step:>4} {best:<14} {best_ap:>10.4f}  {sign}{abs(best_delta):>6.4f}")
        active.append(best); candidates.remove(best); cur_ap = best_ap
        step += 1
    save_csv(history, output_dir / "tables" / "11_greedy_ensemble.csv")
    # Recommend cutoff: stop including models whose addition delta < 0.001
    keep = [history[0]["added"]]
    for h in history[1:]:
        if h["delta"] >= 0.001:
            keep.append(h["added"])
        else:
            break
    print(f"\n  RECOMMENDED stacker INPUT SUBSET ({len(keep)} models): {', '.join(keep)}")
    print(f"  TAKEAWAY: feed only these to the stacker. Adding more is more "
          f"parameters for the stacker without information gain.")
    return {"history": history, "recommended": keep}


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 — Stacker recipe (final consolidation)
# ─────────────────────────────────────────────────────────────────────────────
def section_12_recipe(s4, s6, s7, s11, output_dir):
    hr("SECTION 12 — Stacker recipe (final)", "=")
    print("  ── INPUT MODELS ─────────────────────────────────────────────────")
    if s11 and "recommended" in s11:
        print("  • Use the subset recommended by §11:")
        for n in s11["recommended"]:
            print(f"      - {n}")
        print("  • Other models from your registry can be omitted; their information")
        print("    is already covered by the subset (§9 CMI confirms it).")
    print()
    print("  ── PER-MODEL PRE-PROCESSING ─────────────────────────────────────")
    if s4:
        print("  • Apply CC-min-area drop using §6 best_min_area per model:")
        print("      see tables/06_cc_filter.csv")
    print("  • Apply Gaussian smoothing σ = 1.5 (already in patchcore baseline).")
    print("  • Per-(class, view) empirical-CDF rank normalisation when §3 flags")
    print("    Δp95/p95_mean > 0.30. Fit LUTs on train/good (same shape as the")
    print("    consensus-v3 path in patchcore_baseline_v2.py).")
    print()
    print("  ── STACKER FEATURES (per pixel, in addition to raw scores) ──────")
    print("  • Class one-hot (8 dims)")
    print("  • Anomaly_type one-hot (where known at train time)")
    print("  • view_index one-hot (5 dims)")
    print("  • Per-class spatial-prior heatmap value (from dataset §6 npy)")
    print("  • CC features (computed at p98 threshold of each model's score):")
    print("      - log(CC area)  /  fill ratio  /  compactness")
    print("      - 1[area < min_area] (drop-dust flag)")
    print("  • Multi-view aggregation features (when ≥ 2 views available):")
    print("      - cross-view max score")
    print("      - cross-view mean score")
    print("      - cross-view std (uncertainty)")
    if s7:
        print("    Justified by §7: pos μ > neg μ ⇒ agreement features are reliable.")
    print()
    print("  ── STACKER TRAINING ─────────────────────────────────────────────")
    print("  • Use xgboost_stacker_v3 with --rank-norm per-class for the per-")
    print("    (class, view) cells flagged in §3.")
    print("  • LOAO CV: per-class objective is the right thing because per-class")
    print("    AP varies wildly (§5).")
    print()
    print(f"  Tables saved under: {output_dir / 'tables'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_named(arg: str) -> tuple[str, Path]:
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"--predictions entries must be NAME:PATH, got {arg!r}")
    name, path = arg.split(":", 1)
    return name.strip(), Path(path.strip())


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--predictions", nargs="+", type=parse_named, required=True,
                    help="NAME:/path/to/local_predictions.npz pairs")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="section ids to skip, e.g. 9 10")
    ap.add_argument("--mp-side", type=int, default=64,
                    help="mean-pool side for cross-model sections")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    skip = {str(s) for s in args.skip}

    with tee_to(args.output_dir / "report.txt"):
        hr("SPACEPRESSO PREDICTIONS ANALYSIS v2", "█")
        print(f"  output_dir : {args.output_dir}")
        print(f"  models     : {len(args.predictions)}")
        for n, p in args.predictions:
            print(f"    {n:<14} {p}")
        print(f"  deps       : SciPy={HAS_SCIPY} sklearn={HAS_SKLEARN} "
              f"spearmanr={HAS_SPEARMAN}")
        t0 = time.time()

        named_paths = args.predictions
        section_1_inventory(named_paths, args.output_dir)
        if "2" not in skip:
            section_2_score_distributions(named_paths, args.output_dir)
        if "3" not in skip:
            section_3_calibration_drift(named_paths, args.output_dir)
        s4 = (None if "4" in skip
              else section_4_sparsity(named_paths, args.output_dir))
        s5 = ({} if "5" in skip
              else section_5_per_type_ap(named_paths, args.output_dir))
        s6 = (None if "6" in skip
              else section_6_cc_filter(named_paths, args.output_dir))
        s7 = (None if "7" in skip
              else section_7_crossview_score(named_paths, args.output_dir,
                                              side=args.mp_side))
        if "8" not in skip:
            section_8_mi_score_truth(named_paths, args.output_dir)
        s9 = (None if "9" in skip
              else section_9_redundancy(named_paths, args.output_dir,
                                         side=args.mp_side))
        if "10" not in skip:
            section_10_cross_model_agreement(s9, args.output_dir)
        s11 = (None if "11" in skip
               else section_11_greedy_ensemble(s5, named_paths, args.output_dir,
                                                side=args.mp_side))
        section_12_recipe(s4, s6, s7, s11, args.output_dir)

        hr(f"DONE in {time.time() - t0:.1f}s", "█")


if __name__ == "__main__":
    main()