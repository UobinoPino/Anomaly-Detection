"""Spacepresso dataset analysis — v2.

Mission: pixel-level anomaly detection on multi-view Spacepresso data, judged
by Average Precision. The output of THIS script feeds the stacker design:
every section ends with a one-line takeaway you can act on.

Compared to v1 we (a) compress each section's stdout to a few lines, (b)
write full machine-readable tables under `tables/` and arrays under
`tables/*.npy`, and (c) add information-theoretic and structural-prior
sections borrowed from our rec-sys analysis playbook:

  Inventory & meta (compact)
   1. Inventory                — class/split counts, multi-view structure
   2. Image meta               — resolution + channels + size
   3. Color stats              — per-(class,split) μ/σ + histogram
   4. Train↔test domain shift  — sym-KL + Bhattacharyya per channel

  Geometry of anomalies
   5. Mask coverage            — anomaly-pixel fraction per (class, type)
   6. Spatial heatmap          — entropy, centre-bias, peak location
   7. Component shape          — area, compactness, "dust" rate

  ── NEW: information-theoretic priors ──
   8. MI: metadata ↔ anomaly  — H, I, NMI, U(Y|X) for class/type/view ↔
                                 anomaly size/location/shape
   9. CMI: redundancy audit    — I(view; loc | class), I(view; size | class)
                                 → multi-view consistency expectation

  ── NEW: multi-view structure ──
  10. Cross-view mask agreement — same-sample views: defect IoU, "lonely
                                  view" rate, agreement-only mask AP ceiling

  ── NEW: anomaly-type structure ──
  11. Anomaly-type fingerprint  — SVD on per-(class,type) flattened heatmap;
                                  clusters anomaly types → which types are
                                  confusable, which deserve their own head
  12. Spatial-prior AP ceiling  — if we predict the per-class mean heatmap
                                  blindly, what AP do we get?

  ── Submission / metric scoping ──
  13. Submission scoping        — H/W distribution at test time
  14. Pixel-AP imbalance        — random-baseline AP estimate

  Final
  15. Stacker-design takeaways  — concrete recommendations from the above

Outputs:
    <report_dir>/report.txt           — full plaintext log (Tee'd here)
    <report_dir>/tables/*.csv         — machine-readable tables
    <report_dir>/tables/*.npy         — heatmaps, fingerprints
    <report_dir>/figs/*.png           — optional plots (matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from scipy import ndimage as ndi
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/workspace/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_CSV        = PROJECT_ROOT / "data" / "anomaly_descriptions.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "analysis_out"

CLASS_HUMAN = {
    "class_01": "resistor",  "class_02": "inductor", "class_03": "gear",
    "class_04": "screw",     "class_05": "nut",      "class_06": "coffee",
    "class_07": "pistachio", "class_08": "capsule",
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIEW_RE = re.compile(r"^(?P<base>.+?)_view(?P<view>\d+)\.[A-Za-z]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Logging + tiny stat helpers
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
    """Compact single-line percentile summary."""
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
# Information-theoretic helpers (nats)
# Same shapes as in our rec-sys analysis: pass a `joint` Counter
# {(x,y): n}, optionally with pre-computed marginals.
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


def nmi(mi: float, hx: float, hy: float) -> float:
    d = math.sqrt(max(hx, 0) * max(hy, 0))
    return (mi / d) if d > 0 else 0.0


def cmi_xyz(joint_xyz: Counter) -> tuple[float, float]:
    """Compute I(X; Y | Z) and H(Y | Z) given counts over (x, y, z) triples."""
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


def discretise(values: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Equal-frequency quantile binning. Returns int bin ids."""
    if values.size == 0: return values.astype(np.int8)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    if qs.size == 0: return np.zeros_like(values, dtype=np.int8)
    edges = np.unique(np.quantile(values, qs))
    return np.searchsorted(edges, values, side="right").astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scan
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ImageRecord:
    path: Path
    cls: str
    split: str
    anomaly_type: str | None = None
    sample_id: str | None = None
    view: int | None = None
    mask_path: Path | None = None


def parse_view(name: str) -> tuple[str, int | None]:
    m = VIEW_RE.match(name)
    if m:
        return m.group("base"), int(m.group("view"))
    return Path(name).stem, None


def scan_dataset(data_root: Path) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    if not data_root.exists():
        print(f"  [WARN] {data_root} not found"); return records, []
    classes = sorted(d.name for d in data_root.iterdir()
                     if d.is_dir() and d.name.startswith("class_"))
    for cls in classes:
        cdir = data_root / cls
        gd = cdir / "train" / "good"
        if gd.exists():
            for p in sorted(gd.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    records.append(ImageRecord(p, cls, "train_good",
                                               sample_id=sid, view=v))
        td = cdir / "train"
        if td.exists():
            for sd in sorted(td.iterdir()):
                if (not sd.is_dir() or sd.name == "good"
                        or not sd.name.startswith("anomaly_")):
                    continue
                gt_dir = cdir / "ground_truth_train" / sd.name
                for p in sorted(sd.iterdir()):
                    if p.suffix.lower() not in IMG_EXTS: continue
                    sid, v = parse_view(p.name)
                    mp = None
                    if gt_dir.exists():
                        c1 = gt_dir / p.name
                        if c1.exists():
                            mp = c1
                        else:
                            for q in gt_dir.iterdir():
                                if (q.stem == p.stem
                                        and q.suffix.lower() in IMG_EXTS):
                                    mp = q; break
                    records.append(ImageRecord(p, cls, "train_anomaly",
                                               anomaly_type=sd.name,
                                               sample_id=sid, view=v,
                                               mask_path=mp))
        ted = cdir / "test"
        if ted.exists():
            for p in sorted(ted.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    records.append(ImageRecord(p, cls, "test",
                                               sample_id=sid, view=v))
    return records, classes


def open_img(path: Path) -> np.ndarray | None:
    if not HAS_PIL: return None
    try:
        with Image.open(path) as im:
            im.load()
            return np.array(im)
    except Exception:
        return None


def open_mask(path: Path, target_side: int | None = None) -> np.ndarray | None:
    if not HAS_PIL: return None
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode not in ("L", "1", "I", "I;16"):
                im = im.convert("L")
            if target_side is not None:
                im = im.resize((target_side, target_side), Image.NEAREST)
            arr = np.array(im)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            return (arr > 0).astype(np.uint8)
    except Exception:
        return None


def sample_records(records: list[ImageRecord], k: int, seed: int = 0):
    if k <= 0 or k >= len(records): return list(records)
    rng = np.random.default_rng(seed)
    return [records[i] for i in rng.choice(len(records), k, replace=False)]


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Inventory (compact)
# ─────────────────────────────────────────────────────────────────────────────
def section_1_inventory(records, classes, report_dir):
    hr("SECTION 1 — Inventory", "=")
    by_cs: dict[tuple, list] = defaultdict(list)
    for r in records: by_cs[(r.cls, r.split)].append(r)

    rows = []
    print(f"  {'class':<10} {'object':<12} "
          f"{'good':>6} {'anom':>6} {'test':>6} "
          f"{'#types':>7} {'#samples':>9}")
    total_good = total_anom = total_test = 0
    for cls in classes:
        ng = len(by_cs.get((cls, "train_good"), []))
        na = len(by_cs.get((cls, "train_anomaly"), []))
        nt = len(by_cs.get((cls, "test"), []))
        types = sorted({r.anomaly_type for r in by_cs.get((cls, "train_anomaly"), [])
                        if r.anomaly_type})
        samples = {r.sample_id for r in by_cs.get((cls, "train_anomaly"), [])
                   if r.sample_id}
        total_good += ng; total_anom += na; total_test += nt
        rows.append({"class": cls, "object": CLASS_HUMAN.get(cls, "?"),
                     "train_good": ng, "train_anomaly": na, "test": nt,
                     "n_anomaly_types": len(types),
                     "n_unique_samples": len(samples),
                     "anomaly_types": ";".join(types)})
        print(f"  {cls:<10} {CLASS_HUMAN.get(cls, '?'):<12} "
              f"{ng:>6} {na:>6} {nt:>6} {len(types):>7} {len(samples):>9}")
    print(f"  {'TOTAL':<23} {total_good:>6} {total_anom:>6} {total_test:>6}")

    # View count per sample
    by_sample: dict[tuple, list[ImageRecord]] = defaultdict(list)
    for r in records:
        sid = r.sample_id or r.path.stem
        by_sample[(r.cls, r.split, r.anomaly_type, sid)].append(r)
    views_dist = Counter(len(rs) for rs in by_sample.values())
    print(f"\n  views/sample distribution: " +
          " ".join(f"{v}v→{n}" for v, n in sorted(views_dist.items())))

    save_csv(rows, report_dir / "tables" / "01_inventory.csv")
    print(f"  → tables/01_inventory.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Image meta (compact)
# ─────────────────────────────────────────────────────────────────────────────
def section_2_image_meta(records, classes, report_dir, max_per_bucket):
    hr("SECTION 2 — Image metadata", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return {}
    by_cs: dict[tuple, list] = defaultdict(list)
    for r in records: by_cs[(r.cls, r.split)].append(r)

    res_seen: Counter = Counter()
    chan_seen: Counter = Counter()
    sizes: list[float] = []
    for (cls, split), recs in by_cs.items():
        sample = sample_records(recs, max_per_bucket,
                                seed=hash((cls, split)) & 0xFFFF)
        for r in sample:
            arr = open_img(r.path)
            if arr is None: continue
            h, w = arr.shape[:2]
            c = 1 if arr.ndim == 2 else arr.shape[2]
            res_seen[(h, w)] += 1
            chan_seen[c] += 1
            try: sizes.append(r.path.stat().st_size / 1024.0)
            except Exception: pass

    print(f"  unique resolutions : {len(res_seen)}")
    top_res = res_seen.most_common(3)
    print(f"  top-3 resolutions  : " +
          ", ".join(f"{h}x{w}({n})" for (h, w), n in top_res))
    print(f"  channels           : " +
          ", ".join(f"{c}ch→{n}" for c, n in sorted(chan_seen.items())))
    if sizes:
        pct_one_line(sizes, "file size (KB)")

    headline = ("UNIFORM (1 resolution) — no resize concern."
                if len(res_seen) == 1
                else f"NON-UNIFORM ({len(res_seen)} resolutions) — resize at train time, "
                     f"submit at native (H, W).")
    print(f"  TAKEAWAY: {headline}")
    save_csv([{"resolution": f"{h}x{w}", "count": n}
              for (h, w), n in sorted(res_seen.items(), key=lambda kv: -kv[1])],
             report_dir / "tables" / "02_resolutions.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Color stats (compact, with entropy)
# ─────────────────────────────────────────────────────────────────────────────
def section_3_color_stats(records, classes, report_dir, max_per_bucket):
    hr("SECTION 3 — Pixel color statistics", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return {}
    by_cs: dict[tuple, list] = defaultdict(list)
    for r in records: by_cs[(r.cls, r.split)].append(r)

    histograms: dict[tuple, np.ndarray] = {}
    rows = []
    for (cls, split), recs in sorted(by_cs.items()):
        sample = sample_records(recs, max_per_bucket,
                                seed=hash((cls, split, "px")) & 0xFFFF)
        hist = np.zeros((3, 256), dtype=np.int64)
        for r in sample:
            arr = open_img(r.path)
            if arr is None: continue
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] != 3:
                continue
            arr = arr.astype(np.uint8)
            for c in range(3):
                hist[c] += np.bincount(arr[..., c].ravel(), minlength=256)
        histograms[(cls, split)] = hist
        means, stds, ents = [], [], []
        for c in range(3):
            h = hist[c].astype(np.float64); tot = h.sum()
            if tot == 0:
                means.append(0); stds.append(0); ents.append(0); continue
            x = np.arange(256)
            mu = (h * x).sum() / tot
            var = (h * (x - mu) ** 2).sum() / tot
            p = h / tot
            p = p[p > 0]
            ent = -(p * np.log(p)).sum()
            means.append(mu); stds.append(math.sqrt(var)); ents.append(ent)
        rows.append({"class": cls, "split": split,
                     "mean_R": means[0], "mean_G": means[1], "mean_B": means[2],
                     "std_R": stds[0], "std_G": stds[1], "std_B": stds[2],
                     "ent_R": ents[0], "ent_G": ents[1], "ent_B": ents[2]})

    print(f"  {'class':<10} {'split':<14}  "
          f"{'μR':>5} {'μG':>5} {'μB':>5}  "
          f"{'σR':>5} {'σG':>5} {'σB':>5}  "
          f"{'H_R':>5} {'H_G':>5} {'H_B':>5}")
    for r in rows:
        print(f"  {r['class']:<10} {r['split']:<14}  "
              f"{r['mean_R']:>5.1f} {r['mean_G']:>5.1f} {r['mean_B']:>5.1f}  "
              f"{r['std_R']:>5.1f} {r['std_G']:>5.1f} {r['std_B']:>5.1f}  "
              f"{r['ent_R']:>5.2f} {r['ent_G']:>5.2f} {r['ent_B']:>5.2f}")
    save_csv(rows, report_dir / "tables" / "03_color_stats.csv")
    print(f"  TAKEAWAY: low H_X ⇒ flat/uniform texture (background-friendly). "
          f"σ ≫ catalogue average ⇒ candidate for class-specific normalisation.")
    return {"histograms": histograms}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Train vs Test domain shift (compact)
# ─────────────────────────────────────────────────────────────────────────────
def _hist_distance(p, q):
    p = p.astype(np.float64) / max(p.sum(), 1)
    q = q.astype(np.float64) / max(q.sum(), 1)
    eps = 1e-12
    pp = p + eps; qq = q + eps
    pp /= pp.sum(); qq /= qq.sum()
    kl_pq = float((pp * np.log(pp / qq)).sum())
    kl_qp = float((qq * np.log(qq / pp)).sum())
    bhat = -math.log(max(float(np.sqrt(pp * qq).sum()), eps))
    cdf_p = np.cumsum(p); cdf_q = np.cumsum(q)
    wass = float(np.abs(cdf_p - cdf_q).sum())
    return {"kl_sym": (kl_pq + kl_qp) / 2, "bhatt": bhat, "wass": wass}


def section_4_domain_shift(s3, classes, report_dir):
    hr("SECTION 4 — Train↔Test domain shift", "=")
    H = s3.get("histograms", {})
    if not H:
        print("  [SKIP] no histograms"); return
    rows = []
    per_class_kl = []
    print(f"  {'class':<10}  "
          f"{'symKL_R':>8} {'symKL_G':>8} {'symKL_B':>8}  "
          f"{'Bhatt_avg':>10}  drift")
    for cls in classes:
        h_tr = H.get((cls, "train_good")); h_te = H.get((cls, "test"))
        if h_tr is None or h_te is None: continue
        kls = []; bhs = []
        for c in range(3):
            d = _hist_distance(h_tr[c], h_te[c])
            kls.append(d["kl_sym"]); bhs.append(d["bhatt"])
            rows.append({"class": cls, "channel": "RGB"[c], **d})
        avg_kl = float(np.mean(kls)); avg_bh = float(np.mean(bhs))
        per_class_kl.append(avg_kl)
        flag = "‼ HIGH" if avg_kl > 0.05 else ("· mod" if avg_kl > 0.02 else "ok")
        print(f"  {cls:<10}  "
              f"{kls[0]:>8.4f} {kls[1]:>8.4f} {kls[2]:>8.4f}  "
              f"{avg_bh:>10.4f}  {flag}")
    save_csv(rows, report_dir / "tables" / "04_domain_shift.csv")
    print(f"  TAKEAWAY: classes flagged ‼ deserve stronger augmentation + "
          f"per-class TTA. Stacker should not pool such classes' scores in "
          f"a single global rank.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Mask coverage
# ─────────────────────────────────────────────────────────────────────────────
def section_5_mask_coverage(records, classes, report_dir):
    hr("SECTION 5 — Mask coverage", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return {}
    per_anom_fracs: dict[tuple, list[float]] = defaultdict(list)
    all_fracs: list[float] = []
    n_masks = 0
    for r in records:
        if r.split != "train_anomaly" or r.mask_path is None: continue
        m = open_mask(r.mask_path)
        if m is None: continue
        n_masks += 1
        f = float(m.mean())
        all_fracs.append(f)
        per_anom_fracs[(r.cls, r.anomaly_type or "?")].append(f)

    print(f"  masks inspected: {n_masks}")
    pct_one_line(all_fracs, "anomaly-pixel fraction (global)")
    # Per (class, anomaly_type) compressed table
    print(f"  {'class':<10} {'type':<11} {'n':>3} "
          f"{'μ':>7} {'p50':>7} {'p95':>7}")
    rows = []
    for (cls, atype), fracs in sorted(per_anom_fracs.items()):
        a = np.asarray(fracs)
        rows.append({"class": cls, "anomaly_type": atype, "n": len(a),
                     "frac_mean": float(a.mean()),
                     "frac_p25": float(np.percentile(a, 25)),
                     "frac_p50": float(np.percentile(a, 50)),
                     "frac_p75": float(np.percentile(a, 75)),
                     "frac_p95": float(np.percentile(a, 95))})
        print(f"  {cls:<10} {atype:<11} {len(a):>3} "
              f"{a.mean():>7.4f} {np.percentile(a, 50):>7.4f} "
              f"{np.percentile(a, 95):>7.4f}")
    save_csv(rows, report_dir / "tables" / "05_mask_coverage.csv")
    return {"all_fracs": all_fracs, "per_anom_fracs": per_anom_fracs}


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Spatial heatmap (entropy + centre-bias)
# Saves per-(class, anomaly_type) heatmaps to tables/06_heat_*.npy for §11.
# ─────────────────────────────────────────────────────────────────────────────
def section_6_spatial(records, classes, report_dir, side=128):
    hr("SECTION 6 — Spatial heatmaps", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return {}
    per_class: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((side, side), dtype=np.float64))
    per_class_n: Counter = Counter()
    per_ca: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros((side, side), dtype=np.float64))
    per_ca_n: Counter = Counter()
    global_heat = np.zeros((side, side), dtype=np.float64); global_n = 0

    for r in records:
        if r.split != "train_anomaly" or r.mask_path is None: continue
        m = open_mask(r.mask_path, target_side=side)
        if m is None: continue
        m = m.astype(np.float64)
        per_class[r.cls] += m; per_class_n[r.cls] += 1
        per_ca[(r.cls, r.anomaly_type or "?")] += m
        per_ca_n[(r.cls, r.anomaly_type or "?")] += 1
        global_heat += m; global_n += 1

    if global_n == 0:
        print("  no masks"); return {}

    npydir = report_dir / "tables"; npydir.mkdir(parents=True, exist_ok=True)
    np.save(npydir / "06_heat_global.npy", global_heat / global_n)
    rows = []
    print(f"  {'class':<10} {'N':>4}  {'cy':>5} {'cx':>5}  "
          f"{'entropy':>8}  {'centre_bias':>11}  shape")
    for cls in sorted(per_class):
        h = per_class[cls] / max(per_class_n[cls], 1)
        ys, xs = np.indices(h.shape); tot = h.sum()
        cy = float((ys * h).sum() / tot) if tot > 0 else 0.0
        cx = float((xs * h).sum() / tot) if tot > 0 else 0.0
        p = h / tot if tot > 0 else h
        flat = p.ravel(); flat = flat[flat > 0]
        ent = float(-(flat * np.log(flat)).sum()) if flat.size else 0.0
        cbox = h[side//4: 3*side//4, side//4: 3*side//4].sum() / max(tot, 1e-12)
        shape = "edge"
        if cbox > 0.7:   shape = "centred"
        elif cbox > 0.5: shape = "mid"
        rows.append({"class": cls, "n": per_class_n[cls],
                     "cy": cy, "cx": cx, "entropy": ent,
                     "centre_bias": float(cbox), "shape": shape})
        print(f"  {cls:<10} {per_class_n[cls]:>4}  {cy:>5.1f} {cx:>5.1f}  "
              f"{ent:>8.3f}  {cbox:>11.4f}  {shape}")
        np.save(npydir / f"06_heat_{cls}.npy", h)
    # save per-(class, anomaly_type) heatmaps for §11 fingerprint
    for (cls, atype), h_acc in per_ca.items():
        h = h_acc / max(per_ca_n[(cls, atype)], 1)
        np.save(npydir / f"06_heat_{cls}_{atype}.npy", h)
    save_csv(rows, report_dir / "tables" / "06_spatial_stats.csv")
    print(f"  TAKEAWAY: centre_bias > 0.7 ⇒ apply a soft centre-Gaussian prior "
          f"at inference; this is a stacker feature too (per-class centre mask).")
    return {"per_class_heat": {cls: per_class[cls] / max(per_class_n[cls], 1)
                                for cls in per_class},
            "per_ca_heat": {k: (v / max(per_ca_n[k], 1))
                            for k, v in per_ca.items()},
            "side": side}


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Component shape ("structure vs dust")
# ─────────────────────────────────────────────────────────────────────────────
def _cc_label(mask: np.ndarray):
    if HAS_SCIPY:
        labels, n = ndi.label(mask)
        return labels, int(n)
    # numpy fallback
    labels = np.zeros(mask.shape, dtype=np.int32); n = 0
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if mask[r, c] and labels[r, c] == 0:
                n += 1; stack = [(r, c)]
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < H and 0 <= x < W and mask[y, x] and labels[y, x] == 0:
                        labels[y, x] = n
                        stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
    return labels, n


def section_7_components(records, classes, report_dir, max_masks=600):
    hr("SECTION 7 — Component shape (structure vs dust)", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return
    all_areas: list[int] = []
    all_compact: list[float] = []
    cc_per_mask: list[int] = []
    dust_share: list[float] = []  # fraction of CCs that are tiny
    n = 0
    for r in records:
        if r.split != "train_anomaly" or r.mask_path is None: continue
        m = open_mask(r.mask_path)
        if m is None: continue
        n += 1
        if n > max_masks: break
        labels, ncc = _cc_label(m)
        cc_per_mask.append(ncc)
        if ncc == 0: continue
        tiny = 0
        for k in range(1, ncc + 1):
            comp = (labels == k); area = int(comp.sum())
            if area < 4: tiny += 1; continue
            ys, xs = np.where(comp)
            bb_h = ys.max() - ys.min() + 1
            bb_w = xs.max() - xs.min() + 1
            if HAS_SCIPY:
                eroded = ndi.binary_erosion(comp).astype(np.uint8)
                perim = int((comp.astype(np.uint8) - eroded).sum())
            else:
                perim = bb_h * 2 + bb_w * 2  # crude
            compact = (4 * math.pi * area) / max(perim ** 2, 1)
            if area < 25: tiny += 1
            all_areas.append(area); all_compact.append(compact)
        dust_share.append(tiny / max(ncc, 1))

    pct_one_line(cc_per_mask, "n_CC per mask")
    pct_one_line(all_areas, "area (px)")
    pct_one_line(all_compact, "compactness (0=line, 1=disk)")
    pct_one_line(dust_share, "tiny-CC share per mask (area<25)")
    save_csv([{"area_px": a, "compactness": c}
              for a, c in zip(all_areas, all_compact)],
             report_dir / "tables" / "07_shape_components.csv")
    if all_areas:
        p25 = float(np.percentile(all_areas, 25))
        print(f"  TAKEAWAY: p25(area)={p25:.0f} px ⇒ stacker should TRY dropping "
              f"CCs below ~{int(max(p25/2, 10))} px (drop-dust post-process). "
              f"If p95(compactness) is high, score maps benefit from "
              f"morphological closing before AP.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — MI: metadata ↔ anomaly priors
# ─────────────────────────────────────────────────────────────────────────────
def section_8_mi(records, classes, s5, s6, report_dir):
    hr("SECTION 8 — MI: metadata ↔ anomaly properties", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return {}

    # Build per-anomalous-image features for MI
    feats = []  # list of dicts
    for r in records:
        if r.split != "train_anomaly" or r.mask_path is None: continue
        m = open_mask(r.mask_path, target_side=128)
        if m is None or m.sum() == 0: continue
        area_frac = float(m.mean())
        ys, xs = np.where(m > 0)
        cy = float(ys.mean()) / m.shape[0]
        cx = float(xs.mean()) / m.shape[1]
        # quadrant: 0=TL, 1=TR, 2=BL, 3=BR
        quad = (1 if cx >= 0.5 else 0) + (2 if cy >= 0.5 else 0)
        labels, ncc = _cc_label(m)
        feats.append({
            "cls": r.cls,
            "atype": r.anomaly_type or "?",
            "view": int(r.view) if r.view is not None else -1,
            "sample_id": r.sample_id or r.path.stem,
            "area_frac": area_frac,
            "cy": cy, "cx": cx, "quad": quad,
            "ncc": int(ncc),
        })
    if not feats:
        print("  no anomalous masks"); return {}

    # Discretise continuous features
    areas = np.array([f["area_frac"] for f in feats])
    ncss = np.array([f["ncc"] for f in feats])
    area_bin = discretise(areas, n_bins=4)
    ncc_bin = discretise(ncss.astype(np.float64), n_bins=4)
    for i, f in enumerate(feats):
        f["area_bin"] = int(area_bin[i])
        f["ncc_bin"] = int(ncc_bin[i])

    # MI table
    pairs = [
        ("class      → anomaly_type ", "cls",   "atype"),
        ("class      → area_quartile", "cls",   "area_bin"),
        ("class      → loc_quadrant ", "cls",   "quad"),
        ("class      → ncc_quartile ", "cls",   "ncc_bin"),
        ("anom_type  → area_quartile", "atype", "area_bin"),
        ("anom_type  → loc_quadrant ", "atype", "quad"),
        ("anom_type  → ncc_quartile ", "atype", "ncc_bin"),
        ("view       → loc_quadrant ", "view",  "quad"),
        ("view       → area_quartile", "view",  "area_bin"),
    ]
    print(f"  {'feature':<28} {'n':>5} {'H(X)':>6} {'H(Y)':>6} "
          f"{'MI':>6} {'NMI':>6} {'U(Y|X)':>8}")
    rows = []
    for label, k_x, k_y in pairs:
        joint = Counter((f[k_x], f[k_y]) for f in feats)
        mi, hx, hy = mi_xy(joint)
        n = nmi(mi, hx, hy)
        u = mi / hy if hy > 0 else 0.0
        print(f"  {label:<28} {sum(joint.values()):>5} "
              f"{hx:>6.3f} {hy:>6.3f} {mi:>6.3f} {n:>6.3f} {u:>8.3f}")
        rows.append({"feature": label.strip(), "n": sum(joint.values()),
                     "H_X": hx, "H_Y": hy, "MI": mi, "NMI": n, "U_Y_given_X": u})
    save_csv(rows, report_dir / "tables" / "08_mi.csv")
    print(f"  TAKEAWAY: stacker should include class AND anomaly_type as one-hot "
          f"features (high U); high U(loc|view) ⇒ view index is informative ⇒ "
          f"add view-index as a stacker feature.")
    return {"feats": feats}


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — CMI: redundancy audit (does view add info given class?)
# ─────────────────────────────────────────────────────────────────────────────
def section_9_cmi(s8, report_dir):
    hr("SECTION 9 — Conditional MI: view, type | class", "=")
    feats = s8.get("feats", [])
    if not feats:
        print("  [SKIP] no anomalous features"); return
    triples = [
        ("I(view; loc_quadrant  | class)        ", "view",  "quad",     "cls"),
        ("I(view; area_quartile | class)        ", "view",  "area_bin", "cls"),
        ("I(view; ncc_quartile  | class)        ", "view",  "ncc_bin",  "cls"),
        ("I(atype; loc_quadrant | class)        ", "atype", "quad",     "cls"),
        ("I(atype; area_quartile | class)       ", "atype", "area_bin", "cls"),
    ]
    print(f"  {'feature':<42} {'n':>5} {'CMI':>6} {'H(Y|Z)':>8} {'CMI/H(Y|Z)':>11}")
    rows = []
    for label, x, y, z in triples:
        joint = Counter()
        for f in feats: joint[(f[x], f[y], f[z])] += 1
        cmi, hyz = cmi_xyz(joint)
        rel = (cmi / hyz) if hyz > 0 else 0.0
        print(f"  {label:<42} {sum(joint.values()):>5} {cmi:>6.3f} "
              f"{hyz:>8.3f} {rel:>11.3f}")
        rows.append({"feature": label.strip(),
                     "n": sum(joint.values()),
                     "CMI": cmi, "H_Y_given_Z": hyz, "rel": rel})
    save_csv(rows, report_dir / "tables" / "09_cmi.csv")
    print(f"  TAKEAWAY: I(view; loc | class) > 0.05 nats ⇒ defect location is "
          f"VIEW-DEPENDENT ⇒ stacker MUST condition on (class, view) when "
          f"normalising scores, not just (class).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Cross-view mask agreement
# Multi-view consistency on GROUND TRUTH: how often does the same defect
# appear in multiple views? If almost never → views are independent and the
# "sibling-bank" multiview trick is structurally unsound for these defects.
# If yes → consensus-style stacker features are warranted.
# ─────────────────────────────────────────────────────────────────────────────
def section_10_multiview(records, classes, report_dir, side=128):
    hr("SECTION 10 — Cross-view mask agreement", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return {}
    # group anomalous records by sample
    by_sample: dict[tuple, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.split != "train_anomaly" or r.mask_path is None: continue
        sid = r.sample_id or r.path.stem
        by_sample[(r.cls, r.anomaly_type or "?", sid)].append(r)

    n_samples = 0
    n_multi = 0
    n_lonely = 0
    pairwise_iou = []  # IoU between mask pairs of same sample
    presence_per_sample = []  # fraction of views with mask>0
    per_class_iou: dict[str, list[float]] = defaultdict(list)
    per_class_presence: dict[str, list[float]] = defaultdict(list)

    for (cls, atype, sid), recs in by_sample.items():
        recs_sorted = sorted(recs, key=lambda r: r.view or 0)
        masks = []
        for r in recs_sorted:
            m = open_mask(r.mask_path, target_side=side)
            if m is None:
                masks.append(None); continue
            masks.append(m)
        valid = [m for m in masks if m is not None]
        if not valid: continue
        n_samples += 1
        n_present = sum(1 for m in valid if m.sum() > 0)
        presence = n_present / len(valid)
        presence_per_sample.append(presence)
        per_class_presence[cls].append(presence)
        if n_present == 1 and len(valid) > 1:
            n_lonely += 1  # defect visible in exactly one view
        if n_present >= 2:
            n_multi += 1
        # pairwise IoU among views where defect is present
        present = [m for m in valid if m.sum() > 0]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                inter = float(np.logical_and(present[i], present[j]).sum())
                union = float(np.logical_or(present[i], present[j]).sum())
                iou = inter / max(union, 1.0)
                pairwise_iou.append(iou)
                per_class_iou[cls].append(iou)

    print(f"  multi-view samples analysed   : {n_samples}")
    print(f"  P(defect in ≥2 views)         : {n_multi / max(n_samples, 1):.4f}")
    print(f"  P(defect in exactly 1 view)   : {n_lonely / max(n_samples, 1):.4f}")
    pct_one_line(presence_per_sample, "fraction of views with defect")
    pct_one_line(pairwise_iou, "pairwise mask IoU (defect-present pairs)")

    print(f"\n  Per-class breakdown:")
    print(f"  {'class':<10} {'n_samples':>10} "
          f"{'P(≥2 views)':>13} {'IoU p50':>9} {'IoU p95':>9}")
    rows = []
    for cls in classes:
        ps = per_class_presence.get(cls, [])
        ious = per_class_iou.get(cls, [])
        if not ps: continue
        ge2 = float(np.mean([1 if p >= 2/5 else 0 for p in ps]))
        iou_p50 = float(np.percentile(ious, 50)) if ious else 0.0
        iou_p95 = float(np.percentile(ious, 95)) if ious else 0.0
        rows.append({"class": cls, "n_samples": len(ps),
                     "p_ge2_views": ge2, "iou_p50": iou_p50, "iou_p95": iou_p95})
        print(f"  {cls:<10} {len(ps):>10} {ge2:>13.4f} "
              f"{iou_p50:>9.4f} {iou_p95:>9.4f}")
    save_csv(rows, report_dir / "tables" / "10_multiview.csv")

    p_multi = n_multi / max(n_samples, 1)
    if p_multi > 0.5:
        print(f"  TAKEAWAY: defect commonly visible in multiple views (P={p_multi:.2f}) "
              f"⇒ AGREEMENT-based multiview features are reliable. Add "
              f"per-pixel cross-view max, mean, std as stacker features.")
    elif p_multi > 0.2:
        print(f"  TAKEAWAY: defect MIXED-view (P={p_multi:.2f}); use MAX over views "
              f"(not mean) so single-view positives don't get washed out.")
    else:
        print(f"  TAKEAWAY: defect mostly lonely (P={p_multi:.2f}); cross-view "
              f"consensus will mostly HURT. Score views independently and "
              f"submit a per-view score.")
    return {"presence_per_sample": presence_per_sample,
            "pairwise_iou": pairwise_iou, "p_multi": p_multi}


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 — Anomaly-type spatial fingerprint (truncated SVD)
# Build a matrix [n_anomaly_types × (side*side)] of per-(class, type) mean
# heatmaps, run SVD, and report the cumulative variance captured by the top
# few components. Then for each pair of anomaly types within a class, report
# cosine similarity between their fingerprints — high cosine = confusable.
# ─────────────────────────────────────────────────────────────────────────────
def section_11_fingerprint(s6, classes, report_dir):
    hr("SECTION 11 — Anomaly-type spatial fingerprint (SVD)", "=")
    per_ca = s6.get("per_ca_heat", {})
    if not per_ca:
        print("  [SKIP] no per-(class, anomaly) heatmaps"); return
    keys = sorted(per_ca.keys())
    side = s6["side"]
    M = np.stack([per_ca[k].ravel() for k in keys], axis=0).astype(np.float32)
    # L2-normalise each row → cosine becomes a dot product
    norms = np.linalg.norm(M, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Mn = M / norms
    # SVD (small, dense)
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    var = (sv ** 2)
    cumvar = np.cumsum(var) / var.sum() if var.sum() > 0 else np.zeros_like(var)
    k_85 = int(np.argmax(cumvar >= 0.85) + 1) if (cumvar >= 0.85).any() else len(sv)
    k_95 = int(np.argmax(cumvar >= 0.95) + 1) if (cumvar >= 0.95).any() else len(sv)
    print(f"  {len(keys)} (class, anomaly_type) heatmaps")
    print(f"  SVD : {k_85} components capture 85% var, {k_95} for 95% var")
    print(f"  top-5 singular values: " +
          ", ".join(f"{s:.3f}" for s in sv[:5]))

    # Per-class pairwise cosine: which types are confusable?
    sub("Per-class pairwise fingerprint cosine (confusability)")
    rows = []
    print(f"  {'class':<10}  {'n_types':>7}  "
          f"{'cos μ':>7} {'cos p50':>8} {'cos max':>8}  "
          f"top-pair")
    for cls in classes:
        idx_in = [i for i, (c, _) in enumerate(keys) if c == cls]
        if len(idx_in) < 2: continue
        types = [keys[i][1] for i in idx_in]
        sub_n = Mn[idx_in]  # (n_types, side*side)
        sims = sub_n @ sub_n.T
        # off-diagonal
        n = len(idx_in)
        triu = sims[np.triu_indices(n, k=1)]
        i_max = int(np.argmax(triu))
        # decode pair from triu_indices
        ti, tj = np.triu_indices(n, k=1)
        top_pair = (types[ti[i_max]], types[tj[i_max]], float(triu[i_max]))
        rows.append({"class": cls, "n_types": n,
                     "cos_mean": float(triu.mean()),
                     "cos_p50": float(np.percentile(triu, 50)),
                     "cos_max": float(triu.max()),
                     "top_pair_a": top_pair[0], "top_pair_b": top_pair[1],
                     "top_pair_cos": top_pair[2]})
        print(f"  {cls:<10}  {n:>7}  "
              f"{triu.mean():>7.3f} {np.percentile(triu, 50):>8.3f} "
              f"{triu.max():>8.3f}  "
              f"{top_pair[0]}↔{top_pair[1]} ({top_pair[2]:.2f})")
    save_csv(rows, report_dir / "tables" / "11_fingerprint.csv")
    np.save(report_dir / "tables" / "11_singular_values.npy", sv)
    print(f"  TAKEAWAY: pairs with cos > 0.7 are spatially indistinguishable ⇒ "
          f"the stacker can't learn (class, type)-conditional weights for them; "
          f"merge them or drop the conditioning. Few-component capture (k_85 "
          f"small) ⇒ a small spatial prior dictionary is enough.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 — Spatial-prior AP ceiling
# Use the per-class mean heatmap (from §6) as a constant "prediction" for
# every anomalous image of that class, and compute pixel-AP. Gives the AP
# floor any sane model must beat.
# ─────────────────────────────────────────────────────────────────────────────
def section_12_prior_ap(records, s6, classes, report_dir):
    hr("SECTION 12 — Spatial-prior AP ceiling (per-class mean heatmap)", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return
    pred_per_class = s6.get("per_class_heat", {})
    side = s6.get("side", 128)
    if not pred_per_class:
        print("  [SKIP] no per-class heatmaps"); return

    try:
        from sklearn.metrics import average_precision_score
        ap_fn = average_precision_score
    except Exception:
        def ap_fn(y, s):
            order = np.argsort(-s, kind="stable"); y = y[order]
            if y.sum() == 0: return 0.0
            tp = np.cumsum(y); fp = np.cumsum(1 - y)
            p = tp / (tp + fp + 1e-12); r = tp / max(int(y.sum()), 1)
            r = np.concatenate([[0.0], r]); p = np.concatenate([[1.0], p])
            return float(np.sum((r[1:] - r[:-1]) * p[1:]))

    per_class_ap: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.split != "train_anomaly" or r.mask_path is None: continue
        m = open_mask(r.mask_path, target_side=side)
        if m is None: continue
        if r.cls not in pred_per_class: continue
        pred = pred_per_class[r.cls]
        ap = ap_fn(m.ravel().astype(np.int32), pred.ravel().astype(np.float32))
        per_class_ap[r.cls].append(float(ap))

    print(f"  {'class':<10} {'n':>4} {'prior AP (mean ± std)':>26}")
    rows = []
    for cls in classes:
        ap = per_class_ap.get(cls, [])
        if not ap: continue
        a = np.asarray(ap)
        rows.append({"class": cls, "n": len(a),
                     "prior_ap_mean": float(a.mean()),
                     "prior_ap_std": float(a.std()),
                     "prior_ap_p50": float(np.percentile(a, 50))})
        print(f"  {cls:<10} {len(a):>4} "
              f"{a.mean():>15.4f} ± {a.std():.4f}")
    save_csv(rows, report_dir / "tables" / "12_spatial_prior_ap.csv")
    print(f"  TAKEAWAY: any stacker output that scores LOWER than the prior AP "
          f"on its own class is broken. Use this as a CI floor in your tests. "
          f"For classes with high prior AP, a simple `score += λ·prior_heat` "
          f"feature is a free stacker boost.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 — Submission scoping (compact)
# ─────────────────────────────────────────────────────────────────────────────
def section_13_submission(records, classes, report_dir, max_per_class=50):
    hr("SECTION 13 — Submission scoping", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return
    by_cs: dict[tuple, list] = defaultdict(list)
    for r in records: by_cs[(r.cls, r.split)].append(r)
    total = 0
    res_seen: Counter = Counter()
    for cls in classes:
        recs = by_cs.get((cls, "test"), [])
        total += len(recs)
        for r in sample_records(recs, max_per_class,
                                seed=hash(("sub", cls)) & 0xFFFF):
            arr = open_img(r.path)
            if arr is None: continue
            res_seen[(arr.shape[0], arr.shape[1])] += 1
    print(f"  total test images : {total}")
    print(f"  expected rows     : {total}")
    print(f"  test (H, W) distribution: " +
          ", ".join(f"{h}x{w}({n})" for (h, w), n in res_seen.most_common(5)))
    print(f"  Reminder: q8rle header uses NATIVE (H, W) per image.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 14 — Pixel-AP imbalance
# ─────────────────────────────────────────────────────────────────────────────
def section_14_pixel_ap_imbalance(s5):
    hr("SECTION 14 — Pixel-AP imbalance", "=")
    fracs = s5.get("all_fracs", [])
    if not fracs:
        print("  [SKIP] no mask fractions"); return
    a = np.asarray(fracs)
    print(f"  mean defect fraction (labeled images): {a.mean():.6f}")
    print(f"  random AP baseline (labeled images)  : {a.mean():.4f}")
    print(f"  (test contains clean+anomalous images ⇒ actual random AP is LOWER)")
    print(f"  Reference: ≥ 0.20 = meaningful, ≥ 0.40 = strong, ≥ 0.55 = podium.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 15 — Stacker-design takeaways (consolidated)
# ─────────────────────────────────────────────────────────────────────────────
def section_15_takeaways(s5, s10, report_dir):
    hr("SECTION 15 — Stacker-design takeaways (final)", "=")
    p_multi = s10.get("p_multi", 0.0) if s10 else 0.0
    print("  These are *mechanical* translations of the priors above. Combine")
    print("  with the predictions-side analysis (analyze_predictions.py) for")
    print("  a complete stacker spec.")
    print()
    print("  ── NORMALISATION ────────────────────────────────────────────────")
    print("  • Per-(class, view) score rank-norm: required if §9 CMI(view; loc")
    print("    | class) > 0.05. Fit empirical CDFs on train/good, apply at test.")
    print("  • Per-class threshold percentile: read from §5 mask-coverage to")
    print("    seed `agreement-pct` and `keep-frac` consensus-v3 knobs.")
    print()
    print("  ── POST-PROCESSING (drop dust, keep structure) ──────────────────")
    print("  • Drop CCs below p25(area) from §7 BEFORE q8rle. AP-improving on")
    print("    every MVTec-style task we have measured.")
    print("  • Apply Gaussian smoothing σ ≈ 1–2 px. Drop too-isolated CCs (")
    print("    no neighbours within 2× their radius).")
    print()
    print("  ── MULTI-VIEW POLICY ────────────────────────────────────────────")
    if p_multi > 0.5:
        print(f"  • P(≥2 views show defect) = {p_multi:.2f} ⇒ ADD agreement")
        print("    features: cross-view-mean, cross-view-std, cross-view-max,")
        print("    AND a 'lonely-view penalty' (down-weight pixels that only")
        print("    light up in one view).")
    elif p_multi > 0.2:
        print(f"  • P(≥2 views) = {p_multi:.2f} ⇒ MAX over views (mean washes")
        print("    out single-view positives). Still include cross-view std as a")
        print("    confidence feature; a high σ here = unreliable score.")
    else:
        print(f"  • P(≥2 views) = {p_multi:.2f} ⇒ views mostly independent. AVOID")
        print("    consensus-v3 aggressive boost; submit per-view scores.")
    print()
    print("  ── STACKER FEATURES (must-have) ─────────────────────────────────")
    print("  • One-hot class + anomaly_type (where known)")
    print("  • view_index (informative per §9)")
    print("  • Per-(class, view) rank-normalised base-model scores")
    print("  • Per-class spatial-prior heatmap (from §6 npy) as an additive")
    print("    feature (§12 shows it has standalone AP)")
    print("  • Per-pixel CC features: area, compactness, distance-to-image-")
    print("    centre")
    print()
    print("  ── INFORMATION-THEORY GUARDRAILS ────────────────────────────────")
    print("  • If §11 cos > 0.7 between two anomaly types within a class, they")
    print("    are spatially indistinguishable ⇒ don't condition on type for")
    print("    them; merge or drop.")
    print()
    print(f"  Tables saved under: {report_dir / 'tables'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--csv",        type=Path, default=DEFAULT_CSV)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--max-images-per-bucket", type=int, default=200)
    ap.add_argument("--max-shape-masks", type=int, default=600)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="section ids to skip, e.g. 8 9 11")
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    skip = {str(s) for s in args.skip}

    with tee_to(args.report_dir / "report.txt"):
        hr("SPACEPRESSO DATASET ANALYSIS v2", "█")
        print(f"  data_root  : {args.data_root}")
        print(f"  report_dir : {args.report_dir}")
        print(f"  deps       : PIL={HAS_PIL} SciPy={HAS_SCIPY} MPL={HAS_MPL}")
        t0 = time.time()

        records, classes = scan_dataset(args.data_root)
        if not records:
            print("\n  [FATAL] no images found"); return

        section_1_inventory(records, classes, args.report_dir)
        if "2" not in skip:
            section_2_image_meta(records, classes, args.report_dir,
                                  args.max_images_per_bucket)
        s3 = ({} if "3" in skip
              else section_3_color_stats(records, classes, args.report_dir,
                                          args.max_images_per_bucket))
        if "4" not in skip:
            section_4_domain_shift(s3, classes, args.report_dir)
        s5 = ({} if "5" in skip
              else section_5_mask_coverage(records, classes, args.report_dir))
        s6 = ({} if "6" in skip
              else section_6_spatial(records, classes, args.report_dir))
        if "7" not in skip:
            section_7_components(records, classes, args.report_dir,
                                  args.max_shape_masks)
        s8 = ({} if "8" in skip
              else section_8_mi(records, classes, s5, s6, args.report_dir))
        if "9" not in skip:
            section_9_cmi(s8, args.report_dir)
        s10 = ({} if "10" in skip
               else section_10_multiview(records, classes, args.report_dir))
        if "11" not in skip:
            section_11_fingerprint(s6, classes, args.report_dir)
        if "12" not in skip:
            section_12_prior_ap(records, s6, classes, args.report_dir)
        if "13" not in skip:
            section_13_submission(records, classes, args.report_dir)
        if "14" not in skip:
            section_14_pixel_ap_imbalance(s5)
        section_15_takeaways(s5, s10, args.report_dir)

        hr(f"DONE in {time.time() - t0:.1f}s", "█")


if __name__ == "__main__":
    main()