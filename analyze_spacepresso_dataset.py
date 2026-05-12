"""Comprehensive Spacepresso anomaly-detection dataset analysis.

Mission: Spacepresso (Italian Space Agency, Mars launch in 14 days).
Task   : pixel-level anomaly detection across multi-view images of 8 object
         classes (resistor, inductor, gear, screw, nut, coffee, pistachio,
         capsule). Evaluation: pixel-level Average Precision.

Goal of THIS script: surface every structural prior that matters for
modelling decisions BEFORE writing a single line of model code, and dump
each finding to disk so it can be pasted back into chat / iterated on.

What it computes (each section is self-contained — read top-down):

  Section 0.   Setup
               Path scan, output directory bootstrap, logger that tees to
               both stdout and a .txt report so nothing is lost.

  Section 1.   Dataset inventory
               Per-class file counts (train/good, train/anomaly_*,
               ground_truth_train/anomaly_*, test). Multi-view grouping by
               sample_id (the _viewN suffix). Headline of the dataset.

  Section 2.   Image-level metadata
               Resolution distribution, channels, dtype, file size, mode.
               Stratified by class × split (train_good / train_anomaly /
               test). If resolution is non-uniform, downstream models need
               either resize or pad — this section tells you which.

  Section 3.   Pixel-level color statistics
               Per-channel mean/std/percentiles per class & split. Built to
               diagnose train/test domain shift before it bites you in the
               leaderboard.

  Section 4.   Train vs test domain shift
               KL-style comparison of the per-channel histograms between
               train_good and test for each class. Big shift ⇒ test-time
               augmentation or stronger normalisation is needed.

  Section 5.   Multi-view consistency
               For each sample_id, mean inter-view pixel-MSE and structural
               similarity. Tells you whether multi-view ensembling is worth
               the engineering (highly correlated views ⇒ less added value).

  Section 6.   Mask coverage statistics
               Per (class, anomaly_type): n_examples, anomaly-pixel ratio
               distribution, mask binarity check. Informs class-imbalance
               for AP, and per-anomaly-type difficulty ranking.

  Section 7.   Spatial distribution of anomalies
               Aggregate heatmap of mask occurrence across the image plane,
               per class. Saved as .npy AND .png. Reveals whether anomalies
               concentrate in a region (centre-bias, edges, …) or are
               uniform — affects whether a positional prior helps.

  Section 8.   Anomaly shape / size analysis
               Connected components per mask: count, area, bbox area,
               compactness (4πA / P²), eccentricity, solidity. Drives patch
               size and receptive-field choices.

  Section 9.   Per-class × per-anomaly-type breakdown
               One row per (class, anomaly_type): n_train_examples, mean
               anomaly fraction, median CC area, description summary.
               Sorted by ascending median area = hardness ranking.

  Section 10.  Anomaly description text analysis
               From anomaly_descriptions.csv: description length, generic
               vs. specific descriptions ("Localized visual anomaly..."),
               per-class word frequencies. Quantifies how much textual
               supervision you can leverage (e.g., for a CLIP-conditioned
               head).

  Section 11.  Object/background heuristics
               Edge density, background uniformity proxy, foreground area
               estimate. Tells you whether a foreground-mask preprocessing
               step is likely to help (it usually does for MVTec-style
               data).

  Section 12.  Submission scoping
               Test image count, expected submission rows, per-image (H, W)
               distribution for the q8rle header, sanity checks on the
               provided CSV's anomaly_descriptions example paths.

  Section 13.  Pixel-AP imbalance analysis
               Global positive-pixel fraction over labelled anomalies, and
               implications for AP-vs-IoU choice. Random-baseline AP
               estimate.

  Section 14.  Recommendations
               Mechanical translation of the priors above into concrete
               modelling choices: backbone, input resolution, patch size,
               augmentation policy, loss weights, multi-view fusion, and a
               rough leaderboard-AP target band.

Output:
  REPORT_DIR/report.txt              — full plaintext log (everything below)
  REPORT_DIR/figs/<section>_*.png    — heatmaps, histograms, shape plots
  REPORT_DIR/tables/*.csv            — machine-readable tables for follow-up

Run:
    python analyze_spacepresso_dataset.py \\
        --data-root  /path/to/spacepresso/data \\
        --csv        /path/to/anomaly_descriptions.csv \\
        --report-dir /path/to/output \\
        --max-images-per-bucket 200    # speed knob
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
from typing import Iterable, Optional

import numpy as np

# Optional deps — script gracefully degrades if absent
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

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
# Default paths — wired to the WSL project layout
#   /mnt/c/Users/Francoo/PycharmProjects/Anomaly-Detection/
#       ├── data/
#       │   ├── class_01/  ...  class_08/
#       │   └── anomaly_descriptions.csv     <-- put the uploaded CSV here
#       ├── analysis_out/                    <-- created by this script
#       └── analyze_spacepresso_dataset.py
# Override on the CLI if you reorganise.
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/mnt/c/Users/Francoo/PycharmProjects/Anomaly-Detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_CSV        = PROJECT_ROOT / "data" / "anomaly_descriptions.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "analysis_out"

# Class label → human name (matches anomaly_descriptions.csv `object_name`).
# Override / extend if your dataset uses different class folder names.
CLASS_HUMAN = {
    "class_01": "resistor",
    "class_02": "inductor",
    "class_03": "gear",
    "class_04": "screw",
    "class_05": "nut",
    "class_06": "coffee",
    "class_07": "pistachio",
    "class_08": "capsule",
}

# Image extensions accepted (lowercase)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Regex for filenames like img_<sampleid>_view<k>.png
VIEW_RE = re.compile(r"^(?P<base>.+?)_view(?P<view>\d+)\.[A-Za-z]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger — every print also lands in report.txt
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


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


def hr(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}")
    print(f"  {title}")
    print(f"{char * 78}")


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


def percentile_summary(arr, label: str, indent: str = "    ") -> None:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        print(f"{indent}{label}: (empty)")
        return
    print(f"{indent}{label}: n={len(a)} "
          f"min={a.min():.4g} mean={a.mean():.4g} "
          f"p25={np.percentile(a, 25):.4g} p50={np.percentile(a, 50):.4g} "
          f"p75={np.percentile(a, 75):.4g} p95={np.percentile(a, 95):.4g} "
          f"max={a.max():.4g} std={a.std():.4g}")


def save_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    # ensure all rows have the same keys
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ImageRecord:
    path: Path
    cls: str               # e.g. "class_01"
    split: str             # "train_good" | "train_anomaly" | "test"
    anomaly_type: str | None = None   # e.g. "anomaly_03", or None
    sample_id: str | None = None      # base id stripping _viewK (and ext)
    view: int | None = None
    mask_path: Path | None = None     # only for train_anomaly with a GT mask


@dataclass
class DatasetIndex:
    records: list[ImageRecord] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)

    def by_class_split(self) -> dict[tuple[str, str], list[ImageRecord]]:
        out: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
        for r in self.records:
            out[(r.cls, r.split)].append(r)
        return out

    def by_class_anomaly(self) -> dict[tuple[str, str], list[ImageRecord]]:
        out: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
        for r in self.records:
            if r.split == "train_anomaly":
                out[(r.cls, r.anomaly_type or "?")].append(r)
        return out

    def samples(self) -> dict[tuple[str, str, str | None, str], list[ImageRecord]]:
        """Group by (class, split, anomaly_type, sample_id)."""
        out: dict[tuple[str, str, str | None, str], list[ImageRecord]] = defaultdict(list)
        for r in self.records:
            sid = r.sample_id or r.path.stem
            out[(r.cls, r.split, r.anomaly_type, sid)].append(r)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Inventory
# ─────────────────────────────────────────────────────────────────────────────
def parse_view(name: str) -> tuple[str, int | None]:
    """Strip _viewK from a stem-or-filename. Returns (sample_id, view_or_None)."""
    base_name = name
    m = VIEW_RE.match(base_name)
    if m:
        return m.group("base"), int(m.group("view"))
    # fallback: treat the whole stem as sample_id
    stem = Path(base_name).stem
    return stem, None


def scan_dataset(data_root: Path) -> DatasetIndex:
    idx = DatasetIndex()
    if not data_root.exists():
        print(f"  [WARN] data_root does not exist: {data_root}")
        return idx

    classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()
                      and d.name.startswith("class_")])
    idx.classes = classes
    if not classes:
        print(f"  [WARN] no class_* directories found in {data_root}")
        return idx

    for cls in classes:
        cdir = data_root / cls

        # --- train/good
        good_dir = cdir / "train" / "good"
        if good_dir.exists():
            for p in sorted(good_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    idx.records.append(ImageRecord(
                        path=p, cls=cls, split="train_good",
                        sample_id=sid, view=v))

        # --- train/anomaly_XX (+ matching ground_truth_train/anomaly_XX)
        train_dir = cdir / "train"
        if train_dir.exists():
            for sub_dir in sorted(train_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                if sub_dir.name == "good":
                    continue
                if not sub_dir.name.startswith("anomaly_"):
                    continue
                a_type = sub_dir.name
                gt_dir = cdir / "ground_truth_train" / a_type
                for p in sorted(sub_dir.iterdir()):
                    if p.suffix.lower() not in IMG_EXTS:
                        continue
                    sid, v = parse_view(p.name)
                    mp = None
                    if gt_dir.exists():
                        # try exact same filename in gt_dir
                        cand = gt_dir / p.name
                        if cand.exists():
                            mp = cand
                        else:
                            # try same stem but any extension
                            for q in gt_dir.iterdir():
                                if q.stem == p.stem and q.suffix.lower() in IMG_EXTS:
                                    mp = q
                                    break
                    idx.records.append(ImageRecord(
                        path=p, cls=cls, split="train_anomaly",
                        anomaly_type=a_type, sample_id=sid, view=v,
                        mask_path=mp))

        # --- test
        test_dir = cdir / "test"
        if test_dir.exists():
            # Some MVTec-style layouts nest test/<bucket>/files. Walk.
            for p in sorted(test_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    idx.records.append(ImageRecord(
                        path=p, cls=cls, split="test",
                        sample_id=sid, view=v))

    return idx


def section_1_inventory(idx: DatasetIndex, report_dir: Path) -> None:
    hr("SECTION 1 — Dataset inventory", "=")
    print(f"  classes detected ({len(idx.classes)}): {', '.join(idx.classes)}")
    print(f"  total image records: {len(idx.records)}")

    # Per (class, split) table
    sub("Counts per (class, split)")
    table_rows = []
    by_cs = idx.by_class_split()
    print(f"  {'class':<10} {'human':<12} "
          f"{'train_good':>11} {'train_anom':>11} {'test':>8} "
          f"{'#anom_types':>12}")
    for cls in idx.classes:
        n_g  = len(by_cs.get((cls, "train_good"), []))
        n_a  = len(by_cs.get((cls, "train_anomaly"), []))
        n_t  = len(by_cs.get((cls, "test"), []))
        a_types = sorted({r.anomaly_type for r in by_cs.get((cls, "train_anomaly"), [])
                          if r.anomaly_type})
        table_rows.append({
            "class": cls, "human": CLASS_HUMAN.get(cls, "?"),
            "train_good": n_g, "train_anomaly": n_a, "test": n_t,
            "n_anomaly_types": len(a_types),
            "anomaly_types": ";".join(a_types),
        })
        print(f"  {cls:<10} {CLASS_HUMAN.get(cls, '?'):<12} "
              f"{n_g:>11} {n_a:>11} {n_t:>8} {len(a_types):>12}")
    save_table(table_rows, report_dir / "tables" / "01_inventory_by_class.csv")

    # Multi-view structure
    sub("Multi-view sample structure")
    samples = idx.samples()
    views_per_sample = []
    samples_with_n: Counter = Counter()
    for key, recs in samples.items():
        v = len({r.view for r in recs if r.view is not None})
        # if no _viewN suffix, we treat the sample as 1-view
        v = max(v, 1 if any(r.view is None for r in recs) else v)
        views_per_sample.append(v)
        samples_with_n[v] += 1
    print(f"  total unique sample_ids: {len(samples)}")
    print(f"  views-per-sample distribution:")
    for v in sorted(samples_with_n):
        print(f"    {v} views: {samples_with_n[v]} samples")
    percentile_summary(views_per_sample, "views per sample")

    # Per (class, split): how many samples have all-5 views, how many partial
    sub("Per (class, split) sample-view completeness")
    per_cs_views: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (cls, split, atype, sid), recs in samples.items():
        per_cs_views[(cls, split)].append(len(recs))
    print(f"  {'class':<10} {'split':<14} {'n_samples':>10} {'views_p50':>10} {'has_5views':>11}")
    for (cls, split), counts in sorted(per_cs_views.items()):
        a = np.asarray(counts)
        n5 = int((a == 5).sum())
        print(f"  {cls:<10} {split:<14} {len(a):>10} "
              f"{int(np.percentile(a, 50)):>10} {n5:>11}")

    # Per anomaly_type counts (with mask coverage)
    sub("Anomaly-type counts and mask coverage")
    by_ca = idx.by_class_anomaly()
    rows = []
    print(f"  {'class':<10} {'anom_type':<12} {'n_imgs':>8} {'with_mask':>10} {'sample_ids':>12}")
    for (cls, atype), recs in sorted(by_ca.items()):
        n_w_mask = sum(1 for r in recs if r.mask_path is not None)
        sids = {r.sample_id for r in recs}
        rows.append({"class": cls, "anomaly_type": atype,
                     "n_images": len(recs), "n_with_mask": n_w_mask,
                     "n_unique_sample_ids": len(sids)})
        print(f"  {cls:<10} {atype:<12} {len(recs):>8} {n_w_mask:>10} {len(sids):>12}")
    save_table(rows, report_dir / "tables" / "01_anomaly_type_counts.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Image reading utilities
# ─────────────────────────────────────────────────────────────────────────────
def open_image_array(path: Path) -> np.ndarray | None:
    """Returns HxW or HxWxC uint8 array, or None on failure."""
    if not HAS_PIL:
        return None
    try:
        with Image.open(path) as im:
            im.load()
            arr = np.array(im)
            return arr
    except Exception as e:
        return None


def open_mask_array(path: Path) -> np.ndarray | None:
    """Returns binary HxW uint8 array (0 / 1), or None."""
    if not HAS_PIL:
        return None
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode not in ("L", "1", "I", "I;16"):
                im = im.convert("L")
            arr = np.array(im)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            # binarize: anything > 0 is anomaly
            return (arr > 0).astype(np.uint8)
    except Exception:
        return None


def file_size_kb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024.0
    except Exception:
        return 0.0


def sample_records(records: list[ImageRecord], k: int, seed: int = 0) -> list[ImageRecord]:
    if k <= 0 or k >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(records), size=k, replace=False)
    return [records[i] for i in sel]


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Image-level metadata
# ─────────────────────────────────────────────────────────────────────────────
def section_2_image_meta(idx: DatasetIndex, report_dir: Path,
                         max_per_bucket: int) -> dict:
    hr("SECTION 2 — Image-level metadata", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed — pip install Pillow")
        return {}

    by_cs = idx.by_class_split()
    rows = []
    res_per_class_split: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    chan_per_class_split: dict[tuple[str, str], Counter] = defaultdict(Counter)
    size_per_class_split: dict[tuple[str, str], list[float]] = defaultdict(list)

    for (cls, split), recs in by_cs.items():
        sample = sample_records(recs, max_per_bucket, seed=hash((cls, split)) & 0xFFFF)
        for r in sample:
            arr = open_image_array(r.path)
            if arr is None:
                continue
            h, w = arr.shape[:2]
            c = 1 if arr.ndim == 2 else arr.shape[2]
            res_per_class_split[(cls, split)].append((h, w))
            chan_per_class_split[(cls, split)][c] += 1
            size_per_class_split[(cls, split)].append(file_size_kb(r.path))

    print(f"  Sampled up to {max_per_bucket} images per (class, split).")
    sub("Resolution distribution per (class, split)")
    print(f"  {'class':<10} {'split':<14} {'n':>5} "
          f"{'h_mode':>8} {'w_mode':>8} {'h_p50':>7} {'w_p50':>7} "
          f"{'unique':>7}")
    for (cls, split) in sorted(res_per_class_split):
        rs = res_per_class_split[(cls, split)]
        if not rs:
            continue
        hs = np.asarray([h for h, _ in rs])
        ws = np.asarray([w for _, w in rs])
        h_mode = Counter([h for h, _ in rs]).most_common(1)[0][0]
        w_mode = Counter([w for _, w in rs]).most_common(1)[0][0]
        unique = len(set(rs))
        rows.append({"class": cls, "split": split, "n": len(rs),
                     "h_mode": int(h_mode), "w_mode": int(w_mode),
                     "h_p50": int(np.percentile(hs, 50)),
                     "w_p50": int(np.percentile(ws, 50)),
                     "unique_resolutions": unique,
                     "min_h": int(hs.min()), "max_h": int(hs.max()),
                     "min_w": int(ws.min()), "max_w": int(ws.max())})
        print(f"  {cls:<10} {split:<14} {len(rs):>5} "
              f"{int(h_mode):>8} {int(w_mode):>8} "
              f"{int(np.percentile(hs, 50)):>7} {int(np.percentile(ws, 50)):>7} "
              f"{unique:>7}")
    save_table(rows, report_dir / "tables" / "02_resolutions.csv")

    sub("Channel-count distribution per (class, split)")
    print(f"  {'class':<10} {'split':<14}  channels_dist")
    for (cls, split), cn in sorted(chan_per_class_split.items()):
        items = sorted(cn.items())
        s = ", ".join(f"{c}ch:{n}" for c, n in items)
        print(f"  {cls:<10} {split:<14}  {s}")

    sub("File-size distribution (KB) per (class, split)")
    for (cls, split), szs in sorted(size_per_class_split.items()):
        percentile_summary(szs, f"{cls}/{split}")

    # Headline takeaway
    all_res = []
    for v in res_per_class_split.values():
        all_res.extend(v)
    if all_res:
        unique_global = len(set(all_res))
        print(f"\n  HEADLINE: {unique_global} unique (h, w) pairs across the dataset. "
              f"{'Uniform — easy.' if unique_global == 1 else 'NON-uniform — plan to resize/pad.'}")
    return {"res_per_class_split": res_per_class_split}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Pixel-level color statistics
# ─────────────────────────────────────────────────────────────────────────────
def section_3_pixel_stats(idx: DatasetIndex, report_dir: Path,
                          max_per_bucket: int) -> dict:
    hr("SECTION 3 — Pixel-level color statistics", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed")
        return {}

    by_cs = idx.by_class_split()
    rows = []
    histograms: dict[tuple[str, str], np.ndarray] = {}  # (cls, split) -> (3, 256)

    print(f"  Sampling up to {max_per_bucket} images per (class, split).")
    for (cls, split), recs in sorted(by_cs.items()):
        sample = sample_records(recs, max_per_bucket, seed=hash((cls, split, "px")) & 0xFFFF)
        # streaming Welford
        n_pix = 0
        means = np.zeros(3, dtype=np.float64)
        m2 = np.zeros(3, dtype=np.float64)
        hist = np.zeros((3, 256), dtype=np.int64)
        for r in sample:
            arr = open_image_array(r.path)
            if arr is None:
                continue
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[..., :3]
            elif arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[2] != 3:
                continue
            arr = arr.astype(np.uint8)
            for c in range(3):
                ch = arr[..., c]
                hist[c] += np.bincount(ch.ravel(), minlength=256)
            # update Welford on the per-image channel mean — cheap proxy
            sample_means = arr.reshape(-1, 3).astype(np.float64)
            for c in range(3):
                ch_vals = sample_means[:, c]
                k = ch_vals.size
                if k == 0:
                    continue
                new_n = n_pix + k
                delta = ch_vals.mean() - means[c] if c == 0 else ch_vals.mean() - means[c]
                means[c] = (means[c] * n_pix + ch_vals.sum()) / new_n if new_n > 0 else 0.0
                # variance-via-hist below; skip in-loop variance to save time
            n_pix += sample_means.shape[0]

        histograms[(cls, split)] = hist
        # compute std from histogram for accuracy
        stds = []
        for c in range(3):
            h = hist[c].astype(np.float64)
            tot = h.sum()
            if tot == 0:
                stds.append(0.0); continue
            x = np.arange(256)
            mu = (h * x).sum() / tot
            var = (h * (x - mu) ** 2).sum() / tot
            stds.append(float(np.sqrt(var)))
            means[c] = float(mu)
        rows.append({
            "class": cls, "split": split, "n_pixels": int(n_pix),
            "mean_R": float(means[0]), "mean_G": float(means[1]), "mean_B": float(means[2]),
            "std_R": stds[0], "std_G": stds[1], "std_B": stds[2],
        })

    print(f"\n  {'class':<10} {'split':<14}  "
          f"{'meanR':>7} {'meanG':>7} {'meanB':>7}  "
          f"{'stdR':>6} {'stdG':>6} {'stdB':>6}")
    for r in rows:
        print(f"  {r['class']:<10} {r['split']:<14}  "
              f"{r['mean_R']:>7.2f} {r['mean_G']:>7.2f} {r['mean_B']:>7.2f}  "
              f"{r['std_R']:>6.2f} {r['std_G']:>6.2f} {r['std_B']:>6.2f}")
    save_table(rows, report_dir / "tables" / "03_color_stats.csv")

    # Save histogram plots
    if HAS_MPL:
        figdir = report_dir / "figs"
        figdir.mkdir(parents=True, exist_ok=True)
        for (cls, split), hist in histograms.items():
            fig, ax = plt.subplots(figsize=(7, 3))
            for c, color in enumerate(["red", "green", "blue"]):
                ax.plot(hist[c], color=color, alpha=0.7, label=color[0].upper())
            ax.set_title(f"{cls} / {split} — channel histograms")
            ax.set_xlabel("intensity"); ax.set_ylabel("count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figdir / f"03_hist_{cls}_{split}.png", dpi=110)
            plt.close(fig)
        print(f"\n  Saved per-(class,split) histogram plots to {figdir}")
    return {"histograms": histograms}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Train vs test domain shift
# ─────────────────────────────────────────────────────────────────────────────
def _hist_distance(p: np.ndarray, q: np.ndarray) -> dict:
    """Symmetric KL + Bhattacharyya + Wasserstein-1 between 1d histograms."""
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
    return {"kl_sym": (kl_pq + kl_qp) / 2.0, "bhattacharyya": bhat, "wasserstein": wass}


def section_4_domain_shift(s3: dict, idx: DatasetIndex, report_dir: Path) -> None:
    hr("SECTION 4 — Train vs test domain shift", "=")
    if "histograms" not in s3:
        print("  [SKIP] need section 3 histograms"); return
    H = s3["histograms"]
    print("  For each class, compare per-channel histograms train_good ↔ test.")
    print("  Larger distance ⇒ stronger shift ⇒ stronger normalisation/aug needed.")
    print(f"\n  {'class':<10} {'channel':<7} {'sym-KL':>9} {'Bhatt':>9} {'Wass':>9}")
    rows = []
    for cls in idx.classes:
        h_tr = H.get((cls, "train_good"))
        h_te = H.get((cls, "test"))
        if h_tr is None or h_te is None:
            continue
        for c, name in enumerate(["R", "G", "B"]):
            d = _hist_distance(h_tr[c], h_te[c])
            rows.append({"class": cls, "channel": name, **d})
            print(f"  {cls:<10} {name:<7} "
                  f"{d['kl_sym']:>9.4f} {d['bhattacharyya']:>9.4f} {d['wasserstein']:>9.2f}")
    save_table(rows, report_dir / "tables" / "04_domain_shift.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Multi-view consistency
# ─────────────────────────────────────────────────────────────────────────────
def _common_resize(a: np.ndarray, b: np.ndarray, side: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Cheap nearest-neighbour resize of a/b to (side,side,3) grayscale-friendly."""
    def to_3c(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return np.stack([x] * 3, axis=-1)
        if x.shape[-1] == 4:
            return x[..., :3]
        if x.shape[-1] == 1:
            return np.repeat(x, 3, axis=-1)
        return x[..., :3]
    a = to_3c(a); b = to_3c(b)
    if HAS_PIL:
        ai = Image.fromarray(a).resize((side, side))
        bi = Image.fromarray(b).resize((side, side))
        return np.asarray(ai), np.asarray(bi)
    # numpy fallback (slow)
    def naive_resize(x):
        h, w = x.shape[:2]
        rh = (np.linspace(0, h - 1, side)).astype(int)
        rw = (np.linspace(0, w - 1, side)).astype(int)
        return x[rh][:, rw]
    return naive_resize(a), naive_resize(b)


def section_5_multiview(idx: DatasetIndex, report_dir: Path,
                        max_samples: int = 200) -> None:
    hr("SECTION 5 — Multi-view consistency", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return
    samples = idx.samples()
    multi = [(k, recs) for k, recs in samples.items() if len(recs) > 1]
    print(f"  total multi-view samples: {len(multi)}")
    if not multi:
        return
    if len(multi) > max_samples:
        rng = np.random.default_rng(0)
        multi = [multi[i] for i in rng.choice(len(multi), size=max_samples, replace=False)]
        print(f"  sampled {len(multi)} for analysis")

    mse_per_pair = []
    cos_per_pair = []
    by_class: dict[str, list[float]] = defaultdict(list)
    for (cls, split, atype, sid), recs in multi:
        recs_sorted = sorted(recs, key=lambda r: r.view or 0)
        arrs = []
        for r in recs_sorted:
            a = open_image_array(r.path)
            if a is None:
                continue
            arrs.append(a)
        if len(arrs) < 2:
            continue
        # pairwise on resized 128x128
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                ai, bj = _common_resize(arrs[i], arrs[j], side=128)
                af = ai.astype(np.float32) / 255.0
                bf = bj.astype(np.float32) / 255.0
                mse = float(((af - bf) ** 2).mean())
                # cosine on flat vectors
                fa = af.ravel() - af.mean()
                fb = bf.ravel() - bf.mean()
                denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
                cos = float((fa * fb).sum() / denom) if denom > 1e-9 else 0.0
                mse_per_pair.append(mse); cos_per_pair.append(cos)
                by_class[cls].append(cos)

    sub("Pairwise inter-view similarity (across all classes)")
    percentile_summary(mse_per_pair, "MSE on 128x128 (lower = more similar)")
    percentile_summary(cos_per_pair, "centred cosine (1.0 = identical)")
    sub("Mean centred-cosine per class")
    rows = []
    for cls in sorted(by_class):
        vals = np.asarray(by_class[cls])
        rows.append({"class": cls, "n_pairs": len(vals),
                     "cos_mean": float(vals.mean()),
                     "cos_p50": float(np.percentile(vals, 50)),
                     "cos_p25": float(np.percentile(vals, 25))})
        print(f"  {cls:<10} n_pairs={len(vals):>5}  "
              f"mean={vals.mean():.4f}  p50={np.percentile(vals, 50):.4f}  "
              f"p25={np.percentile(vals, 25):.4f}")
    save_table(rows, report_dir / "tables" / "05_multiview_similarity.csv")
    print("\n  Read: cos > 0.95 ⇒ views are near-duplicates (pure rotation/lighting).")
    print("  cos in [0.5, 0.9] ⇒ partial overlap — multi-view fusion likely helpful.")
    print("  cos < 0.5 ⇒ very different perspectives — fusion might confuse, treat as augmentation.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Mask coverage
# ─────────────────────────────────────────────────────────────────────────────
def section_6_mask_coverage(idx: DatasetIndex, report_dir: Path) -> dict:
    hr("SECTION 6 — Mask coverage statistics", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return {}

    by_ca = idx.by_class_anomaly()
    rows = []
    all_fracs = []
    per_anom_fracs: dict[tuple[str, str], list[float]] = defaultdict(list)
    binarity_violations = 0
    n_masks_inspected = 0

    for (cls, atype), recs in sorted(by_ca.items()):
        for r in recs:
            if r.mask_path is None:
                continue
            mask = open_mask_array(r.mask_path)
            if mask is None:
                continue
            n_masks_inspected += 1
            # binarity check using PIL again
            try:
                with Image.open(r.mask_path) as im:
                    arr_raw = np.asarray(im)
                if arr_raw.ndim == 3:
                    arr_raw = arr_raw.mean(axis=-1)
                u = np.unique(arr_raw)
                if len(u) > 2:
                    binarity_violations += 1
            except Exception:
                pass
            frac = float(mask.mean())
            all_fracs.append(frac)
            per_anom_fracs[(cls, atype)].append(frac)

    print(f"  total masks inspected             : {n_masks_inspected}")
    print(f"  masks with non-binary values      : {binarity_violations}")
    if all_fracs:
        percentile_summary(all_fracs, "anomaly-pixel fraction (global)")

    sub("Per (class, anomaly_type) anomaly-pixel fraction")
    print(f"  {'class':<10} {'anom':<11} {'n':>4} "
          f"{'mean':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
    for (cls, atype), fracs in sorted(per_anom_fracs.items()):
        a = np.asarray(fracs)
        rows.append({"class": cls, "anomaly_type": atype, "n": len(a),
                     "frac_mean": float(a.mean()),
                     "frac_p25": float(np.percentile(a, 25)),
                     "frac_p50": float(np.percentile(a, 50)),
                     "frac_p75": float(np.percentile(a, 75)),
                     "frac_p95": float(np.percentile(a, 95)),
                     "frac_min": float(a.min()),
                     "frac_max": float(a.max())})
        print(f"  {cls:<10} {atype:<11} {len(a):>4} "
              f"{a.mean():>8.4f} {np.percentile(a, 25):>8.4f} "
              f"{np.percentile(a, 50):>8.4f} {np.percentile(a, 75):>8.4f} "
              f"{np.percentile(a, 95):>8.4f}")
    save_table(rows, report_dir / "tables" / "06_mask_coverage.csv")
    return {"all_fracs": all_fracs, "per_anom_fracs": per_anom_fracs}


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Spatial distribution of anomalies
# ─────────────────────────────────────────────────────────────────────────────
def section_7_spatial(idx: DatasetIndex, report_dir: Path,
                      side: int = 256) -> None:
    hr("SECTION 7 — Spatial distribution of anomalies", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return

    by_ca = idx.by_class_anomaly()
    figdir = report_dir / "figs"; figdir.mkdir(parents=True, exist_ok=True)
    npydir = report_dir / "tables"; npydir.mkdir(parents=True, exist_ok=True)
    print(f"  Building per-class heatmaps at {side}x{side} (sum of resized masks).")

    per_class: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((side, side), dtype=np.float64))
    per_class_n: Counter = Counter()
    global_heat = np.zeros((side, side), dtype=np.float64)
    global_n = 0

    for (cls, atype), recs in by_ca.items():
        for r in recs:
            if r.mask_path is None:
                continue
            mask = open_mask_array(r.mask_path)
            if mask is None or mask.size == 0:
                continue
            # resize to (side, side) using PIL nearest
            try:
                im = Image.fromarray(mask * 255)
                im = im.resize((side, side), resample=Image.NEAREST)
                m = (np.asarray(im) > 0).astype(np.float64)
            except Exception:
                m = mask  # fallback (different shape)
                if m.shape != (side, side):
                    continue
            per_class[cls] += m
            per_class_n[cls] += 1
            global_heat += m
            global_n += 1

    if global_n == 0:
        print("  no masks found, skipping plots"); return

    # Save aggregate heatmap (global)
    np.save(npydir / "07_spatial_heatmap_global.npy", global_heat / global_n)
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(global_heat / global_n, cmap="hot")
        ax.set_title(f"Global anomaly heatmap (N={global_n})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(figdir / "07_heatmap_global.png", dpi=130)
        plt.close(fig)

    print(f"\n  Per-class heatmap statistics (centre-of-mass shift, peak loc, entropy):")
    rows = []
    print(f"  {'class':<10} {'N':>5}  {'cy':>6} {'cx':>6}  "
          f"{'peak_y':>7} {'peak_x':>7}  {'entropy':>8}  {'centre_bias':>11}")
    for cls in sorted(per_class):
        h = per_class[cls] / max(per_class_n[cls], 1)
        # centre-of-mass
        ys, xs = np.indices(h.shape)
        tot = h.sum()
        cy = float((ys * h).sum() / tot) if tot > 0 else 0.0
        cx = float((xs * h).sum() / tot) if tot > 0 else 0.0
        peak = np.unravel_index(int(np.argmax(h)), h.shape)
        # entropy of normalised heatmap
        p = h / tot if tot > 0 else h
        flat = p.ravel(); flat = flat[flat > 0]
        ent = float(-(flat * np.log(flat)).sum())
        # centre-bias: fraction of mass within central 50%
        cbox = h[side // 4: 3 * side // 4, side // 4: 3 * side // 4].sum() / max(tot, 1e-12)
        rows.append({"class": cls, "n_masks": per_class_n[cls],
                     "centre_y": cy, "centre_x": cx,
                     "peak_y": int(peak[0]), "peak_x": int(peak[1]),
                     "entropy_nats": ent, "centre_bias_frac": float(cbox)})
        print(f"  {cls:<10} {per_class_n[cls]:>5}  {cy:>6.1f} {cx:>6.1f}  "
              f"{int(peak[0]):>7} {int(peak[1]):>7}  {ent:>8.3f}  {cbox:>11.4f}")
        np.save(npydir / f"07_spatial_heatmap_{cls}.npy", h)
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.imshow(h, cmap="hot")
            ax.set_title(f"{cls} — anomaly heatmap (N={per_class_n[cls]})")
            ax.axis("off"); fig.tight_layout()
            fig.savefig(figdir / f"07_heatmap_{cls}.png", dpi=130)
            plt.close(fig)
    save_table(rows, report_dir / "tables" / "07_spatial_heatmap_stats.csv")
    print("\n  Read: high centre_bias (>0.7) ⇒ object is centred — a positional prior")
    print("  (e.g. centre-cropped patches) helps. Low centre_bias ⇒ anomalies spread,")
    print("  use a fully translation-invariant model.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Anomaly shape / size analysis
# ─────────────────────────────────────────────────────────────────────────────
def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Returns (labels, n) using scipy if available, else 4-connectivity flood-fill."""
    if HAS_SCIPY:
        labels, n = ndi.label(mask)
        return labels, int(n)
    # Pure-numpy fallback (slow but correct)
    labels = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    stack = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] and labels[r, c] == 0:
                n += 1
                stack.append((r, c))
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                        if mask[y, x] and labels[y, x] == 0:
                            labels[y, x] = n
                            stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
    return labels, n


def section_8_shape(idx: DatasetIndex, report_dir: Path,
                    max_masks: int = 600) -> None:
    hr("SECTION 8 — Anomaly shape / size analysis", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return

    by_ca = idx.by_class_anomaly()
    rows = []
    all_areas = []; all_compact = []; all_aspect = []
    cc_per_image = []
    sample_count = 0

    for (cls, atype), recs in by_ca.items():
        for r in recs:
            if r.mask_path is None:
                continue
            mask = open_mask_array(r.mask_path)
            if mask is None:
                continue
            sample_count += 1
            if sample_count > max_masks:
                break
            labels, ncc = _connected_components(mask)
            cc_per_image.append(ncc)
            if ncc == 0:
                continue
            for k in range(1, ncc + 1):
                comp = (labels == k)
                area = int(comp.sum())
                if area < 4:
                    continue
                # bbox
                ys, xs = np.where(comp)
                bb_h = ys.max() - ys.min() + 1
                bb_w = xs.max() - xs.min() + 1
                bb_area = int(bb_h * bb_w)
                fill = area / bb_area if bb_area > 0 else 0.0
                aspect = bb_w / bb_h if bb_h > 0 else 0.0
                # crude perimeter via 4-neighbour edge count
                if HAS_SCIPY:
                    eroded = ndi.binary_erosion(comp).astype(np.uint8)
                    perim = int((comp.astype(np.uint8) - eroded).sum())
                else:
                    pad = np.zeros((comp.shape[0] + 2, comp.shape[1] + 2), dtype=bool)
                    pad[1:-1, 1:-1] = comp
                    perim = int(((pad[1:-1, 1:-1] & ~pad[:-2, 1:-1]).sum()
                                + (pad[1:-1, 1:-1] & ~pad[2:, 1:-1]).sum()
                                + (pad[1:-1, 1:-1] & ~pad[1:-1, :-2]).sum()
                                + (pad[1:-1, 1:-1] & ~pad[1:-1, 2:]).sum()))
                compact = (4 * math.pi * area) / max(perim ** 2, 1)
                rows.append({"class": cls, "anomaly_type": atype,
                             "area_px": area, "bbox_area_px": bb_area,
                             "fill_ratio": float(fill),
                             "aspect_ratio_wh": float(aspect),
                             "perimeter_px": int(perim),
                             "compactness": float(compact),
                             "img_h": int(mask.shape[0]),
                             "img_w": int(mask.shape[1]),
                             "area_frac": float(area / mask.size)})
                all_areas.append(area); all_compact.append(compact)
                all_aspect.append(aspect)
        if sample_count > max_masks:
            break
    save_table(rows, report_dir / "tables" / "08_shape_components.csv")

    print(f"  inspected up to {max_masks} masks; total CCs analysed: {len(rows)}")
    sub("Connected-components per mask")
    percentile_summary(cc_per_image, "n_CC per mask")
    sub("Component-level distributions")
    percentile_summary(all_areas, "area (pixels)")
    percentile_summary([r["area_frac"] for r in rows], "area_frac (of image)")
    percentile_summary(all_compact, "compactness (4πA / P²) — 1=circle, 0=line")
    percentile_summary(all_aspect, "aspect ratio (w/h)")

    # Histogram plots
    if HAS_MPL and rows:
        figdir = report_dir / "figs"
        figdir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        ax[0].hist(np.log10(np.asarray(all_areas) + 1), bins=40)
        ax[0].set_title("log10(area+1)")
        ax[1].hist(all_compact, bins=30)
        ax[1].set_title("compactness")
        ax[2].hist(np.log10(np.asarray(all_aspect) + 1e-3), bins=30)
        ax[2].set_title("log10(aspect)")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "08_shape_hist.png", dpi=120)
        plt.close(fig)

    print("\n  Read: typical area & compactness drive PATCH SIZE in patch-based methods.")
    print("    p95(area) ≈ patch area; small p25 (~10s of px) ⇒ need fine spatial output.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Per-class × per-anomaly-type breakdown
# ─────────────────────────────────────────────────────────────────────────────
def section_9_breakdown(idx: DatasetIndex, s6: dict, report_dir: Path) -> None:
    hr("SECTION 9 — Per-class × per-anomaly-type breakdown (hardness ranking)", "=")
    by_ca = idx.by_class_anomaly()
    per_anom = s6.get("per_anom_fracs", {})
    rows = []
    for (cls, atype), recs in sorted(by_ca.items()):
        fracs = per_anom.get((cls, atype), [])
        n_imgs = len(recs)
        n_with_mask = sum(1 for r in recs if r.mask_path is not None)
        n_views = len({r.view for r in recs if r.view is not None})
        n_samples = len({r.sample_id for r in recs})
        rows.append({"class": cls, "human": CLASS_HUMAN.get(cls, "?"),
                     "anomaly_type": atype, "n_imgs": n_imgs,
                     "n_with_mask": n_with_mask, "n_views": n_views,
                     "n_unique_samples": n_samples,
                     "frac_p50": float(np.percentile(fracs, 50)) if fracs else 0.0,
                     "frac_p95": float(np.percentile(fracs, 95)) if fracs else 0.0})
    rows.sort(key=lambda r: r["frac_p50"])
    print(f"  {'class':<10} {'obj':<11} {'anom':<11} {'n_img':>5} "
          f"{'mask':>4} {'samples':>7} {'views':>5} {'frac_p50':>9} {'frac_p95':>9}")
    for r in rows:
        print(f"  {r['class']:<10} {r['human']:<11} {r['anomaly_type']:<11} "
              f"{r['n_imgs']:>5} {r['n_with_mask']:>4} "
              f"{r['n_unique_samples']:>7} {r['n_views']:>5} "
              f"{r['frac_p50']:>9.4f} {r['frac_p95']:>9.4f}")
    save_table(rows, report_dir / "tables" / "09_per_class_anomaly.csv")
    print("\n  Sorted ascending by median anomaly fraction (top = hardest = smallest defects).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Anomaly description text analysis
# ─────────────────────────────────────────────────────────────────────────────
def _tokenise(s: str) -> list[str]:
    s = s.lower()
    return re.findall(r"[a-z]{3,}", s)


GENERIC_PHRASE = "localized visual anomaly affecting the object surface"

def section_10_descriptions(csv_path: Path, report_dir: Path) -> None:
    hr("SECTION 10 — Anomaly description text analysis", "=")
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found"); return
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({k: (v or "") for k, v in r.items()})
    print(f"  rows in CSV: {len(rows)}")
    if not rows:
        return
    classes = Counter(r["public_class"] for r in rows)
    objects = Counter(r["object_name"] for r in rows)
    print(f"  unique classes : {len(classes)}")
    print(f"  unique objects : {len(objects)}  -> {dict(objects)}")
    desc_lengths = [len(r["description"].split()) for r in rows]
    percentile_summary(desc_lengths, "description length (words)")
    n_generic = sum(1 for r in rows if GENERIC_PHRASE in r["description"].lower())
    print(f"  generic 'Localized visual anomaly...' rows: {n_generic} "
          f"({n_generic / len(rows):.4f})")

    sub("Rich vs generic descriptions per class")
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["public_class"]].append(r)
    print(f"  {'class':<10} {'object':<12} {'n_anom':>7} {'n_rich':>7} {'n_generic':>10}")
    out_rows = []
    for cls, items in sorted(by_class.items()):
        n_rich = sum(1 for r in items if GENERIC_PHRASE not in r["description"].lower())
        n_generic_c = len(items) - n_rich
        out_rows.append({"class": cls, "object": items[0]["object_name"],
                         "n_anom_types": len(items),
                         "n_rich_descriptions": n_rich,
                         "n_generic_descriptions": n_generic_c})
        print(f"  {cls:<10} {items[0]['object_name']:<12} "
              f"{len(items):>7} {n_rich:>7} {n_generic_c:>10}")
    save_table(out_rows, report_dir / "tables" / "10_descriptions_summary.csv")

    sub("Top words per class (excluding stop list)")
    stop = {"the", "and", "for", "with", "are", "have", "has", "may", "this",
            "from", "due", "more", "less", "very", "also", "into", "onto",
            "single", "multiple", "small", "medium", "large", "size", "sizes",
            "variable", "depending", "extent", "areas", "area", "color",
            "colour", "same", "standard", "likely", "cause", "causes", "caused",
            "causing", "during", "leading", "leads", "type", "types", "shape",
            "different", "various", "visual", "cue", "irregular", "blotchy",
            "surface", "object", "linear", "jagged", "patch", "patches",
            "covered", "potential", "possibility", "appears", "manufacturing",
            "handling", "processing", "stress", "damage", "exposure"}
    for cls, items in sorted(by_class.items()):
        c = Counter()
        for r in items:
            for tok in _tokenise(r["description"]):
                if tok not in stop:
                    c[tok] += 1
        top = c.most_common(12)
        toks = ", ".join(f"{t}({n})" for t, n in top)
        print(f"  {cls:<10}  {items[0]['object_name']:<12}  {toks}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 — Object/background heuristics
# ─────────────────────────────────────────────────────────────────────────────
def _edge_density(arr: np.ndarray) -> float:
    """Fraction of pixels with |∇| > a threshold — proxy for texture/edges."""
    if arr.ndim == 3:
        g = arr.mean(axis=-1)
    else:
        g = arr
    g = g.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    dy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    mag = np.sqrt(dx * dx + dy * dy)
    return float((mag > 12.0).mean())


def section_11_background(idx: DatasetIndex, report_dir: Path,
                          max_per_class: int = 80) -> None:
    hr("SECTION 11 — Object / background heuristics", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return

    by_cs = idx.by_class_split()
    rows = []
    print(f"  Sampling up to {max_per_class} train/good images per class.")
    for cls in idx.classes:
        recs = by_cs.get((cls, "train_good"), [])
        if not recs:
            continue
        sample = sample_records(recs, max_per_class, seed=hash(("bg", cls)) & 0xFFFF)
        eds = []
        bg_unifs = []
        for r in sample:
            arr = open_image_array(r.path)
            if arr is None:
                continue
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[..., :3]
            eds.append(_edge_density(arr))
            # BG uniformity: std of the 4 corner crops (each = 10% side)
            h, w = arr.shape[:2]
            sh = max(int(h * 0.1), 4); sw = max(int(w * 0.1), 4)
            corners = [arr[:sh, :sw], arr[:sh, -sw:], arr[-sh:, :sw], arr[-sh:, -sw:]]
            cs = [c.astype(np.float32) for c in corners]
            cs_mean = np.array([c.reshape(-1, c.shape[-1] if c.ndim == 3 else 1).mean(axis=0)
                                for c in cs])
            bg_unifs.append(float(cs_mean.std()))
        rows.append({"class": cls, "human": CLASS_HUMAN.get(cls, "?"),
                     "n_sampled": len(eds),
                     "edge_density_mean": float(np.mean(eds)) if eds else 0.0,
                     "edge_density_p50": float(np.percentile(eds, 50)) if eds else 0.0,
                     "bg_corner_std_mean": float(np.mean(bg_unifs)) if bg_unifs else 0.0})
    print(f"\n  {'class':<10} {'object':<11} {'n':>4} {'edge_d_mean':>12} "
          f"{'edge_d_p50':>11} {'bg_corner_std':>14}")
    for r in rows:
        print(f"  {r['class']:<10} {r['human']:<11} {r['n_sampled']:>4} "
              f"{r['edge_density_mean']:>12.4f} {r['edge_density_p50']:>11.4f} "
              f"{r['bg_corner_std_mean']:>14.3f}")
    save_table(rows, report_dir / "tables" / "11_background.csv")
    print("\n  Read:")
    print("    bg_corner_std small (< 10) ⇒ uniform background, foreground masking is easy")
    print("    edge_density high (> 0.15) ⇒ textured object (gear, threads, beans) — patch")
    print("    methods like PatchCore / EfficientAD will benefit from texture features.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 — Submission scoping
# ─────────────────────────────────────────────────────────────────────────────
def section_12_submission(idx: DatasetIndex, report_dir: Path,
                          max_per_class: int = 50) -> None:
    hr("SECTION 12 — Submission scoping", "=")
    if not HAS_PIL:
        print("  [SKIP] PIL not installed"); return
    by_cs = idx.by_class_split()
    test_total = 0
    rows = []
    res_seen: Counter = Counter()
    for cls in idx.classes:
        recs = by_cs.get((cls, "test"), [])
        test_total += len(recs)
        sample = sample_records(recs, max_per_class, seed=hash(("sub", cls)) & 0xFFFF)
        for r in sample:
            arr = open_image_array(r.path)
            if arr is None:
                continue
            res_seen[(arr.shape[0], arr.shape[1])] += 1
        rows.append({"class": cls, "n_test_images": len(recs),
                     "sampled_resolutions": len(set(res_seen))})
    print(f"  total test images   : {test_total}")
    print(f"  expected submission rows (one per test image): {test_total}")
    print(f"\n  Test image resolution distribution (sampled):")
    for (h, w), n in sorted(res_seen.items(), key=lambda kv: -kv[1]):
        print(f"    {h:>4} x {w:<4}  count={n}")
    print(f"\n  q8rle header reminder: 'q8rle <H> <W> <v> <l> ...' — H is rows, W is cols.")
    print("  Use the EXACT (H, W) of each test image (NOT a rescaled version) when encoding.")
    save_table(rows, report_dir / "tables" / "12_submission_scoping.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 — Pixel-AP imbalance
# ─────────────────────────────────────────────────────────────────────────────
def section_13_pixel_ap(s6: dict, report_dir: Path) -> None:
    hr("SECTION 13 — Pixel-AP imbalance analysis", "=")
    fracs = s6.get("all_fracs", [])
    if not fracs:
        print("  [SKIP] no mask fractions"); return
    a = np.asarray(fracs)
    pos_global = float(a.mean())
    print(f"  mean anomaly-pixel fraction over LABELED anomaly images: {pos_global:.6f}")
    print(f"    => a constant-score predictor that scores 1 everywhere has AP ≈ {pos_global:.6f}")
    print(f"    => a random uniform predictor likewise has AP ≈ {pos_global:.6f}")
    # If clean (good) test images dominate, the actual test positive rate can be ≪ this.
    # We can't see test masks, but an upper bound for AP-baseline:
    print(f"\n  NOTE: actual test set mixes clean and anomalous images, so the")
    print(f"  effective positive-pixel fraction is LOWER — expect random AP < "
          f"{pos_global:.4f}.")
    print(f"  Anything ≥ 0.20 AP is meaningful; ≥ 0.40 is strong; ≥ 0.55 is podium-level.")
    print(f"  (Calibrate against the public leaderboard once you submit a baseline.)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 14 — Recommendations
# ─────────────────────────────────────────────────────────────────────────────
def section_14_recommendations(idx: DatasetIndex, report_dir: Path) -> None:
    hr("SECTION 14 — Modelling recommendations", "=")
    print("  These are mechanical translations of the priors above. After")
    print("  running this script, refine each bullet against the actual numbers.")
    print()
    print("  ── DATA HANDLING ─────────────────────────────────────────────────")
    print("  • Keep ORIGINAL test resolution for q8rle encoding, but train on a")
    print("    fixed common size (e.g. 256x256 or 320x320 if memory allows).")
    print("  • Use TRAIN/GOOD as the unsupervised signal and the per-class set of")
    print("    train/anomaly_*/ as a tiny supervised validation set (one example per")
    print("    type means leave-one-out is the only honest way to validate).")
    print("  • The anomaly_descriptions.csv example_image_relpath gives you 1 named")
    print("    'reference' anomaly per (class, anomaly_type). Use this for visual ")
    print("    inspection AND as test-time templates if you go a few-shot route.")
    print()
    print("  ── ARCHITECTURE FAMILIES (in increasing complexity) ─────────────")
    print("  1. Reconstruction-based (autoencoders, FastFlow): cheap, decent on ")
    print("     uniform-background classes (resistor, screw, capsule).")
    print("  2. Feature-distance / memory-bank (PatchCore): strong default when ")
    print("     train_good per class > ~50 — uses ImageNet-pretrained CNN features ")
    print("     (e.g., WideResNet50 layers 2+3) + nearest-neighbour over a ")
    print("     coreset. Easy to implement, explainable scores.")
    print("  3. Student-teacher (EfficientAD, RD4AD): pretrained teacher + small ")
    print("     student trained only on good images. Pixel-anomaly = student-")
    print("     teacher feature gap. Strong on textured classes (gear, beans).")
    print("  4. Diffusion / DDIM-based (e.g., AnomalyDiffusion): heaviest, ")
    print("     usually needed only for long-tail anomaly types.")
    print()
    print("  ── MULTI-VIEW POLICY ────────────────────────────────────────────")
    print("  • Section 5 cosine tells you the right policy:")
    print("      cos > 0.95 ⇒ treat each view as a duplicate/augmentation of the same scene.")
    print("      0.7 < cos ≤ 0.95 ⇒ ensemble at score level (max-pool sum of per-view AP scores).")
    print("      cos ≤ 0.7 ⇒ treat views as independent and submit per-view masks separately.")
    print()
    print("  ── PATCH SIZE & RECEPTIVE FIELD ─────────────────────────────────")
    print("  • Pick the patch size so that p95(component-area) ≤ patch_area:")
    print("      e.g., if p95 = 4 000 px on 256x256 images, that's 4000/65536 ≈ 6%")
    print("      ⇒ a 32x32 patch (1024 px) is too small; use 64x64 (4096 px).")
    print("  • For very small p25 (< ~30 px), include a high-resolution head.")
    print()
    print("  ── AUGMENTATION ─────────────────────────────────────────────────")
    print("  • Always: small rotation (±5°), small affine, light colour jitter.")
    print("  • If multi-view cos < 0.9, ALSO add stronger geometric aug (the views")
    print("    themselves give you a free perspective-aug curriculum).")
    print("  • For textured classes, add CutPaste/Synthetic-defect aug — paste random")
    print("    noise patches onto train/good and use train/anomaly_* masks as templates.")
    print()
    print("  ── LOSS & POST-PROCESSING ───────────────────────────────────────")
    print("  • Pixel AP rewards high-recall scoring; do NOT threshold. Submit raw")
    print("    [0,1] scores quantised to 8 bits via q8rle (the format does that for you).")
    print("  • Median-blur or Gaussian-smooth your score map BEFORE q8rle: small smoothing")
    print("    (σ ≈ 1–2 px) usually adds 0.5–1 AP point on MVTec-style tasks.")
    print("  • Centre-bias: if Section 7 shows centre_bias > 0.7 for a class,")
    print("    multiply scores by a soft centre-Gaussian prior at inference.")
    print()
    print("  ── EVALUATION & TRACKING ────────────────────────────────────────")
    print("  • Implement a local pixel-AP scorer that reads a held-out subset of")
    print("    train/anomaly_* (with masks) — leave one anomaly_type out per class,")
    print("    compute per-class AP, then mean-AP. Use this for hyperparameter sweeps;")
    print("    do not waste leaderboard submissions on tuning.")
    print("  • The competition gives 10 submissions/day — schedule them as")
    print("    1 ablation/day max to reserve probes for the final week.")
    print()
    print("  ── 14-DAY PLAN ──────────────────────────────────────────────────")
    print("  Days  1–3   : EDA (this script), end-to-end pipeline with PatchCore baseline,")
    print("                local AP harness, first leaderboard probe.")
    print("  Days  4–7   : per-class architecture A/B (PatchCore vs EfficientAD vs RD4AD),")
    print("                pick a winner per class, ensemble at score-map level.")
    print("  Days  8–11  : multi-view fusion, augmentation tuning, score smoothing,")
    print("                centre-bias prior. Probe leaderboard 2–3x.")
    print("  Days 12–13  : freeze, run TTA (flips, multi-scale), final probe.")
    print("  Day  14     : submit, write the report (which is graded), prep oral exam.")
    print()
    print("  Don't forget: every team member needs ≥1 submission to get any points.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--csv",        type=Path, default=DEFAULT_CSV)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--max-images-per-bucket", type=int, default=200,
                    help="cap on images sampled per (class, split) bucket — speed knob")
    ap.add_argument("--max-multiview-samples", type=int, default=200)
    ap.add_argument("--max-shape-masks", type=int, default=600)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="section ids to skip, e.g. 5 7 8")
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "report.txt"
    skip = {str(s) for s in args.skip}

    with tee_to(report_path):
        hr("SPACEPRESSO ANOMALY-DETECTION DATASET ANALYSIS", "█")
        print(f"  data_root  : {args.data_root}")
        print(f"  csv        : {args.csv}")
        print(f"  report_dir : {args.report_dir}")
        print(f"  PIL avail  : {HAS_PIL}    SciPy avail: {HAS_SCIPY}    "
              f"matplotlib: {HAS_MPL}    pandas: {HAS_PANDAS}")
        print(f"  knobs      : max_images_per_bucket={args.max_images_per_bucket}  "
              f"max_multiview={args.max_multiview_samples}  "
              f"max_shape_masks={args.max_shape_masks}")
        t0 = time.time()

        idx = scan_dataset(args.data_root)
        if not idx.records:
            print("\n  [FATAL] no images found — check --data-root path and folder layout.")
            return

        section_1_inventory(idx, args.report_dir)

        s2 = {} if "2" in skip else section_2_image_meta(
            idx, args.report_dir, args.max_images_per_bucket)
        s3 = {} if "3" in skip else section_3_pixel_stats(
            idx, args.report_dir, args.max_images_per_bucket)
        if "4" not in skip:
            section_4_domain_shift(s3, idx, args.report_dir)
        if "5" not in skip:
            section_5_multiview(idx, args.report_dir, args.max_multiview_samples)
        s6 = {} if "6" in skip else section_6_mask_coverage(idx, args.report_dir)
        if "7" not in skip:
            section_7_spatial(idx, args.report_dir)
        if "8" not in skip:
            section_8_shape(idx, args.report_dir, args.max_shape_masks)
        if "9" not in skip:
            section_9_breakdown(idx, s6, args.report_dir)
        if "10" not in skip:
            section_10_descriptions(args.csv, args.report_dir)
        if "11" not in skip:
            section_11_background(idx, args.report_dir)
        if "12" not in skip:
            section_12_submission(idx, args.report_dir)
        if "13" not in skip:
            section_13_pixel_ap(s6, args.report_dir)
        if "14" not in skip:
            section_14_recommendations(idx, args.report_dir)

        hr(f"DONE in {time.time() - t0:.1f}s — paste report.txt back to chat", "█")
        print(f"  Full report  : {report_path}")
        print(f"  Tables (CSV) : {args.report_dir / 'tables'}")
        print(f"  Figures (PNG): {args.report_dir / 'figs'}")


if __name__ == "__main__":
    main()