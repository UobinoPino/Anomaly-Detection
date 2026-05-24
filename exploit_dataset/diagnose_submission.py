"""Pre-submission sanity check for q8rle CSV outputs.

Goal
----
Catch in 30 seconds the kinds of damage that aren't visible from
ensemble_log.txt or local AP:
  - Scores not in [0, 1] when quantised → saturated to 0 or 255.
  - Maps too dense (RLE compression breaking down → huge zip).
  - Per-class score distributions wildly different (one class
    dominating the leaderboard ranking).
  - Average positive-fraction implausibly high (everything looks
    anomalous → leaderboard score collapses).
  - Mass concentrated at the global max (looks like clipping).

Usage
-----
    python diagnose_submission.py PATH/TO/submission.csv
    python diagnose_submission.py PATH/TO/submission.zip
    # optional: also compare against per-method submissions or known-good baseline
    python diagnose_submission.py PATH/TO/submission.csv \\
        --compare BASELINE_SUBMISSION.csv

What the output tells you
-------------------------
- Mean quantised value across ALL pixels: a working submission has
  most pixels close to 0 (background). Number above ~50 is suspicious;
  above ~100 means almost every pixel claims to be anomalous.
- Per-class mean q-value: tells you whether one class is silenced
  (and one is shouting). For a calibrated submission these should be
  roughly comparable (within 20-30%); wildly different means the
  classes are not on the same effective scale.
- RLE compression ratio: pairs-per-image. A healthy AD map has
  20-200 distinct runs per 224x224 image. A noisy map has thousands.
  This is exactly the leading indicator for huge zip sizes.
- Per-class histogram (10 bins): sanity-check the distribution.
  Healthy submissions are HEAVILY left-skewed (most mass near 0).
- Top-1% threshold per class: tells you whether the "tail" — the
  pixels you actually want at the top of the ranking — exists.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Bump csv field-size limit — a single q8rle entry can be > 256 KB when
# the submission is pathologically dense, and Python's default 131072
# raises _csv.Error. We use sys.maxsize / 2 to avoid platform-dependent
# OverflowErrors that bite when using sys.maxsize directly on Windows.
csv.field_size_limit(sys.maxsize // 2)


def q8rle_to_array(s: str) -> np.ndarray:
    """Decode a q8rle string back to uint8 (no scaling)."""
    t = s.split()
    h, w = int(t[1]), int(t[2])
    if len(t) <= 3:
        return np.zeros((h, w), dtype=np.uint8)
    vals = np.array(list(map(int, t[3::2])), dtype=np.uint8)
    lens = np.array(list(map(int, t[4::2])), dtype=np.int64)
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat


def n_runs(s: str) -> int:
    """Number of RLE pairs (proxy for compression ratio)."""
    t = s.split()
    # pairs start at index 3, every 2 tokens = 1 run
    return max(0, (len(t) - 3) // 2)


def parse_class_from_id(id_str: str,
                          id_to_class: dict[str, str] | None = None) -> str:
    """Look up the class for a submission ID.

    Submission IDs (e.g., 'img_000001_view1') contain NO class info.
    The caller must provide an id_to_class mapping built from any of
    the per-method test_predictions.npz files (every method shares the
    same id↔class mapping). Falls back to substring matching if no
    mapping is provided (only useful for debug/synthetic IDs).
    """
    if id_to_class is not None:
        if id_str in id_to_class:
            return id_to_class[id_str]
    # Legacy/debug fallback: substring match
    parts = id_str.replace("/", "_").split("_")
    for i, p in enumerate(parts):
        if p == "class" and i + 1 < len(parts):
            return f"class_{parts[i + 1]}"
    return "unknown"


def load_id_class_map(npz_paths: list[Path]) -> dict[str, str]:
    """Build a single id→class dict by unioning multiple npz files."""
    mapping: dict[str, str] = {}
    for path in npz_paths:
        try:
            data = np.load(path, allow_pickle=True)
            ids = data["ids"]
            classes = data["classes"]
            for k in range(len(ids)):
                mapping[str(ids[k])] = str(classes[k])
        except Exception as e:
            print(f"  WARN: cannot load id-class map from {path}: {e}",
                  file=sys.stderr)
    return mapping


def open_csv(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = [n for n in zf.namelist() if n.endswith(".csv")]
            if not inner:
                raise RuntimeError(f"no .csv inside {path}")
            with zf.open(inner[0]) as f:
                text = f.read().decode("utf-8")
                return io.StringIO(text), inner[0]
    return open(path, "r", encoding="utf-8"), path.name


def histogram_str(values: np.ndarray, n_bins: int = 10) -> str:
    counts, edges = np.histogram(values, bins=n_bins, range=(0, 256))
    total = counts.sum()
    if total == 0:
        return "(empty)"
    parts = []
    for c, e in zip(counts, edges[:-1]):
        pct = c / total * 100
        parts.append(f"[{int(e):>3}-{int(e+25):>3}] {pct:>5.1f}%")
    return "  ".join(parts)


def analyse(csv_stream, label: str,
            id_to_class: dict[str, str] | None = None) -> dict:
    """Read a submission CSV and return a dict of stats per class + global."""
    reader = csv.reader(csv_stream)
    header = next(reader)
    if header[0].lower() != "id" or header[1].lower() != "label":
        print(f"  WARN: unexpected header {header}", file=sys.stderr)

    n_rows = 0
    n_runs_per_image = []
    per_class = defaultdict(lambda: {
        "sum_q": 0,
        "n_pix": 0,
        "max_q": 0,
        "all_q_samples": [],     # sample for histogram
        "n_at_max": 0,
        "n_at_zero": 0,
        "n_images": 0,
    })

    sample_every = 5            # sample every Nth image for histogram
    sample_pix_per_img = 5000   # number of pixels to sample per image

    rng = np.random.default_rng(0)

    for row in reader:
        if len(row) < 2:
            continue
        id_str, q8 = row[0], row[1]
        cls = parse_class_from_id(id_str, id_to_class)
        cls_d = per_class[cls]
        n_rows += 1
        cls_d["n_images"] += 1

        # Cheap: pair count (drives RLE compression)
        n_runs_per_image.append(n_runs(q8))

        # Decode for full pixel stats
        arr = q8rle_to_array(q8)
        cls_d["sum_q"] += int(arr.sum())
        cls_d["n_pix"] += int(arr.size)
        cls_d["max_q"] = max(cls_d["max_q"], int(arr.max()))
        cls_d["n_at_max"] += int((arr == 255).sum())
        cls_d["n_at_zero"] += int((arr == 0).sum())

        # Sample for histogram
        if (n_rows % sample_every) == 0:
            flat = arr.ravel()
            if flat.size > sample_pix_per_img:
                idx = rng.choice(flat.size, sample_pix_per_img,
                                  replace=False)
                cls_d["all_q_samples"].extend(flat[idx].tolist())
            else:
                cls_d["all_q_samples"].extend(flat.tolist())

    n_runs_arr = np.asarray(n_runs_per_image, dtype=np.int64)

    return {
        "label": label,
        "n_rows": n_rows,
        "n_runs": n_runs_arr,
        "per_class": per_class,
    }


def print_report(stats: dict):
    label = stats["label"]
    n_rows = stats["n_rows"]
    n_runs_arr = stats["n_runs"]
    per_class = stats["per_class"]

    print()
    print("=" * 78)
    print(f"  REPORT  —  {label}")
    print("=" * 78)
    print(f"  rows: {n_rows}")

    # ── 1. RLE compression diagnostic ──────────────────────────────────────
    print("\n[1/5] RLE compression (pairs per image — lower = better compression)")
    median = int(np.median(n_runs_arr))
    p99    = int(np.percentile(n_runs_arr, 99))
    p_max  = int(n_runs_arr.max())
    p_min  = int(n_runs_arr.min())
    bytes_est = float(n_runs_arr.sum() * 6) / 1e6  # rough: 6 bytes per pair
    print(f"  pairs/image   min={p_min}  median={median}  "
          f"p99={p99}  max={p_max}")
    print(f"  est. CSV size ~ {bytes_est:.1f} MB (zip ~ 1/3)")
    if median > 5000:
        print(f"  ⚠️  median {median} pairs/image is VERY HIGH. Submission "
              f"will be huge. This usually means your final ECDF or "
              f"per-pixel calibration is forcing every pixel to a "
              f"different bucket — there is no background.")
    elif median > 1000:
        print(f"  ⚠️  median {median} pairs/image is high — your maps "
              f"are quite noisy. Consider a final threshold/sparsify "
              f"step.")
    else:
        print(f"  ✓   median {median} pairs/image is healthy.")

    # ── 2. Per-class q-value mean ──────────────────────────────────────────
    print("\n[2/5] Per-class mean quantised value (out of 255)")
    print(f"  Healthy: most classes 5-40 (mostly background). "
          f"Above 80 → almost-everything-anomalous.")
    print(f"  {'class':<12} {'mean_q':>8}  {'max_q':>5}  "
          f"{'frac=255':>10}  {'frac=0':>8}  {'n':>5}")
    print(f"  {'-'*12} {'-'*8}  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*5}")
    class_means = []
    for cls in sorted(per_class):
        d = per_class[cls]
        if d["n_pix"] == 0:
            continue
        mean_q = d["sum_q"] / d["n_pix"]
        frac_at_max = d["n_at_max"] / d["n_pix"]
        frac_at_zero = d["n_at_zero"] / d["n_pix"]
        class_means.append((cls, mean_q))
        marker = ""
        if mean_q > 100:
            marker = " ⚠️ very dense"
        elif mean_q > 60:
            marker = " ⚠ denser than expected"
        elif mean_q < 3:
            marker = " ⚠ extremely sparse — model silent?"
        print(f"  {cls:<12} {mean_q:>8.2f}  {d['max_q']:>5}  "
              f"{frac_at_max:>10.4%}  {frac_at_zero:>8.2%}  "
              f"{d['n_images']:>5}{marker}")

    if class_means:
        vals = [v for _, v in class_means]
        cv = float(np.std(vals) / max(np.mean(vals), 1e-6))
        print(f"\n  coefficient of variation across classes: {cv:.3f}")
        if cv > 0.5:
            print(f"  ⚠️ classes are on very different scales. Cross-class "
                  f"ranking will be dominated by whichever class shouts.")
        elif cv > 0.3:
            print(f"  ⚠ class scales differ noticeably but not catastrophically.")
        else:
            print(f"  ✓ classes are on comparable scales.")

    # ── 3. Per-class histogram of pixel values ─────────────────────────────
    print("\n[3/5] Pixel-value histogram per class")
    print(f"  Healthy: bin [0-25] >70%, monotonically decreasing.")
    print(f"  Bad: roughly flat distribution → every pixel evenly "
          f"distributed → final ECDF turned the map into uniform noise.")
    for cls in sorted(per_class):
        d = per_class[cls]
        samples = np.asarray(d["all_q_samples"], dtype=np.uint8)
        if samples.size == 0:
            continue
        first_bin_pct = float((samples < 25).sum() / samples.size * 100)
        marker = ""
        if first_bin_pct < 30:
            marker = " ⚠ NOT sparse"
        elif first_bin_pct < 50:
            marker = " ⚠ less sparse than expected"
        print(f"  {cls}  ({samples.size} pix)  "
              f"frac<25={first_bin_pct:.1f}%{marker}")
        print(f"    {histogram_str(samples)}")

    # ── 4. Top-1% threshold (the actual ranking signal) ────────────────────
    print("\n[4/5] Per-class top-1% threshold (where AP's positive "
          "signal lives)")
    print(f"  Healthy: a number well above the class mean. If top-1% is "
          f"only slightly above the mean, the model has nothing to say.")
    for cls in sorted(per_class):
        d = per_class[cls]
        samples = np.asarray(d["all_q_samples"], dtype=np.uint8)
        if samples.size == 0:
            continue
        mean_q = d["sum_q"] / d["n_pix"]
        top99 = float(np.percentile(samples, 99))
        top999 = float(np.percentile(samples, 99.9))
        ratio = top99 / max(mean_q, 1e-6)
        marker = " ✓" if ratio > 3 else " ⚠ flat" if ratio < 1.5 else ""
        print(f"  {cls}  mean={mean_q:.1f}  p99={top99:.1f}  "
              f"p99.9={top999:.1f}  ratio_p99/mean={ratio:.2f}{marker}")

    # ── 5. Summary ─────────────────────────────────────────────────────────
    print("\n[5/5] Summary")
    score_issues = 0
    if int(np.median(n_runs_arr)) > 5000:
        score_issues += 1
        print(f"  ❌ RLE compression: catastrophic ({int(np.median(n_runs_arr))} pairs/img)")
    elif int(np.median(n_runs_arr)) > 1000:
        score_issues += 1
        print(f"  ⚠️ RLE compression: degraded")
    else:
        print(f"  ✓ RLE compression looks healthy")

    overall_mean = sum(d["sum_q"] for d in per_class.values()) / max(
        sum(d["n_pix"] for d in per_class.values()), 1)
    if overall_mean > 80:
        score_issues += 1
        print(f"  ❌ overall mean q={overall_mean:.1f}: maps are very dense")
    elif overall_mean > 40:
        score_issues += 1
        print(f"  ⚠ overall mean q={overall_mean:.1f}: maps denser than ideal")
    else:
        print(f"  ✓ overall mean q={overall_mean:.1f}")

    if score_issues == 0:
        print(f"\n  >>> Submission looks healthy. Upload.")
    else:
        print(f"\n  >>> {score_issues} issue(s) detected. Investigate "
              f"before uploading.")


def compare(stats_a: dict, stats_b: dict):
    """Compare key metrics between two submissions side by side."""
    print()
    print("=" * 78)
    print(f"  COMPARE  —  {stats_a['label']}   vs   {stats_b['label']}")
    print("=" * 78)
    classes = sorted(set(stats_a["per_class"]) | set(stats_b["per_class"]))
    print(f"  {'class':<12}  {'mean_q (A)':>11}  {'mean_q (B)':>11}  "
          f"{'ratio B/A':>10}")
    for cls in classes:
        da = stats_a["per_class"].get(cls)
        db = stats_b["per_class"].get(cls)
        ma = (da["sum_q"] / max(da["n_pix"], 1)) if da else 0
        mb = (db["sum_q"] / max(db["n_pix"], 1)) if db else 0
        ratio = mb / max(ma, 1e-6)
        print(f"  {cls:<12}  {ma:>11.2f}  {mb:>11.2f}  {ratio:>10.2f}")
    print(f"\n  Median pairs/image:  A={int(np.median(stats_a['n_runs']))}  "
          f"B={int(np.median(stats_b['n_runs']))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--compare", type=Path, default=None,
                    help="Optional second submission to compare against.")
    ap.add_argument("--id-class-map", type=Path, action="append",
                    default=[], metavar="TEST_PREDICTIONS_NPZ",
                    help="Path to any test_predictions.npz — its "
                         "(ids, classes) arrays are used to map "
                         "submission IDs to class names. Repeatable. "
                         "If omitted, per-class stats default to "
                         "'unknown'.")
    args = ap.parse_args()

    if not args.submission.exists():
        print(f"file not found: {args.submission}", file=sys.stderr)
        return 1

    id_to_class = (load_id_class_map(args.id_class_map)
                   if args.id_class_map else None)
    if id_to_class:
        print(f"  loaded id→class mapping for {len(id_to_class)} ids "
              f"from {len(args.id_class_map)} file(s)")

    stream, name = open_csv(args.submission)
    stats_a = analyse(stream, name, id_to_class)
    print_report(stats_a)

    if args.compare:
        stream_b, name_b = open_csv(args.compare)
        stats_b = analyse(stream_b, name_b, id_to_class)
        print_report(stats_b)
        compare(stats_a, stats_b)

    return 0


if __name__ == "__main__":
    sys.exit(main())