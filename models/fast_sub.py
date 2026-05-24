"""Simple score fusion: average N submission CSVs into one.

Each submission is a CSV with header `ID,Label` and rows of
`ID, q8rle_encoded_score_map`. We decode each map, average pixel-wise
across all submissions (uniform weights), and re-encode.

Usage:
    uv run python score_fusion_simple.py \\
        --runs sub1.csv sub2.csv sub3.csv \\
        --out fused/submission.csv

Notes:
    - Accepts any number of submissions (>= 2).
    - Uses the intersection of IDs if some submissions are missing rows.
    - Decoding is done one ID at a time to keep memory low (handles
      N=10+ submissions without blowing up).
    - No weights, no rank-normalisation, no class map. Just average.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

# q8rle strings can exceed csv's default 128 KB field limit.
csv.field_size_limit(sys.maxsize)


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
    """Decode q8rle string to (H, W) float32 array in [0, 1]."""
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
# Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_submission(path: Path) -> dict[str, str]:
    """Returns {ID: q8rle_string}."""
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["ID", "Label"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for row in reader:
            if len(row) < 2:
                continue
            out[row[0]] = row[1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path,
                    help="2+ submission.csv paths to average")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output submission.csv path")
    ap.add_argument("--no-zip", action="store_true",
                    help="Skip producing the .zip alongside the .csv")
    args = ap.parse_args()

    if len(args.runs) < 2:
        raise SystemExit("need >= 2 submissions to average")

    # ── Load all submissions (raw q8rle strings only, no decoding yet)
    print(f"Loading {len(args.runs)} submission CSVs...")
    subs: list[dict[str, str]] = []
    for p in args.runs:
        s = load_submission(p)
        print(f"  {p.parent.name}/{p.name}: {len(s)} rows")
        subs.append(s)

    # ── Use intersection of IDs (in case some submissions are missing rows)
    common = set.intersection(*[set(s.keys()) for s in subs])
    if not common:
        raise SystemExit("no IDs in common across submissions")
    n_missing = sum(len(s) - len(common) for s in subs)
    if n_missing > 0:
        print(f"  [warn] {n_missing} IDs were not present in all "
              f"submissions — using intersection ({len(common)} IDs).")
    all_ids = sorted(common)

    # ── Fuse one ID at a time (low memory)
    M = len(subs)
    print(f"\nAveraging {len(all_ids)} images across {M} submissions "
          f"(uniform weights = 1/{M})...")
    fused: dict[str, str] = {}
    t0 = time.time()
    for i, sid in enumerate(all_ids):
        # Decode each method's map for this ID, then average.
        avg: np.ndarray | None = None
        for s in subs:
            m = q8rle_to_float_matrix(s[sid])
            if avg is None:
                avg = m.astype(np.float32, copy=True)
            else:
                if m.shape != avg.shape:
                    raise ValueError(
                        f"shape mismatch on ID {sid}: {m.shape} vs {avg.shape}")
                avg += m
        assert avg is not None
        avg /= M
        avg = np.clip(avg, 0.0, 1.0).astype(np.float32)
        fused[sid] = float_matrix_to_q8rle(avg)
        if (i + 1) % 1000 == 0:
            print(f"  fused {i + 1}/{len(all_ids)}  "
                  f"({time.time() - t0:.1f}s elapsed)", flush=True)
    print(f"  fused all {len(all_ids)} in {time.time() - t0:.1f}s")

    # ── Write
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

    print("\nDone.")


if __name__ == "__main__":
    main()