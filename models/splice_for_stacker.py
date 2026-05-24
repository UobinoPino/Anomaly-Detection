"""Splice TransFusion (covers only some classes) into a base method run
(covers all 8 classes), producing a new "method" that has full coverage but
uses TransFusion's scores for the target classes.

This is the Option-1 splicing from the earlier discussion: for non-target
classes, copy the base method's scores so alignment doesn't break inside the
stacker; for target classes, substitute TransFusion's. The stacker's per-class
`top_methods_per_class` selection then routes correctly.

Inputs:
  --base-run        path to an existing run dir, e.g.
                    .../runs/20260515-100850_wrn50_..._exp5-input384-mb128_1fe0a2
                    (must contain submission.csv AND local_predictions.npz)
  --tf-run          path to the TransFusion run dir produced by transfusion_baseline.py
  --target-classes  classes to substitute (e.g. class_03 class_07)
  --data-root       used to look up id → class for test submission
  --out-dir         destination for the spliced run

Outputs (under --out-dir):
  - submission.csv               (target IDs from tf, others from base)
  - local_predictions.npz        (target val rows from tf, others from base)
  - splice_manifest.json
  - run_log.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


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


def hr(t, c="="): print(f"\n{c * 78}\n  {t}\n{c * 78}")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders matching the stacker's conventions
# ─────────────────────────────────────────────────────────────────────────────
def load_submission(path: Path) -> tuple[list[str], dict[str, str]]:
    """Returns (ordered_ids, dict_id_to_label) preserving original row order
    so the spliced output looks identical to the base on un-changed rows."""
    csv.field_size_limit(sys.maxsize)
    ids: list[str] = []
    d: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["ID", "Label"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for row in reader:
            if len(row) >= 2:
                ids.append(row[0])
                d[row[0]] = row[1]
    return ids, d


def load_local_preds(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
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
    """Map every test image stem → class folder name."""
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


def _resize_nn(arr: np.ndarray, h_target: int, w_target: int,
                  dtype=np.float32) -> np.ndarray:
    h, w = arr.shape
    if (h, w) == (h_target, w_target):
        return arr.astype(dtype, copy=False)
    ys = np.linspace(0, h - 1, h_target).round().astype(np.int64)
    xs = np.linspace(0, w - 1, w_target).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Splicing
# ─────────────────────────────────────────────────────────────────────────────
def splice_submission(base_ids: list[str], base_sub: dict[str, str],
                         tf_sub: dict[str, str], class_map: dict[str, str],
                         target_classes: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    n_tf_used = 0
    n_base_used = 0
    n_tf_missing = 0
    for sid in base_ids:
        cls = class_map.get(sid)
        if cls in target_classes and sid in tf_sub:
            out[sid] = tf_sub[sid]; n_tf_used += 1
        elif cls in target_classes and sid not in tf_sub:
            # TransFusion didn't run for this id (shouldn't happen if TF was
            # invoked on all of the target classes). Fall back to base.
            out[sid] = base_sub[sid]; n_tf_missing += 1
        else:
            out[sid] = base_sub[sid]; n_base_used += 1
    print(f"  submission splice: base={n_base_used}, tf={n_tf_used}, "
          f"tf_missing_fallback_to_base={n_tf_missing}")
    return out


def splice_local_preds(base: dict, tf: dict,
                          target_classes: set[str]) -> dict:
    """Substitute val rows for target classes. Resize TF's scores+masks to
    match base's spatial shape (the stacker's align_local_preds does NN
    resize too, but doing it here keeps things explicit and small.)"""
    base_ids = base["ids"]
    base_classes = base["classes"]
    base_scores = base["scores"]
    base_masks = base["masks"]
    base_atypes = base["anomaly_types"]
    base_paths = base.get("image_paths")
    N = base_ids.shape[0]
    H_b, W_b = base_scores.shape[1], base_scores.shape[2]

    # Index TF by id
    tf_by_id: dict[str, int] = {sid: i for i, sid in enumerate(tf["ids"].tolist())}
    tf_scores = tf["scores"]
    tf_masks = tf["masks"]

    # Output arrays: clone base, overwrite target-class rows
    out_scores = base_scores.copy()
    out_masks = base_masks.copy()
    out_atypes = base_atypes.copy()

    n_substituted = 0
    n_target_missing_in_tf = 0
    for i in range(N):
        cls = str(base_classes[i])
        if cls not in target_classes: continue
        sid = str(base_ids[i])
        if sid not in tf_by_id:
            n_target_missing_in_tf += 1
            # leave base values in place (graceful degrade)
            continue
        j = tf_by_id[sid]
        s_tf = tf_scores[j]
        m_tf = tf_masks[j]
        out_scores[i] = _resize_nn(s_tf, H_b, W_b, np.float32)
        # NN-resize mask too, to be coherent; cast to uint8.
        out_masks[i] = _resize_nn(m_tf.astype(np.float32),
                                      H_b, W_b, np.float32).astype(np.uint8)
        out_atypes[i] = str(tf["anomaly_types"][j])
        n_substituted += 1
    print(f"  local_preds splice: substituted {n_substituted} val rows "
          f"({n_target_missing_in_tf} target rows missing from tf, "
          f"left as base)")
    return {
        "ids": base_ids,
        "classes": base_classes,
        "anomaly_types": out_atypes,
        "scores": out_scores,
        "masks": out_masks,
        "image_paths": base_paths,
    }


def save_local_preds(path: Path, d: dict) -> None:
    kwargs = dict(
        ids=np.asarray(d["ids"], dtype=object),
        classes=np.asarray(d["classes"], dtype=object),
        anomaly_types=np.asarray(d["anomaly_types"], dtype=object),
        scores=d["scores"].astype(np.float32),
        masks=d["masks"].astype(np.uint8),
    )
    if d.get("image_paths") is not None:
        kwargs["image_paths"] = np.asarray(d["image_paths"], dtype=object)
    np.savez_compressed(path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-run", type=Path, required=True,
                    help="existing 14-method run dir to fill the non-target "
                          "classes with (e.g. exp5-input384-mb128)")
    ap.add_argument("--tf-run", type=Path, required=True,
                    help="TransFusion run dir produced by transfusion_baseline.py")
    ap.add_argument("--target-classes", nargs="+", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tee_to(args.out_dir / "run_log.txt"):
        hr(f"SPLICE — TransFusion → base method", "=")
        print(f"  base_run        : {args.base_run}")
        print(f"  tf_run          : {args.tf_run}")
        print(f"  target_classes  : {args.target_classes}")
        print(f"  data_root       : {args.data_root}")
        print(f"  out_dir         : {args.out_dir}")

        # ── Load everything ─────────────────────────────────────────────────
        print(f"\nLoading base submission...")
        base_ids, base_sub = load_submission(args.base_run / "submission.csv")
        print(f"  {len(base_ids)} test rows")

        print(f"\nLoading TransFusion submission...")
        tf_ids, tf_sub = load_submission(args.tf_run / "submission.csv")
        print(f"  {len(tf_ids)} test rows")

        print(f"\nLoading base local_predictions.npz...")
        base_local = load_local_preds(args.base_run / "local_predictions.npz")
        print(f"  {len(base_local['ids'])} val rows; "
              f"scores shape {base_local['scores'].shape}")

        print(f"\nLoading TransFusion local_predictions.npz...")
        tf_local = load_local_preds(args.tf_run / "local_predictions.npz")
        print(f"  {len(tf_local['ids'])} val rows; "
              f"scores shape {tf_local['scores'].shape}")

        print(f"\nBuilding class map from {args.data_root}...")
        class_map = build_class_map_from_data(args.data_root)
        print(f"  {len(class_map)} test ids mapped to classes")

        target_set = set(args.target_classes)

        # ── Splice ──────────────────────────────────────────────────────────
        print(f"\nSplicing submissions...")
        out_sub = splice_submission(base_ids, base_sub, tf_sub,
                                       class_map, target_set)

        print(f"\nSplicing local_predictions...")
        out_local = splice_local_preds(base_local, tf_local, target_set)

        # ── Write ───────────────────────────────────────────────────────────
        sub_path = args.out_dir / "submission.csv"
        with open(sub_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ID", "Label"])
            for sid in base_ids:
                w.writerow([sid, out_sub[sid]])
        print(f"\nWrote spliced submission -> {sub_path}")

        local_path = args.out_dir / "local_predictions.npz"
        save_local_preds(local_path, out_local)
        print(f"Wrote spliced local preds -> {local_path}")

        manifest = {
            "method_name": args.out_dir.name,
            "model_family": "transfusion_spliced",
            "base_run": str(args.base_run),
            "tf_run": str(args.tf_run),
            "target_classes": list(target_set),
            "n_test_rows": len(base_ids),
            "n_val_rows": int(len(base_local["ids"])),
        }
        with open(args.out_dir / "splice_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        hr("DONE", "=")


if __name__ == "__main__":
    main()