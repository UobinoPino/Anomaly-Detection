"""Helper for dumping per-pixel local-validation predictions to disk.

Background
----------
The PatchCore / CutPaste / RD / FastFlow / UniAD / EfficientAD baselines
all compute pixel-AP on a local validation set. To train a stacker
(logistic regression, XGBoost, etc.) we need the raw per-pixel score
maps plus their ground-truth masks.

This module provides a tiny saver utility that any baseline can call
during its local-val loop with three extra lines of code. The output is
one `.npz` file per run:

    <run_dir>/local_predictions.npz
        ids      : (N,) string  — "class_01/anomaly_01/view_01" etc.
        classes  : (N,) string  — "class_01", ...
        anomaly_types : (N,) string — "anomaly_01", ...
        scores   : (N, H, W) float32  — per-pixel anomaly score, ARBITRARY range
        masks    : (N, H, W) uint8    — binary ground truth
        score_range : (2,) float32    — (global_min, global_max) for quick check

Range policy (v2 — IMPORTANT CHANGE FROM v1)
--------------------------------------------
Score arrays are stored AS-IS — the previous version clipped to [0, 1]
"defensively", which silently destroyed any score map whose distribution
extended outside the unit interval. This bit FastFlow, EfficientAD, and
UniAD hard (NLL / MSE scores are unbounded and can be negative). The
training-time AP printed in the log was computed BEFORE the saver clip,
so the log looked fine while the npz on disk was saturated — making
post-hoc analysis and stacking impossible.

Now we:
  - Accept arbitrary float ranges (negative or > 1 both fine).
  - Store float32 unchanged.
  - Warn ONCE per run if scores look pathological (NaN / inf / all-zero).

Downstream consumers (xgboost_stacker.py and analyze_predictions.py)
already apply rank-normalisation or compute distribution-agnostic AP,
so neither cares about absolute scale. The submission-write path does
its own global percentile calibration (`calibrate_to_unit`) so that's
unaffected.

All score maps in one run MUST share the same (H, W). The baselines'
local-val pipelines already resize everything to a fixed evaluation
resolution, so this is satisfied in practice.

Integration into each baseline (one-time edit)
----------------------------------------------
At the top of the baseline (after imports):

    from local_preds_saver import LocalPredSaver
    _local_saver = LocalPredSaver()

Inside the per-(class, anomaly_type, view) local-val loop, RIGHT NEXT TO
the existing `pixel_average_precision(...)` call:

    _local_saver.add(cls, anomaly_type, view_idx, score_map, gt_mask)

After the local-val loop finishes (just before writing `local_eval.csv`):

    _local_saver.save(run_dir / "local_predictions.npz")

Pass the SAME `score_map` you pass to `pixel_average_precision(...)`
(typically the smoothed, but otherwise raw, output of the model). DO NOT
pre-calibrate or clip before handing it to the saver.

CLI sanity check
----------------
    python local_preds_saver.py inspect <path/to/local_predictions.npz>

Prints shape, dtype, per-class counts, mean positive-pixel fraction,
score range and saturation diagnostics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


class LocalPredSaver:
    """Accumulates per-image local-val score maps + GT masks, then
    writes a single npz file. Thread-unsafe; intended for the
    sequential local-val loop inside each baseline.

    Score values are stored AS-IS (float32) — no clipping, no rescaling.
    Anomaly scores from density / reconstruction methods (FastFlow,
    UniAD, EfficientAD) are unbounded by construction, and clipping
    them destroys the very tail that contains the anomaly signal.
    """

    def __init__(self):
        self.ids: list[str] = []
        self.classes: list[str] = []
        self.anomaly_types: list[str] = []
        self.image_paths: list[str] = []
        self.scores: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        # Diagnostics — printed once at save() time.
        self._n_nonfinite_imgs = 0
        self._n_constant_imgs = 0
        self._global_min = float("inf")
        self._global_max = float("-inf")

    def add(self, cls: str, anomaly_type: str, view_idx: int,
            score_map, gt_mask, image_path: str | Path | None = None) -> None:
        """Append one (image, mask) pair.

        Args:
            cls, anomaly_type, view_idx : labels for indexing later.
            score_map : 2-D torch.Tensor or np.ndarray. ARBITRARY real-
                valued range — do NOT clip or rescale before calling.
                Higher = more anomalous.
            gt_mask   : 2-D binary (0/1 or bool).
            image_path : original source image path. Stored as a string
                so the stacker can load the raw RGB for pixel-intensity
                features without having to reconstruct file paths from
                the labels. Optional; pass "" if not available.
        """
        s = _to_numpy_2d(score_map, dtype=np.float32)
        m = _to_numpy_2d(gt_mask, dtype=np.uint8)
        if s.shape != m.shape:
            raise ValueError(
                f"score_map shape {s.shape} != mask shape {m.shape} "
                f"for {cls}/{anomaly_type}/view_{view_idx:02d}")

        # Sanitise non-finite values without changing the rest of the
        # distribution: replace NaN/inf with the per-image finite min/max.
        # This preserves the AP for the affected image (since rank-AP only
        # cares about ordering of finite values) and keeps the array
        # stackable downstream.
        finite_mask = np.isfinite(s)
        if not finite_mask.all():
            self._n_nonfinite_imgs += 1
            if finite_mask.any():
                f_min = float(s[finite_mask].min())
                f_max = float(s[finite_mask].max())
                s = np.where(np.isnan(s), f_min, s)
                s = np.where(np.isposinf(s), f_max, s)
                s = np.where(np.isneginf(s), f_min, s)
            else:
                # Pathological — everything is NaN/inf. Zero it out and
                # let the constant-image check below flag it.
                s = np.zeros_like(s)

        if float(s.max()) == float(s.min()):
            self._n_constant_imgs += 1

        self._global_min = min(self._global_min, float(s.min()))
        self._global_max = max(self._global_max, float(s.max()))

        # Defensive: masks binarised to 0/1.
        if m.max() > 1:
            m = (m > 0).astype(np.uint8)

        self.ids.append(f"{cls}/{anomaly_type}/view_{view_idx:02d}")
        self.classes.append(cls)
        self.anomaly_types.append(anomaly_type)
        self.image_paths.append(str(image_path) if image_path is not None else "")
        self.scores.append(s)
        self.masks.append(m)

    def save(self, path: Path | str) -> Path:
        """Write the accumulator to a compressed .npz file. All score
        maps and masks must share the same (H, W). If any differ, we
        resize-with-numpy via nearest-neighbour to the most common shape
        (and print a warning), because heterogeneous shapes can't be
        stacked into a single ndarray.
        """
        if not self.scores:
            raise RuntimeError("LocalPredSaver.save(): nothing to save")
        path = Path(path)

        # Enforce uniform shape (resize others to the modal shape if
        # there's a mismatch; warn the user since that suggests a bug).
        shapes = [s.shape for s in self.scores]
        if len(set(shapes)) > 1:
            from collections import Counter
            modal_shape = Counter(shapes).most_common(1)[0][0]
            print(f"[local_preds_saver] WARN: heterogeneous shapes "
                  f"{Counter(shapes)}; resizing all to {modal_shape}",
                  file=sys.stderr)
            self.scores = [
                _nn_resize(s, modal_shape, dtype=np.float32)
                for s in self.scores]
            self.masks = [
                _nn_resize(m, modal_shape, dtype=np.uint8)
                for m in self.masks]

        # Stack into single arrays.
        scores = np.stack(self.scores).astype(np.float32)        # (N,H,W)
        masks  = np.stack(self.masks ).astype(np.uint8)          # (N,H,W)
        ids    = np.asarray(self.ids, dtype=object)
        classes = np.asarray(self.classes, dtype=object)
        anomaly_types = np.asarray(self.anomaly_types, dtype=object)
        image_paths = np.asarray(self.image_paths, dtype=object)
        score_range = np.asarray([self._global_min, self._global_max],
                                  dtype=np.float32)

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ids=ids,
            classes=classes,
            anomaly_types=anomaly_types,
            image_paths=image_paths,
            scores=scores,
            masks=masks,
            score_range=score_range,
        )

        # Diagnostics — make it loud if something looked wrong, but
        # don't refuse to save (the user may want to inspect anyway).
        n = len(self.scores)
        n_pos_frac = float(masks.mean())
        print(f"[local_preds_saver] saved {n} predictions "
              f"({scores.shape[1]}x{scores.shape[2]}, "
              f"~{n_pos_frac * 100:.2f}% positive pixels)")
        print(f"[local_preds_saver]   score range: "
              f"[{self._global_min:.4g}, {self._global_max:.4g}]  "
              f"(stored as-is, no clipping)")
        if self._n_nonfinite_imgs:
            print(f"[local_preds_saver]   WARN: {self._n_nonfinite_imgs}/{n} "
                  f"images contained NaN/inf — replaced with finite "
                  f"min/max per image. Investigate the model.",
                  file=sys.stderr)
        if self._n_constant_imgs:
            print(f"[local_preds_saver]   WARN: {self._n_constant_imgs}/{n} "
                  f"images have a constant score map (max == min). "
                  f"Their AP is ill-defined; the model is degenerate "
                  f"on those samples.", file=sys.stderr)
        # Tail-saturation heuristic: if the top 0.1% of pixels are all
        # equal, the score was almost certainly clipped upstream.
        try:
            top = np.percentile(scores.reshape(-1), 99.9)
            top_max = float(scores.max())
            if top == top_max and top_max > 0:
                frac_at_max = float((scores >= top_max).mean())
                if frac_at_max > 0.001:
                    print(f"[local_preds_saver]   WARN: {frac_at_max*100:.2f}% "
                          f"of pixels are tied at the global max "
                          f"({top_max:.4g}). Looks like the score was "
                          f"clipped before being handed to the saver — "
                          f"check that you're passing the RAW model "
                          f"score, not a normalised/calibrated one.",
                          file=sys.stderr)
        except Exception:
            pass
        print(f"[local_preds_saver]   wrote -> {path}")
        return path

    def __len__(self) -> int:
        return len(self.scores)


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────
def _to_numpy_2d(x, dtype) -> np.ndarray:
    if hasattr(x, "detach"):              # torch.Tensor
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:   # (1, H, W)
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {x.shape}")
    return x.astype(dtype, copy=False)


def _nn_resize(arr: np.ndarray, target_shape, dtype) -> np.ndarray:
    """Nearest-neighbour resize without bringing in PIL/scipy. Used only
    in the rare shape-mismatch fallback."""
    th, tw = target_shape
    h, w = arr.shape
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]].astype(dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI: quick inspector
# ─────────────────────────────────────────────────────────────────────────────
def _inspect(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    ids = data["ids"]; classes = data["classes"]
    scores = data["scores"]; masks = data["masks"]
    print(f"file        : {path}")
    print(f"n_images    : {len(ids)}")
    print(f"score shape : {scores.shape}  dtype={scores.dtype}")
    print(f"mask shape  : {masks.shape}  dtype={masks.dtype}")
    print(f"% positive  : {float(masks.mean()) * 100:.3f}")
    s_min = float(scores.min()); s_max = float(scores.max())
    print(f"score range : [{s_min:.4g}, {s_max:.4g}]")
    # Saturation diagnostic
    if s_max > s_min:
        frac_at_max = float((scores >= s_max - 1e-9).mean())
        frac_at_min = float((scores <= s_min + 1e-9).mean())
        print(f"  pixels tied at max: {frac_at_max*100:.3f}%")
        print(f"  pixels tied at min: {frac_at_min*100:.3f}%")
        if frac_at_max > 0.01:
            print(f"  WARN: heavy upper-tail saturation — score map may "
                  f"have been clipped before saving.")
    n_nonfinite = int((~np.isfinite(scores)).sum())
    if n_nonfinite:
        print(f"  WARN: {n_nonfinite} non-finite values present.")
    if "score_range" in data.files:
        sr = data["score_range"]
        print(f"stored score_range: [{float(sr[0]):.4g}, {float(sr[1]):.4g}]")
    from collections import Counter
    cls_counts = Counter(classes.tolist())
    print(f"per-class counts:")
    for cls in sorted(cls_counts):
        print(f"  {cls:<12} {cls_counts[cls]}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        _inspect(Path(sys.argv[2]))
    else:
        print(__doc__)
        sys.exit(1)