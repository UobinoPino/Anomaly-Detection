"""Save raw test-set anomaly scores per-method for cross-model ensembling.

Background
----------
Each baseline currently writes a q8-quantized RLE submission.csv (256
levels). That's lossy for ensembling — you can't recover a clean rank
ordering after passing through an 8-bit bottleneck. This saver writes
the raw float32 scores alongside the submission so submit_ensemble.py
can apply ECDF calibration and weighted geometric mean across methods
at full precision.

The output file format mirrors local_predictions.npz so the loading
code in submit_ensemble.py is uniform across both train_anomaly (with
masks) and test (without).

Integration (two lines per baseline, never re-run training)
-----------------------------------------------------------
At the top of the baseline, with the other imports:

    from test_preds_saver import save_test_predictions

In main(), right BEFORE the existing `write_submission(...)` call:

    save_test_predictions(all_test_results, run_dir)

`all_test_results` is the same list of (record, score_map) tuples that
write_submission() already consumes. NO other code changes.

After patching: re-run each baseline with `--skip-eval` and the memory
banks already on disk will be reused (no re-training). The inference
sweep takes minutes per model, not hours.

Output: <run_dir>/test_predictions.npz with keys
    ids         : (N,) object  — image stem ("sample_view03"), matches
                                 submission.csv ID column
    classes     : (N,) object  — "class_01", ...
    views       : (N,) int32   — view index (-1 if unknown)
    image_paths : (N,) object  — full path to source image
    scores      : (N, H, W) float32 — raw smoothed score at submission
                                       resolution. ARBITRARY range — do
                                       NOT clip or rescale before saving.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np


def save_test_predictions(test_results, run_dir: Path) -> Path:
    """Persist raw test scores from a baseline's inference loop.

    Args:
        test_results : list of (record, score_map) tuples.
            record must have:
              .path (Path)         — original image path
              .cls  (str)          — class name (e.g. "class_01")
              .view (int|None)     — view index for multi-view samples
            score_map : 2-D np.ndarray or torch tensor at the SAME
                resolution write_submission writes (typically 224x224),
                already smoothed but NOT yet calibrated/clipped.
        run_dir : the run directory (will write inside this folder).

    Returns:
        Path to the written npz file.
    """
    if not test_results:
        raise RuntimeError("save_test_predictions: empty test_results")

    ids, classes, views, image_paths, scores = [], [], [], [], []
    for r, sm in test_results:
        # tensor → numpy
        if hasattr(sm, "detach"):
            sm = sm.detach().cpu().numpy()
        sm = np.asarray(sm, dtype=np.float32)
        if sm.ndim == 3 and sm.shape[0] == 1:
            sm = sm[0]
        if sm.ndim != 2:
            raise ValueError(
                f"score_map must be 2-D, got shape {sm.shape} for "
                f"{getattr(r, 'path', '?')}")

        ids.append(r.path.stem)
        classes.append(getattr(r, "cls", "?"))
        v = getattr(r, "view", None)
        views.append(int(v) if v is not None else -1)
        image_paths.append(str(r.path))
        scores.append(sm)

    # Heterogeneous shapes should not happen, but be safe.
    shapes = [s.shape for s in scores]
    if len(set(shapes)) > 1:
        modal_shape = Counter(shapes).most_common(1)[0][0]
        print(f"[test_preds_saver] WARN: heterogeneous shapes "
              f"{Counter(shapes)}; nearest-neighbour resizing all to "
              f"{modal_shape}", file=sys.stderr)
        scores = [_nn_resize(s, modal_shape) for s in scores]

    scores_arr = np.stack(scores).astype(np.float32)
    path = Path(run_dir) / "test_predictions.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=object),
        classes=np.asarray(classes, dtype=object),
        views=np.asarray(views, dtype=np.int32),
        image_paths=np.asarray(image_paths, dtype=object),
        scores=scores_arr,
    )

    s_min = float(scores_arr.min())
    s_max = float(scores_arr.max())
    print(f"    saved {len(test_results)} test predictions "
          f"({scores_arr.shape[1]}x{scores_arr.shape[2]}, "
          f"range [{s_min:.4g}, {s_max:.4g}]) -> {path}")
    return path


def _nn_resize(arr: np.ndarray, target_shape) -> np.ndarray:
    th, tw = target_shape
    h, w = arr.shape
    ys = np.linspace(0, h - 1, th).round().astype(np.int64)
    xs = np.linspace(0, w - 1, tw).round().astype(np.int64)
    return arr[ys[:, None], xs[None, :]]