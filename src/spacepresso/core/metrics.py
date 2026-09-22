"""Evaluation metrics.

The leaderboard metric is **globally pooled pixel-level Average Precision**:
every pixel of every image is thrown into one ranking. That is deliberately
different from the per-image AP averaged over images, which is what local
validation reports — a detector can win the per-image average and still lose
the leaderboard if its scores are not comparable *across* images. Both are
provided here, and named so the difference is visible at the call site.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "average_precision",
    "per_image_ap",
    "pooled_pixel_ap",
]


def _average_precision_numpy(
    scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int8]
) -> float:
    """Dependency-free AP, used when scikit-learn is unavailable."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(int(y.sum()), 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def average_precision(
    scores: npt.NDArray[np.floating], labels: npt.NDArray[np.integer]
) -> float:
    """Average Precision over flat score/label arrays.

    Returns 0.0 when ``labels`` contains no positives — AP is undefined there,
    and 0.0 keeps class means finite.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = (np.asarray(labels).ravel() > 0).astype(np.int8)
    if s.shape != y.shape:
        raise ValueError(f"score/label shape mismatch: {s.shape} vs {y.shape}")
    if y.sum() == 0:
        return 0.0
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return _average_precision_numpy(s, y)
    return float(average_precision_score(y, s))


def per_image_ap(score: npt.NDArray[np.floating], gt: npt.NDArray[np.integer]) -> float:
    """Pixel-AP for a single ``(H, W)`` score map against its mask."""
    return average_precision(score, gt)


def pooled_pixel_ap(
    scores: Sequence[npt.NDArray[np.floating]] | Iterable[npt.NDArray[np.floating]],
    masks: Sequence[npt.NDArray[np.integer]] | Iterable[npt.NDArray[np.integer]],
) -> float:
    """The leaderboard metric: AP over every pixel of every image, pooled.

    Score maps are *not* renormalised — pooling is exactly what exposes
    cross-image scale mismatch, so hiding it here would defeat the purpose.
    """
    flat_scores = np.concatenate([np.asarray(s, np.float64).ravel() for s in scores])
    flat_masks = np.concatenate([np.asarray(m).ravel() for m in masks])
    return average_precision(flat_scores, flat_masks)
