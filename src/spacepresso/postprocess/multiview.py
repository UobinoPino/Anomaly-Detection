"""Per-view score calibration for multi-view samples.

The Spacepresso dataset photographs each object from several fixed
viewpoints. A detector scores each view independently, so view 3 (say, a
close-up) can sit on a systematically different scale from view 0. Per-image
AP never notices — it is rank-based within one image. The leaderboard metric
pools every pixel of every image into one ranking, so it notices a great deal.

This module rescales each view onto a common range using statistics gathered
from the *normal* training images of that view.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import torch

from spacepresso.core.logging import get_logger
from spacepresso.core.records import ImageRecord

__all__ = [
    "ViewStats",
    "apply_view_norm",
    "fit_view_norm",
    "load_view_stats",
    "save_view_stats",
]

logger = get_logger(__name__)

#: ``{view_index: (lo, hi)}`` percentile pair per viewpoint.
ViewStats = dict[int, tuple[float, float]]

#: Scores one ``(B, C, H, W)`` batch to ``(B, H, W)``. The only thing this
#: module needs from a detector.
BatchScorer = Callable[[torch.Tensor], torch.Tensor]

#: Builds a dataloader over records. Injected so this module does not need to
#: know the batch size or worker policy of whoever is calling it.
LoaderFactory = Callable[[Sequence[ImageRecord]], object]

_DEFAULT_VIEW = 0


def fit_view_norm(
    score_batch: BatchScorer,
    make_loader: LoaderFactory,
    train_good: Sequence[ImageRecord],
    *,
    max_images_per_view: int = 200,
    max_pixels_per_view: int = 1_500_000,
    lo_pct: float = 1.0,
    hi_pct: float = 99.5,
    seed: int = 0,
) -> ViewStats:
    """Collect ``(p_lo, p_hi)`` score percentiles per view over normal images.

    Args:
        score_batch: scores one batch of images, un-augmented.
        make_loader: turns a record list into an iterable of
            ``(images, masks, indices)`` batches.
        train_good: the normal training images to calibrate against.

    Subsampling is bounded twice — by image count and by pixel count — because
    a single view of a large class can otherwise contribute tens of millions
    of pixels to one percentile computation.
    """
    by_view: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in train_good:
        by_view[record.view if record.view is not None else _DEFAULT_VIEW].append(
            record
        )

    rng = np.random.default_rng(seed)
    stats: ViewStats = {}

    for view in sorted(by_view):
        records = by_view[view]
        if not records:
            continue
        if len(records) > max_images_per_view:
            chosen = rng.choice(len(records), size=max_images_per_view, replace=False)
            records = [records[i] for i in chosen]

        chunks: list[npt.NDArray[np.float32]] = []
        for images, _masks, _indices in make_loader(records):  # type: ignore[misc]
            scored = score_batch(images)
            chunks.append(np.asarray(scored.cpu(), dtype=np.float32).reshape(-1))

        flat = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        if flat.size > max_pixels_per_view:
            keep = rng.choice(flat.size, size=max_pixels_per_view, replace=False)
            flat = flat[keep]

        lo = float(np.percentile(flat, lo_pct))
        hi = float(np.percentile(flat, hi_pct))
        if hi <= lo:
            hi = lo + 1e-6
        stats[view] = (lo, hi)
        logger.info(
            "      [view-norm] view=%d  images=%d  pixels=%s  p%g=%.5f  p%g=%.5f",
            view,
            len(records),
            f"{flat.size:,}",
            lo_pct,
            lo,
            hi_pct,
            hi,
        )

    return stats


def apply_view_norm(
    score_map: npt.NDArray[np.floating],
    view: int | None,
    stats: ViewStats,
) -> npt.NDArray[np.float32]:
    """Affine-rescale one score map onto its view's calibrated range.

    **No clipping.** The output is a strictly increasing function of the
    input, so per-image AP is exactly unchanged; the gain is purely in
    cross-image comparability. Values may fall below 0 or above 1, and the
    global calibration in :mod:`spacepresso.core.submission` does the final
    clip-and-quantise.

    An unknown view falls back to per-image min-max, which is still monotonic
    but recovers no cross-view comparability — the caller gets a warning-free
    but ineffective normalisation, which is the honest outcome when there is
    no statistic to apply.
    """
    scores = np.asarray(score_map, dtype=np.float32)
    if view is None or view not in stats:
        lo, hi = float(scores.min()), float(scores.max())
    else:
        lo, hi = stats[view]
    return ((scores - lo) / max(hi - lo, 1e-9)).astype(np.float32)


def save_view_stats(stats: ViewStats, path) -> None:
    """Persist view statistics next to the run's other artifacts."""
    views = np.asarray(sorted(stats), dtype=np.int32)
    bounds = np.asarray([stats[int(v)] for v in views], dtype=np.float32)
    np.savez(path, views=views, bounds=bounds)


def load_view_stats(path) -> ViewStats:
    with np.load(path) as handle:
        return {
            int(view): (float(lo), float(hi))
            for view, (lo, hi) in zip(handle["views"], handle["bounds"], strict=True)
        }
