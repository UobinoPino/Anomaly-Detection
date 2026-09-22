"""Spatial post-processing of score maps.

Three operations that ran inside the stacker forks, with two to four slightly
different copies each:

* **Small-component suppression** — real defects are contiguous; isolated hot
  pixels are usually detector noise, and a handful of them can cost a lot of
  precision at the top of a pooled ranking.
* **Spatial priors** — per-class heatmaps of where anomalies historically
  occur, produced by ``spacepresso.analysis.priors``.
* **Within-image ranking** — replace scores by their rank inside the image, to
  remove per-image scale before fusing detectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from spacepresso.core.imaging import resize_bilinear
from spacepresso.core.logging import get_logger

__all__ = [
    "DEFAULT_MIN_CC_PER_FAMILY",
    "apply_spatial_prior",
    "load_spatial_priors",
    "parse_min_cc_spec",
    "suppress_small_components",
    "within_image_rank",
]

logger = get_logger(__name__)

#: Sensible default component-size floors by detector family. Density models
#: produce smoother maps than nearest-neighbour ones, so they tolerate a
#: larger floor before real defects start being erased.
DEFAULT_MIN_CC_PER_FAMILY: dict[str, int] = {
    "patchcore": 8,
    "cutpaste": 8,
    "efficientad": 4,
    "fastflow": 4,
    "reverse_distillation": 4,
    "dino_dpmm": 12,
    "anomalydino": 12,
    "draem": 0,
    "glass": 0,
    "cfa": 4,
    "uniad": 4,
    "winclip": 16,
    "textad": 16,
}


def suppress_small_components(
    score: npt.NDArray[np.floating],
    min_size: int,
    *,
    threshold_pct: float = 99.0,
    fill: float = 0.0,
) -> npt.NDArray[np.float32]:
    """Zero out hot blobs smaller than ``min_size`` pixels.

    "Hot" means at or above the ``threshold_pct`` percentile *of this image*,
    so the operation adapts to each map's own distribution rather than needing
    an absolute threshold.

    ``min_size <= 0`` is a no-op, which is how a detector family opts out.
    """
    scores = np.asarray(score, dtype=np.float32)
    if min_size <= 0:
        return scores

    try:
        from scipy import ndimage
    except ImportError:
        logger.warning("scipy is unavailable; skipping small-component suppression")
        return scores

    threshold = float(np.percentile(scores, threshold_pct))
    hot = scores >= threshold
    if not hot.any():
        return scores

    labels, count = ndimage.label(hot)
    if count == 0:
        return scores

    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero(sizes <= min_size)
    small = small[small > 0]  # label 0 is the background
    if small.size == 0:
        return scores

    out = scores.copy()
    out[np.isin(labels, small)] = fill
    return out


def parse_min_cc_spec(spec: str | None, families: Sequence[str]) -> dict[str, int]:
    """Parse ``"patchcore=8,fastflow=4"`` or a bare ``"6"`` into a per-family map.

    An empty spec falls back to :data:`DEFAULT_MIN_CC_PER_FAMILY`, and an
    unlisted family gets 0 (no suppression) rather than a silent default.
    """
    if not spec:
        return {f: DEFAULT_MIN_CC_PER_FAMILY.get(f, 0) for f in families}

    spec = spec.strip()
    if "=" not in spec:
        try:
            uniform = int(spec)
        except ValueError:
            raise ValueError(
                f"min-cc spec {spec!r} is neither an integer nor "
                f"'family=size,family=size'"
            ) from None
        return dict.fromkeys(families, uniform)

    out = dict.fromkeys(families, 0)
    for item in spec.split(","):
        if not item.strip():
            continue
        family, _, size = item.partition("=")
        family = family.strip()
        if family not in out:
            raise ValueError(
                f"min-cc spec names unknown family {family!r}; "
                f"known: {', '.join(sorted(families))}"
            )
        out[family] = int(size)
    return out


def load_spatial_priors(
    prior_dir: Path | None,
    classes: Sequence[str],
    *,
    pattern: str = "06_heat_{cls}.npy",
) -> dict[str, npt.NDArray[np.float32]]:
    """Load per-class anomaly-location heatmaps.

    A missing directory or a missing class file yields an all-zero prior,
    which is a no-op downstream — the feature stays present so the stacker's
    feature vector keeps a stable width, and the log says which classes fell
    back.
    """
    priors: dict[str, npt.NDArray[np.float32]] = {}
    fallback = np.zeros((128, 128), dtype=np.float32)

    if prior_dir is None or not Path(prior_dir).is_dir():
        if prior_dir is not None:
            logger.warning(
                "spatial-prior directory %s does not exist; using zero priors",
                prior_dir,
            )
        return dict.fromkeys(classes, fallback)

    for cls in classes:
        path = Path(prior_dir) / pattern.format(cls=cls)
        if not path.exists():
            logger.warning("no spatial prior for %s (looked for %s)", cls, path.name)
            priors[cls] = fallback
            continue
        try:
            heat = np.clip(np.load(path).astype(np.float32), 0.0, 1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not load %s: %s; using a zero prior", path, exc)
            priors[cls] = fallback
            continue
        logger.info(
            "    loaded spatial prior for %s: shape %s  max=%.4f",
            cls,
            heat.shape,
            float(heat.max()),
        )
        priors[cls] = heat
    return priors


def apply_spatial_prior(
    score: npt.NDArray[np.floating],
    prior: npt.NDArray[np.floating],
    *,
    weight: float = 0.5,
) -> npt.NDArray[np.float32]:
    """Blend a score map with a per-class location prior.

    ``weight`` is the prior's share: 0 leaves the score untouched, 1 replaces
    it with the prior. The prior is resized to the score's resolution.
    """
    scores = np.asarray(score, dtype=np.float32)
    if weight <= 0:
        return scores
    resized = resize_bilinear(np.asarray(prior, np.float32), scores.shape)
    return ((1.0 - weight) * scores + weight * resized).astype(np.float32)


def within_image_rank(
    score: npt.NDArray[np.floating],
) -> npt.NDArray[np.float32]:
    """Replace each pixel by its rank within the image, scaled to [0, 1].

    Removes per-image scale entirely, which is what you want before averaging
    several detectors whose raw score ranges are incomparable — and is
    exactly what you do *not* want as the final submission transform, since it
    destroys the between-image ordering the pooled metric measures.
    """
    flat = np.asarray(score, dtype=np.float32).ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[order] = np.arange(flat.size, dtype=np.float32)
    return (ranks / max(flat.size - 1, 1)).reshape(score.shape)
