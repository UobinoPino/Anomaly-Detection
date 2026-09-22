"""Score-map post-processing primitives: smoothing, resizing, calibration.

Pure NumPy and SciPy — deliberately no torch. These run on the CPU over small
``(H, W)`` maps at the end of a detector's pipeline, and keeping them
torch-free means the codec, metric and submission tests all run without a
900 MB dependency.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

__all__ = [
    "SUBMISSION_SHAPE",
    "calibrate_to_unit",
    "gaussian_smooth",
    "normalise_to_unit",
    "resize_bilinear",
    "resize_nearest",
    "resize_to_submission",
]

#: Every submission row is a 224x224 map, whatever resolution the detector
#: worked at internally.
SUBMISSION_SHAPE: tuple[int, int] = (224, 224)


def _gaussian_kernel_1d(sigma: float, radius: int) -> npt.NDArray[np.float32]:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def gaussian_smooth(
    score: npt.NDArray[np.floating], sigma: float = 1.5
) -> npt.NDArray[np.float32]:
    """Separable Gaussian blur with reflect padding.

    ``sigma <= 0`` is a no-op. Prefers ``scipy.ndimage`` when available (same
    result, roughly an order of magnitude faster than the previous
    ``np.apply_along_axis`` implementation, which built one Python-level
    closure call per row and per column).
    """
    arr = np.asarray(score, dtype=np.float32)
    if sigma <= 0:
        return arr

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        pass
    else:
        return gaussian_filter(arr, sigma=sigma, mode="reflect").astype(np.float32)

    radius = max(1, int(round(3 * sigma)))
    kernel = _gaussian_kernel_1d(sigma, radius)
    padded = np.pad(arr, ((radius, radius), (0, 0)), mode="reflect")
    out = np.empty_like(arr)
    for col in range(arr.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    padded = np.pad(out, ((0, 0), (radius, radius)), mode="reflect")
    for row in range(arr.shape[0]):
        out[row, :] = np.convolve(padded[row, :], kernel, mode="valid")
    return out


def resize_nearest(
    arr: npt.NDArray, shape: tuple[int, int]
) -> npt.NDArray:
    """Nearest-neighbour resize, dtype-preserving. Correct for masks."""
    src = np.asarray(arr)
    if src.shape == shape:
        return src
    h, w = shape
    rows = (np.arange(h) * src.shape[0] // h).clip(0, src.shape[0] - 1)
    cols = (np.arange(w) * src.shape[1] // w).clip(0, src.shape[1] - 1)
    return src[np.ix_(rows, cols)]


def resize_bilinear(
    arr: npt.NDArray[np.floating], shape: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Bilinear resize of a 2-D float map, matching torch's ``align_corners=False``."""
    src = np.asarray(arr, dtype=np.float32)
    if src.shape == shape:
        return src

    out_h, out_w = shape
    in_h, in_w = src.shape
    # align_corners=False sampling grid.
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * in_h / out_h - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * in_w / out_w - 0.5
    ys = ys.clip(0, in_h - 1)
    xs = xs.clip(0, in_w - 1)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]

    top = src[np.ix_(y0, x0)] * (1 - wx) + src[np.ix_(y0, x1)] * wx
    bottom = src[np.ix_(y1, x0)] * (1 - wx) + src[np.ix_(y1, x1)] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def resize_to_submission(
    score: npt.NDArray[np.floating],
) -> npt.NDArray[np.float32]:
    """Resize a score map to the 224x224 submission resolution."""
    return resize_bilinear(score, SUBMISSION_SHAPE)


def calibrate_to_unit(
    scores: Iterable[npt.NDArray[np.floating]],
    lo_pct: float = 1.0,
    hi_pct: float = 99.5,
) -> tuple[float, float]:
    """Find the global ``(lo, hi)`` percentile pair used to map scores to [0, 1].

    Computed once over *all* test score maps so the mapping is global — a
    per-image mapping would destroy the cross-image ordering the pooled
    leaderboard metric measures.
    """
    flat = np.concatenate([np.asarray(s, np.float32).ravel() for s in scores])
    if flat.size == 0:
        raise ValueError("calibrate_to_unit() received no scores")
    lo = float(np.percentile(flat, lo_pct))
    hi = float(np.percentile(flat, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def normalise_to_unit(
    score: npt.NDArray[np.floating], lo: float, hi: float
) -> npt.NDArray[np.float32]:
    """Apply a calibration pair, clipped to [0, 1]."""
    return np.clip((np.asarray(score, np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(
        np.float32
    )
