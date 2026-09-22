"""Non-learned fusion of several detectors.

Worth keeping alongside the learned stacker for two reasons. It needs no
validation labels, so it works when a class has none. And it is the baseline
the stacker has to beat — if a tuned per-class gradient-boosted model does not
clear a weighted geometric mean by a useful margin, the gain is coming from
the detectors, not the stacking, and the extra machinery is not paying for
itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from spacepresso.stacking.normalisation import rank_transform

__all__ = ["RULES", "FusionRule", "ecdf_transform", "fuse"]

FusionRule = Literal["mean", "geometric", "max", "median", "rank_mean"]
RULES: tuple[FusionRule, ...] = ("mean", "geometric", "max", "median", "rank_mean")


def ecdf_transform(
    values: npt.NDArray[np.floating], reference: npt.NDArray[np.floating]
) -> npt.NDArray[np.float32]:
    """Map ``values`` to their percentile under ``reference``'s distribution.

    The principled way to put detectors on a common scale when a sample of
    normal scores is available: after the transform every method's scores are
    uniform on [0, 1] over normal data, so a fixed threshold means the same
    thing for all of them.
    """
    ordered = np.sort(np.asarray(reference, dtype=np.float32).ravel())
    positions = np.searchsorted(ordered, np.asarray(values, dtype=np.float32).ravel())
    return (
        (positions / max(ordered.size, 1)).astype(np.float32).reshape(np.shape(values))
    )


def fuse(
    scores: npt.NDArray[np.floating],
    rule: FusionRule = "mean",
    *,
    weights: Sequence[float] | None = None,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """Combine per-method scores along ``axis``.

    Args:
        scores: ``(..., M)`` scores, expected already on a common scale.
        rule:
            ``mean`` — the safe default; averages away idiosyncratic noise.
            ``geometric`` — demands agreement: one dissenting method near zero
            pulls the result down, which suits high-precision fusion.
            ``max`` — any method firing is enough; high recall, low precision.
            ``median`` — robust to one badly-calibrated method.
            ``rank_mean`` — ranks each method first, then averages, so a
            method with a heavy tail cannot dominate.
        weights: per-method weights. Not accepted by ``max`` or ``median``,
            where they have no meaning.
    """
    if rule not in RULES:
        raise ValueError(f"fusion rule must be one of {RULES}, got {rule!r}")

    array = np.asarray(scores, dtype=np.float32)
    n_methods = array.shape[axis]

    if weights is not None:
        if rule in ("max", "median"):
            raise ValueError(f"the {rule!r} rule does not accept weights")
        weight_array = np.asarray(weights, dtype=np.float32)
        if weight_array.size != n_methods:
            raise ValueError(f"got {weight_array.size} weights for {n_methods} methods")
        total = float(weight_array.sum())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        weight_array = weight_array / total
        shape = [1] * array.ndim
        shape[axis] = n_methods
        weight_array = weight_array.reshape(shape)
    else:
        weight_array = None

    if rule == "max":
        return array.max(axis=axis).astype(np.float32)
    if rule == "median":
        return np.median(array, axis=axis).astype(np.float32)

    if rule == "rank_mean":
        ranked = np.empty_like(array)
        for method in range(n_methods):
            index = [slice(None)] * array.ndim
            index[axis] = method
            ranked[tuple(index)] = rank_transform(array[tuple(index)])
        array = ranked

    if rule == "geometric":
        # In log space, so one near-zero method dominates as intended without
        # the product underflowing across a dozen methods.
        logs = np.log(np.clip(array, 1e-8, None))
        if weight_array is None:
            return np.exp(logs.mean(axis=axis)).astype(np.float32)
        return np.exp((logs * weight_array).sum(axis=axis)).astype(np.float32)

    if weight_array is None:
        return array.mean(axis=axis).astype(np.float32)
    return (array * weight_array).sum(axis=axis).astype(np.float32)
