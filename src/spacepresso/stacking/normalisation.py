"""Rank normalisation of detector scores.

Detectors produce scores on incomparable scales: PatchCore emits cosine
distances in roughly [0, 0.5], FastFlow emits negative log-likelihoods that
can reach the thousands. Before those can be fused, they have to be put on a
common scale, and *rank* is the only transform that does so without assuming
anything about the distributions.

The scope of the ranking is the decision that matters:

``global``
    Rank every pixel of every image of every class together. Preserves the
    between-image ordering that the pooled leaderboard metric measures, and
    is the right default for the final submission.
``per_class``
    Rank within each class. Removes systematic per-class score offsets —
    useful when one class is intrinsically harder and drags the global
    ranking — but destroys the between-class ordering.
``per_class_view``
    Rank within each ``(class, view)`` group. The finest scope; corrects for
    viewpoint-dependent scale on top of class.
``per_image``
    Rank within each image. Maximum scale removal, and the most destructive:
    every image ends up with the same score distribution, so the pooled
    metric can no longer tell a clean image from a defective one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["SCOPES", "Scope", "rank_normalise", "rank_transform"]

Scope = Literal["none", "global", "per_class", "per_class_view", "per_image"]
SCOPES: tuple[Scope, ...] = (
    "none",
    "global",
    "per_class",
    "per_class_view",
    "per_image",
)


def rank_transform(values: npt.NDArray) -> npt.NDArray[np.float32]:
    """Replace values by their rank, scaled to [0, 1].

    Ties are broken by original position (``kind="stable"``), which matters
    for the large constant regions these score maps contain: a random
    tie-break would inject noise into exactly the pixels the metric is most
    sensitive to.
    """
    flat = np.asarray(values).ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks.reshape(np.shape(values))


def _groups(
    scope: Scope,
    classes: Sequence[str] | None,
    views: Sequence[int] | None,
    n: int,
) -> dict[object, npt.NDArray[np.intp]]:
    """Index groups for a scope. One entry per group of images."""
    if scope == "global":
        return {None: np.arange(n)}
    if scope == "per_image":
        return {i: np.asarray([i]) for i in range(n)}

    if classes is None:
        raise ValueError(f"scope={scope!r} needs per-image class labels")
    if scope == "per_class":
        keys: list[object] = list(classes)
    else:
        if views is None:
            raise ValueError("scope='per_class_view' needs per-image view indices")
        keys = list(zip(classes, views, strict=True))

    groups: dict[object, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return {key: np.asarray(value) for key, value in groups.items()}


def rank_normalise(
    scores: npt.NDArray[np.float32],
    scope: Scope = "global",
    *,
    classes: Sequence[str] | None = None,
    views: Sequence[int] | None = None,
    inplace: bool = False,
) -> npt.NDArray[np.float32]:
    """Rank-normalise ``(N, H, W, M)`` scores, independently per method.

    Args:
        scores: one score map per image per method.
        scope: see the module docstring.
        classes: per-image class labels, for the class-aware scopes.
        views: per-image view indices, for ``per_class_view``.
        inplace: overwrite ``scores``. These arrays are gigabytes for a real
            run, so the stacker uses this; copying is the safer default.

    The original had six separate functions for this — ``*_global_*``,
    ``*_per_class_*``, ``*_per_class_view_*``, each once for validation and
    once for test, with subtly different tie handling.
    """
    if scope == "none":
        return scores

    out = scores if inplace else scores.copy()
    n_images, _, _, n_methods = out.shape

    for indices in _groups(scope, classes, views, n_images).values():
        block = out[indices]
        flat = block.reshape(-1, n_methods)
        for method in range(n_methods):
            flat[:, method] = rank_transform(flat[:, method])
        out[indices] = flat.reshape(block.shape)

    return out
