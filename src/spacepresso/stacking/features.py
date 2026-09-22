"""Per-pixel feature construction for the stacker.

The stacker learns a function from *several detectors' scores at a pixel* to
*is this pixel anomalous*. This module turns a set of aligned score maps into
the feature matrix that function is fitted on.

Feature families, grouped by what they know:

* **per-method** — what one detector says here, and in a neighbourhood.
* **cross-method** — what the detectors say about each other: agreement,
  spread, consensus. This is the part a single detector cannot provide and
  the main reason stacking beats averaging.
* **spatial** — where in the image this pixel is, plus the learned per-class
  prior over defect locations.
* **context** — class and view identity, so one model can serve all classes
  while still specialising.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from spacepresso.core.imaging import resize_bilinear
from spacepresso.stacking.normalisation import rank_transform

__all__ = [
    "FeatureConfig",
    "FeatureContext",
    "feature_names",
    "featurize_image",
]

Plane = npt.NDArray[np.float32]
#: A feature family yields ``(name, plane)`` pairs.
Family = Callable[["FeatureContext"], Iterator[tuple[str, Plane]]]


@dataclass
class FeatureConfig:
    """Which feature families to build."""

    # per-method
    raw_score: bool = True
    per_image_rank: bool = True
    gaussian_sigmas: tuple[float, ...] = (3.0,)
    window_sizes: tuple[int, ...] = (7,)
    window_mean: bool = True
    window_max: bool = False
    window_std: bool = False
    gradient: bool = False
    laplacian: bool = False
    image_aggregates: tuple[str, ...] = ("imgp99", "imgstd")

    # cross-method
    cross_stats: bool = True
    cross_consensus: bool = True
    cross_cv: bool = True
    min_top_k: int = 3

    # spatial and context
    spatial_coords: bool = True
    spatial_prior: bool = True
    class_onehot: bool = True
    view_onehot: bool = True
    n_view_slots: int = 5

    def families(self) -> list[str]:
        """Names of the enabled families, in build order."""
        enabled = ["per_method"]
        if self.cross_stats or self.cross_consensus or self.cross_cv or self.min_top_k:
            enabled.append("cross_method")
        if self.spatial_coords or self.spatial_prior:
            enabled.append("spatial")
        if self.class_onehot or self.view_onehot:
            enabled.append("context")
        return enabled


@dataclass
class FeatureContext:
    """Everything a feature family may read."""

    scores: list[Plane]
    config: FeatureConfig
    shape: tuple[int, int]
    class_id: str | None = None
    all_classes: Sequence[str] = field(default_factory=tuple)
    view: int | None = None
    prior: Plane | None = None
    _cache: dict = field(default_factory=dict)

    @property
    def n_methods(self) -> int:
        return len(self.scores)

    def constant(self, value: float) -> Plane:
        return np.full(self.shape, value, dtype=np.float32)

    def stack(self) -> npt.NDArray[np.float32]:
        """``(M, H, W)`` — cached, because four families all want it."""
        if "stack" not in self._cache:
            self._cache["stack"] = np.stack(self.scores, axis=0).astype(np.float32)
        return self._cache["stack"]


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────
def _ndimage():
    from scipy import ndimage

    return ndimage


def _window_mean(plane: Plane, size: int) -> Plane:
    return (
        _ndimage().uniform_filter(plane, size=size, mode="reflect").astype(np.float32)
    )


def _window_max(plane: Plane, size: int) -> Plane:
    return (
        _ndimage().maximum_filter(plane, size=size, mode="reflect").astype(np.float32)
    )


def _window_std(plane: Plane, size: int) -> Plane:
    ndi = _ndimage()
    mean = ndi.uniform_filter(plane, size=size, mode="reflect")
    mean_sq = ndi.uniform_filter(plane * plane, size=size, mode="reflect")
    return np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None)).astype(np.float32)


def _gaussian(plane: Plane, sigma: float) -> Plane:
    return (
        _ndimage()
        .gaussian_filter(plane, sigma=sigma, mode="reflect")
        .astype(np.float32)
    )


def _gradient_magnitude(plane: Plane) -> Plane:
    ndi = _ndimage()
    dx = ndi.sobel(plane, axis=0, mode="reflect")
    dy = ndi.sobel(plane, axis=1, mode="reflect")
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _laplacian(plane: Plane) -> Plane:
    return _ndimage().laplace(plane, mode="reflect").astype(np.float32)


_AGGREGATES: dict[str, Callable[[Plane], float]] = {
    "imgmax": lambda s: float(s.max()),
    "imgp99": lambda s: float(np.percentile(s, 99)),
    "imgmean": lambda s: float(s.mean()),
    "imgstd": lambda s: float(s.std()),
}


# ─────────────────────────────────────────────────────────────────────────────
# Families
# ─────────────────────────────────────────────────────────────────────────────
def per_method(ctx: FeatureContext) -> Iterator[tuple[str, Plane]]:
    """What each detector says at this pixel, and around it.

    The neighbourhood views matter because detectors disagree about
    *localisation* as much as about presence: a window mean lets the model
    use a detector that fires near the defect but not exactly on it.
    """
    config = ctx.config
    for index, score in enumerate(ctx.scores):
        if score.shape != ctx.shape:
            raise ValueError(
                f"method {index} has shape {score.shape}, expected {ctx.shape}"
            )
        prefix = f"m{index}"
        if config.raw_score:
            yield f"{prefix}_raw", score
        if config.per_image_rank:
            yield f"{prefix}_rank", rank_transform(score)
        for sigma in config.gaussian_sigmas:
            yield f"{prefix}_g{sigma:g}", _gaussian(score, sigma)
        for size in config.window_sizes:
            if config.window_mean:
                yield f"{prefix}_mean{size}", _window_mean(score, size)
            if config.window_max:
                yield f"{prefix}_max{size}", _window_max(score, size)
            if config.window_std:
                yield f"{prefix}_std{size}", _window_std(score, size)
        if config.gradient:
            yield f"{prefix}_grad", _gradient_magnitude(score)
        if config.laplacian:
            yield f"{prefix}_lap", _laplacian(score)
        for name in config.image_aggregates:
            if name not in _AGGREGATES:
                raise ValueError(
                    f"unknown image aggregate {name!r}; "
                    f"known: {', '.join(sorted(_AGGREGATES))}"
                )
            yield f"{prefix}_{name}", ctx.constant(_AGGREGATES[name](score))


def cross_method(ctx: FeatureContext) -> Iterator[tuple[str, Plane]]:
    """How the detectors relate to each other at this pixel.

    This is where stacking earns its keep over a weighted average: a pixel
    two detectors both call anomalous is far more likely to be a real defect
    than one a single detector shouts about, and the spread across methods
    encodes that directly.
    """
    config = ctx.config
    if ctx.n_methods < 2:
        return

    stack = ctx.stack()

    if config.cross_stats:
        maximum = stack.max(axis=0)
        minimum = stack.min(axis=0)
        yield "x_mean", stack.mean(axis=0)
        yield "x_max", maximum
        yield "x_min", minimum
        yield "x_std", stack.std(axis=0)
        yield "x_range", (maximum - minimum).astype(np.float32)

    if config.cross_consensus:
        # Counts assume rank-normalised inputs, where a score is its own
        # percentile — which is what the stacker always feeds in.
        yield "xc_top1pct", (stack >= 0.99).sum(axis=0).astype(np.float32)
        yield "xc_top5pct", (stack >= 0.95).sum(axis=0).astype(np.float32)

    if config.cross_cv:
        mean = stack.mean(axis=0)
        yield "xc_cv", (stack.std(axis=0) / (mean + 1e-6)).astype(np.float32)

    k = config.min_top_k
    if k and ctx.n_methods >= k:
        # The k-th largest: high only where at least k detectors agree, which
        # is a much stricter consensus signal than the mean.
        partitioned = np.partition(stack, -k, axis=0)[-k:]
        yield f"xc_min_top{k}", partitioned.min(axis=0).astype(np.float32)


def spatial(ctx: FeatureContext) -> Iterator[tuple[str, Plane]]:
    """Where in the frame this pixel sits, and how likely defects are there."""
    config = ctx.config
    height, width = ctx.shape

    if config.spatial_coords:
        cache_key = ("coords", ctx.shape)
        if cache_key not in ctx._cache:
            ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
            xs /= max(width - 1, 1)
            ys /= max(height - 1, 1)
            centre = np.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2).astype(np.float32)
            edge = np.minimum.reduce([xs, ys, 1 - xs, 1 - ys]).astype(np.float32)
            ctx._cache[cache_key] = {
                "s_x": xs,
                "s_y": ys,
                "s_dist_center": centre,
                "s_dist_edge": edge,
            }
        yield from ctx._cache[cache_key].items()

    if config.spatial_prior and ctx.prior is not None:
        cache_key = ("prior", ctx.class_id, ctx.shape)
        if cache_key not in ctx._cache:
            ctx._cache[cache_key] = resize_bilinear(ctx.prior, ctx.shape)
        yield "s_prior", ctx._cache[cache_key]


def context(ctx: FeatureContext) -> Iterator[tuple[str, Plane]]:
    """Class and view identity as one-hot constant planes.

    Constant planes are a blunt encoding, but they let one tree ensemble
    specialise per class without training eight separate models — the tree
    splits on the indicator and everything below that split is class-specific.
    """
    config = ctx.config

    if config.class_onehot and ctx.all_classes:
        try:
            active = list(ctx.all_classes).index(ctx.class_id)  # type: ignore[arg-type]
        except ValueError:
            active = -1
        for index, name in enumerate(ctx.all_classes):
            yield f"c_{name}", ctx.constant(1.0 if index == active else 0.0)

    if config.view_onehot:
        view = ctx.view if ctx.view is not None else -1
        for slot in range(config.n_view_slots):
            yield f"v_{slot}", ctx.constant(1.0 if view == slot else 0.0)


_FAMILIES: dict[str, Family] = {
    "per_method": per_method,
    "cross_method": cross_method,
    "spatial": spatial,
    "context": context,
}


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────
def featurize_image(
    scores: Sequence[Plane],
    config: FeatureConfig,
    *,
    class_id: str | None = None,
    all_classes: Sequence[str] = (),
    view: int | None = None,
    prior: Plane | None = None,
    cache: dict | None = None,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Build the ``(H*W, F)`` feature matrix for one image.

    Returns the matrix and the feature names, produced together so they
    cannot fall out of step.
    """
    if not scores:
        raise ValueError("featurize_image() needs at least one score map")

    planes = [np.asarray(s, dtype=np.float32) for s in scores]
    ctx = FeatureContext(
        scores=planes,
        config=config,
        shape=planes[0].shape,
        class_id=class_id,
        all_classes=tuple(all_classes),
        view=view,
        prior=prior,
        _cache=cache if cache is not None else {},
    )

    names: list[str] = []
    columns: list[Plane] = []
    for family in config.families():
        for name, plane in _FAMILIES[family](ctx):
            names.append(name)
            columns.append(plane)

    matrix = np.stack(columns, axis=-1).reshape(-1, len(columns))
    return matrix.astype(np.float32, copy=False), names


def feature_names(
    config: FeatureConfig,
    n_methods: int,
    *,
    all_classes: Sequence[str] = (),
    shape: tuple[int, int] = (8, 8),
) -> list[str]:
    """The feature names this config produces, without building real features."""
    dummy = [np.zeros(shape, dtype=np.float32) for _ in range(n_methods)]
    _, names = featurize_image(
        dummy,
        config,
        class_id=all_classes[0] if all_classes else None,
        all_classes=all_classes,
        view=0,
        prior=np.zeros(shape, dtype=np.float32) if config.spatial_prior else None,
    )
    return names
