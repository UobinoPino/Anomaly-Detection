"""Score-map post-processing: TTA, per-view calibration, spatial filters.

Everything here operates on score maps alone. Nothing imports a detector —
which is what broke the ``patchcore_baseline_v2`` ⇄ ``multiview_consensus``
cycle: the per-view calibrator now takes a scoring callable rather than a live
``PatchCore`` instance.
"""

from spacepresso.postprocess.multiview import (
    ViewStats,
    apply_view_norm,
    fit_view_norm,
)
from spacepresso.postprocess.spatial import (
    apply_spatial_prior,
    load_spatial_priors,
    suppress_small_components,
    within_image_rank,
)
from spacepresso.postprocess.tta import TTA_MODES, TTAMode, apply_tta

__all__ = [
    "TTA_MODES",
    "TTAMode",
    "ViewStats",
    "apply_spatial_prior",
    "apply_tta",
    "apply_view_norm",
    "fit_view_norm",
    "load_spatial_priors",
    "suppress_small_components",
    "within_image_rank",
]
