"""Frozen feature extractors, addressed by name.

    backbone = build_backbone("dinov2_vitb14_reg")
    maps = backbone(x, layers=(3, 6, 9, 11))     # {layer: (B, C, H', W')}

Detectors depend on :class:`~spacepresso.backbones.base.Backbone`, never on a
concrete family, so adding a backbone means adding a row to ``registry`` and a
class here — not editing 14 detectors.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from spacepresso.backbones.base import Backbone, FeatureMaps, patchify_and_combine
from spacepresso.backbones.registry import (
    ALL_BACKBONES,
    BACKBONE_CHANNELS,
    BACKBONE_SHORT,
    DINO_BACKBONES,
    RESNET_BACKBONES,
    BackboneSpec,
    channels_for,
    family_of,
    is_dino,
    n_layers,
    patch_size_of,
    resolve_target_layer,
    short_tag,
    spec_for,
    validate_input_size,
)

__all__ = [
    "ALL_BACKBONES",
    "BACKBONE_CHANNELS",
    "BACKBONE_SHORT",
    "DINO_BACKBONES",
    "RESNET_BACKBONES",
    "Backbone",
    "BackboneSpec",
    "FeatureMaps",
    "build_backbone",
    "channels_for",
    "family_of",
    "is_dino",
    "n_layers",
    "patch_size_of",
    "patchify_and_combine",
    "resolve_target_layer",
    "short_tag",
    "spec_for",
    "validate_input_size",
]


@lru_cache(maxsize=4)
def _build_cached(name: str) -> Backbone:
    """Build and cache a backbone.

    Several detectors construct the same backbone once per class, and a
    DINOv2-L download-and-init is not free. The cache is small on purpose: a
    ViT-g weighs several gigabytes and holding more than a handful would blow
    the memory budget the README commits to.
    """
    spec = spec_for(name)
    if spec.family == "resnet":
        from spacepresso.backbones.resnet import ResNetBackbone

        return ResNetBackbone(spec)

    from spacepresso.backbones.dino import DinoBackbone

    return DinoBackbone(spec)


def build_backbone(
    name: str,
    *,
    device: torch.device | str | None = None,
    cache: bool = True,
) -> Backbone:
    """Instantiate a frozen backbone by name, optionally moving it to ``device``."""
    backbone = _build_cached(name) if cache else _build_uncached(name)
    if device is not None:
        backbone = backbone.to(device)
    return backbone


def _build_uncached(name: str) -> Backbone:
    _build_cached.cache_clear()
    try:
        return _build_cached(name)
    finally:
        _build_cached.cache_clear()
