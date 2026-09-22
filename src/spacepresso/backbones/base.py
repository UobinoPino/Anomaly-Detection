"""The backbone interface.

Replaces ``FeatureExtractor``, which was one class trying to be two models:
it branched on ``self.kind`` at every call and left half its attributes set to
``None`` in each mode (``self.stem = self.layer1 = self.layer2 = None`` for
DINO backbones, ``self.dino = self.info = None`` for ResNets). Any caller
reading an attribute had to know which mode it was in.

There is now one protocol with two implementations. A caller asks for
``backbone(x, layers=(3, 6, 9, 11))`` and gets ``{layer: (B, C, H', W')}``
back, whichever family it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from spacepresso.backbones.registry import BackboneSpec

__all__ = ["Backbone", "FeatureMaps", "patchify_and_combine"]

#: ``{layer_index: (B, C, H', W')}`` feature maps at possibly differing grids.
FeatureMaps = dict[int, torch.Tensor]


class Backbone(nn.Module, ABC):
    """A frozen feature extractor.

    Implementations are always in ``eval()`` mode with ``requires_grad=False``
    — every detector in this project treats the backbone as fixed, and a
    trainable one has never been wanted. Detectors that *do* train something
    (EfficientAD's student, FastFlow's coupling layers, CFA's descriptor)
    attach their trainable module on top rather than unfreezing this one.
    """

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def family(self) -> str:
        return self.spec.family

    @property
    def patch_size(self) -> int | None:
        return self.spec.patch_size

    def channels(self, layer: int) -> int:
        return self.spec.channels[layer]

    def total_channels(self, layers: Sequence[int]) -> int:
        """Channel count after concatenating ``layers``."""
        return sum(self.spec.channels[layer] for layer in layers)

    def freeze(self) -> Backbone:
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)
        return self

    @abstractmethod
    def forward(
        self, x: torch.Tensor, layers: Sequence[int] = (2, 3)
    ) -> FeatureMaps:
        """Return ``{layer: (B, C, H', W')}`` for each requested layer."""
        raise NotImplementedError


def patchify_and_combine(
    maps: FeatureMaps,
    *,
    patch_size: int = 3,
    target_layer: int,
    normalise: bool = True,
) -> torch.Tensor:
    """Local neighbourhood pooling, multi-layer concat, L2-normalise.

    The PatchCore feature construction, shared by every detector that scores
    patch descriptors: average-pool each layer over a ``patch_size`` window to
    give each position some context, resample every layer onto the target
    layer's grid, concatenate along channels, then L2-normalise so that
    cosine and Euclidean distances agree.

    Returns ``(B, H'*W', C_total)``.
    """
    if target_layer not in maps:
        raise KeyError(f"target_layer={target_layer} not in {sorted(maps)}")

    height, width = maps[target_layer].shape[-2:]
    pool = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)

    pooled = []
    for layer in sorted(maps):
        feature = maps[layer]
        if feature.shape[-2:] != (height, width):
            feature = F.interpolate(
                feature, size=(height, width), mode="bilinear", align_corners=False
            )
        pooled.append(pool(feature))

    combined = torch.cat(pooled, dim=1)
    batch, channels = combined.shape[0], combined.shape[1]
    combined = combined.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    if normalise:
        combined = F.normalize(combined, p=2, dim=-1)
    return combined
