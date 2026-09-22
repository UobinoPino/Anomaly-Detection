"""Torchvision ResNet backbones."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    resnet50,
    wide_resnet50_2,
)

from spacepresso.backbones.base import Backbone, FeatureMaps
from spacepresso.backbones.registry import BackboneSpec, spec_for

__all__ = ["ResNetBackbone"]

_BUILDERS = {
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
}


class ResNetBackbone(Backbone):
    """ImageNet-pretrained ResNet, tapped at layers 1–4.

    Layer ``k`` has stride ``2^(k+1)``: at a 224px input, layer 2 gives a
    28x28 grid and layer 3 a 14x14 one.
    """

    def __init__(self, spec: BackboneSpec, *, pretrained: bool = True) -> None:
        super().__init__(spec)
        builder, weights = _BUILDERS[spec.name]
        # pretrained=False exists for the architecture tests, which must run
        # in CI without reaching out to download.pytorch.org.
        model = builder(weights=weights if pretrained else None)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.blocks = nn.ModuleList(
            [model.layer1, model.layer2, model.layer3, model.layer4]
        )
        self.freeze()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, layers: Sequence[int] = (2, 3)) -> FeatureMaps:
        wanted = sorted({int(layer) for layer in layers})
        for layer in wanted:
            if layer not in self.spec.channels:
                raise ValueError(
                    f"{self.name}: layer {layer} out of range "
                    f"{sorted(self.spec.channels)}"
                )

        out: FeatureMaps = {}
        deepest = max(wanted)
        h = self.stem(x)
        for index, block in enumerate(self.blocks, start=1):
            h = block(h)
            if index in wanted:
                out[index] = h
            if index >= deepest:
                break
        return out


def build(name: str) -> ResNetBackbone:
    return ResNetBackbone(spec_for(name))
