"""The backbone catalogue: names, dimensions, strides, short tags.

Pure data and pure functions. No torch import, so the tables can be consulted
during config validation before any model is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ALL_BACKBONES",
    "BACKBONE_CHANNELS",
    "BACKBONE_SHORT",
    "DINOV2_SPECS",
    "DINOV3_CONVNEXT_SPECS",
    "DINOV3_VIT_SPECS",
    "DINO_BACKBONES",
    "RESNET_BACKBONES",
    "BackboneFamily",
    "BackboneSpec",
    "channels_for",
    "family_of",
    "is_dino",
    "n_layers",
    "patch_size_of",
    "short_tag",
    "validate_input_size",
]

BackboneFamily = Literal["resnet", "dinov2", "dinov2_reg", "dinov3", "dinov3_convnext"]


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    """Everything knowable about a backbone without instantiating it."""

    name: str
    family: BackboneFamily
    #: {layer or block or stage index -> channel count}
    channels: dict[int, int]
    #: Effective stride from input pixels to the first feature map. ``None``
    #: for ResNet, where it depends on which layer is tapped.
    patch_size: int | None
    n_layers: int
    n_register: int = 0

    @property
    def embed_dim(self) -> int:
        """Channel count of the last layer."""
        return self.channels[max(self.channels)]


# ─────────────────────────────────────────────────────────────────────────────
# Raw specs
# ─────────────────────────────────────────────────────────────────────────────
RESNET_CHANNELS: dict[str, dict[int, int]] = {
    "wide_resnet50_2": {1: 256, 2: 512, 3: 1024, 4: 2048},
    "resnet50": {1: 256, 2: 512, 3: 1024, 4: 2048},
    "resnet18": {1: 64, 2: 128, 3: 256, 4: 512},
}

DINOV2_SPECS: dict[str, dict[str, int]] = {
    "dinov2_vits14": {"embed": 384, "n_blocks": 12, "patch": 14, "reg": 0},
    "dinov2_vitb14": {"embed": 768, "n_blocks": 12, "patch": 14, "reg": 0},
    "dinov2_vitl14": {"embed": 1024, "n_blocks": 24, "patch": 14, "reg": 0},
    "dinov2_vitg14": {"embed": 1536, "n_blocks": 40, "patch": 14, "reg": 0},
    "dinov2_vits14_reg": {"embed": 384, "n_blocks": 12, "patch": 14, "reg": 4},
    "dinov2_vitb14_reg": {"embed": 768, "n_blocks": 12, "patch": 14, "reg": 4},
    "dinov2_vitl14_reg": {"embed": 1024, "n_blocks": 24, "patch": 14, "reg": 4},
    "dinov2_vitg14_reg": {"embed": 1536, "n_blocks": 40, "patch": 14, "reg": 4},
}

DINOV3_VIT_SPECS: dict[str, dict[str, int]] = {
    "dinov3_vits16": {"embed": 384, "n_blocks": 12, "patch": 16, "reg": 4},
    "dinov3_vits16plus": {"embed": 384, "n_blocks": 12, "patch": 16, "reg": 4},
    "dinov3_vitb16": {"embed": 768, "n_blocks": 12, "patch": 16, "reg": 4},
    "dinov3_vitl16": {"embed": 1024, "n_blocks": 24, "patch": 16, "reg": 4},
    "dinov3_vith16plus": {"embed": 1280, "n_blocks": 32, "patch": 16, "reg": 4},
    "dinov3_vit7b16": {"embed": 4096, "n_blocks": 40, "patch": 16, "reg": 4},
}

DINOV3_CONVNEXT_SPECS: dict[str, dict[str, object]] = {
    "dinov3_convnext_tiny": {"channels": [96, 192, 384, 768], "patch_eff": 4},
    "dinov3_convnext_small": {"channels": [96, 192, 384, 768], "patch_eff": 4},
    "dinov3_convnext_base": {"channels": [128, 256, 512, 1024], "patch_eff": 4},
    "dinov3_convnext_large": {"channels": [192, 384, 768, 1536], "patch_eff": 4},
}

BACKBONE_SHORT: dict[str, str] = {
    "wide_resnet50_2": "wrn50",
    "resnet50": "rn50",
    "resnet18": "rn18",
    "dinov2_vits14": "dnv2s14",
    "dinov2_vitb14": "dnv2b14",
    "dinov2_vitl14": "dnv2l14",
    "dinov2_vitg14": "dnv2g14",
    "dinov2_vits14_reg": "dnv2s14r",
    "dinov2_vitb14_reg": "dnv2b14r",
    "dinov2_vitl14_reg": "dnv2l14r",
    "dinov2_vitg14_reg": "dnv2g14r",
    "dinov3_vits16": "dnv3s16",
    "dinov3_vits16plus": "dnv3sp16",
    "dinov3_vitb16": "dnv3b16",
    "dinov3_vitl16": "dnv3l16",
    "dinov3_vith16plus": "dnv3hp16",
    "dinov3_vit7b16": "dnv37b16",
    "dinov3_convnext_tiny": "dnv3cxt",
    "dinov3_convnext_small": "dnv3cxs",
    "dinov3_convnext_base": "dnv3cxb",
    "dinov3_convnext_large": "dnv3cxl",
}


# ─────────────────────────────────────────────────────────────────────────────
# Derived catalogue
# ─────────────────────────────────────────────────────────────────────────────
def _build_specs() -> dict[str, BackboneSpec]:
    specs: dict[str, BackboneSpec] = {}

    for name, channels in RESNET_CHANNELS.items():
        specs[name] = BackboneSpec(
            name=name,
            family="resnet",
            channels=dict(channels),
            patch_size=None,
            n_layers=len(channels),
        )

    for name, spec in DINOV2_SPECS.items():
        n_blocks = int(spec["n_blocks"])
        specs[name] = BackboneSpec(
            name=name,
            family="dinov2_reg" if spec["reg"] else "dinov2",
            channels={i: int(spec["embed"]) for i in range(n_blocks)},
            patch_size=int(spec["patch"]),
            n_layers=n_blocks,
            n_register=int(spec["reg"]),
        )

    for name, spec in DINOV3_VIT_SPECS.items():
        n_blocks = int(spec["n_blocks"])
        specs[name] = BackboneSpec(
            name=name,
            family="dinov3",
            channels={i: int(spec["embed"]) for i in range(n_blocks)},
            patch_size=int(spec["patch"]),
            n_layers=n_blocks,
            n_register=int(spec["reg"]),
        )

    for name, spec in DINOV3_CONVNEXT_SPECS.items():
        channels = list(spec["channels"])  # type: ignore[arg-type]
        specs[name] = BackboneSpec(
            name=name,
            family="dinov3_convnext",
            channels=dict(enumerate(int(c) for c in channels)),
            patch_size=int(spec["patch_eff"]),  # type: ignore[arg-type]
            n_layers=len(channels),
        )

    return specs


SPECS: dict[str, BackboneSpec] = _build_specs()

RESNET_BACKBONES: frozenset[str] = frozenset(RESNET_CHANNELS)
DINOV2_BACKBONES: frozenset[str] = frozenset(DINOV2_SPECS)
DINOV3_BACKBONES: frozenset[str] = frozenset(DINOV3_VIT_SPECS) | frozenset(
    DINOV3_CONVNEXT_SPECS
)
DINO_BACKBONES: frozenset[str] = DINOV2_BACKBONES | DINOV3_BACKBONES
ALL_BACKBONES: tuple[str, ...] = tuple(sorted(SPECS))

#: ``{backbone: {layer_index: channels}}`` — what multi-scale fusion reads.
BACKBONE_CHANNELS: dict[str, dict[int, int]] = {
    name: spec.channels for name, spec in SPECS.items()
}


def spec_for(name: str) -> BackboneSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown backbone {name!r}. Known: {', '.join(ALL_BACKBONES)}"
        ) from None


def is_dino(name: str) -> bool:
    return name in DINO_BACKBONES


def family_of(name: str) -> BackboneFamily:
    return spec_for(name).family


def channels_for(name: str, layer: int) -> int:
    spec = spec_for(name)
    try:
        return spec.channels[layer]
    except KeyError:
        raise ValueError(
            f"{name} has no layer {layer}; valid: {sorted(spec.channels)}"
        ) from None


def n_layers(name: str) -> int:
    return spec_for(name).n_layers


def patch_size_of(name: str) -> int | None:
    return spec_for(name).patch_size


def short_tag(name: str) -> str:
    return BACKBONE_SHORT.get(name, name)


def validate_input_size(name: str, input_size: int) -> None:
    """Fail fast when the input size is not a multiple of the patch stride."""
    patch = patch_size_of(name)
    if patch is None:
        return
    if input_size % patch != 0:
        raise ValueError(
            f"input_size={input_size} must be a multiple of {patch} for {name} "
            f"(DINOv2 uses 14, DINOv3 uses 16). "
            f"Nearest valid: {input_size // patch * patch} or "
            f"{(input_size // patch + 1) * patch}."
        )


def resolve_target_layer(
    backbone: str, feature_layers: tuple[int, ...], explicit: int | None = None
) -> int:
    """Pick the layer whose spatial grid the fused feature map is resampled to.

    A finer target layer means a higher-resolution score map and more memory.
    Default: the shallowest requested layer, which is the finest one.
    """
    if not feature_layers:
        raise ValueError("feature_layers must not be empty")
    spec = spec_for(backbone)
    for layer in feature_layers:
        if layer not in spec.channels:
            raise ValueError(
                f"{backbone} has no layer {layer}; valid: {sorted(spec.channels)}"
            )
    if explicit is None:
        return min(feature_layers)
    if explicit not in feature_layers:
        raise ValueError(
            f"target_layer={explicit} must be one of feature_layers={list(feature_layers)}"
        )
    return explicit
