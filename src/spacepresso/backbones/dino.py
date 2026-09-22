"""DINOv2 and DINOv3 backbones.

Ported from ``exploit_dataset/dinov3_loader.py``, whose own docstring called it
"the single source of truth" for backbones — while sitting in the
dataset-analysis directory, which is why seven detectors under ``models/``
carried a ``sys.path`` hack and still failed to import it from a clean clone.

Three loading strategies, tried in order for DINOv3:

1. ``torch.hub`` against a local clone of the ``facebookresearch/dinov3`` repo
   plus downloaded weights (``$DINOV3_REPO`` / ``$DINOV3_WEIGHTS``),
2. HuggingFace ``transformers``, wrapped so it exposes the same
   ``get_intermediate_layers`` surface as the hub model,
3. a diagnostic error listing exactly what to set.

DINOv2 comes straight from ``torch.hub``; it is not gated.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from spacepresso.backbones.base import Backbone, FeatureMaps
from spacepresso.backbones.registry import (
    DINOV2_SPECS,
    DINOV3_CONVNEXT_SPECS,
    DINOV3_VIT_SPECS,
    BackboneSpec,
    spec_for,
)
from spacepresso.core.logging import get_logger

__all__ = ["DinoBackbone", "build"]

logger = get_logger(__name__)

_HF_MODEL_IDS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "dinov3_convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "dinov3_convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "dinov3_convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "dinov3_convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}

_DINOV3_SETUP_HELP = """\
Could not load {name}. Two options:

  A — local repo + weights:
      git clone https://github.com/facebookresearch/dinov3 ~/dinov3_repo
      export DINOV3_REPO=~/dinov3_repo
      # request access:
      #   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
      export DINOV3_WEIGHTS=~/dinov3_weights   # put the .pth files here

  B — HuggingFace:
      pip install 'transformers>=4.45'
      # accept the licence at https://huggingface.co/{hf_id}
      huggingface-cli login
"""


# ─────────────────────────────────────────────────────────────────────────────
# Loading strategies
# ─────────────────────────────────────────────────────────────────────────────
def _dinov3_repo_dir() -> Path | None:
    """Locate a local clone of the DINOv3 repo.

    Only ``$DINOV3_REPO`` and the user's home directory are consulted. The
    original also probed ``/work/u10813429/dinov3_repo`` and
    ``/workspace/dinov3_repo`` — two specific machines, hardcoded into a
    library function.
    """
    env = os.environ.get("DINOV3_REPO")
    candidates = [Path(env)] if env else []
    candidates += [Path.home() / "dinov3_repo", Path.home() / "dinov3"]
    for candidate in candidates:
        if (candidate / "hubconf.py").exists():
            return candidate
    return None


def _dinov3_weights_for(name: str) -> str | None:
    """Resolve a weights URL or file for ``name``."""
    explicit = os.environ.get(f"DINOV3_{name.upper()}_URL")
    if explicit:
        return explicit

    weights_dir = os.environ.get("DINOV3_WEIGHTS")
    if not weights_dir:
        return None
    directory = Path(weights_dir)
    if not directory.is_dir():
        return None
    for pattern in (f"{name}*.pth", f"{name}*.pt"):
        for path in sorted(directory.glob(pattern)):
            return str(path)
    return None


def _load_dinov2(name: str) -> nn.Module:
    return torch.hub.load(
        "facebookresearch/dinov2", name, trust_repo=True, source="github"
    )


def _load_dinov3_from_hub(name: str) -> nn.Module | None:
    repo = _dinov3_repo_dir()
    weights = _dinov3_weights_for(name)
    if repo is None or weights is None:
        return None
    try:
        return torch.hub.load(str(repo), name, source="local", weights=weights)
    except Exception as exc:
        logger.warning("DINOv3 torch.hub load failed for %s: %s", name, exc)
        return None


def _load_dinov3_from_hf(spec: BackboneSpec) -> nn.Module | None:
    try:
        from transformers import AutoModel
    except ImportError:
        logger.warning("transformers is not installed; cannot load DINOv3 from HF")
        return None

    model_id = _HF_MODEL_IDS.get(spec.name)
    if model_id is None:
        return None
    try:
        backbone = AutoModel.from_pretrained(model_id)
    except Exception as exc:
        logger.warning("HuggingFace load of %s failed: %s", model_id, exc)
        return None
    backbone.eval()

    if spec.family == "dinov3_convnext":
        return _HFConvNeXtAdapter(backbone)
    return _HFViTAdapter(backbone, spec)


def _load_module(spec: BackboneSpec) -> nn.Module:
    if spec.name in DINOV2_SPECS:
        return _load_dinov2(spec.name)
    if spec.name in DINOV3_VIT_SPECS or spec.name in DINOV3_CONVNEXT_SPECS:
        module = _load_dinov3_from_hub(spec.name) or _load_dinov3_from_hf(spec)
        if module is not None:
            return module
        raise RuntimeError(
            _DINOV3_SETUP_HELP.format(
                name=spec.name, hf_id=_HF_MODEL_IDS.get(spec.name, "…")
            )
        )
    raise ValueError(f"{spec.name} is not a DINO backbone")


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace adapters
# ─────────────────────────────────────────────────────────────────────────────
class _HFViTAdapter(nn.Module):
    """Give a HuggingFace DINOv3 ViT the hub model's ``get_intermediate_layers``."""

    def __init__(self, backbone: nn.Module, spec: BackboneSpec) -> None:
        super().__init__()
        self.backbone = backbone
        self.spec = spec

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int],
        reshape: bool = True,
        norm: bool = True,
        return_class_token: bool = False,
    ) -> list[torch.Tensor]:
        indices = (
            list(range(self.spec.n_layers - n, self.spec.n_layers))
            if isinstance(n, int)
            else list(n)
        )
        batch, _, height, width = x.shape
        patch = self.spec.patch_size
        assert patch is not None
        grid_h, grid_w = height // patch, width // patch

        hidden_states = self.backbone(x, output_hidden_states=True).hidden_states
        skip = 1 + self.spec.n_register  # CLS + register tokens

        outputs = []
        for index in indices:
            tokens = hidden_states[index + 1][:, skip:, :]
            if norm and index == indices[-1]:
                layer_norm = getattr(self.backbone, "layernorm", None)
                if layer_norm is not None:
                    tokens = layer_norm(tokens)
            if reshape:
                tokens = (
                    tokens.reshape(batch, grid_h, grid_w, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            outputs.append(tokens)
        return outputs


class _HFConvNeXtAdapter(nn.Module):
    """HuggingFace DINOv3 ConvNeXt — hidden states are already 4-D maps."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int],
        reshape: bool = True,
        norm: bool = True,
        return_class_token: bool = False,
    ) -> list[torch.Tensor]:
        hidden_states = self.backbone(x, output_hidden_states=True).hidden_states
        indices = (
            list(range(len(hidden_states) - n, len(hidden_states)))
            if isinstance(n, int)
            else list(n)
        )
        return [hidden_states[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────
class DinoBackbone(Backbone):
    """DINOv2 / DINOv3 ViT or ConvNeXt.

    ``layers`` indexes transformer blocks for the ViTs (0 … n_blocks-1) and
    stages for ConvNeXt (0 … 3). CLS and register tokens are stripped before
    the caller sees anything, so a ViT and a ConvNeXt both return
    ``(B, C, H', W')``.
    """

    def __init__(self, spec: BackboneSpec, module: nn.Module | None = None) -> None:
        super().__init__(spec)
        self.model = module if module is not None else _load_module(spec)
        self.freeze()

    @property
    def is_convnext(self) -> bool:
        return self.spec.family == "dinov3_convnext"

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, layers: Sequence[int] = (9,)) -> FeatureMaps:
        wanted = sorted({int(layer) for layer in layers})
        for layer in wanted:
            if not 0 <= layer < self.spec.n_layers:
                kind = "stage" if self.is_convnext else "block"
                raise ValueError(
                    f"{self.name}: {kind} {layer} out of range "
                    f"[0, {self.spec.n_layers - 1}]"
                )

        if not self.is_convnext:
            _, _, height, width = x.shape
            patch = self.spec.patch_size
            assert patch is not None
            if height % patch or width % patch:
                raise ValueError(
                    f"{self.name}: input ({height}, {width}) is not divisible by "
                    f"the patch size {patch}"
                )

        outputs = self.model.get_intermediate_layers(
            x, n=list(wanted), reshape=True, norm=True
        )
        return dict(zip(wanted, outputs, strict=True))


def build(name: str) -> DinoBackbone:
    return DinoBackbone(spec_for(name))
