"""Image and mask preprocessing.

One definition of "how an image becomes a tensor", instead of the ``tx =
transforms.Compose([...])`` block that was re-typed inside six copies of
``InferenceDataset`` and five of ``TrainGoodDataset``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torchvision import transforms

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_transform",
    "load_image",
    "load_mask",
]

#: DINOv2/v3 and the torchvision ResNets share the ImageNet statistics.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_transform(input_size: int, *, normalise: bool = True):
    """Resize → tensor → (optionally) ImageNet-normalise."""
    steps: list = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ]
    if normalise:
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


def load_image(path: Path, transform) -> torch.Tensor:
    """Read an RGB image and apply ``transform``."""
    with Image.open(path) as im:
        return transform(im.convert("RGB"))


def load_mask(
    path: Path | None, input_size: int, threshold: int = 127
) -> npt.NDArray[np.float32]:
    """Read a binary ground-truth mask, or return zeros when there is none.

    Nearest-neighbour resize: a bilinear one would invent fractional labels
    along every defect boundary, which then propagate into the AP.
    """
    if path is None:
        return np.zeros((input_size, input_size), dtype=np.float32)
    with Image.open(path) as mask:
        mask = mask.convert("L").resize((input_size, input_size), Image.NEAREST)
        return (np.asarray(mask) > threshold).astype(np.float32)
