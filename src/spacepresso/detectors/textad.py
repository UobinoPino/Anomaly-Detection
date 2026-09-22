"""TextAD — supervised segmentation on language-grounded synthetic defects.

Not a published method: a pragmatic one that turns out to work well here. The
dataset ships ``anomaly_descriptions.csv``, a prose description of each
``(class, anomaly_type)`` pair. Those descriptions say what the defects
*look* like — "thin linear mark", "dark blotchy patch", "irregular
fragments" — and that is exactly what a synthesiser needs to know.

So: match the descriptions to procedural defect families, paint
class-appropriate defects onto the normal training images, and train a plain
segmentation U-Net on the result. No pretrained backbone, no memory bank, no
density model. It is the most *supervised* detector in the portfolio, and its
errors are correspondingly uncorrelated with the rest — which is what makes it
valuable to the stacker even when its own AP is middling.

The language is a lantern, not a crutch: without the CSV every class falls
back to a default mixture of defect families and the detector still trains.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.paths import project_root
from spacepresso.core.records import ImageRecord
from spacepresso.data.defects import (
    DefectSpec,
    inject_defects,
    load_class_taxonomy,
    make_class_defect_specs,
)
from spacepresso.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from spacepresso.detectors.base import Detector
from spacepresso.detectors.blocks import UNet, dice_loss, focal_loss
from spacepresso.detectors.training import train_loop

__all__ = ["TextAD", "TextADConfig"]


@dataclass
class TextADConfig(DetectorConfig):
    base_channels: int = 32
    #: Depth of the U-Net. Each level halves the resolution, so input_size
    #: must be divisible by 2**depth.
    depth: int = 4

    descriptions_csv: Path | None = None
    n_defects: tuple[int, int] = (1, 3)
    #: Fraction of training images left clean. Without these the network
    #: never sees a normal image during training and calls everything
    #: defective at test time.
    clean_fraction: float = 0.2

    focal_gamma: float = 2.0
    focal_alpha: float = 0.5
    dice_weight: float = 0.5

    total_iters: int | None = 4000
    epochs: int = 200
    train_batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_defects = tuple(self.n_defects)  # type: ignore[assignment]
        divisor = 2**self.depth
        if self.input_size % divisor:
            raise ValueError(
                f"input_size={self.input_size} must be divisible by "
                f"{divisor} for a depth-{self.depth} U-Net"
            )
        if not 0.0 <= self.clean_fraction < 1.0:
            raise ValueError(
                f"clean_fraction must be in [0, 1), got {self.clean_fraction}"
            )
        if self.descriptions_csv is None:
            default = project_root() / "data" / "anomaly_descriptions.csv"
            self.descriptions_csv = default if default.is_file() else None

    def slug_parts(self) -> list[str]:
        parts = [
            f"in{self.input_size}",
            f"b{self.base_channels}",
            f"d{self.depth}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
            f"bs{self.train_batch_size}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


class _SynthesisDataset(Dataset):
    """Normal images with class-appropriate defects painted on.

    Images are decoded once in ``__init__`` and cached as uint8 arrays. On
    Linux the dataloader workers fork and inherit that cache copy-on-write, so
    per-epoch decode cost goes to zero — which matters because the synthesis
    itself is already the expensive part of each item.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        specs: Sequence[DefectSpec],
        config: TextADConfig,
    ) -> None:
        self.config = config
        self.specs = list(specs)
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32)

        size = config.input_size
        self.images: list[np.ndarray] = []
        for record in records:
            with Image.open(record.path) as handle:
                self.images.append(
                    np.asarray(
                        handle.convert("RGB").resize((size, size), Image.BILINEAR),
                        dtype=np.uint8,
                    )
                )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self.images[index].astype(np.float32) / 255.0

        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(
            (
                self.config.seed,
                worker.id if worker is not None else 0,
                index,
                int(torch.randint(0, 2**31, (1,)).item()),
            )
        )

        if rng.random() < self.config.clean_fraction:
            image = clean
            mask = np.zeros(clean.shape[:2], dtype=np.float32)
        else:
            image, mask = inject_defects(
                clean, self.specs, rng, n_defects_range=self.config.n_defects
            )

        normalised = (image - self.mean) / self.std
        return (
            torch.from_numpy(normalised.transpose(2, 0, 1).copy()),
            torch.from_numpy(mask),
        )


class TextAD(Detector[TextADConfig]):
    name: ClassVar[str] = "textad"
    config_type: ClassVar[type[DetectorConfig]] = TextADConfig

    def __init__(self, config: TextADConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.net = self._build_net().to(self.device)
        self.taxonomy = load_class_taxonomy(config.descriptions_csv)
        self._masks: torch.Tensor | None = None

    def _build_net(self) -> nn.Module:
        # Shares DRAEM's U-Net, which is the same architecture at a different
        # depth — one implementation, not two.
        net = UNet(3, 2, self.config.base_channels)
        net.downs = nn.ModuleList(list(net.downs)[: self.config.depth])
        net.ups = nn.ModuleList(list(net.ups)[-self.config.depth :])
        return net

    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        cls = train_good[0].cls
        specs = make_class_defect_specs(cls, self.taxonomy)
        self.log.info(
            "    %s defect families: %s",
            cls,
            ", ".join(spec.family for spec in specs),
        )

        dataset = _SynthesisDataset(train_good, specs, self.config)
        loader = self.make_loader(
            train_good,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
            dataset=dataset,
        )
        train_loop(
            [self.net],
            _MaskCapture(loader, self),
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="textad",
        )

    def _loss(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self._masks is not None
        target = self._masks.to(self.device, non_blocking=True)
        logits = self.net(images)
        l_focal = focal_loss(
            logits, target, self.config.focal_gamma, self.config.focal_alpha
        )
        l_dice = dice_loss(logits, target)
        return {
            "loss": l_focal + self.config.dice_weight * l_dice,
            "L_focal": l_focal,
            "L_dice": l_dice,
        }

    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            logits = self.net(images)
        return F.softmax(logits.float(), dim=1)[:, 1]

    def release(self) -> None:
        self.net = self._build_net().to(self.device)
        self._masks = None
        super().release()


class _MaskCapture:
    """Stashes the synthesised masks so the shared training loop stays generic."""

    def __init__(self, loader, detector: TextAD) -> None:
        self._loader = loader
        self._detector = detector

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        for images, masks in self._loader:
            self._detector._masks = masks
            yield images
