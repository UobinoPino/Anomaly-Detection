"""DRAEM — reconstruct, then discriminate.

Zavrtanik et al., *DRAEM: A discriminatively trained reconstruction embedding
for surface anomaly detection* (ICCV 2021).

Two U-Nets trained together on synthetically corrupted normal images:

* the **reconstructive** net is shown a corrupted image and asked for the
  clean one, so it learns to undo defects it has never seen in reality;
* the **discriminative** net is shown the corrupted image *and* the
  reconstruction, and segments where they disagree.

The second net is the one that scores at test time. Feeding it the pair rather
than the residual is what makes DRAEM robust to reconstruction error that is
not a defect — the net learns which disagreements matter.

Unlike every other detector here, DRAEM's output is already a probability in
[0, 1], so it needs no score calibration of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.records import ImageRecord
from spacepresso.data.synthesis import draem_corruption
from spacepresso.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from spacepresso.detectors.base import Detector
from spacepresso.detectors.blocks import UNet, focal_loss, ssim_loss
from spacepresso.detectors.training import train_loop

__all__ = ["DRAEM", "DRAEMConfig"]


@dataclass
class DRAEMConfig(DetectorConfig):
    base_channels: int = 32
    total_iters: int | None = 2500
    epochs: int = 200
    train_batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0

    ssim_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5

    def slug_parts(self) -> list[str]:
        parts = [
            f"in{self.input_size}",
            f"b{self.base_channels}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
            f"bs{self.train_batch_size}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class _CorruptionDataset(Dataset):
    """Yields ``(corrupted, clean, mask)`` with a fresh corruption each call.

    The RNG is seeded per ``(worker, index, epoch-ish)`` rather than held in a
    module global, so corruption is reproducible from the run seed while still
    differing between every sample and every worker.
    """

    def __init__(self, records: Sequence[ImageRecord], input_size: int, seed: int):
        self.records = list(records)
        self.input_size = input_size
        self.seed = seed
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def _normalise(self, image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            ((image - self.mean) / self.std).transpose(2, 0, 1).copy()
        )

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as handle:
            image = handle.convert("RGB").resize(
                (self.input_size, self.input_size), Image.BILINEAR
            )
        clean = np.asarray(image, dtype=np.float32) / 255.0

        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng(
            (self.seed, worker_id, index, int(torch.randint(0, 2**31, (1,)).item()))
        )
        corrupted, mask = draem_corruption(clean, rng)

        return (
            self._normalise(corrupted),
            self._normalise(clean),
            torch.from_numpy(mask),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────
class DRAEM(Detector[DRAEMConfig]):
    name: ClassVar[str] = "draem"
    config_type: ClassVar[type[DetectorConfig]] = DRAEMConfig

    def __init__(self, config: DRAEMConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.reconstructor = UNet(3, 3, config.base_channels).to(self.device)
        self.discriminator = UNet(6, 2, config.base_channels).to(self.device)
        self._batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        dataset = _CorruptionDataset(
            train_good, self.config.input_size, self.config.seed
        )
        loader = self.make_loader(
            train_good,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
            dataset=dataset,
        )
        train_loop(
            [self.reconstructor, self.discriminator],
            _BatchCapture(loader, self),
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="draem",
        )

    def _loss(self, corrupted: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self._batch is not None
        _, clean, mask = self._batch
        clean = clean.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        reconstruction = self.reconstructor(corrupted)
        logits = self.discriminator(torch.cat([corrupted, reconstruction], dim=1))

        l2 = F.mse_loss(reconstruction, clean)
        with torch.amp.autocast("cuda", enabled=False):
            structural = ssim_loss(reconstruction, clean)
        segmentation = focal_loss(
            logits, mask, self.config.focal_gamma, self.config.focal_alpha
        )

        return {
            "loss": l2 + self.config.ssim_weight * structural + segmentation,
            "L_l2": l2,
            "L_ssim": structural,
            "L_seg": segmentation,
        }

    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            reconstruction = self.reconstructor(images)
            logits = self.discriminator(torch.cat([images, reconstruction], dim=1))
        return F.softmax(logits.float(), dim=1)[:, 1]

    def release(self) -> None:
        self.reconstructor = UNet(3, 3, self.config.base_channels).to(self.device)
        self.discriminator = UNet(6, 2, self.config.base_channels).to(self.device)
        self._batch = None
        super().release()


class _BatchCapture:
    """Adapts the three-tensor training batch to the shared training loop.

    ``train_loop`` hands the loss function the image tensor; DRAEM's loss also
    needs the clean target and the mask. Stashing the full tuple on the
    detector keeps the shared loop's contract simple rather than widening it
    for one detector's benefit.
    """

    def __init__(self, loader, detector: DRAEM) -> None:
        self._loader = loader
        self._detector = detector

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        for batch in self._loader:
            self._detector._batch = batch
            yield batch
