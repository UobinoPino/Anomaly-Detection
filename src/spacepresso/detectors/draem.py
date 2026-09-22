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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from spacepresso.core.records import ImageRecord
from spacepresso.data.synthesis import draem_corruption
from spacepresso.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from spacepresso.detectors.base import Detector
from spacepresso.detectors.training import train_loop
from spacepresso.runner.config import DetectorConfig, RuntimeConfig

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
# U-Net
# ─────────────────────────────────────────────────────────────────────────────
def _conv_block(c_in: int, c_out: int, max_groups: int = 8) -> nn.Sequential:
    # GroupNorm rather than BatchNorm: batches are small (8) and per-class, so
    # batch statistics are noisy and shift between classes.
    #
    # The group count is the largest divisor of c_out up to max_groups. Fixing
    # it at 8 made the whole network fail to construct for any base width not
    # a multiple of 8, which is a needless constraint on a tunable.
    groups = math.gcd(c_out, max_groups) or 1
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=groups, num_channels=c_out),
        nn.ReLU(inplace=True),
    )


class _DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(_conv_block(c_in, c_out), _conv_block(c_out, c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(c_in, c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(c_in + c_out, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """Six-level U-Net, used for both roles: 3→3 and 6→2 channels."""

    def __init__(self, in_channels: int, out_channels: int, base: int = 32) -> None:
        super().__init__()
        widths = [base * 2**i for i in range(6)]
        self.inc = _DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            _Down(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
        )
        self.ups = nn.ModuleList(
            _Up(widths[i + 1], widths[i]) for i in reversed(range(len(widths) - 1))
        )
        self.outc = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        out = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            out = up(out, skip)
        return self.outc(out)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────
def ssim_loss(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5):
    """1 − SSIM, computed in fp32.

    The stability constants are 1e-4 and 9e-4; in fp16 those are close to the
    subnormal range and the loss goes to NaN within a few hundred steps. The
    caller keeps this out of autocast.
    """
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1.0, neginf=0.0)

    channels = x.shape[1]
    half = window // 2
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - half
    gauss = torch.exp(-(coords**2) / (2 * sigma * sigma))
    gauss = gauss / (gauss.sum() + 1e-8)
    kernel = (
        (gauss[:, None] * gauss[None, :])
        .clamp_min(1e-8)
        .expand(channels, 1, window, window)
        .contiguous()
    )

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=half, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    var_x = (blur(x * x) - mu_x2).clamp(min=1e-8)
    var_y = (blur(y * y) - mu_y2).clamp(min=1e-8)
    cov = (blur(x * y) - mu_xy).clamp(-1e6, 1e6)

    c1, c2 = 1e-4, 9e-4
    numerator = (2 * mu_xy + c1) * (2 * cov + c2)
    denominator = ((mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)).clamp_min(1e-8)
    return 1.0 - torch.nan_to_num((numerator / denominator).mean(), nan=1.0)


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.5
) -> torch.Tensor:
    """Focal loss over the two-class segmentation head.

    Defects cover a small fraction of the image, so plain cross-entropy is
    dominated by easy background pixels. The ``(1 - p_t)^gamma`` factor
    down-weights those; ``alpha`` additionally reweights the positive class.
    """
    log_probs = F.log_softmax(logits, dim=1)
    target = target.long()
    log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = log_p_t.exp()
    weight = torch.where(
        target == 1, torch.full_like(p_t, alpha), torch.full_like(p_t, 1.0 - alpha)
    )
    return (-weight * (1.0 - p_t) ** gamma * log_p_t).mean()


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
