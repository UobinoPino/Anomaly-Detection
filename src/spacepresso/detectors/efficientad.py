"""EfficientAD — student-teacher plus autoencoder.

Batzner et al., *EfficientAD: Accurate Visual Anomaly Detection at
Millisecond-Level Latencies* (WACV 2024).

Two complementary signals, averaged after z-scoring:

* **student vs teacher** — a small CNN trained to mimic a frozen teacher on
  normal images only. It fails on anomalies, because it never saw any. Trained
  with hard-pixel mining: only the worst 10% of pixels contribute, which stops
  the easy background from swamping the gradient.
* **student vs autoencoder** — the autoencoder reconstructs the teacher's
  features globally and blurs local detail; a second student head is trained
  to match *it*. Their disagreement catches large structural defects that the
  local student-teacher term misses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from spacepresso.backbones import build_backbone, short_tag, validate_input_size
from spacepresso.backbones.registry import family_of, n_layers
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.detectors.training import train_loop
from spacepresso.runner.config import DetectorConfig, RuntimeConfig

__all__ = ["EfficientAD", "EfficientADConfig"]

#: The student PDN downsamples twice, so everything is compared at H/4.
TARGET_STRIDE = 4


@dataclass
class EfficientADConfig(DetectorConfig):
    teacher_backbone: str = "resnet18"
    teacher_layer: int | None = None

    total_iters: int | None = 2500
    epochs: int = 200
    train_batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    hard_mining_pct: float = 0.10
    ae_base_channels: int = 32
    norm_stat_images: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.teacher_backbone != "resnet18":
            validate_input_size(self.teacher_backbone, self.input_size)
        if not 0 < self.hard_mining_pct <= 1:
            raise ValueError(
                f"hard_mining_pct must be in (0, 1], got {self.hard_mining_pct}"
            )

    def slug_parts(self) -> list[str]:
        parts = [short_tag(self.teacher_backbone)]
        if self.teacher_layer is not None:
            parts.append(f"L{self.teacher_layer}")
        parts.append(f"in{self.input_size}")
        parts.append(
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}"
        )
        parts.append(f"bs{self.train_batch_size}")
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────
def _conv_block(c_in: int, c_out: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class StudentPDN(nn.Module):
    """PDN-S: a four-layer CNN emitting ``2 * C`` channels at H/4.

    The doubled width is the trick that lets one network serve both terms:
    the first ``C`` channels chase the teacher, the second ``C`` chase the
    autoencoder.
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 64),
            nn.AvgPool2d(2, stride=2),
            _conv_block(64, 128),
            nn.AvgPool2d(2, stride=2),
            _conv_block(128, 256),
            _conv_block(256, 256),
            nn.Conv2d(256, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Autoencoder(nn.Module):
    """Bottlenecked encoder-decoder regressing the teacher's feature map."""

    def __init__(self, out_channels: int, base: int = 32) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        encoder: list[nn.Module] = []
        in_channels = 3
        for width in widths:
            encoder += [
                nn.Conv2d(in_channels, width, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
            in_channels = width
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_block(base * 8, base * 4),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_block(base * 4, base * 2),
        )
        self.head = nn.Conv2d(base * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(self.encoder(x)))


class Teacher(nn.Module):
    """Frozen backbone resampled to the student's H/4 grid.

    The resampling matters and is easy to get wrong: at a ViT-friendly input
    like 392 the autoencoder's strided convolutions land on 96 while a
    patch-14 ViT gives 98. Comparing those per-pixel without aligning first
    silently compares different spatial positions, so every tensor is pulled
    onto the student's grid before any arithmetic.
    """

    def __init__(self, backbone_name: str, layer: int | None) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        family = family_of(backbone_name)

        if layer is None:
            if family == "resnet":
                layer = 2
            elif family == "dinov3_convnext":
                layer = 0  # stride 4 natively — no resampling needed
            else:
                layer = max(n_layers(backbone_name) - 3, 0)
        self.layer = int(layer)
        self.out_channels = self.backbone.channels(self.layer)
        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x, layers=(self.layer,))[self.layer]
        height, width = x.shape[-2] // TARGET_STRIDE, x.shape[-1] // TARGET_STRIDE
        if features.shape[-2:] != (height, width):
            features = F.interpolate(
                features, size=(height, width), mode="bilinear", align_corners=False
            )
        return features


def _align_to(reference: torch.Tensor, *tensors: torch.Tensor) -> list[torch.Tensor]:
    size = reference.shape[-2:]
    return [
        t
        if t.shape[-2:] == size
        else F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        for t in tensors
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────
class EfficientAD(Detector[EfficientADConfig]):
    name: ClassVar[str] = "efficientad"
    config_type: ClassVar[type[DetectorConfig]] = EfficientADConfig

    def __init__(self, config: EfficientADConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.teacher = Teacher(config.teacher_backbone, config.teacher_layer).to(
            self.device
        )
        self.channels = self.teacher.out_channels
        self.student = StudentPDN(2 * self.channels).to(self.device)
        self.autoencoder = Autoencoder(
            self.channels, base=config.ae_base_channels
        ).to(self.device)
        self.stats: dict[str, float] | None = None

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        loader = self.make_loader(
            train_good,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.log.info("    teacher: %s, %d channels", self.config.teacher_backbone, self.channels)

        train_loop(
            [self.student, self.autoencoder],
            loader,
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="efficientad",
        )
        self.stats = self._compute_norm_stats(train_good)

    def _loss(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_features = self.teacher(images)
        student_out = self.student(images)
        ae_out = self.autoencoder(images)
        teacher_features, ae_out = _align_to(student_out, teacher_features, ae_out)

        to_teacher = student_out[:, : self.channels]
        to_ae = student_out[:, self.channels :]

        # Hard-pixel mining: back-propagate only the worst fraction of pixels.
        per_pixel = ((to_teacher - teacher_features) ** 2).mean(dim=1)
        k = max(int(self.config.hard_mining_pct * per_pixel.numel()), 1)
        l_student = torch.topk(per_pixel.reshape(-1), k, largest=True).values.mean()

        l_ae = ((ae_out - teacher_features) ** 2).mean()
        l_student_ae = ((to_ae - ae_out.detach()) ** 2).mean()

        return {
            "loss": l_student + l_ae + l_student_ae,
            "L_st": l_student,
            "L_ae": l_ae,
            "L_stae": l_student_ae,
        }

    @torch.inference_mode()
    def _compute_norm_stats(
        self, records: Sequence[ImageRecord]
    ) -> dict[str, float]:
        """Per-term mean and std over normal images.

        The two terms live on unrelated scales; without z-scoring, whichever
        happens to be numerically larger dominates the average and the other
        contributes nothing.
        """
        sample = list(records)
        if len(sample) > self.config.norm_stat_images:
            rng = np.random.default_rng(self.config.seed)
            chosen = rng.choice(
                len(sample), size=self.config.norm_stat_images, replace=False
            )
            sample = [sample[i] for i in chosen]

        student_values: list[np.ndarray] = []
        ae_values: list[np.ndarray] = []
        for images, _masks, _indices in self.make_loader(sample):
            images = images.to(self.device, non_blocking=True)
            student_map, ae_map = self._raw_maps(images)
            student_values.append(student_map.cpu().numpy().reshape(-1))
            ae_values.append(ae_map.cpu().numpy().reshape(-1))

        student_all = np.concatenate(student_values)
        ae_all = np.concatenate(ae_values)
        stats = {
            "st_mean": float(student_all.mean()),
            "st_std": float(student_all.std() + 1e-9),
            "ae_mean": float(ae_all.mean()),
            "ae_std": float(ae_all.std() + 1e-9),
        }
        self.log.info(
            "    norm stats  st=%.4g±%.4g  ae=%.4g±%.4g",
            stats["st_mean"],
            stats["st_std"],
            stats["ae_mean"],
            stats["ae_std"],
        )
        return stats

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _raw_maps(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.autocast():
            teacher_features = self.teacher(images)
            student_out = self.student(images)
            ae_out = self.autoencoder(images)
        teacher_features, ae_out = _align_to(student_out, teacher_features, ae_out)

        to_teacher = student_out[:, : self.channels].float()
        to_ae = student_out[:, self.channels :].float()
        student_map = ((to_teacher - teacher_features.float()) ** 2).mean(dim=1)
        ae_map = ((to_ae - ae_out.float()) ** 2).mean(dim=1)
        return student_map, ae_map

    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            raise RuntimeError("EfficientAD.score_batch() called before fit()")
        student_map, ae_map = self._raw_maps(images)
        combined = 0.5 * (
            (student_map - self.stats["st_mean"]) / self.stats["st_std"]
            + (ae_map - self.stats["ae_mean"]) / self.stats["ae_std"]
        )
        return self.upsample(combined)

    def release(self) -> None:
        self.student = StudentPDN(2 * self.channels).to(self.device)
        self.autoencoder = Autoencoder(
            self.channels, base=self.config.ae_base_channels
        ).to(self.device)
        self.stats = None
        super().release()
