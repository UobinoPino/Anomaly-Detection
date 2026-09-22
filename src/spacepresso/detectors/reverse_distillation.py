"""Reverse Distillation — a student that decodes the teacher's features back.

Deng & Li, *Anomaly Detection via Reverse Distillation from One-Class
Embedding* (CVPR 2022).

The usual student-teacher setup gives both networks the same input, so a
capable student can learn to imitate the teacher on anomalies too. Reverse
distillation removes that shortcut: the student never sees the image. It sees
only a bottlenecked embedding of the teacher's own features and must
reconstruct them from it. On normal data the bottleneck is sufficient; on an
anomaly it is not, and the cosine similarity between teacher and student
collapses exactly where the anomaly is.

Two students, matched to the teacher's shape:

* **ResNet** — the paper's OCBE (one-class bottleneck embedding) fusing layers
  1–3 into a single H/32 tensor, plus a mirrored decoder.
* **ViT** — a single-scale convolutional bottleneck, since transformer blocks
  all share one spatial grid and there is no pyramid to fuse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from spacepresso.backbones import (
    build_backbone,
    channels_for,
    family_of,
    short_tag,
    validate_input_size,
)
from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.detectors.training import train_loop

__all__ = ["ReverseDistillation", "ReverseDistillationConfig"]

WRN_BASE_WIDTH = 128


@dataclass
class ReverseDistillationConfig(DetectorConfig):
    teacher_backbone: str = "wide_resnet50_2"
    teacher_layers: tuple[int, ...] = (1, 2, 3)

    bottleneck_dim: int = 256
    #: "mul" multiplies the per-scale distance maps, "sum" adds them.
    #: Multiplying demands agreement across scales and is much sparser.
    amap_mode: str = "mul"

    total_iters: int | None = 2500
    epochs: int = 100
    train_batch_size: int = 8
    lr: float = 5e-4
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.teacher_layers = tuple(self.teacher_layers)
        if self.amap_mode not in ("mul", "sum"):
            raise ValueError(
                f"amap_mode must be 'mul' or 'sum', got {self.amap_mode!r}"
            )

        family = family_of(self.teacher_backbone)
        if family == "dinov3_convnext":
            raise ValueError(
                "Reverse Distillation has no ConvNeXt student; use a ViT or "
                "wide_resnet50_2 teacher."
            )
        if family != "resnet":
            validate_input_size(self.teacher_backbone, self.input_size)
            if len(self.teacher_layers) != 1:
                raise ValueError(
                    "the ViT student is single-scale: pass exactly one "
                    f"teacher layer, got {list(self.teacher_layers)}"
                )
        elif self.teacher_backbone != "wide_resnet50_2":
            raise ValueError(
                "the ResNet student is sized for wide_resnet50_2's 256/512/1024 "
                f"channels; {self.teacher_backbone} would need its own decoder."
            )

    def slug_parts(self) -> list[str]:
        layers = "_".join(str(layer) for layer in self.teacher_layers)
        parts = [
            short_tag(self.teacher_backbone),
            f"L{layers}",
            f"in{self.input_size}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
            f"bs{self.train_batch_size}",
            f"lr{self.lr:.0e}",
            self.amap_mode,
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────
def _conv1x1(c_in: int, c_out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)


def _conv3x3(c_in: int, c_out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)


def _deconv2x2(c_in: int, c_out: int, stride: int = 2) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(c_in, c_out, 2, stride=stride, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = WRN_BASE_WIDTH,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1, self.bn1 = _conv1x1(inplanes, width), nn.BatchNorm2d(width)
        self.conv2, self.bn2 = _conv3x3(width, width, stride), nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class _DeBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: nn.Module | None = None,
        base_width: int = WRN_BASE_WIDTH,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1, self.bn1 = _conv1x1(inplanes, width), nn.BatchNorm2d(width)
        self.conv2 = (
            _deconv2x2(width, width, 2) if stride == 2 else _conv3x3(width, width)
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.upsample is None else self.upsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNetStudent(nn.Module):
    """OCBE fusion of layers 1–3, then a mirrored decoder back to each scale."""

    def __init__(self, base_width: int = WRN_BASE_WIDTH) -> None:
        super().__init__()
        self.l1_down = nn.Sequential(
            _conv3x3(256, 512, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            _conv3x3(512, 1024, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.l2_down = nn.Sequential(
            _conv3x3(512, 1024, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        downsample = nn.Sequential(_conv1x1(3072, 2048, stride=2), nn.BatchNorm2d(2048))
        self.fuse = nn.Sequential(
            _Bottleneck(
                3072, 512, stride=2, downsample=downsample, base_width=base_width
            ),
            _Bottleneck(2048, 512, base_width=base_width),
            _Bottleneck(2048, 512, base_width=base_width),
        )
        self.up1 = self._layer(2048, 256, 3, base_width)
        self.up2 = self._layer(1024, 128, 4, base_width)
        self.up3 = self._layer(512, 64, 6, base_width)

    @staticmethod
    def _layer(
        inplanes: int, planes: int, blocks: int, base_width: int
    ) -> nn.Sequential:
        out_channels = planes * _DeBottleneck.expansion
        upsample = nn.Sequential(
            _deconv2x2(inplanes, out_channels, stride=2), nn.BatchNorm2d(out_channels)
        )
        layers: list[nn.Module] = [
            _DeBottleneck(inplanes, planes, 2, upsample, base_width)
        ]
        layers += [
            _DeBottleneck(out_channels, planes, base_width=base_width)
            for _ in range(1, blocks)
        ]
        return nn.Sequential(*layers)

    def forward(self, teacher: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        fused = self.fuse(
            torch.cat(
                [self.l1_down(teacher[1]), self.l2_down(teacher[2]), teacher[3]], dim=1
            )
        )
        f3 = self.up1(fused)
        f2 = self.up2(f3)
        f1 = self.up3(f2)
        return {1: f1, 2: f2, 3: f3}


class ViTStudent(nn.Module):
    """Single-scale convolutional bottleneck for transformer teachers."""

    def __init__(self, channels: int, bottleneck_dim: int = 256) -> None:
        super().__init__()
        width = bottleneck_dim
        self.encoder = nn.Sequential(
            _conv1x1(channels, width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            _conv3x3(width, width, 2),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            _conv3x3(width, width, 2),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv3x3(width, width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv3x3(width, width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            _conv1x1(width, channels),
        )

    def forward(self, teacher: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        ((layer, features),) = teacher.items()
        decoded = self.decoder(self.encoder(features))
        if decoded.shape[-2:] != features.shape[-2:]:
            decoded = F.interpolate(
                decoded, size=features.shape[-2:], mode="bilinear", align_corners=False
            )
        return {layer: decoded}


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────
class ReverseDistillation(Detector[ReverseDistillationConfig]):
    name: ClassVar[str] = "reverse_distillation"
    config_type: ClassVar[type[DetectorConfig]] = ReverseDistillationConfig

    def __init__(
        self, config: ReverseDistillationConfig, runtime: RuntimeConfig
    ) -> None:
        super().__init__(config, runtime)
        self.teacher = build_backbone(config.teacher_backbone, device=self.device)
        self.student = self._build_student().to(self.device)

    def _build_student(self) -> nn.Module:
        if family_of(self.config.teacher_backbone) == "resnet":
            return ResNetStudent()
        channels = channels_for(
            self.config.teacher_backbone, self.config.teacher_layers[0]
        )
        return ViTStudent(channels, self.config.bottleneck_dim)

    def _teacher_features(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        maps = self.teacher(images, layers=self.config.teacher_layers)
        # Cloned out of the backbone's inference_mode so the student can
        # back-propagate through the comparison.
        return {
            layer: maps[layer].float().clone() for layer in self.config.teacher_layers
        }

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        loader = self.make_loader(
            train_good,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_loop(
            [self.student],
            loader,
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="reverse-distillation",
        )

    def _loss(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher = self._teacher_features(images)
        student = self.student(teacher)
        total = torch.zeros((), device=images.device)
        for layer, features in teacher.items():
            cosine = F.cosine_similarity(features, student[layer], dim=1, eps=1e-8)
            total = total + (1.0 - cosine).mean()
        return total

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            teacher = self._teacher_features(images)
            student = self.student(teacher)

        size = self.config.input_size
        multiply = self.config.amap_mode == "mul"
        result: torch.Tensor | None = None

        for layer in sorted(teacher):
            cosine = F.cosine_similarity(
                teacher[layer].float(), student[layer].float(), dim=1, eps=1e-8
            )
            distance = F.interpolate(
                (1.0 - cosine).unsqueeze(1),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            if result is None:
                result = distance
            else:
                result = result * distance if multiply else result + distance

        assert result is not None
        return result

    def release(self) -> None:
        self.student = self._build_student().to(self.device)
        super().release()
