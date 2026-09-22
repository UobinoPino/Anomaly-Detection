"""FastFlow — normalizing-flow density estimation on backbone features.

Yu et al., *FastFlow: Unsupervised Anomaly Detection and Localization via 2D
Normalizing Flows* (2021).

One flow per backbone scale learns an invertible map from normal features to a
standard Gaussian. A patch is anomalous when its negative log-likelihood is
high: the flow has only ever been asked to make *normal* features look
Gaussian, so anything else lands in the tails.

Per-scale NLLs are z-scored against statistics gathered on the training set
before being summed, because their magnitudes differ by orders of magnitude
across scales and an un-normalised sum is just the deepest scale.
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
from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.detectors.training import train_loop

__all__ = ["FastFlow", "FastFlowConfig"]


@dataclass
class FastFlowConfig(DetectorConfig):
    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (2, 3)

    n_blocks: int = 8
    hidden_ratio: float = 1.0
    clamp: float = 2.0

    total_iters: int | None = 2500
    epochs: int = 100
    train_batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    norm_stat_images: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.feature_layers = tuple(self.feature_layers)
        validate_input_size(self.backbone, self.input_size)

    def slug_parts(self) -> list[str]:
        layers = "_".join(str(layer) for layer in self.feature_layers)
        parts = [
            short_tag(self.backbone),
            f"L{layers}",
            f"in{self.input_size}",
            f"nb{self.n_blocks}",
            f"hr{self.hidden_ratio:g}",
            f"c{self.clamp:g}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
            f"bs{self.train_batch_size}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


# ─────────────────────────────────────────────────────────────────────────────
# Flow
# ─────────────────────────────────────────────────────────────────────────────
class _Subnet(nn.Module):
    """Predicts the coupling's ``(scale, shift)``.

    3x3 → 1x1 → 3x3 keeps the receptive field local, which is the point of
    FastFlow: a defect should perturb its own neighbourhood, not the whole
    map. The last convolution is zero-initialised so each coupling starts as
    the identity and training begins from an exactly invertible map.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    """Real-NVP affine coupling.

    Half the channels pass through untouched and predict an affine transform
    for the other half, which makes the Jacobian triangular and its
    log-determinant a plain channel sum. ``clamp * tanh(s / clamp)`` bounds
    the log-scale; without it the exponential overflows within a few hundred
    steps.
    """

    def __init__(
        self, channels: int, hidden_ratio: float = 1.0, clamp: float = 2.0
    ) -> None:
        super().__init__()
        self.split = channels // 2
        self.rest = channels - self.split
        self.clamp = clamp
        self.subnet = _Subnet(
            self.split, 2 * self.rest, max(int(channels * hidden_ratio), 16)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        identity, transformed = x[:, : self.split], x[:, self.split :]
        scale, shift = self.subnet(identity).chunk(2, dim=1)
        scale = self.clamp * torch.tanh(scale / self.clamp)
        transformed = transformed * torch.exp(scale) + shift
        return torch.cat([identity, transformed], dim=1), scale.sum(dim=1)


class ScaleFlow(nn.Module):
    """A stack of couplings for one feature scale.

    Between blocks the channels are reversed rather than permuted by a learned
    1x1 convolution. On feature maps this narrow a learned permutation adds
    training instability without improving likelihood.
    """

    def __init__(
        self,
        channels: int,
        n_blocks: int = 8,
        hidden_ratio: float = 1.0,
        clamp: float = 2.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            AffineCoupling(channels, hidden_ratio, clamp) for _ in range(n_blocks)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        log_det = torch.zeros(batch, height, width, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, block_log_det = block(x)
            log_det = log_det + block_log_det
            x = torch.flip(x, dims=[1])
        return x, log_det


class MultiScaleFlow(nn.Module):
    """One :class:`ScaleFlow` per backbone layer."""

    def __init__(
        self,
        channels_per_scale: dict[int, int],
        n_blocks: int = 8,
        hidden_ratio: float = 1.0,
        clamp: float = 2.0,
    ) -> None:
        super().__init__()
        self.scales = sorted(channels_per_scale)
        self.flows = nn.ModuleDict(
            {
                str(scale): ScaleFlow(
                    channels_per_scale[scale], n_blocks, hidden_ratio, clamp
                )
                for scale in self.scales
            }
        )

    def forward(
        self, features: dict[int, torch.Tensor]
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {scale: self.flows[str(scale)](features[scale]) for scale in self.scales}


def nll_per_pixel(z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under a standard Gaussian, per pixel.

    ``-log p(z) - log|det J| = 0.5 ||z||² - log_det``, dropping the constant
    ``0.5 C log(2π)``, which shifts every pixel equally and so affects neither
    the ranking nor the gradient.
    """
    return 0.5 * (z**2).sum(dim=1) - log_det


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────
class FastFlow(Detector[FastFlowConfig]):
    name: ClassVar[str] = "fastflow"
    config_type: ClassVar[type[DetectorConfig]] = FastFlowConfig

    def __init__(self, config: FastFlowConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.backbone = build_backbone(config.backbone, device=self.device)
        channels = {
            layer: self.backbone.channels(layer) for layer in config.feature_layers
        }
        self.flow = MultiScaleFlow(
            channels, config.n_blocks, config.hidden_ratio, config.clamp
        ).to(self.device)
        self.stats: dict[int, dict[str, float]] | None = None

    def _features(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        """Per-scale features, detached from the backbone's inference context.

        The backbone's ``forward`` runs under ``inference_mode``, and inference
        tensors cannot be saved for backward. Cloning here — outside that
        context, which ended when ``forward`` returned — yields ordinary
        tensors the flow can train through.
        """
        maps = self.backbone(images, layers=self.config.feature_layers)
        return {
            layer: maps[layer].float().clone() for layer in self.config.feature_layers
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
            [self.flow],
            loader,
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="fastflow",
        )
        self.stats = self._compute_norm_stats(train_good)

    def _loss(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._features(images)
        total = torch.zeros((), device=images.device)
        for _scale, (z, log_det) in self.flow(features).items():
            total = total + nll_per_pixel(z, log_det).mean()
        return total / max(len(self.flow.scales), 1)

    @torch.inference_mode()
    def _compute_norm_stats(
        self, records: Sequence[ImageRecord]
    ) -> dict[int, dict[str, float]]:
        sample = list(records)
        if len(sample) > self.config.norm_stat_images:
            rng = np.random.default_rng(self.config.seed)
            chosen = rng.choice(
                len(sample), size=self.config.norm_stat_images, replace=False
            )
            sample = [sample[i] for i in chosen]

        buckets: dict[int, list[np.ndarray]] = {s: [] for s in self.flow.scales}
        for images, _masks, _indices in self.make_loader(sample):
            images = images.to(self.device, non_blocking=True)
            with self.autocast():
                features = self._features(images)
            for scale, (z, log_det) in self.flow(features).items():
                buckets[scale].append(
                    nll_per_pixel(z, log_det).float().cpu().numpy().reshape(-1)
                )

        stats: dict[int, dict[str, float]] = {}
        for scale, values in buckets.items():
            merged = np.concatenate(values)
            stats[scale] = {
                "mean": float(merged.mean()),
                "std": float(merged.std() + 1e-9),
            }
            self.log.info(
                "    scale %d  NLL mean=%.4g std=%.4g",
                scale,
                stats[scale]["mean"],
                stats[scale]["std"],
            )
        return stats

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            raise RuntimeError("FastFlow.score_batch() called before fit()")

        with self.autocast():
            features = self._features(images)

        size = self.config.input_size
        total: torch.Tensor | None = None
        for scale, (z, log_det) in self.flow(features).items():
            nll = nll_per_pixel(z, log_det).float()
            nll = (nll - self.stats[scale]["mean"]) / self.stats[scale]["std"]
            upsampled = F.interpolate(
                nll.unsqueeze(1),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            total = upsampled if total is None else total + upsampled

        assert total is not None
        return total / max(len(self.flow.scales), 1)

    def release(self) -> None:
        channels = {
            layer: self.backbone.channels(layer) for layer in self.config.feature_layers
        }
        self.flow = MultiScaleFlow(
            channels, self.config.n_blocks, self.config.hidden_ratio, self.config.clamp
        ).to(self.device)
        self.stats = None
        super().release()
