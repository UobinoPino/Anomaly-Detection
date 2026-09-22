"""GLASS — global and local anomaly synthesis in feature space.

Chen et al., *A Unified Anomaly Synthesis Strategy with Gradient Ascent for
Industrial Anomaly Detection and Localization* (ECCV 2024).

A per-patch discriminator is trained to separate normal features from
anomalous ones, using two complementary sources of synthetic anomalies:

* **local** — a Perlin-masked corruption painted onto the image, which gives
  the discriminator spatially-localised positives with known masks;
* **global** — an L∞-bounded PGD ascent on the discriminator's own logit,
  starting from real normal features. This manufactures the *hardest*
  positives available at each moment: features that are barely off-manifold
  but that the current discriminator already leans towards calling anomalous.

The global term is what stops the discriminator from settling for the easy
decision boundary the local corruptions alone would allow. It is warmed up,
because attacking a randomly-initialised discriminator produces noise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from spacepresso.backbones import (
    build_backbone,
    patchify_and_combine,
    resolve_target_layer,
    short_tag,
    validate_input_size,
)
from spacepresso.core.records import ImageRecord
from spacepresso.data.synthesis import perlin_batch_torch
from spacepresso.detectors.base import Detector
from spacepresso.detectors.training import train_loop
from spacepresso.runner.config import DetectorConfig, RuntimeConfig

__all__ = ["GLASS", "GLASSConfig"]


@dataclass
class GLASSConfig(DetectorConfig):
    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (2, 3)
    target_layer: int | None = None
    patch_size: int = 3

    discriminator_hidden: int = 1024
    discriminator_layers: int = 2
    dropout: float = 0.0

    lambda_local: float = 1.0
    lambda_global: float = 1.0
    warmup_iters: int = 200
    attack_epsilon: float = 0.05
    attack_steps: int = 4
    attack_init_sigma: float = 0.015
    attack_subsample: int = 4096

    synth_intensity: tuple[float, float] = (0.3, 1.0)

    total_iters: int | None = 2500
    epochs: int = 100
    train_batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.feature_layers = tuple(self.feature_layers)
        validate_input_size(self.backbone, self.input_size)
        self.target_layer = resolve_target_layer(
            self.backbone, self.feature_layers, self.target_layer
        )

    def slug_parts(self) -> list[str]:
        layers = "_".join(str(layer) for layer in self.feature_layers)
        parts = [
            short_tag(self.backbone),
            f"L{layers}",
            f"in{self.input_size}",
            f"ll{self.lambda_local:g}",
            f"lg{self.lambda_global:g}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


class Adapter(nn.Module):
    """A near-identity linear map of the frozen features.

    Same width in and out — the adapter's job is to rotate and rescale the
    feature space, not to compress it. Initialised at identity plus a little
    noise so the first iterations pass the backbone's features through almost
    unchanged and the discriminator sees a stable target while it warms up.
    """

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = self.dropout(self.activation(self.linear(flat)))
        return out.reshape(*shape[:-1], -1)


class Discriminator(nn.Module):
    """Per-patch binary classifier.

    BatchNorm across the ``B*P`` patch batch is load-bearing: without it the
    network latches onto the one or two backbone channels with the largest
    norms and ignores the rest.
    """

    def __init__(
        self, dim: int, hidden: int = 1024, n_layers: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(width, hidden),
                nn.BatchNorm1d(hidden),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gradient_ascent_anomaly(
    features: torch.Tensor,
    discriminator: Discriminator,
    *,
    epsilon: float = 0.05,
    n_steps: int = 4,
    step_size: float | None = None,
    init_sigma: float = 0.015,
) -> torch.Tensor:
    """PGD ascent on the discriminator's anomaly logit.

    ``torch.autograd.grad`` is used rather than ``backward`` so the attack
    leaves no gradient on the discriminator's parameters and can run inside a
    training step without disturbing the optimiser.

    The caller puts the discriminator in ``eval()`` first, so its BatchNorm
    running statistics stay fixed while the attack probes it.
    """
    if step_size is None:
        step_size = 2.5 * epsilon / max(n_steps, 1)

    delta = (torch.randn_like(features) * init_sigma).clamp(-epsilon, epsilon)
    delta.requires_grad_(True)

    for _ in range(n_steps):
        logits = discriminator(features + delta)
        gradient = torch.autograd.grad(logits.sum(), delta)[0]
        with torch.no_grad():
            delta = (delta + step_size * gradient.sign()).clamp(-epsilon, epsilon)
        delta = delta.detach().requires_grad_(True)

    return (features + delta).detach()


class GLASS(Detector[GLASSConfig]):
    name: ClassVar[str] = "glass"
    config_type: ClassVar[type[DetectorConfig]] = GLASSConfig

    def __init__(self, config: GLASSConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.backbone = build_backbone(config.backbone, device=self.device)
        dim = self.backbone.total_channels(config.feature_layers)
        self.dim = dim
        self.adapter = Adapter(dim, config.dropout).to(self.device)
        self.discriminator = Discriminator(
            dim, config.discriminator_hidden, config.discriminator_layers, config.dropout
        ).to(self.device)
        self._step = 0
        self._generator = torch.Generator(device=self.device).manual_seed(config.seed)

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.backbone(images, layers=self.config.feature_layers)
        assert self.config.target_layer is not None
        combined = patchify_and_combine(
            maps, patch_size=self.config.patch_size, target_layer=self.config.target_layer
        )
        return combined.float().clone()

    def _synthesise_local(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paint a Perlin-masked corruption onto the batch, on-device."""
        batch, _, height, width = images.shape
        noise = perlin_batch_torch(
            batch, height, width, device=images.device, generator=self._generator
        )
        threshold = noise.flatten(1).quantile(0.8, dim=1).view(batch, 1, 1)
        mask = (noise > threshold).float()

        low, high = self.config.synth_intensity
        intensity = (
            torch.rand(batch, 1, 1, 1, device=images.device, generator=self._generator)
            * (high - low)
            + low
        )
        texture = torch.randn(
            batch, 3, 1, 1, device=images.device, generator=self._generator
        ).expand_as(images)
        blend = intensity * mask.unsqueeze(1)
        return (1 - blend) * images + blend * texture, mask

    @staticmethod
    def _mask_to_patches(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Downsample an image-resolution mask onto the feature grid."""
        pooled = F.adaptive_avg_pool2d(mask.unsqueeze(1), (height, width))
        return pooled.squeeze(1).reshape(mask.shape[0], height * width)

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        loader = self.make_loader(
            train_good,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._step = 0
        train_loop(
            [self.adapter, self.discriminator],
            loader,
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="glass",
        )

    def _loss(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        with torch.no_grad():
            normal_raw = self._features(images)
            corrupted, mask = self._synthesise_local(images)
            local_raw = self._features(corrupted)

        n_patches = normal_raw.shape[1]
        side = int(math.isqrt(n_patches))
        patch_mask = self._mask_to_patches(mask, side, side)

        normal = self.adapter(normal_raw).reshape(batch * n_patches, self.dim)
        local = self.adapter(local_raw).reshape(batch * n_patches, self.dim)

        l_normal = F.binary_cross_entropy_with_logits(
            self.discriminator(normal), torch.zeros(batch * n_patches, 1, device=self.device)
        )
        l_local = F.binary_cross_entropy_with_logits(
            self.discriminator(local), patch_mask.reshape(-1, 1).clamp(0, 1)
        )

        l_global = torch.zeros((), device=self.device)
        self._step += 1
        if self._step >= self.config.warmup_iters and self.config.lambda_global > 0:
            self.discriminator.eval()
            take = min(self.config.attack_subsample, normal.shape[0])
            chosen = torch.randperm(normal.shape[0], device=self.device)[:take]
            adversarial = gradient_ascent_anomaly(
                normal[chosen].detach(),
                self.discriminator,
                epsilon=self.config.attack_epsilon,
                n_steps=self.config.attack_steps,
                init_sigma=self.config.attack_init_sigma,
            )
            self.discriminator.train()
            l_global = F.binary_cross_entropy_with_logits(
                self.discriminator(adversarial),
                torch.ones(adversarial.shape[0], 1, device=self.device),
            )

        loss = (
            l_normal
            + self.config.lambda_local * l_local
            + self.config.lambda_global * l_global
        )
        return {"loss": loss, "L_n": l_normal, "L_l": l_local, "L_g": l_global}

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            features = self._features(images)
        batch, n_patches, _ = features.shape
        side = int(math.isqrt(n_patches))
        adapted = self.adapter(features).reshape(batch * n_patches, self.dim)
        logits = self.discriminator(adapted).float().reshape(batch, side, side)
        return self.upsample(torch.sigmoid(logits))

    def release(self) -> None:
        self.adapter = Adapter(self.dim, self.config.dropout).to(self.device)
        self.discriminator = Discriminator(
            self.dim,
            self.config.discriminator_hidden,
            self.config.discriminator_layers,
            self.config.dropout,
        ).to(self.device)
        self._step = 0
        super().release()
