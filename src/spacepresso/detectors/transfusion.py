"""TransFusion — transparency-conditioned diffusion for anomaly localisation.

Fucka et al., *TransFusion: A Transparency-Based Diffusion Model for Anomaly
Detection* (ECCV 2024).

The forward process is not Gaussian noise but *transparency*: at timestep
``t`` the anomalous region is blended into the normal image at strength
``t/T``, so ``x_t = (1 - a(t)·M)·x_0 + a(t)·M·A``. A time-conditioned U-Net
sees ``x_t`` and predicts both the anomaly mask at that transparency and the
underlying normal image.

Conditioning on transparency is what makes it work on faint defects: a model
trained only on opaque anomalies never learns the low-contrast end, and most
real defects live there. At inference the network is run at several
transparency levels and the predicted masks are averaged.

Synthetic anomalies come from *other classes'* training images rather than an
external texture dataset — domain-adjacent, and with no download to manage.
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

from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.records import TRAIN_GOOD, ImageRecord, scan_dataset
from spacepresso.data.synthesis import sample_perlin_mask
from spacepresso.detectors.base import Detector
from spacepresso.detectors.blocks import focal_loss
from spacepresso.detectors.training import train_loop

__all__ = ["TransFusion", "TransFusionConfig"]


@dataclass
class TransFusionConfig(DetectorConfig):
    base_channels: int = 64
    time_dim: int = 256

    #: Transparency levels to average over at inference. More levels cost
    #: linearly more time; four spans faint to opaque adequately.
    inference_steps: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    blob_probability: float = 0.5
    blend_range: tuple[float, float] = (0.4, 0.95)
    texture_pool_size: int = 256

    mask_weight: float = 1.0
    reconstruction_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5

    total_iters: int | None = 5000
    epochs: int = 200
    train_batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.inference_steps = tuple(self.inference_steps)
        if not self.inference_steps:
            raise ValueError("inference_steps must not be empty")

    def slug_parts(self) -> list[str]:
        parts = [
            f"in{self.input_size}",
            f"b{self.base_channels}",
            f"s{len(self.inference_steps)}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
            f"bs{self.train_batch_size}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    """The standard transformer/diffusion positional encoding of a scalar time."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        angles = t[:, None].float() * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


def _groups(channels: int, maximum: int = 8) -> int:
    return math.gcd(channels, maximum) or 1


class ResBlock(nn.Module):
    """Residual block with the time embedding injected as a per-channel shift."""

    def __init__(self, c_in: int, c_out: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time = nn.Linear(time_dim, c_out)
        self.norm2 = nn.GroupNorm(_groups(c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TransFusionUNet(nn.Module):
    """Predicts the anomaly mask and the underlying normal image.

    Output is 4 channels: one mask logit and three reconstruction channels.
    Jointly predicting both is the point — a mask head alone has no incentive
    to understand what the image *should* look like underneath.
    """

    def __init__(self, base: int = 64, time_dim: int = 256) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1)

        self.encoder = nn.ModuleList(
            nn.ModuleList(
                [
                    ResBlock(
                        widths[max(i - 1, 0)] if i else widths[0], widths[i], time_dim
                    ),
                    ResBlock(widths[i], widths[i], time_dim),
                ]
            )
            for i in range(3)
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(widths[i], widths[i], 3, stride=2, padding=1) for i in range(3)
        )
        self.middle = nn.ModuleList(
            [
                ResBlock(widths[2], widths[3], time_dim),
                ResBlock(widths[3], widths[3], time_dim),
            ]
        )
        self.upsample = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i + 1], 2, stride=2)
            for i in reversed(range(3))
        )
        self.decoder = nn.ModuleList(
            nn.ModuleList(
                [
                    ResBlock(widths[i + 1] + widths[i], widths[i], time_dim),
                    ResBlock(widths[i], widths[i], time_dim),
                ]
            )
            for i in reversed(range(3))
        )
        self.head_norm = nn.GroupNorm(_groups(widths[0]), widths[0])
        self.head = nn.Conv2d(widths[0], 4, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.stem(x)

        skips: list[torch.Tensor] = []
        for blocks, down in zip(self.encoder, self.downsample, strict=True):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        for block in self.middle:
            h = block(h, t_emb)

        for up, blocks, skip in zip(
            self.upsample, self.decoder, reversed(skips), strict=True
        ):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        return self.head(F.silu(self.head_norm(h)))


# ─────────────────────────────────────────────────────────────────────────────
# Texture pool
# ─────────────────────────────────────────────────────────────────────────────
class TexturePool:
    """Images from *other* classes, used as anomaly texture.

    Domain-adjacent by construction: a scratch on a gear that looks like
    pistachio shell is a more useful training signal than one that looks like
    a stock photo, and it avoids depending on an external texture dataset.
    """

    def __init__(
        self, paths: Sequence, size: int, pool_size: int = 256, seed: int = 0
    ) -> None:
        self.size = size
        rng = np.random.default_rng(seed)
        chosen = (
            rng.choice(len(paths), size=min(pool_size, len(paths)), replace=False)
            if len(paths)
            else np.asarray([], dtype=int)
        )
        self.images: list[np.ndarray] = []
        for index in chosen:
            try:
                with Image.open(paths[int(index)]) as handle:
                    image = handle.convert("RGB").resize((size, size), Image.BILINEAR)
                self.images.append(
                    np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
                )
            except OSError:
                continue

    def __len__(self) -> int:
        return len(self.images)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """``(3, H, W)`` texture in [0, 1]."""
        if not self.images:
            return rng.random((3, self.size, self.size)).astype(np.float32)
        texture = self.images[int(rng.integers(0, len(self.images)))].copy()
        for channel in range(3):
            texture[channel] = np.clip(
                texture[channel] * rng.uniform(0.7, 1.3), 0.0, 1.0
            )
        if rng.random() < 0.5:
            texture = texture[:, :, ::-1].copy()
        if rng.random() < 0.5:
            texture = texture[:, ::-1, :].copy()
        return texture


def _blob_mask(height: int, width: int, rng: np.random.Generator, n_blobs: int):
    """A few soft elliptical blobs — the focal counterpart to a Perlin mask.

    Perlin masks are broad and diffuse, which suits stains and
    discolouration; blobs are compact, which suits pits and scratch ends.
    Mixing both in training exposes the model to each.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    ys, xs = np.mgrid[0:height, 0:width]
    for _ in range(n_blobs):
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        ry = rng.uniform(height * 0.03, height * 0.15)
        rx = rng.uniform(width * 0.03, width * 0.15)
        mask[((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1.0] = 1.0
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────
class TransFusion(Detector[TransFusionConfig]):
    name: ClassVar[str] = "transfusion"
    config_type: ClassVar[type[DetectorConfig]] = TransFusionConfig

    def __init__(self, config: TransFusionConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.net = TransFusionUNet(config.base_channels, config.time_dim).to(
            self.device
        )
        self.textures: TexturePool | None = None
        self._rng = np.random.default_rng(config.seed)

    def _build_texture_pool(self, cls: str) -> TexturePool:
        """Training images of every class except this one."""
        records = scan_dataset(self.runtime.data_root)
        paths = [r.path for r in records if r.split == TRAIN_GOOD and r.cls != cls]
        pool = TexturePool(
            paths,
            self.config.input_size,
            self.config.texture_pool_size,
            self.config.seed,
        )
        self.log.info(
            "    texture pool: %d images from %d candidates in other classes",
            len(pool),
            len(paths),
        )
        return pool

    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        self.textures = self._build_texture_pool(train_good[0].cls)
        loader = self.make_loader(
            train_good,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_loop(
            [self.net],
            loader,
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="transfusion",
        )

    def _synthesise(
        self, normal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.textures is not None
        batch, _, height, width = normal.shape
        rng = self._rng

        masks = np.zeros((batch, 1, height, width), dtype=np.float32)
        textures = np.zeros((batch, 3, height, width), dtype=np.float32)
        for index in range(batch):
            masks[index, 0] = (
                _blob_mask(height, width, rng, int(rng.integers(1, 4)))
                if rng.random() < self.config.blob_probability
                else sample_perlin_mask(height, width, rng)
            )
            textures[index] = self.textures.sample(rng)

        mask = torch.from_numpy(masks).to(normal.device)
        texture = torch.from_numpy(textures).to(normal.device)
        low, high = self.config.blend_range
        blend = torch.from_numpy(
            rng.uniform(low, high, size=(batch, 1, 1, 1)).astype(np.float32)
        ).to(normal.device)
        return normal * (1.0 - blend * mask) + texture * (blend * mask), mask, blend

    def _loss(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        anomalous, mask, _blend = self._synthesise(images)

        # Sample a transparency level per image and interpolate towards it,
        # so the network sees the full range from invisible to fully blended.
        t = torch.rand(images.shape[0], device=images.device)
        blended = images + t[:, None, None, None] * (anomalous - images)

        output = self.net(blended, t)
        mask_logits = output[:, :1]
        reconstruction = torch.sigmoid(output[:, 1:])

        # The mask target scales with transparency: at t=0 there is nothing
        # to find, and asking for a full mask there would teach the model to
        # hallucinate defects in clean images.
        target = (mask.squeeze(1) * (t[:, None, None] > 0.05)).long()
        l_mask = focal_loss(
            torch.cat([-mask_logits, mask_logits], dim=1),
            target,
            self.config.focal_gamma,
            self.config.focal_alpha,
        )
        l_reconstruction = F.l1_loss(reconstruction, images)

        loss = (
            self.config.mask_weight * l_mask
            + self.config.reconstruction_weight * l_reconstruction
        )
        return {"loss": loss, "L_mask": l_mask, "L_rec": l_reconstruction}

    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        total: torch.Tensor | None = None
        for step in self.config.inference_steps:
            t = torch.full((images.shape[0],), float(step), device=images.device)
            with self.autocast():
                output = self.net(images, t)
            probability = torch.sigmoid(output[:, 0].float())
            total = probability if total is None else total + probability
        assert total is not None
        return total / len(self.config.inference_steps)

    def release(self) -> None:
        self.net = TransFusionUNet(self.config.base_channels, self.config.time_dim).to(
            self.device
        )
        self.textures = None
        super().release()
