"""UniAD — transformer reconstruction with a neighbour-masked attention.

You et al., *A Unified Model for Multi-class Anomaly Detection* (NeurIPS 2022).

An encoder-decoder transformer is trained to reconstruct a frozen backbone's
feature map from itself. Left alone it would learn the identity, which
reconstructs anomalies perfectly and detects nothing. UniAD's fix is to forbid
each token from attending to itself and its immediate neighbours, so every
patch must be reconstructed from *distant* context. Normal patches are
predictable from their surroundings; anomalous ones are not.

Two further defences against the identity shortcut:

* a learned query embedding rather than the encoder output as the decoder's
  starting point, so information has to pass through the attention;
* feature jitter during training, which makes exact copying unprofitable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from spacepresso.backbones import build_backbone, short_tag, validate_input_size
from spacepresso.backbones.registry import channels_for, patch_size_of
from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.detectors.training import train_loop

__all__ = ["UniAD", "UniADConfig"]


@dataclass
class UniADConfig(DetectorConfig):
    backbone: str = "dinov2_vitb14_reg"
    feature_layer: int = 9

    model_dim: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    #: Chebyshev radius of the forbidden attention neighbourhood. 0 masks only
    #: the diagonal, which is the minimum needed to break the identity map.
    #: 7 is the paper's value and suits the 28x28 grid a patch-14 ViT gives at
    #: 392px; on smaller grids it must come down (see :func:`neighbour_mask`).
    neighbour_radius: int = 7
    jitter_sigma: float = 0.1

    total_iters: int | None = 2500
    epochs: int = 100
    train_batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    norm_stat_images: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_input_size(self.backbone, self.input_size)
        if self.model_dim % self.n_heads:
            raise ValueError(
                f"model_dim={self.model_dim} must be divisible by "
                f"n_heads={self.n_heads}"
            )

    def slug_parts(self) -> list[str]:
        parts = [
            short_tag(self.backbone),
            f"L{self.feature_layer}",
            f"in{self.input_size}",
            f"d{self.model_dim}",
            f"r{self.neighbour_radius}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


def neighbour_mask(height: int, width: int, radius: int) -> torch.Tensor:
    """``(L, L)`` boolean mask, ``True`` where attention is forbidden.

    Token ``(r1, c1)`` may not attend to ``(r2, c2)`` when their Chebyshev
    distance is at most ``radius``.

    Raises:
        ValueError: if the radius leaves any token with nothing to attend to.

            This is worth failing loudly on. Softmax over a fully-masked row
            is NaN, and that NaN propagates through the reconstruction, the
            residual and the score map, only surfacing much later as "Input
            contains NaN" from the metric — with nothing pointing back at the
            radius.

            The binding case is a *central* token, not a corner one: the
            token in the middle of the grid is the one whose farthest
            neighbour is nearest. So the usable radius is bounded by
            ``max(height // 2, width // 2) - 1``, which is 13 on
            the 28x28 grid a patch-14 ViT gives at 392px (the paper's radius
            of 7 fits comfortably) but only 1 on a 4x4 grid.
    """
    rows = torch.arange(height).repeat_interleave(width)
    cols = torch.arange(width).repeat(height)
    chebyshev = torch.maximum(
        (rows.unsqueeze(0) - rows.unsqueeze(1)).abs(),
        (cols.unsqueeze(0) - cols.unsqueeze(1)).abs(),
    )
    mask = chebyshev <= radius

    if bool(mask.all(dim=1).any()):
        largest = max(height // 2, width // 2) - 1
        raise ValueError(
            f"neighbour_radius={radius} leaves some tokens on the {height}x{width} "
            f"feature grid with nothing to attend to, which makes attention NaN. "
            f"Use at most {largest}, or raise input_size / pick a finer feature "
            f"layer to get a larger grid."
        )
    return mask


def _mlp(dim: int, ratio: float, dropout: float) -> nn.Sequential:
    hidden = int(dim * ratio)
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, dim),
        nn.Dropout(dropout),
    )


class EncoderBlock(nn.Module):
    """Pre-LN block with neighbour-masked self-attention."""

    def __init__(self, dim: int, heads: int, ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _mlp(dim, ratio, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        normed = self.norm1(x)
        attended, _ = self.attention(
            normed, normed, normed, attn_mask=attn_mask, need_weights=False
        )
        x = x + attended
        return x + self.mlp(self.norm2(x))


class DecoderBlock(nn.Module):
    """Cross-attend to the encoder, then masked self-attention, then FFN."""

    def __init__(self, dim: int, heads: int, ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = _mlp(dim, ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = self.norm_kv(memory)
        attended, _ = self.cross(self.norm_q(x), kv, kv, need_weights=False)
        x = x + attended
        normed = self.norm_self(x)
        attended, _ = self.self_attention(
            normed, normed, normed, attn_mask=attn_mask, need_weights=False
        )
        x = x + attended
        return x + self.mlp(self.norm_mlp(x))


class ReconstructionTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        grid: tuple[int, int],
        *,
        model_dim: int = 256,
        heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        radius: int = 7,
    ) -> None:
        super().__init__()
        height, width = grid
        self.grid = grid
        n_tokens = height * width

        self.input_proj = nn.Linear(feature_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, feature_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, model_dim))
        self.query_embed = nn.Parameter(torch.zeros(1, n_tokens, model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        self.encoder = nn.ModuleList(
            EncoderBlock(model_dim, heads, mlp_ratio, dropout)
            for _ in range(encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderBlock(model_dim, heads, mlp_ratio, dropout)
            for _ in range(decoder_layers)
        )
        self.norm_out = nn.LayerNorm(model_dim)
        self.register_buffer(
            "attn_mask", neighbour_mask(height, width, radius), persistent=False
        )

    def forward(self, features: torch.Tensor, jitter_sigma: float = 0.0):
        batch, channels, height, width = features.shape
        if (height, width) != self.grid:
            raise ValueError(
                f"feature grid {(height, width)} does not match the grid this "
                f"model was built for, {self.grid}"
            )

        tokens = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        if jitter_sigma > 0 and self.training:
            std = tokens.detach().std(dim=(0, 1), keepdim=True) + 1e-6
            tokens = tokens + torch.randn_like(tokens) * (jitter_sigma * std)

        encoded = self.input_proj(tokens) + self.pos_embed
        for block in self.encoder:
            encoded = block(encoded, attn_mask=self.attn_mask)

        decoded = self.query_embed.expand(batch, -1, -1).clone()
        for block in self.decoder:
            decoded = decoded + self.query_embed
            decoded = block(decoded, encoded, attn_mask=self.attn_mask)

        reconstruction = self.output_proj(self.norm_out(decoded))
        return reconstruction.reshape(batch, height, width, channels).permute(
            0, 3, 1, 2
        )


class UniAD(Detector[UniADConfig]):
    name: ClassVar[str] = "uniad"
    config_type: ClassVar[type[DetectorConfig]] = UniADConfig

    def __init__(self, config: UniADConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.backbone = build_backbone(config.backbone, device=self.device)
        self.feature_dim = channels_for(config.backbone, config.feature_layer)
        self.model = self._build_model().to(self.device)
        self.stats: dict[str, float] | None = None

    def _grid(self) -> tuple[int, int]:
        patch = patch_size_of(self.config.backbone)
        if patch is None:
            # ResNet layer k has stride 2^(k+1).
            stride = 2 ** (self.config.feature_layer + 1)
            side = self.config.input_size // stride
        else:
            side = self.config.input_size // patch
        return side, side

    def _build_model(self) -> ReconstructionTransformer:
        return ReconstructionTransformer(
            self.feature_dim,
            self._grid(),
            model_dim=self.config.model_dim,
            heads=self.config.n_heads,
            encoder_layers=self.config.n_encoder_layers,
            decoder_layers=self.config.n_decoder_layers,
            mlp_ratio=self.config.mlp_ratio,
            dropout=self.config.dropout,
            radius=self.config.neighbour_radius,
        )

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.backbone(images, layers=(self.config.feature_layer,))
        return maps[self.config.feature_layer].float().clone()

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        loader = self.make_loader(
            train_good,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_loop(
            [self.model],
            loader,
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="uniad",
        )
        self.stats = self._compute_norm_stats(train_good)

    def _loss(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._features(images)
        reconstruction = self.model(features, jitter_sigma=self.config.jitter_sigma)
        return torch.nn.functional.mse_loss(reconstruction, features)

    @torch.inference_mode()
    def _compute_norm_stats(self, records: Sequence[ImageRecord]) -> dict[str, float]:
        sample = list(records)
        if len(sample) > self.config.norm_stat_images:
            rng = np.random.default_rng(self.config.seed)
            chosen = rng.choice(
                len(sample), size=self.config.norm_stat_images, replace=False
            )
            sample = [sample[i] for i in chosen]

        values = [
            self._residual(images.to(self.device, non_blocking=True))
            .cpu()
            .numpy()
            .reshape(-1)
            for images, _masks, _indices in self.make_loader(sample)
        ]
        merged = np.concatenate(values)
        return {"mean": float(merged.mean()), "std": float(merged.std() + 1e-9)}

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _residual(self, images: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            features = self._features(images)
            reconstruction = self.model(features)
        return ((features.float() - reconstruction.float()) ** 2).mean(dim=1)

    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            raise RuntimeError("UniAD.score_batch() called before fit()")
        residual = self._residual(images)
        normalised = (residual - self.stats["mean"]) / self.stats["std"]
        return self.upsample(normalised)

    def release(self) -> None:
        self.model = self._build_model().to(self.device)
        self.stats = None
        super().release()
