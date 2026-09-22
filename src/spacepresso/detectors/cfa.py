"""CFA — Coupled-hypersphere Feature Adaptation.

Lee et al., *CFA: Coupled-hypersphere-based Feature Adaptation for
Target-Oriented Anomaly Localization* (IEEE Access 2022).

PatchCore scores against a frozen feature space; CFA learns a small residual
adaptation of that space first. A per-patch MLP is trained so that normal
patches fall *inside* a hypersphere of radius ``r`` around their nearest
memory entries (attraction) while the next-nearest entries are pushed outside
``r + alpha`` (repulsion). The result is a feature space where the
nearest-neighbour distance separates normal from anomalous more sharply than
the raw backbone does.

At test time the score is the mean squared distance to the ``k_test`` nearest
adapted memory entries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

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
from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.logging import now_hms
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.detectors.coreset import greedy_coreset
from spacepresso.detectors.training import train_loop

__all__ = ["CFA", "CFAConfig"]


@dataclass
class CFAConfig(DetectorConfig):
    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (2, 3)
    target_layer: int | None = None
    patch_size: int = 3

    coreset_frac: float = 0.05
    coreset_algorithm: str = "minibatch"
    coreset_batch: int = 64

    hidden_dim: int | None = None
    use_batchnorm: bool = True

    k_attract: int = 3
    k_repel: int = 3
    radius: float = 1e-5
    alpha: float = 1e-3
    k_test: int = 3

    total_iters: int | None = 2500
    epochs: int = 50
    train_batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 5e-4

    #: How often to refresh the cached adapted memory bank, in iterations.
    #: The descriptor barely moves between steps at this learning rate, so
    #: recomputing it every iteration is most of the training cost for no
    #: measurable benefit — the same staleness trick momentum encoders use.
    memory_refresh_every: int = 10
    loss_chunk: int = 8192
    score_chunk: int = 4096

    _excluded: ClassVar[frozenset[str]] = frozenset({"loss_chunk", "score_chunk"})

    def __post_init__(self) -> None:
        super().__post_init__()
        self.feature_layers = tuple(self.feature_layers)
        validate_input_size(self.backbone, self.input_size)
        self.target_layer = resolve_target_layer(
            self.backbone, self.feature_layers, self.target_layer
        )

    def fingerprint(self) -> dict[str, object]:
        return {
            key: value
            for key, value in super().fingerprint().items()
            if key not in self._excluded
        }

    def slug_parts(self) -> list[str]:
        layers = "_".join(str(layer) for layer in self.feature_layers)
        parts = [
            short_tag(self.backbone),
            f"L{layers}",
            f"in{self.input_size}",
            f"cs{int(self.coreset_frac * 100):02d}",
            f"ka{self.k_attract}",
            f"kr{self.k_repel}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


class Descriptor(nn.Module):
    """Per-patch residual MLP, zero-initialised so it begins as the identity.

    Starting from the identity matters: the contrastive loss is very noisy for
    the first few hundred iterations, and a randomly-initialised descriptor
    would have already destroyed the pretrained feature geometry by the time
    the loss became informative.
    """

    def __init__(
        self, dim: int, hidden_dim: int | None = None, use_batchnorm: bool = True
    ) -> None:
        super().__init__()
        hidden = hidden_dim or dim
        layers: list[nn.Module] = [nn.Linear(dim, hidden)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.LeakyReLU(0.1, inplace=True), nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        return (self.net(flat) + flat).reshape(shape)


def contrastive_loss(
    patches: torch.Tensor,
    memory_t: torch.Tensor,
    memory_norm_sq: torch.Tensor,
    *,
    k_attract: int,
    k_repel: int,
    radius_sq: float,
    alpha: float,
    chunk: int = 8192,
) -> dict[str, torch.Tensor]:
    """Attraction to the nearest entries, repulsion of the next nearest.

    Distances use ``‖p-m‖² = ‖p‖² + ‖m‖² - 2p·m`` so the heavy term is one
    matmul. On CUDA that matmul runs in bf16 — same exponent range as fp32,
    ~16x the throughput on tensor cores — while the norms stay fp32, because
    their cancellation against ``2p·m`` is the precision-sensitive step and
    the small distances it produces are exactly what ``topk`` selects.
    """
    if patches.shape[0] == 0:
        zero = torch.zeros((), device=patches.device)
        return {"loss": zero, "L_att": zero, "L_rep": zero}

    k_total = k_attract + k_repel
    use_bf16 = memory_t.dtype == torch.bfloat16
    attract_sum = torch.zeros((), device=patches.device)
    repel_sum = torch.zeros((), device=patches.device)
    n_attract = n_repel = 0

    for start in range(0, patches.shape[0], chunk):
        block = patches[start : start + chunk]
        block_norm_sq = (block * block).sum(dim=1, keepdim=True)
        inner = (block.bfloat16() @ memory_t).float() if use_bf16 else block @ memory_t
        # Cancellation can produce tiny negatives; clamping is gradient-safe
        # because it only bites where d² ≈ 0 < r², inside the attraction zone
        # where relu(d² - r²) already has zero gradient.
        distances = (
            block_norm_sq + memory_norm_sq.unsqueeze(0) - 2.0 * inner
        ).clamp_min(0.0)

        nearest, _ = torch.topk(distances, k_total, dim=1, largest=False)
        attract = nearest[:, :k_attract]
        repel = nearest[:, k_attract:]
        attract_sum = attract_sum + F.relu(attract - radius_sq).sum()
        repel_sum = repel_sum + F.relu(radius_sq + alpha - repel).sum()
        n_attract += attract.numel()
        n_repel += repel.numel()

    l_attract = attract_sum / max(n_attract, 1)
    l_repel = repel_sum / max(n_repel, 1)
    return {"loss": l_attract + l_repel, "L_att": l_attract, "L_rep": l_repel}


class CFA(Detector[CFAConfig]):
    name: ClassVar[str] = "cfa"
    config_type: ClassVar[type[DetectorConfig]] = CFAConfig

    def __init__(self, config: CFAConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.backbone = build_backbone(config.backbone, device=self.device)
        self.descriptor: Descriptor | None = None
        self.memory: torch.Tensor | None = None
        self.adapted_memory: torch.Tensor | None = None
        self._step = 0
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _patch_features(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.backbone(images, layers=self.config.feature_layers)
        assert self.config.target_layer is not None
        features = patchify_and_combine(
            maps,
            patch_size=self.config.patch_size,
            target_layer=self.config.target_layer,
        )
        # Clone out of the backbone's inference_mode: these feed a trainable
        # descriptor and inference tensors cannot be saved for backward.
        return features.float().clone()

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        self.memory = self._build_memory(train_good)
        self.descriptor = Descriptor(
            self.memory.shape[1], self.config.hidden_dim, self.config.use_batchnorm
        ).to(self.device)

        loader = self.make_loader(
            train_good,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._step = 0
        self._cache = None
        train_loop(
            [self.descriptor],
            loader,
            self._loss,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="cfa",
        )

        with torch.no_grad():
            self.adapted_memory = self.descriptor(self.memory).detach()
        self.memory = None
        self._cache = None

    @torch.no_grad()
    def _build_memory(self, records: Sequence[ImageRecord]) -> torch.Tensor:
        """Coreset of raw backbone patch features.

        ``no_grad`` rather than ``inference_mode``: the returned tensor sits on
        the constant side of a distance against grad-tracked patches, and
        PyTorch refuses to save inference tensors for backward even there.
        """
        self.log.info(
            "    [%s] extracting train/good patch features (%d images)",
            now_hms(),
            len(records),
        )
        chunks: list[torch.Tensor] = []
        for images, _masks, _indices in self.make_loader(
            records, batch_size=self.runtime.batch_size
        ):
            images = images.to(self.device, non_blocking=True)
            with self.autocast():
                features = self._patch_features(images)
            chunks.append(features.reshape(-1, features.shape[-1]).detach().cpu())

        features_cpu = torch.cat(chunks, dim=0)
        n_select = max(int(self.config.coreset_frac * features_cpu.shape[0]), 1)
        self.log.info(
            "    [%s] greedy coreset: selecting %d of %d (%.1f%%)",
            now_hms(),
            n_select,
            features_cpu.shape[0],
            self.config.coreset_frac * 100,
        )
        indices = greedy_coreset(
            features_cpu,
            n_select,
            self.device,
            seed=self.config.seed,
            algorithm=self.config.coreset_algorithm,
            batch_size=self.config.coreset_batch,
        )
        # fp32: the descriptor's BatchNorm statistics are unstable in fp16.
        memory = features_cpu[indices].to(self.device).float().contiguous()
        self.log.info(
            "    [%s] memory bank (%d, %d)  %.1f MB",
            now_hms(),
            memory.shape[0],
            memory.shape[1],
            memory.element_size() * memory.numel() / 1e6,
        )
        return memory

    def _refresh_memory_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.descriptor is not None and self.memory is not None
        with torch.no_grad():
            adapted = self.descriptor(self.memory).detach()
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            memory_t = adapted.to(dtype).t().contiguous()
            norm_sq = (adapted.float() ** 2).sum(dim=1)
        return memory_t, norm_sq

    def _loss(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.descriptor is not None
        if self._cache is None or self._step % self.config.memory_refresh_every == 0:
            self._cache = self._refresh_memory_cache()
        self._step += 1

        memory_t, norm_sq = self._cache
        with torch.no_grad():
            features = self._patch_features(images)
        adapted = self.descriptor(features.reshape(-1, features.shape[-1]))
        return contrastive_loss(
            adapted,
            memory_t,
            norm_sq,
            k_attract=self.config.k_attract,
            k_repel=self.config.k_repel,
            radius_sq=self.config.radius,
            alpha=self.config.alpha,
            chunk=self.config.loss_chunk,
        )

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.descriptor is None or self.adapted_memory is None:
            raise RuntimeError("CFA.score_batch() called before fit()")

        with self.autocast():
            features = self._patch_features(images)
        batch, n_patches, channels = features.shape
        side = math.isqrt(n_patches)

        adapted = self.descriptor(features.reshape(-1, channels))
        k = min(self.config.k_test, self.adapted_memory.shape[0])
        out = torch.empty(adapted.shape[0], device=self.device, dtype=torch.float32)

        for start in range(0, adapted.shape[0], self.config.score_chunk):
            end = min(adapted.shape[0], start + self.config.score_chunk)
            squared = torch.cdist(adapted[start:end], self.adapted_memory) ** 2
            out[start:end] = squared.topk(k, dim=1, largest=False).values.mean(dim=1)

        return self.upsample(out.reshape(batch, side, side))

    def release(self) -> None:
        self.descriptor = None
        self.memory = None
        self.adapted_memory = None
        self._cache = None
        super().release()
