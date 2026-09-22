"""WinCLIP — zero-shot and few-shot anomaly detection with CLIP.

Jeong et al., *WinCLIP: Zero-/Few-Shot Anomaly Classification and
Segmentation* (CVPR 2023).

CLIP aligns images and text in one embedding space, so "a photo of a defective
gear" is a usable anomaly detector with no training data at all. Two
refinements make it work at the pixel level:

* **Windows.** A single patch token is too small to carry object-level
  semantics. Average-pooling patch tokens over sliding windows of several
  sizes gives each position a description of its neighbourhood, and the scores
  from each scale are averaged.
* **Few-shot reference matching.** When normal images are available — and here
  they always are — each window is also compared to every window of a handful
  of reference images. That term needs no text and catches defects the prompts
  do not name.

The two terms are blended by ``alpha``. Position-agnostic reference matching
(comparing against every position, not the corresponding one) is what makes
this robust to the multi-view drift in this dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.paths import project_root
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.detectors.prompts import (
    CLASS_FALLBACK_NAME,
    build_prompts,
    load_descriptions,
)

__all__ = ["WinCLIP", "WinCLIPConfig"]

#: CLIP ViT-B/16 was trained at 224px with 16px patches; the positional
#: embedding is tied to that geometry.
CLIP_INPUT_SIZE = 224


@dataclass
class WinCLIPConfig(DetectorConfig):
    input_size: int = CLIP_INPUT_SIZE
    model_name: str = "ViT-B-16"
    pretrained: str = "openai"

    window_sizes: tuple[int, ...] = (2, 3, 5)
    #: Weight of the zero-shot (text) term. 1.0 is pure zero-shot.
    alpha: float = 0.5
    n_references: int = 8

    descriptions_csv: Path | None = None
    reference_chunk: int = 4096

    def __post_init__(self) -> None:
        super().__post_init__()
        self.window_sizes = tuple(self.window_sizes)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.input_size != CLIP_INPUT_SIZE:
            raise ValueError(
                f"WinCLIP requires input_size={CLIP_INPUT_SIZE}; CLIP's "
                f"positional embedding is tied to that geometry."
            )
        if self.descriptions_csv is None:
            default = project_root() / "data" / "anomaly_descriptions.csv"
            self.descriptions_csv = default if default.is_file() else None

    def slug_parts(self) -> list[str]:
        windows = "-".join(str(w) for w in self.window_sizes)
        parts = [
            self.model_name.replace("/", "").replace("-", "").lower(),
            f"w{windows}",
            f"a{self.alpha:g}",
            f"ref{self.n_references}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


class WinCLIP(Detector[WinCLIPConfig]):
    name: ClassVar[str] = "winclip"
    config_type: ClassVar[type[DetectorConfig]] = WinCLIPConfig

    def __init__(self, config: WinCLIPConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.model, self.tokenizer, self.logit_scale = self._load_clip()
        self.text_normal: torch.Tensor | None = None
        self.text_anomaly: torch.Tensor | None = None
        self.reference_windows: dict[int, torch.Tensor] = {}
        self._descriptions = load_descriptions(config.descriptions_csv)

    def _load_clip(self):
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "WinCLIP needs open_clip. Install it with:\n"
                "    pip install 'open_clip_torch>=2.20'"
            ) from exc

        model, _, _ = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=self.device,
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # CLIP's learned temperature. Reusing it keeps the zero-shot softmax
        # calibrated the way the model was trained.
        scale = float(model.logit_scale.detach().exp().item())
        return model, open_clip.get_tokenizer(self.config.model_name), scale

    # ── CLIP plumbing ────────────────────────────────────────────────────
    @torch.inference_mode()
    def _encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, P, D)`` patch tokens in CLIP's *text-aligned* space.

        open_clip's ``encode_image`` returns only the CLS token, so the visual
        forward pass is reproduced here to keep the patch tokens — and the
        final ``ln_post`` and projection are applied to them too, which is
        what puts them in the same space as the text embeddings. Skipping that
        projection is the classic WinCLIP reimplementation bug: the cosine
        similarities come out plausible but meaningless.
        """
        visual = self.model.visual
        with self.autocast():
            x = visual.conv1(images)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
            if getattr(visual, "patch_dropout", None) is not None:
                x = visual.patch_dropout(x)
            if getattr(visual, "ln_pre", None) is not None:
                x = visual.ln_pre(x)
            x = visual.ln_post(visual.transformer(x))
            if getattr(visual, "proj", None) is not None:
                x = x @ visual.proj
        return x[:, 1:].float()

    @torch.inference_mode()
    def _encode_text(self, prompts: Sequence[str], batch_size: int = 128):
        """Encode, L2-normalise, mean-pool, L2-normalise again."""
        total: torch.Tensor | None = None
        count = 0
        for start in range(0, len(prompts), batch_size):
            tokens = self.tokenizer(list(prompts[start : start + batch_size])).to(
                self.device
            )
            with self.autocast():
                features = self.model.encode_text(tokens)
            features = F.normalize(features.float(), dim=-1)
            total = (
                features.sum(dim=0) if total is None else total + features.sum(dim=0)
            )
            count += features.shape[0]
        assert total is not None
        return F.normalize(total / max(count, 1), dim=-1)

    @staticmethod
    def _windows(patches: torch.Tensor, size: int) -> torch.Tensor:
        """Sliding-window average of patch tokens: ``(B, H', W', D)``."""
        batch, n_patches, dim = patches.shape
        side = int(round(n_patches**0.5))
        if side * side != n_patches:
            raise ValueError(f"non-square patch grid: P={n_patches}")
        grid = patches.reshape(batch, side, side, dim).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(grid, kernel_size=size, stride=1)
        return F.normalize(pooled.permute(0, 2, 3, 1), dim=-1)

    # ── fit ──────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        cls = train_good[0].cls
        described = self._descriptions.get(cls)
        object_name = (
            described.object_name
            if described
            else CLASS_FALLBACK_NAME.get(cls, "object")
        )
        normal_prompts, anomaly_prompts = build_prompts(
            object_name, described.per_type if described else None
        )
        self.log.info(
            "    prompts for %s (%r): %d normal, %d anomaly",
            cls,
            object_name,
            len(normal_prompts),
            len(anomaly_prompts),
        )
        self.text_normal = self._encode_text(normal_prompts)
        self.text_anomaly = self._encode_text(anomaly_prompts)

        self.reference_windows = {}
        if self.config.n_references <= 0 or self.config.alpha >= 1.0:
            return

        references = self._select_references(train_good)
        chunks = [
            self._encode_patches(images.to(self.device, non_blocking=True)).cpu()
            for images, _masks, _indices in self.make_loader(references)
        ]
        patches = torch.cat(chunks, dim=0).to(self.device)
        for size in self.config.window_sizes:
            self.reference_windows[size] = self._windows(patches, size).contiguous()
        self.log.info("    %d reference images encoded", len(references))

    def _select_references(
        self, train_good: Sequence[ImageRecord]
    ) -> list[ImageRecord]:
        """Spread the reference set across views rather than taking the first N.

        Taking the first N images of a multi-view class usually means N views
        of one or two objects, which makes the few-shot term blind to the
        other viewpoints.
        """
        by_view: dict[int, list[ImageRecord]] = {}
        for record in train_good:
            by_view.setdefault(
                record.view if record.view is not None else 0, []
            ).append(record)

        selected: list[ImageRecord] = []
        rng = np.random.default_rng(self.config.seed)
        views = sorted(by_view)
        index = 0
        while len(selected) < min(self.config.n_references, len(train_good)):
            bucket = by_view[views[index % len(views)]]
            choice = bucket[int(rng.integers(0, len(bucket)))]
            if choice not in selected:
                selected.append(choice)
            elif len(selected) >= sum(len(b) for b in by_view.values()):
                break
            index += 1
        return selected

    # ── score ────────────────────────────────────────────────────────────
    def _zero_shot(self, windows: torch.Tensor) -> torch.Tensor:
        assert self.text_normal is not None and self.text_anomaly is not None
        batch, height, width, dim = windows.shape
        text = torch.stack([self.text_normal, self.text_anomaly], dim=0)
        logits = (windows.reshape(-1, dim) @ text.T) * self.logit_scale
        return F.softmax(logits, dim=-1)[:, 1].reshape(batch, height, width)

    def _few_shot(
        self, windows: torch.Tensor, references: torch.Tensor
    ) -> torch.Tensor:
        """1 − max cosine similarity to any reference window at any position."""
        batch, height, width, dim = windows.shape
        queries = windows.reshape(-1, dim)
        bank = references.reshape(-1, dim)
        out = torch.empty(queries.shape[0], device=queries.device)
        for start in range(0, queries.shape[0], self.config.reference_chunk):
            end = min(queries.shape[0], start + self.config.reference_chunk)
            out[start:end] = 1.0 - (queries[start:end] @ bank.T).max(dim=-1).values
        return out.reshape(batch, height, width)

    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.text_normal is None:
            raise RuntimeError("WinCLIP.score_batch() called before fit()")

        patches = self._encode_patches(images)
        total = torch.zeros(
            images.shape[0],
            self.config.input_size,
            self.config.input_size,
            device=self.device,
        )

        for size in self.config.window_sizes:
            windows = self._windows(patches, size)
            score = self._zero_shot(windows)
            reference = self.reference_windows.get(size)
            if reference is not None:
                score = self.config.alpha * score + (1.0 - self.config.alpha) * (
                    self._few_shot(windows, reference)
                )
            total = total + F.interpolate(
                score.unsqueeze(1),
                size=(self.config.input_size, self.config.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return total / len(self.config.window_sizes)

    def release(self) -> None:
        self.text_normal = None
        self.text_anomaly = None
        self.reference_windows = {}
        super().release()
