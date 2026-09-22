"""Shared scaffolding for the detectors that train something.

Eight detectors — EfficientAD, FastFlow, Reverse Distillation, CFA, DRAEM,
CutPaste, GLASS, UniAD — each had its own ``train_<name>`` function, and each
opened with the same forty lines: convert an iteration budget into epochs,
build Adam plus a cosine schedule, build an AMP scaler, run the loop, average
the loss components, log every N epochs, switch to eval.

Only the inner ``loss = f(batch)`` differs. That is the part a detector
supplies here.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from spacepresso.core.logging import get_logger, now_hms

__all__ = ["TrainingSchedule", "train_loop"]

logger = get_logger(__name__)

#: Computes the loss for one batch. Returns either a scalar loss, or a mapping
#: of named components whose ``"loss"`` entry is the one to back-propagate —
#: the rest are logged, which is how the per-component curves in the old run
#: logs (``L_st``, ``L_ae``, ``L_stae``) are preserved.
LossFn = Callable[[torch.Tensor], torch.Tensor | Mapping[str, torch.Tensor]]


class TrainingSchedule:
    """Converts an iteration budget into whole epochs.

    Every detector took both ``--epochs`` and ``--total-iters`` and preferred
    the latter when set, because an iteration budget is what actually bounds
    wall-clock time when class sizes differ by 10x. That conversion is here
    once.
    """

    def __init__(
        self,
        loader: DataLoader,
        *,
        total_iters: int | None,
        epochs: int,
    ) -> None:
        self.iters_per_epoch = max(len(loader), 1)
        if total_iters and total_iters > 0:
            self.epochs = max(1, math.ceil(total_iters / self.iters_per_epoch))
            logger.info(
                "    [auto-epoch] total_iters=%d / %d → %d epochs",
                total_iters,
                self.iters_per_epoch,
                self.epochs,
            )
        else:
            self.epochs = epochs
        self.total_iters = self.iters_per_epoch * self.epochs


def train_loop(
    modules: Iterable[nn.Module],
    loader: DataLoader,
    loss_fn: LossFn,
    *,
    device: torch.device,
    lr: float,
    weight_decay: float = 0.0,
    total_iters: int | None = None,
    epochs: int = 100,
    amp: bool = True,
    grad_clip: float | None = None,
    log_divisions: int = 8,
    label: str = "training",
) -> None:
    """Run Adam + cosine annealing over ``loader``, minimising ``loss_fn``.

    ``modules`` are set to train mode, have their parameters collected into
    one optimiser, and are returned in eval mode. Batches are moved to
    ``device`` before ``loss_fn`` sees them.
    """
    modules = list(modules)
    parameters = [
        p for module in modules for p in module.parameters() if p.requires_grad
    ]
    if not parameters:
        raise ValueError(f"{label}: no trainable parameters")

    schedule = TrainingSchedule(loader, total_iters=total_iters, epochs=epochs)
    optimiser = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(schedule.total_iters, 1)
    )
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger.info(
        "    [%s] %s: %d epochs × %d iters (%d total)  amp=%s",
        now_hms(),
        label,
        schedule.epochs,
        schedule.iters_per_epoch,
        schedule.total_iters,
        use_amp,
    )

    log_every = max(1, schedule.epochs // max(log_divisions, 1))
    started = time.time()

    for epoch in range(schedule.epochs):
        for module in modules:
            module.train()

        totals: dict[str, float] = {}
        seen = 0
        for batch in loader:
            images = _batch_images(batch).to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = loss_fn(images)
            components = (
                dict(result) if isinstance(result, Mapping) else {"loss": result}
            )
            loss = components["loss"]

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            size = images.shape[0]
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * size
            seen += size

        if (epoch + 1) % log_every == 0 or epoch == schedule.epochs - 1:
            parts = " ".join(
                f"{key}={total / max(seen, 1):.4f}"
                for key, total in sorted(totals.items())
            )
            logger.info(
                "      epoch %3d/%d  %s  lr=%.2e  elapsed=%.1fs",
                epoch + 1,
                schedule.epochs,
                parts,
                scheduler.get_last_lr()[0],
                time.time() - started,
            )

    for module in modules:
        module.eval()
    logger.info("    [%s] %s done (%.1fs)", now_hms(), label, time.time() - started)


def _batch_images(batch: object) -> torch.Tensor:
    """Pull the image tensor out of whatever the loader yields.

    ``SpacepressoDataset`` yields ``(images, masks, indices)``; the
    augmentation datasets yield a bare tensor or their own tuple whose first
    element is the image.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]  # type: ignore[return-value]
    raise TypeError(f"cannot find images in a batch of type {type(batch)!r}")
