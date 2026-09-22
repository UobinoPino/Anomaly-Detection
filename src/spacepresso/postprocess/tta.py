"""Test-time augmentation.

Every detector shipped its own ``score_batch``: flip the input, score it, flip
the score map back, average. Six copies of ``_score_one_pass`` and six of
``score_batch``, all the same loop with a different single-pass function
inside it.

The loop is the reusable part, so here it takes the single-pass function as an
argument. A detector implements "score one batch, no augmentation" and gets
every TTA mode for free.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Literal

import torch

__all__ = ["TTA_MODES", "TTAMode", "apply_tta"]

TTAMode = Literal["none", "hflip", "vflip", "hvflip", "d4"]
TTA_MODES: tuple[TTAMode, ...] = ("none", "hflip", "vflip", "hvflip", "d4")

#: A function mapping ``(B, C, H, W)`` inputs to ``(B, H, W)`` score maps.
ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def _transforms(
    mode: TTAMode,
) -> Iterator[tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]]:
    """Yield ``(forward, inverse)`` pairs for a TTA mode.

    ``forward`` maps the input image, ``inverse`` maps the resulting score map
    back to the original orientation. Each is its own involution here, but
    they are returned as a pair because ``d4`` includes rotations, where they
    differ.
    """
    identity = lambda t: t  # noqa: E731
    hflip = lambda t: torch.flip(t, dims=[-1])  # noqa: E731
    vflip = lambda t: torch.flip(t, dims=[-2])  # noqa: E731
    hvflip = lambda t: torch.flip(t, dims=[-2, -1])  # noqa: E731

    yield identity, identity
    if mode == "none":
        return
    if mode in ("hflip", "hvflip", "d4"):
        yield hflip, hflip
    if mode in ("vflip", "hvflip", "d4"):
        yield vflip, vflip
    if mode == "d4":
        yield hvflip, hvflip
        # The four 90-degree rotations. rot90(k) is undone by rot90(-k), so
        # forward and inverse genuinely differ for these.
        for k in (1, 2, 3):
            yield (
                lambda t, k=k: torch.rot90(t, k, dims=[-2, -1]),
                lambda t, k=k: torch.rot90(t, -k, dims=[-2, -1]),
            )


def apply_tta(
    score_fn: ScoreFn,
    x: torch.Tensor,
    mode: TTAMode = "none",
) -> torch.Tensor:
    """Average ``score_fn`` over the augmentations in ``mode``.

    Args:
        score_fn: scores one un-augmented batch, returning ``(B, H, W)``.
        x: input batch, ``(B, C, H, W)``.
        mode: ``none`` | ``hflip`` | ``vflip`` | ``hvflip`` | ``d4``.

    Averaging in score space (rather than voting, or taking a max) is what the
    detectors did before and what the calibration downstream assumes: the mean
    of several unbiased score maps stays on the same scale as one of them.
    """
    if mode not in TTA_MODES:
        raise ValueError(f"unknown TTA mode {mode!r}; expected one of {TTA_MODES}")

    total: torch.Tensor | None = None
    count = 0
    for forward, inverse in _transforms(mode):
        scored = inverse(score_fn(forward(x)))
        total = scored.clone() if total is None else total + scored
        count += 1

    assert total is not None
    return total / count
