"""Torch datasets and loaders over :class:`~spacepresso.core.records.ImageRecord`.

There was one ``InferenceDataset`` per detector — six exact clones of one
version and three of another, plus five copies of ``TrainGoodDataset`` — all
differing only in whether they returned a mask. There is now one dataset that
takes a flag, and one loader factory.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spacepresso.core.records import ImageRecord
from spacepresso.data.transforms import build_transform, load_image, load_mask

__all__ = [
    "SpacepressoDataset",
    "make_loader",
    "seed_worker",
]


class SpacepressoDataset(Dataset):
    """Yields ``(image, mask, index)``.

    ``index`` is the position in ``records`` and is what callers use to map a
    batch element back to its :class:`ImageRecord` — needed because the loader
    may be running with ``shuffle=True``.

    When ``load_masks`` is false, or a record has no mask on disk, the mask is
    a zero map of the right shape rather than ``None``, so the collate
    function stays trivial.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        *,
        input_size: int,
        load_masks: bool = False,
        normalise: bool = True,
    ) -> None:
        self.records = list(records)
        self.input_size = input_size
        self.load_masks = load_masks
        self.transform = build_transform(input_size, normalise=normalise)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        image = load_image(record.path, self.transform)
        mask = load_mask(
            record.mask_path if self.load_masks else None, self.input_size
        )
        return image, torch.from_numpy(mask), index


def seed_worker(worker_id: int) -> None:
    """Give each dataloader worker a distinct, run-reproducible seed.

    The previous ``worker_init_fn`` (10 copies) seeded every worker from the
    same base, so augmentation-heavy detectors — CutPaste, DRAEM, GLASS — drew
    correlated "random" anomalies across workers.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    records: Sequence[ImageRecord],
    *,
    batch_size: int,
    input_size: int,
    load_masks: bool = False,
    num_workers: int = 2,
    shuffle: bool = False,
    drop_last: bool = False,
    dataset: Dataset | None = None,
) -> DataLoader:
    """Build a dataloader over records, or over a caller-supplied ``dataset``.

    ``dataset`` is the hook the augmentation-based detectors use: CutPaste,
    DRAEM and GLASS need their own ``__getitem__`` (they synthesise anomalies
    on the fly) but want the same worker seeding, pinning and persistence
    policy as everything else.
    """
    ds = dataset if dataset is not None else SpacepressoDataset(
        records, input_size=input_size, load_masks=load_masks
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
    )
