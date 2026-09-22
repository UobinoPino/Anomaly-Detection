"""Dataset records and filesystem scanning.

The single source of truth for *what an image is* in this project. Previously
this lived inside ``models/patchcore_baseline_v2.py``, which meant every other
detector had to import PatchCore in order to list a directory.

Expected on-disk layout::

    <data_root>/
      class_01/
        train/good/*.png
        train/anomaly_<type>/*.png
        ground_truth_train/anomaly_<type>/*.png     # masks, same stems
        test/*.png
      class_02/
      ...

Filenames may carry a multi-view suffix, ``<sample>_view<NN>.png``. Views of
the same physical object share a ``sample_id``; several detectors and the
stacker exploit that grouping.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "IMG_EXTS",
    "ImageRecord",
    "Split",
    "group_by_sample",
    "parse_view",
    "scan_dataset",
    "select",
]

IMG_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

VIEW_RE = re.compile(r"^(?P<base>.+?)_view(?P<v>\d+)\.[A-Za-z]+$")

#: The three splits a record can belong to. ``train_anomaly`` is the *local
#: validation* set: it is the only split that carries ground-truth masks.
Split = str
TRAIN_GOOD: Split = "train_good"
TRAIN_ANOMALY: Split = "train_anomaly"
TEST: Split = "test"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One image on disk, plus everything derivable from its path.

    Immutable on purpose: records are shared freely between the runner, the
    detectors and the prediction writers, and nothing downstream should be
    able to mutate another component's view of the dataset.
    """

    path: Path
    cls: str
    split: Split
    anomaly_type: str | None = None
    sample_id: str | None = None
    view: int | None = None
    mask_path: Path | None = None

    @property
    def stem(self) -> str:
        """Submission ID for this image (``submission.csv`` ``ID`` column)."""
        return self.path.stem

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None


def parse_view(filename: str) -> tuple[str, int | None]:
    """Split ``sample_view03.png`` into ``("sample", 3)``.

    Files without a view suffix return ``(stem, None)``.
    """
    m = VIEW_RE.match(filename)
    if m:
        return m.group("base"), int(m.group("v"))
    return Path(filename).stem, None


def _find_mask(mask_dir: Path, image: Path) -> Path | None:
    """Locate the ground-truth mask for ``image`` inside ``mask_dir``.

    Tries the exact filename first, then any file sharing the stem — masks are
    often stored with a different extension than the image.
    """
    if not mask_dir.is_dir():
        return None
    exact = mask_dir / image.name
    if exact.exists():
        return exact
    for candidate in sorted(mask_dir.iterdir()):
        if candidate.stem == image.stem and candidate.suffix.lower() in IMG_EXTS:
            return candidate
    return None


def _images_in(directory: Path, recursive: bool = False) -> list[Path]:
    if not directory.is_dir():
        return []
    it = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


def scan_dataset(data_root: Path | str) -> list[ImageRecord]:
    """Walk ``data_root`` and return every image as an :class:`ImageRecord`.

    Raises:
        FileNotFoundError: if ``data_root`` does not exist. The previous
            implementation printed a message and returned an empty list, which
            meant a mistyped ``--data-root`` produced a successful run that
            silently scored nothing.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")

    records: list[ImageRecord] = []
    class_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("class_")
    )
    if not class_dirs:
        raise FileNotFoundError(
            f"no class_* directories under {root} — is this the right --data-root?"
        )

    for cdir in class_dirs:
        cls = cdir.name

        for p in _images_in(cdir / "train" / "good"):
            sid, view = parse_view(p.name)
            records.append(ImageRecord(p, cls, TRAIN_GOOD, sample_id=sid, view=view))

        train_dir = cdir / "train"
        if train_dir.is_dir():
            anomaly_dirs = sorted(
                d
                for d in train_dir.iterdir()
                if d.is_dir() and d.name.startswith("anomaly_")
            )
            for adir in anomaly_dirs:
                mask_dir = cdir / "ground_truth_train" / adir.name
                for p in _images_in(adir):
                    sid, view = parse_view(p.name)
                    records.append(
                        ImageRecord(
                            p,
                            cls,
                            TRAIN_ANOMALY,
                            anomaly_type=adir.name,
                            sample_id=sid,
                            view=view,
                            mask_path=_find_mask(mask_dir, p),
                        )
                    )

        for p in _images_in(cdir / "test", recursive=True):
            sid, view = parse_view(p.name)
            records.append(ImageRecord(p, cls, TEST, sample_id=sid, view=view))

    return records


def select(
    records: Iterable[ImageRecord],
    *,
    cls: str | None = None,
    split: Split | None = None,
) -> list[ImageRecord]:
    """Filter records by class and/or split.

    Replaces the three-line list comprehension that opened all 14 copies of
    ``run_one_class``.
    """
    out = list(records)
    if cls is not None:
        out = [r for r in out if r.cls == cls]
    if split is not None:
        out = [r for r in out if r.split == split]
    return out


def group_by_sample(
    records: Iterable[ImageRecord],
) -> dict[str, list[ImageRecord]]:
    """Group records by ``(class, sample_id)`` — i.e. the views of one object.

    The key is ``"<cls>/<sample_id>"``; records with no ``sample_id`` fall back
    to their stem so they form singleton groups rather than colliding.
    """
    groups: dict[str, list[ImageRecord]] = {}
    for r in records:
        key = f"{r.cls}/{r.sample_id or r.stem}"
        groups.setdefault(key, []).append(r)
    for group in groups.values():
        group.sort(key=lambda r: (r.view if r.view is not None else -1, r.stem))
    return groups


def classes_in(records: Iterable[ImageRecord]) -> list[str]:
    """Sorted list of distinct class names present in ``records``."""
    return sorted({r.cls for r in records})
