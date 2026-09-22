"""Run identity and cross-experiment tracking."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "append_to_master",
    "config_digest",
    "make_run_id",
    "slugify",
]

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(text: str) -> str:
    """Filesystem-safe fragment for a run name."""
    return _SLUG_RE.sub("-", text).strip("-")


def _canonical(value: Any) -> Any:
    """JSON-safe, order-stable rendering of a config value.

    Sets are sorted, tuples become lists, Paths become strings, so the
    digest does not depend on insertion order or sequence type.
    """
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def config_digest(fingerprint: Mapping[str, Any], length: int = 6) -> str:
    """Short SHA-1 over the fields that make a run reproducible.

    Only fields that change the *result* belong in ``fingerprint`` — not
    ``num_workers``, not ``report_dir``, not ``run_tag``. Re-running with the
    same science therefore lands on the same digest.
    """
    payload = json.dumps(_canonical(fingerprint), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:length]


def make_run_id(
    slug: str,
    fingerprint: Mapping[str, Any],
    *,
    run_tag: str = "",
    stamp: str | None = None,
) -> str:
    """``<timestamp>_<slug>[_<tag>]_<digest>``.

    Args:
        slug: the human-readable part a detector builds from its own settings,
            e.g. ``"effad_dnv2b14r_L9_in392_it5000_bs8"``.
        fingerprint: the result-determining settings, hashed into the digest.
        run_tag: free-text label from ``--run-tag``.
        stamp: overrideable for tests.
    """
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    parts = [stamp, slug]
    if run_tag:
        parts.append(slugify(run_tag))
    return "_".join(p for p in parts if p) + "_" + config_digest(fingerprint)


def append_to_master(master_csv: Path, row: Mapping[str, Any]) -> None:
    """Append one run to the cross-experiment CSV, widening columns as needed.

    Rewrites the whole file because the column set grows over time: a new
    detector reporting ``AP_class_09`` must not shift the existing rows.
    """
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    fieldnames: list[str] = []

    if master_csv.exists():
        with open(master_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)

    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

    existing.append({k: str(v) for k, v in row.items()})

    tmp = master_csv.with_suffix(master_csv.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in existing:
            writer.writerow({k: record.get(k, "") for k in fieldnames})
    tmp.replace(master_csv)
