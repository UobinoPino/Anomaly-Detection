"""Project path resolution.

Resolution order, first hit wins:

1. an explicit argument (``--data-root`` on the CLI),
2. ``SPACEPRESSO_DATA_ROOT`` / ``SPACEPRESSO_REPORT_DIR`` in the environment,
3. a well-known location relative to the repository root,
4. the Kaggle mount, if this is running in a Kaggle notebook.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "default_data_root",
    "default_report_dir",
    "project_root",
    "resolve_data_root",
    "resolve_report_dir",
]

ENV_DATA_ROOT = "SPACEPRESSO_DATA_ROOT"
ENV_REPORT_DIR = "SPACEPRESSO_REPORT_DIR"
ENV_PROJECT_ROOT = "SPACEPRESSO_ROOT"

_KAGGLE_INPUT = Path("/kaggle/input/spacepresso")
_KAGGLE_WORKING = Path("/kaggle/working")


def project_root() -> Path:
    """The repository root.

    ``$SPACEPRESSO_ROOT`` wins; otherwise it is derived from this file's
    location (``src/spacepresso/core/paths.py`` → three levels up), which is
    correct for both an editable install and a plain clone.
    """
    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _in_kaggle() -> bool:
    return _KAGGLE_WORKING.is_dir() and "KAGGLE_URL_BASE" in os.environ


def default_data_root() -> Path:
    if env := os.environ.get(ENV_DATA_ROOT):
        return Path(env).expanduser()
    if _in_kaggle():
        for candidate in (_KAGGLE_INPUT / "data", _KAGGLE_INPUT):
            if candidate.is_dir():
                return candidate
    return project_root() / "data"


def default_report_dir() -> Path:
    if env := os.environ.get(ENV_REPORT_DIR):
        return Path(env).expanduser()
    if _in_kaggle():
        return _KAGGLE_WORKING / "baseline_out"
    return project_root() / "baseline_out"


def resolve_data_root(explicit: Path | str | None = None) -> Path:
    """Data root, with the documented precedence. Must exist."""
    root = Path(explicit).expanduser() if explicit else default_data_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"data root not found: {root}\nPass --data-root, or set ${ENV_DATA_ROOT}."
        )
    return root.resolve()


def resolve_report_dir(explicit: Path | str | None = None) -> Path:
    """Output root, with the documented precedence. Created if missing."""
    root = Path(explicit).expanduser() if explicit else default_report_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()
