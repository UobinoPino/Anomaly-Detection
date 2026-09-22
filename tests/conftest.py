"""Shared fixtures.

``tests/fixtures`` holds helpers, not test cases — it is importable because
``tests/`` is on the path via the src layout and this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.dataset import build_dataset


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic Spacepresso dataset, built once per session."""
    return build_dataset(tmp_path_factory.mktemp("data"))


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    out = tmp_path / "baseline_out"
    out.mkdir()
    return out
