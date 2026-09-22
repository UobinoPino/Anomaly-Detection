"""Run logging.

Library code calls ``get_logger(__name__).info(...)``. The runner installs a
handler that writes to both the console and ``<run_dir>/run_log.txt``.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "get_logger",
    "log_section",
    "log_subsection",
    "run_log",
    "setup_logging",
]

_ROOT = "spacepresso"
_RULE_WIDTH = 78


def get_logger(name: str) -> logging.Logger:
    """Logger for a module. Pass ``__name__``."""
    if name == "__main__" or not name.startswith(_ROOT):
        return logging.getLogger(_ROOT)
    return logging.getLogger(name)


class _PlainFormatter(logging.Formatter):
    """Keeps INFO lines bare so run logs stay readable as reports.

    Warnings and errors get a prefix, because those are the lines someone
    greps for afterwards.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno <= logging.INFO:
            return message
        return f"[{record.levelname}] {message}"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Install the console handler. Idempotent."""
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(h, "_spacepresso_console", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_PlainFormatter())
        handler._spacepresso_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


@contextmanager
def run_log(path: Path, level: int = logging.INFO) -> Iterator[logging.Logger]:
    """Mirror everything logged inside the block into ``path``."""
    logger = setup_logging(level)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(_PlainFormatter())
    handler.setLevel(level)
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        handler.flush()
        logger.removeHandler(handler)
        handler.close()


def log_section(logger: logging.Logger, title: str, char: str = "=") -> None:
    """A banner line around a title."""
    rule = char * _RULE_WIDTH
    logger.info("\n%s\n  %s\n%s", rule, title, rule)


def log_subsection(logger: logging.Logger, title: str) -> None:
    """A minor heading."""
    logger.info("\n--- %s ---", title)


def now_hms() -> str:
    """Wall-clock ``HH:MM:SS``, for progress lines inside long loops."""
    return time.strftime("%H:%M:%S")
