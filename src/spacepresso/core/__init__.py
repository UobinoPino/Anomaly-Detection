"""Foundation layer: records, codec, metrics, imaging, submission, logging.

Nothing in ``core`` imports from any other ``spacepresso`` subpackage. That is
the rule that keeps the dependency graph acyclic, and it is enforced by
``tests/unit/test_architecture.py``.
"""

from spacepresso.core import codec, imaging, metrics, paths, tracking
from spacepresso.core.logging import get_logger, log_section, run_log, setup_logging
from spacepresso.core.records import ImageRecord, scan_dataset, select
from spacepresso.core.submission import load_submission, write_submission

__all__ = [
    "ImageRecord",
    "codec",
    "get_logger",
    "imaging",
    "load_submission",
    "log_section",
    "metrics",
    "paths",
    "run_log",
    "scan_dataset",
    "select",
    "setup_logging",
    "tracking",
    "write_submission",
]
