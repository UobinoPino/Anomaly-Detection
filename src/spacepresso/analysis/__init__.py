"""Dataset and submission analysis.

Tools that inform the pipeline rather than being part of it: the spatial
priors the stacker consumes, and the pre-submission sanity check.
"""

from spacepresso.analysis.priors import compute_priors, write_priors
from spacepresso.analysis.submission_check import check_submission

__all__ = ["check_submission", "compute_priors", "write_priors"]
