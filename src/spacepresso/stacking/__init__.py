"""Combining several detectors into one submission.

One configurable stacker, replacing the six forked programs
(``xgboost_stacker_v3`` … ``v8``, ``log_stacker_v2``, ``logreg_stacker``,
``ensemble_tier0``) that between them ran to ~17,500 lines.
"""

from spacepresso.stacking.calibration import Calibrator, fit_calibrator
from spacepresso.stacking.crossval import ClassTrainingData, out_of_fold, pooled_cv_score
from spacepresso.stacking.dataset import AlignedValidation, load_test_scores, load_validation
from spacepresso.stacking.estimators import ESTIMATORS, build_estimator
from spacepresso.stacking.features import FeatureConfig, featurize_image
from spacepresso.stacking.fusion import RULES, fuse
from spacepresso.stacking.normalisation import SCOPES, rank_normalise
from spacepresso.stacking.stacker import StackerConfig, StackerResult, run_stacker

__all__ = [
    "ESTIMATORS",
    "RULES",
    "SCOPES",
    "AlignedValidation",
    "Calibrator",
    "ClassTrainingData",
    "FeatureConfig",
    "StackerConfig",
    "StackerResult",
    "build_estimator",
    "featurize_image",
    "fit_calibrator",
    "fuse",
    "load_test_scores",
    "load_validation",
    "out_of_fold",
    "pooled_cv_score",
    "rank_normalise",
    "run_stacker",
]
