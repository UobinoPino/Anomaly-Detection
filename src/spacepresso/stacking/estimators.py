"""The models the stacker can fit, behind one interface.

``xgboost_stacker_v*.py`` and ``log*_stacker*.py`` were separate programs —
about 4,300 lines between them — that differed only in which estimator they
called. They each had their own ``fit_per_class``, ``fuse_test``,
``tune_global``, ``tune_per_class``, ``pooled_cv_score_one_class`` and
``loao_oof_one_class``, all near-clones.

The estimator is a parameter, so there is one stacker with a ``--estimator``
flag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from spacepresso.core.logging import get_logger

__all__ = [
    "ESTIMATORS",
    "Estimator",
    "EstimatorName",
    "LogisticEstimator",
    "XGBoostEstimator",
    "build_estimator",
    "default_params",
    "suggest_params",
]

logger = get_logger(__name__)

EstimatorName = Literal["xgboost", "logreg"]
ESTIMATORS: tuple[EstimatorName, ...] = ("xgboost", "logreg")


class Estimator(ABC):
    """Fit on a pixel feature matrix, predict an anomaly score per pixel."""

    name: str

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        self.params = dict(params)
        self.seed = seed
        self.model: Any = None

    @abstractmethod
    def fit(
        self, x: npt.NDArray[np.float32], y: npt.NDArray[np.integer]
    ) -> Estimator: ...

    @abstractmethod
    def predict(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

    def feature_importance(self) -> npt.NDArray[np.float32] | None:
        """Per-feature importance, if the estimator exposes one."""
        return None


class XGBoostEstimator(Estimator):
    """Gradient-boosted trees. The workhorse — handles the feature
    interactions (method A agrees with method B *and* we are near the centre)
    that a linear model cannot express."""

    name = "xgboost"

    def fit(self, x, y):
        import xgboost as xgb

        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        # Pixel labels are ~1% positive; without this the trees spend their
        # capacity on the majority class.
        if "scale_pos_weight" not in params:
            positives = max(int((y > 0).sum()), 1)
            params["scale_pos_weight"] = float((y.size - positives) / positives)
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(x, (y > 0).astype(np.int8))
        return self

    def predict(self, x):
        if self.model is None:
            raise RuntimeError("XGBoostEstimator.predict() before fit()")
        return self.model.predict_proba(x)[:, 1].astype(np.float32)

    def feature_importance(self):
        if self.model is None:
            return None
        return np.asarray(self.model.feature_importances_, dtype=np.float32)


class LogisticEstimator(Estimator):
    """Regularised logistic regression on standardised features.

    Faster to fit, far fewer knobs, and a useful sanity check: when the
    boosted model does not beat this by much, the gain is coming from the
    detectors rather than from the stacking.
    """

    name = "logreg"

    def fit(self, x, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        params = dict(self.params)
        l1_ratio = params.get("l1_ratio")
        # saga is the only solver supporting elasticnet; lbfgs is faster for
        # the pure-L2 case, so the solver follows the penalty rather than
        # being a separate knob the caller has to keep consistent.
        if params.get("penalty") == "elasticnet" or (
            l1_ratio is not None and 0.0 < float(l1_ratio) < 1.0
        ):
            params.setdefault("penalty", "elasticnet")
            params["solver"] = "saga"
        else:
            params.pop("l1_ratio", None)
            params.setdefault("penalty", "l2")
            params["solver"] = "lbfgs"

        params.setdefault("max_iter", 1000)
        params.setdefault("class_weight", "balanced")
        params["random_state"] = self.seed

        self.model = Pipeline(
            [
                ("scale", StandardScaler(copy=False)),
                ("clf", LogisticRegression(**params)),
            ]
        )
        self.model.fit(x, (y > 0).astype(np.int8))
        return self

    def predict(self, x):
        if self.model is None:
            raise RuntimeError("LogisticEstimator.predict() before fit()")
        return self.model.predict_proba(x)[:, 1].astype(np.float32)

    def feature_importance(self):
        if self.model is None:
            return None
        return np.abs(self.model.named_steps["clf"].coef_[0]).astype(np.float32)


DEFAULT_PARAMS: dict[EstimatorName, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": 8,
    },
    "logreg": {"C": 1.0, "l1_ratio": 0.0},
}


def default_params(name: EstimatorName, *, device: str = "cpu") -> dict[str, Any]:
    """Default hyperparameters, with the device applied for XGBoost.

    The v8 stacker set the device by mutating ``xgboost_stacker_v6``'s
    module-level ``DEFAULT_XGB_PARAMS`` dict in place, so that "every
    ``_make_xgb`` call site" would pick it up. That made the device a hidden
    piece of cross-module global state. It is now an argument.
    """
    params = dict(DEFAULT_PARAMS[name])
    if name == "xgboost":
        params["device"] = device
    return params


def build_estimator(
    name: EstimatorName, params: dict[str, Any] | None = None, seed: int = 0
) -> Estimator:
    if name == "xgboost":
        return XGBoostEstimator(params or default_params("xgboost"), seed)
    if name == "logreg":
        return LogisticEstimator(params or default_params("logreg"), seed)
    raise ValueError(f"unknown estimator {name!r}; expected one of {ESTIMATORS}")


def suggest_params(name: EstimatorName, trial: Any) -> dict[str, Any]:
    """Optuna search space for an estimator."""
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        }
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }
