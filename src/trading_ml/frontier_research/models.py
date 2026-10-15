"""Small, reproducible estimators for isolated temporal research.

The caller owns chronological splits, label purging and embargo. These adapters
fit only the supplied frame; no scaler or other state is learned at prediction.
Optional boosting packages are imported only when their recipe is requested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_ml.errors import DataError


def _parameters(recipe: Mapping[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(recipe.get("parameters", {}))
    for name, value in fixed.items():
        if name in parameters and parameters[name] != value:
            raise DataError(f"Research requires {name}={value!r}")
    return parameters | fixed


def create_estimator(recipe: Mapping[str, Any], regression: bool = False) -> Any:
    """Create a CPU estimator with seed 42 and a single worker.

    ``recipe`` contains ``kind`` and an optional ``parameters`` mapping. Parameter
    overrides can change the regularized defaults, except seed/CPU/worker flags.
    Logistic regression is a classification recipe only.
    """
    kind = recipe.get("kind")
    if kind == "logistic":
        if regression:
            raise DataError("The logistic recipe requires a binary classification target")
        parameters = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"} | _parameters(
            recipe, {"random_state": 42}
        )
        return Pipeline(
            [("scale", StandardScaler()), ("logistic", LogisticRegression(**parameters))]
        )
    if kind == "random_forest":
        parameters = {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_leaf": 20,
        } | _parameters(recipe, {"random_state": 42, "n_jobs": 1})
        estimator = RandomForestRegressor if regression else RandomForestClassifier
        return estimator(**parameters)
    if kind == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        parameters = {
            "n_estimators": 160,
            "max_depth": 2,
            "learning_rate": 0.03,
            "min_child_weight": 10,
            "reg_lambda": 5.0,
            "tree_method": "hist",
            "eval_metric": "rmse" if regression else "logloss",
        } | _parameters(recipe, {"random_state": 42, "n_jobs": 1, "device": "cpu"})
        estimator = XGBRegressor if regression else XGBClassifier
        return estimator(**parameters)
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as error:
            raise DataError("The lightgbm recipe requires the optional lightgbm package") from error
        parameters = {
            "n_estimators": 160,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 40,
            "learning_rate": 0.03,
            "reg_lambda": 5.0,
            "verbosity": -1,
        } | _parameters(
            recipe,
            {
                "random_state": 42,
                "n_jobs": 1,
                "device_type": "cpu",
                "deterministic": True,
                "force_col_wise": True,
            },
        )
        estimator = LGBMRegressor if regression else LGBMClassifier
        return estimator(**parameters)
    if kind == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as error:
            raise DataError("The catboost recipe requires the optional catboost package") from error
        parameters = {
            "iterations": 160,
            "depth": 3,
            "learning_rate": 0.03,
            "l2_leaf_reg": 10,
            "verbose": False,
            "loss_function": "RMSE" if regression else "Logloss",
        } | _parameters(
            recipe,
            {
                "random_seed": 42,
                "thread_count": 1,
                "task_type": "CPU",
                "allow_writing_files": False,
            },
        )
        estimator = CatBoostRegressor if regression else CatBoostClassifier
        return estimator(**parameters)
    raise DataError(f"Unknown frontier estimator kind: {kind!r}")


def _features(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    names = list(columns)
    if not names or len(names) != len(set(names)):
        raise DataError("Estimator features must be a nonempty list of unique columns")
    if not frame.columns.is_unique:
        raise DataError("Estimator input has duplicate column names")
    missing = set(names) - set(frame.columns)
    if missing:
        raise DataError(f"Missing estimator features: {sorted(missing)}")
    try:
        values = frame.loc[:, names].astype(float)
    except (TypeError, ValueError) as error:
        raise DataError("Estimator features must be numeric") from error
    if not np.isfinite(values.to_numpy()).all():
        raise DataError("Estimator features contain nonfinite values")
    return values


def fit_model(
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str | pd.Series | Sequence[float] | np.ndarray,
    recipe: Mapping[str, Any],
    regression: bool = False,
) -> Any:
    """Fit exactly the provided rows and labels, returning the fitted estimator."""
    values = _features(frame, columns)
    if not len(values):
        raise DataError("Cannot fit an empty training frame")
    if isinstance(target, str):
        if target not in frame:
            raise DataError(f"Missing estimator target: {target}")
        target = frame[target]
    if isinstance(target, pd.Series) and not target.index.equals(frame.index):
        raise DataError("Estimator target index does not match the training frame")
    try:
        labels = np.asarray(target, dtype=float)
    except (TypeError, ValueError) as error:
        raise DataError("Estimator targets must be numeric") from error
    if labels.ndim != 1 or len(labels) != len(values) or not np.isfinite(labels).all():
        raise DataError("Estimator targets must be a finite vector matching the training rows")
    if not regression and set(np.unique(labels)) != {0.0, 1.0}:
        raise DataError("Classification training requires both binary classes 0 and 1")
    model = create_estimator(recipe, regression)
    model.fit(values, labels)
    return model


def predict_probability(model: Any, frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Return the fitted classifier's probability of class one without refitting."""
    values = _features(frame, columns)
    if not hasattr(model, "predict_proba"):
        raise DataError("Probability prediction requires a fitted classifier")
    classes = np.asarray(model.classes_)
    indices = np.flatnonzero(classes == 1)
    if len(indices) != 1:
        raise DataError("Fitted classifier must contain the positive class 1")
    if not len(values):
        return np.empty(0, dtype=float)
    result = np.asarray(model.predict_proba(values)[:, indices[0]], dtype=float)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise DataError("Classifier returned invalid probabilities")
    return result


def predict_value(model: Any, frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Return fitted model values, primarily for regression recipes."""
    values = _features(frame, columns)
    if not len(values):
        return np.empty(0, dtype=float)
    result = np.asarray(model.predict(values), dtype=float)
    if result.ndim != 1 or len(result) != len(values) or not np.isfinite(result).all():
        raise DataError("Estimator returned invalid prediction values")
    return result
