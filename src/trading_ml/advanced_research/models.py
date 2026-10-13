from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from trading_ml.common import frame_hash, read_json, sha256, write_new_json
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.models.calibration import apply_calibration, fit_sigmoid


@dataclass
class Fitted:
    model: Any
    columns: list[str]
    audit: dict[str, Any]
    calibration: dict[str, Any] | None = None
    regression: bool = False

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        if isinstance(self.model, dict):
            state = self.model
            values = (frame[self.columns].to_numpy() - np.array(state["mean"])) / np.array(
                state["scale"]
            )
            score = values @ np.array(state["coef"]) + state["intercept"]
            raw = 1 / (1 + np.exp(-np.clip(score, -700, 700)))
        elif self.regression:
            score = self.model.predict(frame[self.columns])
            scale = self.audit["score_scale"]
            raw = 1 / (1 + np.exp(-np.clip(score / scale, -35, 35)))
        else:
            raw = self.model.predict_proba(frame[self.columns])[:, 1]
        return apply_calibration(raw, self.calibration)

    def expected_return(self, frame: pd.DataFrame) -> np.ndarray:
        if self.regression:
            return self.model.predict(frame[self.columns])
        probability = self.probabilities(frame)
        return (
            probability * self.audit["positive_payoff_mean"]
            + (1 - probability) * self.audit["negative_payoff_mean"]
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        state = {
            "features": self.columns,
            "audit": self.audit,
            "calibration": self.calibration,
            "regression": self.regression,
        }
        if isinstance(self.model, dict):
            state["logistic"] = self.model
        elif isinstance(self.model, Pipeline):
            scaler, logistic = self.model.named_steps["scale"], self.model.named_steps["logistic"]
            state["logistic"] = {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coef": logistic.coef_[0].tolist(),
                "intercept": float(logistic.intercept_[0]),
            }
        else:
            self.model.save_model(directory / "model.ubj")
            state["model_sha256"] = sha256(directory / "model.ubj")
        write_new_json(directory / "fit.json", state)


def load_fitted(
    directory: Path,
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    recipe: dict[str, Any],
    protocol: dict[str, Any],
) -> Fitted:
    state = read_json(directory / "fit.json")
    columns = state["features"]
    training = frame.loc[
        frame.research_ready
        & (frame.reference_time < cutoff)
        & (frame.target_time < cutoff)
        & frame.target.notna()
    ]
    if recipe.get("rolling"):
        training = training.loc[
            training.reference_time
            >= cutoff - pd.Timedelta(days=protocol["research"]["rolling_days"])
        ]
    if recipe.get("reserve_calibration"):
        training = training.loc[
            training.target_time
            < cutoff - pd.Timedelta(days=protocol["research"]["calibration_days"])
        ]
    if (
        state["audit"]["parameters"] != recipe
        or state["audit"]["cutoff"] != cutoff.isoformat()
        or state["audit"]["fit_sha256"] != frame_hash(training[columns + ["target", "target_time"]])
    ):
        raise DataError("Un ajuste cacheado no reproduce el contrato de entrenamiento")
    if state["audit"].get("calibration_sha256"):
        boundary = cutoff - pd.Timedelta(days=protocol["research"]["calibration_days"])
        calibration = frame.loc[
            frame.research_ready
            & (frame.reference_time >= boundary)
            & (frame.reference_time < cutoff)
            & (frame.target_time < cutoff)
            & frame.target.notna()
        ]
        if state["audit"]["calibration_sha256"] != frame_hash(
            calibration[columns + ["target", "target_time"]]
        ):
            raise DataError("Cola de calibración cacheada incompatible")
    if "logistic" in state:
        estimator = state["logistic"]
    else:
        if state.get("model_sha256") and sha256(directory / "model.ubj") != state["model_sha256"]:
            raise DataError("Modelo cacheado alterado")
        estimator = XGBRegressor() if state["regression"] else XGBClassifier()
        estimator.load_model(directory / "model.ubj")
        if estimator.get_booster().feature_names != columns:
            raise DataError("Features de caché incompatibles")
    return Fitted(estimator, columns, state["audit"], state["calibration"], state["regression"])


def fit_recipe(
    frame: pd.DataFrame,
    columns: list[str],
    cutoff: pd.Timestamp,
    recipe: dict[str, Any],
    protocol: dict[str, Any],
    consume_fit: Callable[[], None],
    reused: Fitted | None = None,
) -> Fitted:
    research = protocol["research"]
    train = frame.loc[
        frame.research_ready
        & (frame.reference_time < cutoff)
        & (frame.target_time < cutoff)
        & frame.target.notna()
    ].copy()
    if recipe.get("rolling"):
        train = train.loc[
            train.reference_time >= cutoff - pd.Timedelta(days=research["rolling_days"])
        ]
    calibration = None
    fitting = train
    if recipe.get("reserve_calibration"):
        boundary = cutoff - pd.Timedelta(days=research["calibration_days"])
        fitting = train.loc[train.target_time < boundary]
        calibration = train.loc[train.reference_time >= boundary]
    if len(fitting) < research["minimum_train_rows"] or fitting.target.nunique() != 2:
        raise DataError("Datos insuficientes en entrenamiento avanzado purgado")
    if reused is not None:
        if reused.audit["fit_sha256"] != frame_hash(fitting[columns + ["target", "target_time"]]):
            raise DataError("Reutilización de calibración con train distinto")
        model = reused.model
    elif recipe.get("kind") == "logistic":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(**protocol["logistic"], random_state=research["seed"]),
                ),
            ]
        )
        consume_fit()
        model.fit(fitting[columns], fitting.target.astype(int))
    else:
        parameters = protocol[
            "xgboost_regularized"
            if recipe.get("parameters") == "regularized"
            else "xgboost_reference"
        ].copy()
        if recipe.get("depth"):
            parameters["max_depth"] = recipe["depth"]
        common = {
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": 1,
            "random_state": research["seed"],
        }
        consume_fit()
        if recipe.get("regression"):
            model = XGBRegressor(**parameters, **common, objective="reg:squarederror")
            model.fit(fitting[columns], fitting.net_return_target)
        else:
            model = XGBClassifier(
                **parameters, **common, objective="binary:logistic", eval_metric="logloss"
            )
            model.fit(fitting[columns], fitting.target.astype(int))
    payoff = fitting.dropna(subset=["net_return_target"])
    positive = payoff.loc[payoff.target == 1, "net_return_target"]
    negative = payoff.loc[payoff.target == 0, "net_return_target"]
    audit = {
        "rows": len(fitting),
        "train_last_target": fitting.target_time.max().isoformat(),
        "fit_sha256": frame_hash(fitting[columns + ["target", "target_time"]]),
        "cutoff": cutoff.isoformat(),
        "positive_payoff_mean": float(positive.mean()),
        "negative_payoff_mean": float(negative.mean()),
        "positive_payoff_se": float(positive.std() / np.sqrt(len(positive))),
        "negative_payoff_se": float(negative.std() / np.sqrt(len(negative))),
        "train_prevalence": float(fitting.target.mean()),
        "parameters": recipe,
        "test_or_audited_trades_used_for_selection": False,
        "calibration_rows": len(calibration) if calibration is not None else 0,
    }
    result = Fitted(model, columns, audit, regression=recipe.get("regression", False))
    if result.regression:
        audit["score_scale"] = max(float(fitting.net_return_target.std()), 1e-6)
    if recipe.get("calibrated") or result.regression:
        if calibration is None or len(calibration) < 30 or calibration.target.nunique() != 2:
            raise DataError("Calibración avanzada insuficiente")
        consume_fit()
        result.calibration = fit_sigmoid(
            result.probabilities(calibration), calibration.target.to_numpy(), research["seed"]
        )
        audit["calibration_first_reference"] = calibration.reference_time.min().isoformat()
        audit["calibration_last_target"] = calibration.target_time.max().isoformat()
        audit["calibration_sha256"] = frame_hash(calibration[columns + ["target", "target_time"]])
    audit["training_metrics"] = classification_metrics(
        fitting.target.to_numpy(), result.probabilities(fitting)
    )
    return result
