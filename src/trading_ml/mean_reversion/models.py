from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from trading_ml.common import frame_hash, read_json, sha256, write_new_json
from trading_ml.errors import DataError
from trading_ml.mean_reversion.features import reversion_columns


@dataclass
class ReversionFilter:
    kind: str
    estimator: Pipeline | XGBClassifier | dict[str, Any] | None
    audit: dict[str, Any]

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.zeros(len(frame))  # Abstention sentinel, not an estimated probability.
        ready = frame.mr_ready.to_numpy()
        if self.estimator is None or not ready.any():
            return result
        values = frame.loc[ready, reversion_columns()]
        if not np.isfinite(values.to_numpy()).all():
            raise DataError("Features Mean Reversion no finitas")
        if isinstance(self.estimator, dict):
            state = self.estimator
            standardized = (values.to_numpy() - np.array(state["scaler_mean"])) / np.array(
                state["scaler_scale"]
            )
            logits = standardized @ np.array(state["coefficients"]) + state["intercept"]
            result[ready] = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        else:
            result[ready] = self.estimator.predict_proba(values)[:, 1]
        return result

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        state = {"kind": self.kind, "features": reversion_columns(), "audit": self.audit}
        if self.estimator is not None:
            if self.kind == "logistic":
                if isinstance(self.estimator, dict):
                    state["logistic"] = self.estimator
                else:
                    scaler = self.estimator.named_steps["scale"]
                    model = self.estimator.named_steps["logistic"]
                    state["logistic"] = {
                        "scaler_mean": scaler.mean_.tolist(),
                        "scaler_scale": scaler.scale_.tolist(),
                        "coefficients": model.coef_[0].tolist(),
                        "intercept": float(model.intercept_[0]),
                        "classes": model.classes_.tolist(),
                        "iterations": model.n_iter_.tolist(),
                    }
            else:
                self.estimator.save_model(directory / "model.ubj")
                state["model_sha256"] = sha256(directory / "model.ubj")
        write_new_json(directory / "filter.json", state)


def load_filter(directory: Path) -> ReversionFilter:
    state = read_json(directory / "filter.json")
    if state["features"] != reversion_columns():
        raise DataError("Allowlist Mean Reversion incompatible")
    estimator = None
    if state["audit"]["status"] == "fitted":
        if state["kind"] == "logistic":
            estimator = state["logistic"]
            if (
                estimator["classes"] != [0, 1]
                or not (np.array(estimator["scaler_scale"]) > 0).all()
            ):
                raise DataError("Estado logístico inválido")
        elif state["kind"] == "xgb":
            if sha256(directory / "model.ubj") != state["model_sha256"]:
                raise DataError("XGBoost Mean Reversion alterado")
            estimator = XGBClassifier()
            estimator.load_model(directory / "model.ubj")
            if estimator.get_booster().feature_names != reversion_columns():
                raise DataError("Features del modelo de reversión incompatibles")
        else:
            raise DataError("Tipo de filtro de reversión desconocido")
    return ReversionFilter(state["kind"], estimator, state["audit"])


def fit_filter(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    cutoff: pd.Timestamp,
    kind: str,
    parameters: dict[str, Any],
    minimum_rows: int = 20,
    seed: int = 42,
) -> ReversionFilter:
    if kind not in {"logistic", "xgb"}:
        raise DataError("Filtro Mean Reversion desconocido")
    usable = labels.loc[
        (labels.label_status == "observed")
        & (labels.reference_time < cutoff)
        & (labels.target_time < cutoff)
    ].copy()
    values = frame.loc[usable.source_index, reversion_columns()]
    if not np.isfinite(values.to_numpy()).all():
        raise DataError("Features de ajuste de reversión faltantes")
    audit = {
        "status": "abstain_insufficient_opportunities_or_single_class",
        "rows": len(usable),
        "positive_labels": int(usable.target.sum()),
        "train_prevalence": float(usable.target.mean()) if len(usable) else None,
        "last_target_time": usable.target_time.max().isoformat() if len(usable) else None,
        "cutoff": cutoff.isoformat(),
        "feature_sha256": frame_hash(values),
        "labels_sha256": frame_hash(usable),
        "parameters": parameters,
        "test_used_for_fitting_or_selection": False,
        "probability_target": "anchored_mean_hit_before_stop_or_timeout",
    }
    if len(usable) < minimum_rows or usable.target.nunique() != 2:
        return ReversionFilter(kind, None, audit)
    if kind == "logistic":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(**parameters, solver="lbfgs", random_state=seed)),
            ]
        )
    else:
        estimator = XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=seed,
        )
    estimator.fit(values, usable.target.astype(int))
    audit["status"] = "fitted"
    if kind == "logistic":
        audit["converged"] = bool(
            estimator.named_steps["logistic"].n_iter_.max() < parameters["max_iter"]
        )
    return ReversionFilter(kind, estimator, audit)
