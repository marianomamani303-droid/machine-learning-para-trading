from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from trading_ml.errors import DataError


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    if (
        probability.ndim != 1
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise DataError("Probabilidades inválidas para calibración")
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def fit_sigmoid(probability: np.ndarray, target: np.ndarray, seed: int = 42) -> dict[str, Any]:
    """Ajuste logístico regularizado; requiere datos ajenos al fit de XGBoost.

    Si el ajuste intenta invertir el ranking se colapsa a la prevalencia de
    calibración, para no convertir ruido de una cola corta en señal inversa.
    """
    logits = _logit(probability)
    target = np.asarray(target)
    if (
        target.ndim != 1
        or len(target) != len(logits)
        or len(target) < 20
        or not np.isin(target, [0, 1]).all()
        or len(np.unique(target)) != 2
    ):
        raise DataError("Calibración requiere al menos 20 filas y ambas clases")
    estimator = LogisticRegression(C=1.0, random_state=seed, solver="lbfgs")
    estimator.fit(logits[:, None], target.astype(int))
    slope = float(estimator.coef_[0, 0])
    intercept = float(estimator.intercept_[0])
    collapsed = slope < 0
    if collapsed:
        slope = 0.0
        prevalence = float(target.mean())
        intercept = float(np.log(prevalence / (1 - prevalence)))
    return {
        "method": "sigmoid",
        "slope": slope,
        "intercept": intercept,
        "rows": len(target),
        "ranking_collapsed": collapsed,
        "regularization_C": 1.0,
    }


def apply_calibration(probability: np.ndarray, calibration: dict[str, Any] | None) -> np.ndarray:
    if calibration is None:
        return np.asarray(probability, dtype=float)
    if (
        calibration.get("method") != "sigmoid"
        or not np.isfinite([calibration["slope"], calibration["intercept"]]).all()
        or calibration["slope"] < 0
    ):
        raise DataError("Artefacto de calibración inválido")
    z = np.clip(calibration["slope"] * _logit(probability) + calibration["intercept"], -35, 35)
    return 1 / (1 + np.exp(-z))
