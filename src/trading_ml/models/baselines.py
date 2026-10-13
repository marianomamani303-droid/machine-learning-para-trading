from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trading_ml.common import frame_hash
from trading_ml.errors import DataError

BASELINE_NAMES = ("constant_05", "train_prevalence", "logistic_l2")


@dataclass
class ReferenceModels:
    columns: list[str]
    prevalence: float
    scaler: StandardScaler
    logistic: LogisticRegression
    fit_sha256: str

    def probabilities(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        values = frame[self.columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataError("Features no finitas en baseline")
        return {
            "constant_05": np.full(len(frame), 0.5),
            "train_prevalence": np.full(len(frame), self.prevalence),
            "logistic_l2": self.logistic.predict_proba(self.scaler.transform(values))[:, 1],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.columns,
            "train_prevalence": self.prevalence,
            "fit_sha256": self.fit_sha256,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "coefficients": self.logistic.coef_[0].tolist(),
            "intercept": float(self.logistic.intercept_[0]),
            "C": float(self.logistic.C),
            "solver": self.logistic.solver,
            "iterations": self.logistic.n_iter_.tolist(),
        }


def fit_reference_models(
    train: pd.DataFrame,
    columns: list[str],
    *,
    seed: int = 42,
    C: float = 1.0,
) -> ReferenceModels:
    """All fitted quantities, including scaling and prevalence, use train only."""
    if not np.isfinite(C) or C <= 0 or train.empty or train.target.nunique() != 2:
        raise DataError("Baseline requiere train con ambas clases y C positivo")
    values = train[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataError("Features no finitas en train del baseline")
    scaler = StandardScaler().fit(values)
    logistic = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=seed)
    logistic.fit(scaler.transform(values), train.target.astype(int))
    return ReferenceModels(
        columns.copy(), float(train.target.mean()), scaler, logistic, frame_hash(train)
    )
