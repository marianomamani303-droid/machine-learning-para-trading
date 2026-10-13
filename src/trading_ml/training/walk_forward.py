from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import frame_hash
from trading_ml.errors import DataError, LeakageError


@dataclass(frozen=True)
class WalkForwardConfig:
    boundaries: list[str]
    inner_validation_days: int = 90
    early_stopping_rounds: int = 25
    min_train_rows: int = 365
    min_eval_rows: int = 60

    def validate(self) -> None:
        dates = pd.to_datetime(self.boundaries, utc=True)
        if len(dates) < 3 or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise DataError("Walk-forward requiere al menos dos ventanas consecutivas ordenadas")
        if (
            min(
                self.inner_validation_days,
                self.early_stopping_rounds,
                self.min_train_rows,
                self.min_eval_rows,
            )
            < 1
        ):
            raise DataError("Tamaños walk-forward inválidos")


@dataclass
class WalkForwardFold:
    number: int
    start: pd.Timestamp
    end: pd.Timestamp
    train: pd.DataFrame
    evaluation: pd.DataFrame


def walk_forward_folds(
    frame: pd.DataFrame,
    config: WalkForwardConfig,
    final_start: pd.Timestamp,
    training_start: pd.Timestamp,
    rolling_days: int | None = None,
) -> list[WalkForwardFold]:
    """Evalúa ventanas no superpuestas; cada etiqueta de train termina antes del fold.

    El test final se elimina antes de producir cualquier fold o estadística.
    """
    config.validate()
    if rolling_days is not None and rolling_days < 1:
        raise DataError("Ventana móvil inválida")
    dates = pd.to_datetime(config.boundaries, utc=True)
    if dates[-1] > final_start:
        raise LeakageError("Los folds no pueden entrar en el test final")
    if frame.timestamp.duplicated().any() or not frame.reference_time.is_monotonic_increasing:
        raise DataError("Timeline walk-forward inválido")
    development = frame.loc[
        (frame.reference_time < final_start) & (frame.target_time < final_start)
    ].dropna(subset=["target", "target_time"])
    folds = []
    for number, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True), 1):
        lower = (
            max(training_start, start - pd.Timedelta(days=rolling_days))
            if rolling_days is not None
            else training_start
        )
        train = development.loc[
            (development.reference_time >= lower) & (development.target_time < start)
        ].copy()
        evaluation = development.loc[
            (development.reference_time >= start)
            & (development.reference_time < end)
            & (development.target_time < end)
        ].copy()
        if (
            len(train) < config.min_train_rows
            or len(evaluation) < config.min_eval_rows
            or train.target.nunique() != 2
        ):
            raise DataError(f"Fold {number} insuficiente después de purga")
        folds.append(WalkForwardFold(number, start, end, train, evaluation))
    return folds


def fit_temporal_model(
    train: pd.DataFrame,
    columns: list[str],
    parameters: dict[str, Any],
    seed: int,
    config: WalkForwardConfig,
    early_stop: bool,
) -> tuple[XGBClassifier, dict[str, Any]]:
    """Early stopping usa sólo una cola interior de train, luego refit de train.

    No consulta el fold externo ni el test final para elegir cantidad de árboles.
    """
    if train.empty or train.target.nunique() != 2:
        raise DataError("Entrenamiento temporal vacío o de una sola clase")
    common = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "random_state": seed,
        "n_jobs": 1,
    }
    chosen = parameters.copy()
    details: dict[str, Any] = {
        "train_rows": len(train),
        "train_last_target": train.target_time.max().isoformat(),
        "early_stopping": early_stop,
        "fit_data_sha256": frame_hash(train),
    }
    if early_stop:
        inner_start = train.reference_time.max() - pd.Timedelta(days=config.inner_validation_days)
        fitting = train.loc[train.target_time < inner_start]
        validation = train.loc[train.reference_time >= inner_start]
        if (
            len(fitting) < config.min_train_rows
            or len(validation) < 20
            or fitting.target.nunique() != 2
        ):
            raise DataError("Datos insuficientes para validación interior purgada")
        probe = XGBClassifier(
            **chosen, **common, early_stopping_rounds=config.early_stopping_rounds
        )
        probe.fit(
            fitting[columns],
            fitting.target.astype(int),
            eval_set=[(validation[columns], validation.target.astype(int))],
            verbose=False,
        )
        chosen["n_estimators"] = int(probe.best_iteration) + 1
        details.update(
            {
                "inner_fit_rows": len(fitting),
                "inner_validation_rows": len(validation),
                "inner_fit_last_target": fitting.target_time.max().isoformat(),
                "inner_validation_first_reference": validation.reference_time.min().isoformat(),
                "inner_validation_last_target": validation.target_time.max().isoformat(),
                "best_inner_log_loss": float(probe.best_score),
                "inner_fit_sha256": frame_hash(fitting),
                "inner_validation_sha256": frame_hash(validation),
            }
        )
    model = XGBClassifier(**chosen, **common)
    model.fit(train[columns], train.target.astype(int))
    details["effective_parameters"] = chosen
    return model, details
