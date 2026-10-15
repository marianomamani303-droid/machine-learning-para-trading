from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from trading_ml.advanced_research.models import Fitted
from trading_ml.common import frame_hash
from trading_ml.errors import DataError
from trading_ml.label_research.events import average_uniqueness
from trading_ml.models.calibration import fit_sigmoid
from trading_ml.training.recipe import protect_reserved_labels


class Budget:
    def __init__(self, maximum):
        self.maximum, self.actual = maximum, 0

    def consume(self):
        if self.actual >= self.maximum:
            raise DataError("Presupuesto de ajustes del estudio de etiquetas agotado")
        self.actual += 1


def partitions(frame, start, research, *, reserve=True):
    valid = frame.research_ready & frame.target.notna() & frame.target_time.notna()
    boundary = start - pd.Timedelta(days=research["calibration_days"])
    embargo = pd.Timedelta(days=research["embargo_days"])
    fitting_end = boundary if reserve else start
    train = frame.loc[
        valid & (frame.reference_time < fitting_end) & (frame.target_time < fitting_end - embargo)
    ].copy()
    calibration = (
        frame.loc[
            valid & (frame.reference_time >= boundary) & (frame.target_time < start - embargo)
        ].copy()
        if reserve
        else frame.iloc[:0].copy()
    )
    return train, calibration


def fit_event_model(
    frame,
    start,
    columns,
    protocol,
    parameters,
    budget,
    root,
    *,
    reserve=True,
    weighted=False,
    kind="xgboost",
    meta=False,
):
    r = protocol["research"]
    train, calibration = partitions(frame, start, r, reserve=reserve)
    protect_reserved_labels(root, train, calibration)
    minimum = r["minimum_meta_train_rows"] if meta else r["minimum_train_rows"]
    valid = (
        len(train) >= minimum
        and train.target.nunique() == 2
        and train.target.value_counts().min() >= 5
    )
    if not valid:
        if meta:
            return None, {
                "status": "insufficient_meta_training",
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "cutoff": start.isoformat(),
            }
        raise DataError("Etiquetas insuficientes para el ajuste purgado")
    train_weights = average_uniqueness(train) if weighted else None
    budget.consume()
    if kind == "logistic":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=0.1 if meta else 1.0, max_iter=2000, random_state=r["seed"]
                    ),
                ),
            ]
        )
        model.fit(train[columns], train.target.astype(int))
    else:
        model = XGBClassifier(
            **(
                parameters
                | {
                    "random_state": r["seed"],
                    "n_jobs": 1,
                    "tree_method": "hist",
                    "device": "cpu",
                    "eval_metric": "logloss",
                }
            )
        )
        model.fit(train[columns], train.target.astype(int), sample_weight=train_weights)
    positive = train.loc[train.target == 1, "unit_net_return"]
    negative = train.loc[train.target == 0, "unit_net_return"]
    audit = {
        "status": "fitted",
        "cutoff": start.isoformat(),
        "kind": kind,
        "meta": meta,
        "train_rows": len(train),
        "train_prior": float(train.target.mean()),
        "train_target_end": train.target_time.max().isoformat(),
        "train_first_reference": train.reference_time.min().isoformat(),
        "fit_sha256": frame_hash(train[columns + ["target", "target_time"]]),
        "positive_payoff_mean": float(positive.mean()),
        "negative_payoff_mean": float(negative.mean()),
        "calibration_rows": len(calibration),
        "calibration_status": "not_requested" if not reserve else "insufficient",
        "parameters": parameters if kind == "xgboost" else {"C": 0.1 if meta else 1.0},
        "weighted_by_train_uniqueness": weighted,
        "weights": {"min": float(train_weights.min()), "max": float(train_weights.max())}
        if weighted
        else None,
    }
    calibration_state = None
    required = r["minimum_meta_calibration_rows"] if meta else 20
    if reserve and len(calibration) >= required and calibration.target.nunique() == 2:
        budget.consume()
        calibration_state = fit_sigmoid(
            model.predict_proba(calibration[columns])[:, 1],
            calibration.target.to_numpy(dtype=int),
            r["seed"],
        )
        audit |= {
            "calibration_status": "fitted",
            "calibration_target_end": calibration.target_time.max().isoformat(),
            "calibration_first_reference": calibration.reference_time.min().isoformat(),
            "calibration_sha256": frame_hash(calibration[columns + ["target", "target_time"]]),
        }
    result = Fitted(model, columns, audit, calibration_state)
    return result, audit


def forecasts(fitted, frame):
    ready = (
        frame.research_ready & frame[fitted.columns].notna().all(axis=1)
        if fitted
        else np.zeros(len(frame), dtype=bool)
    )
    raw, calibrated = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    if fitted is not None and ready.any():
        raw[ready] = replace(fitted, calibration=None).probabilities(frame.loc[ready])
        # No calibration data is an explicit abstention for calibrated recipes,
        # not an unannounced substitution of raw probabilities.
        if fitted.calibration is not None:
            calibrated[ready] = fitted.probabilities(frame.loc[ready])
    return raw, calibrated


def meta_pool(frame, oof, events, start, threshold=0.55):
    """Only ex-ante base offers create examples, whether or not a portfolio was busy."""
    offered = oof.loc[(oof.probability >= threshold) & (oof.reference_time < start)]
    if offered.source_index.duplicated().any():
        raise DataError("Predicciones OOF base duplicadas")
    if not {"model_cutoff", "last_train_target"}.issubset(offered):
        raise DataError("Falta procedencia temporal de predicciones OOF")
    cutoff = pd.to_datetime(offered.model_cutoff, utc=True)
    last_label = pd.to_datetime(offered.last_train_target, utc=True)
    if ((last_label >= cutoff) | (cutoff > offered.reference_time)).any():
        raise DataError("Meta-Labeling exige probabilidades base fuera de muestra")
    if not np.array_equal(
        frame.loc[offered.source_index, "reference_time"].to_numpy(),
        offered.reference_time.to_numpy(),
    ):
        raise DataError("Pronósticos OOF desalineados con las features")
    pool = frame.loc[offered.source_index].copy()
    pool["base_probability"] = offered.probability.to_numpy()
    indices = offered.source_index.to_numpy()
    for column in ["target", "target_time", "unit_net_return"]:
        pool[column] = events.loc[indices, column].to_numpy()
    pool.target_time = pd.to_datetime(pool.target_time, utc=True)
    pool["oof_model_file"] = offered.model_file.to_numpy()
    pool.research_ready &= pool.base_probability.notna()
    return pool
