from __future__ import annotations

import pandas as pd

from trading_ml.errors import DataError


def temporal_partitions(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    *,
    embargo_days: int = 6,
    calibration_days: int = 120,
    rolling_days: int | None = None,
):
    if embargo_days < 6 or calibration_days < 60:
        raise DataError("Embargo/calibración inferiores al protocolo")
    boundary = start - pd.Timedelta(days=calibration_days)
    train = frame.loc[
        frame.research_ready
        & frame.target.notna()
        & (frame.reference_time < boundary)
        & (frame.target_time < boundary - pd.Timedelta(days=embargo_days))
    ].copy()
    calibration = frame.loc[
        frame.research_ready
        & frame.target.notna()
        & (frame.reference_time >= boundary)
        & (frame.target_time < start - pd.Timedelta(days=embargo_days))
    ].copy()
    if rolling_days:
        train = train.loc[train.reference_time >= start - pd.Timedelta(days=rolling_days)]
    if len(train) < 120 or len(calibration) < 40 or train.target.nunique() < 2:
        raise DataError("Particiones temporales insuficientes")
    return train, calibration


def evaluation_rows(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    # Targets that cross a fold boundary cannot enter rankings or bootstrap.
    return frame.loc[
        frame.research_ready
        & frame.target.notna()
        & (frame.reference_time >= start)
        & (frame.reference_time < end)
        & (frame.target_time < end)
    ].copy()


def prediction_rows(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    """Execution availability must never depend on a future label or its end."""
    return frame.loc[
        frame.research_ready & (frame.reference_time >= start) & (frame.reference_time < end)
    ].copy()
