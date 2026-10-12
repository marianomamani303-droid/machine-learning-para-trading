from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_ml.errors import DataError

OHLCV = ["open", "high", "low", "close", "volume"]


def clean_ohlcv(frame: pd.DataFrame, delta: pd.Timedelta) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ordena y elimina duplicados idénticos; falla ante conflictos, NaN o huecos.

    Nunca interpola precios: imputarlos inventaría rentabilidades negociables.
    El calendario continuo es específico al mercado cripto elegido.
    """
    required = ["timestamp", *OHLCV]
    if not set(required).issubset(frame.columns):
        raise DataError(f"Faltan columnas OHLCV: {set(required) - set(frame.columns)}")
    frame = frame[required].copy()
    frame.timestamp = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    frame[OHLCV] = frame[OHLCV].apply(pd.to_numeric, errors="coerce")
    invalid = frame.isna().any(axis=1) | ~np.isfinite(frame[OHLCV]).all(axis=1)
    if invalid.any():
        raise DataError(f"Valores faltantes/no finitos en {int(invalid.sum())} filas")
    duplicates = frame[frame.timestamp.duplicated(keep=False)]
    if not duplicates.empty and (duplicates.groupby("timestamp")[OHLCV].nunique() > 1).any().any():
        raise DataError("Duplicados con OHLCV contradictorios")
    removed = int(frame.timestamp.duplicated().sum())
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 2:
        raise DataError("Se requieren al menos dos velas")
    price_bad = (frame[["open", "high", "low", "close"]] <= 0).any(axis=1) | (frame.volume < 0)
    price_bad |= (frame.high < frame[["open", "close", "low"]].max(axis=1)) | (
        frame.low > frame[["open", "close", "high"]].min(axis=1)
    )
    if price_bad.any():
        raise DataError(f"OHLCV inválido en {int(price_bad.sum())} filas")
    if (frame.timestamp != frame.timestamp.dt.floor(delta)).any():
        raise DataError("Timestamps desalineados respecto del intervalo UTC")
    diffs = frame.timestamp.diff().iloc[1:]
    gaps = diffs != delta
    if gaps.any():
        raise DataError(f"Se detectaron {int(gaps.sum())} discontinuidades; no se imputan velas")
    return frame, {
        "rows": len(frame),
        "duplicates_removed": removed,
        "missing_values": 0,
        "gaps": 0,
        "zero_volume_rows": int((frame.volume == 0).sum()),
        "first_timestamp": frame.timestamp.iloc[0].isoformat(),
        "last_timestamp": frame.timestamp.iloc[-1].isoformat(),
    }
