from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from trading_ml.config.settings import INTERVALS
from trading_ml.errors import DataError
from trading_ml.online.config import utc

ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
]
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_candles(frame: pd.DataFrame, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(interval, str) or interval not in INTERVALS:
        raise DataError("Intervalo de mercado no admitido")
    now = utc(now)
    if not set(COLUMNS) <= set(frame.columns):
        raise DataError("Fuente debe contener timestamp, open, high, low, close, volume")
    data = frame[COLUMNS].copy()
    try:
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True, errors="raise")
        if data.timestamp.isna().any():
            raise ValueError("Timestamps ausentes")
        delta = pd.Timedelta(INTERVALS[interval])
        # No utilizar OHLCV de velas abiertas, ni siquiera para validación/features.
        data = data.loc[data.timestamp + delta <= now].copy()
        for column in COLUMNS[1:]:
            if data[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
                raise ValueError("OHLCV booleano no admitido")
            data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise DataError("OHLCV/timestamps inválidos") from exc
    if data.timestamp.isna().any() or data.timestamp.duplicated().any():
        raise DataError("Timestamps ausentes o duplicados")
    values = data[COLUMNS[1:]].to_numpy()
    if (
        not np.isfinite(values).all()
        or (data[COLUMNS[1:5]] <= 0).any().any()
        or (data.volume < 0).any()
    ):
        raise DataError("OHLCV debe ser finito, precios positivos y volumen no negativo")
    if (
        (data.high < data[["open", "close", "low"]].max(axis=1))
        | (data.low > data[["open", "close", "high"]].min(axis=1))
    ).any():
        raise DataError("Rangos OHLC inconsistentes")
    if any(stamp != stamp.floor(delta) for stamp in data.timestamp):
        raise DataError("Velas desalineadas con la grilla UTC")
    return data.sort_values("timestamp").reset_index(drop=True)


def fetch_candles(config: dict, now: pd.Timestamp, last_open: str | None) -> pd.DataFrame:
    market = config["market"]
    now = utc(now)
    # Red de solo lectura. Sin credenciales, fallback histórico ni endpoints de órdenes.
    params = {
        "symbol": market["symbol"],
        "interval": market["interval"],
        "limit": 1000,
        "endTime": int(now.timestamp() * 1000) - 1,
    }
    if last_open is not None:
        params["startTime"] = int(utc(last_open).timestamp() * 1000)
    records = []
    for _page in range(config["online"]["max_fetch_pages"]):
        last_error = None
        for endpoint in ENDPOINTS:
            try:
                response = requests.get(endpoint, params=params, timeout=(5, 15))
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or any(
                    not isinstance(row, list) or len(row) < 6 for row in payload
                ):
                    raise DataError("Respuesta de mercado inválida")
                timestamps = [row[0] for row in payload]
                if any(type(value) is not int or value < 0 for value in timestamps):
                    raise DataError("Timestamps REST deben ser enteros de milisegundos")
                if any(
                    left >= right for left, right in zip(timestamps, timestamps[1:], strict=False)
                ):
                    raise DataError("Página REST desordenada o duplicada")
                if payload and "startTime" in params and timestamps[0] < params["startTime"]:
                    raise DataError("El proveedor no respetó el cursor de paginación")
                break
            except requests.RequestException as exc:
                last_error = exc
        else:
            raise ConnectionError(
                f"Proveedores públicos no disponibles: {last_error}"
            ) from last_error
        records.extend(row[:6] for row in payload)
        if len(payload) < 1000 or last_open is None:
            break
        params["startTime"] = int(payload[-1][0]) + 1
    frame = pd.DataFrame(records, columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    return validate_candles(frame, market["interval"], now)


def features(window: pd.DataFrame) -> list[float]:
    """Fixed, bounded causal transforms; no scaler fitted on future rows."""
    close = window.close.to_numpy(dtype=float)
    last = window.iloc[-1]
    returns = np.diff(np.log(close))
    volume_mean = float(window.volume.mean())
    result = [
        np.log(close[-1] / close[-2]) * 50,
        np.log(close[-1] / close[-4]) * 25,
        np.log(close[-1] / close[-8]) * 15,
        (close[-1] / close.mean() - 1) * 20,
        np.std(returns) * 50,
        (last.high / last.low - 1) * 25,
        (last.close / last.open - 1) * 50,
        (last.volume / volume_mean - 1) if volume_mean > 0 else 0,
    ]
    return np.clip(result, -10, 10).astype(float).tolist()
