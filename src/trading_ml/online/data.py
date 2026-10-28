from __future__ import annotations

import math

import numpy as np
import pandas as pd
import requests

from trading_ml.config.settings import INTERVALS
from trading_ml.errors import DataError
from trading_ml.online.config import FEATURE_SETS, utc

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


def _financial_features(window: pd.DataFrame) -> list[float]:
    """Compute causal microstructure liquidity & Parkinson volatility features on closed bars."""
    if len(window) < 24:
        raise DataError("v2_financial11 requiere al menos 24 velas cerradas en la ventana")
    high = window.high.to_numpy(dtype=float)
    low = window.low.to_numpy(dtype=float)
    close = window.close.to_numpy(dtype=float)
    volume = window.volume.to_numpy(dtype=float)

    # 1. Corwin-Schultz (2012) high-low spread estimator (20-pair causal average)
    hl_tail = min(len(window), 21)
    h_sub = high[-hl_tail:]
    l_sub = low[-hl_tail:]
    log_hl_sq = np.log(h_sub / l_sub) ** 2
    beta = log_hl_sq[:-1] + log_hl_sq[1:]
    h_max = np.maximum(h_sub[:-1], h_sub[1:])
    l_min = np.minimum(l_sub[:-1], l_sub[1:])
    gamma = np.log(h_max / l_min) ** 2
    denom = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    exp_a = np.exp(np.clip(alpha, -10.0, 10.0))
    spreads = np.maximum(0.0, 2.0 * (exp_a - 1.0) / (1.0 + exp_a))
    cs_spread_scaled = float(np.mean(spreads) * 50.0)

    # 2. Range-to-dollar-volume liquidity impact z-score (up to 64 bars causal window)
    rel_range = (high - low) / np.maximum(close, 1e-8)
    dollar_vol = close * np.maximum(volume, 0.0)
    log_impact = np.log((rel_range + 1e-8) / (dollar_vol + 1.0))
    imp_std = float(np.std(log_impact))
    imp_mean = float(np.mean(log_impact))
    impact_z = float((log_impact[-1] - imp_mean) / imp_std) if imp_std > 1e-8 else 0.0

    # 3. Parkinson short-to-long range volatility ratio (6-bar vs 24-bar)
    park_Factor = 1.0 / (4.0 * math.log(2.0))
    park_bars = park_Factor * (np.log(high / low) ** 2)
    short_vol = math.sqrt(max(float(np.mean(park_bars[-6:])), 1e-12))
    long_vol = math.sqrt(max(float(np.mean(park_bars[-24:])), 1e-12))
    park_ratio = (short_vol / long_vol - 1.0) * 5.0 if long_vol > 1e-10 else 0.0

    return [cs_spread_scaled, impact_z, park_ratio]


def features(window: pd.DataFrame, feature_set: str = "v1_base8") -> list[float]:
    """Fixed, bounded causal transforms; no scaler fitted on future rows."""
    if feature_set not in FEATURE_SETS:
        raise DataError(f"feature_set desconocido: {feature_set}")
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
    if feature_set == "v2_financial11":
        result.extend(_financial_features(window))
    return np.clip(result, -10, 10).astype(float).tolist()


def realized_target_metrics(
    reference_close: float,
    target_bars: pd.DataFrame,
) -> tuple[float, float]:
    """Compute causal confirmed log-return and blended Parkinson/squared-return realized variance."""
    target_close = float(target_bars["close"].iloc[-1])
    log_ret = math.log(target_close / float(reference_close))
    high_max = float(target_bars["high"].max())
    low_min = float(target_bars["low"].min())
    park_var = (math.log(high_max / low_min) ** 2) / (4.0 * math.log(2.0)) if low_min > 0 else 0.0
    blended_rv = max(1e-8, 0.5 * (log_ret * log_ret) + 0.5 * park_var)
    return float(log_ret), float(blended_rv)
