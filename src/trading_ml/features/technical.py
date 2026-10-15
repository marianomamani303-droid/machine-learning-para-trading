from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.config.settings import Features
from trading_ml.smc.detectors import SMCConfig, detect_smc, smc_columns


def feature_columns(config: Features) -> list[str]:
    columns = [
        "return_1",
        "return_3",
        "return_7",
        "range_ratio",
        "body_ratio",
        "volume_ratio",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility",
        *[f"sma_ratio_{window}" for window in config.sma_windows],
    ]
    if config.enhanced:
        columns.extend(
            [
                "return_21",
                "return_lag_1",
                "return_lag_2",
                "momentum_vol_ratio",
                "volatility_ratio",
                "atr_ratio",
                "close_location",
                "volume_change_1",
            ]
        )
    if config.smc:
        columns.extend(smc_columns())
    return columns


def build_features(
    frame: pd.DataFrame, config: Features, *, include_volume: bool = True
) -> pd.DataFrame:
    """Indicadores disponibles al cierre de cada vela; sin ajustes globales."""
    result = frame.copy()
    close = frame.close
    for lag in [1, 3, 7]:
        result[f"return_{lag}"] = close.pct_change(lag, fill_method=None)
    result["range_ratio"] = (frame.high - frame.low) / close
    result["body_ratio"] = (close - frame.open) / frame.open
    if include_volume:
        result["volume_ratio"] = frame.volume / frame.volume.rolling(
            20, min_periods=20
        ).mean().replace(0, np.nan)
    change = close.diff()
    gain = (
        change.clip(lower=0)
        .ewm(alpha=1 / config.rsi_window, adjust=False, min_periods=config.rsi_window)
        .mean()
    )
    loss = (
        (-change.clip(upper=0))
        .ewm(alpha=1 / config.rsi_window, adjust=False, min_periods=config.rsi_window)
        .mean()
    )
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    result["rsi"] = rsi / 100
    fast = close.ewm(span=config.macd_fast, adjust=False, min_periods=config.macd_fast).mean()
    slow = close.ewm(span=config.macd_slow, adjust=False, min_periods=config.macd_slow).mean()
    macd = fast - slow
    signal = macd.ewm(span=config.macd_signal, adjust=False, min_periods=config.macd_signal).mean()
    result["macd"] = macd / close
    result["macd_signal"] = signal / close
    result["macd_hist"] = (macd - signal) / close
    result["volatility"] = result.return_1.rolling(
        config.volatility_window, min_periods=config.volatility_window
    ).std()
    for window in config.sma_windows:
        result[f"sma_ratio_{window}"] = close / close.rolling(window, min_periods=window).mean() - 1
    if config.enhanced:
        result["return_21"] = close.pct_change(21, fill_method=None)
        result["return_lag_1"] = result.return_1.shift(1)
        result["return_lag_2"] = result.return_1.shift(2)
        short_vol = result.return_1.rolling(5, min_periods=5).std()
        long_vol = result.return_1.rolling(60, min_periods=60).std()
        result["momentum_vol_ratio"] = result.return_7 / (
            result.volatility.replace(0, np.nan) * np.sqrt(7)
        )
        result["volatility_ratio"] = short_vol / long_vol.replace(0, np.nan)
        true_range = pd.concat(
            [
                frame.high - frame.low,
                (frame.high - close.shift(1)).abs(),
                (frame.low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        result["atr_ratio"] = true_range.rolling(14, min_periods=14).mean() / close
        result["close_location"] = (
            (close - frame.low) / (frame.high - frame.low).replace(0, np.nan)
        ).fillna(0.5)
        if include_volume:
            result["volume_change_1"] = np.log1p(frame.volume).diff()
    if config.smc:
        patterns = detect_smc(frame, SMCConfig.from_features(config))
        result = pd.concat([result, patterns], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)
