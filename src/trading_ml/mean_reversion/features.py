from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.mean_reversion.config import ReversionConfig
from trading_ml.momentum.features import causal_regimes


def reversion_columns() -> list[str]:
    return [
        "mr_z",
        "mr_log_z",
        "mr_distance",
        "mr_std_ratio",
        "mr_rsi",
        "mr_volatility",
        "mr_volatility_ratio",
        "mr_atr_ratio",
        "mr_return_1",
        "mr_return_3",
        "mr_return_7",
        "mr_mean_slope",
        "mr_return_autocorrelation",
        "mr_down_fraction",
        "mr_regime_score",
    ]


def build_reversion_features(frame: pd.DataFrame, config: ReversionConfig) -> pd.DataFrame:
    """Reference distribution excludes current close; no global fitted quantities."""
    config.validate()
    result = frame.copy()
    close = frame.close
    prior = close.shift(1)
    mean = prior.rolling(config.lookback).mean()
    std = prior.rolling(config.lookback).std(ddof=1)
    result["mr_mean"] = mean
    result["mr_std"] = std
    result["mr_z"] = (close - mean) / std.replace(0, np.nan)
    result["mr_z"] = result.mr_z.mask((std == 0) & (close == mean), 0)
    log_close = np.log(close)
    log_mean = log_close.shift(1).rolling(config.lookback).mean()
    log_std = log_close.shift(1).rolling(config.lookback).std(ddof=1)
    result["mr_log_z"] = (log_close - log_mean) / log_std.replace(0, np.nan)
    result["mr_log_z"] = result.mr_log_z.mask((log_std == 0) & (log_close == log_mean), 0)
    result["mr_distance"] = close / mean - 1
    result["mr_std_ratio"] = std / mean
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
    rsi = 1 - 1 / (1 + gain / loss.replace(0, np.nan))
    result["mr_rsi"] = rsi.mask((loss == 0) & (gain > 0), 1).mask((loss == 0) & (gain == 0), 0.5)
    log_returns = log_close.diff()
    vol = log_returns.rolling(config.volatility_window).std()
    result["mr_volatility"] = vol
    long_vol = log_returns.rolling(60).std()
    result["mr_volatility_ratio"] = (vol / long_vol.replace(0, np.nan)).mask(
        (vol == 0) & (long_vol == 0), 1
    )
    tr = pd.concat(
        [
            frame.high - frame.low,
            (frame.high - close.shift(1)).abs(),
            (frame.low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["mr_atr"] = tr.rolling(config.atr_window).mean()
    result["mr_atr_ratio"] = result.mr_atr / close
    for lag in [1, 3, 7]:
        result[f"mr_return_{lag}"] = close.pct_change(lag, fill_method=None)
    result["mr_mean_slope"] = mean.pct_change(5, fill_method=None) / 5
    autocorrelation = log_returns.rolling(config.lookback).corr(log_returns.shift(1))
    result["mr_return_autocorrelation"] = autocorrelation.mask(vol == 0, 0)
    result["mr_down_fraction"] = (
        (log_returns < 0).astype(float).where(log_returns.notna()).rolling(config.lookback).mean()
    )
    score = log_close.diff(60) / (log_returns.rolling(20).std().replace(0, np.nan) * np.sqrt(60))
    score = score.mask((log_returns.rolling(20).std() == 0) & (log_close.diff(60) == 0), 0)
    result["mr_regime_score"] = score
    result["regime"] = causal_regimes(frame).mask(score == 0, "sideways")
    result = result.replace([np.inf, -np.inf], np.nan)
    result["mr_ready"] = result[reversion_columns()].notna().all(axis=1) & (result.mr_atr > 0)
    return result
