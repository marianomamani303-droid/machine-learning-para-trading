from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.momentum.config import MomentumConfig


def momentum_columns() -> list[str]:
    """Dimensionless features, explicit allowlist excluding future labels."""
    return [
        "mom_return",
        "mom_log_return",
        "mom_normalized",
        "mom_volatility",
        "mom_fast_distance",
        "mom_slow_distance",
        "mom_fast_slope",
        "mom_slow_slope",
        "mom_efficiency",
        "mom_signed_efficiency",
        "mom_positive_fraction",
        "mom_trend_persistence",
        "mom_above_run",
        "mom_high_distance",
        "mom_low_distance",
        "mom_breakout_up",
        "mom_breakout_down",
        "mom_macd",
        "mom_macd_hist",
    ]


def build_momentum_features(frame: pd.DataFrame, config: MomentumConfig) -> pd.DataFrame:
    """Only current closed candle and history; prior extrema exclude current candle."""
    config.validate()
    result = frame.copy()
    close = frame.close
    log_close = np.log(close)
    log_return = log_close.diff()
    window = config.lookback
    volatility = log_return.rolling(config.volatility_window).std()
    fast = close.rolling(config.fast_window).mean()
    slow = close.rolling(config.slow_window).mean()
    result["mom_return"] = close.pct_change(window, fill_method=None)
    result["mom_log_return"] = log_close.diff(window)
    result["mom_normalized"] = result.mom_log_return / (
        volatility.replace(0, np.nan) * np.sqrt(window)
    )
    # A flat market has no momentum rather than an infinite score.
    result["mom_normalized"] = result.mom_normalized.mask(
        (volatility == 0) & (result.mom_log_return == 0), 0
    )
    result["mom_volatility"] = volatility
    result["mom_fast_distance"] = close / fast - 1
    result["mom_slow_distance"] = close / slow - 1
    result["mom_fast_slope"] = (
        fast.pct_change(config.slope_window, fill_method=None) / config.slope_window
    )
    result["mom_slow_slope"] = (
        slow.pct_change(config.slope_window, fill_method=None) / config.slope_window
    )
    path = log_return.abs().rolling(window).sum()
    efficiency = result.mom_log_return / path.replace(0, np.nan)
    efficiency = efficiency.mask(path == 0, 0)
    result["mom_signed_efficiency"] = efficiency
    result["mom_efficiency"] = efficiency.abs()
    result["mom_positive_fraction"] = (
        (log_return > 0).astype(float).where(log_return.notna()).rolling(window).mean()
    )
    above = (close > slow).where(slow.notna())
    result["mom_trend_persistence"] = above.astype(float).rolling(window).mean()
    above_bool = above.fillna(False).astype(bool)
    run = above_bool.groupby((~above_bool).cumsum()).cumsum()
    result["mom_above_run"] = run.clip(upper=window) / window
    previous_high = frame.high.shift(1).rolling(window).max()
    previous_low = frame.low.shift(1).rolling(window).min()
    result["mom_high_distance"] = close / previous_high - 1
    result["mom_low_distance"] = close / previous_low - 1
    result["mom_breakout_up"] = (close > previous_high).astype(float).where(previous_high.notna())
    result["mom_breakout_down"] = (close < previous_low).astype(float).where(previous_low.notna())
    macd_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    macd_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = macd_fast - macd_slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["mom_macd"] = macd / close
    result["mom_macd_hist"] = (macd - signal) / close
    result = result.replace([np.inf, -np.inf], np.nan)
    result["mom_ready"] = result[momentum_columns()].notna().all(axis=1)
    return result


def causal_regimes(frame: pd.DataFrame, lookback: int = 60, band: float = 1.0) -> pd.Series:
    """Trailing risk-normalized direction at close; never full-period hindsight."""
    r = np.log(frame.close).diff()
    score = np.log(frame.close).diff(lookback) / (
        r.rolling(20).std().replace(0, np.nan) * np.sqrt(lookback)
    )
    return pd.Series(
        np.where(score > band, "bull", np.where(score < -band, "bear", "sideways")),
        index=frame.index,
    ).where(score.notna(), "unknown")
