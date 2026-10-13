from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.config.settings import Features
from trading_ml.features import build_features, feature_columns
from trading_ml.mean_reversion import ReversionConfig, build_reversion_features
from trading_ml.momentum.features import causal_regimes
from trading_ml.risk_overlay.simulator import risk_features
from trading_ml.smc.detectors import smc_columns

ORIGINAL = feature_columns(Features())
CORE = [
    "return_1",
    "return_3",
    "range_ratio",
    "body_ratio",
    "volume_ratio",
    "rsi",
    "macd_hist",
    "volatility",
    "sma_ratio_20",
    "sma_ratio_50",
    "volatility_ratio",
    "atr_ratio",
    "close_location",
    "volume_change_1",
]
MOMENTUM = [
    "adv_return_20",
    "adv_return_60",
    "adv_momentum_20",
    "adv_momentum_60",
    "adv_slope_20",
    "adv_efficiency_20",
]
REVERSION = ["mr_z", "mr_log_z", "mr_std_ratio", "mr_return_autocorrelation", "mr_down_fraction"]
BREAKOUT = [
    "adv_high_distance",
    "adv_low_distance",
    "adv_breakout_up",
    "adv_breakout_down",
    "adv_range_vol_ratio",
]
FLOW = [
    "flow_buy_share",
    "flow_pressure_3",
    "flow_pressure_10",
    "flow_pressure_change",
    "flow_quote_ratio",
    "flow_trades_ratio",
    "flow_mean_trade_ratio",
    "flow_vwap_distance",
]
META = (
    CORE + MOMENTUM + FLOW + ["mr_z", "adv_high_distance", "adv_regime_score", "base_probability"]
)


def family_columns(family: str) -> list[str]:
    groups = {
        "original": ORIGINAL,
        "core": CORE,
        "momentum": CORE + MOMENTUM,
        "reversion": CORE + REVERSION,
        "breakout": CORE + BREAKOUT,
        "flow": CORE + FLOW,
        "smc": CORE + smc_columns(),
        "flow_momentum": CORE + FLOW + MOMENTUM,
    }
    return list(dict.fromkeys(groups[family]))


def build_research_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = build_features(frame, Features(enhanced=True, smc=True))
    mr = build_reversion_features(frame, ReversionConfig())
    for column in REVERSION:
        result[column] = mr[column]
    log_close = np.log(frame.close)
    log_return = log_close.diff()
    vol = log_return.rolling(20).std()
    for horizon in [20, 60]:
        result[f"adv_return_{horizon}"] = frame.close.pct_change(horizon, fill_method=None)
        result[f"adv_momentum_{horizon}"] = log_close.diff(horizon) / (
            vol.replace(0, np.nan) * np.sqrt(horizon)
        )
    sma20 = frame.close.rolling(20).mean()
    result["adv_slope_20"] = sma20.pct_change(5, fill_method=None) / 5
    result["adv_efficiency_20"] = log_close.diff(20).abs() / log_return.abs().rolling(
        20
    ).sum().replace(0, np.nan)
    high = frame.high.shift(1).rolling(20).max()
    low = frame.low.shift(1).rolling(20).min()
    result["adv_high_distance"] = frame.close / high - 1
    result["adv_low_distance"] = frame.close / low - 1
    result["adv_breakout_up"] = (frame.close > high).astype(float)
    result["adv_breakout_down"] = (frame.close < low).astype(float)
    result["adv_range_vol_ratio"] = (
        (frame.high - frame.low) / frame.close
    ) / result.volatility.replace(0, np.nan)
    quote = frame.quote_volume.replace(0, np.nan)
    result["flow_buy_share"] = frame.taker_quote / quote
    for window in [3, 10]:
        result[f"flow_pressure_{window}"] = (
            2 * frame.taker_quote.rolling(window).sum() / quote.rolling(window).sum() - 1
        )
    result["flow_pressure_change"] = result.flow_buy_share.diff()
    result["flow_quote_ratio"] = quote / quote.shift(1).rolling(20).mean()
    result["flow_trades_ratio"] = frame.trades / frame.trades.shift(1).rolling(20).mean().replace(
        0, np.nan
    )
    mean_trade = quote / frame.trades.replace(0, np.nan)
    result["flow_mean_trade_ratio"] = mean_trade / mean_trade.shift(1).rolling(20).mean()
    result["flow_vwap_distance"] = frame.close / (quote / frame.volume.replace(0, np.nan)) - 1
    result["adv_regime_score"] = log_close.diff(60) / (vol.replace(0, np.nan) * np.sqrt(60))
    result["adv_vol_limit"] = result.volatility.shift(1).rolling(60).quantile(0.90)
    result["regime"] = causal_regimes(frame)
    result = risk_features(result).replace([np.inf, -np.inf], np.nan)
    columns = list(
        dict.fromkeys(ORIGINAL + CORE + MOMENTUM + REVERSION + BREAKOUT + FLOW + smc_columns())
    )
    result["research_ready"] = (
        result[columns + ["adv_regime_score", "adv_vol_limit"]].notna().all(axis=1)
    )
    return result
