from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.config.settings import Features
from trading_ml.features.technical import build_features, feature_columns
from trading_ml.multimarket.markets import instrument

PRICE = [column for column in feature_columns(Features()) if column != "volume_ratio"]
GROUPS = {
    "momentum": ["acceleration", "trend_efficiency"],
    "reversion": ["zscore_20", "range_location_20"],
    "regime": ["volatility_ratio", "breakout_distance", "atr_ratio"],
    "session": [
        "london_open",
        "new_york_open",
        "tokyo_open",
        "hour_sin",
        "hour_cos",
        "week_sin",
        "week_cos",
    ],
    "spread": ["spread_fraction", "spread_change"],
}


def features(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Close-known features. No volume proxy, centered pivots or full-sample scaling."""
    asset = instrument(symbol)
    f = build_features(frame, Features(), include_volume=False)
    close = f.close
    f["acceleration"] = f.return_3 - f.return_3.shift(3)
    f["trend_efficiency"] = (close - close.shift(20)) / close.diff().abs().rolling(
        20
    ).sum().replace(0, np.nan)
    f["zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
    high, low = f.high.rolling(20).max(), f.low.rolling(20).min()
    f["range_location_20"] = (close - low) / (high - low).replace(0, np.nan)
    f["volatility_ratio"] = f.return_1.rolling(5).std() / f.return_1.rolling(60).std().replace(
        0, np.nan
    )
    f["breakout_distance"] = (close - f.high.rolling(20).max().shift(1)) / close
    tr = pd.concat(
        [f.high - f.low, (f.high - close.shift()).abs(), (f.low - close.shift()).abs()], axis=1
    ).max(axis=1)
    f["atr_ratio"] = tr.rolling(14).mean() / close
    clock = pd.DatetimeIndex(f.reference_time)
    for name, zone, opening, closing in [
        ("london_open", "Europe/London", 8, 17),
        ("new_york_open", "America/New_York", 8, 17),
        ("tokyo_open", "Asia/Tokyo", 9, 18),
    ]:
        hour = clock.tz_convert(zone).hour
        f[name] = ((hour >= opening) & (hour < closing)).astype(float)
    f["hour_sin"] = np.sin(2 * np.pi * clock.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * clock.hour / 24)
    f["week_sin"] = np.sin(2 * np.pi * clock.dayofweek / 7)
    f["week_cos"] = np.cos(2 * np.pi * clock.dayofweek / 7)
    if asset.market == "forex":
        f["spread_fraction"] = (f.close_ask - f.close_bid) / f.close_bid
        f["spread_change"] = f.spread_fraction / f.spread_fraction.rolling(20).median().replace(
            0, np.nan
        )
        if "crossed_close" in f:
            # Invalid contemporaneous quote, known at signal close; abstain.
            f.loc[f.crossed_close, ["spread_fraction", "spread_change"]] = np.nan
    f["source_index"] = np.arange(len(f))
    return f.replace([np.inf, -np.inf], np.nan)


def recipes(symbol: str) -> dict[str, list[str]]:
    extensions = [*GROUPS["momentum"], *GROUPS["reversion"], *GROUPS["regime"], *GROUPS["session"]]
    if instrument(symbol).market == "forex":
        extensions += GROUPS["spread"]
    return {"price": PRICE.copy(), "compact": PRICE + extensions}
