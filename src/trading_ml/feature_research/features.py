from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.advanced_research.features import build_research_features
from trading_ml.errors import DataError

# Small, hypothesis-led families. Never infer model columns from the whole frame:
# it also carries future labels for scoring.
FAMILIES = {
    "momentum": ["fe_acceleration_3", "adv_efficiency_20", "adv_momentum_60"],
    "volatility": ["fe_vol_ratio_5_60", "fe_parkinson_ratio_20", "fe_vol_of_vol_20"],
    "flow": ["fe_imbalance_1", "fe_flow_innovation_3_20", "flow_mean_trade_ratio"],
    "reversion": ["fe_log_z_20", "fe_vwap_distance_20"],
    "breakout": ["fe_high_atr_20", "fe_low_atr_20", "fe_squeeze_release"],
    "structure": ["smc_structure", "fe_sweep_net", "fe_bos_net", "smc_structure_location"],
    "timeframes": ["fe_week_momentum_4", "fe_week_trend_8", "fe_daily_week_gap"],
    "cross_asset": [
        "fe_peer_return_1",
        "fe_relative_return_20",
        "fe_residual_return_1",
        "fe_correlation_change",
    ],
}
ORIGINAL_GROUPS = {
    "momentum": [
        "return_1",
        "return_3",
        "return_7",
        "macd",
        "macd_signal",
        "macd_hist",
        "sma_ratio_10",
        "sma_ratio_20",
        "sma_ratio_50",
    ],
    "shape": ["range_ratio", "body_ratio"],
    "volume": ["volume_ratio"],
    "volatility": ["volatility"],
    "rsi": ["rsi"],
}


def closed_weekly_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Monday 00:00 UTC availability; incomplete weeks never become features."""
    week = frame.timestamp.dt.normalize() - pd.to_timedelta(frame.timestamp.dt.weekday, unit="D")
    weekly = (
        frame.assign(week=week)
        .groupby("week", sort=True)
        .agg(close=("close", "last"), count=("close", "size"))
    )
    weekly = weekly.loc[weekly["count"] == 7].copy()
    log_price = np.log(weekly.close)
    sigma = log_price.diff().rolling(8, min_periods=8).std().replace(0, np.nan)
    weekly["fe_week_momentum_4"] = log_price.diff(4) / (sigma * 2)
    ema = weekly.close.ewm(span=8, adjust=False, min_periods=8).mean()
    weekly["fe_week_trend_8"] = np.log(weekly.close / ema) / sigma
    weekly["weekly_available_time"] = weekly.index + pd.Timedelta(days=7)
    columns = ["weekly_available_time", "fe_week_momentum_4", "fe_week_trend_8"]
    return pd.merge_asof(
        frame[["reference_time"]],
        weekly[columns].reset_index(drop=True),
        left_on="reference_time",
        right_on="weekly_available_time",
        direction="backward",
        tolerance=pd.Timedelta(days=6),
    )


def build_feature_panel(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if set(raw) != {"BTCUSDT", "ETHUSDT"}:
        raise DataError("El estudio requiere BTCUSDT y ETHUSDT diarios")
    built = {}
    for symbol, source in raw.items():
        frame = source.reset_index(drop=True).copy()
        required = {
            "timestamp",
            "reference_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
        }
        if not required.issubset(frame):
            raise DataError("Faltan datos OHLCV/flujo reales; no se imputan")
        if (
            frame.reference_time.duplicated().any()
            or not frame.reference_time.is_monotonic_increasing
        ):
            raise DataError("Referencias desordenadas o duplicadas")
        if (frame.timestamp.diff().iloc[1:] != pd.Timedelta(days=1)).any():
            raise DataError("Las features requieren velas diarias contiguas")
        if (frame.reference_time != frame.timestamp + pd.Timedelta(days=1)).any():
            raise DataError("reference_time debe ser el cierre diario UTC")
        if (frame.timestamp != frame.timestamp.dt.normalize()).any():
            raise DataError("Velas desalineadas respecto de medianoche UTC")
        f = build_research_features(frame)
        log_price = np.log(frame.close)
        ret = log_price.diff()
        sigma = ret.rolling(20).std().replace(0, np.nan)
        short, long = ret.rolling(5).std(), ret.rolling(60).std().replace(0, np.nan)
        f["fe_acceleration_3"] = (log_price.diff(3) - log_price.diff(3).shift(3)) / (
            sigma * np.sqrt(6)
        )
        f["fe_vol_ratio_5_60"] = short / long
        parkinson = np.sqrt(
            np.log(frame.high / frame.low).pow(2).rolling(20).mean() / (4 * np.log(2))
        )
        f["fe_parkinson_ratio_20"] = parkinson / sigma
        f["fe_vol_of_vol_20"] = short.rolling(20).std() / short.rolling(20).mean().replace(
            0, np.nan
        )
        quote = frame.quote_volume.replace(0, np.nan)
        f["fe_imbalance_1"] = 2 * frame.taker_quote / quote - 1
        pressure3 = 2 * frame.taker_quote.rolling(3).sum() / quote.rolling(3).sum() - 1
        prior20 = (
            2 * frame.taker_quote.shift(1).rolling(20).sum() / quote.shift(1).rolling(20).sum() - 1
        )
        f["fe_flow_innovation_3_20"] = pressure3 - prior20
        prior_mean = log_price.shift(1).rolling(20).mean()
        prior_std = log_price.shift(1).rolling(20).std().replace(0, np.nan)
        f["fe_log_z_20"] = (log_price - prior_mean) / prior_std
        vwap20 = quote.rolling(20).sum() / frame.volume.rolling(20).sum().replace(0, np.nan)
        f["fe_vwap_distance_20"] = np.log(frame.close / vwap20) / sigma
        tr = pd.concat(
            [
                frame.high - frame.low,
                (frame.high - frame.close.shift(1)).abs(),
                (frame.low - frame.close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        prior_atr = tr.shift(1).rolling(14).mean().replace(0, np.nan)
        f["fe_high_atr_20"] = (frame.close - frame.high.shift(1).rolling(20).max()) / prior_atr
        f["fe_low_atr_20"] = (frame.close - frame.low.shift(1).rolling(20).min()) / prior_atr
        f["fe_squeeze_release"] = tr / prior_atr * (long / short.replace(0, np.nan)).shift(1)
        f["fe_sweep_net"] = f.smc_sweep_low - f.smc_sweep_high
        f["fe_bos_net"] = f.smc_bos_up - f.smc_bos_down
        weekly = closed_weekly_features(frame)
        for name in weekly.columns.drop("reference_time"):
            f[name] = weekly[name].to_numpy()
        f["fe_daily_week_gap"] = log_price.diff(5) / (sigma * np.sqrt(5)) - f.fe_week_momentum_4
        # Regimes use only past returns and thresholds fixed before the experiment.
        score = log_price.diff(60) / (sigma * np.sqrt(60))
        f["fe_regime"] = np.select([score > 1, score < -1], ["bull", "bear"], default="sideways")
        limit = sigma.shift(1).rolling(120).quantile(0.8)
        f["fe_vol_regime"] = np.where(sigma > limit, "high", "normal")
        built[symbol] = f

    output = {}
    for symbol, f in built.items():
        peer = built[next(s for s in built if s != symbol)]
        # Exact references, without forward filling or truncating to a future joint mask.
        joined = f.merge(
            peer[["reference_time", "close"]],
            on="reference_time",
            how="left",
            suffixes=("", "_peer"),
            validate="one_to_one",
        )
        r, p = np.log(joined.close).diff(), np.log(joined.close_peer).diff()
        beta = r.shift(1).rolling(60).cov(p.shift(1)) / p.shift(1).rolling(60).var().replace(
            0, np.nan
        )
        joined["fe_peer_return_1"] = p
        joined["fe_relative_return_20"] = r.rolling(20).sum() - p.rolling(20).sum()
        joined["fe_residual_return_1"] = (r - beta * p) / r.shift(1).rolling(60).std().replace(
            0, np.nan
        )
        joined["fe_correlation_change"] = r.rolling(20).corr(p) - r.rolling(60).corr(p)
        future = joined.close.shift(-1)
        joined["target"] = (future > joined.close).astype(float).where(future.notna())
        joined["target_time"] = joined.reference_time.shift(-1)
        output[symbol] = joined.replace([np.inf, -np.inf], np.nan)
    return output
