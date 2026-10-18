from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, sidecar
from trading_ml.errors import DataError

PRICE = (
    "ret_bar",
    "ret_day",
    "ret_week",
    "range_sigma",
    "body_range",
    "close_location",
    "volatility",
    "vol_ratio",
    "efficiency",
    "hour_sin",
    "hour_cos",
)
EXTRA = (
    "parkinson_ratio",
    "relative_volume",
    "amihud_z",
    "signed_volume",
    "breakout",
    "downside_ratio",
)
DELTAS = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def load_development(path: Path, start: str, end: str) -> pd.DataFrame:
    """Stop CSV iteration at quarantine boundary; never use later prices/features."""
    meta = read_json(sidecar(path))
    if sha256(path) != meta["sha256"]:
        raise DataError("Historical source checksum mismatch")
    begin, stop = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    records = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            timestamp = pd.Timestamp(row["timestamp"])
            if timestamp >= stop:
                break
            if timestamp >= begin:
                records.append(
                    {
                        "timestamp": timestamp,
                        **{k: float(row[k]) for k in ["open", "high", "low", "close", "volume"]},
                    }
                )
    frame = pd.DataFrame(records)
    if (
        frame.empty
        or frame.timestamp.duplicated().any()
        or not frame.timestamp.is_monotonic_increasing
    ):
        raise DataError("Missing, duplicate or unordered development data")
    numbers = frame[["open", "high", "low", "close", "volume"]]
    if (
        not np.isfinite(numbers).all().all()
        or (numbers.iloc[:, :4] <= 0).any().any()
        or (frame.volume < 0).any()
    ):
        raise DataError("Invalid prices/volume")
    if (
        (frame.high < numbers.iloc[:, :4].max(axis=1))
        | (frame.low > numbers.iloc[:, :4].min(axis=1))
    ).any():
        raise DataError("Invalid OHLC")
    return frame


def causal_features(raw: pd.DataFrame, interval: str) -> pd.DataFrame:
    delta = DELTAS[interval]
    day = int(pd.Timedelta(days=1) / delta)
    week, month = 7 * day, 30 * day
    raw = raw.copy().reset_index(drop=True)
    segment = raw.timestamp.diff().ne(delta).cumsum()
    pieces = []
    for _, f in raw.groupby(segment, sort=False):
        f = f.copy()
        ret = np.log(f.close / f.close.shift())
        sigma = ret.rolling(week).std().clip(lower=1e-8)
        rng = (f.high - f.low).clip(lower=1e-8)
        f["ret_bar"] = ret / sigma
        f["ret_day"] = np.log(f.close / f.close.shift(day)) / (sigma * np.sqrt(day))
        f["ret_week"] = np.log(f.close / f.close.shift(week)) / (sigma * np.sqrt(week))
        f["range_sigma"] = np.log(f.high / f.low) / sigma
        f["body_range"] = (f.close - f.open) / rng
        f["close_location"] = (2 * f.close - f.high - f.low) / rng
        f["volatility"] = sigma
        f["vol_ratio"] = ret.rolling(max(day, 2)).std() / sigma
        f["efficiency"] = (f.close - f.close.shift(week)).abs() / f.close.diff().abs().rolling(
            week
        ).sum().clip(lower=1e-8)
        hour = (f.timestamp + delta).dt.hour
        f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        f["parkinson_ratio"] = (
            np.sqrt(np.log(f.high / f.low).pow(2).rolling(week).mean() / (4 * np.log(2))) / sigma
        )
        f["relative_volume"] = f.volume / f.volume.rolling(week).mean().clip(lower=1e-8)
        illiquidity = ret.abs() / (f.volume * f.close).clip(lower=1)
        mean = illiquidity.rolling(month).mean()
        f["amihud_z"] = (illiquidity - mean) / illiquidity.rolling(month).std().clip(lower=1e-16)
        f["signed_volume"] = f.body_range * f.relative_volume
        previous_high = f.high.rolling(week).max().shift()
        previous_low = f.low.rolling(week).min().shift()
        f["breakout"] = (
            (f.close - previous_high).clip(lower=0) + (f.close - previous_low).clip(upper=0)
        ) / (f.close * sigma)
        f["downside_ratio"] = ret.clip(upper=0).pow(2).rolling(week).mean() / ret.pow(2).rolling(
            week
        ).mean().clip(lower=1e-16)
        f["trend_long"] = f.close > f.close.ewm(span=month, adjust=False, min_periods=month).mean()
        pieces.append(f)
    result = pd.concat(pieces).sort_index()
    result["reference_time"] = result.timestamp + delta
    return result


def supervised(raw: pd.DataFrame, interval: str, horizon: int, end: str) -> pd.DataFrame:
    f = causal_features(raw, interval)
    f["entry_time"] = f.timestamp.shift(-1)
    f["target_time"] = f.timestamp.shift(-horizon - 1)
    f["gross_return"] = f.open.shift(-horizon - 1) / f.open.shift(-1) - 1
    delta = DELTAS[interval]
    continuous = (f.entry_time == f.reference_time) & (
        f.target_time == f.reference_time + horizon * delta
    )
    f["eligible"] = (
        continuous
        & (f.target_time < pd.Timestamp(end, tz="UTC"))
        & np.isfinite(f[list(PRICE + EXTRA)]).all(axis=1)
    )
    f["target"] = (f.gross_return > 0).astype(int)
    return f


def partitions(frame: pd.DataFrame, start, end, validation_days=90, embargo_hours=48):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    gap = pd.Timedelta(hours=embargo_hours)
    validation_start = start - pd.Timedelta(days=validation_days)
    fit = frame.loc[frame.eligible & (frame.target_time < validation_start - gap)]
    validation = frame.loc[
        frame.eligible
        & (frame.reference_time >= validation_start)
        & (frame.target_time < start - gap)
    ]
    test = frame.loc[frame.eligible & (frame.reference_time >= start) & (frame.target_time < end)]
    if min(len(fit), len(validation), len(test)) < 40:
        raise DataError("Insufficient rows for purged partitions")
    if not (
        fit.target_time.max() < validation.reference_time.min() - gap
        and validation.target_time.max() < test.reference_time.min() - gap
    ):
        raise DataError("Temporal purge violated")
    return fit, validation, test
