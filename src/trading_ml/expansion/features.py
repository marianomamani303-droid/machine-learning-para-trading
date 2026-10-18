from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.dmn.data import Panel, causal_features
from trading_ml.errors import DataError, LeakageError

FLOW = ("aggressive_imbalance", "imbalance_ewm7", "quote_volume_z63", "trade_size_z63")
DERIVATIVES = ("funding_1d", "funding_7d", "funding_z63", "perp_premium")
CROSS = ("crypto_momentum_mean", "crypto_momentum_dispersion", "crypto_breadth")


def point_in_time_join(
    references: pd.DatetimeIndex, events: pd.DataFrame, values: list[str], tolerance: pd.Timedelta
) -> pd.DataFrame:
    """Revision-safe availability join; observation times need not be monotone.

    For macro revisions, an older observation revised today must never displace a
    newer observation. Use latest_observations below before exporting a state stream.
    """
    data = events.copy()
    if not {"event_time", "available_at", *values}.issubset(data):
        raise DataError("Datos externos requieren evento, disponibilidad y valores")
    for col in ["event_time", "available_at"]:
        data[col] = pd.to_datetime(data[col], utc=True)
    if (data.available_at < data.event_time).any():
        raise LeakageError("Disponibilidad anterior al evento")
    if data.available_at.duplicated().any() or data[values].isna().any().any():
        raise DataError("Disponibilidades duplicadas o valores faltantes")
    left = pd.DataFrame({"reference_time": pd.DatetimeIndex(references)})
    if not left.reference_time.is_monotonic_increasing or left.reference_time.duplicated().any():
        raise DataError("Referencias de join desordenadas/duplicadas")
    return pd.merge_asof(
        left,
        data.sort_values("available_at"),
        left_on="reference_time",
        right_on="available_at",
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )


def latest_observations(events: pd.DataFrame, references: pd.DatetimeIndex) -> pd.DataFrame:
    """At each decision pick latest observation, then latest vintage known then."""
    data = events.copy()
    for col in ["event_time", "available_at"]:
        data[col] = pd.to_datetime(data[col], utc=True)
    if (data.available_at < data.event_time).any() or data.duplicated(
        ["event_time", "available_at"]
    ).any():
        raise LeakageError("Vintages ambiguos o retroactivos")
    rows = []
    for reference in references:
        available = data.loc[data.available_at <= reference].sort_values(
            ["event_time", "available_at"]
        )
        if available.empty:
            rows.append({"reference_time": reference})
        else:
            rows.append({**available.iloc[-1].to_dict(), "reference_time": reference})
    return pd.DataFrame(rows)


def zscore_past(series: pd.Series, window=63) -> pd.Series:
    past = series.shift(1).rolling(window, min_periods=window)
    return (series - past.mean()) / past.std().clip(lower=1e-8)


def open_interest_features(events: pd.DataFrame) -> pd.DataFrame:
    """No OI history is fabricated: caller must supply measured availability."""
    data = events.sort_values("available_at").copy()
    if (data.open_interest <= 0).any() or not np.isfinite(data.open_interest).all():
        raise DataError("Open interest inválido")
    data["oi_change"] = np.log(data.open_interest).diff()
    data["oi_z63"] = zscore_past(np.log(data.open_interest))
    return data


def crypto_features(spot: pd.DataFrame, mark: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    frame = causal_features(spot)
    # Preserve exchange flow columns: the OHLCV cleaner intentionally removes extras.
    frame = frame.merge(
        spot[["timestamp", "quote_volume", "taker_quote", "trades"]],
        on="timestamp",
        validate="one_to_one",
    )
    if (frame.quote_volume <= 0).any() or (frame.trades <= 0).any():
        raise DataError("Sesión sin volumen/trades: no se inventa flujo")
    frame["aggressive_imbalance"] = 2 * frame.taker_quote / frame.quote_volume - 1
    if (frame.aggressive_imbalance.abs() > 1 + 1e-6).any():
        raise DataError("Volumen agresor imposible")
    frame["imbalance_ewm7"] = frame.aggressive_imbalance.ewm(span=7, adjust=False).mean()
    frame["quote_volume_z63"] = zscore_past(np.log(frame.quote_volume))
    frame["trade_size_z63"] = zscore_past(np.log(frame.quote_volume / frame.trades))
    # Daily published state only after all settlements in that UTC day are known.
    settlements = funding.assign(day=funding.available_at.dt.floor("D"))
    daily = settlements.groupby("day").funding_rate.agg(["sum", "count"])
    if (daily["count"] < 1).any():
        raise DataError("Día sin funding")
    daily["funding_1d"] = daily["sum"]
    daily["funding_7d"] = daily["sum"].rolling(7).sum()
    daily["funding_z63"] = zscore_past(daily["sum"])
    daily["available_at"] = daily.index + pd.Timedelta(days=1)
    daily["event_time"] = daily.available_at
    joined = point_in_time_join(
        pd.DatetimeIndex(frame.reference_time),
        daily.reset_index(drop=True).dropna(),
        list(DERIVATIVES[:3]),
        pd.Timedelta(days=1),
    )
    for col in DERIVATIVES[:3]:
        frame[col] = joined[col].to_numpy()
    marks = mark.rename(columns={"timestamp": "event_time", "close": "mark_close"})
    joined = point_in_time_join(
        pd.DatetimeIndex(frame.reference_time),
        marks[["event_time", "available_at", "mark_close"]],
        ["mark_close"],
        pd.Timedelta(seconds=1),
    )
    frame["perp_premium"] = joined.mark_close.to_numpy() / frame.close - 1
    return frame


def attach_context(
    frames: dict[str, pd.DataFrame], forex: dict[str, pd.DataFrame]
) -> tuple[dict, tuple]:
    times = pd.DatetimeIndex(next(iter(frames.values())).reference_time)
    if any(not times.equals(pd.DatetimeIndex(f.reference_time)) for f in frames.values()):
        raise DataError("Criptoactivos desalineados")
    momentum = np.stack([f.return_21.to_numpy() for f in frames.values()], axis=1)
    shared = {
        "crypto_momentum_mean": momentum.mean(axis=1),
        "crypto_momentum_dispersion": momentum.std(axis=1),
        "crypto_breadth": (momentum > 0).mean(axis=1),
    }
    names = list(CROSS)
    strengths = []
    for symbol, raw in sorted(forex.items()):
        f = raw.sort_values("reference_time").copy()
        ret = f.close.pct_change(fill_method=None)
        vol = ret.ewm(span=60, adjust=False, min_periods=60).std().clip(lower=1e-5)
        f["fx_momentum"] = f.close.pct_change(21, fill_method=None) / (vol * np.sqrt(21))
        f["fx_return"] = ret / vol
        f["event_time"] = f.reference_time
        f["available_at"] = f.reference_time + pd.Timedelta(seconds=1)
        f = f.dropna(subset=["fx_momentum", "fx_return"])
        joined = point_in_time_join(
            times,
            f[["event_time", "available_at", "fx_momentum", "fx_return"]],
            ["fx_momentum", "fx_return"],
            pd.Timedelta(days=4),
        )
        stale = joined.fx_momentum.isna()
        for col in ["fx_momentum", "fx_return"]:
            key = f"{symbol}_{col}"
            # Missing external information is a known state, not an invented price.
            shared[key] = joined[col].fillna(0).to_numpy()
            names.append(key)
        key = f"{symbol}_unavailable"
        shared[key] = stale.astype(float).to_numpy()
        names.append(key)
        strengths.append(shared[f"{symbol}_fx_momentum"] * (1 if symbol.startswith("USD") else -1))
    if strengths:
        shared["usd_strength_proxy"] = np.mean(strengths, axis=0)
        names.append("usd_strength_proxy")
    for f in frames.values():
        for key, values in shared.items():
            f[key] = values
    return frames, tuple(names)


def make_panel(
    frames: dict[str, pd.DataFrame],
    columns: tuple[str, ...],
    sequence=63,
    *,
    noise=False,
    noise_dimensions=0,
) -> Panel:
    # Same warmup for all ablations. The caller crops each frame by the full schema.
    first = next(iter(frames.values()))
    values = np.stack(
        [f[list(columns)].to_numpy(dtype=np.float32) for f in frames.values()], axis=1
    )
    names = columns
    if noise:
        rng = np.random.default_rng(20261016)
        values = np.concatenate(
            [values, rng.normal(size=(*values.shape[:2], noise_dimensions)).astype(np.float32)],
            axis=2,
        )
        names = (*columns, *(f"noise_{n}" for n in range(noise_dimensions)))
    if not np.isfinite(values).all():
        raise DataError("Features no finitas después de warmup común")
    x = np.lib.stride_tricks.sliding_window_view(values, sequence, axis=0).transpose(0, 1, 3, 2)
    tail = sequence - 1
    vol = np.stack([f.daily_volatility.to_numpy()[tail:] for f in frames.values()], axis=1)
    returns = np.stack([f.asset_return.to_numpy()[tail:] for f in frames.values()], axis=1)
    baselines = {
        "cash": np.zeros_like(vol),
        "long_scaled": np.ones_like(vol),
        "tsmom": np.stack(
            [np.sign(f.return_252.to_numpy()[tail:]) for f in frames.values()], axis=1
        ),
    }
    return Panel(
        pd.DatetimeIndex(first.reference_time.iloc[tail:]),
        pd.DatetimeIndex(first.target_time.iloc[tail:]),
        tuple(frames),
        x,
        returns,
        vol,
        baselines,
        names,
    )
