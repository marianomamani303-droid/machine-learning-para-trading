from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trading_ml.common import read_json, sha256, sidecar
from trading_ml.config.settings import Features
from trading_ml.errors import DataError, LeakageError
from trading_ml.expansion.features import zscore_past
from trading_ml.features.technical import build_features, feature_columns
from trading_ml.training.recipe import protect_reserved_labels

DAY = pd.Timedelta(days=1)
FINANCIAL = [
    "fin_liq_corwin_schultz_20",
    "fin_liq_range_impact_z63",
    "fin_vol_parkinson_ratio",
]
COLUMNS = feature_columns(Features()) + FINANCIAL


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Exact top_core_3 formulas; targets/prices are never included in agent states."""
    f = raw.copy().reset_index(drop=True)
    for c in ("timestamp", "available_at"):
        f[c] = pd.to_datetime(f[c], utc=True)
    if (
        len(f) < 200
        or not f.timestamp.is_monotonic_increasing
        or f.timestamp.duplicated().any()
        or not f.timestamp.diff().iloc[1:].eq(DAY).all()
        or not f.available_at.eq(f.timestamp + DAY).all()
        or not np.isfinite(f[["open", "high", "low", "close", "volume", "quote_volume"]])
        .all()
        .all()
        or (f[["open", "high", "low", "close"]] <= 0).any().any()
        or (f.high < f[["open", "close", "low"]].max(axis=1)).any()
        or (f.low > f[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise DataError("Daily spot source must be continuous, valid and close-available")
    f["reference_time"] = f.available_at
    f = build_features(f, Features())
    vol = np.log(f.close / f.close.shift(1)).rolling(20, min_periods=20).std().clip(lower=1e-8)
    hl = np.log((f.high / f.low).clip(lower=1.0))
    parkinson = np.sqrt(hl.pow(2).rolling(20, min_periods=20).mean() / (4 * np.log(2)))
    f[FINANCIAL[2]] = parkinson / vol - 1
    impact = (
        (np.log((f.high / f.low).clip(lower=1e-8)) / f.quote_volume.clip(lower=1))
        .rolling(5, min_periods=5)
        .mean()
    )
    f[FINANCIAL[1]] = zscore_past(np.log(impact.clip(lower=1e-18)), window=63)
    beta = hl.pow(2) + hl.shift(1).pow(2)
    high2 = pd.concat([f.high, f.high.shift(1)], axis=1).max(axis=1)
    low2 = pd.concat([f.low, f.low.shift(1)], axis=1).min(axis=1)
    gamma = np.log((high2 / low2).clip(lower=1.0)).pow(2)
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt((gamma / k).clip(lower=0))
    estimate = 2 * (np.exp(alpha.clip(lower=0)) - 1) / (1 + np.exp(alpha.clip(lower=0)))
    f[FINANCIAL[0]] = estimate.rolling(20, min_periods=20).mean() / vol.clip(lower=1e-6)
    f["sigma"] = vol
    f["target"] = (f.close.shift(-1) > f.close).astype(float).where(f.close.shift(-1).notna())
    f["target_time"] = f.reference_time.shift(-1)
    # Signal at close t is first executable at next open t+1, at reference_time.
    f["entry_time"] = f.timestamp.shift(-1)
    f["entry_open"] = f.open.shift(-1)
    f["exit_time"] = f.timestamp.shift(-2)
    f["exit_open"] = f.open.shift(-2)
    # Purge for the slowest scenario; delayed target must have an in-fold exit.
    f["latest_exit_time"] = f.timestamp.shift(-3)
    return f


def load_panel(config: dict) -> tuple[pd.DataFrame, dict]:
    data = config["data"]
    index = Path(data["source_index"])
    source = Path(read_json(index)["assets"][data["symbol"]]["spot"])
    meta = read_json(sidecar(source))
    if sha256(source) != meta["sha256"] or meta.get("frequency") != "1d":
        raise DataError("Invalid source hash/frequency")
    if meta.get("instrument_type") != "crypto_spot":
        raise DataError("Only verified spot source supported")
    stop = pd.Timestamp(data["stop"], tz="UTC")
    if stop > pd.Timestamp("2025-07-01", tz="UTC"):
        raise LeakageError("Historical development quarantine cannot be extended")
    raw = pd.read_csv(source, float_precision="round_trip")
    raw.timestamp = pd.to_datetime(raw.timestamp, utc=True)
    raw.available_at = pd.to_datetime(raw.available_at, utc=True)
    # Crop BEFORE any features, forward labels or execution arrays.
    raw = raw.loc[
        (raw.timestamp >= pd.Timestamp(data["start"], tz="UTC")) & (raw.available_at < stop)
    ]
    f = build_panel(raw)
    protect_reserved_labels(Path("artifacts"), f.dropna(subset=["target_time"]))
    return f, {
        "source": str(source),
        "source_sha256": sha256(source),
        "metadata_sha256": sha256(sidecar(source)),
        "index_sha256": sha256(index),
        "rows": len(raw),
        "last_available_at": raw.available_at.max().isoformat(),
    }


def supervised_fit(frame: pd.DataFrame, start: pd.Timestamp, config: dict):
    cutoff = start - pd.Timedelta(days=config["data"]["embargo_days"])
    train = frame.loc[(frame.target_time < cutoff) & np.isfinite(frame[COLUMNS]).all(axis=1)]
    p = config["predictor"]
    if len(train) < p["minimum_rows"] or train.target.nunique() != 2:
        raise DataError("Insufficient prior-only supervised history")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=p["C"], solver="lbfgs", max_iter=p["max_iter"], random_state=p["seed"]
        ),
    )
    model.fit(train[COLUMNS].to_numpy(), train.target.to_numpy(dtype=int))
    return model, {
        "fit_at": start.isoformat(),
        "cutoff_exclusive": cutoff.isoformat(),
        "last_train_reference": train.reference_time.max().isoformat(),
        "last_train_label": train.target_time.max().isoformat(),
        "train_rows": len(train),
        "columns": COLUMNS,
    }


def eligible(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (
        (frame.reference_time >= start)
        & (frame.latest_exit_time < end)
        & frame.target.notna()
        & np.isfinite(frame[COLUMNS + ["sigma", "entry_open", "exit_open"]]).all(axis=1)
    )
    return frame.loc[mask].copy().reset_index(drop=True)


def predicted_block(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, config: dict):
    model, contract = supervised_fit(frame, start, config)
    block = eligible(frame, start, end)
    if len(block) < 2:
        raise DataError("Empty prediction block")
    block["probability"] = model.predict_proba(block[COLUMNS].to_numpy())[:, 1]
    block["predictor_fit_at"] = pd.Timestamp(contract["fit_at"])
    block["predictor_last_label"] = pd.Timestamp(contract["last_train_label"])
    return block, model, contract


def training_predictions(frame: pd.DataFrame, end: pd.Timestamp, config: dict):
    """Inner walk-forward only; no backward predictions from a full-history fit."""
    start = pd.Timestamp(config["data"]["agent_history_start"], tz="UTC")
    pieces, fits = [], []
    while start < end:
        next_start = min(start + config["data"]["inner_block_days"] * DAY, end)
        model, contract = supervised_fit(frame, start, config)
        # Do not purge at INTERNAL predictor refresh boundaries: positions persist.
        block = eligible(frame, start, end)
        block = block.loc[block.reference_time < next_start].copy()
        if len(block):
            block["probability"] = model.predict_proba(block[COLUMNS].to_numpy())[:, 1]
            block["predictor_fit_at"] = pd.Timestamp(contract["fit_at"])
            block["predictor_last_label"] = pd.Timestamp(contract["last_train_label"])
            pieces.append(block)
            fits.append((model, contract))
        start = next_start
    if not pieces:
        raise DataError("Empty RL training history")
    result = pd.concat(pieces, ignore_index=True)
    if not result.reference_time.diff().iloc[1:].eq(DAY).all():
        raise LeakageError("Non-contiguous RL training predictions")
    return result, fits
