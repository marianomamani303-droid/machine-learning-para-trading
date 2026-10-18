from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from trading_ml.errors import DataError
from trading_ml.smc.detectors import SMCConfig, smc_columns

EXTENSIONS = [
    "sr_body_atr",
    "sr_impulse_up",
    "sr_impulse_down",
    "sr_support_rejection",
    "sr_resistance_rejection",
    "sr_break_retest_up",
    "sr_break_retest_down",
    "sr_retest_impulse_up",
    "sr_retest_impulse_down",
    "sr_ob_impulse_up",
    "sr_ob_impulse_down",
]
STRUCTURE = smc_columns() + EXTENSIONS
DIRECTIONS = ["bear", "sideways", "bull"]
VOLATILITIES = ["low", "normal", "high"]
RULE_COLUMNS = [f"sr_direction_{x}" for x in DIRECTIONS] + [f"sr_vol_{x}" for x in VOLATILITIES]
CLUSTER_INPUT = ["adv_momentum_60", "adv_efficiency_20", "volatility"]
CLUSTER_COLUMNS = [f"sr_cluster_{i}" for i in range(3)]


def extend_structure(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Consume existing confirmed SMC events, without detecting pivots/OB again."""
    if (
        not isinstance(config["retest_bars"], int)
        or config["retest_bars"] < 1
        or not np.isfinite(list(config.values())).all()
        or config["tolerance_atr"] < 0
        or config["impulse_atr"] <= 0
        or not 0.5 < config["impulse_close_location"] <= 1
    ):
        raise DataError("Extensión de estructura inválida")
    f = frame.copy()
    defaults = SMCConfig()
    tr = pd.concat(
        [f.high - f.low, (f.high - f.close.shift()).abs(), (f.low - f.close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.shift().rolling(defaults.atr_window).mean().replace(0, np.nan)
    body = f.close - f.open
    location = (f.close - f.low) / (f.high - f.low).replace(0, np.nan)
    f["sr_body_atr"] = (body / atr).clip(-10, 10)
    f["sr_impulse_up"] = (
        (body >= config["impulse_atr"] * atr) & (location >= config["impulse_close_location"])
    ).astype(float)
    f["sr_impulse_down"] = (
        (-body >= config["impulse_atr"] * atr) & (location <= 1 - config["impulse_close_location"])
    ).astype(float)
    # Event in t already certifies the pivot at t-right. Never move it back to j.
    resistance = f.high.shift(defaults.swing_right).where(f.smc_pivot_high_confirmed == 1).ffill()
    support = f.low.shift(defaults.swing_right).where(f.smc_pivot_low_confirmed == 1).ffill()
    tolerance = config["tolerance_atr"] * atr
    f["sr_support_rejection"] = (
        (f.low <= support.shift() + tolerance)
        & (f.high >= support.shift() - tolerance)
        & (f.close > support.shift())
    ).astype(float)
    f["sr_resistance_rejection"] = (
        (f.low <= resistance.shift() + tolerance)
        & (f.high >= resistance.shift() - tolerance)
        & (f.close < resistance.shift())
    ).astype(float)
    for direction, level, names, suffix in [
        (1, resistance, ["smc_bos_up", "smc_choch_up"], "up"),
        (-1, support, ["smc_bos_down", "smc_choch_down"], "down"),
    ]:
        events = f[names].max(axis=1).to_numpy()
        retests = np.zeros(len(f))
        pending = None
        for i in range(len(f)):
            if pending is not None:
                born, price = pending
                invalid = direction * (f.close.iloc[i] - price) < -tolerance.iloc[i]
                if i - born > config["retest_bars"] or invalid:
                    pending = None
                elif (
                    f.low.iloc[i] <= price + tolerance.iloc[i]
                    and f.high.iloc[i] >= price - tolerance.iloc[i]
                    and direction * (f.close.iloc[i] - price) >= 0
                ):
                    retests[i] = 1
                    pending = None  # First valid retest only; never the birth candle.
            if events[i] and np.isfinite(level.iloc[i]):
                pending = (i, level.iloc[i])
        f[f"sr_break_retest_{suffix}"] = retests
        f[f"sr_retest_impulse_{suffix}"] = retests * f[f"sr_impulse_{suffix}"]
        side = "bull" if direction > 0 else "bear"
        f[f"sr_ob_impulse_{suffix}"] = f[f"smc_ob_{side}_touch"] * f[f"sr_impulse_{suffix}"]
    sigma = np.log(f.close).diff().rolling(20).std().replace(0, np.nan)
    low = sigma.shift().rolling(120).quantile(0.2)
    high = sigma.shift().rolling(120).quantile(0.8)
    # Direction comes from the existing causal detector in build_research_features.
    f["sr_direction"] = f.regime
    f["sr_vol"] = np.select([sigma < low, sigma > high], ["low", "high"], default="normal")
    f.loc[low.isna() | high.isna(), "sr_vol"] = "unknown"
    for name in DIRECTIONS:
        f[f"sr_direction_{name}"] = (f.sr_direction == name).astype(float)
    for name in VOLATILITIES:
        f[f"sr_vol_{name}"] = (f.sr_vol == name).astype(float)
    f["sr_state"] = f.sr_direction + "_" + f.sr_vol
    f["sr_ready"] = (
        f[EXTENSIONS + CLUSTER_INPUT].notna().all(axis=1)
        & (f.sr_vol != "unknown")
        & (f.sr_direction != "unknown")
    )
    return f


def structure_gate(frame: pd.DataFrame) -> np.ndarray:
    confirmed = frame[["sr_retest_impulse_up", "sr_ob_impulse_up"]].max(axis=1) > 0
    rejection = (frame.sr_support_rejection > 0) & (frame.sr_impulse_up > 0)
    return (confirmed | rejection).to_numpy()


def fit_clusters(train: pd.DataFrame, seed: int):
    """No labels, evaluation rows, smoothing or transductive full-history fit."""
    scaler = StandardScaler().fit(train[CLUSTER_INPUT])
    cluster = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(
        scaler.transform(train[CLUSTER_INPUT])
    )
    return scaler, cluster


def apply_clusters(frame: pd.DataFrame, scaler, cluster) -> pd.DataFrame:
    result = frame.copy()
    states = cluster.predict(scaler.transform(frame[CLUSTER_INPUT]))
    result["sr_cluster_state"] = states.astype(str)
    for i, column in enumerate(CLUSTER_COLUMNS):
        result[column] = (states == i).astype(float)
    return result
