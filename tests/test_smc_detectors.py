import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_ml.config import Settings
from trading_ml.errors import DataError
from trading_ml.features import build_features, feature_columns
from trading_ml.smc import (
    SMCConfig,
    detect_smc,
    directional_states,
    rule_probabilities,
    smc_columns,
)


def frame_from_prices(opening, high, low, close):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=len(close), freq="1D", tz="UTC"),
            "open": opening,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        }
    )


def test_pivot_confirmation_bos_choch_and_ob_birth():
    frame = frame_from_prices(
        [10, 13, 11, 11.5, 14.5, 16, 13, 8],
        [11, 15, 13, 12, 17, 17.5, 14, 9],
        [9, 12, 10, 10.5, 14, 12, 7, 6],
        [10, 14, 12, 11, 16, 13, 8, 7],
    )
    config = SMCConfig(swing_left=1, swing_right=2, atr_window=2, liquidity_window=3)
    result = detect_smc(frame, config)
    assert result.smc_pivot_high_confirmed.iloc[:3].sum() == 0
    assert result.smc_pivot_high_confirmed.iloc[3] == 1
    assert result.smc_bos_up.iloc[4] == 1
    assert result.smc_choch_up.iloc[4] == 0
    assert result.smc_ob_up.iloc[:4].sum() == 0
    assert result.smc_ob_up.iloc[4] == 1
    ob = next(zone for zone in json.loads(result.smc_active_zones.iloc[4]) if zone["kind"] == "ob")
    assert ob["source_position"] == 3
    assert ob["created_position"] == 4
    assert result.smc_ob_bull_touch.iloc[4] == 0
    assert result.smc_ob_bull_touch.iloc[5] == 1
    assert result.smc_choch_down.iloc[6] == 1
    assert result.smc_bos_down.iloc[6] == 0
    assert result.smc_ob_bull_active.iloc[6] == 0
    assert result.smc_choch_down.iloc[7] == 0
    assert result.smc_trend.iloc[6] == -1


def test_liquidity_sweeps_use_previous_window_and_conflicts_abstain():
    frame = frame_from_prices(
        [10] * 5, [11, 11, 11, 12, 13], [9, 9, 9, 8.5, 9.5], [10, 10, 10, 10.5, 12]
    )
    result = detect_smc(
        frame, SMCConfig(swing_left=1, swing_right=1, atr_window=2, liquidity_window=3)
    )
    assert result.smc_sweep_high.iloc[3] == 1
    assert result.smc_sweep_low.iloc[3] == 1
    assert result.smc_sweep_high.iloc[4] == 0
    probability, state = rule_probabilities(result)
    assert probability[3] == 0.5
    assert state[3] == "neutral"


@pytest.mark.parametrize("bearish", [False, True])
def test_fvg_confirmation_partial_touch_and_full_mitigation(bearish):
    frame = frame_from_prices([10.0] * 20, [11.0] * 20, [9.0] * 20, [10.0] * 20)
    frame.loc[15, ["open", "high", "low", "close"]] = [10, 15, 9.5, 14]
    frame.loc[16, ["open", "high", "low", "close"]] = [14, 15.5, 13, 14.5]
    frame.loc[17, ["open", "high", "low", "close"]] = [14, 14.5, 12, 13.5]
    frame.loc[18, ["open", "high", "low", "close"]] = [12, 13, 10.5, 11.5]
    if bearish:
        original = frame.copy()
        for field in ["open", "close"]:
            frame[field] = 25 - original[field]
        frame["high"], frame["low"] = 25 - original.low, 25 - original.high
    result = detect_smc(frame, SMCConfig(atr_window=2))
    direction, zone = ("down", "bear") if bearish else ("up", "bull")
    assert result[f"smc_fvg_{direction}"].iloc[16] == 1
    assert result[f"smc_fvg_{zone}_touch"].iloc[16] == 0
    assert result[f"smc_fvg_{zone}_touch"].iloc[17] == 1
    assert result[f"smc_fvg_{zone}_active"].iloc[17] == 1
    assert result[f"smc_fvg_{zone}_active"].iloc[18] == 0


def test_structure_higher_highs_and_higher_lows():
    frame = frame_from_prices(
        [8, 9, 8, 10, 9, 11, 10],
        [10, 12, 11, 13, 12, 14, 13],
        [5, 7, 6, 8, 7, 9, 8],
        [8, 9, 8, 10, 9, 11, 10],
    )
    result = detect_smc(
        frame, SMCConfig(swing_left=1, swing_right=1, atr_window=2, liquidity_window=2)
    )
    assert result.smc_structure.iloc[5] == 1
    assert result.smc_higher_high.iloc[5] == 1
    assert result.smc_higher_low.iloc[5] == 1


@pytest.mark.parametrize("cutoff", [50, 150, 300])
def test_smc_prefixes_and_future_mutation_are_identical(raw, cutoff):
    full = detect_smc(raw)
    prefix = detect_smc(raw.iloc[: cutoff + 1])
    pd.testing.assert_frame_equal(full.iloc[: cutoff + 1], prefix)
    changed = raw.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close", "volume"]] *= 10
    pd.testing.assert_frame_equal(full.iloc[: cutoff + 1], detect_smc(changed).iloc[: cutoff + 1])


def test_smc_feature_integration_and_old_contracts(raw):
    base = Settings()
    assert not any(key.startswith("smc") for key in base.contract()["features"])
    enabled = replace(base.features, smc=True)
    result = build_features(raw, enabled)
    assert len(feature_columns(enabled)) == 14 + len(smc_columns())
    assert np.isfinite(result[smc_columns()].to_numpy()).all()
    assert replace(base, features=enabled).contract()["feature_schema"] == 3
    assert (
        replace(base, features=replace(base.features, enhanced=True)).contract()["feature_schema"]
        == 2
    )


def test_neutral_band_boundaries_and_insufficient_evidence():
    assert directional_states(np.array([0.45, 0.5, 0.55])).tolist() == [
        "bearish",
        "neutral",
        "bullish",
    ]
    assert directional_states(np.array([0.9]), ready=np.array([False])).tolist() == ["neutral"]
    with pytest.raises(DataError):
        directional_states(np.array([np.nan]))


def test_invalid_ohlcv_and_smc_config(raw):
    with pytest.raises(DataError):
        detect_smc(raw, SMCConfig(swing_right=0))
    raw.loc[10, "high"] = 0
    with pytest.raises(DataError):
        detect_smc(raw)


def test_zones_expire_and_capacity_is_bounded(raw):
    result = detect_smc(raw, SMCConfig(zone_age=1, max_zones_per_type=1))
    for text in result.smc_active_zones:
        zones = json.loads(text)
        assert all(zone["age_bars"] <= 1 for zone in zones)
        assert all(
            sum(z["kind"] == kind and z["direction"] == direction for z in zones) <= 1
            for kind in ["fvg", "ob"]
            for direction in ["bullish", "bearish"]
        )


def test_bearish_structure_is_symmetric():
    frame = frame_from_prices(
        [10, 13, 11, 11.5, 14.5, 16, 13, 8],
        [11, 15, 13, 12, 17, 17.5, 14, 9],
        [9, 12, 10, 10.5, 14, 12, 7, 6],
        [10, 14, 12, 11, 16, 13, 8, 7],
    )
    original = frame.copy()
    frame["open"], frame["close"] = 30 - original.open, 30 - original.close
    frame["high"], frame["low"] = 30 - original.low, 30 - original.high
    result = detect_smc(
        frame, SMCConfig(swing_left=1, swing_right=2, atr_window=2, liquidity_window=3)
    )
    assert result.smc_bos_down.iloc[4] == 1
    assert result.smc_ob_down.iloc[4] == 1
    assert result.smc_choch_up.iloc[6] == 1
    assert result.smc_trend.iloc[6] == 1
