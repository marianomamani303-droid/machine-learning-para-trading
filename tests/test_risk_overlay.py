from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import raw_file

from trading_ml.backtesting.simulator import simulate
from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.config.settings import Backtest
from trading_ml.errors import LeakageError
from trading_ml.momentum.config import ResearchCosts
from trading_ml.risk_overlay.config import RiskPolicy, entry_fraction
from trading_ml.risk_overlay.simulator import risk_features, run_overlay
from trading_ml.risk_overlay.study import run_risk_study
from trading_ml.training.pipeline import train


def timeline(raw):
    frame = raw.copy().reset_index(drop=True)
    frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
    return risk_features(frame)


def controlled_frame(raw):
    frame = timeline(raw.iloc[:20])
    frame["open"] = 100.0
    frame["close"] = 100.0
    frame["risk_atr"] = 2.0
    frame["risk_volatility"] = 0.02
    return frame


def test_original_overlay_reproduces_existing_simulator(raw):
    frame = timeline(raw)
    p = np.where(np.arange(len(frame)) % 3 == 0, 0.65, 0.45)
    p[198] = 0.0
    start, end = frame.timestamp.iloc[100], frame.timestamp.iloc[200]
    signals = frame.iloc[99:198][["timestamp", "reference_time"]].copy()
    signals["probability_up"] = p[99:198]
    native = simulate(frame, signals, 1, Backtest(), frame.reference_time.iloc[98])
    overlay = run_overlay(frame, p, start, end, ResearchCosts(spread_bps=0), RiskPolicy())
    result = overlay.simulation
    assert result.metrics["net_return"] == pytest.approx(
        native.summary["strategy_metrics"]["cumulative_return"]
    )
    assert result.metrics["max_drawdown"] == pytest.approx(
        native.summary["strategy_metrics"]["max_drawdown"]
    )
    pd.testing.assert_series_equal(result.trades.entry_time, native.trades.entry_time)
    pd.testing.assert_series_equal(result.trades.exit_time, native.trades.exit_time)
    np.testing.assert_allclose(result.trades.net_return, native.trades.net_return)
    assert result.metrics["execution_cost_cash"] == pytest.approx(
        overlay.events.execution_cost_cash.sum()
    )


def test_risk_features_and_simulation_prefix_ignore_future(raw):
    frame = timeline(raw)
    changed = raw.copy()
    changed.loc[350:, ["open", "high", "low", "close"]] *= 6
    altered = timeline(changed)
    pd.testing.assert_frame_equal(frame.iloc[:350], altered.iloc[:350])
    p = np.full(len(frame), 0.65)
    policy = RiskPolicy(
        name="combined",
        capital_cap=0.25,
        sizing="combined",
        atr_stop=True,
        soft_drawdown=0.05,
        hard_drawdown=0.1,
        persistent=True,
        max_holding_bars=5,
    )
    start, end = frame.timestamp.iloc[100], frame.timestamp.iloc[300]
    original = run_overlay(frame, p, start, end, ResearchCosts(), policy)
    second = run_overlay(altered, p, start, end, ResearchCosts(), policy)
    pd.testing.assert_frame_equal(original.simulation.equity, second.simulation.equity)
    pd.testing.assert_frame_equal(original.events, second.events)


def test_caps_are_enforced_and_dynamic_sizing_falls_with_risk(raw):
    frame = controlled_frame(raw)
    frame["open"] = np.arange(100, 120)
    frame["close"] = frame.open
    p = np.full(len(frame), 0.65)
    for cap in [0.1, 0.25, 0.5]:
        policy = RiskPolicy(name="cap", capital_cap=cap, persistent=True, max_holding_bars=5)
        result = run_overlay(
            frame, p, frame.timestamp.iloc[1], frame.timestamp.iloc[19], ResearchCosts(), policy
        )
        assert result.simulation.metrics["max_open_capital_exposure"] <= cap + 1e-12
        assert result.simulation.metrics["cap_reductions"] > 0
    policy = RiskPolicy(name="dynamic", capital_cap=0.5, sizing="combined")
    assert entry_fraction(policy, 0.05, 0.1, 0) < entry_fraction(policy, 0.02, 0.02, 0)


def test_atr_stop_uses_closed_candle_not_intravela_low(raw):
    frame = controlled_frame(raw)
    frame.loc[1:2, "low"] = 1.0
    frame.loc[3, "close"] = 93.0
    frame.loc[4, "open"] = 80.0
    p = np.zeros(len(frame))
    p[0] = 0.65
    policy = RiskPolicy(name="stop", capital_cap=0.5, atr_stop=True, max_holding_bars=10)
    result = run_overlay(
        frame, p, frame.timestamp.iloc[1], frame.timestamp.iloc[18], ResearchCosts(), policy
    )
    trade = result.simulation.trades.iloc[0]
    assert trade.entry_time == frame.timestamp.iloc[1]
    assert trade.exit_time == frame.timestamp.iloc[4]
    assert trade.exit_reason == "atr_close_stop"
    assert trade.net_return < -0.2  # Gap exceeds nominal ATR stop loss.


def test_one_bar_atr_stop_does_not_magically_anticipate_exit(raw):
    frame = controlled_frame(raw)
    frame.loc[1, "close"] = 90
    frame.loc[2, "open"] = 80
    p = np.full(len(frame), 0.65)
    atr = RiskPolicy(name="atr", capital_cap=0.5, sizing="atr")
    a = run_overlay(
        frame, p, frame.timestamp.iloc[1], frame.timestamp.iloc[18], ResearchCosts(), atr
    )
    b = run_overlay(
        frame,
        p,
        frame.timestamp.iloc[1],
        frame.timestamp.iloc[18],
        ResearchCosts(),
        replace(atr, name="stop", atr_stop=True),
    )
    np.testing.assert_allclose(a.simulation.equity.equity, b.simulation.equity.equity)


def test_drawdown_halt_is_latched_and_gross_replays_same_decisions(raw):
    frame = controlled_frame(raw)
    policy = RiskPolicy(name="dd", soft_drawdown=0.05, hard_drawdown=0.1)
    result = run_overlay(
        frame,
        np.full(len(frame), 0.65),
        frame.timestamp.iloc[1],
        frame.timestamp.iloc[19],
        ResearchCosts(commission_bps=500, slippage_bps=0, spread_bps=0),
        policy,
    )
    curve = result.simulation.equity
    assert curve.halted.any()
    first_halt = curve.index[curve.halted][0]
    assert curve.halted.iloc[first_halt:].all()
    assert (curve.capital_exposure.iloc[first_halt:] == 0).all()
    assert result.simulation.metrics["gross_return"] == pytest.approx(0)
    assert result.simulation.metrics["net_return"] < -0.1
    assert result.simulation.metrics["drawdown_rejections"] > 0


def test_soft_drawdown_reduces_existing_exposure(raw):
    frame = controlled_frame(raw)
    frame.loc[1, "close"] = 80
    frame.loc[2:, "open"] = 80
    frame.loc[2:, "close"] = 80
    policy = RiskPolicy(
        name="dd",
        capital_cap=0.5,
        soft_drawdown=0.05,
        hard_drawdown=0.3,
        persistent=True,
        max_holding_bars=10,
    )
    result = run_overlay(
        frame,
        np.full(len(frame), 0.65),
        frame.timestamp.iloc[1],
        frame.timestamp.iloc[19],
        ResearchCosts(),
        policy,
    )
    row = result.simulation.equity.loc[
        result.simulation.equity.timestamp == frame.timestamp.iloc[2]
    ].iloc[0]
    assert row.capital_exposure <= 0.25 + 1e-12
    assert result.simulation.metrics["cap_reductions"] > 0


def test_persistence_and_cooldown_reduce_churn_and_ignore_in_sample_confirmation(raw):
    frame = controlled_frame(raw)
    p = np.full(len(frame), 0.65)
    start, end = frame.timestamp.iloc[1], frame.timestamp.iloc[19]
    original = run_overlay(frame, p, start, end, ResearchCosts(), RiskPolicy())
    policy = RiskPolicy(
        name="persistent",
        capital_cap=0.25,
        confirmations=2,
        cooldown_bars=2,
        persistent=True,
        max_holding_bars=5,
    )
    reduced = run_overlay(frame, p, start, end, ResearchCosts(), policy)
    assert reduced.simulation.trades.entry_time.iloc[0] == frame.timestamp.iloc[2]
    assert reduced.simulation.metrics["trade_count"] < original.simulation.metrics["trade_count"]
    assert (
        reduced.simulation.metrics["execution_cost_cash"]
        < original.simulation.metrics["execution_cost_cash"]
    )


def test_rejects_signal_before_its_close(raw):
    frame = controlled_frame(raw)
    frame.loc[0, "reference_time"] += pd.Timedelta(hours=1)
    with pytest.raises(LeakageError):
        run_overlay(
            frame,
            np.ones(len(frame)),
            frame.timestamp.iloc[1],
            frame.timestamp.iloc[19],
            ResearchCosts(),
            RiskPolicy(),
        )


def test_study_protects_reservations_models_and_selects_no_test_parameters(config, raw, prepared):
    source = raw_file(config, raw)
    version = train(config, prepared)
    future_plan = config.root / "momentum_studies" / "reserved" / "plan.json"
    write_new_json(
        future_plan,
        {"prospective": {"start": "2026-10-14T00:00:00Z", "end": "2027-04-12T00:00:00Z"}},
    )
    future_hash = sha256(future_plan)
    source_hash = sha256(source)
    protocol = Path("config/risk_study.toml").read_text(encoding="utf-8")
    protocol = protocol.replace(
        next(line for line in protocol.splitlines() if line.startswith("reference_versions =")),
        f'reference_versions = ["{version}"]',
    )
    protocol = protocol.replace(
        'primary_reference = "v20261012T130540_d2bb6ac0"', f'primary_reference = "{version}"'
    )
    protocol = protocol.replace(
        next(line for line in protocol.splitlines() if line.startswith("boundaries =")),
        'boundaries = ["2021-01-01", "2021-03-01", "2021-05-05"]',
    )
    protocol = protocol.replace('historical_end = "2026-10-01"', 'historical_end = "2021-08-23"')
    path = config.root / "risk.toml"
    path.write_text(protocol, encoding="utf-8")
    summary = run_risk_study(config, path, source)
    assert summary["actual_fits"] == 2
    assert summary["prospective_evaluated"] is False
    assert summary["prospective_unchanged"] is True
    assert summary["diagnosis"]["predictive_capacity_changed"] is False
    assert sha256(source) == source_hash
    assert sha256(future_plan) == future_hash
    directory = Path(summary["directory"])
    assert read_json(directory / "matched_caps.json")["test_used"] is False
    for relative, expected in read_json(directory / "artifacts.json").items():
        assert sha256(directory / relative) == expected
    # A protocol crossing the reserved window must fail before evaluation.
    changed = protocol.replace('historical_end = "2021-08-23"', 'historical_end = "2026-11-01"')
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(LeakageError):
        run_risk_study(config, path, source)
