from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import candles

from trading_ml.advanced_research.features import ORIGINAL
from trading_ml.backtesting.simulator import simulate
from trading_ml.config import Settings
from trading_ml.config.settings import Backtest, Features
from trading_ml.errors import DataError, LeakageError
from trading_ml.features import build_features
from trading_ml.label_research.events import (
    ExitRule,
    attach_labels,
    average_uniqueness,
    causal_scales,
    label_events,
)
from trading_ml.label_research.execution import simulate_events
from trading_ml.label_research.models import (
    Budget,
    fit_event_model,
    forecasts,
    meta_pool,
    partitions,
)
from trading_ml.label_research.study import validate_protocol
from trading_ml.momentum.config import ResearchCosts


def flat_frame(rows=90):
    times = pd.date_range("2020-01-01", periods=rows, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": times,
            "reference_time": times + pd.Timedelta(days=1),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1000,
            "label_sigma": 0.02,
            "label_atr_fraction": 0.03,
            "research_ready": True,
        }
    )


def zero_costs():
    return ResearchCosts(commission_bps=0, slippage_bps=0, spread_bps=0)


def test_intrabar_ties_keep_all_events_and_bound_outcomes():
    frame = flat_frame()
    frame.loc[31, ["high", "low"]] = [104, 97]
    rule = ExitRule("tb", 5, "percent", 0.03, 0.02)
    lower = label_events(frame, rule, zero_costs())
    upper = label_events(frame, rule, zero_costs(), tie="take_first")
    assert lower.loc[30, "ambiguous"] and upper.loc[30, "ambiguous"]
    assert lower.loc[30, "event_label"] == -1 and upper.loc[30, "event_label"] == 1
    assert lower.loc[30, "unit_net_return"] == pytest.approx(-0.02)
    assert upper.loc[30, "unit_net_return"] == pytest.approx(0.03)
    assert lower.loc[30, "target_time"] == frame.reference_time.iloc[31]


def test_stop_gaps_fill_at_open_and_tp_gaps_are_capped():
    frame = flat_frame()
    frame.loc[32, ["open", "high", "low", "close"]] = [95, 96, 94, 95]
    rule = ExitRule("tb", 5, "percent", 0.03, 0.02)
    labels = label_events(frame, rule, zero_costs())
    assert labels.loc[30, "reason"] == "sl_gap"
    assert labels.loc[30, "exit_price"] == 95
    frame.loc[32, ["open", "high", "low", "close"]] = [105, 106, 104, 105]
    labels = label_events(frame, rule, zero_costs())
    assert labels.loc[30, "reason"] == "tp_gap" and labels.loc[30, "exit_price"] == 103


def test_timeout_and_early_hits_without_full_future_horizon():
    frame = flat_frame()
    rule = ExitRule("tb", 5, "percent", 0.03, 0.02)
    labels = label_events(frame, rule, zero_costs())
    assert labels.loc[30, "reason"] == "timeout" and labels.loc[30, "exit_index"] == 36
    assert labels.loc[30, "target_time"] == frame.timestamp.iloc[36]
    assert pd.isna(labels.target.iloc[-2])
    frame.loc[len(frame) - 1, "high"] = 104
    labels = label_events(frame, rule, zero_costs())
    assert labels.target.iloc[-2] == 1  # Known early hit; do not censor unnecessarily.
    assert pd.isna(labels.target.iloc[-1])


def test_tp_event_is_not_necessarily_a_profitable_trade():
    frame = flat_frame()
    rule = ExitRule("tiny", 3, "percent", 0.001, 0.02)
    labels = label_events(frame, rule, ResearchCosts())
    assert labels.loc[30, "event_label"] == 1
    assert labels.loc[30, "unit_net_return"] < 0 and labels.loc[30, "target"] == 0


def test_close_control_does_not_pretend_to_observe_intrabar_order():
    frame = flat_frame()
    frame.loc[31, ["high", "low"]] = [110, 90]
    labels = label_events(frame, ExitRule("close", 3, "close"), zero_costs())
    assert labels.loc[30, "reason"] == "timeout" and not labels.loc[30, "ambiguous"]


def test_volatility_widths_are_prefix_causal_and_frozen():
    frame = candles(200)
    frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
    before = causal_scales(frame)
    changed = frame.copy()
    changed.loc[100:, ["open", "high", "low", "close"]] *= 5
    after = causal_scales(changed)
    pd.testing.assert_frame_equal(before.iloc[:100], after.iloc[:100])
    rule = ExitRule("vol", 5, "vol")
    a, b = label_events(before, rule, zero_costs()), label_events(after, rule, zero_costs())
    pd.testing.assert_frame_equal(
        a.loc[:99, ["up_fraction", "down_fraction"]], b.loc[:99, ["up_fraction", "down_fraction"]]
    )
    assert a.loc[99, "up_fraction"] == pytest.approx(1.5 * before.label_sigma.iloc[99])


@pytest.mark.parametrize(
    "rule",
    [
        ExitRule("fixed", 1),
        ExitRule("vol", 5, "vol"),
        ExitRule("atr", 5, "atr"),
        ExitRule("close", 5, "close"),
    ],
)
def test_labels_and_forward_execution_have_identical_payoffs(rule):
    frame = flat_frame()
    frame.loc[33, ["high", "low", "close"]] = [104, 99, 103.5]
    costs = ResearchCosts(annual_holding_bps=100)
    labels = label_events(frame, rule, costs)
    p = np.zeros(len(frame))
    p[29] = 0.9
    result = simulate_events(
        frame, p, rule, frame.timestamp.iloc[30], frame.timestamp.iloc[70], costs
    )
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade.net_return == pytest.approx(labels.loc[29, "unit_net_return"])
    assert trade.exit_time == labels.loc[29, "target_time"]
    assert trade.exit_price == labels.loc[29, "exit_price"]
    assert result.metrics["net_return"] == pytest.approx(trade.net_return)
    assert result.metrics["execution_cost_cash"] > 0 and result.metrics["holding_cost_cash"] > 0


def test_future_labels_never_change_signals_or_execution():
    frame = flat_frame()
    p = np.full(len(frame), 0.8)
    start, end = frame.timestamp.iloc[30], frame.timestamp.iloc[70]
    rule = ExitRule("vol", 5, "vol")
    before = simulate_events(frame, p, rule, start, end, ResearchCosts())
    changed = frame.assign(
        target=1.0,
        target_time=pd.Timestamp("2030-01-01", tz="UTC"),
        event_label=-1,
        ambiguous=True,
        exit_index=0,
    )
    after = simulate_events(changed, p, rule, start, end, ResearchCosts())
    pd.testing.assert_frame_equal(before.curve, after.curve)
    pd.testing.assert_frame_equal(before.trades, after.trades)
    pd.testing.assert_frame_equal(before.decisions, after.decisions)
    assert before.trades.signal_reference.max() < end - pd.Timedelta(days=11)


def test_intrabar_exit_cannot_free_a_position_for_the_same_open():
    frame = flat_frame()
    p = np.full(len(frame), 0.8)
    start, end = frame.timestamp.iloc[30], frame.timestamp.iloc[70]
    rule = ExitRule("tb", 5, "percent", 0.03, 0.02)
    frame.loc[31, "high"] = 104
    result = simulate_events(frame, p, rule, start, end, zero_costs())
    day = result.decisions.loc[result.decisions.reference_time == frame.timestamp.iloc[31]].iloc[0]
    assert day.reason == "position_open"
    assert 31 not in result.trades.entry_index.to_list()


def test_original_h1_tradebook_matches_existing_simulator():
    frame = candles(100)
    frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
    frame = causal_scales(frame)
    p = np.full(len(frame), 0.8)
    start, end = frame.timestamp.iloc[30], frame.timestamp.iloc[90]
    costs = ResearchCosts()
    new = simulate_events(frame, p, ExitRule("net_h1", 1), start, end, costs)
    selected = (frame.reference_time >= start) & (
        frame.reference_time < end - pd.Timedelta(days=11)
    )
    signals = frame.loc[selected, ["timestamp", "reference_time"]].assign(
        probability_up=p[selected]
    )
    old = simulate(
        frame,
        signals,
        1,
        Backtest(
            commission_bps=costs.commission_bps,
            slippage_bps=costs.slippage_bps + costs.spread_bps / 2,
        ),
        start - pd.Timedelta(days=1),
    )
    np.testing.assert_allclose(old.trades.net_return, new.trades.net_return, atol=1e-12)
    np.testing.assert_array_equal(old.trades.entry_time, new.trades.entry_time)
    np.testing.assert_array_equal(old.trades.exit_time, new.trades.exit_time)
    assert old.summary["strategy_metrics"]["cumulative_return"] == pytest.approx(
        new.metrics["net_return"]
    )


def test_overlap_uniqueness_downweights_crowded_training_events():
    events = pd.DataFrame(
        {
            "reference_time": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-10"], utc=True),
            "target_time": pd.to_datetime(["2020-01-03", "2020-01-03", "2020-01-11"], utc=True),
        }
    )
    weights = average_uniqueness(events)
    assert weights[0] == weights[1] < weights[2] and weights.mean() == pytest.approx(1)


def test_meta_pool_requires_genuinely_oos_parent_scores():
    frame = flat_frame()
    events = label_events(frame, ExitRule("fixed", 1), zero_costs())
    selected = frame.iloc[30:60]
    oof = pd.DataFrame(
        {
            "source_index": selected.index,
            "reference_time": selected.reference_time,
            "probability": 0.6,
            "model_file": "model",
            "model_cutoff": selected.reference_time.iloc[0],
            "last_train_target": selected.reference_time.iloc[0] - pd.Timedelta(days=7),
        }
    )
    pool = meta_pool(frame, oof, events, frame.reference_time.iloc[50])
    assert pool.index.min() == 30 and pool.index.max() == 49
    oof["last_train_target"] = oof.model_cutoff
    with pytest.raises(DataError, match="fuera de muestra"):
        meta_pool(frame, oof, events, frame.reference_time.iloc[50])


def test_fitting_and_calibration_are_temporally_disjoint_and_future_invariant(tmp_path):
    protocol = tomllib.loads(Path("config/label_study.toml").read_text(encoding="utf-8"))
    frame = build_features(candles(700), Features())
    frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
    frame = causal_scales(frame)
    frame["research_ready"] = frame[ORIGINAL].notna().all(axis=1)
    frame = attach_labels(frame, label_events(frame, ExitRule("tb", 5, "vol"), ResearchCosts()))
    start = frame.reference_time.iloc[600]
    parameters = {"n_estimators": 4, "max_depth": 2, "learning_rate": 0.1}
    a, audit = fit_event_model(frame, start, ORIGINAL, protocol, parameters, Budget(2), tmp_path)
    train, cal = partitions(frame, start, protocol["research"])
    assert train.target_time.max() < cal.reference_time.min() - pd.Timedelta(days=6)
    assert cal.target_time.max() < start - pd.Timedelta(days=6)
    changed = frame.copy()
    changed.loc[changed.reference_time >= start, "target"] = 1 - changed.target
    changed.loc[changed.reference_time >= start, ORIGINAL] *= 3
    b, second = fit_event_model(changed, start, ORIGINAL, protocol, parameters, Budget(2), tmp_path)
    assert audit == second and a.calibration == b.calibration
    np.testing.assert_allclose(forecasts(a, frame)[0], forecasts(b, frame)[0])


def test_protocol_and_reserved_data_guards_before_market_reads(tmp_path, monkeypatch):
    import trading_ml.label_research.study as study
    from trading_ml.common import write_new_json

    protocol = tomllib.loads(Path("config/label_study.toml").read_text(encoding="utf-8"))
    assert validate_protocol(protocol)[2] == 490
    write_new_json(
        tmp_path / "frontier_studies" / "frozen" / "prospective.json",
        {"start": "2025-01-01", "end": "2027-01-01"},
    )
    monkeypatch.setattr(
        study, "_load_panel", lambda *args: pytest.fail("Se leyó mercado reservado")
    )
    with pytest.raises(LeakageError):
        study.run_label_study(replace(Settings(), root=tmp_path), Path("config/label_study.toml"))


def test_fold_aggregation_compounds_wealth_and_scales_nominal_costs():
    from trading_ml.label_research.report import aggregate

    curve = pd.DataFrame(
        {
            "fold": [0, 1],
            "timestamp": pd.to_datetime(["2022-01-02", "2022-01-03"], utc=True),
            "equity": [9000.0, 12000.0],
            "gross_equity": [9100.0, 12100.0],
            "net_return": [-0.1, 0.2],
            "gross_return": [-0.09, 0.21],
            "exposed": True,
            "capital_exposure": 1.0,
            "execution_cost_cash": [100.0, 200.0],
            "commission_cash": [60.0, 120.0],
            "slippage_spread_cash": [40.0, 80.0],
            "turnover": 2.0,
        }
    )
    trades = pd.DataFrame(
        {
            "fold": [0, 1],
            "pnl_cash": [-1000.0, 2000.0],
            "capital_before": 10000.0,
            "capital_after": [9000.0, 12000.0],
            "holding_cost_cash": 0.0,
            "net_return": [-0.1, 0.2],
            "ambiguous": False,
            "holding_days": 1.0,
        }
    )
    metrics = aggregate(curve, trades, 10000.0)
    assert metrics["net_return"] == pytest.approx(0.08)
    assert metrics["gross_return"] == pytest.approx(0.91 * 1.21 - 1)
    assert metrics["execution_cost_cash"] == pytest.approx(100 + 200 * 0.9)
    assert metrics["profit_factor"] == pytest.approx(1.8)


def test_cash_like_filter_cannot_demonstrate_positive_edge_vs_cash():
    from trading_ml.label_research.report import economic_intervals

    dates = pd.date_range("2022-01-01", periods=70, tz="UTC")
    rows = []
    for strategy, daily in [("original", -0.001), ("candidate", 0.0), ("cash", 0.0)]:
        rows.append(
            pd.DataFrame(
                {
                    "fold": np.repeat([0, 1], 35),
                    "timestamp": dates,
                    "strategy": strategy,
                    "net_return": daily,
                }
            )
        )
    result = economic_intervals(
        pd.concat(rows),
        ["candidate"],
        {"candidate": "original"},
        {"samples": 100, "block_lengths": [14, 30], "seed": 42, "confidence": 0.95},
    )
    for block in result["blocks"].values():
        assert block["candidate__minus__cash"]["lower"] == 0
        assert block["candidate__minus__original"]["lower"] > 0


@pytest.mark.parametrize("relative", ["summary.json", "../escaped.json"])
def test_label_verifier_rejects_changed_or_escaped_artifacts_before_model_reads(tmp_path, relative):
    from trading_ml.common import write_new_json
    from trading_ml.label_research.verify import verify_label_study

    write_new_json(tmp_path / "artifacts.json", {relative: "bad-digest"})
    write_new_json(tmp_path / "summary.json", {"changed": True})
    with pytest.raises(DataError, match="alterado"):
        verify_label_study(Settings(), tmp_path)


def test_censored_label_timestamps_survive_csv_verification():
    from trading_ml.label_research.verify import assert_times

    left = pd.to_datetime(["2022-01-01", None], utc=True)
    right = pd.Series(left.as_unit("ns"))
    assert_times(left, right)
    with pytest.raises(AssertionError):
        assert_times(left, pd.to_datetime(["2022-01-02", None], utc=True))
