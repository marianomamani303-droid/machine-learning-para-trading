from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_ml.errors import LeakageError
from trading_ml.momentum import (
    MomentumConfig,
    ResearchCosts,
    build_momentum_features,
    momentum_columns,
    momentum_signals,
)
from trading_ml.momentum.simulator import opportunity_labels, simulate_policy
from trading_ml.momentum.study import fit_momentum_filter


def timeline(raw):
    frame = raw.copy()
    frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
    return frame


def test_features_and_signals_are_prefix_causal(raw):
    cfg = MomentumConfig(lookback=20)
    full = build_momentum_features(timeline(raw), cfg)
    prefix = build_momentum_features(timeline(raw.iloc[:350]), cfg)
    pd.testing.assert_frame_equal(full.iloc[:350], prefix)
    changed = timeline(raw)
    changed.loc[350:, ["open", "high", "low", "close"]] *= 7
    adversarial = build_momentum_features(changed, cfg)
    pd.testing.assert_frame_equal(full.iloc[:350], adversarial.iloc[:350])
    pd.testing.assert_frame_equal(
        momentum_signals(full, cfg).iloc[:350], momentum_signals(adversarial, cfg).iloc[:350]
    )
    assert not {"target", "target_time", "trade_net_return", "future_return"} & set(
        momentum_columns()
    )


def test_prior_breakout_excludes_current_high_and_low(raw):
    cfg = MomentumConfig(lookback=20)
    source = timeline(raw)
    expected = source.close.iloc[100] / source.high.iloc[80:100].max() - 1
    source.loc[100, "high"] *= 50
    source.loc[100, "low"] /= 50
    features = build_momentum_features(source, cfg)
    assert features.mom_high_distance.iloc[100] == pytest.approx(expected)
    assert features.mom_low_distance.iloc[100] == pytest.approx(
        source.close.iloc[100] / source.low.iloc[80:100].min() - 1
    )


def test_next_open_persistence_exit_and_costs(raw):
    frame = timeline(raw.iloc[:12]).reset_index(drop=True)
    frame["open"] = np.arange(100.0, 112.0)
    signals = pd.DataFrame({"entry": False, "exit": False}, index=frame.index)
    signals.loc[1, "entry"] = True
    signals.loc[5, "exit"] = True
    costs = ResearchCosts()
    result = simulate_policy(
        frame, signals, frame.timestamp.iloc[2], frame.timestamp.iloc[11], costs, 20
    )
    trade = result.trades.iloc[0]
    assert trade.entry_time == frame.timestamp.iloc[2]
    assert trade.exit_time == frame.timestamp.iloc[6]
    assert trade.holding_bars == 4
    assert trade.exit_reason == "trend_exit"
    expected = 106 / 102 * costs.sell_factor / costs.buy_factor - 1
    assert trade.net_return == pytest.approx(expected)
    assert result.metrics["net_return"] == pytest.approx(expected)
    assert result.metrics["max_drawdown"] < 0  # Includes entry costs against initial capital.
    labels = opportunity_labels(frame, signals, 20, costs)
    assert labels.trade_net_return.iloc[0] == pytest.approx(expected)
    assert labels.target_time.iloc[0] == trade.exit_time


def test_ml_only_gates_entry_and_max_holding_is_enforced(raw):
    frame = timeline(raw.iloc[:15]).reset_index(drop=True)
    signals = pd.DataFrame({"entry": False, "exit": False}, index=frame.index)
    signals.loc[1, "entry"] = True
    result = simulate_policy(
        frame, signals, frame.timestamp.iloc[2], frame.timestamp.iloc[12], ResearchCosts(), 3
    )
    assert result.trades.holding_bars.iloc[0] == 3
    assert result.trades.exit_reason.iloc[0] == "max_holding"
    blocked = simulate_policy(
        frame,
        signals,
        frame.timestamp.iloc[2],
        frame.timestamp.iloc[12],
        ResearchCosts(),
        3,
        probabilities=np.zeros(len(frame)),
    )
    assert blocked.trades.empty
    assert blocked.metrics["net_return"] == 0
    assert blocked.metrics["turnover"] == 0


def test_censored_labels_are_not_forced_profitable_at_boundary(raw):
    frame = timeline(raw.iloc[:12]).reset_index(drop=True)
    signals = pd.DataFrame({"entry": False, "exit": False}, index=frame.index)
    signals.loc[9, "entry"] = True
    assert opportunity_labels(frame, signals, 20, ResearchCosts()).empty
    result = simulate_policy(
        frame, signals, frame.timestamp.iloc[8], frame.timestamp.iloc[11], ResearchCosts(), 20
    )
    assert result.trades.empty  # Final opening never starts a new position.


def test_train_purges_actual_exit_and_is_invariant_to_future(raw):
    frame = build_momentum_features(timeline(raw), MomentumConfig(lookback=20))
    indices = np.arange(100, 400, 10)
    labels = pd.DataFrame(
        {
            "source_index": indices,
            "reference_time": frame.reference_time.iloc[indices].to_numpy(),
            "target_time": frame.timestamp.iloc[indices + 15].to_numpy(),
            "target": np.arange(len(indices)) % 2,
        }
    )
    cutoff = frame.timestamp.iloc[300]
    params = {"n_estimators": 5, "max_depth": 2}
    model, audit = fit_momentum_filter(frame, labels, cutoff, params, 42, 5)
    assert pd.Timestamp(audit["last_target_time"]) < cutoff
    future_labels = labels.copy()
    future_labels.loc[future_labels.target_time >= cutoff, "target"] = 1
    altered = frame.copy()
    altered.loc[300:, momentum_columns()] = 999
    model2, audit2 = fit_momentum_filter(altered, future_labels, cutoff, params, 42, 5)
    assert audit == audit2
    np.testing.assert_array_equal(
        model.predict_proba(frame.loc[100:250, momentum_columns()]),
        model2.predict_proba(frame.loc[100:250, momentum_columns()]),
    )


def test_rejects_execution_before_signal_available(raw):
    frame = timeline(raw.iloc[:20]).reset_index(drop=True)
    frame.loc[2, "reference_time"] += pd.Timedelta(hours=1)
    signals = pd.DataFrame({"entry": True, "exit": False}, index=frame.index)
    with pytest.raises(LeakageError):
        simulate_policy(
            frame, signals, frame.timestamp.iloc[3], frame.timestamp.iloc[12], ResearchCosts(), 3
        )


def test_financing_uses_elapsed_calendar_and_labels_match(raw):
    frame = timeline(raw.iloc[:20]).reset_index(drop=True)
    frame["open"] = 100.0
    # A session gap is allowed by the provider-independent engine.
    frame.loc[6:, "timestamp"] += pd.Timedelta(days=2)
    frame.loc[6:, "reference_time"] += pd.Timedelta(days=2)
    signals = pd.DataFrame({"entry": False, "exit": False}, index=frame.index)
    signals.loc[1, "entry"] = True
    costs = replace(ResearchCosts(), annual_holding_bps=1000)
    result = simulate_policy(
        frame, signals, frame.timestamp.iloc[2], frame.timestamp.iloc[14], costs, 5
    )
    labels = opportunity_labels(frame, signals, 5, costs)
    assert result.trades.net_return.iloc[0] == pytest.approx(labels.trade_net_return.iloc[0])
    assert result.trades.holding_bars.iloc[0] == 5


def test_insufficient_training_abstains(raw):
    frame = build_momentum_features(timeline(raw), MomentumConfig(lookback=20))
    labels = pd.DataFrame(columns=["source_index", "reference_time", "target_time", "target"])
    model, audit = fit_momentum_filter(
        frame, labels, frame.timestamp.iloc[300], {"n_estimators": 5}, 42, 15
    )
    assert model is None
    assert audit["status"].startswith("abstain")


def test_momentum_future_reservation_is_protected_by_existing_training_guard(tmp_path):
    from trading_ml.common import write_new_json
    from trading_ml.training.recipe import protect_reserved_labels

    write_new_json(
        tmp_path / "momentum_studies" / "run" / "plan.json",
        {"prospective": {"start": "2026-10-14T00:00:00Z", "end": "2027-04-12T00:00:00Z"}},
    )
    frame = pd.DataFrame(
        {
            "reference_time": pd.to_datetime(["2026-10-01"], utc=True),
            "target_time": pd.to_datetime(["2026-10-15"], utc=True),
        }
    )
    with pytest.raises(LeakageError):
        protect_reserved_labels(tmp_path, frame)
    frame["target_time"] = pd.to_datetime(["2026-10-13"], utc=True)
    protect_reserved_labels(tmp_path, frame)


def test_complete_study_records_all_scenarios_and_preserves_registry(config, raw):
    from pathlib import Path

    from conftest import raw_file

    from trading_ml.common import read_json, sha256, write_new_json
    from trading_ml.momentum.study import run_momentum_study

    source = raw_file(config, raw)
    source_hash = sha256(source)
    pointer = config.root / "models" / "latest.json"
    write_new_json(pointer, {"version": "unchanged"})
    pointer_hash = sha256(pointer)
    protocol = Path("config/momentum_study.toml").read_text(encoding="utf-8")
    protocol = protocol.replace("lookbacks = [20, 60, 120, 250]", "lookbacks = [8]")
    protocol = protocol.replace(
        next(line for line in protocol.splitlines() if line.startswith("boundaries =")),
        'boundaries = ["2020-09-01", "2021-03-01", "2021-06-01"]',
    )
    protocol = protocol.replace('historical_end = "2026-10-01"', 'historical_end = "2021-08-23"')
    protocol = (
        protocol.replace(
            next(line for line in protocol.splitlines() if line.startswith("reference_versions =")),
            "reference_versions = []",
        )
        .replace("n_estimators = 120", "n_estimators = 5")
        .replace("samples = 2000", "samples = 100")
    )
    path = config.root / "protocol.toml"
    path.write_text(protocol, encoding="utf-8")
    summary = run_momentum_study(config, path, source)
    directory = Path(summary["directory"])
    assert summary["experiment_records"] == 25  # 3 windows × 4 systems × 2 costs + frozen model.
    assert summary["promoted"] is False
    assert summary["independent_advantage_confirmed"] is False
    assert sha256(source) == source_hash
    assert sha256(pointer) == pointer_hash
    for relative, expected in read_json(directory / "artifacts.json").items():
        assert sha256(directory / relative) == expected
    assert (directory / "REPORT.md").exists()
    for period in [
        "development",
        "development_stress",
        "observed_historical",
        "observed_historical_stress",
    ]:
        assert len(summary["periods"][period]["systems"]) == 4
        assert "calendar_quarters" in summary["periods"][period]["systems"]["cash"]
