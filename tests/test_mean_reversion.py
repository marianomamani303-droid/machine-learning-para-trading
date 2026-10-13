from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import raw_file

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.errors import LeakageError
from trading_ml.mean_reversion import (
    ResearchCosts,
    ReversionConfig,
    build_reversion_features,
    reversion_columns,
    reversion_opportunities,
)
from trading_ml.mean_reversion.models import fit_filter, load_filter
from trading_ml.mean_reversion.simulator import entry_plan, opportunity_labels, simulate_reversion
from trading_ml.mean_reversion.study import run_reversion_study
from trading_ml.training.recipe import protect_reserved_labels


def timeline(raw):
    result = raw.copy().reset_index(drop=True)
    result["reference_time"] = result.timestamp + pd.Timedelta(days=1)
    return result


def execution_frame(raw):
    frame = timeline(raw.iloc[:16])
    frame["open"] = 100.0
    frame["close"] = 100.0
    frame["mr_mean"] = 120.0
    frame["mr_atr"] = 2.0
    frame["regime"] = "sideways"
    opportunities = pd.Series(False, index=frame.index)
    opportunities.iloc[1] = True
    return frame, opportunities


def test_features_and_opportunities_are_prefix_causal(raw):
    config = ReversionConfig()
    frame = timeline(raw)
    features = build_reversion_features(frame, config)
    prefix = build_reversion_features(frame.iloc[:350], config)
    pd.testing.assert_frame_equal(features.iloc[:350], prefix)
    changed = frame.copy()
    changed.loc[350:, ["open", "high", "low", "close"]] *= 8
    changed_features = build_reversion_features(changed, config)
    pd.testing.assert_frame_equal(features.iloc[:350], changed_features.iloc[:350])
    pd.testing.assert_series_equal(
        reversion_opportunities(features, config).iloc[:350],
        reversion_opportunities(changed_features, config).iloc[:350],
    )
    assert not {"target", "target_time", "mr_mean", "mr_std", "trade_net_return"} & set(
        reversion_columns()
    )


def test_zscore_uses_previous_window_and_is_scale_invariant(raw):
    frame = timeline(raw)
    features = build_reversion_features(frame, ReversionConfig())
    expected = (frame.close.iloc[100] - frame.close.iloc[80:100].mean()) / frame.close.iloc[
        80:100
    ].std(ddof=1)
    assert features.mr_z.iloc[100] == pytest.approx(expected)
    scaled = frame.copy()
    scaled[["open", "high", "low", "close"]] *= 100
    other = build_reversion_features(scaled, ReversionConfig())
    np.testing.assert_allclose(features.mr_z, other.mr_z, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_flat_market_has_no_extreme_opportunities(raw):
    frame = timeline(raw)
    frame[["open", "high", "low", "close"]] = 100.0
    features = build_reversion_features(frame, ReversionConfig())
    assert not reversion_opportunities(features, ReversionConfig()).any()
    assert (features.mr_z.dropna() == 0).all()


def test_stop_gaps_can_exceed_risk_budget_and_label_matches_trade(raw):
    frame, opportunities = execution_frame(raw)
    frame.loc[2, "close"] = 95
    frame.loc[3, "open"] = 90
    config, costs = ReversionConfig(), ResearchCosts()
    plan = entry_plan(frame, 1, config, costs)
    assert plan["planned_loss_fraction"] <= config.risk_fraction + 1e-12
    result = simulate_reversion(
        frame, opportunities, frame.timestamp.iloc[2], frame.timestamp.iloc[14], config, costs
    )
    label = opportunity_labels(frame, opportunities, config, costs).iloc[0]
    trade = result.trades.iloc[0]
    assert trade.entry_time == frame.timestamp.iloc[2]
    assert trade.exit_time == frame.timestamp.iloc[3]
    assert trade.exit_reason == "stop"
    assert trade.risk_budget_exceeded
    assert label.target == 0
    assert label.target_time == trade.exit_time
    assert trade.net_return == pytest.approx(label.trade_net_return)
    assert trade.portfolio_contribution == pytest.approx(label.portfolio_contribution)
    assert result.metrics["net_return"] == pytest.approx(trade.portfolio_contribution)


def test_reversion_is_not_a_guarantee_of_net_profit(raw):
    frame, opportunities = execution_frame(raw)
    frame.loc[2, "close"] = 121  # Target reached at a closed candle.
    frame.loc[3, "open"] = 90  # Exit gap loses money despite correct event forecast.
    config, costs = ReversionConfig(), ResearchCosts()
    labels = opportunity_labels(frame, opportunities, config, costs)
    result = simulate_reversion(
        frame, opportunities, frame.timestamp.iloc[2], frame.timestamp.iloc[14], config, costs
    )
    assert labels.target.iloc[0] == 1
    assert labels.trade_net_return.iloc[0] < 0
    assert result.metrics["win_rate"] == 0
    assert result.metrics["reversion_exit_fraction"] == 1


def test_timeout_censoring_and_anchored_target(raw):
    frame, opportunities = execution_frame(raw)
    frame.loc[2:, "mr_mean"] = 50  # Drifting MA must not cause an apparent reversion.
    config = replace(ReversionConfig(), max_holding_bars=4)
    labels = opportunity_labels(frame, opportunities, config, ResearchCosts())
    assert labels.exit_reason.iloc[0] == "timeout"
    assert labels.target_time.iloc[0] == frame.timestamp.iloc[6]
    result = simulate_reversion(
        frame,
        opportunities,
        frame.timestamp.iloc[2],
        frame.timestamp.iloc[5],
        config,
        ResearchCosts(),
    )
    assert result.trades.exit_reason.iloc[0] == "window_end"
    truncated = frame.iloc[:5]
    censored = opportunity_labels(truncated, opportunities.iloc[:5], config, ResearchCosts())
    assert censored.label_status.iloc[0] == "right_censored"
    assert pd.isna(censored.target.iloc[0])


def test_ml_only_gates_entries_and_gap_above_target_is_rejected(raw):
    frame, opportunities = execution_frame(raw)
    config, costs = ReversionConfig(), ResearchCosts()
    blocked = simulate_reversion(
        frame,
        opportunities,
        frame.timestamp.iloc[2],
        frame.timestamp.iloc[14],
        config,
        costs,
        np.zeros(len(frame)),
    )
    assert blocked.trades.empty
    assert blocked.metrics["rejected_by_probability"] == 1
    assert blocked.metrics["net_return"] == 0
    frame.loc[2, "open"] = 125
    result = simulate_reversion(
        frame, opportunities, frame.timestamp.iloc[2], frame.timestamp.iloc[14], config, costs
    )
    assert result.trades.empty
    assert result.metrics["nonexecutable_at_open"] == 1


@pytest.mark.parametrize(
    "kind,parameters",
    [("logistic", {"C": 1.0, "max_iter": 2000}), ("xgb", {"n_estimators": 5, "max_depth": 2})],
)
def test_temporal_models_scaler_and_serialization_ignore_future(raw, tmp_path, kind, parameters):
    frame = build_reversion_features(timeline(raw), ReversionConfig())
    indices = np.arange(100, 400, 10)
    labels = pd.DataFrame(
        {
            "source_index": indices,
            "reference_time": frame.reference_time.iloc[indices].to_numpy(),
            "target_time": frame.timestamp.iloc[indices + 15].to_numpy(),
            "target": np.arange(len(indices)) % 2,
            "label_status": "observed",
        }
    )
    cutoff = frame.timestamp.iloc[300]
    model = fit_filter(frame, labels, cutoff, kind, parameters, 5)
    assert pd.Timestamp(model.audit["last_target_time"]) < cutoff
    changed = frame.copy()
    changed.loc[300:, reversion_columns()] = 1000
    changed_labels = labels.copy()
    changed_labels.loc[labels.target_time >= cutoff, "target"] = 1
    second = fit_filter(changed, changed_labels, cutoff, kind, parameters, 5)
    assert model.audit == second.audit
    np.testing.assert_array_equal(
        model.probabilities(frame.iloc[:250]), second.probabilities(frame.iloc[:250])
    )
    path = tmp_path / kind
    model.save(path)
    restored = load_filter(path)
    np.testing.assert_allclose(
        model.probabilities(frame), restored.probabilities(frame), rtol=1e-12, atol=1e-12
    )


def test_rejects_unclosed_signal_and_protects_future_reservation(raw, tmp_path):
    frame, opportunities = execution_frame(raw)
    frame.loc[1, "reference_time"] += pd.Timedelta(hours=1)
    with pytest.raises(LeakageError):
        simulate_reversion(
            frame,
            opportunities,
            frame.timestamp.iloc[2],
            frame.timestamp.iloc[14],
            ReversionConfig(),
            ResearchCosts(),
        )
    write_new_json(
        tmp_path / "mean_reversion_studies" / "run" / "plan.json",
        {"prospective": {"start": "2026-10-14T00:00:00Z", "end": "2027-04-12T00:00:00Z"}},
    )
    labels = pd.DataFrame(
        {
            "reference_time": pd.to_datetime(["2026-10-01"], utc=True),
            "target_time": pd.to_datetime(["2026-10-15"], utc=True),
        }
    )
    with pytest.raises(LeakageError):
        protect_reserved_labels(tmp_path, labels)


def test_complete_study_keeps_models_and_every_variant(config, raw):
    source = raw_file(config, raw)
    source_hash = sha256(source)
    pointer = config.root / "models" / "latest.json"
    write_new_json(pointer, {"version": "preserved"})
    pointer_hash = sha256(pointer)
    protocol = Path("config/mean_reversion_study.toml").read_text(encoding="utf-8")
    protocol = protocol.replace("lookbacks = [20, 60]", "lookbacks = [20]").replace(
        'regime_filters = ["all", "sideways"]', 'regime_filters = ["all"]'
    )
    protocol = protocol.replace(
        next(line for line in protocol.splitlines() if line.startswith("boundaries =")),
        'boundaries = ["2020-09-01", "2021-03-01", "2021-06-01"]',
    )
    protocol = protocol.replace('historical_end = "2026-10-01"', 'historical_end = "2021-08-23"')
    protocol = protocol.replace("n_estimators = 120", "n_estimators = 5").replace(
        "samples = 2000", "samples = 100"
    )
    path = config.root / "protocol.toml"
    path.write_text(protocol, encoding="utf-8")
    summary = run_reversion_study(config, path, source)
    directory = Path(summary["directory"])
    assert (
        summary["experiment_records"] == 32
    )  # 3 windows × 5 systems × 2 costs + 2 frozen filters.
    assert summary["promoted"] is False
    assert summary["independent_advantage_confirmed"] is False
    assert sha256(source) == source_hash
    assert sha256(pointer) == pointer_hash
    for relative, expected in read_json(directory / "artifacts.json").items():
        assert sha256(directory / relative) == expected
    for period in summary["periods"].values():
        assert len(period["systems"]) == 5
    assert (directory / "REPORT.md").exists()
