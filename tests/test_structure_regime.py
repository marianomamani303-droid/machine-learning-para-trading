from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import candles

from trading_ml.advanced_research.features import build_research_features
from trading_ml.errors import DataError
from trading_ml.feature_research.study import FitBudget, temporal_score, temporal_train
from trading_ml.structure_regime.execution import economic_summary, execute
from trading_ml.structure_regime.features import (
    CLUSTER_COLUMNS,
    CLUSTER_INPUT,
    EXTENSIONS,
    RULE_COLUMNS,
    apply_clusters,
    extend_structure,
    fit_clusters,
)
from trading_ml.structure_regime.report import simultaneous_economics
from trading_ml.structure_regime.study import fit_experts, snapshot, validate_protocol


@pytest.fixture
def protocol():
    return tomllib.loads(Path("config/structure_regime.toml").read_text(encoding="utf-8"))


def enriched(rows=650):
    frame = candles(rows)
    frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
    frame["quote_volume"] = frame.close * frame.volume
    frame["taker_quote"] = frame.quote_volume * 0.5
    frame["taker_base"] = frame.volume * 0.5
    frame["trades"] = 100
    return frame


def build(raw, protocol):
    return extend_structure(build_research_features(raw), protocol["structure"])


def test_prefix_and_future_ohlcv_mutation_do_not_change_features(protocol):
    raw = enriched()
    full = build(raw, protocol)
    stop = 413
    prefix = build(raw.iloc[:stop], protocol)
    changed = raw.copy()
    changed.loc[stop:, ["open", "high", "low", "close"]] *= 3
    changed = build(changed, protocol)
    columns = EXTENSIONS + RULE_COLUMNS + ["sr_state", "sr_ready"]
    pd.testing.assert_frame_equal(full.loc[: stop - 1, columns], prefix[columns])
    pd.testing.assert_frame_equal(full.loc[: stop - 1, columns], changed.loc[: stop - 1, columns])
    assert (full.sr_vol.iloc[:140] == "unknown").all()


def test_retest_uses_confirmed_level_and_not_birth_candle(protocol):
    raw = enriched(50)
    raw[["open", "close"]] = 100.0
    raw["high"], raw["low"] = 101.0, 99.0
    frame = build_research_features(raw)
    # Input events stand for the already-tested detector; extension must consume them.
    for name in ["smc_pivot_high_confirmed", "smc_bos_up", "smc_choch_up"]:
        frame[name] = 0.0
    frame.loc[18, "high"] = 105
    frame.loc[20, "smc_pivot_high_confirmed"] = 1
    frame.loc[22, ["open", "close", "high", "low"]] = [104, 106, 107, 104]
    frame.loc[22, "smc_bos_up"] = 1
    frame.loc[23:24, ["open", "close", "high", "low"]] = [104.8, 107, 107, 104.8]
    result = extend_structure(frame, protocol["structure"])
    assert result.sr_break_retest_up.iloc[22] == 0
    assert result.sr_break_retest_up.iloc[23] == 1
    assert result.sr_retest_impulse_up.iloc[23] == 1
    assert result.sr_break_retest_up.iloc[24] == 0
    assert result.sr_resistance_rejection.iloc[:20].sum() == 0


def test_clusters_fit_only_train_and_are_causal_at_inference(protocol):
    frame = build(enriched(), protocol).dropna(subset=CLUSTER_INPUT)
    train, scoring = frame.iloc[:300], frame.iloc[300:]
    scaler, cluster = fit_clusters(train, 42)
    np.testing.assert_allclose(scaler.mean_, train[CLUSTER_INPUT].mean())
    first = apply_clusters(scoring, scaler, cluster)
    changed = scoring.copy()
    changed.loc[changed.index[20] :, CLUSTER_INPUT] *= 100
    second = apply_clusters(changed, scaler, cluster)
    pd.testing.assert_frame_equal(
        first[CLUSTER_COLUMNS].iloc[:20], second[CLUSTER_COLUMNS].iloc[:20]
    )


def test_experts_fallback_and_never_use_evaluation_labels(tmp_path, protocol):
    raw = enriched()
    f = build(raw, protocol)
    f["target"] = (f.close.shift(-1) > f.close).astype(float)
    f["target_time"] = f.reference_time.shift(-1)
    f = f.loc[f.sr_ready]
    train, evaluation = f.iloc[:300].copy(), f.iloc[300:].copy()
    train["state"] = "main"
    evaluation["state"] = "unseen"
    config = protocol["research"]
    baseline = np.full(len(evaluation), 0.51)
    parameters = {"n_estimators": 3, "max_depth": 1}
    first, audit = fit_experts(
        train,
        evaluation,
        "state",
        ["return_1"],
        baseline,
        FitBudget(10, parameters),
        42,
        config,
        tmp_path,
    )
    np.testing.assert_array_equal(first, baseline)
    assert next(a for a in audit if a["state"] == "unseen")["fallback_global"]
    evaluation["state"] = "main"
    location = tmp_path / "second"
    location.mkdir()
    p, _ = fit_experts(
        train,
        evaluation,
        "state",
        ["return_1"],
        baseline,
        FitBudget(10, parameters),
        42,
        config,
        location,
    )
    evaluation["target"] = 1 - evaluation.target
    evaluation["target_time"] = pd.Timestamp("2099-01-01", tz="UTC")
    location = tmp_path / "third"
    location.mkdir()
    q, _ = fit_experts(
        train,
        evaluation,
        "state",
        ["return_1"],
        baseline,
        FitBudget(10, parameters),
        42,
        config,
        location,
    )
    np.testing.assert_array_equal(p, q)


def test_execution_calendar_gate_spread_and_no_target_access(protocol):
    frame = enriched(60)
    frame[["open", "high", "low", "close"]] = 100.0
    rows = frame.iloc[20:40].copy()
    probability = np.full(len(rows), 0.6)
    base = execute(frame, rows, probability, protocol["costs"])
    gated = execute(frame, rows, probability, protocol["costs"], gate=np.zeros(len(rows), bool))
    legacy = execute(frame, rows, probability, protocol["costs"], spread=False)
    double = execute(frame, rows, probability, protocol["costs"], multiplier=2)
    pd.testing.assert_series_equal(base.equity.timestamp, gated.equity.timestamp)
    assert gated.summary["trade_count"] == 0
    assert len(base.trades) == len(rows)
    assert (base.trades.entry_time >= base.trades.signal_time).all()
    assert (
        double.equity.equity.iloc[-1] < base.equity.equity.iloc[-1] < legacy.equity.equity.iloc[-1]
    )
    fee, slip = 10 / 10000, 6 / 10000
    np.testing.assert_allclose(
        base.trades.net_return, (1 - slip) * (1 - fee) / ((1 + slip) * (1 + fee)) - 1
    )
    # Initial capital must be included when the very first trade loses money.
    assert economic_summary(base.equity, base.trades)["max_drawdown"] < 0


def test_bootstrap_identical_policies_never_imply_advantage(protocol):
    config = protocol["bootstrap"] | {"samples": 100}
    values = np.tile([0.01, -0.015], 100)
    result = simultaneous_economics(
        {"a": values, "b": values}, np.repeat([0, 1], 100), [("a", "b")], config
    )
    for pairs in result["blocks"].values():
        for ci in pairs["a__minus__b"].values():
            assert ci["estimate"] == ci["lower"] == ci["upper"] == 0


def test_embargo_and_outer_target_purge(protocol):
    frame = enriched()
    frame["target"] = np.arange(len(frame)) % 2
    frame["target_time"] = frame.reference_time + pd.Timedelta(days=1)
    start, end = frame.reference_time.iloc[400], frame.reference_time.iloc[500]
    train = temporal_train(frame, start, 6, 240)
    evaluation = temporal_score(frame, start, end)
    assert train.target_time.max() < start - pd.Timedelta(days=6)
    assert evaluation.target_time.max() < end
    assert evaluation.reference_time.min() >= start


def test_protocol_and_preservation_snapshot(tmp_path, protocol):
    validate_protocol(protocol)
    bad = protocol | {"research": protocol["research"] | {"embargo_days": 0}}
    with pytest.raises(DataError):
        validate_protocol(bad)
    old = tmp_path / "old.json"
    old.write_text("old")
    before = snapshot(tmp_path)
    new = tmp_path / "new"
    new.mkdir()
    (new / "result.json").write_text("new")
    assert snapshot(tmp_path, new) == before
    old.write_text("changed")
    assert snapshot(tmp_path, new) != before
