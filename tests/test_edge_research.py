from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.backtesting.simulator import simulate
from trading_ml.common import sha256, write_new_json
from trading_ml.config.settings import Backtest
from trading_ml.dashboard.edge_candidates import research_candidates
from trading_ml.edge_research.execution import execute, payoff_threshold
from trading_ml.edge_research.features import (
    EXTRA,
    PRICE,
    causal_features,
    load_development,
    partitions,
    supervised,
)
from trading_ml.edge_research.models import fit_or_load, predict
from trading_ml.edge_research.study import paired_test
from trading_ml.errors import DataError


def candles(n=3500):
    rng = np.random.default_rng(123)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"),
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price * 1.001,
            "volume": rng.uniform(20, 200, n),
        }
    )


@pytest.fixture
def protocol():
    return {
        "commission_bps": 10,
        "slippage_bps": 5,
        "half_spread_bps": 2,
        "threshold": 0.55,
        "target_annual_volatility": 0.20,
        "payoff_cost_margin": 1.5,
        "seed": 42,
        "logistic_c": 0.1,
        "xgb_estimators": 5,
    }


def test_future_prices_cannot_change_features_or_fitted_prediction(tmp_path, protocol):
    raw = candles()
    a = supervised(raw, "1h", 4, "2023-01-01")
    changed = raw.copy()
    changed.loc[2200:, ["open", "high", "low", "close"]] *= 3
    b = supervised(changed, "1h", 4, "2023-01-01")
    pd.testing.assert_frame_equal(
        a.loc[:2199, list(PRICE + EXTRA)], b.loc[:2199, list(PRICE + EXTRA)]
    )
    fit, validation = (
        a.loc[a.eligible & (a.index < 1600)],
        a.loc[a.eligible & (a.index >= 1700) & (a.index < 2000)],
    )
    for recipe in ["logistic_price", "logistic_full", "logistic_noise", "xgb_full"]:
        state, count = fit_or_load(tmp_path / recipe, fit, validation, recipe, protocol)
        before = predict(tmp_path / recipe, state, a.loc[[2100]])
        after = predict(tmp_path / recipe, state, b.loc[[2100]])
        np.testing.assert_array_equal(before, after)
        assert count == (2 if recipe == "xgb_full" else 1)
        recovered, count = fit_or_load(tmp_path / recipe, fit, validation, recipe, protocol)
        assert recovered == state and count == 0


def test_gap_excludes_spanning_targets_and_resets_feature_history():
    raw = candles().drop(index=1800).reset_index(drop=True)
    f = supervised(raw, "1h", 4, "2023-01-01")
    assert not f.loc[1795:1799, "eligible"].any()
    assert not f.loc[1800:2000, "eligible"].any()
    assert f.eligible.iloc[-10]
    assert f.loc[f.eligible, "target_time"].lt(pd.Timestamp("2023-01-01", tz="UTC")).all()


def test_full_size_ledger_matches_existing_simulator_and_delays_are_inside_fold(protocol):
    f = supervised(candles(), "1h", 4, "2023-01-01")
    rows = f.iloc[900:1000]
    p = np.linspace(0.4, 0.8, len(rows))
    curve, trades = execute(f, rows, p, 4, protocol)
    signals = rows[["timestamp", "reference_time"]].assign(probability_up=p)
    old = simulate(
        f,
        signals,
        4,
        Backtest(threshold=0.55, commission_bps=10, slippage_bps=7, initial_capital=1),
        rows.reference_time.min() - pd.Timedelta(seconds=1),
    )
    np.testing.assert_allclose(curve.equity, old.equity.equity.iloc[1:], rtol=0, atol=1e-12)
    assert len(trades) == len(old.trades)
    delayed, delayed_trades = execute(f, rows, p, 4, protocol, delay=1)
    assert (
        pd.to_datetime(delayed_trades.entry_time) - pd.to_datetime(delayed_trades.signal_time)
        == pd.Timedelta(hours=1)
    ).all()
    assert pd.to_datetime(delayed_trades.exit_time).max() <= delayed.timestamp.max()
    with pytest.raises(DataError):
        execute(f, rows, np.full(len(rows), np.nan), 4, protocol)


def test_purge_uses_execution_exit_and_validation_payoff_only(protocol):
    f = supervised(candles(20000), "1h", 4, "2025-01-01")
    fit, validation, test = partitions(
        f, pd.Timestamp("2023-07-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")
    )
    assert fit.target_time.max() < validation.reference_time.min() - pd.Timedelta(hours=48)
    assert validation.target_time.max() < test.reference_time.min() - pd.Timedelta(hours=48)
    before = payoff_threshold(validation, protocol)
    f.loc[test.index, "gross_return"] = 999
    assert payoff_threshold(validation, protocol) == before
    assert before >= 0.5


def test_constant_daily_features_do_not_invent_intraday_phase():
    raw = candles(2000).iloc[::24].copy().reset_index(drop=True)
    f = causal_features(raw, "1d")
    assert np.isfinite(f[list(PRICE + EXTRA)].iloc[-1]).all()
    assert f.hour_sin.eq(0).all()


def test_dashboard_preserves_rejections_and_checks_research_evidence(tmp_path):
    directory = tmp_path / "edge_research" / "runs" / "fixed"
    write_new_json(
        directory / "shadow_manifest.json",
        {
            "candidates": [
                {
                    "task": "BTCUSDT_1h_h4",
                    "recipe": "logistic_full",
                    "eligible_for_shadow": False,
                    "predictive_pass": False,
                    "economic_pass": False,
                }
            ]
        },
    )
    write_new_json(
        directory / "plan.json",
        {"protocol": {"commission_bps": 10, "slippage_bps": 5, "half_spread_bps": 2}},
    )
    write_new_json(directory / "summary.json", {})
    write_new_json(
        directory / "artifacts.json", {p.name: sha256(p) for p in directory.glob("*.json")}
    )
    rows = research_candidates(tmp_path)
    assert rows[0]["Rentabilidad"] == "Sin confirmar"
    assert rows[0]["Estado"] == "No supera criterios de seguimiento"
    (directory / "summary.json").write_text("{}\n", encoding="utf-8")
    # Change actual bytes, not merely the JSON object order.
    (directory / "summary.json").write_text('{"altered": true}', encoding="utf-8")
    with pytest.raises(DataError):
        research_candidates(tmp_path)


def test_quarantined_values_are_never_parsed_or_used(tmp_path):
    source = tmp_path / "bounded.csv"
    source.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2025-06-30 22:00:00+00:00,100,101,99,100,20\n"
        "2025-06-30 23:00:00+00:00,100,101,99,100,20\n"
        "2025-07-01 00:00:00+00:00,FORBIDDEN,FORBIDDEN,FORBIDDEN,FORBIDDEN,FORBIDDEN\n",
        encoding="utf-8",
    )
    write_new_json(source.with_suffix(".csv.json"), {"sha256": sha256(source)})
    frame = load_development(source, "2025-06-30", "2025-07-01")
    assert len(frame) == 2
    assert frame.close.eq(100).all()


def test_paired_bootstrap_distinguishes_zero_and_constant_positive_gain():
    frame = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=120, tz="UTC"),
            "fold": np.repeat([0, 1], 60),
            "gain": 0.0,
        }
    )
    protocol = {"bootstrap_days": [7, 30], "bootstrap_samples": 100, "seed": 42}
    assert paired_test(frame, "gain", protocol)["p"] == 1
    frame["gain"] = 1.0
    result = paired_test(frame, "gain", protocol)
    assert result["p"] < 0.05
    assert all(interval["lower"] == 1 for interval in result["intervals"])


def test_independent_cash_share_audit_catches_missing_trades(protocol):
    spec = spec_from_file_location(
        "edge_verifier", Path(__file__).parents[1] / "scripts" / "verify_edge_research.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    f = supervised(candles(), "1h", 4, "2023-01-01")
    rows, p = f.iloc[900:1000], np.full(100, 0.9)
    for policy in ["fixed_055", "payoff_risk", "trend_risk"]:
        for multiplier, delay in [(0, 0), (1, 0), (2, 0), (1, 1)]:
            curve, trades = execute(
                f, rows, p, 4, protocol, policy=policy, tau=0.55, multiplier=multiplier, delay=delay
            )
            independent = module.audit_ledger(
                f, rows, p, trades, 4, protocol, policy, 0.55, multiplier, delay
            )
            np.testing.assert_allclose(independent.equity, curve.equity, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                independent.buy_hold_equity, curve.buy_hold_equity, rtol=0, atol=1e-12
            )
    curve, trades = execute(f, rows, p, 4, protocol)
    with pytest.raises(DataError):
        module.audit_ledger(f, rows, p, trades.iloc[1:], 4, protocol, "fixed_055", 0.55, 1, 0)


def test_independent_audit_preserves_relative_precision_near_zero_capital(protocol):
    spec = spec_from_file_location(
        "edge_verifier_precision", Path(__file__).parents[1] / "scripts" / "verify_edge_research.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = candles(6500)
    raw[["open", "close"]] = 100.0
    raw["high"], raw["low"] = 101.0, 99.0
    f = supervised(raw, "1h", 1, "2025-01-01")
    rows, p = f.iloc[800:5800], np.ones(5000)
    curve, trades = execute(f, rows, p, 1, protocol, multiplier=2)
    independent = module.audit_ledger(f, rows, p, trades, 1, protocol, "fixed_055", 0.55, 2, 0)
    assert curve.equity.iloc[-1] < 1e-12
    np.testing.assert_allclose(independent.equity, curve.equity, rtol=1e-12, atol=0)
    from io import StringIO

    native = np.array([0.5638946294784546, 0.371234], dtype=np.float32)
    stored = pd.read_csv(StringIO(pd.DataFrame({"p": native}).to_csv(index=False))).p.to_numpy()
    csv_error, native_error = module.check_probability_roundtrip(native, stored)
    assert csv_error > 1e-12 and native_error == 0
    altered = np.nextafter(native, np.ones_like(native)).astype(float)
    with pytest.raises(DataError):
        module.check_probability_roundtrip(native, altered)
