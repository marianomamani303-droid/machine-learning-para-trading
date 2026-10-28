from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from trading_ml.errors import DataError, LeakageError
from trading_ml.rl_signal.data import COLUMNS, build_panel, load_panel, training_predictions
from trading_ml.rl_signal.dqn import Normalizer, fit_dqn
from trading_ml.rl_signal.environment import Account, TradingEnvironment, rollout
from trading_ml.rl_signal.evaluation import holm, paired_inference, reconstruct


@pytest.fixture
def config():
    return tomllib.loads(Path("config/rl_signal.toml").read_text(encoding="utf-8"))


def episode(n=40, probability=0.6, sigma=0.01, growth=0.0):
    times = pd.date_range("2023-01-01", periods=n, tz="UTC")
    prices = 20000 * (1 + growth) ** np.arange(n + 1)
    return pd.DataFrame(
        {
            "reference_time": times,
            "entry_time": times,
            "exit_time": times + pd.Timedelta(days=1),
            "entry_open": prices[:-1],
            "exit_open": prices[1:],
            "close": prices[:-1] * 0.999,
            "probability": probability,
            "sigma": sigma,
        }
    )


def source(n=1500):
    rng = np.random.default_rng(199)
    close = 20000 * np.exp(np.cumsum(rng.normal(0.0003, 0.025, n)))
    opening = np.r_[close[0], close[:-1]]
    times = pd.date_range("2021-01-01", periods=n, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": times,
            "available_at": times + pd.Timedelta(days=1),
            "open": opening,
            "close": close,
            "high": np.maximum(opening, close) * 1.02,
            "low": np.minimum(opening, close) * 0.98,
            "volume": rng.uniform(1000, 10000, n),
            "quote_volume": rng.uniform(1e8, 1e9, n),
        }
    )


def test_financial_features_are_prefix_invariant():
    raw = source()
    full = build_panel(raw)
    prefix = build_panel(raw.iloc[:900])
    np.testing.assert_allclose(
        full[COLUMNS].iloc[:900], prefix[COLUMNS], equal_nan=True, rtol=0, atol=0
    )
    changed = raw.copy()
    changed.loc[900:, ["open", "close", "high", "low", "volume", "quote_volume"]] *= 3
    altered = build_panel(changed)
    np.testing.assert_allclose(
        full[COLUMNS].iloc[:900], altered[COLUMNS].iloc[:900], equal_nan=True, rtol=0, atol=0
    )


def test_inner_predictions_use_prior_labels_and_identical_frozen_recipe(config):
    panel = build_panel(source())
    end = pd.Timestamp("2022-08-01", tz="UTC")
    frame, fits = training_predictions(panel, end, config)
    assert (frame.predictor_last_label < frame.predictor_fit_at - pd.Timedelta(days=6)).all()
    assert (frame.predictor_fit_at <= frame.reference_time).all()
    assert frame.latest_exit_time.max() < end
    assert len(fits) == 3
    assert all(model[-1].C == 1 and contract["columns"] == COLUMNS for model, contract in fits)
    # Mutating all not-yet-observed outcomes cannot change predictions in this prefix.
    changed = source()
    cut = changed.timestamp >= end
    changed.loc[cut, ["open", "close", "high", "low"]] *= 4
    other, _ = training_predictions(build_panel(changed), end, config)
    np.testing.assert_array_equal(frame.probability, other.probability)


def test_quarantine_cannot_be_extended(config):
    config["data"]["stop"] = "2026-01-01"
    with pytest.raises(LeakageError):
        load_panel(config)


def test_repeating_target_preserves_units_and_charges_only_real_delta(config):
    curve, orders = rollout(episode(), config, "fixed")
    filled = orders.loc[orders.signed_quantity != 0]
    assert len(filled) == 2
    assert filled.signed_quantity.iloc[0] > 0 > filled.signed_quantity.iloc[1]
    np.testing.assert_allclose(curve.units.iloc[:-1], curve.units.iloc[0], atol=0, rtol=0)
    assert curve.cash.min() > 0
    assert curve.exposure.max() <= 0.5
    result = reconstruct(curve, orders, config, "base")
    assert result["max_equity_error"] < 1e-8
    assert result["gross_return_same_units"] == pytest.approx(0, abs=1e-12)
    assert curve.log_return.sum() < 0


def test_partial_rebalance_never_closes_and_reopens(config):
    env = TradingEnvironment(episode(), config)
    env.step(1)
    first_units = env.account.units
    env.step(2)
    assert env.account.units > first_units
    assert len(env.orders) == 2
    assert all(o["signed_quantity"] > 0 for o in env.orders)
    last = env.orders[-1]
    assert last["units_before"] == first_units
    assert last["cost"] == pytest.approx(
        last["commission"]
        + abs(last["signed_quantity"]) * abs(last["fill_price"] - last["reference_price"])
    )


def test_hysteresis_uses_predeclared_entry_and_exit(config):
    frame = episode(8)
    frame["probability"] = [0.56, 0.54, 0.51, 0.49, 0.54, 0.56, 0.52, 0.49]
    fixed, _ = rollout(frame, config, "fixed")
    band, _ = rollout(frame, config, "band")
    assert fixed.action.tolist() == [2, 0, 0, 0, 0, 2, 0, 0]
    assert band.action.tolist() == [2, 2, 2, 0, 0, 2, 2, 0]
    assert band.filled_orders.sum() == fixed.filled_orders.sum()


def test_weight_drift_is_marked_and_trimmed_to_causal_limit(config):
    env = TradingEnvironment(episode(growth=0.10), config)
    env.step(2)
    units = env.account.units
    assert env.state()[3] > 0.5
    env.step(2)
    assert env.account.units < units
    assert env.ledger[-1]["exposure"] <= 0.50003


def test_action_mask_and_bad_actions_rejected(config):
    env = TradingEnvironment(episode(sigma=0.08), config)
    assert env.mask().tolist() == [True, False, False]
    for action in (1, 2, -1, 3, 0.5):
        with pytest.raises(DataError):
            env.step(action)


def test_cost_stress_and_terminal_liquidation(config):
    frame = episode()
    base, b_orders = rollout(frame, config, "fixed")
    double, d_orders = rollout(frame, config, "fixed", "double")
    assert double.equity.iloc[-1] < base.equity.iloc[-1]
    assert d_orders.cost.sum() > b_orders.cost.sum()
    assert base.units.iloc[-1] < config["portfolio"]["quantity_step"]
    assert b_orders.terminal.iloc[-1]
    np.testing.assert_allclose(
        base.reward, base.log_return - 0.5 * (base.exposure * base.sigma) ** 2
    )
    assert base.log_return.iloc[-1] < 0
    reconstruct(double, d_orders, config, "double")


def test_minimums_rounding_and_unliquidatable_dust(config):
    p = config["portfolio"]
    account = Account(9999, 0.000105)
    order = account.rebalance(0.0, 20000, p, terminal=True)
    assert order["status"] == "minimum_order"
    assert account.units == 0.000105  # No fictitious dust liquidation.
    account = Account(10000)
    order = account.rebalance(0.25, 27123.123, p)
    assert order["signed_quantity"] / p["quantity_step"] == pytest.approx(
        round(order["signed_quantity"] / p["quantity_step"])
    )
    assert account.units * 27123.123 / account.equity(27123.123) <= 0.25


def test_delayed_action_queue_keeps_original_order(config):
    env = TradingEnvironment(episode(), config, "delay")
    env.step(2)
    assert env.account.units == 0
    env.step(0)
    assert env.account.units > 0  # Yesterday's buy executes, not today's zero target.
    order = env.orders[-1]
    assert order["execution_time"] == order["signal_time"] + pd.Timedelta(days=1)
    env.step(0)
    assert env.account.units < config["portfolio"]["quantity_step"]
    curve, orders = rollout(episode(), config, "fixed", "delay")
    assert curve.exposure.iloc[0] == 0
    reconstruct(curve, orders, config, "delay")


def test_execution_uses_next_open_and_independent_reconstruction_detects_corruption(config):
    frame = episode(growth=0.005)
    frame["close"] *= 0.5  # Causal close can differ; fill must remain the next OPEN.
    curve, orders = rollout(frame, config, "fixed")
    assert orders.reference_price.iloc[0] == frame.entry_open.iloc[0]
    reconstruct(curve, orders, config, "base")
    corrupted = orders.copy()
    corrupted.loc[0, "commission"] *= 2
    with pytest.raises(AssertionError):
        reconstruct(curve, corrupted, config, "base")


def test_normalizer_and_inference_never_update_from_validation_or_test(config, tmp_path):
    small = copy.deepcopy(config)
    small["training"].update(maximum_steps=300, checkpoint_every=100, warmup_steps=64)
    train = episode(80, growth=0.001)
    validation = episode(30, probability=0.54, growth=-0.001)
    model, norm, summary = fit_dqn(train, validation, small, 42, tmp_path / "first")
    expected = Normalizer.fit(train, small)
    np.testing.assert_array_equal(norm.mean, expected.mean)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mean = norm.mean.copy()
    rollout(episode(30, probability=0.99, sigma=0.04), small, "dqn", normalizer=norm, network=model)
    np.testing.assert_array_equal(norm.mean, mean)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    candidates = summary["checkpoints"]
    assert (
        summary["selected_step"]
        == max(candidates, key=lambda r: (r["validation_reward"], -r["step"]))["step"]
    )
    with np.load(tmp_path / "first/training_replay.npz") as replay:
        assert replay["row_indices"].max() < len(train)
        assert replay["states"].shape == (300, 5)
    # Exact same seed and fixed budget reproduces selected weights.
    repeated, _, _ = fit_dqn(train, validation, small, 42, tmp_path / "repeat")
    assert all(torch.equal(before[k], v) for k, v in repeated.state_dict().items())


def test_paired_bootstrap_preserves_null_and_all_24_comparisons(config):
    config["inference"]["samples"] = 100
    curves = {}
    f = pd.DataFrame(
        {"log_return": np.linspace(-0.01, 0.01, 120), "fold": np.repeat([0, 1, 2], 40)}
    )
    for scenario in config["inference"]["scenarios"]:
        for reference in config["inference"]["comparisons"]:
            curves[(scenario, reference, 0)] = f.copy()
        for seed in config["training"]["seeds"]:
            curves[(scenario, "dqn", seed)] = f.copy()
    rows, seed_rows = paired_inference(curves, config)
    assert len(rows) == 24 and len(seed_rows) == 18
    for row in rows:
        assert row["log_gain"] == pytest.approx(0, abs=1e-15)
        assert row["simultaneous_low"] == pytest.approx(0, abs=1e-15)
        assert row["p_holm_24"] == 1
    np.testing.assert_allclose(holm([0.01, 0.02, 0.5]), [0.03, 0.04, 0.5])
