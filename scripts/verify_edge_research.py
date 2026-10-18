"""Reload models and reconstruct every strategy from original development data."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.edge_research.execution import economic_metrics, friction, payoff_threshold
from trading_ml.edge_research.features import load_development, partitions, supervised
from trading_ml.edge_research.models import predict
from trading_ml.edge_research.study import concatenate_curves, verify
from trading_ml.errors import DataError


def check_probability_roundtrip(predicted, stored):
    predicted, stored = np.asarray(predicted), np.asarray(stored)
    restored = stored.astype(predicted.dtype)
    # pandas writes the shortest float32 decimal, which roundtrips exactly to
    # float32 but differs from the original when parsed as float64.
    native_error = float(np.max(np.abs(predicted - restored)))
    tolerance = 0.0 if predicted.dtype == np.float32 else 1e-12
    if native_error > tolerance:
        raise DataError(f"Native probability not reproduced: {native_error}")
    return float(np.max(np.abs(predicted.astype(float) - stored))), native_error


def audit_ledger(
    frame, rows, probability, trades, horizon, protocol, policy, tau, multiplier, delay
):
    """Independent cash/share reconstruction; does not call the backtest simulator."""
    start, stop = int(rows.index.min()) + 1, int(rows.index.max()) + horizon + 1
    delta = (frame.reference_time - frame.timestamp).iloc[0]
    buy, sell = friction(protocol, multiplier)
    clock = pd.DatetimeIndex(frame.timestamp)
    prices = frame.open.to_numpy(dtype=float)
    probability = np.asarray(probability)
    allowed = probability >= (protocol["threshold"] if policy == "fixed_055" else tau)
    if policy != "fixed_055":
        allowed &= rows.trend_long.to_numpy() & (rows.relative_volume.to_numpy() >= 0.25)
    expected = []
    free_at = -1
    for j in np.flatnonzero(allowed):
        source = int(rows.index[j])
        entry, exit_at = source + 1 + delay, source + 1 + delay + horizon
        if (
            exit_at > stop
            or clock[entry] != frame.reference_time.iloc[source] + delay * delta
            or clock[exit_at] != clock[entry] + horizon * delta
        ):
            continue
        if entry >= free_at:
            expected.append((j, source, entry, exit_at))
            free_at = exit_at
    if len(expected) != len(trades):
        raise DataError("Trade omissions, excess trades or non-overlap violation")
    equity = np.ones(stop - start + 1)
    previous_exit = 0
    capital = 1.0
    for (j, source, entry, exit_at), trade in zip(expected, trades.itertuples(), strict=True):
        size = (
            1.0
            if policy == "fixed_055"
            else min(
                1.0,
                protocol["target_annual_volatility"]
                / (frame.volatility.iloc[source] * np.sqrt(float(pd.Timedelta(days=365) / delta))),
            )
        )
        if (
            pd.Timestamp(trade.entry_time) != clock[entry]
            or pd.Timestamp(trade.exit_time) != clock[exit_at]
            or pd.Timestamp(trade.signal_time) != frame.reference_time.iloc[source]
        ):
            raise DataError("Execution clock/label mismatch")
        if (
            not np.isclose(trade.size, size, rtol=0, atol=1e-12)
            or not np.isclose(trade.probability, probability[j], rtol=0, atol=1e-12)
            or not np.isclose(trade.equity_before, capital, rtol=0, atol=1e-10)
        ):
            raise DataError("Trade risk, score or initial capital mismatch")
        quantity = capital * size / (prices[entry] * buy)
        proceeds = quantity * prices[exit_at] * sell
        # Blockwise cash + shares avoids cancellation of 1 - nearly 1 when
        # thousands of costly trades drive capital close to zero.
        equity[previous_exit : entry - start] = capital
        equity[entry - start : exit_at - start] = (
            capital * (1 - size) + quantity * prices[entry:exit_at]
        )
        after = capital * (1 - size) + proceeds
        if not np.isclose(trade.equity_after, after, rtol=0, atol=1e-10) or not np.isclose(
            trade.net_return, after / capital - 1, rtol=0, atol=1e-12
        ):
            raise DataError("Trade profit/cost identity mismatch")
        capital = after
        previous_exit = exit_at - start
    equity[previous_exit:] = capital
    benchmark = prices[start : stop + 1] / prices[start] / buy
    benchmark[-1] *= sell
    return pd.DataFrame(
        {
            "timestamp": clock[start : stop + 1],
            "equity": equity,
            "net_return": equity / np.r_[1.0, equity[:-1]] - 1,
            "buy_hold_equity": benchmark,
            "buy_hold_return": benchmark / np.r_[1.0, benchmark[:-1]] - 1,
        }
    )


def reconstruct(directory):
    result = verify(directory)
    plan = read_json(directory / "plan.json")
    protocol = plan["protocol"]
    summary = read_json(directory / "summary.json")
    frames, probabilities = {}, {}
    max_error, native_error, reconstructed = 0.0, 0.0, 0
    for path in sorted(directory.glob("*/fold_*/predictions.csv")):
        task, fold = path.parent.parent.name, int(path.parent.name.split("_")[1])
        symbol, interval, horizon = task.split("_")
        horizon = int(horizon[1:])
        if task not in frames:
            source = Path(plan["inputs"][f"{symbol}:{interval}"]["path"])
            raw = load_development(source, protocol["start"], protocol["end_exclusive"])
            frames[task] = supervised(raw, interval, horizon, protocol["end_exclusive"])
        frame = frames[task]
        saved = pd.read_csv(
            path, parse_dates=["timestamp", "reference_time", "entry_time", "target_time"]
        )
        indexed = frame.set_index("timestamp", drop=False)
        rows = indexed.loc[saved.timestamp].copy()
        rows.index = frame.index[frame.timestamp.isin(saved.timestamp)]
        if not rows.eligible.all() or not np.array_equal(rows.target, saved.target):
            raise DataError("Stored target differs from executable label")
        if not np.allclose(rows.gross_return, saved.gross_return, rtol=0, atol=1e-14):
            raise DataError("Stored executable return not reproduced")
        for recipe in protocol["recipes"]:
            state = read_json(path.parent / recipe / "state.json")
            p = predict(path.parent / recipe, state, rows)
            error, local_native_error = check_probability_roundtrip(p, saved[recipe].to_numpy())
            max_error = max(max_error, error)
            native_error = max(native_error, local_native_error)
            probabilities[(task, fold, recipe)] = (rows, p)
            reconstructed += 1
    scenarios = {"gross": (0, 0), "base": (1, 0), "double": (2, 0), "delay": (1, 1)}
    for stat in summary["economic"]:
        task, recipe = stat["task"], stat["recipe"]
        horizon = int(task.split("_h")[1])
        multiplier, delay = scenarios[stat["scenario"]]
        prefix = directory / task / "portfolios" / f"{recipe}_{stat['policy']}_{stat['scenario']}"
        saved_trades = pd.read_csv(prefix.with_suffix(".trades.csv"))
        parts = []
        for fold in range(3):
            rows, p = probabilities[(task, fold, "logistic_price" if recipe == "trend" else recipe)]
            split = read_json(directory / task / f"fold_{fold}" / "split.json")
            boundaries = pd.to_datetime(protocol["boundaries"], utc=True)
            _, validation, _ = partitions(
                frames[task],
                boundaries[fold],
                boundaries[fold + 1],
                protocol["validation_days"],
                protocol["embargo_hours"],
            )
            if not np.isclose(
                payoff_threshold(validation, protocol), split["payoff_tau"], rtol=0, atol=1e-14
            ):
                raise DataError("Validation-only payoff threshold not reproduced")
            trades = saved_trades.loc[saved_trades.fold.eq(fold)]
            curve = audit_ledger(
                frames[task],
                rows,
                np.ones(len(rows)) if recipe == "trend" else p,
                trades,
                horizon,
                protocol,
                stat["policy"],
                0.5 if recipe == "trend" else split["payoff_tau"],
                multiplier,
                delay,
            )
            parts.append(curve)
        curve = concatenate_curves(parts)
        computed = economic_metrics(curve, saved_trades)
        if any(not np.isclose(computed[k], stat[k], rtol=0, atol=1e-10) for k in computed):
            raise DataError("Reconstructed execution differs")
        saved_path = (
            directory / task / "portfolios" / f"{recipe}_{stat['policy']}_{stat['scenario']}.csv"
        )
        saved_curve = pd.read_csv(saved_path)
        np.testing.assert_allclose(
            curve[["equity", "net_return", "buy_hold_equity"]],
            saved_curve[["equity", "net_return", "buy_hold_equity"]],
            rtol=0,
            atol=1e-10,
        )
    result.update(
        {
            "reloaded_models": reconstructed,
            "max_probability_error": max_error,
            "max_native_probability_error": native_error,
            "float32_predictions_reproduced_exactly": True,
            "portfolios_reconstructed_from_prices": len(summary["economic"]),
            "sealed_prices_used": False,
            "new_fits": 0,
            "verification_method": "Independent cash and share flows from raw opens, scores and validation thresholds; no reuse of execution simulator",
            "verifier_sha256": sha256(Path(__file__)),
        }
    )
    output = (
        Path("artifacts/edge_research/verification")
        / f"{directory.name}_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    )
    write_new_json(output, result)
    print({"output": str(output), **result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    reconstruct(parser.parse_args().study)
