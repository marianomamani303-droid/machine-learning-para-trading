from dataclasses import replace

import pandas as pd
import pytest

from trading_ml.backtesting import simulate
from trading_ml.config.settings import Backtest
from trading_ml.errors import LeakageError


def market(prices):
    timestamps = pd.date_range("2020-01-01", periods=len(prices), freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "reference_time": timestamps + pd.Timedelta(days=1),
            "open": prices,
        }
    )


def signal(frame, indices, probability=0.8):
    result = frame.loc[indices, ["timestamp", "reference_time"]].copy()
    result["probability_up"] = probability
    return result


def test_next_open_execution_and_cost_math():
    frame = market([100, 200, 220, 300])
    costs = Backtest(commission_bps=10, slippage_bps=5)
    result = simulate(frame, signal(frame, [0]), 1, costs, pd.Timestamp("2019-12-01", tz="UTC"))
    expected = 220 / 200 * (1 - 0.0005) / (1 + 0.0005) * (1 - 0.001) / (1 + 0.001) - 1
    assert result.summary["strategy_metrics"]["cumulative_return"] == pytest.approx(expected)
    assert result.trades.entry_fill.iloc[0] == pytest.approx(200 * 1.0005)
    assert result.trades.exit_time.iloc[0] == frame.timestamp.iloc[2]
    assert result.summary["buy_hold_metrics"]["cumulative_return"] == pytest.approx(expected)


def test_no_trade_preserves_cash_and_win_rate_undefined():
    frame = market([100, 100, 90, 80])
    result = simulate(
        frame, signal(frame, [0, 1], 0.1), 1, Backtest(), pd.Timestamp("2019-01-01", tz="UTC")
    )
    assert result.summary["trade_count"] == 0
    assert result.summary["win_rate"] is None
    assert result.summary["strategy_metrics"]["cumulative_return"] == 0
    assert result.summary["buy_hold_metrics"]["cumulative_return"] < 0


def test_mark_to_market_drawdown_and_no_overlap():
    frame = market([100, 100, 50, 110, 100, 120])
    result = simulate(
        frame,
        signal(frame, [0, 1, 2, 3]),
        2,
        replace(Backtest(), commission_bps=0, slippage_bps=0),
        pd.Timestamp("2019-01-01", tz="UTC"),
    )
    assert result.summary["trade_count"] == 2
    assert result.summary["signals_skipped_while_in_position"] == 1
    assert result.summary["strategy_metrics"]["max_drawdown"] == pytest.approx(-0.5)
    assert result.trades.entry_time.iloc[1] >= result.trades.exit_time.iloc[0]


def test_flat_price_loses_costs():
    frame = market([100, 100, 100])
    result = simulate(
        frame, signal(frame, [0]), 1, Backtest(), pd.Timestamp("2019-01-01", tz="UTC")
    )
    assert result.summary["win_rate"] == 0
    assert result.summary["strategy_metrics"]["cumulative_return"] < 0


def test_missing_exit_excluded_and_leakage_rejected():
    frame = market([100, 100, 110, 120])
    result = simulate(
        frame, signal(frame, [0, 2]), 1, Backtest(), pd.Timestamp("2019-01-01", tz="UTC")
    )
    assert result.summary["signals_without_exit_price"] == 1
    with pytest.raises(LeakageError):
        simulate(frame, signal(frame, [0]), 1, Backtest(), frame.reference_time.iloc[0])
