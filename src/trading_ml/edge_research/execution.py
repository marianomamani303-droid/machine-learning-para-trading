from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.errors import DataError


def friction(protocol, multiplier=1):
    fee = protocol["commission_bps"] * multiplier / 10000
    slip = (protocol["slippage_bps"] + protocol["half_spread_bps"]) * multiplier / 10000
    return (1 + fee) * (1 + slip), (1 - fee) * (1 - slip)


def payoff_threshold(validation, protocol):
    r = validation.gross_return.to_numpy()
    up, down = r[r > 0], -r[r <= 0]
    if min(len(up), len(down)) < 20:
        return 1.0
    buy, sell = friction(protocol)
    hurdle = protocol["payoff_cost_margin"] * (buy / sell - 1)
    return float(np.clip((down.mean() + hurdle) / (up.mean() + down.mean()), 0.5, 1))


def execute(
    frame,
    rows,
    probability,
    horizon,
    protocol,
    *,
    policy="fixed_055",
    tau=None,
    multiplier=1,
    delay=0,
):
    """Fractional, self-financing fixed-hold trades with every-open equity marks."""
    p = np.asarray(probability, dtype=float)
    if len(rows) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise DataError("Misaligned/invalid probabilities")
    if policy not in {"fixed_055", "payoff_risk", "trend_risk"} or horizon < 1 or delay < 0:
        raise DataError("Invalid execution policy")
    delta = (frame.reference_time - frame.timestamp).iloc[0]
    buy, sell = friction(protocol, multiplier)
    start = int(rows.index.min()) + 1
    stop = int(rows.index.max()) + horizon + 1
    prices = frame.open.to_numpy(dtype=float)
    if stop >= len(frame) or start >= stop:
        raise DataError("Execution without terminal price")
    allowed = p >= (
        protocol["threshold"] if policy == "fixed_055" else (tau if tau is not None else 0.5)
    )
    if policy in {"payoff_risk", "trend_risk"}:
        allowed &= rows.trend_long.to_numpy() & (rows.relative_volume.to_numpy() >= 0.25)
    annualization = float(pd.Timedelta(days=365) / delta)
    sizes = np.ones(len(rows))
    if policy != "fixed_055":
        sizes = np.clip(
            protocol["target_annual_volatility"]
            / (rows.volatility.to_numpy() * np.sqrt(annualization)),
            0,
            1,
        )
    entries = {}
    rejected_delay = 0
    for j, idx in enumerate(rows.index):
        entry, exit_at = int(idx) + 1 + delay, int(idx) + 1 + delay + horizon
        if (
            exit_at > stop
            or frame.timestamp.iloc[entry] != frame.reference_time.iloc[idx] + delay * delta
            or frame.timestamp.iloc[exit_at] != frame.timestamp.iloc[entry] + horizon * delta
        ):
            rejected_delay += int(allowed[j])
            continue
        if allowed[j]:
            entries[entry] = (exit_at, float(sizes[j]), float(p[j]), int(idx))
    cash, quantity, exit_index, prior = 1.0, 0.0, -1, 1.0
    records, trades = [], []
    active = {}
    for index in range(start, stop + 1):
        price = prices[index]
        if quantity > 0 and index == exit_index:
            cash += quantity * price * sell
            quantity = 0.0
            trades.append(
                {
                    **active,
                    "exit_time": frame.timestamp.iloc[index],
                    "equity_after": cash,
                    "net_return": cash / active["equity_before"] - 1,
                }
            )
        if index in entries and quantity == 0:
            exit_index, size, prob, source = entries[index]
            before = cash
            quantity = cash * size / (price * buy)
            cash *= 1 - size
            active = {
                "signal_time": frame.reference_time.iloc[source],
                "entry_time": frame.timestamp.iloc[index],
                "equity_before": before,
                "size": size,
                "probability": prob,
            }
        equity = cash + quantity * price
        records.append(
            {
                "timestamp": frame.timestamp.iloc[index],
                "equity": equity,
                "net_return": equity / prior - 1,
            }
        )
        prior = equity
    if quantity > 0:
        raise DataError("Unliquidated position at fold boundary")
    curve = pd.DataFrame(records)
    curve["buy_hold_equity"] = prices[start : stop + 1] / prices[start] / buy
    curve.loc[curve.index[-1], "buy_hold_equity"] *= sell
    curve["buy_hold_return"] = curve.buy_hold_equity.pct_change().fillna(
        curve.buy_hold_equity.iloc[0] - 1
    )
    curve["delay_rejected_signals"] = rejected_delay
    return curve, pd.DataFrame(trades)


def daily_returns(curve):
    f = curve.copy()
    # End-of-day dated by execution timestamps; each opening return is included once.
    return f.groupby(f.timestamp.dt.floor("D"))[["net_return", "buy_hold_return"]].agg(
        lambda x: np.prod(1 + x) - 1
    )


def economic_metrics(curve, trades):
    daily = daily_returns(curve).net_return
    equity = np.r_[1.0, curve.equity.to_numpy()]
    std = daily.std(ddof=1)
    return {
        "net_return": float(equity[-1] - 1),
        "sharpe": float(daily.mean() / std * np.sqrt(365)) if std > 0 else 0.0,
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "trades": len(trades),
        "buy_hold_return": float(curve.buy_hold_equity.iloc[-1] - 1),
    }
