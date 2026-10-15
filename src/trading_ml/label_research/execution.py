from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ml.errors import DataError
from trading_ml.label_research.events import ExitRule, observed_exit, widths
from trading_ml.momentum.config import ResearchCosts
from trading_ml.momentum.simulator import validate_timeline


@dataclass
class EventSimulation:
    curve: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    metrics: dict


def summarize(curve, trades):
    equity = curve.equity.to_numpy(dtype=float)
    pnl = trades.pnl_cash.to_numpy() if len(trades) else np.array([])
    positive, negative = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "net_return": float(equity[-1] / equity[0] - 1),
        "gross_return": float(curve.gross_equity.iloc[-1] / curve.gross_equity.iloc[0] - 1),
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "trade_count": len(trades),
        "win_rate": float((trades.net_return > 0).mean()) if len(trades) else None,
        "mean_net_trade": float(trades.net_return.mean()) if len(trades) else None,
        "profit_factor": float(positive / negative) if negative > 0 else None,
        "no_losing_trades": bool(len(trades) and negative == 0),
        "mean_holding_days": float(trades.holding_days.mean()) if len(trades) else None,
        "ambiguous_trades": int(trades.ambiguous.sum()) if len(trades) else 0,
        "end_bar_exposure": float(curve.capital_exposure.iloc[1:].mean()),
        "fraction_bars_exposed": float(curve.exposed.iloc[1:].mean()),
        "execution_cost_cash": float(curve.execution_cost_cash.sum()),
        "commission_cash": float(curve.commission_cash.sum()),
        "slippage_spread_cash": float(curve.slippage_spread_cash.sum()),
        "holding_cost_cash": float(trades.holding_cost_cash.sum()) if len(trades) else 0.0,
        "turnover": float(curve.turnover.sum()),
    }


def simulate_events(
    frame,
    probability,
    rule: ExitRule,
    start,
    end,
    costs: ResearchCosts,
    *,
    threshold=0.55,
    gate=None,
    maximum_horizon=10,
    tie="stop_first",
):
    """Chronological long/cash, 100% fixed allocation, no label-dependent masks.

    All recipes stop offering new entries at the same known calendar cutoff.
    They may reenter after an open exit, never after an intrabar exit that day.
    Labels, event end times and precomputed exit indices are never read here.
    """
    rule.validate()
    costs.validate()
    validate_timeline(frame)
    if not 0 <= threshold <= 1 or maximum_horizon < rule.horizon:
        raise DataError("Umbral/horizonte de ejecución inválido")
    p = np.asarray(probability, dtype=float)
    if p.shape != (len(frame),) or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise DataError("Probabilidades desalineadas o inválidas")
    g = np.ones(len(frame), dtype=bool) if gate is None else np.asarray(gate)
    if g.shape != p.shape or not np.isin(g, [True, False]).all():
        raise DataError("Filtro de entrada inválido")
    cutoff = end - pd.Timedelta(days=maximum_horizon + 1)
    indices = np.flatnonzero((frame.timestamp >= start) & (frame.timestamp < end))
    if len(indices) < maximum_horizon + 2 or indices[0] < 1:
        raise DataError("Ventana sin cobertura para ejecución")
    opening, high, low, close, sigma, atr = (
        frame[c].to_numpy(dtype=float)
        for c in ["open", "high", "low", "close", "label_sigma", "label_atr_fraction"]
    )
    timestamps, references = list(frame.timestamp), list(frame.reference_time)
    cash = gross_cash = costs.initial_capital
    active = None
    previous = previous_gross = costs.initial_capital
    fee, slip = costs.commission_bps / 10000, (costs.slippage_bps + costs.spread_bps / 2) / 10000
    curve = [
        {
            "timestamp": start,
            "equity": cash,
            "gross_equity": gross_cash,
            "net_return": 0.0,
            "gross_return": 0.0,
            "exposed": False,
            "capital_exposure": 0.0,
            "execution_cost_cash": 0.0,
            "commission_cash": 0.0,
            "slippage_spread_cash": 0.0,
            "turnover": 0.0,
        }
    ]
    trades, decisions = [], []
    for i in indices:
        i, source = int(i), int(i) - 1
        day_fee = day_slip = turnover = 0.0
        exposed = active is not None

        def sell(result, exit_index=i):
            nonlocal active, cash, gross_cash, day_fee, day_slip, turnover
            price, reason, phase, ambiguous = result
            time = timestamps[exit_index] if phase == "open" else references[exit_index]
            years = (time - active["entry_time"]).total_seconds() / (365.25 * 86400)
            carry = np.exp(-costs.annual_holding_bps / 10000 * years)
            notional = active["quantity"] * price * carry
            day_fee += notional * (1 - slip) * fee
            day_slip += notional * slip
            turnover += 1.0
            cash = notional * costs.sell_factor
            gross_cash = active["gross_quantity"] * price * carry
            trades.append(
                {k: v for k, v in active.items() if k not in {"quantity", "gross_quantity"}}
                | {
                    "exit_time": time,
                    "exit_index": exit_index,
                    "exit_price": price,
                    "reason": reason,
                    "phase": phase,
                    "ambiguous": ambiguous,
                    "holding_days": years * 365.25,
                    "holding_cost_cash": active["quantity"] * price * (1 - carry),
                    "net_return": cash / active["capital_before"] - 1,
                    "pnl_cash": cash - active["capital_before"],
                    "capital_after": cash,
                }
            )
            active = None

        # At the open, only opening/previous close are observed. In particular,
        # today's eventual high/low must not decide whether we are free to enter.
        if active is not None:
            result = observed_exit(
                rule,
                age=i - active["entry_index"],
                opening=opening[i],
                high=-np.inf,
                low=np.inf,
                previous_close=close[source],
                upper=active["upper"],
                lower=active["lower"],
                tie=tie,
            )
            if result is not None:
                sell(result)
        up, down = widths(rule, sigma[source], atr[source])
        in_time = start <= references[source] < cutoff
        reason = "accepted"
        if not in_time:
            reason = "calendar_cutoff"
        elif not np.isfinite(p[source]):
            reason = "probability_unavailable"
        elif p[source] < threshold:
            reason = "below_threshold"
        elif not bool(g[source]):
            reason = "filter_rejected"
        elif active is not None:
            reason = "position_open"
        elif not np.isfinite([up, down]).all():
            reason = "scale_unavailable"
        if reason == "accepted":
            price = opening[i]
            quantity = cash / (price * costs.buy_factor)
            notional = quantity * price
            day_fee += notional * (1 + slip) * fee
            day_slip += notional * slip
            turnover += 1.0
            active = {
                "signal_reference": references[source],
                "source_index": source,
                "entry_time": timestamps[i],
                "entry_index": i,
                "entry_price": price,
                "upper": price * (1 + up),
                "lower": price * (1 - down),
                "up_fraction": up,
                "down_fraction": down,
                "capital_before": cash,
                "probability": float(p[source]),
                "quantity": quantity,
                "gross_quantity": gross_cash / price,
            }
            cash = gross_cash = 0.0
            exposed = True
        decisions.append(
            {
                "reference_time": references[source],
                "probability": float(p[source]) if np.isfinite(p[source]) else None,
                "gate": bool(g[source]),
                "calendar_eligible": in_time,
                "reason": reason,
                "up_fraction": up,
                "down_fraction": down,
            }
        )
        if active is not None:
            result = observed_exit(
                rule,
                age=i - active["entry_index"],
                opening=opening[i],
                high=high[i],
                low=low[i],
                previous_close=close[source],
                upper=active["upper"],
                lower=active["lower"],
                tie=tie,
            )
            if result is not None:
                sell(result)
        if active is not None:
            elapsed = (references[i] - active["entry_time"]).total_seconds() / (365.25 * 86400)
            carry = np.exp(-costs.annual_holding_bps / 10000 * elapsed)
            value = active["quantity"] * close[i] * carry
            gross_value = active["gross_quantity"] * close[i] * carry
        else:
            if not exposed:
                cash *= (1 + costs.annual_cash_rate) ** (1 / 365.25)
                gross_cash *= (1 + costs.annual_cash_rate) ** (1 / 365.25)
            value, gross_value = cash, gross_cash
        curve.append(
            {
                "timestamp": references[i],
                "equity": value,
                "gross_equity": gross_value,
                "net_return": value / previous - 1,
                "gross_return": gross_value / previous_gross - 1,
                "exposed": exposed,
                "capital_exposure": float(active is not None),
                "execution_cost_cash": day_fee + day_slip,
                "commission_cash": day_fee,
                "slippage_spread_cash": day_slip,
                "turnover": turnover,
            }
        )
        previous, previous_gross = value, gross_value
    if active is not None:
        raise DataError("Posición sin liquidar: el corte calendario no cubre el horizonte")
    curve = pd.DataFrame(curve)
    trades = pd.DataFrame(
        trades,
        columns=[
            "signal_reference",
            "source_index",
            "entry_time",
            "entry_index",
            "entry_price",
            "upper",
            "lower",
            "up_fraction",
            "down_fraction",
            "capital_before",
            "probability",
            "exit_time",
            "exit_index",
            "exit_price",
            "reason",
            "phase",
            "ambiguous",
            "holding_days",
            "holding_cost_cash",
            "net_return",
            "pnl_cash",
            "capital_after",
        ],
    )
    return EventSimulation(curve, trades, pd.DataFrame(decisions), summarize(curve, trades))
