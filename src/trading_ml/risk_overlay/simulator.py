from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.errors import DataError
from trading_ml.momentum.config import ResearchCosts
from trading_ml.momentum.simulator import YEAR_SECONDS, Simulation, performance, validate_timeline
from trading_ml.risk_overlay.config import RiskPolicy, entry_fraction


@dataclass
class RiskSimulation:
    simulation: Simulation
    events: pd.DataFrame


def risk_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["risk_volatility"] = frame.close.pct_change(fill_method=None).rolling(20).std()
    true_range = pd.concat(
        [
            frame.high - frame.low,
            (frame.high - frame.close.shift(1)).abs(),
            (frame.low - frame.close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["risk_atr"] = true_range.rolling(14).mean()
    return result


def run_overlay(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    costs: ResearchCosts,
    policy: RiskPolicy,
    horizon: int = 1,
    threshold: float = 0.55,
    *,
    entry_gate: np.ndarray | None = None,
) -> RiskSimulation:
    """Policies change allocation/execution, never model probabilities.

    Stops/DD use ONLY the preceding closed candle; fill at current open with costs.
    Drawdown circuit breaker latches until the window ends. Gross book replays the
    net book's exact decisions/fractions, so fee-driven DD does not change gross trades.
    """
    policy.validate()
    costs.validate()
    validate_timeline(frame)
    p = np.asarray(probabilities, dtype=float)
    gate = np.ones(len(frame), dtype=bool) if entry_gate is None else np.asarray(entry_gate)
    if gate.shape != (len(frame),) or not np.isin(gate, [True, False]).all():
        raise DataError("Filtro de entrada desalineado o inválido")
    if p.shape != (len(frame),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise DataError("Probabilidades del overlay inválidas")
    if (
        not 0.5 <= threshold <= 1
        or policy.exit_threshold > threshold
        or not isinstance(horizon, int)
        or horizon < 1
    ):
        raise DataError("Horizonte o umbral de riesgo inválidos")
    if not np.isfinite(frame.close).all() or (frame.close <= 0).any():
        raise DataError("Cierres de riesgo inválidos")
    indices = np.flatnonzero(((frame.timestamp >= start) & (frame.timestamp < end)).to_numpy())
    if len(indices) < 2 or indices[0] < 1:
        raise DataError("Ventana del overlay insuficiente")
    first, last = int(indices[0]), int(indices[-1])
    initial = costs.initial_capital
    cash, quantity, gross_cash, gross_quantity = initial, 0.0, initial, 0.0
    peak_close = initial
    halted = False
    last_exit = -(10**9)
    active: dict[str, Any] = {}
    events = []
    trades = []
    rows = [
        {
            "timestamp": frame.timestamp.iloc[first],
            "equity": initial,
            "gross_equity": initial,
            "net_return": 0.0,
            "gross_return": 0.0,
            "turnover": 0.0,
            "exposed": False,
            "capital_exposure": 0.0,
            "decision_drawdown": 0.0,
            "halted": False,
            "execution_cost_cash": 0.0,
            "commission_cash": 0.0,
            "slippage_cash": 0.0,
            "regime": "initial",
        }
    ]
    previous_equity, previous_gross = initial, initial
    counts = {
        "eligible_signals": 0,
        "confirmation_rejections": 0,
        "quality_rejections": 0,
        "cooldown_rejections": 0,
        "drawdown_rejections": 0,
        "signals_while_in_position": 0,
        "invalid_sizing_rejections": 0,
        "terminal_signal_exclusions": 0,
        "cap_reductions": 0,
        "drawdown_halts": 0,
    }
    fee = costs.commission_bps / 10000
    slip = (costs.slippage_bps + costs.spread_bps / 2) / 10000
    maximum = policy.max_holding_bars or horizon
    for index in indices:
        i = int(index)
        price = float(frame.open.iloc[i])
        source = i - 1
        # Account held after the previous open marked at its closed candle.
        close_equity = cash + quantity * float(frame.close.iloc[source])
        peak_close = max(peak_close, close_equity)
        drawdown = close_equity / peak_close - 1
        if policy.hard_drawdown is not None and drawdown <= -policy.hard_drawdown and not halted:
            halted = True
            counts["drawdown_halts"] += 1
        if i > first:
            years = (
                frame.timestamp.iloc[i] - frame.timestamp.iloc[i - 1]
            ).total_seconds() / YEAR_SECONDS
            cash *= (1 + costs.annual_cash_rate) ** years
            gross_cash *= (1 + costs.annual_cash_rate) ** years
            quantity *= np.exp(-costs.annual_holding_bps / 10000 * years)
            # Gross includes financing; excludes execution friction only.
            gross_quantity *= np.exp(-costs.annual_holding_bps / 10000 * years)
        pre_equity = cash + quantity * price
        turnover = 0.0
        commissions = 0.0
        slippage = 0.0

        def sell(
            fraction: float,
            reason: str,
            sale_price: float = price,
            sale_equity: float = pre_equity,
            sale_index: int = i,
            source_index: int = source,
            trade: dict[str, Any] = active,
        ) -> float:
            nonlocal cash, quantity, gross_cash, gross_quantity, turnover, commissions, slippage
            units = quantity * fraction
            mid_notional = units * sale_price
            proceeds = mid_notional * costs.sell_factor
            cash += proceeds
            quantity -= units
            gross_proceeds = gross_quantity * fraction * sale_price
            gross_cash += gross_proceeds
            gross_quantity *= 1 - fraction
            turnover += mid_notional / sale_equity
            commissions += mid_notional * (1 - slip) * fee
            slippage += mid_notional * slip
            events.append(
                {
                    "timestamp": frame.timestamp.iloc[sale_index],
                    "source_reference": frame.reference_time.iloc[source_index],
                    "side": "sell",
                    "fraction": fraction,
                    "reason": reason,
                    "mid_notional": mid_notional,
                    "execution_cost_cash": mid_notional - proceeds,
                }
            )
            trade["realized_proceeds"] += proceeds
            trade["gross_realized_proceeds"] += gross_proceeds
            return proceeds

        reason = None
        if quantity > 0:
            if halted:
                reason = "drawdown_halt"
            elif policy.atr_stop and float(frame.close.iloc[source]) <= active["stop_price"]:
                reason = "atr_close_stop"
            elif policy.persistent and p[source] < policy.exit_threshold:
                reason = "probability_exit"
            elif i - active["entry_index"] >= maximum:
                reason = "holding_limit"
            elif i == last:
                reason = "window_end"
            if reason:
                sell(1.0, reason)
                quantity = 0.0
                gross_quantity = 0.0
                pnl = active["realized_proceeds"] - active["entry_capital"]
                gross_pnl = active["gross_realized_proceeds"] - active["gross_entry_capital"]
                trades.append(
                    {
                        **active,
                        "exit_time": frame.timestamp.iloc[i],
                        "exit_open": price,
                        "net_return": pnl / active["entry_capital"],
                        "pnl_cash": pnl,
                        "gross_trade_return": gross_pnl / active["gross_entry_capital"],
                        "gross_pnl_cash": gross_pnl,
                        "holding_bars": i - active["entry_index"],
                        "exit_reason": reason,
                    }
                )
                last_exit = i
        if quantity > 0:
            cap = policy.capital_cap
            if policy.soft_drawdown is not None and drawdown <= -policy.soft_drawdown:
                cap *= policy.reduced_fraction
            # Reduction only: enforce cap at executable open, including sale friction.
            equity_before_trim = cash + quantity * price
            if quantity * price / equity_before_trim > cap + 1e-12:
                drag = 1 - costs.sell_factor
                remaining_notional = (
                    cap * (equity_before_trim - quantity * price * drag) / (1 - cap * drag)
                )
                reduction = 1 - max(0.0, remaining_notional) / (quantity * price)
                sell(reduction, "exposure_cap_reduction")
                counts["cap_reductions"] += 1
        in_window = start <= frame.reference_time.iloc[source] < end
        signal = in_window and p[source] >= threshold
        if signal:
            counts["eligible_signals"] += 1
            lower = source - policy.confirmations + 1
            confirmed = (
                lower >= 0
                and (frame.reference_time.iloc[lower] >= start)
                and bool((p[lower : source + 1] >= threshold).all())
            )
            if i == last:
                counts["terminal_signal_exclusions"] += 1
            elif halted:
                counts["drawdown_rejections"] += 1
            elif not confirmed:
                counts["confirmation_rejections"] += 1
            elif not gate[source]:
                counts["quality_rejections"] += 1
            elif quantity > 0:
                counts["signals_while_in_position"] += 1
            elif i - last_exit <= policy.cooldown_bars and (
                policy.cooldown_bars > 0 or policy.persistent
            ):
                counts["cooldown_rejections"] += 1
            else:
                atr = float(frame.risk_atr.iloc[source])
                fraction = entry_fraction(
                    policy,
                    float(frame.risk_volatility.iloc[source]),
                    atr / float(frame.close.iloc[source]),
                    drawdown,
                )
                if fraction <= 0 or (
                    policy.atr_stop
                    and (not np.isfinite(atr) or price - policy.atr_multiple * atr <= 0)
                ):
                    counts["invalid_sizing_rejections"] += 1
                else:
                    equity_before_buy = cash
                    allocated = cash * fraction
                    quantity = allocated / (price * costs.buy_factor)
                    cash -= allocated
                    gross_allocated = gross_cash * fraction
                    gross_quantity = gross_allocated / price
                    gross_cash -= gross_allocated
                    mid_notional = quantity * price
                    turnover += mid_notional / pre_equity
                    commissions += mid_notional * (1 + slip) * fee
                    slippage += mid_notional * slip
                    active = {
                        "signal_reference": frame.reference_time.iloc[source],
                        "entry_time": frame.timestamp.iloc[i],
                        "entry_index": i,
                        "entry_open": price,
                        "entry_capital": allocated,
                        "allocation": fraction,
                        "probability": float(p[source]),
                        "stop_price": price - policy.atr_multiple * atr
                        if np.isfinite(atr)
                        else None,
                        "realized_proceeds": 0.0,
                        "portfolio_capital": equity_before_buy,
                    }
                    active["gross_entry_capital"] = gross_allocated
                    active["gross_realized_proceeds"] = 0.0
                    events.append(
                        {
                            "timestamp": frame.timestamp.iloc[i],
                            "source_reference": frame.reference_time.iloc[source],
                            "side": "buy",
                            "fraction": fraction,
                            "reason": "entry",
                            "mid_notional": mid_notional,
                            "execution_cost_cash": allocated - mid_notional,
                        }
                    )
        equity = cash + quantity * price
        gross_equity = gross_cash + gross_quantity * price
        regime_index = i - 2 if i > first else source
        rows.append(
            {
                "timestamp": frame.timestamp.iloc[i],
                "equity": equity,
                "gross_equity": gross_equity,
                "net_return": equity / previous_equity - 1,
                "gross_return": gross_equity / previous_gross - 1,
                "turnover": turnover,
                "exposed": quantity > 0,
                "capital_exposure": quantity * price / equity,
                "decision_drawdown": drawdown,
                "halted": halted,
                "execution_cost_cash": commissions + slippage,
                "commission_cash": commissions,
                "slippage_cash": slippage,
                "regime": frame.regime.iloc[regime_index] if "regime" in frame else "unknown",
            }
        )
        previous_equity, previous_gross = equity, gross_equity
    curve = pd.DataFrame(rows)
    trade_frame = pd.DataFrame(
        trades,
        columns=[
            "signal_reference",
            "entry_time",
            "entry_index",
            "entry_open",
            "entry_capital",
            "allocation",
            "probability",
            "stop_price",
            "realized_proceeds",
            "portfolio_capital",
            "gross_entry_capital",
            "gross_realized_proceeds",
            "exit_time",
            "exit_open",
            "net_return",
            "pnl_cash",
            "gross_trade_return",
            "gross_pnl_cash",
            "holding_bars",
            "exit_reason",
        ],
    )
    metrics = {**performance(curve, trade_frame), **risk_statistics(curve, trade_frame), **counts}
    event_frame = pd.DataFrame(
        events,
        columns=[
            "timestamp",
            "source_reference",
            "side",
            "fraction",
            "reason",
            "mid_notional",
            "execution_cost_cash",
        ],
    )
    return RiskSimulation(Simulation(metrics, curve, trade_frame), event_frame)


def risk_statistics(curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    gross = curve.gross_equity.to_numpy()
    durations = curve.timestamp.diff().dt.total_seconds().fillna(0).to_numpy()[1:]
    allocations = curve.capital_exposure.shift(1).fillna(0).to_numpy()[1:]
    time = durations.sum()
    losses = -trades.loc[trades.pnl_cash < 0, "pnl_cash"].sum()
    gains = trades.loc[trades.pnl_cash > 0, "pnl_cash"].sum()
    gross_losses = -trades.loc[trades.gross_pnl_cash < 0, "gross_pnl_cash"].sum()
    gross_gains = trades.loc[trades.gross_pnl_cash > 0, "gross_pnl_cash"].sum()
    return {
        "gross_return": float(gross[-1] / gross[0] - 1),
        "gross_drawdown": float((gross / np.maximum.accumulate(gross) - 1).min()),
        "execution_cost_cash": float(curve.execution_cost_cash.sum()),
        "commission_cash": float(curve.commission_cash.sum()),
        "slippage_cash": float(curve.slippage_cash.sum()),
        "cost_drag_return": float(
            gross[-1] / gross[0] - curve.equity.iloc[-1] / curve.equity.iloc[0]
        ),
        "mean_capital_exposure": float(np.sum(durations * allocations) / time) if time > 0 else 0.0,
        "time_in_market": float(np.sum(durations * (allocations > 0)) / time) if time > 0 else 0.0,
        "max_open_capital_exposure": float(curve.capital_exposure.max()),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "gross_profit_factor": float(gross_gains / gross_losses) if gross_losses > 0 else None,
        "gross_win_rate": float((trades.gross_trade_return > 0).mean()) if len(trades) else None,
        "mean_gross_unit_trade_return": float(trades.gross_trade_return.mean())
        if len(trades)
        else None,
        "expectancy_net_unit": float(trades.net_return.mean()) if len(trades) else None,
        "exit_counts": {str(k): int(v) for k, v in trades.exit_reason.value_counts().items()},
        "hard_drawdown_triggered": bool(curve.halted.any()),
    }
