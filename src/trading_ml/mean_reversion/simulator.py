from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.errors import DataError
from trading_ml.mean_reversion.config import ResearchCosts, ReversionConfig
from trading_ml.mean_reversion.rules import exit_reason
from trading_ml.momentum.simulator import YEAR_SECONDS, Simulation, performance, validate_timeline


def _validate(
    frame: pd.DataFrame, opportunities: pd.Series, config: ReversionConfig, costs: ResearchCosts
) -> None:
    config.validate()
    costs.validate()
    validate_timeline(frame)
    if (
        not frame.index.equals(opportunities.index)
        or opportunities.isna().any()
        or not opportunities.isin([True, False]).all()
    ):
        raise DataError("Oportunidades de reversión faltantes o desalineadas")
    if not np.isfinite(frame.close).all() or (frame.close <= 0).any():
        raise DataError("Cierres de reversión inválidos")


def entry_plan(
    frame: pd.DataFrame, source: int, config: ReversionConfig, costs: ResearchCosts
) -> dict[str, Any] | None:
    """Known next-open admission and risk sizing, shared by simulator and labels."""
    entry = source + 1
    if entry >= len(frame):
        return None
    price = float(frame.open.iloc[entry])
    target = float(frame.mr_mean.iloc[source])
    atr = float(frame.mr_atr.iloc[source])
    if not np.isfinite([target, atr]).all() or atr <= 0:
        return None
    stop = price - config.stop_atr * atr
    # Avoid buying a gap already beyond the target; potential target gain must pay costs.
    if stop <= 0 or target * costs.sell_factor <= price * costs.buy_factor:
        return None
    stop_loss_fraction = 1 - stop * costs.sell_factor / (price * costs.buy_factor)
    fraction = min(config.max_allocation, config.risk_fraction / stop_loss_fraction)
    return {
        "source_index": source,
        "signal_reference": frame.reference_time.iloc[source],
        "entry_index": entry,
        "entry_time": frame.timestamp.iloc[entry],
        "entry_open": price,
        "target_price": target,
        "stop_price": stop,
        "allocation": float(fraction),
        "planned_loss_fraction": float(fraction * stop_loss_fraction),
        "entry_regime": frame.regime.iloc[source] if "regime" in frame else "unknown",
    }


def opportunity_labels(
    frame: pd.DataFrame, opportunities: pd.Series, config: ReversionConfig, costs: ResearchCosts
) -> pd.DataFrame:
    """Reversion = anchored target reached before close-stop/timeout; censored != failure.

    Only labels inspect future closes/opens. A target hit can still lose after costs
    or a gap at the exit open; reversion probability and profitability are distinct.
    """
    _validate(frame, opportunities, config, costs)
    rows = []
    for source in np.flatnonzero(opportunities.to_numpy()):
        plan = entry_plan(frame, int(source), config, costs)
        row = {
            "source_index": int(source),
            "reference_time": frame.reference_time.iloc[source],
            "target_time": pd.NaT,
            "target": np.nan,
            "trade_net_return": np.nan,
            "portfolio_contribution": np.nan,
            "exit_reason": None,
            "label_status": "right_censored",
        }
        if plan is None:
            rows.append({**row, "label_status": "entry_not_executable"})
            continue
        entry = plan["entry_index"]
        for i in range(entry + 1, min(entry + config.max_holding_bars + 1, len(frame))):
            reason = exit_reason(
                float(frame.close.iloc[i - 1]),
                plan["target_price"],
                plan["stop_price"],
                i - entry,
                config.max_holding_bars,
            )
            if reason:
                years = (
                    frame.timestamp.iloc[i] - frame.timestamp.iloc[entry]
                ).total_seconds() / YEAR_SECONDS
                net = (
                    float(frame.open.iloc[i])
                    / plan["entry_open"]
                    * costs.sell_factor
                    / costs.buy_factor
                    * np.exp(-costs.annual_holding_bps / 10000 * years)
                    - 1
                )
                row.update(
                    {
                        "target_time": frame.timestamp.iloc[i],
                        "target": int(reason == "reversion"),
                        "trade_net_return": float(net),
                        "portfolio_contribution": plan["allocation"] * float(net),
                        "exit_reason": reason,
                        "label_status": "observed",
                    }
                )
                break
        rows.append(row)
    result = pd.DataFrame(
        rows,
        columns=[
            "source_index",
            "reference_time",
            "target_time",
            "target",
            "trade_net_return",
            "portfolio_contribution",
            "exit_reason",
            "label_status",
        ],
    )
    result["target_time"] = pd.to_datetime(result.target_time, utc=True)
    result["reference_time"] = pd.to_datetime(result.reference_time, utc=True)
    return result


def simulate_reversion(
    frame: pd.DataFrame,
    opportunities: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: ReversionConfig,
    costs: ResearchCosts,
    probabilities: np.ndarray | None = None,
    threshold: float = 0.55,
) -> Simulation:
    _validate(frame, opportunities, config, costs)
    p = np.ones(len(frame)) if probabilities is None else np.asarray(probabilities, dtype=float)
    if (
        p.shape != (len(frame),)
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
        or not 0 <= threshold <= 1
    ):
        raise DataError("Probabilidades o umbral Mean Reversion inválidos")
    indices = np.flatnonzero(((frame.timestamp >= start) & (frame.timestamp < end)).to_numpy())
    if len(indices) < 2 or indices[0] < 1:
        raise DataError("Ventana Mean Reversion insuficiente o sin señal causal anterior")
    first, last = int(indices[0]), int(indices[-1])
    cash, quantity = costs.initial_capital, 0.0
    initial = cash
    previous_equity = initial
    active: dict[str, Any] = {}
    rows = [
        {
            "timestamp": frame.timestamp.iloc[first],
            "equity": initial,
            "net_return": 0.0,
            "turnover": 0.0,
            "exposed": False,
            "allocation": 0.0,
            "regime": "initial",
        }
    ]
    trades = []
    counts = {
        "opportunities": 0,
        "rejected_by_probability": 0,
        "skipped_while_in_position": 0,
        "nonexecutable_at_open": 0,
        "skipped_on_exit_open": 0,
        "terminal_opportunities": 0,
    }
    for index in indices:
        i = int(index)
        price = float(frame.open.iloc[i])
        exposed = quantity > 0
        if i > first:
            years = (
                frame.timestamp.iloc[i] - frame.timestamp.iloc[i - 1]
            ).total_seconds() / YEAR_SECONDS
            quantity *= np.exp(-costs.annual_holding_bps / 10000 * years)
            cash *= (1 + costs.annual_cash_rate) ** years
        pre_equity = cash + quantity * price
        turnover = 0.0
        reason = None
        if quantity > 0:
            reason = exit_reason(
                float(frame.close.iloc[i - 1]),
                active["target_price"],
                active["stop_price"],
                i - active["entry_index"],
                config.max_holding_bars,
            )
            if i == last and reason is None:
                reason = "window_end"
            if reason:
                proceeds = quantity * price * costs.sell_factor
                turnover += quantity * price / pre_equity
                cash += proceeds
                unit_return = proceeds / active["position_capital"] - 1
                trades.append(
                    {
                        **active,
                        "exit_time": frame.timestamp.iloc[i],
                        "exit_open": price,
                        "holding_bars": i - active["entry_index"],
                        "net_return": unit_return,
                        "portfolio_contribution": active["allocation"] * unit_return,
                        "exit_reason": reason,
                        "risk_budget_exceeded": active["allocation"] * unit_return
                        < -config.risk_fraction,
                    }
                )
                quantity = 0.0
        if opportunities.iloc[i - 1]:
            counts["opportunities"] += 1
            if i == last:
                counts["terminal_opportunities"] += 1
            elif p[i - 1] < threshold:
                counts["rejected_by_probability"] += 1
            elif exposed:
                counts["skipped_on_exit_open" if reason else "skipped_while_in_position"] += 1
            else:
                plan = entry_plan(frame, i - 1, config, costs)
                if plan is None:
                    counts["nonexecutable_at_open"] += 1
                else:
                    allocated = cash * plan["allocation"]
                    quantity = allocated / (price * costs.buy_factor)
                    turnover += quantity * price / pre_equity
                    cash -= allocated
                    active = {**plan, "position_capital": allocated, "probability": float(p[i - 1])}
        value = cash + quantity * price
        regime_index = i - 2 if i > first else i - 1
        rows.append(
            {
                "timestamp": frame.timestamp.iloc[i],
                "equity": value,
                "net_return": value / previous_equity - 1,
                "turnover": turnover,
                "exposed": exposed,
                "allocation": quantity * price / value,
                "regime": frame.regime.iloc[regime_index] if "regime" in frame else "unknown",
            }
        )
        previous_equity = value
    curve = pd.DataFrame(rows)
    curve.attrs["costs"] = asdict(costs)
    trade_frame = pd.DataFrame(
        trades,
        columns=[
            "source_index",
            "signal_reference",
            "entry_index",
            "entry_time",
            "entry_open",
            "target_price",
            "stop_price",
            "allocation",
            "planned_loss_fraction",
            "entry_regime",
            "position_capital",
            "probability",
            "exit_time",
            "exit_open",
            "holding_bars",
            "net_return",
            "portfolio_contribution",
            "exit_reason",
            "risk_budget_exceeded",
        ],
    )
    metrics = {**performance(curve, trade_frame), **trade_statistics(trade_frame), **counts}
    return Simulation(metrics, curve, trade_frame)


def trade_statistics(trades: pd.DataFrame) -> dict[str, Any]:
    n = len(trades)
    if not n:
        return {
            "expectancy_net_per_trade": None,
            "expectancy_portfolio_contribution": None,
            "profit_factor": None,
            "reversion_exit_fraction": None,
            "risk_budget_exceeded_count": 0,
            "exit_counts": {},
            "entry_regimes": {},
        }
    positive = trades.loc[trades.net_return > 0, "net_return"].sum()
    negative = -trades.loc[trades.net_return < 0, "net_return"].sum()
    return {
        "expectancy_net_per_trade": float(trades.net_return.mean()),
        "expectancy_portfolio_contribution": float(trades.portfolio_contribution.mean()),
        "profit_factor": float(positive / negative) if negative > 0 else None,
        "reversion_exit_fraction": float((trades.exit_reason == "reversion").mean()),
        "risk_budget_exceeded_count": int(trades.risk_budget_exceeded.sum()),
        "exit_counts": {str(k): int(v) for k, v in trades.exit_reason.value_counts().items()},
        "entry_regimes": {
            str(r): {
                "trades": len(g),
                "expectancy_net_per_trade": float(g.net_return.mean()),
                "reversion_exit_fraction": float((g.exit_reason == "reversion").mean()),
            }
            for r, g in trades.groupby("entry_regime")
        },
    }
