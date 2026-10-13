from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.errors import DataError, LeakageError
from trading_ml.momentum.config import ResearchCosts

YEAR_SECONDS = 365.25 * 86400


@dataclass
class Simulation:
    metrics: dict[str, Any]
    equity: pd.DataFrame
    trades: pd.DataFrame


def performance(curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    values = curve.equity.to_numpy()
    seconds = (curve.timestamp.iloc[-1] - curve.timestamp.iloc[0]).total_seconds()
    years = seconds / YEAR_SECONDS
    returns = curve.net_return.to_numpy()
    return {
        "net_return": float(values[-1] / values[0] - 1),
        "cagr": float((values[-1] / values[0]) ** (1 / years) - 1) if years > 0 else None,
        "max_drawdown": float((values / np.maximum.accumulate(values) - 1).min()),
        "turnover": float(curve.turnover.sum()),
        "annual_turnover": float(curve.turnover.sum() / years) if years > 0 else None,
        "trade_count": len(trades),
        "win_rate": float((trades.net_return > 0).mean()) if len(trades) else None,
        "mean_holding_bars": float(trades.holding_bars.mean()) if len(trades) else None,
        "exposure_fraction": float(curve.exposed.iloc[1:].mean()),
        "return_observations": len(returns) - 1,
        "settings": curve.attrs.get("costs"),
    }


def validate_timeline(frame: pd.DataFrame) -> None:
    if (
        len(frame) < 3
        or not frame.timestamp.is_monotonic_increasing
        or frame.timestamp.duplicated().any()
    ):
        raise DataError("Timeline Momentum vacío, desordenado o duplicado")
    if frame.reference_time.isna().any() or not frame.reference_time.is_monotonic_increasing:
        raise DataError("Referencias Momentum inválidas")
    if (frame.reference_time <= frame.timestamp).any() or (
        frame.reference_time.iloc[:-1].to_numpy() > frame.timestamp.iloc[1:].to_numpy()
    ).any():
        raise LeakageError("La vela de señal no está cerrada a la apertura siguiente")
    if not np.isfinite(frame.open).all() or (frame.open <= 0).any():
        raise DataError("Precios de ejecución Momentum inválidos")


def simulate_policy(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    costs: ResearchCosts,
    max_holding_bars: int,
    *,
    probabilities: np.ndarray | None = None,
    threshold: float = 0.55,
    reenter_on_exit: bool = False,
) -> Simulation:
    """Long/cash: signal at previous close, fill at next open, terminal liquidation.

    Windows reset to cash and liquidate at their final available opening. ML gates
    entry only; neither probabilities nor future prices alter holding/exit rules.
    Timestamps may contain session gaps, allowing external Forex/futures adapters.
    """
    costs.validate()
    validate_timeline(frame)
    if not isinstance(max_holding_bars, int) or max_holding_bars < 1 or not 0 <= threshold <= 1:
        raise DataError("Permanencia o umbral inválidos")
    if not signals.index.equals(frame.index) or not set(["entry", "exit"]).issubset(signals):
        raise DataError("Señales Momentum desalineadas")
    if signals[["entry", "exit"]].isna().any().any():
        raise DataError("Señales Momentum faltantes")
    p = np.ones(len(frame)) if probabilities is None else np.asarray(probabilities, dtype=float)
    if p.shape != (len(frame),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise DataError("Probabilidades Momentum inválidas")
    indices = np.flatnonzero(((frame.timestamp >= start) & (frame.timestamp < end)).to_numpy())
    if len(indices) < 2 or indices[0] < 1:
        raise DataError("Ventana de ejecución insuficiente o sin vela causal anterior")
    cash, quantity = costs.initial_capital, 0.0
    initial = cash
    first, last = int(indices[0]), int(indices[-1])
    rows = [
        {
            "timestamp": frame.timestamp.iloc[first],
            "equity": initial,
            "net_return": 0.0,
            "turnover": 0.0,
            "exposed": False,
            "regime": "initial",
        }
    ]
    trades: list[dict[str, Any]] = []
    active: dict[str, Any] = {}
    previous_equity = initial
    for i in indices:
        i = int(i)
        price = float(frame.open.iloc[i])
        exposed = quantity > 0
        if i > first:
            elapsed = (
                frame.timestamp.iloc[i] - frame.timestamp.iloc[i - 1]
            ).total_seconds() / YEAR_SECONDS
            if exposed:
                quantity *= np.exp(-costs.annual_holding_bps / 10000 * elapsed)
            else:
                cash *= (1 + costs.annual_cash_rate) ** elapsed
        pre_trade_equity = cash + quantity * price
        turnover = 0.0
        exiting = quantity > 0 and (
            bool(signals.exit.iloc[i - 1])
            or i - active["entry_index"] >= max_holding_bars
            or i == last
        )
        if exiting:
            turnover += quantity * price / pre_trade_equity
            cash = quantity * price * costs.sell_factor
            trades.append(
                {
                    **active,
                    "exit_time": frame.timestamp.iloc[i],
                    "exit_open": price,
                    "holding_bars": i - active["entry_index"],
                    "net_return": cash / active["capital_before"] - 1,
                    "capital_after": cash,
                    "exit_reason": "window_end"
                    if i == last
                    else ("trend_exit" if signals.exit.iloc[i - 1] else "max_holding"),
                }
            )
            quantity = 0.0
        if i != last and quantity == 0 and (not exiting or reenter_on_exit):
            if bool(signals.entry.iloc[i - 1]) and p[i - 1] >= threshold:
                active = {
                    "signal_reference": frame.reference_time.iloc[i - 1],
                    "entry_index": i,
                    "entry_time": frame.timestamp.iloc[i],
                    "entry_open": price,
                    "capital_before": cash,
                    "probability": float(p[i - 1]),
                }
                quantity = cash / (price * costs.buy_factor)
                turnover += quantity * price / pre_trade_equity
                cash = 0.0
        value = cash + quantity * price
        regime_index = i - 2 if i > first else i - 1
        regime = frame.regime.iloc[regime_index] if "regime" in frame else "unknown"
        rows.append(
            {
                "timestamp": frame.timestamp.iloc[i],
                "equity": value,
                "net_return": value / previous_equity - 1,
                "turnover": turnover,
                "exposed": exposed,
                "regime": regime,
            }
        )
        previous_equity = value
    curve = pd.DataFrame(rows)
    curve.attrs["costs"] = asdict(costs)
    trade_frame = pd.DataFrame(
        trades,
        columns=[
            "signal_reference",
            "entry_index",
            "entry_time",
            "entry_open",
            "capital_before",
            "probability",
            "exit_time",
            "exit_open",
            "holding_bars",
            "net_return",
            "capital_after",
            "exit_reason",
        ],
    )
    return Simulation(performance(curve, trade_frame), curve, trade_frame)


def opportunity_labels(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    max_holding_bars: int,
    costs: ResearchCosts,
) -> pd.DataFrame:
    """Potential trade for EACH causal opportunity, including overlapping ones.

    Labels alone use future data. Natural exits must be observed; right-censored
    entries stay unlabelled, never assigned a fictitious boundary liquidation.
    Actual label exit timestamp is the purge boundary used by model training.
    """
    costs.validate()
    validate_timeline(frame)
    if not isinstance(max_holding_bars, int) or max_holding_bars < 1:
        raise DataError("Permanencia de etiqueta inválida")
    if not signals.index.equals(frame.index) or signals[["entry", "exit"]].isna().any().any():
        raise DataError("Señales de etiqueta faltantes o desalineadas")
    rows = []
    opens = frame.open.to_numpy()
    exits = np.flatnonzero(signals.exit.to_numpy()) + 1
    for source in np.flatnonzero(signals.entry.to_numpy()):
        entry = int(source) + 1
        future_exits = exits[exits > entry]
        exit_index = min(
            entry + max_holding_bars, int(future_exits[0]) if len(future_exits) else len(frame)
        )
        if entry >= len(frame) or exit_index >= len(frame):
            continue
        years = (
            frame.timestamp.iloc[exit_index] - frame.timestamp.iloc[entry]
        ).total_seconds() / YEAR_SECONDS
        net = (
            opens[exit_index]
            / opens[entry]
            * costs.sell_factor
            / costs.buy_factor
            * np.exp(-costs.annual_holding_bps / 10000 * years)
            - 1
        )
        rows.append(
            {
                "source_index": int(source),
                "reference_time": frame.reference_time.iloc[source],
                "target_time": frame.timestamp.iloc[exit_index],
                "target": int(net > 0),
                "trade_net_return": float(net),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["source_index", "reference_time", "target_time", "target", "trade_net_return"],
    )
