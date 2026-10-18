from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.backtesting.simulator import simulate
from trading_ml.config.settings import Backtest
from trading_ml.errors import DataError


def execute(full, rows, probability, costs, *, multiplier=1.0, gate=None, spread=True):
    """Adapt the existing simulator, preserving all signal/calendar rows under a gate."""
    p = np.asarray(probability, dtype=float).copy()
    if gate is not None:
        allowed = np.asarray(gate, dtype=bool)
        if allowed.shape != p.shape:
            raise DataError("Filtro desalineado")
        p[~allowed] = 0.0  # Execution only; never changes stored predictive scores.
    signals = rows[["timestamp", "reference_time"]].copy()
    signals["probability_up"] = p
    settings = Backtest(
        threshold=costs["threshold"],
        commission_bps=costs["commission_bps"] * multiplier,
        slippage_bps=(costs["slippage_bps"] + (costs["spread_bps"] / 2 if spread else 0))
        * multiplier,
        initial_capital=costs["initial_capital"],
    )
    return simulate(
        full, signals, 1, settings, rows.reference_time.min() - pd.Timedelta(nanoseconds=1)
    )


def economic_summary(curve, trades):
    values = curve.equity.to_numpy(dtype=float)
    return {
        "net_return": float(values[-1] / values[0] - 1),
        "max_drawdown": float((values / np.maximum.accumulate(values) - 1).min()),
        "trade_count": len(trades),
    }


def chain_curves(curves):
    capital = 1.0
    output = []
    for fold, curve in curves:
        part = curve.copy()
        part["equity"] = capital * part.equity / part.equity.iloc[0]
        part["fold"] = fold
        capital = part.equity.iloc[-1]
        output.append(part)
    return pd.concat(output, ignore_index=True)
