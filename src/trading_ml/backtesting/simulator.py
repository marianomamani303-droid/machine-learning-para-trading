from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.config.settings import Backtest
from trading_ml.errors import DataError, LeakageError


@dataclass
class BacktestResult:
    summary: dict[str, Any]
    equity: pd.DataFrame
    trades: pd.DataFrame


def _performance(equity: np.ndarray) -> dict[str, float]:
    return {
        "cumulative_return": float(equity[-1] / equity[0] - 1),
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
    }


def simulate(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int,
    costs: Backtest,
    allowed_after: pd.Timestamp,
) -> BacktestResult:
    """Long/cash sin solapamiento: señal al cierre, compra próxima apertura,
    venta h aperturas después. Valuación en cada apertura y costos por lado.
    No asume que el retorno close/close de la etiqueta sea ejecutable.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise DataError("Horizonte inválido")
    if signals.empty or signals.reference_time.min() <= allowed_after:
        raise LeakageError("Las señales deben ser posteriores al período de selección")
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise DataError("Velas desordenadas o duplicadas")
    if signals.timestamp.duplicated().any() or not signals.timestamp.is_monotonic_increasing:
        raise DataError("Señales desordenadas o duplicadas")
    if not np.isfinite(frame.open.to_numpy()).all() or (frame.open <= 0).any():
        raise DataError("Precios de apertura inválidos")
    if (
        not np.isfinite(signals.probability_up).all()
        or ((signals.probability_up < 0) | (signals.probability_up > 1)).any()
    ):
        raise DataError("Probabilidades inválidas")
    if not (
        all(
            np.isfinite(x)
            for x in [
                costs.threshold,
                costs.commission_bps,
                costs.slippage_bps,
                costs.initial_capital,
            ]
        )
        and 0.5 <= costs.threshold <= 1
        and 0 <= costs.commission_bps < 10000
        and 0 <= costs.slippage_bps < 10000
        and costs.initial_capital > 0
    ):
        raise DataError("Configuración de backtesting inválida")
    frame = frame.reset_index(drop=True)
    locations = {timestamp: index for index, timestamp in enumerate(frame.timestamp)}
    eligible: dict[int, Any] = {}
    excluded = 0
    for signal in signals.itertuples():
        if signal.timestamp not in locations:
            raise DataError("Señal sin vela correspondiente")
        source = locations[signal.timestamp]
        if signal.reference_time != frame.reference_time.iloc[source]:
            raise DataError("Referencia de señal incompatible con la vela")
        entry = source + 1
        if entry + horizon >= len(frame):
            excluded += 1
            continue
        if frame.timestamp.iloc[entry] < signal.reference_time:
            raise LeakageError("Entrada anterior al momento en que la señal está disponible")
        eligible[entry] = signal
    if not eligible:
        raise DataError("No hay aperturas suficientes para entrada y salida")
    start = min(eligible)
    stop = max(eligible) + horizon
    fee = costs.commission_bps / 10000
    slip = costs.slippage_bps / 10000
    cash = costs.initial_capital
    quantity = 0.0
    exit_index = -1
    entry_cash = 0.0
    active: dict[str, Any] = {}
    trades = []
    curve = [
        {
            "timestamp": frame.timestamp.iloc[start],
            "phase": "initial",
            "equity": cash,
            "buy_hold_equity": cash,
        }
    ]
    benchmark_quantity = cash / (float(frame.open.iloc[start]) * (1 + slip) * (1 + fee))
    skipped = 0
    for index in range(start, stop + 1):
        price = float(frame.open.iloc[index])
        if quantity > 0 and index == exit_index:
            cash = quantity * price * (1 - slip) * (1 - fee)
            trades.append(
                {
                    **active,
                    "exit_time": frame.timestamp.iloc[index],
                    "exit_fill": price * (1 - slip),
                    "capital_after": cash,
                    "net_return": cash / entry_cash - 1,
                }
            )
            quantity = 0.0
        signal = eligible.get(index)
        if signal is not None and signal.probability_up >= costs.threshold:
            if quantity > 0:
                skipped += 1
            else:
                entry_cash = cash
                quantity = cash / (price * (1 + slip) * (1 + fee))
                cash = 0.0
                exit_index = index + horizon
                active = {
                    "signal_time": signal.reference_time,
                    "entry_time": frame.timestamp.iloc[index],
                    "entry_fill": price * (1 + slip),
                    "probability_up": float(signal.probability_up),
                    "capital_before": entry_cash,
                }
        benchmark = benchmark_quantity * price
        if index == stop:
            benchmark *= (1 - slip) * (1 - fee)
        curve.append(
            {
                "timestamp": frame.timestamp.iloc[index],
                "phase": "open",
                "equity": cash + quantity * price,
                "buy_hold_equity": benchmark,
            }
        )
    if quantity != 0:
        raise DataError("Simulación terminó con una posición sin liquidar")
    equity = pd.DataFrame(curve)
    trade_frame = pd.DataFrame(
        trades,
        columns=[
            "signal_time",
            "entry_time",
            "entry_fill",
            "probability_up",
            "capital_before",
            "exit_time",
            "exit_fill",
            "capital_after",
            "net_return",
        ],
    )
    summary = {
        "strategy": "long_cash_next_open_fixed_horizon",
        "horizon_bars": horizon,
        "settings": asdict(costs),
        "first_execution": frame.timestamp.iloc[start].isoformat(),
        "last_execution": frame.timestamp.iloc[stop].isoformat(),
        "strategy_metrics": _performance(equity.equity.to_numpy()),
        "buy_hold_metrics": _performance(equity.buy_hold_equity.to_numpy()),
        "final_capital": float(equity.equity.iloc[-1]),
        "trade_count": len(trades),
        "win_rate": float((trade_frame.net_return > 0).mean()) if trades else None,
        "signals_without_exit_price": excluded,
        "signals_skipped_while_in_position": skipped,
        "warnings": [
            "Simulación a aperturas: no incluye liquidez, latencia ni drawdown intravela.",
            "No se toman posiciones short; la probabilidad bajista conserva efectivo.",
            "Los costos se aplican a compra y venta del benchmark también.",
        ],
    }
    return BacktestResult(summary, equity, trade_frame)
