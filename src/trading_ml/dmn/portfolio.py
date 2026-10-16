from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.errors import DataError


def target_weights(
    signal: np.ndarray, volatility: np.ndarray, risk: dict, periods: int = 365
) -> np.ndarray:
    signal, volatility = np.asarray(signal, dtype=float), np.asarray(volatility, dtype=float)
    if (
        signal.shape != volatility.shape
        or signal.ndim != 2
        or signal.shape[1] < 1
        or not np.isfinite(signal).all()
        or not np.isfinite(volatility).all()
        or (volatility <= 0).any()
        or (np.abs(signal) > 1 + 1e-6).any()
    ):
        raise DataError("Señal/volatilidad de cartera DMN inválida")
    signal = signal.clip(0 if risk["position_mode"] == "long_only" else -1, 1)
    scale = risk["target_volatility"] / (
        np.maximum(volatility, risk["daily_volatility_floor"]) * np.sqrt(periods)
    )
    return np.clip(signal * scale, -risk["maximum_gross"], risk["maximum_gross"]) / signal.shape[1]


def simulate(
    weights: np.ndarray,
    returns: np.ndarray,
    times: pd.DatetimeIndex,
    costs: dict,
    periods: int = 365,
    *,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """Self financing target weights after fees; actual holdings drift between opens.

    Each call starts flat and ends flat. Independent folds therefore include both
    boundary trades. Cost rate is charged per unit of absolute traded notional.
    """
    weights, returns = np.asarray(weights, dtype=float), np.asarray(returns, dtype=float)
    times = pd.DatetimeIndex(times)
    if (
        weights.shape != returns.shape
        or weights.ndim != 2
        or len(weights) != len(times)
        or not len(weights)
        or not np.isfinite(weights).all()
        or not np.isfinite(returns).all()
        or (returns <= -1).any()
        or (np.abs(weights).sum(axis=1) > 1 + 1e-8).any()
        or times.has_duplicates
        or not times.is_monotonic_increasing
        or not np.isfinite(multiplier)
        or multiplier < 0
    ):
        raise DataError("Entradas del backtest DMN inválidas")
    if not np.isfinite(list(costs.values())).all() or min(costs.values()) < 0 or periods <= 0:
        raise DataError("Costos/calendario DMN inválidos")
    rate = (
        multiplier
        * sum(costs[k] for k in ["commission_bps", "slippage_bps", "half_spread_bps"])
        / 10000
    )
    if not 0 <= rate < 0.1:
        raise DataError("Costos de ejecución excesivos")
    if len(times) > 1 and not ((times[1:] - times[:-1]) == pd.Timedelta(days=1)).all():
        raise DataError("Backtest diario con huecos; no se salta riesgo de calendario")
    wealth, holdings = 1.0, np.zeros(weights.shape[1])
    records = []
    for n, (w, ret) in enumerate(zip(weights, returns, strict=True)):
        before = wealth
        h = holdings / before
        # q + c * |w*q - h| = 1. Contraction for gross<=1 and c<0.1.
        q = 1.0
        for _ in range(20):
            q = 1 - rate * np.abs(w * q - h).sum()
        traded = np.abs(w * q - h).sum()
        funding = (
            multiplier
            * (
                np.maximum(-w, 0).sum() * costs["short_borrow_annual_bps"]
                + max(w.sum() - 1, 0) * costs["cash_borrow_annual_bps"]
            )
            / (10000 * periods)
        )
        after_fees = before * q
        wealth = after_fees * (1 + np.dot(w, ret) - funding)
        holdings = after_fees * w * (1 + ret)
        if wealth <= 0:
            raise DataError("Cartera insolvente; simulación detenida")
        exit_cost = 0.0
        if n == len(weights) - 1:
            liquidation_notional = np.abs(holdings).sum()
            exit_cost = rate * liquidation_notional / before
            traded += liquidation_notional / before
            wealth -= rate * liquidation_notional
            if wealth <= 0:
                raise DataError("Cartera insolvente al liquidar")
        records.append(
            {
                "reference_time": times[n],
                "exit_time": times[n] + pd.Timedelta(days=1),
                "gross_return": float(np.dot(w, ret)),
                "net_return": float(wealth / before - 1),
                "equity": wealth,
                "turnover": float(traded),
                "transaction_cost": float(1 - q + exit_cost),
                "funding_cost": float(q * funding),
                "gross_exposure": float(np.abs(w).sum()),
                "net_exposure": float(w.sum()),
            }
        )
    return pd.DataFrame(records)


def metrics(frame: pd.DataFrame, periods: int = 365) -> dict:
    r = frame.net_return.to_numpy(dtype=float)
    equity = np.r_[1.0, np.cumprod(1 + r)]
    vol = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)))
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1))
    annual_mean = float(r.mean() * periods)
    return {
        "dates": len(r),
        "total_return": float(equity[-1] - 1),
        "cagr": float(equity[-1] ** (periods / len(r)) - 1),
        "annual_mean_return": annual_mean,
        "annual_volatility": vol * np.sqrt(periods),
        "sharpe": float(r.mean() / vol * np.sqrt(periods)) if vol > 1e-12 else None,
        "sortino": float(r.mean() / downside * np.sqrt(periods)) if downside > 1e-12 else None,
        "max_drawdown": drawdown,
        "calmar": annual_mean / abs(drawdown) if drawdown < -1e-12 else None,
        "positive_date_fraction": float((r > 0).mean()),
        "mean_daily_turnover": float(frame.turnover.mean()),
        "total_turnover": float(frame.turnover.sum()),
        "summed_cost_fractions": float((frame.transaction_cost + frame.funding_cost).sum()),
        "mean_gross_exposure": float(frame.gross_exposure.mean()),
        "annualization_note": "365 eligible daily observations; boundary omissions disclosed separately",
    }
