from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trading_ml.errors import DataError
from trading_ml.label_research.events import ExitRule, observed_exit
from trading_ml.multimarket.markets import fx_open, instrument


@dataclass(frozen=True)
class Costs:
    forex_commission_bps: float = 0.35  # each side, $7 round-trip / $100k
    forex_slippage_pips: float = 0.1
    forex_swap_annual_bps: float = 100.0  # adverse scenario, not a historical broker quote
    forex_additional_spread_pips: float = 0.0  # execution-cost scenario, never a data quote
    crypto_commission_bps: float = 10.0
    crypto_slippage_bps: float = 5.0
    risk_fraction: float = 0.01
    maximum_notional_fraction: float = 1.0
    maximum_entry_spread_pips: float = 5.0
    initial_capital: float = 10000.0

    def validate(self):
        values = list(asdict(self).values())
        if (
            not np.isfinite(values).all()
            or min(values) < 0
            or not 0 < self.risk_fraction <= 0.1
            or not 0 < self.maximum_notional_fraction <= 1
            or self.initial_capital <= 0
        ):
            raise DataError("Costos/riesgo multimercado inválidos")


def conversion(symbol, bid, ask, amount):
    asset = instrument(symbol)
    if asset.quote in {"USD", "USDT"}:
        return float(amount)
    if asset.base == "USD":
        return float(amount / (ask if amount >= 0 else bid))
    raise DataError("Falta conversión USD point-in-time")


def _swap_rollovers(start, end):
    """NY17 rollover Mon-Fri, Wednesday triple; conservative adverse carry only."""
    first = start.tz_convert("America/New_York").normalize().tz_localize(None)
    last = end.tz_convert("America/New_York").normalize().tz_localize(None)
    candidates = (
        (pd.date_range(first, last, freq="D") + pd.Timedelta(hours=17))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    return sum(
        3 if t.tz_convert("America/New_York").weekday() == 2 else 1
        for t in candidates
        if start < t <= end and t.tz_convert("America/New_York").weekday() < 5
    )


class Engine:
    """Shared event kernel for historical outcomes and forward long/cash portfolios."""

    def __init__(
        self,
        frame,
        symbol,
        costs: Costs,
        *,
        execution_frame=None,
        bars_per_horizon=1,
        expiry_policy="timeout_first",
    ):
        costs.validate()
        if expiry_policy not in {"timeout_first", "barrier_first"}:
            raise DataError("Prioridad de vencimiento inválida")
        self.expiry_policy = expiry_policy
        self.frame, self.symbol, self.asset, self.costs = frame, symbol, instrument(symbol), costs
        self.fx = self.asset.market == "forex"
        self.execution = execution_frame if execution_frame is not None else frame
        if not isinstance(bars_per_horizon, int) or bars_per_horizon < 1:
            raise DataError("Escala temporal de ejecución inválida")
        self.bars_per_horizon = bars_per_horizon
        self.time = list(self.execution.timestamp)
        self.reference = list(self.execution.reference_time)
        self.entries = pd.DatetimeIndex(self.execution.timestamp).searchsorted(
            pd.DatetimeIndex(frame.reference_time)
        )
        self.atr = frame.atr_ratio.to_numpy(float)
        self.market_open = (
            fx_open(pd.DatetimeIndex(self.execution.timestamp)).to_numpy()
            if self.fx
            else np.ones(len(self.execution), dtype=bool)
        )
        self.bid = {
            k: self.execution[f"{k}_bid" if self.fx else k].to_numpy(float)
            for k in ["open", "high", "low", "close"]
        }
        self.ask = {
            k: self.execution[f"{k}_ask" if self.fx else k].to_numpy(float)
            for k in ["open", "close"]
        }
        self.ask_high = self.execution.high_ask.to_numpy(float) if self.fx else None
        if any(self.time[i + 1] < self.reference[i] for i in range(len(self.execution) - 1)):
            raise DataError("Velas de ejecución solapadas")
        self.rollover_open = self.rollover_close = np.zeros(len(self.execution), dtype=int)
        if self.fx:
            days = pd.date_range(
                self.time[0].tz_convert("America/New_York").tz_localize(None).normalize(),
                self.reference[-1].tz_convert("America/New_York").tz_localize(None).normalize(),
                freq="D",
            )
            dates = (
                (days + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("UTC")
            )
            weights = np.where(days.dayofweek < 5, np.where(days.dayofweek == 2, 3, 1), 0)
            cumulative = np.r_[0, np.cumsum(weights)]
            self.rollover_open = cumulative[
                np.searchsorted(
                    dates.as_unit("ns").asi8,
                    pd.DatetimeIndex(self.time).as_unit("ns").asi8,
                    side="right",
                )
            ]
            self.rollover_close = cumulative[
                np.searchsorted(
                    dates.as_unit("ns").asi8,
                    pd.DatetimeIndex(self.reference).as_unit("ns").asi8,
                    side="right",
                )
            ]

    def entry(self, index):
        ask = self.ask["open"][index]
        slip = (
            self.asset.pip * self.costs.forex_slippage_pips
            if self.fx
            else ask * self.costs.crypto_slippage_bps / 10000
        )
        markup = self.asset.pip * self.costs.forex_additional_spread_pips if self.fx else 0.0
        return ask + slip + markup

    def _commission(self):
        return (
            self.costs.forex_commission_bps if self.fx else self.costs.crypto_commission_bps
        ) / 10000

    def notional(self, units, price):
        return units if self.asset.base == "USD" else units * price

    def pnl(self, units, entry, exit_bid, exit_ask):
        return conversion(self.symbol, exit_bid, exit_ask, units * (exit_bid - entry))

    def event(self, source, horizon, barrier):
        entry_index = int(self.entries[source])
        steps = horizon * self.bars_per_horizon
        if entry_index + steps >= len(self.execution):
            return None
        if self.fx and self.ask["open"][entry_index] < self.bid["open"][entry_index]:
            return None
        scale = float(self.atr[source])
        if not np.isfinite(scale) or not 0 < scale < 0.3:
            return None
        entry = self.entry(entry_index)
        rule = ExitRule("fx_event", steps, "atr" if barrier else "fixed", 1.5, 1.0)
        upper, lower = entry * (1 + 1.5 * scale), entry * (1 - scale)
        for j in range(entry_index, entry_index + steps + 1):
            result = observed_exit(
                rule,
                age=j - entry_index,
                opening=self.bid["open"][j],
                high=self.bid["high"][j],
                low=self.bid["low"][j],
                previous_close=self.bid["close"][j - 1],
                upper=upper,
                lower=lower,
            )
            if barrier and self.expiry_policy == "barrier_first" and j - entry_index >= steps:
                if self.bid["open"][j] >= upper:
                    result = (float(upper), "tp_gap_at_timeout", "open", False)
                elif self.bid["open"][j] <= lower:
                    result = (float(self.bid["open"][j]), "sl_gap_at_timeout", "open", False)
            if result is not None:
                price, reason, phase, ambiguous = result
                when = self.time[j] if phase == "open" else self.reference[j]
                ask = self.ask["open" if phase == "open" else "close"][j]
                # For quote->USD conversion at an unknown intrabar instant, use
                # the least favorable observed ask high / bid low bound.
                if self.fx and self.asset.base == "USD" and phase != "open":
                    ask = float(self.ask_high[j])
                slip = (
                    self.asset.pip * self.costs.forex_slippage_pips
                    if self.fx
                    else price * self.costs.crypto_slippage_bps / 10000
                )
                exit_price = price - slip
                conversion_bid = self.bid["low"][j] if phase != "open" else self.bid["open"][j]
                pnl = conversion(self.symbol, conversion_bid, ask, exit_price - entry)
                notional = self.notional(1, entry)
                rollovers = int(
                    (self.rollover_open if phase == "open" else self.rollover_close)[j]
                    - self.rollover_open[entry_index]
                )
                carry = notional * self.costs.forex_swap_annual_bps / 10000 / 365 * rollovers
                fees = (
                    notional * self._commission()
                    + self.notional(1, exit_price) * self._commission()
                )
                return {
                    "source_index": source,
                    "entry_index": entry_index,
                    "exit_index": j,
                    "entry_time": self.time[entry_index],
                    "target_time": when,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "exit_ask": ask,
                    "conversion_bid": conversion_bid,
                    "pnl_per_unit": pnl - fees - carry,
                    "unit_net_return": (pnl - fees - carry) / notional,
                    "target": int(pnl - fees - carry > 0),
                    "event_label": 1
                    if reason.startswith("tp")
                    else (-1 if reason.startswith("sl") else 0),
                    "reason": reason,
                    "phase": phase,
                    "ambiguous": ambiguous,
                    "swap_rollovers": rollovers,
                    "fees_per_unit": fees,
                    "swap_per_unit": carry,
                    "risk_width": scale,
                }
        raise DataError("Evento sin salida temporal")

    def portfolio(
        self,
        signals: pd.DataFrame,
        horizon: int,
        barrier: bool,
        threshold: float,
        end: pd.Timestamp,
    ):
        """Entry eligibility uses ONLY current signal, spread, equity and time.

        End-of-fold cutoff is known before the run, never based on realized labels.
        Execution events are advanced bar by bar; future event exit times are never
        used to accept a signal or to size its position.
        """
        if signals.empty:
            return (
                {
                    "trades": 0,
                    "net_return": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": None,
                    "profit_factor": None,
                },
                pd.DataFrame(),
                pd.DataFrame(),
            )
        allowed = {int(r.source_index): float(r.probability) for r in signals.itertuples()}
        first = min(allowed)
        # The signal is known at feature-bar close. Enter the next observed H1
        # quote, even if its H4/D1 bucket later turns out to be incomplete.
        entry_signals = {}
        for source, probability in allowed.items():
            entry_signals[int(self.entries[source])] = (source, probability)
        cash = self.costs.initial_capital
        trades, curve = [], []
        position = None
        # Calendar buffer is fixed ex ante (3 calendar days per daily horizon
        # plus a week) and does not consult whether an outcome hits a barrier.
        duration = self.frame.reference_time.iloc[first] - self.frame.timestamp.iloc[first]
        buffer = max(pd.Timedelta(days=7), (horizon + 2) * duration * 3)
        entry_cutoff = end - buffer
        for j in range(int(self.entries[first]), len(self.execution)):
            if self.time[j] >= end:
                break
            if position is not None and j == position["event"]["exit_index"]:
                event, units, entered_equity = (
                    position["event"],
                    position["units"],
                    position["equity"],
                )
                profit = units * event["pnl_per_unit"]
                cash += profit
                trades.append(
                    {
                        **event,
                        "units": units,
                        "profit_usd": profit,
                        "equity_before": entered_equity,
                        "equity_after": cash,
                        "probability": position["probability"],
                    }
                )
                intrabar_exit = event["phase"] != "open"
                position = None
            else:
                intrabar_exit = False
            source, probability = entry_signals.get(j, (-1, 0.0))
            if (
                position is None
                and not intrabar_exit
                and source >= 0
                and probability >= threshold
                and self.frame.reference_time.iloc[source] < entry_cutoff
                and cash > 0
            ):
                spread_ok = self.market_open[j] and (
                    not self.fx
                    or 0
                    <= (self.ask["open"][j] - self.bid["open"][j]) / self.asset.pip
                    <= self.costs.maximum_entry_spread_pips
                    - self.costs.forex_additional_spread_pips
                )
                if spread_ok:
                    scale = float(self.atr[source])
                    entry = self.entry(j)
                    if np.isfinite(scale) and scale > 0:
                        notional = self.notional(1, entry)
                        risk_per_unit = abs(
                            conversion(
                                self.symbol, self.bid["open"][j], self.ask["open"][j], entry * scale
                            )
                        )
                        units = min(
                            cash * self.costs.risk_fraction / risk_per_unit,
                            cash * self.costs.maximum_notional_fraction / notional,
                        )
                        units = np.floor(units / self.asset.unit_step) * self.asset.unit_step
                        if units >= self.asset.minimum_units:
                            event = self.event(source, horizon, barrier)
                            if event is None:
                                raise DataError(
                                    "Cobertura futura insuficiente: no se omite una entrada por etiqueta censurada"
                                )
                            position = {
                                "event": event,
                                "units": units,
                                "equity": cash,
                                "probability": probability,
                            }
                            if event["exit_index"] == j:
                                profit = units * event["pnl_per_unit"]
                                cash += profit
                                trades.append(
                                    {
                                        **event,
                                        "units": units,
                                        "profit_usd": profit,
                                        "equity_before": position["equity"],
                                        "equity_after": cash,
                                        "probability": allowed[source],
                                    }
                                )
                                position = None
            marked = cash
            if position is not None:
                marked += self.pnl(
                    position["units"],
                    position["event"]["entry_price"],
                    self.bid["close"][j],
                    self.ask["close"][j],
                )
                marked -= (
                    position["units"]
                    * self.notional(1, position["event"]["entry_price"])
                    * self._commission()
                )
                if self.fx:
                    count = (
                        self.rollover_close[j]
                        - self.rollover_open[position["event"]["entry_index"]]
                    )
                    marked -= (
                        position["units"]
                        * self.notional(1, position["event"]["entry_price"])
                        * self.costs.forex_swap_annual_bps
                        / 10000
                        / 365
                        * count
                    )
            curve.append({"timestamp": self.reference[j], "equity": marked})
        if position is not None:
            raise DataError("Posición abierta al final del fold; no se censura silenciosamente")
        outcomes = pd.DataFrame(trades)
        equities = np.r_[self.costs.initial_capital, [r["equity"] for r in curve]]
        profits = outcomes.profit_usd.to_numpy() if len(outcomes) else np.array([])
        gains, losses = profits[profits > 0].sum(), -profits[profits < 0].sum()
        summary = {
            "trades": len(outcomes),
            "net_return": float(cash / self.costs.initial_capital - 1),
            "max_drawdown": float((equities / np.maximum.accumulate(equities) - 1).min()),
            "win_rate": float((profits > 0).mean()) if len(profits) else None,
            "profit_factor": float(gains / losses) if losses > 0 else None,
            "profit_factor_undefined_no_losses": bool(len(profits) and losses == 0),
            "ambiguous_trades": int(outcomes.ambiguous.sum()) if len(outcomes) else 0,
            "entry_cutoff": entry_cutoff.isoformat(),
        }
        return summary, outcomes, pd.DataFrame(curve)
