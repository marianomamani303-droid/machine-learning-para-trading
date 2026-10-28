from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

import numpy as np
import pandas as pd

from trading_ml.errors import DataError


def rounded_down(value: float, step: float) -> float:
    return float(
        (Decimal(str(max(0.0, value))) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
        * Decimal(str(step))
    )


@dataclass
class Account:
    cash: float
    units: float = 0.0

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def rebalance(self, weight: float, price: float, p: dict, multiplier=1.0, terminal=False):
        if not 0 <= weight <= 0.5 or not np.isfinite(price) or price <= 0:
            raise DataError("Invalid target/price")
        before_cash, before_units = self.cash, self.units
        equity = self.equity(price)
        current = self.units * price / equity
        fee = p["commission_bps"] * multiplier / 10000
        adverse = (p["half_spread_bps"] + p["slippage_bps"]) * multiplier / 10000
        direction = 1 if weight > current else -1
        fill = price * (1 + direction * adverse)
        # Solve q*price = w * equity_after_costs, retaining actual prior units.
        friction = (1 + adverse) * (1 + fee) - 1 if direction > 0 else 1 - (1 - adverse) * (1 - fee)
        target_notional = (
            weight
            * (equity + direction * self.units * price * friction)
            / (1 + direction * weight * friction)
        )
        quantity = rounded_down(abs(target_notional / price - self.units), p["quantity_step"])
        # Buy floor ensures no leverage. Sell floor cannot sell more than inventory.
        if direction > 0:
            quantity = min(
                quantity, rounded_down(self.cash / (fill * (1 + fee)), p["quantity_step"])
            )
        else:
            quantity = min(quantity, rounded_down(self.units, p["quantity_step"]))
        status = "filled"
        # A positive target is permitted to drift within the common dead band;
        # any overshoot above its target is trimmed if it exceeds the risk ceiling
        # by the caller's force flag (terminal=True also forces a risk reduction).
        if not terminal and weight > 0 and abs(weight - current) < p["rebalance_band"]:
            status = "rebalance_band"
        elif quantity < p["minimum_quantity"] or quantity * fill < p["minimum_notional"]:
            status = "minimum_order"
        if status != "filled":
            quantity = 0.0
        notional = quantity * fill
        commission = notional * fee
        self.units = float(Decimal(str(self.units)) + Decimal(str(direction * quantity)))
        self.cash -= direction * notional + commission
        if self.cash < -1e-7 or self.units < -1e-10:
            raise DataError("Account borrowed cash or units")
        cost = commission + quantity * abs(fill - price)
        return {
            "cash_before": before_cash,
            "units_before": before_units,
            "cash_after": self.cash,
            "units_after": self.units,
            "reference_price": price,
            "fill_price": fill,
            "signed_quantity": direction * quantity,
            "notional": notional,
            "commission": commission,
            "spread_slippage": cost - commission,
            "cost": cost,
            "equity_before": equity,
            "equity_after": self.equity(price),
            "target_weight": weight,
            "status": status,
        }


def allowed_actions(sigma: float, p: dict) -> np.ndarray:
    actions = np.array(p["actions"], dtype=float)
    return actions * max(sigma, 1e-8) * np.sqrt(365) <= p["annual_risk_budget"] + 1e-12


class TradingEnvironment:
    """One self-financing account per episode; no learning or normalization here."""

    def __init__(self, frame: pd.DataFrame, config: dict, scenario="base"):
        if scenario not in ("base", "double", "delay") or len(frame) < 2:
            raise DataError("Invalid scenario or insufficient episode")
        self.frame = frame.reset_index(drop=True)
        self.p = config["portfolio"]
        self.penalty = config["training"]["risk_penalty"]
        self.scenario = scenario
        self.multiplier = 2.0 if scenario == "double" else 1.0
        if not self.frame.reference_time.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise DataError("Episode must be daily and contiguous")
        if not (self.frame.reference_time <= self.frame.entry_time).all():
            raise DataError("Execution precedes signal availability")
        if not np.allclose(
            self.frame.exit_open.iloc[:-1], self.frame.entry_open.iloc[1:], rtol=0, atol=0
        ):
            raise DataError("Mark-to-market boundaries must match")
        self.reset()

    def reset(self):
        self.i = 0
        self.account = Account(self.p["initial_capital"])
        self.age = 0
        self.pending = None
        self.ledger = []
        self.orders = []
        self.nominal_target = 0.0
        return self.state()

    def state(self):
        row = self.frame.iloc[self.i]
        equity = self.account.equity(row.close)
        cost = (
            (self.p["commission_bps"] + self.p["half_spread_bps"] + self.p["slippage_bps"])
            * self.multiplier
            / 10000
        )
        return np.array(
            [
                row.probability,
                row.sigma,
                cost,
                self.account.units * row.close / equity,
                min(self.age, self.p["age_scale_days"]),
            ],
            dtype=np.float64,
        )

    def mask(self):
        return allowed_actions(float(self.frame.iloc[self.i].sigma), self.p)

    def step(self, action: int):
        if (
            not isinstance(action, (int, np.integer))
            or action not in range(3)
            or not self.mask()[action]
        ):
            raise DataError("Action outside predefined exposure/risk budget")
        row = self.frame.iloc[self.i]
        signal_exposure = float(self.state()[3])
        weight = self.p["actions"][action]
        self.nominal_target = weight
        decision = (weight, row.reference_time, float(row.sigma))
        if self.scenario == "delay":
            execute = self.pending
            self.pending = decision
        else:
            execute = decision
        before = self.account.equity(row.entry_open)
        held_before = self.account.units > 0
        orders = []
        if execute is not None:
            w, signal_time, signal_sigma = execute
            # In delay stress only information recorded with that queued order is used.
            ceiling = max(np.array(self.p["actions"])[allowed_actions(signal_sigma, self.p)])
            current = self.account.units * row.entry_open / before
            order = self.account.rebalance(
                w, row.entry_open, self.p, self.multiplier, terminal=current > ceiling
            )
            order.update(signal_time=signal_time, execution_time=row.entry_time, terminal=False)
            orders.append(order)
        after = self.account.equity(row.entry_open)
        exposure = self.account.units * row.entry_open / after
        held = self.account.units > 0
        self.age = self.age + 1 if held and held_before else int(held)
        end = self.account.equity(row.exit_open)
        done = self.i == len(self.frame) - 1
        if done:
            order = self.account.rebalance(
                0.0, row.exit_open, self.p, self.multiplier, terminal=True
            )
            order.update(
                signal_time=row.reference_time, execution_time=row.exit_time, terminal=True
            )
            orders.append(order)
            end = self.account.equity(row.exit_open)
        log_return = float(np.log(end / before))
        risk = self.penalty * (exposure * float(row.sigma)) ** 2
        cost = sum(o["cost"] for o in orders)
        turnover = sum(o["notional"] / o["equity_before"] for o in orders)
        record = {
            "reference_time": row.reference_time,
            "entry_time": row.entry_time,
            "exit_time": row.exit_time,
            "entry_open": float(row.entry_open),
            "exit_open": float(row.exit_open),
            "action": int(action),
            "probability": float(row.probability),
            "sigma": float(row.sigma),
            "exposure": exposure,
            "signal_exposure": signal_exposure,
            "age_days": self.age,
            "equity_before": before,
            "equity": end,
            "cash": self.account.cash,
            "units": self.account.units,
            "log_return": log_return,
            "reward": log_return - risk,
            "risk_penalty": risk,
            "cost": cost,
            "turnover": turnover,
            "filled_orders": sum(o["signed_quantity"] != 0 for o in orders),
        }
        self.orders.extend(orders)
        self.ledger.append(record)
        if not done:
            self.i += 1
        next_state = self.state()
        return next_state, log_return - risk, done


def rule_action(env: TradingEnvironment, policy: str) -> int:
    available = np.flatnonzero(env.mask())
    largest = int(available[-1])
    probability = env.state()[0]
    if policy == "cash":
        return 0
    if policy == "long_scaled":
        return largest
    if policy == "fixed":
        return largest if probability >= env.p["fixed_entry"] else 0
    if policy == "band":
        if probability >= env.p["fixed_entry"]:
            return largest
        if probability < env.p["band_exit"]:
            return 0
        return min(largest, int(np.argmin(abs(np.array(env.p["actions"]) - env.nominal_target))))
    raise DataError(f"Unknown policy: {policy}")


def rollout(frame, config, policy, scenario="base", normalizer=None, network=None):
    import torch

    env = TradingEnvironment(frame, config, scenario)
    while True:
        if policy == "dqn":
            with torch.no_grad():
                q = network(torch.from_numpy(normalizer.transform(env.state())[None])).numpy()[0]
            action = int(np.argmax(np.where(env.mask(), q, -np.inf)))
        else:
            action = rule_action(env, policy)
        _, _, done = env.step(action)
        if done:
            break
    return pd.DataFrame(env.ledger), pd.DataFrame(env.orders)
