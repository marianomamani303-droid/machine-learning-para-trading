from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_ml.errors import DataError


@dataclass(frozen=True)
class RiskPolicy:
    name: str = "original"
    capital_cap: float = 1.0
    sizing: str = "fixed"
    volatility_target_per_bar: float = 0.01
    risk_budget: float = 0.01
    atr_multiple: float = 3.0
    atr_stop: bool = False
    soft_drawdown: float | None = None
    hard_drawdown: float | None = None
    reduced_fraction: float = 0.5
    confirmations: int = 1
    cooldown_bars: int = 0
    persistent: bool = False
    exit_threshold: float = 0.50
    max_holding_bars: int | None = None

    def validate(self) -> None:
        values = [
            self.capital_cap,
            self.volatility_target_per_bar,
            self.risk_budget,
            self.atr_multiple,
            self.reduced_fraction,
            self.exit_threshold,
        ]
        if (
            not self.name
            or not np.isfinite(values).all()
            or not (
                0 < self.capital_cap <= 1
                and self.sizing in {"fixed", "volatility", "atr", "combined"}
                and self.volatility_target_per_bar > 0
                and 0 < self.risk_budget <= 1
                and self.atr_multiple > 0
                and 0 < self.reduced_fraction < 1
                and 0 <= self.exit_threshold <= 1
            )
        ):
            raise DataError("Política de riesgo inválida")
        for value in [self.confirmations, self.cooldown_bars]:
            if not isinstance(value, int) or isinstance(value, bool):
                raise DataError("Confirmación/cooldown inválidos")
        if self.confirmations < 1 or self.cooldown_bars < 0:
            raise DataError("Confirmación/cooldown inválidos")
        if self.max_holding_bars is not None and (
            not isinstance(self.max_holding_bars, int) or self.max_holding_bars < 1
        ):
            raise DataError("Permanencia de riesgo inválida")
        if (self.soft_drawdown is None) != (self.hard_drawdown is None):
            raise DataError("Límites de drawdown incompletos")
        if self.soft_drawdown is not None and not 0 < self.soft_drawdown < self.hard_drawdown < 1:
            raise DataError("Límites de drawdown inválidos")


def entry_fraction(
    policy: RiskPolicy, volatility: float, atr_ratio: float, drawdown: float
) -> float:
    fraction = policy.capital_cap
    if policy.sizing in {"volatility", "combined"}:
        if not np.isfinite(volatility) or volatility <= 0:
            return 0.0
        fraction = min(fraction, policy.volatility_target_per_bar / volatility)
    if policy.sizing in {"atr", "combined"}:
        if not np.isfinite(atr_ratio) or atr_ratio <= 0:
            return 0.0
        fraction = min(fraction, policy.risk_budget / (policy.atr_multiple * atr_ratio))
    if policy.soft_drawdown is not None and drawdown <= -policy.soft_drawdown:
        fraction *= policy.reduced_fraction
    return float(fraction)
