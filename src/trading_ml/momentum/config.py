from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_ml.errors import DataError


@dataclass(frozen=True)
class MomentumConfig:
    lookback: int = 60
    volatility_window: int = 20
    slope_window: int = 5
    minimum_normalized_momentum: float = 0.5
    minimum_efficiency: float = 0.10
    minimum_persistence: float = 0.55
    max_holding_multiplier: int = 2

    @property
    def fast_window(self) -> int:
        return max(5, self.lookback // 2)

    @property
    def slow_window(self) -> int:
        return self.lookback * 2

    @property
    def max_holding_bars(self) -> int:
        return self.lookback * self.max_holding_multiplier

    def validate(self) -> None:
        if (
            any(
                isinstance(x, bool) or not isinstance(x, int) or x < 2
                for x in [self.lookback, self.volatility_window, self.slope_window]
            )
            or not isinstance(self.max_holding_multiplier, int)
            or self.max_holding_multiplier < 1
        ):
            raise DataError("Ventanas Momentum inválidas")
        if not np.isfinite(
            [self.minimum_normalized_momentum, self.minimum_efficiency, self.minimum_persistence]
        ).all() or not (
            self.minimum_normalized_momentum >= 0
            and 0 <= self.minimum_efficiency <= 1
            and 0.5 <= self.minimum_persistence <= 1
        ):
            raise DataError("Umbrales Momentum inválidos")


@dataclass(frozen=True)
class ResearchCosts:
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    spread_bps: float = 2.0  # Full quoted spread; half on each side.
    annual_holding_bps: float = 0.0
    annual_cash_rate: float = 0.0
    initial_capital: float = 10000.0

    def validate(self) -> None:
        if not np.isfinite(list(self.__dict__.values())).all() or not (
            self.initial_capital > 0
            and 0 <= self.commission_bps < 10000
            and 0 <= self.slippage_bps + self.spread_bps / 2 < 10000
            and self.slippage_bps >= 0
            and self.spread_bps >= 0
            and self.annual_holding_bps >= 0
            and self.annual_cash_rate > -1
        ):
            raise DataError("Costos Momentum inválidos")

    @property
    def buy_factor(self) -> float:
        return (1 + (self.slippage_bps + self.spread_bps / 2) / 10000) * (
            1 + self.commission_bps / 10000
        )

    @property
    def sell_factor(self) -> float:
        return (1 - (self.slippage_bps + self.spread_bps / 2) / 10000) * (
            1 - self.commission_bps / 10000
        )
