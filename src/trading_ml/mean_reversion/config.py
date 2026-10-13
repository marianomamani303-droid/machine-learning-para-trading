from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_ml.errors import DataError
from trading_ml.momentum.config import ResearchCosts


@dataclass(frozen=True)
class ReversionConfig:
    lookback: int = 20
    rsi_window: int = 14
    volatility_window: int = 20
    atr_window: int = 14
    entry_z: float = -2.0
    maximum_rsi: float = 0.40
    stop_atr: float = 2.0
    max_holding_bars: int = 10
    risk_fraction: float = 0.01
    max_allocation: float = 0.50
    sideways_only: bool = False

    def validate(self) -> None:
        for x in [
            self.lookback,
            self.rsi_window,
            self.volatility_window,
            self.atr_window,
            self.max_holding_bars,
        ]:
            if isinstance(x, bool) or not isinstance(x, int) or x < 2:
                raise DataError("Ventanas Mean Reversion inválidas")
        values = [
            self.entry_z,
            self.maximum_rsi,
            self.stop_atr,
            self.risk_fraction,
            self.max_allocation,
        ]
        if not np.isfinite(values).all() or not (
            self.entry_z < 0
            and 0 < self.maximum_rsi < 0.5
            and self.stop_atr > 0
            and 0 < self.risk_fraction <= self.max_allocation <= 1
            and isinstance(self.sideways_only, bool)
        ):
            raise DataError("Umbrales o riesgo Mean Reversion inválidos")


__all__ = ["ResearchCosts", "ReversionConfig"]
