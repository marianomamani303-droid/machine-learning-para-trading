from __future__ import annotations

import pandas as pd

from trading_ml.mean_reversion.config import ReversionConfig


def reversion_opportunities(frame: pd.DataFrame, config: ReversionConfig) -> pd.Series:
    config.validate()
    opportunities = (
        frame.mr_ready & (frame.mr_z <= config.entry_z) & (frame.mr_rsi <= config.maximum_rsi)
    )
    if config.sideways_only:
        opportunities &= frame.regime == "sideways"
    return opportunities.rename("opportunity")


def exit_reason(
    close: float, target: float, stop: float, holding_bars: int, maximum: int
) -> str | None:
    """Close-triggered exits executed at next open; never promise a hard stop fill."""
    if close <= stop:
        return "stop"
    if close >= target:
        return "reversion"
    if holding_bars >= maximum:
        return "timeout"
    return None
