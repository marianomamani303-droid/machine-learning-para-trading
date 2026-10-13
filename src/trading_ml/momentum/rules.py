from __future__ import annotations

import pandas as pd

from trading_ml.momentum.config import MomentumConfig


def momentum_signals(frame: pd.DataFrame, config: MomentumConfig) -> pd.DataFrame:
    """Entry filter only; exit and holding rules are shared by rules and ML."""
    config.validate()
    entry = (
        frame.mom_ready
        & (frame.mom_normalized >= config.minimum_normalized_momentum)
        & (frame.mom_fast_distance > 0)
        & (frame.mom_slow_distance > 0)
        & (frame.mom_slow_slope > 0)
        & (frame.mom_efficiency >= config.minimum_efficiency)
        & (frame.mom_trend_persistence >= config.minimum_persistence)
        & (frame.mom_breakout_up == 1)
    )
    exit_signal = (
        (frame.mom_log_return <= 0)
        | (frame.mom_slow_distance <= 0)
        | (frame.mom_breakout_down == 1)
    )
    return pd.DataFrame({"entry": entry, "exit": exit_signal}, index=frame.index)
