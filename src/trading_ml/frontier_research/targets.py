from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.advanced_research.targets import build_targets
from trading_ml.momentum.config import ResearchCosts


def research_targets(frame: pd.DataFrame, costs: ResearchCosts) -> dict[str, pd.DataFrame]:
    output = build_targets(frame, costs)
    output.pop("net_h5")
    barrier = frame.copy()
    y, times, payoffs, reasons = [], [], [], []
    # Close-monitored competing barriers, executed at following open. Daily high/low
    # cannot identify TP/SL order, so this deliberately estimates a different event.
    for i in range(len(frame)):
        if i + 6 >= len(frame) or not np.isfinite(frame.risk_atr.iloc[i]):
            y.append(np.nan)
            times.append(pd.NaT)
            payoffs.append(np.nan)
            reasons.append("censored")
            continue
        entry = frame.open.iloc[i + 1]
        atr = frame.risk_atr.iloc[i]
        upper, lower = entry + 2 * atr, entry - 2 * atr
        exit_index, reason = i + 6, "timeout"
        for j in range(i + 1, i + 6):
            if frame.close.iloc[j] >= upper:
                exit_index, reason = j + 1, "tp_close"
                break
            if frame.close.iloc[j] <= lower:
                exit_index, reason = j + 1, "sl_close"
                break
        y.append(float(reason == "tp_close"))
        times.append(frame.timestamp.iloc[exit_index])
        payoffs.append(
            frame.open.iloc[exit_index] / entry * costs.sell_factor / costs.buy_factor - 1
        )
        reasons.append(reason)
    barrier["target"] = y
    barrier["target_time"] = pd.to_datetime(times, utc=True)
    barrier["net_return_target"] = payoffs
    barrier["barrier_outcome"] = reasons
    output["barrier_close_h5"] = barrier
    return output
