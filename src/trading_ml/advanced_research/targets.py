from __future__ import annotations

import pandas as pd

from trading_ml.momentum.config import ResearchCosts


def build_targets(frame: pd.DataFrame, costs: ResearchCosts) -> dict[str, pd.DataFrame]:
    targets = {}
    for horizon in [1, 3, 5]:
        output = frame.copy()
        entry = frame.open.shift(-1)
        exit_price = frame.open.shift(-(horizon + 1))
        output["net_return_target"] = exit_price / entry * costs.sell_factor / costs.buy_factor - 1
        output["target_time"] = frame.timestamp.shift(-(horizon + 1))
        if horizon == 1:
            future = frame.close.shift(-1)
            output["target"] = (future > frame.close).astype(float).where(future.notna())
            output["target_time"] = frame.reference_time.shift(-1)
            key = "direction_h1"
        else:
            output["target"] = (
                (output.net_return_target > 0).astype(float).where(exit_price.notna())
            )
            key = f"net_h{horizon}"
        targets[key] = output
    return targets
