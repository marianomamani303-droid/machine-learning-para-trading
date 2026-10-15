from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.advanced_research.features import CORE, FLOW, MOMENTUM, build_research_features
from trading_ml.errors import DataError

LOCAL = CORE + FLOW + MOMENTUM
CROSS = [
    "peer_return_1",
    "peer_return_3",
    "peer_momentum_20",
    "relative_return_20",
    "cross_correlation_60",
    "cross_beta_60",
    "spread_z_60",
]


def multi_asset_features(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if len(raw) != 2:
        raise DataError("La investigación cruzada requiere exactamente dos activos")
    built = {symbol: build_research_features(frame) for symbol, frame in raw.items()}
    output = {}
    for symbol, own in built.items():
        peer = built[next(s for s in built if s != symbol)]
        if own.reference_time.duplicated().any() or peer.reference_time.duplicated().any():
            raise DataError("Referencias cruzadas duplicadas")
        # Exact closed-candle alignment: no forward filling a stale peer observation.
        aligned = own.merge(
            peer[["reference_time", "close", "return_1", "return_3", "adv_momentum_20"]],
            on="reference_time",
            how="left",
            suffixes=("", "_peer"),
            validate="one_to_one",
        )
        aligned["peer_return_1"] = aligned.return_1_peer
        aligned["peer_return_3"] = aligned.return_3_peer
        aligned["peer_momentum_20"] = aligned.adv_momentum_20_peer
        r, q = np.log(aligned.close).diff(), np.log(aligned.close_peer).diff()
        aligned["relative_return_20"] = r.rolling(20).sum() - q.rolling(20).sum()
        aligned["cross_correlation_60"] = r.rolling(60).corr(q)
        aligned["cross_beta_60"] = r.rolling(60).cov(q) / q.rolling(60).var().replace(0, np.nan)
        spread = np.log(aligned.close / aligned.close_peer)
        aligned["spread_z_60"] = (spread - spread.shift(1).rolling(60).mean()) / (
            spread.shift(1).rolling(60).std().replace(0, np.nan)
        )
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        aligned["research_ready"] &= aligned[LOCAL + CROSS].notna().all(axis=1)
        output[symbol] = aligned
    return output
