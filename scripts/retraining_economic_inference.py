"""Resample paired realized policy returns; do not rerun execution on shuffled bars."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, write_new_json
from trading_ml.evaluation.uncertainty import moving_block_indices
from trading_ml.multimarket.markets import session_end

OUT = Path("artifacts/retraining_comparison/20261017_controlled")


def daily_curve(path, fx=False, neural=False):
    f = pd.read_csv(path)
    if neural:
        f["time"] = pd.to_datetime(f.reference_time, utc=True)
        return f[["time", "net_return"]]
    f["time"] = pd.to_datetime(f.timestamp, utc=True)
    if fx:
        f["time"] = session_end(pd.DatetimeIndex(f.time) - pd.Timedelta(nanoseconds=1))
    # Initial row + multiple marks may share a date. Preserve starting capital separately.
    initial = 10000.0
    f = f.groupby("time", as_index=False).equity.last()
    f["net_return"] = f.equity / np.r_[initial, f.equity.to_numpy()[:-1]] - 1
    return f[["time", "net_return"]]


def main():
    groups = defaultdict(list)
    for section in ["fits", "events", "forex", "neural"]:
        for p in (OUT / section).rglob("complete.json"):
            m = read_json(p.parent / "fit.json")
            key = (m.get("symbol", "BTC_ETH_PANEL"), m.get("interval", "1d"), m["recipe"])
            for scenario in ["base", "double"]:
                f = daily_curve(
                    p.parent / f"{scenario}_curve.csv",
                    fx=section == "forex",
                    neural=section == "neural",
                )
                f["fold"] = m["fold"]
                f["seed"] = m["seed"]
                f["arm"] = m["arm"]
                groups[key + (scenario,)].append(f)
    results = {}
    for key, parts in groups.items():
        f = pd.concat(parts, ignore_index=True)
        w = f.pivot(
            index=["time", "fold"], columns=["arm", "seed"], values="net_return"
        ).sort_index()
        # Omit entire unavailable fold/seed policies based on past calibration status.
        both = set(f.loc[f.arm == "A", "fold"]) & set(f.loc[f.arm == "B", "fold"])
        w = w.loc[w.index.get_level_values("fold").isin(both)]
        assert not w.isna().any().any(), key
        seedlist = sorted(f.seed.unique())
        pa = np.column_stack([w[("A", s)].to_numpy() for s in seedlist])
        pb = np.column_stack([w[("B", s)].to_numpy() for s in seedlist])
        years = 252 if key[0] == "EURUSD" else 365

        def calc(r, years=years):
            total = np.prod(1 + r, axis=-1) - 1
            std = r.std(axis=-1, ddof=1)
            sharpe = np.divide(
                r.mean(axis=-1) * np.sqrt(years),
                std,
                out=np.full_like(std, np.nan),
                where=std > 1e-12,
            )
            return total, sharpe

        point_a = calc(pa.T)
        point_b = calc(pb.T)
        output = {}
        for block in [7, 30]:
            idx = moving_block_indices(w.index.get_level_values("fold").to_numpy(), 2000, block, 42)
            draws_r = []
            draws_s = []
            for j in range(len(seedlist)):
                ra, sa = calc(pa[:, j][idx])
                rb, sb = calc(pb[:, j][idx])
                draws_r.append(rb - ra)
                draws_s.append(sb - sa)
            dr = np.mean(draws_r, axis=0)
            ds = np.mean(draws_s, axis=0)
            output[str(block)] = {
                "net_return_A": float(np.mean(point_a[0])),
                "net_return_B": float(np.mean(point_b[0])),
                "net_return_delta": float(np.mean(point_b[0] - point_a[0])),
                "net_return_delta_ci95": np.quantile(dr, [0.025, 0.975]).tolist(),
                "sharpe_A": float(np.nanmean(point_a[1]))
                if np.isfinite(point_a[1]).any()
                else None,
                "sharpe_B": float(np.nanmean(point_b[1]))
                if np.isfinite(point_b[1]).any()
                else None,
                "sharpe_delta_ci95": np.nanquantile(ds, [0.025, 0.975]).tolist()
                if np.isfinite(ds).any()
                else None,
                "dates": len(w),
                "seeds": len(seedlist),
                "pointwise_only": True,
            }
        results["|".join(key)] = output
    write_new_json(OUT / "economic_inference.json", results)
    print("Paired economic policies:", len(results), flush=True)


if __name__ == "__main__":
    main()
