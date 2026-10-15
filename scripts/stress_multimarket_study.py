"""Cost stress of every saved policy; no fits, no threshold selection."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.multimarket.data import execution_source, load_snapshot
from trading_ml.multimarket.execution import Costs, Engine
from trading_ml.multimarket.features import features
from trading_ml.multimarket.markets import instrument


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for study in args.studies:
        report = read_json(study / "summary.json")
        for key, asset in report["results"].items():
            directory = study / key
            raw, origin = load_snapshot(Path(asset["data"]))
            frame = features(raw, asset["symbol"])
            execution, multiplier = execution_source(raw, origin)
            costs = Costs(**read_json(directory / "direction_price_h1_fold1.json")["costs"])
            stressed = replace(
                costs,
                forex_commission_bps=costs.forex_commission_bps * 2,
                forex_slippage_pips=costs.forex_slippage_pips * 2,
                forex_swap_annual_bps=300,
                crypto_commission_bps=costs.crypto_commission_bps * 2,
                crypto_slippage_bps=costs.crypto_slippage_bps * 2,
            )
            scenarios = {"double_friction_swap300": stressed}
            if instrument(asset["symbol"]).market == "forex":
                scenarios["double_friction_swap300_extra1pip"] = replace(
                    stressed, forex_additional_spread_pips=1.0
                )
            for scenario, parameters in scenarios.items():
                engine = Engine(
                    frame,
                    asset["symbol"],
                    parameters,
                    execution_frame=execution,
                    bars_per_horizon=multiplier,
                    expiry_policy="barrier_first",
                )
                for name, data in asset["experiments"].items():
                    prediction_path = directory / f"{name}_predictions.csv"
                    prediction = pd.read_csv(prediction_path)
                    threshold = int(name.rsplit("_t", 1)[1]) / 100 if "_t" in name else 0.55
                    horizon = 3 if name.startswith("tb_") or name == "direction_compact_h3" else 1
                    folds = []
                    for number, part in prediction.groupby("fold", sort=True):
                        end = pd.to_datetime(
                            next(f["end"] for f in data["folds"] if f["fold"] == number), utc=True
                        )
                        summary, _, _ = engine.portfolio(
                            part, horizon, name.startswith("tb_"), threshold, end
                        )
                        folds.append(summary)
                    rows.append(
                        {
                            "run_id": report["run_id"],
                            "symbol": asset["symbol"],
                            "interval": asset["interval"],
                            "experiment": name,
                            "scenario": scenario,
                            "base_net_return": data["net_return"],
                            "stress_net_return": float(
                                np.prod([1 + f["net_return"] for f in folds]) - 1
                            ),
                            "stress_positive_folds": sum(f["net_return"] > 0 for f in folds),
                            "folds": len(folds),
                            "stress_trades": sum(f["trades"] for f in folds),
                            "forecast_sha256": sha256(prediction_path),
                            "expiry_policy": "barrier_first",
                        }
                    )
                print(key, scenario, "completed", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "cost_sensitivity.csv", index=False)
    write_new_json(
        args.output / "protocol.json",
        {
            "studies": [str(s.resolve()) for s in args.studies],
            "double_friction": True,
            "forex_swap_annual_bps": 300,
            "additional_spread_scenario_pips": 1.0,
            "expiry_policy": "barrier_first",
            "all_saved_policies": True,
            "new_fits": 0,
            "selection": "exploratory sensitivity, no new model or threshold chosen",
            "rows": len(table),
        },
    )


if __name__ == "__main__":
    main()
