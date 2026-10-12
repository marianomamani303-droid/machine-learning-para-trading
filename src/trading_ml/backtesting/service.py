from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_ml.backtesting.simulator import BacktestResult, simulate
from trading_ml.common import write_json
from trading_ml.config.settings import Backtest
from trading_ml.evaluation.service import out_of_sample
from trading_ml.models import ModelBundle
from trading_ml.preprocessing import load_prepared


def backtest(
    bundle: ModelBundle, dataset: Path, costs: Backtest | None = None, output: Path | None = None
) -> BacktestResult:
    frame, provenance = load_prepared(dataset, bundle.metadata["contract"])
    test = out_of_sample(bundle, frame)
    signals = test[["timestamp", "reference_time"]].copy()
    signals["probability_up"] = bundle.probabilities(test)
    costs = costs or Backtest(**bundle.metadata["config"]["backtest"])
    result = simulate(
        frame,
        signals,
        bundle.metadata["contract"]["horizon"],
        costs,
        pd.Timestamp(bundle.metadata["selection_label_end"]),
    )
    result.summary.update(
        {
            "model_version": bundle.metadata["version"],
            "dataset_sha256": provenance["sha256"],
            "selection_label_end": bundle.metadata["selection_label_end"],
        }
    )
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "summary.json", result.summary)
        result.equity.to_csv(output / "equity.csv", index=False)
        result.trades.to_csv(output / "trades.csv", index=False)
    return result
