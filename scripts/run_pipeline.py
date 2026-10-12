"""Ejecuta el pipeline real y conserva un manifiesto de resultados."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from trading_ml.backtesting import backtest
from trading_ml.common import environment, read_json, sidecar, write_json
from trading_ml.config import load_settings
from trading_ml.data import download
from trading_ml.evaluation import evaluate
from trading_ml.models import load_model
from trading_ml.prediction import predict
from trading_ml.preprocessing import prepare
from trading_ml.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings(args.config)
    raw = download(settings, refresh=args.refresh)
    prepared = prepare(settings, raw)
    version = train(settings, prepared)
    bundle = load_model(settings.root, version)
    destination = settings.root / "reports" / version
    evaluation = evaluate(bundle, prepared, destination / "evaluation.json")
    prediction = predict(bundle, prepared)
    write_json(destination / "prediction.json", prediction)
    result = backtest(bundle, prepared, output=destination / "backtest")
    write_json(
        destination / "run.json",
        {
            "version": version,
            "environment": environment(),
            "raw_data": read_json(sidecar(raw)),
            "prepared_data": read_json(sidecar(prepared)),
            "evaluation": evaluation,
            "prediction": prediction,
            "backtest": result.summary,
        },
    )
    print(f"Pipeline completo verificado: {version}; reportes en {destination}")


if __name__ == "__main__":
    main()
