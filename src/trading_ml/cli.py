from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from trading_ml.backtesting import backtest
from trading_ml.common import read_json, write_json
from trading_ml.config import load_settings
from trading_ml.data import download
from trading_ml.errors import TradingMLError
from trading_ml.evaluation import evaluate
from trading_ml.evaluation.comparison import compare
from trading_ml.evaluation.smc_prospective import evaluate_smc_prospective
from trading_ml.models import load_model
from trading_ml.prediction import predict
from trading_ml.prediction.smc_analysis import analyze_smc
from trading_ml.preprocessing import prepare
from trading_ml.training import retrain, train
from trading_ml.training.optimization import optimize
from trading_ml.training.smc_study import run_smc_study


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Motor independiente de ML para investigación de trading"
    )
    result.add_argument("--config", type=Path, default=Path("config/default.toml"))
    result.add_argument("--verbose", action="store_true")
    commands = result.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Descargar meses cerrados con checksum")
    fetch.add_argument("--output", type=Path)
    fetch.add_argument("--refresh", action="store_true")
    process = commands.add_parser("process", help="Validar y generar features/etiquetas")
    process.add_argument("--input", type=Path)
    process.add_argument("--output", type=Path)
    fit = commands.add_parser("train", help="Entrenar y evaluar test temporal")
    fit.add_argument("--data", type=Path)
    for name in ["evaluate", "predict", "backtest", "retrain"]:
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path)
        command.add_argument("--version", default="latest")
        if name != "retrain":
            command.add_argument("--output", type=Path)
        if name == "predict":
            command.add_argument("--reference", help="Referencia de cierre UTC, p.ej. 2026-10-01")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--data", type=Path)
    comparison.add_argument("--versions", nargs="+", required=True)
    comparison.add_argument("--output", type=Path)
    optimization = commands.add_parser(
        "optimize", help="Máximo cinco recetas y walk-forward con purga"
    )
    optimization.add_argument("--protocol", type=Path, default=Path("config/optimization.toml"))
    optimization.add_argument("--version", default="latest")
    smc_study = commands.add_parser(
        "smc-study", help="Comparar original, SMC reglas y XGBoost+SMC sin tocar el test observado"
    )
    smc_study.add_argument("--protocol", type=Path, default=Path("config/smc_study.toml"))
    smc_study.add_argument("--version", default="latest")
    analysis = commands.add_parser(
        "smc-analyze",
        help="Analizar estructura y probabilidad con el candidato SMC de investigación",
    )
    analysis.add_argument("--version")
    analysis.add_argument("--data", type=Path)
    analysis.add_argument("--reference")
    analysis.add_argument("--output", type=Path)
    future = commands.add_parser(
        "smc-future-evaluate",
        help="Evaluar la reserva futura completa con los tres sistemas congelados",
    )
    future.add_argument("--raw", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        settings = load_settings(args.config)
        dataset = getattr(args, "data", None) or settings.root / "data" / "prepared.csv"
        if args.command == "download":
            result = {"raw_data": str(download(settings, args.output, args.refresh))}
        elif args.command == "process":
            result = {"prepared_data": str(prepare(settings, args.input, args.output))}
        elif args.command == "train":
            result = {"model_version": train(settings, dataset)}
        elif args.command == "retrain":
            result = {"model_version": retrain(settings.root, dataset, args.version)}
        elif args.command == "smc-study":
            report = run_smc_study(settings, args.protocol, args.version)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "reference_version",
                    "candidate_version",
                    "candidate_dataset",
                    "selected_recipe",
                    "executed_experiments",
                    "stop_reason",
                    "passed_walk_forward",
                    "promoted",
                    "old_test_evaluated",
                    "prospective",
                ]
            }
        elif args.command == "smc-analyze":
            version = args.version
            if version is None:
                pointer = read_json(settings.root / "smc_studies" / "latest.json")
                version = read_json(Path(pointer["summary"]))["candidate_version"]
            bundle = load_model(settings.root, version)
            data = args.data or Path(bundle.metadata["dataset_path"])
            result = analyze_smc(bundle, data, args.reference)
            if args.output:
                write_json(args.output, result)
        elif args.command == "smc-future-evaluate":
            result = evaluate_smc_prospective(settings.root, args.raw)
        elif args.command == "optimize":
            report = optimize(settings, args.protocol, args.version)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "previous_version",
                    "candidate_version",
                    "champion_version",
                    "selected_experiment",
                    "executed_experiments",
                    "stop_reason",
                    "passed_walk_forward",
                    "promoted",
                    "comparison",
                ]
            }
        elif args.command == "compare":
            result = compare(settings.root, dataset, args.versions)
            if args.output:
                write_json(args.output, result)
        else:
            bundle = load_model(settings.root, args.version)
            if args.command == "evaluate":
                result = evaluate(bundle, dataset, args.output)
            elif args.command == "predict":
                result = predict(bundle, dataset, args.reference)
                if args.output:
                    write_json(args.output, result)
            else:
                destination = (
                    args.output
                    or settings.root / "reports" / bundle.metadata["version"] / "backtest"
                )
                result = backtest(bundle, dataset, settings.backtest, destination).summary
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (TradingMLError, OSError, ValueError, KeyError, TypeError) as exc:
        logging.error("%s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
