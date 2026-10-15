from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from trading_ml.advanced_research.study import run_advanced_study
from trading_ml.backtesting import backtest
from trading_ml.common import read_json, write_json
from trading_ml.config import load_settings
from trading_ml.data import download
from trading_ml.errors import TradingMLError
from trading_ml.evaluation import evaluate
from trading_ml.evaluation.comparison import compare
from trading_ml.evaluation.smc_prospective import evaluate_smc_prospective
from trading_ml.mean_reversion.study import run_reversion_study
from trading_ml.models import load_model
from trading_ml.models.lifecycle import promote
from trading_ml.momentum.study import run_momentum_study
from trading_ml.prediction import predict
from trading_ml.prediction.smc_analysis import analyze_smc
from trading_ml.preprocessing import prepare
from trading_ml.risk_overlay.study import run_risk_study
from trading_ml.training import retrain, train
from trading_ml.training.optimization import optimize
from trading_ml.training.predictive_study import run_predictive_study
from trading_ml.training.smc_study import run_smc_study


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Motor independiente de ML para investigación de trading"
    )
    result.add_argument("--config", type=Path, default=Path("config/default.toml"))
    result.add_argument("--verbose", action="store_true")
    commands = result.add_subparsers(dest="command", required=True)
    from trading_ml.multimarket.cli import add_commands

    add_commands(commands)
    fetch = commands.add_parser("download", help="Descargar meses cerrados con checksum")
    fetch.add_argument("--output", type=Path)
    fetch.add_argument("--refresh", action="store_true")
    process = commands.add_parser("process", help="Validar y generar features/etiquetas")
    process.add_argument("--input", type=Path)
    process.add_argument("--output", type=Path)
    fit = commands.add_parser(
        "train", help="Entrenar y registrar candidato sin evaluar test ni activar"
    )
    fit.add_argument("--data", type=Path)
    promotion = commands.add_parser("promote", help="Activar explícitamente una versión evaluada")
    promotion.add_argument("--version", required=True)
    promotion.add_argument("--evaluation", type=Path, required=True)
    promotion.add_argument("--reason", required=True)
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
    predictive = commands.add_parser(
        "predictive-study", help="Baselines, ablaciones y bootstrap temporal sin promoción"
    )
    predictive.add_argument("--protocol", type=Path, default=Path("config/predictive_study.toml"))
    predictive.add_argument("--version", default="latest")
    smc_study = commands.add_parser(
        "smc-study", help="Comparar original, SMC reglas y XGBoost+SMC sin tocar el test observado"
    )
    smc_study.add_argument("--protocol", type=Path, default=Path("config/smc_study.toml"))
    smc_study.add_argument("--version", default="latest")
    momentum = commands.add_parser(
        "momentum-study", help="Momentum reglas/XGBoost, horizontes fijos y costos sin promoción"
    )
    momentum.add_argument("--protocol", type=Path, default=Path("config/momentum_study.toml"))
    momentum.add_argument("--raw", type=Path)
    reversion = commands.add_parser(
        "mean-reversion-study", help="Reversión: reglas, logística y XGBoost sin promoción"
    )
    reversion.add_argument(
        "--protocol", type=Path, default=Path("config/mean_reversion_study.toml")
    )
    reversion.add_argument("--raw", type=Path)
    risk = commands.add_parser(
        "risk-study", help="Sizing, costos, stops y drawdown sin alterar estrategias"
    )
    risk.add_argument("--protocol", type=Path, default=Path("config/risk_study.toml"))
    risk.add_argument("--raw", type=Path)
    advanced = commands.add_parser(
        "advanced-study", help="Investigación predictiva y meta-filtros con presupuesto cerrado"
    )
    advanced.add_argument("--protocol", type=Path, default=Path("config/advanced_research.toml"))
    advanced.add_argument("--raw", type=Path)
    feature_study = commands.add_parser(
        "feature-study", help="Features causales, selección temporal anidada y ablaciones"
    )
    feature_study.add_argument("--protocol", type=Path, default=Path("config/feature_study.toml"))
    label_study = commands.add_parser(
        "label-study", help="Triple Barrier, Meta-Labeling y costos en experimentos aislados"
    )
    label_study.add_argument("--protocol", type=Path, default=Path("config/label_study.toml"))
    advanced.add_argument(
        "--cache-from",
        type=Path,
        help="Recuperar ajustes de una corrida técnica fallida del mismo protocolo",
    )
    frontier = commands.add_parser(
        "frontier-study", help="Modelos multi-activo con embargo y test futuro sellado"
    )
    frontier.add_argument("--protocol", type=Path, default=Path("config/frontier_research.toml"))
    frontier.add_argument("--cache-from", type=Path)
    frontier_future = commands.add_parser(
        "frontier-future-evaluate", help="Abrir una vez el test futuro de modelos congelados"
    )
    frontier_future.add_argument("--study", type=Path, required=True)
    frontier_future.add_argument("--btc-raw", type=Path, required=True)
    frontier_future.add_argument("--eth-raw", type=Path, required=True)
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
        if args.command.startswith("market-"):
            from trading_ml.multimarket.cli import dispatch

            result = dispatch(settings, args)
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        dataset = getattr(args, "data", None) or settings.root / "data" / "prepared.csv"
        if args.command == "download":
            result = {"raw_data": str(download(settings, args.output, args.refresh))}
        elif args.command == "process":
            result = {"prepared_data": str(prepare(settings, args.input, args.output))}
        elif args.command == "train":
            version = train(settings, dataset)
            result = {
                "model_version": version,
                "baseline_gate": load_model(settings.root, version).metadata["baseline_gate"],
            }
        elif args.command == "retrain":
            version = retrain(settings.root, dataset, args.version)
            result = {
                "model_version": version,
                "baseline_gate": load_model(settings.root, version).metadata["baseline_gate"],
            }
        elif args.command == "promote":
            result = promote(settings.root, args.version, args.evaluation, args.reason)
        elif args.command == "predictive-study":
            report = run_predictive_study(settings, args.protocol, args.version)
            result = {
                "run_id": report["run_id"],
                "directory": report["directory"],
                "actual_fits": report["actual_fits"],
                "models": {
                    name: {"metrics": data["metrics"], "decision": data.get("decision")}
                    for name, data in report["development"]["models"].items()
                },
                "promoted": False,
            }
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
        elif args.command == "momentum-study":
            report = run_momentum_study(settings, args.protocol, args.raw)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "actual_xgboost_fits",
                    "experiment_records",
                    "independent_advantage_confirmed",
                    "promoted",
                    "live_trading",
                    "prospective",
                ]
            }
        elif args.command == "mean-reversion-study":
            report = run_reversion_study(settings, args.protocol, args.raw)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "actual_fits",
                    "experiment_records",
                    "existing_models_preserved",
                    "independent_advantage_confirmed",
                    "promoted",
                    "live_trading",
                    "prospective",
                ]
            }
        elif args.command == "risk-study":
            report = run_risk_study(settings, args.protocol, args.raw)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "actual_fits",
                    "experiment_records",
                    "diagnosis",
                    "existing_models_preserved",
                    "prospective_unchanged",
                    "prospective_evaluated",
                    "promoted",
                    "live_trading",
                ]
            }
        elif args.command == "advanced-study":
            report = run_advanced_study(settings, args.protocol, args.raw, args.cache_from)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "comparison",
                    "actual_fits",
                    "experiment_records",
                    "existing_models_preserved",
                    "prospective_unchanged",
                    "prospective_evaluated",
                    "promoted",
                    "live_trading",
                ]
            }
        elif args.command == "label-study":
            from trading_ml.label_research.study import run_label_study

            report = run_label_study(settings, args.protocol)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "actual_fits",
                    "historical_qualified_candidates",
                    "conclusion",
                    "existing_models_preserved",
                    "prospective_unchanged",
                    "promoted",
                ]
            }
        elif args.command == "feature-study":
            from trading_ml.feature_research.study import run_feature_study

            report = run_feature_study(settings, args.protocol)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "actual_fits",
                    "historical_qualified_recipes",
                    "conclusion",
                    "existing_models_preserved",
                    "prospective_unchanged",
                    "prospective_evaluated",
                    "promoted",
                ]
            }
        elif args.command == "frontier-study":
            from trading_ml.frontier_research.study import run_frontier_study

            report = run_frontier_study(settings, args.protocol, args.cache_from)
            result = {
                key: report[key]
                for key in [
                    "run_id",
                    "directory",
                    "actual_fits",
                    "records",
                    "selection",
                    "conclusion",
                    "prospective_evaluated",
                    "promoted",
                    "live_trading",
                ]
            }
        elif args.command == "frontier-future-evaluate":
            from trading_ml.frontier_research.prospective import evaluate_frontier_future

            result = evaluate_frontier_future(settings, args.study, args.btc_raw, args.eth_raw)
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
                result = backtest(bundle, dataset, output=destination).summary
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (TradingMLError, OSError, ValueError, KeyError, TypeError) as exc:
        logging.error("%s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
