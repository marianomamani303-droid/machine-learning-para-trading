from __future__ import annotations

from pathlib import Path


def add_commands(commands):
    download = commands.add_parser(
        "market-download", help="Descargar/verificar Forex bid/ask o reutilizar cripto"
    )
    download.add_argument("--symbol", required=True)
    download.add_argument("--interval", choices=["1h", "4h", "1d"], default="1d")
    download.add_argument("--protocol", type=Path, default=Path("config/multimarket.toml"))
    download.add_argument("--refresh", action="store_true")
    study = commands.add_parser(
        "market-study", help="Experimentos multimercado aislados; primero piloto EURUSD"
    )
    study.add_argument("--protocol", type=Path, default=Path("config/multimarket.toml"))
    study.add_argument("--symbols", nargs="+")
    study.add_argument("--intervals", nargs="+", choices=["1h", "4h", "1d"])
    study.add_argument("--pilot", type=Path)
    study.add_argument("--refresh", action="store_true")
    train = commands.add_parser(
        "market-train", help="Entrenar candidato independiente sin promoción"
    )
    train.add_argument("--protocol", type=Path, default=Path("config/multimarket.toml"))
    train.add_argument("--symbol", required=True)
    train.add_argument("--interval", default="1d", choices=["1h", "4h", "1d"])
    train.add_argument("--cutoff", required=True)
    train.add_argument("--recipe", choices=["price", "compact"], default="price")
    train.add_argument(
        "--target", choices=["direction", "net_profit", "triple_barrier"], default="direction"
    )
    train.add_argument("--horizon", type=int, default=1)
    train.add_argument("--data", type=Path)
    for command in ["market-predict", "market-evaluate", "market-backtest"]:
        sub = commands.add_parser(command)
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--calibrated", action="store_true")
        if command == "market-predict":
            sub.add_argument("--reference")
        else:
            sub.add_argument("--start", required=True)
            sub.add_argument("--end", required=True)
            sub.add_argument("--output", type=Path)


def dispatch(settings, args):
    from trading_ml.multimarket.protocol import load_protocol
    from trading_ml.multimarket.service import evaluate_market, predict_market, train_market
    from trading_ml.multimarket.study import dataset, run_study

    if args.command == "market-download":
        path = dataset(
            settings, load_protocol(args.protocol), args.symbol, args.interval, args.refresh
        )
        return {"dataset": str(path.resolve())}
    if args.command == "market-study":
        report = run_study(
            settings, args.protocol, args.symbols, args.intervals, args.pilot, args.refresh
        )
        return {
            k: report[k]
            for k in [
                "run_id",
                "directory",
                "pilot_pipeline_validated",
                "actual_fits",
                "frozen_unchanged",
                "adopted",
            ]
        }
    if args.command == "market-train":
        return train_market(
            settings,
            args.protocol,
            args.symbol,
            args.interval,
            args.cutoff,
            args.recipe,
            args.target,
            args.horizon,
            args.data,
        )
    if args.command == "market-predict":
        return predict_market(args.model, args.data, args.reference, args.calibrated)
    return evaluate_market(
        settings, args.model, args.data, args.start, args.end, args.calibrated, args.output
    )
