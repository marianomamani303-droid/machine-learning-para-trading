"""PowerShell-friendly commands for the durable, prediction-only online engine."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import time
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from trading_ml.errors import TradingMLError

LOGGER = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo")
    return number


def _positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Debe ser un número finito mayor que cero")
    return number


def _prediction_limit(value: str) -> int:
    number = _positive_int(value)
    if number > 1000:
        raise argparse.ArgumentTypeError("El límite no puede superar 1000 predicciones")
    return number


def _common_options(parser: argparse.ArgumentParser, *, inherited: bool = False) -> None:
    parser.add_argument(
        "--config",
        dest="online_config",
        type=Path,
        default=argparse.SUPPRESS if inherited else Path("config/online.toml"),
        help="Protocolo TOML online (independiente de config/default.toml)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=argparse.SUPPRESS if inherited else None,
        help="Base SQLite; por defecto online.state_path del protocolo; usar otra para replay",
    )


def add_commands(commands: Any) -> None:
    online = commands.add_parser(
        "online", help="Aprendizaje en línea y seguimiento de predicciones sin órdenes"
    )
    _common_options(online)
    actions = online.add_subparsers(dest="online_command", required=True)
    descriptions = {
        "init": "Crear el estado sin descargar datos ni entrenar con historia",
        "run": "Procesar nuevas velas cerradas y aprender de resultados confirmados",
        "status": "Consultar modelos, pendientes y estado recuperable",
        "predictions": "Consultar probabilidades y resultados registrados sin modificar el estado",
        "report": "Comparar modelos sobre resultados confirmados comunes",
        "verify": "Comprobar la integridad del estado persistido",
        "backup": "Crear una copia coherente del estado SQLite",
        "restore": "Restaurar una copia en una ubicación nueva",
        "catalog": "Consultar el catálogo local de modelos históricos y estrategias",
        "infer": "Inferencia causal de solo lectura con un modelo histórico explícito",
    }
    for name, description in descriptions.items():
        action = actions.add_parser(name, help=description)
        _common_options(action, inherited=True)
        if name == "run":
            mode = action.add_mutually_exclusive_group()
            mode.add_argument("--once", action="store_true", help="Un ciclo y terminar")
            mode.add_argument("--cycles", type=_positive_int, help="Cantidad máxima de ciclos")
            action.add_argument("--source", type=Path, help="CSV OHLCV local explícito")
            action.add_argument(
                "--replay",
                action="store_true",
                help="Simulación histórica aislada; requiere --source",
            )
            action.add_argument(
                "--now", help="Reloj UTC simulado; disponible solamente con --replay"
            )
            action.add_argument(
                "--poll-seconds", type=_positive_seconds, help="Espera entre ciclos; reemplaza TOML"
            )
        elif name == "predictions":
            action.add_argument("--limit", type=_prediction_limit, default=20)
            action.add_argument("--status", choices=["pending", "evaluated", "invalid_gap"])
            action.add_argument("--model-id", help="Filtrar un modelo incremental por ID")
        elif name == "backup":
            action.add_argument("--output", type=Path, required=True)
        elif name == "restore":
            action.add_argument("--backup", dest="backup_path", type=Path, required=True)
        elif name in {"catalog", "infer"}:
            action.add_argument("--root", type=Path, default=Path("artifacts"))
            action.add_argument("--symbol", default=None if name == "catalog" else "BTCUSDT")
            action.add_argument("--interval", default=None if name == "catalog" else "1d")
            if name == "catalog":
                action.add_argument("--include-unavailable", action="store_true")
            else:
                action.add_argument("--model-id", action="append", required=True)
                action.add_argument("--source", type=Path, required=True)
                action.add_argument("--peer-source", type=Path)
                action.add_argument("--reference", help="Cierre UTC que se desea analizar")


def _emit(payload: dict[str, Any]) -> None:
    # One complete JSON object per line is directly consumable by PowerShell pipelines.
    # JSON escapes also preserve Unicode across Windows console/pipe codepages.
    print(json.dumps(payload, ensure_ascii=True, allow_nan=False), flush=True)


def _recoverable(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (requests.RequestException, ConnectionError, TimeoutError)):
            return True
        if isinstance(current, sqlite3.OperationalError):
            if any(word in str(current).lower() for word in ("locked", "busy")):
                return True
        current = current.__cause__
    return False


def _poll_seconds(args: argparse.Namespace) -> float:
    if args.poll_seconds is not None:
        return args.poll_seconds
    with args.online_config.open("rb") as stream:
        protocol = tomllib.load(stream)
    value = protocol.get("runtime", {}).get(
        "poll_seconds", protocol.get("online", {}).get("poll_seconds", 30.0)
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("poll_seconds debe ser un número")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("poll_seconds debe ser finito y mayor que cero")
    return float(value)


def dispatch(args: argparse.Namespace) -> int:
    from trading_ml.online import service

    try:
        if args.online_command in {"catalog", "infer"}:
            return _historical_snapshot(args)
        if args.state is None:
            with args.online_config.open("rb") as stream:
                protocol = tomllib.load(stream)
            args.state = Path(
                protocol.get("online", {}).get("state_path", "artifacts/online/state.sqlite3")
            )
        if args.online_command == "init":
            _emit(service.init_state(args.online_config, args.state))
        elif args.online_command == "status":
            _emit(service.status(args.state))
        elif args.online_command == "predictions":
            _emit(
                service.predictions(
                    args.state,
                    limit=args.limit,
                    status_filter=args.status,
                    model_id=args.model_id,
                )
            )
        elif args.online_command == "report":
            _emit(service.report(args.state))
        elif args.online_command == "verify":
            result = service.verify(args.state)
            _emit(result)
            if result.get("valid") is False or result.get("ok") is False:
                return 1
        elif args.online_command == "backup":
            _emit(service.backup(args.state, args.output))
        elif args.online_command == "restore":
            _emit(service.restore(args.backup_path, args.state))
        else:
            return _run(args, service)
        return 0
    except KeyboardInterrupt:
        _emit({"event": "stopped", "reason": "keyboard_interrupt"})
        return 0
    except (TradingMLError, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        LOGGER.error("%s", exc, exc_info=args.verbose)
        _emit({"event": "error", "error": str(exc), "recoverable": False})
        return 1


def _run(args: argparse.Namespace, service: Any) -> int:
    if args.replay and args.source is None:
        raise ValueError("--replay requiere --source con un CSV explícito")
    if args.now is not None and not args.replay:
        raise ValueError("--now requiere --replay; el seguimiento live usa el reloj real")
    poll = _poll_seconds(args)
    limit = 1 if args.once else args.cycles
    attempts = failures = 0
    while limit is None or attempts < limit:
        attempts += 1
        try:
            result = service.run_once(
                args.online_config,
                args.state,
                source=args.source,
                replay=args.replay,
                now=args.now,
            )
            _emit(result)
            failures = 0
        except Exception as exc:
            retry = _recoverable(exc)
            failures += 1
            LOGGER.error("Ciclo online interrumpido: %s", exc, exc_info=args.verbose)
            _emit(
                {
                    "event": "cycle_error",
                    "error": str(exc),
                    "recoverable": retry,
                    "attempt": attempts,
                }
            )
            if not retry or args.once:
                return 1
        if limit is None or attempts < limit:
            time.sleep(min(300.0, poll * (2 ** min(failures, 5))))
    return 1 if failures else 0


def _historical_snapshot(args: argparse.Namespace) -> int:
    from trading_ml.dashboard.models_catalog import discover_all_models

    models = discover_all_models(
        args.root, include_non_deployable=getattr(args, "include_unavailable", False)
    )
    selected = [
        model
        for model in models
        if (args.symbol is None or model.symbol.upper() == args.symbol.upper())
        and (args.interval is None or model.interval == args.interval)
    ]
    if args.online_command == "catalog":
        columns = (
            "model_id",
            "display_name",
            "version",
            "architecture",
            "category",
            "stage",
            "source_kind",
            "availability_status",
            "is_deployable",
            "unavailable_reason",
            "symbol",
            "interval",
            "horizon",
            "output_type",
            "selection_label_end",
            "known_limitations",
            "artifact_path",
            "artifact_sha256",
        )
        _emit(
            {
                "event": "catalog",
                "models": [{key: getattr(m, key) for key in columns} for m in selected],
            }
        )
        return 0

    import pandas as pd

    from trading_ml.dashboard.inference_engine import run_batch_inference

    wanted = set(args.model_id)
    selected = [model for model in selected if model.model_id in wanted]
    missing = wanted - {model.model_id for model in selected}
    if missing:
        raise ValueError(f"Modelos no disponibles para el activo/intervalo: {sorted(missing)}")
    candles = pd.read_csv(args.source)
    peer = pd.read_csv(args.peer_source) if args.peer_source else None
    outcomes = run_batch_inference(
        selected,
        candles,
        peer_closed_candles=peer,
        reference_ts=pd.Timestamp(args.reference) if args.reference else None,
        use_cache=False,
    )
    _emit(
        {
            "event": "historical_inference_snapshot",
            "evidence": "read_only_snapshot_not_prospective_validation",
            "results": [
                {
                    "model_id": outcome.descriptor.model_id,
                    "result": asdict(outcome.result) if outcome.result is not None else None,
                    "error": outcome.error,
                }
                for outcome in outcomes
            ],
        }
    )
    return 0 if all(outcome.succeeded for outcome in outcomes) else 1
