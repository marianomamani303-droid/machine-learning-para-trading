"""Data-only CLI; deliberately has no model/study execution path."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import object_hash, read_json, sha256, sidecar, write_new_json
from trading_ml.data.history import aggregate, export_forex, export_legacy, validate
from trading_ml.data.history_sources import acquire_binance, acquire_forex, acquire_index, listing
from trading_ml.errors import DataError

LOG = logging.getLogger(__name__)
ACQUISITION_LOGS = {
    "history_acquisition.log",
    "history_indices.log",
    "history_cboe.log",
    "history_fx_daily.log",
    "history_repair.log",
}


def verify_inventory(path: Path) -> dict:
    inventory = read_json(path)
    total = 0
    sources = {}
    grouped = defaultdict(dict)
    for entry in inventory["datasets"]:
        source = Path(entry["path"])
        if sha256(source) != entry["sha256"]:
            raise DataError(f"Dataset alterado: {source}")
        if entry.get("manifest_sha256") and sha256(sidecar(source)) != entry["manifest_sha256"]:
            raise DataError("Manifiesto histórico alterado")
        meta = read_json(sidecar(source))
        if meta["sha256"] != entry["sha256"]:
            raise DataError("Manifiesto alterado")
        frame = pd.read_csv(source, float_precision="round_trip")
        for col in ["timestamp", "available_at"]:
            frame[col] = pd.to_datetime(frame[col], utc=True)
        quality = validate(frame, entry["interval"])
        if meta["market_type"] == "crypto_dated_futures":
            delivery = pd.Timestamp("20" + meta["symbol"][-6:], tz="UTC") + pd.Timedelta(hours=8)
            if (frame.available_at > delivery).any():
                raise DataError("Velas posteriores a liquidación en un contrato fechado")
        if meta["market_type"] == "cash_index_context" and (frame.high <= frame.low).any():
            raise DataError("Índice con OHLC plano sin rango diario verificable")
        if quality != meta["quality"]:
            raise DataError("Calidad guardada no reproducida")
        for source in meta.get("sources", []):
            if source.get("payload"):
                sources[source["payload"]] = source["sha256"]
        grouped[entry["provider"], entry["symbol"]][entry["interval"]] = Path(entry["path"])
        total += len(frame)
    comparisons = 0
    for group in grouped.values():
        if "15m" not in group:
            continue
        original = pd.read_csv(group["15m"], float_precision="round_trip")
        original.timestamp = pd.to_datetime(original.timestamp, utc=True)
        original.available_at = pd.to_datetime(original.available_at, utc=True)
        for interval, source_path in group.items():
            if interval in {"15m", "1d_ny"}:
                continue
            expected, _ = aggregate(original, "15m", interval)
            observed = pd.read_csv(source_path, float_precision="round_trip")
            for column in ["timestamp", "available_at"]:
                observed[column] = pd.to_datetime(observed[column], utc=True)
            if not expected.timestamp.equals(
                observed.timestamp
            ) or not expected.available_at.equals(observed.available_at):
                raise DataError("Temporalidades inconsistentes en fechas")
            numeric = [c for c in expected if c not in {"timestamp", "available_at"}]
            if not np.allclose(expected[numeric], observed[numeric], rtol=1e-12, atol=1e-10):
                raise DataError("Temporalidades inconsistentes en OHLC/volumen")
            comparisons += 1
    for source_path, digest in sources.items():
        if sha256(Path(source_path)) != digest:
            raise DataError(f"Original del proveedor alterado: {source_path}")
    return {
        "datasets": len(inventory["datasets"]),
        "rows": total,
        "all_checksums_and_quality_verified": True,
        "models_trained": 0,
        "source_payloads_verified": len(sources),
        "timeframe_aggregations_reconstructed": comparisons,
    }


def acquire(config_path: Path, root: Path, only: str | None = None) -> Path:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    end = pd.Timestamp(config["data"]["end"], tz="UTC")
    if end > pd.Timestamp.now(tz="UTC").normalize().replace(day=1) or end.day != 1:
        raise DataError("Se requieren meses cerrados y fin exclusivo en primer día del mes")
    if not 1 <= config["data"]["workers"] <= 4:
        raise DataError("Máximo cuatro conexiones por proveedor")
    identity = object_hash(config)
    base = root / "historical" / "acquisitions" / identity[:20]
    base.mkdir(parents=True, exist_ok=True)
    plan = base / "plan.json"
    if not plan.exists():
        # Capture names/size/mtime of every existing artifact without reading test results.
        previous = {}
        for directory, folders, files in os.walk(root):
            folders[:] = [
                name for name in folders if name not in {"historical", "cache", "__pycache__"}
            ]
            for name in files:
                if Path(directory).resolve() == root.resolve() and name in ACQUISITION_LOGS:
                    continue
                path = Path(directory) / name
                stat = path.stat()
                previous[str(path.resolve())] = {
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        write_new_json(
            plan,
            {
                "protocol": config,
                "protocol_sha256": identity,
                "started_at": datetime.now(UTC).isoformat(),
                "previous_artifacts": previous,
                "models_trained": 0,
            },
        )

    def job(key, callback):
        receipt = base / "completed" / (key + ".json")
        if receipt.exists():
            rows = read_json(receipt)["datasets"]
            for row in rows:
                if sha256(Path(row["path"])) != row["sha256"]:
                    raise DataError("Dataset completado alterado")
            LOG.info("Reutilizar %s: %d datasets", key, len(rows))
            return rows
        LOG.info("Iniciar %s", key)
        rows = callback()
        write_new_json(receipt, {"datasets": rows, "completed_at": datetime.now(UTC).isoformat()})
        LOG.info(
            "Completado %s: %d datasets, %d registros", key, len(rows), sum(r["rows"] for r in rows)
        )
        return rows

    failures = []

    def attempt(key, callback):
        try:
            return job(key, callback)
        except Exception as exc:
            LOG.exception("No adquirido: %s", key)
            failure = {"job": key, "error": str(exc), "type": type(exc).__name__}
            failures.append(failure)
            write_new_json(
                base / "failures" / f"{key}_{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json", failure
            )
            return []

    def crypto():
        rows = []
        for symbol in config["crypto"]["spot"]:
            rows += attempt(
                "spot_" + symbol, lambda s=symbol: acquire_binance(root, s, "spot", config)
            )
        for symbol in config["crypto"]["perpetual"]:
            rows += attempt(
                "perp_" + symbol, lambda s=symbol: acquire_binance(root, s, "futures/um", config)
            )
        try:
            prefixes = listing(root, "data/futures/cm/monthly/klines/", delimiter="/")
            pattern = (
                "(?:"
                + "|".join(re.escape(s) for s in config["crypto"]["dated_futures"])
                + ")_[0-9]{6}"
            )
            symbols = [p.rstrip("/").split("/")[-1] for p in prefixes]
            symbols = [s for s in symbols if re.fullmatch(pattern, s)]
            for symbol in symbols:
                if pd.Timestamp("20" + symbol[-6:], tz="UTC") >= end:
                    continue
                rows += attempt(
                    "dated_" + symbol + "_delivery_bound",
                    lambda s=symbol: acquire_binance(root, s, "futures/cm", config),
                )
        except Exception as exc:
            failures.append({"job": "dated_listing", "error": str(exc)})
            LOG.exception("Listado futuros fallido")
        return rows

    def forex():
        rows = []
        for symbol in config["forex"]["symbols"]:
            rows += attempt("fx_" + symbol, lambda s=symbol: acquire_forex(root, s, config))
        return rows

    def indices():
        rows = []
        for symbol in config["indices"]["symbols"]:
            rows += attempt(
                "index_" + symbol.replace("^", "") + "_nonzero_range",
                lambda s=symbol: acquire_index(root, s, config),
            )
        return rows

    providers = {"crypto": crypto, "forex": forex, "indices": indices}
    datasets = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(callback)
            for name, callback in providers.items()
            if only is None or name == only
        ]
        for future in futures:
            datasets.extend(future.result())
    prior = read_json(plan)["previous_artifacts"]
    # Early plans may contain the mutable log created by that same acquisition.
    # Exempt only this task's logs, never models/checkpoints/results.
    own_logs = {str((root / name).resolve()) for name in ACQUISITION_LOGS}
    excluded_logs = sorted(set(prior) & own_logs)
    prior = {name: stat for name, stat in prior.items() if name not in own_logs}
    altered = [
        name
        for name, stat in prior.items()
        if not Path(name).exists()
        or Path(name).stat().st_size != stat["bytes"]
        or Path(name).stat().st_mtime_ns != stat["mtime_ns"]
    ]
    if altered:
        raise DataError(f"Artefactos anteriores cambiaron: {altered[:3]}")
    path = base / f"inventory_{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json"
    write_new_json(
        path,
        {
            "datasets": datasets,
            "failures": failures,
            "protocol": config,
            "protocol_sha256": identity,
            "models_trained": 0,
            "previous_artifacts_unchanged": True,
            "previous_artifacts_checked_by_size_and_mtime": len(prior),
            "mutable_acquisition_logs_excluded": excluded_logs,
            "rows_including_derived_timeframes": sum(r["rows"] for r in datasets),
            "unique_base_rows": sum(
                r["rows"]
                for r in datasets
                if r["interval"] == "15m" or r["market_type"] == "cash_index_context"
            ),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    LOG.info(
        "Inventario: %s; %d datasets; %d registros; %d fallos",
        path,
        len(datasets),
        sum(r["rows"] for r in datasets),
        len(failures),
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "verify", "export"])
    parser.add_argument("--config", type=Path, default=Path("config/historical.toml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--only", choices=["crypto", "forex", "indices"])
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument(
        "--forex", action="store_true", help="Exportar H1 bid/ask al adaptador multimercado"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    if args.command == "acquire":
        print(acquire(args.config, args.root, args.only))
    elif args.command == "verify":
        if args.inventory is None:
            parser.error("verify requiere --inventory")
        result = verify_inventory(args.inventory)
        if args.output:
            write_new_json(args.output, result)
        print(result)
    else:
        if any(value is None for value in [args.source, args.start, args.end]) or (
            not args.forex and args.output is None
        ):
            parser.error("export requiere --source, --output, --start y --end")
        if args.forex:
            print(export_forex(args.source, root=args.root, start=args.start, end=args.end))
        else:
            print(
                export_legacy(
                    args.source, args.output, root=args.root, start=args.start, end=args.end
                )
            )


if __name__ == "__main__":
    main()
