"""Consolidate explicit acquisition inventories without touching old experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, sidecar, write_new_json
from trading_ml.data.history import DEVELOPMENT_END


def summarize(inventories: list[Path], output: Path, prefer_longest=False):
    entries = {}
    failures = []
    for inventory in inventories:
        value = read_json(inventory)
        failures.extend(value.get("failures", []))
        for entry in value["datasets"]:
            key = entry["provider"], entry["symbol"], entry["interval"]
            if key in entries and entries[key]["sha256"] != entry["sha256"]:
                old = entries[key]
                if not prefer_longest:
                    raise ValueError("Inventarios con versiones diferentes; elegir explícitamente")
                candidate, shorter = (
                    (entry, old)
                    if entry["first_open"] <= old["first_open"]
                    and entry["last_close"] >= old["last_close"]
                    else (old, entry)
                )
                if entry["market_type"] == "cash_index_context":
                    new_policy = read_json(sidecar(Path(entry["path"]))).get("ohlc_policy")
                    old_policy = read_json(sidecar(Path(old["path"]))).get("ohlc_policy")
                    if new_policy == "observed_nonzero_daily_range" and old_policy != new_policy:
                        candidate, shorter = entry, old
                    elif old_policy == "observed_nonzero_daily_range" and new_policy != old_policy:
                        candidate, shorter = old, entry
                complete = pd.read_csv(candidate["path"], float_precision="round_trip")
                subset = pd.read_csv(shorter["path"], float_precision="round_trip")
                if shorter["market_type"] == "cash_index_context":
                    subset = subset.loc[subset.high > subset.low]
                overlap = subset.merge(
                    complete, on="timestamp", suffixes=("_old", "_new"), validate="one_to_one"
                )
                if len(overlap) != len(subset) or any(
                    not np.allclose(
                        overlap[col + "_old"], overlap[col + "_new"], rtol=1e-12, atol=1e-10
                    )
                    for col in ["open", "high", "low", "close"]
                ):
                    raise ValueError(
                        "Las versiones no son extensiones consistentes: revisar la fuente"
                    )
                entry = candidate
            entries[key] = entry
    groups = defaultdict(list)
    unique_rows = development = quarantined = 0
    exclusions = defaultdict(int)
    cutoff = pd.Timestamp(DEVELOPMENT_END, tz="UTC")
    for entry in entries.values():
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Dataset modificado")
        meta = read_json(sidecar(path))
        clocks = pd.to_datetime(pd.read_csv(path, usecols=["available_at"]).available_at, utc=True)
        dev = int((clocks <= cutoff).sum())
        entry = {
            **entry,
            "manifest_sha256": sha256(sidecar(path)),
            "development_rows_before_label_purge": dev,
            "quarantined_rows": len(clocks) - dev,
        }
        groups[entry["market_type"]].append(entry)
        development += dev
        quarantined += len(clocks) - dev
        if entry["interval"] == "15m" or entry["market_type"] in {
            "cash_index_context",
            "exchange_dated_futures",
        }:
            unique_rows += entry["rows"]
            exclusions["binance_partial_or_invalid_observations"] += meta.get(
                "excluded_invalid_observations", 0
            )
            exclusions["index_invalid_ohlc_observations"] += len(
                meta.get("excluded_invalid_ohlc_dates", [])
            )
            exclusions["cboe_nontraded_or_invalid_observations"] += len(
                meta.get("source_quality", {}).get("excluded_nontraded_or_invalid_ohlc_dates", [])
            )
    summary = []
    datasets = []
    for market, rows in sorted(groups.items()):
        datasets.extend(rows)
        summary.append(
            {
                "market": market,
                "instruments_or_individual_contracts": len(
                    {(r["provider"], r["symbol"]) for r in rows}
                ),
                "datasets": len(rows),
                "rows": sum(r["rows"] for r in rows),
                "first_open": min(r["first_open"] for r in rows),
                "last_close": max(r["last_close"] for r in rows),
                "intervals": sorted({r["interval"] for r in rows}),
                "development_rows_before_label_purge": sum(
                    r["development_rows_before_label_purge"] for r in rows
                ),
                "quarantined_rows": sum(r["quarantined_rows"] for r in rows),
            }
        )
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "inventories": {str(p.resolve()): sha256(p) for p in inventories},
        "datasets": datasets,
        "summary": summary,
        "failures": failures,
        "rows_including_derived_timeframes": development + quarantined,
        "base_observations_excluding_derived_timeframes": unique_rows,
        "development_rows_before_label_purge": development,
        "quarantined_rows": quarantined,
        "development_end": DEVELOPMENT_END,
        "exclusions": dict(exclusions),
        "models_trained": 0,
        "independent_evaluation_performed": False,
        "note": "Derived timeframes reuse source observations. Splits must be chronological and purged; recent data remain quarantined. Individual futures are not continuous tradable series.",
    }
    write_new_json(output.with_suffix(".json"), result)
    lines = [
        "# Resultado de ampliación histórica (2026-10-17)",
        "",
        "Adquisición y validación únicamente. **Ningún modelo entrenado.**",
        "",
        "| Mercado | Activos/contratos | Datasets | Registros | Primer dato UTC | Último cierre UTC |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['market']} | {row['instruments_or_individual_contracts']} | {row['datasets']} | {row['rows']:,} | {row['first_open'][:10]} | {row['last_close'][:10]} |"
        )
    lines += [
        "",
        f"Total: **{development + quarantined:,} registros**, con **{unique_rows:,} observaciones base** sin contar de nuevo las temporalidades derivadas.",
        f"Desarrollo hasta {DEVELOPMENT_END}: {development:,} filas antes de purgar etiquetas. Cuarentena posterior: {quarantined:,} filas.",
        "",
        f"Trabajos no adquiridos: {len(failures)}. Exclusiones: {dict(exclusions)}.",
        "",
        "Forex: ventanas UTC completas, bid/ask indicativo sin volumen global ni swaps. Índices/Cboe sólo diarios. Perpetuos requieren funding y futuros fechados un protocolo de margen/roll. No hay ES/NQ/GC/CL licenciados. Huecos y sesiones parciales no se rellenan. Los CSV/manifiestos identifican la cobertura efectiva de cada activo.",
        "",
        "Comandos, adaptadores y controles: [HISTORICAL_DATA.md](HISTORICAL_DATA.md).",
    ]
    with output.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print({k: v for k, v in result.items() if k not in {"datasets", "inventories"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Base name without extension")
    parser.add_argument(
        "--prefer-longest",
        action="store_true",
        help="Elegir cobertura mayor sólo si reproduce todo el tramo común",
    )
    args = parser.parse_args()
    summarize(args.inventory, args.output, args.prefer_longest)
