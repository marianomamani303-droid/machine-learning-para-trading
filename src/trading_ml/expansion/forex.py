from __future__ import annotations

import logging
import tomllib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from trading_ml.common import code_provenance, read_json, sha256, write_new_json
from trading_ml.dmn.study import (
    _environment,
    _source_archive,
    _verify_files,
    _write_once,
    _write_text_once,
    protected_files,
)
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.expansion.study import forex_sources
from trading_ml.multimarket.data import execution_source, load_snapshot
from trading_ml.multimarket.execution import Costs, Engine
from trading_ml.multimarket.features import features, recipes
from trading_ml.multimarket.study import _fit, labelled, require_outcomes
from trading_ml.training.recipe import protect_reserved_labels

LOG = logging.getLogger(__name__)


def run_forex(settings, protocol_path: Path, *, resume: Path | None = None) -> dict:
    with protocol_path.open("rb") as stream:
        p = tomllib.load(stream)
    bounds = pd.to_datetime(p["boundaries"], utc=True)
    if (
        p["symbols"] != ["USDCAD", "NZDUSD"]
        or p["maximum_fits"] != 24
        or len(bounds) != 4
        or bounds.has_duplicates
        or not bounds.is_monotonic_increasing
        or bounds[-1] > pd.Timestamp("2026-10-01", tz="UTC")
    ):
        raise DataError("Protocolo Forex fuera del estudio fijo")
    costs = Costs(**p["costs"])
    costs.validate()
    sources = forex_sources(settings.root, p["symbols"])
    files = {str(path.resolve()): sha256(path) for path in sources.values()}
    code = code_provenance()
    if resume:
        directory = resume.resolve()
        plan = read_json(directory / "plan.json")
        if (
            plan["protocol"] != p
            or plan["sources"] != files
            or plan["code"]["source_sha256"] != code["source_sha256"]
            or plan["environment"] != _environment()
        ):
            raise DataError("Fuente, código, protocolo o entorno de Forex cambiaron")
        _verify_files(plan["protected"])
        if (directory / "artifacts.json").exists():
            _verify_files(
                {str(directory / k): v for k, v in read_json(directory / "artifacts.json").items()}
            )
            return read_json(directory / "summary.json")
    else:
        ident = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        directory = (settings.root / "expansion" / "forex_studies" / ident).resolve()
        directory.mkdir(parents=True)
        plan = {
            "protocol": p,
            "sources": files,
            "code": code,
            "environment": _environment(),
            "protected": protected_files(settings.root),
            "historical_development_only": True,
            "automatic_promotion": False,
        }
        write_new_json(directory / "plan.json", plan)
        _source_archive(directory, protocol_path)
    LOG.info("Forex expansion directory=%s", directory)
    results = []
    fits = 0
    for symbol, path in sources.items():
        raw, meta = load_snapshot(path)
        frame = features(raw, symbol)
        execution, multiplier = execution_source(raw, meta)
        engine = Engine(
            frame, symbol, costs, execution_frame=execution, bars_per_horizon=multiplier
        )
        data = labelled(frame, engine, 1, "direction", False)
        columns = recipes(symbol)
        common = columns["compact"]
        for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            valstart = start - pd.Timedelta(days=p["embargo_days"] + p["calibration_days"])
            train = data.loc[
                data.target_time < valstart - pd.Timedelta(days=p["embargo_days"])
            ].dropna(subset=common + ["target"])
            val = data.loc[
                (data.reference_time >= valstart)
                & (data.target_time < start - pd.Timedelta(days=p["embargo_days"]))
            ].dropna(subset=common + ["target"])
            out = data.loc[
                (data.reference_time >= start) & (data.reference_time < end - pd.Timedelta(days=20))
            ].dropna(subset=common)
            require_outcomes(out, engine, "direction")
            if (
                len(train) < p["min_train_rows"]
                or len(val) < 20
                or len(out) < 40
                or out.target_time.max() >= end
            ):
                raise DataError("Cobertura/particiones de Forex insuficientes")
            protect_reserved_labels(settings.root, train, val, out)
            for recipe in ["price", "compact"]:
                for seed in p["seeds"]:
                    fits += 1
                    destination = directory / f"{symbol}_{recipe}_fold{fold}_seed{seed}"
                    destination.mkdir(exist_ok=True)
                    record = destination / "report.json"
                    if record.exists():
                        r = read_json(record)
                        _verify_files({str(destination / k): v for k, v in r["files"].items()})
                        results.append(r)
                        continue
                    model = _fit(train, columns[recipe], {**p, "seed": seed})
                    model.save_model(destination / "model.ubj")
                    pred = out[["reference_time", "target_time", "target", "source_index"]].copy()
                    pred["probability"] = model.predict_proba(out[columns[recipe]])[:, 1]
                    validation = classification_metrics(
                        val.target.astype(int).to_numpy(),
                        model.predict_proba(val[columns[recipe]])[:, 1],
                    )
                    observed = classification_metrics(
                        pred.target.astype(int).to_numpy(), pred.probability.to_numpy()
                    )
                    prevalence = np.full(len(pred), train.target.mean())
                    baseline = classification_metrics(
                        pred.target.astype(int).to_numpy(), prevalence
                    )
                    net, trades, curve = engine.portfolio(pred, 1, False, p["threshold"], end)
                    stress_costs = replace(
                        costs,
                        forex_commission_bps=2 * costs.forex_commission_bps,
                        forex_slippage_pips=2 * costs.forex_slippage_pips,
                        forex_swap_annual_bps=2 * costs.forex_swap_annual_bps,
                    )
                    stressed = Engine(
                        frame,
                        symbol,
                        stress_costs,
                        execution_frame=execution,
                        bars_per_horizon=multiplier,
                    )
                    stress, _, _ = stressed.portfolio(pred, 1, False, p["threshold"], end)
                    _write_text_once(destination / "predictions.csv", pred.to_csv(index=False))
                    _write_text_once(destination / "trades.csv", trades.to_csv(index=False))
                    _write_text_once(destination / "equity.csv", curve.to_csv(index=False))
                    r = {
                        "symbol": symbol,
                        "recipe": recipe,
                        "fold": fold,
                        "seed": seed,
                        "metrics": observed,
                        "validation": validation,
                        "baseline_prevalence": baseline,
                        "net": net,
                        "stress": stress,
                        "columns": columns[recipe],
                        "fit_dates": len(train),
                        "train_last_target": train.target_time.max().isoformat(),
                        "validation_last_target": val.target_time.max().isoformat(),
                        "oos_first": out.reference_time.min().isoformat(),
                        "files": {f.name: sha256(f) for f in destination.iterdir() if f.is_file()},
                    }
                    _write_once(record, r)
                    results.append(r)
                    LOG.info(
                        "%s %s fold=%s seed=%s AUC=%s trades=%s",
                        symbol,
                        recipe,
                        fold,
                        seed,
                        observed["roc_auc"],
                        net["trades"],
                    )
    _verify_files(plan["protected"])
    report = {
        "directory": str(directory),
        "actual_fits": fits,
        "results": results,
        "independent_advantage_confirmed": False,
        "promoted": False,
        "historical_development_only": True,
        "protected_artifacts_unchanged": True,
        "limitations": "Indicative FXCM bid/ask; swaps are cost scenarios; D1 direction label differs from 24 H1 execution horizon",
    }
    _write_once(directory / "summary.json", report)
    write_new_json(
        directory / "artifacts.json",
        {str(f.relative_to(directory)): sha256(f) for f in directory.rglob("*") if f.is_file()},
    )
    return report


def verify_forex(directory: Path) -> dict:
    from xgboost import XGBClassifier

    manifest = read_json(directory / "artifacts.json")
    if any(not (directory / k).resolve().is_relative_to(directory.resolve()) for k in manifest):
        raise DataError("Ruta Forex externa al estudio")
    _verify_files({str(directory / k): v for k, v in manifest.items()})
    plan = read_json(directory / "plan.json")
    _verify_files(plan["sources"])
    _verify_files(plan["protected"])
    report = read_json(directory / "summary.json")
    protocol = plan["protocol"]
    by_symbol = {}
    for source in plan["sources"]:
        raw, meta = load_snapshot(Path(source))
        symbol = meta["instrument"]["symbol"]
        frame = features(raw, symbol)
        execution, multiplier = execution_source(raw, meta)
        engine = Engine(
            frame,
            symbol,
            Costs(**protocol["costs"]),
            execution_frame=execution,
            bars_per_horizon=multiplier,
        )
        by_symbol[symbol] = (frame, engine)
    for result in report["results"]:
        key = f"{result['symbol']}_{result['recipe']}_fold{result['fold']}_seed{result['seed']}"
        path = directory / key
        pred = pd.read_csv(path / "predictions.csv", float_precision="round_trip")
        pred["reference_time"] = pd.to_datetime(pred.reference_time, utc=True)
        pred["target_time"] = pd.to_datetime(pred.target_time, utc=True)
        pred["probability"] = pred.probability.astype(np.float32)
        frame, engine = by_symbol[result["symbol"]]
        source = frame.iloc[pred.source_index.to_numpy(int)]
        model = XGBClassifier()
        model.load_model(path / "model.ubj")
        probability = model.predict_proba(source[result["columns"]])[:, 1]
        if not np.allclose(probability, pred.probability, atol=1e-7, rtol=0):
            raise DataError("Predicciones Forex no reproducidas")
        labels = labelled(frame, engine, 1, "direction", False)
        if not np.array_equal(labels.iloc[pred.source_index.to_numpy(int)].target, pred.target):
            raise DataError("Etiquetas Forex no reproducidas")
        end = pd.to_datetime(protocol["boundaries"][result["fold"] + 1], utc=True)
        net, trades, curve = engine.portfolio(pred, 1, False, protocol["threshold"], end)
        saved_curve = pd.read_csv(path / "equity.csv", float_precision="round_trip")
        if not np.allclose(curve.equity, saved_curve.equity, rtol=1e-10, atol=1e-8):
            raise DataError("Equity Forex no reproducida")
        if len(trades) != result["net"]["trades"] or not np.isclose(
            net["net_return"], result["net"]["net_return"], atol=1e-12
        ):
            raise DataError("Operaciones Forex no reproducidas")
    return {
        "intact": True,
        "verified_files": len(manifest),
        "reproduced_models": len(report["results"]),
        "reconstructed_portfolios": len(report["results"]),
    }
