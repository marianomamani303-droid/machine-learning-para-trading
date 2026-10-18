from __future__ import annotations

import logging
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from trading_ml.common import (
    code_provenance,
    read_json,
    sha256,
    sidecar,
    write_new_json,
)
from trading_ml.dmn.data import FEATURES, temporal_split
from trading_ml.dmn.portfolio import metrics, simulate, target_weights
from trading_ml.dmn.report import calendar_curve, paired_intervals
from trading_ml.dmn.study import (
    _environment,
    _fit_or_recover,
    _source_archive,
    _verify_files,
    _write_once,
    _write_text_once,
    protected_files,
)
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.expansion.data import load_snapshot
from trading_ml.expansion.features import (
    DERIVATIVES,
    FLOW,
    attach_context,
    crypto_features,
    make_panel,
)
from trading_ml.multimarket.data import load_snapshot as load_fx
from trading_ml.training.recipe import protect_reserved_labels

LOG = logging.getLogger(__name__)
SCENARIOS = (
    "anchor_price",
    "expanded_price",
    "expanded_flow",
    "expanded_derivatives",
    "expanded_context",
    "expanded_noise",
)


def load_protocol(path: Path) -> dict:
    with path.open("rb") as stream:
        p = tomllib.load(stream)
    r, d, t = p["research"], p["data"], p["training"]
    dates = pd.to_datetime(r["boundaries"], utc=True)
    if (
        p["schema"] != 1
        or len(dates) < 2
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or dates[-1] != pd.to_datetime(d["end"], utc=True)
        or dates[-1] > pd.Timestamp("2026-10-01", tz="UTC")
        or dates[-1] > pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
        or not set(d["anchors"]) <= set(d["symbols"])
        or len(set(d["symbols"])) != len(d["symbols"])
        or not 1 <= t["epochs"] <= 100
        or not 1 <= t["patience"] <= 30
        or t["embargo_days"] < 1
        or not 2 <= t["sequence_length"] <= 252
        or r["architectures"] != ["lstm", "xgboost"]
        or len(set(r["seeds"])) != len(r["seeds"])
        or r["maximum_fits"] != len(SCENARIOS) * 2 * len(r["seeds"]) * (len(dates) - 1)
    ):
        raise DataError("Protocolo de expansión inválido")
    return p


def forex_sources(root: Path, symbols: list[str]) -> dict:
    sources = {}
    for symbol in symbols:
        candidates = []
        for path in (root / "multimarket" / "data" / symbol).glob("1d_*.csv.json"):
            meta = read_json(path)
            if (
                meta["instrument"]["provider"] == "fxcm"
                and meta.get("start", "")[:10] == "2020-01-01"
                and meta.get("end_exclusive", "")[:10] == "2025-07-01"
            ):
                candidates.append(path.with_suffix(""))
        if not candidates:
            raise DataError(f"Falta histórico FXCM auditado de {symbol}")
        sources[symbol] = sorted(candidates)[0]
    return sources


def prepare(root: Path, p: dict, index: dict) -> tuple[dict, dict]:
    if index["start"] != p["data"]["start"] or index["end"] != p["data"]["end"]:
        raise DataError("Índice de datos no coincide con el protocolo")
    frames, files = {}, {}
    for symbol in p["data"]["symbols"]:
        sources = index["assets"][symbol]
        data = {}
        for kind in ["spot", "perp", "funding"]:
            path = Path(sources[kind])
            data[kind], _ = load_snapshot(path)
            files[str(path)] = sha256(path)
            files[str(sidecar(path))] = sha256(sidecar(path))
        frames[symbol] = crypto_features(data["spot"], data["perp"], data["funding"])
    forex = {}
    for symbol, path in forex_sources(root, p["data"]["forex"]).items():
        forex[symbol], _ = load_fx(path)
        files[str(path.resolve())] = sha256(path)
        files[str(sidecar(path).resolve())] = sha256(sidecar(path))
    frames, context = attach_context(frames, forex)
    full = (*FEATURES, *FLOW, *DERIVATIVES, *context)
    valid = np.logical_and.reduce(
        [np.isfinite(f[list(full)]).all(axis=1).to_numpy() for f in frames.values()]
    )
    if not valid.any():
        raise DataError("No hay datos con todas las familias después de warmup")
    first = int(np.flatnonzero(valid)[0])
    if not valid[first:].all():
        raise DataError("Huecos internos de features; no seleccionar fechas por resultados")
    frames = {s: f.iloc[first:].reset_index(drop=True) for s, f in frames.items()}
    for frame in frames.values():
        protect_reserved_labels(root, frame)
    panels = {}
    for scenario in SCENARIOS:
        selected = {
            s: f
            for s, f in frames.items()
            if scenario != "anchor_price" or s in p["data"]["anchors"]
        }
        cols = FEATURES
        if scenario in {"expanded_flow", "expanded_derivatives", "expanded_context"}:
            cols = (*cols, *FLOW)
        if scenario in {"expanded_derivatives", "expanded_context"}:
            cols = (*cols, *DERIVATIVES)
        if scenario == "expanded_context":
            cols = (*cols, *context)
        panels[scenario] = make_panel(
            selected,
            cols,
            p["training"]["sequence_length"],
            noise=scenario == "expanded_noise",
            noise_dimensions=len(full) - len(FEATURES),
        )
    return panels, files


def run_study(
    settings, protocol_path: Path, dataset_index: Path, *, resume: Path | None = None
) -> dict:
    p = load_protocol(protocol_path)
    panels, inputs = prepare(settings.root, p, read_json(dataset_index))
    inputs[str(dataset_index.resolve())] = sha256(dataset_index)
    code, env = code_provenance(), _environment()
    if resume:
        directory = resume.resolve()
        plan = read_json(directory / "plan.json")
        if (
            plan["protocol"] != p
            or plan["code"]["source_sha256"] != code["source_sha256"]
            or plan["environment"] != env
            or plan["inputs"] != inputs
        ):
            raise DataError("Reanudación requiere código, entorno, protocolo y fuentes congelados")
        _verify_files(plan["protected"])
        if (directory / "artifacts.json").exists():
            _verify_files(
                {str(directory / k): v for k, v in read_json(directory / "artifacts.json").items()}
            )
            return read_json(directory / "summary.json")
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        directory = (settings.root / "expansion" / "studies" / run_id).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        protected = protected_files(settings.root)
        for path in (settings.root / "dmn_studies").glob("*/artifacts.json"):
            protected[str(path.resolve())] = sha256(path)
        plan = {
            "run_id": run_id,
            "protocol": p,
            "inputs": inputs,
            "protected": protected,
            "code": code,
            "environment": env,
            "historical_development_only": True,
            "independent_advantage_confirmed": False,
        }
        write_new_json(directory / "plan.json", plan)
        _source_archive(directory, protocol_path)
    LOG.info("EXPANSION run=%s directory=%s", plan["run_id"], directory)
    curves, stresses, folds, fits = {}, {}, [], 0
    dates = pd.to_datetime(p["research"]["boundaries"], utc=True)
    for fold, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True)):
        LOG.info("fold %s %s..%s", fold, start.date(), end.date())
        fold_dir = directory / f"fold_{fold:02d}"
        fold_dir.mkdir(exist_ok=True)
        records, predictive = {}, {}
        for scenario, panel in panels.items():
            partitions = temporal_split(panel, start, end, p["training"])
            _, _, test = partitions
            anchor_ids = [panel.symbols.index(s) for s in p["data"]["anchors"]]
            for architecture in p["research"]["architectures"]:
                key = f"{scenario}__{architecture}"
                recipe = {
                    "id": key,
                    "architecture": architecture,
                    "objective": "binary" if architecture == "xgboost" else "sharpe",
                    "net_training": architecture == "lstm",
                }
                predictions = []
                probabilities = []
                for seed in p["research"]["seeds"]:
                    fits += 1
                    if fits > p["research"]["maximum_fits"]:
                        raise DataError("Presupuesto agotado")
                    model_dir = fold_dir / f"{key}__seed_{seed}"
                    signal, meta = _fit_or_recover(
                        panel, partitions, recipe, p, seed, model_dir, recovering=resume is not None
                    )
                    predictions.append(signal)
                    if architecture == "xgboost":
                        from xgboost import XGBClassifier

                        model = XGBClassifier()
                        model.load_model(model_dir / "model.ubj")
                        prob = model.predict_proba(
                            panel.x[test, :, -1, :].reshape(-1, len(panel.feature_names))
                        )[:, 1].reshape(len(test), -1)
                        probabilities.append(prob)
                    records[f"{key}__seed_{seed}"] = {
                        "selected_epoch": meta.get("selected_epoch"),
                        "features": len(panel.feature_names),
                    }
                    _evaluate(
                        curves,
                        stresses,
                        fold_dir,
                        fold,
                        f"{key}__seed_{seed}",
                        panel,
                        test,
                        signal,
                        anchor_ids,
                        p,
                    )
                signal = np.mean(predictions, axis=0)
                _evaluate(curves, stresses, fold_dir, fold, key, panel, test, signal, anchor_ids, p)
                if probabilities:
                    probability = np.mean(probabilities, axis=0)[:, anchor_ids]
                    y = (panel.returns[test][:, anchor_ids] > 0).astype(int)
                    predictive[key] = classification_metrics(y.ravel(), probability.ravel())
                    pred = pd.DataFrame(
                        {
                            "reference_time": np.repeat(panel.times[test], len(anchor_ids)),
                            "symbol": np.tile(p["data"]["anchors"], len(test)),
                            "target": y.ravel(),
                            "probability": probability.ravel(),
                        }
                    )
                    _write_text_once(
                        fold_dir / f"{key}_probabilities.csv", pred.to_csv(index=False)
                    )
        baseline = panels["anchor_price"]
        _, _, test = temporal_split(baseline, start, end, p["training"])
        for name, signal in baseline.baselines.items():
            _evaluate(
                curves,
                stresses,
                fold_dir,
                fold,
                name,
                baseline,
                test,
                signal[test],
                list(range(len(baseline.symbols))),
                p,
            )
        record = {
            "fold": fold,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "fits": records,
            "predictive": predictive,
        }
        _write_once(fold_dir / "report.json", record)
        folds.append(record)
    combined = {k: calendar_curve(pd.concat(v, ignore_index=True)) for k, v in curves.items()}
    double = {k: calendar_curve(pd.concat(v, ignore_index=True)) for k, v in stresses.items()}
    models = {
        k: {"net": metrics(v), "double_cost": metrics(double[k])} for k, v in combined.items()
    }
    pairs = {}
    for architecture in p["research"]["architectures"]:
        base = f"anchor_price__{architecture}__anchor"
        candidates = [f"{s}__{architecture}__anchor" for s in SCENARIOS if s != "anchor_price"]
        # Reuse synchronous date-block bootstrap with a named strategy as reference.
        pairs[architecture] = paired_intervals(
            {"tsmom": combined[base], **{k: combined[k] for k in candidates}},
            candidates,
            p["bootstrap"],
        )
        pairs[architecture]["reference"] = base
    _verify_files(plan["protected"])
    summary = {
        "run_id": plan["run_id"],
        "directory": str(directory),
        "actual_fits": fits,
        "models": models,
        "folds": folds,
        "paired_uncertainty": pairs,
        "historical_development_only": True,
        "independent_advantage_confirmed": False,
        "promoted": False,
        "protected_artifacts_unchanged": True,
    }
    for name, curve in combined.items():
        _write_text_once(directory / f"{name}_equity.csv", curve.to_csv(index=False))
    _write_once(directory / "summary.json", summary)
    write_new_json(
        directory / "artifacts.json",
        {
            str(f.relative_to(directory)): sha256(f)
            for f in sorted(directory.rglob("*"))
            if f.is_file()
        },
    )
    return summary


def _evaluate(curves, stresses, directory, fold, key, panel, test, signal, anchor_ids, p):
    for scope, ids in [("anchor", anchor_ids), ("universe", list(range(len(panel.symbols))))]:
        name = f"{key}__{scope}"
        selected = signal[:, ids]
        returns = panel.returns[test][:, ids]
        vol = panel.volatility[test][:, ids]
        w = target_weights(selected, vol, p["risk"])
        curve = simulate(w, returns, panel.times[test], p["costs"])
        stress = simulate(w, returns, panel.times[test], p["costs"], multiplier=2)
        curve["fold"], stress["fold"] = fold, fold
        curves.setdefault(name, []).append(curve)
        stresses.setdefault(name, []).append(stress)
        for j, i in enumerate(ids):
            curve[f"weight_{panel.symbols[i]}"] = w[:, j]
            curve[f"return_{panel.symbols[i]}"] = returns[:, j]
        _write_text_once(directory / f"{name}_positions.csv", curve.to_csv(index=False))


def verify(directory: Path) -> dict:
    manifest = read_json(directory / "artifacts.json")
    if any(not (directory / k).resolve().is_relative_to(directory.resolve()) for k in manifest):
        raise DataError("Ruta externa al estudio")
    _verify_files({str(directory / k): v for k, v in manifest.items()})
    plan = read_json(directory / "plan.json")
    _verify_files(plan["inputs"])
    _verify_files(plan["protected"])
    count = 0
    assembled = {"net": {}, "double_cost": {}}
    for path in directory.glob("fold_*/*_positions.csv"):
        f = pd.read_csv(path, float_precision="round_trip")
        symbols = [k.removeprefix("weight_") for k in f if k.startswith("weight_")]
        w = f[[f"weight_{s}" for s in symbols]].to_numpy()
        r = f[[f"return_{s}" for s in symbols]].to_numpy()
        actual = simulate(
            w,
            r,
            pd.DatetimeIndex(pd.to_datetime(f.reference_time, utc=True)),
            plan["protocol"]["costs"],
        )
        if not np.allclose(actual.net_return, f.net_return, rtol=1e-10, atol=1e-12):
            raise DataError("Retornos de expansión no reproducidos")
        name = path.stem.removesuffix("_positions")
        actual["fold"] = int(path.parent.name.removeprefix("fold_"))
        assembled["net"].setdefault(name, []).append(actual)
        stressed = simulate(
            w,
            r,
            pd.DatetimeIndex(pd.to_datetime(f.reference_time, utc=True)),
            plan["protocol"]["costs"],
            multiplier=2,
        )
        stressed["fold"] = actual["fold"].iloc[0]
        assembled["double_cost"].setdefault(name, []).append(stressed)
        count += 1
    summary = read_json(directory / "summary.json")
    for mode, strategies in assembled.items():
        for name, parts in strategies.items():
            recomputed = metrics(calendar_curve(pd.concat(parts, ignore_index=True)))
            expected = summary["models"][name][mode]
            for key, value in recomputed.items():
                if isinstance(value, (int, float)):
                    if expected[key] is None or not np.isclose(
                        value, expected[key], rtol=1e-10, atol=1e-12
                    ):
                        raise DataError(f"Métrica agregada no reproducida: {name}/{mode}/{key}")
                elif value != expected[key]:
                    raise DataError("Métrica indefinida o contrato no reproducido")
    return {
        "intact": True,
        "verified_files": len(manifest),
        "reconstructed_portfolios": count,
        "reconstructed_cost_scenarios": count * 2,
    }
