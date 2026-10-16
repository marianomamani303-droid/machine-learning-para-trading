from __future__ import annotations

import logging
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from trading_ml.common import (
    code_provenance,
    environment,
    object_hash,
    read_json,
    sha256,
    write_new_json,
)
from trading_ml.dmn.config import load_protocol
from trading_ml.dmn.data import build_panel, load_raw, temporal_split
from trading_ml.dmn.portfolio import metrics, simulate, target_weights
from trading_ml.dmn.report import calendar_curve, markdown, summarize
from trading_ml.errors import DataError
from trading_ml.training.recipe import protect_reserved_labels

LOG = logging.getLogger(__name__)


def protected_files(root: Path) -> dict:
    paths = list((root / "models").rglob("*"))
    for name in ["smc_studies", "momentum_studies", "mean_reversion_studies", "frontier_studies"]:
        for pattern in ["*/prospective.json", "*/plan.json", "latest.json"]:
            paths.extend((root / name).glob(pattern))
    return {str(p.resolve()): sha256(p) for p in paths if p.is_file()}


def _verify_files(manifest: dict) -> None:
    if any(not Path(p).is_file() or sha256(Path(p)) != digest for p, digest in manifest.items()):
        raise DataError("Artefacto protegido o de experimento alterado")


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        if object_hash(read_json(path)) != object_hash(value):
            raise DataError(f"Resultado previo no reproducido: {path}")
    else:
        write_new_json(path, value)


def _write_text_once(path: Path, value: str) -> None:
    if path.exists():
        if path.read_bytes() != value.encode("utf-8"):
            raise DataError(f"Resultado previo no reproducido: {path}")
    else:
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(value)


def _source_archive(directory: Path, protocol_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(
        directory / "source.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for folder in ["src", "tests", "scripts"]:
            for path in sorted((repo / folder).rglob("*.py")):
                archive.write(path, str(path.relative_to(repo)))
        for name in [
            "pyproject.toml",
            "requirements-dmn-lock.txt",
            "DMN_RESEARCH.md",
            "CODEX_PROGRESS.md",
        ]:
            path = repo / name
            if path.is_file():
                archive.write(path, name)
        archive.write(protocol_path, "frozen_protocol.toml")


def _fit_or_recover(panel, partitions, recipe, protocol, seed, directory, *, recovering):
    from trading_ml.dmn.models import apply_scaler, fit_network, fit_xgboost, predict_network

    completed = directory / "completed.json"
    if completed.exists():
        record = read_json(completed)
        _verify_files({str(directory / name): digest for name, digest in record["files"].items()})
        return np.load(directory / "signals.npy", allow_pickle=False), record["metadata"]
    if directory.exists():
        if not recovering:
            raise DataError("Directorio de ajuste preexistente")
        # Keep every interrupted attempt; never silently erase failed evidence.
        archive = directory.parent / f"{directory.name}_interrupted_{uuid4().hex[:8]}"
        directory.rename(archive)
    attempts = directory.parent / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = f"{directory.name}_{uuid4().hex[:8]}"
    write_new_json(
        attempts / f"{attempt}_started.json",
        {
            "recipe": recipe,
            "seed": seed,
            "started_utc": datetime.now(UTC).isoformat(),
        },
    )
    train, validation, test = partitions
    if recipe["architecture"] == "xgboost":
        signal, meta = fit_xgboost(panel, train, validation, test, protocol, seed, directory)
    else:
        model, scaler, meta = fit_network(
            panel, train, validation, recipe, protocol, seed, directory
        )
        signal = predict_network(
            model, apply_scaler(panel.x[test], scaler), recipe, protocol["risk"]
        )
    np.save(directory / "signals.npy", signal, allow_pickle=False)
    files = {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}
    write_new_json(completed, {"metadata": meta, "files": files})
    write_new_json(
        attempts / f"{attempt}_finished.json",
        {
            "finished_utc": datetime.now(UTC).isoformat(),
            "completed_sha256": sha256(completed),
        },
    )
    return signal, meta


def run_study(
    settings, protocol_path: Path, *, resume: Path | None = None, fetch: bool = False
) -> dict:
    protocol = load_protocol(protocol_path)
    code = code_provenance()
    protected = protected_files(settings.root)
    if resume:
        directory = resume.resolve()
        plan = read_json(directory / "plan.json")
        if (
            plan["protocol"] != protocol
            or plan["code"]["source_sha256"] != code["source_sha256"]
            or plan["environment"] != _environment()
        ):
            raise DataError("Reanudar exige protocolo, código y entorno congelados idénticos")
        _verify_files(plan["protected_files"])
        if (directory / "artifacts.json").exists():
            _verify_files(
                {
                    str(directory / name): digest
                    for name, digest in read_json(directory / "artifacts.json").items()
                }
            )
            return read_json(directory / "summary.json")
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        directory = (settings.root / "dmn_studies" / run_id).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        plan = {
            "run_id": run_id,
            "protocol": protocol,
            "code": code,
            "environment": _environment(),
            "protected_files": protected,
            "created_utc": datetime.now(UTC).isoformat(),
            "historical_development_only": True,
            "prospective_evaluated": False,
            "independent_advantage_confirmed": False,
        }
        write_new_json(directory / "plan.json", plan)
        _source_archive(directory, protocol_path)
    LOG.info("DMN run=%s directory=%s", plan["run_id"], directory)
    inputs = directory / "inputs.json"
    if inputs.exists():
        _verify_files({str(directory / name): digest for name, digest in read_json(inputs).items()})
    else:
        raw, sources = load_raw(settings, protocol, directory / "downloads", fetch=fetch)
        _write_once(directory / "sources.json", sources)
        for symbol, frame in raw.items():
            _write_text_once(directory / f"{symbol}_raw.csv", frame.to_csv(index=False))
        write_new_json(
            inputs,
            {
                p.name: sha256(p)
                for p in directory.iterdir()
                if p.is_file()
                and (p.name.endswith("_raw.csv") or p.name in {"sources.json", "source.zip"})
            },
        )
    raw = {}
    for symbol in protocol["data"]["symbols"]:
        frame = pd.read_csv(directory / f"{symbol}_raw.csv", float_precision="round_trip")
        frame.timestamp = pd.to_datetime(frame.timestamp, utc=True)
        guard = pd.DataFrame(
            {
                "reference_time": frame.timestamp + pd.Timedelta(days=1),
                "target_time": frame.timestamp + pd.Timedelta(days=2),
            }
        )
        protect_reserved_labels(settings.root, guard)
        raw[symbol] = frame
    panel = build_panel(
        raw, protocol["training"]["sequence_length"], protocol["risk"]["daily_volatility_floor"]
    )
    dates = pd.to_datetime(protocol["research"]["boundaries"], utc=True)
    folds, outputs, stresses, fit_count = [], {}, {}, 0
    for fold, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True)):
        partitions = temporal_split(panel, start, end, protocol["training"])
        train, validation, test = partitions
        fold_dir = directory / f"fold_{fold:02d}"
        fold_dir.mkdir(exist_ok=True)
        LOG.info(
            "fold=%s %s..%s train=%s validation=%s test=%s",
            fold,
            start.date(),
            end.date(),
            len(train),
            len(validation),
            len(test),
        )
        signals = {name: value[test] for name, value in panel.baselines.items()}
        fit_metadata = {}
        for recipe in protocol["recipes"]:
            seed_signals = []
            for seed in protocol["research"]["seeds"]:
                fit_count += 1
                if fit_count > protocol["research"]["maximum_fits"]:
                    raise DataError("Presupuesto DMN agotado")
                key = f"{recipe['id']}__seed_{seed}"
                model_dir = fold_dir / key
                signal, meta = _fit_or_recover(
                    panel,
                    partitions,
                    recipe,
                    protocol,
                    seed,
                    model_dir,
                    recovering=resume is not None,
                )
                signals[key] = signal
                fit_metadata[key] = meta
                seed_signals.append(signal)
            signals[recipe["id"]] = np.mean(seed_signals, axis=0)
        evaluations, by_asset = {}, {}
        for name, signal in signals.items():
            w = target_weights(signal, panel.volatility[test], protocol["risk"])
            if name == "buy_hold_rebalanced":
                w = np.full_like(w, 1 / len(panel.symbols))
            curve = simulate(w, panel.returns[test], panel.times[test], protocol["costs"])
            stress = simulate(
                w, panel.returns[test], panel.times[test], protocol["costs"], multiplier=2
            )
            curve["fold"], stress["fold"] = fold, fold
            outputs.setdefault(name, []).append(curve)
            stresses.setdefault(name, []).append(stress)
            evaluations[name] = {"net": metrics(curve), "double_cost": metrics(stress)}
            evaluations[name]["gross"] = metrics(
                simulate(w, panel.returns[test], panel.times[test], protocol["costs"], multiplier=0)
            )
            by_asset[name] = {}
            for j, symbol in enumerate(panel.symbols):
                # Standalone sleeve uses full capital to expose concentration effects.
                sleeve = simulate(
                    w[:, j : j + 1] * len(panel.symbols),
                    panel.returns[test, j : j + 1],
                    panel.times[test],
                    protocol["costs"],
                )
                by_asset[name][symbol] = metrics(sleeve)
            save = curve.copy()
            for j, symbol in enumerate(panel.symbols):
                save[f"signal_{symbol}"] = signal[:, j]
                save[f"weight_{symbol}"] = w[:, j]
                save[f"asset_return_{symbol}"] = panel.returns[test, j]
            path = fold_dir / f"{name}_positions.csv"
            _write_text_once(path, save.to_csv(index=False))
        fold_record = {
            "fold": fold,
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "train_dates": len(train),
            "validation_dates": len(validation),
            "test_dates": len(test),
            "train_last_target": panel.target_times[train[-1]].isoformat(),
            "validation_first_reference": panel.times[validation[0]].isoformat(),
            "validation_last_target": panel.target_times[validation[-1]].isoformat(),
            "test_first_reference": panel.times[test[0]].isoformat(),
            "test_last_target": panel.target_times[test[-1]].isoformat(),
            "models": evaluations,
            "by_asset_standalone": by_asset,
            "fit_metadata": fit_metadata,
        }
        path = fold_dir / "report.json"
        if path.exists():
            if object_hash(read_json(path)) != object_hash(fold_record):
                raise DataError("El fold recuperado no reproduce resultados congelados")
        else:
            write_new_json(path, fold_record)
        folds.append(fold_record)
    curves = {
        name: calendar_curve(pd.concat(values, ignore_index=True))
        for name, values in outputs.items()
    }
    stress_curves = {
        name: calendar_curve(pd.concat(values, ignore_index=True))
        for name, values in stresses.items()
    }
    summary = {
        "run_id": plan["run_id"],
        "directory": str(directory),
        "actual_fits": fit_count,
        "fit_attempts": len(list(directory.glob("fold_*/attempts/*_started.json"))),
        "folds": folds,
        "protocol_sha256": object_hash(protocol),
        "cash_boundary_dates": int(next(iter(curves.values())).cash_boundary_date.sum()),
        **summarize(curves, stress_curves, folds, protocol),
    }
    _verify_files(plan["protected_files"])
    summary["protected_artifacts_unchanged"] = True
    for name, curve in curves.items():
        _write_text_once(directory / f"{name}_equity.csv", curve.to_csv(index=False))
    _write_once(directory / "summary.json", summary)
    _write_text_once(directory / "REPORT.md", markdown(summary, str(directory)))
    write_new_json(
        directory / "artifacts.json",
        {
            str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        },
    )
    LOG.info("DMN complete: %s", directory)
    return summary


def _environment() -> dict:
    import torch

    return {**environment(), "torch": str(torch.__version__)}


def verify_study(directory: Path) -> dict:
    manifest = read_json(directory / "artifacts.json")
    for name in manifest:
        if not (directory / name).resolve().is_relative_to(directory.resolve()):
            raise DataError("Ruta fuera del estudio DMN")
    _verify_files({str(directory / name): digest for name, digest in manifest.items()})
    plan = read_json(directory / "plan.json")
    _verify_files(plan["protected_files"])
    summary = read_json(directory / "summary.json")
    reconstructed = 0
    for fold in summary["folds"]:
        for name, recorded in fold["models"].items():
            positions = pd.read_csv(
                directory / f"fold_{fold['fold']:02d}" / f"{name}_positions.csv",
                float_precision="round_trip",
            )
            symbols = plan["protocol"]["data"]["symbols"]
            weights = positions[[f"weight_{s}" for s in symbols]].to_numpy()
            returns = positions[[f"asset_return_{s}" for s in symbols]].to_numpy()
            times = pd.DatetimeIndex(pd.to_datetime(positions.reference_time, utc=True))
            for mode, multiplier in [("gross", 0), ("net", 1), ("double_cost", 2)]:
                curve = simulate(
                    weights, returns, times, plan["protocol"]["costs"], multiplier=multiplier
                )
                actual = metrics(curve)
                for key, value in actual.items():
                    other = recorded[mode][key]
                    if isinstance(value, (float, int)):
                        if other is None or not np.isclose(value, other, rtol=1e-10, atol=1e-12):
                            raise DataError(
                                f"Contabilidad no reproducida: fold {fold['fold']} {name} {mode} {key}"
                            )
                    elif value != other:
                        raise DataError("Métrica DMN no reproducida")
                if multiplier == 1 and not np.allclose(
                    curve.net_return, positions.net_return, rtol=1e-10, atol=1e-12
                ):
                    raise DataError("Retornos diarios DMN no reproducidos")
                reconstructed += 1
    return {
        "directory": str(directory.resolve()),
        "verified_files": len(manifest),
        "protected_files": len(plan["protected_files"]),
        "reconstructed_portfolios": reconstructed,
        "intact": True,
    }
