from __future__ import annotations

import logging
import os
import tomllib
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.backtesting import simulate
from trading_ml.common import environment, frame_hash, read_json, sha256, sidecar, write_json
from trading_ml.config import Settings
from trading_ml.config.settings import settings_from_dict
from trading_ml.data import download
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.selection import Acceptance, assess_consistency, summarize_folds
from trading_ml.features import feature_columns
from trading_ml.models import load_model
from trading_ml.models.registry import publish_candidate
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.training.walk_forward import (
    WalkForwardConfig,
    fit_temporal_model,
    walk_forward_folds,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    enhanced: bool
    extended_history: bool
    early_stop: bool
    parameters: dict[str, Any]
    rolling_days: int | None = None


def fixed_experiments(
    reference_parameters: dict[str, Any], specification: dict[str, Any]
) -> list[Experiment]:
    moderate = {
        **reference_parameters,
        "n_estimators": specification["max_trees"],
        **{
            key: specification[key]
            for key in [
                "max_depth",
                "learning_rate",
                "reg_lambda",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
            ]
        },
    }
    return [
        Experiment(
            "E0",
            "Receta anterior reentrenada antes de cada fold",
            False,
            False,
            False,
            reference_parameters.copy(),
        ),
        Experiment(
            "E1", "Regularización moderada y early stopping interior", False, False, True, moderate
        ),
        Experiment(
            "E2",
            "E1 más ocho features causales de momentum, riesgo y volumen",
            True,
            False,
            True,
            moderate,
        ),
        Experiment(
            "E3",
            "E2 incorporando historia desde el inicio spot de BTCUSDT",
            True,
            True,
            True,
            moderate,
        ),
        Experiment(
            "E4",
            "E2 con train móvil de 730 días para evaluar recencia",
            True,
            False,
            True,
            moderate,
            specification["rolling_train_days"],
        ),
    ]


def _bounds(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "first_reference": frame.reference_time.min().isoformat(),
        "last_reference": frame.reference_time.max().isoformat(),
        "last_target_time": frame.target_time.max().isoformat(),
    }


def evaluate_walk_forward(
    frame: pd.DataFrame,
    settings: Settings,
    experiment: Experiment,
    config: WalkForwardConfig,
    final_start: pd.Timestamp,
    output: Path,
) -> dict[str, Any]:
    """Produce exclusivamente resultados de desarrollo y artefactos por fold."""
    columns = feature_columns(settings.features)
    folds = walk_forward_folds(
        frame,
        config,
        final_start,
        pd.Timestamp(settings.market.start, tz="UTC"),
        experiment.rolling_days,
    )
    reports = []
    predictions = []
    for fold in folds:
        model, fitting = fit_temporal_model(
            fold.train,
            columns,
            experiment.parameters,
            settings.training.seed,
            config,
            experiment.early_stop,
        )
        probability = model.predict_proba(fold.evaluation[columns])[:, 1]
        metrics = classification_metrics(fold.evaluation.target.to_numpy(), probability)
        signals = fold.evaluation[["timestamp", "reference_time"]].copy()
        signals["probability_up"] = probability
        # No precio de ejecución ni etiqueta de folds posteriores/final entra en este backtest.
        candles = frame.loc[frame.reference_time < fold.end]
        result = simulate(
            candles,
            signals,
            settings.training.horizon,
            settings.backtest,
            fold.train.target_time.max(),
        )
        destination = output / f"fold_{fold.number}"
        destination.mkdir(parents=True, exist_ok=True)
        model.save_model(destination / "model.ubj")
        result.equity.to_csv(destination / "equity.csv", index=False)
        result.trades.to_csv(destination / "trades.csv", index=False)
        prediction = fold.evaluation[
            ["timestamp", "reference_time", "target_time", "target"]
        ].copy()
        prediction["probability_up"] = probability
        prediction["constant_probability_up"] = float(fold.train.target.mean())
        prediction["fold"] = fold.number
        predictions.append(prediction)
        reports.append(
            {
                "fold": fold.number,
                "first_reference": fold.evaluation.reference_time.min().isoformat(),
                "last_reference": fold.evaluation.reference_time.max().isoformat(),
                "train": _bounds(fold.train),
                "fitting": fitting,
                "metrics": metrics,
                "backtest": result.summary,
            }
        )
        LOGGER.info(
            "%s fold %s: log-loss %.5f, AUC %s, retorno neto %.2f%%",
            experiment.name,
            fold.number,
            metrics["log_loss"],
            metrics["roc_auc"],
            100 * result.summary["strategy_metrics"]["cumulative_return"],
        )
    oof = pd.concat(predictions, ignore_index=True)
    if oof.reference_time.duplicated().any() or oof.target_time.max() >= final_start:
        raise LeakageError("Predicciones OOF solapadas o dentro del test final")
    oof.to_csv(output / "oof_predictions.csv", index=False)
    report = {
        "experiment": asdict(experiment),
        "config": settings.to_dict(),
        "folds": reports,
        "summary": summarize_folds(
            reports,
            classification_metrics(oof.target.to_numpy(), oof.probability_up.to_numpy()),
            classification_metrics(oof.target.to_numpy(), oof.constant_probability_up.to_numpy()),
        ),
    }
    write_json(output / "walk_forward.json", report)
    return report


def _publish_audit(model: Any, root: Path, metadata: dict[str, Any]) -> str:
    return publish_candidate(model, root, metadata)


def _candidate_dataset(
    base: Settings, source: Path, experiment: Experiment, directory: Path, extended_start: str
) -> tuple[Settings, Path]:
    settings = replace(base, features=replace(base.features, enhanced=experiment.enhanced))
    if experiment.extended_history:
        settings = replace(settings, market=replace(settings.market, start=extended_start))
        extended = directory.parent / "extended_raw.csv"
        if not extended.exists():
            download(settings, extended)
        # Detectar revisiones del proveedor: el período compartido debe ser idéntico.
        old = pd.read_csv(source)
        newer = pd.read_csv(extended)
        shared = newer.loc[newer.timestamp.isin(old.timestamp)].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(old.reset_index(drop=True), shared)
        except AssertionError as exc:
            raise DataError(
                "El histórico extendido modificó el período original; no es una comparación controlada"
            ) from exc
        source = extended
    destination = directory / "prepared.csv"
    return settings, prepare(settings, source, destination)


def optimize(
    settings: Settings, protocol: Path = Path("config/optimization.toml"), version: str = "latest"
) -> dict[str, Any]:
    """Hasta cinco recetas prefijadas, parada por consistencia y test final una vez.

    Conserva el campeón anterior si falla validación o aceptación final. Cada
    reentrenamiento de fold es parte de una receta, no una búsqueda adicional.
    """
    with protocol.open("rb") as stream:
        specification = tomllib.load(stream)
    wf = WalkForwardConfig(**specification["walk_forward"])
    wf.validate()
    rule = Acceptance(**specification["acceptance"])
    rule.validate()
    maximum, patience = (
        specification["budget"]["max_experiments"],
        specification["budget"]["patience"],
    )
    if (
        not isinstance(maximum, int)
        or not 2 <= maximum <= 5
        or not isinstance(patience, int)
        or patience < 1
    ):
        raise DataError("Presupuesto: 2 a 5 experimentos totales y patience >= 1")
    previous = load_model(settings.root, version)
    original = settings_from_dict(previous.metadata["config"])
    original = replace(original, root=settings.root)
    if original.features.enhanced or original.market.interval != "1d":
        raise DataError(
            "Este protocolo está definido para la receta original diaria; requiere un protocolo nuevo para otro contrato"
        )
    final_start = pd.Timestamp(previous.metadata["splits"]["test"]["first_reference"])
    if pd.Timestamp(wf.boundaries[-1], tz="UTC") != final_start:
        raise LeakageError(
            "El último límite walk-forward debe coincidir con el test original reservado"
        )
    raw = settings.root / "data" / "raw.csv"
    if (
        read_json(sidecar(raw))["sha256"] != previous.metadata["raw_sha256"]
        or sha256(raw) != previous.metadata["raw_sha256"]
    ):
        raise DataError("El protocolo requiere el snapshot crudo exacto del modelo anterior")
    prior_pointer = settings.root / "optimization" / "latest.json"
    if prior_pointer.exists():
        prior_report = read_json(Path(read_json(prior_pointer)["summary"]))
        prior_plan = read_json(Path(prior_report["directory"]) / "plan.json")
        if prior_report["previous_version"] == previous.metadata["version"] and prior_plan[
            "protocol_sha256"
        ] == sha256(protocol):
            load_model(settings.root, prior_report["candidate_version"])
            LOGGER.info(
                "Ronda ya completada: %s; se devuelve el informe sin nuevos experimentos",
                prior_report["run_id"],
            )
            return prior_report
    recipes = fixed_experiments(
        previous.metadata["selected_parameters"], specification["experiments"]
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = settings.root / "optimization" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    # Se escribe antes de entrenar, descargar historia adicional o observar resultados.
    plan = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "previous_version": previous.metadata["version"],
        "previous_model_sha256": previous.metadata["model_sha256"],
        "protocol": specification,
        "protocol_sha256": sha256(protocol),
        "final_start": final_start.isoformat(),
        "recipes": [asdict(recipe) for recipe in recipes[:maximum]],
        "final_test_previously_observed": True,
        "final_acceptance": "Sólo el candidato elegido en desarrollo; log-loss y Brier menores que el anterior y retorno neto no menor. Sin probar un segundo candidato si falla.",
    }
    write_json(directory / "plan.json", plan)
    reports: list[dict[str, Any]] = []
    datasets: dict[str, tuple[Settings, Path]] = {}
    failures = 0
    stop_reason = "budget_exhausted"
    for experiment in recipes[:maximum]:
        config, dataset = _candidate_dataset(
            original,
            raw,
            experiment,
            directory / experiment.name,
            specification["data"]["extended_start"],
        )
        datasets[experiment.name] = config, dataset
        frame, _ = load_prepared(dataset, config.contract())
        report = evaluate_walk_forward(
            frame, config, experiment, wf, final_start, directory / experiment.name
        )
        if reports:
            report["acceptance"] = assess_consistency(reports[0], report, rule)
            failures = 0 if report["acceptance"]["accepted"] else failures + 1
        reports.append(report)
        write_json(directory / experiment.name / "walk_forward.json", report)
        # Registra el progreso antes de continuar; el test final no se carga en el selector.
        write_json(
            directory / "development.json",
            {"experiments": reports, "consecutive_failures": failures},
        )
        if failures >= patience:
            stop_reason = "no_consistent_improvement"
            break
    eligible = [report for report in reports[1:] if report["acceptance"]["accepted"]]
    pool = eligible or reports[1:]
    selected = min(pool, key=lambda report: report["summary"]["metrics"]["log_loss"])
    name = selected["experiment"]["name"]
    write_json(
        directory / "selection.json",
        {
            "selected_experiment": name,
            "passed_walk_forward": bool(eligible),
            "criterion": "lowest OOF log_loss among consistent candidates; if none, audit the lowest-loss variant without promoting",
            "selection_frozen_before_final": True,
            "stop_reason": stop_reason,
            "executed_experiments": len(reports),
        },
    )
    config, dataset = datasets[name]
    frame, provenance = load_prepared(dataset, config.contract())
    columns = feature_columns(config.features)
    train = frame.loc[
        (frame.target_time < final_start)
        & (frame.reference_time >= pd.Timestamp(config.market.start, tz="UTC"))
    ].dropna(subset=["target"])
    rolling = selected["experiment"]["rolling_days"]
    if rolling is not None:
        train = train.loc[train.reference_time >= final_start - pd.Timedelta(days=rolling)]
    model, fitting = fit_temporal_model(
        train,
        columns,
        selected["experiment"]["parameters"],
        config.training.seed,
        wf,
        selected["experiment"]["early_stop"],
    )
    # El test original se consulta sólo después de congelar la receta elegida.
    original_frame, _ = load_prepared(
        settings.root / "data" / "prepared.csv", previous.metadata["contract"]
    )
    last_reference = pd.Timestamp(previous.metadata["splits"]["test"]["last_reference"])
    test = frame.loc[
        (frame.reference_time >= final_start) & (frame.reference_time <= last_reference)
    ].dropna(subset=["target", "target_time"])
    old_test = original_frame.loc[original_frame.reference_time.isin(test.reference_time)]
    if not test.reference_time.reset_index(drop=True).equals(
        old_test.reference_time.reset_index(drop=True)
    ) or not np.array_equal(test.target.to_numpy(), old_test.target.to_numpy()):
        raise DataError(
            "El test final debe contener exactamente las mismas referencias y etiquetas"
        )
    old_probability = previous.probabilities(old_test)
    new_probability = model.predict_proba(test[columns])[:, 1]
    comparison = {}
    for label, evaluation, probability, candles, cutoff in [
        (
            "previous",
            old_test,
            old_probability,
            original_frame,
            pd.Timestamp(previous.metadata["selection_label_end"]),
        ),
        ("candidate", test, new_probability, frame, train.target_time.max()),
    ]:
        signals = evaluation[["timestamp", "reference_time"]].copy()
        signals["probability_up"] = probability
        simulation = simulate(candles, signals, config.training.horizon, original.backtest, cutoff)
        destination = directory / "final" / label
        destination.mkdir(parents=True, exist_ok=True)
        simulation.equity.to_csv(destination / "equity.csv", index=False)
        simulation.trades.to_csv(destination / "trades.csv", index=False)
        signals["target"] = evaluation.target.to_numpy()
        signals.to_csv(destination / "predictions.csv", index=False)
        comparison[label] = {
            "metrics": classification_metrics(evaluation.target.to_numpy(), probability),
            "backtest": simulation.summary,
        }
    before, after = comparison["previous"], comparison["candidate"]
    final_checks = {
        "log_loss": after["metrics"]["log_loss"] < before["metrics"]["log_loss"],
        "brier_score": after["metrics"]["brier_score"] < before["metrics"]["brier_score"],
        "net_return": after["backtest"]["strategy_metrics"]["cumulative_return"]
        >= before["backtest"]["strategy_metrics"]["cumulative_return"],
    }
    promoted = bool(eligible) and all(final_checks.values())
    version_new = _publish_audit(
        model,
        settings.root,
        {
            "parent_version": previous.metadata["version"],
            "config": replace(
                config, model=fitting["effective_parameters"], search={"enabled": False}
            ).to_dict(),
            "contract": config.contract(),
            "features": columns,
            "selected_parameters": fitting["effective_parameters"],
            "fitting": fitting,
            "splits": {
                "train": _bounds(train),
                "validation": {"method": "purged_walk_forward", "folds": len(wf.boundaries) - 1},
                "test": _bounds(test),
            },
            "selection_label_end": train.target_time.max().isoformat(),
            "training_data_sha256": frame_hash(train),
            "prepared_sha256": provenance["sha256"],
            "raw_sha256": provenance["raw_sha256"],
            "dataset_path": str(dataset.resolve()),
            "test_metrics": after["metrics"],
            "optimization": {
                "run_id": run_id,
                "experiment": name,
                "passed_walk_forward": bool(eligible),
                "promoted": promoted,
                "plan_path": str((directory / "plan.json").resolve()),
            },
        },
    )
    champion = version_new if promoted else previous.metadata["version"]
    if promoted:
        temporary = settings.root / "models" / f".latest_{uuid.uuid4().hex}.json"
        write_json(temporary, {"version": champion})
        os.replace(temporary, settings.root / "models" / "latest.json")
    write_json(
        settings.root / "models" / "champion.json",
        {
            "version": champion,
            "run_id": run_id,
            "criterion": "purged walk-forward consistency and single final confirmation; otherwise retain previous",
        },
    )
    summary = {
        "run_id": run_id,
        "directory": str(directory),
        "plan_sha256": sha256(directory / "plan.json"),
        "environment": environment(),
        "previous_version": previous.metadata["version"],
        "candidate_version": version_new,
        "champion_version": champion,
        "selected_experiment": name,
        "selected_dataset": str(dataset),
        "executed_experiments": len(reports),
        "stop_reason": stop_reason,
        "passed_walk_forward": bool(eligible),
        "final_checks": final_checks,
        "promoted": promoted,
        "final_test_previously_observed": True,
        "comparison": comparison,
        "experiments": reports,
        "more_history_evaluated": any(
            report["experiment"]["extended_history"] for report in reports
        ),
    }
    write_json(directory / "summary.json", summary)
    write_json(
        settings.root / "optimization" / "latest.json",
        {"run_id": run_id, "summary": str(directory / "summary.json")},
    )
    return summary
