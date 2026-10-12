from __future__ import annotations

import logging
import tomllib
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from trading_ml.backtesting import simulate
from trading_ml.common import environment, frame_hash, read_json, sha256, sidecar, write_json
from trading_ml.config import Settings
from trading_ml.config.settings import settings_from_dict
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.selection import Acceptance, assess_consistency, summarize_folds
from trading_ml.evaluation.signals import signal_quality
from trading_ml.features import feature_columns
from trading_ml.models import load_model
from trading_ml.models.calibration import apply_calibration, fit_sigmoid
from trading_ml.models.registry import publish_candidate
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.smc.detectors import SMCConfig
from trading_ml.smc.rules import directional_states, rule_probabilities
from trading_ml.training.walk_forward import (
    WalkForwardConfig,
    fit_temporal_model,
    walk_forward_folds,
)

LOGGER = logging.getLogger(__name__)


def fit_smc_model(
    train: pd.DataFrame,
    columns: list[str],
    parameters: dict[str, Any],
    wf: WalkForwardConfig,
    seed: int,
    early_stop: bool,
    calibrated: bool,
    calibration_days: int,
) -> tuple[XGBClassifier, dict[str, Any], dict[str, Any] | None]:
    """Fit/ES/calibración disjuntos y purgados; no refit tras calibrar."""
    calibration = None
    fitting = train
    calibration_frame = None
    if calibrated:
        if not isinstance(calibration_days, int) or calibration_days < 20:
            raise DataError("Ventana de calibración insuficiente")
        start = train.reference_time.max() - pd.Timedelta(days=calibration_days)
        fitting = train.loc[train.target_time < start]
        calibration_frame = train.loc[train.reference_time >= start]
        if fitting.empty or len(calibration_frame) < 20:
            raise DataError("Datos insuficientes para fit/calibración disjuntos")
    model, details = fit_temporal_model(fitting, columns, parameters, seed, wf, early_stop)
    if calibration_frame is not None:
        raw = model.predict_proba(calibration_frame[columns])[:, 1]
        calibration = fit_sigmoid(raw, calibration_frame.target.to_numpy(), seed)
        calibration.update(
            {
                "first_reference": calibration_frame.reference_time.min().isoformat(),
                "last_reference": calibration_frame.reference_time.max().isoformat(),
                "last_target_time": calibration_frame.target_time.max().isoformat(),
                "fit_last_target_time": fitting.target_time.max().isoformat(),
                "disjoint_from_model_fit": True,
            }
        )
        details["calibration"] = calibration
    details["all_known_labels_end"] = train.target_time.max().isoformat()
    return model, details, calibration


def evaluate_smc_recipe(
    frame: pd.DataFrame,
    settings: Settings,
    recipe: dict[str, Any],
    wf: WalkForwardConfig,
    final_start: pd.Timestamp,
    destination: Path,
    threshold: float,
    calibration_days: int,
) -> dict[str, Any]:
    """Sólo desarrollo; reglas no aprenden de etiquetas, S0 no usa columnas SMC."""
    features = replace(settings.features, smc=recipe["kind"] != "original")
    columns = feature_columns(features)
    folds = walk_forward_folds(
        frame, wf, final_start, pd.Timestamp(settings.market.start, tz="UTC")
    )
    reports = []
    predictions = []
    for fold in folds:
        calibration = None
        details: dict[str, Any] = {
            "train_rows": len(fold.train),
            "train_last_target": fold.train.target_time.max().isoformat(),
        }
        directory = destination / f"fold_{fold.number}"
        directory.mkdir(parents=True, exist_ok=True)
        if recipe["kind"] == "rules":
            probability, states = rule_probabilities(fold.evaluation)
            calibration_status = "heuristic_uncalibrated"
        else:
            model, details, calibration = fit_smc_model(
                fold.train,
                columns,
                recipe["parameters"],
                wf,
                settings.training.seed,
                recipe["early_stop"],
                recipe["calibrated"],
                calibration_days,
            )
            probability = apply_calibration(
                model.predict_proba(fold.evaluation[columns])[:, 1], calibration
            )
            states = directional_states(
                probability,
                threshold,
                fold.evaluation.smc_ready.to_numpy() if recipe["kind"] != "original" else None,
            )
            model.save_model(directory / "model.ubj")
            calibration_status = (
                "sigmoid_fitted_on_disjoint_past_window" if calibration else "uncalibrated"
            )
        signals = fold.evaluation[["timestamp", "reference_time"]].copy()
        signals["probability_up"] = probability
        # Es simulación long/cash; señales bajistas se evalúan pero no abren shorts.
        simulation = simulate(
            frame.loc[frame.reference_time < fold.end],
            signals,
            settings.training.horizon,
            settings.backtest,
            fold.train.target_time.max(),
        )
        simulation.equity.to_csv(directory / "equity.csv", index=False)
        simulation.trades.to_csv(directory / "trades.csv", index=False)
        prediction = fold.evaluation[
            ["timestamp", "reference_time", "target_time", "target"]
        ].copy()
        prediction["probability_up"] = probability
        prediction["state"] = states
        prediction["constant_probability_up"] = float(fold.train.target.mean())
        prediction["fold"] = fold.number
        predictions.append(prediction)
        metrics = classification_metrics(fold.evaluation.target.to_numpy(), probability)
        reports.append(
            {
                "fold": fold.number,
                "first_reference": fold.evaluation.reference_time.min().isoformat(),
                "last_reference": fold.evaluation.reference_time.max().isoformat(),
                "fitting": details,
                "calibration_status": calibration_status,
                "metrics": metrics,
                "signal_quality": signal_quality(
                    fold.evaluation.target.to_numpy(), probability, states
                ),
                "backtest": simulation.summary,
            }
        )
        write_json(directory / "evaluation.json", reports[-1])
        LOGGER.info(
            "%s fold %s: AUC=%s log-loss=%.5f señales=%s retorno=%.2f%%",
            recipe["id"],
            fold.number,
            metrics["roc_auc"],
            metrics["log_loss"],
            reports[-1]["signal_quality"]["directional_signals"],
            100 * simulation.summary["strategy_metrics"]["cumulative_return"],
        )
    oof = pd.concat(predictions, ignore_index=True)
    if oof.target_time.max() >= final_start or oof.reference_time.duplicated().any():
        raise LeakageError("OOF SMC invade final o duplica referencias")
    oof.to_csv(destination / "oof_predictions.csv", index=False)
    selected_rows = frame.loc[frame.reference_time.isin(oof.reference_time)]
    report = {
        "recipe": recipe,
        "features": columns,
        "folds": reports,
        "summary": summarize_folds(
            reports,
            classification_metrics(oof.target.to_numpy(), oof.probability_up.to_numpy()),
            classification_metrics(oof.target.to_numpy(), oof.constant_probability_up.to_numpy()),
        ),
        "signal_quality": signal_quality(
            oof.target.to_numpy(), oof.probability_up.to_numpy(), oof.state.to_numpy()
        ),
        "smc_pattern_counts": {
            name: int(selected_rows[name].sum())
            for name in [
                "smc_bos_up",
                "smc_bos_down",
                "smc_choch_up",
                "smc_choch_down",
                "smc_sweep_low",
                "smc_sweep_high",
                "smc_fvg_up",
                "smc_fvg_down",
                "smc_ob_up",
                "smc_ob_down",
            ]
        },
    }
    write_json(destination / "report.json", report)
    return report


def run_smc_study(
    settings: Settings, protocol: Path = Path("config/smc_study.toml"), version: str = "latest"
) -> dict[str, Any]:
    """Ronda acotada; no consulta test observado y jamás modifica latest/champion."""
    specification = tomllib.loads(protocol.read_text(encoding="utf-8"))
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
        or not 3 <= maximum <= 5
        or not isinstance(patience, int)
        or patience < 1
    ):
        raise DataError("SMC admite 3 a 5 recetas y paciencia positiva")
    threshold = specification["signals"]["threshold"]
    if not 0.5 < threshold < 1:
        raise DataError("Umbral neutral inválido")
    calibration_days = specification["calibration"]["days"]
    prospective_days = specification["prospective"]["window_days"]
    if (
        not isinstance(calibration_days, int)
        or isinstance(calibration_days, bool)
        or calibration_days < 20
    ):
        raise DataError("La calibración requiere al menos 20 días")
    if (
        not isinstance(prospective_days, int)
        or isinstance(prospective_days, bool)
        or prospective_days < 1
    ):
        raise DataError("La reserva prospectiva requiere una cantidad positiva de días")
    if set(specification["smc"]) - {
        "swing_left",
        "swing_right",
        "liquidity_window",
        "ob_window",
        "zone_age",
    }:
        raise DataError("Parámetros SMC no soportados por el contrato de esta ronda")
    smc_config = SMCConfig(**specification["smc"])
    smc_config.validate()
    reference = load_model(settings.root, version)
    base = settings_from_dict(reference.metadata["config"])
    base = replace(base, root=settings.root)
    if (
        base.market.interval != "1d"
        or base.features.smc
        or base.features.enhanced
        or threshold != base.backtest.threshold
    ):
        raise DataError(
            "Esta ronda requiere referencia original diaria y el mismo umbral de costos/señales"
        )
    final_start = pd.Timestamp(reference.metadata["splits"]["test"]["first_reference"])
    if pd.Timestamp(wf.boundaries[-1], tz="UTC") != final_start:
        raise LeakageError("Las ventanas deben terminar en el inicio del test histórico excluido")
    raw = base.root / "data" / "raw.csv"
    if (
        sha256(raw) != reference.metadata["raw_sha256"]
        or read_json(sidecar(raw))["sha256"] != reference.metadata["raw_sha256"]
    ):
        raise DataError("La comparación requiere el snapshot original exacto")
    pointer = base.root / "smc_studies" / "latest.json"
    if pointer.exists():
        cached = read_json(Path(read_json(pointer)["summary"]))
        if cached["reference_version"] == reference.metadata["version"] and cached[
            "protocol_sha256"
        ] == sha256(protocol):
            load_model(base.root, cached["candidate_version"])
            return cached
    original_parameters = reference.metadata["selected_parameters"]
    moderate = {**original_parameters, **specification["regularized"]}
    replace(base, model=moderate).validate()
    recipes = [
        {
            "id": "S0",
            "kind": "original",
            "description": "XGBoost original reentrenado por ventana",
            "parameters": original_parameters,
            "early_stop": False,
            "calibrated": False,
        },
        {
            "id": "S1",
            "kind": "rules",
            "description": "SMC determinista: confianza heurística sin ML",
            "parameters": {},
            "early_stop": False,
            "calibrated": False,
        },
        {
            "id": "S2",
            "kind": "combined",
            "description": "XGBoost original + SMC, mismo train e hiperparámetros",
            "parameters": original_parameters,
            "early_stop": False,
            "calibrated": False,
        },
        {
            "id": "S3",
            "kind": "combined",
            "description": "SMC + regularización, ES interior y calibración temporal independiente",
            "parameters": moderate,
            "early_stop": True,
            "calibrated": True,
        },
        {
            "id": "S4",
            "kind": "combined",
            "description": "SMC con confirmación más lenta (left=5, right=3)",
            "parameters": original_parameters,
            "early_stop": False,
            "calibrated": False,
            "smc_override": {"smc_swing_left": 5, "smc_swing_right": 3},
        },
    ]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = base.root / "smc_studies" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    # Las fronteras futuras se congelan en esta invocación, antes de observar resultados.
    raw_last = (
        pd.to_datetime(pd.read_csv(raw, usecols=["timestamp"]).timestamp, utc=True).max()
        + base.delta
    )
    future_start = max(pd.Timestamp.now(tz="UTC").normalize() + base.delta, raw_last + base.delta)
    future_end = future_start + pd.Timedelta(days=prospective_days)
    plan = {
        "smc_detector_config": asdict(smc_config),
        "smc_definition_version": "smc_v1",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "reference_version": reference.metadata["version"],
        "reference_model_sha256": reference.metadata["model_sha256"],
        "raw_sha256": sha256(raw),
        "protocol_sha256": sha256(protocol),
        "protocol": specification,
        "recipes": recipes[:maximum],
        "excluded_observed_test_start": final_start.isoformat(),
        "prospective_reference_start": future_start.isoformat(),
        "prospective_reference_end_exclusive": future_end.isoformat(),
        "automatic_promotion": False,
        "final_evaluation_status": "pending_future_data",
    }
    write_json(destination / "plan.json", plan)
    (destination / "raw_snapshot.csv").write_bytes(raw.read_bytes())
    write_json(sidecar(destination / "raw_snapshot.csv"), read_json(sidecar(raw)))
    detector_features = replace(
        base.features,
        smc=True,
        **{f"smc_{key}": value for key, value in specification["smc"].items()},
    )
    common = replace(base, features=detector_features)
    dataset = prepare(common, raw, destination / "prepared_smc.csv")
    frame, _ = load_prepared(dataset, common.contract())
    reports = []
    failures = 0
    stop_reason = "budget_exhausted"
    datasets: dict[str, tuple[Settings, Path]] = {}
    for recipe in recipes[:maximum]:
        current, path = common, dataset
        if "smc_override" in recipe:
            current = replace(common, features=replace(common.features, **recipe["smc_override"]))
            path = prepare(current, raw, destination / f"prepared_{recipe['id']}.csv")
        datasets[recipe["id"]] = current, path
        current_frame, _ = load_prepared(path, current.contract())
        report = evaluate_smc_recipe(
            current_frame,
            current,
            recipe,
            wf,
            final_start,
            destination / recipe["id"],
            threshold,
            calibration_days,
        )
        if recipe["kind"] == "combined":
            report["acceptance"] = assess_consistency(reports[0], report, rule)
            failures = 0 if report["acceptance"]["accepted"] else failures + 1
        reports.append(report)
        write_json(destination / recipe["id"] / "report.json", report)
        write_json(
            destination / "development.json",
            {"reports": reports, "consecutive_ml_failures": failures},
        )
        if failures >= patience:
            stop_reason = "no_consistent_smc_improvement"
            break
    candidates = [report for report in reports if report["recipe"]["kind"] == "combined"]
    qualified = [report for report in candidates if report["acceptance"]["accepted"]]
    selected = min(
        qualified or candidates, key=lambda report: report["summary"]["metrics"]["log_loss"]
    )
    chosen = selected["recipe"]
    write_json(
        destination / "selection.json",
        {
            "selected_recipe": chosen["id"],
            "passed_walk_forward": bool(qualified),
            "criterion": "lowest development OOF log_loss among qualified recipes, otherwise research-only candidate",
            "excluded_test_never_evaluated": True,
        },
    )
    chosen_settings, chosen_dataset = datasets[chosen["id"]]
    chosen_frame, provenance = load_prepared(chosen_dataset, chosen_settings.contract())
    training = chosen_frame.loc[chosen_frame.target_time < final_start].dropna(
        subset=["target", "target_time"]
    )
    columns = feature_columns(chosen_settings.features)
    model, fitting, calibration = fit_smc_model(
        training,
        columns,
        chosen["parameters"],
        wf,
        base.training.seed,
        chosen["early_stop"],
        chosen["calibrated"],
        calibration_days,
    )
    cutoff = training.target_time.max().isoformat()
    candidate_version = publish_candidate(
        model,
        base.root,
        {
            "parent_version": reference.metadata["version"],
            "config": replace(
                chosen_settings, model=fitting["effective_parameters"], search={"enabled": False}
            ).to_dict(),
            "contract": chosen_settings.contract(),
            "features": columns,
            "selected_parameters": fitting["effective_parameters"],
            "fitting": fitting,
            "calibration": calibration,
            "training_data_sha256": frame_hash(training),
            "prepared_sha256": provenance["sha256"],
            "raw_sha256": provenance["raw_sha256"],
            "dataset_path": str(chosen_dataset.resolve()),
            "selection_label_end": cutoff,
            "splits": {
                "train": {
                    "rows": len(training),
                    "first_reference": training.reference_time.min().isoformat(),
                    "last_reference": training.reference_time.max().isoformat(),
                    "last_target_time": cutoff,
                },
                "validation": {"method": "purged_walk_forward", "folds": len(wf.boundaries) - 1},
                "test": {
                    "status": "pending_future_data",
                    "first_reference": future_start.isoformat(),
                    "last_reference": (future_end - base.delta).isoformat(),
                    "last_target_time": (
                        future_end - base.delta + base.training.horizon * base.delta
                    ).isoformat(),
                },
            },
            "test_metrics": {"status": "pending_future_data"},
            "smc_study": {
                "run_id": run_id,
                "recipe": chosen["id"],
                "passed_walk_forward": bool(qualified),
                "promoted": False,
                "signal_threshold": threshold,
            },
        },
    )
    prospective = {
        "status": "pending_future_data",
        "first_reference": future_start.isoformat(),
        "end_reference_exclusive": future_end.isoformat(),
        "candidate_version": candidate_version,
        "reference_version": reference.metadata["version"],
        "rules_version": "smc_rules_v1",
        "protocol_sha256": sha256(protocol),
        "horizon_bars": base.training.horizon,
        "interval": base.market.interval,
        "requires_new_unobserved_ohlcv": True,
    }
    write_json(destination / "prospective.json", prospective)
    summary = {
        "run_id": run_id,
        "directory": str(destination),
        "plan_sha256": sha256(destination / "plan.json"),
        "protocol_sha256": sha256(protocol),
        "reference_version": reference.metadata["version"],
        "candidate_version": candidate_version,
        "candidate_dataset": str(chosen_dataset),
        "selected_recipe": chosen["id"],
        "executed_experiments": len(reports),
        "stop_reason": stop_reason,
        "passed_walk_forward": bool(qualified),
        "promoted": False,
        "old_test_evaluated": False,
        "prospective": prospective,
        "environment": environment(),
        "reports": reports,
        "comparison": {"original": reports[0], "smc_rules": reports[1], "combined": selected},
    }
    write_json(destination / "summary.json", summary)
    write_json(pointer, {"run_id": run_id, "summary": str(destination / "summary.json")})
    return summary
