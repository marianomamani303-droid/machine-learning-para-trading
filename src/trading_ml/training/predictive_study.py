from __future__ import annotations

import math
import tomllib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import (
    code_provenance,
    environment,
    frame_hash,
    object_hash,
    read_json,
    sha256,
    write_new_json,
)
from trading_ml.config import Settings
from trading_ml.config.settings import settings_from_dict
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.baseline_gate import assess_baseline_gate
from trading_ml.evaluation.diagnostics import probability_diagnostics
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.research_report import write_research_report
from trading_ml.evaluation.uncertainty import BootstrapConfig, block_bootstrap_comparison
from trading_ml.features import feature_columns
from trading_ml.models import ModelBundle, load_model
from trading_ml.models.baselines import BASELINE_NAMES, fit_reference_models
from trading_ml.models.calibration import apply_calibration
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.training.recipe import protect_reserved_labels
from trading_ml.training.walk_forward import WalkForwardConfig, walk_forward_folds


def feature_families(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "momentum": [c for c in columns if c.startswith(("return_", "macd", "sma_ratio_"))],
        "price_shape": [c for c in columns if c in {"range_ratio", "body_ratio"}],
        "volatility": [c for c in columns if c == "volatility"],
        "volume": [c for c in columns if c == "volume_ratio"],
        "rsi": [c for c in columns if c == "rsi"],
        "smc": [c for c in columns if c.startswith("smc_")],
    }
    assigned = [c for group in groups.values() for c in group]
    if sorted(assigned) != sorted(columns) or len(assigned) != len(set(assigned)):
        raise DataError("Las familias deben cubrir exactamente las features sin duplicados")
    return groups


def fixed_ablations(columns: list[str]) -> dict[str, list[str]]:
    groups = feature_families(columns)
    if any(not group for group in groups.values()):
        raise DataError("Este protocolo requiere features originales y SMC")
    core = [c for c in columns if not c.startswith("smc_")]
    return {
        "xgboost": core,
        "xgboost_with_smc": columns.copy(),
        "xgboost_smc_only": groups["smc"],
        **{
            f"without_{name}": [c for c in core if c not in group]
            for name, group in groups.items()
            if name != "smc"
        },
    }


def _bounds(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "first_reference": frame.reference_time.min().isoformat(),
        "last_reference": frame.reference_time.max().isoformat(),
        "last_target_time": frame.target_time.max().isoformat(),
        "frame_sha256": frame_hash(frame),
    }


def evaluate_development(
    frame: pd.DataFrame,
    settings: Settings,
    parameters: dict[str, Any],
    specification: dict[str, Any],
    final_start: pd.Timestamp,
    output: Path,
) -> dict[str, Any]:
    wf = WalkForwardConfig(**specification["walk_forward"])
    columns = feature_columns(settings.features)
    recipes = fixed_ablations(columns)
    folds = walk_forward_folds(
        frame, wf, final_start, pd.Timestamp(settings.market.start, tz="UTC")
    )
    budget = specification["budget"]
    fits = len(folds) * (len(recipes) + 1) + int(
        specification["diagnostics"]["historical_train_validation_test"]
    )
    if (
        len(recipes) > budget["max_xgboost_recipes"]
        or fits > budget["max_fits"]
        or parameters["n_estimators"] > budget["max_trees"]
    ):
        raise DataError("El protocolo excede el presupuesto prefijado de recetas/ajustes/árboles")
    if (
        budget["cpu_threads"] != 1
        or not 1 <= budget["max_trees"] <= 400
        or not 1 <= budget["max_fits"] <= 128
    ):
        raise DataError("Presupuesto CPU inválido")
    bootstrap = BootstrapConfig(
        **{
            **specification["bootstrap"],
            "block_lengths": tuple(specification["bootstrap"]["block_lengths"]),
        }
    )
    bootstrap.validate()
    if not 0 < specification["acceptance"]["required_fold_fraction"] <= 1:
        raise DataError("Fracción de aceptación inválida")
    records = []
    predictions = []
    for fold in folds:
        protect_reserved_labels(settings.root, fold.train)
        directory = output / f"fold_{fold.number}"
        directory.mkdir(parents=True, exist_ok=False)
        references = fit_reference_models(
            fold.train,
            recipes["xgboost"],
            seed=settings.training.seed,
            C=specification["baselines"]["logistic_C"],
        )
        write_new_json(directory / "baselines.json", references.to_dict())
        probabilities = references.probabilities(fold.evaluation)
        train_probabilities = references.probabilities(fold.train)
        for name, features in recipes.items():
            model = XGBClassifier(
                **parameters,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                random_state=settings.training.seed,
                n_jobs=1,
            )
            model.fit(fold.train[features], fold.train.target.astype(int))
            model.save_model(directory / (name + ".ubj"))
            probabilities[name] = model.predict_proba(fold.evaluation[features])[:, 1]
            train_probabilities[name] = model.predict_proba(fold.train[features])[:, 1]
        prediction = fold.evaluation[
            ["timestamp", "reference_time", "target_time", "target"]
        ].copy()
        prediction["fold"] = fold.number
        for name, probability in probabilities.items():
            prediction[name] = probability
        predictions.append(prediction)
        record = {
            "fold": fold.number,
            "train": _bounds(fold.train),
            "evaluation": _bounds(fold.evaluation),
            "identical_rows_for_all_models": True,
            "models": {
                name: {
                    "train": classification_metrics(
                        fold.train.target.to_numpy(), train_probabilities[name]
                    ),
                    "validation": classification_metrics(fold.evaluation.target.to_numpy(), p),
                    "probabilities": probability_diagnostics(
                        fold.evaluation.target.to_numpy(), p, settings.backtest.threshold
                    ),
                }
                for name, p in probabilities.items()
            },
        }
        records.append(record)
        write_new_json(directory / "evaluation.json", record)
    oof = pd.concat(predictions, ignore_index=True)
    if oof.reference_time.duplicated().any() or oof.target_time.max() >= final_start:
        raise LeakageError("Predicciones de desarrollo duplicadas o dentro del test observado")
    output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    probabilities = {name: oof[name].to_numpy() for name in [*BASELINE_NAMES, *recipes]}
    pairs = [(name, baseline) for name in recipes for baseline in BASELINE_NAMES]
    pairs += [(name, "xgboost") for name in recipes if name != "xgboost"]
    uncertainty = block_bootstrap_comparison(
        oof.target.to_numpy(), probabilities, oof.fold.to_numpy(), pairs, bootstrap
    )
    models = {}
    for name, probability in probabilities.items():
        metric = classification_metrics(oof.target.to_numpy(), probability)
        aucs = [r["models"][name]["validation"]["roc_auc"] for r in records]
        models[name] = {
            "metrics": metric,
            "mean_fold_auc": float(np.mean(aucs)) if all(a is not None for a in aucs) else None,
            "probabilities": probability_diagnostics(
                oof.target.to_numpy(), probability, settings.backtest.threshold
            ),
            "mean_train_log_loss": float(
                np.mean([r["models"][name]["train"]["log_loss"] for r in records])
            ),
            "mean_validation_log_loss": float(
                np.mean([r["models"][name]["validation"]["log_loss"] for r in records])
            ),
        }
        if name in recipes:
            decision = assess_baseline_gate(
                probabilities,
                oof.target.to_numpy(),
                uncertainty,
                name,
                minimum_loss_gain=specification["acceptance"]["minimum_log_loss_gain"],
            )
            required = math.ceil(
                specification["acceptance"]["required_fold_fraction"] * len(records)
            )
            wins = {
                baseline: sum(
                    r["models"][name]["validation"]["log_loss"]
                    < r["models"][baseline]["validation"]["log_loss"]
                    and r["models"][name]["validation"]["brier_score"]
                    < r["models"][baseline]["validation"]["brier_score"]
                    for r in records
                )
                for baseline in BASELINE_NAMES
            }
            decision["winning_folds_vs_baselines"] = wins
            decision["required_winning_folds"] = required
            if any(value < required for value in wins.values()):
                decision["accepted"] = False
                decision["status"] = "rejected_by_baselines"
                decision["failed_checks"].append("fold_consistency")
            models[name]["decision"] = decision
    return {
        "scope": "previously_observed_development_only",
        "rows": len(oof),
        "folds": records,
        "models": models,
        "uncertainty": uncertainty,
        "families": feature_families(columns),
        "recipes": recipes,
        "fits": {"xgboost": len(folds) * len(recipes), "logistic": len(folds)},
        "test_excluded_from_fit_selection_and_ablations": True,
        "limitations": [
            "Original hyperparameters were historically selected; these are retrospective development comparisons.",
            "Ablations are conditional on correlated remaining features and one fixed seed; they do not establish market causality.",
            "Pooled AUC ranks scores across models/folds; mean-fold AUC is reported separately.",
        ],
    }


def historical_split_diagnostics(
    bundle: ModelBundle, frame: pd.DataFrame, specification: dict[str, Any], output: Path
) -> dict[str, Any]:
    partitions = {}
    for name in ["train", "validation", "test"]:
        bounds = bundle.metadata["splits"][name]
        partition = frame.loc[
            (frame.reference_time >= pd.Timestamp(bounds["first_reference"]))
            & (frame.reference_time <= pd.Timestamp(bounds["last_reference"]))
        ].dropna(subset=["target", "target_time"])
        if len(partition) != bounds["rows"]:
            raise DataError("Las particiones históricas no coinciden con las registradas")
        partitions[name] = partition
    if (
        partitions["train"].target_time.max() >= partitions["validation"].reference_time.min()
        or partitions["validation"].target_time.max() >= partitions["test"].reference_time.min()
    ):
        raise LeakageError("Particiones históricas no purgadas")
    references = fit_reference_models(
        partitions["train"],
        bundle.metadata["features"],
        seed=bundle.metadata["config"]["training"]["seed"],
        C=specification["baselines"]["logistic_C"],
    )
    write_new_json(output / "historical_baselines.json", references.to_dict())
    reports = {}
    predictions = []
    for name, partition in partitions.items():
        probabilities = {
            "xgboost": bundle.probabilities(partition),
            **references.probabilities(partition),
        }
        reports[name] = {
            "bounds": _bounds(partition),
            "models": {
                key: probability_diagnostics(partition.target.to_numpy(), p)
                for key, p in probabilities.items()
            },
        }
        if name != "train":
            config = BootstrapConfig(
                **{
                    **specification["bootstrap"],
                    "block_lengths": tuple(specification["bootstrap"]["block_lengths"]),
                }
            )
            reports[name]["uncertainty"] = block_bootstrap_comparison(
                partition.target.to_numpy(),
                probabilities,
                np.zeros(len(partition), dtype=int),
                [("xgboost", baseline) for baseline in BASELINE_NAMES],
                config,
            )
        prediction = partition[["reference_time", "target_time", "target"]].copy()
        prediction["partition"] = name
        for key, p in probabilities.items():
            prediction[key] = p
        predictions.append(prediction)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / "historical_predictions.csv", index=False
    )
    return {
        "scope": "descriptive_fixed_baselines_on_already_observed_test",
        "used_for_selection_or_rejection": False,
        "partitions": reports,
        "generalization_gaps": {
            model: {
                "validation_minus_train_log_loss": reports["validation"]["models"][model][
                    "metrics"
                ]["log_loss"]
                - reports["train"]["models"][model]["metrics"]["log_loss"],
                "test_minus_train_log_loss": reports["test"]["models"][model]["metrics"]["log_loss"]
                - reports["train"]["models"][model]["metrics"]["log_loss"],
            }
            for model in ["xgboost", *BASELINE_NAMES]
        },
        "warning": "The XGBoost recipe was selected on validation and test was previously observed; this is not new blind evidence.",
    }


def existing_smc_diagnostics(root: Path, final_start: pd.Timestamp) -> dict[str, Any]:
    pointer = root / "smc_studies" / "latest.json"
    if not pointer.exists():
        return {"status": "no_historical_smc_study"}
    summary_path = Path(read_json(pointer)["summary"])
    summary = read_json(summary_path)
    selected = next(
        r for r in summary["reports"] if r["recipe"]["id"] == summary["selected_recipe"]
    )
    directory = summary_path.parent / summary["selected_recipe"]
    oof = pd.read_csv(directory / "oof_predictions.csv")
    if pd.to_datetime(oof.target_time, utc=True).max() >= final_start:
        raise LeakageError("El diagnóstico SMC no puede consumir test observado")
    frame, _ = load_prepared(Path(summary["candidate_dataset"]))
    details = []
    for number, group in oof.groupby("fold", sort=True):
        recorded = selected["folds"][int(number) - 1]["fitting"]
        references = pd.to_datetime(group.reference_time, utc=True)
        rows = frame.loc[frame.reference_time.isin(references)]
        if not np.array_equal(
            rows.target.to_numpy(), group.target.to_numpy()
        ) or not np.array_equal(rows.reference_time.to_numpy(), references.to_numpy()):
            raise DataError("El OOF SMC no coincide con su snapshot")
        if pd.Timestamp(recorded["train_last_target"]) >= rows.reference_time.min():
            raise LeakageError("Modelo SMC ajustado después del inicio de su fold")
        path = directory / f"fold_{number}" / "model.ubj"
        model = XGBClassifier()
        model.load_model(path)
        raw = model.predict_proba(rows[model.get_booster().feature_names])[:, 1]
        calibration = recorded.get("calibration")
        calibrated = apply_calibration(raw, calibration)
        if not np.allclose(calibrated, group.probability_up.to_numpy(), rtol=1e-12, atol=1e-12):
            raise DataError("Las probabilidades SMC no se reproducen desde el modelo y calibrador")
        details.append(
            {
                "fold": int(number),
                "calibration": calibration,
                "effective_parameters": recorded.get("effective_parameters"),
                "model_sha256": sha256(path),
                "raw_diagnostics": probability_diagnostics(group.target.to_numpy(), raw),
                "diagnostics": probability_diagnostics(group.target.to_numpy(), calibrated),
            }
        )
    return {
        "status": "read_only_reuse",
        "source_sha256": sha256(directory / "oof_predictions.csv"),
        "not_part_of_current_baseline_gate": True,
        "folds": details,
        "interpretation": "Calibrated score levels and collapse describe prevalence/abstention, not necessarily opportunity discrimination.",
    }


def _protected(root: Path) -> dict[str, str]:
    paths = list((root / "models").rglob("*"))
    paths += [
        p
        for p in (root / "smc_studies").rglob("*")
        if p.name in {"plan.json", "prospective.json", "latest.json"}
    ]
    return {str(p.resolve()): sha256(p) for p in paths if p.is_file()}


def run_predictive_study(
    settings: Settings,
    protocol: Path = Path("config/predictive_study.toml"),
    version: str = "latest",
) -> dict[str, Any]:
    specification = tomllib.loads(protocol.read_text(encoding="utf8"))
    reference = load_model(settings.root, version)
    base = replace(settings_from_dict(reference.metadata["config"]), root=settings.root)
    parameters = specification["xgboost"]
    replace(base, model=parameters, search={"enabled": False}).validate()
    if base.features.smc or base.features.enhanced or base.market.interval != "1d":
        raise DataError(
            "Este protocolo requiere una referencia original diaria con particiones registradas"
        )
    final_start = pd.Timestamp(reference.metadata["splits"]["test"]["first_reference"])
    if pd.to_datetime(specification["walk_forward"]["boundaries"][-1], utc=True) != final_start:
        raise DataError("Las ventanas deben terminar antes del test histórico observado")
    raw = settings.root / "data" / "raw.csv"
    if sha256(raw) != reference.metadata["raw_sha256"]:
        raise DataError("El estudio requiere el snapshot crudo exacto de la referencia")
    protected = _protected(settings.root)
    code = code_provenance()
    identity = {
        "protocol_sha256": sha256(protocol),
        "raw_sha256": sha256(raw),
        "model_sha256": reference.metadata["model_sha256"],
        "metadata_sha256": sha256(reference.directory / "metadata.json"),
        "code_sha256": code["source_sha256"],
        "environment": environment(),
    }
    run_id = object_hash(identity)[:24]
    directory = settings.root / "predictive_studies" / run_id
    receipt = directory / "completed.json"
    if receipt.exists():
        completion = read_json(receipt)
        if any(sha256(directory / p) != digest for p, digest in completion["artifacts"].items()):
            raise DataError("El informe guardado está alterado")
        return read_json(directory / "summary.json")
    if directory.exists():
        raise DataError("Hay una ejecución incompleta: preservá su evidencia y revisá el fallo")
    plan = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "identity": identity,
        "protocol": specification,
        "reference_version": reference.metadata["version"],
        "parameters": parameters,
        "code": code,
        "protected_sha256": protected,
        "observed_test_start": final_start.isoformat(),
        "test_role": "descriptive_only_after_development_is_frozen",
        "automatic_promotion": False,
        "hyperparameter_search": False,
    }
    write_new_json(directory / "plan.json", plan)
    start_time = perf_counter()
    combined = replace(base, features=replace(base.features, smc=True))
    dataset = prepare(combined, raw, directory / "prepared.csv")
    frame, _ = load_prepared(dataset, combined.contract())
    original, _ = load_prepared(Path(reference.metadata["dataset_path"]), base.contract())
    pd.testing.assert_frame_equal(frame[original.columns], original, check_exact=True)
    development = evaluate_development(
        frame,
        combined,
        parameters,
        specification,
        final_start,
        directory / "development",
    )
    # Freeze all development decisions before any descriptive test inspection.
    write_new_json(directory / "development.json", development)
    historical = (
        historical_split_diagnostics(reference, frame, specification, directory)
        if specification["diagnostics"]["historical_train_validation_test"]
        else {"status": "disabled"}
    )
    smc = (
        existing_smc_diagnostics(settings.root, final_start)
        if specification["diagnostics"]["reuse_smc_development_scores"]
        else {"status": "disabled"}
    )
    if any(sha256(Path(p)) != digest for p, digest in protected.items()):
        raise DataError("Un modelo o reserva prospectiva cambió durante el estudio")
    summary = {
        "run_id": run_id,
        "directory": str(directory.resolve()),
        "plan_sha256": sha256(directory / "plan.json"),
        "reference_version": reference.metadata["version"],
        "development": development,
        "historical_diagnostics": historical,
        "smc_diagnostics": smc,
        "elapsed_seconds": perf_counter() - start_time,
        "actual_fits": {
            "xgboost": development["fits"]["xgboost"],
            "logistic": development["fits"]["logistic"]
            + int(specification["diagnostics"]["historical_train_validation_test"]),
        },
        "protected_artifacts_unchanged": True,
        "promoted": False,
    }
    write_new_json(directory / "summary.json", summary)
    write_research_report(directory / "RESULTS.md", summary)
    artifacts = {
        str(p.relative_to(directory)): sha256(p) for p in directory.rglob("*") if p.is_file()
    }
    write_new_json(receipt, {"artifacts": artifacts, "run_id": run_id})
    return summary
