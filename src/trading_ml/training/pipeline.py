from __future__ import annotations

import itertools
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import frame_hash, object_hash, sha256
from trading_ml.config import Settings
from trading_ml.config.settings import Market, settings_from_dict
from trading_ml.errors import DataError
from trading_ml.evaluation.baseline_gate import validation_baseline_report
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.models import load_model
from trading_ml.models.calibration import apply_calibration
from trading_ml.models.registry import publish_candidate
from trading_ml.preprocessing import load_prepared
from trading_ml.training.recipe import chronological_recipe, protect_reserved_labels, resolve_recipe
from trading_ml.training.split import chronological_split
from trading_ml.training.walk_forward import WalkForwardConfig

LOGGER = logging.getLogger(__name__)


def _bounds(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "first_reference": frame.reference_time.iloc[0].isoformat(),
        "last_reference": frame.reference_time.iloc[-1].isoformat(),
        "last_target_time": frame.target_time.max().isoformat(),
    }


def train(
    settings: Settings,
    dataset: Path | None = None,
    parent_version: str | None = None,
    *,
    recipe: dict[str, Any] | None = None,
) -> str:
    """Fit/select using past data and register a candidate. Never evaluate or activate."""
    settings.validate()
    dataset = dataset or settings.root / "data" / "prepared.csv"
    frame, provenance = load_prepared(dataset, settings.contract())
    if "market" not in provenance:
        raise DataError("El manifiesto preparado no registra el rango de mercado; ejecutá process")
    settings = replace(settings, market=Market(**provenance["market"]))
    recipe = recipe or chronological_recipe(settings)
    if recipe.get("schema") != 1 or recipe.get("training") != settings.to_dict()["training"]:
        raise DataError("Receta incompatible con la configuración temporal")
    if recipe.get("signal_threshold") != settings.backtest.threshold:
        raise DataError("Umbral de receta incompatible con el backtest registrado")
    replace(settings, model=recipe["parameters"], search=recipe["search"]).validate()
    settings = replace(settings, model=recipe["parameters"], search=recipe["search"])
    columns = provenance["features"]
    trials = []
    calibration = None
    fitting: dict[str, Any] = {}
    if recipe["kind"] == "chronological_search":
        split = chronological_split(frame, settings.training)
        protect_reserved_labels(settings.root, split.train, split.validation)
        candidates = [recipe["parameters"].copy()]
        search = recipe["search"]
        if search.get("enabled", False):
            candidates = [
                {**recipe["parameters"], "max_depth": depth, "learning_rate": rate}
                for depth, rate in itertools.product(
                    search.get("max_depth", [recipe["parameters"]["max_depth"]]),
                    search.get("learning_rate", [recipe["parameters"]["learning_rate"]]),
                )
            ]
        selected = None
        best_loss = float("inf")
        parameters = None
        for params in candidates:
            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                random_state=settings.training.seed,
                n_jobs=1,
            )
            model.fit(split.train[columns], split.train.target.astype(int))
            metrics = classification_metrics(
                split.validation.target.to_numpy(),
                model.predict_proba(split.validation[columns])[:, 1],
            )
            trials.append({"parameters": params, "validation": metrics})
            if metrics["log_loss"] < best_loss:
                best_loss, selected, parameters = metrics["log_loss"], model, params
        if selected is None:
            raise DataError("No se pudo entrenar ningún candidato")
        known = fit_frame = split.train
        cutoff = split.validation.target_time.max()
        partitions = {
            "train": _bounds(split.train),
            "validation": _bounds(split.validation),
            "test": _bounds(split.test),
        }
        purged = split.purged_rows
        hashes = {"fit": frame_hash(fit_frame), "selection": frame_hash(split.validation)}
        baseline_report = validation_baseline_report(
            split.train,
            split.validation,
            columns,
            selected.predict_proba(split.validation[columns])[:, 1],
            settings.training.seed,
        )
    elif recipe["kind"] == "temporal_refit":
        from trading_ml.training.smc_study import fit_smc_model

        known = frame.dropna(subset=["target", "target_time"])
        start = pd.Timestamp(recipe["partition"]["training_start"], tz="UTC")
        rolling = recipe["partition"].get("rolling_days")
        if rolling is not None:
            start = max(start, known.target_time.max() - pd.Timedelta(days=rolling))
        known = known.loc[known.reference_time >= start]
        protect_reserved_labels(settings.root, known)
        wf = WalkForwardConfig(**recipe["walk_forward"])
        wf.validate()
        # Independent outer tail for the recipe gate; ES and calibration stay
        # inside the preceding block. The final refit below uses all known past.
        gate_start = known.reference_time.max() - pd.Timedelta(days=wf.inner_validation_days)
        gate_train = known.loc[known.target_time < gate_start]
        gate_validation = known.loc[known.reference_time >= gate_start]
        gate_model, _, gate_calibration = fit_smc_model(
            gate_train,
            columns,
            recipe["parameters"],
            wf,
            settings.training.seed,
            recipe["early_stop"],
            recipe["calibrated"],
            recipe["calibration_days"],
        )
        baseline_fit = gate_train
        if gate_calibration:
            baseline_fit = gate_train.loc[
                gate_train.target_time < pd.Timestamp(gate_calibration["first_reference"])
            ]
        baseline_report = validation_baseline_report(
            baseline_fit,
            gate_validation,
            columns,
            apply_calibration(
                gate_model.predict_proba(gate_validation[columns])[:, 1], gate_calibration
            ),
            settings.training.seed,
            scope="heldout_tail_recipe_validation_before_final_refit",
        )
        selected, fitting, calibration = fit_smc_model(
            known,
            columns,
            recipe["parameters"],
            wf,
            settings.training.seed,
            recipe["early_stop"],
            recipe["calibrated"],
            recipe["calibration_days"],
        )
        parameters = fitting["effective_parameters"]
        cutoff = known.target_time.max()
        fit_frame = known
        if calibration:
            fit_frame = known.loc[known.target_time < pd.Timestamp(calibration["first_reference"])]
        future_start = cutoff + settings.delta
        partitions = {
            "train": _bounds(fit_frame),
            "validation": {"method": "purged_inner_temporal", **fitting},
            "test": {
                "status": "pending_future_data",
                "first_reference": future_start.isoformat(),
                "last_reference": None,
                "last_target_time": None,
            },
        }
        if calibration:
            partitions["calibration"] = calibration.copy()
        purged = len(known) - len(fit_frame) - (calibration["rows"] if calibration else 0)
        hashes = {"fit": frame_hash(fit_frame), "known_labels": frame_hash(known)}
        if calibration:
            hashes["calibration"] = frame_hash(
                known.loc[known.reference_time >= pd.Timestamp(calibration["first_reference"])]
            )
    else:
        raise DataError("Tipo de receta no soportado")
    metadata = {
        "parent_version": parent_version,
        "config": settings.to_dict(),
        "contract": settings.contract(),
        "features": columns,
        "selected_parameters": parameters,
        "training_recipe": recipe,
        "recipe_sha256": object_hash(recipe),
        "fitting": fitting,
        "calibration": calibration,
        "selection_criterion": "minimum validation log_loss; temporal recipe uses disjoint inner windows",
        "training_data_sha256": frame_hash(fit_frame),
        "partition_sha256": hashes,
        "prepared_sha256": provenance["sha256"],
        "raw_sha256": provenance["raw_sha256"],
        "dataset_path": str(dataset.resolve()),
        "dataset_reference_end": frame.reference_time.max().isoformat(),
        "dataset_manifest": provenance,
        "splits": partitions,
        "purged_rows": purged,
        "selection_label_end": cutoff.isoformat(),
        "trials": trials,
        "test_metrics": {"status": "not_evaluated"},
        "baseline_validation": baseline_report,
        "baseline_gate": baseline_report["decision"],
        "registration_status": "candidate"
        if baseline_report["decision"]["accepted"]
        else "rejected_by_baselines",
        "feature_importance": dict(
            zip(columns, selected.feature_importances_.astype(float).tolist(), strict=True)
        ),
    }
    version = publish_candidate(selected, settings.root, metadata)
    LOGGER.info("Candidato registrado sin evaluación de test ni activación: %s", version)
    return version


def retrain(root: Path, dataset: Path, version: str = "latest") -> str:
    """Reconstruct the complete parent recipe and register a new, inactive candidate."""
    previous = load_model(root, version)
    settings = replace(settings_from_dict(previous.metadata["config"]), root=root)
    frame, _ = load_prepared(dataset, settings.contract())
    if frame.reference_time.max() <= pd.Timestamp(previous.metadata["dataset_reference_end"]):
        raise DataError("Reentrenamiento requiere datos más recientes que el dataset original")
    parent_dataset = Path(previous.metadata["dataset_path"])
    if (
        not parent_dataset.exists()
        or sha256(parent_dataset) != previous.metadata["prepared_sha256"]
    ):
        raise DataError(
            "El snapshot preparado del padre falta o fue sobrescrito; conservá snapshots por versión"
        )
    original, _ = load_prepared(parent_dataset, settings.contract())
    shared = frame.loc[frame.timestamp.isin(original.timestamp)].reset_index(drop=True)
    causal_columns = [
        "timestamp",
        "reference_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *previous.metadata["features"],
    ]
    try:
        pd.testing.assert_frame_equal(
            original[causal_columns].reset_index(drop=True),
            shared[causal_columns],
            check_exact=True,
        )
        labelled = original.target.notna().to_numpy()
        pd.testing.assert_frame_equal(
            original.loc[labelled, ["target", "target_time", "future_return"]].reset_index(
                drop=True
            ),
            shared.loc[labelled, ["target", "target_time", "future_return"]].reset_index(drop=True),
            check_exact=True,
        )
    except (AssertionError, IndexError) as exc:
        raise DataError(
            "El dataset ampliado modificó o eliminó el histórico del modelo padre"
        ) from exc
    recipe = resolve_recipe(root, previous.metadata, settings)
    return train(settings, dataset, parent_version=previous.metadata["version"], recipe=recipe)
