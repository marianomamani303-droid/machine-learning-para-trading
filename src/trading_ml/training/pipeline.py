from __future__ import annotations

import itertools
import logging
import os
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import environment, frame_hash, sha256, write_json
from trading_ml.config import Settings
from trading_ml.config.settings import Market, settings_from_dict
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.models import load_model
from trading_ml.preprocessing import load_prepared
from trading_ml.training.split import chronological_split

LOGGER = logging.getLogger(__name__)


def _bounds(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "first_reference": frame.reference_time.iloc[0].isoformat(),
        "last_reference": frame.reference_time.iloc[-1].isoformat(),
        "last_target_time": frame.target_time.max().isoformat(),
    }


def train(
    settings: Settings, dataset: Path | None = None, parent_version: str | None = None
) -> str:
    """Selecciona por log-loss de validación, mantiene el modelo entrenado sólo en train.

    El test se evalúa una vez después de seleccionar; nunca guía parámetros ni umbral.
    """
    settings.validate()
    dataset = dataset or settings.root / "data" / "prepared.csv"
    frame, provenance = load_prepared(dataset, settings.contract())
    if "market" not in provenance:
        raise DataError(
            "El manifiesto preparado no registra el rango de mercado; ejecutá process nuevamente"
        )
    settings = replace(settings, market=Market(**provenance["market"]))
    split = chronological_split(frame, settings.training)
    columns = provenance["features"]
    candidates = [settings.model.copy()]
    if settings.search.get("enabled", False):
        candidates = [
            {**settings.model, "max_depth": depth, "learning_rate": rate}
            for depth, rate in itertools.product(
                settings.search.get("max_depth", [settings.model["max_depth"]]),
                settings.search.get("learning_rate", [settings.model["learning_rate"]]),
            )
        ]
    selected = None
    best_loss = float("inf")
    selected_params = None
    trials = []
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
            split.validation.target.to_numpy(), model.predict_proba(split.validation[columns])[:, 1]
        )
        trials.append({"parameters": params, "validation": metrics})
        LOGGER.info("Candidato %s; validation log-loss=%.6f", params, metrics["log_loss"])
        if metrics["log_loss"] < best_loss:
            best_loss, selected, selected_params = metrics["log_loss"], model, params
    if selected is None:
        raise DataError("No se pudo entrenar ningún candidato")
    probability = selected.predict_proba(split.test[columns])[:, 1]
    metrics = classification_metrics(split.test.target.to_numpy(), probability)
    baseline = classification_metrics(
        split.test.target.to_numpy(), np.full(len(split.test), float(split.train.target.mean()))
    )
    version = "v" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    models = settings.root / "models"
    temporary = models / (".tmp_" + version)
    temporary.mkdir(parents=True, exist_ok=False)
    selected.save_model(temporary / "model.ubj")
    metadata = {
        "version": version,
        "parent_version": parent_version,
        "created_at": datetime.now(UTC).isoformat(),
        "engine_version": "0.1.0",
        "config": settings.to_dict(),
        "contract": settings.contract(),
        "features": columns,
        "selected_parameters": selected_params,
        "selection_criterion": "minimum validation log_loss; ties retain first candidate",
        "training_data_sha256": frame_hash(split.train),
        "prepared_sha256": provenance["sha256"],
        "raw_sha256": provenance["raw_sha256"],
        "dataset_path": str(dataset.resolve()),
        "environment": environment(),
        "splits": {
            "train": _bounds(split.train),
            "validation": _bounds(split.validation),
            "test": _bounds(split.test),
        },
        "purged_rows": split.purged_rows,
        "selection_label_end": split.validation.target_time.max().isoformat(),
        "trials": trials,
        "test_metrics": metrics,
        "baseline_test_metrics": baseline,
        "model_sha256": sha256(temporary / "model.ubj"),
        "feature_importance": dict(
            zip(columns, selected.feature_importances_.astype(float).tolist(), strict=True)
        ),
    }
    write_json(temporary / "metadata.json", metadata)
    write_json(
        temporary / "test_metrics.json",
        {
            "version": version,
            "model": metrics,
            "constant_train_prevalence_baseline": baseline,
            "first_reference": split.test.reference_time.iloc[0].isoformat(),
            "last_reference": split.test.reference_time.iloc[-1].isoformat(),
        },
    )
    predictions = split.test[
        ["timestamp", "reference_time", "target_time", "close", "target", "future_return"]
    ].copy()
    predictions["probability_up"] = probability
    predictions.to_csv(temporary / "test_predictions.csv", index=False)
    os.replace(temporary, models / version)
    latest_temp = models / f".latest_{uuid.uuid4().hex}.json"
    write_json(latest_temp, {"version": version})
    os.replace(latest_temp, models / "latest.json")
    LOGGER.info("Modelo publicado: %s; test ROC-AUC=%s", version, metrics["roc_auc"])
    return version


def retrain(root: Path, dataset: Path, version: str = "latest") -> str:
    """Reentrena desde cero y exige nuevas velas posteriores al dataset original."""
    previous = load_model(root, version)
    settings = settings_from_dict(previous.metadata["config"])
    frame, provenance = load_prepared(dataset, settings.contract())
    if "market" not in provenance:
        raise DataError(
            "El manifiesto preparado no registra el rango de mercado; ejecutá process nuevamente"
        )
    settings = replace(settings, root=root, market=Market(**provenance["market"]))
    if frame.reference_time.max() <= pd.Timestamp(
        previous.metadata["splits"]["test"]["last_target_time"]
    ):
        raise DataError("Reentrenamiento requiere datos más recientes que el dataset original")
    return train(settings, dataset, parent_version=previous.metadata["version"])
