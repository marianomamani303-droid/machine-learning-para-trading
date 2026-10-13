from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import (
    code_provenance,
    environment,
    object_hash,
    read_json,
    sha256,
    sidecar,
    write_json,
)
from trading_ml.errors import DataError
from trading_ml.models.calibration import apply_calibration
from trading_ml.models.metadata import normalize_metadata


@dataclass
class ModelBundle:
    model: XGBClassifier
    metadata: dict[str, Any]
    directory: Path

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        columns = self.metadata["features"]
        if not set(columns).issubset(frame.columns):
            raise DataError("Faltan variables del modelo")
        values = frame[columns]
        if not np.isfinite(values.to_numpy()).all():
            raise DataError("Variables no finitas")
        raw = self.model.predict_proba(values)[:, 1]
        return apply_calibration(raw, self.metadata.get("calibration"))


def load_model(root: Path, version: str = "latest") -> ModelBundle:
    models = root / "models"
    if version in {"latest", "active", "champion"}:
        pointer = models / "active.json"
        if not pointer.exists():
            pointer = models / ("champion.json" if version == "champion" else "latest.json")
        if not pointer.exists():
            raise DataError("No hay modelo activo; evaluá y promoví explícitamente una versión")
        version = read_json(pointer)["version"]
    if not re.fullmatch(r"v[0-9A-Za-z_-]+", version):
        raise DataError("Versión de modelo inválida")
    directory = models / version
    metadata = read_json(directory / "metadata.json")
    if (
        metadata["version"] != version
        or sha256(directory / "model.ubj") != metadata["model_sha256"]
    ):
        raise DataError("El artefacto de modelo no coincide con sus metadatos")
    if (
        metadata.get("config_sha256")
        and object_hash(metadata["config"]) != metadata["config_sha256"]
    ):
        raise DataError("La configuración del modelo cambió después del registro")
    if (
        metadata.get("recipe_sha256")
        and object_hash(metadata["training_recipe"]) != metadata["recipe_sha256"]
    ):
        raise DataError("La receta del modelo cambió después del registro")
    model = XGBClassifier()
    model.load_model(directory / "model.ubj")
    if model.get_booster().feature_names != metadata["features"]:
        raise DataError("Las variables del modelo no coinciden con sus metadatos")
    normalized = normalize_metadata(root, metadata)
    active_pointer = models / "active.json"
    if not active_pointer.exists():
        active_pointer = models / "latest.json"
    active_version = read_json(active_pointer)["version"] if active_pointer.exists() else None
    normalized["activation_status"] = "active" if active_version == version else "inactive"
    return ModelBundle(model, normalized, directory)


def publish_candidate(model: XGBClassifier, root: Path, metadata: dict[str, Any]) -> str:
    """Guarda un candidato íntegro sin cambiar latest ni el campeón."""
    if "baseline_gate" not in metadata:
        metadata = {
            **metadata,
            "baseline_gate": {
                "accepted": False,
                "status": "required_baselines_not_evaluated",
                "failed_checks": ["missing_validation_against_constant_prevalence_logistic"],
                "test_used": False,
                "automatic_promotion": False,
            },
            "registration_status": "research_only_baselines_pending",
        }
    version = "v" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    temporary = root / "models" / (".tmp_" + version)
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_model(temporary / "model.ubj")
    metadata = {
        **metadata,
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "model_sha256": sha256(temporary / "model.ubj"),
        "environment": environment(),
        "engine_version": "0.1.0",
        "metadata_schema": 2,
        "registration_status": metadata.get("registration_status", "candidate"),
        "code_provenance": code_provenance(),
        "config_sha256": object_hash(metadata["config"]),
        "dataset_reference_end": metadata.get(
            "dataset_reference_end", metadata["config"]["market"]["end"]
        ),
    }
    if metadata.get("training_recipe"):
        metadata["recipe_sha256"] = object_hash(metadata["training_recipe"])
    # New versions own their dataset snapshot, so processing an expanded input
    # cannot silently overwrite the parent's training/evaluation evidence.
    if metadata.get("dataset_path"):
        source = Path(metadata["dataset_path"])
        manifest = read_json(sidecar(source))
        if (
            manifest.get("sha256") != sha256(source)
            or metadata.get("prepared_sha256") != manifest["sha256"]
        ):
            raise DataError("El dataset de registro no coincide con su procedencia")
        snapshot = temporary / "data" / "prepared.csv"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        write_json(sidecar(snapshot), manifest)
        metadata["source_dataset_path"] = str(source.resolve())
        metadata["dataset_path"] = str(
            (root / "models" / version / "data" / "prepared.csv").resolve()
        )
        metadata["dataset_manifest"] = manifest
    write_json(temporary / "metadata.json", metadata)
    write_json(
        temporary / "test_metrics.json", metadata.get("test_metrics", {"status": "not_evaluated"})
    )
    os.replace(temporary, root / "models" / version)
    return version
