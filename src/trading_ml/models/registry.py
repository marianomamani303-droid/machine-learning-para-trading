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

from trading_ml.common import environment, read_json, sha256, write_json
from trading_ml.errors import DataError
from trading_ml.models.calibration import apply_calibration


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
    if version == "latest":
        version = read_json(models / "latest.json")["version"]
    if not re.fullmatch(r"v[0-9A-Za-z_-]+", version):
        raise DataError("Versión de modelo inválida")
    directory = models / version
    metadata = read_json(directory / "metadata.json")
    if (
        metadata["version"] != version
        or sha256(directory / "model.ubj") != metadata["model_sha256"]
    ):
        raise DataError("El artefacto de modelo no coincide con sus metadatos")
    model = XGBClassifier()
    model.load_model(directory / "model.ubj")
    if model.get_booster().feature_names != metadata["features"]:
        raise DataError("Las variables del modelo no coinciden con sus metadatos")
    return ModelBundle(model, metadata, directory)


def publish_candidate(model: XGBClassifier, root: Path, metadata: dict[str, Any]) -> str:
    """Guarda un candidato íntegro sin cambiar latest ni el campeón."""
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
    }
    write_json(temporary / "metadata.json", metadata)
    write_json(
        temporary / "test_metrics.json", metadata.get("test_metrics", {"status": "not_evaluated"})
    )
    os.replace(temporary, root / "models" / version)
    return version
