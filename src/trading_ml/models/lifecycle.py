from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.common import object_hash, read_json, sha256, write_json, write_new_json
from trading_ml.errors import DataError, LeakageError
from trading_ml.models.registry import load_model
from trading_ml.preprocessing import load_prepared


def promote(root: Path, version: str, evaluation: Path, reason: str) -> dict[str, Any]:
    """Explicit research activation backed by an unchanged, registered OOS report.

    This records an operator's decision, not a claim of profitability. A single
    authoritative pointer avoids partial updates of latest/champion aliases.
    """
    if not reason.strip():
        raise DataError("La promoción requiere un motivo explícito")
    if version in {"latest", "active", "champion"}:
        raise DataError("La promoción requiere un ID de versión concreto")
    bundle = load_model(root, version)
    if bundle.metadata.get("baseline_gate", {}).get("accepted") is False:
        raise DataError("El candidato fue rechazado por los baselines de validación")
    report = read_json(evaluation)
    evaluation_id = report.get("evaluation_id", "")
    if (
        not isinstance(evaluation_id, str)
        or len(evaluation_id) != 32
        or any(c not in "0123456789abcdef" for c in evaluation_id)
    ):
        raise DataError("Evaluación sin identificador registrado")
    receipt_path = root / "evaluations" / (evaluation_id + ".json")
    if not receipt_path.exists():
        raise DataError("La evaluación no fue registrada por evaluate")
    receipt = read_json(receipt_path)
    expected = {
        "kind": "out_of_sample_evaluation",
        "version": version,
        "model_sha256": sha256(bundle.directory / "model.ubj"),
        "metadata_sha256": sha256(bundle.directory / "metadata.json"),
        "config_sha256": object_hash(bundle.metadata["config"]),
        "contract": bundle.metadata["contract"],
        "selection_label_end": bundle.metadata["selection_label_end"],
        "backtest_settings": bundle.metadata["config"]["backtest"],
    }
    if (
        receipt.get("report_sha256") != sha256(evaluation)
        or receipt.get("version") != version
        or any(report.get(k) != v for k, v in expected.items())
    ):
        raise DataError("Evaluación alterada o incompatible con el modelo/configuración")
    frame, provenance = load_prepared(Path(report["dataset_path"]), bundle.metadata["contract"])
    if provenance["sha256"] != report["dataset_sha256"]:
        raise DataError("El dataset evaluado cambió después de la evaluación")
    first = pd.Timestamp(report["first_reference"])
    last = pd.Timestamp(report["last_reference"])
    if first <= pd.Timestamp(bundle.metadata["selection_label_end"]) or first < pd.Timestamp(
        bundle.metadata["splits"]["test"]["first_reference"]
    ):
        raise LeakageError("La evaluación de promoción invade entrenamiento/selección")
    observed = frame.loc[(frame.reference_time >= first) & (frame.reference_time <= last)].dropna(
        subset=["target", "target_time"]
    )
    if observed.empty or len(observed) != report["metrics"]["rows"]:
        raise DataError("La ventana evaluada no coincide con el dataset")
    # Pending frozen candidates require their complete reserved window.
    test = bundle.metadata["splits"]["test"]
    if test.get("status", "").startswith("pending") and test.get("last_reference"):
        if first != pd.Timestamp(test["first_reference"]) or last != pd.Timestamp(
            test["last_reference"]
        ):
            raise DataError("La promoción requiere la ventana prospectiva completa")
    models = root / "models"
    active = models / "active.json"
    previous = (
        read_json(active)["version"]
        if active.exists()
        else (
            read_json(models / "latest.json")["version"]
            if (models / "latest.json").exists()
            else None
        )
    )
    event = {
        "promotion_id": uuid.uuid4().hex,
        "version": version,
        "previous_version": previous,
        "promoted_at": datetime.now(UTC).isoformat(),
        "reason": reason.strip(),
        "evaluation_id": evaluation_id,
        "evaluation_sha256": sha256(evaluation),
        "model_sha256": expected["model_sha256"],
        "decision": "explicit_research_activation",
    }
    write_new_json(root / "promotions" / (event["promotion_id"] + ".json"), event)
    temporary = models / (".active_" + uuid.uuid4().hex + ".json")
    write_json(temporary, event)
    os.replace(temporary, active)
    return event
