from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.common import object_hash, sha256, write_new_json
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.models import ModelBundle
from trading_ml.preprocessing import load_prepared


def out_of_sample(bundle: ModelBundle, frame: pd.DataFrame, labelled: bool = True) -> pd.DataFrame:
    cutoff = pd.Timestamp(bundle.metadata["selection_label_end"])
    test_start = pd.Timestamp(bundle.metadata["splits"]["test"]["first_reference"])
    result = frame.loc[
        (frame.reference_time > cutoff) & (frame.reference_time >= test_start)
    ].copy()
    if labelled:
        result = result.dropna(subset=["target", "target_time"])
    test = bundle.metadata["splits"]["test"]
    if test.get("status", "").startswith("pending") and test.get("last_reference"):
        last = pd.Timestamp(test["last_reference"])
        result = result.loc[result.reference_time <= last]
        if labelled:
            from trading_ml.config.settings import INTERVALS

            expected = pd.date_range(
                test_start, last, freq=INTERVALS[bundle.metadata["contract"]["interval"]]
            )
            if (
                len(result) != len(expected)
                or not (pd.DatetimeIndex(result.reference_time) == expected).all()
            ):
                raise DataError(
                    "Evaluación pendiente: falta completar la ventana prospectiva reservada"
                )
    if result.empty:
        raise LeakageError("No hay filas posteriores a entrenamiento y selección para evaluar")
    return result


def evaluate(bundle: ModelBundle, dataset: Path, output: Path | None = None) -> dict[str, Any]:
    frame, provenance = load_prepared(dataset, bundle.metadata["contract"])
    test = out_of_sample(bundle, frame)
    metrics = classification_metrics(test.target.to_numpy(), bundle.probabilities(test))
    report = {
        "kind": "out_of_sample_evaluation",
        "evaluation_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "version": bundle.metadata["version"],
        "model_sha256": sha256(bundle.directory / "model.ubj"),
        "metadata_sha256": sha256(bundle.directory / "metadata.json"),
        "config_sha256": object_hash(bundle.metadata["config"]),
        "dataset_sha256": provenance["sha256"],
        "dataset_path": str(dataset.resolve()),
        "contract": bundle.metadata["contract"],
        "selection_label_end": bundle.metadata["selection_label_end"],
        "backtest_settings": bundle.metadata["config"]["backtest"],
        "first_reference": test.reference_time.iloc[0].isoformat(),
        "last_reference": test.reference_time.iloc[-1].isoformat(),
        "last_target_time": test.target_time.max().isoformat(),
        "metrics": metrics,
    }
    if output:
        write_new_json(output, report)
        receipt = (
            bundle.directory.parent.parent / "evaluations" / (report["evaluation_id"] + ".json")
        )
        write_new_json(
            receipt,
            {
                "report_sha256": sha256(output),
                "report_path": str(output.resolve()),
                "version": report["version"],
            },
        )
    return report
