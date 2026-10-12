from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.common import write_json
from trading_ml.errors import LeakageError
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
    if result.empty:
        raise LeakageError("No hay filas posteriores a entrenamiento y selección para evaluar")
    return result


def evaluate(bundle: ModelBundle, dataset: Path, output: Path | None = None) -> dict[str, Any]:
    frame, provenance = load_prepared(dataset, bundle.metadata["contract"])
    test = out_of_sample(bundle, frame)
    metrics = classification_metrics(test.target.to_numpy(), bundle.probabilities(test))
    report = {
        "version": bundle.metadata["version"],
        "dataset_sha256": provenance["sha256"],
        "first_reference": test.reference_time.iloc[0].isoformat(),
        "last_reference": test.reference_time.iloc[-1].isoformat(),
        "metrics": metrics,
    }
    if output:
        write_json(output, report)
    return report
