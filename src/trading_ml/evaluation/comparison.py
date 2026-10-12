from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.models import load_model
from trading_ml.preprocessing import load_prepared


def compare(root: Path, dataset: Path, versions: list[str]) -> dict[str, Any]:
    """Compara versiones sobre exactamente las mismas filas no usadas por ninguna."""
    if len(versions) < 2:
        raise DataError("compare requiere al menos dos versiones")
    bundles = [load_model(root, version) for version in versions]
    contract = bundles[0].metadata["contract"]
    if any(bundle.metadata["contract"] != contract for bundle in bundles):
        raise DataError("Versiones con contratos incompatibles")
    frame, provenance = load_prepared(dataset, contract)
    cutoff = max(pd.Timestamp(bundle.metadata["selection_label_end"]) for bundle in bundles)
    start = max(
        pd.Timestamp(bundle.metadata["splits"]["test"]["first_reference"]) for bundle in bundles
    )
    common = frame.loc[(frame.reference_time > cutoff) & (frame.reference_time >= start)].dropna(
        subset=["target", "target_time"]
    )
    if common.empty:
        raise LeakageError("No existe un período común fuera de muestra")
    return {
        "dataset_sha256": provenance["sha256"],
        "rows": len(common),
        "first_reference": common.reference_time.iloc[0].isoformat(),
        "last_reference": common.reference_time.iloc[-1].isoformat(),
        "models": [
            {
                "version": bundle.metadata["version"],
                "metrics": classification_metrics(
                    common.target.to_numpy(), bundle.probabilities(common)
                ),
            }
            for bundle in bundles
        ],
    }
