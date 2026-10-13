from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.common import read_json, sha256
from trading_ml.config.settings import INTERVALS
from trading_ml.errors import DataError


def legacy_plan(root: Path, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Recover recorded recipes from evidence, never from effective tree counts."""
    if "smc_study" in metadata:
        directory = root / "smc_studies" / metadata["smc_study"]["run_id"]
    elif "optimization" in metadata:
        directory = root / "optimization" / metadata["optimization"]["run_id"]
    elif metadata.get("plan_sha256") and metadata.get("dataset_path"):
        directory = Path(metadata.get("source_dataset_path", metadata["dataset_path"])).parent
    else:
        return None
    path = directory / "plan.json"
    if not path.exists():
        raise DataError("Falta el plan histórico necesario para reconstruir la receta")
    expected = metadata.get("plan_sha256")
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        expected = expected or summary.get("plan_sha256")
    if expected is None or sha256(path) != expected:
        raise DataError("El plan histórico no coincide con su digest registrado")
    return read_json(path)


def normalize_metadata(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Compatibility view only: historical files remain byte-for-byte intact."""
    result = deepcopy(metadata)
    result.setdefault("metadata_schema", 1)
    result.setdefault("dataset_reference_end", result["config"]["market"]["end"])
    result["dataset_reference_end"] = pd.to_datetime(
        result["dataset_reference_end"], utc=True
    ).isoformat()
    test = result["splits"]["test"]
    if not test.get("last_target_time"):
        plan = legacy_plan(root, result)
        end = (plan or {}).get("independent_final_end_exclusive")
        if end:
            delta = pd.Timedelta(INTERVALS[result["contract"]["interval"]])
            last = pd.to_datetime(end, utc=True) - delta
            test.update(
                {
                    "status": "pending_independent_future_data",
                    "end_reference_exclusive": pd.to_datetime(end, utc=True).isoformat(),
                    "last_reference": last.isoformat(),
                    "last_target_time": (last + result["contract"]["horizon"] * delta).isoformat(),
                }
            )
    return result
