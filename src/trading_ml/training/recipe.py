from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.common import read_json
from trading_ml.config import Settings
from trading_ml.errors import DataError, LeakageError
from trading_ml.models.metadata import legacy_plan
from trading_ml.training.walk_forward import WalkForwardConfig


def chronological_recipe(settings: Settings) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "chronological_search",
        "parameters": settings.model.copy(),
        "search": settings.search.copy(),
        "training": asdict(settings.training),
        "partition": {"method": "chronological_fractions"},
        "early_stop": False,
        "calibrated": False,
        "signal_threshold": settings.backtest.threshold,
    }


def temporal_recipe(
    settings: Settings,
    parameters: dict[str, Any],
    wf: WalkForwardConfig,
    early_stop: bool,
    calibrated: bool = False,
    calibration_days: int = 90,
    rolling_days: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "temporal_refit",
        "parameters": parameters.copy(),
        "training": asdict(settings.training),
        "search": {"enabled": False},
        "partition": {
            "method": "available_labels_then_future_evaluation",
            "training_start": settings.market.start,
            "rolling_days": rolling_days,
        },
        "walk_forward": asdict(wf),
        "early_stop": early_stop,
        "calibrated": calibrated,
        "calibration_days": calibration_days,
        "signal_threshold": settings.backtest.threshold,
    }


def resolve_recipe(root: Path, metadata: dict[str, Any], settings: Settings) -> dict[str, Any]:
    if metadata.get("training_recipe"):
        return metadata["training_recipe"]
    if not metadata.get("fitting") and not metadata.get("calibration"):
        return chronological_recipe(settings)
    plan = legacy_plan(root, metadata)
    if plan is None:
        raise DataError("El modelo no registra una receta completa; falta el plan de entrenamiento")
    if metadata.get("optimization"):
        chosen = metadata["optimization"]["experiment"]
        recipe = next((r for r in plan["recipes"] if r["name"] == chosen), None)
        if recipe is None:
            raise DataError("La receta seleccionada no existe en el plan")
        protocol = plan["protocol"]
        return temporal_recipe(
            settings,
            recipe["parameters"],
            WalkForwardConfig(**protocol["walk_forward"]),
            recipe["early_stop"],
            rolling_days=recipe.get("rolling_days"),
        )
    if metadata.get("smc_study"):
        chosen = metadata["smc_study"]["recipe"]
        recipe = next((r for r in plan["recipes"] if r["id"] == chosen), None)
        protocol = plan["protocol"]
        calibration_days = protocol["calibration"]["days"]
        wf = WalkForwardConfig(**protocol["walk_forward"])
    else:
        directory = Path(metadata.get("source_dataset_path", metadata["dataset_path"])).parent
        summary = read_json(directory / "summary.json")
        chosen = next((k for k, v in summary["versions"].items() if v == metadata["version"]), None)
        recipe = next((r for r in plan["recipes"] if r["id"] == chosen), None)
        calibration_days = plan["calibration_days"]
        wf = WalkForwardConfig(**plan["walk_forward"])
    if recipe is None:
        raise DataError("La versión no está asociada a una receta en el plan histórico")
    return temporal_recipe(
        settings,
        recipe["parameters"],
        wf,
        recipe["early_stop"],
        recipe["calibrated"],
        calibration_days,
    )


def protect_reserved_labels(root: Path, *frames: pd.DataFrame) -> None:
    """The frozen prospective window may never become fitting/selection data."""
    reservations = [read_json(path) for path in (root / "smc_studies").glob("*/prospective.json")]
    research_plans = list((root / "momentum_studies").glob("*/plan.json"))
    research_plans.extend((root / "mean_reversion_studies").glob("*/plan.json"))
    for path in research_plans:
        future = read_json(path).get("prospective")
        if future:
            reservations.append(
                {
                    "first_reference": future["start"],
                    "end_reference_exclusive": future["end"],
                }
            )
    pointer = root / "smc_studies" / "latest.json"
    if pointer.exists():
        summary = read_json(Path(read_json(pointer)["summary"]))
        reservations.append(summary["prospective"])
    for reservation in reservations:
        start = pd.Timestamp(reservation["first_reference"])
        end = pd.Timestamp(reservation["end_reference_exclusive"])
        for frame in frames:
            if ((frame.reference_time < end) & (frame.target_time >= start)).any():
                raise LeakageError("El ajuste o la selección invade el test prospectivo reservado")
