from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.config.settings import INTERVALS
from trading_ml.errors import LeakageError
from trading_ml.models import ModelBundle
from trading_ml.preprocessing import load_prepared


def predict(
    bundle: ModelBundle,
    dataset: Path,
    reference: str | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Predice una vela cerrada posterior a selección; incluye advertencia de antigüedad."""
    frame, _ = load_prepared(dataset, bundle.metadata["contract"])
    row = (
        frame.iloc[[-1]]
        if reference is None
        else frame.loc[frame.reference_time == pd.to_datetime(reference, utc=True)]
    )
    if len(row) != 1:
        raise LeakageError("La referencia debe identificar exactamente una vela preparada")
    timestamp = row.reference_time.iloc[0]
    cutoff = pd.Timestamp(bundle.metadata["selection_label_end"])
    if timestamp <= cutoff:
        raise LeakageError(
            "No se permite presentar como nueva una predicción del período usado para entrenar o seleccionar"
        )
    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if timestamp > current:
        raise LeakageError("La referencia todavía no está cerrada")
    probability = float(bundle.probabilities(row)[0])
    delta = pd.Timedelta(INTERVALS[bundle.metadata["contract"]["interval"]])
    horizon = bundle.metadata["contract"]["horizon"]
    calibration = bundle.metadata.get("calibration")
    calibration_status = "sigmoid_fitted_on_disjoint_past_window" if calibration else "uncalibrated"
    warnings = ["La clase bajada incluye precios sin cambio; no estima magnitud ni rentabilidad."]
    if calibration:
        warnings.insert(
            0,
            "Calibración sigmoide ajustada en datos pasados disjuntos; puede degradarse por cambios de régimen.",
        )
        if calibration.get("ranking_collapsed"):
            warnings.append(
                "La calibración colapsó a una probabilidad constante; no discrimina dirección."
            )
    else:
        warnings.insert(
            0, "Probabilidades sin calibración posterior; consultar Brier y reliability bins."
        )
    if current - timestamp > delta * 2:
        warnings.append(
            "Datos históricos desactualizados para una decisión actual; descargar y procesar datos recientes."
        )
    return {
        "asset": bundle.metadata["contract"]["symbol"],
        "reference_time": timestamp.isoformat(),
        "interval": bundle.metadata["contract"]["interval"],
        "horizon_bars": horizon,
        "horizon_seconds": int(delta.total_seconds() * horizon),
        "target_reference_time": (timestamp + delta * horizon).isoformat(),
        "probability_up": probability,
        "probability_down": 1 - probability,
        "model_version": bundle.metadata["version"],
        "calibration_status": calibration_status,
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "generated_at": datetime.now(UTC).isoformat(),
    }
