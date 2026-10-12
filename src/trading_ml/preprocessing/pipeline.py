from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, sidecar, write_json
from trading_ml.config import Settings
from trading_ml.config.settings import INTERVALS, Features
from trading_ml.errors import DataError
from trading_ml.features import build_features, feature_columns
from trading_ml.preprocessing.quality import clean_ohlcv


def prepare(
    settings: Settings,
    source: Path | None = None,
    output: Path | None = None,
    now: pd.Timestamp | None = None,
) -> Path:
    source = source or settings.root / "data" / "raw.csv"
    provenance = read_json(sidecar(source))
    if provenance.get("kind") != "raw" or provenance.get("sha256") != sha256(source):
        raise DataError("El CSV crudo no coincide con su manifiesto")
    if any(
        provenance["market"][key] != getattr(settings.market, key) for key in ["symbol", "interval"]
    ):
        raise DataError("Activo/intervalo del CSV no coincide con la configuración")
    frame, quality = clean_ohlcv(pd.read_csv(source), settings.delta)
    requested_start = pd.Timestamp(provenance["market"]["start"], tz="UTC").ceil(settings.delta)
    requested_end = pd.Timestamp(provenance["market"]["end"], tz="UTC")
    expected_last = requested_end.ceil(settings.delta) - settings.delta
    if frame.timestamp.iloc[0] != requested_start or frame.timestamp.iloc[-1] != expected_last:
        raise DataError("Cobertura incompleta en los extremos del intervalo solicitado")
    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if (frame.timestamp + settings.delta > current).any():
        raise DataError("No se aceptan velas abiertas ni timestamps futuros")
    result = build_features(frame, settings.features)
    result["reference_time"] = result.timestamp + settings.delta
    future_close = result.close.shift(-settings.training.horizon)
    result["future_return"] = future_close / result.close - 1
    result["target"] = (future_close > result.close).astype(float).where(future_close.notna())
    result["target_time"] = result.reference_time.shift(-settings.training.horizon)
    columns = feature_columns(settings.features)
    complete = result[columns].notna().all(axis=1)
    if complete.any() and not complete.loc[complete.idxmax() :].all():
        raise DataError(
            "Variables faltantes después del calentamiento; no se descartan velas intermedias"
        )
    result = result.loc[complete].reset_index(drop=True)
    if result.empty:
        raise DataError("Datos insuficientes para el calentamiento de indicadores")
    output = output or settings.root / "data" / "prepared.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    write_json(
        sidecar(output),
        {
            "kind": "prepared",
            "market": provenance["market"],
            "contract": settings.contract(),
            "features": columns,
            "sha256": sha256(output),
            "raw_sha256": provenance["sha256"],
            "quality": quality,
            "rows": len(result),
            "warmup_rows_removed": len(frame) - len(result),
            "unlabelled_rows": int(result.target.isna().sum()),
            "flat_label_rows": int((result.future_return == 0).sum()),
        },
    )
    return output


def load_prepared(
    path: Path, contract: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = read_json(sidecar(path))
    if metadata.get("kind") != "prepared" or metadata.get("sha256") != sha256(path):
        raise DataError("El dataset preparado no coincide con su manifiesto")
    if contract is not None and metadata["contract"] != contract:
        raise DataError("Activo, intervalo, indicadores u horizonte incompatibles con el modelo")
    frame = pd.read_csv(path)
    for column in ["timestamp", "reference_time", "target_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if (
        frame.empty
        or not frame.timestamp.is_monotonic_increasing
        or frame.timestamp.duplicated().any()
    ):
        raise DataError("Dataset vacío, desordenado o duplicado")
    if frame[metadata["features"]].isna().any().any():
        raise DataError("Variables predictivas faltantes")
    expected_features = feature_columns(Features(**metadata["contract"]["features"]))
    if metadata["features"] != expected_features:
        raise DataError("Esquema de variables incompatible con los indicadores configurados")
    if not np.isfinite(frame[expected_features].to_numpy()).all():
        raise DataError("Variables predictivas no finitas")
    delta = pd.Timedelta(INTERVALS[metadata["contract"]["interval"]])
    if (frame.timestamp.diff().iloc[1:] != delta).any() or (
        frame.reference_time != frame.timestamp + delta
    ).any():
        raise DataError("Timeline preparado discontinuo o referencias inválidas")
    labelled = frame.dropna(subset=["target"])
    if (
        not labelled.target.isin([0, 1]).all()
        or (
            labelled.target_time
            != labelled.reference_time + metadata["contract"]["horizon"] * delta
        ).any()
    ):
        raise DataError("Etiquetas o fechas objetivo inválidas")
    return frame, metadata
