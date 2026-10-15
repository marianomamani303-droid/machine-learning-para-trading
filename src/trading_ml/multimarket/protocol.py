from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd

from trading_ml.config.settings import Settings
from trading_ml.errors import DataError
from trading_ml.multimarket.execution import Costs
from trading_ml.multimarket.markets import instrument
from trading_ml.training.walk_forward import WalkForwardConfig


def load_protocol(path: Path) -> dict:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    allowed = {
        "schema",
        "start",
        "end",
        "boundaries",
        "symbols",
        "intervals",
        "pilot_intervals",
        "seed",
        "embargo_days",
        "calibration_days",
        "min_train_rows",
        "bootstrap_samples",
        "bootstrap_blocks",
        "model",
        "costs",
        "dukascopy_transport",
        "forex_provider",
    }
    if set(value) - allowed or value.get("schema") != 1:
        raise DataError("Protocolo multimercado desconocido")
    WalkForwardConfig(value["boundaries"]).validate()
    dates = pd.to_datetime(value["boundaries"], utc=True)
    if (
        dates[-1] != pd.to_datetime(value["end"], utc=True)
        or pd.to_datetime(value["start"], utc=True) >= dates[0]
    ):
        raise DataError("Fechas del protocolo inconsistentes")
    if not value["symbols"] or len(set(value["symbols"])) != len(value["symbols"]):
        raise DataError("Lista de activos inválida")
    for symbol in value["symbols"]:
        instrument(symbol)
    for key in ["intervals", "pilot_intervals"]:
        if not value[key] or not set(value[key]) <= {"1h", "4h", "1d"}:
            raise DataError("Intervalos no soportados")
    for key in ["embargo_days", "calibration_days", "min_train_rows", "bootstrap_samples"]:
        if not isinstance(value[key], int) or value[key] < 1:
            raise DataError(f"{key} inválido")
    Settings(model=value["model"]).validate()
    Costs(**value["costs"]).validate()
    if value.get("dukascopy_transport", "https") not in {"http", "https"}:
        raise DataError("Transporte Dukascopy inválido")
    if value.get("forex_provider", "dukascopy") not in {"dukascopy", "fxcm"}:
        raise DataError("Proveedor Forex no soportado")
    return value
