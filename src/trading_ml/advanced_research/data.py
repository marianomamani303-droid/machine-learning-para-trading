from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, sidecar
from trading_ml.config import Settings
from trading_ml.data.binance import parse_archive
from trading_ml.errors import DataError
from trading_ml.preprocessing.quality import OHLCV, clean_ohlcv

FLOW_RAW = ["quote_volume", "trades", "taker_base", "taker_quote"]


def recover_microstructure(settings: Settings, source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read verified existing archives only; never refresh or invent missing flow."""
    manifest = read_json(sidecar(source))
    if manifest.get("kind") != "raw" or manifest.get("sha256") != sha256(source):
        raise DataError("Historial de investigación sin procedencia íntegra")
    if any(
        manifest["market"][key] != getattr(settings.market, key) for key in ["symbol", "interval"]
    ):
        raise DataError("Mercado incompatible para recuperación de flujo")
    original, quality = clean_ohlcv(pd.read_csv(source), settings.delta)
    parts = []
    archives = []
    for entry in manifest["sources"]:
        name = entry["url"].rsplit("/", 1)[-1]
        if Path(name).name != name or not name.startswith(
            f"{settings.market.symbol}-{settings.market.interval}-"
        ):
            raise DataError("Nombre de archivo de investigación inválido")
        path = settings.root / "cache" / name
        digest = sha256(path)
        if (
            digest != entry["sha256"]
            or path.with_suffix(".zip.CHECKSUM").read_text(encoding="ascii").strip().split()[0]
            != digest
        ):
            raise DataError("Caché de microestructura alterada")
        parts.append(parse_archive(path.read_bytes(), include_microstructure=True))
        archives.append({"path": str(path.resolve()), "sha256": digest})
    expanded = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    expanded = expanded.loc[expanded.timestamp.isin(original.timestamp)].reset_index(drop=True)
    if (
        len(expanded) != len(original)
        or expanded.timestamp.duplicated().any()
        or not (expanded.timestamp == original.timestamp).all()
    ):
        raise DataError("Flujo faltante o desalineado con OHLCV original")
    if not np.allclose(expanded[OHLCV], original[OHLCV], rtol=1e-11, atol=1e-10):
        raise DataError("Los ZIP no reproducen los precios/volúmenes existentes")
    values = expanded[FLOW_RAW].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise DataError("Microestructura faltante o inválida")
    if (
        (expanded.trades % 1 != 0).any()
        or (expanded.taker_base > expanded.volume + 1e-8).any()
        or (expanded.taker_quote > expanded.quote_volume + 1e-6).any()
    ):
        raise DataError("Volumen taker/trades inconsistente")
    expanded["reference_time"] = expanded.timestamp + settings.delta
    if (expanded.reference_time > pd.Timestamp.now(tz="UTC")).any():
        raise DataError("No se aceptan velas abiertas para flujo")
    return expanded, {
        "source_sha256": sha256(source),
        "archives": archives,
        "quality": quality,
        "missing_flow_imputed": False,
        "original_csv_modified": False,
    }
