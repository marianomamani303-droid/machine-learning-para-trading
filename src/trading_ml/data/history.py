"""Versioned multi-provider history. Acquisition/validation only, never fitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, sidecar, write_new_json
from trading_ml.errors import DataError, LeakageError
from trading_ml.training.recipe import protect_reserved_labels

INTERVALS = {"1m": "1min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D", "1d_ny": "1D"}
PRICES = ["open", "high", "low", "close"]
DEVELOPMENT_END = "2025-07-01"


def validate(frame: pd.DataFrame, interval: str, *, continuous=False) -> dict:
    if frame.empty or interval not in INTERVALS:
        raise DataError("Historia vacía o intervalo desconocido")
    clock = pd.DatetimeIndex(frame.timestamp)
    delta = pd.Timedelta(INTERVALS[interval])
    if clock.tz is None or str(clock.tz) != "UTC":
        raise DataError("La historia debe usar UTC explícito")
    if clock.has_duplicates or not clock.is_monotonic_increasing or clock.hasnans:
        raise DataError("Fechas inválidas, duplicadas o desordenadas")
    local = clock.tz_convert("America/New_York")
    aligned = (
        ((local.hour == 17) & (local.minute == 0) & (local.second == 0) & (local.microsecond == 0))
        if interval == "1d_ny"
        else clock == clock.floor(delta)
    )
    if not aligned.all():
        raise DataError("Fechas fuera de la grilla temporal")
    numeric = PRICES + [c for c in frame if c.startswith("volume") or c.endswith(("_bid", "_ask"))]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise DataError("Valores faltantes o no finitos")
    for suffix in [""] + (["_bid", "_ask"] if "open_bid" in frame else []):
        o, h, low, c = (frame[k + suffix] for k in PRICES)
        if ((low <= 0) | (low > np.minimum(o, c)) | (h < np.maximum(o, c))).any():
            raise DataError("Precios no positivos u OHLC inconsistentes")
    for col in [c for c in frame if c.startswith("volume")]:
        if (frame[col] < 0).any():
            raise DataError("Volumen negativo")
    if (frame.available_at != frame.timestamp + delta).any():
        raise DataError("Disponibilidad anterior o distinta del cierre")
    if (frame.available_at > pd.Timestamp.now(tz="UTC")).any():
        raise DataError("Velas abiertas o fechas futuras")
    gaps = clock.to_series().diff().dropna()
    spacing = (gaps / delta).round() if interval == "1d_ny" else gaps / delta
    missing = int((spacing.astype(int) - 1).clip(lower=0).sum())
    if continuous and missing:
        raise DataError("Historia continua incompleta")
    return {
        "rows": len(frame),
        "first_open": clock[0].isoformat(),
        "last_close": frame.available_at.iloc[-1].isoformat(),
        "missing_grid_slots": missing,
        "gap_count": int((gaps > delta).sum()),
        "max_gap_seconds": float(gaps.max().total_seconds()) if len(gaps) else 0,
        "duplicates": 0,
        "missing_values": 0,
        "imputed_rows": 0,
        "crossed_open_quotes": int((frame.open_ask < frame.open_bid).sum())
        if "open_bid" in frame
        else 0,
        "crossed_close_quotes": int((frame.close_ask < frame.close_bid).sum())
        if "open_bid" in frame
        else 0,
        "calendar_note": "Native NY17 session dates; weekends/DST are not provider outages."
        if interval == "1d_ny"
        else "Grid gaps are measured, not assumed exchange outages; closures need the market calendar.",
    }


def aggregate(frame: pd.DataFrame, source: str, target: str) -> tuple[pd.DataFrame, dict]:
    """Aggregate observed complete UTC windows only. Never upsample or fill."""
    if "1d_ny" in {source, target}:
        raise DataError("NY17 requiere fuente nativa o agregador de sesiones, no ventanas UTC")
    small, large = pd.Timedelta(INTERVALS[source]), pd.Timedelta(INTERVALS[target])
    if large < small or large % small:
        raise DataError("No se puede generar una temporalidad menor ni fraccionaria")
    validate(frame, source)
    data = frame.set_index("timestamp")
    mapping = {}
    for suffix in [""] + (["_bid", "_ask"] if "open_bid" in frame else []):
        mapping.update(
            {
                "open" + suffix: "first",
                "high" + suffix: "max",
                "low" + suffix: "min",
                "close" + suffix: "last",
            }
        )
    mapping.update({c: "sum" for c in frame if c.startswith("volume")})
    groups = data.resample(large, closed="left", label="left", origin="epoch")
    result = groups.agg(mapping)
    count = groups.size()
    complete = count == int(large / small)
    result = result.loc[complete].reset_index()
    result["available_at"] = result.timestamp + large
    return result, {
        "incomplete_windows_excluded": int(((count > 0) & ~complete).sum()),
        "source_interval": source,
        "alignment": "UTC; full observed windows only",
    }


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicates = frame.loc[frame.timestamp.duplicated(keep=False)]
    columns = [c for c in frame if c != "timestamp"]
    if (
        len(duplicates)
        and (duplicates.groupby("timestamp")[columns].nunique(dropna=False) > 1).any().any()
    ):
        raise DataError("Observaciones duplicadas contradictorias")
    removed = int(frame.timestamp.duplicated().sum())
    return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(
        drop=True
    ), removed


def save(root: Path, frame: pd.DataFrame, metadata: dict) -> dict:
    quality = validate(frame, metadata["interval"])
    payload = frame.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    directory = (
        root
        / "historical"
        / "datasets"
        / metadata["provider"]
        / metadata["symbol"].replace("^", "INDEX_")
    )
    path = directory / f"{metadata['interval']}_{digest[:20]}.csv"
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != digest or (
            sidecar(path).exists() and read_json(sidecar(path))["sha256"] != digest
        ):
            raise DataError("Snapshot histórico alterado")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    if not sidecar(path).exists():
        write_new_json(
            sidecar(path),
            {
                **metadata,
                "schema": "historical_ohlc_v1",
                "sha256": digest,
                "quality": quality,
                "timezone": "UTC",
                "no_imputation": True,
            },
        )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "symbol": metadata["symbol"],
        "provider": metadata["provider"],
        "market_type": metadata["market_type"],
        "interval": metadata["interval"],
        **quality,
    }


def load_history(
    path: Path, *, root: Path, start: str | None = None, end: str, horizon: int = 1
) -> tuple[pd.DataFrame, dict]:
    """Explicit opt-in adapter. Restrict inputs BEFORE features and future labels.

    There is intentionally no training bypass for quarantined recent history.
    Session markets retain real gaps and require a calendar-aware experiment.
    """
    meta = read_json(sidecar(path))
    if meta.get("schema") != "historical_ohlc_v1" or sha256(path) != meta["sha256"]:
        raise DataError("Integridad de dataset histórico inválida")
    stop = pd.to_datetime(end, utc=True)
    cutoff = min(
        pd.to_datetime(meta["development_end"], utc=True), pd.Timestamp(DEVELOPMENT_END, tz="UTC")
    )
    if stop > cutoff or horizon < 1:
        raise LeakageError("Rango invade datos en cuarentena o horizonte inválido")
    data = pd.read_csv(path, float_precision="round_trip")
    for col in ["timestamp", "available_at"]:
        data[col] = pd.to_datetime(data[col], utc=True)
    mask = data.available_at <= stop
    if start is not None:
        mask &= data.timestamp >= pd.to_datetime(start, utc=True)
    data = data.loc[mask].reset_index(drop=True)
    validate(data, meta["interval"])
    if meta["market_type"] == "crypto_dated_futures":
        delivery = pd.Timestamp("20" + meta["symbol"][-6:], tz="UTC") + pd.Timedelta(hours=8)
        if (data.available_at > delivery).any():
            raise DataError(
                "El histórico incluye velas posteriores al vencimiento; usar la versión depurada"
            )
    if meta["market_type"] == "cash_index_context" and (data.high <= data.low).any():
        raise DataError("Índice con rango OHLC diario no verificable; usar la versión depurada")
    delta = pd.Timedelta(INTERVALS[meta["interval"]])
    guard = pd.DataFrame(
        {"reference_time": data.available_at, "target_time": data.available_at.shift(-horizon)}
    )
    # Tail labels are unavailable; conservative boundary guard for all rows.
    guard["target_time"] = guard.target_time.fillna(data.available_at + horizon * delta)
    protect_reserved_labels(root, guard)
    return data, meta


def export_legacy(path: Path, output: Path, *, root: Path, start: str, end: str) -> Path:
    """Produce an immutable raw input compatible with the existing prepare()."""
    data, meta = load_history(path, root=root, start=start, end=end)
    if meta["market_type"] != "crypto_spot" or "volume" not in data:
        raise DataError("El pipeline OHLCV continuo sólo admite spot con volumen negociado")
    validate(data, meta["interval"], continuous=True)
    first, stop = pd.to_datetime([start, end], utc=True)
    if data.timestamp.iloc[0] != first or data.available_at.iloc[-1] != stop:
        raise DataError("El export requiere cobertura completa en ambos extremos")
    payload = data[["timestamp", *PRICES, "volume"]].to_csv(index=False).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    write_new_json(
        sidecar(output),
        {
            "kind": "raw",
            "provider": meta["provider"],
            "market": {
                "symbol": meta["symbol"],
                "interval": meta["interval"],
                "start": start,
                "end": end,
            },
            "rows": len(data),
            "sha256": sha256(output),
            "historical_source": str(path.resolve()),
            "historical_sha256": meta["sha256"],
        },
    )
    return output


def export_forex(path: Path, *, root: Path, start: str, end: str) -> Path:
    """Adapt H1 bid/ask to the existing calendar-aware multimarket pipeline.

    The legacy daily adapter still aggregates at NY17. A historical UTC D1
    snapshot is never silently substituted for that different session definition.
    """
    from trading_ml.multimarket.data import _snapshot
    from trading_ml.multimarket.markets import instrument

    data, meta = load_history(path, root=root, start=start, end=end)
    if meta["market_type"] != "indicative_otc_forex" or meta["interval"] != "1h":
        raise DataError("El adaptador Forex requiere H1 bid/ask real")
    if not all(col + suffix in data for suffix in ["_bid", "_ask"] for col in PRICES):
        raise DataError("Faltan cotizaciones bid/ask")
    data = data.rename(columns={"available_at": "reference_time"})
    data["crossed_open"] = data.open_ask < data.open_bid
    data["crossed_close"] = data.close_ask < data.close_bid
    return _snapshot(
        root / "historical" / "adapters",
        meta["symbol"],
        data,
        {
            "schema": "multimarket_bidask_v1",
            "instrument": instrument(meta["symbol"]).contract() | {"provider": "fxcm"},
            "interval": "1h",
            "start": start,
            "end_exclusive": end,
            "quality": meta["quality"],
            "historical_parent": str(path.resolve()),
            "historical_parent_sha256": meta["sha256"],
        },
    )
