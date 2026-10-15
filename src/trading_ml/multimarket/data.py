from __future__ import annotations

import hashlib
import logging
import lzma
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from trading_ml.common import read_json, sha256, sidecar, write_json, write_new_json
from trading_ml.errors import DataError
from trading_ml.multimarket.markets import fx_open, instrument, session_end

LOG = logging.getLogger(__name__)
FIELDS = ["open", "high", "low", "close"]
# Legacy public BI5 service. Explicit transport provenance: local checksums
# detect later corruption but do not authenticate an HTTP download.
ENDPOINT = "https://datafeed.dukascopy.com/datafeed"
_THREAD = threading.local()
_NETWORK_LOCK = threading.Lock()
_RATE_LIMITED = threading.Event()
_NEXT_GET = 0.0


def decode_hourly(payload: bytes, month: pd.Timestamp, scale: int) -> pd.DataFrame:
    if not isinstance(scale, int) or scale <= 0:
        raise DataError("Escala de precios inválida")
    raw = lzma.decompress(payload)
    if not raw or len(raw) % 24:
        raise DataError("Archivo BI5 vacío o longitud de registro inválida")
    records = np.array(list(struct.iter_unpack(">5if", raw)), dtype=float)
    end = month + pd.offsets.MonthBegin(1)
    seconds = records[:, 0]
    if (
        (seconds % 3600 != 0).any()
        or (seconds < 0).any()
        or (seconds >= (end - month).total_seconds()).any()
    ):
        raise DataError("Timestamps BI5 fuera del mes/grid H1")
    frame = pd.DataFrame(
        {
            "timestamp": month + pd.to_timedelta(seconds, unit="s"),
            "open": records[:, 1] / scale,
            "close": records[:, 2] / scale,
            "low": records[:, 3] / scale,
            "high": records[:, 4] / scale,
            "quote_volume": records[:, 5],
        }
    )
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise DataError("BI5 duplicado o desordenado")
    values = frame[FIELDS + ["quote_volume"]].to_numpy()
    if (
        not np.isfinite(values).all()
        or (frame[FIELDS] <= 0).any().any()
        or (frame.quote_volume < 0).any()
    ):
        raise DataError("BI5 contiene valores inválidos")
    if (
        (frame.low > frame[["open", "close"]].min(axis=1))
        | (frame.high < frame[["open", "close"]].max(axis=1))
        | (frame.high < frame.low)
    ).any():
        raise DataError("OHLC BI5 inconsistente")
    return frame


def _month_file(
    cache: Path,
    symbol: str,
    month: pd.Timestamp,
    side: str,
    refresh: bool,
    transport: str = "https",
) -> tuple[pd.DataFrame, dict]:
    global _NEXT_GET
    endpoint = ENDPOINT.replace("https://", f"{transport}://")
    url = f"{endpoint}/{symbol}/{month.year}/{month.month - 1:02d}/{side}_candles_hour_1.bi5"
    path = cache / symbol / f"{month:%Y-%m}_{side}.bi5"
    manifest_path = sidecar(path)
    mutable = month >= pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
    if path.exists() and manifest_path.exists() and not refresh and not mutable:
        evidence = read_json(manifest_path)
        if sha256(path) != evidence["sha256"] or evidence["url"] != url:
            raise DataError(f"Checksum o procedencia de cache inválido: {path}")
        return decode_hourly(path.read_bytes(), month, instrument(symbol).price_scale), evidence
    for attempt in range(3):
        try:
            if _RATE_LIMITED.is_set():
                raise DataError(
                    "Dukascopy limitado por HTTP429; descarga detenida, reutilice cache o reintente más tarde"
                )
            if not hasattr(_THREAD, "session"):
                _THREAD.session = requests.Session()
            with _NETWORK_LOCK:
                pause = max(0.0, _NEXT_GET - time.monotonic())
                if pause:
                    time.sleep(pause)
                _NEXT_GET = time.monotonic() + 0.5
            response = _THREAD.session.get(url, timeout=(10, 40))
            if response.status_code == 429:
                _RATE_LIMITED.set()
                raise DataError(
                    f"Dukascopy HTTP429: descarga detenida; Retry-After={response.headers.get('Retry-After', 'no informado')}; {url}"
                )
            response.raise_for_status()
            payload = response.content
            frame = decode_hourly(payload, month, instrument(symbol).price_scale)
            break
        except (requests.RequestException, lzma.LZMAError) as exc:
            if attempt == 2:
                raise DataError(f"No se pudo descargar {url}: {type(exc).__name__}") from exc
            time.sleep(1 + attempt)
    evidence = {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "provider": "dukascopy",
        "upstream_checksum_available": False,
        "authenticated_transport": url.startswith("https://"),
        "final_url": response.url,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.parent / "versions" / f"{evidence['sha256']}.bi5"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with archive.open("xb") as stream:
            stream.write(payload)
    elif sha256(archive) != evidence["sha256"]:
        raise DataError("Archivo de proveedor versionado alterado")
    evidence["immutable_archive"] = str(archive.resolve())
    temporary = path.with_suffix(".pending")
    temporary.write_bytes(payload)
    temporary.replace(path)
    write_json(manifest_path, evidence)
    return frame, evidence


def download_forex(
    root: Path,
    symbol: str,
    start: str,
    end: str,
    refresh: bool = False,
    *,
    transport: str = "https",
) -> Path:
    asset = instrument(symbol)
    if asset.market != "forex":
        raise DataError("Este proveedor sólo descarga Forex")
    if transport not in {"https", "http"}:
        raise DataError("Transporte de datos inválido")
    first, last = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if first >= last or last > pd.Timestamp.now(tz="UTC").floor("h"):
        raise DataError("Rango inválido o incluye velas aún abiertas")
    months = pd.date_range(
        first.normalize().replace(day=1),
        (last - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1),
        freq="MS",
    )
    work = [(month, side) for month in months for side in ["BID", "ASK"]]
    LOG.info(
        "Descarga %s: %d archivos mensuales bid/ask (cache incremental)", asset.symbol, len(work)
    )

    def fetch(item):
        month, side = item
        return side, _month_file(
            root / "cache" / "dukascopy", asset.symbol, month, side, refresh, transport
        )

    frames = {"BID": [], "ASK": []}
    evidence = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        iterator = iter(work)
        pending = [pool.submit(fetch, next(iterator)) for _ in range(min(2, len(work)))]
        index = 0
        while pending:
            side, (frame, record) = pending.pop(0).result()
            index += 1
            frames[side].append(frame)
            evidence.append(record)
            if index % 24 == 0:
                LOG.info("%s: %d/%d archivos verificados", asset.symbol, index, len(work))
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(fetch, item))
    bid, ask = [pd.concat(frames[side], ignore_index=True) for side in ["BID", "ASK"]]
    merged = bid.merge(
        ask,
        on="timestamp",
        suffixes=("_bid", "_ask"),
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not (merged._merge == "both").all():
        raise DataError("Horas bid/ask no coinciden")
    merged = merged.drop(columns="_merge").sort_values("timestamp").reset_index(drop=True)
    in_range = (merged.timestamp >= first) & (merged.timestamp + pd.Timedelta(hours=1) <= last)
    zero = (merged.quote_volume_bid == 0) | (merged.quote_volume_ask == 0)
    flat_zero = (
        zero
        & (merged[[f"{f}_bid" for f in FIELDS]].nunique(axis=1) == 1)
        & (merged[[f"{f}_ask" for f in FIELDS]].nunique(axis=1) == 1)
    )
    if (in_range & zero & ~flat_zero).any():
        raise DataError("Cotización con volumen cero pero movimiento: requiere inspección")
    frame = merged.loc[in_range & ~zero].copy()
    # Executable side OHLC, never average independently timed bid/ask extremes.
    for field in FIELDS:
        frame[field] = frame[f"{field}_bid"]
    if ((frame.open_ask < frame.open_bid) | (frame.close_ask < frame.close_bid)).any():
        raise DataError("Spread negativo en apertura/cierre")
    frame["reference_time"] = frame.timestamp + pd.Timedelta(hours=1)
    expected = pd.date_range(first.ceil("h"), last, freq="h", inclusive="left")
    expected = expected[fx_open(expected).to_numpy()]
    actual = pd.DatetimeIndex(frame.timestamp)
    missing = expected.difference(actual)
    outside = actual[~fx_open(actual).to_numpy()]
    diagnostics = {
        "expected_active_hours": len(expected),
        "observed_hours": len(frame),
        "zero_quote_placeholders_removed": int((in_range & zero).sum()),
        "missing_nominal_hours": len(missing),
        "missing_timestamps": [t.isoformat() for t in missing],
        "observed_outside_nominal_calendar": len(outside),
        "outside_timestamps": [t.isoformat() for t in outside],
        "calendar": "NY Sunday17-Friday17; UTC Jan1/Dec25 excluded; holiday early closures remain diagnostic gaps",
        "imputed_rows": 0,
        "volume_semantics": "supplier quote/liquidity volume; not exchange or global FX traded volume; not a model feature",
    }
    return _snapshot(
        root,
        asset.symbol,
        frame,
        {
            "instrument": asset.contract(),
            "start": first.isoformat(),
            "end_exclusive": last.isoformat(),
            "sources": evidence,
            "quality": diagnostics,
            "interval": "1h",
            "schema": "multimarket_bidask_v1",
        },
    )


def _snapshot(root: Path, symbol: str, frame: pd.DataFrame, metadata: dict) -> Path:
    content = frame.to_csv(index=False, float_format="%.12g").encode()
    digest = hashlib.sha256(content).hexdigest()
    path = root / "multimarket" / "data" / symbol / f"{metadata['interval']}_{digest[:20]}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(content)
        write_new_json(
            sidecar(path),
            {
                **metadata,
                "csv_sha256": digest,
                "rows": len(frame),
                "created_utc": datetime.now(UTC).isoformat(),
            },
        )
    elif sha256(path) != digest or read_json(sidecar(path))["csv_sha256"] != digest:
        raise DataError("Snapshot alterado")
    return path


def load_snapshot(path: Path) -> tuple[pd.DataFrame, dict]:
    metadata = read_json(sidecar(path))
    if sha256(path) != metadata["csv_sha256"]:
        raise DataError("Checksum del dataset multimercado inválido")
    frame = pd.read_csv(path)
    for column in ["timestamp", "reference_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame.timestamp.duplicated().any() or not frame.reference_time.is_monotonic_increasing:
        raise DataError("Timeline multimercado inválida")
    return frame, metadata


def execution_source(frame: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, int]:
    """Feature aggregation never removes real prices from the execution path."""
    if metadata["instrument"]["market"] != "forex" or metadata["interval"] == "1h":
        return frame, 1
    parent = Path(metadata["parent_csv"])
    if sha256(parent) != metadata["parent_sha256"]:
        raise DataError("Fuente H1 de ejecución alterada")
    hourly, origin = load_snapshot(parent)
    if origin["interval"] != "1h" or origin["instrument"] != metadata["instrument"]:
        raise DataError("Fuente H1 de ejecución incompatible")
    return hourly, {"4h": 4, "1d": 24}[metadata["interval"]]


def aggregate_forex(root: Path, path: Path, interval: str) -> Path:
    if interval not in ["1h", "4h", "1d"]:
        raise DataError("Forex admite 1h, 4h UTC y 1d NY17; no se inventan subvelas")
    if interval == "1h":
        return path
    frame, metadata = load_snapshot(path)
    if metadata["interval"] != "1h":
        raise DataError("Agregación requiere fuente H1")
    frame["group"] = (
        session_end(pd.DatetimeIndex(frame.timestamp))
        if interval == "1d"
        else frame.timestamp.dt.floor("4h") + pd.Timedelta(hours=4)
    )
    records = []
    incomplete = []
    for end, group in frame.groupby("group", sort=True):
        if interval == "1d":
            local_end = end.tz_convert("America/New_York").tz_localize(None)
            start = (
                (local_end - pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
            )
        else:
            start = end - pd.Timedelta(hours=4)
        required = pd.date_range(start, end, freq="h", inclusive="left")
        # Demand all actual hours in the bucket: never disguise holiday fragments as full bars.
        if not pd.DatetimeIndex(group.timestamp).equals(required):
            incomplete.append(
                {"start": start.isoformat(), "end": end.isoformat(), "hours": len(group)}
            )
            continue
        record = {"timestamp": start, "reference_time": end}
        for suffix in ["", "_bid", "_ask"]:
            record.update(
                {
                    f"open{suffix}": group[f"open{suffix}"].iloc[0],
                    f"high{suffix}": group[f"high{suffix}"].max(),
                    f"low{suffix}": group[f"low{suffix}"].min(),
                    f"close{suffix}": group[f"close{suffix}"].iloc[-1],
                }
            )
        if "quote_volume_bid" in group:
            record["quote_volume_bid"] = float(group.quote_volume_bid.sum())
            record["quote_volume_ask"] = float(group.quote_volume_ask.sum())
        if "crossed_open" in group:
            record["crossed_open"] = bool(group.crossed_open.iloc[0])
            record["crossed_close"] = bool(group.crossed_close.iloc[-1])
        records.append(record)
    if not records:
        raise DataError("No hay velas completas para agregar")
    quality = {
        **metadata["quality"],
        "incomplete_buckets_excluded": incomplete,
        "complete_bars": len(records),
    }
    return _snapshot(
        root,
        metadata["instrument"]["symbol"],
        pd.DataFrame(records),
        {
            **metadata,
            "interval": interval,
            "parent_csv": str(path.resolve()),
            "parent_sha256": sha256(path),
            "quality": quality,
        },
    )
