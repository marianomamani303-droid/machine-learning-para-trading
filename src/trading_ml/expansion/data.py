from __future__ import annotations

import hashlib
import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from trading_ml.common import read_json, sha256, sidecar, write_json, write_new_json
from trading_ml.data.binance import parse_archive
from trading_ml.errors import DataError
from trading_ml.preprocessing.quality import clean_ohlcv

LOG = logging.getLogger(__name__)
BASE = "https://data.binance.vision/data"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")


def archive(root: Path, url: str, *, refresh=False) -> tuple[bytes, dict]:
    """Content addressed payloads and immutable retrieval receipts; no credentials."""
    directory = root / "expansion" / "cache" / hashlib.sha256(url.encode()).hexdigest()
    pointer = directory / "current.json"
    if pointer.exists() and not refresh:
        meta = read_json(pointer)
        path = directory / (meta["sha256"] + ".zip")
        if sha256(path) != meta["sha256"]:
            raise DataError("Archivo externo alterado")
        return path.read_bytes(), meta
    for attempt in range(3):
        try:
            checksum = requests.get(url + ".CHECKSUM", timeout=(5, 25))
            if checksum.status_code == 429:
                raise DataError("Proveedor limitado HTTP429; detener y reanudar más tarde")
            checksum.raise_for_status()
            digest = checksum.text.split()[0].lower()
            response = requests.get(url, timeout=(5, 25))
            if response.status_code == 429:
                raise DataError("Proveedor limitado HTTP429; detener y reanudar más tarde")
            response.raise_for_status()
            payload = response.content
            if hashlib.sha256(payload).hexdigest() != digest:
                raise DataError("SHA256 externo no coincide con el proveedor")
            break
        except requests.RequestException as exc:
            if attempt == 2 or (
                getattr(exc, "response", None) is not None and exc.response.status_code == 404
            ):
                raise DataError(f"Fuente no disponible: {url} ({type(exc).__name__})") from exc
            time.sleep(attempt + 1)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + ".zip")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(payload)
    elif sha256(path) != digest:
        raise DataError("Versión histórica externa alterada")
    meta = {
        "url": url,
        "sha256": digest,
        "upstream_sha256": digest,
        "bytes": len(payload),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "transport": "https",
        "payload": str(path.resolve()),
        "timezone": "UTC",
    }
    receipt = directory / f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{digest[:12]}.json"
    if not receipt.exists():
        write_new_json(receipt, meta)
    write_json(pointer, meta)
    return payload, meta


def csv_in_zip(payload: bytes, *, header=True) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        names = [n for n in zipped.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise DataError("ZIP externo sin un único CSV")
        return pd.read_csv(zipped.open(names[0]), header=0 if header else None)


def parse_mark_archive(payload: bytes) -> pd.DataFrame:
    f = csv_in_zip(payload, header=False)
    if f.shape[1] != 12:
        raise DataError("Formato de precios de marca inesperado")
    if str(f.iloc[0, 0]).lower() in {"open_time", "opentime"}:
        f = f.iloc[1:].reset_index(drop=True)
    numbers = f.iloc[:, :5].apply(pd.to_numeric, errors="raise")
    numbers.columns = ["timestamp", "open", "high", "low", "close"]
    unit = "us" if numbers.timestamp.median() > 100_000_000_000_000 else "ms"
    numbers["timestamp"] = pd.to_datetime(numbers.timestamp, unit=unit, utc=True)
    return numbers


def snapshot(root: Path, symbol: str, kind: str, frame: pd.DataFrame, metadata: dict) -> Path:
    content = frame.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    path = root / "expansion" / "data" / symbol / f"{kind}_{digest[:20]}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != digest or read_json(sidecar(path))["sha256"] != digest:
            raise DataError("Snapshot externo alterado")
    else:
        with path.open("xb") as stream:
            stream.write(content)
        write_new_json(
            sidecar(path),
            {
                **metadata,
                "sha256": digest,
                "rows": len(frame),
                "kind": kind,
                "symbol": symbol,
                "timezone": "UTC",
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    return path


def load_snapshot(path: Path) -> tuple[pd.DataFrame, dict]:
    meta = read_json(sidecar(path))
    if sha256(path) != meta["sha256"]:
        raise DataError("SHA256 de snapshot de expansión inválido")
    f = pd.read_csv(path, float_precision="round_trip")
    for col in ["timestamp", "event_time", "available_at", "reference_time"]:
        if col in f:
            f[col] = pd.to_datetime(f[col], utc=True, format="ISO8601")
    if meta.get("kind") == "funding":
        validate_funding(f)
    return f, meta


def validate_funding(frame: pd.DataFrame) -> None:
    interval = frame.funding_interval_hours
    permitted = pd.concat([interval, interval.shift(1)], axis=1).max(axis=1)
    gaps = frame.event_time.diff().dt.total_seconds() / 3600
    if (
        not np.isfinite(frame.funding_rate).all()
        or not np.isfinite(interval).all()
        or (interval <= 0).any()
        or frame.event_time.duplicated().any()
        or not frame.event_time.is_monotonic_increasing
        or (gaps > permitted + 1 / 60).any()
    ):
        raise DataError("Funding incompleto o inválido respecto de su frecuencia; no se rellena")


def acquire_crypto(root: Path, symbol: str, start: str, end: str) -> dict:
    first, last = pd.to_datetime([start, end], utc=True)
    if (
        symbol not in SYMBOLS
        or first >= last
        or last > pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
        or last > pd.Timestamp("2026-10-01", tz="UTC")
    ):
        raise DataError("Universo/rango de adquisición no permitido para desarrollo")
    months = pd.period_range(
        first.tz_localize(None).to_period("M"),
        (last - pd.Timedelta(days=1)).tz_localize(None).to_period("M"),
        freq="M",
    )
    result = {}
    for kind in ["spot", "perp", "funding"]:

        def fetch(month, kind=kind):
            name = f"{symbol}-{'fundingRate-' if kind == 'funding' else '1d-'}{month}.zip"
            if kind == "spot":
                legacy = root / "cache" / name
                if legacy.exists() and legacy.with_suffix(".zip.CHECKSUM").exists():
                    digest = sha256(legacy)
                    if (
                        legacy.with_suffix(".zip.CHECKSUM").read_text(encoding="ascii").split()[0]
                        != digest
                    ):
                        raise DataError("Cache spot anterior alterada")
                    return legacy.read_bytes(), {
                        "url": f"{BASE}/spot/monthly/klines/{symbol}/1d/{name}",
                        "sha256": digest,
                        "payload": str(legacy.resolve()),
                    }
                url = f"{BASE}/spot/monthly/klines/{symbol}/1d/{name}"
            elif kind == "perp":
                url = f"{BASE}/futures/um/monthly/klines/{symbol}/1d/{name}"
            else:
                url = f"{BASE}/futures/um/monthly/fundingRate/{symbol}/{name}"
            return archive(root, url)

        LOG.info("Adquirir %s %s: %s meses, máximo 3 conexiones", symbol, kind, len(months))
        # Bounded workers; worker exceptions are propagated, never silently skip months.
        with ThreadPoolExecutor(max_workers=3) as pool:
            items = list(pool.map(fetch, months))
        frames = []
        for payload, _ in items:
            if kind == "spot":
                f = parse_archive(payload, include_microstructure=True)
            elif kind == "perp":
                f = parse_mark_archive(payload)
            else:
                f = csv_in_zip(payload).rename(
                    columns={"calc_time": "event_time", "last_funding_rate": "funding_rate"}
                )
                f["event_time"] = pd.to_datetime(f.event_time, unit="ms", utc=True)
            frames.append(f)
        f = pd.concat(frames, ignore_index=True)
        clock = "event_time" if kind == "funding" else "timestamp"
        f = f.loc[(f[clock] >= first) & (f[clock] < last)].sort_values(clock).reset_index(drop=True)
        recovered = []
        if kind != "funding":
            expected = pd.date_range(first, last, inclusive="left", freq="D")
            missing = expected.difference(pd.DatetimeIndex(f.timestamp))
            for day in missing:
                folder = "spot" if kind == "spot" else "futures/um"
                url = f"{BASE}/{folder}/daily/klines/{symbol}/1d/{symbol}-1d-{day:%Y-%m-%d}.zip"
                payload, evidence = archive(root, url)
                daily = (
                    parse_archive(payload, include_microstructure=True)
                    if kind == "spot"
                    else parse_mark_archive(payload)
                )
                if len(daily) != 1 or daily.timestamp.iloc[0] != day:
                    raise DataError("Recuperación diaria no corresponde al hueco declarado")
                f = pd.concat([f, daily], ignore_index=True)
                items.append((payload, evidence))
                recovered.append(day.isoformat())
            f = f.sort_values(clock).reset_index(drop=True)
        if f[clock].duplicated().any() or f.empty:
            raise DataError("Fuente vacía o timestamps duplicados")
        if kind == "spot":
            clean, quality = clean_ohlcv(f, pd.Timedelta(days=1))
            if not np.allclose(
                clean[["open", "high", "low", "close", "volume"]],
                f[["open", "high", "low", "close", "volume"]],
            ):
                raise DataError("Limpieza cambia datos spot")
            if (
                not np.isfinite(f[["quote_volume", "trades", "taker_base", "taker_quote"]])
                .all()
                .all()
                or (f[["quote_volume", "trades", "taker_base", "taker_quote"]] < 0).any().any()
                or (f.trades % 1 != 0).any()
                or (f.taker_base < 0)
                | (f.taker_base > f.volume + 1e-8)
                | (f.taker_quote > f.quote_volume + 1e-6)
                | (f.trades < 0)
            ).any():
                raise DataError("Flujo spot inconsistente")
        elif kind == "perp":
            _, quality = clean_ohlcv(f.assign(volume=0), pd.Timedelta(days=1))
        else:
            validate_funding(f)
            if (
                not np.isfinite(f.funding_rate).all()
                or (f.funding_interval_hours <= 0).any()
                or (f.event_time.diff().dropna() > pd.Timedelta(hours=25)).any()
                or f.event_time.min() - first > pd.Timedelta(hours=25)
                or last - f.event_time.max() > pd.Timedelta(hours=25)
            ):
                raise DataError("Funding incompleto o inválido; no se inventan liquidaciones")
            quality = {
                "max_gap_hours": float(f.event_time.diff().max().total_seconds() / 3600),
                "missing_imputed": 0,
            }
        if kind != "funding":
            if f.timestamp.iloc[0] != first or f.timestamp.iloc[-1] + pd.Timedelta(days=1) != last:
                raise DataError("Cobertura diaria incompleta")
            f["available_at"] = f.timestamp + pd.Timedelta(days=1)
        else:
            f["available_at"] = f.event_time + pd.Timedelta(seconds=1)
        result[kind] = str(
            snapshot(
                root,
                symbol,
                kind,
                f,
                {
                    "start": first.isoformat(),
                    "end_exclusive": last.isoformat(),
                    "frequency": "event" if kind == "funding" else "1d",
                    "instrument_type": "crypto_spot"
                    if kind == "spot"
                    else "crypto_perpetual_information",
                    "availability_policy": "closed daily bar"
                    if kind != "funding"
                    else "settled event plus 1 second assumed dissemination latency",
                    "sources": [m for _, m in items],
                    "quality": {
                        **quality,
                        "missing_days_recovered_from_official_daily_files": recovered,
                    },
                    "legal_source": "Binance public data; retain upstream terms; no redistribution",
                },
            ).resolve()
        )
    return result


def acquire(root: Path, start="2021-01-01", end="2025-07-01") -> Path:
    assets = {}
    for symbol in SYMBOLS:
        assets[symbol] = acquire_crypto(root, symbol, start, end)
    identity = hashlib.sha256(str(assets).encode()).hexdigest()[:16]
    path = root / "expansion" / "datasets" / f"crypto_{identity}.json"
    value = {"start": start, "end": end, "assets": assets}
    if path.exists():
        if read_json(path) != value:
            raise DataError("Índice de datasets inconsistente")
    else:
        write_new_json(path, value)
    return path
