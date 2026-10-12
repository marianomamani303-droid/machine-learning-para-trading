from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from trading_ml.common import sha256, sidecar, write_json
from trading_ml.config import Settings
from trading_ml.errors import DataError

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


def parse_archive(payload: bytes) -> pd.DataFrame:
    """Lee un ZIP sin extraerlo; reconoce milisegundos y microsegundos."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise DataError("El archivo debe contener exactamente un CSV")
        frame = pd.read_csv(archive.open(names[0]), header=None, names=COLUMNS)
    ticks = pd.to_numeric(frame["timestamp"], errors="raise")
    unit = "us" if ticks.median() > 100_000_000_000_000 else "ms"
    frame["timestamp"] = pd.to_datetime(ticks, unit=unit, utc=True)
    return frame[["timestamp", "open", "high", "low", "close", "volume"]]


def _month(settings: Settings, month: str, refresh: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    name = f"{settings.market.symbol}-{settings.market.interval}-{month}.zip"
    cache = settings.root / "cache" / name
    checksum_path = cache.with_suffix(".zip.CHECKSUM")
    url = f"{BASE_URL}/{settings.market.symbol}/{settings.market.interval}/{name}"
    with requests.Session() as session:
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        try:
            if refresh or not cache.exists() or not checksum_path.exists():
                response = session.get(url + ".CHECKSUM", timeout=(10, 60))
                response.raise_for_status()
                checksum = response.text.split()[0].lower()
                response = session.get(url, timeout=(10, 60))
                response.raise_for_status()
                payload = response.content
                if hashlib.sha256(payload).hexdigest() != checksum:
                    raise DataError(f"Checksum incorrecto: {url}")
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(payload)
                checksum_path.write_text(checksum, encoding="ascii")
            else:
                payload = cache.read_bytes()
                checksum = checksum_path.read_text(encoding="ascii").strip()
                if hashlib.sha256(payload).hexdigest() != checksum:
                    raise DataError(f"Caché corrupta: {cache}; descargá con --refresh")
        except requests.RequestException as exc:
            raise DataError(
                f"No se pudo descargar {url}: {exc}. Los archivos mensuales se publican luego del cierre del mes; elegí un end anterior o reintentá."
            ) from exc
    LOGGER.info("Archivo verificado: %s", name)
    return parse_archive(payload), {"url": url, "sha256": checksum}


def download(settings: Settings, output: Path | None = None, refresh: bool = False) -> Path:
    """Descarga archivos mensuales reales en [start, end), sin velas abiertas."""
    settings.validate()
    start = pd.Timestamp(settings.market.start, tz="UTC")
    end = pd.Timestamp(settings.market.end, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    if end > now.normalize().replace(day=1):
        raise DataError(
            "Este proveedor usa meses cerrados: end no puede superar el inicio del mes actual UTC"
        )
    months = pd.period_range(
        start.tz_localize(None).to_period("M"),
        (end - pd.Timedelta(nanoseconds=1)).tz_localize(None).to_period("M"),
        freq="M",
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(lambda month: _month(settings, str(month), refresh), months))
    frame = pd.concat([part[0] for part in parts], ignore_index=True).sort_values("timestamp")
    frame = frame.loc[
        (frame.timestamp >= start)
        & (frame.timestamp < end)
        & (frame.timestamp + settings.delta <= now)
    ]
    if frame.empty:
        raise DataError("La descarga no produjo datos")
    output = output or settings.root / "data" / "raw.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    write_json(
        sidecar(output),
        {
            "kind": "raw",
            "provider": "binance-public-data",
            "market": settings.to_dict()["market"],
            "downloaded_at": datetime.now(UTC).isoformat(),
            "rows": len(frame),
            "sha256": sha256(output),
            "sources": [part[1] for part in parts],
        },
    )
    return output
