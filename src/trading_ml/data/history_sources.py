"""Official Binance/FXCM archives and vendor index observations, with receipts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from trading_ml.common import read_json, sha256, write_json, write_new_json
from trading_ml.data.history import PRICES, aggregate, deduplicate, save, validate
from trading_ml.errors import DataError

LOG = logging.getLogger(__name__)
BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE = "https://data.binance.vision/"
_LOCAL = threading.local()
_FX_LOCK = threading.Lock()
_FX_NEXT = 0.0


def session():
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
        _LOCAL.session.headers["User-Agent"] = "historical-research/1.0"
    return _LOCAL.session


def fetch(root: Path, url: str, *, upstream=False, fx=False) -> tuple[bytes | None, dict]:
    """Immutable payload + receipt. Only explicit HTTP404 can be skipped."""
    global _FX_NEXT
    folder = root / "historical" / "cache" / hashlib.sha256(url.encode()).hexdigest()
    pointer = folder / "current.json"
    if pointer.exists():
        meta = read_json(pointer)
        if meta.get("status") == 404:
            return None, meta
        source = Path(meta["payload"])
        if sha256(source) != meta["sha256"] or meta["url"] != url:
            raise DataError("Cache histórico alterado")
        return source.read_bytes(), meta
    for attempt in range(3):
        if fx:
            with _FX_LOCK:
                time.sleep(max(0, _FX_NEXT - time.monotonic()))
                _FX_NEXT = time.monotonic() + 0.18
        try:
            response = session().get(url, timeout=(8, 35))
            if response.status_code == 429:
                raise DataError(f"HTTP429: detener proveedor y reanudar; {url}")
            if response.status_code == 404:
                meta = {"url": url, "status": 404, "retrieved_at": datetime.now(UTC).isoformat()}
                write_json(pointer, meta)
                return None, meta
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 2:
                raise DataError(f"Descarga fallida {url}: {type(exc).__name__}") from exc
            time.sleep(2**attempt)
    payload = response.content
    digest = hashlib.sha256(payload).hexdigest()
    if upstream:
        checksum = session().get(url + ".CHECKSUM", timeout=(8, 35))
        checksum.raise_for_status()
        if checksum.text.split()[0].lower() != digest:
            raise DataError("Checksum de Binance no coincide")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (digest + ".payload")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(payload)
    elif sha256(path) != digest:
        raise DataError("Payload histórico alterado")
    meta = {
        "url": url,
        "sha256": digest,
        "bytes": len(payload),
        "payload": str(path.resolve()),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "upstream_checksum": upstream,
        "transport": "HTTPS",
    }
    receipt = folder / (digest + ".json")
    if not receipt.exists():
        write_new_json(receipt, meta)
    write_json(pointer, meta)
    return payload, meta


def listing(root: Path, prefix: str, *, delimiter="") -> list[str]:
    keys, marker = [], ""
    while True:
        params = {"prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        url = requests.Request("GET", BUCKET, params=params).prepare().url
        payload, _ = fetch(root, url)
        if payload is None:
            raise DataError("Listado de archivos oficiales no disponible")
        tree = ET.fromstring(payload)
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        current = [
            e.text
            for e in tree.findall(
                "s:CommonPrefixes/s:Prefix" if delimiter else "s:Contents/s:Key", ns
            )
        ]
        keys.extend(current)
        if tree.findtext("s:IsTruncated", namespaces=ns) != "true":
            return keys
        marker = tree.findtext("s:NextMarker", namespaces=ns) or current[-1]


def parse_binance(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        files = [n for n in zipped.namelist() if n.endswith(".csv")]
        if len(files) != 1:
            raise DataError("ZIP sin un único CSV")
        f = pd.read_csv(zipped.open(files[0]), header=None)
    if f.shape[1] != 12:
        raise DataError("Schema Binance inválido")
    if str(f.iloc[0, 0]).lower() in {"open_time", "opentime"}:
        f = f.iloc[1:].reset_index(drop=True)
    values = f.iloc[:, :6].apply(pd.to_numeric, errors="raise")
    values.columns = ["timestamp", *PRICES, "volume"]
    unit = "us" if values.timestamp.median() > 100_000_000_000_000 else "ms"
    values.timestamp = pd.to_datetime(values.timestamp, unit=unit, utc=True)
    close_time = pd.to_datetime(pd.to_numeric(f.iloc[:, 6]), unit=unit, utc=True)
    expected_close = values.timestamp + pd.Timedelta(minutes=15)
    invalid_clock = (
        (close_time < expected_close - pd.Timedelta(milliseconds=1))
        | (close_time >= expected_close)
        | (values.timestamp != values.timestamp.dt.floor("15min"))
    )
    invalid_price = (
        ~np.isfinite(values[[*PRICES, "volume"]].to_numpy(float)).all(axis=1)
        | (values[PRICES] <= 0).any(axis=1)
        | (values.volume < 0)
        | (values.low > values[["open", "close"]].min(axis=1))
        | (values.high < values[["open", "close"]].max(axis=1))
    )
    invalid = invalid_clock | invalid_price
    if invalid.all():
        raise DataError("Archivo Binance sin velas válidas; se conserva crudo y se rechaza")
    excluded = [
        {
            "open_time": str(values.timestamp.iloc[i]),
            "close_time": str(close_time.iloc[i]),
            "invalid_clock": bool(invalid_clock.iloc[i]),
            "invalid_price": bool(invalid_price.iloc[i]),
        }
        for i in np.flatnonzero(invalid)
    ]
    values["available_at"] = expected_close
    values = values.loc[~invalid].reset_index(drop=True)
    validate(values, "15m")
    values.attrs["excluded_observations"] = excluded
    return values


def save_timeframes(root, frame, meta, source, intervals):
    rows = []
    for interval in intervals:
        data, aggregation = (
            (frame, {}) if interval == source else aggregate(frame, source, interval)
        )
        if data.empty:
            continue
        rows.append(save(root, data, {**meta, "interval": interval, "aggregation": aggregation}))
    return rows


def acquire_binance(root: Path, symbol: str, segment: str, config: dict) -> list[dict]:
    prefix = f"data/{segment}/monthly/klines/{symbol}/15m/"
    start = pd.Timestamp(config["crypto"]["start"], tz="UTC")
    end = pd.Timestamp(config["data"]["end"], tz="UTC")
    keys = [
        k
        for k in listing(root, prefix)
        if k.endswith(".zip") and start.strftime("%Y-%m") <= k[-11:-4] < end.strftime("%Y-%m")
    ]
    if not keys:
        raise DataError(f"No hay archivos M15 de {symbol}")

    def download(key):
        payload, receipt = fetch(root, BINANCE + key, upstream=True)
        if payload is None:
            raise DataError("Archivo listado por Binance devolvió HTTP404")
        frame = parse_binance(payload)
        return frame, {**receipt, "excluded_observations": frame.attrs["excluded_observations"]}

    frames, sources = [], []
    with ThreadPoolExecutor(max_workers=config["data"]["workers"]) as pool:
        for index, (frame, receipt) in enumerate(pool.map(download, keys), 1):
            frames.append(frame)
            sources.append(receipt)
            if index % 24 == 0:
                LOG.info("Binance %s %s: %d/%d meses", segment, symbol, index, len(keys))
    frame, removed = deduplicate(pd.concat(frames, ignore_index=True))
    frame = frame.loc[(frame.timestamp >= start) & (frame.available_at <= end)].reset_index(
        drop=True
    )
    market = (
        "crypto_spot"
        if segment == "spot"
        else "crypto_perpetual"
        if symbol.endswith("USDT")
        else "crypto_dated_futures"
    )
    expiry = (
        pd.Timestamp("20" + symbol[-6:], tz="UTC").isoformat()
        if market == "crypto_dated_futures"
        else None
    )
    post_delivery_excluded = 0
    post_delivery_with_volume = 0
    if expiry:
        delivery = pd.Timestamp(expiry) + pd.Timedelta(hours=8)
        after_delivery = frame.available_at > delivery
        post_delivery_with_volume = int((frame.loc[after_delivery, "volume"] > 0).sum())
        post_delivery_excluded = int(after_delivery.sum())
        frame = frame.loc[~after_delivery].reset_index(drop=True)
    meta = {
        "provider": "binance_" + segment.replace("/", "_"),
        "symbol": symbol,
        "market_type": market,
        "development_end": config["data"]["development_end"],
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "sources": sources,
        "excluded_invalid_observations": sum(len(s["excluded_observations"]) for s in sources),
        "identical_duplicates_removed": removed,
        "expiry_date": expiry,
        "scheduled_delivery_time_utc": (pd.Timestamp(expiry) + pd.Timedelta(hours=8)).isoformat()
        if expiry
        else None,
        "post_scheduled_delivery_candles_quarantined": post_delivery_excluded,
        "post_scheduled_delivery_candles_with_reported_volume": post_delivery_with_volume,
        "delivery_tail_policy": "Conservative scheduled 08:00 cutoff; terminal/settlement/late observations retained in original source; actual delayed delivery not assumed",
        "contract_policy": "Individual dated contract, no stitching, no back adjustment, no automatic rolls"
        if expiry
        else "Exchange-specific traded candles",
        "volume_semantics": "number of contracts; BTCUSD=100 USD, ETHUSD=10 USD face value; inverse settlement"
        if segment == "futures/cm"
        else "base asset units traded",
        "limitations": "Surviving assets selected ex post; not a historical constituent universe. Recent dates quarantined. Futures require contract-aware P&L/costs and funding for perpetuals; not legacy spot returns.",
        "legal_source": "https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md",
    }
    return save_timeframes(root, frame, meta, "15m", ["15m", "1h", "4h", "1d"])


def parse_fxcm_minute(payload: bytes) -> pd.DataFrame:
    raw = gzip.decompress(payload)
    columns = ["DateTime"] + [side + col.capitalize() for side in ["Bid", "Ask"] for col in PRICES]
    f = pd.read_csv(io.BytesIO(raw)) if raw.strip() else pd.DataFrame(columns=columns)
    if set(f) != set(columns):
        raise DataError("Schema FXCM inesperado")
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(f.DateTime, format="%m/%d/%Y %H:%M:%S.%f", utc=True)}
    )
    for side in ["bid", "ask"]:
        for col in PRICES:
            frame[f"{col}_{side}"] = pd.to_numeric(
                f[side.capitalize() + col.capitalize()], errors="raise"
            )
    for col in PRICES:
        frame[col] = frame[f"{col}_bid"]
    frame["available_at"] = frame.timestamp + pd.Timedelta(minutes=1)
    return frame


def acquire_forex(root: Path, symbol: str, config: dict) -> list[dict]:
    start = pd.Timestamp(config["forex"]["start"], tz="UTC")
    end = pd.Timestamp(config["data"]["end"], tz="UTC")
    frames, sources = [], []
    # A year at a time bounds memory; M1 remains in compressed immutable cache.
    for year in range(start.year, end.year + 1):
        jobs = list(range(1, 54 if year < end.year else min(54, end.dayofyear // 7 + 3)))

        def download(week, year=year):
            url = f"https://candledata.fxcorporate.com/m1/{symbol}/{year}/{week}.csv.gz"
            payload, receipt = fetch(root, url, fx=True)
            return parse_fxcm_minute(payload) if payload is not None else None, receipt

        parts = []
        with ThreadPoolExecutor(max_workers=config["data"]["workers"]) as pool:
            for frame, receipt in pool.map(download, jobs):
                sources.append(receipt)
                if frame is not None and len(frame):
                    parts.append(frame)
        if parts:
            minute, removed = deduplicate(pd.concat(parts, ignore_index=True))
            minute = minute.loc[(minute.timestamp >= start) & (minute.available_at <= end)]
            if len(minute):
                validate(minute, "1m")
                m15, quality = aggregate(minute, "1m", "15m")
                if len(m15):
                    frames.append(m15)
                LOG.info(
                    "FXCM %s %d: %d minutos reales, %d M15 completos, %d parciales excluidos",
                    symbol,
                    year,
                    len(minute),
                    len(m15),
                    quality["incomplete_windows_excluded"],
                )
    if not frames:
        raise DataError(f"FXCM no devolvió historia utilizable: {symbol}")
    frame, removed = deduplicate(pd.concat(frames, ignore_index=True))
    meta = {
        "provider": "fxcm",
        "symbol": symbol,
        "market_type": "indicative_otc_forex",
        "development_end": config["data"]["development_end"],
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "sources": sources,
        "identical_duplicates_removed": removed,
        "volume_semantics": "No global traded volume supplied; no fabricated volume column",
        "limitations": "Indicative Active Trader bid/ask, crossed quotes retained and counted. Strict complete UTC windows exclude partial sessions and holidays, not a NY-close D1 series. No historical swaps. Calendar-aware execution required.",
        "legal_source": "https://github.com/fxcm/MarketData/tree/master/CandleData; personal research EULA",
    }
    return save_timeframes(root, frame, meta, "15m", ["15m", "1h", "4h", "1d"])


def acquire_index(root: Path, symbol: str, config: dict) -> list[dict]:
    start = pd.Timestamp(config["indices"]["start"], tz="UTC")
    end = pd.Timestamp(config["data"]["end"], tz="UTC")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?interval=1d&period1={int(start.timestamp())}&period2={int(end.timestamp())}"
    payload, receipt = fetch(root, url)
    if payload is None:
        raise DataError("Índice no disponible")
    value = json.loads(payload)
    if value["chart"].get("error"):
        raise DataError("Error del proveedor de índices")
    result = value["chart"]["result"][0]
    if result["meta"]["symbol"] != symbol or result["meta"]["instrumentType"] != "INDEX":
        raise DataError("La fuente no corresponde al índice solicitado")
    times = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    # Session date in the exchange timezone, not a fabricated tradable UTC price.
    dates = (
        times.tz_convert(result["meta"]["exchangeTimezoneName"])
        .tz_localize(None)
        .normalize()
        .tz_localize("UTC")
    )
    prices = result["indicators"]["quote"][0]
    frame = pd.DataFrame({"timestamp": dates, **{col: prices[col] for col in PRICES}})
    values = frame[PRICES].to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    valid &= (frame.low <= frame[["open", "close"]].min(axis=1)) & (
        frame.high >= frame[["open", "close"]].max(axis=1)
    )
    # Long early vendor histories often repeat the daily close in all OHLC
    # fields. Keep their original payload, not an unverified intraday range.
    zero_range_dates = frame.loc[frame.high == frame.low, "timestamp"].astype(str).tolist()
    valid &= frame.high > frame.low
    invalid_dates = frame.loc[~valid, "timestamp"].astype(str).tolist()
    frame = frame.loc[valid & (frame.timestamp >= start) & (frame.timestamp < end)].copy()
    frame["available_at"] = frame.timestamp + pd.Timedelta(days=1)
    frame, removed = deduplicate(frame)
    meta = {
        "provider": "yahoo",
        "symbol": symbol,
        "market_type": "cash_index_context",
        "interval": "1d",
        "development_end": config["data"]["development_end"],
        "sources": [receipt],
        "exchange_metadata": result["meta"],
        "excluded_invalid_ohlc_dates": invalid_dates,
        "excluded_unverified_zero_range_dates": zero_range_dates,
        "ohlc_policy": "observed_nonzero_daily_range",
        "identical_duplicates_removed": removed,
        "availability_policy": "Conservative next UTC midnight after local session date; context only, no same-session execution",
        "volume_semantics": "Index has no directly traded units; vendor volume deliberately not used",
        "limitations": "Vendor observations, no official upstream checksum, revisions possible. Cash index is non-tradable context, not an ETF or future. Daily only; no invented intraday bars. Exchange calendars and publication times require a separate protocol.",
        "legal_source": "Yahoo Finance; personal research, no redistribution",
    }
    return [save(root, frame, meta)]
