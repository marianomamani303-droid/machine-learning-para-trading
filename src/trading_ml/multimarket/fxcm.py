"""Official FXCM public H1 bid/ask sample archives, without accounts or credentials."""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from trading_ml.common import frame_hash, read_json, sha256, sidecar, write_new_json
from trading_ml.errors import DataError
from trading_ml.multimarket.data import FIELDS, _snapshot
from trading_ml.multimarket.markets import fx_open, instrument

LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_NEXT = 0.0
_LOCAL = threading.local()


def parse_fxcm(payload: bytes):
    required = ["DateTime"] + [
        side + field.capitalize() for side in ["Bid", "Ask"] for field in FIELDS
    ]
    raw = gzip.decompress(payload)
    frame = pd.read_csv(io.BytesIO(raw)) if raw.strip() else pd.DataFrame(columns=required)
    if set(frame.columns) != set(required):
        raise DataError("Formato FXCM H1 inesperado")
    result = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                frame.DateTime, format="%m/%d/%Y %H:%M:%S.%f", utc=True, errors="raise"
            )
        }
    )
    for side in ["Bid", "Ask"]:
        for field in FIELDS:
            result[f"{field}_{side.lower()}"] = pd.to_numeric(
                frame[side + field.capitalize()], errors="raise"
            )
        values = result[[f"{field}_{side.lower()}" for field in FIELDS]]
        if not np.isfinite(values).all().all() or (values <= 0).any().any():
            raise DataError("Precios FXCM faltantes o inválidos")
        low, high = result[f"low_{side.lower()}"], result[f"high_{side.lower()}"]
        if ((low > values.min(axis=1)) | (high < values.max(axis=1))).any():
            raise DataError("OHLC FXCM inconsistente")
    if (
        result.timestamp.duplicated().any()
        or not result.timestamp.is_monotonic_increasing
        or (result.timestamp != result.timestamp.dt.floor("h")).any()
    ):
        raise DataError("Timeline H1 FXCM inválida")
    # Keep indicative crossed observations, never rewrite their prices or remove
    # an entire future hour. Eligibility is tested at each observable instant.
    result["crossed_open"] = result.open_ask < result.open_bid
    result["crossed_close"] = result.close_ask < result.close_bid
    for field in FIELDS:
        result[field] = result[f"{field}_bid"]
    result["reference_time"] = result.timestamp + pd.Timedelta(hours=1)
    return result


def _archive(root, symbol, year, week, refresh):
    global _NEXT
    url = f"https://candledata.fxcorporate.com/H1/{symbol}/{year}/{week}.csv.gz"
    path = root / "cache" / "fxcm" / symbol / f"{year}_{week:02d}.csv.gz"
    now = pd.Timestamp.now(tz="UTC")
    recent = year == now.year and week >= int(now.strftime("%U")) - 2
    missing_path = path.with_suffix(".missing.json")
    if (
        missing_path.exists()
        and not path.exists()
        and not refresh
        and not recent
        and year <= now.year
    ):
        return None, read_json(missing_path)
    if path.exists() and sidecar(path).exists() and not refresh and not recent:
        evidence = read_json(sidecar(path))
        if evidence["sha256"] != sha256(path) or evidence["url"] != url:
            raise DataError("Cache FXCM alterada")
        return parse_fxcm(path.read_bytes()), evidence
    for attempt in range(3):
        # A global bound across both workers; never a burst of hundreds of GETs.
        with _LOCK:
            pause = max(0.0, _NEXT - time.monotonic())
            if pause:
                time.sleep(pause)
            _NEXT = time.monotonic() + 0.3
        if not hasattr(_LOCAL, "session"):
            _LOCAL.session = requests.Session()
        try:
            response = _LOCAL.session.get(url, timeout=(10, 30))
            if response.status_code == 404:
                missing = {
                    "url": url,
                    "status": 404,
                    "retrieved_utc": datetime.now(UTC).isoformat(),
                    "interpretation": "missing archive; not assumed a market closure",
                }
                if not recent and year <= now.year and not missing_path.exists():
                    write_new_json(missing_path, missing)
                return None, missing
            if response.status_code == 429:
                raise DataError(
                    f"FXCM HTTP429: descarga detenida; Retry-After={response.headers.get('Retry-After', 'no informado')}; {url}"
                )
            response.raise_for_status()
            payload = response.content
            frame = parse_fxcm(payload)
            break
        except requests.RequestException as exc:
            if attempt == 2:
                raise DataError(f"FXCM no disponible: {url}: {type(exc).__name__}") from exc
            time.sleep(2**attempt)
    evidence = {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "provider": "fxcm",
        "authenticated_transport": True,
        "upstream_checksum_available": False,
        "indicative_prices": True,
        "rows": len(frame),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    immutable = path.parent / "versions" / f"{evidence['sha256']}.csv.gz"
    immutable.parent.mkdir(exist_ok=True)
    if not immutable.exists():
        with immutable.open("xb") as stream:
            stream.write(payload)
    elif sha256(immutable) != evidence["sha256"]:
        raise DataError("Archivo versionado FXCM alterado")
    evidence["immutable_archive"] = str(immutable.resolve())
    # Refresh changes only a cache pointer. Historical snapshots and versions stay immutable.
    temporary = path.with_suffix(".pending")
    temporary.write_bytes(payload)
    temporary.replace(path)
    from trading_ml.common import write_json

    write_json(sidecar(path), evidence)
    return frame, evidence


def download_fxcm(root: Path, symbol: str, start: str, end: str, refresh=False):
    asset = instrument(symbol)
    if asset.market != "forex":
        raise DataError("FXCM requiere un par Forex")
    first, last = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if first >= last or last > pd.Timestamp.now(tz="UTC").floor("h"):
        raise DataError("Rango FXCM inválido")
    # Provider week numbers are not assumed to equal ISO weeks. Request all
    # published slots in each touched year; retain actual UTC timestamps only.
    jobs = [
        (year, week)
        for year in range(first.year - 1, last.year + 1)
        for week in range(1, 54)
        if (year >= first.year or week >= 51)
        and (year < last.year or week <= min(53, last.dayofyear // 7 + 3))
    ]
    parts, evidence = [], []
    LOG.info(
        "FXCM %s: %d archivos semanales, máximo 2 conexiones y 3,3 GET/s", asset.symbol, len(jobs)
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        # Bounded submission: failure stops scheduling new network calls.
        iterator = iter(jobs)
        pending = [
            pool.submit(_archive, root, asset.symbol, *job, refresh)
            for job in [next(iterator), next(iterator)]
        ]
        index = 0
        while pending:
            frame, record = pending.pop(0).result()
            evidence.append(record)
            if frame is not None:
                parts.append(frame)
            index += 1
            if index % 52 == 0:
                LOG.info("FXCM %s: %d/%d archivos revisados", asset.symbol, index, len(jobs))
            job = next(iterator, None)
            if job is not None:
                pending.append(pool.submit(_archive, root, asset.symbol, *job, refresh))
    if not parts:
        raise DataError("FXCM no devolvió datos")
    data = pd.concat(parts, ignore_index=True)
    duplicate = data.loc[data.timestamp.duplicated(keep=False)]
    values = [f"{field}_{side}" for side in ["bid", "ask"] for field in FIELDS]
    if len(duplicate) and (duplicate.groupby("timestamp")[values].nunique() > 1).any().any():
        raise DataError("Archivos FXCM con precios duplicados contradictorios")
    removed = int(data.timestamp.duplicated().sum())
    data = data.drop_duplicates("timestamp").sort_values("timestamp")
    data = data.loc[(data.timestamp >= first) & (data.reference_time <= last)].reset_index(
        drop=True
    )
    if len(data) < 120:
        raise DataError("Cobertura FXCM insuficiente")
    expected = pd.date_range(first.ceil("h"), last, freq="h", inclusive="left")
    expected = expected[fx_open(expected).to_numpy()]
    observed = pd.DatetimeIndex(data.timestamp)
    missing = expected.difference(observed)
    # This is a quality gate, never a license to fill missing prices. Rejected
    # histories remain unavailable instead of producing a selective backtest.
    quality = {
        "provider": "fxcm",
        "indicative_active_trader_prices": True,
        "observed_hours": len(data),
        "expected_active_hours": len(expected),
        "missing_nominal_hours": len(missing),
        "missing_timestamps": [t.isoformat() for t in missing],
        "observed_outside_nominal_calendar": int((~fx_open(observed)).sum()),
        "crossed_open_quotes": int(data.crossed_open.sum()),
        "crossed_close_quotes": int(data.crossed_close.sum()),
        "identical_duplicates_removed": removed,
        "missing_archives": [r for r in evidence if r.get("status") == 404],
        "empty_archives": [r for r in evidence if r.get("rows") == 0],
        "volume_semantics": "not supplied; no volume columns fabricated",
        "imputed_rows": 0,
    }
    if len(missing) / max(len(expected), 1) > 0.02:
        rejected = (
            root
            / "multimarket"
            / "data_rejections"
            / f"{asset.symbol}_{first:%Y%m%d}_{last:%Y%m%d}_{frame_hash(data)[:12]}.json"
        )
        if not rejected.exists():
            write_new_json(
                rejected,
                {
                    "provider": "fxcm",
                    "symbol": asset.symbol,
                    "start": first.isoformat(),
                    "end_exclusive": last.isoformat(),
                    "quality": quality,
                    "sources": evidence,
                    "maximum_missing_nominal_fraction": 0.02,
                    "decision": "rejected before any model fit; no imputation",
                },
            )
        raise DataError(
            f"FXCM cobertura insuficiente: {len(missing)}/{len(expected)} horas nominales ausentes; evidencia: {rejected}"
        )
    contract = asset.contract() | {"provider": "fxcm"}
    return _snapshot(
        root,
        asset.symbol,
        data,
        {
            "instrument": contract,
            "start": first.isoformat(),
            "end_exclusive": last.isoformat(),
            "interval": "1h",
            "schema": "multimarket_bidask_v1",
            "sources": evidence,
            "quality": quality,
        },
    )
