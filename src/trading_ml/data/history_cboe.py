"""Cboe VX monthly dated futures. Daily traded OHLC, never a continuous proxy."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.data.history import PRICES, deduplicate, save
from trading_ml.data.history_sources import fetch
from trading_ml.errors import DataError

LOG = logging.getLogger(__name__)


def parse_cboe(payload: bytes, expiry: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(io.BytesIO(payload))
    required = [
        "Trade Date",
        "Futures",
        "Open",
        "High",
        "Low",
        "Close",
        "Settle",
        "Total Volume",
        "Open Interest",
    ]
    if not set(required) <= set(raw):
        raise DataError("Formato Cboe inesperado")
    dates = pd.to_datetime(raw["Trade Date"], utc=True)
    if dates.max() != expiry or (dates > expiry).any():
        raise DataError("El último día del contrato no confirma el vencimiento del archivo")
    # The filename and the contract descriptor must independently agree.
    expected_month = expiry.strftime("%b %Y")
    if not raw.Futures.astype(str).str.contains(expected_month, regex=False).all():
        raise DataError("Descriptor Cboe pertenece a otro vencimiento")
    f = pd.DataFrame(
        {
            "timestamp": dates,
            **{k: pd.to_numeric(raw[k.capitalize()], errors="raise") for k in PRICES},
            "volume": pd.to_numeric(raw["Total Volume"], errors="raise"),
            "settlement": pd.to_numeric(raw.Settle, errors="raise"),
            "open_interest": pd.to_numeric(raw["Open Interest"], errors="raise"),
        }
    )
    values = f[[*PRICES, "volume", "settlement", "open_interest"]].to_numpy(float)
    valid = (
        np.isfinite(values).all(axis=1)
        & (f[PRICES] > 0).all(axis=1)
        & (f.volume > 0)
        & (f.open_interest >= 0)
        & (f.settlement > 0)
    )
    valid &= (f.low <= f[["open", "close"]].min(axis=1)) & (
        f.high >= f[["open", "close"]].max(axis=1)
    )
    # Settlement is a separate exchange observation, never substituted for a trade.
    excluded = f.loc[~valid, "timestamp"].astype(str).tolist()
    f = f.loc[valid].copy()
    f["available_at"] = f.timestamp + pd.Timedelta(days=1)
    f, removed = deduplicate(f)
    return f, {
        "source_rows": len(raw),
        "excluded_nontraded_or_invalid_ohlc_dates": excluded,
        "identical_duplicates_removed": removed,
        "settlement_not_substituted_for_ohlc": True,
    }


def acquire(root: Path, start="2013-01-01", end="2026-10-01", development_end="2025-07-01") -> Path:
    first, last = pd.to_datetime([start, end], utc=True)
    if first >= last or last > pd.Timestamp.now(tz="UTC").normalize():
        raise DataError("Rango Cboe inválido")
    directory = root / "historical" / "cboe" / f"{first:%Y%m%d}_{last:%Y%m%d}"
    rows, failures = [], []

    payload, listing_source = fetch(
        root,
        "https://www-api.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/",
    )
    if payload is None:
        raise DataError("Listado oficial Cboe no disponible")
    products = [
        p
        for group in json.loads(payload).values()
        for p in group
        if p["duration_type"] == "M" and first <= pd.Timestamp(p["expire_date"], tz="UTC") < last
    ]

    def contract(product):
        candidate = pd.Timestamp(product["expire_date"], tz="UTC")
        receipt_path = directory / "completed" / f"VX_{candidate:%Y%m%d}.json"
        if receipt_path.exists():
            entries = read_json(receipt_path)["datasets"]
            for entry in entries:
                if sha256(Path(entry["path"])) != entry["sha256"]:
                    raise DataError("Contrato Cboe alterado")
            return entries
        source_path = product["path"]
        if (
            not source_path.startswith("data/us/futures/market_statistics/historical_data/VX/VX_")
            or ".." in source_path
        ):
            raise DataError("Ruta de contrato Cboe inesperada")
        payload, source = fetch(root, "https://cdn.cboe.com/" + source_path)
        if payload is None:
            raise DataError("Archivo listado por Cboe no disponible")
        frame, report = parse_cboe(payload, candidate)
        frame = frame.loc[(frame.timestamp >= first) & (frame.available_at <= last)]
        if frame.empty:
            return []
        entry = save(
            root,
            frame,
            {
                "provider": "cboe",
                "symbol": f"VX_{candidate:%Y%m%d}",
                "market_type": "exchange_dated_futures",
                "interval": "1d",
                "development_end": development_end,
                "sources": [source, listing_source],
                "exchange_contract": product,
                "expiry_date": candidate.date().isoformat(),
                "contract_multiplier_usd_per_index_point": 1000,
                "volume_semantics": "exchange traded contracts",
                "source_quality": report,
                "availability_policy": "Conservative UTC midnight after CFE trade date",
                "contract_policy": "Individual monthly VX contract, unadjusted; no stitching, no implicit roll, no synthetic OHLC on nontraded days",
                "limitations": "Daily traded OHLC only. Nontraded/invalid OHLC observations are excluded, raw settlements retained in original archive. No intraday history. Requires calendar, margin, multiplier, execution and roll protocol before fitting/backtesting.",
                "legal_source": "https://www.cboe.com/markets/us/futures/market-statistics/historical-data/futures; exchange data terms, no redistribution",
            },
        )
        write_new_json(receipt_path, {"datasets": [entry], "source_quality": report})
        LOG.info("Cboe %s: %d observaciones negociadas", entry["symbol"], entry["rows"])
        return [entry]

    def attempt(product):
        try:
            return contract(product)
        except Exception as exc:
            failures.append({"contract": product["expire_date"], "error": str(exc)})
            LOG.error("Cboe %s: %s", product["expire_date"], exc)
            return []

    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(attempt, products):
            rows.extend(result)
    path = directory / f"inventory_{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json"
    write_new_json(
        path,
        {
            "datasets": rows,
            "failures": failures,
            "models_trained": 0,
            "rows_including_derived_timeframes": sum(r["rows"] for r in rows),
            "unique_base_rows": sum(r["rows"] for r in rows),
            "universe": "Monthly VX expiries only; weekly VX excluded explicitly",
            "start": start,
            "end_exclusive": end,
        },
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2026-10-01")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
    )
    print(acquire(args.root, args.start, args.end))
