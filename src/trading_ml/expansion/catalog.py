from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, sidecar, write_new_json
from trading_ml.errors import DataError
from trading_ml.expansion.data import BASE, SYMBOLS, archive, csv_in_zip, snapshot


def catalogue(root: Path) -> dict:
    rows = []
    for symbol in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]:
        files = []
        for path in (root / "multimarket" / "data" / symbol).glob("1d_*.csv.json"):
            m = read_json(path)
            if m["instrument"]["provider"] == "fxcm":
                csv = path.with_suffix("")
                if sha256(csv) != m["csv_sha256"]:
                    raise DataError("Catálogo encontró histórico Forex alterado")
                files.append(
                    {
                        "path": str(csv.resolve()),
                        "rows": m["rows"],
                        "start": m["start"],
                        "end": m["end_exclusive"],
                    }
                )
        rows.append(
            {
                "symbol": symbol,
                "type": "indicative_otc_fx_bidask",
                "provider": "FXCM",
                "sources": files,
                "available": bool(files),
                "limitations": "personal research EULA; indicative broker quotes, no global traded volume; historical swaps absent",
            }
        )
    for symbol in SYMBOLS:
        files = []
        for path in (root / "expansion" / "data" / symbol).glob("spot_*.csv"):
            meta = read_json(sidecar(path))
            if sha256(path) != meta["sha256"]:
                raise DataError("Catálogo encontró spot alterado")
            files.append(
                {
                    "path": str(path.resolve()),
                    "rows": meta["rows"],
                    "start": meta["start"],
                    "end": meta["end_exclusive"],
                }
            )
        rows.append(
            {
                "symbol": symbol,
                "type": "crypto_spot",
                "provider": "Binance",
                "sources": files,
                "available": bool(files),
                "limitations": "USDT proxy for USD; ex-post surviving universe; exchange-specific flow",
            }
        )
    for symbol, description in [
        ("ES", "S&P 500"),
        ("NQ", "Nasdaq 100"),
        ("YM", "Dow Jones"),
        ("RTY", "Russell 2000"),
        ("GC", "gold"),
        ("SI", "silver"),
        ("CL", "WTI crude"),
        ("NG", "natural gas"),
        ("BRN", "Brent crude"),
    ]:
        rows.append(
            {
                "symbol": symbol,
                "description": description,
                "type": "exchange_dated_futures",
                "available": False,
                "provider": "CME DataMine / ICE for Brent; licensed historical data not supplied",
                "required": [
                    "license/source provenance",
                    "individual contract prices",
                    "expiry and multiplier",
                    "known-in-advance roll policy",
                    "calendar",
                    "costs",
                ],
                "limitations": "no ETF/CFD/front-month ticker substitution; negative futures prices require cash P&L, not generic percentage returns",
            }
        )
    rows.append(
        {
            "symbol": "SPY/QQQ/DIA/IWM",
            "type": "ETF_proxy",
            "available": False,
            "limitations": "not index futures; corporate actions, distributions, exchange calendar and data rights required",
        }
    )
    rows.append(
        {
            "symbol": "USA500IDXUSD/XAUUSD/etc",
            "type": "provider_cfd_or_otc_quote",
            "available": False,
            "limitations": "Dukascopy HTTPS probe timed out; FXCM SPX500 price archive returned 404; not exchange futures",
        }
    )
    return {
        "instruments": rows,
        "datasets_not_redistributed": True,
        "deferred_alternatives": {
            "macro": "ALFRED vintage/release availability required; current revised FRED history excluded",
            "options": "licensed historical chains/IV with expiry/delta and quote times required",
            "ETF_flows": "publication timestamps, revisions and historical constituents required",
            "onchain": "point-in-time chain indexing/confirmation and address-label vintages required",
            "news": "historical publication/update times and licensed archive required",
            "orderbook_liquidations": "event-level reliable archive and completeness required; not inferred from candles",
        },
    }


def acquire_oi_pilot(root: Path, symbol="BTCUSDT", day="2023-01-01") -> Path:
    # One measured pilot day; not silently stretched into a long historical feature.
    if symbol not in SYMBOLS or not pd.Timestamp("2021-01-01") <= pd.Timestamp(day) < pd.Timestamp(
        "2025-07-01"
    ):
        raise DataError("Piloto OI fuera del protocolo")
    url = f"{BASE}/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"
    payload, meta = archive(root, url)
    f = csv_in_zip(payload)
    if not (f.symbol == symbol).all():
        raise DataError("OI de otro contrato")
    f["event_time"] = pd.to_datetime(f.create_time, utc=True)
    f["available_at"] = f.event_time + pd.Timedelta(minutes=5)
    f["open_interest"] = pd.to_numeric(f.sum_open_interest)
    if (
        (f.open_interest <= 0).any()
        or not np.isfinite(f.open_interest).all()
        or f.event_time.duplicated().any()
    ):
        raise DataError("Open interest inválido")
    return snapshot(
        root,
        symbol,
        "oi_pilot",
        f,
        {
            "sources": [meta],
            "frequency": "5min",
            "availability_policy": "event plus five minutes conservative assumed latency; not historical publication proof",
            "used_for_model_selection": False,
            "coverage_scope": "pilot single day; no interpolation/extension",
        },
    )


def save_catalogue(root: Path, output: Path) -> dict:
    result = catalogue(root)
    write_new_json(output, result)
    return result
