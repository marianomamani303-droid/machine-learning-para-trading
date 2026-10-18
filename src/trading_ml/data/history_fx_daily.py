"""Native FXCM NY17 daily archives, distinct from strictly aggregated UTC days."""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.data.history import deduplicate, save
from trading_ml.data.history_sources import fetch, parse_fxcm_minute
from trading_ml.errors import DataError

LOG = logging.getLogger(__name__)


def acquire(root: Path, config: dict) -> Path:
    start = pd.Timestamp(config["forex"]["start"], tz="UTC")
    end = pd.Timestamp(config["data"]["end"], tz="UTC")
    directory = root / "historical" / "fxcm_native_daily" / f"{start:%Y%m%d}_{end:%Y%m%d}"
    results, failures = [], []
    for symbol in config["forex"]["symbols"]:
        receipt_path = directory / "completed" / (symbol + ".json")
        if receipt_path.exists():
            entry = read_json(receipt_path)["datasets"][0]
            if sha256(Path(entry["path"])) != entry["sha256"]:
                raise DataError("D1 nativo alterado")
            results.append(entry)
            continue

        def year_download(year, symbol=symbol):
            url = f"https://candledata.fxcorporate.com/D1/{symbol}/{year}.csv.gz"
            payload, source = fetch(root, url, fx=True)
            if payload is None:
                return None, source
            frame = parse_fxcm_minute(payload)
            frame["available_at"] = frame.timestamp + pd.Timedelta(days=1)
            return frame, source

        try:
            frames, sources = [], []
            with ThreadPoolExecutor(max_workers=3) as pool:
                for frame, source in pool.map(year_download, range(start.year, end.year + 1)):
                    sources.append(source)
                    if frame is not None and len(frame):
                        frames.append(frame)
            if not frames:
                raise DataError("No hay archivos diarios nativos")
            frame, removed = deduplicate(pd.concat(frames, ignore_index=True))
            frame = frame.loc[(frame.timestamp >= start) & (frame.available_at <= end)].reset_index(
                drop=True
            )
            local = frame.timestamp.dt.tz_convert("America/New_York")
            active = local.dt.dayofweek.isin([6, 0, 1, 2, 3])
            excluded_closed = frame.loc[~active, "timestamp"].astype(str).tolist()
            frame = frame.loc[active].reset_index(drop=True)
            entry = save(
                root,
                frame,
                {
                    "provider": "fxcm",
                    "symbol": symbol,
                    "market_type": "indicative_otc_forex",
                    "interval": "1d_ny",
                    "development_end": config["data"]["development_end"],
                    "sources": sources,
                    "session_timezone": "America/New_York",
                    "session_open_hour": 17,
                    "bar_origin": "provider_native_daily",
                    "identical_duplicates_removed": removed,
                    "volume_semantics": "Not provided, no proxy fabricated",
                    "availability_policy": "Native session open plus 24h; NY17 on business sessions; actual UTC timestamps retained",
                    "limitations": "Native annual archive currently not available after 2021. Different calendar/source from UTC days aggregated from M1. Keep separate; do not relabel as UTC midnight or substitute a native daily bar for H1 execution.",
                    "legal_source": "https://github.com/fxcm/MarketData/tree/master/CandleData; personal research EULA",
                },
            )
            write_new_json(
                receipt_path,
                {
                    "datasets": [entry],
                    "excluded_observations_outside_weekly_session": excluded_closed,
                },
            )
            results.append(entry)
            LOG.info(
                "FXCM %s D1 NY17: %d filas; %s a %s",
                symbol,
                entry["rows"],
                entry["first_open"],
                entry["last_close"],
            )
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
            LOG.error("D1 NY17 %s: %s", symbol, exc)
    path = directory / f"inventory_{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json"
    write_new_json(
        path,
        {
            "datasets": results,
            "failures": failures,
            "models_trained": 0,
            "note": "Native daily NY17; explicitly separate from UTC D1. Missing annual files are retained in sources.",
        },
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("config/historical.toml"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
    )
    print(acquire(args.root, tomllib.loads(args.config.read_text(encoding="utf-8"))))
