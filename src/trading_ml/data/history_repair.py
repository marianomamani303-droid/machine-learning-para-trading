"""Recover genuine archive omissions from official daily Binance M15 files."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, sidecar, write_new_json
from trading_ml.data.history import deduplicate
from trading_ml.data.history_sources import BINANCE, fetch, parse_binance, save_timeframes
from trading_ml.errors import DataError

LOG = logging.getLogger(__name__)


def repair(root: Path, receipt: Path) -> Path:
    entries = read_json(receipt)["datasets"]
    if len({(entry["provider"], entry["symbol"]) for entry in entries}) != 1:
        raise DataError("Usar el comprobante de un solo instrumento para reparar")
    base = next(r for r in entries if r["interval"] == "15m")
    source = Path(base["path"])
    meta = read_json(sidecar(source))
    if sha256(source) != base["sha256"]:
        raise DataError("Histórico original alterado")
    segment = {
        "binance_spot": "spot",
        "binance_futures_um": "futures/um",
        "binance_futures_cm": "futures/cm",
    }[meta["provider"]]
    frame = pd.read_csv(source, float_precision="round_trip")
    for col in ["timestamp", "available_at"]:
        frame[col] = pd.to_datetime(frame[col], utc=True)
    missing = pd.date_range(
        frame.timestamp.iloc[0], frame.timestamp.iloc[-1], freq="15min"
    ).difference(pd.DatetimeIndex(frame.timestamp))
    recovered, evidence, unavailable = [], [], []
    numeric = ["open", "high", "low", "close", "volume"]
    for day in missing.normalize().unique():
        symbol = meta["symbol"]
        url = f"{BINANCE}data/{segment}/daily/klines/{symbol}/15m/{symbol}-15m-{day:%Y-%m-%d}.zip"
        try:
            payload, record = fetch(root, url, upstream=True)
            if payload is None:
                unavailable.append(record)
                continue
            daily = parse_binance(payload)
            if (daily.timestamp.dt.normalize() != day).any():
                raise DataError("Archivo diario no coincide con la fecha del hueco")
            overlap = frame.merge(daily, on="timestamp", suffixes=("_old", "_new"))
            if any(
                not np.allclose(overlap[c + "_old"], overlap[c + "_new"], rtol=1e-12, atol=1e-10)
                for c in numeric
            ):
                raise DataError(
                    "Fuente diaria contradice precios mensuales; no se mezclan versiones"
                )
            rescued = daily.loc[daily.timestamp.isin(missing)].copy()
            if len(rescued):
                rescued.attrs = {}
                frame = pd.concat([frame, rescued], ignore_index=True)
                recovered.extend(rescued.timestamp.astype(str).tolist())
                evidence.append(
                    {
                        **record,
                        "excluded_observations": daily.attrs.get("excluded_observations", []),
                    }
                )
                LOG.info("Recuperadas %s %s: %d velas oficiales", symbol, day.date(), len(rescued))
        except Exception as exc:
            unavailable.append({"url": url, "reason": str(exc)})
    frame, _ = deduplicate(frame)
    if recovered:
        entries = save_timeframes(
            root,
            frame,
            {
                **meta,
                "sources": meta["sources"] + evidence,
                "repair": {
                    "parent_csv": str(source.resolve()),
                    "parent_sha256": base["sha256"],
                    "recovered_rows": len(recovered),
                    "recovered_timestamps": recovered,
                    "policy": "Official daily source only; no interpolation; overlapping observations must agree",
                },
            },
            "15m",
            ["15m", "1h", "4h", "1d"],
        )
    if sha256(source) != base["sha256"]:
        raise DataError("La reparación modificó el histórico de origen")
    path = (
        root
        / "historical"
        / "repairs"
        / f"{meta['symbol']}_{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json"
    )
    write_new_json(
        path,
        {
            "datasets": entries,
            "failures": [],
            "unavailable_or_conflicting_daily_files": unavailable,
            "original_gap_slots": len(missing),
            "recovered_rows": len(recovered),
            "remaining_gap_slots": len(missing) - len(recovered),
            "original_snapshot_unchanged": True,
            "models_trained": 0,
        },
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
    )
    print(repair(args.root, args.receipt))
