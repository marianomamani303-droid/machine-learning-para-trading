"""Quarantine terminal/settlement candles after scheduled delivery, preserving originals."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_ml.common import read_json, sha256, sidecar, write_new_json
from trading_ml.data.history_sources import save_timeframes
from trading_ml.errors import DataError


def bound(root: Path, inventory: Path) -> Path:
    original = read_json(inventory)
    entries = [r for r in original["datasets"] if r["market_type"] != "crypto_dated_futures"]
    removed, modified = 0, 0
    for entry in original["datasets"]:
        if entry["market_type"] != "crypto_dated_futures" or entry["interval"] != "15m":
            continue
        source = Path(entry["path"])
        if sha256(source) != entry["sha256"]:
            raise DataError("Fuente previa alterada")
        meta = read_json(sidecar(source))
        delivery = pd.Timestamp("20" + entry["symbol"][-6:], tz="UTC") + pd.Timedelta(hours=8)
        frame = pd.read_csv(source, float_precision="round_trip")
        for col in ["timestamp", "available_at"]:
            frame[col] = pd.to_datetime(frame[col], utc=True)
        invalid = frame.available_at > delivery
        with_volume = int((frame.loc[invalid, "volume"] > 0).sum())
        count = int(invalid.sum())
        removed += count
        modified += int(count > 0)
        entries.extend(
            save_timeframes(
                root,
                frame.loc[~invalid].reset_index(drop=True),
                {
                    **meta,
                    "scheduled_delivery_time_utc": delivery.isoformat(),
                    "post_scheduled_delivery_candles_quarantined": count,
                    "post_scheduled_delivery_candles_with_reported_volume": with_volume,
                    "delivery_tail_policy": "Conservative nominal 08:00 cutoff; terminal/settlement/late observations retained in original source; actual delayed delivery not assumed",
                    "delivery_bound_parent": str(source.resolve()),
                    "delivery_bound_parent_sha256": entry["sha256"],
                },
                "15m",
                ["15m", "1h", "4h", "1d"],
            )
        )
        if sha256(source) != entry["sha256"]:
            raise DataError("La depuración cambió el original")
    path = (
        root
        / "historical"
        / "delivery_bounds"
        / f"inventory_{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json"
    )
    write_new_json(
        path,
        {
            "datasets": entries,
            "failures": original.get("failures", []),
            "parent_inventory": str(inventory.resolve()),
            "parent_inventory_sha256": sha256(inventory),
            "post_scheduled_delivery_candles_quarantined": removed,
            "contracts_changed": modified,
            "models_trained": 0,
            "original_snapshots_unchanged": True,
        },
    )
    print(
        {
            "inventory": str(path),
            "post_scheduled_delivery_candles_quarantined": removed,
            "contracts_changed": modified,
        }
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    bound(args.root, args.inventory)
