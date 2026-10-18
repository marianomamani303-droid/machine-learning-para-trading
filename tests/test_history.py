import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.common import read_json, sidecar, write_new_json
from trading_ml.config import Settings
from trading_ml.config.settings import Market
from trading_ml.data.history import (
    aggregate,
    deduplicate,
    export_forex,
    export_legacy,
    load_history,
    save,
    validate,
)
from trading_ml.data.history_sources import parse_binance
from trading_ml.errors import DataError, LeakageError
from trading_ml.preprocessing.pipeline import prepare


def candles(start="2020-01-01", periods=384):
    times = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    close = 100 + np.sin(np.arange(periods) / 9) + np.arange(periods) / 20
    return pd.DataFrame(
        {
            "timestamp": times,
            "open": close - 0.1,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
            "available_at": times + pd.Timedelta(minutes=15),
        }
    )


def stored(tmp_path, frame=None, **kwargs):
    meta = {
        "provider": "test",
        "symbol": "BTCUSDT",
        "interval": "15m",
        "market_type": "crypto_spot",
        "development_end": "2025-07-01",
        **kwargs,
    }
    return save(tmp_path, candles() if frame is None else frame, meta)


def test_incomplete_windows_are_excluded_without_filling_and_never_upsampled():
    frame = candles(periods=16).drop(index=3)
    hourly, report = aggregate(frame, "15m", "1h")
    assert list(hourly.timestamp) == list(
        pd.date_range("2020-01-01 01:00", periods=3, freq="h", tz="UTC")
    )
    assert report["incomplete_windows_excluded"] == 1
    assert hourly.volume.tolist() == [40, 40, 40]
    assert hourly.open.iloc[0] == frame.loc[4, "open"]
    assert hourly.high.iloc[0] == frame.loc[4:7, "high"].max()
    assert hourly.close.iloc[0] == frame.loc[7, "close"]
    assert (hourly.available_at == hourly.timestamp + pd.Timedelta(hours=1)).all()
    with pytest.raises(DataError, match="menor"):
        aggregate(hourly, "1h", "15m")


@pytest.mark.parametrize(
    "defect",
    ["nan", "infinity", "negative_volume", "ohlc", "duplicate", "clock", "availability", "future"],
)
def test_rejects_invalid_observations(defect):
    f = candles(periods=12)
    if defect == "nan":
        f.loc[0, "close"] = np.nan
    elif defect == "infinity":
        f.loc[0, "close"] = np.inf
    elif defect == "negative_volume":
        f.loc[0, "volume"] = -1
    elif defect == "ohlc":
        f.loc[0, "low"] = f.high.iloc[0] + 2
    elif defect == "duplicate":
        f = pd.concat([f, f.iloc[:1]])
    elif defect == "clock":
        f.loc[0, "timestamp"] += pd.Timedelta(seconds=1)
    elif defect == "availability":
        f.loc[0, "available_at"] = f.timestamp.iloc[0]
    else:
        f = candles("2099-01-01", periods=12)
    with pytest.raises(DataError):
        validate(f, "15m")


def test_duplicate_prices_must_agree():
    f = candles(periods=12)
    clean, removed = deduplicate(pd.concat([f, f.iloc[:1]]))
    assert len(clean) == 12 and removed == 1
    bad = f.iloc[:1].copy()
    bad.loc[0, "volume"] = 50
    with pytest.raises(DataError, match="contradictorias"):
        deduplicate(pd.concat([f, bad]))


def test_loader_quarantine_and_real_prospective_reservations(tmp_path):
    entry = stored(tmp_path)
    from pathlib import Path

    path = Path(entry["path"])
    with pytest.raises(LeakageError, match="cuarentena"):
        load_history(path, root=tmp_path, end="2025-07-02")
    f, _ = load_history(path, root=tmp_path, end="2020-01-02")
    assert len(f) == 96 and f.available_at.max() == pd.Timestamp("2020-01-02", tz="UTC")
    write_new_json(
        tmp_path / "smc_studies" / "old" / "prospective.json",
        {"first_reference": "2020-01-02", "end_reference_exclusive": "2020-01-03"},
    )
    with pytest.raises(LeakageError, match="reservado"):
        load_history(path, root=tmp_path, end="2020-01-02")


def test_snapshot_immutable_and_corruption_detected(tmp_path):
    from pathlib import Path

    entry = stored(tmp_path)
    path = Path(entry["path"])
    assert stored(tmp_path)["path"] == str(path)
    path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(DataError, match="Integridad"):
        load_history(path, root=tmp_path, end="2020-01-02")


def test_new_spot_can_feed_existing_prepare_without_training(tmp_path):
    from pathlib import Path

    entry = stored(tmp_path)
    source = export_legacy(
        Path(entry["path"]),
        tmp_path / "export.csv",
        root=tmp_path,
        start="2020-01-01",
        end="2020-01-05",
    )
    settings = Settings(root=tmp_path, market=Market("BTCUSDT", "15m", "2020-01-01", "2020-01-05"))
    output = prepare(settings, source, tmp_path / "prepared.csv")
    assert read_json(sidecar(output))["kind"] == "prepared"
    with pytest.raises(FileExistsError):
        export_legacy(
            Path(entry["path"]), source, root=tmp_path, start="2020-01-01", end="2020-01-05"
        )


def test_session_markets_are_not_silently_exported_as_crypto(tmp_path):
    from pathlib import Path

    entry = stored(tmp_path, market_type="crypto_dated_futures", symbol="BTCUSD_200626")
    with pytest.raises(DataError, match="sólo admite spot"):
        export_legacy(
            Path(entry["path"]),
            tmp_path / "bad.csv",
            root=tmp_path,
            start="2020-01-01",
            end="2020-01-02",
        )


@pytest.mark.parametrize("unit,header", [("ms", False), ("us", False), ("ms", True)])
def test_binance_time_units_headers_and_partial_maintenance_bars(unit, header):
    times = pd.date_range("2020-01-01", periods=200, freq="15min", tz="UTC")
    scale = pd.Timedelta(milliseconds=1) if unit == "ms" else pd.Timedelta(microseconds=1)
    ticks = [int(t.timestamp() * (1000 if unit == "ms" else 1_000_000)) for t in times]
    ends = [
        int(
            (t + pd.Timedelta(minutes=15) - scale).timestamp()
            * (1000 if unit == "ms" else 1_000_000)
        )
        for t in times
    ]
    ends[1] -= 120_000 if unit == "ms" else 120_000_000
    rows = pd.DataFrame(
        [
            [tick, 100, 110, 90, 105, 10, end, 1000, 5, 1, 100, 0]
            for tick, end in zip(ticks, ends, strict=True)
        ]
    )
    rows.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "q",
        "n",
        "tb",
        "tq",
        "ignore",
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("candles.csv", rows.to_csv(index=False, header=header))
    frame = parse_binance(buffer.getvalue())
    assert len(frame) == 199
    assert times[1] not in list(frame.timestamp)
    assert frame.timestamp.iloc[0] == times[0]
    assert len(frame.attrs["excluded_observations"]) == 1
    assert frame.attrs["excluded_observations"][0]["invalid_clock"]


def test_aggregation_is_causal_under_future_price_changes():
    f = candles(periods=96)
    first, _ = aggregate(f, "15m", "4h")
    altered = f.copy()
    altered.loc[64:, ["open", "high", "low", "close"]] *= 2
    second, _ = aggregate(altered, "15m", "4h")
    pd.testing.assert_frame_equal(first.iloc[:4], second.iloc[:4])


def test_forex_adapter_preserves_quotes_and_ny_session_pipeline(tmp_path):
    from trading_ml.multimarket.data import aggregate_forex, load_snapshot
    from trading_ml.multimarket.features import features

    minute = candles(periods=96 * 12)
    for col in ["open", "high", "low", "close"]:
        minute[col + "_bid"] = minute[col]
        minute[col + "_ask"] = minute[col] + 0.02
    minute = minute.drop(columns="volume")
    hourly, _ = aggregate(minute, "15m", "1h")
    entry = stored(
        tmp_path, hourly, symbol="EURGBP", interval="1h", market_type="indicative_otc_forex"
    )
    output = export_forex(Path(entry["path"]), root=tmp_path, start="2020-01-01", end="2020-01-12")
    source, meta = load_snapshot(output)
    assert meta["instrument"]["provider"] == "fxcm"
    assert "volume" not in source
    assert not source.crossed_close.any()
    assert len(features(source, "EURGBP")) == len(source)
    daily_path = aggregate_forex(tmp_path / "legacy", output, "1d")
    daily, _ = load_snapshot(daily_path)
    assert (daily.reference_time.dt.tz_convert("America/New_York").dt.hour == 17).all()


def test_cboe_keeps_settlement_distinct_and_excludes_nontraded_bars():
    from trading_ml.data.history_cboe import parse_cboe

    payload = b"Trade Date,Futures,Open,High,Low,Close,Settle,Total Volume,Open Interest\n2025-06-17,M (Jun 2025),20,22,19,21,23,100,200\n2025-06-18,M (Jun 2025),0,0,0,0,22,0,0\n"
    frame, quality = parse_cboe(payload, pd.Timestamp("2025-06-18", tz="UTC"))
    assert len(frame) == 1
    assert frame.close.iloc[0] == 21 and frame.settlement.iloc[0] == 23
    assert len(quality["excluded_nontraded_or_invalid_ohlc_dates"]) == 1
    with pytest.raises(DataError, match="vencimiento"):
        parse_cboe(payload, pd.Timestamp("2025-06-19", tz="UTC"))


def test_historical_network_checksums_and_cache_corruption(tmp_path, monkeypatch):
    import hashlib
    from unittest.mock import Mock

    from trading_ml.data.history_sources import fetch, session

    payload = b"official compressed source"
    digest = hashlib.sha256(payload).hexdigest()
    get = Mock(
        side_effect=[
            Mock(status_code=200, content=payload),
            Mock(status_code=200, text=digest + " file.zip"),
        ]
    )
    monkeypatch.setattr(session(), "get", get)
    content, receipt = fetch(tmp_path, "https://example.test/file.zip", upstream=True)
    assert content == payload and receipt["upstream_checksum"]
    fetch(tmp_path, "https://example.test/file.zip", upstream=True)
    assert get.call_count == 2
    Path(receipt["payload"]).write_bytes(b"corrupt")
    with pytest.raises(DataError, match="alterado"):
        fetch(tmp_path, "https://example.test/file.zip", upstream=True)


def test_historical_upstream_checksum_mismatch_never_persists_payload(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from trading_ml.data.history_sources import fetch, session

    get = Mock(
        side_effect=[Mock(status_code=200, content=b"bad"), Mock(status_code=200, text="0" * 64)]
    )
    monkeypatch.setattr(session(), "get", get)
    with pytest.raises(DataError, match="Checksum"):
        fetch(tmp_path, "https://example.test/file.zip", upstream=True)
    assert not list(tmp_path.rglob("*.payload"))


def test_inventory_reconstructs_timeframes_and_rejects_inconsistent_valid_bars(tmp_path):
    from trading_ml.data.history_cli import verify_inventory

    base = candles()
    first = stored(tmp_path, base)
    hourly, _ = aggregate(base, "15m", "1h")
    second = stored(tmp_path, hourly, interval="1h")
    inventory = tmp_path / "inventory.json"
    write_new_json(inventory, {"datasets": [first, second]})
    result = verify_inventory(inventory)
    assert result["timeframe_aggregations_reconstructed"] == 1
    assert result["rows"] == 480
    hourly.loc[0, "high"] += 20
    conflicting = stored(tmp_path, hourly, interval="1h")
    invalid_inventory = tmp_path / "invalid_inventory.json"
    write_new_json(invalid_inventory, {"datasets": [first, conflicting]})
    with pytest.raises(DataError, match="inconsistentes"):
        verify_inventory(invalid_inventory)


def test_vendor_close_only_index_history_is_not_accepted_as_daily_ohlc(tmp_path, monkeypatch):
    import json

    from trading_ml.data.history_sources import acquire_index

    data = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "^TEST",
                        "instrumentType": "INDEX",
                        "exchangeTimezoneName": "UTC",
                    },
                    "timestamp": [1577836800, 1577923200],
                    "indicators": {
                        "quote": [
                            {"open": [10, 11], "high": [10, 13], "low": [10, 9], "close": [10, 12]}
                        ]
                    },
                }
            ],
        }
    }
    monkeypatch.setattr(
        "trading_ml.data.history_sources.fetch",
        lambda *a, **k: (json.dumps(data).encode(), {"url": "https://example.test/index"}),
    )
    rows = acquire_index(
        tmp_path,
        "^TEST",
        {
            "indices": {"start": "1900-01-01"},
            "data": {"end": "2020-02-01", "development_end": "2025-07-01"},
        },
    )
    assert rows[0]["rows"] == 1
    meta = read_json(sidecar(Path(rows[0]["path"])))
    assert len(meta["excluded_unverified_zero_range_dates"]) == 1


def test_native_ny_daily_calendar_is_not_utc_midnight():
    winter = candles(periods=1).drop(columns="volume")
    winter["timestamp"] = pd.to_datetime(["2020-01-05T22:00:00Z"])
    winter["available_at"] = winter.timestamp + pd.Timedelta(days=1)
    assert validate(winter, "1d_ny")["rows"] == 1
    summer = winter.copy()
    summer.timestamp = pd.to_datetime(["2020-07-05T21:00:00Z"])
    summer.available_at = summer.timestamp + pd.Timedelta(days=1)
    validate(summer, "1d_ny")
    with pytest.raises(DataError, match="grilla"):
        validate(summer, "1d")
    with pytest.raises(DataError, match="NY17"):
        aggregate(summer, "1d_ny", "1d")


def test_futures_loader_enforces_scheduled_delivery_boundary(tmp_path):
    f = candles("2020-09-25 07:45", periods=2)
    entry = stored(tmp_path, f, symbol="BTCUSD_200925", market_type="crypto_dated_futures")
    with pytest.raises(DataError, match="vencimiento"):
        load_history(Path(entry["path"]), root=tmp_path, end="2020-09-26")
    safe = stored(tmp_path, f.iloc[:1], symbol="BTCUSD_200925", market_type="crypto_dated_futures")
    data, _ = load_history(Path(safe["path"]), root=tmp_path, end="2020-09-26")
    assert data.available_at.iloc[-1] == pd.Timestamp("2020-09-25 08:00", tz="UTC")


@pytest.mark.parametrize("conflict", [False, True])
def test_daily_repair_recovers_only_official_missing_rows_and_rejects_conflicts(
    tmp_path, monkeypatch, conflict
):
    from trading_ml.data.history_repair import repair

    original = candles(periods=96)
    incomplete = original.drop(index=[20, 21, 22, 23]).reset_index(drop=True)
    entry = stored(
        tmp_path,
        incomplete,
        provider="binance_futures_um",
        market_type="crypto_perpetual",
        sources=[],
    )
    receipt = tmp_path / "completed.json"
    write_new_json(receipt, {"datasets": [entry]})
    zipped = io.BytesIO()
    raw = []
    for i, row in original.iterrows():
        tick = int(row.timestamp.timestamp() * 1000)
        high = row.high + (1 if conflict and i == 0 else 0)
        raw.append(
            [
                tick,
                row.open,
                high,
                row.low,
                row.close,
                row.volume,
                tick + 900000 - 1,
                1000,
                10,
                1,
                100,
                0,
            ]
        )
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("daily.csv", pd.DataFrame(raw).to_csv(index=False, header=False))
    monkeypatch.setattr(
        "trading_ml.data.history_repair.fetch",
        lambda *a, **k: (
            zipped.getvalue(),
            {"url": "https://example.test/official-daily.zip", "sha256": "fixture"},
        ),
    )
    result = read_json(repair(tmp_path, receipt))
    assert result["original_snapshot_unchanged"]
    assert result["recovered_rows"] == (0 if conflict else 4)
    assert result["remaining_gap_slots"] == (4 if conflict else 0)
    if not conflict:
        assert {r["interval"] for r in result["datasets"]} == {"15m", "1h", "4h", "1d"}
        assert next(r for r in result["datasets"] if r["interval"] == "15m")["rows"] == 96
