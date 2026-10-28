from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from test_online_service import _at, _candles, _config, _payloads, _rows, _write_source

from trading_ml.errors import TradingMLError
from trading_ml.online import service, store


def test_provider_delay_retains_pending_forecasts_until_target_data_arrives(tmp_path):
    config = _config(tmp_path)
    state = tmp_path / "state.sqlite3"
    frame = _candles(42)
    source = _write_source(tmp_path, frame.iloc[:40])
    service.run_once(config, state, source=source, now=_at(40, 30))
    assert len(_rows(state, "predictions")) == 3
    service.run_once(config, state, source=source, now=_at(41, 30))
    delayed = _rows(state, "predictions")
    assert {row["status"] for row in delayed} == {"pending"}
    assert all(model["updates"] == 0 for model in _payloads(state).values())
    _write_source(tmp_path, frame)
    service.run_once(config, state, source=source, now=_at(42, 30))
    recovered = _rows(state, "predictions")
    old = [row for row in recovered if pd.Timestamp(row["reference_time"]) == pd.Timestamp(_at(40))]
    assert {row["status"] for row in old} == {"evaluated"}
    assert all(model["updates"] == 1 for model in _payloads(state).values())
    assert not any(
        pd.Timestamp(row["reference_time"]) == pd.Timestamp(_at(41)) for row in recovered
    )
    service.verify(state)


def test_wallclock_cannot_move_backwards_within_the_same_closed_candle(tmp_path):
    config = _config(tmp_path)
    state = tmp_path / "state.sqlite3"
    source = _write_source(tmp_path, _candles(40))
    service.run_once(config, state, source=source, now=_at(40, 60))
    before = {table: _rows(state, table) for table in ("predictions", "models", "meta", "events")}
    with pytest.raises(TradingMLError):
        service.run_once(config, state, source=source, now=_at(40, 30))
    assert {table: _rows(state, table) for table in before} == before


def test_concurrent_cycles_commit_one_confirmation_and_one_forecast_per_model(tmp_path):
    config = _config(tmp_path)
    state = tmp_path / "state.sqlite3"
    source = _write_source(tmp_path, _candles(42))
    service.run_once(config, state, source=source, now=_at(40, 30))

    def cycle():
        return service.run_once(config, state, source=source, now=_at(41, 30))

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda _: cycle(), range(2)))
    assert sum(row["new_predictions"] for row in reports) == 3
    assert sum(row["confirmed_predictions"] for row in reports) == 3
    forecasts = _rows(state, "predictions")
    assert len(forecasts) == 6
    assert len({(row["model_id"], row["reference_time"]) for row in forecasts}) == 6
    assert all(model["updates"] == 1 for model in _payloads(state).values())
    service.verify(state)


class _TimestampClock:
    """Replace only service.pd, leaving the shared pandas module untouched."""

    def __init__(self, clock):
        self.clock = clock

    def __call__(self, *args, **kwargs):
        return pd.Timestamp(*args, **kwargs)

    def now(self, *, tz=None):
        stamp = self.clock[0]
        return stamp.tz_convert(tz) if tz is not None else stamp.tz_localize(None)


class _PandasClock:
    def __init__(self, clock):
        self.Timestamp = _TimestampClock(clock)

    def __getattr__(self, name):
        return getattr(pd, name)


@pytest.mark.parametrize("interval,late_seconds", [("1h", 121), ("1m", 61)])
def test_inference_latency_cannot_backdate_forecast_past_delay_or_target(
    tmp_path, monkeypatch, interval, late_seconds
):
    config = _config(tmp_path)
    if interval == "1m":
        config.write_text(
            config.read_text(encoding="utf-8").replace('interval = "1h"', 'interval = "1m"'),
            encoding="utf-8",
        )
    frame = _candles(40)
    frequency = "1min" if interval == "1m" else interval
    frame["timestamp"] = pd.date_range("2024-01-01", periods=40, freq=frequency, tz="UTC")
    reference = frame.timestamp.iloc[-1] + pd.Timedelta(frequency)
    source = _write_source(tmp_path, frame)
    state = tmp_path / "state.sqlite3"
    clock = [reference + pd.Timedelta(seconds=30)]
    monkeypatch.setattr(service, "pd", _PandasClock(clock))
    original_predict = service.predict

    def delayed_predict(model, values):
        probability = original_predict(model, values)
        clock[0] = reference + pd.Timedelta(seconds=late_seconds)
        return probability

    monkeypatch.setattr(service, "predict", delayed_predict)
    summary = service.run_once(config, state, source=source)
    assert summary["new_predictions"] == 0
    assert _rows(state, "predictions") == []
    assert all(model["updates"] == 0 for model in _payloads(state).values())
    service.verify(state)


def test_live_forecast_records_actual_issuance_after_inference(tmp_path, monkeypatch):
    config = _config(tmp_path)
    state = tmp_path / "state.sqlite3"
    source = _write_source(tmp_path, _candles(40))
    reference = pd.Timestamp(_at(40))
    clock = [reference + pd.Timedelta(seconds=30)]
    monkeypatch.setattr(service, "pd", _PandasClock(clock))
    original_predict = service.predict

    def delayed_predict(model, values):
        probability = original_predict(model, values)
        clock[0] = reference + pd.Timedelta(seconds=90)
        return probability

    monkeypatch.setattr(service, "predict", delayed_predict)
    service.run_once(config, state, source=source)
    forecasts = _rows(state, "predictions")
    assert len(forecasts) == 3
    assert {row["created_at"] for row in forecasts} == {clock[0].isoformat()}
    assert all(
        pd.Timestamp(row["created_at"]) < pd.Timestamp(row["target_time"]) for row in forecasts
    )
    service.verify(state)


@pytest.mark.parametrize("operation", ["init", "run", "config", "backup", "restore"])
def test_state_and_backup_paths_outside_workspace_are_rejected_before_any_creation(
    tmp_path, operation
):
    config = _config(tmp_path)
    existing_state = tmp_path / "source.sqlite3"
    service.init_state(config, existing_state)
    outside_parent = store.WORKSPACE.parent / f"codex-online-test-{uuid4().hex}"
    outside_state = outside_parent / "state.sqlite3"
    assert not outside_parent.exists()
    if operation == "config":
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                'state_path = "artifacts/online/state.sqlite3"',
                f'state_path = "{outside_state.as_posix()}"',
            ),
            encoding="utf-8",
        )
    with pytest.raises(TradingMLError, match="proyecto local"):
        if operation == "init":
            service.init_state(config, outside_state)
        elif operation == "run":
            service.run_once(config, outside_state)
        elif operation == "config":
            service.init_state(config)
        elif operation == "backup":
            service.backup(existing_state, outside_state)
        else:
            service.restore(existing_state, outside_state)
    assert not outside_parent.exists()
    assert not outside_state.exists()
    service.verify(existing_state)


@pytest.mark.parametrize("lookalike_tables", [False, True])
def test_init_preserves_unknown_existing_sqlite_byte_for_byte(tmp_path: Path, lookalike_tables):
    config = _config(tmp_path)
    unrelated = tmp_path / "existing_user_database.sqlite3"
    with sqlite3.connect(unrelated) as connection:
        connection.execute("CREATE TABLE user_records(id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO user_records VALUES(1, 'preserve existing user data')")
        if lookalike_tables:
            # Generic table names alone must not identify an unrelated database
            # as an initialized online state.
            connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE models(model_id TEXT PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE events(id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)"
            )
    before = unrelated.read_bytes()
    with pytest.raises((TradingMLError, sqlite3.DatabaseError)):
        service.init_state(config, unrelated)
    assert unrelated.read_bytes() == before
    with sqlite3.connect(unrelated) as connection:
        assert connection.execute("SELECT * FROM user_records").fetchall() == [
            (1, "preserve existing user data")
        ]
