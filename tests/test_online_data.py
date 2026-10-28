from __future__ import annotations

import copy
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from trading_ml.errors import DataError
from trading_ml.online.config import load_config, protocol
from trading_ml.online.data import COLUMNS, features, fetch_candles, validate_candles

BASE_CONFIG = """[market]
symbol = "BTCUSDT"
interval = "1h"
horizon = 1
[online]
start_at = "2026-10-28T00:00:00+00:00"
"""


def _config(tmp_path, extra=""):
    path = tmp_path / "online.toml"
    path.write_text(BASE_CONFIG + extra, encoding="utf-8")
    return load_config(path)


def _response(rows):
    response = Mock()
    response.json.return_value = rows
    return response


def _klines(count, start="2026-01-01T00:00:00+00:00"):
    return [
        [int(stamp.timestamp() * 1000), "100", "102", "99", "101", "10"]
        for stamp in pd.date_range(start, periods=count, freq="1h")
    ]


@pytest.mark.parametrize(
    "extra",
    [
        "poll_seconds = true",
        "poll_seconds = nan",
        "poll_seconds = inf",
        "max_prediction_delay_seconds = false",
        "seed = 4294967296",
        "warmup_bars = true",
        "state_path = ''",
        "start_art = 'typo'",
        "[evaluation]\nalpha = nan",
        "[evaluation]\nalpha = inf",
        "[evaluation]\nalpha = false",
        "[evaluation]\nmin_effect = nan",
        "[evaluation]\nmin_effect = true",
        "[evaluation]\npatience = false",
        "[evaluation]\nmin_sample = 20",
        "[unknown]\nvalue=1",
    ],
)
def test_config_rejects_nonfinite_boolean_or_unknown_parameters(tmp_path, extra):
    with pytest.raises(DataError, match="Configuración online inválida"):
        _config(tmp_path, extra)


def test_protocol_preserves_defaults_and_tracks_statistical_changes(tmp_path):
    config = _config(tmp_path)
    initial = protocol(config)
    changed = copy.deepcopy(config)
    changed["evaluation"]["alpha"] = 0.01
    assert protocol(changed) != initial
    changed = copy.deepcopy(config)
    changed["online"]["poll_seconds"] = 60
    assert protocol(changed) == initial


def test_fetch_paginates_forward_from_last_committed_bar(monkeypatch, tmp_path):
    config = _config(tmp_path)
    first = _klines(1000)
    second = _klines(4, start="2026-02-11T16:00:00+00:00")
    calls = []
    pages = iter([first, second])

    def get(url, *, params, timeout):
        calls.append((url, dict(params), timeout))
        return _response(next(pages))

    monkeypatch.setattr("trading_ml.online.data.requests.get", get)
    now = pd.Timestamp("2026-03-01T00:00:00Z")
    frame = fetch_candles(config, now, "2026-01-01T00:00:00Z")
    assert len(frame) == 1004
    assert len(calls) == 2
    assert calls[1][1]["startTime"] == first[-1][0] + 1
    assert calls[0][1]["endTime"] == int(now.timestamp() * 1000) - 1
    assert calls[0][2] == (5, 15)


def test_bootstrap_uses_only_one_recent_page(monkeypatch, tmp_path):
    get = Mock(return_value=_response(_klines(1000)))
    monkeypatch.setattr("trading_ml.online.data.requests.get", get)
    frame = fetch_candles(_config(tmp_path), pd.Timestamp("2026-03-01T00:00:00Z"), None)
    assert len(frame) == 1000
    get.assert_called_once()


def test_provider_failures_raise_instead_of_using_local_or_stale_data(monkeypatch, tmp_path):
    get = Mock(side_effect=requests.ConnectionError("offline"))
    monkeypatch.setattr("trading_ml.online.data.requests.get", get)
    with pytest.raises(ConnectionError, match="Proveedores públicos no disponibles"):
        fetch_candles(_config(tmp_path), pd.Timestamp("2026-03-01T00:00:00Z"), None)
    assert get.call_count == 2


def test_second_public_provider_can_recover_transport_failure(monkeypatch, tmp_path):
    get = Mock(side_effect=[requests.Timeout("timeout"), _response(_klines(2))])
    monkeypatch.setattr("trading_ml.online.data.requests.get", get)
    frame = fetch_candles(_config(tmp_path), pd.Timestamp("2026-03-01T00:00:00Z"), None)
    assert len(frame) == 2
    assert get.call_count == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"code": -1},
        [[True, 1, 1, 1, 1, 1]],
        [[1, 1]],
        [[2, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
    ],
)
def test_invalid_rest_payload_is_not_silently_replaced(monkeypatch, tmp_path, payload):
    get = Mock(return_value=_response(payload))
    monkeypatch.setattr("trading_ml.online.data.requests.get", get)
    with pytest.raises(DataError):
        fetch_candles(_config(tmp_path), pd.Timestamp("2026-03-01T00:00:00Z"), None)
    get.assert_called_once()


def test_provider_cannot_ignore_resume_cursor(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "trading_ml.online.data.requests.get", Mock(return_value=_response(_klines(2)))
    )
    with pytest.raises(DataError, match="cursor"):
        fetch_candles(
            _config(tmp_path), pd.Timestamp("2026-03-01T00:00:00Z"), "2026-02-01T00:00:00Z"
        )


def _frame():
    frame = pd.DataFrame(_klines(10), columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    return frame


@pytest.mark.parametrize(
    "column,value",
    [
        ("timestamp", pd.NaT),
        ("open", True),
        ("close", float("nan")),
        ("high", 90),
        ("low", 110),
        ("volume", -1),
    ],
)
def test_invalid_closed_candles_are_rejected(column, value):
    frame = _frame().astype(object)
    frame.loc[0, column] = value
    with pytest.raises(DataError):
        validate_candles(frame, "1h", pd.Timestamp("2026-03-01T00:00:00Z"))


def test_open_candle_ohlcv_never_reaches_feature_computation():
    frame = _frame().astype(object)
    now = pd.Timestamp(frame.loc[9, "timestamp"])
    closed = validate_candles(frame, "1h", now)
    frame.loc[9, ["open", "high", "low", "close", "volume"]] = "incomplete"
    after = validate_candles(frame, "1h", now)
    pd.testing.assert_frame_equal(closed, after)
    assert features(after) == features(closed)
    assert len(closed) == 9


def test_duplicate_and_misaligned_candles_fail():
    frame = _frame()
    with pytest.raises(DataError, match="duplicados"):
        validate_candles(
            pd.concat([frame, frame.iloc[:1]]), "1h", pd.Timestamp("2026-03-01T00:00:00Z")
        )
    frame.loc[0, "timestamp"] += pd.Timedelta(minutes=1)
    with pytest.raises(DataError, match="desalineadas"):
        validate_candles(frame, "1h", pd.Timestamp("2026-03-01T00:00:00Z"))
