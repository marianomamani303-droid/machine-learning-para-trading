from __future__ import annotations

import gzip
import lzma
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.common import read_json, sidecar
from trading_ml.config.settings import Features
from trading_ml.errors import DataError, LeakageError
from trading_ml.features.technical import build_features
from trading_ml.multimarket.data import (
    _month_file,
    _snapshot,
    aggregate_forex,
    decode_hourly,
    load_snapshot,
)
from trading_ml.multimarket.execution import Costs, Engine, _swap_rollovers, conversion
from trading_ml.multimarket.features import features, recipes
from trading_ml.multimarket.fxcm import parse_fxcm
from trading_ml.multimarket.markets import fx_open, instrument, session_end
from trading_ml.multimarket.protocol import load_protocol
from trading_ml.multimarket.service import predict_market
from trading_ml.multimarket.study import labelled, require_outcomes


def quotes(rows=180, start="2024-01-08T00:00Z"):
    """Synthetic deterministic fixtures ONLY; never used as market history."""
    time = pd.date_range(start, periods=rows, freq="h")
    rng = np.random.default_rng(12)
    price = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.0005, rows)))
    frame = pd.DataFrame(
        {
            "timestamp": time,
            "reference_time": time + pd.Timedelta(hours=1),
            "open": price,
            "close": price * 1.0001,
            "high": price * 1.001,
            "low": price * 0.999,
        }
    )
    for field in ["open", "high", "low", "close"]:
        frame[f"{field}_bid"] = frame[field]
        frame[f"{field}_ask"] = frame[field] + 0.0002
    frame["quote_volume_bid"] = 1.0
    frame["quote_volume_ask"] = 1.0
    return frame


def test_binary_decode_order_scale_and_validation():
    month = pd.Timestamp("2020-01-01", tz="UTC")
    packed = lzma.compress(struct.pack(">5if", 3600, 110000, 110010, 109990, 110020, 4.0))
    frame = decode_hourly(packed, month, 100000)
    assert frame.timestamp.iloc[0] == month + pd.Timedelta(hours=1)
    assert frame.close.iloc[0] == 1.1001
    assert frame.low.iloc[0] == 1.0999
    jpy = decode_hourly(packed, month, 1000)
    assert jpy.open.iloc[0] == 110
    with pytest.raises(DataError):
        decode_hourly(lzma.compress(b"bad"), month, 100000)
    with pytest.raises(DataError):
        decode_hourly(packed, month, 0)


def test_duplicate_and_invalid_ohlc_bi5():
    month = pd.Timestamp("2020-01-01", tz="UTC")
    record = struct.pack(">5if", 0, 100, 101, 99, 102, 1)
    with pytest.raises(DataError, match="duplicado"):
        decode_hourly(lzma.compress(record * 2), month, 100)
    bad = struct.pack(">5if", 0, 100, 110, 99, 102, 1)
    with pytest.raises(DataError, match="OHLC"):
        decode_hourly(lzma.compress(bad), month, 100)


def test_incremental_cache_no_network_and_detects_corruption(tmp_path, monkeypatch):
    month = pd.Timestamp("2020-01-01", tz="UTC")
    packed = lzma.compress(struct.pack(">5if", 0, 110000, 110010, 109990, 110020, 4.0))

    class Response:
        content = packed
        status_code = 200
        url = "https://test.invalid/fixture"

        def raise_for_status(self):
            pass

    calls = []

    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            return Response()

    from trading_ml.multimarket import data

    monkeypatch.setattr(data._THREAD, "session", Session(), raising=False)
    _month_file(tmp_path, "EURUSD", month, "BID", False)
    _month_file(tmp_path, "EURUSD", month, "BID", False)
    assert len(calls) == 1
    (tmp_path / "EURUSD" / "2020-01_BID.bi5").write_bytes(b"tampered")
    with pytest.raises(DataError, match="Checksum"):
        _month_file(tmp_path, "EURUSD", month, "BID", False)


def test_calendar_weekends_and_dst():
    values = pd.to_datetime(
        [
            "2024-03-08T21:00Z",
            "2024-03-08T22:00Z",
            "2024-03-10T20:00Z",
            "2024-03-10T21:00Z",
            "2024-11-03T21:00Z",
            "2024-11-03T22:00Z",
        ]
    )
    assert fx_open(values).tolist() == [True, False, False, True, False, True]
    end = session_end(pd.to_datetime(["2024-03-11T00:00Z", "2024-11-04T00:00Z"]))
    assert end.tolist() == list(pd.to_datetime(["2024-03-11T21:00Z", "2024-11-04T22:00Z"]))


def test_aggregation_rejects_fragments_does_not_impute(tmp_path):
    frame = quotes(8)
    metadata = {
        "interval": "1h",
        "instrument": instrument("EURUSD").contract(),
        "quality": {},
        "start": frame.timestamp.min().isoformat(),
        "end_exclusive": frame.reference_time.max().isoformat(),
    }
    path = _snapshot(tmp_path, "EURUSD", frame.drop(index=2), metadata)
    aggregated = aggregate_forex(tmp_path, path, "4h")
    result, record = load_snapshot(aggregated)
    assert len(result) == 1
    assert result.timestamp.iloc[0] == frame.timestamp.iloc[4]
    assert len(record["quality"]["incomplete_buckets_excluded"]) == 1
    assert result.open_ask.iloc[0] == pytest.approx(frame.open_ask.iloc[4])
    assert result.close_bid.iloc[0] == pytest.approx(frame.close_bid.iloc[7])
    assert aggregate_forex(tmp_path, path, "1h") == path


def test_snapshot_hash_is_enforced(tmp_path):
    path = _snapshot(tmp_path, "EURUSD", quotes(4), {"interval": "1h"})
    with path.open("a") as stream:
        stream.write("changed")
    with pytest.raises(DataError, match="Checksum"):
        load_snapshot(path)


def test_features_prefix_invariance_and_no_volume():
    raw = quotes(180)
    first = features(raw.iloc[:120], "EURUSD")
    changed = raw.copy()
    changed.loc[120:, ["open", "high", "low", "close", "close_ask", "close_bid"]] *= 2
    second = features(changed, "EURUSD").iloc[:120]
    columns = recipes("EURUSD")["compact"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert not any("volume" in c for c in columns)
    assert "volume_ratio" not in first
    assert "spread_fraction" in columns


def test_legacy_features_default_unchanged_with_explicit_volume():
    raw = quotes(100)
    raw["volume"] = np.arange(100) + 1
    original = build_features(raw, Features(enhanced=True))
    without = build_features(
        raw.drop(columns="volume"), Features(enhanced=True), include_volume=False
    )
    assert "volume_ratio" in original and "volume_change_1" in original
    pd.testing.assert_frame_equal(
        original.drop(columns=["volume", "volume_ratio", "volume_change_1"]), without
    )


@pytest.mark.parametrize(
    "symbol,bid,ask,amount,expected",
    [
        ("EURUSD", 1.1, 1.2, 10, 10),
        ("USDJPY", 100, 101, 101, 1),
        ("USDJPY", 100, 101, -100, -1),
        ("USDCHF", 0.9, 0.91, 0.91, 1),
        ("BTCUSDT", 10000, 10001, 10, 10),
    ],
)
def test_conversion_uses_correct_observed_side(symbol, bid, ask, amount, expected):
    assert conversion(symbol, bid, ask, amount) == pytest.approx(expected)


def test_rollover_triple_wednesday_and_weekend():
    assert (
        _swap_rollovers(pd.Timestamp("2024-01-10T21:00Z"), pd.Timestamp("2024-01-10T23:00Z")) == 3
    )
    assert (
        _swap_rollovers(pd.Timestamp("2024-01-12T23:00Z"), pd.Timestamp("2024-01-14T23:00Z")) == 0
    )


def test_spread_and_costs_make_flat_price_unprofitable():
    frame = quotes(8)
    frame[["open", "high", "low", "close", "open_bid", "high_bid", "low_bid", "close_bid"]] = 1.1
    frame[["open_ask", "high_ask", "low_ask", "close_ask"]] = 1.1002
    frame["atr_ratio"] = 0.01
    event = Engine(frame, "EURUSD", Costs()).event(0, 1, False)
    assert event["entry_price"] > 1.1002
    assert event["exit_price"] < 1.1
    assert event["target"] == 0
    assert event["unit_net_return"] < -0.00018


def test_barrier_ambiguous_stop_first_and_gap_execution():
    frame = quotes(8)
    frame["atr_ratio"] = 0.001
    frame.loc[1, "low_bid"] = 1.0
    frame.loc[1, "high_bid"] = 1.2
    engine = Engine(frame, "EURUSD", Costs())
    event = engine.event(0, 3, True)
    assert event["ambiguous"] and event["reason"] == "sl"
    frame.loc[1, "low_bid"] = frame.open_bid.iloc[1]
    frame.loc[1, "high_bid"] = frame.open_bid.iloc[1]
    frame.loc[2, "open_bid"] = 0.9
    event = Engine(frame, "EURUSD", Costs()).event(0, 3, True)
    assert event["reason"] == "sl_gap"
    assert event["exit_price"] < 0.9


def test_direction_label_and_next_open_event_are_distinct():
    frame = features(quotes(120), "EURUSD")
    engine = Engine(frame, "EURUSD", Costs())
    direction = labelled(frame, engine, 1, "direction", False)
    net = labelled(frame, engine, 1, "net_profit", False)
    index = 80
    assert direction.target_time.iloc[index] == frame.reference_time.iloc[index + 1]
    assert net.target_time.iloc[index] == frame.timestamp.iloc[index + 2]


def test_sizing_and_entry_do_not_consult_future_outcomes():
    frame = features(quotes(300), "EURUSD")
    signals = frame.iloc[[80]][["source_index", "reference_time"]].copy()
    signals["probability"] = 0.8
    engine = Engine(frame, "EURUSD", Costs())
    end = frame.reference_time.iloc[-1]
    _, trades, _ = engine.portfolio(signals, 1, False, 0.55, end)
    assert len(trades) == 1
    changed = frame.copy()
    changed.loc[82:, ["open_bid", "close_bid", "high_bid", "low_bid", "open_ask", "close_ask"]] *= (
        1.01
    )
    _, other, _ = Engine(changed, "EURUSD", Costs()).portfolio(signals, 1, False, 0.55, end)
    assert other.units.iloc[0] == trades.units.iloc[0]
    assert other.entry_price.iloc[0] == trades.entry_price.iloc[0]
    assert other.profit_usd.iloc[0] != trades.profit_usd.iloc[0]
    assert trades.units.iloc[0] * trades.entry_price.iloc[0] <= Costs().initial_capital


def test_current_spread_restriction_and_no_entry_on_weekend():
    frame = features(quotes(300), "EURUSD")
    signals = frame.iloc[[80]][["source_index", "reference_time"]].copy()
    signals["probability"] = 0.8
    summary, _, _ = Engine(
        frame, "EURUSD", replace(Costs(), maximum_entry_spread_pips=1)
    ).portfolio(signals, 1, False, 0.55, frame.reference_time.iloc[-1])
    assert summary["trades"] == 0


def test_predict_never_builds_labels_or_uses_future_prices(tmp_path, monkeypatch):
    from trading_ml.multimarket import service

    raw = quotes(180)
    path = _snapshot(
        tmp_path, "EURUSD", raw, {"interval": "1h", "instrument": instrument("EURUSD").contract()}
    )
    model_path = tmp_path / "model.ubj"
    model_path.write_bytes(b"test-double")
    metadata = {
        "symbol": "EURUSD",
        "interval": "1h",
        "last_train_target": raw.reference_time.iloc[50].isoformat(),
        "last_calibration_target": raw.reference_time.iloc[70].isoformat(),
        "specification": {"target": "direction"},
    }
    monkeypatch.setattr(service, "load_candidate", lambda _: (None, metadata))
    monkeypatch.setattr(
        service, "labelled", lambda *a, **k: pytest.fail("Prediction constructed labels")
    )
    seen = []

    def p(_, frame, calibrated):
        seen.append(frame.reference_time.max())
        return np.array([0.6]), None

    monkeypatch.setattr(service, "probabilities", p)
    when = raw.reference_time.iloc[100]
    output = predict_market(model_path, path, when.isoformat())
    assert output["probability"] == 0.6 and seen == [when]
    assert output["orders_submitted"] == 0
    with pytest.raises(LeakageError):
        predict_market(model_path, path, raw.reference_time.iloc[60].isoformat())


def test_protocol_and_supported_markets():
    protocol = load_protocol(Path("config/multimarket.toml"))
    assert len(protocol["symbols"]) == 7
    assert instrument("USD/JPY").pip == 0.01
    assert instrument("BTC/USDT").market == "crypto"
    with pytest.raises(DataError):
        instrument("EURGBP")


def test_snapshot_is_append_only(tmp_path):
    frame = quotes(4)
    meta = {"interval": "1h"}
    path = _snapshot(tmp_path, "EURUSD", frame, meta)
    before = read_json(sidecar(path))
    assert _snapshot(tmp_path, "EURUSD", frame, meta) == path
    assert read_json(sidecar(path)) == before


def test_execution_enters_h1_even_if_next_feature_bucket_is_incomplete():
    hourly = quotes(600)
    coarse = hourly.iloc[::4].copy().reset_index(drop=True)
    coarse["reference_time"] = coarse.timestamp + pd.Timedelta(hours=4)
    # A later incomplete feature bucket may be absent from the model dataset.
    # Its actual H1 prices must remain executable, regardless of that outcome.
    coarse = coarse.drop(index=66).reset_index(drop=True)
    coarse = features(coarse, "EURUSD")
    coarse["atr_ratio"] = 0.001
    hourly.loc[265, "low_bid"] = 1.0
    engine = Engine(coarse, "EURUSD", Costs(), execution_frame=hourly, bars_per_horizon=4)
    event = engine.event(65, 3, True)
    assert event["entry_index"] == 264
    assert event["entry_time"] == hourly.timestamp.iloc[264]
    assert event["entry_time"] < coarse.timestamp.iloc[66]
    assert event["exit_index"] in [264, 265]


def test_fast_rollover_counts_match_calendar_across_dst():
    raw = quotes(400, start="2024-03-07T00:00Z")
    frame = features(raw, "EURUSD")
    engine = Engine(frame, "EURUSD", Costs())
    for start, end in [(0, 200), (40, 90), (100, 300)]:
        assert engine.rollover_close[end] - engine.rollover_open[start] == _swap_rollovers(
            engine.time[start], engine.reference[end]
        )


def test_fxcm_parser_keeps_real_crossed_quotes_and_has_no_volume():
    frame = quotes(120)
    source = pd.DataFrame({"DateTime": frame.timestamp.dt.strftime("%m/%d/%Y %H:%M:%S.%f")})
    for side in ["Bid", "Ask"]:
        for field in ["open", "high", "low", "close"]:
            source[side + field.capitalize()] = frame[field + "_" + side.lower()]
    source.loc[90, "AskOpen"] = source.loc[90, "BidOpen"] - 0.00001
    source.loc[90, "AskClose"] = source.loc[90, "BidClose"] - 0.00001
    parsed = parse_fxcm(gzip.compress(source.to_csv(index=False).encode()))
    assert parsed.crossed_open.iloc[90] and parsed.crossed_close.iloc[90]
    assert parsed.open_ask.iloc[90] < parsed.open_bid.iloc[90]
    assert not any("volume" in c for c in parsed)
    computed = features(parsed, "EURUSD")
    assert pd.isna(computed.spread_fraction.iloc[90])
    assert Engine(computed, "EURUSD", Costs()).event(89, 1, False) is None


@pytest.mark.parametrize(
    "content",
    [b"", b"DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose\r\n"],
)
def test_fxcm_empty_archive_does_not_create_prices(content):
    frame = parse_fxcm(gzip.compress(content))
    assert frame.empty
    assert frame.timestamp.dtype.tz is not None


def test_dukascopy_429_stops_without_immediate_retries(tmp_path, monkeypatch):
    import threading

    from trading_ml.multimarket import data

    calls = []

    class Response:
        status_code = 429
        headers = {"Retry-After": "60"}

    class Session:
        def get(self, *a, **k):
            calls.append(1)
            return Response()

    monkeypatch.setattr(data, "_RATE_LIMITED", threading.Event())
    monkeypatch.setattr(data, "_NEXT_GET", 0)
    monkeypatch.setattr(data._THREAD, "session", Session(), raising=False)
    for month in ["2020-01-01", "2020-02-01"]:
        with pytest.raises(DataError, match="429"):
            _month_file(tmp_path, "EURUSD", pd.Timestamp(month, tz="UTC"), "BID", False)
    assert len(calls) == 1


def test_unknown_outcomes_abort_instead_of_hiding_future_trades():
    frame = features(quotes(120), "EURUSD")
    engine = Engine(frame, "EURUSD", Costs())
    rows = pd.DataFrame({"source_index": [119], "target": [np.nan]})
    with pytest.raises(DataError, match="Outcomes incompletos"):
        require_outcomes(rows, engine, "direction")
    with pytest.raises(DataError, match="Outcomes incompletos"):
        require_outcomes(rows, engine, "net_profit")


def test_tp_limit_sensitivity_at_temporal_expiry():
    frame = features(quotes(120), "EURUSD")
    frame["atr_ratio"] = 0.01
    frame.loc[82, "open_bid"] = 1.2
    ordinary = Engine(frame, "EURUSD", Costs()).event(80, 1, True)
    strict = Engine(frame, "EURUSD", Costs(), expiry_policy="barrier_first").event(80, 1, True)
    assert ordinary["reason"] == "timeout"
    assert strict["reason"] == "tp_gap_at_timeout"
    assert strict["exit_price"] < strict["entry_price"] * 1.015
    assert strict["unit_net_return"] < ordinary["unit_net_return"]


def test_spread_markup_is_a_cost_not_a_fabricated_quote():
    frame = features(quotes(120), "EURUSD")
    original = Engine(frame, "EURUSD", Costs()).event(80, 1, False)
    marked = Engine(frame, "EURUSD", replace(Costs(), forex_additional_spread_pips=1)).event(
        80, 1, False
    )
    assert marked["entry_price"] - original["entry_price"] == pytest.approx(0.0001)
    assert marked["unit_net_return"] < original["unit_net_return"]
    assert frame.open_ask.iloc[81] == quotes(120).open_ask.iloc[81]
