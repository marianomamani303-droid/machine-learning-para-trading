from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.errors import DataError, LeakageError
from trading_ml.expansion.data import archive, load_snapshot, parse_mark_archive, snapshot
from trading_ml.expansion.features import (
    DERIVATIVES,
    FLOW,
    attach_context,
    crypto_features,
    latest_observations,
    make_panel,
    open_interest_features,
    point_in_time_join,
)


def market(rows=1100):
    rng = np.random.default_rng(17)
    opening = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, rows)))
    close = opening * np.exp(rng.normal(0, 0.01, rows))
    f = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=rows, tz="UTC"),
            "open": opening,
            "close": close,
            "high": np.maximum(opening, close) * 1.02,
            "low": np.minimum(opening, close) * 0.98,
            "volume": rng.uniform(10, 100, rows),
            "trades": rng.integers(100, 2000, rows),
        }
    )
    f["quote_volume"] = f.volume * f.close
    f["taker_quote"] = f.quote_volume * rng.uniform(0.1, 0.9, rows)
    f["taker_base"] = f.volume * 0.5
    f["available_at"] = f.timestamp + pd.Timedelta(days=1)
    perp = f[["timestamp", "open", "high", "low", "close", "available_at"]].copy()
    perp[["open", "high", "low", "close"]] *= 1.001
    event = pd.date_range("2020-01-01", periods=3 * rows, freq="8h", tz="UTC")
    fund = pd.DataFrame(
        {
            "event_time": event,
            "available_at": event + pd.Timedelta(seconds=1),
            "funding_rate": rng.normal(0.0001, 0.00005, len(event)),
            "funding_interval_hours": 8,
        }
    )
    return f, perp, fund


def test_derivative_header_and_headerless():
    rows = "1577836800000,10,12,9,11,0,1577923199999,0,1,0,0,0\n"
    for content in [
        rows,
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        + rows,
    ]:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as z:
            z.writestr("data.csv", content)
        actual = parse_mark_archive(stream.getvalue())
        assert len(actual) == 1 and actual.close.iloc[0] == 11
        assert actual.timestamp.iloc[0] == pd.Timestamp("2020-01-01", tz="UTC")


def test_availability_not_observation_date_and_revisions():
    events = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01"], utc=True),
            "available_at": pd.to_datetime(["2020-02-10", "2020-03-10", "2020-04-10"], utc=True),
            "value": [1.0, 2.0, 999.0],
        }
    )
    refs = pd.to_datetime(["2020-02-01", "2020-03-15", "2020-04-15"], utc=True)
    result = latest_observations(events, refs)
    assert pd.isna(result.value.iloc[0])
    assert result.value.iloc[1:].tolist() == [2.0, 2.0]
    bad = events.copy()
    bad.loc[0, "available_at"] = pd.Timestamp("2019-01-01", tz="UTC")
    with pytest.raises(LeakageError):
        latest_observations(bad, refs)
    joined = point_in_time_join(refs, events.iloc[:2], ["value"], pd.Timedelta(days=30))
    assert pd.isna(joined.value.iloc[0]) and joined.value.iloc[1] == 2
    assert pd.isna(joined.value.iloc[2])  # stale information stays unavailable


def test_all_features_are_prefix_invariant():
    spot, perp, fund = market()
    before = crypto_features(spot, perp, fund)
    cut = spot.timestamp.iloc[850]
    changed_spot = spot.copy()
    changed_perp = perp.copy()
    changed_fund = fund.copy()
    changed_spot.loc[850:, ["open", "high", "low", "close"]] *= 3
    changed_spot.loc[850:, ["quote_volume", "taker_quote"]] *= 4
    changed_perp.loc[850:, ["open", "high", "low", "close"]] *= 5
    changed_fund.loc[changed_fund.event_time >= cut, "funding_rate"] = 0.5
    after = crypto_features(changed_spot, changed_perp, changed_fund)
    columns = [*FLOW, *DERIVATIVES]
    mask = before.timestamp < cut
    pd.testing.assert_frame_equal(before.loc[mask, columns], after.loc[mask, columns])
    for day in [before.iloc[100], before.iloc[300]]:
        known = fund.loc[
            (fund.available_at >= day.reference_time - pd.Timedelta(days=1))
            & (fund.available_at < day.reference_time)
        ]
        assert day.funding_1d == pytest.approx(known.funding_rate.sum())
    assert before.perp_premium.dropna().mean() == pytest.approx(0.001)


def test_forex_context_closes_and_staleness():
    spot, perp, fund = market()
    f = crypto_features(spot, perp, fund)
    fx = spot.copy()
    fx["reference_time"] = fx.timestamp + pd.Timedelta(hours=22)
    fx = fx.iloc[:750]
    out, names = attach_context({"BTC": f.copy(), "ETH": f.copy()}, {"EURUSD": fx})
    assert "EURUSD_unavailable" in names
    assert out["BTC"].EURUSD_unavailable.iloc[-1] == 1
    assert out["BTC"].EURUSD_fx_return.iloc[-1] == 0
    expected = out["BTC"].iloc[:300][list(names)].copy()
    fx.loc[fx.index >= 800, "close"] *= 9
    repeated, _ = attach_context({"BTC": f.copy(), "ETH": f.copy()}, {"EURUSD": fx})
    pd.testing.assert_frame_equal(expected, repeated["BTC"].iloc[:300][list(names)])


def test_oi_transform_and_snapshot_integrity(tmp_path):
    f = pd.DataFrame(
        {
            "available_at": pd.date_range("2020-01-01", periods=100, tz="UTC"),
            "open_interest": np.arange(1, 101),
        }
    )
    out = open_interest_features(f)
    assert out.oi_change.iloc[-1] == pytest.approx(np.log(100 / 99))
    assert np.isfinite(out.oi_z63.iloc[-1])
    path = snapshot(tmp_path, "BTC", "oi", out, {"frequency": "1d"})
    loaded, _ = load_snapshot(path)
    assert len(loaded) == 100
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(DataError):
        load_snapshot(path)


def test_cache_checksum_and_reuse(tmp_path, monkeypatch):
    import hashlib

    import requests

    payload = b"test immutable payload"
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        r = requests.Response()
        r.status_code = 200
        r._content = (
            (hashlib.sha256(payload).hexdigest() + " data.zip").encode()
            if url.endswith("CHECKSUM")
            else payload
        )
        return r

    monkeypatch.setattr(requests, "get", get)
    p, meta = archive(tmp_path, "https://example.test/data.zip")
    p2, m2 = archive(tmp_path, "https://example.test/data.zip")
    assert p == p2 == payload and meta == m2 and len(calls) == 2
    Path(meta["payload"]).write_bytes(b"bad")
    with pytest.raises(DataError):
        archive(tmp_path, "https://example.test/data.zip")


def test_funding_variable_precision_and_missing_event(tmp_path):
    from trading_ml.expansion.data import validate_funding

    _, _, fund = market(100)
    fund.loc[1, "event_time"] += pd.Timedelta(milliseconds=8)
    path = snapshot(tmp_path, "BTC", "funding", fund, {})
    loaded, _ = load_snapshot(path)
    assert loaded.event_time.iloc[1].microsecond == 8000
    with pytest.raises(DataError, match="Funding incompleto"):
        validate_funding(loaded.drop(index=50))


def test_dynamic_model_schema_and_future_invariance(tmp_path):
    torch = pytest.importorskip("torch")
    import tomllib

    from trading_ml.dmn.data import FEATURES, temporal_split
    from trading_ml.dmn.models import fit_network, load_network

    with Path("config/expansion.toml").open("rb") as stream:
        p = tomllib.load(stream)
    p["training"].update(epochs=1, hidden_size=4, minimum_train_dates=30)
    raw, perp, fund = market()
    f = crypto_features(raw, perp, fund).iloc[64:].reset_index(drop=True)
    panel = make_panel({"BTC": f}, (*FEATURES, *FLOW), sequence=8)
    tr, val, test = temporal_split(
        panel,
        pd.Timestamp("2022-07-01", tz="UTC"),
        pd.Timestamp("2023-01-01", tz="UTC"),
        p["training"],
    )
    rec = {"id": "dynamic", "architecture": "lstm", "objective": "sharpe", "net_training": True}
    m, _, meta = fit_network(panel, tr, val, rec, p, 42, tmp_path / "a")
    loaded, _ = load_network(tmp_path / "a", expected_features=panel.feature_names)
    for key, v in m.state_dict().items():
        torch.testing.assert_close(v, loaded.state_dict()[key])
    with pytest.raises(DataError):
        load_network(tmp_path / "a")
    before = {key: value.clone() for key, value in m.state_dict().items()}
    panel.x = panel.x.copy()
    panel.x[test] *= -50
    panel.returns[test] = 0.9
    changed, _, meta2 = fit_network(panel, tr, val, rec, p, 42, tmp_path / "b")
    assert meta["history"] == meta2["history"]
    for key, v in changed.state_dict().items():
        torch.testing.assert_close(v, before[key], rtol=0, atol=0)


def test_expansion_study_end_to_end_and_resume(tmp_path):
    pytest.importorskip("torch")
    from trading_ml.common import sha256, write_new_json
    from trading_ml.config import Settings
    from trading_ml.expansion.study import run_study, verify

    spot, perp, funding = market(1096)
    index = {"start": "2020-01-01", "end": "2023-01-01", "assets": {}}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        index["assets"][symbol] = {
            kind: str(snapshot(tmp_path, symbol, kind, frame, {}).resolve())
            for kind, frame in [("spot", spot), ("perp", perp), ("funding", funding)]
        }
    source = tmp_path / "dataset.json"
    write_new_json(source, index)
    text = Path("config/expansion.toml").read_text(encoding="utf-8")
    text = text.replace('start = "2021-01-01"', 'start = "2020-01-01"').replace(
        'end = "2025-07-01"', 'end = "2023-01-01"'
    )
    text = text.replace(
        'symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]',
        'symbols = ["BTCUSDT", "ETHUSDT"]',
    )
    text = text.replace(
        'forex = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]',
        "forex = []",
    )
    text = text.replace(
        'boundaries = ["2023-07-01", "2024-01-01", "2025-01-01", "2025-07-01"]',
        'boundaries = ["2022-07-01", "2023-01-01"]',
    )
    text = text.replace("seeds = [42, 1729]", "seeds = [42]").replace(
        "maximum_fits = 72", "maximum_fits = 12"
    )
    text = text.replace("epochs = 30", "epochs = 1").replace("hidden_size = 20", "hidden_size = 4")
    cfg = tmp_path / "protocol.toml"
    cfg.write_text(text, encoding="utf-8")
    report = run_study(Settings(root=tmp_path), cfg, source)
    directory = Path(report["directory"])
    assert report["actual_fits"] == 12
    assert verify(directory)["intact"]
    before = {str(path): sha256(path) for path in directory.rglob("*") if path.is_file()}
    assert run_study(Settings(root=tmp_path), cfg, source, resume=directory) == report
    after = {str(path): sha256(path) for path in directory.rglob("*") if path.is_file()}
    assert before == after
