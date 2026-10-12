import io
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_ml.common import read_json, sidecar, write_json
from trading_ml.config import Settings
from trading_ml.config.settings import Market
from trading_ml.data.binance import parse_archive
from trading_ml.errors import DataError, TradingMLError
from trading_ml.features import build_features, feature_columns
from trading_ml.preprocessing import clean_ohlcv, load_prepared, prepare


def test_duplicate_removed_and_sorted(raw):
    messy = pd.concat([raw.iloc[::-1], raw.iloc[[20]]], ignore_index=True)
    clean, quality = clean_ohlcv(messy, pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(clean, raw)
    assert quality["duplicates_removed"] == 1


def test_conflicting_duplicate_rejected(raw):
    extra = raw.iloc[[20]].copy()
    extra["volume"] += 1
    with pytest.raises(DataError, match="contradictorios"):
        clean_ohlcv(pd.concat([raw, extra]), pd.Timedelta(days=1))


@pytest.mark.parametrize("problem", ["gap", "missing", "infinite", "price", "ohlc", "alignment"])
def test_bad_data_rejected(raw, problem):
    if problem == "gap":
        raw = raw.drop(index=20)
    elif problem == "missing":
        raw.loc[20, "volume"] = np.nan
    elif problem == "infinite":
        raw.loc[20, "close"] = np.inf
    elif problem == "price":
        raw.loc[20, "open"] = -1
    elif problem == "ohlc":
        raw.loc[20, "high"] = 1
    else:
        raw.loc[20, "timestamp"] += pd.Timedelta(minutes=1)
    with pytest.raises(DataError):
        clean_ohlcv(raw, pd.Timedelta(days=1))


def test_features_do_not_change_when_future_changes(raw):
    settings = Settings().features
    original = build_features(raw, settings)
    changed = raw.copy()
    changed.loc[301:, ["open", "high", "low", "close", "volume"]] *= 7
    modified = build_features(changed, settings)
    columns = feature_columns(settings)
    pd.testing.assert_frame_equal(original.loc[:300, columns], modified.loc[:300, columns])
    prefix = build_features(raw.iloc[:301], settings)
    pd.testing.assert_frame_equal(original.loc[:300, columns], prefix[columns])


def test_rsi_constant_and_uptrend(raw):
    for column in ["open", "high", "low", "close"]:
        raw[column] = 100.0
    result = build_features(raw, Settings().features)
    assert (result.rsi.dropna() == 0.5).all()
    raw["close"] = np.arange(len(raw)) + 100.0
    assert (build_features(raw, Settings().features).rsi.dropna() == 1).all()


@pytest.mark.parametrize("unit,multiplier", [("ms", 1), ("us", 1000)])
def test_archive_timestamp_units(unit, multiplier):
    timestamp = 1577836800000 * multiplier
    row = f"{timestamp},100,110,90,105,1000,{timestamp},0,1,0,0,0\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("candles.csv", row)
    frame = parse_archive(buffer.getvalue())
    assert frame.timestamp.iloc[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert frame.close.iloc[0] == 105


def test_prepared_labels_and_tail(prepared, config):
    frame, metadata = load_prepared(prepared, config.contract())
    assert len(frame) == 551
    assert frame.target.isna().sum() == 1
    assert frame.target_time.isna().sum() == 1
    np.testing.assert_allclose(
        frame.future_return.iloc[:-1], frame.close.shift(-1).iloc[:-1] / frame.close.iloc[:-1] - 1
    )
    assert (frame.target_time.dropna() - frame.reference_time.iloc[:-1] == config.delta).all()
    assert metadata["warmup_rows_removed"] == 49


def test_checksum_and_contract_enforced(prepared):
    with pytest.raises(DataError, match="incompatibles"):
        load_prepared(prepared, {"wrong": True})
    prepared.write_text(prepared.read_text() + "\n")
    with pytest.raises(DataError, match="manifiesto"):
        load_prepared(prepared)


def test_incomplete_coverage_rejected(config, raw):
    from conftest import raw_file

    with pytest.raises(DataError, match="Cobertura"):
        prepare(config, raw_file(config, raw.iloc[1:]))


def test_interior_feature_gaps_rejected(config, raw):
    from conftest import raw_file

    raw.loc[100:150, "volume"] = 0.0
    with pytest.raises(DataError, match="calentamiento"):
        prepare(config, raw_file(config, raw))


def test_raw_manifest_enforced(config, raw):
    from conftest import raw_file

    path = raw_file(config, raw)
    metadata = read_json(sidecar(path))
    metadata["sha256"] = "bad"
    write_json(sidecar(path), metadata)
    with pytest.raises(DataError, match="manifiesto"):
        prepare(config, path)


def test_invalid_config():
    with pytest.raises(TradingMLError):
        replace(Settings(), market=Market(symbol="../../escape")).validate()
    with pytest.raises(TradingMLError):
        replace(Settings(), search={"enabled": True, "max_depth": []}).validate()
