import hashlib
import io
import zipfile
from unittest.mock import Mock

import pytest
import requests

from trading_ml.data.binance import _month
from trading_ml.errors import DataError


def archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        output.writestr("data.csv", "1577836800000,100,110,90,105,1000,1577923199999,0,1,0,0,0\n")
    return buffer.getvalue()


def test_download_checksum_cache_and_corruption(config, monkeypatch):
    payload = archive()
    checksum = hashlib.sha256(payload).hexdigest()
    response_checksum = Mock(text=checksum + "  data.zip")
    response_data = Mock(content=payload)
    get = Mock(side_effect=[response_checksum, response_data])
    monkeypatch.setattr(requests.Session, "get", get)
    frame, manifest = _month(config, "2020-01", False)
    assert len(frame) == 1
    assert manifest["sha256"] == checksum
    assert get.call_count == 2
    _month(config, "2020-01", False)
    assert get.call_count == 2
    cache = config.root / "cache" / "BTCUSDT-1d-2020-01.zip"
    cache.write_bytes(b"bad")
    with pytest.raises(DataError, match="corrupta"):
        _month(config, "2020-01", False)
    get.side_effect = [response_checksum, response_data]
    _month(config, "2020-01", True)
    assert get.call_count == 4
    assert cache.read_bytes() == payload


def test_remote_checksum_mismatch_rejected(config, monkeypatch):
    get = Mock(side_effect=[Mock(text="0" * 64), Mock(content=archive())])
    monkeypatch.setattr(requests.Session, "get", get)
    with pytest.raises(DataError, match="Checksum incorrecto"):
        _month(config, "2020-01", False)
    assert not (config.root / "cache" / "BTCUSDT-1d-2020-01.zip").exists()


def test_http_failure_is_actionable(config, monkeypatch):
    monkeypatch.setattr(requests.Session, "get", Mock(side_effect=requests.HTTPError("404")))
    with pytest.raises(DataError, match="end anterior"):
        _month(config, "2020-01", False)
