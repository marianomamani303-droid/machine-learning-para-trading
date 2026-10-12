from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_ml.common import sha256, sidecar, write_json
from trading_ml.config import Settings
from trading_ml.config.settings import Market, Training
from trading_ml.preprocessing import prepare


def candles(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, rows)))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq="1D", tz="UTC"),
            "open": opening,
            "high": np.maximum(close, opening) * 1.01,
            "low": np.minimum(close, opening) * 0.99,
            "close": close,
            "volume": rng.uniform(100, 500, rows),
        }
    )


def raw_file(config: Settings, frame: pd.DataFrame):
    path = config.root / "data" / "raw.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    write_json(
        sidecar(path), {"kind": "raw", "market": config.to_dict()["market"], "sha256": sha256(path)}
    )
    return path


@pytest.fixture
def raw():
    return candles()


@pytest.fixture
def config(tmp_path):
    return replace(
        Settings(),
        root=tmp_path,
        market=Market(end="2021-08-23"),
        training=Training(min_partition_rows=30),
        model={**Settings().model, "n_estimators": 20},
        search={"enabled": True, "max_depth": [2, 3], "learning_rate": [0.03]},
    )


@pytest.fixture
def prepared(config, raw):
    return prepare(config, raw_file(config, raw))
