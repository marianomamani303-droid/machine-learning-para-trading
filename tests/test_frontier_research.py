from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_ml.common import write_new_json
from trading_ml.errors import DataError, LeakageError
from trading_ml.frontier_research.features import CROSS, LOCAL, multi_asset_features
from trading_ml.frontier_research.prospective import (
    require_future_available,
    validate_future_history,
)
from trading_ml.frontier_research.study import metrics
from trading_ml.frontier_research.targets import research_targets
from trading_ml.frontier_research.validation import (
    evaluation_rows,
    prediction_rows,
    temporal_partitions,
)
from trading_ml.momentum.config import ResearchCosts
from trading_ml.training.recipe import protect_reserved_labels


def raw(n=500, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    timestamp = pd.date_range("2020-01-01", periods=n, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "reference_time": timestamp + pd.Timedelta(days=1),
            "open": close * 0.997,
            "close": close,
            "high": close * 1.03,
            "low": close * 0.97,
            "volume": rng.uniform(100, 1000, n),
            "quote_volume": rng.uniform(10000, 30000, n),
            "taker_quote": rng.uniform(3000, 8000, n),
            "trades": rng.integers(100, 1000, n),
        }
    )


def test_cross_asset_prefix_is_causal_and_missing_peer_abstains():
    assets = {"BTCUSDT": raw(), "ETHUSDT": raw(seed=43)}
    full = multi_asset_features(assets)
    short = multi_asset_features({s: f.iloc[:300].copy() for s, f in assets.items()})
    for symbol in assets:
        np.testing.assert_allclose(
            full[symbol].iloc[:300][LOCAL + CROSS], short[symbol][LOCAL + CROSS], equal_nan=True
        )
    assets["ETHUSDT"] = assets["ETHUSDT"].drop(index=250)
    gap = multi_asset_features(assets)["BTCUSDT"]
    assert pd.isna(gap.peer_return_1.iloc[250])
    assert not gap.research_ready.iloc[250]


def test_targets_close_barrier_next_open_and_censoring():
    frame = raw(40)
    frame["risk_atr"] = 1.0
    frame.loc[1, "open"] = 100.0
    frame.loc[1, "close"] = 105.0
    frame.loc[2, "open"] = 97.0
    output = research_targets(frame, ResearchCosts())["barrier_close_h5"]
    assert output.target.iloc[0] == 1
    assert output.target_time.iloc[0] == frame.timestamp.iloc[2]
    assert output.net_return_target.iloc[0] < 0  # TP event can lose on an opening gap.
    assert output.target.iloc[-6:].isna().all()
    assert output.barrier_outcome.iloc[0] == "tp_close"


def test_purge_embargo_exact_availability_and_rolling():
    frame = raw()
    frame["research_ready"] = True
    frame["target"] = np.arange(len(frame)) % 2
    frame["target_time"] = frame.reference_time + pd.Timedelta(days=5)
    cutoff = frame.reference_time.iloc[450]
    train, cal = temporal_partitions(frame, cutoff, rolling_days=400)
    assert train.target_time.max() < cal.reference_time.min() - pd.Timedelta(days=6)
    assert cal.target_time.max() < cutoff - pd.Timedelta(days=6)
    assert train.reference_time.min() >= cutoff - pd.Timedelta(days=400)
    eval_rows = evaluation_rows(frame, cutoff, frame.reference_time.iloc[-1])
    assert eval_rows.target_time.max() < frame.reference_time.iloc[-1]
    with pytest.raises(DataError):
        temporal_partitions(frame, cutoff, embargo_days=1)


def test_final_test_cannot_be_opened_before_end():
    plan = {"end": "2027-10-13"}
    with pytest.raises(LeakageError, match="sellado"):
        require_future_available(plan, pd.Timestamp("2026-10-14", tz="UTC"))
    require_future_available(plan, pd.Timestamp("2027-10-13", tz="UTC"))


def test_balanced_accuracy_measures_majority_only_predictor():
    y = np.array([1] * 90 + [0] * 10)
    result = metrics(y, np.repeat(0.9, 100))
    assert result["accuracy"] == 0.9
    assert result["balanced_accuracy"] == 0.5


def test_future_reservation_protects_existing_training_commands(tmp_path):
    write_new_json(
        tmp_path / "frontier_studies" / "study" / "prospective.json",
        {"start": "2027-04-13", "end": "2027-10-13"},
    )
    leaked = pd.DataFrame(
        {
            "reference_time": pd.to_datetime(["2027-04-12"], utc=True),
            "target_time": pd.to_datetime(["2027-04-13"], utc=True),
        }
    )
    with pytest.raises(LeakageError):
        protect_reserved_labels(tmp_path, leaked)


def test_execution_availability_does_not_depend_on_future_barrier_outcome():
    frame = raw(30)
    frame["research_ready"] = True
    frame["target"] = 1.0
    frame["target_time"] = frame.reference_time + pd.Timedelta(days=1)
    start, stop = frame.reference_time.iloc[20], frame.reference_time.iloc[25]
    short = prediction_rows(frame, start, stop)
    scored_short = evaluation_rows(frame, start, stop)
    frame["target_time"] = frame.reference_time + pd.Timedelta(days=6)
    pd.testing.assert_series_equal(
        prediction_rows(frame, start, stop).reference_time, short.reference_time
    )
    assert len(evaluation_rows(frame, start, stop)) < len(scored_short)


def test_future_data_must_include_warmup_and_full_reserved_window():
    snapshot = raw(100)
    late_only = raw(200)
    late_only["timestamp"] += pd.Timedelta(days=365)
    late_only["reference_time"] += pd.Timedelta(days=365)
    with pytest.raises(DataError, match="warmup"):
        validate_future_history(
            late_only,
            snapshot,
            late_only.reference_time.iloc[10],
            late_only.reference_time.iloc[150],
        )
