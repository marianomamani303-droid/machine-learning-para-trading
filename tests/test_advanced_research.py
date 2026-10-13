from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from conftest import raw_file

from trading_ml.advanced_research.data import recover_microstructure
from trading_ml.advanced_research.features import (
    CORE,
    build_research_features,
    family_columns,
)
from trading_ml.advanced_research.meta import causal_opportunity_labels, fit_meta
from trading_ml.advanced_research.models import fit_recipe, load_fitted
from trading_ml.advanced_research.statistics import paired_predictive
from trading_ml.advanced_research.targets import build_targets
from trading_ml.common import read_json, sha256, sidecar, write_json
from trading_ml.data.binance import parse_archive
from trading_ml.errors import DataError
from trading_ml.momentum.config import ResearchCosts
from trading_ml.risk_overlay.config import RiskPolicy
from trading_ml.risk_overlay.simulator import run_overlay


def flow_frame(raw):
    result = raw.copy()
    result["reference_time"] = result.timestamp + pd.Timedelta(days=1)
    rng = np.random.default_rng(41)
    result["quote_volume"] = result.volume * result.close
    result["trades"] = rng.integers(100, 500, len(result))
    share = rng.uniform(0.3, 0.7, len(result))
    result["taker_base"] = result.volume * share
    result["taker_quote"] = result.quote_volume * share
    return result


def test_flow_features_and_smc_are_prefix_causal(raw):
    raw = flow_frame(raw)
    features = build_research_features(raw)
    prefix = build_research_features(raw.iloc[:350])
    pd.testing.assert_frame_equal(features.iloc[:350], prefix)
    changed = raw.copy()
    changed.loc[350:, ["open", "high", "low", "close", "quote_volume", "taker_quote"]] *= 4
    second = build_research_features(changed)
    pd.testing.assert_frame_equal(features.iloc[:350], second.iloc[:350])
    for family in [
        "original",
        "core",
        "momentum",
        "reversion",
        "breakout",
        "flow",
        "smc",
        "flow_momentum",
    ]:
        assert not {"target", "target_time", "net_return_target", "future_return"} & set(
            family_columns(family)
        )


@pytest.mark.parametrize("unit,multiplier", [("ms", 1000), ("us", 1000000)])
def test_microstructure_reader_preserves_legacy_schema(unit, multiplier):
    ticks = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * multiplier)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "data.csv",
            f"{ticks},100,110,90,105,1000,{ticks + 86400 * multiplier - 1},105000,100,400,42000,0\n",
        )
    legacy = parse_archive(stream.getvalue())
    expanded = parse_archive(stream.getvalue(), include_microstructure=True)
    assert list(legacy) == ["timestamp", "open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(legacy, expanded[list(legacy)])
    assert expanded.taker_quote.iloc[0] == 42000
    assert expanded.timestamp.iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")


def test_archive_recovery_requires_verified_original_fields(config, raw):
    source = raw_file(config, raw)
    expanded = flow_frame(raw)
    numeric = expanded.drop(columns=["reference_time"])
    records = []
    for row in numeric.itertuples():
        ticks = int(row.timestamp.timestamp() * 1000)
        records.append(
            [
                ticks,
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
                ticks + 86400000 - 1,
                row.quote_volume,
                row.trades,
                row.taker_base,
                row.taker_quote,
                0,
            ]
        )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("data.csv", pd.DataFrame(records).to_csv(index=False, header=False))
    payload = stream.getvalue()
    archive_path = config.root / "cache" / "BTCUSDT-1d-2020-01.zip"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    archive_path.with_suffix(".zip.CHECKSUM").write_text(digest, encoding="ascii")
    manifest = read_json(sidecar(source))
    manifest["sources"] = [
        {"url": "https://data.binance.vision/BTCUSDT-1d-2020-01.zip", "sha256": digest}
    ]
    write_json(sidecar(source), manifest)
    before = sha256(source)
    recovered, provenance = recover_microstructure(config, source)
    assert len(recovered) == len(raw)
    assert provenance["missing_flow_imputed"] is False
    assert sha256(source) == before
    archive_path.write_bytes(b"corrupt")
    with pytest.raises(DataError):
        recover_microstructure(config, source)


def test_net_target_executes_next_open_with_costs(raw):
    frame = build_research_features(flow_frame(raw))
    costs = ResearchCosts()
    targets = build_targets(frame, costs)
    index = 100
    for horizon in [3, 5]:
        dataset = targets[f"net_h{horizon}"]
        expected = (
            raw.open.iloc[index + 1 + horizon]
            / raw.open.iloc[index + 1]
            * costs.sell_factor
            / costs.buy_factor
            - 1
        )
        assert dataset.net_return_target.iloc[index] == pytest.approx(expected)
        assert dataset.target_time.iloc[index] == raw.timestamp.iloc[index + 1 + horizon]
        assert dataset.target.iloc[index] == int(expected > 0)


@pytest.mark.parametrize("kind", ["xgb", "logistic", "regression"])
def test_models_scalers_and_calibration_ignore_post_cutoff_data(raw, kind):
    frame = build_research_features(flow_frame(raw))
    datasets = build_targets(frame, ResearchCosts())
    key = "net_h3" if kind == "regression" else "direction_h1"
    dataset = datasets[key]
    cutoff = dataset.reference_time.iloc[450]
    protocol = {
        "research": {
            "minimum_train_rows": 120,
            "rolling_days": 730,
            "calibration_days": 90,
            "seed": 42,
        },
        "xgboost_reference": {"n_estimators": 5, "max_depth": 2},
        "xgboost_regularized": {"n_estimators": 5, "max_depth": 2},
        "logistic": {"C": 1.0, "max_iter": 2000},
    }
    recipe = {
        "id": "test",
        "kind": "logistic" if kind == "logistic" else "xgb",
        "regression": kind == "regression",
        "reserve_calibration": kind == "regression",
    }
    model = fit_recipe(dataset, CORE, cutoff, recipe, protocol, lambda: None)
    altered = dataset.copy()
    future = altered.reference_time >= cutoff
    altered.loc[future, CORE] = 999
    altered.loc[altered.target_time >= cutoff, "target"] = 1
    altered.loc[altered.target_time >= cutoff, "net_return_target"] = 3
    second = fit_recipe(altered, CORE, cutoff, recipe, protocol, lambda: None)
    assert model.audit == second.audit
    assert model.calibration == second.calibration
    evaluation = dataset.loc[dataset.research_ready & (dataset.reference_time < cutoff)]
    np.testing.assert_array_equal(model.probabilities(evaluation), second.probabilities(evaluation))


def test_meta_censors_unforecasted_exit_and_purges_actual_label_time(raw):
    frame = build_research_features(flow_frame(raw))
    base = np.full(len(frame), 0.65)
    policy = RiskPolicy(
        name="combined", atr_stop=True, persistent=True, confirmations=2, max_holding_bars=5
    )
    start, stop = frame.reference_time.iloc[100], frame.reference_time.iloc[200]
    labels = causal_opportunity_labels(frame, base, start, stop, ResearchCosts(), policy)
    assert (labels.loc[labels.status == "observed", "target_time"] < stop).all()
    assert (labels.status == "censored").any()
    assert not labels.reference_time.eq(start).any()  # Previous in-sample confirmation forbidden.
    protocol = {"research": {"minimum_meta_rows": 5000, "seed": 42}}
    model = fit_meta(
        frame, labels, frame.reference_time.iloc[180], "logistic", protocol, lambda: None
    )
    assert model.model is None
    assert pd.Timestamp(model.audit["last_target"]) < frame.reference_time.iloc[180]
    assert model.audit["past_oos_forecasts_only"] is True


def test_entry_gate_does_not_change_base_exit_probabilities(raw):
    frame = build_research_features(flow_frame(raw))
    p = np.zeros(len(frame))
    p[100:105] = 0.65
    p[105] = 0.45
    start, stop = frame.reference_time.iloc[100], frame.reference_time.iloc[130]
    policy = RiskPolicy(
        name="persistent", capital_cap=0.25, persistent=True, confirmations=2, max_holding_bars=10
    )
    normal = run_overlay(frame, p, start, stop, ResearchCosts(), policy)
    gate = np.zeros(len(frame), dtype=bool)
    gate[101] = True
    filtered = run_overlay(frame, p, start, stop, ResearchCosts(), policy, entry_gate=gate)
    assert (
        filtered.simulation.trades.exit_time.iloc[0] == normal.simulation.trades.exit_time.iloc[0]
    )
    assert filtered.simulation.trades.exit_reason.iloc[0] == "probability_exit"
    assert filtered.simulation.trades.probability.iloc[0] == p[101]
    blocked = run_overlay(
        frame, p, start, stop, ResearchCosts(), policy, entry_gate=np.zeros(len(frame), dtype=bool)
    )
    assert blocked.simulation.trades.empty
    assert blocked.simulation.metrics["quality_rejections"] > 0


def test_simultaneous_bootstrap_identical_models_have_zero_difference():
    target = np.arange(100) % 2
    p = np.where(target, 0.65, 0.35)
    result = paired_predictive(
        target,
        {"a": p, "b": p.copy()},
        np.repeat([1, 2], 50),
        [("a", "b")],
        {"samples": 100, "block_lengths": [7, 30], "confidence": 0.95, "seed": 42},
    )
    for block in result["blocks"].values():
        for interval in block["a__minus__b"].values():
            assert interval["estimate"] == 0
            assert interval["lower"] == 0
            assert interval["upper"] == 0
    assert result["method"].endswith("max_abs_t")


@pytest.mark.parametrize("kind", ["logistic", "xgb"])
def test_meta_trains_on_numeric_oos_probability_after_pool_concat(raw, kind):
    frame = build_research_features(flow_frame(raw))
    indices = np.arange(100, 220, 5)
    pool = pd.DataFrame(
        {
            "source_index": indices,
            "reference_time": frame.reference_time.iloc[indices].to_numpy(),
            "target_time": frame.timestamp.iloc[indices + 5].to_numpy(),
            "target": np.arange(len(indices)) % 2,
            "unit_net_return": np.where(np.arange(len(indices)) % 2, 0.02, -0.02),
            "base_probability": pd.Series(np.full(len(indices), 0.6), dtype=object),
            "status": "observed",
        }
    )
    protocol = {
        "research": {"minimum_meta_rows": 5, "seed": 42},
        "logistic": {"C": 1.0, "max_iter": 2000},
        "xgboost_regularized": {"n_estimators": 5, "max_depth": 2},
    }
    model = fit_meta(frame, pool, frame.reference_time.iloc[300], kind, protocol, lambda: None)
    assert model.model is not None
    assert np.isfinite(model.probabilities(frame, np.full(len(frame), 0.6))).all()


@pytest.mark.parametrize("kind", ["logistic", "xgb"])
def test_recovered_model_matches_contract_and_probabilities(raw, tmp_path, kind):
    frame = build_targets(build_research_features(flow_frame(raw)), ResearchCosts())["direction_h1"]
    protocol = {
        "research": {
            "minimum_train_rows": 120,
            "rolling_days": 730,
            "calibration_days": 90,
            "seed": 42,
        },
        "xgboost_reference": {"n_estimators": 5, "max_depth": 2},
        "logistic": {"C": 1.0, "max_iter": 2000},
    }
    recipe = {"id": "cached", "kind": kind}
    cutoff = frame.reference_time.iloc[450]
    model = fit_recipe(frame, CORE, cutoff, recipe, protocol, lambda: None)
    directory = tmp_path / "cached"
    model.save(directory)
    recovered = load_fitted(directory, frame, cutoff, recipe, protocol)
    valid = frame.loc[frame.research_ready]
    np.testing.assert_allclose(
        model.probabilities(valid), recovered.probabilities(valid), rtol=1e-12, atol=1e-12
    )
    with pytest.raises(DataError):
        load_fitted(directory, frame, cutoff, {**recipe, "id": "other"}, protocol)
