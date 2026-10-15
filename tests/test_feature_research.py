from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import candles
from sklearn.linear_model import LogisticRegression

from trading_ml.advanced_research.features import ORIGINAL
from trading_ml.errors import DataError
from trading_ml.feature_research.diagnostics import (
    block_permutation,
    permutation_diagnostic,
    prune_correlated,
)
from trading_ml.feature_research.features import FAMILIES, build_feature_panel
from trading_ml.feature_research.study import (
    ALL,
    FitBudget,
    _inner_selection,
    choose_inner,
    fixed_recipes,
    temporal_score,
    temporal_train,
    validate_protocol,
)


def raw_panel(rows=850):
    output = {}
    rng = np.random.default_rng(71)
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        frame = candles(rows)
        # Distinct peer path and actual-looking microstructure are synthetic test
        # fixtures only. Experiments must recover these fields from verified ZIPs.
        multiplier = np.exp(np.cumsum(rng.normal(0, 0.003, rows)))
        frame[["open", "high", "low", "close"]] *= multiplier[:, None]
        frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
        frame["quote_volume"] = frame.volume * frame.close
        frame["trades"] = rng.integers(50, 300, rows)
        share = rng.uniform(0.3, 0.7, rows)
        frame["taker_quote"] = frame.quote_volume * share
        frame["taker_base"] = frame.volume * share
        output[symbol] = frame
    return output


@pytest.fixture(scope="module")
def panel():
    raw = raw_panel()
    return raw, build_feature_panel(raw)


def test_features_prefix_and_future_mutation_invariance(panel):
    raw, built = panel
    stop = 403  # Incomplete calendar week: resampling must not reveal its end.
    prefix = build_feature_panel({s: f.iloc[:stop] for s, f in raw.items()})
    changed = {s: f.copy() for s, f in raw.items()}
    for frame in changed.values():
        frame.loc[stop:, ["open", "high", "low", "close", "quote_volume", "taker_quote"]] *= 3
        frame.loc[stop:, "trades"] *= 2
    second = build_feature_panel(changed)
    for symbol, f in built.items():
        columns = ["reference_time", "weekly_available_time", *ALL]
        pd.testing.assert_frame_equal(f.loc[: stop - 1, columns], prefix[symbol][columns])
        pd.testing.assert_frame_equal(
            f.loc[: stop - 1, columns], second[symbol].loc[: stop - 1, columns]
        )


def test_weekly_features_only_publish_completed_utc_weeks(panel):
    _, built = panel
    for f in built.values():
        available = f.weekly_available_time.dropna()
        assert (available.dt.weekday == 0).all()
        assert (available <= f.loc[available.index, "reference_time"]).all()
        assert (f.loc[available.index, "reference_time"] - available <= pd.Timedelta(days=6)).all()
        # First partial week begins Wednesday Jan 1; publish first full week Jan 13.
        assert available.min() == pd.Timestamp("2020-01-13", tz="UTC")
        for _, week in f.loc[f.fe_week_momentum_4.notna()].groupby("weekly_available_time"):
            assert week.fe_week_momentum_4.nunique() == 1
            assert week.fe_week_trend_8.nunique() == 1


def test_missing_peer_is_not_forward_filled(panel):
    raw, _ = panel
    shortened = raw | {"ETHUSDT": raw["ETHUSDT"].iloc[:-12]}
    btc = build_feature_panel(shortened)["BTCUSDT"]
    assert btc.fe_peer_return_1.iloc[-12:].isna().all()
    assert btc.fe_residual_return_1.iloc[-12:].isna().all()
    assert btc.return_1.iloc[-12:].notna().all()


def test_flow_never_invented_and_alignment_rejected(panel):
    raw, _ = panel
    with pytest.raises(DataError, match="no se imputan"):
        build_feature_panel(raw | {"ETHUSDT": raw["ETHUSDT"].drop(columns="taker_quote")})
    with pytest.raises(DataError, match="contiguas"):
        build_feature_panel(raw | {"ETHUSDT": raw["ETHUSDT"].drop(index=400)})
    shifted = raw["ETHUSDT"].copy()
    shifted.reference_time += pd.Timedelta(hours=1)
    with pytest.raises(DataError, match="cierre diario"):
        build_feature_panel(raw | {"ETHUSDT": shifted})


def test_label_tail_and_feature_allowlists(panel):
    _, built = panel
    columns = set(c for recipe in fixed_recipes().values() for c in recipe)
    assert not columns & {"target", "target_time", "future_return", "reference_time", "close_peer"}
    assert len(ALL) == len(set(ALL))
    assert set(ORIGINAL).isdisjoint(c for group in FAMILIES.values() for c in group)
    for f in built.values():
        assert pd.isna(f.target.iloc[-1]) and pd.isna(f.target_time.iloc[-1])
        assert f[ALL].iloc[-1].notna().all()  # Still usable for prediction.
        np.testing.assert_array_equal(f.target.iloc[:-1], (f.close.shift(-1) > f.close).iloc[:-1])


def test_temporal_purge_uses_actual_label_end(panel):
    _, built = panel
    f = built["BTCUSDT"]
    start, end = f.reference_time.iloc[500], f.reference_time.iloc[650]
    train = temporal_train(f, start)
    assert (train.target_time < start - pd.Timedelta(days=6)).all()
    delayed = f.copy()
    delayed.loc[100, "target_time"] = start
    assert 100 not in temporal_train(delayed, start).index
    score = temporal_score(f, start, end)
    assert (score.reference_time >= start).all() and (score.target_time < end).all()
    assert 649 not in score.index  # Target ending exactly at next fold is purged.


def test_redundancy_filter_fits_only_supplied_train():
    x = np.arange(200, dtype=float)
    train = pd.DataFrame({"a": x, "copy": -x, "constant": 1, "other": np.sin(x)})
    kept, removed = prune_correlated(train, list(train))
    assert kept == ["a", "other"]
    assert {r["feature"] for r in removed} == {"copy", "constant"}
    preferred, _ = prune_correlated(train, list(train), priority={"copy": 1})
    assert "copy" in preferred and "a" not in preferred


def test_block_permutation_preserves_blocks_and_all_observations():
    order = block_permutation(95, 7, np.random.default_rng(42))
    np.testing.assert_array_equal(np.sort(order), np.arange(95))
    for start in range(0, 95, 7):
        indices = np.arange(start, min(start + 7, 95))
        positions = [int(np.flatnonzero(order == i)[0]) for i in indices]
        assert np.all(np.diff(positions) == 1)
    with pytest.raises(DataError):
        block_permutation(12, 7, np.random.default_rng(42))


def test_group_permutation_reports_predictive_information_oos():
    rng = np.random.default_rng(5)
    train_x, test_x = rng.normal(size=600), rng.normal(size=300)
    model = LogisticRegression(C=5).fit(np.column_stack([train_x, train_x]), train_x > 0)
    frame = pd.DataFrame({"a": test_x, "b": test_x, "target": (test_x > 0).astype(int)})
    rows = permutation_diagnostic(
        model, frame, ["a", "b"], {"a": ["a"], "both": ["a", "b"]}, blocks=[7], repeats=5
    )
    values = {r["name"]: r["delta_loss_mean"] for r in rows}
    assert values["both"] > values["a"] > 0
    assert rows[1]["members"] == ["a", "b"]


def test_inner_selection_requires_repeated_improvement(panel):
    _, built = panel
    r = {"inner_minimum_gain": 0.001, "correlation_threshold": 0.9, "compact_max_features": 20}
    scores = {"original": [0.7, 0.7], "add_flow": [0.65, 0.71], "add_momentum": [0.69, 0.69]}
    importance = dict.fromkeys(ALL, [-0.01, 0.01, 0.01, 0.01])
    choice = choose_inner(scores, importance, built["BTCUSDT"].iloc[200:500], r)
    assert choice["chosen"] == "add_momentum"
    assert choice["compact_fallback_original"]
    assert choice["compact_columns"] == ORIGINAL


def test_inner_selection_unaffected_by_outer_future_labels(panel):
    _, built = panel
    frame = built["BTCUSDT"].loc[built["BTCUSDT"][ALL].notna().all(axis=1)].copy()
    start = pd.Timestamp("2022-01-01", tz="UTC")
    r = {
        "seed": 42,
        "embargo_days": 6,
        "inner_days": 90,
        "minimum_train_rows": 120,
        "inner_minimum_gain": 0.001,
        "correlation_threshold": 0.9,
        "compact_max_features": 20,
    }
    imp = {"block_lengths": [7], "inner_repeats": 2}
    parameters = {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.1}
    train = temporal_train(frame, start, minimum=120)
    before = _inner_selection(frame, start, train, FitBudget(20, parameters), r, imp)
    changed = frame.copy()
    changed.loc[changed.reference_time >= start, "target"] = 1 - changed.target
    # Outer features may change too: neither is an input to selection.
    changed.loc[changed.reference_time >= start, ALL] *= 10
    after = _inner_selection(changed, start, train, FitBudget(20, parameters), r, imp)
    assert before == after


def test_fixed_protocol_budget_and_no_unbounded_search():
    protocol = tomllib.loads(Path("config/feature_study.toml").read_text(encoding="utf-8"))
    dates, planned = validate_protocol(protocol)
    assert len(dates) == 7 and planned == 648
    protocol["research"]["maximum_fits"] = 500
    with pytest.raises(DataError, match="Presupuesto"):
        validate_protocol(protocol)


def test_feature_study_rejects_reserved_window_before_loading_market_data(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import trading_ml.feature_research.study as study
    from trading_ml.common import write_new_json
    from trading_ml.config import Settings
    from trading_ml.errors import LeakageError

    settings = replace(Settings(), root=tmp_path)
    write_new_json(
        tmp_path / "frontier_studies" / "future" / "prospective.json",
        {"start": "2025-01-01", "end": "2027-01-01"},
    )
    monkeypatch.setattr(
        study,
        "load_model",
        lambda *args: SimpleNamespace(
            metadata={"features": ORIGINAL, "selected_parameters": {"n_estimators": 3}}
        ),
    )

    def forbidden(*args):
        pytest.fail("Se intentó leer datos reservados")

    monkeypatch.setattr(study, "_load_panel", forbidden)
    with pytest.raises(LeakageError, match="prospectivo"):
        study.run_feature_study(settings, Path("config/feature_study.toml"))


def test_positive_point_estimate_does_not_override_uncertainty_or_stability():
    from trading_ml.feature_research.report import _gate

    metrics = {
        "original": {"log_loss": 0.70, "roc_auc": 0.50},
        "compact_inner": {"log_loss": 0.69, "roc_auc": 0.52},
    }
    stats = {
        "metrics": metrics,
        "blocks": {
            "7": {
                "compact_inner__minus__original": {
                    "log_loss": {"lower": -0.02, "upper": 0.01},
                    "roc_auc": {"lower": -0.01, "upper": 0.03},
                }
            }
        },
    }
    fits = [
        {"symbol": "BTCUSDT", "seed": 42, "fold": fold, "recipe": name, "oos": values}
        for fold in range(3)
        for name, values in metrics.items()
    ]
    protocol = {
        "acceptance": {
            "minimum_log_loss_gain": 0.003,
            "minimum_auc_gain": 0.005,
            "required_fold_fraction": 2 / 3,
        }
    }
    result = _gate("BTCUSDT", 42, "compact_inner", stats, fits, protocol, baseline_checks=False)
    assert result["loss_gain"] > 0 and result["auc_gain"] > 0
    assert not result["passed"]
    assert "simultaneous_loss_block_7" in result["failed"]


def test_statistical_comparison_rejects_different_oos_rows():
    from trading_ml.feature_research.report import _paired

    dates = pd.date_range("2022-01-01", periods=10, tz="UTC")
    original = pd.DataFrame(
        {
            "fold": 0,
            "reference_time": dates,
            "target_time": dates + pd.Timedelta(days=1),
            "target": np.arange(10) % 2,
            "recipe": "original",
            "probability": 0.5,
        }
    )
    candidate = original.iloc[:-1].assign(recipe="compact_inner")
    with pytest.raises(ValueError, match="exactamente"):
        _paired(pd.concat([original, candidate]), {}, False)


@pytest.mark.parametrize("relative", ["summary.json", "../escaped.json"])
def test_verifier_rejects_corruption_and_paths_before_loading_models(tmp_path, relative):
    from trading_ml.common import write_new_json
    from trading_ml.config import Settings
    from trading_ml.feature_research.verify import verify_feature_study

    write_new_json(tmp_path / "artifacts.json", {relative: "invalid-digest"})
    write_new_json(tmp_path / "summary.json", {"run_id": "changed"})
    with pytest.raises(DataError, match="alterado"):
        verify_feature_study(Settings(), tmp_path)
