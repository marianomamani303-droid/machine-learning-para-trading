from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_ml.config import Settings
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation import evaluate
from trading_ml.evaluation.selection import Acceptance, assess_consistency
from trading_ml.features import build_features, feature_columns
from trading_ml.models import load_model
from trading_ml.models.lifecycle import promote
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.training import train
from trading_ml.training.optimization import Experiment, evaluate_walk_forward, optimize
from trading_ml.training.walk_forward import (
    WalkForwardConfig,
    fit_temporal_model,
    walk_forward_folds,
)


def config_for_test(final):
    return WalkForwardConfig(
        ["2020-12-01", "2021-03-01", final.strftime("%Y-%m-%d")],
        inner_validation_days=30,
        early_stopping_rounds=5,
        min_train_rows=80,
        min_eval_rows=10,
    )


@pytest.mark.parametrize("horizon", [1, 3, 10])
def test_walk_forward_purge_and_final_exclusion(config, raw, horizon):
    from conftest import raw_file

    config = replace(config, training=replace(config.training, horizon=horizon))
    dataset = prepare(config, raw_file(config, raw))
    frame, _ = load_prepared(dataset)
    final = pd.Timestamp("2021-05-01", tz="UTC")
    folds = walk_forward_folds(
        frame, config_for_test(final), final, pd.Timestamp("2020-01-01", tz="UTC")
    )
    assert len(folds) == 2
    for fold in folds:
        assert fold.train.target_time.max() < fold.evaluation.reference_time.min()
        assert fold.evaluation.target_time.max() < fold.end <= final
    assert set(folds[0].evaluation.index).isdisjoint(folds[1].evaluation.index)
    with pytest.raises(LeakageError):
        walk_forward_folds(
            frame, config_for_test(final + pd.Timedelta(days=1)), final, frame.reference_time.min()
        )


def test_rolling_window_and_inner_early_stopping(config, prepared):
    frame, metadata = load_prepared(prepared)
    final = pd.Timestamp("2021-06-01", tz="UTC")
    wf = config_for_test(final)
    folds = walk_forward_folds(frame, wf, final, frame.reference_time.min(), rolling_days=250)
    for fold in folds:
        assert fold.train.reference_time.min() >= fold.start - pd.Timedelta(days=250)
        model, details = fit_temporal_model(
            fold.train, metadata["features"], {**config.model, "n_estimators": 40}, 42, wf, True
        )
        assert pd.Timestamp(details["inner_fit_last_target"]) < pd.Timestamp(
            details["inner_validation_first_reference"]
        )
        assert pd.Timestamp(details["inner_validation_last_target"]) < fold.start
        assert 1 <= details["effective_parameters"]["n_estimators"] <= 40
        assert model.n_estimators == details["effective_parameters"]["n_estimators"]


def test_enhanced_features_causal_and_legacy_contract(raw):
    enhanced = replace(Settings().features, enhanced=True)
    assert len(feature_columns(enhanced)) == 22
    original = build_features(raw, enhanced)
    changed = raw.copy()
    changed.loc[301:, ["open", "high", "low", "close", "volume"]] *= 3
    modified = build_features(changed, enhanced)
    pd.testing.assert_frame_equal(
        original.loc[:300, feature_columns(enhanced)], modified.loc[:300, feature_columns(enhanced)]
    )
    assert "enhanced" not in Settings().contract()["features"]
    assert Settings().contract()["feature_schema"] == 1
    assert replace(Settings(), features=enhanced).contract()["feature_schema"] == 2


def test_final_prices_labels_and_features_cannot_affect_walk_forward(config, prepared):
    frame, _ = load_prepared(prepared)
    final = pd.Timestamp("2021-05-01", tz="UTC")
    wf = config_for_test(final)
    recipe = Experiment("E1", "test", False, False, True, {**config.model, "n_estimators": 40})
    a = evaluate_walk_forward(frame, config, recipe, wf, final, config.root / "a")
    changed = frame.copy()
    mask = changed.reference_time >= final
    changed.loc[mask, "target"] = 1 - changed.loc[mask, "target"]
    changed.loc[mask, feature_columns(config.features)] *= 5
    changed.loc[mask, "open"] *= 2
    b = evaluate_walk_forward(changed, config, recipe, wf, final, config.root / "b")
    assert a == b


def fold_report(loss, brier, auc, net, rows=100):
    return {
        "first_reference": "2020-01-01",
        "last_reference": "2020-06-01",
        "metrics": {"log_loss": loss, "brier_score": brier, "roc_auc": auc, "rows": rows},
        "backtest": {"strategy_metrics": {"cumulative_return": net}},
    }


def test_acceptance_requires_consistency_not_one_lucky_fold():
    reference = {
        "folds": [fold_report(0.7, 0.26, 0.49, -0.1) for _ in range(5)],
        "summary": {"metrics": {"log_loss": 0.7}, "mean_fold_auc": 0.49, "compounded_return": -0.3},
    }
    candidate = {
        "folds": [fold_report(0.69, 0.25, 0.51, 0.01) for _ in range(5)],
        "summary": {
            "metrics": {"log_loss": 0.69},
            "mean_fold_auc": 0.51,
            "compounded_return": 0.1,
            "trade_count": 100,
            "constant_metrics": {"log_loss": 0.693},
        },
    }
    assert assess_consistency(reference, candidate, Acceptance())["accepted"]
    candidate["folds"][0:4] = [fold_report(0.71, 0.27, 0.48, -0.2) for _ in range(4)]
    assert not assess_consistency(reference, candidate, Acceptance())["accepted"]
    candidate["folds"][0]["metrics"]["rows"] += 1
    with pytest.raises(DataError):
        assess_consistency(reference, candidate, Acceptance())


def test_optimization_budget_stop_and_champion_retained(config, prepared, accepted_training):
    version = train(config, prepared)
    evaluation_path = config.root / "reference_evaluation.json"
    evaluate(load_model(config.root, version), prepared, evaluation_path)
    promote(config.root, version, evaluation_path, "Synthetic reference")
    final = pd.Timestamp(
        load_model(config.root, version).metadata["splits"]["test"]["first_reference"]
    )
    protocol = config.root / "protocol.toml"
    protocol.write_text(
        f'''[walk_forward]
boundaries = ["2020-12-01", "2021-03-01", "{final.strftime("%Y-%m-%d")}"]
inner_validation_days = 30
early_stopping_rounds = 5
min_train_rows = 80
min_eval_rows = 10
[budget]
max_experiments = 5
patience = 1
[acceptance]
min_log_loss_gain = 10.0
[data]
extended_start = "2017-08-17"
[experiments]
max_trees = 40
learning_rate = 0.025
max_depth = 2
reg_lambda = 20.0
min_child_weight = 20
subsample = 0.8
colsample_bytree = 0.8
rolling_train_days = 730
''',
        encoding="utf-8",
    )
    result = optimize(config, protocol, version)
    assert result["executed_experiments"] == 2
    assert result["stop_reason"] == "no_consistent_improvement"
    assert not result["promoted"]
    assert result["champion_version"] == version
    assert load_model(config.root).metadata["version"] == version
    candidate = load_model(config.root, result["candidate_version"])
    assert candidate.metadata["parent_version"] == version
    assert (
        candidate.metadata["splits"]["train"]["last_target_time"]
        < candidate.metadata["splits"]["test"]["first_reference"]
    )
    assert np.isfinite(candidate.probabilities(load_prepared(prepared)[0])).all()
    replay = optimize(config, protocol, version)
    assert replay["run_id"] == result["run_id"]
    assert replay["executed_experiments"] == result["executed_experiments"]
    protocol.write_text(protocol.read_text().replace("max_experiments = 5", "max_experiments = 6"))
    with pytest.raises(DataError, match="Presupuesto"):
        optimize(config, protocol, version)
