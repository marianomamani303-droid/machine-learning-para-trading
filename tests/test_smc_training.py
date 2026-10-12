import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.common import read_json, sha256, sidecar, write_json
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.signals import signal_quality
from trading_ml.features import feature_columns
from trading_ml.models import load_model
from trading_ml.models.calibration import apply_calibration, fit_sigmoid
from trading_ml.prediction.smc_analysis import analyze_smc
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.training import train
from trading_ml.training.smc_study import evaluate_smc_recipe, fit_smc_model, run_smc_study
from trading_ml.training.walk_forward import WalkForwardConfig, walk_forward_folds


def small_wf(final):
    return WalkForwardConfig(
        ["2020-12-01", "2021-03-01", final.strftime("%Y-%m-%d")],
        inner_validation_days=30,
        early_stopping_rounds=5,
        min_train_rows=80,
        min_eval_rows=10,
    )


def write_protocol(path, final):
    path.write_text(
        f'''[walk_forward]
boundaries = ["2020-12-01", "2021-03-01", "{final.strftime("%Y-%m-%d")}"]
inner_validation_days = 30
early_stopping_rounds = 5
min_train_rows = 80
min_eval_rows = 10
[budget]
max_experiments = 5
patience = 2
[acceptance]
min_log_loss_gain = 10.0
[smc]
swing_left = 1
swing_right = 1
liquidity_window = 5
ob_window = 5
zone_age = 10
[signals]
threshold = 0.55
[calibration]
days = 30
[regularized]
n_estimators = 30
max_depth = 2
learning_rate = 0.025
subsample = 0.8
colsample_bytree = 0.8
reg_lambda = 20.0
min_child_weight = 20
[prospective]
window_days = 90
''',
        encoding="utf-8",
    )


def test_sigmoid_serialization_validity_and_monotonicity():
    probability = np.linspace(0.1, 0.9, 100)
    target = np.r_[np.zeros(50), np.ones(50)]
    calibration = fit_sigmoid(probability, target)
    restored = json.loads(json.dumps(calibration))
    result = apply_calibration(probability, restored)
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    assert (np.diff(result) >= 0).all()
    inverted = fit_sigmoid(probability, 1 - target)
    assert inverted["ranking_collapsed"]
    np.testing.assert_allclose(apply_calibration(probability, inverted), 0.5)
    with pytest.raises(DataError):
        fit_sigmoid(probability, np.ones(100))


def test_calibration_is_disjoint_from_model_and_outer_fold(config, raw):
    from conftest import raw_file

    config = replace(config, features=replace(config.features, smc=True))
    frame, _ = load_prepared(prepare(config, raw_file(config, raw)))
    final = pd.Timestamp("2021-05-01", tz="UTC")
    wf = small_wf(final)
    fold = walk_forward_folds(frame, wf, final, frame.reference_time.min())[0]
    model, fitting, calibration = fit_smc_model(
        fold.train,
        feature_columns(config.features),
        {**config.model, "n_estimators": 40},
        wf,
        42,
        True,
        True,
        30,
    )
    assert pd.Timestamp(calibration["fit_last_target_time"]) < pd.Timestamp(
        calibration["first_reference"]
    )
    assert pd.Timestamp(calibration["last_target_time"]) < fold.start
    assert pd.Timestamp(fitting["inner_validation_last_target"]) < pd.Timestamp(
        calibration["first_reference"]
    )
    assert model.n_estimators >= 1


def test_neutral_signals_do_not_count_as_correct():
    quality = signal_quality(np.array([1, 0, 1, 0]), np.array([0.7, 0.3, 0.5, 0.5]))
    assert quality["bullish"] == quality["bearish"] == 1
    assert quality["neutral"] == 2
    assert quality["coverage"] == 0.5
    assert quality["direction_accuracy"] == 1
    assert quality["bullish_precision"] == quality["bearish_precision"] == 1
    none = signal_quality(np.array([1, 0]), np.array([0.5, 0.5]))
    assert none["direction_accuracy"] is None


@pytest.mark.parametrize("kind", ["rules", "combined"])
def test_excluded_test_cannot_affect_smc_fold_results(config, raw, kind):
    from conftest import raw_file

    config = replace(config, features=replace(config.features, smc=True))
    path = prepare(config, raw_file(config, raw))
    frame, _ = load_prepared(path)
    final = pd.Timestamp("2021-05-01", tz="UTC")
    recipe = {
        "id": "test",
        "kind": kind,
        "parameters": config.model,
        "early_stop": False,
        "calibrated": False,
    }
    first = evaluate_smc_recipe(
        frame, config, recipe, small_wf(final), final, config.root / "a", 0.55, 30
    )
    changed = frame.copy()
    mask = changed.reference_time >= final
    changed.loc[mask, "target"] = 1 - changed.loc[mask, "target"]
    changed.loc[mask, "open"] *= 3
    changed.loc[mask, feature_columns(config.features)] *= 7
    second = evaluate_smc_recipe(
        changed, config, recipe, small_wf(final), final, config.root / "b", 0.55, 30
    )
    assert first == second


def test_smc_study_training_inference_preservation_and_no_final(config, prepared):
    old = train(config, prepared)
    final = pd.Timestamp(load_model(config.root, old).metadata["splits"]["test"]["first_reference"])
    protocol = config.root / "smc.toml"
    write_protocol(protocol, final)
    report = run_smc_study(config, protocol, old)
    assert report["executed_experiments"] == 4
    assert report["stop_reason"] == "no_consistent_smc_improvement"
    assert not report["old_test_evaluated"]
    assert report["prospective"]["status"] == "pending_future_data"
    assert not report["promoted"]
    assert load_model(config.root).metadata["version"] == old
    model = load_model(config.root, report["candidate_version"])
    assert model.metadata["test_metrics"]["status"] == "pending_future_data"
    prediction = analyze_smc(
        model, Path(report["candidate_dataset"]), now=pd.Timestamp("2021-09-01", tz="UTC")
    )
    assert prediction["signal_state"] in ["bullish", "bearish", "neutral"]
    assert prediction["probability_up"] + prediction["probability_down"] == pytest.approx(1)
    assert isinstance(prediction["patterns"], list)
    assert prediction["smc_rules"]["calibration_status"] == "heuristic_uncalibrated"
    with pytest.raises(LeakageError):
        analyze_smc(model, Path(report["candidate_dataset"]), reference="2020-12-01")
    with pytest.raises(DataError):
        analyze_smc(load_model(config.root, old), prepared)
    same = run_smc_study(config, protocol, old)
    assert same["run_id"] == report["run_id"]
    snapshot = read_json(Path(report["directory"]) / "plan.json")
    assert snapshot["automatic_promotion"] is False
    assert (
        pd.Timestamp(snapshot["prospective_reference_start"])
        > pd.Timestamp(snapshot["created_at"]).normalize()
    )
    from trading_ml.evaluation.smc_prospective import evaluate_smc_prospective

    with pytest.raises(DataError, match="pendiente"):
        evaluate_smc_prospective(config.root, config.root / "data" / "raw.csv")
    assert not (Path(report["directory"]) / "prospective_evaluation" / "summary.json").exists()
    # Emulación temporal sintética aislada: no son métricas experimentales reales.
    original_raw = config.root / "data" / "raw.csv"
    original_frame = pd.read_csv(original_raw)
    future_end = pd.Timestamp(report["prospective"]["end_reference_exclusive"])
    dates = pd.date_range(
        pd.to_datetime(original_frame.timestamp.iloc[-1], utc=True) + config.delta,
        future_end,
        freq=config.delta,
    )
    price = float(original_frame.close.iloc[-1])
    tail = pd.DataFrame(
        {
            "timestamp": dates,
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 100.0,
        }
    )
    future_raw = config.root / "synthetic_future.csv"
    future_raw.write_bytes(
        original_raw.read_bytes() + tail.to_csv(index=False, header=False).encode("utf-8")
    )
    manifest = read_json(sidecar(original_raw))
    manifest["market"]["end"] = (future_end + config.delta).strftime("%Y-%m-%d")
    manifest["sha256"] = sha256(future_raw)
    write_json(sidecar(future_raw), manifest)
    evaluation = evaluate_smc_prospective(
        config.root, future_raw, now=future_end + pd.Timedelta(days=2)
    )
    assert evaluation["status"] == "complete_frozen_prospective_window"
    assert not evaluation["automatic_promotion"]
    assert set(evaluation["comparison"]) == {"original", "smc_rules", "combined"}
    assert all(item["metrics"]["rows"] == 90 for item in evaluation["comparison"].values())
    assert load_model(config.root).metadata["version"] == old
    protocol.write_text(protocol.read_text().replace("max_experiments = 5", "max_experiments = 6"))
    with pytest.raises(DataError, match="3 a 5"):
        run_smc_study(config, protocol, old)


def test_invalid_temporal_protocol_rejected_before_experiments(config, prepared):
    version = train(config, prepared)
    final = pd.Timestamp(
        load_model(config.root, version).metadata["splits"]["test"]["first_reference"]
    )
    protocol = config.root / "invalid_smc.toml"
    write_protocol(protocol, final)
    original = protocol.read_text()
    protocol.write_text(original.replace("window_days = 90", "window_days = 0"))
    with pytest.raises(DataError, match="prospectiva"):
        run_smc_study(config, protocol, version)
    protocol.write_text(original.replace("days = 30\n[regularized]", "days = 0\n[regularized]"))
    with pytest.raises(DataError, match="calibración"):
        run_smc_study(config, protocol, version)
    assert not (config.root / "smc_studies").exists()
