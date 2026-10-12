import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.backtesting import backtest
from trading_ml.cli import main
from trading_ml.common import sha256
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation import classification_metrics, evaluate
from trading_ml.evaluation.comparison import compare
from trading_ml.models import load_model
from trading_ml.prediction import predict
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.training import retrain, train
from trading_ml.training.split import chronological_split


@pytest.mark.parametrize("horizon", [1, 3, 10])
def test_split_purges_label_overlap(config, raw, horizon):
    from conftest import raw_file

    config = replace(config, training=replace(config.training, horizon=horizon))
    path = prepare(config, raw_file(config, raw))
    frame, _ = load_prepared(path)
    split = chronological_split(frame, config.training)
    assert split.train.target_time.max() < split.validation.reference_time.min()
    assert split.validation.target_time.max() < split.test.reference_time.min()
    assert split.purged_rows == 2 * horizon
    assert set(split.train.index).isdisjoint(split.test.index)


def test_metrics_single_class_and_calibration():
    metrics = classification_metrics(np.ones(4), np.array([1.0, 0.8, 0.9, 0.7]))
    assert metrics["roc_auc"] is None
    assert metrics["brier_score"] == pytest.approx(0.035)
    assert sum(item["count"] for item in metrics["reliability_bins"]) == 4
    with pytest.raises(DataError):
        classification_metrics(np.array([0, 1]), np.array([0.1, np.nan]))


def test_full_pipeline_persistence_and_reproducibility(config, prepared):
    first = train(config, prepared)
    second = train(config, prepared)
    assert first != second
    a = load_model(config.root, first)
    b = load_model(config.root, second)
    frame, _ = load_prepared(prepared)
    np.testing.assert_array_equal(a.probabilities(frame), b.probabilities(frame))
    assert sha256(a.directory / "model.ubj") == sha256(b.directory / "model.ubj")
    report = evaluate(a, prepared)
    assert report["metrics"] == a.metadata["test_metrics"]
    prediction = predict(a, prepared, now=pd.Timestamp("2021-08-24", tz="UTC"))
    assert prediction["probability_up"] + prediction["probability_down"] == pytest.approx(1)
    assert prediction["asset"] == "BTCUSDT"
    assert prediction["horizon_bars"] == 1
    historical_reference = frame.reference_time.iloc[10].strftime("%Y-%m-%d")
    with pytest.raises(LeakageError):
        predict(a, prepared, historical_reference)
    result = backtest(a, prepared, output=config.root / "backtest")
    assert result.summary["trade_count"] >= 0
    assert result.equity.equity.iloc[0] == 10000
    comparison = compare(config.root, prepared, [first, second])
    assert comparison["models"][0]["metrics"] == comparison["models"][1]["metrics"]
    assert load_model(config.root).metadata["version"] == second
    with pytest.raises(DataError, match="más recientes"):
        retrain(config.root, prepared)
    with pytest.raises(DataError, match="Versión"):
        load_model(config.root, "../escape")


def test_retraining_requires_and_uses_new_data(config, prepared):
    from conftest import candles, raw_file

    old = train(config, prepared)
    newer_config = replace(config, market=replace(config.market, end="2021-12-01"))
    newer = prepare(newer_config, raw_file(newer_config, candles(700)))
    new = retrain(config.root, newer, old)
    bundle = load_model(config.root, new)
    assert bundle.metadata["parent_version"] == old
    assert bundle.metadata["config"]["market"]["end"] == newer_config.market.end
    assert pd.Timestamp(bundle.metadata["splits"]["test"]["last_target_time"]) > pd.Timestamp(
        load_model(config.root, old).metadata["splits"]["test"]["last_target_time"]
    )
    report = compare(config.root, newer, [old, new])
    assert report["rows"] > 0


def test_artifact_corruption_rejected(config, prepared):
    version = train(config, prepared)
    bundle = load_model(config.root, version)
    model = bundle.directory / "model.ubj"
    model.write_bytes(model.read_bytes() + b"corrupt")
    with pytest.raises(DataError, match="artefacto"):
        load_model(config.root, version)


def test_no_out_of_sample_data_rejected(config, prepared):
    from trading_ml.evaluation import out_of_sample

    version = train(config, prepared)
    bundle = load_model(config.root, version)
    frame, _ = load_prepared(prepared)
    with pytest.raises(LeakageError):
        out_of_sample(bundle, frame.iloc[:20])


def test_cli_error_exit(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.toml"), "train"]) == 1
    assert capsys.readouterr().out == ""


def test_changing_test_prices_cannot_change_selected_model(config, prepared, raw):
    from conftest import raw_file

    first = train(config, prepared)
    original = load_model(config.root, first)
    cutoff = pd.Timestamp(original.metadata["splits"]["test"]["first_reference"])
    changed = raw.copy()
    after_cutoff = changed.timestamp + config.delta >= cutoff
    changed.loc[after_cutoff, ["open", "high", "low", "close"]] *= 1.5
    modified = prepare(config, raw_file(config, changed), config.root / "modified.csv")
    second = train(config, modified)
    new = load_model(config.root, second)
    assert new.metadata["selected_parameters"] == original.metadata["selected_parameters"]
    assert new.metadata["trials"] == original.metadata["trials"]
    assert new.metadata["model_sha256"] == original.metadata["model_sha256"]


def test_cli_success_workflow(config, prepared, capsys):
    text = Path("config/default.toml").read_text(encoding="utf-8")
    text = text.replace('root = "artifacts"', f'root = "{config.root.as_posix()}"')
    text = text.replace('end = "2026-10-01"', f'end = "{config.market.end}"')
    text = text.replace("n_estimators = 160", "n_estimators = 20")
    path = config.root / "cli.toml"
    path.write_text(text, encoding="utf-8")
    prefix = ["--config", str(path)]
    assert main([*prefix, "train", "--data", str(prepared)]) == 0
    first = json.loads(capsys.readouterr().out)["model_version"]
    assert main([*prefix, "train", "--data", str(prepared)]) == 0
    second = json.loads(capsys.readouterr().out)["model_version"]
    for command in ["evaluate", "predict", "backtest"]:
        assert main([*prefix, command, "--version", first, "--data", str(prepared)]) == 0
        assert isinstance(json.loads(capsys.readouterr().out), dict)
    assert main([*prefix, "compare", "--versions", first, second, "--data", str(prepared)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] > 0
