import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_ml.backtesting import backtest
from trading_ml.cli import main
from trading_ml.common import read_json, sha256, write_json
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation import evaluate
from trading_ml.models import load_model
from trading_ml.models.lifecycle import promote
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.smc import directional_states, rule_probabilities
from trading_ml.training import retrain, train
from trading_ml.training.recipe import resolve_recipe, temporal_recipe
from trading_ml.training.walk_forward import WalkForwardConfig
from trading_ml.verification import run_checks


def expanded_dataset(config, raw, rows=700):
    from conftest import candles, raw_file

    expanded = candles(rows)
    expanded.iloc[: len(raw)] = raw.to_numpy()
    newer_config = replace(
        config,
        market=replace(
            config.market, end=(expanded.timestamp.iloc[-1] + config.delta).strftime("%Y-%m-%d")
        ),
    )
    return prepare(newer_config, raw_file(newer_config, expanded), config.root / "expanded.csv")


def test_training_never_activates_or_evaluates_test(config, prepared, monkeypatch):
    # All metric calls must originate in validation, not the test partition.
    import trading_ml.training.pipeline as pipeline

    original = pipeline.classification_metrics
    frame, _ = load_prepared(prepared)
    from trading_ml.training.split import chronological_split

    split = chronological_split(frame, config.training)
    lengths = []

    def metrics(y, p):
        lengths.append(len(y))
        return original(y, p)

    monkeypatch.setattr(pipeline, "classification_metrics", metrics)
    version = train(config, prepared)
    assert lengths and set(lengths) == {len(split.validation)}
    assert not (config.root / "models/latest.json").exists()
    assert not (config.root / "models/active.json").exists()
    bundle = load_model(config.root, version)
    assert bundle.metadata["test_metrics"]["status"] == "not_evaluated"
    assert not (bundle.directory / "test_predictions.csv").exists()
    with pytest.raises(DataError, match="activo"):
        load_model(config.root)


def test_explicit_promotion_keeps_candidate_and_historical_pointers(
    config, prepared, accepted_training
):
    old = train(config, prepared)
    write_json(config.root / "models/latest.json", {"version": old})
    write_json(config.root / "models/champion.json", {"version": old})
    before = {p: sha256(p) for p in (config.root / "models").rglob("*") if p.is_file()}
    new = train(config, prepared)
    assert load_model(config.root).metadata["version"] == old
    bundle = load_model(config.root, new)
    metadata_digest = sha256(bundle.directory / "metadata.json")
    report = config.root / "candidate_evaluation.json"
    evaluate(bundle, prepared, report)
    result = promote(config.root, new, report, "Explicit synthetic decision")
    assert result["previous_version"] == old
    for alias in ["latest", "active", "champion"]:
        assert load_model(config.root, alias).metadata["version"] == new
    assert all(sha256(p) == digest for p, digest in before.items())
    assert sha256(bundle.directory / "metadata.json") == metadata_digest


@pytest.mark.parametrize(
    "problem",
    ["modified_report", "wrong_version", "missing_receipt", "modified_metadata", "empty_reason"],
)
def test_promotion_rejects_invalid_evidence(config, prepared, problem, accepted_training):
    version = train(config, prepared)
    bundle = load_model(config.root, version)
    output = config.root / "evaluation.json"
    report = evaluate(bundle, prepared, output)
    reason = "Synthetic decision"
    if problem == "modified_report":
        report["metrics"]["log_loss"] = 0.01
        write_json(output, report)
    elif problem == "wrong_version":
        version = train(config, prepared)
    elif problem == "missing_receipt":
        (config.root / "evaluations" / (report["evaluation_id"] + ".json")).unlink()
    elif problem == "modified_metadata":
        metadata = read_json(bundle.directory / "metadata.json")
        metadata["config"]["backtest"]["commission_bps"] = 0
        write_json(bundle.directory / "metadata.json", metadata)
    else:
        reason = " "
    with pytest.raises(DataError):
        promote(config.root, version, output, reason)
    assert not (config.root / "models/active.json").exists()


def test_evaluation_is_append_only(config, prepared):
    bundle = load_model(config.root, train(config, prepared))
    path = config.root / "evaluation.json"
    evaluate(bundle, prepared, path)
    digest = sha256(path)
    with pytest.raises(FileExistsError):
        evaluate(bundle, prepared, path)
    assert sha256(path) == digest


def test_model_owns_snapshot_when_input_is_overwritten(config, prepared, raw):
    old = train(config, prepared)
    bundle = load_model(config.root, old)
    snapshot = Path(bundle.metadata["dataset_path"])
    digest = sha256(snapshot)
    assert snapshot != prepared.resolve()
    expanded = expanded_dataset(config, raw)
    prepared.write_bytes(expanded.read_bytes())
    from trading_ml.common import sidecar

    sidecar(prepared).write_bytes(sidecar(expanded).read_bytes())
    new = retrain(config.root, prepared, old)
    assert sha256(snapshot) == digest
    assert load_model(config.root, new).metadata["parent_version"] == old


@pytest.mark.parametrize("calibrated", [False, True])
def test_retrain_preserves_temporal_recipe_and_disjoint_partitions(config, raw, calibrated):
    from conftest import raw_file

    config = replace(
        config, features=replace(config.features, smc=calibrated), search={"enabled": False}
    )
    path = prepare(config, raw_file(config, raw))
    wf = WalkForwardConfig(
        ["2020-12-01", "2021-03-01", "2021-05-01"],
        inner_validation_days=30,
        early_stopping_rounds=5,
        min_train_rows=80,
        min_eval_rows=10,
    )
    recipe = temporal_recipe(config, {**config.model, "n_estimators": 40}, wf, True, calibrated, 30)
    old = train(config, path, recipe=recipe)
    parent = load_model(config.root, old)
    newer = expanded_dataset(config, raw)
    new = retrain(config.root, newer, old)
    bundle = load_model(config.root, new)
    assert bundle.metadata["training_recipe"] == recipe
    assert bundle.metadata["training_recipe"]["parameters"]["n_estimators"] == 40
    assert bundle.metadata["selected_parameters"]["n_estimators"] <= 40
    assert bundle.metadata["fitting"]["early_stopping"]
    assert bundle.metadata["parent_version"] == old
    assert pd.Timestamp(bundle.metadata["selection_label_end"]) > pd.Timestamp(
        parent.metadata["selection_label_end"]
    )
    assert "last_target_time" in bundle.metadata["splits"]["test"]
    assert bundle.metadata["training_data_sha256"] == bundle.metadata["fitting"]["fit_data_sha256"]
    assert bundle.metadata["code_provenance"]["source_sha256"]
    if calibrated:
        calibration = bundle.metadata["calibration"]
        assert pd.Timestamp(calibration["fit_last_target_time"]) < pd.Timestamp(
            calibration["first_reference"]
        )
        assert pd.Timestamp(
            bundle.metadata["fitting"]["inner_validation_last_target"]
        ) < pd.Timestamp(calibration["first_reference"])
        assert (
            calibration["calibration_data_sha256"]
            == bundle.metadata["partition_sha256"]["calibration"]
        )
    else:
        assert bundle.metadata["calibration"] is None


def test_retrain_rejects_revised_parent_history(config, prepared, raw):
    old = train(config, prepared)
    newer = expanded_dataset(config, raw)
    from trading_ml.common import sidecar

    revised = pd.read_csv(newer)
    revised.loc[20, "volume"] *= 2
    revised.to_csv(newer, index=False)
    manifest = read_json(sidecar(newer))
    manifest["sha256"] = sha256(newer)
    write_json(sidecar(newer), manifest)
    with pytest.raises(DataError, match="histórico"):
        retrain(config.root, newer, old)


def test_legacy_short_candidate_recovers_recipe_without_rewriting(config, prepared, raw):
    old = train(config, prepared)
    bundle = load_model(config.root, old)
    metadata = deepcopy(bundle.metadata)
    metadata.pop("training_recipe")
    metadata.pop("recipe_sha256")
    metadata.pop("dataset_reference_end")
    metadata["fitting"] = {
        "early_stopping": True,
        "effective_parameters": {**config.model, "n_estimators": 1},
    }
    metadata["selected_parameters"]["n_estimators"] = 1
    metadata["splits"]["test"] = {"first_reference": "2022-01-01T00:00:00+00:00"}
    metadata["source_dataset_path"] = str(prepared.resolve())
    wf = WalkForwardConfig(
        ["2020-12-01", "2021-03-01", "2021-05-01"],
        inner_validation_days=30,
        early_stopping_rounds=5,
        min_train_rows=80,
        min_eval_rows=10,
    )
    plan_path = prepared.parent / "plan.json"
    write_json(
        plan_path,
        {
            "walk_forward": wf.__dict__,
            "recipes": [
                {
                    "id": "original_new",
                    "parameters": {**config.model, "n_estimators": 40},
                    "early_stop": True,
                    "calibrated": False,
                }
            ],
            "calibration_days": 30,
            "independent_final_end_exclusive": "2022-04-01T00:00:00+00:00",
        },
    )
    metadata["plan_sha256"] = sha256(plan_path)
    write_json(
        prepared.parent / "summary.json",
        {"versions": {"original_new": old}, "plan_sha256": sha256(plan_path)},
    )
    write_json(bundle.directory / "metadata.json", metadata)
    digest = sha256(bundle.directory / "metadata.json")
    restored = load_model(config.root, old)
    assert restored.metadata["splits"]["test"]["last_target_time"] == "2022-04-01T00:00:00+00:00"
    recovered = resolve_recipe(config.root, restored.metadata, config)
    assert recovered["parameters"]["n_estimators"] == 40
    newer = expanded_dataset(config, raw)
    new = retrain(config.root, newer, old)
    assert load_model(config.root, new).metadata["fitting"]["early_stopping"]
    assert sha256(bundle.directory / "metadata.json") == digest


def test_reserved_labels_are_rejected_before_fitting(config, prepared, monkeypatch):
    summary_path = config.root / "reservation.json"
    write_json(
        summary_path,
        {
            "prospective": {
                "first_reference": "2021-01-01T00:00:00+00:00",
                "end_reference_exclusive": "2021-02-01T00:00:00+00:00",
            }
        },
    )
    write_json(config.root / "smc_studies/latest.json", {"summary": str(summary_path)})
    with patch("trading_ml.training.pipeline.XGBClassifier") as model:
        with pytest.raises(LeakageError, match="reservado"):
            train(config, prepared)
        model.assert_not_called()


@pytest.mark.parametrize("threshold", [0.55, 0.65, 0.7, 0.8])
def test_smc_rule_states_use_configured_threshold(raw, threshold):
    from trading_ml.smc import detect_smc

    patterns = detect_smc(raw)
    probability, states = rule_probabilities(patterns, threshold)
    np.testing.assert_array_equal(states, directional_states(probability, threshold))


def test_cli_and_api_use_registered_costs(config, prepared, capsys):
    config = replace(
        config,
        backtest=replace(config.backtest, threshold=0.65, commission_bps=27, slippage_bps=13),
    )
    version = train(config, prepared)
    bundle = load_model(config.root, version)
    expected = backtest(bundle, prepared).summary
    text = (
        Path("config/default.toml")
        .read_text(encoding="utf8")
        .replace('root = "artifacts"', f'root = "{config.root.as_posix()}"')
    )
    cli_config = config.root / "cli.toml"
    cli_config.write_text(text, encoding="utf8")
    assert (
        main(
            [
                "--config",
                str(cli_config),
                "backtest",
                "--version",
                version,
                "--data",
                str(prepared),
                "--output",
                str(config.root / "bt"),
            ]
        )
        == 0
    )
    actual = json.loads(capsys.readouterr().out)
    assert actual == expected
    with pytest.raises(DataError, match="registrado"):
        backtest(bundle, prepared, replace(config.backtest, commission_bps=0))


def test_verification_reports_real_subprocess_failures(tmp_path):
    results = [
        subprocess.CompletedProcess([], code, stdout=f"real output {code}", stderr="")
        for code in [0, 1, 0, 0]
    ]
    with (
        patch("trading_ml.verification.subprocess.run", side_effect=results) as execute,
        patch("trading_ml.verification.code_provenance", return_value={}),
    ):
        report = run_checks(tmp_path)
    assert execute.call_count == 4
    assert not report["passed"]
    assert report["checks"][1]["exit_code"] == 1
    assert report["checks"][1]["stdout"] == "real output 1"
