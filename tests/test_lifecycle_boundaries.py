from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_ml.common import read_json, sha256, write_json
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation import evaluate
from trading_ml.models import load_model
from trading_ml.prediction.smc_analysis import analyze_smc
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.training import train
from trading_ml.training.optimization import Experiment, optimize
from trading_ml.training.recipe import protect_reserved_labels
from trading_ml.verification import verify_recent_study


def test_completed_acceptance_never_automatically_promotes(config, prepared):
    from test_walk_forward import config_for_test

    version = train(config, prepared)
    bundle = load_model(config.root, version)
    write_json(config.root / "models/latest.json", {"version": version})
    write_json(config.root / "models/champion.json", {"version": version})
    digests = {p: sha256(p) for p in (config.root / "models").glob("*.json")}
    final = pd.Timestamp(bundle.metadata["splits"]["test"]["first_reference"])
    protocol = config.root / "optimization.toml"
    wf = config_for_test(final)
    protocol.write_text(
        f'''[walk_forward]
boundaries = ["2020-12-01", "2021-03-01", "{final.strftime("%Y-%m-%d")}"]
inner_validation_days = 30
early_stopping_rounds = 5
min_train_rows = 80
min_eval_rows = 10
[budget]
max_experiments = 2
patience = 1
[acceptance]
[data]
extended_start = "2017-08-17"
[experiments]
''',
        encoding="utf8",
    )
    parameters = bundle.metadata["selected_parameters"]
    experiments = [Experiment(name, name, False, False, False, parameters) for name in ["E0", "E1"]]
    import trading_ml.training.optimization as module

    original_metrics = module.classification_metrics
    original_simulate = module.simulate
    final_calls = []

    def metrics(target, probability):
        result = original_metrics(target, probability)
        if len(target) == bundle.metadata["splits"]["test"]["rows"]:
            final_calls.append(1)
            result["log_loss"] = 0.8 if len(final_calls) == 1 else 0.7
            result["brier_score"] = 0.3 if len(final_calls) == 1 else 0.25
        return result

    def simulation(*args, **kwargs):
        result = original_simulate(*args, **kwargs)
        result.summary["strategy_metrics"]["cumulative_return"] = 0.1
        return result

    with (
        patch.object(module, "fixed_experiments", return_value=experiments),
        patch.object(module, "assess_consistency", return_value={"accepted": True}),
        patch.object(module, "classification_metrics", side_effect=metrics),
        patch.object(module, "simulate", side_effect=simulation),
    ):
        result = optimize(config, protocol, version)
    assert len(wf.boundaries) == 3
    assert result["promotion_eligible"]
    assert not result["promoted"]
    assert all(sha256(p) == digest for p, digest in digests.items())
    assert load_model(config.root).metadata["version"] == version


def test_all_frozen_reservations_are_protected(config, prepared):
    frame, _ = load_prepared(prepared)
    first = frame.reference_time.iloc[100]
    write_json(
        config.root / "smc_studies/older/prospective.json",
        {
            "first_reference": first.isoformat(),
            "end_reference_exclusive": (first + config.delta * 10).isoformat(),
        },
    )
    with pytest.raises(LeakageError, match="reservado"):
        protect_reserved_labels(config.root, frame.dropna(subset=["target"]))


def test_partial_prospective_evaluation_produces_no_report(config, prepared):
    version = train(config, prepared)
    bundle = load_model(config.root, version)
    test = bundle.metadata["splits"]["test"]
    test["status"] = "pending_future_data"
    test["last_reference"] = (pd.Timestamp(test["last_reference"]) + config.delta).isoformat()
    output = config.root / "incomplete.json"
    with pytest.raises(DataError, match="pendiente"):
        evaluate(bundle, prepared, output)
    assert not output.exists()


def test_smc_analysis_and_backtest_share_threshold(config, raw):
    from conftest import raw_file

    config = replace(
        config,
        features=replace(config.features, smc=True),
        backtest=replace(config.backtest, threshold=0.7),
    )
    dataset = prepare(config, raw_file(config, raw))
    bundle = load_model(config.root, train(config, dataset))
    result = analyze_smc(bundle, dataset)
    assert result["neutral_band"]["bullish_at_or_above"] == 0.7
    from trading_ml.smc.rules import directional_states

    assert (
        result["smc_rules"]["signal_state"]
        == directional_states(np.array([result["smc_rules"]["probability_up"]]), 0.7)[0]
    )


def test_registered_recipe_digest_detects_corruption(config, prepared):
    version = train(config, prepared)
    path = config.root / "models" / version / "metadata.json"
    metadata = read_json(path)
    metadata["training_recipe"]["parameters"]["n_estimators"] += 1
    write_json(path, metadata)
    with pytest.raises(DataError, match="receta"):
        load_model(config.root, version)


def test_historical_verifier_has_no_writes():
    # Local evidence is optional in clean checkouts; the implementation remains
    # covered independently by subprocess failure tests.
    directory = Path("artifacts/recent_retraining/20261012_short")
    if not directory.exists():
        pytest.skip("Historical local artifacts are not part of Git")
    before = {p: sha256(p) for p in directory.rglob("*") if p.is_file()}
    result = verify_recent_study(Path("artifacts"), directory)
    assert result["matching_features_and_labels"]
    assert all(sha256(p) == digest for p, digest in before.items())
