from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from trading_ml.common import sha256
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.baseline_gate import assess_baseline_gate, validation_baseline_report
from trading_ml.evaluation.diagnostics import probability_diagnostics
from trading_ml.evaluation.uncertainty import (
    BootstrapConfig,
    _auc,
    block_bootstrap_comparison,
    moving_block_indices,
)
from trading_ml.models import load_model
from trading_ml.models.baselines import BASELINE_NAMES, fit_reference_models
from trading_ml.models.lifecycle import promote
from trading_ml.preprocessing import load_prepared, prepare
from trading_ml.training import train
from trading_ml.training.predictive_study import (
    evaluate_development,
    feature_families,
    fixed_ablations,
    run_predictive_study,
)


def protocol(final):
    return {
        "walk_forward": {
            "boundaries": ["2020-12-01", "2021-03-01", final.strftime("%Y-%m-%d")],
            "inner_validation_days": 30,
            "early_stopping_rounds": 5,
            "min_train_rows": 80,
            "min_eval_rows": 10,
        },
        "budget": {"max_xgboost_recipes": 8, "max_fits": 19, "max_trees": 40, "cpu_threads": 1},
        "baselines": {"logistic_C": 1.0},
        "bootstrap": {"samples": 100, "block_lengths": [7, 14], "confidence": 0.95, "seed": 42},
        "acceptance": {"minimum_log_loss_gain": 0.001, "required_fold_fraction": 0.6},
        "diagnostics": {
            "historical_train_validation_test": True,
            "reuse_smc_development_scores": False,
        },
    }


def test_reference_models_fit_only_past(config, prepared):
    frame, _ = load_prepared(prepared)
    columns = ["return_1", "rsi", "volatility"]
    fitting = frame.iloc[:300]
    references = fit_reference_models(fitting, columns)
    np.testing.assert_allclose(references.scaler.mean_, fitting[columns].mean())
    validation = frame.iloc[301:401].copy()
    before = references.to_dict()
    predictions = references.probabilities(validation)
    np.testing.assert_allclose(predictions["constant_05"], 0.5)
    np.testing.assert_allclose(predictions["train_prevalence"], fitting.target.mean())
    validation[columns] *= 100
    validation["target"] = 1 - validation.target
    references.probabilities(validation)
    assert references.to_dict() == before


@pytest.mark.parametrize("seed", [1, 42])
def test_block_indices_preserve_fold_and_contiguous_blocks(seed):
    groups = np.repeat([1, 2], [40, 50])
    indices = moving_block_indices(groups, 100, 7, seed)
    np.testing.assert_array_equal(groups[indices], np.broadcast_to(groups, indices.shape))
    for start, size in [(0, 40), (40, 50)]:
        local = indices[:, start : start + size]
        for offset in range(0, size, 7):
            assert (np.diff(local[:, offset : min(offset + 7, size)], axis=1) == 1).all()
    np.testing.assert_array_equal(indices, moving_block_indices(groups, 100, 7, seed))


@pytest.mark.parametrize(
    "groups,block", [(np.array([0, 1, 0, 1]), 1), (np.repeat([0, 1], [2, 20]), 3)]
)
def test_invalid_bootstrap_groups_rejected(groups, block):
    with pytest.raises(DataError):
        moving_block_indices(groups, 100, block, 42)


@pytest.mark.parametrize("ties", [False, True])
def test_auc_matches_sklearn(ties):
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    if ties:
        p = p.round(1)
    assert _auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert _auc(y, np.full(len(y), 0.5)) == 0.5


def test_paired_bootstrap_identical_predictions_have_zero_difference():
    y = np.tile([0, 1], 50)
    p = np.linspace(0.1, 0.9, 100)
    result = block_bootstrap_comparison(
        y,
        {"a": p, "b": p.copy()},
        np.zeros(100),
        [("a", "b")],
        BootstrapConfig(samples=100, block_lengths=(7, 14)),
    )
    for block in result["blocks"].values():
        for metric in block["paired"]["a__minus__b"].values():
            assert metric["estimate"] == metric["lower"] == metric["upper"] == 0


def test_gate_accepts_strong_signal_and_rejects_constant():
    y = np.tile([0, 1], 100)
    baselines = {name: np.full(200, 0.5) for name in BASELINE_NAMES}
    for p, accepted in [(np.where(y, 0.9, 0.1), True), (np.full(200, 0.5), False)]:
        probabilities = {"xgboost": p, **baselines}
        intervals = block_bootstrap_comparison(
            y,
            probabilities,
            np.zeros(200),
            [("xgboost", name) for name in BASELINE_NAMES],
            BootstrapConfig(samples=100, block_lengths=(7, 30)),
            include_auc=False,
        )
        decision = assess_baseline_gate(probabilities, y, intervals)
        assert decision["accepted"] is accepted
        assert not decision["test_used"]


def test_baseline_validation_purge_enforced(config, prepared):
    frame, _ = load_prepared(prepared)
    with pytest.raises(LeakageError, match="purgado"):
        validation_baseline_report(
            frame.iloc[:100], frame.iloc[99:200], ["return_1"], np.full(101, 0.5), 42
        )


def test_smc_constant_diagnostics_distinguish_bias_from_resolution():
    y = np.tile([0, 1], 100)
    diagnostic = probability_diagnostics(y, np.full(200, 0.43956043956))
    assert diagnostic["span"] == 0
    assert diagnostic["unique_scores"] == 1
    assert diagnostic["bearish_count"] == 200
    assert diagnostic["metrics"]["roc_auc"] == 0.5
    assert diagnostic["brier_grouped_decomposition"]["resolution"] == 0


def test_development_and_gates_do_not_change_when_test_changes(config, raw):
    from conftest import raw_file

    config = replace(config, features=replace(config.features, smc=True))
    dataset = prepare(config, raw_file(config, raw))
    frame, _ = load_prepared(dataset)
    final = pd.Timestamp("2021-05-01", tz="UTC")
    specification = protocol(final)
    first = evaluate_development(
        frame, config, config.model, specification, final, config.root / "a"
    )
    changed = frame.copy()
    mask = changed.reference_time >= final
    changed.loc[mask, "target"] = 1 - changed.loc[mask, "target"]
    changed.loc[mask, "open"] *= 3
    # Mutate every causal feature of the excluded test.
    from trading_ml.features import feature_columns

    changed.loc[mask, feature_columns(config.features)] *= 100
    second = evaluate_development(
        changed, config, config.model, specification, final, config.root / "b"
    )
    assert first == second
    for model in first["models"].values():
        if "decision" in model:
            assert not model["decision"]["test_used"]


def test_budget_is_enforced_before_any_fit(config, raw):
    from conftest import raw_file

    config = replace(config, features=replace(config.features, smc=True))
    frame, _ = load_prepared(prepare(config, raw_file(config, raw)))
    final = pd.Timestamp("2021-05-01", tz="UTC")
    spec = protocol(final)
    spec["budget"]["max_fits"] = 1
    with pytest.raises(DataError, match="presupuesto"):
        evaluate_development(frame, config, config.model, spec, final, config.root / "unwritten")
    assert not (config.root / "unwritten").exists()


def test_new_rejected_model_cannot_be_promoted(config, prepared, monkeypatch):
    import trading_ml.training.pipeline as pipeline

    original = pipeline.validation_baseline_report

    def rejected(*args, **kwargs):
        result = original(*args, **kwargs)
        result["decision"]["accepted"] = False
        return result

    monkeypatch.setattr(pipeline, "validation_baseline_report", rejected)
    version = train(config, prepared)
    bundle = load_model(config.root, version)
    assert bundle.metadata["registration_status"] == "rejected_by_baselines"
    with pytest.raises(DataError, match="baselines"):
        promote(config.root, version, config.root / "no_report.json", "cannot override gate")


def test_study_cache_preserves_artifacts_and_runs_no_new_fits(config, prepared, monkeypatch):
    version = train(config, prepared)
    bundle = load_model(config.root, version)
    final = pd.Timestamp(bundle.metadata["splits"]["test"]["first_reference"])
    path = config.root / "protocol.toml"
    # Small fixed protocol mirrors the real one, without historical SMC files.
    path.write_text(
        f'''[walk_forward]
boundaries = ["2020-12-01", "2021-03-01", "{final.strftime("%Y-%m-%d")}"]
inner_validation_days = 30
early_stopping_rounds = 5
min_train_rows = 80
min_eval_rows = 10
[budget]
max_xgboost_recipes = 8
max_fits = 19
max_trees = 40
cpu_threads = 1
[baselines]
logistic_C = 1.0
[xgboost]
n_estimators = 20
max_depth = 2
learning_rate = 0.03
subsample = 0.85
colsample_bytree = 0.85
reg_lambda = 5.0
min_child_weight = 10
[bootstrap]
samples = 100
block_lengths = [7, 14]
confidence = 0.95
seed = 42
[acceptance]
minimum_log_loss_gain = 0.001
required_fold_fraction = 0.60
[diagnostics]
historical_train_validation_test = true
reuse_smc_development_scores = false
''',
        encoding="utf8",
    )
    before = {p: sha256(p) for p in (config.root / "models").rglob("*") if p.is_file()}
    report = run_predictive_study(config, path, version)
    assert report["actual_fits"] == {"xgboost": 16, "logistic": 3}
    assert report["historical_diagnostics"]["used_for_selection_or_rejection"] is False
    assert all(sha256(p) == digest for p, digest in before.items())
    import trading_ml.training.predictive_study as study

    monkeypatch.setattr(
        study, "evaluate_development", lambda *a, **k: pytest.fail("cache retrained")
    )
    assert run_predictive_study(config, path, version) == report


def test_feature_families_cover_contract_and_ablation_is_explicit(config):
    from trading_ml.features import feature_columns

    columns = feature_columns(replace(config.features, smc=True))
    families = feature_families(columns)
    assert sorted(c for group in families.values() for c in group) == sorted(columns)
    recipes = fixed_ablations(columns)
    assert len(recipes) == 8
    assert not set(families["momentum"]) & set(recipes["without_momentum"])
    assert recipes["xgboost_smc_only"] == families["smc"]
    with pytest.raises(DataError, match="familias"):
        feature_families([*columns, "target"])
