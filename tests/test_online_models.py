from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from trading_ml.online.evaluation import evaluate_models
from trading_ml.online.models import learn, new_models, predict, validate_model


def _predictions(count=250, candidate_p=0.99, baseline_p=0.01, horizon=1):
    rows = []
    for index in range(count):
        reference = pd.Timestamp("2026-10-01", tz="UTC") + pd.Timedelta(hours=index)
        for name, probability in (("baseline", baseline_p), ("logistic_fast", candidate_p)):
            rows.append(
                {
                    "model_id": name,
                    "reference_time": reference.isoformat(),
                    "target_time": (reference + pd.Timedelta(hours=horizon)).isoformat(),
                    "horizon": horizon,
                    "probability": probability,
                    "label": 0,
                    "status": "evaluated",
                }
            )
    return rows


def _policy(**overrides):
    return {
        "eval_window": 50,
        "min_samples": 100,
        "patience": 3,
        "alpha": 0.05,
        "min_effect": 0.001,
        "horizon": 1,
        **overrides,
    }


def test_predictions_are_read_only_and_learning_is_incremental_and_serializable():
    models = new_models(2)
    before = copy.deepcopy(models)
    for model in models.values():
        assert predict(model, [0.5, -0.2]) == 0.5
    assert models == before
    for model in models.values():
        learn(model, [0.5, -0.2], 1)
        assert model["updates"] == 1
        assert predict(model, [0.5, -0.2]) > 0.5
    assert json.loads(json.dumps(models)) == models


def test_prequential_probability_is_stored_before_a_label_changes_weights():
    model = new_models(1)["logistic_fast"]
    stored_forecast = predict(model, [1.0])
    learn(model, [1.0], 1)
    assert stored_forecast == 0.5
    assert predict(model, [1.0]) > stored_forecast
    for _ in range(100):
        learn(model, [1.0], 0)
    assert stored_forecast == 0.5


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [1.0, 2.0]])
def test_invalid_features_are_rejected_without_a_model_update(values):
    model = new_models(1)["logistic_fast"]
    before = copy.deepcopy(model)
    with pytest.raises(ValueError):
        learn(model, values, 1)
    assert model == before


def test_small_or_unconfirmed_samples_cannot_disable_or_validate():
    models = new_models(1)
    rows = _predictions(90)
    assert evaluate_models(rows, models, _policy()) == []
    for row in _predictions(300)[len(rows) :]:
        row["status"] = "pending"
        row["label"] = None
        rows.append(row)
    assert evaluate_models(rows, models, _policy()) == []
    assert models["logistic_fast"]["status"] == "active"
    assert models["logistic_fast"]["evaluation"]["validation_state"] == "unvalidated"


def test_persistent_inferiority_disables_without_deleting_and_is_restart_idempotent():
    models = new_models(1)
    rows = _predictions(150)
    reports = evaluate_models(rows, models, _policy())
    assert [row["decision"] for row in reports] == ["inferior_block", "inferior_block"]
    assert models["logistic_fast"]["status"] == "active"
    models = json.loads(json.dumps(models))
    rows = _predictions(200)
    reports = evaluate_models(rows, models, _policy())
    assert reports[-1]["decision"] == "disabled"
    assert models["logistic_fast"]["status"] == "disabled"
    before = copy.deepcopy(models)
    assert evaluate_models(rows, models, _policy()) == []
    learn(models["logistic_fast"], [1.0], 1)
    assert models == before


def test_superiority_requires_persistent_disjoint_blocks_and_never_validates_profitability():
    models = new_models(1)
    reports = evaluate_models(
        _predictions(200, candidate_p=0.01, baseline_p=0.99), models, _policy()
    )
    assert [row["decision"] for row in reports] == [
        "superior_block",
        "superior_block",
        "validated_predictive",
    ]
    assert models["logistic_fast"]["evaluation"]["validation_state"] == "validated_predictive"
    assert all(row["profitability_validated"] is False for row in reports)
    assert all(
        left["last_target"] <= right["first_reference"]
        for left, right in zip(reports, reports[1:], strict=False)
    )


def test_overlapping_horizons_are_thinned_and_only_common_targets_count():
    models = new_models(1)
    rows = _predictions(150, horizon=3)
    reports = evaluate_models(rows, models, _policy(horizon=3))
    assert reports == []
    assert models["logistic_fast"]["evaluation"]["total_pairs"] == 50
    assert models["logistic_fast"]["status"] == "active"
    models = new_models(1)
    rows = [row for row in _predictions(250) if row["model_id"] != "baseline"]
    assert evaluate_models(rows, models, _policy()) == []
    assert models["logistic_fast"]["evaluation"]["total_pairs"] == 0


def test_market_or_label_mismatches_cannot_supply_paired_evidence():
    rows = _predictions(200)
    for row in rows:
        row["symbol"] = "BTCUSDT" if row["model_id"] == "baseline" else "ETHUSDT"
    models = new_models(1)
    assert evaluate_models(rows, models, _policy()) == []
    rows = _predictions(200)
    rows[1]["label"] = 1
    with pytest.raises(ValueError, match="Etiquetas incompatibles"):
        evaluate_models(rows, models, _policy())


def test_global_alpha_budget_accounts_for_models_directions_and_all_looks():
    reports = evaluate_models(_predictions(5000), new_models(1), _policy(patience=1000))
    spent_for_one_direction_one_model = sum(row["alpha_spent"] for row in reports)
    assert spent_for_one_direction_one_model * 2 * 2 < 0.05
    assert all(
        left["alpha_spent"] > right["alpha_spent"]
        for left, right in zip(reports, reports[1:], strict=False)
    )


def test_nonpersistent_bad_windows_do_not_accumulate_a_disable_streak():
    rows = _predictions(200)
    for row in rows[200:300]:
        row["probability"] = 0.5
    models = new_models(1)
    reports = evaluate_models(rows, models, _policy(patience=2))
    assert [row["decision"] for row in reports] == [
        "inferior_block",
        "insufficient_evidence",
        "inferior_block",
    ]
    assert models["logistic_fast"]["status"] == "active"


def test_duplicate_forecasts_are_rejected_instead_of_inflating_evidence():
    rows = _predictions(200)
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="duplicada"):
        evaluate_models(rows, new_models(1), _policy())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("updates", True),
        ("updates", -1),
        ("n_features", 0),
        ("seed", 2**32),
        ("status", "unknown"),
        ("kind", "unknown"),
    ],
)
def test_model_checkpoint_rejects_invalid_schema(key, value):
    model = new_models(1)["logistic_fast"]
    model[key] = value
    with pytest.raises(ValueError):
        validate_model(model)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("weights", [float("inf")]),
        ("weights", [float("nan")]),
        ("weights", []),
        ("weights", [True]),
        ("intercept", float("inf")),
        ("learning_rate", -1),
        ("learning_rate", True),
        ("l2", float("nan")),
        ("l2", -0.1),
    ],
)
def test_prediction_and_learning_reject_invalid_parameters_before_mutation(key, value):
    model = new_models(1)["logistic_fast"]
    model["params"][key] = value
    before = json.dumps(model)
    with pytest.raises(ValueError):
        predict(model, [1.0])
    with pytest.raises(ValueError):
        learn(model, [1.0], 1)
    assert json.dumps(model) == before


def test_overflowing_score_does_not_modify_a_finite_checkpoint():
    model = new_models(1)["logistic_fast"]
    model["params"]["weights"] = [1e308]
    before = copy.deepcopy(model)
    with pytest.raises(ValueError, match="desbordó"):
        learn(model, [10.0], 1)
    assert model == before


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0, -1, True])
def test_baseline_checkpoint_rejects_invalid_beta_counts(value):
    model = new_models(1)["baseline"]
    model["params"]["positive"] = value
    with pytest.raises(ValueError):
        validate_model(model)


def test_significant_deterioration_revokes_validation_but_retains_its_history():
    models = new_models(1)
    good = _predictions(200, candidate_p=0.01, baseline_p=0.99)
    evaluate_models(good, models, _policy())
    state = models["logistic_fast"]["evaluation"]
    validated_at = state["last_validated_target"]
    assert state["validation_state"] == "validated_predictive"
    # One inconclusive block does not prove that a historical edge disappeared.
    neutral = _predictions(250, candidate_p=0.5, baseline_p=0.5)[len(good) :]
    reports = evaluate_models(good + neutral, models, _policy())
    assert reports[-1]["decision"] == "insufficient_evidence"
    assert state["validation_state"] == "validated_predictive"
    bad = _predictions(300)[len(good + neutral) :]
    reports = evaluate_models(good + neutral + bad, models, _policy())
    assert reports[-1]["decision"] == "validation_revoked"
    assert state["validation_state"] == "unvalidated"
    assert state["last_validated_target"] == validated_at
    assert state["validation_revoked_at"] == bad[-1]["target_time"]
    assert models["logistic_fast"]["status"] == "active"
    validate_model(models["logistic_fast"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("look", True),
        ("look", -1),
        ("worse_streak", 1.5),
        ("total_pairs", -1),
        ("last_target", "2026-01-01T00:00:00"),
        ("last_validated_target", "NaT"),
        ("validation_revoked_at", "2026-01-01T00:00:00"),
    ],
)
def test_model_checkpoint_validates_evaluation_counters_and_dates(key, value):
    models = new_models(1)
    evaluate_models(_predictions(200), models, _policy())
    model = models["logistic_fast"]
    model["evaluation"][key] = value
    with pytest.raises(ValueError):
        validate_model(model)
