from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

from trading_ml.errors import DataError
from trading_ml.frontier_research.models import (
    create_estimator,
    fit_model,
    predict_probability,
    predict_value,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(9)
    first = rng.normal(size=240)
    second = rng.normal(size=240)
    return pd.DataFrame(
        {
            "first": first,
            "second": second,
            "target": (first + 0.2 * second > 0).astype(int),
            "return": 0.01 * first + 0.002 * second,
        },
        index=pd.date_range("2020-01-01", periods=240),
    )


@pytest.mark.parametrize("kind", ["logistic", "random_forest", "xgboost"])
def test_future_perturbation_does_not_change_fit_or_prefix(frame, kind):
    columns = ["first", "second"]
    recipe = {"kind": kind}
    model = fit_model(frame.iloc[:160], columns, "target", recipe)
    before = predict_probability(model, frame.iloc[160:], columns)
    altered = frame.copy()
    altered.iloc[200:, :2] += 10000
    altered.iloc[200:, altered.columns.get_loc("target")] ^= 1
    second = fit_model(altered.iloc[:160], columns, "target", recipe)
    np.testing.assert_array_equal(
        before[:40], predict_probability(second, altered.iloc[160:200], columns)
    )
    np.testing.assert_array_equal(before, predict_probability(second, frame.iloc[160:], columns))
    if kind == "logistic":
        scaler = model.named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, frame.iloc[:160][columns].mean())
        state = scaler.mean_.copy()
        predict_probability(model, altered.iloc[200:], columns)
        np.testing.assert_array_equal(state, scaler.mean_)
        np.testing.assert_array_equal(state, second.named_steps["scale"].mean_)
    assert np.isfinite(before).all()
    assert ((before >= 0) & (before <= 1)).all()


@pytest.mark.parametrize("kind", ["random_forest", "xgboost"])
def test_regression_outputs_are_reproducible_and_not_probabilities(frame, kind):
    recipe = {"kind": kind}
    columns = ["first", "second"]
    model = fit_model(frame.iloc[:160], columns, "return", recipe, regression=True)
    other = fit_model(
        frame.iloc[:160], columns, frame.iloc[:160]["return"], recipe, regression=True
    )
    predictions = predict_value(model, frame.iloc[160:], columns)
    np.testing.assert_array_equal(predictions, predict_value(other, frame.iloc[160:], columns))
    assert (predictions < 0).any()
    with pytest.raises(DataError, match="classifier"):
        predict_probability(model, frame.iloc[160:], columns)


def test_estimator_configuration_keeps_regularization_and_single_cpu():
    forest = create_estimator({"kind": "random_forest"})
    assert forest.n_estimators == 200
    assert forest.min_samples_leaf == 20
    assert forest.max_depth == 5
    assert forest.n_jobs == 1
    assert forest.random_state == 42
    boost = create_estimator({"kind": "xgboost", "parameters": {"max_depth": 3}})
    parameters = boost.get_params()
    assert parameters["max_depth"] == 3
    assert parameters["min_child_weight"] == 10
    assert parameters["reg_lambda"] == 5
    assert parameters["device"] == "cpu"
    assert parameters["n_jobs"] == 1
    with pytest.raises(DataError, match="n_jobs=1"):
        create_estimator({"kind": "random_forest", "parameters": {"n_jobs": 4}})
    with pytest.raises(DataError, match="binary classification"):
        create_estimator({"kind": "logistic"}, regression=True)
    with pytest.raises(DataError, match="Unknown"):
        create_estimator({"kind": "neural_network"})


@pytest.mark.parametrize("kind", ["lightgbm", "catboost"])
def test_missing_optional_boosting_package_has_explicit_error(monkeypatch, kind):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == kind:
            raise ImportError("optional package deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(DataError, match=f"optional {kind} package"):
        create_estimator({"kind": kind})
    create_estimator({"kind": "logistic"})


def test_model_rejects_misaligned_targets_and_invalid_features(frame):
    recipe = {"kind": "logistic"}
    with pytest.raises(DataError, match="index"):
        fit_model(frame, ["first"], frame.target.iloc[::-1], recipe)
    with pytest.raises(DataError, match="both binary classes"):
        fit_model(frame, ["first"], np.ones(len(frame)), recipe)
    invalid = frame.copy()
    invalid.loc[invalid.index[0], "first"] = np.inf
    with pytest.raises(DataError, match="nonfinite"):
        fit_model(invalid, ["first"], "target", recipe)
    with pytest.raises(DataError, match="unique columns"):
        fit_model(frame, ["first", "first"], "target", recipe)


def test_numeric_features_and_empty_prediction(frame):
    columns = ["first", "second"]
    numeric_objects = frame.astype({"first": object, "second": object})
    model = fit_model(numeric_objects, columns, "target", {"kind": "logistic"})
    np.testing.assert_array_equal(
        predict_probability(model, frame, columns),
        predict_probability(model, numeric_objects, columns),
    )
    assert predict_probability(model, frame.iloc[:0], columns).shape == (0,)
