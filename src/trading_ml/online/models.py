"""Small, serializable incremental estimators for delayed prequential learning.

The caller owns confirmation and exactly-once processing of labels. No estimator
reads data files, future observations, or the historical model registry.
"""

from __future__ import annotations

import math
from typing import Any

PROBABILITY_EPSILON = 1e-6


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_model(model: dict[str, Any]) -> None:
    """Reject malformed or nonfinite checkpoints before inference or mutation."""
    if not isinstance(model, dict):
        raise ValueError("El estado de un modelo debe ser un objeto")
    for name, minimum in (("n_features", 1), ("updates", 0), ("seed", 0)):
        if type(model.get(name)) is not int or model[name] < minimum:
            raise ValueError(f"Estado de modelo: {name} debe ser entero >= {minimum}")
    if model["seed"] >= 2**32:
        raise ValueError("Estado de modelo: seed debe ser menor que 2**32")
    if model.get("status") not in ("active", "disabled"):
        raise ValueError("Estado de modelo: status inválido")
    params = model.get("params")
    if not isinstance(params, dict):
        raise ValueError("Estado de modelo: params debe ser un objeto")
    if model.get("kind") == "beta_frequency":
        if set(params) != {"positive", "negative"} or not all(
            _finite_number(value) and value > 0 for value in params.values()
        ):
            raise ValueError("Contadores beta deben ser positivos y finitos")
        if not math.isfinite(params["positive"] + params["negative"]):
            raise ValueError("La suma de contadores beta debe ser finita")
    elif model.get("kind") == "logistic":
        if set(params) != {"weights", "intercept", "learning_rate", "l2"}:
            raise ValueError("Parámetros logísticos ausentes o desconocidos")
        weights = params["weights"]
        if (
            not isinstance(weights, list)
            or len(weights) != model["n_features"]
            or not all(_finite_number(value) for value in weights)
        ):
            raise ValueError("Los pesos deben ser finitos y respetar n_features")
        if not _finite_number(params["intercept"]):
            raise ValueError("El intercepto debe ser finito")
        if not _finite_number(params["learning_rate"]) or not 0 < params["learning_rate"] <= 1:
            raise ValueError("learning_rate debe ser finito y estar en (0, 1]")
        if not _finite_number(params["l2"]) or not 0 <= params["l2"] <= 1:
            raise ValueError("l2 debe ser finito y estar en [0, 1]")
    else:
        raise ValueError("Modelo incremental desconocido")
    if "evaluation" in model:
        evaluation = model["evaluation"]
        if not isinstance(evaluation, dict):
            raise ValueError("El estado de evaluación debe ser un objeto")
        for name in ("look", "worse_streak", "better_streak", "total_pairs"):
            if type(evaluation.get(name)) is not int or evaluation[name] < 0:
                raise ValueError(f"Contador de evaluación inválido: {name}")
        if evaluation.get("validation_state") not in (
            "unvalidated",
            "validated_predictive",
            "inferior_predictive",
        ):
            raise ValueError("Estado de validación predictiva inválido")
        from datetime import datetime

        for name in ("last_target", "last_validated_target", "validation_revoked_at"):
            if evaluation.get(name) is not None:
                try:
                    stamp = datetime.fromisoformat(evaluation[name])
                    if stamp.tzinfo is None:
                        raise ValueError("Timestamp sin zona")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Fecha de evaluación inválida: {name}") from exc


def new_models(n_features: int, seed: int = 42) -> dict[str, dict[str, Any]]:
    """Return a fixed cohort; zero initialization deliberately needs no RNG."""
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
        raise ValueError("n_features debe ser un entero positivo")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed debe ser un entero entre 0 y 2**32-1")
    shared = {"status": "active", "updates": 0, "n_features": n_features, "seed": seed}
    models = {
        "baseline": {
            **shared,
            "kind": "beta_frequency",
            "params": {"positive": 1.0, "negative": 1.0},
        }
    }
    for name, rate in (("logistic_slow", 0.01), ("logistic_fast", 0.1)):
        models[name] = {
            **shared,
            "kind": "logistic",
            "params": {
                "weights": [0.0] * n_features,
                "intercept": 0.0,
                "learning_rate": rate,
                "l2": 0.001,
            },
        }
    return models


def _features(model: dict[str, Any], features: list[float]) -> list[float]:
    if len(features) != model["n_features"]:
        raise ValueError("Dimensión de features incompatible con el modelo")
    values = [float(value) for value in features]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Las features deben ser finitas")
    # Fixed clipping has no fitted state and cannot consume future statistics.
    return [max(-10.0, min(10.0, value)) for value in values]


def predict(model: dict[str, Any], features: list[float]) -> float:
    """Predict without changing weights, counters, or preprocessing state."""
    validate_model(model)
    values = _features(model, features)
    params = model["params"]
    if model["kind"] == "beta_frequency":
        probability = params["positive"] / (params["positive"] + params["negative"])
    elif model["kind"] == "logistic":
        score = params["intercept"] + sum(
            weight * value for weight, value in zip(params["weights"], values, strict=True)
        )
        if not math.isfinite(score):
            raise ValueError("El estado del modelo desbordó el score logístico")
        # The two branches avoid exp overflow for extreme but finite weights.
        probability = (
            1.0 / (1.0 + math.exp(-score))
            if score >= 0
            else (math.exp(score) / (1.0 + math.exp(score)))
        )
    else:
        raise ValueError(f"Modelo incremental desconocido: {model['kind']}")
    if not math.isfinite(probability):
        raise ValueError("El estado del modelo produjo una probabilidad no finita")
    return max(PROBABILITY_EPSILON, min(1.0 - PROBABILITY_EPSILON, probability))


def learn(model: dict[str, Any], features: list[float], label: int) -> None:
    """Apply one confirmed label; disabled models retain their last state."""
    validate_model(model)
    if label not in (0, 1):
        raise ValueError("La etiqueta confirmada debe ser 0 o 1")
    values = _features(model, features)
    if model["status"] != "active":
        return
    params = dict(model["params"])
    if model["kind"] == "beta_frequency":
        key = "positive" if label == 1 else "negative"
        params[key] += 1.0
    elif model["kind"] == "logistic":
        error = predict(model, features) - label
        rate = params["learning_rate"]
        penalty = params["l2"]
        params["weights"] = [
            weight - rate * (error * value + penalty * weight)
            for weight, value in zip(params["weights"], values, strict=True)
        ]
        params["intercept"] -= rate * error
    else:
        raise ValueError(f"Modelo incremental desconocido: {model['kind']}")
    updated = {**model, "params": params, "updates": model["updates"] + 1}
    validate_model(updated)
    model.update(params=params, updates=updated["updates"])
