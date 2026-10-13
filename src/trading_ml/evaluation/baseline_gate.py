from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_ml.errors import LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import BootstrapConfig, block_bootstrap_comparison
from trading_ml.models.baselines import BASELINE_NAMES, fit_reference_models


def assess_baseline_gate(
    probabilities: dict[str, np.ndarray],
    target: np.ndarray,
    uncertainty: dict[str, Any],
    candidate: str = "xgboost",
    *,
    minimum_loss_gain: float = 0.001,
) -> dict[str, Any]:
    metrics = {name: classification_metrics(target, p) for name, p in probabilities.items()}
    checks = {}
    for baseline in BASELINE_NAMES:
        key = f"{candidate}__minus__{baseline}"
        checks[baseline] = {
            "log_loss_gain": metrics[baseline]["log_loss"] - metrics[candidate]["log_loss"]
            >= minimum_loss_gain,
            "brier_improves": metrics[candidate]["brier_score"] < metrics[baseline]["brier_score"],
            "uncertainty_supports_loss": all(
                block["paired"][key]["log_loss"]["upper"] < 0
                for block in uncertainty["blocks"].values()
            ),
            "uncertainty_supports_brier": all(
                block["paired"][key]["brier_score"]["upper"] < 0
                for block in uncertainty["blocks"].values()
            ),
        }
    accepted = all(all(check.values()) for check in checks.values())
    return {
        "accepted": accepted,
        "status": "passed_baselines" if accepted else "rejected_by_baselines",
        "checks": checks,
        "minimum_log_loss_gain": minimum_loss_gain,
        "failed_checks": [
            f"{name}.{check}"
            for name, items in checks.items()
            for check, passed in items.items()
            if not passed
        ],
        "test_used": False,
        "automatic_promotion": False,
    }


def validation_baseline_report(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    probability: np.ndarray,
    seed: int,
    *,
    scope: str = "selection_validation_not_independent",
) -> dict[str, Any]:
    if train.target_time.max() >= validation.reference_time.min():
        raise LeakageError("Baseline train/validación no está purgado")
    references = fit_reference_models(train, columns, seed=seed)
    probabilities = {"xgboost": probability, **references.probabilities(validation)}
    blocks = tuple(x for x in (7, 30) if x <= len(validation))
    config = BootstrapConfig(samples=500, block_lengths=blocks or (1,), seed=seed)
    intervals = block_bootstrap_comparison(
        validation.target.to_numpy(),
        probabilities,
        np.zeros(len(validation), dtype=int),
        [("xgboost", name) for name in BASELINE_NAMES],
        config,
        include_auc=False,
    )
    return {
        "scope": scope,
        "train_sha256": references.fit_sha256,
        "first_validation_reference": validation.reference_time.min().isoformat(),
        "last_validation_target": validation.target_time.max().isoformat(),
        "baselines": references.to_dict(),
        "metrics": {
            name: classification_metrics(validation.target.to_numpy(), p)
            for name, p in probabilities.items()
        },
        "uncertainty": intervals,
        "decision": assess_baseline_gate(probabilities, validation.target.to_numpy(), intervals),
    }
