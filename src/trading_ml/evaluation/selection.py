from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from trading_ml.errors import DataError


@dataclass(frozen=True)
class Acceptance:
    min_log_loss_gain: float = 0.003
    min_auc_gain: float = 0.005
    required_fold_fraction: float = 0.6
    min_total_trades: int = 30
    require_positive_compounded_return: bool = True
    require_beating_constant_log_loss: bool = True

    def validate(self) -> None:
        if (
            not 0 < self.required_fold_fraction <= 1
            or min(self.min_log_loss_gain, self.min_auc_gain, self.min_total_trades) < 0
        ):
            raise DataError("Regla de aceptación inválida")


def assess_consistency(
    reference: dict[str, Any], candidate: dict[str, Any], rule: Acceptance
) -> dict[str, Any]:
    """Regla predefinida sobre folds de desarrollo; nunca recibe el test final."""
    rule.validate()
    previous, current = reference["folds"], candidate["folds"]
    if (
        not previous
        or len(previous) != len(current)
        or any(
            a["first_reference"] != b["first_reference"]
            or a["last_reference"] != b["last_reference"]
            or a["metrics"]["rows"] != b["metrics"]["rows"]
            for a, b in zip(previous, current, strict=True)
        )
    ):
        raise DataError("Comparación walk-forward requiere ventanas idénticas")
    required = math.ceil(rule.required_fold_fraction * len(current))
    probabilistic_wins = sum(
        b["metrics"]["log_loss"] < a["metrics"]["log_loss"]
        and b["metrics"]["brier_score"] < a["metrics"]["brier_score"]
        for a, b in zip(previous, current, strict=True)
    )
    economic_wins = sum(
        b["backtest"]["strategy_metrics"]["cumulative_return"]
        > a["backtest"]["strategy_metrics"]["cumulative_return"]
        for a, b in zip(previous, current, strict=True)
    )
    ref, cand = reference["summary"], candidate["summary"]
    checks = {
        "log_loss_gain": ref["metrics"]["log_loss"] - cand["metrics"]["log_loss"]
        >= rule.min_log_loss_gain,
        "mean_fold_auc_gain": cand["mean_fold_auc"] is not None
        and ref["mean_fold_auc"] is not None
        and cand["mean_fold_auc"] - ref["mean_fold_auc"] >= rule.min_auc_gain,
        "probabilistic_consistency": probabilistic_wins >= required,
        "economic_consistency": economic_wins >= required,
        "compounded_return_not_worse": cand["compounded_return"] >= ref["compounded_return"],
        "positive_compounded_return": not rule.require_positive_compounded_return
        or cand["compounded_return"] > 0,
        "minimum_trades": cand["trade_count"] >= rule.min_total_trades,
        "beats_constant_log_loss": not rule.require_beating_constant_log_loss
        or cand["metrics"]["log_loss"] < cand["constant_metrics"]["log_loss"],
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "probabilistic_winning_folds": int(probabilistic_wins),
        "economic_winning_folds": int(economic_wins),
        "required_winning_folds": required,
        "rule": asdict(rule),
    }


def summarize_folds(
    folds: list[dict[str, Any]], metrics: dict[str, Any], constant_metrics: dict[str, Any]
) -> dict[str, Any]:
    aucs = [fold["metrics"]["roc_auc"] for fold in folds]
    returns = [fold["backtest"]["strategy_metrics"]["cumulative_return"] for fold in folds]
    return {
        "metrics": metrics,
        "constant_metrics": constant_metrics,
        "mean_fold_auc": float(np.mean(aucs)) if all(value is not None for value in aucs) else None,
        "compounded_return": float(np.prod(1 + np.array(returns)) - 1),
        "positive_return_folds": sum(value > 0 for value in returns),
        "trade_count": sum(fold["backtest"]["trade_count"] for fold in folds),
        "mean_fold_return": float(np.mean(returns)),
    }
