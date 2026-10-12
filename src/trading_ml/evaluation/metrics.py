from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from trading_ml.errors import DataError


def classification_metrics(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> dict[str, Any]:
    """Métricas y reliability bins fijos; ROC-AUC nulo si hay una sola clase."""
    target = np.asarray(target)
    probability = np.asarray(probability, dtype=float)
    if (
        target.ndim != 1
        or probability.ndim != 1
        or not isinstance(bins, int)
        or bins < 1
        or len(target) == 0
        or len(target) != len(probability)
        or not np.isin(target, [0, 1]).all()
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise DataError("Etiquetas o probabilidades inválidas")
    target = target.astype(int)
    predicted = probability >= 0.5
    reliability = []
    ece = 0.0
    indices = np.minimum((probability * bins).astype(int), bins - 1)
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        if count:
            mean_probability = float(probability[mask].mean())
            observed = float(target[mask].mean())
            ece += count / len(target) * abs(mean_probability - observed)
            reliability.append(
                {
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "count": count,
                    "mean_probability": mean_probability,
                    "observed_up_fraction": observed,
                }
            )
    return {
        "rows": len(target),
        "up_fraction": float(target.mean()),
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability))
        if len(np.unique(target)) == 2
        else None,
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "ece_10_bins": float(ece),
        "reliability_bins": reliability,
        "warnings": ["ROC-AUC no definido: una sola clase"] if len(np.unique(target)) < 2 else [],
    }
