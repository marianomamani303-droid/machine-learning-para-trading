from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import _auc, moving_block_indices


def paired_predictive(
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
    groups: np.ndarray,
    pairs: list[tuple[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Simultaneous bootstrap max-|t| intervals across pairs, metrics and blocks.

    Centers paired draws at their estimates; uses a 95% joint critical value,
    avoiding poorly resolved Bonferroni extreme quantiles. This is conditional
    bootstrap inference, not a post-selection or independent future guarantee.
    """
    metrics = {name: classification_metrics(target, p) for name, p in probabilities.items()}
    family = max(1, len(pairs) * 3 * len(config["block_lengths"]))
    result = {}
    standardized = []
    intervals = []
    for block in config["block_lengths"]:
        indices = moving_block_indices(groups, config["samples"], block, config["seed"])
        draws = {}
        for name, raw in probabilities.items():
            p = np.clip(raw, 1e-12, 1 - 1e-12)
            loss = -(target * np.log(p) + (1 - target) * np.log1p(-p))
            draws[name] = {
                "log_loss": loss[indices].mean(axis=1),
                "accuracy": ((raw >= 0.5) == target)[indices].mean(axis=1),
                "roc_auc": np.array([_auc(target[row], raw[row]) for row in indices]),
            }
        result[str(block)] = {}
        for a, b in pairs:
            pair = {}
            for metric in ["log_loss", "accuracy", "roc_auc"]:
                values = draws[a][metric] - draws[b][metric]
                values = values[np.isfinite(values)]
                point = (
                    metrics[a][metric] - metrics[b][metric]
                    if metrics[a][metric] is not None and metrics[b][metric] is not None
                    else None
                )
                se = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                interval = {
                    "estimate": point,
                    "lower": None,
                    "upper": None,
                    "valid_draws": len(values),
                    "bootstrap_se": se,
                }
                if point is not None and len(values) == config["samples"]:
                    standardized.append(
                        np.abs((values - point) / se) if se > 1e-12 else np.zeros(len(values))
                    )
                    intervals.append(interval)
                pair[metric] = interval
            result[str(block)][a + "__minus__" + b] = pair
    critical = (
        float(np.quantile(np.max(np.array(standardized), axis=0), config["confidence"]))
        if standardized
        else None
    )
    for interval in intervals:
        interval["lower"] = interval["estimate"] - critical * interval["bootstrap_se"]
        interval["upper"] = interval["estimate"] + critical * interval["bootstrap_se"]
    return {
        "metrics": metrics,
        "blocks": result,
        "family_comparisons": family,
        "simultaneous_confidence": config["confidence"],
        "max_t_critical": critical,
        "samples": config["samples"],
        "method": "paired_non_circular_moving_blocks_bootstrap_max_abs_t",
        "limitations": "Conditional centered bootstrap, not a post-selection guarantee; historical selection and shared asset/regimes are not independent future confirmation",
    }


def drift_diagnostic(
    train: pd.DataFrame, evaluation: pd.DataFrame, columns: list[str]
) -> dict[str, Any]:
    """Descriptive train-bin PSI; never uses evaluation to change model/threshold."""
    output = {}
    for name in columns:
        edges = np.unique(np.quantile(train[name], np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        previous = np.histogram(train[name], edges)[0] / len(train)
        current = np.histogram(evaluation[name], edges)[0] / len(evaluation)
        previous, current = np.clip(previous, 1e-6, None), np.clip(current, 1e-6, None)
        output[name] = float(np.sum((current - previous) * np.log(current / previous)))
    return {"psi": output, "model_or_threshold_changed": False, "train_bins_only": True}
