from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics


@dataclass(frozen=True)
class BootstrapConfig:
    samples: int = 1000
    block_lengths: tuple[int, ...] = (7, 30)
    confidence: float = 0.95
    seed: int = 42

    def validate(self) -> None:
        if not isinstance(self.samples, int) or not 100 <= self.samples <= 5000:
            raise DataError("Bootstrap admite entre 100 y 5000 remuestras")
        if not self.block_lengths or any(
            not isinstance(x, int) or x < 1 for x in self.block_lengths
        ):
            raise DataError("Longitudes de bloque inválidas")
        if len(set(self.block_lengths)) != len(self.block_lengths) or not 0 < self.confidence < 1:
            raise DataError("Configuración de bootstrap inválida")


def moving_block_indices(
    groups: np.ndarray,
    samples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Non-circular blocks sampled separately per contiguous fold, never across folds."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) < block_length or block_length < 1:
        raise DataError("Muestra insuficiente para bootstrap por bloques")
    runs = np.split(np.arange(len(groups)), np.flatnonzero(groups[1:] != groups[:-1]) + 1)
    if len({groups[r[0]] for r in runs}) != len(runs) or any(len(r) < block_length for r in runs):
        raise DataError("Folds repetidos/no contiguos o menores que el bloque")
    rng = np.random.default_rng(seed)
    chunks = []
    for run in runs:
        count = int(np.ceil(len(run) / block_length))
        starts = rng.integers(0, len(run) - block_length + 1, size=(samples, count))
        local = (starts[:, :, None] + np.arange(block_length)).reshape(samples, -1)[:, : len(run)]
        chunks.append(run[local])
    return np.concatenate(chunks, axis=1)


def _auc(target: np.ndarray, probability: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; checked against sklearn."""
    positives = int(target.sum())
    if positives == 0 or positives == len(target):
        return np.nan
    order = np.argsort(probability, kind="stable")
    sorted_p = probability[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    ends = np.r_[starts[1:], len(target)]
    ranks = np.repeat((starts + ends + 1) / 2, ends - starts)
    return float(
        (ranks[target[order] == 1].sum() - positives * (positives + 1) / 2)
        / (positives * (len(target) - positives))
    )


def _losses(
    target: np.ndarray, probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.clip(probability, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return (
        -(target * np.log(p) + (1 - target) * np.log1p(-p)),
        (probability - target) ** 2,
        ((probability >= 0.5) == target).astype(float),
    )


def _interval(point: float | None, values: np.ndarray, confidence: float) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    tail = (1 - confidence) / 2
    return {
        "estimate": point,
        "lower": float(np.quantile(finite, tail)) if len(finite) else None,
        "upper": float(np.quantile(finite, 1 - tail)) if len(finite) else None,
        "valid_draws": len(finite),
        "undefined_draws": len(values) - len(finite),
    }


def block_bootstrap_comparison(
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
    groups: np.ndarray,
    pairs: list[tuple[str, str]],
    config: BootstrapConfig,
    *,
    include_auc: bool = True,
) -> dict[str, Any]:
    config.validate()
    target = np.asarray(target)
    if len(groups) != len(target) or not probabilities:
        raise DataError("Grupos/probabilidades incompatibles")
    metrics = {name: classification_metrics(target, p) for name, p in probabilities.items()}
    target = target.astype(int)
    if any(a not in metrics or b not in metrics for a, b in pairs):
        raise DataError("Par de bootstrap desconocido")
    losses = {
        name: _losses(target, np.asarray(p, dtype=float)) for name, p in probabilities.items()
    }
    results = {}
    for block in config.block_lengths:
        indices = moving_block_indices(groups, config.samples, block, config.seed)
        draws = {}
        for name, p in probabilities.items():
            loss, brier, accuracy = losses[name]
            draws[name] = {
                "log_loss": loss[indices].mean(axis=1),
                "brier_score": brier[indices].mean(axis=1),
                "accuracy": accuracy[indices].mean(axis=1),
            }
            if include_auc:
                p = np.asarray(p)
                draws[name]["roc_auc"] = np.array([_auc(target[row], p[row]) for row in indices])
        results[str(block)] = {
            "models": {
                name: {
                    metric: _interval(metrics[name][metric], values, config.confidence)
                    for metric, values in draw.items()
                }
                for name, draw in draws.items()
            },
            "paired": {
                f"{a}__minus__{b}": {
                    metric: _interval(
                        metrics[a][metric] - metrics[b][metric]
                        if metrics[a][metric] is not None and metrics[b][metric] is not None
                        else None,
                        draws[a][metric] - draws[b][metric],
                        config.confidence,
                    )
                    for metric in draws[a]
                }
                for a, b in pairs
            },
        }
    return {
        "method": "noncircular_moving_blocks_within_fold",
        "samples": config.samples,
        "confidence": config.confidence,
        "seed": config.seed,
        "blocks": results,
        "difference_convention": "candidate minus reference; negative loss/Brier is better",
        "limitations": [
            "Conditional on these historical periods and fixed predictions; not a post-selection or future-regime guarantee.",
            "Intervals are marginal, not a multiplicity-adjusted familywise claim.",
        ],
    }
