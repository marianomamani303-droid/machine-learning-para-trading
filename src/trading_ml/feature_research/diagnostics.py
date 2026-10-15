from __future__ import annotations

import numpy as np

from trading_ml.errors import DataError


def binary_loss(target, probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(target)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def block_permutation(rows: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Reorder contiguous blocks without crossing evaluation windows."""
    if block < 1 or rows < 2 * block:
        raise DataError("Se requieren al menos dos bloques para permutación")
    chunks = [np.arange(start, min(start + block, rows)) for start in range(0, rows, block)]
    order = rng.permutation(len(chunks))
    if np.array_equal(order, np.arange(len(chunks))):
        order = np.roll(order, 1)
    return np.concatenate([chunks[i] for i in order])


def permutation_diagnostic(model, frame, columns, groups, *, blocks=(7, 30), repeats=8, seed=42):
    """Positive Δloss = useful to this fitted model; repeats are NOT confidence intervals."""
    x = frame[columns].to_numpy(dtype=float)
    y = frame.target.to_numpy(dtype=int)
    base = float(binary_loss(y, model.predict_proba(x)[:, 1]).mean())
    result = []
    for block in blocks:
        rng = np.random.default_rng(seed + block)
        orders = [block_permutation(len(frame), block, rng) for _ in range(repeats)]
        for name, members in groups.items():
            indices = [columns.index(c) for c in members if c in columns]
            if not indices:
                continue
            values = []
            for order in orders:
                changed = x.copy()
                # All members share a permutation: preserve within-family dependence.
                changed[:, indices] = x[order][:, indices]
                loss = binary_loss(y, model.predict_proba(changed)[:, 1]).mean()
                values.append(float(loss - base))
            result.append(
                {
                    "name": name,
                    "members": [columns[i] for i in indices],
                    "block": block,
                    "delta_loss_mean": float(np.mean(values)),
                    "permutation_std": float(np.std(values, ddof=1)) if repeats > 1 else 0.0,
                    "draws": values,
                }
            )
    return result


def prune_correlated(train, columns, threshold=0.90, priority=None):
    """Greedy Spearman filter fitted on training only; deterministic column preference."""
    if not 0 < threshold <= 1 or not columns or len(set(columns)) != len(columns):
        raise DataError("Filtro de redundancia inválido")
    correlations = train[columns].corr(method="spearman").abs()
    order = sorted(columns, key=lambda c: -(priority or {}).get(c, 0.0))
    kept, removed = [], []
    for name in order:
        if train[name].nunique() < 2:
            removed.append({"feature": name, "reason": "constant_train"})
            continue
        match = next((c for c in kept if correlations.loc[name, c] >= threshold), None)
        if match:
            removed.append(
                {
                    "feature": name,
                    "reason": "train_spearman",
                    "representative": match,
                    "absolute_correlation": float(correlations.loc[name, match]),
                }
            )
        else:
            kept.append(name)
    return kept, removed


def correlation_pairs(train, columns, threshold=0.85):
    corr = train[columns].corr(method="spearman")
    return [
        {"a": a, "b": b, "spearman": float(corr.loc[a, b])}
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
        if abs(corr.loc[a, b]) >= threshold
    ]
