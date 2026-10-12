from __future__ import annotations

from typing import Any

import numpy as np

from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.smc.rules import directional_states


def signal_quality(
    target: np.ndarray,
    probability: np.ndarray,
    states: np.ndarray | None = None,
    threshold: float = 0.55,
) -> dict[str, Any]:
    """No cuenta neutrales como aciertos; informa cobertura y calidad selectiva."""
    target = np.asarray(target)
    states = directional_states(probability, threshold) if states is None else np.asarray(states)
    if (
        len(target) != len(states)
        or len(target) == 0
        or not np.isin(states, ["bullish", "bearish", "neutral"]).all()
        or not np.isin(target, [0, 1]).all()
    ):
        raise DataError("Señales o etiquetas incompatibles")
    bull, bear = states == "bullish", states == "bearish"
    active = bull | bear
    return {
        "rows": len(target),
        "bullish": int(bull.sum()),
        "bearish": int(bear.sum()),
        "neutral": int((~active).sum()),
        "directional_signals": int(active.sum()),
        "coverage": float(active.mean()),
        "bullish_precision": float(target[bull].mean()) if bull.any() else None,
        "bearish_precision": float((1 - target[bear]).mean()) if bear.any() else None,
        "direction_accuracy": float(
            ((states[active] == "bullish").astype(int) == target[active]).mean()
        )
        if active.any()
        else None,
        "active_probability_metrics": classification_metrics(
            target[active], np.asarray(probability)[active]
        )
        if active.any()
        else None,
        "warnings": ["Pocas señales direccionales: estimación selectiva inestable"]
        if active.sum() < 30
        else [],
    }
