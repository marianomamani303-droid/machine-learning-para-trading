from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.errors import DataError


def directional_states(
    probability: np.ndarray, threshold: float = 0.55, ready: np.ndarray | None = None
) -> np.ndarray:
    """Abstención simétrica: [1-threshold, threshold] es neutral salvo extremos."""
    probability = np.asarray(probability, dtype=float)
    if (
        probability.ndim != 1
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
        or not 0.5 < threshold < 1
    ):
        raise DataError("Probabilidades o umbral de abstención inválidos")
    lower = np.nextafter(1 - threshold, np.inf)
    result = np.where(
        probability >= threshold, "bullish", np.where(probability <= lower, "bearish", "neutral")
    )
    if ready is not None:
        readiness = np.asarray(ready, dtype=bool)
        if readiness.shape != probability.shape:
            raise DataError("Estado de calentamiento incompatible")
        result = np.where(readiness, result, "neutral")
    return result


def rule_probabilities(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Confianza heurística, no frecuencia calibrada ni modelo aprendido.

    Suma evidencia actual; conflictos de direcciones y sólo tendencia abstienen.
    """
    weights = {"bos": 2.0, "choch": 2.0, "sweep": 1.5, "fvg": 1.0, "ob": 1.0}
    bull = np.zeros(len(frame))
    bear = np.zeros(len(frame))
    for event, weight in weights.items():
        up, down = ("low", "high") if event == "sweep" else ("up", "down")
        bull += weight * frame[f"smc_{event}_{up}"].to_numpy()
        bear += weight * frame[f"smc_{event}_{down}"].to_numpy()
    for zone in ["fvg", "ob"]:
        bull += frame[f"smc_{zone}_bull_touch"].to_numpy()
        bear += frame[f"smc_{zone}_bear_touch"].to_numpy()
    score = bull - bear + 0.5 * frame.smc_trend.to_numpy()
    eligible = (frame.smc_ready.to_numpy() > 0) & ((bull > 0) ^ (bear > 0)) & (np.abs(score) >= 1)
    probability = np.where(eligible, 0.5 + 0.1 * np.clip(score, -3, 3), 0.5)
    return probability, directional_states(probability)
