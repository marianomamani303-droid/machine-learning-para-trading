"""Prospective paired comparisons with conservative sequential error control.

Scores come only from probabilities stored before the outcome was available.
Disjoint target intervals avoid mechanically overlapping labels. The one-sided
Hoeffding-Azuma bound tests a *conditional* mean Brier loss null, allowing adapted
forecasts; it does not assume IID market returns or establish profitability.
Alpha is shared across a fixed cohort, both directions, and all future looks.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Timestamp de evaluación inválido")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _key(row: dict[str, Any], horizon: int) -> tuple:
    return (
        row.get("symbol", row.get("asset", "")),
        row.get("interval", ""),
        int(row.get("horizon", horizon)),
        _timestamp(row["reference_time"]),
        _timestamp(row["target_time"]),
    )


def _paired_rows(predictions: list[dict[str, Any]], candidate: str, horizon: int) -> list[tuple]:
    selected: dict[str, dict[tuple, dict]] = {"baseline": {}, candidate: {}}
    for row in predictions:
        name = row.get("model_id")
        if name not in selected or row.get("status") != "evaluated":
            continue
        if row.get("label") not in (0, 1):
            raise ValueError("Una predicción evaluada requiere etiqueta binaria confirmada")
        probability = float(row["probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probabilidad de evaluación inválida")
        key = _key(row, horizon)
        if key[2] != horizon:
            continue
        if key[4] <= key[3]:
            raise ValueError("El objetivo debe ser posterior a la predicción")
        if key in selected[name]:
            raise ValueError("Predicción duplicada en la comparación pareada")
        selected[name][key] = row
    baseline = selected["baseline"]
    common = sorted(set(baseline) & set(selected[candidate]), key=lambda key: (key[3], key[4]))
    paired = []
    last_target = None
    for key in common:
        control, challenger = baseline[key], selected[candidate][key]
        if control["label"] != challenger["label"]:
            raise ValueError("Etiquetas incompatibles para el mismo objetivo")
        if last_target is not None and key[3] < last_target:
            continue
        label = int(control["label"])
        difference = (float(challenger["probability"]) - label) ** 2 - (
            float(control["probability"]) - label
        ) ** 2
        paired.append((key[3], key[4], difference))
        last_target = key[4]
    return paired


def evaluate_models(
    predictions: list[dict[str, Any]], models: dict[str, dict], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    """Consume complete, disjoint blocks once and persist each lifecycle decision.

    ``validation_state='validated_predictive'`` means sustained improvement in
    Brier score against the online frequency baseline only. Disabled model state
    and its stored forecasts are retained. A significantly inferior block revokes
    current validation while preserving its historical date. Inconclusive blocks
    alone do not establish deterioration. Repeating a call is idempotent.
    """
    window = int(policy.get("eval_window", 50))
    minimum = int(policy.get("min_samples", 100))
    patience = int(policy.get("patience", 3))
    horizon = int(policy.get("horizon", 1))
    alpha = float(policy.get("alpha", 0.05))
    effect = float(policy.get("min_effect", 0.001))
    if min(window, minimum, patience, horizon) < 1:
        raise ValueError("Ventanas, muestra, horizonte y persistencia deben ser positivos")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1")
    if not math.isfinite(effect) or not 0 <= effect < 1:
        raise ValueError("min_effect debe estar entre 0 y 1")
    if "baseline" not in models:
        raise ValueError("La comparación requiere el baseline de frecuencia")
    competitors = [name for name in models if name != "baseline"]
    reports = []
    for name in competitors:
        model = models[name]
        if model["status"] != "active":
            continue
        state = model.setdefault(
            "evaluation",
            {
                "look": 0,
                "last_target": None,
                "worse_streak": 0,
                "better_streak": 0,
                "total_pairs": 0,
                "validation_state": "unvalidated",
                "last_validated_target": None,
                "validation_revoked_at": None,
            },
        )
        rows = _paired_rows(predictions, name, horizon)
        if state["last_target"] is not None:
            last = _timestamp(state["last_target"])
            rows = [row for row in rows if row[0] >= last]
        while len(rows) >= window:
            block, rows = rows[:window], rows[window:]
            state["total_pairs"] += window
            state["last_target"] = block[-1][1].isoformat()
            difference = sum(row[2] for row in block) / window
            if state["total_pairs"] < minimum:
                continue
            state["look"] += 1
            look = state["look"]
            # Sum_{k>=1} 1/(k(k+1)) = 1. Include two test directions.
            threshold = alpha / (2 * len(competitors) * look * (look + 1))
            worse_p = math.exp(-window * max(0.0, difference - effect) ** 2 / 2)
            better_p = math.exp(-window * max(0.0, -difference - effect) ** 2 / 2)
            worse = worse_p <= threshold
            better = better_p <= threshold
            state["worse_streak"] = state["worse_streak"] + 1 if worse else 0
            state["better_streak"] = state["better_streak"] + 1 if better else 0
            decision = "insufficient_evidence"
            if worse and state["validation_state"] == "validated_predictive":
                state["validation_state"] = "unvalidated"
                state["validation_revoked_at"] = block[-1][1].isoformat()
                decision = "validation_revoked"
            if state["worse_streak"] >= patience:
                model["status"] = "disabled"
                state["validation_state"] = "inferior_predictive"
                decision = "disabled"
            elif state["better_streak"] >= patience:
                state["validation_state"] = "validated_predictive"
                state["last_validated_target"] = block[-1][1].isoformat()
                state["validation_revoked_at"] = None
                decision = "validated_predictive"
            elif worse and decision != "validation_revoked":
                decision = "inferior_block"
            elif better:
                decision = "superior_block"
            reports.append(
                {
                    "candidate": name,
                    "baseline": "baseline",
                    "decision": decision,
                    "paired_n": state["total_pairs"],
                    "block_n": window,
                    "first_reference": block[0][0].isoformat(),
                    "last_target": block[-1][1].isoformat(),
                    "mean_brier_delta": difference,
                    "p_upper_bound_inferior": worse_p,
                    "p_upper_bound_superior": better_p,
                    "alpha_spent": threshold,
                    "look": look,
                    "method": "paired_brier_conditional_hoeffding_azuma_alpha_spending",
                    "validation_state": state["validation_state"],
                    "last_validated_target": state.get("last_validated_target"),
                    "validation_revoked_at": state.get("validation_revoked_at"),
                    "profitability_validated": False,
                }
            )
            if model["status"] == "disabled":
                break
    return reports
