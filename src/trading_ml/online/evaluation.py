"""Prospective paired comparisons with conservative sequential error control.

Scores come only from probabilities stored before the outcome was available.
Disjoint target intervals avoid mechanically overlapping labels. The one-sided
Hoeffding-Azuma bound tests a *conditional* mean Brier loss null, allowing adapted
forecasts; it does not assume IID market returns or establish profitability.
Alpha is shared across a fixed cohort, both directions, and all future looks.

Includes:
1. Legacy Hoeffding-Azuma paired Brier evaluator (preserved for schema v2 states).
2. Predictable Empirical Bernstein / HAC variance-adaptive sequential evaluator.
3. Multi-target proper scoring rules (Direction, Expected Return, Volatility, Quantiles).
4. Champion-Challenger selection per (symbol, interval, horizon, target_type).
5. Analytical MDE / sample-size audit and 6-scenario Monte Carlo simulation benchmark.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.online.adaptive import MODEL_SPECS


@functools.lru_cache(maxsize=32768)
def _timestamp_cached(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Timestamp de evaluación inválido")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _timestamp(value: Any) -> pd.Timestamp:
    if isinstance(value, str):
        return _timestamp_cached(value)
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


def _paired_rows(
    predictions: list[dict[str, Any]],
    candidate: str,
    horizon: int,
    *,
    baseline_id: str = "baseline",
) -> list[tuple]:
    selected: dict[str, dict[tuple, dict]] = {baseline_id: {}, candidate: {}}
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
    baseline = selected[baseline_id]
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
        p_chal = float(challenger["probability"])
        p_ctrl = float(control["probability"])
        difference = (p_chal - label) ** 2 - (p_ctrl - label) ** 2
        # Conditional range of d_t given F_t before y_t in {0,1} is 2 * |p_chal - p_ctrl|
        predictable_range = 2.0 * abs(p_chal - p_ctrl)
        paired.append((key[3], key[4], difference, predictable_range))
        last_target = key[4]
    return paired


def _hac_variance(diffs: list[float], predictable_ranges: list[float] | None = None) -> tuple[float, float]:
    """Compute Newey-West lag-1 HAC variance and predictable scale bound c_max."""
    n = len(diffs)
    if n <= 1:
        return 0.01, 2.0
    mean_d = sum(diffs) / n
    centered = [d - mean_d for d in diffs]
    gamma_0 = sum(c * c for c in centered) / max(1, n - 1)
    gamma_1 = (
        sum(centered[i] * centered[i - 1] for i in range(1, n)) / max(1, n - 1)
        if n >= 3
        else 0.0
    )
    hac_var = max(gamma_0, gamma_0 + max(0.0, gamma_1), 1e-6)
    if predictable_ranges:
        c_max = max(max(predictable_ranges), 0.02)
        # Predictable quadratic variation floor from Azuma conditional ranges
        pred_qv = sum((0.5 * r) ** 2 for r in predictable_ranges) / n
        eff_var = max(hac_var, 0.25 * pred_qv, 1e-6)
    else:
        c_max = 2.0
        eff_var = hac_var
    return float(min(1.0, eff_var)), float(min(2.0, c_max))


def empirical_bernstein_one_sided_pvalue(
    n: int,
    excess_effect: float,
    hac_variance: float,
    scale_bound: float = 2.0,
) -> float:
    """One-sided predictable Empirical Bernstein / Freedman concentration bound."""
    if n < 2 or excess_effect <= 0.0:
        return 1.0
    var_term = 2.0 * max(1e-6, float(hac_variance))
    scale_term = (4.0 / 3.0) * max(0.02, float(scale_bound)) * float(excess_effect)
    exponent = -(float(n) * (float(excess_effect) ** 2)) / (var_term + scale_term)
    return float(math.exp(max(-100.0, exponent)))


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
    method_cfg = str(policy.get("method", "hoeffding_azuma"))
    if min(window, minimum, patience, horizon) < 1:
        raise ValueError("Ventanas, muestra, horizonte y persistencia deben ser positivos")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1")
    if not math.isfinite(effect) or not 0 <= effect < 1:
        raise ValueError("min_effect debe estar entre 0 y 1")
    if "baseline" not in models:
        raise ValueError("La comparación requiere el baseline de frecuencia")

    # Only compare directional/probability models against 'baseline' in Brier evaluation;
    # pure volatility models emit constant probability=0.5 and are evaluated in their own target family.
    competitors = [
        name
        for name, m in models.items()
        if name != "baseline"
        and m.get("target_type", "direction") in ("direction", "expected_return", "quantiles")
        and m.get("kind") not in ("return_baseline", "volatility_baseline", "quantile_baseline")
    ]
    if not competitors:
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
        all_rows_for_candidate = list(rows)
        if state["last_target"] is not None:
            last = _timestamp(state["last_target"])
            rows = [row for row in rows if row[0] >= last]
        while len(rows) >= window:
            block, rows = rows[:window], rows[window:]
            state["total_pairs"] += window
            state["last_target"] = block[-1][1].isoformat()
            diffs = [row[2] for row in block]
            pred_ranges = [row[3] for row in block]
            difference = sum(diffs) / window
            if state["total_pairs"] < minimum:
                continue
            state["look"] += 1
            look = state["look"]
            # Sum_{k>=1} 1/(k(k+1)) = 1. Include two test directions.
            threshold = alpha / (2 * len(competitors) * look * (look + 1))
            if method_cfg == "empirical_bernstein":
                # Use cumulative non-overlapping trajectory up to current block end for anytime-valid power,
                # plus local block test for rapid regime-shift degradation detection.
                cum_rows = [r for r in all_rows_for_candidate if r[1] <= block[-1][1]]
                cum_diffs = [r[2] for r in cum_rows]
                cum_ranges = [r[3] for r in cum_rows]
                n_cum = len(cum_diffs)
                cum_mean = sum(cum_diffs) / max(1, n_cum)
                hac_var_cum, c_max_cum = _hac_variance(cum_diffs, cum_ranges)
                hac_var_blk, c_max_blk = _hac_variance(diffs, pred_ranges)
                # Scale min_effect proportionally to predictable quadratic variation when variance is small
                eff_margin = min(effect, 0.15 * hac_var_cum)
                worse_p_cum = empirical_bernstein_one_sided_pvalue(
                    n_cum, max(0.0, cum_mean - eff_margin), hac_var_cum, c_max_cum
                )
                worse_p_blk = empirical_bernstein_one_sided_pvalue(
                    window, max(0.0, difference - eff_margin), hac_var_blk, c_max_blk
                )
                worse_p = min(worse_p_cum, worse_p_blk)
                better_p = empirical_bernstein_one_sided_pvalue(
                    n_cum, max(0.0, -cum_mean - eff_margin), hac_var_cum, c_max_cum
                )
                # If the local block is actively worse (difference > 0), prevent superiority streak increment
                if difference >= 0.0:
                    better_p = max(better_p, threshold * 2.0)
                hac_var = hac_var_cum
                c_max = c_max_cum
                method_name = "paired_brier_predictable_empirical_bernstein_alpha_spending"
            else:
                worse_p = math.exp(-window * max(0.0, difference - effect) ** 2 / 2)
                better_p = math.exp(-window * max(0.0, -difference - effect) ** 2 / 2)
                method_name = "paired_brier_conditional_hoeffding_azuma_alpha_spending"
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
            report_item = {
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
                "method": method_name,
                "validation_state": state["validation_state"],
                "last_validated_target": state.get("last_validated_target"),
                "validation_revoked_at": state.get("validation_revoked_at"),
                "profitability_validated": False,
            }
            if method_cfg == "empirical_bernstein":
                report_item["empirical_hac_variance"] = hac_var
                report_item["predictable_scale_bound"] = c_max
            reports.append(report_item)
            if model["status"] == "disabled":
                break
    return reports


# ---------------------------------------------------------------------------
# Mathematical Audit: Required Sample Size & Minimum Detectable Effect (MDE)
# ---------------------------------------------------------------------------


def required_sample_size_hoeffding(
    delta_brier: float,
    min_effect: float = 0.001,
    alpha_spent: float = 0.00625,
) -> int:
    """Analytical sample size n required by Hoeffding-Azuma exp(-n*(delta-eps)^2 / 2) <= alpha."""
    excess = max(1e-9, abs(delta_brier) - min_effect)
    return int(math.ceil(2.0 * math.log(1.0 / alpha_spent) / (excess * excess)))


def required_sample_size_empirical_bernstein(
    delta_brier: float,
    empirical_variance: float = 0.004,
    scale_bound: float = 0.25,
    min_effect: float = 0.001,
    alpha_spent: float = 0.00625,
) -> int:
    """Analytical sample size n required by Predictable Empirical Bernstein."""
    excess = max(1e-9, abs(delta_brier) - min_effect)
    numerator = 2.0 * max(1e-6, empirical_variance) + (4.0 / 3.0) * scale_bound * excess
    return int(math.ceil(numerator * math.log(1.0 / alpha_spent) / (excess * excess)))


def minimum_detectable_effect_hoeffding(
    n: int,
    min_effect: float = 0.001,
    alpha_spent: float = 0.00625,
) -> float:
    return float(min_effect + math.sqrt(2.0 * math.log(1.0 / alpha_spent) / max(1, n)))


def minimum_detectable_effect_empirical_bernstein(
    n: int,
    empirical_variance: float = 0.004,
    scale_bound: float = 0.25,
    min_effect: float = 0.001,
    alpha_spent: float = 0.00625,
) -> float:
    L = math.log(1.0 / alpha_spent)
    # Solve n * excess^2 - (4/3)*c*L*excess - 2*v*L = 0
    b = (4.0 / 3.0) * scale_bound * L
    c_term = 2.0 * max(1e-6, empirical_variance) * L
    excess = (b + math.sqrt(b * b + 4.0 * max(1, n) * c_term)) / (2.0 * max(1, n))
    return float(min_effect + excess)


def audit_sequential_evaluator_math() -> dict[str, Any]:
    """Generate an analytical comparison table between Hoeffding-Azuma and Empirical Bernstein."""
    alpha_look1 = 0.05 / (2 * 2 * 1 * 2)  # alpha=0.05, M=2 competitors, look=1 -> 0.00625
    comparisons = []
    for delta in (0.003, 0.005, 0.010, 0.020, 0.050):
        n_hoeff = required_sample_size_hoeffding(delta, min_effect=0.001, alpha_spent=alpha_look1)
        n_eb = required_sample_size_empirical_bernstein(
            delta,
            empirical_variance=0.0035,
            scale_bound=0.20,
            min_effect=0.001,
            alpha_spent=alpha_look1,
        )
        comparisons.append(
            {
                "true_brier_gain": delta,
                "excess_over_min_effect": round(delta - 0.001, 5),
                "required_n_hoeffding_azuma": n_hoeff,
                "required_n_empirical_bernstein": n_eb,
                "sample_efficiency_ratio": round(n_hoeff / max(1, n_eb), 2),
            }
        )
    window_audit = []
    for n in (50, 100, 250, 500, 1000, 2500):
        window_audit.append(
            {
                "window_n": n,
                "mde_hoeffding_azuma": round(
                    minimum_detectable_effect_hoeffding(n, 0.001, alpha_look1), 5
                ),
                "mde_empirical_bernstein": round(
                    minimum_detectable_effect_empirical_bernstein(
                        n, 0.0035, 0.20, 0.001, alpha_look1
                    ),
                    5,
                ),
            }
        )
    return {
        "alpha": 0.05,
        "min_effect": 0.001,
        "alpha_spent_look_1_two_competitors": alpha_look1,
        "diagnosis": (
            "Hoeffding-Azuma asume el peor rango teórico d_t in [-1, 1] (varianza sub-gaussiana 1.0), "
            "mientras que en probabilidades calibradas cercanas a 0.50 el rango condicional predecible es "
            "2*|p_c - p_b| <= 0.25 y la varianza empírica Var(d_t) ~ 0.002-0.006. Como consecuencia, "
            "en bloques cortos (N=50..250) Hoeffding-Azuma tiene MDE > 0.20 (imposible de cruzar salvo "
            "colapso extremo) y requiere cientos de miles de observaciones para bordes reales de 0.003-0.005."
        ),
        "sample_size_by_effect": comparisons,
        "mde_by_window_size": window_audit,
    }


def run_evaluator_simulation_study(
    n_trials: int = 200,
    n_steps: int = 600,
    eval_window: int = 100,
    min_samples: int = 100,
    patience: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run controlled Monte Carlo benchmark across 6 mandatory statistical scenarios."""
    rng = np.random.default_rng(seed)
    scenarios = (
        "no_edge",
        "small_edge",
        "moderate_edge",
        "regime_shift",
        "serial_dependence",
        "miscalibrated_overconfident",
    )
    results: dict[str, dict[str, Any]] = {}

    for scenario in scenarios:
        for method in ("hoeffding_azuma", "empirical_bernstein"):
            promotions = 0
            disablings = 0
            revocations = 0
            superior_blocks = 0
            inferior_blocks = 0
            first_decision_samples: list[int] = []

            for _ in range(n_trials):
                # Generate synthetic sequence of outcomes and predictions
                if scenario == "serial_dependence":
                    # AR(1) latent process with zero candidate edge over baseline
                    y_list = []
                    state_z = 0.0
                    for _t in range(n_steps):
                        state_z = 0.35 * state_z + float(rng.normal(0.0, 0.9))
                        p_true = 1.0 / (1.0 + math.exp(-0.5 * state_z))
                        y_list.append(int(rng.random() < p_true))
                    labels = np.array(y_list, dtype=int)
                    p_base = np.full(n_steps, 0.50, dtype=float)
                    # Candidate adds pure noise around 0.50
                    p_cand = np.clip(0.50 + rng.normal(0.0, 0.04, size=n_steps), 0.05, 0.95)
                else:
                    # Latent true probability p_true_t driven by unit-variance factor z_t
                    z_true = rng.normal(0.0, 1.0, size=n_steps)
                    z_orth = rng.normal(0.0, 1.0, size=n_steps)
                    p_true = 1.0 / (1.0 + np.exp(-1.1 * z_true))
                    labels = (rng.random(size=n_steps) < p_true).astype(int)
                    p_base = np.full(n_steps, 0.50, dtype=float)

                    if scenario == "no_edge":
                        # Tiny well-behaved fluctuations around 0.50 with zero correlation to z_true
                        p_cand = np.clip(0.50 + 0.01 * z_orth, 0.45, 0.55)
                    elif scenario == "small_edge":
                        # Calibrated forecaster with correlation rho = 0.40 to latent factor (Delta Brier ~ -0.0045)
                        z_hat = 0.40 * z_true + math.sqrt(1.0 - 0.40**2) * z_orth
                        p_cand = 1.0 / (1.0 + np.exp(-0.44 * z_hat))
                    elif scenario == "moderate_edge":
                        # Calibrated forecaster with correlation rho = 0.72 to latent factor (Delta Brier ~ -0.015)
                        z_hat = 0.72 * z_true + math.sqrt(1.0 - 0.72**2) * z_orth
                        p_cand = 1.0 / (1.0 + np.exp(-0.79 * z_hat))
                    elif scenario == "regime_shift":
                        # First half strong edge (rho = 0.75), second half regime inversion
                        half = n_steps // 2
                        p_first = 1.0 / (1.0 + np.exp(-0.82 * z_true[:half]))
                        p_second = 1.0 / (1.0 + np.exp(0.95 * z_true[half:]))
                        p_cand = np.concatenate([p_first, p_second])
                    elif scenario == "miscalibrated_overconfident":
                        # Overconfident uncalibrated model emitting extreme probabilities on noise
                        p_cand = np.clip(1.0 / (1.0 + np.exp(-1.8 * z_orth)), 0.05, 0.95)
                    else:
                        p_cand = p_base.copy()

                # Evaluate blocks sequentially
                diffs = (p_cand - labels) ** 2 - (p_base - labels) ** 2
                pred_ranges = 2.0 * np.abs(p_cand - p_base)
                look = 0
                worse_streak = 0
                better_streak = 0
                val_state = "unvalidated"
                decided_at: int | None = None

                for start_idx in range(0, n_steps - eval_window + 1, eval_window):
                    end_idx = start_idx + eval_window
                    if end_idx < min_samples:
                        continue
                    look += 1
                    b_diffs = diffs[start_idx:end_idx].tolist()
                    b_ranges = pred_ranges[start_idx:end_idx].tolist()
                    mean_d = sum(b_diffs) / eval_window
                    threshold = 0.05 / (2 * 1 * look * (look + 1))
                    if method == "empirical_bernstein":
                        cum_diffs = diffs[:end_idx].tolist()
                        cum_ranges = pred_ranges[:end_idx].tolist()
                        n_cum = len(cum_diffs)
                        cum_mean = sum(cum_diffs) / n_cum
                        hac_v_cum, c_max_cum = _hac_variance(cum_diffs, cum_ranges)
                        hac_v_blk, c_max_blk = _hac_variance(b_diffs, b_ranges)
                        eff_margin = min(0.001, 0.15 * hac_v_cum)
                        wp_cum = empirical_bernstein_one_sided_pvalue(
                            n_cum, max(0.0, cum_mean - eff_margin), hac_v_cum, c_max_cum
                        )
                        wp_blk = empirical_bernstein_one_sided_pvalue(
                            eval_window, max(0.0, mean_d - eff_margin), hac_v_blk, c_max_blk
                        )
                        wp = min(wp_cum, wp_blk)
                        bp = empirical_bernstein_one_sided_pvalue(
                            n_cum, max(0.0, -cum_mean - eff_margin), hac_v_cum, c_max_cum
                        )
                        if mean_d >= 0.0:
                            bp = max(bp, threshold * 2.0)
                    else:
                        wp = math.exp(-eval_window * max(0.0, mean_d - 0.001) ** 2 / 2.0)
                        bp = math.exp(-eval_window * max(0.0, -mean_d - 0.001) ** 2 / 2.0)
                    worse = wp <= threshold
                    better = bp <= threshold
                    if worse:
                        inferior_blocks += 1
                    if better:
                        superior_blocks += 1
                    worse_streak = worse_streak + 1 if worse else 0
                    better_streak = better_streak + 1 if better else 0
                    if worse and val_state == "validated_predictive":
                        val_state = "unvalidated"
                        revocations += 1
                    if worse_streak >= patience:
                        disablings += 1
                        if decided_at is None:
                            decided_at = end_idx
                        break
                    if better_streak >= patience and val_state != "validated_predictive":
                        val_state = "validated_predictive"
                        promotions += 1
                        if decided_at is None:
                            decided_at = end_idx

                if decided_at is not None:
                    first_decision_samples.append(decided_at)

            key = f"{scenario}__{method}"
            results[key] = {
                "scenario": scenario,
                "method": method,
                "trials": n_trials,
                "steps_per_trial": n_steps,
                "eval_window": eval_window,
                "patience": patience,
                "promotion_rate": round(promotions / n_trials, 4),
                "disabling_rate": round(disablings / n_trials, 4),
                "revocation_rate": round(revocations / n_trials, 4),
                "superior_block_rate": round(
                    superior_blocks / max(1, n_trials * (n_steps // eval_window)), 4
                ),
                "inferior_block_rate": round(
                    inferior_blocks / max(1, n_trials * (n_steps // eval_window)), 4
                ),
                "mean_samples_to_decision": (
                    round(float(np.mean(first_decision_samples)), 1)
                    if first_decision_samples
                    else None
                ),
            }

    return {
        "analytical_audit": audit_sequential_evaluator_math(),
        "simulation_results": results,
    }


# ---------------------------------------------------------------------------
# Multi-Target Evaluation & Champion-Challenger Selection
# ---------------------------------------------------------------------------


def _roc_auc_binary(y_true: list[int], y_prob: list[float]) -> float | None:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None
    pos = [p for y, p in zip(y_true, y_prob, strict=True) if y == 1]
    neg = [p for y, p in zip(y_true, y_prob, strict=True) if y == 0]
    if not pos or not neg:
        return None
    # Mann-Whitney U statistic
    arr_p = np.array(y_prob, dtype=float)
    arr_y = np.array(y_true, dtype=int)
    order = np.argsort(arr_p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr_p) + 1, dtype=float)
    # Handle ties with average rank
    _, inv, counts = np.unique(arr_p, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for val_idx in np.where(counts > 1)[0]:
            mask = inv == val_idx
            ranks[mask] = float(np.mean(ranks[mask]))
    n_pos = float(np.sum(arr_y == 1))
    n_neg = float(np.sum(arr_y == 0))
    u_stat = float(np.sum(ranks[arr_y == 1])) - n_pos * (n_pos + 1.0) / 2.0
    return float(u_stat / (n_pos * n_neg))


def _ece_10_bins(y_true: list[int], y_prob: list[float]) -> float | None:
    n = len(y_true)
    if n == 0:
        return None
    arr_y = np.array(y_true, dtype=float)
    arr_p = np.array(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        if i < 9:
            mask = (arr_p >= bins[i]) & (arr_p < bins[i + 1])
        else:
            mask = (arr_p >= bins[i]) & (arr_p <= bins[i + 1])
        count = int(np.sum(mask))
        if count > 0:
            avg_p = float(np.mean(arr_p[mask]))
            avg_y = float(np.mean(arr_y[mask]))
            ece += (count / n) * abs(avg_p - avg_y)
    return float(ece)


def compute_economic_metrics(
    probabilities: list[float],
    realized_returns: list[float],
    *,
    threshold: float = 0.55,
    cost_bps: float = 10.0,
    bars_per_year: float = 8760.0,
) -> dict[str, Any]:
    """Compute realistic paper-trading economic metrics with transaction costs and turnover."""
    n = len(probabilities)
    if n == 0 or len(realized_returns) != n:
        return {
            "trades": 0,
            "exposure_rate": 0.0,
            "turnover": 0.0,
            "gross_return_pct": 0.0,
            "net_return_pct": 0.0,
            "sharpe_net": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_active": None,
            "profit_factor": None,
        }
    cost_rate = float(cost_bps) / 10000.0
    positions: list[int] = []
    for p in probabilities:
        if p >= threshold:
            positions.append(1)
        elif p <= 1.0 - threshold:
            positions.append(-1)
        else:
            positions.append(0)

    gross_pnl: list[float] = []
    net_pnl: list[float] = []
    prev_pos = 0
    turnover = 0.0
    trades = 0
    active_wins = 0
    active_bars = 0
    gross_gains = 0.0
    gross_losses = 0.0

    for pos, ret in zip(positions, realized_returns, strict=True):
        delta_pos = abs(pos - prev_pos)
        if delta_pos > 0:
            trades += 1
        turnover += float(delta_pos)
        cost = float(delta_pos) * cost_rate
        g_bar = float(pos) * float(ret)
        n_bar = g_bar - cost
        gross_pnl.append(g_bar)
        net_pnl.append(n_bar)
        if pos != 0:
            active_bars += 1
            if n_bar > 0:
                active_wins += 1
                gross_gains += n_bar
            elif n_bar < 0:
                gross_losses += abs(n_bar)
        prev_pos = pos

    gross_arr = np.array(gross_pnl, dtype=float)
    net_arr = np.array(net_pnl, dtype=float)
    gross_cum = float(np.prod(1.0 + np.clip(gross_arr, -0.95, 5.0)) - 1.0)
    equity_curve = np.cumprod(1.0 + np.clip(net_arr, -0.95, 5.0))
    net_cum = float(equity_curve[-1] - 1.0)
    peaks = np.maximum.accumulate(np.insert(equity_curve, 0, 1.0))[1:]
    drawdowns = (equity_curve - peaks) / np.maximum(peaks, 1e-8)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) else 0.0

    net_std = float(np.std(net_arr))
    sharpe = (
        float((np.mean(net_arr) / net_std) * math.sqrt(bars_per_year))
        if net_std > 1e-10 and active_bars > 0
        else 0.0
    )
    pf = float(gross_gains / gross_losses) if gross_losses > 1e-10 else None
    return {
        "trades": int(trades),
        "active_bars": int(active_bars),
        "exposure_rate": round(active_bars / n, 4),
        "turnover": round(turnover, 2),
        "gross_return_pct": round(gross_cum * 100.0, 4),
        "net_return_pct": round(net_cum * 100.0, 4),
        "sharpe_net": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "win_rate_active": round(active_wins / active_bars, 4) if active_bars > 0 else None,
        "profit_factor": round(pf, 4) if pf is not None else None,
    }


def evaluate_target_specific_records(
    records: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    horizon: int = 1,
    alpha: float = 0.05,
    min_effect: float = 0.0005,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Evaluate all models grouped by target_type on identical confirmed observations and select Champions."""
    bars_per_year = 8760.0 if interval == "1h" else (365.0 if interval == "1d" else 2190.0)
    by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    for rec in records:
        mid = rec.get("model_id")
        if mid in by_model and rec.get("status") == "evaluated":
            by_model[mid].append(rec)

    for mid in by_model:
        by_model[mid].sort(key=lambda r: str(r["reference_time"]))

    direction_metrics: list[dict[str, Any]] = []
    return_metrics: list[dict[str, Any]] = []
    volatility_metrics: list[dict[str, Any]] = []
    quantile_metrics: list[dict[str, Any]] = []

    # Build lookup by reference_time per model for paired tests
    by_model_time: dict[str, dict[str, dict[str, Any]]] = {
        mid: {str(r["reference_time"]): r for r in rows} for mid, rows in by_model.items()
    }

    for mid, model in models.items():
        rows = by_model.get(mid, [])
        n = len(rows)
        spec = MODEL_SPECS.get(mid)
        target_type = model.get("target_type", spec.target_type if spec else "direction")
        if n == 0:
            continue

        probs = [float(r["probability"]) for r in rows]
        labels = [int(r["label"]) for r in rows]
        returns = [
            float(
                r.get(
                    "realized_return",
                    math.log(float(r["target_close"]) / float(r["reference_close"]))
                    if r.get("target_close") and r.get("reference_close")
                    else (0.005 if int(r["label"]) == 1 else -0.005),
                )
            )
            for r in rows
        ]
        rvs = [
            float(r.get("realized_variance", max(1e-8, ret * ret)))
            for r, ret in zip(rows, returns, strict=True)
        ]

        # 1. Direction evaluation (for direction models + directional projections of Kalman/Quantile)
        if target_type in ("direction", "expected_return", "quantiles") and model["kind"] not in (
            "return_baseline",
            "volatility_baseline",
            "quantile_baseline",
        ):
            brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / n
            ll = (
                -sum(
                    y * math.log(max(1e-6, p)) + (1 - y) * math.log(max(1e-6, 1.0 - p))
                    for p, y in zip(probs, labels, strict=True)
                )
                / n
            )
            acc = sum(int(p >= 0.5) == y for p, y in zip(probs, labels, strict=True)) / n
            auc = _roc_auc_binary(labels, probs)
            ece = _ece_10_bins(labels, probs)
            econ = compute_economic_metrics(
                probs, returns, threshold=0.55, cost_bps=cost_bps, bars_per_year=bars_per_year
            )

            # Paired Brier difference against 'baseline'
            base_map = by_model_time.get("baseline", {})
            common_ts = sorted(set(by_model_time[mid]) & set(base_map))
            b_diffs = [
                (float(by_model_time[mid][t]["probability"]) - int(base_map[t]["label"])) ** 2
                - (float(base_map[t]["probability"]) - int(base_map[t]["label"])) ** 2
                for t in common_ts
            ]
            b_ranges = [
                2.0
                * abs(
                    float(by_model_time[mid][t]["probability"])
                    - float(base_map[t]["probability"])
                )
                for t in common_ts
            ]
            mean_d = sum(b_diffs) / len(b_diffs) if b_diffs else 0.0
            hac_v, c_max = _hac_variance(b_diffs, b_ranges) if b_diffs else (0.01, 2.0)
            p_eb_sup = (
                empirical_bernstein_one_sided_pvalue(
                    len(b_diffs), max(0.0, -mean_d - min_effect), hac_v, c_max
                )
                if b_diffs and mid != "baseline"
                else 1.0
            )
            p_hoeff_sup = (
                math.exp(-len(b_diffs) * max(0.0, -mean_d - min_effect) ** 2 / 2.0)
                if b_diffs and mid != "baseline"
                else 1.0
            )
            direction_metrics.append(
                {
                    "model_id": mid,
                    "kind": model["kind"],
                    "target_type": target_type,
                    "status": model["status"],
                    "n": n,
                    "brier": round(brier, 6),
                    "log_loss": round(ll, 6),
                    "accuracy": round(acc, 5),
                    "roc_auc": round(auc, 5) if auc is not None else None,
                    "ece_10": round(ece, 5) if ece is not None else None,
                    "mean_brier_delta_vs_baseline": round(mean_d, 6),
                    "p_empirical_bernstein_superior": round(p_eb_sup, 6),
                    "p_hoeffding_superior": round(p_hoeff_sup, 6),
                    "economic": econ,
                }
            )

        # 2. Expected Return evaluation
        if target_type == "expected_return":
            pred_rets = [float(r.get("expected_return", 0.0)) for r in rows]
            pred_stds = [max(1e-5, float(r.get("predictive_std", 0.01))) for r in rows]
            mse = sum((pr - r) ** 2 for pr, r in zip(pred_rets, returns, strict=True)) / n
            mae = sum(abs(pr - r) for pr, r in zip(pred_rets, returns, strict=True)) / n
            nll = (
                sum(
                    0.5 * math.log(2.0 * math.pi * (s * s)) + ((r - pr) ** 2) / (2.0 * s * s)
                    for pr, s, r in zip(pred_rets, pred_stds, returns, strict=True)
                )
                / n
            )
            z_scores = [(r - pr) / s for pr, s, r in zip(pred_rets, pred_stds, returns, strict=True)]
            z_var = float(np.var(z_scores)) if len(z_scores) > 1 else 1.0
            ic = (
                float(np.corrcoef(pred_rets, returns)[0, 1])
                if len(pred_rets) > 2 and np.std(pred_rets) > 1e-12 and np.std(returns) > 1e-12
                else 0.0
            )
            base_ret_map = by_model_time.get("return_naive_baseline", {})
            common_ts = sorted(set(by_model_time[mid]) & set(base_ret_map))
            mse_diffs = []
            for t in common_ts:
                r_true = float(by_model_time[mid][t].get("realized_return", 0.0))
                pr_m = float(by_model_time[mid][t].get("expected_return", 0.0))
                pr_b = float(base_ret_map[t].get("expected_return", 0.0))
                mse_diffs.append((pr_m - r_true) ** 2 - (pr_b - r_true) ** 2)
            mean_mse_delta = sum(mse_diffs) / len(mse_diffs) if mse_diffs else 0.0
            hac_v, _ = _hac_variance(mse_diffs) if mse_diffs else (1e-6, 0.1)
            p_sup = (
                empirical_bernstein_one_sided_pvalue(
                    len(mse_diffs), max(0.0, -mean_mse_delta), hac_v, 0.05
                )
                if mse_diffs and mid != "return_naive_baseline"
                else 1.0
            )
            return_metrics.append(
                {
                    "model_id": mid,
                    "kind": model["kind"],
                    "status": model["status"],
                    "n": n,
                    "return_mse": round(mse, 8),
                    "return_rmse": round(math.sqrt(mse), 6),
                    "return_mae": round(mae, 6),
                    "gaussian_nll": round(nll, 5),
                    "pearson_ic": round(ic, 5),
                    "standardized_innovation_var": round(z_var, 4),
                    "mean_mse_delta_vs_baseline": round(mean_mse_delta, 8),
                    "p_empirical_bernstein_superior": round(p_sup, 6),
                }
            )

        # 3. Volatility evaluation
        if target_type == "volatility":
            pred_vars = [max(1e-8, float(r.get("pred_variance", 1e-4))) for r in rows]
            pred_vols = [math.sqrt(pv) for pv in pred_vars]
            true_vols = [math.sqrt(rv) for rv in rvs]
            qlike = (
                sum(
                    math.log(pv) + (rv / pv)
                    for pv, rv in zip(pred_vars, rvs, strict=True)
                )
                / n
            )
            vol_mae = sum(abs(pv - tv) for pv, tv in zip(pred_vols, true_vols, strict=True)) / n
            vol_mse = sum((pv - tv) ** 2 for pv, tv in zip(pred_vols, true_vols, strict=True)) / n

            base_vol_map = by_model_time.get("volatility_naive_baseline", {})
            common_ts = sorted(set(by_model_time[mid]) & set(base_vol_map))
            qlike_diffs = []
            for t in common_ts:
                rv_t = max(1e-8, float(by_model_time[mid][t].get("realized_variance", 1e-4)))
                pv_m = max(1e-8, float(by_model_time[mid][t].get("pred_variance", 1e-4)))
                pv_b = max(1e-8, float(base_vol_map[t].get("pred_variance", 1e-4)))
                ql_m = math.log(pv_m) + rv_t / pv_m
                ql_b = math.log(pv_b) + rv_t / pv_b
                qlike_diffs.append(ql_m - ql_b)
            mean_ql_delta = sum(qlike_diffs) / len(qlike_diffs) if qlike_diffs else 0.0
            hac_v, _ = _hac_variance(qlike_diffs) if qlike_diffs else (0.1, 2.0)
            p_sup = (
                empirical_bernstein_one_sided_pvalue(
                    len(qlike_diffs), max(0.0, -mean_ql_delta - 0.005), hac_v, 1.5
                )
                if qlike_diffs and mid != "volatility_naive_baseline"
                else 1.0
            )
            volatility_metrics.append(
                {
                    "model_id": mid,
                    "kind": model["kind"],
                    "status": model["status"],
                    "n": n,
                    "qlike": round(qlike, 5),
                    "volatility_mae": round(vol_mae, 6),
                    "volatility_rmse": round(math.sqrt(vol_mse), 6),
                    "mean_qlike_delta_vs_baseline": round(mean_ql_delta, 5),
                    "p_empirical_bernstein_superior": round(p_sup, 6),
                }
            )

        # 4. Quantiles evaluation
        if target_type == "quantiles":
            q10s = [float(r.get("q10", -0.01)) for r in rows]
            q50s = [float(r.get("q50", 0.0)) for r in rows]
            q90s = [float(r.get("q90", 0.01)) for r in rows]
            pinball_sum = 0.0
            covered = 0
            winkler_sum = 0.0
            post_crossings = 0
            for q10, q50, q90, r_true in zip(q10s, q50s, q90s, returns, strict=True):
                if q10 > q50 or q50 > q90:
                    post_crossings += 1
                for tau, q_val in ((0.10, q10), (0.50, q50), (0.90, q90)):
                    u = r_true - q_val
                    pinball_sum += u * (tau - (1.0 if u < 0.0 else 0.0))
                if q10 <= r_true <= q90:
                    covered += 1
                width = max(1e-6, q90 - q10)
                w_score = width
                if r_true < q10:
                    w_score += (2.0 / 0.20) * (q10 - r_true)
                elif r_true > q90:
                    w_score += (2.0 / 0.20) * (r_true - q90)
                winkler_sum += w_score

            mean_pinball = pinball_sum / (3.0 * n)
            coverage_80 = covered / n
            mean_width = sum(q90 - q10 for q10, q90 in zip(q10s, q90s, strict=True)) / n
            mean_winkler = winkler_sum / n

            base_q_map = by_model_time.get("quantile_historical_baseline", {})
            common_ts = sorted(set(by_model_time[mid]) & set(base_q_map))
            pb_diffs = []
            for t in common_ts:
                r_t = float(by_model_time[mid][t].get("realized_return", 0.0))
                rec_m = by_model_time[mid][t]
                rec_b = base_q_map[t]
                pb_m = sum(
                    (r_t - float(rec_m.get(k, 0.0)))
                    * (tau - (1.0 if r_t < float(rec_m.get(k, 0.0)) else 0.0))
                    for tau, k in ((0.10, "q10"), (0.50, "q50"), (0.90, "q90"))
                ) / 3.0
                pb_b = sum(
                    (r_t - float(rec_b.get(k, 0.0)))
                    * (tau - (1.0 if r_t < float(rec_b.get(k, 0.0)) else 0.0))
                    for tau, k in ((0.10, "q10"), (0.50, "q50"), (0.90, "q90"))
                ) / 3.0
                pb_diffs.append(pb_m - pb_b)
            mean_pb_delta = sum(pb_diffs) / len(pb_diffs) if pb_diffs else 0.0
            hac_v, _ = _hac_variance(pb_diffs) if pb_diffs else (1e-5, 0.05)
            p_sup = (
                empirical_bernstein_one_sided_pvalue(
                    len(pb_diffs), max(0.0, -mean_pb_delta), hac_v, 0.05
                )
                if pb_diffs and mid != "quantile_historical_baseline"
                else 1.0
            )
            quantile_metrics.append(
                {
                    "model_id": mid,
                    "kind": model["kind"],
                    "status": model["status"],
                    "n": n,
                    "mean_pinball_loss": round(mean_pinball, 6),
                    "coverage_80": round(coverage_80, 4),
                    "coverage_error_vs_80": round(abs(coverage_80 - 0.80), 4),
                    "mean_interval_width": round(mean_width, 6),
                    "winkler_score_80": round(mean_winkler, 6),
                    "post_rearrangement_crossing_rate": round(post_crossings / n, 6),
                    "raw_crossings_before_rearrangement": int(
                        model["params"].get("raw_crossings", 0)
                    ),
                    "mean_pinball_delta_vs_baseline": round(mean_pb_delta, 6),
                    "p_empirical_bernstein_superior": round(p_sup, 6),
                }
            )

    champions = select_champions_by_target(
        symbol=symbol,
        interval=interval,
        horizon=horizon,
        direction_metrics=direction_metrics,
        return_metrics=return_metrics,
        volatility_metrics=volatility_metrics,
        quantile_metrics=quantile_metrics,
        alpha=alpha,
    )
    return {
        "symbol": symbol,
        "interval": interval,
        "horizon": horizon,
        "targets": {
            "direction": sorted(direction_metrics, key=lambda r: r["brier"]),
            "expected_return": sorted(return_metrics, key=lambda r: r["return_mse"]),
            "volatility": sorted(volatility_metrics, key=lambda r: r["qlike"]),
            "quantiles": sorted(quantile_metrics, key=lambda r: r["mean_pinball_loss"]),
        },
        "champions": champions,
    }


def select_champions_by_target(
    *,
    symbol: str,
    interval: str,
    horizon: int,
    direction_metrics: list[dict[str, Any]],
    return_metrics: list[dict[str, Any]],
    volatility_metrics: list[dict[str, Any]],
    quantile_metrics: list[dict[str, Any]],
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Select Champion vs Baseline per (symbol, interval, horizon, target_type).

    Explicitly separates:
    1. descriptive_leader (lowest empirical loss on confirmed history),
    2. statistically_validated_champion (passes sequential Empirical Bernstein test vs baseline),
    3. economically_validated (positive net return & Sharpe > 0 after realistic friction).
    """
    champions: dict[str, dict[str, Any]] = {}

    def _pick(
        target_type: str,
        baseline_id: str,
        rows: list[dict[str, Any]],
        loss_key: str,
        delta_key: str,
    ) -> dict[str, Any]:
        active_rows = [r for r in rows if r.get("status") == "active"]
        if not active_rows:
            return {
                "symbol": symbol,
                "interval": interval,
                "horizon": horizon,
                "target_type": target_type,
                "baseline_id": baseline_id,
                "active_champion": baseline_id,
                "descriptive_leader": None,
                "statistically_validated": False,
                "profitability_validated": False,
                "reason": "sin_modelos_activos",
            }
        ordered = sorted(active_rows, key=lambda r: float(r[loss_key]))
        leader = ordered[0]
        # Adjust threshold across challengers in this target family
        n_challengers = max(1, sum(1 for r in active_rows if r["model_id"] != baseline_id))
        adj_alpha = alpha / n_challengers

        validated_challengers = [
            r
            for r in ordered
            if r["model_id"] != baseline_id
            and float(r.get(delta_key, 0.0)) < 0.0
            and float(r.get("p_empirical_bernstein_superior", 1.0)) <= adj_alpha
        ]
        if validated_challengers:
            champ = validated_challengers[0]
            stat_valid = True
            reason = (
                f"Supera estadísticamente a {baseline_id} en {loss_key} "
                f"(delta={champ[delta_key]}, p_EB={champ['p_empirical_bernstein_superior']} <= {adj_alpha:.4f})"
            )
        else:
            champ_id = baseline_id if any(r["model_id"] == baseline_id for r in active_rows) else leader["model_id"]
            champ = next((r for r in active_rows if r["model_id"] == champ_id), leader)
            stat_valid = False
            reason = (
                f"Líder descriptivo: {leader['model_id']} ({loss_key}={leader[loss_key]}), "
                f"pero ningún retador supera la barrera secuencial (alpha_adj={adj_alpha:.4f}); "
                f"se retiene control {champ['model_id']} como Champion conservador."
            )
        econ = champ.get("economic", {})
        prof_valid = bool(
            stat_valid
            and econ.get("trades", 0) >= 5
            and float(econ.get("net_return_pct", 0.0)) > 0.0
            and float(econ.get("sharpe_net", 0.0)) > 0.25
        )
        return {
            "symbol": symbol,
            "interval": interval,
            "horizon": horizon,
            "target_type": target_type,
            "baseline_id": baseline_id,
            "active_champion": champ["model_id"],
            "descriptive_leader": leader["model_id"],
            "descriptive_leader_loss": leader[loss_key],
            "loss_metric": loss_key,
            "statistically_validated": stat_valid,
            "profitability_validated": prof_valid,
            "reason": reason,
        }

    champions["direction"] = _pick(
        "direction", "baseline", direction_metrics, "brier", "mean_brier_delta_vs_baseline"
    )
    champions["expected_return"] = _pick(
        "expected_return",
        "return_naive_baseline",
        return_metrics,
        "return_mse",
        "mean_mse_delta_vs_baseline",
    )
    champions["volatility"] = _pick(
        "volatility",
        "volatility_naive_baseline",
        volatility_metrics,
        "qlike",
        "mean_qlike_delta_vs_baseline",
    )
    champions["quantiles"] = _pick(
        "quantiles",
        "quantile_historical_baseline",
        quantile_metrics,
        "mean_pinball_loss",
        "mean_pinball_delta_vs_baseline",
    )
    return champions
