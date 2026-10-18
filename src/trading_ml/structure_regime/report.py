from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ml.advanced_research.statistics import paired_predictive
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import moving_block_indices
from trading_ml.structure_regime.execution import economic_summary

BASELINES = ["constant_05", "train_prevalence", "logistic_l2"]
REGIME_RECIPES = ["regime_rules", "specialists_rules", "regime_cluster", "specialists_cluster"]
PREDICTIVE_RECIPES = ["original", "smc_control", "structure", *REGIME_RECIPES]


def simultaneous_economics(returns, groups, pairs, config):
    """Paired block resampling of realized daily policy returns, conditional on fits.

    Does not retrain models or recreate execution on a scrambled price path.
    Intervals concern historical policy performance, not prospective execution.
    """
    result, intervals, standardized = {}, [], []
    for block in config["block_lengths"]:
        indices = moving_block_indices(groups, config["samples"], block, config["seed"])
        points, draws = {}, {}
        for name, values in returns.items():
            log_returns = np.log1p(values)
            equity = np.exp(np.r_[0.0, np.cumsum(log_returns)])
            samples = np.cumsum(log_returns[indices], axis=1)
            peak = np.maximum.accumulate(np.maximum(samples, 0), axis=1)
            points[name] = {
                "net_return": float(equity[-1] - 1),
                "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
            }
            draws[name] = {
                "net_return": np.expm1(samples[:, -1]),
                "max_drawdown": np.expm1((samples - peak).min(axis=1)),
            }
        result[str(block)] = {}
        for a, b in pairs:
            row = {}
            for metric in ["net_return", "max_drawdown"]:
                values = draws[a][metric] - draws[b][metric]
                estimate = points[a][metric] - points[b][metric]
                se = float(values.std(ddof=1))
                interval = {"estimate": estimate, "lower": None, "upper": None, "bootstrap_se": se}
                intervals.append(interval)
                standardized.append(
                    np.abs((values - estimate) / se) if se > 1e-12 else np.zeros(len(values))
                )
                row[metric] = interval
            result[str(block)][a + "__minus__" + b] = row
    critical = float(np.quantile(np.max(standardized, axis=0), config["confidence"]))
    for row in intervals:
        row["lower"] = row["estimate"] - critical * row["bootstrap_se"]
        row["upper"] = row["estimate"] + critical * row["bootstrap_se"]
    return {
        "blocks": result,
        "max_t_critical": critical,
        "simultaneous_confidence": config["confidence"],
        "limitations": "Conditional realized policy returns; no model refitting or path replay",
    }


def analyze(predictions, curves, trades, protocol):
    """All recipes must have identical timestamps, labels and fold assignments."""
    metrics, economic, inference, gates, by_regime = [], [], {}, {}, []
    r, a = protocol["research"], protocol["acceptance"]
    # 2 assets × 2 seeds × 2 metric families, plus alpha reserved for optional stage 3.
    boot = protocol["bootstrap"] | {
        "confidence": 1 - (1 - protocol["bootstrap"]["confidence"]) / (2 * len(r["seeds"]) * 2 * 2)
    }
    for (symbol, seed), group in predictions.groupby(["symbol", "seed"], sort=True):
        keys = ["fold", "reference_time", "target_time", "target"]
        if group.duplicated(keys + ["recipe"]).any():
            raise DataError("Predicciones repetidas")
        wide = group.pivot(index=keys, columns="recipe", values="probability").sort_index()
        if wide.isna().any().any():
            raise DataError("Recetas con distintos períodos de evaluación")
        probabilities = {name: wide[name].to_numpy() for name in wide}
        pairs = [
            (name, base)
            for name in wide
            if name not in ["original", *BASELINES]
            for base in ["original", *BASELINES]
        ]
        if "combined" in wide:
            pairs += [("combined", name) for name in ["structure", *REGIME_RECIPES]]
        predictive = paired_predictive(
            wide.index.get_level_values("target").to_numpy(dtype=int),
            probabilities,
            wide.index.get_level_values("fold").to_numpy(),
            pairs,
            boot,
        )
        for (fold, recipe), part in group.groupby(["fold", "recipe"]):
            metrics.append(
                {
                    "symbol": symbol,
                    "seed": int(seed),
                    "fold": int(fold),
                    "recipe": recipe,
                    **classification_metrics(part.target, part.probability),
                }
            )
        for (state, recipe), part in group.groupby(["sr_state", "recipe"]):
            by_regime.append(
                {
                    "symbol": symbol,
                    "seed": int(seed),
                    "state": state,
                    "recipe": recipe,
                    **classification_metrics(part.target, part.probability),
                }
            )
        local_curves = curves.loc[(curves.symbol == symbol) & (curves.seed == seed)]
        local_trades = trades.loc[(trades.symbol == symbol) & (trades.seed == seed)]
        policy_returns = {}
        for (scenario, recipe), curve in local_curves.groupby(["scenario", "recipe"]):
            curve = curve.sort_values(["fold", "step"])
            ledger = local_trades.loc[
                (local_trades.scenario == scenario) & (local_trades.recipe == recipe)
            ]
            # Include initial zero-return row in every fold, equally across policies.
            values = curve.groupby("fold", sort=False).equity.pct_change().fillna(0).to_numpy()
            eq = np.cumprod(np.r_[1.0, 1 + values])
            item = {
                "symbol": symbol,
                "seed": int(seed),
                "scenario": scenario,
                "recipe": recipe,
                **economic_summary(pd.DataFrame({"equity": eq}), ledger),
            }
            economic.append(item)
            if scenario == "base":
                policy_returns[recipe] = values
        origin = local_curves.loc[
            (local_curves.scenario == "base") & (local_curves.recipe == "original")
        ]
        origin = origin.sort_values(["fold", "step"])
        for recipe in policy_returns:
            comparison = local_curves.loc[
                (local_curves.scenario == "base") & (local_curves.recipe == recipe)
            ]
            comparison = comparison.sort_values(["fold", "step"])
            if not np.array_equal(
                comparison[["fold", "step", "timestamp"]].to_numpy(),
                origin[["fold", "step", "timestamp"]].to_numpy(),
            ):
                raise DataError("Calendarios económicos desalineados")
        econ_pairs = [(name, "original") for name in policy_returns if name != "original"]
        if "combined" in policy_returns:
            econ_pairs += [("combined", name) for name in ["structure", *REGIME_RECIPES]]
        economics = simultaneous_economics(policy_returns, origin.fold.to_numpy(), econ_pairs, boot)
        key = f"{symbol}_seed{seed}"
        inference[key] = {"predictive": predictive, "economic": economics}
        gates[key] = {}
        for recipe in ["structure", *REGIME_RECIPES, *(["combined"] if "combined" in wide else [])]:
            point, baseline = predictive["metrics"][recipe], predictive["metrics"]["original"]
            loss_gain = baseline["log_loss"] - point["log_loss"]
            auc_gain = point["roc_auc"] - baseline["roc_auc"]
            fold_rows = [m for m in metrics if m["symbol"] == symbol and m["seed"] == seed]
            original = {m["fold"]: m for m in fold_rows if m["recipe"] == "original"}
            improved = [
                m["log_loss"] < original[m["fold"]]["log_loss"]
                and m["roc_auc"] > original[m["fold"]]["roc_auc"]
                for m in fold_rows
                if m["recipe"] == recipe
            ]
            e = {
                m["scenario"]: m
                for m in economic
                if m["symbol"] == symbol and m["seed"] == seed and m["recipe"] == recipe
            }
            ref = next(
                m
                for m in economic
                if m["symbol"] == symbol
                and m["seed"] == seed
                and m["recipe"] == "original"
                and m["scenario"] == "base"
            )
            checks = {
                "minimum_loss_gain": loss_gain >= a["minimum_log_loss_gain"],
                "minimum_auc_gain": auc_gain >= a["minimum_auc_gain"],
                "fold_stability": float(np.mean(improved)) >= a["required_fold_fraction"],
                "positive_net_return": e["base"]["net_return"] > 0,
                "minimum_trades": e["base"]["trade_count"] >= a["minimum_trades"],
                "drawdown_not_worse": e["base"]["max_drawdown"] >= ref["max_drawdown"],
                "stress_positive": e["double_cost"]["net_return"] > 0,
            }
            for block in boot["block_lengths"]:
                delta = predictive["blocks"][str(block)][f"{recipe}__minus__original"]
                checks[f"loss_ci_{block}"] = (
                    delta["log_loss"]["upper"] is not None and delta["log_loss"]["upper"] < 0
                )
                checks[f"auc_ci_{block}"] = (
                    delta["roc_auc"]["lower"] is not None and delta["roc_auc"]["lower"] > 0
                )
                for name in BASELINES:
                    ci = predictive["blocks"][str(block)][f"{recipe}__minus__{name}"]["log_loss"]
                    checks[f"beats_{name}_{block}"] = ci["upper"] is not None and ci["upper"] < 0
                ci = economics["blocks"][str(block)][f"{recipe}__minus__original"]["net_return"]
                checks[f"net_return_ci_{block}"] = ci["lower"] > 0
            gates[key][recipe] = {
                "passed": all(checks.values()),
                "checks": checks,
                "loss_gain": loss_gain,
                "auc_gain": auc_gain,
                "folds_improved": int(sum(improved)),
                "failed": [k for k, v in checks.items() if not v],
            }
    qualified = [
        name
        for name in [
            "structure",
            *REGIME_RECIPES,
            *(["combined"] if "combined" in predictions.recipe.values else []),
        ]
        if all(local[name]["passed"] for local in gates.values())
    ]
    return {
        "fold_metrics": metrics,
        "economic_metrics": economic,
        "by_regime": by_regime,
        "inference": inference,
        "acceptance": gates,
        "qualified": qualified,
        "multiple_testing": {
            "family_confidence": boot["confidence"],
            "optional_combination_alpha_reserved": True,
            "historical_project_search_adjusted": False,
        },
    }


def markdown(summary):
    lines = [
        "# Resultados: estructura y regímenes",
        "",
        f"Run: `{summary['run_id']}`; ajustes: {summary['actual_fits']}. Datos históricos ya observados.",
        "",
        "No se confirma una ventaja independiente; no se promovieron modelos.",
        "",
        "Retornos compuestos de ventanas concatenadas, con costos base; drawdown a aperturas.",
        "",
        "| Activo | Semilla | Variante | Accuracy | AUC | Log-loss | Brier | Neto | DD | Trades |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, data in summary["inference"].items():
        symbol, seed = key.split("_seed")
        for name, values in data["predictive"]["metrics"].items():
            econ = next(
                e
                for e in summary["economic_metrics"]
                if e["symbol"] == symbol
                and e["seed"] == int(seed)
                and e["recipe"] == name
                and e["scenario"] == "base"
            )
            lines.append(
                f"| {symbol} | {seed} | {name} | {values['accuracy']:.2%} | {values['roc_auc']:.4f} | "
                f"{values['log_loss']:.6f} | {values['brier_score']:.6f} | "
                f"{econ['net_return']:.2%} | {econ['max_drawdown']:.2%} | {econ['trade_count']} |"
            )
    lines += [
        "",
        f"Combinación: **{summary['combination']['status']}**.",
        "",
        f"Recetas que pasan todos los filtros históricos: {summary['qualified']}.",
        "",
        "`summary.json` conserva cada criterio fallido, intervalos simultáneos, métricas por fold/estado y escenarios. "
        "`fits.json` conserva tamaños de expertos, fallback, pesos, hash de entrenamiento y brecha train/OOS. "
        "`predictions.csv`, `curves.csv`, `trades.csv` permiten reconstruir los resultados.",
        "",
        "El filtro de estructura conserva las probabilidades originales; su resultado económico se incluye en summary.json. "
        "Los resultados del modelo registrado se separan de los refits, sobre fechas posteriores a su selección.",
        "",
        "Los intervalos son aproximados y condicionales a modelos y períodos observados. "
        "No corrigen todas las rondas históricas anteriores. El spread constante y el escenario de costos dobles "
        "no sustituyen cotizaciones ni un modelo de impacto. No se mide drawdown intravela.",
    ]
    return "\n".join(lines) + "\n"
