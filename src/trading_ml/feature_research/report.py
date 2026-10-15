from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.advanced_research.statistics import paired_predictive
from trading_ml.common import write_new_json
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.feature_research.diagnostics import binary_loss


def _paired(frame, protocol, primary):
    keys = ["fold", "reference_time", "target_time", "target"]
    if frame.duplicated(keys + ["recipe"]).any():
        raise ValueError("Predicciones duplicadas")
    wide = frame.pivot(index=keys, columns="recipe", values="probability").sort_index()
    if wide.isna().any().any():
        raise ValueError("Las recetas deben compartir exactamente las observaciones OOS")
    probabilities = {name: wide[name].to_numpy() for name in wide}
    pairs = [(name, "original") for name in wide if name != "original"]
    if primary:
        pairs += [
            (name, baseline)
            for name in ["all", "compact_inner", "nested_choice"]
            for baseline in ["constant_05", "train_prevalence", "logistic_l2"]
        ]
        pairs += [(name, "all") for name in wide if name.startswith("all_without_")]
    # Each asset gets alpha/2: do not treat correlated BTC/ETH as independent
    # confirmations. max-|t| covers all primary pairs/metrics/blocks within asset.
    config = protocol["bootstrap"] | {
        "confidence": 1 - (1 - protocol["bootstrap"]["confidence"]) / 2
    }
    return paired_predictive(
        wide.index.get_level_values("target").to_numpy(dtype=int),
        probabilities,
        wide.index.get_level_values("fold").to_numpy(),
        pairs,
        config,
    )


def _gate(asset, seed, recipe, stats, fold_metrics, protocol, *, baseline_checks):
    checks = {}
    a = protocol["acceptance"]
    metrics = stats["metrics"]
    loss_gain = metrics["original"]["log_loss"] - metrics[recipe]["log_loss"]
    auc_gain = metrics[recipe]["roc_auc"] - metrics["original"]["roc_auc"]
    checks["minimum_log_loss_gain"] = loss_gain >= a["minimum_log_loss_gain"]
    checks["minimum_auc_gain"] = auc_gain >= a["minimum_auc_gain"]
    values = [x for x in fold_metrics if x["symbol"] == asset and x["seed"] == seed]
    original = {x["fold"]: x for x in values if x["recipe"] == "original"}
    candidate = [x for x in values if x["recipe"] == recipe]
    improved = [
        x["oos"]["log_loss"] < original[x["fold"]]["oos"]["log_loss"]
        and x["oos"]["roc_auc"] > original[x["fold"]]["oos"]["roc_auc"]
        for x in candidate
    ]
    fraction = float(np.mean(improved)) if improved else 0.0
    checks["fold_stability_loss_and_auc"] = fraction >= a["required_fold_fraction"]
    for block, pairs in stats["blocks"].items():
        delta = pairs[f"{recipe}__minus__original"]
        checks[f"simultaneous_loss_block_{block}"] = (
            delta["log_loss"]["upper"] is not None and delta["log_loss"]["upper"] < 0
        )
        checks[f"simultaneous_auc_block_{block}"] = (
            delta["roc_auc"]["lower"] is not None and delta["roc_auc"]["lower"] > 0
        )
        if baseline_checks:
            for baseline in ["constant_05", "train_prevalence", "logistic_l2"]:
                interval = pairs[f"{recipe}__minus__{baseline}"]["log_loss"]
                checks[f"beats_{baseline}_block_{block}"] = (
                    interval["upper"] is not None and interval["upper"] < 0
                )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "loss_gain": loss_gain,
        "auc_gain": auc_gain,
        "fraction_folds_improved_loss_and_auc": fraction,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def analyze(predictions, fits, permutations, protocol):
    r = protocol["research"]
    predictive, acceptance, regimes, stability = {}, {}, [], []
    for asset, rows in predictions.groupby("symbol", sort=True):
        predictive[asset] = {}
        for seed, sample in rows.groupby("seed", sort=True):
            predictive[asset][str(seed)] = _paired(sample, protocol, seed == r["seed"])
        acceptance[asset] = {}
        for recipe in ["all", "compact_inner", "nested_choice"]:
            acceptance[asset][recipe] = {
                str(seed): _gate(
                    asset,
                    seed,
                    recipe,
                    predictive[asset][str(seed)],
                    fits,
                    protocol,
                    baseline_checks=seed == r["seed"],
                )
                for seed in [r["seed"], r["secondary_seed"]]
            }
        primary = rows.loc[rows.seed == r["seed"]]
        for dimension in ["fe_regime", "fe_vol_regime"]:
            for (regime, recipe), group in primary.groupby([dimension, "recipe"]):
                regimes.append(
                    {
                        "symbol": asset,
                        "dimension": dimension,
                        "regime": regime,
                        "recipe": recipe,
                        **classification_metrics(group.target, group.probability),
                    }
                )
    for recipe in ["all", "compact_inner", "nested_choice"]:
        accepted = all(
            gate["passed"] for asset in acceptance.values() for gate in asset[recipe].values()
        )
        stability.append({"recipe": recipe, "historical_acceptance": accepted})
    # Family evidence reports add-to-original AND removal-from-all, rather than
    # presenting gain importance as proof of incremental information.
    family_effects = []
    from trading_ml.feature_research.features import FAMILIES

    for asset, rows in predictions.loc[predictions.seed == r["seed"]].groupby("symbol"):
        wide = rows.pivot(
            index=["fold", "reference_time", "target"], columns="recipe", values="probability"
        ).sort_index()
        target = wide.index.get_level_values("target").to_numpy()
        for family in FAMILIES:
            add_loss = binary_loss(target, wide[f"add_{family}"]) - binary_loss(
                target, wide.original
            )
            removal_loss = binary_loss(target, wide[f"all_without_{family}"]) - binary_loss(
                target, wide["all"]
            )
            folds = wide.index.get_level_values("fold").to_numpy()
            family_effects.append(
                {
                    "symbol": asset,
                    "family": family,
                    "add_minus_original_loss": float(add_loss.mean()),
                    "remove_minus_all_loss": float(removal_loss.mean()),
                    "add_improved_folds": sum(
                        float(add_loss[folds == f].mean()) < 0 for f in np.unique(folds)
                    ),
                    "removal_harmed_folds": sum(
                        float(removal_loss[folds == f].mean()) > 0 for f in np.unique(folds)
                    ),
                    "folds": len(np.unique(folds)),
                }
            )
    importance_summary = []
    table = pd.DataFrame(permutations)
    for (asset, recipe, name, block), group in table.groupby(["symbol", "recipe", "name", "block"]):
        importance_summary.append(
            {
                "symbol": asset,
                "recipe": recipe,
                "name": name,
                "block": int(block),
                "folds": len(group),
                "positive_folds": int((group.delta_loss_mean > 0).sum()),
                "mean_delta_loss": float(group.delta_loss_mean.mean()),
                "min_fold_delta_loss": float(group.delta_loss_mean.min()),
                "max_fold_delta_loss": float(group.delta_loss_mean.max()),
            }
        )
    passed = [s["recipe"] for s in stability if s["historical_acceptance"]]
    return {
        "predictive": predictive,
        "fold_metrics": fits,
        "acceptance": acceptance,
        "stability": stability,
        "family_effects": family_effects,
        "importance_summary": importance_summary,
        "regimes": regimes,
        "historical_qualified_recipes": passed,
        "retained_production_features": "original",
        "conclusion": "Sin mejora consistente demostrada; se conserva el contrato original."
        if not passed
        else "Hay candidatos históricos; necesitan confirmación futura antes de integrarse.",
    }


def write_report(path: Path, summary, protocol):
    r = protocol["research"]
    lines = [
        "# Investigación de feature engineering",
        "",
        f"Run: `{summary['run_id']}`. Ajustes realizados: {summary['actual_fits']} "
        f"(cota previa {summary['maximum_planned_fits']}).",
        "",
        summary["conclusion"],
        "",
        "Historial previamente observado: este estudio mide generalización temporal histórica, "
        "sin constituir confirmación prospectiva independiente. Se preservaron modelos y reservas.",
        "",
        "## Comparación sobre las mismas filas",
        "",
        "Δloss = candidato − original; negativo es mejor. Una loss menor también puede reflejar "
        "menor exceso de confianza: se exige mejora de AUC y estabilidad conjuntamente.",
        "",
        "| Activo | Receta | Features medias | Loss OOS | Δloss | AUC | Brier | Brecha loss train/OOS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for asset, results in summary["predictive"].items():
        metrics = results[str(r["seed"])]["metrics"]
        original_loss = metrics["original"]["log_loss"]
        for recipe, m in metrics.items():
            fits = [
                f
                for f in summary["fold_metrics"]
                if f["symbol"] == asset and f["seed"] == r["seed"] and f["recipe"] == recipe
            ]
            count = f"{np.mean([len(f['columns']) for f in fits]):.1f}" if fits else "—"
            gap = f"{np.mean([f['loss_generalization_gap'] for f in fits]):.4f}" if fits else "—"
            lines.append(
                f"| {asset} | {recipe} | {count} | {m['log_loss']:.5f} | "
                f"{m['log_loss'] - original_loss:+.5f} | {m['roc_auc']:.4f} | {m['brier_score']:.5f} | {gap} |"
            )
    lines += [
        "",
        "## Información incremental por familia",
        "",
        "Añadir <0 favorece la familia sobre original. Quitar >0 favorece la familia dentro del "
        "control completo. Son efectos descriptivos; significación y aceptación constan en summary.json.",
        "",
        "| Activo | Familia | Añadir: Δloss | Folds mejora | Quitar: Δloss | Folds deterioro |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["family_effects"]:
        lines.append(
            f"| {row['symbol']} | {row['family']} | {row['add_minus_original_loss']:+.5f} | "
            f"{row['add_improved_folds']}/{row['folds']} | {row['remove_minus_all_loss']:+.5f} | "
            f"{row['removal_harmed_folds']}/{row['folds']} |"
        )
    lines += [
        "",
        "## Candidatos seleccionados sin consultar el fold externo",
        "",
        "| Activo | Candidato | Semilla | Ganancia loss | Ganancia AUC | Folds mejora conjunta | Aceptado |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for asset, recipes in summary["acceptance"].items():
        for recipe, seeds in recipes.items():
            for seed, result in seeds.items():
                lines.append(
                    f"| {asset} | {recipe} | {seed} | {result['loss_gain']:+.5f} | "
                    f"{result['auc_gain']:+.5f} | {result['fraction_folds_improved_loss_and_auc']:.0%} | "
                    f"{result['passed']} |"
                )
    lines += [
        "",
        "## Protocolo y límites",
        "",
        f"Folds externos: {', '.join(r['boundaries'])}. Dos validaciones internas disjuntas "
        f"de {r['inner_days']} días; purga por target_time y embargo de {r['embargo_days']} días. "
        "Cada selección usa solamente información interna anterior al fold. Ventanas expansivas; "
        "sin calibración, early stopping, ajuste de umbrales ni cambio de objetivo.",
        "",
        "Original usa la receta XGBoost ya registrada, reajustada sobre el mismo train que cada "
        "candidato. La referencia congelada no se aplica a folds anteriores a su entrenamiento. "
        "Se incluyen controles enhanced, SMC existente y local BTC/ETH de investigaciones previas.",
        "",
        "original_pruned elimina constantes y correlaciones Spearman ≥0,90 usando train; "
        "compact_inner exige importancia por permutación positiva en ambas ventanas internas "
        "y ambos tamaños de bloque, elimina redundancias y limita a 20 variables. Sin dos variables "
        "válidas vuelve al original. nested_choice exige mejora en ambas ventanas y ganancia "
        "media ≥0,001 antes de elegir entre familias predefinidas. all es un control diagnóstico "
        "para ablaciones, sin presunción de superioridad por acumular variables.",
        "",
        "Permutation importance usa bloques contiguos de 7/30 días dentro de cada ventana; "
        "las familias se permutan juntas. Sus repeticiones miden variación de permutación, "
        "no intervalos de confianza. La permutación marginal puede crear estados improbables "
        "con features correlacionadas: se combina con correlación en train y ablación con reentrenamiento. "
        "La importancia externa y PSI sólo describen resultados; no vuelven a seleccionar columnas.",
        "",
        "Bootstrap pareado por bloques no circulares dentro del fold, 2.000 remuestras, intervalos "
        "max-|t| simultáneos para pares, métricas y bloques; alpha/2 por activo. El resultado es "
        "condicional a estas predicciones y no corrige todas las investigaciones históricas previas. "
        "Los activos y regímenes están correlacionados. Aceptación exige ganancia loss ≥0,003, "
        "AUC ≥0,005, mejora conjunta en al menos 2/3 de folds, intervalos favorables frente a original "
        "y loss superior a tres baselines. Debe mantenerse en BTC, ETH y segunda semilla.",
        "",
        "Todas las variables usan cierres observados. La semana UTC sólo se publica tras siete "
        "velas, el lunes siguiente; se rechazan semanas parciales. BTC/ETH se unen por cierre exacto, "
        "sin imputación de peers. Beta residual se ajusta hasta t−1. Extremos, z-score y ATR de "
        "ruptura usan ventanas previas. Pivotes SMC entran al confirmarse, nunca se retroescriben. "
        "Taker flow mide agresión agregada; no identifica inversores institucionales ni libros de órdenes.",
        "",
        "El objetivo sigue siendo dirección close/close h=1. No se atribuye rentabilidad económica "
        "a una mejora predictiva: este estudio no cambia ejecución, sizing ni costos. No hay datos "
        "de order book, derivados, funding u open interest y no se inventan proxies de esas fuentes.",
        "",
        "## Artefactos",
        "",
        "plan.json registra protocolo, hashes, fuentes y entorno antes del primer ajuste. "
        "predictions.csv conserva todas las predicciones OOS; selections.json las decisiones internas; "
        "importance.json las permutaciones; diagnostics.json correlaciones y PSI de cada train; "
        "summary.json incluye métricas por fold, régimen, semilla, intervalos y rechazos. "
        "Los .ubj y baseline.json permiten reconstruir cada predicción. artifacts.json verifica integridad.",
        "",
        "Fuentes: [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), "
        "[features correlacionadas](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html), "
        "[particiones temporales y gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), "
        "[datos públicos Binance](https://github.com/binance/binance-public-data).",
        "",
    ]
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    write_new_json(path.with_name("family_effects.json"), summary["family_effects"])
