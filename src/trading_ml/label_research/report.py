from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import moving_block_indices
from trading_ml.label_research.execution import summarize


def _metrics(target, p):
    if not len(target):
        return {"rows": 0, "status": "no_observable_probabilities"}
    return classification_metrics(target, p) | {
        "balanced_accuracy": float(balanced_accuracy_score(target, np.asarray(p) >= 0.5))
    }


def aggregate(curves, trades, initial):
    curve_parts, trade_parts = [], []
    capital = initial
    for fold, group in curves.groupby("fold", sort=True):
        c = group.sort_values("timestamp").copy()
        t = trades.loc[trades.fold == fold].copy()
        scale = capital / initial
        for name in ["equity", "execution_cost_cash", "commission_cash", "slippage_spread_cash"]:
            c[name] *= scale
        # Net and gross compound their own previous fold wealth with identical decisions.
        c["gross_equity"] = (
            initial
            * (1 + pd.concat(curve_parts + [c]).gross_return).cumprod().iloc[-len(c) :].to_numpy()
        )
        for name in ["pnl_cash", "capital_before", "capital_after", "holding_cost_cash"]:
            t[name] *= scale
        capital = float(c.equity.iloc[-1])
        curve_parts.append(c)
        trade_parts.append(t)
    curve = pd.concat(curve_parts, ignore_index=True)
    first = curve.iloc[[0]].copy()
    first["timestamp"] = curve.timestamp.iloc[0] - pd.Timedelta(days=1)
    first[["equity", "gross_equity"]] = initial
    first[
        [
            "net_return",
            "gross_return",
            "exposed",
            "capital_exposure",
            "execution_cost_cash",
            "commission_cash",
            "slippage_spread_cash",
            "turnover",
        ]
    ] = 0
    curve = pd.concat([first, curve], ignore_index=True)
    trades = pd.concat(trade_parts, ignore_index=True)
    return summarize(curve, trades)


def economic_intervals(rows, candidates, parents, bootstrap):
    """Joint max-|t| paired log-growth CIs; resample calendar days within folds.

    Counts @ returns avoids materializing (samples, dates, all_strategies).
    This is conditional historical inference, not independent future confirmation.
    """
    wide = rows.pivot(
        index=["fold", "timestamp"], columns="strategy", values="net_return"
    ).sort_index()
    required = sorted(set(candidates + [parents[n] for n in candidates] + ["original", "cash"]))
    if wide[required].isna().any().any():
        raise ValueError("Comparación económica con calendarios diferentes")
    values = np.log1p(wide[required].to_numpy())
    if not np.isfinite(values).all():
        raise ValueError("Retornos inválidos para inferencia")
    lookup = {name: i for i, name in enumerate(required)}
    pairs = sorted(set((c, ref) for c in candidates for ref in ["original", parents[c], "cash"]))
    point = values.mean(axis=0) * 365.25
    result, standardized, pending = {}, [], []
    for block in bootstrap["block_lengths"]:
        indices = moving_block_indices(
            wide.index.get_level_values("fold").to_numpy(),
            bootstrap["samples"],
            block,
            bootstrap["seed"],
        )
        samples, n = indices.shape
        offsets = np.arange(samples)[:, None] * n
        counts = np.bincount((indices + offsets).ravel(), minlength=samples * n).reshape(samples, n)
        draws = counts @ values / n * 365.25
        result[str(block)] = {}
        for candidate, reference in pairs:
            delta = draws[:, lookup[candidate]] - draws[:, lookup[reference]]
            estimate = float(point[lookup[candidate]] - point[lookup[reference]])
            se = float(delta.std(ddof=1))
            interval = {"estimate": estimate, "lower": None, "upper": None, "bootstrap_se": se}
            result[str(block)][candidate + "__minus__" + reference] = interval
            standardized.append(
                np.abs((delta - estimate) / se) if se > 1e-12 else np.zeros(samples)
            )
            pending.append(interval)
    # Bonferroni alpha/2 across BTC/ETH; max-t joint within each asset and blocks.
    confidence = 1 - (1 - bootstrap["confidence"]) / 2
    critical = float(np.quantile(np.max(standardized, axis=0), confidence))
    for row in pending:
        row["lower"] = row["estimate"] - critical * row["bootstrap_se"]
        row["upper"] = row["estimate"] + critical * row["bootstrap_se"]
    return {
        "method": "paired_moving_blocks_joint_max_abs_t_annualized_log_growth",
        "samples": bootstrap["samples"],
        "asset_confidence": confidence,
        "joint_critical": critical,
        "blocks": result,
        "limitations": "Conditional fixed predictions, daily marks, correlated assets; does not correct earlier historical research or guarantee future performance",
    }


def analyze(predictions, curves, trades, folds, protocol):
    r = protocol["research"]
    eligible = predictions.loc[predictions.calendar_eligible & predictions.target.notna()].copy()
    predictive, calibration_pairs, confidence = [], [], []
    for (symbol, forecast_id), sample in eligible.groupby(["symbol", "forecast_id"], sort=True):
        if forecast_id.startswith("meta_"):
            sample = sample.loc[sample.base_offer]
        valid = sample.loc[sample.probability.notna()]
        metric = _metrics(valid.target.to_numpy(dtype=int), valid.probability.to_numpy())
        prior = _metrics(valid.target.to_numpy(dtype=int), valid.train_prior.to_numpy())
        predictive.append(
            {
                "symbol": symbol,
                "forecast_id": forecast_id,
                "target_name": sample.target_name.iloc[0] if len(sample) else "unknown",
                "eligible_rows": len(sample),
                "forecast_coverage": len(valid) / len(sample) if len(sample) else 0.0,
                "metrics": metric,
                "training_prior_metrics": prior,
            }
        )
        # Reliability of probability-selected opportunities, not portfolio trade win rate.
        for threshold in r["thresholds"]:
            selected = valid.loc[valid.probability >= threshold]
            confidence.append(
                {
                    "symbol": symbol,
                    "forecast_id": forecast_id,
                    "target_name": sample.target_name.iloc[0] if len(sample) else "unknown",
                    "threshold": threshold,
                    "opportunities": len(selected),
                    "coverage": len(selected) / len(sample) if len(sample) else 0.0,
                    "realized_positive_fraction": float(selected.target.mean())
                    if len(selected)
                    else None,
                    "mean_predicted_probability": float(selected.probability.mean())
                    if len(selected)
                    else None,
                    "mean_unit_net_return": float(selected.unit_net_return.mean())
                    if len(selected)
                    else None,
                    "overlapping_hypothetical_opportunities": True,
                }
            )
    for (symbol, target), sample in eligible.groupby(["symbol", "target_name"]):
        names = sample.forecast_id.unique()
        for cal_id in [n for n in names if n.endswith("_cal")]:
            raw_id = cal_id.removesuffix("_cal") + "_raw"
            if raw_id not in names:
                continue
            rows = sample.loc[sample.forecast_id.isin([raw_id, cal_id])]
            if cal_id.startswith("meta_"):
                rows = rows.loc[rows.base_offer]
            wide = rows.pivot(
                index=["fold", "reference_time", "target"],
                columns="forecast_id",
                values="probability",
            ).dropna()
            if wide.empty:
                continue
            y = wide.index.get_level_values("target").to_numpy(dtype=int)
            raw_m, cal_m = _metrics(y, wide[raw_id]), _metrics(y, wide[cal_id])
            calibration_pairs.append(
                {
                    "symbol": symbol,
                    "target": target,
                    "calibrated": cal_id,
                    "common_rows": len(y),
                    "raw": raw_m,
                    "cal": cal_m,
                    "delta_log_loss": cal_m["log_loss"] - raw_m["log_loss"],
                    "delta_brier": cal_m["brier_score"] - raw_m["brier_score"],
                }
            )
    economic, intervals, acceptance = {}, {}, {}
    for symbol, rows in curves.groupby("symbol", sort=True):
        economic[symbol] = {}
        for (scenario, strategy), sample in rows.groupby(["scenario", "strategy"], sort=True):
            selected_trades = trades.loc[
                (trades.symbol == symbol)
                & (trades.scenario == scenario)
                & (trades.strategy == strategy)
            ]
            economic[symbol].setdefault(scenario, {})[strategy] = aggregate(
                sample, selected_trades, protocol["costs"]["initial_capital"]
            )
        primary = r["primary_candidates"]
        primary_folds = [f for f in folds if f["symbol"] == symbol and f["scenario"] == "main"]
        parents = {f["strategy"]: f["parent"] for f in primary_folds}
        intervals[symbol] = economic_intervals(
            rows.loc[rows.scenario == "main"], primary, parents, protocol["bootstrap"]
        )
        acceptance[symbol] = {}
        for name in primary:
            m = economic[symbol]["main"][name]
            parent = economic[symbol]["main"][parents[name]]
            stress = economic[symbol]["stress"][name]
            candidate_folds = [f["metrics"] for f in primary_folds if f["strategy"] == name]
            positive_fraction = float(np.mean([m["net_return"] > 0 for m in candidate_folds]))
            coverage = m["trade_count"] / parent["trade_count"] if parent["trade_count"] else 0.0
            forecast_id = name.removesuffix("_55") if name.startswith("meta_") else "tb_vol_h5_cal"
            pred = next(
                p for p in predictive if p["symbol"] == symbol and p["forecast_id"] == forecast_id
            )
            pm, prior = pred["metrics"], pred["training_prior_metrics"]
            checks = {
                "minimum_trades": m["trade_count"] >= protocol["acceptance"]["minimum_trades"],
                "positive_net_return": m["net_return"] > 0,
                "positive_stressed_return": stress["net_return"] > 0,
                "positive_fold_fraction": positive_fraction
                >= protocol["acceptance"]["minimum_positive_fold_fraction"],
                "coverage_vs_parent": coverage
                >= protocol["acceptance"]["minimum_coverage_vs_parent"],
                "positive_mean_trade": m["mean_net_trade"] is not None
                and m["mean_net_trade"] > protocol["acceptance"]["minimum_average_net_trade"],
                "improved_parent_mean_trade": m["mean_net_trade"] is not None
                and parent["mean_net_trade"] is not None
                and m["mean_net_trade"] > parent["mean_net_trade"],
                "improved_parent_precision": m["win_rate"] is not None
                and parent["win_rate"] is not None
                and m["win_rate"] > parent["win_rate"],
                "predictive_skill": pm["rows"] > 0
                and pm["log_loss"] < prior["log_loss"]
                and pm["brier_score"] < prior["brier_score"]
                and pm["roc_auc"] is not None
                and pm["roc_auc"] > 0.5,
            }
            for block, pairs in intervals[symbol]["blocks"].items():
                for reference in set(["original", parents[name], "cash"]):
                    checks[f"joint_log_growth_vs_{reference}_block_{block}"] = (
                        pairs[name + "__minus__" + reference]["lower"] > 0
                    )
            acceptance[symbol][name] = {
                "passed": all(checks.values()),
                "checks": checks,
                "failed": [k for k, v in checks.items() if not v],
                "parent": parents[name],
                "positive_fold_fraction": positive_fraction,
                "trade_coverage_vs_parent": coverage,
            }
    qualified = [
        c for c in r["primary_candidates"] if all(a[c]["passed"] for a in acceptance.values())
    ]
    return {
        "predictive": predictive,
        "calibration_pairs": calibration_pairs,
        "confidence_opportunities": confidence,
        "economic": economic,
        "economic_intervals": intervals,
        "acceptance": acceptance,
        "historical_qualified_candidates": qualified,
        "retained_approach": "original",
        "conclusion": "No hay mejora consistente fuera de muestra; se conserva el enfoque original."
        if not qualified
        else "Candidatos históricos cualificados; el original sigue vigente hasta confirmación prospectiva.",
    }


def write_report(path: Path, summary, protocol):
    lines = [
        "# Triple Barrier y Meta-Labeling: viabilidad",
        "",
        summary["conclusion"],
        "",
        f"Run `{summary['run_id']}`; {summary['actual_fits']} ajustes de cota previa {summary['planned_maximum_fits']}.",
        "",
        "Historial ya observado. Las nuevas probabilidades de rentabilidad no son probabilidades de subida. "
        "No se comparan accuracies entre objetivos distintos como prueba de una mejora.",
        "",
        "## Comparación económica común",
        "",
        "Cada estrategia utiliza las mismas fechas, 100% de capital fijo long/cash y un único trade abierto. "
        "El original se reajusta antes de cada fold; no se aplica un modelo entrenado posteriormente hacia el pasado. "
        "Los últimos once días de cada fold bloquean nuevas entradas para todos, según calendario conocido.",
        "",
        "| Activo | Estrategia | Neto | Bruto | DD cierre | Trades | Win rate | Media neta/trade | PF monetario | Exposición cierre | Costos |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol, scenarios in summary["economic"].items():
        for name, m in scenarios["main"].items():
            wr = f"{m['win_rate']:.1%}" if m["win_rate"] is not None else "—"
            mean = f"{m['mean_net_trade']:.3%}" if m["mean_net_trade"] is not None else "—"
            pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "—"
            lines.append(
                f"| {symbol} | {name} | {m['net_return']:.2%} | {m['gross_return']:.2%} | "
                f"{m['max_drawdown']:.2%} | {m['trade_count']} | {wr} | {mean} | {pf} | "
                f"{m['end_bar_exposure']:.1%} | {m['execution_cost_cash']:.2f} |"
            )
    lines += [
        "",
        "## Candidatos fijados previamente",
        "",
        "| Activo | Candidato | Aceptado | Folds netos positivos | Cobertura vs padre | Criterios fallidos |",
        "|---|---|---|---:|---:|---|",
    ]
    for symbol, candidates in summary["acceptance"].items():
        for name, result in candidates.items():
            lines.append(
                f"| {symbol} | {name} | {result['passed']} | {result['positive_fold_fraction']:.0%} | "
                f"{result['trade_coverage_vs_parent']:.0%} | {', '.join(result['failed'])} |"
            )
    lines += [
        "",
        "## Qué se prueba",
        "",
        "Horizontes 1/3/5/10; TP=1,5σ20, SL=1σ20; control 2σ/1σ h5; control porcentual 3%/2% h5; "
        "ATR fraccional 14 h5 y vigilancia al cierre h5. σ es log-return histórico 20, conocido al cierre de señal; "
        "las distancias se congelan y se anclan en la apertura de entrada. No se reajustan con volatilidad futura. "
        "Se preserva evento −1/0/+1 (SL/tiempo/TP); el objetivo financiero es 1 si el trade modelado gana después de costos, "
        "incluyendo el signo de timeouts. Alcanzar TP no garantiza rentabilidad neta.",
        "",
        "La base tiene las 14 features originales, semilla 42 y parámetros registrados. Raw/cal comparten modelo y train; "
        "calibración sigmoide usa 120 días separados con purga y embargo. Colapsos de ranking y falta de datos se registran. "
        "Sin calibración válida las recetas calibradas se abstienen. Isotonic no se ajusta sobre colas pequeñas.",
        "",
        "El meta-modelo usa las 14 features más la probabilidad base. Sólo oportunidades propuestas por pronósticos "
        "OOF anteriores con p_base≥0,55 forman su entrenamiento; etiquetas deben haberse observado antes del cutoff. "
        "La calibración meta tiene sus propias oportunidades disjuntas. Se prueban logística regularizada y XGBoost "
        "conservador, filtrando el mismo padre con salida h1 o bracket h5. El padre sigue determinando la dirección long. "
        "Los ejemplos pueden solaparse; el portfolio no permite posiciones superpuestas. Un control usa unicidad "
        "media de eventos aprendida solamente en train.",
        "",
        "Confianza: umbrales fijos 0,50/0,55/0,60/0,65 y filtro EV calculado con payoffs de train. Los nuevos "
        "modelos operan a 0,55/0,60 o EV>0; los barridos descriptivos no seleccionan un ganador posterior. "
        "EV es una estimación con incertidumbre y no una garantía, especialmente con pocos eventos.",
        "",
        "## Ejecución, costos e incertidumbre",
        "",
        "OHLC diario no determina el orden de TP/SL dentro de la misma vela. El supuesto principal ejecuta SL primero "
        "en empates; take_first es sensibilidad con las MISMAS probabilidades, sin reentrenar. Los gaps de SL ejecutan "
        "en apertura, nunca en un stop imposible; TP se limita al precio límite incluso con gap favorable. Timeout "
        "ejecuta h aperturas después de entrar. Los eventos intravela se consideran observables sólo al cierre de esa "
        "vela; no se inventan timestamps intradiarios. Las señales al cierre entran en la apertura siguiente, con "
        "slippage y supuesto de disponibilidad de órdenes bracket; los datos no certifican fills reales ni liquidez.",
        "",
        "Comisión 10 bps/lado, slippage 5 bps/lado y spread completo 2 bps (1 por lado), aplicados por igual al baseline. "
        "Costos de mantenimiento configurables; cero funding/financiación en spot sin deuda. No se inventa un gasto "
        "de infraestructura desconocido. Stress duplica las tres fricciones sin cambiar señales. La curva bruta usa "
        "exactamente las decisiones netas, quitando sólo fricción de ejecución. DD y exposición al cierre no miden "
        "riesgo ni tiempo de exposición intravela; se informa también fracción de velas con posición.",
        "",
        "Los retornos se componen entre folds; PnL y costos nominales se escalan con el capital acumulado. "
        "Bootstrap económico pareado en bloques 14/30 días dentro de cada fold, max-|t| simultáneo sobre los contrastes "
        "principales contra original/padre/efectivo, alpha/2 por activo. Se exige retorno y stress positivos, ≥30 trades, "
        "≥2/3 folds positivos, ≥25% de operaciones del padre, mayor precisión y media neta/trade, skill probabilístico "
        "frente a prevalencia de train e intervalos de crecimiento favorables en ambos activos. Reducir exposición "
        "o no operar no basta para aceptar una mejora. Toda inferencia es histórica y condicional.",
        "",
        "## Evidencia y reproducción",
        "",
        "plan.json fija todo antes del primer ajuste; causal_features.csv separa inputs; labels_*.csv contiene futuros "
        "sólo para investigación; base_oof.csv y meta_pool_*.csv permiten auditar cada meta-label. Los modelos y sus "
        "calibraciones se guardan junto a hashes/fechas de train y calibración. predictions.csv conserva forecasts, "
        "fold_results.json métricas individuales y summary.json métricas por objetivo, umbral, calibración, costos, "
        "rechazos e intervalos. curves.csv/trades.csv y decisiones guardan la ejecución. No se promueven modelos ni "
        "se abre una reserva prospectiva.",
        "",
        "Fuentes: [Triple Barrier y Meta-Labeling](https://mlfinpy.readthedocs.io/en/stable/Labelling.html), "
        "[solapamiento y unicidad](https://mlfinpy.readthedocs.io/en/latest/Sampling.html), "
        "[calibración disjunta](https://scikit-learn.org/stable/modules/calibration.html).",
        "",
    ]
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
