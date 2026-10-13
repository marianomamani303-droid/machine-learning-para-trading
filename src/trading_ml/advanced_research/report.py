from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(path: Path, summary: dict[str, Any]) -> None:
    c = summary["comparison"]
    lines = [
        "# Investigación avanzada de predicción y selección",
        "",
        f"Run: `{summary['run_id']}`.",
        "",
        f"Candidato examinado: **{c['candidate']}**; aceptación de desarrollo: **{c['candidate_accepted']}**. "
        f"Recomendación conservada: **{c['retained_recommendation']}**. No se promovió ningún modelo.",
        "",
        "Todo el historial es desarrollo observado. La reserva prospectiva existente se "
        "preservó y no se evaluó. Las 19 operaciones se auditaron para formular hipótesis, "
        "no para probar filtros diseñados con sus propios resultados. La selección se "
        "congeló antes de reejecutar ese período y usa únicamente los siete folds anteriores.",
        "",
        "## Auditoría individual",
        "",
        f"Ganadoras: {summary['audit']['winners']}; perdedoras: {summary['audit']['losers']}. "
        "Cada operación, su retorno y PnL, RSI, ATR, volatilidad, momentum, pendiente, "
        "volumen y señales SMC al cierre anterior están en nineteen_trade_audit.csv/json. "
        "Las diferencias de media/mediana son descriptivas, condicionadas a 19 entradas "
        "seleccionadas. No se presentan p-values iid ni se concluye que un filtro hubiera "
        "evitado pérdidas de forma independiente. Una característica común sólo origina "
        "una hipótesis que requiere ventanas distintas y finalmente datos futuros.",
        "",
        "## Datos e hipótesis",
        "",
        "El lector antiguo conserva sólo OHLCV. La extensión opcional recupera quote_volume, "
        "trades, taker_base y taker_quote de ZIP existentes verificados por SHA-256. "
        "No se inventan datos de flujo ni se descargan archivos nuevos; se comprueba "
        "que esos mismos ZIP reproduzcan los precios/volúmenes del CSV original. "
        "Ese CSV y todos los archivos de modelos/reservas permanecen intactos.",
        "",
        "Familias independientes: core sin MACD/signal y distancias redundantes; "
        "momentum 20/60, pendiente/eficiencia; z-score/reversión; ruptura de extremos "
        "anteriores y rango/volatilidad; flujo taker, quote volume, intensidad y "
        "tamaño medio de operaciones; SMC causal con pivotes confirmados. Sólo la "
        "combinación flow+momentum se fija a priori, por su hipótesis de persistencia "
        "de presión compradora; no se ensamblan todas las estrategias automáticamente.",
        "",
        "Flow es volumen taker agregado al cierre, no order book ni número de compras "
        "individuales. Los ratios evitan niveles de precio/volumen y escalas monetarias "
        "dominantes. Ablaciones usan las mismas filas de train/evaluación y etiquetas. "
        "Se guardan entrenamiento, brecha de overfitting, bins de calibración y PSI "
        "con bins de train; detectar drift no cambia retrospectivamente un parámetro.",
        "",
        "Optimización moderada: profundidad 2/3 y dos grados de regularización, "
        "ventana expansiva/móvil de 730 días, calibración sigmoide en una cola purgada "
        "de 90 días. Calibrated y su control raw comparten exactamente el mismo fit. "
        "Los scalers de logística se ajustan sólo en train. No hay redes neuronales "
        "ni búsqueda adicional tras observar los resultados.",
        "",
        "Direction_h1 conserva el objetivo original close/close. Net_h3 y net_h5 "
        "usan próxima apertura/exit open con comisiones y slippage. Return_magnitude_h3 "
        "regresa el retorno neto y calibra el score con etiquetas de la cola disjunta. "
        "Cada objetivo tiene baselines propios: no se comparan accuracy/AUC de "
        "etiquetas u horizontes distintos como si fueran equivalentes.",
        "",
        "## Reglas de evaluación",
        "",
        "Train se purga por fecha real de target. Predicciones de la evaluación "
        "no participan de ningún scaler, calibraje, payoff medio o threshold. "
        "Se compara con XGBoost original refit sobre las mismas filas (control de "
        "información/entrenamiento), además del modelo defensivo registrado. La "
        "comparación contra el registrado por sí sola no identifica el efecto "
        "de features, pues su fecha de ajuste es anterior; el control refit lo documenta.",
        "",
        "Bootstrap pareado por bloques de 7/30 velas dentro de cada fold. Los "
        "intervalos simultáneos usan max-|t| bootstrap centrado sobre todos los "
        "pares, métricas y bloques de cada objetivo. Son aproximados, condicionales "
        "a este historial y estas predicciones, y no garantizan control post-selección "
        "sobre todas las decisiones anteriores del proyecto ni confirmación futura.",
        "",
        "Aceptación predefinida exige ganancia de log-loss y AUC, evidencia "
        "simultánea contra original/logística/prevalencia/p=0,5, estabilidad entre "
        "folds, retorno neto positivo, al menos 30 operaciones y cobertura suficiente "
        "frente a la referencia. Una cartera en efectivo o con pocas entradas no "
        "se acepta por aparentar un drawdown o una rentabilidad mejores.",
        "",
        "## Probabilidades y filtros",
        "",
        "Thresholds de entrada 0,55/0,60 fijados, control económico con payoffs de "
        "train y abstención. Se mantienen las salidas, sizing, costos, límites y "
        "drawdown defensivos. Los net-horizon models declaran su plazo diferente "
        "y no participan de la selección principal direction_h1.",
        "",
        "Meta logística/XGBoost entrenan exclusivamente con oportunidades de "
        "predicciones base OOS anteriores, incluidas dos ventanas semilla de 2021. "
        "Las etiquetas evalúan retorno unitario al salir por ATR/probabilidad/"
        "máximo cinco velas; se censuran salidas que cruzan una ventana sin forecasts. "
        "No se etiqueta con una probabilidad in-sample del modelo base. El label "
        "aislado no promete reproducir el drawdown de una cartera ni trims parciales.",
        "",
        "El filtro modifica únicamente la entrada. Su p estima rentabilidad neta "
        "de la oportunidad; el control económico combina medias/se de ganancias "
        "y pérdidas de train. El límite inferior de expectativa es un escenario "
        "aproximado y no una garantía: los trades hipotéticos solapados no son iid. "
        "Sin muestra suficiente, se registra abstención, no una probabilidad cero "
        "interpretada como certeza. Cada aceptación/rechazo y su motivo conocido "
        "al cierre se conserva en *_decisions.csv, diferenciando oportunidades "
        "hipotéticas de las entradas realmente ejecutadas por la referencia.",
        "",
        "## Comparaciones",
        "",
    ]
    for phase, period in summary["periods"].items():
        lines += [
            f"### {phase}",
            "",
            "| Experimento | Neto | DD | Trades | PF | Capital medio | Tiempo invertido |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, data in period["economic"].items():
            m = data["metrics"]
            pf = "N/D" if m["profit_factor"] is None else f"{m['profit_factor']:.3f}"
            lines.append(
                f"| {name} | {m['net_return']:.2%} | {m['max_drawdown']:.2%} | {m['trade_count']} | "
                f"{pf} | {m['mean_capital_exposure']:.2%} | {m['time_in_market']:.2%} |"
            )
        for target, group in period["predictive"].items():
            lines += [
                "",
                f"Objetivo **{target}**; comparaciones válidas dentro de esta tabla.",
                "",
                "| Modelo | Accuracy | AUC | Log-loss | Brier |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for name, m in group["metrics"].items():
                auc = "N/D" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
                lines.append(
                    f"| {name} | {m['accuracy']:.2%} | {auc} | {m['log_loss']:.5f} | {m['brier_score']:.5f} |"
                )
        lines += [""]
    lines += ["## Decisiones de selección anteriores al período auditado", ""]
    for name, decision in summary["selection"].items():
        lines.append(
            f"- {name}: aceptación={decision['accepted']}; fallas: "
            + (", ".join(decision["failed_checks"]) or "ninguna")
            + "."
        )
    lines += [
        "",
        "El control de exposición equivalente del período auditado usa un "
        "multiplicador fijado con exposición media de desarrollo, aplicado a cap, "
        "riesgo ATR y objetivo de volatilidad. Es aproximado; la exposición realizada "
        "puede diferir por régimen y frecuencia. La comparación principal conserva "
        "el mismo presupuesto por operación y reporta capital/tiempo, sin llamar "
        "capacidad predictiva a una pérdida menor por invertir menos.",
        "",
        "Más detalles: selection_economic_uncertainty.json, selection_frozen.json, "
        "summary.json, archivos predictive por fold, modelos UBJSON/coeficientes JSON, "
        "auditoría de las 19 operaciones y decisions/trades/equity/events de cada "
        "variante. Se preservan todas las variantes rechazadas para auditar el "
        "presupuesto de búsqueda; no se convierten en modelos activos.",
        "",
        f"Ajustes realizados: {summary['actual_fits']}/{summary['protocol']['budget']['max_fits']}; "
        f"registros: {summary['experiment_records']}. Todos los ajustes usan un hilo "
        "CPU y hasta 160 árboles. No se repite la búsqueda hasta encontrar ganancias.",
        "",
        "Fuentes primarias: [Binance, datos públicos](https://github.com/binance/binance-public-data), "
        "[Spot API: campos de klines](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data), "
        "[scikit-learn: calibración disjunta](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).",
        "",
        "No se evaluó la reserva prospectiva, no hubo promoción y no hay operaciones reales.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
