from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(path: Path, summary: dict[str, Any]) -> None:
    protocol = summary["protocol"]
    rules = protocol["rules"]
    research = protocol["research"]
    historical = summary["periods"]["observed_historical"]["systems"]
    horizons = research["lookbacks"]
    historical_rules = [
        historical[f"momentum_L{h}_rules"]["metrics"]["net_return"] for h in horizons
    ]
    evidence_passed = any(
        d["historical_evidence_passed"]
        for period in summary["periods"].values()
        for d in period["decisions"].values()
    )
    lines = [
        "# Investigación Momentum",
        "",
        f"Run: `{summary['run_id']}`.",
        "",
        "**Conclusión: no hay ventaja confirmada en datos futuros independientes.** "
        "Las comparaciones históricas siguientes son descriptivas; todos los horizontes "
        "y umbrales se congelaron antes de ejecutar esta ronda. No se elige un ganador.",
        "",
        (
            "Ninguna variante cumple simultáneamente los criterios históricos prefijados. "
            if not evidence_passed
            else "Hay variantes con evidencia histórica exploratoria; requieren confirmación futura. "
        )
        + f"En el historial posterior a desarrollo las reglas retornan entre {min(historical_rules):.2%} "
        + f"y {max(historical_rules):.2%}, frente a buy and hold {historical['buy_hold']['metrics']['net_return']:.2%} "
        + f"y efectivo {historical['cash']['metrics']['net_return']:.2%}. "
        + "Menor drawdown que buy and hold puede ser útil como control de exposición, "
        + "pero no demuestra exceso de rentabilidad ni capacidad predictiva adicional de XGBoost.",
        "",
        "## Protocolo y causalidad",
        "",
        f"Momentum long/efectivo por instrumento. Entrada al cierre si retorno normalizado ≥{rules['minimum_normalized_momentum']}, "
        f"precio sobre SMA rápida y lenta, pendiente lenta positiva, eficiencia ≥{rules['minimum_efficiency']}, "
        f"persistencia sobre SMA lenta ≥{rules['minimum_persistence']} y cierre rompe el máximo HIGH de las L velas "
        "anteriores (shift de una vela). Ejecución en la próxima apertura.",
        "",
        f"SMA rápida=max(5,L//2); SMA lenta=2L; pendientes como cambio relativo de la media "
        f"en {rules['slope_window']} velas dividido por esa ventana. Persistencia: proporción "
        "de las últimas L velas sobre SMA lenta. Eficiencia: |log retorno L| dividido "
        "por suma de |log retornos de una vela|. Warmup puede alcanzar aproximadamente "
        "3L velas para calcular persistencia; sin readiness no se opera.",
        "",
        "Permanece mientras el retorno acumulado sea positivo, precio sobre SMA lenta "
        "y no rompa el mínimo LOW anterior; sale en próxima apertura ante cualquiera "
        f"de esas invalidaciones o al cumplir {rules['max_holding_multiplier']}L velas. No se reingresa en la misma apertura "
        f"de salida. ML sólo filtra entradas con P(trade neto rentable) ≥{research['ml_threshold']}; mantiene las "
        "mismas salidas. La etiqueta replica un trade hipotético completo open/open con "
        "costos; etiquetas sin salida natural observada se excluyen.",
        "",
        "Train se purga por la fecha real de salida de cada etiqueta (< inicio del fold). "
        "Las oportunidades de entrenamiento pueden solaparse; su cantidad no representa "
        f"operaciones independientes. Modelo fijo de {protocol['xgboost']['n_estimators']} árboles/profundidad "
        f"{protocol['xgboost']['max_depth']}, sin búsqueda, "
        "early stopping ni calibración posterior. Pocas oportunidades o una sola clase "
        "producen abstención registrada, no un ajuste con datos posteriores.",
        "",
        f"L={'/'.join(map(str, horizons))} **velas**, no meses equivalentes entre mercados; en BTC diario son "
        "días de calendario. La literatura estudia horizontes de 1–12 meses, pero una "
        "cartera diversificada de futuros long/short no prueba una ventaja en BTC long/efectivo.",
        "",
        "Fuentes primarias: [Moskowitz, Ooi y Pedersen (2012)]"
        "(https://doi.org/10.1016/j.jfineco.2011.11.003), "
        "[Hurst, Ooi y Pedersen (2017)]"
        "(https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing).",
        "",
        "## Features actuales",
        "",
        "Schema original: return_1/3/7, SMA ratios 10/20/50, MACD/signal/hist divididos "
        "por close, volatilidad rolling20 y RSI. Schema ampliado: return_21 y "
        "momentum_vol_ratio = return_7/(volatilidad×√7), entre otras variables. "
        "Son causales al cierre y no cambian en esta implementación. Momentum añade "
        "retorno simple/log acumulado por L, volatilidad log, momentum normalizado, "
        "pendientes relativas SMA, eficiencia del recorrido, fracción de retornos positivos, "
        "persistencia/run sobre media y distancias/rupturas de extremos anteriores. "
        "Las variables de etiqueta nunca entran al allowlist de features.",
        "",
        "## Rentabilidad neta y riesgo",
        "",
        "Cada ventana inicia en efectivo y liquida al último open disponible dentro del "
        "intervalo. Todos los sistemas y benchmarks usan los mismos opens y costos. "
        "La concatenación walk-forward representa ventanas reiniciadas, con costos de "
        "liquidación entre folds; no es una única cartera continua sin reinicios.",
        "",
    ]
    for period, data in summary["periods"].items():
        lines += [
            f"### {period}",
            "",
            "| Sistema | Neto | Drawdown | Turnover | Trades | Folds positivos |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, row in data["systems"].items():
            m = row["metrics"]
            lines.append(
                f"| {name} | {m['net_return']:.2%} | {m['max_drawdown']:.2%} | "
                f"{m['turnover']:.2f} | {m['trade_count']} | {row['positive_fold_fraction']:.1%} |"
            )
        lines += [
            "",
            "### Regímenes de esa ventana",
            "",
            "Regímenes definidos causalmente con momentum de 60 velas normalizado: "
            ">1 alcista, <-1 bajista, resto lateral. Se atribuye cada retorno al "
            "régimen conocido al inicio de su intervalo; los costos se incluyen. "
            "La composición por régimen reúne intervalos discontinuos y no es una "
            "cartera negociable separada.",
            "",
            "| Sistema | Alcista: neto (n) | Bajista: neto (n) | Lateral: neto (n) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, row in data["systems"].items():
            cells = []
            for regime in ["bull", "bear", "sideways"]:
                r = row["regimes"][regime]
                cells.append(f"{r['compounded_return']:.2%} ({r['observations']})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "### Estabilidad y evidencia", ""]
        for name, decision in data["decisions"].items():
            lines.append(
                f"- {name}: evidencia histórica consistente = {decision['historical_evidence_passed']}; "
                f"condiciones fallidas: {', '.join(decision['failed_checks']) or 'ninguna'}."
            )
        lines += [
            "",
            f"Intervalos pareados por bootstrap de bloques de {protocol['bootstrap']['block_lengths']} velas, "
            "sin cruzar folds. Se comparan retornos log netos contra efectivo y buy and hold. "
            f"Bonferroni cubre {2 * len(horizons)} sistemas × dos benchmarks × {len(protocol['bootstrap']['block_lengths'])} bloques. "
            "No corrige todas las decisiones históricas del proyecto ni constituye "
            "evidencia prospectiva. La evaluación por trimestres y todos los folds están "
            "en calendar_quarters/folds de summary.json; no se elige horizonte usando esas tablas.",
            "",
        ]
    lines += [
        "## Costos y límites",
        "",
        f"Costos base: `{summary['costs']}`. Stress prefijado multiplica por {protocol['stress']['cost_multiplier']} la comisión, "
        "slippage, spread y costo de tenencia; conserva modelos, señales y umbrales. "
        "Spread y slippage declarados son supuestos de investigación, no "
        "mediciones del libro ni garantía de fill. Costos por lado; benchmark también "
        "paga entrada/salida. Efectivo usa la tasa declarada (0 por defecto).",
        "",
        "Drawdown marcado a cada apertura: no incluye mínimos intravela. Sin latencia, "
        "impacto dependiente del tamaño, impuestos, lotes, financiación variable ni "
        "riesgo de contraparte. El módulo permite spread y financiación configurables "
        "y gaps de sesiones; Forex/futuros necesitan adaptadores de calendario, "
        "rolls/contratos y costos adecuados. No se afirma validación en esos mercados.",
        "",
        "Los modelos actuales originales se cargan sin modificarlos para el historial "
        "posterior a su selección. En desarrollo se ajusta una referencia con sus "
        "features y cantidad fija de árboles histórica, y recalibración temporal "
        "disjunta si corresponde; no se presenta como réplica exacta de la búsqueda "
        "histórica de cada receta. Sus parámetros heredados también tienen sesgo de "
        "selección histórica. No se usan modelos entrenados en el futuro de un fold.",
        "",
        "## Reserva prospectiva",
        "",
        f"Inicio UTC: {summary['prospective']['start']}; fin exclusivo: "
        f"{summary['prospective']['end']}. Estado: pending_future_data. "
        "Se guardan los cuatro modelos Momentum congelados y los hashes del protocolo, "
        "datos y código. No hay promoción ni operaciones reales.",
        "",
        "Cada experimento, abstención, fallo y stress se conserva en experiments.jsonl "
        "y archivos individuales. Signals, labels, trades, equity y UBJSON permiten "
        "auditar resultados negativos y reproducir la ronda.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
