from __future__ import annotations

from pathlib import Path
from typing import Any


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    protocol = summary["protocol"]
    rules, research = protocol["rules"], protocol["research"]
    passed = [
        n
        for p in summary["periods"].values()
        for n, d in p["decisions"].items()
        if d["historical_evidence_passed"]
    ]
    lines = [
        "# Investigación estadística de Mean Reversion",
        "",
        f"Run: `{summary['run_id']}`.",
        "",
        (
            "**Ninguna variante supera todos los criterios históricos prefijados.**"
            if not passed
            else "**Hay evidencia histórica exploratoria en algunas variantes; requiere confirmación futura.**"
        ),
        "",
        "No hay ventaja confirmada en un test futuro independiente. Todos los umbrales, ventanas "
        "y filtros de régimen se fijaron antes de ejecutar esta ronda; no se selecciona un ganador. "
        "El historial de junio de 2025 a septiembre de 2026 ya fue observado por estudios previos "
        "y se informa como descripción histórica, sin presentarlo como prueba ciega.",
        "",
        "## Auditoría de variables existentes",
        "",
        "`features/technical.py` calcula RSI con medias exponenciales de ganancias/pérdidas, "
        "normalizado a [0,1], y SMA ratios close/SMA−1 para 10/20/50 velas. Estas distancias "
        "no están expresadas en unidades de desviación estándar. `volatility` mide desviación "
        "de retornos simples de una vela, no de niveles del precio. La versión ampliada añade "
        "ATR relativo y momentum ajustado por volatilidad; ninguna equivale a un z-score del "
        "precio. Se conservan intactas esas features y los modelos que las usan.",
        "",
        "## Definición matemática y riesgo",
        "",
        "Para cierre Pₜ, μₜ=media(Pₜ₋₁,…,Pₜ₋L), sₜ=desviación muestral de esos L precios "
        "(ddof=1), zₜ=(Pₜ−μₜ)/sₜ. La distribución de referencia excluye el cierre actual. "
        "Se calculan también z-score logarítmico, distancia porcentual, s/μ, RSI, "
        "volatilidad de log retornos, ratio de volatilidad, ATR/precio, pendientes, "
        "autocorrelación de retornos y proporción de velas bajistas.",
        "",
        f"Entrada long si z≤{rules['entry_z']} y RSI≤{rules['maximum_rsi']}, con features completas. "
        f"Ventanas: {research['lookbacks']} velas. Filtros fijos: {research['regime_filters']}. "
        "La condición extrema es una distancia descriptiva: no se asume normalidad ni una "
        "probabilidad universal para z=−2. Tampoco se supone estacionariedad del precio.",
        "",
        "Entrada en la apertura posterior al cierre de señal. La media μ de ese cierre "
        "queda fija como objetivo de la operación. Una media que se desplaza hacia el "
        "precio no basta para declarar reversión. Se descarta una apertura cuyo gap ya "
        "superó el objetivo o cuyo beneficio al objetivo no cubre costos.",
        "",
        f"Stop de referencia = apertura de entrada − {rules['stop_atr']}×ATR conocido al cierre anterior. "
        f"Fracción de cartera = min({rules['max_allocation']}, {rules['risk_fraction']}/pérdida neta "
        "unitaria prevista al stop). El presupuesto incluye costos de entrada/salida; "
        "sin leverage, averaging down ni posiciones superpuestas. Al cierre, se evalúa "
        "primero stop, luego objetivo y luego timeout; se ejecuta en la próxima apertura. "
        f"Plazo máximo: {rules['max_holding_bars']} velas. No se reingresa en la apertura de salida.",
        "",
        "El stop es una señal al cierre, **no una garantía de pérdida máxima**: un gap o "
        "un movimiento intravela puede exceder el presupuesto. Se registra cada exceso "
        "realizado. El drawdown se marca a cada apertura y no incluye mínimos intravela. "
        "Al final de cada fold se liquida; estas salidas administrativas no se usan "
        "como etiquetas artificiales de reversión.",
        "",
        "## Etiquetas y modelos",
        "",
        "Y=1 si el cierre alcanza la media anclada antes del stop o timeout; Y=0 si "
        "sale por stop/timeout. Las etiquetas sin desenlace observado permanecen censuradas, "
        "y las oportunidades con entrada no ejecutable se registran por separado. "
        "Una reversión puede perder después de costos o del gap de salida; se informan "
        "por separado frecuencia de reversión, win rate y rentabilidad.",
        "",
        "Se entrena sólo con oportunidades anteriores al fold cuyas aperturas reales "
        "de salida sean estrictamente anteriores a su inicio. Purga por target_time, "
        "incluyendo stops y timeouts. Las oportunidades hipotéticas pueden solaparse; "
        "su número no equivale al número de operaciones independientes. Folds cronológicos "
        "expansivos; nunca se ajusta scaler o modelo con datos de validación/test.",
        "",
        f"Logística L2: C={protocol['logistic']['C']}, StandardScaler ajustado exclusivamente "
        f"en train. XGBoost: {protocol['xgboost']['n_estimators']} árboles, profundidad "
        f"{protocol['xgboost']['max_depth']}, parámetros fijos. Ambos estiman P(Y=1) y "
        f"filtran la misma oportunidad con umbral {research['probability_threshold']}; "
        "comparten salidas y sizing. No predicen una ganancia neta ni garantizan una "
        "expectativa positiva por superar ese umbral. Si falta muestra o hay una sola "
        "clase, se abstienen y no inventan probabilidades.",
        "",
        "Se conservan log-loss, Brier, AUC y reliability bins por fold, junto a "
        "baselines p=0,5 y prevalencia de train. Los coeficientes/logística se guardan "
        "en JSON y XGBoost en UBJSON; no se usa pickle ni se promueve ningún modelo.",
        "",
        "## Resultados netos, expectativa y estabilidad",
        "",
        "Todos los sistemas comparten las mismas aperturas por fold. Buy and hold "
        "invierte 100%, mientras MR limita exposición y conserva efectivo: comparar "
        "drawdowns no equivale a demostrar alpha a riesgo igual. La concatenación "
        "walk-forward reinicia en efectivo cada fold y paga costos de liquidación.",
        "",
    ]
    for period, data in summary["periods"].items():
        lines += [
            f"### {period}",
            "",
            "| Sistema | Neto cartera | Drawdown | Trades | Expectativa por trade¹ | Turnover | Folds positivos |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, row in data["systems"].items():
            m = row["metrics"]
            lines.append(
                f"| {name} | {_pct(m['net_return'])} | {_pct(m['max_drawdown'])} | {m['trade_count']} | "
                f"{_pct(m.get('expectancy_net_per_trade'))} | {m['turnover']:.2f} | {row['positive_fold_fraction']:.1%} |"
            )
        lines += [
            "",
            "¹Promedio del retorno neto sobre el capital asignado a cada trade, "
            "sin confundirlo con retorno de toda la cartera. Expectativa de contribución "
            "a cartera, profit factor, win rate, duración, salidas y excesos de riesgo "
            "están en summary.json. Pocas operaciones pueden producir promedios muy inestables.",
            "",
            "### Mercado al momento de entrar",
            "",
            "Alcista/bajista/lateral: momentum log de 60 velas dividido por volatilidad "
            "de 20×√60, bandas +1/−1. Es causal al cierre de señal y compartido por "
            "las variantes; el filtro lateral se fijó antes de la ronda. Las tablas "
            "no buscan un régimen ganador después de observar resultados.",
            "",
            "| Sistema | Alcista: trades, expectativa | Bajista: trades, expectativa | Lateral: trades, expectativa |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, row in data["systems"].items():
            if not name.startswith("mr_"):
                continue
            cells = []
            for regime in ["bull", "bear", "sideways"]:
                r = row["metrics"]["entry_regimes"].get(regime, {})
                cells.append(f"{r.get('trades', 0)}, {_pct(r.get('expectancy_net_per_trade'))}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Equity por régimen conocido al inicio del intervalo, retornos "
            "por trimestre y métricas por fold también se conservan; los regímenes "
            "reúnen intervalos discontinuos y no representan carteras separadas.",
            "",
            "### Evidencia estadística",
            "",
        ]
        for name, decision in data["decisions"].items():
            lines.append(
                f"- {name}: criterio histórico={decision['historical_evidence_passed']}; "
                f"aporte del modelo frente a reglas={decision['incremental_model_evidence_vs_rules']}; "
                f"fallas: {', '.join(decision['failed_checks']) or 'ninguna'}."
            )
        lines += [
            "",
            "Bootstrap pareado de retornos log netos, bloques dentro de folds "
            f"de {protocol['bootstrap']['block_lengths']} velas y {protocol['bootstrap']['samples']} remuestras. "
            "Comparaciones contra efectivo y buy and hold, y ML contra sus propias reglas. "
            f"Bonferroni por período sobre {data['statistical_evidence']['family_per_period']} comparaciones×bloques. "
            "Los intervalos son aproximados y condicionales: las colas extremas tienen "
            "pocas remuestras y no se corrigen todas las decisiones históricas del proyecto. "
            "No constituyen una prueba futura independiente ni intervalos iid por trade.",
            "",
        ]
    lines += [
        "## Frecuencia de oportunidades",
        "",
        "Se cuentan todas las señales extremas, "
        "episodios nuevos y etiquetas realmente observadas, incluidos los casos censurados. "
        "La tasa de reversión siguiente se refiere a trades hipotéticos solapados, no a "
        "operaciones ejecutadas independientes; no se interpreta como probabilidad iid.",
        "",
    ]
    for period, recipes in summary["opportunity_frequency"].items():
        lines += [
            f"### {period}",
            "",
            "| Receta / fold | Oportunidades | Episodios | Etiquetas observadas | Reversión | Censuradas/cruzan límite |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for recipe, folds in recipes.items():
            for row in folds:
                lines.append(
                    f"| {recipe} / {row['fold']} | {row['opportunities']} | {row['episodes']} | "
                    f"{row['observed_labels_within_window']} | {_pct(row['empirical_reversion_fraction'])} | "
                    f"{row['right_censored_or_cross_boundary']} |"
                )
    lines += [
        "",
        "Se guarda además frecuencia por año, proporción de velas elegibles y régimen. "
        "Diagnóstico AR(1) en log-precio usa únicamente el pasado de cada fold. Su φ y "
        "vida media condicional son descriptivos; no son un test ADF ni prueban "
        "estacionariedad, y no se usan para seleccionar señales o umbrales.",
        "",
        "## Costos, alcance y reserva futura",
        "",
        f"Costos base: `{summary['costs']}`. Stress prefijado ×{protocol['stress']['cost_multiplier']} "
        "en comisión, slippage, spread y tenencia; no reentrena modelos ni cambia umbrales. "
        "Spread y slippage son supuestos explícitos, no mediciones del libro. Ambos "
        "benchmarks pagan sus costos; efectivo aplica la tasa declarada.",
        "",
        "La literatura relaciona reversión corta en acciones con liquidez; no prueba "
        "el mismo efecto en BTC ni que un precio individual sea estacionario. Fuentes "
        "primarias consultadas: [Nagel, Evaporating Liquidity](https://www.nber.org/papers/w17653), "
        "[Dai et al., Reversals and the Returns to Liquidity Provision](https://www.nber.org/papers/w30917). "
        "Referencia sobre la hipótesis de raíz unitaria: "
        "[documentación de ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html).",
        "",
        "OHLCV, UTC y gaps de sesiones pueden provenir de adaptadores externos. Forex "
        "y futuros requieren calendario, conversión monetaria, lotes, rolls y financiación "
        "reales; el prototipo es long/efectivo sin apalancamiento. No se afirma validación "
        "en esos mercados. Tampoco modela liquidez, impacto dependiente del tamaño o impuestos.",
        "",
        f"Reserva prospectiva UTC: {summary['prospective']['start']} a {summary['prospective']['end']} "
        "exclusivo. Estado pending_future_data. Se congelan todas las variantes, incluso "
        "las rechazadas; no se elige una usando el historial. El guard de entrenamiento "
        "protege las etiquetas reservadas. No se ejecutó una evaluación futura.",
        "",
        f"Se conservaron {summary['experiment_records']} registros y {summary['actual_fits']} "
        "ajustes en esta ejecución. Se preservaron hashes de modelos/punteros existentes. "
        "Cada fallo, pérdida y abstención queda en experiments.jsonl y archivos individuales; "
        "datos, etiquetas, modelos, trades, equity, plan y manifiesto permiten auditarlos. "
        "No hay activación ni operaciones reales.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
