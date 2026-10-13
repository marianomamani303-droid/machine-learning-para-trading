from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(path: Path, summary: dict[str, Any]) -> None:
    diagnosis = summary["diagnosis"]
    before, after = diagnosis["before"], diagnosis["after"]
    lines = [
        "# Investigación de gestión de riesgo",
        "",
        f"Run: `{summary['run_id']}`.",
        "",
        f"**{diagnosis['classification']}** — comparación principal prefijada: "
        f"{diagnosis['reference']} / {diagnosis['policy']}. Neto "
        f"{before['net_return']:.2%} → {after['net_return']:.2%}; drawdown "
        f"{before['max_drawdown']:.2%} → {after['max_drawdown']:.2%}.",
        "",
        "Las probabilidades del modelo son idénticas en todas las políticas. La pérdida "
        "menor puede provenir de exposición reducida, distinto sizing y menor rotación; "
        "no acredita mejor capacidad predictiva. La política principal quedó fijada "
        "en el TOML antes de ejecutar el estudio y no se seleccionó usando el historial "
        "final. No hay confirmación futura independiente.",
        "",
        "## Auditoría de pérdidas",
        "",
        "Se evalúan tres modelos registrados: original, ampliado y SMC. El original "
        "invierte todo el capital por señal y usa salida a plazo fijo. Entradas "
        "consecutivas pueden cerrar/reabrir en la misma apertura, pagando comisión "
        "y slippage por ambos lados incluso si la dirección prevista no cambió. "
        "Un retorno bruto negativo revela pérdidas que no se explican sólo por costos; "
        "la diferencia bruto/neto mide su arrastre compuesto en las mismas decisiones.",
        "",
        "Las métricas de clasificación (AUC, log-loss, Brier, accuracy) se guardan "
        "en prediction_audits. El objetivo predictivo close/close del modelo difiere "
        "del retorno ejecutable open/open; una clasificación correcta puede perder "
        "después de costos. PF bruto/neto, win rate bruto/neto y retorno unitario por "
        "trade permiten distinguir señales deficientes de costos y de pesos distintos.",
        "",
        "Momentum y Mean Reversion existentes quedan intactos. Sus informes previos "
        "ya documentan pérdidas con baja rotación y pocas operaciones: reducir costos "
        "no basta para asumir que el problema de esas estrategias desapareció. Se "
        "registran hashes de esos estudios en el plan, sin volver a entrenarlos.",
        "",
        "## Políticas simples y causales",
        "",
        "Controles fijos: exposición máxima 10%, 25%, 50% frente al 100% original. "
        "Sizing por volatilidad: min(cap, 0,01/σ20), donde σ20 es desviación de "
        "retornos simples disponible al cierre. Sizing ATR: min(cap, 0,01/(3ATR14/P)). "
        "Sizing combinado toma el menor de ambos. Sin apalancamiento. Los objetivos "
        "son por vela; con datos diarios 0,01 corresponde a 1% diario, no a un objetivo "
        "anual. Parámetros completos por política en el protocolo copiado.",
        "",
        "Stop fijo de referencia: apertura de entrada−3ATR conocido al cierre de señal. "
        "Se dispara si un cierre posterior cae bajo ese nivel, y se vende en la "
        "próxima apertura con fricción. No se usa high/low posterior para afirmar "
        "un fill intravela. Gaps pueden exceder el stop. Con holding de una vela, "
        "el stop no puede anticipar la salida ya prevista: una comparación nula es "
        "un resultado válido y se conserva.",
        "",
        "Drawdown causal: marca de cartera al cierre conocido, comparada con el "
        "máximo alcanzado hasta entonces. Al caer 5% se reduce la exposición máxima "
        "a la mitad; al caer 10% se liquida en la próxima apertura y se permanece "
        "en efectivo hasta el final del fold. El límite no garantiza drawdown ≤10% "
        "por gaps. El estado se reinicia sólo al comenzar otra ventana, como "
        "declara el protocolo. No se resetea para esconder una pérdida dentro del fold.",
        "",
        "La política de persistencia exige dos cierres consecutivos con p≥0,55, "
        "mantiene mientras p≥0,50 hasta cinco velas y aplica dos velas de cooldown "
        "tras salir. La política combinada agrega sizing, stop y drawdown con cap "
        "25%. Las confirmaciones anteriores al inicio del fold no se usan como "
        "predicciones fuera de muestra. La reducción de rotación es una decisión "
        "de ejecución distinta, no una recalibración de XGBoost.",
        "",
        "Los caps se cumplen tras cada apertura ejecutada; si una posición crece "
        "por encima del límite, se reduce con costos, sin agregar exposición entre "
        "entradas. No se garantiza un cap continuo ante gaps intravela. El sizing "
        "ATR es un presupuesto aproximado basado en información de cierre, no un "
        "límite contractual de pérdida neta.",
        "",
        "## Validación temporal y controles",
        "",
        "Siete folds cronológicos expansivos de desarrollo, con labels de train "
        "purgadas por target_time. Se reajustan referencias usando features y "
        "cantidades fijas de árboles históricas; la referencia SMC recalibra con "
        "una cola temporal disjunta de train. Esos parámetros heredados tienen "
        "sesgo de selección histórica, por lo que desarrollo no es evidencia nueva "
        "ciega. En el período posterior se cargan los modelos existentes sin alterarlos.",
        "",
        "Las ventanas comienzan en efectivo y liquidan al último open disponible. "
        "La agregación compone retornos de ventanas reiniciadas; no representa "
        "una cartera continua sin reinicios. Los costos en dinero y PF agregados "
        "suman presupuestos independientes de 10.000 por fold. En el período "
        "histórico de una ventana, los montos son directamente comparables.",
        "",
        "Control matched_original: cap original fijado exclusivamente con el ratio "
        "de exposición media combinada/original en desarrollo, antes de evaluar "
        "historial final. Es un control aproximado de exposición, no igualdad "
        "exacta de riesgo ni una normalización realizada mirando el test. Los "
        "controles cap_10/25/50 mantienen las decisiones originales, para aislar "
        "el efecto mecánico de invertir menos.",
        "",
        "Bruto: replay de las mismas entradas, salidas, reducciones y fracciones "
        "del libro neto, eliminando sólo comisión/spread/slippage. No se vuelven "
        "a decidir stops o drawdowns con una cartera bruta diferente. Intereses "
        "y financiación declarados se mantienen en ambos libros. Costos base y "
        "stress prefijados; no cambian parámetros ni probabilidades.",
        "",
    ]
    for period, data in summary["periods"].items():
        lines += [
            f"## {period}",
            "",
            "| Sistema | Bruto | Neto | Drawdown neto | PF neto | Trades | Capital medio | Tiempo invertido | Costos ejecución |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, row in data["systems"].items():
            m = row["metrics"]
            pf = "—" if m["profit_factor"] is None else f"{m['profit_factor']:.3f}"
            lines.append(
                f"| {name} | {m['gross_return']:.2%} | {m['net_return']:.2%} | {m['max_drawdown']:.2%} | "
                f"{pf} | {m['trade_count']} | {m['mean_capital_exposure']:.2%} | {m['time_in_market']:.2%} | {m['execution_cost_cash']:.2f} |"
            )
        lines += [
            "",
            "Estabilidad por fold, trimestre y régimen en summary.json. La "
            "exposición de capital es promedio ponderado por tiempo; tiempo invertido "
            "es fracción de intervalos con alguna posición. No son la misma métrica. "
            "Costos incluyen compras, ventas y trims; su monto no es exactamente la "
            "diferencia de capital bruto/neto debido a reinversión y capitalizaciones.",
            "",
        ]
    lines += [
        "## Diagnóstico y límites",
        "",
        f"Estado principal: {diagnosis['classification']}. Desarrollo y test histórico "
        "pueden dar signos diferentes; mejorar un historial negativo respecto a una "
        "exposición del 100% no implica superar efectivo ni poseer una ventaja predictiva.",
        "",
        f"Costos base: `{summary['costs']}`; stress: `{summary['stress_costs']}`. "
        "Son supuestos de ejecución, no mediciones del libro, y no cubren impuestos, "
        "latencia, lotes o impacto de mercado dependiente del tamaño. No hay trading real.",
        "",
        "Como referencia de investigación, [Moreira y Muir, Volatility Managed Portfolios]"
        "(https://www.nber.org/papers/w22208) estudian gestión de volatilidad en carteras; "
        "esos resultados no prueban que este overlay diario sobre BTC mejore las "
        "predicciones del modelo ni validan nuestros umbrales.",
        "",
        f"Se conservaron {summary['experiment_records']} registros, incluidos nulos y "
        f"negativos, con {summary['actual_fits']} refits temporales de XGBoost. Se "
        "verificaron por hash modelos, punteros, datos y planes/modelos congelados "
        "de las reservas existentes. No se modificó ni evaluó la reserva prospectiva. "
        "Toda selección adicional exigiría un protocolo nuevo y no convertiría "
        "este historial ya observado en un test independiente.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
