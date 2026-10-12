"""Documenta una ronda SMC existente; no entrena ni evalúa el test observado."""

from __future__ import annotations

from pathlib import Path

from trading_ml.common import read_json, sha256


def main() -> None:
    report = read_json(Path(read_json(Path("artifacts/smc_studies/latest.json"))["summary"]))
    directory = Path(report["directory"])
    plan = read_json(directory / "plan.json")
    if (
        report["executed_experiments"] > 5
        or report["old_test_evaluated"]
        or sha256(directory / "plan.json") != report["plan_sha256"]
    ):
        raise ValueError("La ronda no coincide con el protocolo declarado")
    path = Path("EXPERIMENTS.md")
    marker = f"## Smart Money Concepts — {report['run_id']}"
    previous_text = path.read_text(encoding="utf-8")
    if marker in previous_text:
        print("La ronda SMC ya está documentada")
        return
    comparison = report["comparison"]
    original = comparison["original"]
    metadata = read_json(Path("artifacts/models") / report["candidate_version"] / "metadata.json")
    lines = [
        "",
        marker,
        "",
        "Fecha: 2026-10-12, America/Buenos_Aires. OHLCV reales BTCUSDT spot del snapshot 2020-01-01 a 2026-09-30; 2,465 velas. Las métricas siguientes son exclusivamente de desarrollo walk-forward; no son evaluación final independiente.",
        "",
        "### Reglas, datos y diseño fijado",
        "",
        "Definiciones matemáticas completas y límites en [SMC_RULES.md](SMC_RULES.md). Se detectan swings confirmados, estructura/tendencia, BOS, CHoCH, sweeps, FVG y OB. No hay señales retroactivas: pivots se publican en j+right y OB en la ruptura que los confirma. Ningún patrón demuestra quién operó ni identifica órdenes institucionales.",
        "",
        f"32 features SMC más las 14 anteriores: {len(metadata['features'])} columnas, schema 3. Los contratos 1/2 siguen compatibles. Parámetros de detección congelados en `{directory / 'plan.json'}`; SHA-256 `{report['plan_sha256']}`. Dataset preparado y snapshot OHLCV preservados en ese directorio.",
        "",
        "Se usan las mismas siete ventanas purgadas de la etapa anterior: 2022-01-01 a 2025-06-05 (límite exclusivo), 1,244 referencias etiquetadas. Se vuelve a entrenar la receta original antes de cada ventana: no se usa un modelo del futuro para simular el pasado. Estos períodos ya son desarrollo previamente estudiado; no se presentan como muestras nuevas independientes.",
        "",
        "**El test observado desde 2025-06-05 se excluyó de entrenamiento, early stopping, calibración, selección y evaluación de esta ronda.** Se emplean sus velas ya cerradas sólo para construir el ejemplo de análisis histórico, sin medir sus resultados. La evaluación final se reserva a datos futuros nuevos.",
        "",
        "Hasta cinco recetas incluyendo referencia y reglas; parada tras dos variantes ML consecutivas sin cumplir la regla completa. No hubo una grilla adicional ni ajustes guiados por el test. El umbral neutral queda fijo en 0.55/0.45. No se promueve automáticamente ningún modelo aunque cumpla validación.",
        "",
        "### Experimentos ejecutados",
        "",
        "| ID | Configuración | Accuracy | ROC-AUC OOF global | AUC media por fold | Log-loss | Brier | ECE 10 bins | Retorno compuesto neto | Trades |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["reports"]:
        summary = item["summary"]
        metrics = summary["metrics"]
        lines.append(
            f"| {item['recipe']['id']} | {item['recipe']['description']} | {100 * metrics['accuracy']:.2f}% | {metrics['roc_auc']:.6f} | {summary['mean_fold_auc']:.6f} | {metrics['log_loss']:.6f} | {metrics['brier_score']:.6f} | {metrics['ece_10_bins']:.6f} | {100 * summary['compounded_return']:.2f}% | {summary['trade_count']} |"
        )
    lines.extend(
        [
            "",
            "ROC-AUC OOF global agrupa scores de modelos/períodos diferentes; AUC media por fold promedia rankings dentro de cada ventana. La calibración puede desplazar niveles de probabilidad entre ventanas, por lo que las dos medidas pueden diferir. Se reportan ambas y la regla prefijada usa la media por fold.",
            "",
            "S2 mantiene exactamente train, features técnicas e hiperparámetros originales y agrega sólo SMC: es el control principal del aporte incremental de esas features. S3 agrega también regularización y calibración; no permite atribuir todas sus diferencias a SMC por sí solo.",
            "",
            "S3 reserva 90 días para calibración y, antes, otros 90 días para early stopping interior con purga. XGBoost se reentrena sólo sobre el bloque anterior a calibración con la cantidad de árboles elegida. La calibración aprende una sigmoide regularizada de logit(p) en filas disjuntas; no hay refit de XGBoost después. Una pendiente negativa colapsa a prevalencia y se informa, sin invertir el ranking.",
            "",
            "### Comparación de los tres sistemas solicitados",
            "",
            f"El combinado seleccionado por menor log-loss de desarrollo es {report['selected_recipe']}. Selección de investigación: {'cumplió' if report['passed_walk_forward'] else 'no cumplió'} la regla completa; promoción automática: **no**.",
            "",
            "| Métrica | XGBoost original | SMC reglas sin ML | XGBoost+SMC seleccionado |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    trio = [comparison[key] for key in ["original", "smc_rules", "combined"]]
    for key, title in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision binaria a p>=0.5"),
        ("recall", "Recall binario a p>=0.5"),
        ("f1", "F1 binario"),
        ("roc_auc", "ROC-AUC OOF global"),
        ("log_loss", "Log-loss"),
        ("brier_score", "Brier"),
        ("ece_10_bins", "ECE 10 bins"),
    ]:
        lines.append(
            f"| {title} | "
            + " | ".join(f"{item['summary']['metrics'][key]:.6f}" for item in trio)
            + " |"
        )
    for key, title in [
        ("bullish", "Señales alcistas"),
        ("bearish", "Señales bajistas"),
        ("neutral", "Neutrales"),
        ("directional_signals", "Total direccionales"),
    ]:
        lines.append(
            f"| {title} | " + " | ".join(str(item["signal_quality"][key]) for item in trio) + " |"
        )
    for key, title in [
        ("coverage", "Cobertura direccional"),
        ("bullish_precision", "Acierto alcista selectivo"),
        ("bearish_precision", "Acierto bajista selectivo"),
        ("direction_accuracy", "Acierto de señales direccionales"),
    ]:
        values = [
            "—"
            if item["signal_quality"][key] is None
            else f"{100 * item['signal_quality'][key]:.2f}%"
            for item in trio
        ]
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Los neutrales **no** cuentan como aciertos ni se eliminan para mejorar las métricas probabilísticas: ROC-AUC/log-loss/Brier/ECE usan todas las 1,244 filas. La accuracy/precision binaria usa p>=0.5 únicamente para comparación convencional; los scores neutrales 0.5 de las reglas quedan empatados a subida en esa métrica y explican su recall alto. Para interpretar las señales de la herramienta usar cobertura y precisión selectiva, que excluyen abstenciones y muestran el denominador.",
            "",
            "Estado de calibración: original y S2 sin calibración posterior; SMC reglas es confianza heurística **no calibrada** (pesos fijados, no aprendidos); S3 tiene sigmoide ajustada en pasado disjunto. Ajustar un calibrador no garantiza calibración futura bajo drift. Reliability bins completos de todos los modelos/folds están en los JSON, no sólo ECE agregado.",
            "",
            "### Resultados por ventana",
            "",
            "| Primera referencia | Última referencia | Filas | AUC original | AUC reglas | AUC combinado | Log-loss original | Log-loss reglas | Log-loss combinado | Señales original/reglas/combinado |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for index, fold in enumerate(original["folds"]):
        entries = [item["folds"][index] for item in trio]
        aucs = " | ".join(f"{entry['metrics']['roc_auc']:.4f}" for entry in entries)
        losses = " | ".join(f"{entry['metrics']['log_loss']:.4f}" for entry in entries)
        counts = "/".join(str(entry["signal_quality"]["directional_signals"]) for entry in entries)
        lines.append(
            f"| {fold['first_reference'][:10]} | {fold['last_reference'][:10]} | {fold['metrics']['rows']} | {aucs} | {losses} | {counts} |"
        )
    lines.extend(
        [
            "",
            "### Backtesting de análisis con costos",
            "",
            "Simulación long/cash de investigación, compra a próxima apertura y salida tras una vela. Comisión 10 bps y slippage 5 bps por lado; sin shorts ni órdenes reales. Los neutros conservan efectivo. La calidad bajista se mide contra etiquetas, pero el simulador no toma posiciones bajistas. Cada fold liquida dentro de su ventana y excluye señales sin precio de salida, evitando precios del fold siguiente. El compuesto encadena retornos de ventanas cerradas; no es un backtest de un modelo único desplegado sin cambios.",
            "",
            "| Ventana | Original neto | SMC reglas neto | Combinado neto | Buy and hold con costos |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, fold in enumerate(original["folds"]):
        values = " | ".join(
            f"{100 * item['folds'][index]['backtest']['strategy_metrics']['cumulative_return']:.2f}%"
            for item in trio
        )
        lines.append(
            f"| {fold['first_reference'][:10]} a {fold['last_reference'][:10]} | {values} | {100 * fold['backtest']['buy_hold_metrics']['cumulative_return']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "### Patrones y condición de parada",
            "",
            "| Patrón | Alcistas | Bajistas |",
            "| --- | ---: | ---: |",
        ]
    )
    patterns = comparison["smc_rules"]["smc_pattern_counts"]
    for key, name, up, down in [
        ("bos", "BOS", "up", "down"),
        ("choch", "CHoCH", "up", "down"),
        ("sweep", "Liquidity sweep", "low", "high"),
        ("fvg", "FVG nuevos", "up", "down"),
        ("ob", "OB nuevos", "up", "down"),
    ]:
        lines.append(
            f"| {name} | {patterns[f'smc_{key}_{up}']} | {patterns[f'smc_{key}_{down}']} |"
        )
    for item in report["reports"]:
        if "acceptance" in item:
            a = item["acceptance"]
            lines.extend(
                [
                    "",
                    f"{item['recipe']['id']}: mejoras simultáneas log-loss/Brier en {a['probabilistic_winning_folds']}/7 folds y retorno superior en {a['economic_winning_folds']}/7 (se exigían {a['required_winning_folds']}). Checks fallidos: `{', '.join(a['failed_checks']) or 'ninguno'}`.",
                ]
            )
    lines.extend(
        [
            "",
            f"Ejecutadas {report['executed_experiments']} de hasta 5 recetas; motivo `{report['stop_reason']}`. Recetas no ejecutadas: {', '.join(recipe['id'] for recipe in plan['recipes'] if recipe['id'] not in {item['recipe']['id'] for item in report['reports']}) or 'ninguna'}. La parada se respetó: no se cambió la definición ni se añadió un sexto experimento después de ver resultados.",
            "",
            "### Aporte incremental y problemas encontrados",
            "",
        ]
    )
    pure = next(item for item in report["reports"] if item["recipe"]["id"] == "S2")
    loss_gain = original["summary"]["metrics"]["log_loss"] - pure["summary"]["metrics"]["log_loss"]
    auc_gain = pure["summary"]["mean_fold_auc"] - original["summary"]["mean_fold_auc"]
    lines.extend(
        [
            f"Agregar sólo SMC (S2) cambió log-loss en {loss_gain:+.6f} de mejora y AUC media por fold en {auc_gain:+.6f}. Esto no cumple los mínimos prefijados ni la consistencia entre períodos. No demuestra aporte predictivo robusto de los patrones.",
            "",
            "S3 mejoró calidad probabilística y redujo exposición, pero su retorno sigue negativo y no ganó económicamente en suficientes ventanas. En seis de siete períodos su simulación long/cash no abrió operaciones; las 135 operaciones se concentran en un solo fold. Por eso un drawdown o pérdida menor no demuestra una estrategia rentable ni que SMC por sí solo agregó alpha. Las mejoras mezclan regularización, menor complejidad y calibración: faltan controles adicionales para atribuirlas específicamente a SMC, que no se añadieron para respetar el presupuesto.",
            "",
            "Las reglas discretas están sobreconfiadas para el siguiente cierre diario y rinden cerca/debajo del azar. Los patrones describen estructura y zonas, mientras la etiqueta sigue siendo dirección close/close de una sola vela. La agregación diaria no modela libro de órdenes, trayectoria intravela ni ejecución real. Una estructura alcista puede coexistir con probabilidad neutral a un día. La validación reutiliza desarrollo histórico y no incluye inferencia causal, intervalos de confianza ni una muestra nueva independiente.",
            "",
            f"Referencia conservada: `{report['reference_version']}`; candidato SMC: `{report['candidate_version']}`; seleccionado `{report['selected_recipe']}`. **No se cambió latest ni champion.json**. El candidato está guardado para investigación y no valida la rentabilidad del anterior.",
            "",
            "### Análisis funcional de ejemplo",
            "",
        ]
    )
    analysis_path = directory / "latest_analysis.json"
    if analysis_path.exists():
        analysis = read_json(analysis_path)
        lines.extend(
            [
                f"Se ejecutó `smc-analyze` con referencia histórica `{analysis['reference_time']}`, horizonte {analysis['horizon_bars']} vela(s). Tendencia `{analysis['trend']}`, estructura `{analysis['market_structure']}`, patrones {', '.join(item['name'] for item in analysis['patterns']) or 'ninguno'}, P(up)={analysis['probability_up']:.6f}, P(down/no cambio)={analysis['probability_down']:.6f}, estado `{analysis['signal_state']}`, calibración `{analysis['calibration_status']}`.",
                "",
                "El JSON incluye zonas activas con límites, origen y referencia de confirmación, además de advertencias de antigüedad. Las reglas y ML se muestran separados; no son instrucciones de compra/venta. Este ejemplo no evalúa ni afirma el acierto del test observado.",
                "",
            ]
        )
    lines.extend(
        [
            "### Reserva final independiente y reproducibilidad",
            "",
            f"Ventana prospectiva congelada: referencias desde `{report['prospective']['first_reference']}` hasta `{report['prospective']['end_reference_exclusive']}` exclusivo (90 referencias diarias). Estado: **pending_future_data**. No hay resultados finales nuevos para inventar o afirmar una mejora. La última apertura de salida también debe estar disponible y cerrada en el dataset de evaluación.",
            "",
            "`smc-future-evaluate` exige la ventana completa, etiquetas, aperturas de salida, integridad del plan y conservación exacta del snapshot anterior. Evalúa los tres sistemas congelados; no reentrena/recalibra ni promueve. Con el CSV actual rechaza la evaluación como pendiente y no produce métricas. La ruta completa se verifica sólo con datos sintéticos y reloj simulado dentro de tests temporales aislados: esas métricas nunca se registran como resultados reales.",
            "",
            f"Artefactos en `{directory}`: `plan.json`, `development.json`, `selection.json`, `summary.json`, `prospective.json`, `raw_snapshot.csv`, dataset preparado y reportes/modelos/predicciones/equity/trades por receta y fold. La repetición del mismo protocolo recupera el reporte sin nuevas recetas.",
            "",
            "Tests de esta etapa: detección matemática y simetría, confirmación sin retroactividad, mitigación/expiración/capacidad de zonas, igualdad de prefijos y mutaciones futuras, ES/calibración disjuntos, persistencia, neutral/coverage, exclusión del test, presupuesto/parada, conservación de referencia, inferencia y bloqueo/recorrido prospectivo. Evidencia completa de pytest/Ruff/pip check en `artifacts/verification/checks.json`.",
            "",
            "Conclusión: **no se demostró una mejora predictiva consistente atribuible a SMC**. Los detectores y el analizador sí quedaron funcionales. Esperar la ventana prospectiva antes de afirmar generalización; no continuar afinando reglas sobre el test observado. Cualquier investigación de otros horizontes, intradía o calibración adicional necesita un protocolo nuevo y presupuesto propio.",
            "",
        ]
    )
    path.write_text(previous_text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
    print(f"Ronda documentada: {report['run_id']}")


if __name__ == "__main__":
    main()
