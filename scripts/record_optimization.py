"""Registra resultados existentes; no entrena ni decide usando el test final."""

from __future__ import annotations

from pathlib import Path

from trading_ml.common import read_json, sha256


def main() -> None:
    pointer = read_json(Path("artifacts/optimization/latest.json"))
    report = read_json(Path(pointer["summary"]))
    directory = Path(report["directory"])
    if (
        report["executed_experiments"] > 5
        or sha256(directory / "plan.json") != report["plan_sha256"]
    ):
        raise ValueError("El reporte no cumple el presupuesto o no coincide con el plan congelado")
    marker = f"## Optimización moderada — {report['run_id']}"
    path = Path("EXPERIMENTS.md")
    existing = path.read_text(encoding="utf-8")
    if marker in existing:
        print("Este experimento ya está registrado")
        return
    selected = next(
        item
        for item in report["experiments"]
        if item["experiment"]["name"] == report["selected_experiment"]
    )
    previous, candidate = report["comparison"]["previous"], report["comparison"]["candidate"]
    metadata = read_json(Path("artifacts/models") / report["candidate_version"] / "metadata.json")
    plan = read_json(directory / "plan.json")
    lines = [
        "",
        marker,
        "",
        "Fecha de ejecución: 2026-10-12, America/Buenos_Aires. Los resultados siguientes provienen de históricos reales; las pruebas sintéticas sólo verifican invariantes.",
        "",
        "### Protocolo previo y aislamiento del test",
        "",
        f"Plan escrito antes de entrenar: `{directory / 'plan.json'}`. SHA-256 del plan: `{report['plan_sha256']}`. Configuración: `config/optimization.toml`.",
        "",
        "Máximo cinco recetas **incluida E0**; siete ventanas consecutivas desde 2022-01-01 hasta 2025-06-05; parada después de tres variantes consecutivas sin aceptación. Cada receta incluye sus ajustes por fold; no se ejecuta una grilla adicional.",
        "",
        "El período final 2025-06-05 a 2026-09-30, 483 referencias, se excluye de selección y early stopping. **Ya había sido observado en la etapa anterior**: esta confirmación es retrospectiva, no un test nuevo ciego. No hay nuevas velas posteriores a septiembre en el proveedor mensual del snapshot. La receta de referencia también fue elegida históricamente en la etapa previa; su walk-forward es una comparación retrospectiva de una receta fija, no el historial de un sistema desplegado.",
        "",
        "Cada fold entrena sólo con etiquetas conocidas antes de su primera referencia. Early stopping usa una cola interior de 90 días con purga; después se reentrena sobre el train exterior con la cantidad de árboles seleccionada. Ninguna etiqueta, feature ni precio de backtest con referencia en el fold siguiente entra en la selección. Las señales sin apertura de salida dentro de la ventana se excluyen y registran.",
        "",
        "Regla fijada: reducción OOF de log-loss >= 0.003, aumento de AUC media por fold >= 0.005, mejora de log-loss **y** Brier en al menos 5/7 ventanas, rentabilidad neta superior a referencia en al menos 5/7, retorno compuesto no peor y positivo, al menos 30 trades y log-loss inferior al baseline constante del train de cada fold. No se suavizó la regla tras observar los resultados.",
        "",
        "Mismos costos y estrategia para todas las recetas: long/cash, umbral 0.55, 10 bps de comisión y 5 bps de slippage por lado. Retorno compuesto de desarrollo = producto de los retornos netos de folds cerrados; hay cierres/reaperturas en límites y no posiciones entre ventanas. No se compara ese retorno de desarrollo con el retorno del período final como si fueran el mismo tramo.",
        "",
        "### Recetas y resultados de desarrollo",
        "",
        "| ID | Cambio | Log-loss OOF | Brier OOF | AUC media por fold | Retorno compuesto neto | Trades | Aceptado |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["experiments"]:
        summary = item["summary"]
        accepted = (
            "Referencia"
            if item["experiment"]["name"] == "E0"
            else ("Sí" if item["acceptance"]["accepted"] else "No")
        )
        lines.append(
            f"| {item['experiment']['name']} | {item['experiment']['description']} | {summary['metrics']['log_loss']:.6f} | {summary['metrics']['brier_score']:.6f} | {summary['mean_fold_auc']:.6f} | {100 * summary['compounded_return']:.2f}% | {summary['trade_count']} | {accepted} |"
        )
    lines.extend(
        [
            "",
            "E1/E2/E3: máximo 400 árboles, profundidad 2, learning rate 0.025, lambda 20, min_child_weight 20, subsample/colsample 0.8 y selección de árboles exclusivamente interior. E2 agrega retorno a 21 velas, dos retornos rezagados, momentum/volatilidad, ratio de volatilidades 5/60, ATR relativo, ubicación del cierre en el rango y cambio logarítmico de volumen. Son causales y su contrato es schema 2; los modelos originales conservan schema 1.",
            "",
            f"{selected['summary']['metrics']['rows']} observaciones OOF comunes. El candidato elegido **antes de consultar el final** fue {report['selected_experiment']}, por menor log-loss OOF entre las variantes, para auditoría aunque ninguna quedó aceptada.",
            "",
            "| ID | Ventanas con log-loss y Brier mejores | Ventanas con retorno neto mejor | Log-loss baseline constante | Motivos de rechazo |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["experiments"][1:]:
        acceptance = item["acceptance"]
        lines.append(
            f"| {item['experiment']['name']} | {acceptance['probabilistic_winning_folds']}/7 | {acceptance['economic_winning_folds']}/7 | {item['summary']['constant_metrics']['log_loss']:.6f} | {', '.join(acceptance['failed_checks'])} |"
        )
    lines.extend(
        [
            "",
            "### Rentabilidad neta por ventana",
            "",
            "| Ventana | E0 | E1 | E2 | E3 | Buy and hold con costos |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, reference in enumerate(report["experiments"][0]["folds"]):
        returns = [
            f"{100 * item['folds'][index]['backtest']['strategy_metrics']['cumulative_return']:.2f}%"
            for item in report["experiments"]
        ]
        lines.append(
            f"| {reference['first_reference'][:10]} a {reference['last_reference'][:10]} | "
            + " | ".join(returns)
            + f" | {100 * reference['backtest']['buy_hold_metrics']['cumulative_return']:.2f}% |"
        )
    if report["more_history_evaluated"]:
        quality = read_json(directory / "E3/prepared.csv.json")["quality"]
        lines.extend(
            [
                "",
                "### Historia adicional",
                "",
                f"Se descargó y verificó historia desde 2017-08-17: {quality['rows']:,} velas crudas frente a 2,465, incorporando 867 anteriores a 2020. Sin huecos, duplicados ni faltantes. Se verificó que las velas compartidas son idénticas a las originales. No se cambió el tramo de evaluación.",
                "",
                "E3 mejoró el retorno compuesto respecto de E2, pero empeoró log-loss/Brier/AUC y siguió perdiendo. No superó el baseline constante ni la regla de consistencia económica. Esta prueba no justifica incorporar automáticamente toda la historia vieja; tampoco demuestra que más datos recientes o de otras fuentes no puedan aportar. El mercado spot anterior a 2017-08-17 no existe en este proveedor.",
            ]
        )
    lines.extend(
        [
            "",
            "### Parada y comparación final única",
            "",
            f"Ejecutados **{report['executed_experiments']} de un máximo de 5**. Motivo: `{report['stop_reason']}`. E4 (ventana móvil de 730 días) quedó **sin ejecutar**, respetando la parada; no se añadió otra receta después de conocer el final.",
            "",
            f"Anterior: `{report['previous_version']}`. Candidato reentrenado para auditoría: `{report['candidate_version']}` ({report['selected_experiment']}, {metadata['selected_parameters']['n_estimators']} árboles efectivos, {metadata['fitting']['train_rows']} filas de train). Última etiqueta utilizada: {metadata['selection_label_end']}; primera referencia final: {plan['final_start']}.",
            "",
            "| Métrica en las mismas 483 referencias finales | Anterior | Candidato auditado |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, title in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("log_loss", "Log-loss"),
        ("brier_score", "Brier score"),
        ("ece_10_bins", "ECE 10 bins"),
    ]:
        lines.append(
            f"| {title} | {previous['metrics'][key]:.6f} | {candidate['metrics'][key]:.6f} |"
        )
    for key, title in [("cumulative_return", "Retorno neto"), ("max_drawdown", "Drawdown máximo")]:
        lines.append(
            f"| {title} | {100 * previous['backtest']['strategy_metrics'][key]:.2f}% | {100 * candidate['backtest']['strategy_metrics'][key]:.2f}% |"
        )
    lines.extend(
        [
            f"| Trades | {previous['backtest']['trade_count']} | {candidate['backtest']['trade_count']} |",
            f"| Win rate neto | {100 * previous['backtest']['win_rate']:.2f}% | {100 * candidate['backtest']['win_rate']:.2f}% |",
            "",
            "Buy and hold en ese mismo tramo perdió 20.33% con costos para ambas comparaciones. El candidato redujo la pérdida y mostró probabilidades menos extremas, pero realizó sólo 17 trades frente a 116; menor exposición/costos puede explicar parte de la diferencia. La mejora de log-loss/Brier no se acompañó de mayor discriminación: accuracy y ROC-AUC empeoraron. No se afirma una ventaja rentable.",
            "",
            "El candidato final también incorpora entrenamiento hasta junio de 2025, frente al modelo anterior entrenado hasta febrero de 2024: la comparación combina reentrenamiento, regularización y features, y no identifica el efecto causal de cada cambio. Los folds E0 reentrenados antes de cada ventana sirven como control comparable de la receta anterior.",
            "",
            f"**No se promueve el candidato**: falló la regla walk-forward aunque cumplió los tres checks descriptivos finales. `latest` y `champion.json` conservan `{report['champion_version']}` como referencia operativa de investigación. Esto no valida la rentabilidad del modelo anterior: sólo evita reemplazarlo por una variante sin evidencia suficiente. Se conserva también el candidato auditado para reproducir sus resultados, sin usarlo como campeón.",
            "",
            "### Evidencia y limitaciones",
            "",
            f"Directorio: `{directory}`. Plan, selección congelada y resumen: `plan.json`, `selection.json`, `summary.json`. Cada receta conserva `walk_forward.json`, `oof_predictions.csv` y modelos/equity/trades por fold. Comparación final: `final/previous` y `final/candidate`. Dataset del candidato: `{report['selected_dataset']}`. El modelo incluye configuración, hashes, padre y estado de aceptación en sus metadatos.",
            "",
            "Los tests verifican purga exterior/interior, historia móvil, features nuevas causales, preservación de contratos viejos, presupuesto, parada y conservación del campeón. La prueba adversarial que altera precios/features/etiquetas del final deja idénticos los resultados walk-forward. Evidencia de controles: `artifacts/verification/checks.json`.",
            "",
            "Diagnóstico respaldado por esta ronda: señales débiles frente a un baseline constante; sensibilidad entre regímenes (especialmente 2022); probabilidades cercanas a 0.5; costos y pocas oportunidades al umbral fijo. El aprendizaje de dirección close/close no garantiza retorno neto open/open. Los datos adicionales antiguos no corrigieron estos problemas. No se investigó una nueva estrategia, otro horizonte, calibración aprendida, shorts o señales externas para mantener el presupuesto.",
            "",
            "Próximo trabajo: esperar una ventana nueva no observada para evaluación prospectiva, calibración en una ventana independiente, monitoreo de drift y señales distintas a indicadores de precio/volumen. La ventana móvil quedó pendiente; no se seguirá buscando sobre este test para lograr una métrica mejor.",
            "",
        ]
    )
    path.write_text(existing.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
    print(f"Resultados registrados: {report['run_id']}")


if __name__ == "__main__":
    main()
