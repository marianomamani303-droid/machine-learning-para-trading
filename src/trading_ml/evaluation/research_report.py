from __future__ import annotations

from pathlib import Path
from typing import Any


def _number(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_research_report(path: Path, report: dict[str, Any]) -> None:
    development = report["development"]
    lines = [
        "# Evaluación predictiva: baselines, ablaciones e incertidumbre",
        "",
        f"Referencia: `{report['reference_version']}`. Run: `{report['run_id']}`.",
        "",
        "Protocolo fijado antes de observar resultados. CPU de un hilo, sin búsqueda de hiperparámetros ni promoción. Todos los modelos usan las mismas filas temporales purgadas. El histórico de test observado se excluye del desarrollo, las ablaciones y los rechazos.",
        "",
        f"Ajustes: {report['actual_fits']['xgboost']} XGBoost y {report['actual_fits']['logistic']} logística. Tiempo registrado: {report['elapsed_seconds']:.2f} segundos.",
        "",
        "## Desarrollo walk-forward",
        "",
        f"{development['rows']} predicciones en {len(development['folds'])} folds. El predictor p=0,5 usa la misma regla binaria p>=0,5, por lo que clasifica a subida; su accuracy no es una simulación de moneda aleatoria.",
        "",
        "| Modelo | Accuracy | AUC global | AUC media fold | Log-loss | Brier | ECE | Rechazo/aceptación |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, model in development["models"].items():
        m = model["metrics"]
        status = model.get("decision", {}).get("status", "baseline")
        lines.append(
            f"| {name} | {m['accuracy']:.6f} | {_number(m['roc_auc'])} | {_number(model['mean_fold_auc'])} | {m['log_loss']:.6f} | {m['brier_score']:.6f} | {m['ece_10_bins']:.6f} | {status} |"
        )
    lines += [
        "",
        "La AUC global mezcla niveles de scores de modelos/folds diferentes. La AUC media por fold informa rankings dentro de cada período. La regla de rechazo utiliza pérdidas probabilísticas e incertidumbre, no AUC global ni accuracy aislada.",
        "",
        "## Incertidumbre pareada",
        "",
        "Bloques móviles no circulares, muestreados dentro de cada fold; mismos índices para cada modelo y su referencia. Diferencia=candidato−referencia: menor que cero favorece al candidato en log-loss/Brier.",
        "",
        "| Comparación | Bloque | Delta log-loss | Intervalo 95% | Delta Brier | Intervalo 95% |",
        "|---|---:|---:|---|---:|---|",
    ]
    for length, block in development["uncertainty"]["blocks"].items():
        for pair, values in block["paired"].items():
            loss, brier = values["log_loss"], values["brier_score"]
            lines.append(
                f"| {pair} | {length} | {_number(loss['estimate'])} | [{_number(loss['lower'])}; {_number(loss['upper'])}] | {_number(brier['estimate'])} | [{_number(brier['lower'])}; {_number(brier['upper'])}] |"
            )
    lines += [
        "",
        "Los intervalos son marginales y condicionados a predicciones/regímenes históricos. No corrigen selección anterior ni todas las comparaciones múltiples. No acreditan generalización futura.",
        "",
        "## Ablaciones y experimentos negativos",
        "",
    ]
    for name, model in development["models"].items():
        if "decision" in model:
            decision = model["decision"]
            lines.append(
                f"- **{name}**: {decision['status']}. Checks fallidos: {', '.join(decision['failed_checks']) or 'ninguno'}; folds ganadores: {decision['winning_folds_vs_baselines']}."
            )
    lines += [
        "",
        "Se reentrena al retirar cada familia; los hiperparámetros no cambian. Las familias correlacionadas pueden sustituirse entre sí. Una mejora al retirar variables es evidencia condicional en este presupuesto, no una prueba de irrelevancia causal ni una receta seleccionada para despliegue.",
        "",
        "## Diagnóstico train/validación/test histórico",
        "",
    ]
    historical = report["historical_diagnostics"]
    if "partitions" in historical:
        lines += [
            "Baselines ajustados una única vez con el train original y C=1 fijo. El XGBoost guardado se reutiliza. Esta inspección es descriptiva y no alimenta ningún rechazo ni ajuste.",
            "",
            "| Partición | Modelo | Filas | Accuracy | AUC | Log-loss | Brier | ECE |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for partition, values in historical["partitions"].items():
            for name, diagnostic in values["models"].items():
                m = diagnostic["metrics"]
                lines.append(
                    f"| {partition} | {name} | {m['rows']} | {m['accuracy']:.6f} | {_number(m['roc_auc'])} | {m['log_loss']:.6f} | {m['brier_score']:.6f} | {m['ece_10_bins']:.6f} |"
                )
        lines += [
            "",
            "Las diferencias entre train y ventanas posteriores combinan sobreajuste potencial y deriva. El gap no permite aislar automáticamente ambas causas.",
        ]
    lines += ["", "## Probabilidades SMC casi constantes", ""]
    smc = report["smc_diagnostics"]
    if "folds" in smc:
        lines += [
            "Reproducción de scores de desarrollo desde UBJSON y calibrador históricos, sin nuevos ajustes. Estos modelos usaron particiones interiores de ES/calibración diferentes; este diagnóstico se presenta separado del control de aporte incremental SMC.",
            "",
            "| Fold | Árboles | Pendiente | Colapso | Span raw | Span calibrado | Std calibrada | AUC raw | AUC calibrada |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
        for fold in smc["folds"]:
            calibration = fold["calibration"] or {}
            raw = fold.get("raw_diagnostics", {})
            diag = fold["diagnostics"]
            lines.append(
                f"| {fold['fold']} | {fold['effective_parameters']['n_estimators']} | {_number(calibration.get('slope'))} | {calibration.get('ranking_collapsed')} | {_number(raw.get('span'))} | {diag['span']:.6f} | {diag['std']:.6f} | {_number(raw.get('metrics', {}).get('roc_auc'))} | {_number(diag['metrics']['roc_auc'])} |"
            )
        lines += [
            "",
            "Una sigmoide con pendiente positiva conserva el ranking, pero puede comprimir mucho las probabilidades. Una pendiente cero elimina discriminación. La cobertura de estados puede cambiar por desplazamiento de prevalencia o cruce del umbral, aun con scores casi constantes. El Brier agrupado, su resolución y su residual de binning están en JSON.",
        ]
    else:
        lines.append(f"Estado: {smc['status']}.")
    lines += [
        "",
        "## Reproducibilidad y límites",
        "",
        "Plan/decisiones se escriben antes de la inspección del test. Los CSV contienen cada predicción, fila y fold; los UBJSON y coeficientes/escalas permiten reproducir inferencia. completed.json verifica los hashes de todo el run; repetir el mismo código, datos, entorno y protocolo recupera el resultado sin entrenar nuevamente.",
        "",
        "No se modifican modelos anteriores ni la reserva prospectiva. El rendimiento predictivo no prueba rentabilidad. La logística no se calibró/tuneó usando el test y no se promociona como sustituto por este informe.",
        "",
        "Fuentes metodológicas: [calibración y Brier](https://scikit-learn.org/stable/modules/calibration.html), [regresión logística](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [bootstrap de series temporales](https://arch.readthedocs.io/en/stable/bootstrap/timeseries-bootstraps.html). No se instaló ARCH; el remuestreo usa NumPy.",
        "",
    ]
    with path.open("x", encoding="utf8") as stream:
        stream.write("\n".join(lines))
