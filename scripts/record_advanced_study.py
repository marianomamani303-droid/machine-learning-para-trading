"""Publish validated research details; never fits, selects or promotes a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from trading_ml.common import read_json, sha256
from trading_ml.errors import DataError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ADVANCED_RESEARCH.md"))
    args = parser.parse_args()
    for relative, expected in read_json(args.study / "artifacts.json").items():
        if sha256(args.study / relative) != expected:
            raise DataError(f"Artefacto avanzado modificado: {relative}")
    summary = read_json(args.study / "summary.json")
    report = (args.study / "REPORT.md").read_text(encoding="utf-8")
    analysis = args.study.parent.parent / "advanced_analyses" / summary["run_id"]
    extra = ["", "## Análisis complementario, sin ajustes adicionales", ""]
    if (analysis / "meta_ranking.json").exists():
        ranking = read_json(analysis / "meta_ranking.json")
        extra += [
            f"Se evaluaron {ranking['opportunities']} oportunidades OOS anteriores al "
            "período de las 19 operaciones, incluyendo oportunidades solapadas. "
            "Bootstrap sobre calendario diario de cada fold, enmascarado a etiquetas "
            "observadas, para evitar tratar siete oportunidades espaciadas como "
            "si fueran siete días consecutivos.",
            "",
            "| Meta predictor | Accuracy | AUC | Log-loss |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, metrics in ranking["metrics"].items():
            extra.append(
                f"| {name} | {metrics['accuracy']:.2%} | {metrics['roc_auc']:.4f} | {metrics['log_loss']:.5f} |"
            )
        extra += [
            "",
            "El intercepto se lee del XGBoost congelado de cada fold: es un prior "
            "ajustado únicamente en train, sin usar features de la oportunidad. "
            "No se estima con las 19 operaciones. Véase "
            "[documentación oficial de base_score](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters). "
            "Estos resultados no son comparables con la tabla direction_h1: aquí "
            "el objetivo es rentabilidad de una oportunidad, no dirección de la "
            "próxima vela. Sus intervalos/criterios están en meta_ranking.json y "
            "meta_acceptance_diagnostic.json.",
            "",
        ]
        for name, decision in read_json(analysis / "meta_acceptance_diagnostic.json").items():
            extra.append(
                f"- {name}: aceptación={decision['accepted']}; fallas: {', '.join(decision['failed_checks'])}."
            )
        extra += [
            "",
            "Rechazos de las 19 entradas originales (sólo descriptivos):",
            "",
            "| Filtro | Entradas originales admitidas | Ganadoras rechazadas | Perdedoras rechazadas |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, value in read_json(analysis / "nineteen_rejection_summary.json").items():
            extra.append(
                f"| {name} | {value['accepted_reference_entries']} | {value['rejected_winners']} | {value['rejected_losers']} |"
            )
        extra += [
            "",
            "Las decisiones se generaron antes de conocer los resultados; cruzarlas "
            "ahora con ganadoras/perdedoras describe el resultado, no ajusta un "
            "nuevo filtro ni lo valida de forma independiente. Cambiar entradas "
            "cambia las posiciones/cooldowns posteriores, de modo que los trades "
            "de una cartera filtrada no tienen que ser un subconjunto de las 19.",
            "",
        ]
    phase = summary["periods"]["audited_observed_development"]["economic"]
    if "reference_matched_exposure" in phase:
        old = phase["reference_matched_exposure"]["metrics"]
        new = phase["candidate_matched_exposure"]["metrics"]
        extra += [
            "## Control de exposición fijado con desarrollo",
            "",
            f"Referencia ajustada: capital medio {old['mean_capital_exposure']:.2%}, "
            f"neto {old['net_return']:.2%}, DD {old['max_drawdown']:.2%}; candidato "
            f"ajustado: capital medio {new['mean_capital_exposure']:.2%}, neto "
            f"{new['net_return']:.2%}, DD {new['max_drawdown']:.2%}. Los factores "
            "fueron congelados antes del período auditado. No es una igualdad "
            "exacta de exposición realizada, pero evita atribuir la pérdida menor "
            "frente a la referencia íntegra automáticamente a mejor selección.",
            "",
        ]
    extra += [
        "## Presupuesto y reproducibilidad",
        "",
        f"{summary['actual_fits']} ajustes computados contra límite "
        f"{summary['protocol']['budget']['max_fits']}, incluidos 26 intentos de la "
        f"corrida técnica fallida. {summary['fits_in_this_run']} ajustes nuevos y "
        f"{summary['reused_models']} modelos recuperados verificando train, parámetros, "
        "corte y cola de calibración. El fallo fue de dtype del pool meta, no un "
        "resultado que causara cambio de hipótesis. La corrida fallida se conserva.",
        "",
        "La búsqueda quedó cerrada: no se agregaron recetas ni se eligieron "
        "umbrales para mejorar el período auditado. El candidato seleccionado "
        "por log-loss anterior fue rechazado por los criterios conjuntos. La "
        "referencia defensiva permanece vigente y ninguna variante se promovió.",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -m trading_ml advanced-study",
        "# Recuperar modelos de una corrida técnica fallida del mismo protocolo:",
        ".\\.venv\\Scripts\\python.exe -m trading_ml advanced-study --cache-from artifacts/advanced_studies/20261013T224439_cdea5362",
        "# Analizar una ejecución completada, sin ajustes:",
        f".\\.venv\\Scripts\\python.exe scripts/analyze_advanced_results.py {args.study.as_posix()}",
        f".\\.venv\\Scripts\\python.exe scripts/record_advanced_study.py {args.study.as_posix()}",
        "```",
        "",
        "Cada ejecución y análisis tienen directorios nuevos, sin sobrescribir "
        "artifacts/models, datos originales o reservas. El análisis complementario "
        "rechaza reemplazar su carpeta existente. Las probabilidades/modelos se "
        "guardan en JSON/UBJSON, sin pickle. Los módulos advanced_research separan "
        "datos, features, targets, modelos, meta-labels, incertidumbre, auditoría "
        "y orquestación. run_overlay admite entry_gate opcional para filtrar "
        "entradas conservando las probabilidades y reglas de salida de la base.",
        "",
    ]
    args.output.write_text(report + "\n".join(extra), encoding="utf-8")
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
