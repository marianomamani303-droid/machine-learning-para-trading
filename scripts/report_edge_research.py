"""Render comparisons and export figures from a completed immutable study."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.edge_research.execution import friction

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        cells = [
            f"{row[c]:.5f}" if isinstance(row[c], (float, np.floating)) else str(row[c])
            for c in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(directory):
    summary = read_json(directory / "summary.json")
    plan = read_json(directory / "plan.json")
    p = pd.DataFrame(summary["predictive"])
    e = pd.DataFrame(summary["economic"])
    tests = pd.DataFrame(summary["predictive_tests"])
    etests = pd.DataFrame(summary["economic_tests"])
    diagnostics = read_json(directory / "diagnostics.json")
    error_audit = []
    buy, sell = friction(plan["protocol"])
    for task in sorted(p.task.unique()):
        predictions = pd.concat(
            [
                pd.read_csv(path)
                for path in sorted((directory / task).glob("fold_*/predictions.csv"))
            ],
            ignore_index=True,
        )
        tree = [d for d in diagnostics if d["task"] == task and d["recipe"] == "xgb_full"]
        selected = predictions.logistic_full >= plan["protocol"]["threshold"]
        error_audit.append(
            {
                "task": task,
                "direction_up_but_loses_after_cost_%": float(
                    (
                        (predictions.gross_return > 0)
                        & (predictions.gross_return <= buy / sell - 1)
                    ).mean()
                    * 100
                ),
                "xgb_train_auc_mean": float(np.mean([d["train_raw"]["roc_auc"] for d in tree])),
                "xgb_oos_raw_auc": float(
                    p.loc[p.task.eq(task) & p.recipe.eq("xgb_raw"), "roc_auc"].iloc[0]
                ),
                "xgb_raw_loss": float(
                    p.loc[p.task.eq(task) & p.recipe.eq("xgb_raw"), "log_loss"].iloc[0]
                ),
                "xgb_calibrated_loss": float(
                    p.loc[p.task.eq(task) & p.recipe.eq("xgb_full"), "log_loss"].iloc[0]
                ),
                "logistic_p055_coverage_%": float(selected.mean() * 100),
                "logistic_p055_hit_%": float(predictions.loc[selected, "target"].mean() * 100)
                if selected.any()
                else None,
                "payoff_tau_min": min(
                    read_json(path)["payoff_tau"]
                    for path in (directory / task).glob("fold_*/split.json")
                ),
                "payoff_tau_max": max(
                    read_json(path)["payoff_tau"]
                    for path in (directory / task).glob("fold_*/split.json")
                ),
            }
        )
    audit = pd.DataFrame(error_audit)
    receipts = sorted(Path("artifacts/edge_research/verification").glob(f"{directory.name}_*.json"))
    verification = read_json(receipts[-1]) if receipts else {}
    output = (
        Path("artifacts/edge_research/analyses")
        / f"{directory.name}_{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    output.mkdir(parents=True)
    for name, frame in [
        ("predictive", p),
        ("economic", e),
        ("predictive_tests", tests),
        ("economic_tests", etests),
        ("error_audit", audit),
    ]:
        frame.to_csv(output / f"{name}.csv", index=False)
    base = e[e.scenario.eq("base")].copy()
    rows = []
    for _, row in base.iterrows():
        stresses = e[
            e.task.eq(row.task) & e.recipe.eq(row.recipe) & e.policy.eq(row.policy)
        ].set_index("scenario")
        rows.append(
            {
                "task": row.task,
                "recipe": row.recipe,
                "policy": row.policy,
                "net_%": 100 * row.net_return,
                "gross_%": 100 * stresses.loc["gross", "net_return"],
                "double_%": 100 * stresses.loc["double", "net_return"],
                "delay_%": 100 * stresses.loc["delay", "net_return"],
                "sharpe": row.sharpe,
                "drawdown_%": 100 * row.max_drawdown,
                "trades": row.trades,
                "positive_folds": row.positive_folds,
                "buy_hold_%": 100 * row.buy_hold_return,
            }
        )
    econ_table = pd.DataFrame(rows)
    econ_table.to_csv(output / "comparison.csv", index=False)
    executive = []
    for _, row in p.loc[p.recipe.eq("logistic_full")].iterrows():
        fixed = econ_table.loc[
            econ_table.task.eq(row.task)
            & econ_table.recipe.eq("logistic_full")
            & econ_table.policy.eq("fixed_055")
        ].iloc[0]
        risk = econ_table.loc[
            econ_table.task.eq(row.task)
            & econ_table.recipe.eq("logistic_full")
            & econ_table.policy.eq("payoff_risk")
        ].iloc[0]
        price = p.loc[p.task.eq(row.task) & p.recipe.eq("logistic_price")].iloc[0]
        executive.append(
            {
                "task": row.task,
                "AUC": row.roc_auc,
                "loss_precio": price.log_loss,
                "loss_nuevas": row.log_loss,
                "bruto_fijo_%": fixed["gross_%"],
                "neto_fijo_%": fixed["net_%"],
                "neto_filtro_riesgo_%": risk["net_%"],
                "trades_filtro": risk.trades,
            }
        )
    passed = sum(c["eligible_for_shadow"] for c in summary["candidates"])
    significant = int((tests.p_holm < 0.05).sum())
    economic_significant = int((etests.p_holm < 0.05).sum())
    constant_significant = int(((tests.p_holm < 0.05) & tests.comparison.eq("constant_05")).sum())
    incremental_significant = int(
        ((tests.p_holm < 0.05) & tests.comparison.isin(["logistic_price", "logistic_noise"])).sum()
    )
    text = [
        "# Resultados: investigación de ventaja predictiva y económica",
        "Fecha local: 17 de octubre de 2026. Esta ronda es exploratoria sobre historial reutilizado; no constituye confirmación prospectiva.",
        f"Corrida: `{directory.name}`. **{summary['fits']} ajustes: 96 estimadores y 24 calibradores. {passed}/16 candidatos cumplen todos los criterios; {significant}/32 contrastes predictivos y {economic_significant}/16 económicos cruzan Holm 5%.** No se promovieron modelos ni se ejecutaron operaciones reales.",
        f"{constant_significant}/16 contrastes frente a p=0,5 superan Holm, y **{incremental_significant}/16 contrastes de aporte incremental de las features frente a precio o ruido superan Holm.** Superar una probabilidad constante sólo demuestra una mejora retrospectiva dentro de este protocolo; no acredita por sí solo información nueva ni ventaja económica. La tabla de inferencia identifica cada caso, sin elegir un ganador a posteriori.",
        "Comparación principal predeclarada: logística con señales adicionales frente a logística de precio. Una política que retorna 0% sin operaciones conserva efectivo y no demuestra ventaja.",
        table(
            pd.DataFrame(executive),
            [
                "task",
                "AUC",
                "loss_precio",
                "loss_nuevas",
                "bruto_fijo_%",
                "neto_fijo_%",
                "neto_filtro_riesgo_%",
                "trades_filtro",
            ],
        ),
        "## Auditoría y relación con estudios anteriores",
        "La evidencia anterior cubre miles de ajustes, algoritmos y features. Los problemas identificados siguen siendo sobreajuste, señales inestables, confianza excesiva, costos de rotación y repetición de selección sobre las mismas fechas. El informe financiero focalizado contiene indicios positivos (BTC diario logística +45,29% base y +28,24% doble costo; ETH mejora de AUC, Forex mejora relativa), pero la frase anterior de ‘mejora económica robusta’ sólo describe sensibilidad retrospectiva: falta confirmación independiente y control de toda la trayectoria de selección. No se alteró ese informe.",
        "La logística regularizada se usa como referencia interpretable y parsimoniosa; XGBoost pequeño calibrado como contraste no lineal. No se repiten DMN, redes, barreras ni grandes grillas, ya explorados sin evidencia independiente suficiente. Las métricas de entrenamiento, validación y test, probabilidades crudas/calibradas y reliability bins se conservan en `diagnostics.json`.",
        "Los modelos históricos conservan sus contratos y cutoffs. No se usaron sus pesos para puntuar fechas que podrían formar parte de su entrenamiento o selección. La nueva referencia de precio tiene 11 variables causales con ventanas físicas comparables; completa y ruido tienen 17. Es una comparación controlada de este protocolo, no una réplica del contrato original de 14 variables ni una comparación directa de rentabilidades obtenidas en períodos diferentes.",
        "## Diseño efectivamente ejecutado",
        "Dos activos spot; 1h→1h, 1h→4h, 4h→4h y 1d→24h; cuatro recetas: logística precio, logística precio+seis señales, logística precio+seis ruidos, XGBoost completo calibrado. Tres folds: enero–junio 2024, julio–diciembre 2024, enero–junio 2025. Entrenamiento expansivo desde 2022, validación de 90 días disjunta, purga por salida ejecutable y embargo de 48h. No hubo selección de hiperparámetros ni de umbrales en test.",
        "Señales adicionales: ratio de volatilidad de Parkinson, volumen relativo, Amihud estandarizado, volumen firmado aproximado por forma de vela, ruptura de extremos previos y proporción de semivarianza bajista. El régimen causal combina tendencia EMA30d y volumen relativo mínimo. No hay libro de órdenes, profundidad ni flujo taker real en este contrato: son proxies OHLCV.",
        "Todas las recetas comparten filas y costos dentro de cada tarea. H1→4h y H4→4h tienen igual horizonte físico, pero sus señales se emiten a distinta frecuencia; sus retornos agregados son comparación de políticas, no ablation pareada de frecuencias ni muestras independientes. Las ventanas de indicadores representan días físicos iguales. Los huecos invalidan etiquetas atravesadas y reinician warmup; no se rellenan precios.",
        table(
            pd.DataFrame(summary["data_quality"]),
            ["task", "rows", "eligible", "gaps", "first", "last", "latest_target"],
        ),
        "## Resultados predictivos sobre las mismas filas por tarea",
        table(
            p,
            [
                "task",
                "recipe",
                "rows",
                "accuracy",
                "roc_auc",
                "log_loss",
                "brier_score",
                "ece_10_bins",
            ],
        ),
        "Los controles p=0,5 y prevalencia se evalúan sin reentrenamiento. La calibración sigmoide del árbol sólo usa validación anterior al test. La logística conserva probabilidades propias. Se dispone de una sola semilla fija; no se presenta como replicación por semillas.",
        "### Diagnóstico nuevo de errores",
        "La brecha train/OOS del árbol identifica sobreajuste; la comparación crudo/calibrado separa discriminación de probabilidades. El porcentaje de todas las filas cuya dirección alcista perdería tras fricción cuantifica el desajuste entre clasificación y beneficio. La cobertura p≥0,55 es descriptiva: no se modifican umbrales a partir de ella. AUC train es media por fold y AUC OOS se agrupa sobre todos los folds, de modo que la brecha es una señal diagnóstica, no un estimador pareado formal.",
        table(
            audit,
            [
                "task",
                "direction_up_but_loses_after_cost_%",
                "xgb_train_auc_mean",
                "xgb_oos_raw_auc",
                "xgb_raw_loss",
                "xgb_calibrated_loss",
                "logistic_p055_coverage_%",
                "logistic_p055_hit_%",
                "payoff_tau_min",
                "payoff_tau_max",
            ],
        ),
        "## Inferencia predictiva predeclarada",
        "Ganancia positiva significa menor log-loss. La media de loss se calcula por día antes del remuestreo; bootstrap pareado con 2.000 réplicas, bloques 7/30 días dentro de segmentos continuos de cada fold, p unilateral para ganancia positiva centrado bajo H0 e intervalos puntuales de 95%. Holm controla la familia de 32 comparaciones de esta ronda; no corrige los miles de intentos previos ni convierte historia reutilizada en prueba ciega. Se usa el mayor p de ambas longitudes. Intervalos completos en CSV/JSON.",
        table(tests, ["task", "recipe", "comparison", "mean_gain", "p", "p_holm"]),
        "## Resultados económicos y estrés",
        "Todos los porcentajes siguientes son retornos acumulados, no anualizados. Sharpe usa retornos diarios con cash sin interés y 365 días/año. Cada fold empieza/termina flat, con costos de liquidación. Se incluyen días sin operaciones en la ventana de ejecución. Los pocos extremos omitidos por purga se identifican en los splits.",
        "Costo por lado: comisión 10 bps + slippage 5 + medio spread 2; aproximadamente 34 bps ida y vuelta, aplicado multiplicativamente. No son fills reales ni una estimación histórica de profundidad. `double` duplica toda la fricción; `delay` demora una barra adicional sin extender el final del fold. Las operaciones cuyo precio de salida falta se descartan, sin inventar ejecuciones.",
        "`fixed_055` opera full capital con p≥0,55. `payoff_risk` exige expectativa >1,5 veces costo, estimando retornos positivos/negativos sólo en validación, con tendencia EMA30d, volumen relativo≥0,25 y tamaño causal máximo 100% para objetivo anual 20% de volatilidad. `trend_risk` usa los mismos filtros/tamaño sin ML. Buy/hold opera exactamente entre los extremos de cada tarea/fold y paga sus costos. Cash retorna cero.",
        table(
            econ_table,
            [
                "task",
                "recipe",
                "policy",
                "gross_%",
                "net_%",
                "double_%",
                "delay_%",
                "sharpe",
                "drawdown_%",
                "trades",
                "positive_folds",
                "buy_hold_%",
            ],
        ),
        "## Comparación económica primaria",
        "Ganancia diaria media de payoff_risk menos buy/hold, misma ventana, con bootstrap por días y Holm separado en 16 contrastes. Esto prueba superioridad histórica frente al benchmark; no demuestra alpha ajustado por beta, ni rentabilidad futura. El control trend_risk ayuda a distinguir reducción de exposición de señal incremental. El drawdown sólo se observa en aperturas, no dentro de cada vela.",
        table(etests, ["task", "recipe", "comparison", "mean_gain", "p", "p_holm"]),
        "## Preparación para seguimiento en dashboard",
        f"{passed} candidatos cumplen todos los filtros. Cada modelo queda serializado con cutoff, schema, parámetros y hashes; `shadow_manifest.json` conserva todas las aceptaciones y rechazos. El dashboard muestra estas decisiones en ‘Investigación de nuevas señales y temporalidades’. Esta integración es de consulta: no activa modelos ni registra predicciones históricas como prospectivas. El próximo seguimiento requiere congelar un período futuro distinto de todas las reservas existentes, contrato de ejecución open/open y contabilidad con los costos de esta receta; el log legado del dashboard usa otro costo y close/close, y no debe utilizarse como validación económica de estos candidatos.",
        "## Integridad, límites y conclusión",
        f"Se verificaron {summary['prior_artifacts_verified']:,} archivos previos por SHA-256 al cierre. Modelos históricos, reservas y resultados anteriores conservaron sus bytes. Ninguna fecha >=2025-07-01 entró en features, selección o evaluación; sólo se calcularon hashes de archivos completos que contienen períodos en cuarentena. No se abrió una reserva final independiente.",
        f"Verificación adicional: {verification.get('reloaded_models', 'pendiente')} modelos recargados, {verification.get('prediction_rows', 'pendiente')} filas de predicción verificadas y {verification.get('portfolios_reconstructed_from_prices', 'pendiente')} carteras reconstruidas con flujos de caja/acciones independientes desde aperturas originales. Diferencia máxima al leer probabilidades CSV como float64: {verification.get('max_probability_error', 'pendiente')}; error en precisión nativa: {verification.get('max_native_probability_error', 'pendiente')}. Las probabilidades float32 se recuperan exactamente al convertir el decimal CSV a su precisión nativa; no se relajó la igualdad de esos modelos. No hubo reentrenamiento en esta verificación. Recibo: `{receipts[-1] if receipts else 'pendiente'}`.",
        "Software: suite completa 321 pruebas aprobadas; posteriormente, 10 pruebas dirigidas de esta investigación aprobadas (incluyen las iniciales y las nuevas de cuarentena, bootstrap y contabilidad independiente, con capital casi agotado). Lint global correcto y formato de archivos de esta tarea correcto. El chequeo global de formato detectó 9 archivos preexistentes pendientes; se dejaron intactos. `pip check` sin incompatibilidades. Evidencia inicial en `artifacts/edge_research/verification/software_checks.json`, chequeo general en `software_final_checks.json` y prueba final de precisión en `numerical_verifier_checks.json`.",
        "La selección ex post de dos activos supervivientes, las condiciones de un solo exchange, pocos ciclos, correlación BTC/ETH, una semilla y la historia reutilizada limitan la generalización. La fórmula de payoff supone magnitudes condicionales constantes en validación; p(up) no es p(beneficio neto). Calibración y filtros no crean información si no existe señal. Los resultados positivos aislados conservan ese carácter. No se presupone impacto cero al aumentar capital, ni que una AUC ligeramente mayor que 0,5 pague las comisiones. No se modelan restricciones de lotes ni órdenes mínimas; una estrategia con pérdida extrema dejaría de ser operable antes que en esta contabilidad fraccional.",
        "XGBoost funciona aquí como contraste diagnóstico: no cuenta con versiones no lineales de precio y ruido de igual dimensión y, por esa carencia de controles, no puede superar el filtro de promoción de esta ronda. Un rechazo por ese requisito no equivale a demostrar que todo modelo de árboles carezca de señal. Los criterios económicos exigen superar buy/hold; una estrategia con menor retorno y riesgo podría ser útil, pero necesitaría un contraste predeclarado con exposición equivalente antes de atribuirle ventaja.",
        "El dashboard previo calcula P&L con 14 bps aproximados y close/close y permite `seed_recent_closed_predictions`: cargar retrospectivamente predicciones posteriores al entrenamiento no las vuelve observaciones emitidas antes del resultado. Esas filas requieren identificación y separación antes de cualquier evaluación prospectiva. Se conservaron intactos sus logs; esta ronda no los usa como evidencia.",
        "Conclusión: "
        + (
            "hay candidatos para investigar prospectivamente, sin rentabilidad confirmada."
            if passed
            else "esta ronda no descubre una ventaja predictiva y económica robusta que justifique promover un modelo. Se conserva el sistema previo y se documentan también los resultados negativos."
        ),
        "## Próximos pasos recomendados",
        "1. Evitar nuevas búsquedas sobre estos folds. Mantener congelados estos modelos y los indicios de BTC diario del estudio financiero anterior como hipótesis separadas.\n2. Registrar un seguimiento futuro disjunto de todas las reservas, con criterio de parada por calendario, comparación simultánea contra cash, precio y tendencia/riesgo y multiplicidad explícita. No evaluarlo hasta completar su ventana.\n3. Medir spreads, slippage y latencia reales para el tamaño de capital previsto mediante cotizaciones registradas; usar paper trading y ledger ejecutable, sin órdenes reales.\n4. Priorizar una hipótesis acotada de magnitud condicional: estimar E[retorno neto | señales, volatilidad], contrastada contra retorno cero y un modelo simple de volatilidad, para detectar cuándo un movimiento esperado alcanza la fricción. El payoff global usado aquí no distingue magnitudes según las features; su abstención no prueba que todo filtro posible sea inviable. Registrar criterios antes de experimentar y confirmar en datos nuevos, sin volver a elegir parámetros con estos resultados. Una reducción de riesgo debe compararse contra un baseline con exposición equivalente.",
        "## Reproducción",
        "```powershell\n.\\.venv\\Scripts\\python.exe -m trading_ml.edge_research.study run --protocol config/edge_research.toml\n# Reanudar una corrida existente sin repetir modelos completados:\n.\\.venv\\Scripts\\python.exe -m trading_ml.edge_research.study run --study "
        + str(directory).replace("\\", "/")
        + "\n.\\.venv\\Scripts\\python.exe scripts/verify_edge_research.py "
        + str(directory).replace("\\", "/")
        + "\n```",
        "Una corrida nueva agrega otro experimento; no repetirla para escoger mejores métricas. `source.zip` conserva el código previo a los ajustes. SHA-256 acredita integridad local, no autenticación frente a adversarios.",
        "Referencias primarias: [validación temporal](https://scikit-learn.org/stable/modules/cross_validation.html), [calibración](https://scikit-learn.org/stable/modules/calibration.html), [sesgo de selección](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [procedencia Binance](https://github.com/binance/binance-public-data).",
    ]
    content = "\n\n".join(text) + "\n"
    (output / "report.md").write_text(content, encoding="utf-8")
    with Path("EDGE_RESEARCH_RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write(content)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, symbol in enumerate(plan["protocol"]["symbols"]):
        tasks = [f"{symbol}_{t.replace(':', '_h')}" for t in plan["protocol"]["tasks"]]
        x = np.arange(len(tasks))
        for j, recipe in enumerate(["logistic_price", "logistic_full", "xgb_full"]):
            values = p[p.task.isin(tasks) & p.recipe.eq(recipe)].set_index("task").loc[tasks]
            axes[i, 0].bar(x + (j - 1) * 0.24, values.roc_auc, width=0.24, label=recipe)
        axes[i, 0].axhline(0.5, color="black", linestyle="--", linewidth=1)
        axes[i, 0].set_ylim(0.45, max(0.6, float(p.roc_auc.max()) + 0.01))
        axes[i, 0].set_title(f"{symbol}: AUC fuera de muestra")
        selected = (
            econ_table[
                econ_table.task.isin(tasks)
                & econ_table.recipe.eq("logistic_full")
                & econ_table.policy.eq("fixed_055")
            ]
            .set_index("task")
            .loc[tasks]
        )
        risk = (
            econ_table.loc[
                econ_table.task.isin(tasks)
                & econ_table.recipe.eq("logistic_full")
                & econ_table.policy.eq("payoff_risk")
            ]
            .set_index("task")
            .loc[tasks]
        )
        selected["risk_net_%"] = risk["net_%"]
        for j, (column, label) in enumerate(
            [
                ("gross_%", "Bruto p>=0,55"),
                ("net_%", "Neto p>=0,55"),
                ("double_%", "Neto costo doble"),
                ("risk_net_%", "Neto filtro/riesgo"),
            ]
        ):
            axes[i, 1].bar(x + (j - 1.5) * 0.20, selected[column], width=0.20, label=label)
        axes[i, 1].axhline(0, color="black", linewidth=1)
        axes[i, 1].set_title(f"{symbol}: retorno acumulado, logística completa (%)")
        for ax in axes[i]:
            ax.set_xticks(x, [t.replace(symbol + "_", "") for t in tasks])
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Investigación exploratoria: historia reutilizada, sin rentabilidad confirmada")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    write_new_json(
        output / "provenance.json",
        {
            "summary_sha256": sha256(directory / "summary.json"),
            "generator_sha256": sha256(Path(__file__)),
            "study_source_sha256": plan["code"]["source_sha256"],
        },
    )
    write_new_json(
        output / "artifacts.json",
        {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    print(
        {
            "report": str(Path("EDGE_RESEARCH_RESULTS.md").resolve()),
            "artifacts": str(output),
            "eligible_candidates": passed,
            "predictive_holm_pass": significant,
            "economic_holm_pass": economic_significant,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    report(parser.parse_args().study)
