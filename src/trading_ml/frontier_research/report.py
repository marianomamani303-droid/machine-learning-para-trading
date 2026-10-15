from __future__ import annotations

from pathlib import Path

from trading_ml.common import read_json


def write_report(directory: Path, summary: dict):
    plan = read_json(directory / "plan.json")
    lines = [
        "# Investigación predictiva multi-activo",
        "",
        "Toda la evidencia histórica es de desarrollo. El test final independiente queda sellado hasta octubre de 2027; no existe confirmación futura todavía.",
        "",
        f"Corrida: `{summary['run_id']}`. Ajustes: {summary['actual_fits']} / {plan['protocol']['research']['maximum_fits']}. Registros: {summary['records']}.",
        "",
        "## Auditoría previa",
        "",
        "ETH no tenía historial registrado: se incorporó en una ruta separada, con archivos públicos Binance y SHA256. BTC, modelos originales, punteros activos y reservas anteriores permanecen intactos.",
        "",
        "Momentum, Mean Reversion y SMC son módulos existentes. Breakout era una familia de variables; MIXTO era una clasificación de resultados de riesgo. No se presupone una estrategia independiente MIXTO.",
        "",
        "El Profit Factor previo de Mean Reversion sumaba retornos unitarios; esta fase utiliza ganancias/pérdidas monetarias. Exposición de capital y tiempo en mercado se reportan por separado. El objetivo close/close original no equivale al retorno ejecutable open/open.",
        "",
        "## Protocolo",
        "",
        "Tres folds walk-forward, purga por disponibilidad exacta y embargo de seis días entre etiquetas de train, calibración y evaluación. Escalado sólo en train. Calibración en cola temporal independiente de 120 días. Stacking sobre probabilidades de esa cola, fuera del ajuste de cada componente. Cada fold conserva recetas, hashes, modelos, métricas y operaciones.",
        "",
        "Ocho recetas tabulares comparan features locales contra features cruzadas BTC/ETH, regularización, desbalance y ventana móvil. Logistic, XGBoost, Random Forest, LightGBM y CatBoost; ensamble equiponderado y stacking. Calibración sin invertir rankings. Redes profundas excluidas por pocos miles de observaciones diarias correlacionadas y escasos ciclos independientes.",
        "",
        "Cross-asset usa la misma referencia de cierre, sin rellenar precios faltantes. Correlación y beta móviles no prueban cointegración. Los regímenes se calculan causalmente y sus métricas son descriptivas, sin optimizar cortes después de observar resultados.",
        "",
        "Dirección h1, retorno neto h3, barreras de ±2 ATR supervisadas al cierre con vencimiento a cinco velas. TP/SL se ejecutan en la apertura siguiente: no se estima orden intravela. Timeout significa TP no alcanzado, no pérdida obligatoria. Magnitud y retorno/volatilidad se evalúan como regresiones con MAE/MSE frente al promedio de train.",
        "",
        "Todas las probabilidades se prueban con la misma política defensiva y umbrales .55/.60; selección sólo en folds anteriores. El overlay mantiene hasta cinco velas y no utiliza el TP del etiquetado: sus retornos miden utilidad de señal bajo una política común, no reproducción exacta del evento objetivo.",
        "",
        "Costos por lado: comisión 10 bps, slippage 5 bps, medio spread 1 bp; spread completo 2 bps. Supuestos reproducibles, no mediciones del libro. Cash sin interés; buy-and-hold con fricción. Ninguna receta se activa para trading real.",
        "",
        "Criterios: mejora de loss > .003 sobre baselines, AUC > .505, mejora en al menos dos de tres folds, neto positivo, ≥30 operaciones, PF>1 y superioridad de loss mediante intervalos simultáneos por bloques 7/30. Bootstrap condicional por objetivo/activo, no evidencia independiente ni corrección global entre todas las fases investigadas.",
        "",
        "## Selección anterior al período descriptivo",
        "",
        "| Activo / objetivo | Receta | Accuracy | Bal. accuracy | AUC | Loss | Neto selección | Operaciones | Aceptada |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, chosen in summary["selection"].items():
        m, e = chosen["selection_metrics"], chosen["selection_economic"]
        lines.append(
            f"| {key} | {chosen['recipe']} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['roc_auc']:.3f} | {m['log_loss']:.4f} | {e['net_return']:.2%} | {e['trade_count']} | {chosen['accepted']} |"
        )
    lines += [
        "",
        "## Comparación descriptiva posterior",
        "",
        "No es test independiente: fechas de BTC ya fueron observadas. Se conservaron las selecciones anteriores sin cambiarlas por estos resultados.",
        "",
        "| Activo / objetivo | Sistema | Acc. | AUC | Loss | Neto | DD | PF | Trades | Capital medio | Costos |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, period in summary["periods"]["historical_description"].items():
        selected = summary["selection"][key]
        for name in ["xgboost_local", selected["recipe"]]:
            m = period["predictive"][name]
            e = period["economic"][f"{name}__{selected['threshold']}"]["metrics"]
            pf = "n/a" if e["profit_factor"] is None else f"{e['profit_factor']:.2f}"
            lines.append(
                f"| {key} | {name} | {m['accuracy']:.3f} | {m['roc_auc']:.3f} | {m['log_loss']:.4f} | {e['net_return']:.2%} | {e['max_drawdown']:.2%} | {pf} | {e['trade_count']} | {e['mean_capital_exposure']:.2%} | {e['execution_cost_cash']:.2f} |"
            )
    lines += [
        "",
        "## Evidencia y límites",
        "",
        "`summary.json` conserva métricas de cada modelo/activo/objetivo, precision/recall/F1/Brier, métricas por régimen y estabilidad por fold. `experiments.jsonl` incluye variantes perdedoras y regresiones. `uncertainty.json` registra todos los contrastes emparejados. CSV de cada corrida conserva curvas y operaciones, con costos brutos/netos sobre decisiones idénticas.",
        "",
        "Una pérdida menor con capital medio inferior no demuestra mejor predicción. Los distintos objetivos no se comparan por accuracy o loss entre sí. BTC y ETH comparten shocks: no duplican el número de observaciones independientes. Los intervalos históricos no eliminan sesgo por investigaciones anteriores.",
        "",
        "## Test final reservado",
        "",
        f"Ventana nueva {plan['prospective_start']} → {plan['prospective_end']}; evita reemplazar o modificar reservas anteriores. `prospective.json` congela recetas, umbrales, modelos y calibraciones ajustados sólo con datos anteriores al 1/10/2026. Evaluación de una sola apertura con control de fecha y hashes; ningún dato futuro disponible fue leído.",
        "",
        "Conclusión: INCONCLUSO hasta validación prospectiva. Conservar modelos actuales; no hay autorización automática de promoción.",
        "",
        "Fuentes técnicas: [gap temporal de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html), [CatBoost](https://catboost.ai/docs/en/concepts/python-installation), [archivos Binance](https://github.com/binance/binance-public-data).",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
