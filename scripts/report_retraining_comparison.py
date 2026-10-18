"""Render the report only from recorded experiments; no fitting or selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json

OUT = Path("artifacts/retraining_comparison/20261017_controlled")


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        + ["| " + " | ".join(str(x).replace("|", " / ") for x in row) + " |" for row in rows]
    )


def n(value, digits=5):
    return "N/A" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def pct(value):
    return "N/A" if value is None or pd.isna(value) else f"{value * 100:.2f}%"


def main():
    c = pd.read_csv(OUT / "comparison.csv")
    f = pd.read_csv(OUT / "fold_metrics.csv")
    econ = pd.read_csv(OUT / "economic_comparison.csv")
    inf = read_json(OUT / "paired_inference.json")
    skill = read_json(OUT / "baseline_skill_inference.json")
    ei = read_json(OUT / "economic_inference.json")
    ledger = read_json(OUT / "training_ledger.json")
    verification = read_json(OUT / "verification.json")
    details = read_json(OUT / "all_predictive_metrics.json")
    idx = read_json(Path("artifacts/historical/validated_20261017.json"))
    hist = read_json(OUT / "historical_results.json")
    data_sources = read_json(OUT / "sources.json")

    def row(symbol, recipe, metric, interval="1d"):
        return c.loc[
            (c.symbol == symbol)
            & (c.recipe == recipe)
            & (c.interval == interval)
            & (c.metric == metric)
        ].iloc[0]

    btc = c.loc[
        (c.symbol == "BTCUSDT") & (c.target_kind == "close_direction") & (c.metric == "roc_auc")
    ]
    winner_a = btc.loc[btc.before.idxmax()]
    winner_b = btc.loc[btc.after.idxmax()]
    # Winners are descriptive OOS ranks, never a deployment/selection decision.
    supported = [name for name, d in inf.items() if d["statistically_supported_improvement"]]
    text = [
        "# RETRAINING COMPARISON REPORT",
        "Fecha: 17 de octubre de 2026. Investigación retrospectiva, sin promoción ni operaciones reales.",
        "## A. Resumen ejecutivo",
        "**¿Mejoraron después de entrenar con más datos? No se demostró una mejora estadísticamente significativa ni robusta. Hay mejoras numéricas y deterioros, pero ninguna comparación supera el criterio corregido.**",
        f"Se completaron **{ledger['total_fits']} ajustes** de estimadores/calibradores, incluidos 72 modelos OOF y los expertos de régimen. Se conservaron los resultados anteriores. Se compararon {len(inf)} combinaciones de modelo, activo, temporalidad y objetivo, más LSTM sobre una cartera BTC/ETH.",
        f"**{len(supported)}/{len(inf)} comparaciones** muestran una reducción de log-loss con p de Holm < 0,05 e intervalos positivos en ambos bloques. Esa evidencia es condicional a estos ajustes y ventanas ya investigadas; no constituye confirmación prospectiva ni una ventaja robusta de trading.",
        f"En BTC diario, el mejor A descriptivo por AUC es **{winner_a.recipe}** ({n(winner_a.before)}); el mejor B es **{winner_b.recipe}** ({n(winner_b.after)}). Son rangos retrospectivos, no modelos seleccionados para producción.",
        "**Qué significa “antes”:** A reconstruido con las recetas existentes y los datos originales desde 2020, evaluado en las mismas ventanas que B. No se resta un resultado antiguo de 2025–2026 de uno nuevo de 2023–2025. Esos períodos distintos no forman una comparación válida.",
        "## B. Datos utilizados",
        "El inventario incorporado contiene 491 datasets, 11.749.803 registros incluyendo temporalidades derivadas y 8.983.894 observaciones base. Los recuentos base tampoco equivalen a observaciones estadísticamente independientes. Se verificaron los hashes de los 491 CSV y sus manifiestos; la continuidad/duplicación de todo el inventario se recuperó de los manifiestos validados, y los datos elegidos se verificaron directamente. La auditoría no volvió a interpretar fines de semana como errores del proveedor.",
        table(
            ["Mercado", "Activos/contratos", "Datasets", "Filas", "Cobertura disponible"],
            [
                [
                    d["market"],
                    d["instruments_or_individual_contracts"],
                    d["datasets"],
                    f"{d['rows']:,}",
                    d["first_open"][:10] + " a " + d["last_close"][:10],
                ]
                for d in idx["summary"]
            ],
        ),
        "**Cuarentena:** 1.552.659 filas recientes permanecen excluidas del ajuste, selección, calibración y evaluación. Se impuso el límite 2025-07-01 y se respetaron las reservas previas. No se abrió una nueva prueba sellada.",
        table(
            [
                "Activo",
                "A: filas diarias",
                "B: filas diarias",
                "Añadidas",
                "Días recuperados",
                "Discrepancias con agregación antigua",
            ],
            [
                [
                    s,
                    m["old_rows"],
                    m["expanded_rows"],
                    m["extra_rows"],
                    m["extra_rows_missing_in_incorporated"],
                    str(m["common_native_aggregation_mismatch"]),
                ]
                for s, m in data_sources.items()
            ],
        ),
        "A utiliza 2020-01-01–2025-06-30. B añade 2017-08-18–2019-12-31. Las 2.008 filas comunes de cada activo, incluidas todas sus features y etiquetas, se conservaron idénticas. La intersección de features deja 948/1.313/1.679 filas de ajuste A y 1.674/2.039/2.405 B en los tres folds tabulares.",
        "**Desviación necesaria y predeclarada antes de entrenar:** los nuevos D1 agregados desde M15 tenían huecos y no incluían flujo. Se recuperaron 58 ZIP mensuales D1 oficiales, con checksum, para obtener precios y volúmenes taker reales de 2017–2019. No hubo interpolación. Por tanto, este experimento evalúa historia ampliada con reparación nativa documentada; no permite atribuir resultados exclusivamente a los CSV inicialmente incorporados, a datos de 2026 o a nuevos activos.",
        "Forex: EUR/USD indicativo FXCM, bid/ask H1 como fuente y ejecución, agregación NY17 con DST para diario y UTC para H4. La fuente nueva extiende a 2012, pero sólo suma 130 D1 utilizables al primer ajuste; H4 pasa de 3.534 a 11.999 filas de entrenamiento. No se sustituyó D1 NY17 por D1 UTC. No se imputaron sesiones incompletas.",
        "Los 12 pares spot y 10 Forex, sus intervalos, cobertura, huecos, cuarentena y razones de inclusión/exclusión están en [dataset_compatibility.csv](artifacts/retraining_comparison/20261017_controlled/dataset_compatibility.csv). BTC/ETH y EUR/USD se eligieron por continuidad de las recetas y representatividad, antes de observar resultados; los demás activos no recibieron una conclusión de mejora.",
        "Perpetuos y futuros con vencimiento carecen de una receta operativa anterior comparable de margen, funding y rollover en esta investigación. Los contratos individuales no se concatenaron. Los índices de contado no son instrumentos ejecutables y requieren disponibilidad al cierre local. No se crearon modelos para esos mercados ni se mezclaron sus volúmenes con spot/Forex.",
        "## C. Modelos y procedimientos",
        table(
            ["Receta", "Configuración conservada", "Features/objetivo"],
            [
                [
                    "original",
                    "XGBoost 160 árboles, profundidad 2, lr 0,03; subsample/colsample 0,85; lambda 5; min_child_weight 10",
                    "14 originales; dirección close/close siguiente",
                ],
                [
                    "logistic",
                    "StandardScaler sólo en train; LogisticRegression C=1, lbfgs, max_iter=1000",
                    "14 originales; misma dirección",
                ],
                [
                    "random_forest_cross",
                    "200 árboles; profundidad 5; min_samples_leaf=20; CPU, 1 worker",
                    "LOCAL+CROSS existentes, incluido flujo real",
                ],
                [
                    "lightgbm_cross",
                    "160 árboles; 7 hojas; profundidad 3; lr 0,03; lambda 5; min_child_samples 40",
                    "Misma receta cruzada existente",
                ],
                [
                    "catboost_cross",
                    "160 iteraciones; profundidad 3; lr 0,03; l2_leaf_reg 10; CPU",
                    "Misma receta cruzada existente",
                ],
                [
                    "smc / structure / regime_rules",
                    "Mismo XGBoost y detectores causales existentes",
                    "14 + SMC / estructura / régimen",
                ],
                [
                    "specialists_rules",
                    "XGBoost global + expertos existentes; mínimo 80 filas, 20/clase, shrinkage 200",
                    "14 originales por estado; fallback global",
                ],
                [
                    "tb_vol_h5_cal",
                    "Triple Barrier volatilidad: TP 1,5σ, SL 1σ, 5 barras, stop-first; XGBoost original",
                    "Rentabilidad neta del evento; calibración 120 días",
                ],
                [
                    "meta_net_h1_logistic_cal_55",
                    "OOF trimestral desde 2021; oferta base ≥0,55; logística C=0,1 + calibrador",
                    "14 + probabilidad base OOF; beneficio neto, no dirección",
                ],
                [
                    "fx_price / fx_compact",
                    "XGBoost 160, profundidad 3, lr 0,03; calibrador en pasado disjunto",
                    "Features price/compact existentes, calendario y spread propios",
                ],
                [
                    "lstm_sharpe_net",
                    "LSTM 20 unidades; secuencia 63; dropout 0,2; Adam 0,001; 30 epochs, patience 7; batch 128; clipping 1",
                    "8 features DMN; Sharpe neto, cartera long/cash; validación 180 días",
                ],
            ],
        ),
        "No se añadieron arquitecturas. La receta specialists_rules ya es una combinación de modelo global y expertos. No se inventó un híbrido nuevo. Los filtros de momentum/reversión con objetivos de evento diferentes no se reinterpretaron como clasificadores de dirección.",
        "**Walk-forward:** 2023, 2024 y primer semestre de 2025; ambos entrenamientos terminan antes de cada evaluación. Purga por target_time y embargo: 6 días tabular/eventos, 7 Forex y 1 DMN. Las ventanas interiores y el calibrador se mantienen en pasado. No hay búsqueda de hiperparámetros ni umbrales sobre OOS. La evaluación predictiva tabular excluye una cola calendario de 12 días y Forex una de 20 días; la ejecución usa además sus reglas originales de liquidación. Son ventanas de evaluación retrospectiva, no un test independiente nuevo.",
        "Semillas 42 y 1729 para clasificadores estocásticos y LSTM. Logística, especialistas y Forex usan la semilla original 42. No se eligió una semilla ganadora. Las métricas finales son promedios de métricas por semilla, no un ensemble inventado; cada fecha se remuestrea conjuntamente entre semillas.",
        "LSTM conserva la elección de época por Sharpe neto de validación. B dispone también de secuencias antiguas y del contexto que faltaba al calentamiento inicial de A; sus secuencias, retornos y volatilidades compartidos con A se fijaron idénticos. Sus asignaciones no son probabilidades calibradas: log-loss, Brier y calibración probabilística no son aplicables sin cambiar la receta.",
        "**Experimento A histórico:** se cargaron y verificaron ocho modelos guardados y se recuperaron sus recetas originales, ventanas y semillas en registered_model_audit.json; historical_results.json conserva sus métricas reales. Seis se evaluaron en un tramo común válido de 15 observaciones (2025-06-05 a 2025-06-19), previa igualdad de sus features con el snapshot. Dos modelos ajustados hasta 2026 fueron excluidos por solapamiento temporal. No se usó ese tramo pequeño para elegir el mejor modelo ni se lo confundió con los folds principales.",
        table(
            [
                "Versión histórica",
                "Fin de selección",
                "Features",
                "AUC guardada",
                "Accuracy guardada",
                "Log-loss guardado",
            ],
            [
                [
                    m["metadata"]["version"],
                    m["metadata"]["selection_label_end"][:10],
                    len(m["metadata"]["features"]),
                    n(m["metadata"].get("test_metrics", {}).get("roc_auc")),
                    n(m["metadata"].get("test_metrics", {}).get("accuracy")),
                    n(m["metadata"].get("test_metrics", {}).get("log_loss")),
                ]
                for m in hist["registered_models"]
            ],
        ),
        "N/A en los metadatos históricos significa prueba pendiente, no resultado cero. Sus métricas corresponden a las fechas guardadas en splits y no son restables directamente contra las nuevas ventanas.",
        "## D. Comparación predictiva",
        "Todas las diferencias siguientes son B − A sobre las mismas observaciones. Accuracy se expresa como fracción; multiplicar su diferencia por 100 da puntos porcentuales. En log-loss, una diferencia negativa es favorable. La evaluación estadística de cada receta se basa en log-loss, el endpoint primario fijado antes de ajustar; no se intercambió por el indicador que saliera mejor.",
    ]
    for (symbol, interval, target), part in c.groupby(["symbol", "interval", "target_kind"]):
        text += [
            f"### {symbol} — {interval} — {target}",
            table(
                ["Modelo", "Métrica", "Antes A", "Después B", "Diferencia", "Evaluación"],
                [
                    [
                        r.recipe,
                        r.metric,
                        n(r.before),
                        n(r.after),
                        n(r.difference),
                        "log-loss respaldado, retrospectivo"
                        if inf[f"{r.symbol}|{r.interval}|{r.recipe}|{r.target_kind}"][
                            "statistically_supported_improvement"
                        ]
                        else "Sin mejora respaldada del endpoint primario",
                    ]
                    for r in part.loc[
                        part.metric.isin(["roc_auc", "accuracy", "log_loss"])
                    ].itertuples()
                ],
            ),
        ]
    text += [
        "La tabla completa con Balanced Accuracy, Brier, precision, recall, F1, ECE, diferencias relativas y número de fechas/semillas está en [comparison.csv](artifacts/retraining_comparison/20261017_controlled/comparison.csv). Las matrices y bins de calibración por semilla están en all_predictive_metrics.json. Net_event_profit mide beneficio de la operación, no si el precio sube al siguiente cierre; sus AUC no se comparan con close_direction.",
        "![Diferencias predictivas](artifacts/retraining_comparison/20261017_controlled/predictive_differences.png)",
        "![Calibración](artifacts/retraining_comparison/20261017_controlled/calibration.png)",
        "### Baselines y matrices de confusión",
        "constant_05 usa probabilidades 0,5 (log-loss 0,693147; su decisión con empate predice subida). train_prevalence y majority_class se obtienen exclusivamente de train. La moneda aleatoria con semilla 42 se presenta sólo como decisión, sin tratar sus salidas 0/1 como probabilidades ciertas. La logística existente controla capacidad predictiva simple. En Forex, price es también el control de compact.",
        table(
            [
                "Activo",
                "Receta B",
                "Control",
                "Ganancia log-loss vs control",
                "IC95 bloque largo",
                "p Holm controles",
            ],
            [
                [
                    name.split("|")[0],
                    name.split("|")[2],
                    name.split("|")[-1],
                    n(list(d["blocks"].values())[-1]["gain"]),
                    str([round(x, 5) for x in list(d["blocks"].values())[-1]["ci95"]]),
                    n(d["holm_p"]),
                ]
                for name, d in skill.items()
                if name.split("|")[2] in ["original", "fx_compact"]
            ],
        ),
        table(
            ["Activo", "Modelo, semilla 42", "Matriz A [[TN,FP],[FN,TP]]", "Matriz B"],
            [
                [
                    s,
                    r,
                    str(details[f"{s}|1d|{r}|close_direction|A|42"]["confusion_matrix"]),
                    str(details[f"{s}|1d|{r}|close_direction|B|42"]["confusion_matrix"]),
                ]
                for s in ["BTCUSDT", "ETHUSDT"]
                for r in ["original", "logistic", "random_forest_cross"]
            ],
        ),
        "## E. Comparación económica",
        "Spot: long/cash, señal al cierre, ejecución futura a apertura, 10 bps de comisión + 5 de slippage + 1 de medio spread por lado; sin posiciones solapadas. Mismo umbral 0,55 y protocolo de salida por receta. Se encadenan folds liquidando posiciones. Los resultados son promedios entre semillas; no una cartera que elige retrospectivamente la mejor semilla.",
        "Forex usa bid/ask observado, comisión 0,35 bps/lado, slippage 0,1 pip y swap adverso de 100 bps/año como escenario, no cotización histórica de un broker. El estrés duplica comisión/slippage/swap y añade 1 pip de spread; no se presenta como duplicación exacta del spread observado. Sharpe Forex se calcula sobre sesiones NY17 y 252 sesiones/año; spot sobre 365. Los precios son indicativos, sin garantía de fills, liquidez ni drawdown intravela.",
        "Las predicciones direccionales son close/close y la ejecución opera a aperturas: son objetivos distintos. Mejor Accuracy no implica mayor retorno. Los costos compuestos suman los flujos reales de cada fold escalados al capital encadenado. En Forex la exposición es tiempo en mercado; en spot la fracción de barras expuestas.",
        "### Retornos y riesgo de los modelos tabulares/de eventos",
    ]
    erows = []
    ops = []
    for (s, interval, r, _target), part in econ.groupby(
        ["symbol", "interval", "recipe", "target_kind"]
    ):
        a = part.loc[(part.arm == "A") & (part.scenario == "base")].mean(numeric_only=True)
        b = part.loc[(part.arm == "B") & (part.scenario == "base")].mean(numeric_only=True)
        stress = part.loc[(part.arm == "B") & (part.scenario == "double")].net_return.mean()
        erows.append(
            [
                s + " " + interval,
                r,
                pct(a.net_return),
                pct(b.net_return),
                f"{(b.net_return - a.net_return) * 100:+.2f} pp",
                n(a.sharpe, 3) + " → " + n(b.sharpe, 3),
                pct(b.max_drawdown),
                pct(stress),
            ]
        )
        ops.append(
            [
                s + " " + interval,
                r,
                n(b.trades, 1),
                pct(b.win_rate),
                n(b.profit_factor, 3),
                pct(b.exposure),
                n(b.execution_cost_cash_compounded, 2),
            ]
        )
    text += [
        table(
            [
                "Activo/TF",
                "Modelo",
                "Retorno A",
                "Retorno B",
                "Δ retorno",
                "Sharpe A → B",
                "MDD B",
                "Retorno B estrés",
            ],
            erows,
        ),
        table(
            [
                "Activo/TF",
                "Modelo B",
                "Operaciones",
                "Aciertos",
                "Profit Factor",
                "Exposición",
                "Costo ejecución, capital inicial 10.000",
            ],
            ops,
        ),
        "Los detalles de A/B, drawdown y costos por semilla y escenario se conservan en economic_comparison.csv y en los CSV de curvas y operaciones. En meta-labeling se comparan sólo folds con calibración disponible en ambos brazos; no se atribuye ese retorno a toda la historia.",
        "### LSTM/DMN y controles económicos",
        "La LSTM predice asignaciones de riesgo; su ROC-AUC siguiente es una medida del ranking de la asignación frente al signo del retorno open/open, no de una probabilidad de dirección close/close. Accuracy usa asignación ≥0,5 como regla descriptiva fija. No se calculó log-loss/Brier artificial a partir de pesos de cartera.",
    ]
    nd = pd.DataFrame(read_json(OUT / "neural_direction_metrics.json"))
    ndrows = []
    for s, g in nd.groupby("symbol"):
        a = g.loc[g.arm == "A"].mean(numeric_only=True)
        b = g.loc[g.arm == "B"].mean(numeric_only=True)
        ndrows.append(
            [
                s,
                n(a.roc_auc),
                n(b.roc_auc),
                n(b.roc_auc - a.roc_auc),
                pct(a.accuracy),
                pct(b.accuracy),
                "N/A",
            ]
        )
    text += [
        table(
            [
                "Activo",
                "AUC score A",
                "AUC score B",
                "Δ AUC",
                "Accuracy A",
                "Accuracy B",
                "Log-loss",
            ],
            ndrows,
        )
    ]
    nr = ei["BTC_ETH_PANEL|1d|lstm_sharpe_net|base"]["30"]
    ns = ei["BTC_ETH_PANEL|1d|lstm_sharpe_net|double"]["30"]
    text += [
        f"LSTM BTC/ETH, media de semillas: retorno {pct(nr['net_return_A'])} → {pct(nr['net_return_B'])} (Δ {nr['net_return_delta'] * 100:+.2f} pp); Sharpe {n(nr['sharpe_A'], 3)} → {n(nr['sharpe_B'], 3)}. IC95 de Δ retorno, bloques 30: [{nr['net_return_delta_ci95'][0] * 100:.2f}; {nr['net_return_delta_ci95'][1] * 100:.2f}] pp. Retorno B con costos dobles: {pct(ns['net_return_B'])}.",
        table(
            ["Control DMN, mismo panel", "Retorno neto base encadenado"],
            [
                [
                    name,
                    pct(
                        np.prod(
                            [
                                1 + d["metrics"]["total_return"]
                                for d in read_json(OUT / "neural_baselines.json")
                                if d["name"] == name and d["scenario"] == "base"
                            ]
                        )
                        - 1
                    ),
                ]
                for name in ["cash", "long_scaled", "tsmom", "macd"]
            ],
        ),
        "Los controles cash, long escalado por volatilidad, TSMOM y MACD se conservaron con curvas por fold en neural_baselines. La evaluación usa la adaptación spot existente; no es una réplica sobre futuros diversificados.",
        "## F. Significancia y robustez",
        "Bootstrap pareado de bloques móviles: 2.000 réplicas, bloques 7/30 observaciones diarias; H4 usa 42/180 barras para aproximar 7/30 días operados. No se cruzan folds y las semillas comparten el mismo remuestreo de fechas. Se informa el promedio de métricas por semilla y los intervalos son condicionales a los modelos ya ajustados. No incluyen incertidumbre de volver a entrenar ni son garantía de desempeño futuro.",
        "Prueba primaria: ganancia de log-loss A − B, alternativa positiva fijada previamente, bootstrap centrado. Se toma el mayor p entre bloques y se aplica Holm a todos los endpoints modelo/mercado/TF/objetivo. Accuracy, AUC y Brier tienen IC puntuales secundarios, no inferencia simultánea. Los controles forman otra familia Holm. Los intervalos económicos son exploratorios puntuales y no autorizan una selección. La multiplicidad histórica de investigaciones previas impide interpretar estos folds ya observados como confirmación independiente.",
        "Una ganancia numérica es sólo un punto favorable. Una ganancia estadísticamente respaldada exige el endpoint primario corregido. “Robusta” exigiría además estabilidad de signo, tamaño material, ambas semillas, varios mercados y confirmación independiente; esta última no está disponible.",
        table(
            [
                "Endpoint",
                "Ganancia log-loss",
                "IC95 bloque largo",
                "p Holm",
                "Folds con ganancia / pares",
                "Semillas evaluadas",
            ],
            [
                [
                    name,
                    n(list(d["inference"].values())[-1]["log_loss_gain"]),
                    str(
                        [
                            round(x, 5)
                            for x in list(d["inference"].values())[-1]["log_loss_gain_ci95"]
                        ]
                    ),
                    n(d["holm_p"]),
                    stability(f, name),
                    d["seeds"],
                ]
                for name, d in inf.items()
            ],
        ),
        "![Intervalos log-loss](artifacts/retraining_comparison/20261017_controlled/logloss_intervals.png)",
        "**Calibración no disponible:** cuatro políticas meta no cumplieron el mínimo. En BTC, A y B carecen del tercer fold; en ETH, B carece de los dos primeros. Se guardan sus auditorías y se excluyen esos pares del contraste predictivo; no se filtraron por resultado OOS. BTC-meta tiene 2023–2024; ETH-meta sólo primer semestre de 2025. No se reclama robustez temporal para esas comparaciones.",
        "El entrenamiento bruto se llegó a realizar en esos cuatro casos, pero no quedó una política calibrada válida; se incluyen en el conteo de 441 ajustes. Los modelos OOF y las políticas completas sí tienen checkpoints conservados. Los cuatro estimadores brutos descartados no se presentan como modelos completos disponibles.",
        "## G. Diagnóstico",
        "### Respuestas a las ocho preguntas",
        "1. **Mejoras numéricas:** CatBoost BTC/ETH, Random Forest BTC, LightGBM BTC y logística/estructura/SMC ETH presentan aumentos de AUC con menor log-loss. Ninguna mejora del endpoint primario queda estadísticamente respaldada tras Holm. En LightGBM BTC, Accuracy baja pese a mejorar AUC y log-loss.",
        "2. **Deterioros:** LightGBM ETH pierde 0,02689 de AUC y 1,37 pp de Accuracy; log-loss aumenta 0,00468. Los especialistas por reglas empeoran log-loss en ambos activos. Logística BTC pierde 1,94 pp de Accuracy pese a mejorar su backtest.",
        "3. **Cambios pequeños:** SMC BTC cambia AUC +0,00061 y estructura BTC +0,00091; ambos empeoran ligeramente log-loss. No se probó equivalencia estadística: sólo son diferencias puntuales pequeñas.",
        "4. **Mercados prometedores, con cautela:** BTC alcanza el mayor AUC direccional nuevo, 0,55319 en Random Forest; ETH alcanza 0,53518 con logística. En EUR/USD, los mejores AUC rondan 0,515. No hay superioridad corregida frente a los baselines fuertes en estos endpoints.",
        "5. **Temporalidades:** no hay evidencia que establezca una temporalidad ganadora. H4 EUR/USD no ofrece una mejora consistente; más barras no equivalen a más días independientes. Las cifras cripto D1 y Forex H4 representan horizontes/mercados diferentes.",
        "6. **Origen del cambio:** la comparación mantiene las recetas y observaciones compartidas; evalúa la ampliación histórica específica y la reparación nativa predeclarada. No informa sobre entrenamiento con 2025–2026 en cuarentena ni sobre transferencia a los activos no evaluados.",
        "7. **Superioridad a baselines:** no quedó ningún contraste de log-loss B frente a controles con p Holm <0,05. Esto no prueba ausencia absoluta de señal, pero impide afirmar una ventaja validada.",
        "8. **Utilidad económica:** hay backtests positivos, pero no una mejora robusta. Logística BTC pasa de +45,52% a +109,40%, aunque el IC95 de su diferencia incluye cero y su dirección empeora. El Random Forest BTC nuevo, mejor por AUC, pierde 7,08% neto; LSTM baja de +23,62% a +21,36%. Los costos dobles y las diferencias entre semillas impiden una recomendación operativa.",
        "La ampliación no es un multiplicador uniforme de señal. Aporta regímenes anteriores y modifica frecuencias de entrenamiento, particiones de árboles y coeficientes/calibración. Eso puede reducir varianza o introducir una distribución menos pertinente para la ventana posterior. Las diferencias por fold y semilla permiten observar esa inestabilidad; no identifican por sí solas un mecanismo causal de mercado.",
        "En esta comparación tabular se mantuvieron fijas arquitectura, hiperparámetros, columnas, umbrales, costos y todas las filas comunes. La diferencia es la historia agregada con su procedencia nativa declarada. No se atribuye el resultado a “millones de datos nuevos”, porque la mayor parte del inventario corresponde a otros mercados, otras temporalidades dependientes o fechas en cuarentena. Tampoco se confunde mejor calibración/log-loss con mejor ranking direccional o rentabilidad.",
        "La mezcla global/experto, los targets netos de Triple Barrier y el objetivo Sharpe de DMN tienen finalidades diferentes. Sus mejores cifras no son intercambiables. La escasez de ofertas OOF de meta-labeling evidencia que más registros no garantiza una calibración operativa suficiente.",
        "## H. Recomendaciones y decisión",
        "No activar ni promover un modelo usando este informe. Mantener los controles simples y congelar cualquier hipótesis que se quiera confirmar sobre un período futuro disjunto de las reservas existentes. Sólo merecen confirmación adicional las mejoras que superen controles fuertes y mantengan dirección, magnitud y costos en varias ventanas; no ampliar una búsqueda por una mejora aislada.",
        "Los resultados negativos deben permanecer disponibles. No repetir cientos de ajustes variando semillas, umbrales o features sobre estos mismos folds. Perpetuos, contratos e índices requieren primero sus contratos de ejecución/disponibilidad; añadirlos ahora impediría aislar el efecto de los datos.",
        "### Reproducción, recursos y preservación",
        "Equipo: 8 procesadores lógicos, 7.778.787.328 bytes de RAM (~7,24 GiB), ejecución CPU. El piloto XGBoost tardó 0,47 s A y 0,41 s B y confirmó 353 observaciones pareadas antes de ampliar. Árboles con 1 worker y LSTM con 2 threads; no se hizo una búsqueda masiva. La ampliación se decidió por integridad y costo del piloto, no por su score.",
        f"**Preservación verificada:** {verification['old_artifact_files_verified']:,} archivos anteriores comprobados por SHA-256; {len(verification['changed_old_artifacts'])} modificados. Versiones y reservas no se promovieron ni sobrescribieron. Todos los datos/predicciones/curvas/configuraciones nuevos están en artifacts/retraining_comparison/20261017_controlled. Artifacts está excluido de Git y requiere resguardo separado.",
        "**Software:** 55 pruebas dirigidas aprobadas; luego 4 pruebas de inferencia/contabilidad aprobadas tras completar el análisis. Ruff de los scripts nuevos aprobado. Los tests cubren purga/reservas, modelos existentes, ausencia de pseudorreplicación por semillas, Holm e inclusión del costo inicial al agregar curvas. No se confunden con una prueba de rentabilidad futura.",
        "En Windows PowerShell, los avisos PerformanceWarning del runner de eventos se envolvieron como NativeCommandError y el shell informó código 1. No hay traceback Python en ese log: se verificó la finalización de las 24 políticas planeadas (20 completas y 4 abstenciones auditadas), 72 checkpoints OOF y sus artefactos. No se ocultó ese estado del shell.",
        "Comandos (las carpetas son append-only; para una repetición independiente usar un nuevo identificador OUT en los scripts y conservar configuración/fuentes):",
        "```powershell\n.\\.venv\\Scripts\\python.exe scripts/retraining_comparison.py audit\n.\\.venv\\Scripts\\python.exe scripts/retraining_comparison.py pilot\n.\\.venv\\Scripts\\python.exe scripts/retraining_comparison.py full\n.\\.venv\\Scripts\\python.exe scripts/retraining_comparison.py events\n.\\.venv\\Scripts\\python.exe scripts/retraining_comparison.py neural\n.\\.venv\\Scripts\\python.exe scripts/retraining_comparison.py forex\n.\\.venv\\Scripts\\python.exe scripts/retraining_economic_inference.py\n.\\.venv\\Scripts\\python.exe scripts/analyze_retraining_comparison.py\n.\\.venv\\Scripts\\python.exe scripts/report_retraining_comparison.py\n```",
        "La modificación de alcance previa a cualquier ajuste está en protocol_amendment_before_fitting.json: EUR/USD representativo y expertos sólo semilla 42, límite 500. El código y entorno finales se conservan en el manifiesto de cierre; plan.json conserva el protocolo inicial y no fue sobrescrito. Las correcciones de contabilidad e inferencia posteriores no cambiaron modelos ni hiperparámetros ni escogieron resultados.",
        "Archivos principales: comparison.csv, fold_metrics.csv, baselines.csv, paired_inference.json, baseline_skill_inference.json, economic_comparison.csv, economic_inference.json, all_predictive_metrics.json, training_ledger.json, stored_model_suffix_evaluation.json y verification.json. Los hashes, ventanas, parámetros, semillas y predicciones por ajuste están junto a cada modelo.",
        "### Respuesta final verificable",
        "La respuesta es **no se demostró una mejora significativa**: 0/26 endpoints primarios y 0 contrastes corregidos contra controles resultaron favorables. No debe interpretarse como una demostración de equivalencia o de inexistencia absoluta de señal.",
        "Los mejores descriptivos en BTC diario, evaluados sobre las mismas 876 observaciones, son A: XGBoost con SMC (AUC 0,55249) y B: Random Forest cruzado (AUC 0,55319). La diferencia entre ganadores es +0,00071 de AUC; cambiar de receta en esa resta no aísla el efecto de añadir datos.",
        f"Comparación del **mismo mejor modelo anterior de BTC ({winner_a.recipe})**, sin cambiar de receta al calcular diferencias:",
        table(
            ["Métrica", "Antes A", "Después B", "B − A"],
            [
                [
                    m,
                    n(row("BTCUSDT", winner_a.recipe, m).before),
                    n(row("BTCUSDT", winner_a.recipe, m).after),
                    n(row("BTCUSDT", winner_a.recipe, m).difference),
                ]
                for m in ["roc_auc", "accuracy", "log_loss"]
            ],
        ),
        f"Mejor modelo nuevo descriptivo BTC: {winner_b.recipe}. El listado de mejoras respaldadas del endpoint log-loss es: "
        + (", ".join(supported) if supported else "**ninguno**")
        + ".",
        "La conclusión sobre dirección debe considerar además AUC/Accuracy y controles, no sólo log-loss. La conclusión económica debe considerar retorno neto, incertidumbre, estrés de costos y estabilidad. No hay confirmación prospectiva de una mejora robusta.",
        "Comparación del **mismo Random Forest BTC**: AUC 0,54202 → 0,55319 (Δ +0,01117), Accuracy 53,14% → 54,39% (Δ +1,26 pp), log-loss 0,68949 → 0,68788 (Δ −0,00161); retorno neto −10,82% → −7,08% (Δ +3,74 pp). IC95 de Δ AUC, bloque 30: [−0,01927; +0,02609]; IC95 de ganancia log-loss [−0,00340; +0,00429]; p Holm = 1. La mejora es numérica, no respaldada ni robusta ni rentable en este backtest.",
    ]
    report = Path("RETRAINING_COMPARISON_REPORT.md")
    with report.open("x", encoding="utf-8") as stream:
        stream.write("\n\n".join(text) + "\n")
    print(report.resolve())


def stability(frame, name):
    symbol, interval, recipe, target = name.split("|")
    g = frame.loc[
        (frame.symbol == symbol)
        & (frame.interval == interval)
        & (frame.recipe == recipe)
        & (frame.target_kind == target)
    ]
    w = g.pivot(index=["fold", "seed"], columns="arm", values="log_loss").dropna()
    return f"{int((w.B < w.A).sum())}/{len(w)}"


if __name__ == "__main__":
    main()
