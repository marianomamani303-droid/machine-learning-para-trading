# Plan de investigación: estructura y regímenes

Protocolo definido el 16/10/2026 antes de ejecutar ajustes nuevos. Estudio histórico exploratorio: estos datos ya se observaron en muchas rondas; ningún resultado constituye confirmación independiente. No se abrirán reservas prospectivas ni se promoverán modelos.

## Auditoría y reutilización

Se revisaron contratos, construcción de variables, selección temporal, simulación, persistencia y estudios previos:

| Área | Implementación existente | Decisión |
|---|---|---|
| Original | `features/technical.py`, 14 variables; XGBoost y contrato versionado | Refit de la receta registrada en las mismas filas de cada fold; modelo registrado intacto |
| Estructura | `smc/detectors.py`: pivots confirmados j+2, BOS/CHoCH únicos por nivel, OB nacidos en confirmación, touch e invalidación, FVG y sweeps | Reutilizar las 32 variables y los eventos; no crear otro detector |
| Reglas SMC | `smc/rules.py`, `training/smc_study.py`; siete ventanas históricas sin evidencia consistente | Control existente, sin reinterpretar score heurístico como probabilidad calibrada |
| Regímenes | `momentum/features.py:causal_regimes`, `feature_research/features.py`: dirección normalizada trailing 60 y volatilidad trailing | Reutilizar dirección; ampliar volatilidad a tres niveles con cuantiles pasados |
| Condicionamiento previo | `frontier_research/study.py`: prior por régimen; advanced: filtros trend/volatility; diagnósticos por régimen | Ya existen priors y filtros; falta comparar especialistas entrenados por estado con global y global+estado |
| Features investigadas | Feature study: 635 ajustes, estructura incremental débil; advanced, momentum, reversión, labels y frontier | No repetir búsqueda de familias, algoritmos o hiperparámetros |
| Validación | Walk-forward purgado, embargo, baselines escalados en train, bootstrap por bloques, reservas protegidas | Reutilizar helpers temporales y estadísticos |
| Ejecución | `backtesting/simulator.py`: próxima apertura, h=1, long/cash sin solapamiento | Reutilizar; agregar medio spread al slippage mediante configuración local |
| Patrimonio previo | Modelos nativos, metadatos, punteros, datasets y resultados en `artifacts/` | Hash de todos los archivos previos antes/después, escrituras sólo en directorio nuevo |

Limitaciones encontradas: soporte/resistencia se representa por distancias a swings, pero falta rechazo/retest explícito de niveles rotos. OB ya requiere impulso y tiene touch, pero falta confirmación del touch mediante impulso. Los diagnósticos por régimen no equivalen a especialistas. El simulador original no separa spread; los costos son supuestos constantes, no cotizaciones históricas. Labels close/close y ejecución open/open tienen objetivos distintos. Un régimen trailing describe el presente y puede cambiar; no es una clasificación verdadera observada ni una predicción garantizada. La suite existente se ejecuta antes y después del cambio.

## Experimentos predefinidos

1. Original refit; control original+SMC existente; original+SMC+extensiones. Las extensiones utilizan exclusivamente eventos existentes: niveles de swings confirmados, rechazo de soporte/resistencia anterior, primer retest válido dentro de 20 velas de BOS/CHoCH, cuerpo/ATR previo y cierre en extremo para confirmar impulso, touch OB confirmado por impulso. Se conserva la fecha de confirmación y nunca se reescribe la vela fuente. Además, filtro de entradas original con confirmación alcista, sin alterar su predicción probabilística.
2. Global original; global+one-hot de dirección y volatilidad; especialistas en las nueve intersecciones dirección×volatilidad. Alternativa acotada: KMeans de tres estados sobre momentum/eficiencia/volatilidad existentes, con scaler y centroides ajustados sólo en train; global+cluster y especialistas por cluster. Una configuración fija, sin selección de K ni de umbrales. Expertos con menos de 80 filas o 20 ejemplos de una clase usan el global; en otro caso mezcla `n/(n+200)` experto + resto global. Se documentan tamaños, fallback y transiciones.
3. Sólo si estructura y al menos un método de régimen satisfacen TODOS los criterios históricos en ambos activos y semillas: original+estructura+one-hot del régimen elegible. Se compara contra original, estructura y régimen en las mismas filas. Si no, se registra explícitamente «no ejecutado por falta de evidencia»; no se rellena una cuarta fila con resultados ficticios. La combinación será exploratoria por la selección histórica previa.

## Diseño común y presupuesto

- BTCUSDT y ETHUSDT diarios verificados, 2020-01-01 a 2026-09-30. Objetivo original h=1: próximo cierre mayor que cierre actual.
- Fronteras 2022-01-01, 2022-07-01, 2023-01-01, 2023-07-01, 2024-01-01, 2025-01-01, 2026-10-01. Train expansivo; purga por target_time y embargo de seis días; filas comunes para todas las recetas. Sin tuning: hiperparámetros congelados del original; dos semillas 42/1729. Ningún early stopping o scaler toca el fold evaluado.
- Hasta 500 ajustes, incluyendo clustering y logística; máximo teórico 480 con combinación. No se agregan variantes tras observar resultados. El plan JSON se escribe antes del primer ajuste.
- Accuracy, ROC-AUC, log-loss y Brier en todas las observaciones OOS; métricas por fold y régimen; brecha train/OOS.
- Ejecución h=1 próxima apertura, umbral fijo 0,55, capital reinvertido, long/cash. Igual calendario con/sin filtro. Comisiones 10 bps/lado, slippage 5 bps/lado y spread completo 2 bps. Escenarios sin spread (compatibilidad histórica) y costos dobles son sensibilidad; sólo base determina comparación principal. Retorno compuesto, drawdown con capital inicial, operaciones, equity y ledger persistidos.
- Baselines p=0,5, prevalencia y logística L2 con scaler sólo train. Comparación del modelo registrado únicamente en su propio mercado y referencias posteriores a selección, separada del refit para evitar contaminar ventanas antiguas.
- Bootstrap pareado no circular dentro de folds, bloques de 7/30 días, 2.000 remuestras. Intervalos max-|t| simultáneos para todos los pares/accuracy/AUC/loss, y familia económica separada para retorno/drawdown; Bonferroni entre dos activos, dos semillas y dos familias. Brier se informa como métrica descriptiva. La familia incluye controles y comparaciones reservadas para combinación.
- Aceptación: mejora loss ≥0,003 y AUC ≥0,005; mejora conjunta en al menos 4/6 folds; intervalos loss<0 y AUC>0 en ambos bloques; superar loss de los tres baselines; retorno neto positivo, al menos 30 operaciones, retorno incremental con intervalo>0 y drawdown no peor; mantener retorno positivo a costos dobles. Se exige en ambos activos y semillas. No se acepta ahorro de pérdidas por quedarse casi siempre en efectivo.

## Fuentes y alcance

[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) documenta orden temporal y gap; aquí la purga usa disponibilidad real de cada etiqueta, más embargo. [Statsmodels Markov switching](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_regression.html) distingue inferencia de estados: smoothing utiliza observaciones posteriores; se descarta para features online. HMM/Markov quedan como alternativa investigada, sin añadir otra búsqueda ante la evidencia histórica débil. [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) aporta una alternativa disponible sin dependencia nueva; sus estados no tienen semántica económica garantizada. [Bailey y López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) motivan el control de selección múltiple; no se presenta un Sharpe seleccionado como prueba. El bootstrap de esta ronda no corrige todas las decisiones históricas del proyecto.

Las definiciones SMC son operativas, no universales; OHLCV no demuestra actividad institucional. Se documentarán resultados favorables y adversos, incluidos fallbacks, escasez de eventos y combinación omitida.
