# Investigación y optimización de features

Estudio ejecutado el **15/10/2026** sobre BTCUSDT y ETHUSDT diarios. La implementación está en `src/trading_ml/feature_research/` y el protocolo congelado en [config/feature_study.toml](config/feature_study.toml). El objetivo permanece idéntico al original: probabilidad de que el próximo cierre supere al cierre actual, horizonte de una vela.

**Resultado: ninguna mejora pasó los criterios. Se conserva el modelo y el contrato original.** La selección compacta redujo la log-loss, pero sólo mejoró simultáneamente loss/AUC en tres de seis folds, sus intervalos incluyen cero y no superó el baseline p=0,5. Los candidatos quedan como experimentos documentados, sin incorporarse a las predicciones de producción.

## Resultados de la corrida

Run `20261015T162842_a5404a4b`: **635 ajustes de un límite fijado de 650**, por debajo de la cota previa de 648 gracias a reutilizar modelos idénticos dentro de cada fold. Se realizaron 623 ajustes XGBoost —240 internos y 383 externos— y 12 logísticas. Se verificaron 430 artefactos y se reprodujeron **432 series de predicciones** desde modelos guardados, sin nuevos ajustes.

Datos reales desde 2020-01-01 hasta 2026-09-30: 2.465 velas por activo; 2.398 con todas las features de comparación disponibles. La evaluación externa utiliza **1.728 referencias comunes por activo**: 180/183/180/183/365/637 por fold. No se cambian las filas según la receta. El último fold es más largo: la métrica agregada pondera observaciones; la estabilidad cuenta folds y los gráficos muestran cada fold por separado.

| Activo | Receta | Features medias | Log-loss OOS ↓ | AUC OOS ↑ | Brier ↓ |
|---|---|---:|---:|---:|---:|
| BTC | Original | 14 | 0,703466 | 0,512768 | 0,254900 |
| BTC | Original sin correlaciones ≥0,90 | 9 | 0,702305 | 0,514669 | 0,254322 |
| BTC | Todas las familias | 39 | 0,705522 | 0,510413 | 0,255888 |
| BTC | Familia elegida dentro de train | 16,5 | 0,704165 | 0,518859 | 0,255101 |
| BTC | Selección compacta interna | 5,5 | 0,699809 | 0,518394 | 0,253208 |
| BTC | p=0,5 | — | **0,693147** | 0,500000 | **0,250000** |
| BTC | Logística L2 original | 14 | 0,694004 | 0,511660 | 0,250432 |
| ETH | Original | 14 | 0,705528 | 0,501349 | 0,255906 |
| ETH | Original sin correlaciones ≥0,90 | 9 | 0,707612 | 0,489731 | 0,256874 |
| ETH | Todas las familias | 39 | 0,704530 | 0,509072 | 0,255345 |
| ETH | Familia elegida dentro de train | 17 | 0,705383 | 0,500709 | 0,255837 |
| ETH | Selección compacta interna | 5,33 | 0,701537 | 0,502768 | 0,254097 |
| ETH | p=0,5 | — | **0,693147** | 0,500000 | **0,250000** |
| ETH | Logística L2 original | 14 | 0,703052 | 0,504180 | 0,254432 |

Semilla principal 42; medias de cantidad de features entre folds. La selección compacta eligió 2–8 variables en BTC y 2–9 en ETH. La segunda semilla mantiene las columnas seleccionadas y reproduce la mejora puntual de loss: ganancias de 0,003143 BTC y 0,004510 ETH. Tampoco pasa los requisitos de estabilidad, incertidumbre y AUC en ETH.

Intervalos simultáneos aproximados para compacto−original, bloques de 30 días, semilla 42:

| Activo | Δloss | Intervalo Δloss | ΔAUC | Intervalo ΔAUC |
|---|---:|---|---:|---|
| BTC | −0,003657 | [−0,020094; +0,012779] | +0,005625 | [−0,054451; +0,065702] |
| ETH | −0,003991 | [−0,019816; +0,011835] | +0,001419 | [−0,064737; +0,067575] |

Estos intervalos no respaldan una mejora distinguible de cero. El informe conserva también los bloques de siete días y todos los contrastes restantes.

La brecha media por fold entre loss OOS y train aumenta al pasar de original a completo: **0,05692 → 0,08636 en BTC** y **0,06299 → 0,07677 en ETH**. Compacto reduce las brechas a 0,04917 y 0,04730, respectivamente; eso es compatible con menor sobreajuste, sin establecer una ventaja predictiva suficiente.

## Información incremental encontrada y descartada

| Familia añadida al original | Δloss BTC | Folds BTC con menor loss | Δloss ETH | Folds ETH con menor loss |
|---|---:|---:|---:|---:|
| Momentum/tendencia | −0,001719 | 5/6 | −0,001010 | 3/6 |
| Volatilidad | −0,002337 | 4/6 | −0,000235 | 3/6 |
| Flujo | −0,002417 | 4/6 | +0,000619 | 3/6 |
| Reversión | −0,000517 | 3/6 | −0,000327 | 4/6 |
| Breakout | +0,001018 | 1/6 | −0,000262 | 4/6 |
| Estructura/SMC | −0,000216 | 4/6 | −0,000533 | 3/6 |
| Semanal/diario | −0,000548 | 3/6 | −0,001152 | 4/6 |
| BTC/ETH | +0,002425 | 3/6 | −0,000563 | 3/6 |

Estos efectos son descriptivos y no bastan para seleccionar una familia. La ablación desde el conjunto completo contradice varias mejoras aisladas: quitar momentum reduce la loss en ambos activos, aunque añadirlo al original la reduce. La contribución depende de otras variables y del ajuste del árbol; no es una propiedad intrínseca del indicador. Flujo ofrece una mejora puntual en BTC y empeora ETH. Agregar todas las relaciones cruzadas empeora BTC.

Eliminar redundancias tampoco se adopta automáticamente: pasar de 14 a nueve features mejora ligeramente BTC y empeora ETH. En este modelo, una correlación alta no equivale a una ablación inocua.

`body_ratio` recibe gain medio 0,0920 BTC y 0,0867 ETH, entre los mayores del original. Sin embargo, su permutación externa media con bloques 30 produce Δloss **−0,000819 BTC / −0,000426 ETH**; `range_ratio` produce **−0,001168 / −0,001996**. Son señales de uso en train sin utilidad externa media, sujetas a dependencia entre features. Ni siquiera las variables con permutación media positiva mantienen el signo en todos los folds: no hay una lista estable respaldada por esa medida por sí sola.

Los cambios de condición también importan: el compacto pierde frente al original en el régimen de volatilidad alta, tanto en BTC (loss 0,70139 contra 0,69986) como ETH (0,70451 contra 0,69775). En BTC bear también empeora (0,70404 contra 0,69718), aunque mejora bull. Son diagnósticos posteriores; no se construyó un filtro con esas observaciones. PSI muestra drift considerable en el último fold, por ejemplo 1,4540 para MACD signal BTC y 0,4692 para volatilidad ETH; los bins proceden exclusivamente de train y no se usan para afirmar causalidad ni para cambiar retrospectivamente la receta.

Informe completo: [report.md](artifacts/feature_studies/20261015T162842_a5404a4b/report.md). Métricas, intervalos y rechazos: [summary.json](artifacts/feature_studies/20261015T162842_a5404a4b/summary.json). Verificación sin entrenamiento: [verification.json](artifacts/feature_analyses/20261015T162842_a5404a4b/verification.json).

![Comparación por fold](artifacts/feature_analyses/20261015T162842_a5404a4b/fold_comparison.png)

![Importancia por permutación fuera de muestra](artifacts/feature_analyses/20261015T162842_a5404a4b/original_permutation.png)

Los gráficos usan media simple de folds; la tabla principal agrega todas las observaciones. No se infiere rentabilidad a partir de estas métricas: la ronda conserva dirección close/close, ejecución y costos previos, y no añade una simulación económica con parámetros elegidos después de ver resultados. No se alteraron modelos, punteros ni reservas prospectivas.

SHA-256 del manifiesto de la corrida: `979ce10b35beea1e2568b6b7d8b44e01cd3a0cddedab5163e3f95c3e2bd14fc0`. SHA-256 del plan: `b6e66b84b1fc2eaddd385bcc1dbb68c0769adad3d58cd9c9937039163138df6c`.

## Auditoría del sistema existente

El contrato original contiene 14 variables: retornos 1/3/7, rango, cuerpo, volumen relativo, RSI, MACD/signal/histograma, volatilidad y distancias SMA 10/20/50. `enhanced` añade ocho variables; SMC original añade 32. Las rondas anteriores ya investigaron momentum, reversión, breakouts, flujo taker, BTC/ETH y varios modelos. Sus resultados y reservas siguen intactos.

La construcción original es causal al cierre: rolling/ewm no centrados y sin escalado global. SMC confirma un pivot de la vela j sólo en j+right; no lo escribe retrospectivamente en j. La nueva ronda reutiliza esos cálculos en vez de crear otra definición incompatible.

Se detectaron redundancias concretas:

- `macd = macd_signal + macd_hist` es una identidad algebraica con la normalización actual. La correlación por pares no detecta necesariamente toda dependencia multivariada.
- En estas velas diarias, `body_ratio` y `return_1` tienen Spearman aproximadamente 0,99999: la apertura casi coincide con el cierre anterior. Son casi duplicados en este mercado, aunque no una identidad general entre proveedores.
- En todos los trains externos, el filtro a 0,90 también relaciona MACD/signal, retorno 7/SMA 10 y RSI/SMA 20/50. El filtro determinista deja nueve variables originales. Esa redundancia estadística no demuestra que eliminarlas mejore XGBoost.
- El CSV OHLCV habitual descarta campos taker y número de trades. Se recuperan desde los ZIP originales verificados; no se infiere presión compradora de velas alcistas ni se inventan datos de order book.

La importancia `gain` explica uso dentro del entrenamiento. Para valorar generalización se añade permutación sobre datos posteriores y ablación con reentrenamiento, de acuerdo con la [documentación de permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html). La correlación puede ocultar contribuciones individuales; por eso se incluyen permutaciones conjuntas de familias y el [análisis de multicolinealidad](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html).

## Hipótesis acotadas

Son **25 variables candidatas en ocho familias**, además de las 14 originales. El control completo de 39 variables existe para medir interacciones y ablaciones; no se presume que acumular variables mejore el modelo.

| Familia | Variables | Hipótesis incremental |
|---|---|---|
| Momentum y tendencia | Aceleración de retornos logarítmicos 3 contra los 3 anteriores, eficiencia del recorrido 20, momentum 60 ajustado por volatilidad | Cambios de velocidad, persistencia y fuerza de tendencia más allá de retornos cortos |
| Volatilidad y régimen | RV 5/60, Parkinson 20/RV 20, variabilidad de RV 5 en 20 sesiones | Compresión, expansión y desacuerdo entre variación intravela y cierre/cierre |
| Flujo | Desequilibrio taker diario, presión 3 contra presión previa 20, tamaño medio de trade relativo | Innovación en agresión compradora/vendedora y composición del volumen |
| Mean Reversion | Z-score logarítmico contra media/desviación previas 20, distancia a VWAP agregado 20 en unidades de volatilidad | Extensión respecto de referencias de precio y volumen |
| Volatility Breakout | Distancias a máximo/mínimo previos 20 en ATR previo 14; rango actual/ATR multiplicado por compresión previa | Ruptura de niveles anteriores y salida de compresión |
| Estructura/SMC | Estructura confirmada, sweep low−high, BOS up−down, posición entre swings confirmados | Rupturas y rechazos cuantificados sin pivotes conocidos anticipadamente |
| Marcos temporales | Momentum semanal 4 normalizado por RV semanal 8; distancia a EMA semanal 8; desacuerdo diario/semanal | Contexto de tendencia superior completamente cerrado |
| BTC/ETH | Retorno del otro activo, retorno relativo 20, retorno residual con beta previa 60, cambio correlación 20−60 | Componente común, fuerza relativa y cambio de acoplamiento |

Sea r el retorno logarítmico y σ20 su desviación de 20 velas: la aceleración se normaliza por `σ20 * sqrt(6)`; eficiencia 20 es `abs(sum(r,20))/sum(abs(r),20)`. Parkinson utiliza `sqrt(mean(log(H/L)^2,20)/(4*log(2)))`. El residual es `(r_t − beta_(t−1)*r_peer,t)/sigma_(t−1,60)`. No se atribuye cointegración a un cociente BTC/ETH.

Los regímenes descriptivos bull/bear/sideways utilizan momentum 60 normalizado con umbrales ±1 fijados previamente. El régimen de volatilidad alta compara σ20 con su percentil 80 de las 120 observaciones anteriores. PSI usa bordes aprendidos en train. Estos diagnósticos no reajustan el modelo tras observar su evaluación.

La literatura de [time series momentum](https://fairmodel.econ.yale.edu/ec439/mosk.pdf) motiva hipótesis de persistencia en otros instrumentos y horizontes; no demuestra capacidad de predecir la dirección diaria de BTC/ETH. Los conceptos SMC se tratan como patrones OHLC cuantificables: estos datos no identifican participantes institucionales.

## Disponibilidad real y ausencia de leakage

`reference_time` es el final UTC de una vela diaria. Una predicción requiere recibir ambas velas finalizadas y sus campos de volumen. Las variables no presuponen conocer el cierre siguiente.

- Los agregados semanales sólo se publican el lunes UTC posterior a siete velas completas. La primera semana parcial se descarta. Se verifica que `weekly_available_time <= reference_time` y no se arrastra una semana más allá de su período correspondiente.
- BTC y ETH se unen por referencia exacta. Un peer ausente produce NaN y abstención por falta de features; no se arrastra su último precio. El estudio usa las mismas filas disponibles para todos los modelos.
- Extremos de ruptura, ATR de normalización y distribución del z-score usan ventanas previas. La beta del residual termina en t−1.
- Las columnas del modelo proceden de listas explícitas. `target`, `target_time`, precios futuros y máscaras basadas en desenlaces nunca son features.
- La última fila conserva features para inferencia y carece de etiqueta. Los tests alteran datos futuros y truncan el historial, incluida una semana parcial, para verificar invariancia de todo el pasado.
- El estudio rechaza ventanas reservadas antes de cargar precios y comprueba hashes de modelos, punteros y reservas al finalizar.

## Experimentos controlados

Se fijaron **29 recetas principales**: original; ocho adiciones individuales; completo; ocho exclusiones del completo; cinco exclusiones de familias originales; controles enhanced/SMC/local de rondas previas; original filtrado por correlación; selección temporal de familia y selección compacta por permutación. Hay tres baselines adicionales: p=0,5, prevalencia de train y regresión logística L2 con escalado de train.

Todas las recetas usan las mismas filas, etiquetas, semilla principal y parámetros de la referencia registrada: 160 árboles, profundidad 2, learning rate 0,03, lambda 5, min_child_weight 10 y subsampling de filas/columnas 0,85. Se reajusta la receta original antes de cada fold; el modelo registrado, entrenado posteriormente a varios folds, no se aplica hacia atrás en el tiempo.

Seis folds externos expansivos por activo, con límites 2022-01-01, 2022-07-01, 2023-01-01, 2023-07-01, 2024-01-01, 2025-01-01 y 2026-10-01. Se purgan etiquetas por `target_time` y se añade embargo de seis días. Cada train externo contiene dos validaciones internas disjuntas de 120 días, también purgadas y con embargo.

La selección interna compara original, cada familia añadida y completo. Exige mejora en ambas ventanas y al menos 0,001 de ganancia media de log-loss para elegir otra receta. La selección compacta exige permutación individual positiva en las dos ventanas y ambos bloques, elimina correlaciones de train ≥0,90 y limita a 20 features; sin dos variables válidas vuelve al original. Ni el fold externo ni sus importancias participan de esa decisión. La segunda semilla, 1729, repite original, completo y ambos procedimientos de selección, manteniendo fijas sus columnas.

Permutation importance reordena bloques contiguos de 7/30 días dentro de cada ventana y permuta cada familia conjuntamente. Ocho repeticiones externas y cinco internas. La variación entre permutaciones no se presenta como intervalo de confianza. Al poder crear combinaciones improbables entre features correlacionadas, se interpreta junto con las ablaciones y el rendimiento real del modelo.

Bootstrap pareado no circular por bloques de 7/30 días, dentro de cada fold, con 2.000 remuestras. Se usan intervalos aproximados simultáneos max-|t| sobre pares, métricas y bloques de cada activo, con alpha/2 por activo. BTC y ETH no se tratan como réplicas independientes. Se exige ganancia de log-loss ≥0,003, ganancia AUC ≥0,005, mejora conjunta en al menos cuatro de seis folds e intervalos favorables frente al original; los candidatos principales también deben superar la loss de los tres baselines, ambos activos y la segunda semilla.

Este ajuste estadístico cubre los contrastes declarados de esta ronda, condicionado a las predicciones históricas. No corrige todas las decisiones de investigación anteriores del proyecto. El historial ya observado sigue siendo desarrollo; no se presenta como un test prospectivo nuevo. La [investigación sobre backtest overfitting](https://carmamaths.org/jon/backtest2.pdf) explica por qué ampliar búsquedas sobre el mismo historial aumenta falsos descubrimientos. No se calculó PBO/CSCV ni se afirma haberlo hecho.

## Reproducción

```powershell
.\.venv\Scripts\python.exe -m trading_ml feature-study --protocol config/feature_study.toml
# Verificar una corrida guardada y exportar gráficos, sin nuevos ajustes:
.\.venv\Scripts\python.exe scripts/record_feature_study.py artifacts/feature_studies/20261015T162842_a5404a4b
```

Se requieren los CSV y ZIP BTC/ETH existentes con sus checksums; el estudio no los sustituye por fixtures. Datos y esquema oficial: [Binance public data](https://github.com/binance/binance-public-data). La fuente de las features nuevas es el mismo historial de los controles. Los artefactos se crean en una carpeta nueva por corrida y `artifacts.json` conserva sus hashes.

`plan.json` registra protocolo, entorno, hashes de código y procedencia antes del primer ajuste. `predictions.csv` conserva las probabilidades externas; `selection.json`/`selections.json`, cada decisión interna; `fits.json`, columnas, métricas train/OOS, gain y modelo; `importance.json`, permutaciones; `diagnostics.json`, correlaciones de train y drift; `summary.json`, intervalos, métricas por régimen/semilla y motivos de rechazo. El verificador reconstruye features y etiquetas desde archivos verificados y reproduce predicciones guardadas sin entrenar.

Los gráficos requieren Matplotlib, como los scripts de informes anteriores. El comando `feature-study` y sus tests sólo usan las dependencias existentes del motor. La construcción de features queda en el espacio de investigación hasta que exista evidencia suficiente para cambiar un contrato de producción versionado.

Validación final: **189 tests pasaron**, incluyendo 17 nuevos de causalidad, alineación, selección temporal, incertidumbre e integridad. Ruff check, formato de los 126 archivos Python y `pip check` también pasaron. Registro completo de comandos, versiones, hashes y salidas en [checks.json](artifacts/feature_analyses/20261015T162842_a5404a4b/checks.json).
