# Estructura de mercado y modelos condicionados por régimen

Investigación del 16/10/2026. Plan previo: [STRUCTURE_REGIME_PLAN.md](STRUCTURE_REGIME_PLAN.md). Protocolo ejecutable: [config/structure_regime.toml](config/structure_regime.toml). Implementación aislada: `src/trading_ml/structure_regime/`. No cambia los contratos, detectores, modelos o predicciones de producción.

**Resultado: ninguna variante satisface los criterios históricos de aceptación. No se reemplaza ningún modelo. La combinación no se ejecuta porque los experimentos 1 y 2 no justifican continuar.** Hay mejoras puntuales de clasificación, especialmente SMC en BTC, que no superan la incertidumbre ni se convierten en rentabilidad neta.

## Resultados ejecutados

Run `20261016T152046_8f7d1886`: **360 ajustes de un límite de 500**, frente a una cota teórica de 480 incluyendo la combinación opcional. Incluye 312 XGBoost, 24 KMeans y 24 logísticas. Los fallbacks evitan ajustes sin muestra suficiente. Dos activos × seis folds × dos semillas, sin ajustes adicionales después de observar los resultados.

Se utilizaron 2.465 velas reales por activo; 2.325 disponen de todas las variables comunes. La evaluación suma **1.728 observaciones por activo y semilla**, distribuidas en 180/183/180/183/365/637 por fold. La comparación principal siguiente usa semilla 42, costos base y reinversión entre ventanas. El [informe completo](artifacts/structure_regime_studies/20261016T152046_8f7d1886/report.md) incluye ambas semillas y todos los baselines.

| Activo | Variante | Accuracy | AUC | Log-loss | Brier | Neto | DD | Operaciones |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTC | Original refit | 51,33% | 0,5147 | 0,701408 | 0,253936 | −83,17% | −85,36% | 524 |
| BTC | Control SMC existente | 52,31% | 0,5283 | 0,698622 | 0,252558 | −85,96% | −88,20% | 525 |
| BTC | Estructura con extensiones | 52,72% | 0,5264 | 0,699337 | 0,252910 | −86,92% | −89,73% | 554 |
| BTC | Global + régimen por reglas | 51,56% | 0,5130 | 0,701695 | 0,254079 | −87,26% | −89,17% | 517 |
| BTC | Especialistas por reglas | 50,52% | 0,5122 | 0,704291 | 0,255316 | −84,79% | −86,38% | 591 |
| BTC | Global + cluster | 51,91% | 0,5188 | 0,701106 | 0,253756 | −78,15% | −81,44% | 513 |
| BTC | Especialistas por cluster | 52,03% | 0,5137 | 0,703989 | 0,255152 | −87,14% | −89,14% | 555 |
| ETH | Original refit | 51,10% | 0,4972 | 0,707697 | 0,256933 | −93,71% | −94,98% | 543 |
| ETH | Control SMC existente | 49,88% | 0,4985 | 0,704532 | 0,255492 | −85,77% | −87,94% | 541 |
| ETH | Estructura con extensiones | 49,83% | 0,4979 | 0,705017 | 0,255733 | −93,18% | −94,20% | 564 |
| ETH | Global + régimen por reglas | 51,04% | 0,4979 | 0,707097 | 0,256649 | −91,74% | −92,93% | 535 |
| ETH | Especialistas por reglas | 49,54% | 0,4961 | 0,710403 | 0,258213 | −94,76% | −95,21% | 638 |
| ETH | Global + cluster | 50,87% | 0,4963 | 0,707385 | 0,256793 | −93,79% | −94,87% | 545 |
| ETH | Especialistas por cluster | 50,06% | 0,4968 | 0,711458 | 0,258725 | −95,77% | −96,14% | 615 |

En todos estos casos la loss supera **0,693147 de p=0,5**, cuyo Brier es 0,25. La logística obtiene loss 0,694170 BTC y 0,704969 ETH; sus retornos también son negativos. No se interpreta una mejora frente a un original deficiente como ventaja suficiente.

### Experimento 1: mejora puntual, sin aceptación

Estructura reduce loss en BTC 0,002071 y aumenta AUC 0,011618; mejora simultáneamente ambas métricas en 5/6 folds. La segunda semilla conserva ganancias 0,002160 y 0,010736, con 4/6 folds. Esto es un resultado favorable descriptivo, pero la reducción de loss no alcanza el mínimo 0,003 y no supera los intervalos corregidos ni los baselines. El retorno base empeora en BTC con semilla 42 y sigue siendo negativo con ambas semillas.

En ETH la ganancia de AUC es sólo 0,000664 y la mejora conjunta ocurre en 3/6 folds; la segunda semilla obtiene +0,000328 de AUC. Las nuevas extensiones tampoco mejoran la loss puntual del **control SMC ya existente** en la semilla principal: la empeoran 0,000715 BTC y 0,000485 ETH. Ese contraste es descriptivo, sin afirmar una diferencia estadística entre ambos. No se justifica incorporar las once variables nuevas al contrato de producción.

Intervalos simultáneos aproximados para estructura−original, bloques de treinta días, semilla 42:

| Activo | Δloss | Intervalo Δloss | ΔAUC | Intervalo ΔAUC |
|---|---:|---|---:|---|
| BTC | −0,002071 | [−0,007919; +0,003777] | +0,011618 | [−0,013178; +0,036414] |
| ETH | −0,002680 | [−0,008567; +0,003207] | +0,000664 | [−0,021654; +0,022983] |

Los intervalos incluyen cero; los bloques de siete días tampoco sostienen aceptación. La brecha media train/OOS de loss, promediando folds y semillas, pasa de 0,05718 a 0,05860 en BTC y de 0,06841 a 0,06942 en ETH al añadir estructura completa. No hay evidencia de que sumar estas variables resuelva el sobreajuste.

El filtro de confirmaciones toma sólo **6 trades BTC / 8 ETH** con semilla 42: netos −3,63% / −12,05%, DD −6,21% / −12,36%. Con semilla 1729 produce +4,57% BTC con siete trades y −9,36% ETH con siete. Se documenta el caso positivo, pero no cumple la muestra mínima ni se replica entre activos. La pérdida mucho menor respecto del original está asociada con una reducción extrema de actividad; no demuestra mejor predicción.

En todo el historial con features disponibles se observaron 70/72 primeros retesteos alcistas de ruptura en BTC/ETH y sólo 16/16 confirmados por impulso. Estos conteos describen escasez de eventos, no el número de operaciones independientes ni una estimación de rentabilidad.

### Experimento 2: especialización no justificada

Añadir reglas cambia muy poco la loss y no mejora de manera consistente ambos activos y semillas. Global+cluster reduce la loss BTC sólo 0,000302 con semilla 42; con semilla 1729 la empeora 0,000048. En ETH ocurre el mismo cambio de signo entre semillas. La menor pérdida de BTC con clusters (−78,15% frente a −83,17%) no alcanza retorno positivo ni evidencia estadística de ventaja.

Los especialistas por reglas y por clusters empeoran la loss agregada en ambos activos y ambas semillas. Dividir la muestra no revela aquí una ventaja de especialización. Sumando ambos seeds, 636/3.456 pronósticos BTC y 684/3.456 ETH de especialistas por reglas usan fallback por muestra insuficiente; las demás observaciones siguen contrayéndose hacia el global. Los expertos de clusters casi siempre pueden ajustarse, pero tampoco mejoran: disponer de filas suficientes no garantiza señal.

Con costos dobles, original/estructura/global+reglas/global+cluster retornan respectivamente **−96,85% / −97,78% / −97,56% / −95,77% en BTC** y **−98,89% / −98,88% / −98,51% / −98,91% en ETH**, semilla 42. El escenario sin spread mantiene resultados negativos para todas esas recetas. No se ajustó el umbral ni la ejecución para rescatar una variante.

### Experimento 3: condición de continuación incumplida

| Variante solicitada | Estado |
|---|---|
| Baseline actual, reentrenado temporalmente | Ejecutado |
| Estructura solamente, añadida al baseline | Ejecutado |
| Régimen solamente, por reglas y clusters, global y especialistas | Ejecutado |
| Estructura + régimen | **No ejecutado:** ni estructura ni régimen satisfacen los filtros previos |

La rama de combinación está implementada y presupuestada, con prioridad determinista del método de régimen y comparaciones emparejadas. No se consumieron ajustes para producir una cuarta cifra después de que los experimentos anteriores fueran rechazados.

### Comparación separada con el modelo guardado

Sólo BTC, **482 referencias del 05/06/2025 al 29/09/2026**, posteriores al final de información de selección del original. Entrada/salida ejecutable entre 05/06/2025 y 30/09/2026; mismo calendario y costos base para todos. Este segmento ya se observó en otras investigaciones.

| Modelo | AUC | Log-loss | Neto | DD | Trades |
|---|---:|---:|---:|---:|---:|
| Original registrado intacto | 0,4782 | 0,707467 | −46,57% | −48,07% | 116 |
| Original refit | 0,4798 | 0,704905 | −35,46% | −44,83% | 97 |
| Estructura | 0,5119 | 0,699340 | −27,41% | −44,89% | 105 |
| Global + reglas | 0,4842 | 0,703585 | −36,44% | −46,44% | 94 |
| Global + cluster | 0,4814 | 0,705906 | −23,11% | −39,27% | 87 |

Las pérdidas son menores en algunos candidatos, sin rentabilidad positiva ni prueba independiente. Estos números difieren del test publicado original por la fila final excluida para ejecutar dentro del calendario común y por añadir spread. El JSON conserva accuracy/Brier y todos los demás candidatos para este mismo segmento. No se usó el modelo guardado para producir falsas predicciones OOS en 2022–2024.

![Comparación histórica de clasificación y capital](artifacts/structure_regime_analyses/20261016T152046_8f7d1886/comparison.png)

## Evidencia, preservación y controles

- [Plan JSON anterior a los ajustes](artifacts/structure_regime_studies/20261016T152046_8f7d1886/plan.json), [summary con todos los criterios e intervalos](artifacts/structure_regime_studies/20261016T152046_8f7d1886/summary.json) y [manifiesto SHA-256](artifacts/structure_regime_studies/20261016T152046_8f7d1886/artifacts.json).
- **34.042 archivos anteriores conservados byte por byte**, incluidos datasets, modelos, resultados y reservas. Hash conjunto de preservación: `a5661e8d454edd3e6f5d2a9827d22b8b1837173ea6e294abe219dfa482d796a8`. No se abrió ninguna reserva ni se promovió un candidato.
- [Verificación independiente del almacenamiento](artifacts/structure_regime_verifications/20261016T152046_8f7d1886.json): 372 hashes de artefactos nuevos, 240 series de predicciones reconstruidas con error máximo 1,23×10⁻¹⁵, 132 series económicas comprobadas y 429 archivos de modelos/reservas revalidados; cero ajustes nuevos.
- Suite completa: **250 pruebas aprobadas** (242 existentes y ocho nuevas); Ruff, formato y `pip check` aprobados. Las nuevas pruebas incluyen invariancia al prefijo y mutación futura de OHLCV, nacimiento/retest, clustering sólo train, etiquetas externas adulteradas, fallback, costos y calendario de abstención.
- SHA-256 del manifiesto de la corrida: `9d8f0d5e5f4087cf62225c8cf08e96f8c8513fb9a222df8dcc0775e3d6d08277`.

## Auditoría profunda: lo que ya existía

El original tiene 14 features, XGBoost de profundidad elegida en validación cronológica y horizonte de una vela. Enhanced suma ocho features y SMC suma 32, con contratos versionados. Se verificaron las listas explícitas de features: los datasets llevan etiquetas futuras para evaluar, pero esas columnas no se entregan a los estimadores. RSI/EMA/rolling usan el cierre actual y el pasado. No se encontraron ventanas centradas ni pivots escritos retrospectivamente en la ruta reutilizada.

`smc/detectors.py` ya implementa swings confirmados con left=3/right=2, HH/HL/LH/LL, BOS, CHoCH, sweeps, FVG y OB. El swing sólo se publica al cierre j+2; BOS/CHoCH requieren cruce del cierre y se consumen por nivel. OB requiere ruptura y cuerpo ≥0,5 ATR de las 14 velas previas; nace al confirmar, aunque su vela fuente sea anterior. La invalidación y el touch ocurren hacia adelante y el nacimiento no cuenta como retest. Por eso **no se añadió un segundo detector**. Los niveles de soporte/resistencia se extraen de esos mismos eventos confirmados.

La dirección bull/bear/sideways de `momentum/features.py:causal_regimes` usa retorno logarítmico 60 dividido por volatilidad logarítmica 20×√60, con bandas ±1. `feature_research/features.py` ya clasificaba volatilidad alta contra el percentil 80 de las 120 volatilidades anteriores. La nueva ronda reutiliza dirección y añade el extremo bajo del mismo esquema. Estos estados se conocen al cerrar la vela, sin suavizarlos usando datos posteriores.

Los estudios previos ya probaron reglas SMC, calibración sigmoide temporal, momentum, reversión, breakouts, flujo, features cruzadas, múltiples algoritmos, ensambles y stacking. `frontier_research/study.py` incluye un prior por régimen con contracción de veinte observaciones hacia la prevalencia global; `advanced_research` incluye filtros de tendencia y volatilidad. Ninguno equivale a entrenar y comparar especialistas supervisados por estado contra global y global+estado sobre las mismas observaciones. Esa es la ampliación de modelos de esta ronda; se reutiliza XGBoost, sin otro clasificador ni nueva búsqueda de hiperparámetros.

| Investigación existente | Hallazgo relevante para esta ronda |
|---|---|
| Original | Test publicado AUC 0,4787; retorno neto −45,32% con sus costos/período originales |
| SMC | No mostró aporte predictivo consistente en las siete ventanas anteriores |
| Feature engineering | 635 ajustes; añadir estructura redujo loss sólo 0,000216 BTC / 0,000533 ETH; evidencia insuficiente |
| Momentum y reversión | No cumplieron los criterios conjuntos; pocas oportunidades y dependencia temporal limitan inferencias |
| Riesgo | −45,32% → −1,51% con overlay en su comparación principal, con probabilidades idénticas; mejora de exposición, no de predicción |
| Advanced | Casos puntuales favorables como calibración, sin ventaja independiente; filtros ya disponibles |
| Frontier | 453 ajustes; ninguna receta aceptada; ya incluye prior por régimen, cinco familias de estimadores y ensambles |
| Triple Barrier / meta-labeling | 486 ajustes, ninguna mejora consistente que justifique reemplazar el original |
| Multimercado | Proveedores y ejecución FX tienen calendarios, spreads y objetivos propios; no se mezclan con el experimento diario cripto |

Los números de esta tabla son antecedentes de [EXPERIMENTS.md](EXPERIMENTS.md), [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md), [RISK_RESEARCH.md](RISK_RESEARCH.md), [ADVANCED_RESEARCH.md](ADVANCED_RESEARCH.md), [FRONTIER_RESEARCH.md](FRONTIER_RESEARCH.md) y [TRIPLE_BARRIER_RESEARCH.md](TRIPLE_BARRIER_RESEARCH.md); **sus retornos no son comparaciones emparejadas con esta ronda**. Se reajusta la receta original dentro de cada fold para la comparación causal de información.

## Qué se amplía

Las once extensiones de estructura son cuerpo/ATR previo, impulso alcista/bajista, rechazo de soporte/resistencia anterior, primer retest posterior de ruptura en cada dirección, retest confirmado por impulso y touch de OB confirmado por impulso. Con las 32 features SMC existentes y las 14 originales, el modelo estructura usa 57 columnas.

Un impulso exige cuerpo ≥0,5 ATR previo y cierre en el cuarto superior/inferior del rango actual. El soporte y la resistencia son los últimos swings confirmados. Su rechazo exige solapar el nivel conocido en el cierre anterior con tolerancia 0,25 ATR y cerrar del lado de rechazo. Una ruptura BOS/CHoCH guarda su nivel al confirmarse; el primer retorno posterior válido puede ocurrir dentro de veinte velas, con tolerancia 0,25 ATR y cierre del lado roto. Un cierre más allá de la tolerancia en sentido opuesto invalida el retest pendiente. Una nueva ruptura reemplaza el pendiente de su dirección. No se exige ni se conoce la supervivencia futura de una zona.

El filtro de selección acepta una entrada original únicamente con retest alcista+impulso, touch OB alcista+impulso o rechazo de soporte+impulso. No altera las probabilidades utilizadas para medir clasificación y no elimina filas del calendario de ejecución; las abstenciones conservan efectivo.

Régimen por reglas: tres direcciones × tres niveles de volatilidad. Baja/alta compara σ20 actual con percentiles 20/80 de las 120 σ anteriores; el resto es normal. Durante el calentamiento se marca unknown y no se usa la observación en ninguna receta. El modelo global recibe seis indicadores one-hot, no una codificación ordinal arbitraria.

Régimen por clustering: KMeans de tres clusters sobre momentum normalizado 60, eficiencia 20 y volatilidad existentes. StandardScaler y centroides se ajustan sólo en train, quedan congelados durante el fold y se guardan como números. Los identificadores de cluster son locales a cada fold; no se presentan como bull/bear/sideways ni se trasladan modelos entre clusters de folds diferentes. Se guardan centroides en unidades originales, tamaños y tasas de transición para interpretar su inestabilidad.

Los especialistas usan las catorce features originales y sólo las filas históricas de su estado. Se requieren ≥80 filas y ≥20 observaciones de cada clase. Sin ellas usan el global. Con ellas, su peso es n/(n+200) y el resto corresponde al global. Se reportan por separado la predicción del experto, la mezcla y el fallback; no se trata el número de filas como número de ciclos independientes.

## Validación y límites de interpretación

Se utilizan seis folds expansivos, BTC/ETH, dos semillas y filas comunes; train se purga por target_time, más seis días de embargo. La receta se hereda del modelo existente: 160 árboles, profundidad 2, learning rate 0,03, subsample/colsample 0,85, lambda 5 y min_child_weight 10. Es una receta elegida históricamente con estos datos: el walk-forward no elimina ese sesgo de diseño previo. Ninguna etiqueta externa participa de nuevos ajustes, escalado, clustering, pesos o umbrales. No se presenta el historial como test ciego.

Accuracy/AUC/log-loss/Brier miden el objetivo original close/close. La ejecución usa siguiente apertura y salida una apertura después, con costos: comisión 10 bps/lado, slippage 5 bps/lado y medio spread 1 bp/lado. Base y doble costo comparten señales y calendario. La variante sin spread permite contrastar el arrastre incremental del supuesto nuevo. Los spreads son supuestos constantes; no existen aquí cotizaciones históricas bid/ask, latencia o impacto. Las entradas consecutivas pagan la salida y la nueva entrada, igual que el original.

Las curvas se componen entre ventanas que comienzan en efectivo; no representan una cartera sin cortes. El drawdown incluye capital inicial, se mide a aperturas y omite variación intravela. Los intervalos económicos remuestrean los retornos diarios realizados de políticas fijas: no reentrenan modelos ni simulan ejecución sobre precios barajados.

Se reutiliza bootstrap pareado por bloques no circulares dentro de cada fold, con 2.000 remuestras y bloques 7/30. Max-|t| corrige simultáneamente pares y métricas dentro de cada familia. Se divide alfa entre dos activos, dos semillas, familias predictiva/económica y se reserva otra mitad para la combinación opcional: confianza por familia 99,6875%. Brier y diagnósticos de estados se informan descriptivamente; no se usan para extraer una significancia aislada. Esto no corrige las miles de decisiones de todas las rondas previas.

La aceptación histórica exige simultáneamente loss −0,003, AUC +0,005 frente al original, mejora conjunta en ≥4/6 folds, intervalos loss<0 y AUC>0 en ambos bloques, superar loss de p=0,5/prevalencia/logística, neto positivo y ≥30 trades, exceso de retorno con intervalo positivo, DD no peor y retorno positivo con costos dobles. Debe cumplirse en ambos activos y semillas. Una mejora de AUC aislada o una pérdida menor con menos operaciones no constituye una ventaja real.

Fuentes metodológicas: [validación temporal y gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [métricas probabilísticas](https://scikit-learn.org/stable/modules/model_evaluation.html), [Markov switching](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_regression.html), [sesgo de selección y múltiples pruebas](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf). Markov/HMM se investigó como alternativa: no se incorpora smoothing que use observaciones posteriores, ni se añade otra búsqueda con escasos ciclos. Las estructuras SMC son definiciones operativas sobre OHLCV y no identifican órdenes institucionales.

## Reproducción

```powershell
.\.venv\Scripts\python.exe -m trading_ml structure-regime-study --protocol config/structure_regime.toml
# Sustituir <run> por el directorio devuelto:
.\.venv\Scripts\python.exe scripts/verify_structure_regime.py artifacts/structure_regime_studies/<run>
.\.venv\Scripts\python.exe scripts/plot_structure_regime.py artifacts/structure_regime_studies/<run>
.\.venv\Scripts\python.exe -m pytest -q
```

Cada invocación de entrenamiento crea una corrida nueva: consume otro presupuesto y no se debe usar para buscar un ganador. El verificador reconstruye las predicciones desde los modelos nativos, scaler/centroides y coeficientes guardados, sin ajustes nuevos. No sobrescribe la corrida; guarda su recibo en otra carpeta. El script de gráficos requiere matplotlib, ya disponible en el entorno de investigación, y también usa una carpeta nueva.
