# Experimentos ejecutados

Fecha: 2026-10-12, America/Buenos_Aires. Todos los resultados de esta página corresponden a OHLCV reales de Binance spot. Los datasets sintéticos usados en tests no intervienen en estas métricas.

## Datos y entorno

- Activo: BTCUSDT; intervalo diario UTC; 81 ZIP mensuales oficiales con SHA-256 verificado.
- Inicio de velas: 2020-01-01; última vela: 2026-09-30; `end=2026-10-01` exclusivo.
- 2,465 velas crudas; 0 duplicados, 0 huecos, 0 valores faltantes, 0 velas de volumen cero.
- 49 filas de calentamiento eliminadas; 2,416 filas preparadas; 2,415 etiquetas disponibles; 1 fila final sin etiqueta. No hubo empates de precio en las etiquetas de este experimento.
- 14 variables causales; horizonte 1 vela; semilla 42; CPU hist y un hilo.
- Windows 11, Python 3.14.3, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, XGBoost 3.4.1 y requests 2.34.2. Entorno completo en `requirements-lock.txt`.

| Artefacto | SHA-256 |
| --- | --- |
| CSV crudo | `4be50c51df210f6dae7a5bde79bc7113a0201fb44f003f2aff76f4741e4a61d2` |
| CSV preparado | `403c6593e2816e1aad5fac3737f077bf2952955844445646cd579db41fade61c` |
| Modelo UBJSON, corte completo | `22b5b90c478011342900752cf93d23f0c66b89b23aee694951b9e6f97cf825fe` |

Procedencia por mes: `artifacts/data/raw.csv.json`. Calidad: `artifacts/data/prepared.csv.json`.

## Primer entrenamiento

Versión inicial: `v20261012T125624_1576ecf1`. Configuración guardada en su `metadata.json`. No se cambió la estrategia para mejorar el test después de ver los resultados.

| Partición | Filas después de purga | Primera referencia UTC | Última referencia UTC | Última etiqueta conocida UTC |
| --- | ---: | --- | --- | --- |
| Train | 1,448 | 2020-02-20 | 2024-02-06 | 2024-02-07 |
| Validación | 482 | 2024-02-08 | 2025-06-03 | 2025-06-04 |
| Test | 483 | 2025-06-05 | 2026-09-30 | 2026-10-01 |

Proporciones iniciales 60/20/20; dos filas purgadas. Sólo train ajusta los árboles. Validación selecciona por menor log-loss; test se calcula después de seleccionar.

| Profundidad | Learning rate | Log-loss validación |
| ---: | ---: | ---: |
| 2 | 0.03 | 0.699051 |
| 2 | 0.07 | 0.709701 |
| 3 | 0.03 | 0.705322 |
| 3 | 0.07 | 0.716413 |

Seleccionado: profundidad 2, learning rate 0.03, 160 árboles, subsample 0.85, colsample_bytree 0.85, reg_lambda 5.0 y min_child_weight 10. Sin early stopping, sin calibración posterior y sin reentrenar con validación.

## Evaluación fuera de muestra

Baseline: probabilidad constante igual a la proporción de subidas del train, 0.50828729. No usa la frecuencia del test para ajustar su probabilidad.

| Métrica | XGBoost | Baseline constante |
| --- | ---: | ---: |
| Accuracy | 0.492754 | 0.494824 |
| Precision | 0.487903 | 0.494824 |
| Recall | 0.506276 | 1.000000 |
| F1 | 0.496920 | 0.662050 |
| ROC-AUC | 0.478702 | 0.500000 |
| Brier score (menor es mejor) | 0.256962 | 0.250154 |
| Log-loss (menor es mejor) | 0.707352 | 0.693456 |
| ECE, 10 bins fijos | 0.062700 | 0.013463 |

La frecuencia real de subidas del test fue 0.494824. El baseline predice siempre subida al aplicar el umbral 0.5; su F1 mayor no prueba utilidad económica. XGBoost no supera este baseline en discriminación ni calidad probabilística en este período.

| Intervalo de probabilidad | Observaciones | P(up) promedio | Subidas observadas |
| --- | ---: | ---: | ---: |
| [0.3, 0.4) | 34 | 0.384929 | 0.500000 |
| [0.4, 0.5) | 201 | 0.453482 | 0.502488 |
| [0.5, 0.6) | 212 | 0.542480 | 0.504717 |
| [0.6, 0.7) | 36 | 0.625431 | 0.388889 |

Los bins vacíos no se reportan. El último bin tiene sólo 36 casos: no conviene interpretar su discrepancia como una regla estable. La calibración fue evaluada, no corregida mediante un ajuste aprendido.

Evidencia: `artifacts/models/v20261012T125624_1576ecf1/test_metrics.json`, `test_predictions.csv` y `artifacts/reports/initial_evaluation.json`.

## Predicción histórica de prueba

Referencia 2026-10-01 00:00 UTC, para el cierre del 2026-10-02 00:00 UTC; horizonte 1 día:

- P(subida): 0.5179736614.
- P(bajada o sin cambio): 0.4820263386.
- Estado `warning`: probabilidades sin calibración posterior y datos históricos antiguos respecto de la fecha de ejecución.

El cierre objetivo no está incluido en el dataset descargado; esta predicción demuestra la inferencia de la última fila sin etiqueta. No se afirma que haya acertado ni se interpreta como una señal actual. JSON original: `artifacts/reports/initial_prediction.json`. Se verificó también el argumento CLI `--reference 2026-10-01T00:00:00Z`.

## Backtest inicial

Long/cash, umbral 0.55 fijado en configuración, capital inicial 10,000 USDT, comisión 10 bps y slippage 5 bps **por lado**. Aperturas desde 2025-06-05 hasta 2026-09-30; salida un día después de la entrada; sin short ni leverage.

| Resultado | Estrategia | Buy and hold con costos |
| --- | ---: | ---: |
| Rentabilidad acumulada | -45.317348% | -20.328991% |
| Drawdown máximo a aperturas | -46.887085% | -52.971766% |
| Capital final de estrategia | 5,468.265203 USDT | — |
| Operaciones cerradas | 116 | — |
| Operaciones con retorno neto positivo | 40.517241% | — |

Una señal se excluyó por no disponer de la apertura necesaria para su salida. No hubo señales descartadas por posición abierta para h=1. Las operaciones negativas no se sustituyeron ni ocultaron. El backtest perdió más que buy and hold en rentabilidad; un drawdown menor no vuelve rentable la estrategia. No incluye liquidez, latencia, impuestos, financiación, spreads variables ni drawdown intravela.

Evidencia inicial: `artifacts/reports/v20261012T125624_1576ecf1/backtest/summary.json`, `equity.csv` y `trades.csv`.

## Reentrenamiento real y comparación

Ejecutado `scripts/run_retraining_demo.py` con un snapshot anterior a julio de 2026 y el dataset completo:

- Padre: `v20261012T130539_2a7f3cde`, datos hasta 2026-06-30.
- Reentrenado: `v20261012T130540_d2bb6ac0`, incorpora julio-septiembre de 2026; registra el padre, conserva indicadores/hiperparámetros y guarda el rango temporal ampliado en su configuración.
- La comparación usa exactamente las mismas 483 filas fuera de entrenamiento y selección para ambos, 2025-06-05 a 2026-09-30.

| Métrica sobre período común | Padre | Reentrenado |
| --- | ---: | ---: |
| Accuracy | 0.472050 | 0.492754 |
| ROC-AUC | 0.479611 | 0.478702 |
| Brier | 0.255988 | 0.256962 |
| Log-loss | 0.705324 | 0.707352 |

Es una demostración retrospectiva del mecanismo, no una evaluación prospectiva de una política desplegada. El nuevo dataset desplaza los cortes temporales de train/validation/test. No se eligió un ganador ni se ajustó el umbral usando esta tabla. El modelo reentrenado con el dataset completo tiene exactamente el mismo SHA-256 que el primero.

Evidencia: `artifacts/reports/retraining_demo.json` y `artifacts/reports/comparison.json`. Una solicitud de reentrenamiento sin nuevas velas se rechaza y está cubierta por pruebas.

## Recorrido reproducible y verificaciones finales

También se ejecutó el script completo `scripts/run_pipeline.py`, desde descarga/cache verificada hasta predicción/backtest: versión `v20261012T130154_6893e846`. El modelo tiene el mismo hash y los mismos resultados que el inicial; los IDs y timestamps de ejecución cambian intencionalmente. Después se verificó nuevamente el reentrenamiento: `latest` apunta a `v20261012T130540_d2bb6ac0` al finalizar esta entrega, con el mismo hash y fechas del dataset completo correctamente registradas.

Manifiesto completo: `artifacts/reports/v20261012T130154_6893e846/run.json`. Reportes separados en ese mismo directorio: `evaluation.json`, `prediction.json`, `backtest/summary.json`, `backtest/equity.csv`, `backtest/trades.csv`.

Controles ejecutados mediante `scripts/verify.py`:

- 37 pruebas automatizadas pasaron, sin warnings: datos inválidos, huecos y cobertura, duplicados, ms/us, checksum remoto y caché, indicadores causales, cola sin etiqueta, purga h=1/3/10, clasificación/calibración, persistencia, reproducibilidad, corrupción, predicción histórica, rechazo de leakage, reentrenamiento, comparación y CLI.
- Prueba adversarial: alterar OHLCV de test no cambia bytes de modelo, selección ni métricas de validación.
- Backtesting probado contra fórmulas exactas de costos, ejecución posterior, pérdidas con precio plano, drawdown durante posiciones y exclusión por salida faltante.
- Ruff lint y formato correctos; `pip check` sin dependencias rotas.

La salida original de estos controles se conserva en `artifacts/verification/checks.json`. Los resultados de esta página son descriptivos, sin intervalos de confianza ni significancia estadística: walk-forward, calibración independiente y evaluación prospectiva quedan pendientes.

## Próxima versión

Priorizar validación walk-forward con purga y varias ventanas, una ventana independiente para calibración, datos diarios incrementales y monitoreo de drift. Estudiar si las variables aportan información antes de aumentar la búsqueda de parámetros; una grilla mayor por sí sola puede aumentar el sobreajuste. Incorporar costos más realistas y un período prospectivo de simulación una vez definida la estrategia.

## Optimización moderada — 20261012T131625_fe8294f6

Fecha de ejecución: 2026-10-12, America/Buenos_Aires. Los resultados siguientes provienen de históricos reales; las pruebas sintéticas sólo verifican invariantes.

### Protocolo previo y aislamiento del test

Plan escrito antes de entrenar: `artifacts\optimization\20261012T131625_fe8294f6\plan.json`. SHA-256 del plan: `982bbe4f0d544c48eb17fd3f114bfa115c549ffe008a21f005045b16e337fedb`. Configuración: `config/optimization.toml`.

Máximo cinco recetas **incluida E0**; siete ventanas consecutivas desde 2022-01-01 hasta 2025-06-05; parada después de tres variantes consecutivas sin aceptación. Cada receta incluye sus ajustes por fold; no se ejecuta una grilla adicional.

El período final 2025-06-05 a 2026-09-30, 483 referencias, se excluye de selección y early stopping. **Ya había sido observado en la etapa anterior**: esta confirmación es retrospectiva, no un test nuevo ciego. No hay nuevas velas posteriores a septiembre en el proveedor mensual del snapshot. La receta de referencia también fue elegida históricamente en la etapa previa; su walk-forward es una comparación retrospectiva de una receta fija, no el historial de un sistema desplegado.

Cada fold entrena sólo con etiquetas conocidas antes de su primera referencia. Early stopping usa una cola interior de 90 días con purga; después se reentrena sobre el train exterior con la cantidad de árboles seleccionada. Ninguna etiqueta, feature ni precio de backtest con referencia en el fold siguiente entra en la selección. Las señales sin apertura de salida dentro de la ventana se excluyen y registran.

Regla fijada: reducción OOF de log-loss >= 0.003, aumento de AUC media por fold >= 0.005, mejora de log-loss **y** Brier en al menos 5/7 ventanas, rentabilidad neta superior a referencia en al menos 5/7, retorno compuesto no peor y positivo, al menos 30 trades y log-loss inferior al baseline constante del train de cada fold. No se suavizó la regla tras observar los resultados.

Mismos costos y estrategia para todas las recetas: long/cash, umbral 0.55, 10 bps de comisión y 5 bps de slippage por lado. Retorno compuesto de desarrollo = producto de los retornos netos de folds cerrados; hay cierres/reaperturas en límites y no posiciones entre ventanas. No se compara ese retorno de desarrollo con el retorno del período final como si fueran el mismo tramo.

### Recetas y resultados de desarrollo

| ID | Cambio | Log-loss OOF | Brier OOF | AUC media por fold | Retorno compuesto neto | Trades | Aceptado |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E0 | Receta anterior reentrenada antes de cada fold | 0.701267 | 0.253818 | 0.529518 | -68.68% | 420 | Referencia |
| E1 | Regularización moderada y early stopping interior | 0.696831 | 0.251785 | 0.536215 | -67.13% | 299 | No |
| E2 | E1 más ocho features causales de momentum, riesgo y volumen | 0.695062 | 0.250921 | 0.539473 | -70.71% | 234 | No |
| E3 | E2 incorporando historia desde el inicio spot de BTCUSDT | 0.696092 | 0.251448 | 0.532720 | -61.62% | 227 | No |

E1/E2/E3: máximo 400 árboles, profundidad 2, learning rate 0.025, lambda 20, min_child_weight 20, subsample/colsample 0.8 y selección de árboles exclusivamente interior. E2 agrega retorno a 21 velas, dos retornos rezagados, momentum/volatilidad, ratio de volatilidades 5/60, ATR relativo, ubicación del cierre en el rango y cambio logarítmico de volumen. Son causales y su contrato es schema 2; los modelos originales conservan schema 1.

1244 observaciones OOF comunes. El candidato elegido **antes de consultar el final** fue E2, por menor log-loss OOF entre las variantes, para auditoría aunque ninguna quedó aceptada.

| ID | Ventanas con log-loss y Brier mejores | Ventanas con retorno neto mejor | Log-loss baseline constante | Motivos de rechazo |
| --- | ---: | ---: | ---: | --- |
| E1 | 6/7 | 5/7 | 0.694962 | positive_compounded_return, beats_constant_log_loss |
| E2 | 6/7 | 2/7 | 0.695055 | economic_consistency, compounded_return_not_worse, positive_compounded_return, beats_constant_log_loss |
| E3 | 6/7 | 4/7 | 0.694512 | mean_fold_auc_gain, economic_consistency, positive_compounded_return, beats_constant_log_loss |

### Rentabilidad neta por ventana

| Ventana | E0 | E1 | E2 | E3 | Buy and hold con costos |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-01-01 a 2022-06-29 | -35.67% | -48.14% | -53.61% | -32.39% | -56.25% |
| 2022-07-01 a 2022-12-30 | -37.20% | -33.70% | -37.83% | -34.56% | -16.84% |
| 2023-01-01 a 2023-06-29 | 1.66% | 2.37% | -5.10% | 5.47% | 81.28% |
| 2023-07-01 a 2023-12-30 | -20.40% | -4.03% | 11.86% | 2.53% | 37.64% |
| 2024-01-01 a 2024-06-29 | 1.37% | 4.46% | 0.00% | 0.00% | 42.48% |
| 2024-07-01 a 2024-12-30 | -16.89% | -6.84% | -14.90% | -19.78% | 48.88% |
| 2025-01-01 a 2025-06-03 | 13.71% | 0.00% | 12.42% | 0.00% | 12.79% |

### Historia adicional

Se descargó y verificó historia desde 2017-08-17: 3,332 velas crudas frente a 2,465, incorporando 867 anteriores a 2020. Sin huecos, duplicados ni faltantes. Se verificó que las velas compartidas son idénticas a las originales. No se cambió el tramo de evaluación.

E3 mejoró el retorno compuesto respecto de E2, pero empeoró log-loss/Brier/AUC y siguió perdiendo. No superó el baseline constante ni la regla de consistencia económica. Esta prueba no justifica incorporar automáticamente toda la historia vieja; tampoco demuestra que más datos recientes o de otras fuentes no puedan aportar. El mercado spot anterior a 2017-08-17 no existe en este proveedor.

### Parada y comparación final única

Ejecutados **4 de un máximo de 5**. Motivo: `no_consistent_improvement`. E4 (ventana móvil de 730 días) quedó **sin ejecutar**, respetando la parada; no se añadió otra receta después de conocer el final.

Anterior: `v20261012T130540_d2bb6ac0`. Candidato reentrenado para auditoría: `v20261012T131650_5106daf4` (E2, 31 árboles efectivos, 1920 filas de train). Última etiqueta utilizada: 2025-06-04T00:00:00+00:00; primera referencia final: 2025-06-05T00:00:00+00:00.

| Métrica en las mismas 483 referencias finales | Anterior | Candidato auditado |
| --- | ---: | ---: |
| Accuracy | 0.492754 | 0.490683 |
| Precision | 0.487903 | 0.488215 |
| Recall | 0.506276 | 0.606695 |
| F1 | 0.496920 | 0.541045 |
| ROC-AUC | 0.478702 | 0.475718 |
| Log-loss | 0.707352 | 0.696816 |
| Brier score | 0.256962 | 0.251829 |
| ECE 10 bins | 0.062700 | 0.029960 |
| Retorno neto | -45.32% | -13.68% |
| Drawdown máximo | -46.89% | -21.92% |
| Trades | 116 | 17 |
| Win rate neto | 40.52% | 41.18% |

Buy and hold en ese mismo tramo perdió 20.33% con costos para ambas comparaciones. El candidato redujo la pérdida y mostró probabilidades menos extremas, pero realizó sólo 17 trades frente a 116; menor exposición/costos puede explicar parte de la diferencia. La mejora de log-loss/Brier no se acompañó de mayor discriminación: accuracy y ROC-AUC empeoraron. No se afirma una ventaja rentable.

El candidato final también incorpora entrenamiento hasta junio de 2025, frente al modelo anterior entrenado hasta febrero de 2024: la comparación combina reentrenamiento, regularización y features, y no identifica el efecto causal de cada cambio. Los folds E0 reentrenados antes de cada ventana sirven como control comparable de la receta anterior.

**No se promueve el candidato**: falló la regla walk-forward aunque cumplió los tres checks descriptivos finales. `latest` y `champion.json` conservan `v20261012T130540_d2bb6ac0` como referencia operativa de investigación. Esto no valida la rentabilidad del modelo anterior: sólo evita reemplazarlo por una variante sin evidencia suficiente. Se conserva también el candidato auditado para reproducir sus resultados, sin usarlo como campeón.

### Evidencia y limitaciones

Directorio: `artifacts\optimization\20261012T131625_fe8294f6`. Plan, selección congelada y resumen: `plan.json`, `selection.json`, `summary.json`. Cada receta conserva `walk_forward.json`, `oof_predictions.csv` y modelos/equity/trades por fold. Comparación final: `final/previous` y `final/candidate`. Dataset del candidato: `artifacts\optimization\20261012T131625_fe8294f6\E2\prepared.csv`. El modelo incluye configuración, hashes, padre y estado de aceptación en sus metadatos.

Los tests verifican purga exterior/interior, historia móvil, features nuevas causales, preservación de contratos viejos, presupuesto, parada y conservación del campeón. La prueba adversarial que altera precios/features/etiquetas del final deja idénticos los resultados walk-forward. Evidencia de controles: `artifacts/verification/checks.json`.

Verificación final de esta ronda: **45 pruebas pasaron sin warnings**, Ruff lint y formato correctos, `pip check` sin dependencias rotas. La recuperación del informe completado mantuvo exactamente cuatro recetas y 28 modelos de folds, sin nuevos experimentos. Se cargó el candidato guardado y se generó una predicción de prueba con su dataset de schema 2; JSON en `final/candidate/latest_prediction.json` (histórica, con advertencias de antigüedad y ausencia de calibración posterior).

Diagnóstico respaldado por esta ronda: señales débiles frente a un baseline constante; sensibilidad entre regímenes (especialmente 2022); probabilidades cercanas a 0.5; costos y pocas oportunidades al umbral fijo. El aprendizaje de dirección close/close no garantiza retorno neto open/open. Los datos adicionales antiguos no corrigieron estos problemas. No se investigó una nueva estrategia, otro horizonte, calibración aprendida, shorts o señales externas para mantener el presupuesto.

Próximo trabajo: esperar una ventana nueva no observada para evaluación prospectiva, calibración en una ventana independiente, monitoreo de drift y señales distintas a indicadores de precio/volumen. La ventana móvil quedó pendiente; no se seguirá buscando sobre este test para lograr una métrica mejor.

## Smart Money Concepts — 20261012T134906_05753740

Fecha: 2026-10-12, America/Buenos_Aires. OHLCV reales BTCUSDT spot del snapshot 2020-01-01 a 2026-09-30; 2,465 velas. Las métricas siguientes son exclusivamente de desarrollo walk-forward; no son evaluación final independiente.

### Reglas, datos y diseño fijado

Definiciones matemáticas completas y límites en [SMC_RULES.md](SMC_RULES.md). Se detectan swings confirmados, estructura/tendencia, BOS, CHoCH, sweeps, FVG y OB. No hay señales retroactivas: pivots se publican en j+right y OB en la ruptura que los confirma. Ningún patrón demuestra quién operó ni identifica órdenes institucionales.

32 features SMC más las 14 anteriores: 46 columnas, schema 3. Los contratos 1/2 siguen compatibles. Parámetros de detección congelados en `artifacts\smc_studies\20261012T134906_05753740\plan.json`; SHA-256 `8b7dab3dd0218119dc5fcf58432f0fd59bb3c5f0caa987cfeda785237b23cce1`. Dataset preparado y snapshot OHLCV preservados en ese directorio.

Se usan las mismas siete ventanas purgadas de la etapa anterior: 2022-01-01 a 2025-06-05 (límite exclusivo), 1,244 referencias etiquetadas. Se vuelve a entrenar la receta original antes de cada ventana: no se usa un modelo del futuro para simular el pasado. Estos períodos ya son desarrollo previamente estudiado; no se presentan como muestras nuevas independientes.

**El test observado desde 2025-06-05 se excluyó de entrenamiento, early stopping, calibración, selección y evaluación de esta ronda.** Se emplean sus velas ya cerradas sólo para construir el ejemplo de análisis histórico, sin medir sus resultados. La evaluación final se reserva a datos futuros nuevos.

Hasta cinco recetas incluyendo referencia y reglas; parada tras dos variantes ML consecutivas sin cumplir la regla completa. No hubo una grilla adicional ni ajustes guiados por el test. El umbral neutral queda fijo en 0.55/0.45. No se promueve automáticamente ningún modelo aunque cumpla validación.

### Experimentos ejecutados

| ID | Configuración | Accuracy | ROC-AUC OOF global | AUC media por fold | Log-loss | Brier | ECE 10 bins | Retorno compuesto neto | Trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | XGBoost original reentrenado por ventana | 52.89% | 0.523523 | 0.529518 | 0.701267 | 0.253818 | 0.035429 | -68.68% | 420 |
| S1 | SMC determinista: confianza heurística sin ML | 47.67% | 0.482711 | 0.477550 | 0.725901 | 0.264835 | 0.083923 | -48.42% | 235 |
| S2 | XGBoost original + SMC, mismo train e hiperparámetros | 51.53% | 0.525866 | 0.529278 | 0.700360 | 0.253372 | 0.047189 | -69.76% | 426 |
| S3 | SMC + regularización, ES interior y calibración temporal independiente | 50.64% | 0.520139 | 0.540610 | 0.693903 | 0.250374 | 0.030893 | -10.21% | 135 |

ROC-AUC OOF global agrupa scores de modelos/períodos diferentes; AUC media por fold promedia rankings dentro de cada ventana. La calibración puede desplazar niveles de probabilidad entre ventanas, por lo que las dos medidas pueden diferir. Se reportan ambas y la regla prefijada usa la media por fold.

S2 mantiene exactamente train, features técnicas e hiperparámetros originales y agrega sólo SMC: es el control principal del aporte incremental de esas features. S3 agrega también regularización y calibración; no permite atribuir todas sus diferencias a SMC por sí solo.

S3 reserva 90 días para calibración y, antes, otros 90 días para early stopping interior con purga. XGBoost se reentrena sólo sobre el bloque anterior a calibración con la cantidad de árboles elegida. La calibración aprende una sigmoide regularizada de logit(p) en filas disjuntas; no hay refit de XGBoost después. Una pendiente negativa colapsa a prevalencia y se informa, sin invertir el ranking.

### Comparación de los tres sistemas solicitados

El combinado seleccionado por menor log-loss de desarrollo es S3. Selección de investigación: no cumplió la regla completa; promoción automática: **no**.

| Métrica | XGBoost original | SMC reglas sin ML | XGBoost+SMC seleccionado |
| --- | ---: | ---: | ---: |
| Accuracy | 0.528939 | 0.476688 | 0.506431 |
| Precision binaria a p>=0.5 | 0.521146 | 0.483716 | 0.503198 |
| Recall binario a p>=0.5 | 0.619125 | 0.818476 | 0.382496 |
| F1 binario | 0.565926 | 0.608067 | 0.434622 |
| ROC-AUC OOF global | 0.523523 | 0.482711 | 0.520139 |
| Log-loss | 0.701267 | 0.725901 | 0.693903 |
| Brier | 0.253818 | 0.264835 | 0.250374 |
| ECE 10 bins | 0.035429 | 0.083923 | 0.030893 |
| Señales alcistas | 422 | 236 | 136 |
| Señales bajistas | 237 | 200 | 264 |
| Neutrales | 585 | 808 | 844 |
| Total direccionales | 659 | 436 | 400 |
| Cobertura direccional | 52.97% | 35.05% | 32.15% |
| Acierto alcista selectivo | 51.18% | 49.15% | 52.94% |
| Acierto bajista selectivo | 54.43% | 44.00% | 53.41% |
| Acierto de señales direccionales | 52.35% | 46.79% | 53.25% |

Los neutrales **no** cuentan como aciertos ni se eliminan para mejorar las métricas probabilísticas: ROC-AUC/log-loss/Brier/ECE usan todas las 1,244 filas. La accuracy/precision binaria usa p>=0.5 únicamente para comparación convencional; los scores neutrales 0.5 de las reglas quedan empatados a subida en esa métrica y explican su recall alto. Para interpretar las señales de la herramienta usar cobertura y precisión selectiva, que excluyen abstenciones y muestran el denominador.

Estado de calibración: original y S2 sin calibración posterior; SMC reglas es confianza heurística **no calibrada** (pesos fijados, no aprendidos); S3 tiene sigmoide ajustada en pasado disjunto. Ajustar un calibrador no garantiza calibración futura bajo drift. Reliability bins completos de todos los modelos/folds están en los JSON, no sólo ECE agregado.

### Resultados por ventana

| Primera referencia | Última referencia | Filas | AUC original | AUC reglas | AUC combinado | Log-loss original | Log-loss reglas | Log-loss combinado | Señales original/reglas/combinado |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2022-01-01 | 2022-06-29 | 180 | 0.5514 | 0.4697 | 0.5467 | 0.6992 | 0.7247 | 0.6924 | 130/62/0 |
| 2022-07-01 | 2022-12-30 | 183 | 0.5136 | 0.5025 | 0.5671 | 0.7114 | 0.7093 | 0.6855 | 111/66/48 |
| 2023-01-01 | 2023-06-29 | 180 | 0.6262 | 0.4728 | 0.5473 | 0.6772 | 0.7305 | 0.7032 | 74/60/180 |
| 2023-07-01 | 2023-12-30 | 183 | 0.4777 | 0.4679 | 0.5447 | 0.7256 | 0.7168 | 0.6933 | 113/35/0 |
| 2024-01-01 | 2024-06-29 | 181 | 0.5399 | 0.4635 | 0.5744 | 0.6928 | 0.7539 | 0.6905 | 92/87/136 |
| 2024-07-01 | 2024-12-30 | 183 | 0.4871 | 0.4547 | 0.5042 | 0.7036 | 0.7277 | 0.6987 | 70/70/36 |
| 2025-01-01 | 2025-06-03 | 154 | 0.5108 | 0.5118 | 0.5000 | 0.6979 | 0.7175 | 0.6939 | 69/56/0 |

### Backtesting de análisis con costos

Simulación long/cash de investigación, compra a próxima apertura y salida tras una vela. Comisión 10 bps y slippage 5 bps por lado; sin shorts ni órdenes reales. Los neutros conservan efectivo. La calidad bajista se mide contra etiquetas, pero el simulador no toma posiciones bajistas. Cada fold liquida dentro de su ventana y excluye señales sin precio de salida, evitando precios del fold siguiente. El compuesto encadena retornos de ventanas cerradas; no es un backtest de un modelo único desplegado sin cambios.

| Ventana | Original neto | SMC reglas neto | Combinado neto | Buy and hold con costos |
| --- | ---: | ---: | ---: | ---: |
| 2022-01-01 a 2022-06-29 | -35.67% | -27.53% | 0.00% | -56.25% |
| 2022-07-01 a 2022-12-30 | -37.20% | -14.70% | 0.00% | -16.84% |
| 2023-01-01 a 2023-06-29 | 1.66% | 3.61% | 0.00% | 81.28% |
| 2023-07-01 a 2023-12-30 | -20.40% | 9.70% | 0.00% | 37.64% |
| 2024-01-01 a 2024-06-29 | 1.37% | -11.88% | -10.21% | 42.48% |
| 2024-07-01 a 2024-12-30 | -16.89% | -6.26% | 0.00% | 48.88% |
| 2025-01-01 a 2025-06-03 | 13.71% | -11.13% | 0.00% | 12.79% |

### Patrones y condición de parada

| Patrón | Alcistas | Bajistas |
| --- | ---: | ---: |
| BOS | 33 | 25 |
| CHoCH | 21 | 21 |
| Liquidity sweep | 50 | 87 |
| FVG nuevos | 95 | 78 |
| OB nuevos | 47 | 39 |

S2: mejoras simultáneas log-loss/Brier en 4/7 folds y retorno superior en 4/7 (se exigían 5). Checks fallidos: `log_loss_gain, mean_fold_auc_gain, probabilistic_consistency, economic_consistency, compounded_return_not_worse, positive_compounded_return, beats_constant_log_loss`.

S3: mejoras simultáneas log-loss/Brier en 6/7 folds y retorno superior en 4/7 (se exigían 5). Checks fallidos: `economic_consistency, positive_compounded_return`.

Ejecutadas 4 de hasta 5 recetas; motivo `no_consistent_smc_improvement`. Recetas no ejecutadas: S4. La parada se respetó: no se cambió la definición ni se añadió un sexto experimento después de ver resultados.

### Aporte incremental y problemas encontrados

Agregar sólo SMC (S2) cambió log-loss en +0.000908 de mejora y AUC media por fold en -0.000240. Esto no cumple los mínimos prefijados ni la consistencia entre períodos. No demuestra aporte predictivo robusto de los patrones.

S3 mejoró calidad probabilística y redujo exposición, pero su retorno sigue negativo y no ganó económicamente en suficientes ventanas. En seis de siete períodos su simulación long/cash no abrió operaciones; las 135 operaciones se concentran en un solo fold. Por eso un drawdown o pérdida menor no demuestra una estrategia rentable ni que SMC por sí solo agregó alpha. Las mejoras mezclan regularización, menor complejidad y calibración: faltan controles adicionales para atribuirlas específicamente a SMC, que no se añadieron para respetar el presupuesto.

Las reglas discretas están sobreconfiadas para el siguiente cierre diario y rinden cerca/debajo del azar. Los patrones describen estructura y zonas, mientras la etiqueta sigue siendo dirección close/close de una sola vela. La agregación diaria no modela libro de órdenes, trayectoria intravela ni ejecución real. Una estructura alcista puede coexistir con probabilidad neutral a un día. La validación reutiliza desarrollo histórico y no incluye inferencia causal, intervalos de confianza ni una muestra nueva independiente.

Referencia conservada: `v20261012T130540_d2bb6ac0`; candidato SMC: `v20261012T134917_f88d6f93`; seleccionado `S3`. **No se cambió latest ni champion.json**. El candidato está guardado para investigación y no valida la rentabilidad del anterior.

### Análisis funcional de ejemplo

Se ejecutó `smc-analyze` con referencia histórica `2026-10-01T00:00:00+00:00`, horizonte 1 vela(s). Tendencia `bullish`, estructura `higher_highs_higher_lows`, patrones pivot_low_confirmed, fvg_bull_touch, P(up)=0.527523, P(down/no cambio)=0.472477, estado `neutral`, calibración `sigmoid_fitted_on_disjoint_past_window`.

El JSON incluye zonas activas con límites, origen y referencia de confirmación, además de advertencias de antigüedad. Las reglas y ML se muestran separados; no son instrucciones de compra/venta. Este ejemplo no evalúa ni afirma el acierto del test observado.

### Reserva final independiente y reproducibilidad

Ventana prospectiva congelada: referencias desde `2026-10-13T00:00:00+00:00` hasta `2027-01-11T00:00:00+00:00` exclusivo (90 referencias diarias). Estado: **pending_future_data**. No hay resultados finales nuevos para inventar o afirmar una mejora. La última apertura de salida también debe estar disponible y cerrada en el dataset de evaluación.

`smc-future-evaluate` exige la ventana completa, etiquetas, aperturas de salida, integridad del plan y conservación exacta del snapshot anterior. Evalúa los tres sistemas congelados; no reentrena/recalibra ni promueve. Con el CSV actual rechaza la evaluación como pendiente y no produce métricas. La ruta completa se verifica sólo con datos sintéticos y reloj simulado dentro de tests temporales aislados: esas métricas nunca se registran como resultados reales.

Artefactos en `artifacts\smc_studies\20261012T134906_05753740`: `plan.json`, `development.json`, `selection.json`, `summary.json`, `prospective.json`, `raw_snapshot.csv`, dataset preparado y reportes/modelos/predicciones/equity/trades por receta y fold. La repetición del mismo protocolo recupera el reporte sin nuevas recetas.

Tests de esta etapa: detección matemática y simetría, confirmación sin retroactividad, mitigación/expiración/capacidad de zonas, igualdad de prefijos y mutaciones futuras, ES/calibración disjuntos, persistencia, neutral/coverage, exclusión del test, presupuesto/parada, conservación de referencia, inferencia y bloqueo/recorrido prospectivo. Evidencia completa de pytest/Ruff/pip check en `artifacts/verification/checks.json`.

Verificación final: **65 pruebas aprobadas sin warnings**, Ruff lint/formato correctos y `pip check` sin dependencias rotas. El recorrido completo prospectivo se probó con reloj y datos sintéticos en directorios temporales; la ejecución con OHLCV actuales devolvió el error esperado de reserva incompleta, sin escribir métricas finales. Se verificaron 21 modelos de folds, el candidato cargable, análisis CLI, integridad del plan/snapshot, ausencia de S4 y conservación exacta de `latest` y `champion.json`.

Conclusión: **no se demostró una mejora predictiva consistente atribuible a SMC**. Los detectores y el analizador sí quedaron funcionales. Esperar la ventana prospectiva antes de afirmar generalización; no continuar afinando reglas sobre el test observado. Cualquier investigación de otros horizontes, intradía o calibración adicional necesita un protocolo nuevo y presupuesto propio.

## Reentrenamiento corto con historia reciente — 2026-10-12

Pedido posterior: descargar cantidad moderada, reentrenar ambas familias con pocos recursos y comparar contra versiones anteriores sin cambiar arquitectura ni funcionalidades. Se ejecutaron **dos recetas fijas**, sin búsqueda ni promoción. Evidencia completa, resultados de cada fold y limitaciones: [RESULTS.md](artifacts/recent_retraining/20261012_short/RESULTS.md); cifras completas: [summary.json](artifacts/recent_retraining/20261012_short/summary.json).

Se descargaron nuevamente 33 archivos mensuales SHA-256 verificados: BTC/USDT diario, 2024-01-01 a 2026-09-30, 1,004 velas, 955 filas preparadas y 954 etiquetas. El tramo coincide exactamente con el snapshot previo: **no incorpora fechas adicionales** porque el proveedor mensual existente ya tenía septiembre y todavía no consume octubre. No se implementó otro proveedor ni una descarga diaria incremental.

Tres folds comunes, expansivos y purgados: 2025-06-05 a 2025-10-01, 2025-10-01 a 2026-04-01 y 2026-04-01 a 2026-10-01, límites finales exclusivos. 480 referencias externas comunes (117/181/182). Versiones previas estáticas frente a nuevos modelos reentrenados antes de cada fold; las etiquetas posteriores nunca intervienen en ese ajuste. ES interior 60 días, paciencia 10; SMC sigmoid en 90 días separados. CPU hist, un hilo, profundidad 2 y máximo 80 árboles. Ajuste de candidatos completos y seis modelos de folds más evaluación: 1.78 s de pared, 1.56 s de CPU; descarga 12.56 s.

**Evaluación retrospectiva, no test final nuevo**: estas fechas incluyen el antiguo test observado. Recetas congeladas antes de calcular resultados, sin selección ni variantes posteriores basadas en estas métricas. No se permite promoción con esta evidencia. La reserva prospectiva anterior 2026-10-13 a 2027-01-11 exclusivo, sus modelos congelados y su plan siguen intactos y pendientes.

| Sistema | Accuracy | AUC OOF | Log-loss | Brier | ECE | Neto con costos | Trades | Señales |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original anterior | 49.17% | 0.477759 | 0.707562 | 0.257065 | 0.063747 | -45.32% | 116 | 238 |
| Original nuevo | 47.71% | 0.458027 | 0.702967 | 0.254825 | 0.047193 | -35.26% | 75 | 92 |
| SMC anterior | 49.58% | 0.490833 | 0.695126 | 0.250988 | 0.031665 | 0.00% | 0 | 0 |
| SMC nuevo | 48.33% | 0.493776 | 0.697617 | 0.252221 | 0.048358 | 0.00% | 0 | 182 |

Costos por lado: comisión 10 bps + slippage 5 bps; long/cash, entrada siguiente apertura, horizonte un día, sin shorts ni operaciones reales. Neto compuesto entre folds independientes. El 0% SMC significa **cero operaciones**, no una estrategia rentable. Sus 182 señales nuevas son bajistas, todas en el último fold, con acierto 48.35%.

Original: log-loss/Brier mejoran agregados pero accuracy y AUC empeoran; sólo mejora ambas métricas probabilísticas en 1/3 folds. Menor pérdida acompaña menor exposición; en el único fold donde opera pierde 35.26%. SMC: empeoran log-loss, Brier y ECE; sólo mejora log-loss/Brier en 1/3 folds. La calibración del último fold colapsa a una constante por pendiente negativa, sin discriminación. Ambos nuevos pierden en log-loss/Brier frente a la prevalencia histórica. No se acredita aporte consistente de SMC; el diseño no aísla SMC de regularización/calibración.

Parada al completar las dos recetas: sin más experimentos y **sin reemplazo**. Original activo `v20261012T130540_d2bb6ac0`, SMC previo `v20261012T134917_f88d6f93`; candidatos nuevos `v20261012T143622_ff1c464c` (26 árboles) y `v20261012T143623_0d9b01f0` (2 árboles). Los candidatos completos no se presentan como evaluados con los resultados de sus modelos por fold. Metadatos guardan contratos propios, procedencia y estado de test pendiente; inferencia pública rechaza la referencia ya observada y exige datos posteriores al cutoff.

65 tests aprobados en 16.67 s; Ruff lint/formato correctos. `verification.json` registra contratos/datasets iguales, coincidencia del raw previo, modelos cargables y guardia de inferencia. Se verificó por hashes que código/configuración, versiones anteriores, snapshots y latest/champion siguen intactos. Sólo se añadieron artefactos y se actualizaron los registros de experimentos y roadmap.

## Investigación predictiva multi-activo — 14/10/2026

Protocolo: `config/frontier_research.toml`. Corrida completa: `artifacts/frontier_studies/20261014T134646_e17d06e3`. BTC y ETH diarios, 2465 velas por activo, archivos Binance y checksums. Ocho recetas de logística/XGBoost/Random Forest/LightGBM/CatBoost, ensamble y stacking; tres objetivos, tres folds de selección y un período histórico descriptivo. Purga exacta y embargo de seis días; calibración y stacking temporales fuera del ajuste de componentes. Costos por lado 10 bps de comisión + 5 bps slippage + 1 bp de medio spread.

Se conservaron **892 registros**, **381 ajustes nuevos** y se cargaron **453/500** al presupuesto. Se reutilizaron 32 modelos con calibración. El prototipo `20261014T134018_d1e1df02` se abortó por usar disponibilidad futura de etiquetas de barreras en la máscara de ejecución; sus retornos se marcaron inválidos. Conserva 70 ajustes completados y un cargo conservador de 72. La corrección separa los forecasts causales de las etiquetas usadas para scoring y mantiene intactas las hipótesis.

Ninguna selección cumple todos los criterios. BTC dirección h1 seleccionó Random Forest cruzado: en el período posterior accuracy 49,17%, AUC 0,4924, loss 0,6964 y cero operaciones. La referencia original con la misma política defensiva y costos actuales perdió 1,55%, DD 2,38%, 19 trades, PF 0,5475, capital medio 1,55% y costos 63,64/10000. Pasar a efectivo no acredita mejor predicción. ETH dirección obtuvo AUC 0,4616 y también cero operaciones. Se preservan negativos, resultados de regresiones, regímenes, estabilidad, stress y controles de exposición.

**INCONCLUSO**; ningún reemplazo ni activación. Modelos originales y reservas anteriores conservados por hash. Nueva reserva sellada: 13/04/2027 a 13/10/2027 UTC, con modelos congelados y guardas contra entrenamiento y apertura anticipada. La suite completa pasó 172 pruebas; Ruff y dependencias verificados. Informe, comparación con los tres modelos registrados y gráficos: [FRONTIER_RESEARCH.md](FRONTIER_RESEARCH.md). La fase previa de features/meta-filtros conserva sus 524 registros y 198/210 ajustes en [ADVANCED_RESEARCH.md](ADVANCED_RESEARCH.md).

## Feature engineering anidado BTC/ETH — 15/10/2026

Se ejecutó `feature-study` con protocolo fijo [config/feature_study.toml](config/feature_study.toml): 25 features candidatas en ocho familias, 29 recetas, seis folds externos por activo y dos ventanas internas de 120 días por fold. Purga por disponibilidad de etiquetas y embargo de seis días; mismos datos, objetivo h1 e hiperparámetros XGBoost en todos los controles. Se incluyeron gain, permutación individual/conjunta por bloques 7/30, correlaciones de train, adiciones, ablaciones, baselines, PSI y regímenes. Selección de familias y selección compacta ocurren exclusivamente dentro de train; semilla adicional 1729 para los candidatos principales.

Run `20261015T162842_a5404a4b`: **635/650 ajustes**, incluyendo 623 XGBoost y 12 logísticas; cota prevista 648. 1.728 observaciones externas comunes por activo. Original → compacto: BTC loss 0,703466 → 0,699809, AUC 0,512768 → 0,518394; ETH loss 0,705528 → 0,701537, AUC 0,501349 → 0,502768. El compacto mejora conjuntamente loss/AUC en sólo tres de seis folds y sus intervalos simultáneos incluyen cero. p=0,5 tiene menor loss (0,693147) en ambos activos. El control completo de 39 variables aumenta la brecha train/OOS; eliminar correlaciones mejora ligeramente BTC y empeora ETH.

**Ninguna receta aceptada; se conserva el contrato original.** Historial previamente observado, sin test prospectivo nuevo ni promoción. Se verificaron 430 artefactos y se reconstruyeron 432 series de probabilidades desde modelos guardados y features regeneradas de ZIP verificados, sin ajustes adicionales. Modelos, punteros y reservas permanecen intactos por hash. Validación final: 189 tests, Ruff check/formato y dependencias pasaron; comandos y salidas registrados en `artifacts/feature_analyses/20261015T162842_a5404a4b/checks.json`. Informe y gráficos: [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md); evidencia completa en `artifacts/feature_studies/20261015T162842_a5404a4b/`.

## Triple Barrier y Meta-Labeling — 15/10/2026

Run `20261015T170214_699ffc3e`: **486 ajustes de cota previa 490**, incluidos 238 XGBoost, 36 logísticas y 212 calibraciones. Se compararon 83 variantes de decisión sobre BTC/ETH, seis folds comunes 2022-01-01 a 2026-10-01, purga por final real del evento y embargo de seis días. Se conservaron las 14 features originales para aislar el objetivo. Barreras TP/SL/tiempo con σ histórica, ATR y porcentajes; horizontes 1/3/5/10; Meta-Labeling logístico/XGBoost entrenado sólo sobre propuestas OOS de la base. Raw/cal comparten train; calibración disjunta de 120 días. Se exige rentabilidad neta, estabilidad, cobertura, skill frente a prevalencia y stress, además de crecimiento superior a original/padre/efectivo.

Costos iguales para todas las estrategias: comisión 10 bps/lado, slippage 5 bps/lado y spread completo 2 bps. Long/cash al 100%, una posición, entradas bloqueadas por calendario en los últimos once días de cada fold. Baseline original refit: neto −82,50% BTC / −93,31% ETH, 506/521 trades. Triple Barrier σ h5 calibrado con EV: −80,36% / −89,32%, 530/471 trades. Meta-logística calibrada h1: −4,04% / −37,59%, 16/89 trades; meta TB: −8,58% / −34,72%, 11/63. Menor exposición y operaciones no acreditan rentabilidad. Estas fechas/recetas difieren del backtest estático anterior; no se comparan directamente sus porcentajes.

La variante exploratoria al cierre h5, calibrada y p≥0,60, es positiva en ambos activos (+12,22%/+2,27%), pero gana en sólo 1/6 folds por activo y pierde con costos duplicados (−14,23%/−7,98%). Meta h1 raw gana +13,32% BTC con 22 trades y pierde −37,62% ETH; net_h10 calibrado gana +90,59% BTC y pierde −68,87% ETH. Ninguna de las 83 variantes mantiene neto positivo en ambos bajo stress. La incertidumbre intravela es material: meta TB BTC cambia de −8,58% a +4,96% con TP primero; se conserva SL primero y no se acepta el supuesto favorable como evidencia.

**Ningún candidato principal aceptado; el original permanece.** Histórico previamente observado, sin apertura de reservas futuras ni promoción. Se verificaron 9.141 artefactos, contratos de 274 modelos, 468 series externas con abstenciones, 4.198 forecasts base OOS y 80.302 retornos de trades, sin nuevos ajustes. La suite pasó 211 tests; Ruff check/formato y dependencias correctos. Informe completo, audit del objetivo original, calibración, horizontes, costos, positivos y negativos: [TRIPLE_BARRIER_RESEARCH.md](TRIPLE_BARRIER_RESEARCH.md). Evidencia en `artifacts/label_studies/20261015T170214_699ffc3e/` y verificación en `artifacts/label_analyses/20261015T170214_699ffc3e/`.
