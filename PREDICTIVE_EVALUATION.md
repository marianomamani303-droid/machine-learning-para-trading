# Evaluación predictiva — 2026-10-13

Se agregaron baselines constantes y logísticos, intervalos por bloques, ablaciones fijas y rechazo de candidatos por validación. La ronda ejecutada **no demostró una ventaja probabilística consistente de XGBoost**. Se conservaron los modelos anteriores y la reserva prospectiva; no se promovió ninguna versión.

## Implementación

- `models/baselines.py`: P(up)=0,5, prevalencia del train y logística L2 con C=1, lbfgs y escalado ajustado exclusivamente en train. Coeficientes, intercepto y escalas se guardan como JSON; no se usa pickle.
- `evaluation/uncertainty.py`: bloques móviles no circulares dentro de cada fold, con remuestras pareadas para accuracy, AUC, log-loss y Brier. Informa AUC indefinida si una remuestra tiene una sola clase. Los intervalos no incluyen incertidumbre del refit ni corrección por todas las comparaciones múltiples.
- `evaluation/diagnostics.py`: reliability bins/ECE, dispersión de scores, cobertura y descomposición de Brier agrupada con residual explícito de binning.
- `evaluation/baseline_gate.py`: compara el candidato con los tres baselines usando validación y diferencias pareadas de pérdidas, sin leer el test.
- `training/predictive_study.py`: estudio temporal reproducible con recetas fijas, presupuesto previo, modelos por fold, CSV de predicciones, diagnóstico separado de test observado y recibo de hashes.
- `evaluation/research_report.py`: informe Markdown completo, incluidos experimentos rechazados.

`train` registra la evaluación de baselines en sus metadatos. Una receta cronológica se evalúa sobre su validación purgada, que también participa en la selección de XGBoost: esa evidencia se identifica como desarrollo, no como validación independiente de selección. Una receta temporal utiliza una cola exterior anterior al refit final; ES y calibración se ajustan dentro del bloque previo. Los baselines se ajustan con las mismas filas que el XGBoost de la sonda, excluyendo su bloque de calibración. El modelo final conserva su receta y reajuste originales.

Los candidatos rechazados se conservan para investigación con estado `rejected_by_baselines`. `promote` los bloquea. Un registro directo sin evidencia de los tres baselines queda `research_only_baselines_pending` y tampoco puede promoverse. Los metadatos históricos no se modifican ni se les atribuye una decisión nueva retrospectivamente.

## Protocolo y presupuesto ejecutados

Configuración: [config/predictive_study.toml](config/predictive_study.toml), fijada antes de ejecutar. Referencia: `v20261012T130540_d2bb6ac0`.

Siete folds expansivos/purgados: enero–junio 2022, julio–diciembre 2022, enero–junio 2023, julio–diciembre 2023, enero–junio 2024, julio–diciembre 2024 y enero–principios de junio 2025. Las etiquetas y referencias exteriores terminan antes de `2025-06-05`, inicio del test previamente observado. **1.244 predicciones comunes**, sin modificar filas entre modelos.

Se usó una receta XGBoost fija, declarada en TOML, igual a la referencia actual: 160 árboles, profundidad 2, learning rate 0,03, lambda 5 y min_child_weight 10. No hubo selección de parámetros ni árboles dentro del estudio. Las recetas y períodos son investigación retrospectiva sobre desarrollo ya examinado; no representan una simulación ciega de despliegue histórico.

Ocho recetas XGBoost por fold: técnicas originales, técnicas+SMC, solo SMC y eliminación individual de momentum, forma de vela, volatilidad, volumen y RSI. La eliminación de SMC reutiliza el control técnico, evitando ajustes duplicados. Logística C=1 fija por fold; un ajuste adicional con el train original para el diagnóstico histórico.

**56 ajustes XGBoost y 8 logísticos; 64 en total.** XGBoost usa CPU con un hilo. No se instalaron dependencias. La ronda tardó **40,66 segundos**, incluido preparado y bootstrap. Se usaron 1.000 remuestras por longitud de bloque, de 7 y 30 días, semilla 42 y confianza 95%.

Artefactos completos: [RESULTS.md](artifacts/predictive_studies/9d55ef8f3ddf5feb40e56f0c/RESULTS.md), [summary.json](artifacts/predictive_studies/9d55ef8f3ddf5feb40e56f0c/summary.json), plan y `completed.json` en el mismo directorio. Repetir la ejecución recupera el resultado sin nuevos ajustes; se comprobó con el entrenador bloqueado.

## Resultados de desarrollo

| Modelo | Accuracy | AUC global | Log-loss | Brier |
|---|---:|---:|---:|---:|
| P(up)=0,5 | 49,60% | 0,50000 | 0,693147 | 0,250000 |
| Prevalencia de train | 49,60% | 0,47941 | 0,694962 | 0,250906 |
| Logística L2 | 50,96% | 0,52598 | 0,692674 | 0,249758 |
| XGBoost técnico | 52,89% | 0,52352 | 0,701267 | 0,253818 |
| XGBoost técnico+SMC | 51,53% | 0,52587 | 0,700360 | 0,253372 |
| XGBoost solo SMC | 50,40% | 0,51510 | 0,697015 | 0,251900 |

La prevalencia es constante **dentro** del fold y tiene AUC 0,5 en cada ventana. Su AUC global difiere porque combina constantes distintas entre folds. Se reporta también AUC media por fold: logística 0,53103, XGBoost técnico 0,52952 y técnico+SMC 0,52928.

XGBoost supera accuracy de logística pero pierde en log-loss y Brier: las probabilidades y su confianza son menos útiles que el control lineal. La diferencia de log-loss XGBoost−logística es **+0,008594**, con intervalo [0,001548; 0,015792] para bloques de 7 y [0,001426; 0,015326] para bloques de 30. Es evidencia relativa condicionada a este desarrollo, con intervalos marginales.

La mejora de logística frente a p=0,5 es apenas **−0,000473** de log-loss. Sus intervalos incluyen cero: [−0,005753; 0,003925] y [−0,005162; 0,002950]. Estos intervalos se obtienen restando log(2) a los intervalos de logística: el log-loss del control p=0,5 es idéntico en cada remuestra. No se demuestra una ventaja probabilística robusta de logística frente al control sin información.

La AUC XGBoost tiene intervalos [0,4970; 0,5564] y [0,4991; 0,5555]. Logística tiene [0,4989; 0,5622] y [0,5013; 0,5617]: la conclusión frente a 0,5 es sensible al tamaño de bloque. No se transforma esta señal débil en una afirmación de rentabilidad.

## Ablaciones

| Cambio respecto de XGBoost técnico | Log-loss | Delta log-loss |
|---|---:|---:|
| Retirar momentum | 0,702523 | +0,001256 |
| Retirar forma de vela | 0,701430 | +0,000163 |
| Retirar volatilidad | 0,698490 | −0,002777 |
| Retirar volumen | 0,701633 | +0,000365 |
| Retirar RSI | 0,702703 | +0,001436 |
| Agregar SMC | 0,700360 | −0,000908 |

Los intervalos de momentum, forma, volumen y agregado SMC incluyen cero. Retirar volatilidad mejora el punto estimado, pero su intervalo también incluye cero para ambas longitudes: es una hipótesis de simplificación, no una mejora aceptada.

Retirar RSI empeora log-loss con intervalos marginales positivos: [0,000113; 0,002743] y [0,000270; 0,002499]. Es la señal condicional más clara de esta ablación, pequeña y sin corrección familiar por multiplicidad. No demuestra utilidad suficiente del sistema ni independencia respecto de otras variables correlacionadas.

**Las ocho recetas XGBoost fallaron la regla completa de baselines.** No se eliminó una familia ni se seleccionó la variante de mejor resultado para evaluar/promover sobre test.

## Train, validación y test observado

El XGBoost histórico se cargó sin reajuste. La logística y prevalencia se ajustaron una vez con exactamente sus 1.448 filas de train; los cuatro modelos se midieron sobre las mismas 482 filas de validación y 483 de test. El test se inspeccionó después de congelar desarrollo y se mantuvo fuera de selección, ablaciones y rechazo.

| Partición | XGBoost AUC | XGBoost log-loss | Logística AUC | Logística log-loss |
|---|---:|---:|---:|---:|
| Train | 0,68638 | 0,656659 | 0,56004 | 0,687149 |
| Validación | 0,50805 | 0,699051 | 0,55002 | 0,688842 |
| Test observado | 0,47870 | 0,707352 | 0,49739 | 0,694631 |

XGBoost tiene una brecha de log-loss train→validación de +0,042392 y train→test de +0,050693; logística, +0,001693 y +0,007482. La diferencia respalda una mayor brecha de generalización de XGBoost, compatible con sobreajuste y deriva; no separa automáticamente sus causas.

En test, ambos pierden frente a p=0,5 (0,693147). La mejora descriptiva de logística frente a XGBoost no constituye una nueva confirmación ciega. Las curvas de calibración, intervalos y CSV de todas las particiones se conservan en el informe.

## SMC y probabilidades casi constantes

Se reprodujeron los siete modelos S3 y sus calibradores originales sobre sus scores de desarrollo. Los rangos raw→calibrado son, por ejemplo, 0,008484→0,001106 en fold 2 y 0,003513→0,000113 en fold 4. En fold 7 el calibrador tiene pendiente cero y elimina el ranking: span 0 y AUC 0,5. La casi constancia combina poca variación del modelo base y compresión de la sigmoide; el desplazamiento de prevalencia puede producir estados sin discriminar oportunidades.

Los candidatos completos se inspeccionaron separadamente sobre sus etiquetas ya usadas, sin nuevos ajustes ni métricas independientes: [diagnóstico de SMC](artifacts/predictive_diagnostics/smc-models-final-20261013.json). Este análisis no modifica umbrales ni evalúa el test prospectivo.

## Uso y criterio de rechazo

```powershell
.\.venv\Scripts\python.exe -m trading_ml predictive-study --version v20261012T130540_d2bb6ac0
.\.venv\Scripts\python.exe scripts/diagnose_smc_probabilities.py --versions v20261012T134917_f88d6f93 v20261012T143623_0d9b01f0
```

La regla exige ganar al p=0,5, prevalencia y logística: mejora mínima de log-loss de 0,001, Brier inferior y límites superiores negativos de las diferencias pareadas de log-loss/Brier para ambas longitudes de bloque. En el estudio exige además ganar simultáneamente pérdidas en al menos 5/7 folds frente a cada baseline. Es un filtro conservador de desarrollo, no un certificado de rentabilidad. La promoción continúa siendo explícita y está bloqueada para un candidato rechazado o sin evaluación requerida.

Los tests cubren ajuste de scaler/prevalencia en pasado, purga, bloques sin cruces de fold, AUC con empates contra sklearn, intervalos pareados, casos positivos/negativos del gate, invariancia al modificar test, cobertura de familias, presupuesto, caché y bloqueo de promoción.

Fuentes: [calibración y Brier](https://scikit-learn.org/stable/modules/calibration.html), [logística regularizada](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [bootstrap temporal](https://arch.readthedocs.io/en/stable/bootstrap/timeseries-bootstraps.html). Calibración, discriminación y utilidad económica son propiedades distintas; estos resultados evalúan las primeras dos.
