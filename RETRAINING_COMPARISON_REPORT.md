# RETRAINING COMPARISON REPORT

Fecha: 17 de octubre de 2026. Investigación retrospectiva, sin promoción ni operaciones reales.

## A. Resumen ejecutivo

**¿Mejoraron después de entrenar con más datos? No se demostró una mejora estadísticamente significativa ni robusta. Hay mejoras numéricas y deterioros, pero ninguna comparación supera el criterio corregido.**

Se completaron **441 ajustes** de estimadores/calibradores, incluidos 72 modelos OOF y los expertos de régimen. Se conservaron los resultados anteriores. Se compararon 26 combinaciones de modelo, activo, temporalidad y objetivo, más LSTM sobre una cartera BTC/ETH.

**0/26 comparaciones** muestran una reducción de log-loss con p de Holm < 0,05 e intervalos positivos en ambos bloques. Esa evidencia es condicional a estos ajustes y ventanas ya investigadas; no constituye confirmación prospectiva ni una ventaja robusta de trading.

En BTC diario, el mejor A descriptivo por AUC es **smc** (0.55249); el mejor B es **random_forest_cross** (0.55319). Son rangos retrospectivos, no modelos seleccionados para producción.

**Qué significa “antes”:** A reconstruido con las recetas existentes y los datos originales desde 2020, evaluado en las mismas ventanas que B. No se resta un resultado antiguo de 2025–2026 de uno nuevo de 2023–2025. Esos períodos distintos no forman una comparación válida.

## B. Datos utilizados

El inventario incorporado contiene 491 datasets, 11.749.803 registros incluyendo temporalidades derivadas y 8.983.894 observaciones base. Los recuentos base tampoco equivalen a observaciones estadísticamente independientes. Se verificaron los hashes de los 491 CSV y sus manifiestos; la continuidad/duplicación de todo el inventario se recuperó de los manifiestos validados, y los datos elegidos se verificaron directamente. La auditoría no volvió a interpretar fines de semana como errores del proveedor.

| Mercado | Activos/contratos | Datasets | Filas | Cobertura disponible |
|---|---|---|---|---|
| cash_index_context | 8 | 8 | 82,707 | 1962-01-02 a 2026-10-01 |
| crypto_dated_futures | 50 | 200 | 1,126,654 | 2020-06-11 a 2026-09-25 |
| crypto_perpetual | 6 | 24 | 1,814,908 | 2020-01-01 a 2026-10-01 |
| crypto_spot | 12 | 48 | 4,301,790 | 2017-08-17 a 2026-10-01 |
| exchange_dated_futures | 161 | 161 | 28,696 | 2013-05-20 a 2026-09-17 |
| indicative_otc_forex | 10 | 50 | 4,395,048 | 2012-01-01 a 2026-10-01 |

**Cuarentena:** 1.552.659 filas recientes permanecen excluidas del ajuste, selección, calibración y evaluación. Se impuso el límite 2025-07-01 y se respetaron las reservas previas. No se abrió una nueva prueba sellada.

| Activo | A: filas diarias | B: filas diarias | Añadidas | Días recuperados | Discrepancias con agregación antigua |
|---|---|---|---|---|---|
| BTCUSDT | 2008 | 2874 | 866 | 18 | {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 5} |
| ETHUSDT | 2008 | 2874 | 866 | 18 | {'open': 3, 'high': 0, 'low': 0, 'close': 0, 'volume': 4} |

A utiliza 2020-01-01–2025-06-30. B añade 2017-08-18–2019-12-31. Las 2.008 filas comunes de cada activo, incluidas todas sus features y etiquetas, se conservaron idénticas. La intersección de features deja 948/1.313/1.679 filas de ajuste A y 1.674/2.039/2.405 B en los tres folds tabulares.

**Desviación necesaria y predeclarada antes de entrenar:** los nuevos D1 agregados desde M15 tenían huecos y no incluían flujo. Se recuperaron 58 ZIP mensuales D1 oficiales, con checksum, para obtener precios y volúmenes taker reales de 2017–2019. No hubo interpolación. Por tanto, este experimento evalúa historia ampliada con reparación nativa documentada; no permite atribuir resultados exclusivamente a los CSV inicialmente incorporados, a datos de 2026 o a nuevos activos.

Forex: EUR/USD indicativo FXCM, bid/ask H1 como fuente y ejecución, agregación NY17 con DST para diario y UTC para H4. La fuente nueva extiende a 2012, pero sólo suma 130 D1 utilizables al primer ajuste; H4 pasa de 3.534 a 11.999 filas de entrenamiento. No se sustituyó D1 NY17 por D1 UTC. No se imputaron sesiones incompletas.

Los 12 pares spot y 10 Forex, sus intervalos, cobertura, huecos, cuarentena y razones de inclusión/exclusión están en [dataset_compatibility.csv](artifacts/retraining_comparison/20261017_controlled/dataset_compatibility.csv). BTC/ETH y EUR/USD se eligieron por continuidad de las recetas y representatividad, antes de observar resultados; los demás activos no recibieron una conclusión de mejora.

Perpetuos y futuros con vencimiento carecen de una receta operativa anterior comparable de margen, funding y rollover en esta investigación. Los contratos individuales no se concatenaron. Los índices de contado no son instrumentos ejecutables y requieren disponibilidad al cierre local. No se crearon modelos para esos mercados ni se mezclaron sus volúmenes con spot/Forex.

## C. Modelos y procedimientos

| Receta | Configuración conservada | Features/objetivo |
|---|---|---|
| original | XGBoost 160 árboles, profundidad 2, lr 0,03; subsample/colsample 0,85; lambda 5; min_child_weight 10 | 14 originales; dirección close/close siguiente |
| logistic | StandardScaler sólo en train; LogisticRegression C=1, lbfgs, max_iter=1000 | 14 originales; misma dirección |
| random_forest_cross | 200 árboles; profundidad 5; min_samples_leaf=20; CPU, 1 worker | LOCAL+CROSS existentes, incluido flujo real |
| lightgbm_cross | 160 árboles; 7 hojas; profundidad 3; lr 0,03; lambda 5; min_child_samples 40 | Misma receta cruzada existente |
| catboost_cross | 160 iteraciones; profundidad 3; lr 0,03; l2_leaf_reg 10; CPU | Misma receta cruzada existente |
| smc / structure / regime_rules | Mismo XGBoost y detectores causales existentes | 14 + SMC / estructura / régimen |
| specialists_rules | XGBoost global + expertos existentes; mínimo 80 filas, 20/clase, shrinkage 200 | 14 originales por estado; fallback global |
| tb_vol_h5_cal | Triple Barrier volatilidad: TP 1,5σ, SL 1σ, 5 barras, stop-first; XGBoost original | Rentabilidad neta del evento; calibración 120 días |
| meta_net_h1_logistic_cal_55 | OOF trimestral desde 2021; oferta base ≥0,55; logística C=0,1 + calibrador | 14 + probabilidad base OOF; beneficio neto, no dirección |
| fx_price / fx_compact | XGBoost 160, profundidad 3, lr 0,03; calibrador en pasado disjunto | Features price/compact existentes, calendario y spread propios |
| lstm_sharpe_net | LSTM 20 unidades; secuencia 63; dropout 0,2; Adam 0,001; 30 epochs, patience 7; batch 128; clipping 1 | 8 features DMN; Sharpe neto, cartera long/cash; validación 180 días |

No se añadieron arquitecturas. La receta specialists_rules ya es una combinación de modelo global y expertos. No se inventó un híbrido nuevo. Los filtros de momentum/reversión con objetivos de evento diferentes no se reinterpretaron como clasificadores de dirección.

**Walk-forward:** 2023, 2024 y primer semestre de 2025; ambos entrenamientos terminan antes de cada evaluación. Purga por target_time y embargo: 6 días tabular/eventos, 7 Forex y 1 DMN. Las ventanas interiores y el calibrador se mantienen en pasado. No hay búsqueda de hiperparámetros ni umbrales sobre OOS. La evaluación predictiva tabular excluye una cola calendario de 12 días y Forex una de 20 días; la ejecución usa además sus reglas originales de liquidación. Son ventanas de evaluación retrospectiva, no un test independiente nuevo.

Semillas 42 y 1729 para clasificadores estocásticos y LSTM. Logística, especialistas y Forex usan la semilla original 42. No se eligió una semilla ganadora. Las métricas finales son promedios de métricas por semilla, no un ensemble inventado; cada fecha se remuestrea conjuntamente entre semillas.

LSTM conserva la elección de época por Sharpe neto de validación. B dispone también de secuencias antiguas y del contexto que faltaba al calentamiento inicial de A; sus secuencias, retornos y volatilidades compartidos con A se fijaron idénticos. Sus asignaciones no son probabilidades calibradas: log-loss, Brier y calibración probabilística no son aplicables sin cambiar la receta.

**Experimento A histórico:** se cargaron y verificaron ocho modelos guardados y se recuperaron sus recetas originales, ventanas y semillas en registered_model_audit.json; historical_results.json conserva sus métricas reales. Seis se evaluaron en un tramo común válido de 15 observaciones (2025-06-05 a 2025-06-19), previa igualdad de sus features con el snapshot. Dos modelos ajustados hasta 2026 fueron excluidos por solapamiento temporal. No se usó ese tramo pequeño para elegir el mejor modelo ni se lo confundió con los folds principales.

| Versión histórica | Fin de selección | Features | AUC guardada | Accuracy guardada | Log-loss guardado |
|---|---|---|---|---|---|
| v20261012T125624_1576ecf1 | 2025-06-04 | 14 | 0.47870 | 0.49275 | 0.70735 |
| v20261012T130154_6893e846 | 2025-06-04 | 14 | 0.47870 | 0.49275 | 0.70735 |
| v20261012T130539_2a7f3cde | 2025-03-22 | 14 | 0.47915 | 0.48172 | 0.70553 |
| v20261012T130540_d2bb6ac0 | 2025-06-04 | 14 | 0.47870 | 0.49275 | 0.70735 |
| v20261012T131650_5106daf4 | 2025-06-04 | 22 | 0.47572 | 0.49068 | 0.69682 |
| v20261012T134917_f88d6f93 | 2025-06-04 | 46 | N/A | N/A | N/A |
| v20261012T143622_ff1c464c | 2026-10-01 | 14 | N/A | N/A | N/A |
| v20261012T143623_0d9b01f0 | 2026-10-01 | 46 | N/A | N/A | N/A |

N/A en los metadatos históricos significa prueba pendiente, no resultado cero. Sus métricas corresponden a las fechas guardadas en splits y no son restables directamente contra las nuevas ventanas.

## D. Comparación predictiva

Todas las diferencias siguientes son B − A sobre las mismas observaciones. Accuracy se expresa como fracción; multiplicar su diferencia por 100 da puntos porcentuales. En log-loss, una diferencia negativa es favorable. La evaluación estadística de cada receta se basa en log-loss, el endpoint primario fijado antes de ajustar; no se intercambió por el indicador que saliera mejor.

### BTCUSDT — 1d — close_direction

| Modelo | Métrica | Antes A | Después B | Diferencia | Evaluación |
|---|---|---|---|---|---|
| catboost_cross | roc_auc | 0.53132 | 0.54706 | 0.01574 | Sin mejora respaldada del endpoint primario |
| catboost_cross | accuracy | 0.52169 | 0.53196 | 0.01027 | Sin mejora respaldada del endpoint primario |
| catboost_cross | log_loss | 0.69195 | 0.68916 | -0.00279 | Sin mejora respaldada del endpoint primario |
| lightgbm_cross | roc_auc | 0.52032 | 0.53642 | 0.01611 | Sin mejora respaldada del endpoint primario |
| lightgbm_cross | accuracy | 0.52626 | 0.52397 | -0.00228 | Sin mejora respaldada del endpoint primario |
| lightgbm_cross | log_loss | 0.70129 | 0.69393 | -0.00736 | Sin mejora respaldada del endpoint primario |
| logistic | roc_auc | 0.54396 | 0.54001 | -0.00395 | Sin mejora respaldada del endpoint primario |
| logistic | accuracy | 0.53995 | 0.52055 | -0.01941 | Sin mejora respaldada del endpoint primario |
| logistic | log_loss | 0.68925 | 0.68955 | 0.00031 | Sin mejora respaldada del endpoint primario |
| original | roc_auc | 0.54586 | 0.54814 | 0.00228 | Sin mejora respaldada del endpoint primario |
| original | accuracy | 0.54167 | 0.53824 | -0.00342 | Sin mejora respaldada del endpoint primario |
| original | log_loss | 0.69113 | 0.69148 | 0.00034 | Sin mejora respaldada del endpoint primario |
| random_forest_cross | roc_auc | 0.54202 | 0.55319 | 0.01117 | Sin mejora respaldada del endpoint primario |
| random_forest_cross | accuracy | 0.53139 | 0.54395 | 0.01256 | Sin mejora respaldada del endpoint primario |
| random_forest_cross | log_loss | 0.68949 | 0.68788 | -0.00161 | Sin mejora respaldada del endpoint primario |
| regime_rules | roc_auc | 0.53937 | 0.54338 | 0.00401 | Sin mejora respaldada del endpoint primario |
| regime_rules | accuracy | 0.52397 | 0.53710 | 0.01313 | Sin mejora respaldada del endpoint primario |
| regime_rules | log_loss | 0.69200 | 0.69230 | 0.00029 | Sin mejora respaldada del endpoint primario |
| smc | roc_auc | 0.55249 | 0.55310 | 0.00061 | Sin mejora respaldada del endpoint primario |
| smc | accuracy | 0.54224 | 0.53653 | -0.00571 | Sin mejora respaldada del endpoint primario |
| smc | log_loss | 0.68950 | 0.69021 | 0.00071 | Sin mejora respaldada del endpoint primario |
| specialists_rules | roc_auc | 0.53703 | 0.52910 | -0.00793 | Sin mejora respaldada del endpoint primario |
| specialists_rules | accuracy | 0.52511 | 0.52397 | -0.00114 | Sin mejora respaldada del endpoint primario |
| specialists_rules | log_loss | 0.69462 | 0.69845 | 0.00383 | Sin mejora respaldada del endpoint primario |
| structure | roc_auc | 0.55223 | 0.55313 | 0.00091 | Sin mejora respaldada del endpoint primario |
| structure | accuracy | 0.53368 | 0.53938 | 0.00571 | Sin mejora respaldada del endpoint primario |
| structure | log_loss | 0.68960 | 0.68991 | 0.00031 | Sin mejora respaldada del endpoint primario |

### BTCUSDT — 1d — net_event_profit

| Modelo | Métrica | Antes A | Después B | Diferencia | Evaluación |
|---|---|---|---|---|---|
| tb_vol_h5_cal | roc_auc | 0.48456 | 0.49248 | 0.00792 | Sin mejora respaldada del endpoint primario |
| tb_vol_h5_cal | accuracy | 0.50913 | 0.51027 | 0.00114 | Sin mejora respaldada del endpoint primario |
| tb_vol_h5_cal | log_loss | 0.70093 | 0.70052 | -0.00040 | Sin mejora respaldada del endpoint primario |
| meta_net_h1_logistic_cal_55 | roc_auc | 0.53420 | 0.53269 | -0.00151 | Sin mejora respaldada del endpoint primario |
| meta_net_h1_logistic_cal_55 | accuracy | 0.58557 | 0.57426 | -0.01132 | Sin mejora respaldada del endpoint primario |
| meta_net_h1_logistic_cal_55 | log_loss | 0.68265 | 0.68183 | -0.00082 | Sin mejora respaldada del endpoint primario |

### ETHUSDT — 1d — close_direction

| Modelo | Métrica | Antes A | Después B | Diferencia | Evaluación |
|---|---|---|---|---|---|
| catboost_cross | roc_auc | 0.50756 | 0.52875 | 0.02119 | Sin mejora respaldada del endpoint primario |
| catboost_cross | accuracy | 0.50228 | 0.52397 | 0.02169 | Sin mejora respaldada del endpoint primario |
| catboost_cross | log_loss | 0.69665 | 0.69240 | -0.00424 | Sin mejora respaldada del endpoint primario |
| lightgbm_cross | roc_auc | 0.52630 | 0.49941 | -0.02689 | Sin mejora respaldada del endpoint primario |
| lightgbm_cross | accuracy | 0.50685 | 0.49315 | -0.01370 | Sin mejora respaldada del endpoint primario |
| lightgbm_cross | log_loss | 0.69795 | 0.70263 | 0.00468 | Sin mejora respaldada del endpoint primario |
| logistic | roc_auc | 0.51951 | 0.53518 | 0.01567 | Sin mejora respaldada del endpoint primario |
| logistic | accuracy | 0.51142 | 0.53311 | 0.02169 | Sin mejora respaldada del endpoint primario |
| logistic | log_loss | 0.69304 | 0.69156 | -0.00148 | Sin mejora respaldada del endpoint primario |
| original | roc_auc | 0.51628 | 0.50759 | -0.00869 | Sin mejora respaldada del endpoint primario |
| original | accuracy | 0.51313 | 0.51427 | 0.00114 | Sin mejora respaldada del endpoint primario |
| original | log_loss | 0.70053 | 0.70017 | -0.00036 | Sin mejora respaldada del endpoint primario |
| random_forest_cross | roc_auc | 0.52185 | 0.51580 | -0.00605 | Sin mejora respaldada del endpoint primario |
| random_forest_cross | accuracy | 0.50799 | 0.50628 | -0.00171 | Sin mejora respaldada del endpoint primario |
| random_forest_cross | log_loss | 0.69372 | 0.69358 | -0.00014 | Sin mejora respaldada del endpoint primario |
| regime_rules | roc_auc | 0.50717 | 0.50401 | -0.00316 | Sin mejora respaldada del endpoint primario |
| regime_rules | accuracy | 0.51427 | 0.50457 | -0.00970 | Sin mejora respaldada del endpoint primario |
| regime_rules | log_loss | 0.70143 | 0.70019 | -0.00125 | Sin mejora respaldada del endpoint primario |
| smc | roc_auc | 0.50633 | 0.51849 | 0.01216 | Sin mejora respaldada del endpoint primario |
| smc | accuracy | 0.49201 | 0.51712 | 0.02511 | Sin mejora respaldada del endpoint primario |
| smc | log_loss | 0.69977 | 0.69811 | -0.00167 | Sin mejora respaldada del endpoint primario |
| specialists_rules | roc_auc | 0.50500 | 0.50430 | -0.00069 | Sin mejora respaldada del endpoint primario |
| specialists_rules | accuracy | 0.50114 | 0.49772 | -0.00342 | Sin mejora respaldada del endpoint primario |
| specialists_rules | log_loss | 0.70314 | 0.70594 | 0.00280 | Sin mejora respaldada del endpoint primario |
| structure | roc_auc | 0.50744 | 0.52976 | 0.02231 | Sin mejora respaldada del endpoint primario |
| structure | accuracy | 0.50114 | 0.52169 | 0.02055 | Sin mejora respaldada del endpoint primario |
| structure | log_loss | 0.69969 | 0.69548 | -0.00421 | Sin mejora respaldada del endpoint primario |

### ETHUSDT — 1d — net_event_profit

| Modelo | Métrica | Antes A | Después B | Diferencia | Evaluación |
|---|---|---|---|---|---|
| meta_net_h1_logistic_cal_55 | roc_auc | 0.50000 | 0.57895 | 0.07895 | Sin mejora respaldada del endpoint primario |
| meta_net_h1_logistic_cal_55 | accuracy | 0.44970 | 0.55030 | 0.10059 | Sin mejora respaldada del endpoint primario |
| meta_net_h1_logistic_cal_55 | log_loss | 0.70290 | 0.68579 | -0.01712 | Sin mejora respaldada del endpoint primario |
| tb_vol_h5_cal | roc_auc | 0.51487 | 0.52447 | 0.00960 | Sin mejora respaldada del endpoint primario |
| tb_vol_h5_cal | accuracy | 0.54338 | 0.53767 | -0.00571 | Sin mejora respaldada del endpoint primario |
| tb_vol_h5_cal | log_loss | 0.68499 | 0.68411 | -0.00088 | Sin mejora respaldada del endpoint primario |

### EURUSD — 1d — close_direction

| Modelo | Métrica | Antes A | Después B | Diferencia | Evaluación |
|---|---|---|---|---|---|
| fx_compact | roc_auc | 0.50993 | 0.51343 | 0.00350 | Sin mejora respaldada del endpoint primario |
| fx_compact | accuracy | 0.50934 | 0.51783 | 0.00849 | Sin mejora respaldada del endpoint primario |
| fx_compact | log_loss | 0.69388 | 0.69325 | -0.00064 | Sin mejora respaldada del endpoint primario |
| fx_price | roc_auc | 0.50324 | 0.50205 | -0.00119 | Sin mejora respaldada del endpoint primario |
| fx_price | accuracy | 0.49745 | 0.51104 | 0.01358 | Sin mejora respaldada del endpoint primario |
| fx_price | log_loss | 0.69470 | 0.69499 | 0.00029 | Sin mejora respaldada del endpoint primario |

### EURUSD — 4h — close_direction

| Modelo | Métrica | Antes A | Después B | Diferencia | Evaluación |
|---|---|---|---|---|---|
| fx_compact | roc_auc | 0.51566 | 0.51506 | -0.00060 | Sin mejora respaldada del endpoint primario |
| fx_compact | accuracy | 0.51036 | 0.50891 | -0.00146 | Sin mejora respaldada del endpoint primario |
| fx_compact | log_loss | 0.69279 | 0.69307 | 0.00028 | Sin mejora respaldada del endpoint primario |
| fx_price | roc_auc | 0.50949 | 0.51394 | 0.00445 | Sin mejora respaldada del endpoint primario |
| fx_price | accuracy | 0.51533 | 0.50365 | -0.01168 | Sin mejora respaldada del endpoint primario |
| fx_price | log_loss | 0.69323 | 0.69292 | -0.00031 | Sin mejora respaldada del endpoint primario |

La tabla completa con Balanced Accuracy, Brier, precision, recall, F1, ECE, diferencias relativas y número de fechas/semillas está en [comparison.csv](artifacts/retraining_comparison/20261017_controlled/comparison.csv). Las matrices y bins de calibración por semilla están en all_predictive_metrics.json. Net_event_profit mide beneficio de la operación, no si el precio sube al siguiente cierre; sus AUC no se comparan con close_direction.

![Diferencias predictivas](artifacts/retraining_comparison/20261017_controlled/predictive_differences.png)

![Calibración](artifacts/retraining_comparison/20261017_controlled/calibration.png)

### Baselines y matrices de confusión

constant_05 usa probabilidades 0,5 (log-loss 0,693147; su decisión con empate predice subida). train_prevalence y majority_class se obtienen exclusivamente de train. La moneda aleatoria con semilla 42 se presenta sólo como decisión, sin tratar sus salidas 0/1 como probabilidades ciertas. La logística existente controla capacidad predictiva simple. En Forex, price es también el control de compact.

| Activo | Receta B | Control | Ganancia log-loss vs control | IC95 bloque largo | p Holm controles |
|---|---|---|---|---|---|
| ETHUSDT | original | constant_05 | -0.00703 | [-0.0127, 0.00514] | 1.00000 |
| ETHUSDT | original | train_prevalence | -0.00711 | [-0.01309, 0.00594] | 1.00000 |
| ETHUSDT | original | logistic_l2 | -0.00861 | [-0.01414, 0.00039] | 1.00000 |
| BTCUSDT | original | constant_05 | 0.00167 | [-0.00705, 0.01092] | 1.00000 |
| BTCUSDT | original | train_prevalence | 0.00138 | [-0.00695, 0.01093] | 1.00000 |
| BTCUSDT | original | logistic_l2 | -0.00192 | [-0.00898, 0.00621] | 1.00000 |
| EURUSD | fx_compact | constant_05 | 0.00008 | [-0.00111, 0.00093] | 1.00000 |
| EURUSD | fx_compact | train_prevalence | 0.00008 | [-0.00109, 0.00091] | 1.00000 |
| EURUSD | fx_compact | constant_05 | -0.00010 | [-0.00474, 0.00472] | 1.00000 |
| EURUSD | fx_compact | train_prevalence | -0.00027 | [-0.00623, 0.00512] | 1.00000 |

| Activo | Modelo, semilla 42 | Matriz A [[TN,FP],[FN,TP]] | Matriz B |
|---|---|---|---|
| BTCUSDT | original | [[209, 216], [184, 267]] | [[181, 244], [168, 283]] |
| BTCUSDT | logistic | [[173, 252], [151, 300]] | [[147, 278], [142, 309]] |
| BTCUSDT | random_forest_cross | [[208, 217], [193, 258]] | [[196, 229], [169, 282]] |
| ETHUSDT | original | [[262, 169], [255, 190]] | [[299, 132], [294, 151]] |
| ETHUSDT | logistic | [[122, 309], [119, 326]] | [[246, 185], [224, 221]] |
| ETHUSDT | random_forest_cross | [[185, 246], [185, 260]] | [[278, 153], [269, 176]] |

## E. Comparación económica

Spot: long/cash, señal al cierre, ejecución futura a apertura, 10 bps de comisión + 5 de slippage + 1 de medio spread por lado; sin posiciones solapadas. Mismo umbral 0,55 y protocolo de salida por receta. Se encadenan folds liquidando posiciones. Los resultados son promedios entre semillas; no una cartera que elige retrospectivamente la mejor semilla.

Forex usa bid/ask observado, comisión 0,35 bps/lado, slippage 0,1 pip y swap adverso de 100 bps/año como escenario, no cotización histórica de un broker. El estrés duplica comisión/slippage/swap y añade 1 pip de spread; no se presenta como duplicación exacta del spread observado. Sharpe Forex se calcula sobre sesiones NY17 y 252 sesiones/año; spot sobre 365. Los precios son indicativos, sin garantía de fills, liquidez ni drawdown intravela.

Las predicciones direccionales son close/close y la ejecución opera a aperturas: son objetivos distintos. Mejor Accuracy no implica mayor retorno. Los costos compuestos suman los flujos reales de cada fold escalados al capital encadenado. En Forex la exposición es tiempo en mercado; en spot la fracción de barras expuestas.

### Retornos y riesgo de los modelos tabulares/de eventos

| Activo/TF | Modelo | Retorno A | Retorno B | Δ retorno | Sharpe A → B | MDD B | Retorno B estrés |
|---|---|---|---|---|---|---|---|
| BTCUSDT 1d | catboost_cross | -1.89% | -22.70% | -20.81 pp | 0.086 → -0.270 | -43.80% | -63.99% |
| BTCUSDT 1d | lightgbm_cross | -31.38% | -19.56% | +11.82 pp | -0.358 → -0.140 | -47.91% | -73.33% |
| BTCUSDT 1d | logistic | 45.52% | 109.40% | +63.88 pp | 0.905 → 1.491 | -12.08% | 31.24% |
| BTCUSDT 1d | meta_net_h1_logistic_cal_55 | 0.00% | 0.00% | +0.00 pp | N/A → N/A | 0.00% | 0.00% |
| BTCUSDT 1d | original | -6.66% | 9.05% | +15.71 pp | 0.021 → 0.248 | -35.17% | -50.50% |
| BTCUSDT 1d | random_forest_cross | -10.82% | -7.08% | +3.74 pp | -0.074 → 0.006 | -36.53% | -52.24% |
| BTCUSDT 1d | regime_rules | -10.19% | 1.70% | +11.88 pp | -0.063 → 0.149 | -42.70% | -54.30% |
| BTCUSDT 1d | smc | -19.85% | -20.50% | -0.65 pp | -0.200 → -0.198 | -46.04% | -67.86% |
| BTCUSDT 1d | specialists_rules | -12.55% | 0.38% | +12.93 pp | -0.076 → 0.155 | -38.52% | -63.37% |
| BTCUSDT 1d | structure | -21.30% | -31.03% | -9.73 pp | -0.226 → -0.397 | -50.36% | -71.79% |
| BTCUSDT 1d | tb_vol_h5_cal | -30.34% | -16.10% | +14.25 pp | -0.476 → -0.134 | -45.45% | -50.83% |
| ETHUSDT 1d | catboost_cross | -45.74% | -15.20% | +30.53 pp | -0.536 → -0.082 | -44.64% | -47.48% |
| ETHUSDT 1d | lightgbm_cross | -44.36% | -7.01% | +37.35 pp | -0.402 → 0.066 | -28.08% | -51.43% |
| ETHUSDT 1d | logistic | -11.94% | 12.35% | +24.29 pp | -0.057 → 0.318 | -27.99% | -13.30% |
| ETHUSDT 1d | meta_net_h1_logistic_cal_55 | 0.00% | 0.00% | +0.00 pp | N/A → N/A | 0.00% | 0.00% |
| ETHUSDT 1d | original | -32.09% | -2.81% | +29.28 pp | -0.319 → 0.093 | -38.95% | -39.91% |
| ETHUSDT 1d | random_forest_cross | -11.25% | 2.96% | +14.21 pp | 0.038 → 0.172 | -37.24% | -35.05% |
| ETHUSDT 1d | regime_rules | -17.90% | -14.70% | +3.20 pp | -0.094 → -0.050 | -35.44% | -48.94% |
| ETHUSDT 1d | smc | -3.01% | -14.37% | -11.36 pp | 0.111 → -0.046 | -40.61% | -47.80% |
| ETHUSDT 1d | specialists_rules | -44.56% | -17.16% | +27.40 pp | -0.493 → -0.041 | -43.74% | -61.57% |
| ETHUSDT 1d | structure | -26.52% | -3.21% | +23.31 pp | -0.213 → 0.114 | -43.32% | -41.75% |
| ETHUSDT 1d | tb_vol_h5_cal | -5.46% | -4.07% | +1.39 pp | -0.557 → -0.633 | -4.07% | -4.38% |
| EURUSD 1d | fx_compact | -2.28% | -0.80% | +1.48 pp | -0.386 → -0.161 | -1.85% | -1.09% |
| EURUSD 1d | fx_price | 0.00% | -2.31% | -2.31 pp | N/A → -0.666 | -3.25% | -2.18% |
| EURUSD 4h | fx_compact | -0.01% | -0.13% | -0.12 pp | -0.070 → -0.484 | -0.21% | -0.23% |
| EURUSD 4h | fx_price | 0.00% | 0.00% | +0.00 pp | N/A → N/A | 0.00% | 0.00% |

| Activo/TF | Modelo B | Operaciones | Aciertos | Profit Factor | Exposición | Costo ejecución, capital inicial 10.000 |
|---|---|---|---|---|---|---|
| BTCUSDT 1d | catboost_cross | 238.5 | 46.75% | 0.878 | 42.79% | 6380.59 |
| BTCUSDT 1d | lightgbm_cross | 345.0 | 46.09% | 0.933 | 54.91% | 10356.56 |
| BTCUSDT 1d | logistic | 146.0 | 57.53% | 1.649 | 23.77% | 7354.79 |
| BTCUSDT 1d | meta_net_h1_logistic_cal_55 | 0.0 | N/A | N/A | 0.00% | 0.00 |
| BTCUSDT 1d | original | 247.5 | 49.69% | 1.040 | 40.87% | 8096.04 |
| BTCUSDT 1d | random_forest_cross | 208.0 | 48.79% | 0.960 | 37.56% | 6007.64 |
| BTCUSDT 1d | regime_rules | 250.0 | 48.80% | 1.009 | 40.05% | 8012.77 |
| BTCUSDT 1d | smc | 283.0 | 47.70% | 0.910 | 45.01% | 8001.84 |
| BTCUSDT 1d | specialists_rules | 315.0 | 47.94% | 1.001 | 48.48% | 10240.07 |
| BTCUSDT 1d | structure | 279.5 | 47.59% | 0.863 | 43.36% | 7662.85 |
| BTCUSDT 1d | tb_vol_h5_cal | 167.0 | 43.71% | 0.932 | 32.24% | 4828.69 |
| ETHUSDT 1d | catboost_cross | 149.5 | 51.16% | 0.928 | 31.08% | 4906.62 |
| ETHUSDT 1d | lightgbm_cross | 203.0 | 47.78% | 0.974 | 38.90% | 6809.92 |
| ETHUSDT 1d | logistic | 81.0 | 55.56% | 1.088 | 17.30% | 3198.40 |
| ETHUSDT 1d | meta_net_h1_logistic_cal_55 | 0.0 | N/A | N/A | 0.00% | 0.00 |
| ETHUSDT 1d | original | 151.0 | 51.37% | 0.982 | 29.98% | 5006.22 |
| ETHUSDT 1d | random_forest_cross | 144.0 | 54.17% | 1.009 | 30.26% | 5060.19 |
| ETHUSDT 1d | regime_rules | 161.0 | 51.88% | 0.936 | 31.67% | 5205.85 |
| ETHUSDT 1d | smc | 154.5 | 51.43% | 0.938 | 29.93% | 5193.00 |
| ETHUSDT 1d | specialists_rules | 240.0 | 49.58% | 0.946 | 42.47% | 7914.68 |
| ETHUSDT 1d | structure | 158.5 | 54.25% | 0.986 | 30.85% | 5498.36 |
| ETHUSDT 1d | tb_vol_h5_cal | 1.0 | 0.00% | 0.000 | 0.18% | 31.35 |
| EURUSD 1d | fx_compact | 44.0 | 54.55% | 0.898 | 6.94% | 90.42 |
| EURUSD 1d | fx_price | 25.0 | 44.00% | 0.594 | 3.69% | 51.19 |
| EURUSD 4h | fx_compact | 5.0 | 20.00% | 0.488 | 0.52% | 5.15 |
| EURUSD 4h | fx_price | 0.0 | N/A | N/A | 0.00% | 0.00 |

Los detalles de A/B, drawdown y costos por semilla y escenario se conservan en economic_comparison.csv y en los CSV de curvas y operaciones. En meta-labeling se comparan sólo folds con calibración disponible en ambos brazos; no se atribuye ese retorno a toda la historia.

### LSTM/DMN y controles económicos

La LSTM predice asignaciones de riesgo; su ROC-AUC siguiente es una medida del ranking de la asignación frente al signo del retorno open/open, no de una probabilidad de dirección close/close. Accuracy usa asignación ≥0,5 como regla descriptiva fija. No se calculó log-loss/Brier artificial a partir de pesos de cartera.

| Activo | AUC score A | AUC score B | Δ AUC | Accuracy A | Accuracy B | Log-loss |
|---|---|---|---|---|---|---|
| BTCUSDT | 0.51989 | 0.51961 | -0.00028 | 49.50% | 50.22% | N/A |
| ETHUSDT | 0.52035 | 0.52825 | 0.00790 | 48.68% | 50.44% | N/A |

LSTM BTC/ETH, media de semillas: retorno 23.62% → 21.36% (Δ -2.27 pp); Sharpe 1.337 → 1.433. IC95 de Δ retorno, bloques 30: [-12.31; 5.28] pp. Retorno B con costos dobles: 20.23%.

| Control DMN, mismo panel | Retorno neto base encadenado |
|---|---|
| cash | 0.00% |
| long_scaled | 63.51% |
| tsmom | 38.08% |
| macd | 18.10% |

Los controles cash, long escalado por volatilidad, TSMOM y MACD se conservaron con curvas por fold en neural_baselines. La evaluación usa la adaptación spot existente; no es una réplica sobre futuros diversificados.

## F. Significancia y robustez

Bootstrap pareado de bloques móviles: 2.000 réplicas, bloques 7/30 observaciones diarias; H4 usa 42/180 barras para aproximar 7/30 días operados. No se cruzan folds y las semillas comparten el mismo remuestreo de fechas. Se informa el promedio de métricas por semilla y los intervalos son condicionales a los modelos ya ajustados. No incluyen incertidumbre de volver a entrenar ni son garantía de desempeño futuro.

Prueba primaria: ganancia de log-loss A − B, alternativa positiva fijada previamente, bootstrap centrado. Se toma el mayor p entre bloques y se aplica Holm a todos los endpoints modelo/mercado/TF/objetivo. Accuracy, AUC y Brier tienen IC puntuales secundarios, no inferencia simultánea. Los controles forman otra familia Holm. Los intervalos económicos son exploratorios puntuales y no autorizan una selección. La multiplicidad histórica de investigaciones previas impide interpretar estos folds ya observados como confirmación independiente.

Una ganancia numérica es sólo un punto favorable. Una ganancia estadísticamente respaldada exige el endpoint primario corregido. “Robusta” exigiría además estabilidad de signo, tamaño material, ambas semillas, varios mercados y confirmación independiente; esta última no está disponible.

| Endpoint | Ganancia log-loss | IC95 bloque largo | p Holm | Folds con ganancia / pares | Semillas evaluadas |
|---|---|---|---|---|---|
| ETHUSDT / 1d / catboost_cross / close_direction | 0.00424 | [-0.00039, 0.00996] | 1.00000 | 6/6 | 2 |
| ETHUSDT / 1d / lightgbm_cross / close_direction | -0.00468 | [-0.01221, 0.00661] | 1.00000 | 4/6 | 2 |
| ETHUSDT / 1d / logistic / close_direction | 0.00148 | [-0.00275, 0.00821] | 1.00000 | 2/3 | 1 |
| ETHUSDT / 1d / original / close_direction | 0.00036 | [-0.00503, 0.00787] | 1.00000 | 3/6 | 2 |
| ETHUSDT / 1d / random_forest_cross / close_direction | 0.00014 | [-0.00406, 0.00642] | 1.00000 | 2/6 | 2 |
| ETHUSDT / 1d / regime_rules / close_direction | 0.00125 | [-0.00358, 0.0084] | 1.00000 | 4/6 | 2 |
| ETHUSDT / 1d / smc / close_direction | 0.00167 | [-0.00382, 0.0102] | 1.00000 | 5/6 | 2 |
| ETHUSDT / 1d / specialists_rules / close_direction | -0.00280 | [-0.00783, 0.00519] | 1.00000 | 1/3 | 1 |
| ETHUSDT / 1d / structure / close_direction | 0.00421 | [-0.00075, 0.01234] | 1.00000 | 5/6 | 2 |
| BTCUSDT / 1d / catboost_cross / close_direction | 0.00279 | [-0.00188, 0.00715] | 1.00000 | 5/6 | 2 |
| BTCUSDT / 1d / lightgbm_cross / close_direction | 0.00736 | [-0.00281, 0.01293] | 1.00000 | 6/6 | 2 |
| BTCUSDT / 1d / logistic / close_direction | -0.00031 | [-0.00257, 0.00203] | 1.00000 | 1/3 | 1 |
| BTCUSDT / 1d / original / close_direction | -0.00034 | [-0.00357, 0.003] | 1.00000 | 3/6 | 2 |
| BTCUSDT / 1d / random_forest_cross / close_direction | 0.00161 | [-0.0034, 0.00429] | 1.00000 | 5/6 | 2 |
| BTCUSDT / 1d / regime_rules / close_direction | -0.00029 | [-0.00419, 0.00324] | 1.00000 | 2/6 | 2 |
| BTCUSDT / 1d / smc / close_direction | -0.00071 | [-0.00494, 0.00396] | 1.00000 | 3/6 | 2 |
| BTCUSDT / 1d / specialists_rules / close_direction | -0.00383 | [-0.01013, 0.00112] | 1.00000 | 1/3 | 1 |
| BTCUSDT / 1d / structure / close_direction | -0.00031 | [-0.0047, 0.00493] | 1.00000 | 3/6 | 2 |
| ETHUSDT / 1d / meta_net_h1_logistic_cal_55 / net_event_profit | 0.01712 | [0.00938, 0.03347] | 1.00000 | 1/1 | 1 |
| ETHUSDT / 1d / tb_vol_h5_cal / net_event_profit | 0.00088 | [-0.00049, 0.00189] | 1.00000 | 1/3 | 1 |
| BTCUSDT / 1d / tb_vol_h5_cal / net_event_profit | 0.00040 | [-0.00193, 0.00312] | 1.00000 | 1/3 | 1 |
| BTCUSDT / 1d / meta_net_h1_logistic_cal_55 / net_event_profit | 0.00082 | [-0.00092, 0.00372] | 1.00000 | 1/2 | 1 |
| EURUSD / 4h / fx_compact / close_direction | -0.00028 | [-0.00165, 0.00033] | 1.00000 | 1/3 | 1 |
| EURUSD / 4h / fx_price / close_direction | 0.00031 | [-0.00027, 0.00082] | 1.00000 | 3/3 | 1 |
| EURUSD / 1d / fx_compact / close_direction | 0.00064 | [-0.00174, 0.00333] | 1.00000 | 1/3 | 1 |
| EURUSD / 1d / fx_price / close_direction | -0.00029 | [-0.00306, 0.00195] | 1.00000 | 1/3 | 1 |

![Intervalos log-loss](artifacts/retraining_comparison/20261017_controlled/logloss_intervals.png)

**Calibración no disponible:** cuatro políticas meta no cumplieron el mínimo. En BTC, A y B carecen del tercer fold; en ETH, B carece de los dos primeros. Se guardan sus auditorías y se excluyen esos pares del contraste predictivo; no se filtraron por resultado OOS. BTC-meta tiene 2023–2024; ETH-meta sólo primer semestre de 2025. No se reclama robustez temporal para esas comparaciones.

El entrenamiento bruto se llegó a realizar en esos cuatro casos, pero no quedó una política calibrada válida; se incluyen en el conteo de 441 ajustes. Los modelos OOF y las políticas completas sí tienen checkpoints conservados. Los cuatro estimadores brutos descartados no se presentan como modelos completos disponibles.

## G. Diagnóstico

### Respuestas a las ocho preguntas

1. **Mejoras numéricas:** CatBoost BTC/ETH, Random Forest BTC, LightGBM BTC y logística/estructura/SMC ETH presentan aumentos de AUC con menor log-loss. Ninguna mejora del endpoint primario queda estadísticamente respaldada tras Holm. En LightGBM BTC, Accuracy baja pese a mejorar AUC y log-loss.

2. **Deterioros:** LightGBM ETH pierde 0,02689 de AUC y 1,37 pp de Accuracy; log-loss aumenta 0,00468. Los especialistas por reglas empeoran log-loss en ambos activos. Logística BTC pierde 1,94 pp de Accuracy pese a mejorar su backtest.

3. **Cambios pequeños:** SMC BTC cambia AUC +0,00061 y estructura BTC +0,00091; ambos empeoran ligeramente log-loss. No se probó equivalencia estadística: sólo son diferencias puntuales pequeñas.

4. **Mercados prometedores, con cautela:** BTC alcanza el mayor AUC direccional nuevo, 0,55319 en Random Forest; ETH alcanza 0,53518 con logística. En EUR/USD, los mejores AUC rondan 0,515. No hay superioridad corregida frente a los baselines fuertes en estos endpoints.

5. **Temporalidades:** no hay evidencia que establezca una temporalidad ganadora. H4 EUR/USD no ofrece una mejora consistente; más barras no equivalen a más días independientes. Las cifras cripto D1 y Forex H4 representan horizontes/mercados diferentes.

6. **Origen del cambio:** la comparación mantiene las recetas y observaciones compartidas; evalúa la ampliación histórica específica y la reparación nativa predeclarada. No informa sobre entrenamiento con 2025–2026 en cuarentena ni sobre transferencia a los activos no evaluados.

7. **Superioridad a baselines:** no quedó ningún contraste de log-loss B frente a controles con p Holm <0,05. Esto no prueba ausencia absoluta de señal, pero impide afirmar una ventaja validada.

8. **Utilidad económica:** hay backtests positivos, pero no una mejora robusta. Logística BTC pasa de +45,52% a +109,40%, aunque el IC95 de su diferencia incluye cero y su dirección empeora. El Random Forest BTC nuevo, mejor por AUC, pierde 7,08% neto; LSTM baja de +23,62% a +21,36%. Los costos dobles y las diferencias entre semillas impiden una recomendación operativa.

La ampliación no es un multiplicador uniforme de señal. Aporta regímenes anteriores y modifica frecuencias de entrenamiento, particiones de árboles y coeficientes/calibración. Eso puede reducir varianza o introducir una distribución menos pertinente para la ventana posterior. Las diferencias por fold y semilla permiten observar esa inestabilidad; no identifican por sí solas un mecanismo causal de mercado.

En esta comparación tabular se mantuvieron fijas arquitectura, hiperparámetros, columnas, umbrales, costos y todas las filas comunes. La diferencia es la historia agregada con su procedencia nativa declarada. No se atribuye el resultado a “millones de datos nuevos”, porque la mayor parte del inventario corresponde a otros mercados, otras temporalidades dependientes o fechas en cuarentena. Tampoco se confunde mejor calibración/log-loss con mejor ranking direccional o rentabilidad.

La mezcla global/experto, los targets netos de Triple Barrier y el objetivo Sharpe de DMN tienen finalidades diferentes. Sus mejores cifras no son intercambiables. La escasez de ofertas OOF de meta-labeling evidencia que más registros no garantiza una calibración operativa suficiente.

## H. Recomendaciones y decisión

No activar ni promover un modelo usando este informe. Mantener los controles simples y congelar cualquier hipótesis que se quiera confirmar sobre un período futuro disjunto de las reservas existentes. Sólo merecen confirmación adicional las mejoras que superen controles fuertes y mantengan dirección, magnitud y costos en varias ventanas; no ampliar una búsqueda por una mejora aislada.

Los resultados negativos deben permanecer disponibles. No repetir cientos de ajustes variando semillas, umbrales o features sobre estos mismos folds. Perpetuos, contratos e índices requieren primero sus contratos de ejecución/disponibilidad; añadirlos ahora impediría aislar el efecto de los datos.

### Reproducción, recursos y preservación

Equipo: 8 procesadores lógicos, 7.778.787.328 bytes de RAM (~7,24 GiB), ejecución CPU. El piloto XGBoost tardó 0,47 s A y 0,41 s B y confirmó 353 observaciones pareadas antes de ampliar. Árboles con 1 worker y LSTM con 2 threads; no se hizo una búsqueda masiva. La ampliación se decidió por integridad y costo del piloto, no por su score.

**Preservación verificada:** 32,658 archivos anteriores comprobados por SHA-256; 0 modificados. Versiones y reservas no se promovieron ni sobrescribieron. Todos los datos/predicciones/curvas/configuraciones nuevos están en artifacts/retraining_comparison/20261017_controlled. Artifacts está excluido de Git y requiere resguardo separado.

**Software:** 55 pruebas dirigidas aprobadas; luego 4 pruebas de inferencia/contabilidad aprobadas tras completar el análisis. Ruff de los scripts nuevos aprobado. Los tests cubren purga/reservas, modelos existentes, ausencia de pseudorreplicación por semillas, Holm e inclusión del costo inicial al agregar curvas. No se confunden con una prueba de rentabilidad futura.

En Windows PowerShell, los avisos PerformanceWarning del runner de eventos se envolvieron como NativeCommandError y el shell informó código 1. No hay traceback Python en ese log: se verificó la finalización de las 24 políticas planeadas (20 completas y 4 abstenciones auditadas), 72 checkpoints OOF y sus artefactos. No se ocultó ese estado del shell.

Comandos (las carpetas son append-only; para una repetición independiente usar un nuevo identificador OUT en los scripts y conservar configuración/fuentes):

```powershell
.\.venv\Scripts\python.exe scripts/retraining_comparison.py audit
.\.venv\Scripts\python.exe scripts/retraining_comparison.py pilot
.\.venv\Scripts\python.exe scripts/retraining_comparison.py full
.\.venv\Scripts\python.exe scripts/retraining_comparison.py events
.\.venv\Scripts\python.exe scripts/retraining_comparison.py neural
.\.venv\Scripts\python.exe scripts/retraining_comparison.py forex
.\.venv\Scripts\python.exe scripts/retraining_economic_inference.py
.\.venv\Scripts\python.exe scripts/analyze_retraining_comparison.py
.\.venv\Scripts\python.exe scripts/report_retraining_comparison.py
```

La modificación de alcance previa a cualquier ajuste está en protocol_amendment_before_fitting.json: EUR/USD representativo y expertos sólo semilla 42, límite 500. El código y entorno finales se conservan en el manifiesto de cierre; plan.json conserva el protocolo inicial y no fue sobrescrito. Las correcciones de contabilidad e inferencia posteriores no cambiaron modelos ni hiperparámetros ni escogieron resultados.

Archivos principales: comparison.csv, fold_metrics.csv, baselines.csv, paired_inference.json, baseline_skill_inference.json, economic_comparison.csv, economic_inference.json, all_predictive_metrics.json, training_ledger.json, stored_model_suffix_evaluation.json y verification.json. Los hashes, ventanas, parámetros, semillas y predicciones por ajuste están junto a cada modelo.

### Respuesta final verificable

La respuesta es **no se demostró una mejora significativa**: 0/26 endpoints primarios y 0 contrastes corregidos contra controles resultaron favorables. No debe interpretarse como una demostración de equivalencia o de inexistencia absoluta de señal.

Los mejores descriptivos en BTC diario, evaluados sobre las mismas 876 observaciones, son A: XGBoost con SMC (AUC 0,55249) y B: Random Forest cruzado (AUC 0,55319). La diferencia entre ganadores es +0,00071 de AUC; cambiar de receta en esa resta no aísla el efecto de añadir datos.

Comparación del **mismo mejor modelo anterior de BTC (smc)**, sin cambiar de receta al calcular diferencias:

| Métrica | Antes A | Después B | B − A |
|---|---|---|---|
| roc_auc | 0.55249 | 0.55310 | 0.00061 |
| accuracy | 0.54224 | 0.53653 | -0.00571 |
| log_loss | 0.68950 | 0.69021 | 0.00071 |

Mejor modelo nuevo descriptivo BTC: random_forest_cross. El listado de mejoras respaldadas del endpoint log-loss es: **ninguno**.

La conclusión sobre dirección debe considerar además AUC/Accuracy y controles, no sólo log-loss. La conclusión económica debe considerar retorno neto, incertidumbre, estrés de costos y estabilidad. No hay confirmación prospectiva de una mejora robusta.

Comparación del **mismo Random Forest BTC**: AUC 0,54202 → 0,55319 (Δ +0,01117), Accuracy 53,14% → 54,39% (Δ +1,26 pp), log-loss 0,68949 → 0,68788 (Δ −0,00161); retorno neto −10,82% → −7,08% (Δ +3,74 pp). IC95 de Δ AUC, bloque 30: [−0,01927; +0,02609]; IC95 de ganancia log-loss [−0,00340; +0,00429]; p Holm = 1. La mejora es numérica, no respaldada ni robusta ni rentable en este backtest.

---

## I. Estudios de expansión de mercados y señales (recuperación y verificación sin reentrenar)

Además del estudio controlado de extensión histórica (441 ajustes, secciones A–H), se recuperaron, verificaron por SHA-256/contabilidad y analizaron sin reentrenar los dos estudios predeclarados de expansión de universo/señales que habían quedado terminados en disco:

1. **Expansión multimercado/derivados/flujo (`20261016T211432_bcec9ebe`)**: **72 ajustes completados** (6 escenarios × 2 arquitecturas [`xgboost`, `lstm`] × 2 semillas [`42`, `1729`] × 3 folds walk-forward: `2023-07-01..2024-01-01`, `2024-01-01..2025-01-01`, `2025-01-01..2025-07-01`; validación interior 180 días, embargo 1 día, secuencia 63). Verificación: **768 archivos intactos**, **234 carteras reconstruidas** (468 escenarios de costo) y reproducción exacta de probabilidades XGBoost desde `model.ubj` (error máximo `2,98e-8`). Artefactos de inferencia en `artifacts/expansion_analysis/20261016T211432_bcec9ebe/20261017T221013/`.
2. **Expansión Forex en nuevos pares (`20261016T210703_c9107a5c`)**: **24 ajustes completados** (`USDCAD` y `NZDUSD` D1 NY17 con ejecución H1 bid/ask FXCM, recetas `price` vs `compact`, 2 semillas × 3 folds `2023-07-01..2025-07-01`, embargo 7 días, calibración 180 días). Verificación: **123 archivos intactos**, **24 modelos y 24 carteras reproducidos**. Artefactos en `artifacts/expansion_analysis/20261016T210703_c9107a5c/20261017T221546/`.

### I.1. Ablación de transferencia cross-asset y nuevas fuentes sobre BTC/ETH (72 ajustes, 728 fechas × 2 activos = 1.456 observaciones OOS)

Todos los escenarios se evalúan exactamente sobre las mismas 1.456 observaciones de `BTCUSDT` y `ETHUSDT`:

- `anchor_price`: entrenado sólo con BTC/ETH y 8 features de precio/volatilidad DMN.
- `expanded_price`: entrenado en panel con 5 criptos (`BTC`, `ETH`, `SOL`, `BNB`, `XRP`).
- `expanded_flow`: añade flujo agresor spot (`taker_buy_ratio`, anomalías de volumen/trades).
- `expanded_derivatives`: añade `funding_rate` liquidado y prima perpetuo/spot negociada.
- `expanded_context`: añade retornos/dispersión cross-crypto y contexto de 7 pares Forex (`EURUSD`, `GBPUSD`, `USDJPY`, `USDCHF`, `AUDUSD`, `USDCAD`, `NZDUSD`).
- `expanded_noise`: control negativo con el mismo número de columnas que `expanded_context`, reemplazando las features nuevas por ruido gaussiano causal independiente.

#### Métricas predictivas direccionales — XGBoost (media de semillas 42 y 1729 sobre BTC+ETH, 1.456 filas)

| Escenario | Accuracy | Balanced Acc. | ROC-AUC | Log-loss | Brier | Precision | Recall | F1 | ECE | Δ Log-loss (A−B) | IC95 ganancia (bloque 30) | p Holm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `anchor_price` (base) | 0.49245 | 0.50183 | 0.50754 | 0.69468 | 0.25076 | 0.53137 | 0.04643 | 0.08528 | 0.02705 | 0.00000 | Referencia | — |
| `expanded_price` (5 cripto) | 0.51202 | 0.51809 | 0.52508 | 0.69732 | 0.25200 | 0.55468 | 0.22342 | 0.31501 | 0.03673 | -0.00264 | [-0.00675, +0.00117] | 1.00000 |
| `expanded_flow` (+flujo spot) | 0.51545 | 0.52034 | 0.51793 | 0.69553 | 0.25116 | 0.54904 | 0.28331 | 0.37375 | 0.03132 | -0.00085 | [-0.00337, +0.00158] | 1.00000 |
| `expanded_derivatives` (+funding/perp) | 0.52163 | 0.52738 | 0.52370 | 0.69477 | 0.25080 | 0.57224 | 0.24832 | 0.34615 | 0.02982 | -0.00009 | [-0.00227, +0.00199] | 1.00000 |
| `expanded_context` (+cross/7 FX) | 0.50240 | 0.50950 | 0.51358 | 0.69401 | 0.25043 | 0.53882 | 0.16487 | 0.25132 | 0.02663 | +0.00066 | [-0.00161, +0.00293] | 1.00000 |
| `expanded_noise` (control ruido) | 0.50412 | 0.51078 | 0.52754 | 0.69356 | 0.25020 | 0.54071 | 0.18775 | 0.27857 | 0.03009 | +0.00112 | [-0.00062, +0.00285] | 0.68966 |
| Control `constant_05` | 0.51030 | 0.50000 | 0.50000 | 0.69315 | 0.25000 | 0.51030 | 1.00000 | 0.67576 | 0.01030 | — | — | — |
| Control `train_prevalence` | 0.48970 | 0.50000 | 0.50000 | 0.69422 | 0.25054 | 0.00000 | 0.00000 | 0.00000 | 0.02697 | — | — | — |

**Hallazgo crítico de control experimental:** aunque añadir 5 criptos, flujo y derivados (perpetuos + funding) eleva el Accuracy de XGBoost de `49,24%` a `52,16%` (+2,92 pp) y el ROC-AUC de `0,50754` a `0,52370` (+0,01616), **el control con ruido gaussiano (`expanded_noise`) alcanza el mayor ROC-AUC (`0,52754`) y el menor log-loss (`0,69356`) de las seis variantes**, porque las columnas extra actúan como regularización implícita por submuestreo de columnas (`colsample_bytree = 0,85`) y frenado temprano en validación. Ninguna de las seis variantes mejora el log-loss del baseline trivial `constant_05` (`0,69315`), y ningún contraste supera Holm (`p >= 0,6897`).

#### Desglose por activo (XGBoost, 728 fechas por activo)

| Activo | Escenario | Accuracy | Balanced Acc. | ROC-AUC | Log-loss | Brier | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | `anchor_price` | 0.49107 | 0.50632 | 0.51535 | 0.69460 | 0.25072 | 0.59987 | 0.04388 | 0.08178 |
| BTCUSDT | `expanded_price` | 0.51030 | 0.52054 | 0.52985 | 0.69851 | 0.25257 | 0.56939 | 0.21011 | 0.30401 |
| BTCUSDT | `expanded_flow` | 0.51442 | 0.52267 | 0.51741 | 0.69629 | 0.25154 | 0.56163 | 0.27261 | 0.36705 |
| BTCUSDT | `expanded_derivatives` | 0.52129 | 0.53113 | 0.52825 | 0.69519 | 0.25100 | 0.59363 | 0.23271 | 0.33422 |
| BTCUSDT | `expanded_context` | 0.49931 | 0.51099 | 0.51946 | 0.69411 | 0.25048 | 0.55008 | 0.15691 | 0.24278 |
| BTCUSDT | `expanded_noise` | 0.50275 | 0.51399 | 0.52557 | 0.69427 | 0.25056 | 0.56042 | 0.17287 | 0.26418 |
| ETHUSDT | `anchor_price` | 0.49382 | 0.49751 | 0.50039 | 0.69476 | 0.25080 | 0.48051 | 0.04905 | 0.08862 |
| ETHUSDT | `expanded_price` | 0.51374 | 0.51604 | 0.52137 | 0.69612 | 0.25142 | 0.54209 | 0.23706 | 0.32573 |
| ETHUSDT | `expanded_flow` | 0.51648 | 0.51833 | 0.51866 | 0.69476 | 0.25078 | 0.53777 | 0.29428 | 0.38034 |
| ETHUSDT | `expanded_derivatives` | 0.52198 | 0.52412 | 0.52053 | 0.69435 | 0.25059 | 0.55414 | 0.26431 | 0.35765 |
| ETHUSDT | `expanded_context` | 0.50549 | 0.50826 | 0.50773 | 0.69392 | 0.25039 | 0.52856 | 0.17302 | 0.25973 |
| ETHUSDT | `expanded_noise` | 0.50549 | 0.50801 | 0.52904 | 0.69284 | 0.24985 | 0.52452 | 0.20300 | 0.29242 |

#### Desempeño económico y ranking direccional — LSTM y XGBoost sobre cartera ancla BTC/ETH (730 fechas)

| Escenario | Arquitectura | AUC score direccional | Accuracy (asignación ≥0,5) | Retorno neto (ancla) | Sharpe neto | Max Drawdown | Turnover total | Retorno costo doble | IC95 Δ Sharpe vs `anchor_price` (bloque 30) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `anchor_price` | LSTM | 0.50021 | 0.49657 | +13.70% | 1.076 | -6.64% | 2.95 | +13.17% | Referencia |
| `expanded_price` | LSTM | 0.50255 | 0.49176 | +14.23% | 1.057 | -7.43% | 3.33 | +13.62% | [-0.309, +0.280] |
| `expanded_flow` | LSTM | 0.49798 | 0.50859 | +15.60% | 1.035 | -7.06% | 4.39 | +14.79% | [-0.394, +0.323] |
| `expanded_derivatives` | LSTM | 0.50388 | 0.49382 | +12.06% | 0.976 | -6.21% | 4.44 | +11.27% | [-0.540, +0.354] |
| `expanded_context` | LSTM | 0.49332 | 0.48455 | +10.66% | 0.887 | -6.23% | 4.28 | +9.91% | [-0.443, +0.036] |
| `expanded_noise` | LSTM | 0.50578 | 0.49313 | +13.00% | 1.056 | -6.48% | 8.17 | +11.53% | [-0.218, +0.189] |
| `anchor_price` | XGBoost | 0.50754 | 0.49245 | -0.69% | -0.067 | -6.68% | 19.80 | -3.79% | Referencia |
| `expanded_price` | XGBoost | 0.52508 | 0.51202 | +0.28% | 0.056 | -10.12% | 60.65 | -8.99% | [-1.077, +1.289] |
| `expanded_flow` | XGBoost | 0.51793 | 0.51545 | -0.16% | 0.031 | -10.49% | 72.87 | -11.15% | [-1.144, +1.360] |
| `expanded_derivatives` | XGBoost | 0.52370 | 0.52163 | +3.49% | 0.262 | -10.49% | 64.82 | -6.70% | [-0.887, +1.503] |
| `expanded_context` | XGBoost | 0.51358 | 0.50240 | -6.52% | -0.624 | -10.55% | 41.49 | -12.52% | [-2.228, +1.218] |
| `expanded_noise` | XGBoost | 0.52754 | 0.50412 | -3.50% | -0.267 | -10.47% | 53.06 | -11.35% | [-1.670, +1.184] |
| `long_scaled` | Baseline | — | — | +38.05% | 1.149 | -13.25% | 7.05 | +36.50% | — |
| `tsmom` | Baseline | — | — | +31.46% | 1.086 | -10.41% | 8.89 | +29.60% | — |

Ningún variante de LSTM o XGBoost supera el Sharpe ni el retorno neto de `long_scaled` (`+38,05%`, Sharpe `1,149`) o `tsmom` (`+31,46%`, Sharpe `1,086`) sobre las mismas fechas; todos los intervalos de confianza al 95% de diferencia de Sharpe frente a `anchor_price` incluyen cero.

### I.2. Nuevos pares Forex (`USDCAD` y `NZDUSD` D1, 24 ajustes, 2023-07-01..2025-07-01)

| Par | Receta | Filas OOS | Accuracy | Balanced Acc. | ROC-AUC | Log-loss | Brier | Precision | Recall | F1 | Log-loss prevalencia | Retorno neto | Retorno estrés | Operaciones | Profit Factor medio | Max DD | Folds con mejora vs prevalencia |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| USDCAD | `price` (base) | 459 | 0.51307 | 0.51454 | 0.49787 | 0.71490 | 0.26028 | 0.52874 | 0.45319 | 0.48800 | 0.69329 | +2.98% | +1.85% | 88.5 | 1.240 | -2.06% | 0/3 |
| USDCAD | `compact` (ampliada) | 459 | 0.48148 | 0.48348 | 0.46852 | 0.72740 | 0.26614 | 0.49225 | 0.40000 | 0.44111 | 0.69329 | +1.66% | +0.78% | 69.5 | 1.144 | -2.26% | 0/3 |
| NZDUSD | `price` (base) | 458 | 0.50655 | 0.50896 | 0.50641 | 0.70407 | 0.25526 | 0.49832 | 0.64667 | 0.56288 | 0.69540 | -10.32% | -12.03% | 137.5 | 0.765 | -9.25% | 0/3 |
| NZDUSD | `compact` (ampliada) | 458 | 0.50655 | 0.51041 | 0.49809 | 0.71017 | 0.25822 | 0.49848 | 0.73111 | 0.59276 | 0.69540 | -14.92% | -17.19% | 188.0 | 0.699 | -13.22% | 0/3 |

En ambos pares nuevos, el modelo empeora el log-loss respecto de la prevalencia de entrenamiento (`0,704–0,727` frente a `0,693–0,695`, en **0/3 folds** para las dos semillas, `p_Holm = 1,0`), y la receta ampliada `compact` empeora tanto el ROC-AUC como el log-loss frente a `price` (`Δ log-loss = +0,01250` en `USDCAD`, `+0,00610` en `NZDUSD`).

### I.3. Tabla resumen consolidada exigida (antes vs. después sobre observaciones comunes)

| Modelo | Mercado | Accuracy antes | Accuracy después | AUC antes | AUC después | Log-loss antes | Log-loss después | Conclusión |
|---|---|---:|---:|---:|---:|---:|---:|---|
| XGBoost (`original`) | BTC/USDT D1 | 0.54167 | 0.53824 | 0.54586 | 0.54814 | 0.69113 | 0.69148 | Mixto (AUC +0,00228; Acc −0,34 pp; log-loss +0,00034; p Holm = 1,0) |
| Random Forest (`random_forest_cross`) | BTC/USDT D1 | 0.53139 | 0.54395 | 0.54202 | 0.55319 | 0.68949 | 0.68788 | Mejoró numéricamente (Acc +1,26 pp; AUC +0,01117; p Holm = 1,0; retorno −7,08%) |
| Regresión Logística (`logistic`) | BTC/USDT D1 | 0.53995 | 0.52055 | 0.54396 | 0.54001 | 0.68925 | 0.68955 | Empeoró en predicción (Acc −1,94 pp; AUC −0,00395), aunque mejoró retorno (+109,40%) |
| LightGBM (`lightgbm_cross`) | BTC/USDT D1 | 0.52626 | 0.52397 | 0.52032 | 0.53642 | 0.70129 | 0.69393 | Mixto (AUC +0,01611 y log-loss −0,00736; Acc −0,23 pp; p Holm = 1,0) |
| CatBoost (`catboost_cross`) | BTC/USDT D1 | 0.52169 | 0.53196 | 0.53132 | 0.54706 | 0.69195 | 0.68916 | Mejoró numéricamente (Acc +1,03 pp; AUC +0,01574; p Holm = 1,0; retorno −22,70%) |
| XGBoost + SMC (`smc`) | BTC/USDT D1 | 0.54224 | 0.53653 | 0.55249 | 0.55310 | 0.68950 | 0.69021 | Sin cambio relevante (AUC +0,00061; Acc −0,57 pp; log-loss +0,00071) |
| XGBoost + Estructura (`structure`) | BTC/USDT D1 | 0.53368 | 0.53938 | 0.55223 | 0.55313 | 0.68960 | 0.68991 | Mejora marginal en Acc (+0,57 pp) y AUC (+0,00091); log-loss +0,00031 |
| XGBoost + Régimen (`regime_rules`) | BTC/USDT D1 | 0.52397 | 0.53710 | 0.53937 | 0.54338 | 0.69200 | 0.69230 | Mejoró Acc (+1,31 pp) y AUC (+0,00401); log-loss +0,00029; p Holm = 1,0 |
| Híbrido Especialistas (`specialists_rules`) | BTC/USDT D1 | 0.52511 | 0.52397 | 0.53703 | 0.52910 | 0.69462 | 0.69845 | Empeoró en todas las métricas predictivas |
| Triple Barrier (`tb_vol_h5_cal`, evento neto) | BTC/USDT D1 | 0.50913 | 0.51027 | 0.48456 | 0.49248 | 0.70093 | 0.70052 | Sigue bajo azar (AUC < 0,50; p Holm = 1,0) |
| Meta-labeling (`meta_net_h1_logistic_cal_55`) | BTC/USDT D1 | 0.58557 | 0.57426 | 0.53420 | 0.53269 | 0.68265 | 0.68183 | Empeoró Acc (−1,13 pp) y AUC (−0,00151); 0 operaciones ejecutadas |
| XGBoost (`original`) | ETH/USDT D1 | 0.51313 | 0.51427 | 0.51628 | 0.50759 | 0.70053 | 0.70017 | Empeoró AUC (−0,00869); log-loss peor que 0,5 (`0,70017`) |
| Random Forest (`random_forest_cross`) | ETH/USDT D1 | 0.50799 | 0.50628 | 0.52185 | 0.51580 | 0.69372 | 0.69358 | Empeoró Acc (−0,17 pp) y AUC (−0,00605); p Holm = 1,0 |
| Regresión Logística (`logistic`) | ETH/USDT D1 | 0.51142 | 0.53311 | 0.51951 | 0.53518 | 0.69304 | 0.69156 | Mejoró numéricamente (Acc +2,17 pp; AUC +0,01567; log-loss −0,00148; p Holm = 1,0) |
| LightGBM (`lightgbm_cross`) | ETH/USDT D1 | 0.50685 | 0.49315 | 0.52630 | 0.49941 | 0.69795 | 0.70263 | Empeoró claramente (Acc −1,37 pp; AUC −0,02689; log-loss +0,00468) |
| CatBoost (`catboost_cross`) | ETH/USDT D1 | 0.50228 | 0.52397 | 0.50756 | 0.52875 | 0.69665 | 0.69240 | Mejoró numéricamente (Acc +2,17 pp; AUC +0,02119; 6/6 fold-semillas; p Holm = 1,0) |
| XGBoost + SMC (`smc`) | ETH/USDT D1 | 0.49201 | 0.51712 | 0.50633 | 0.51849 | 0.69977 | 0.69811 | Mejoró numéricamente (Acc +2,51 pp; AUC +0,01216), pero log-loss > 0,693 |
| XGBoost + Estructura (`structure`) | ETH/USDT D1 | 0.50114 | 0.52169 | 0.50744 | 0.52976 | 0.69969 | 0.69548 | Mejoró numéricamente (Acc +2,05 pp; AUC +0,02231), pero log-loss > 0,693 |
| XGBoost + Régimen (`regime_rules`) | ETH/USDT D1 | 0.51427 | 0.50457 | 0.50717 | 0.50401 | 0.70143 | 0.70019 | Empeoró Acc (−0,97 pp) y AUC (−0,00316) |
| Híbrido Especialistas (`specialists_rules`) | ETH/USDT D1 | 0.50114 | 0.49772 | 0.50500 | 0.50430 | 0.70314 | 0.70594 | Empeoró en todas las métricas |
| Triple Barrier (`tb_vol_h5_cal`, evento neto) | ETH/USDT D1 | 0.54338 | 0.53767 | 0.51487 | 0.52447 | 0.68499 | 0.68411 | Mixto (AUC +0,00960; Acc −0,57 pp; sólo 1 operación) |
| Meta-labeling (`meta_net_h1_logistic_cal_55`, 1 fold) | ETH/USDT D1 | 0.44970 | 0.55030 | 0.50000 | 0.57895 | 0.70290 | 0.68579 | Artefacto de muestra mínima (169 días, 1 fold, recall 0%, 0 operaciones; p Holm = 1,0) |
| LSTM / DMN (`lstm_sharpe_net`, historia 2017 vs 2020) | BTC/USDT D1 | 0.49500 | 0.50220 | 0.51989 | 0.51961 | N/A | N/A | Sin cambio (AUC −0,00028; Acc +0,72 pp; retorno cartera 23,62% → 21,36%) |
| LSTM / DMN (`lstm_sharpe_net`, historia 2017 vs 2020) | ETH/USDT D1 | 0.48680 | 0.50440 | 0.52035 | 0.52825 | N/A | N/A | Ligera mejora descriptiva de ranking (AUC +0,00790; Acc +1,76 pp) |
| XGBoost (`fx_compact`) | EUR/USD D1 | 0.50934 | 0.51783 | 0.50993 | 0.51343 | 0.69388 | 0.69325 | Mejora numérica leve (Acc +0,85 pp; AUC +0,00350; log-loss > 0,69315; p Holm = 1,0) |
| XGBoost (`fx_price`) | EUR/USD D1 | 0.49745 | 0.51104 | 0.50324 | 0.50205 | 0.69470 | 0.69499 | Mixto (Acc +1,36 pp; AUC −0,00119; log-loss +0,00029) |
| XGBoost (`fx_compact`) | EUR/USD H4 | 0.51036 | 0.50891 | 0.51566 | 0.51506 | 0.69279 | 0.69307 | Empeoró levemente (Acc −0,15 pp; AUC −0,00060; log-loss +0,00028) |
| XGBoost (`fx_price`) | EUR/USD H4 | 0.51533 | 0.50365 | 0.50949 | 0.51394 | 0.69323 | 0.69292 | Mixto (Acc −1,17 pp; AUC +0,00445; 0 operaciones) |
| XGBoost panel (`anchor_price` → `expanded_derivatives`) | BTC+ETH D1 (c/ perp+funding) | 0.49245 | 0.52163 | 0.50754 | 0.52370 | 0.69468 | 0.69477 | Mejoró Acc (+2,92 pp) y AUC (+0,01616), pero `expanded_noise` obtuvo AUC `0,52754` y mejor log-loss (`0,69356`) |
| LSTM panel (`anchor_price` → `expanded_flow`) | BTC+ETH D1 (5 cripto + flujo) | 0.49657 | 0.50859 | 0.50021 | 0.49798 | N/A | N/A | Retorno neto +13,70% → +15,60%, pero Sharpe baja (1,076 → 1,035) y sigue bajo `long_scaled` (+38,05%) |
| XGBoost (`price` → `compact`) | USD/CAD D1 | 0.51307 | 0.48148 | 0.49787 | 0.46852 | 0.71490 | 0.72740 | Empeoró (Acc −3,16 pp; AUC −0,02935; peor que prevalencia en 3/3 folds) |
| XGBoost (`price` → `compact`) | NZD/USD D1 | 0.50655 | 0.50655 | 0.50641 | 0.49809 | 0.70407 | 0.71017 | Empeoró (AUC −0,00832; log-loss +0,00610; peor que prevalencia en 3/3 folds) |

El informe definitivo consolidado con las respuestas directas a las preguntas de investigación se encuentra en [FINAL_MODEL_IMPROVEMENT_REPORT.md](FINAL_MODEL_IMPROVEMENT_REPORT.md).

