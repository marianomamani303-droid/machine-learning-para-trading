# Resultados: investigación de ventaja predictiva y económica

Fecha local: 17 de octubre de 2026. Esta ronda es exploratoria sobre historial reutilizado; no constituye confirmación prospectiva.

Corrida: `20261018T015816_ca7033e2`. **120 ajustes: 96 estimadores y 24 calibradores. 0/16 candidatos cumplen todos los criterios; 5/32 contrastes predictivos y 0/16 económicos cruzan Holm 5%.** No se promovieron modelos ni se ejecutaron operaciones reales.

5/16 contrastes frente a p=0,5 superan Holm, y **0/16 contrastes de aporte incremental de las features frente a precio o ruido superan Holm.** Superar una probabilidad constante sólo demuestra una mejora retrospectiva dentro de este protocolo; no acredita por sí solo información nueva ni ventaja económica. La tabla de inferencia identifica cada caso, sin elegir un ganador a posteriori.

Comparación principal predeclarada: logística con señales adicionales frente a logística de precio. Una política que retorna 0% sin operaciones conserva efectivo y no demuestra ventaja.

| task | AUC | loss_precio | loss_nuevas | bruto_fijo_% | neto_fijo_% | neto_filtro_riesgo_% | trades_filtro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT_1h_h1 | 0.54699 | 0.69030 | 0.69037 | 142.58680 | -99.99653 | 0.00000 | 0 |
| BTCUSDT_1h_h4 | 0.54124 | 0.69038 | 0.69047 | 86.74454 | -96.96885 | 0.00000 | 0 |
| BTCUSDT_4h_h1 | 0.55811 | 0.68773 | 0.68911 | 115.46236 | -93.70229 | 0.00000 | 0 |
| BTCUSDT_1d_h1 | 0.51528 | 0.69937 | 0.70945 | 4.73779 | -26.70738 | -19.25905 | 27 |
| ETHUSDT_1h_h1 | 0.55048 | 0.68925 | 0.68932 | 70.86551 | -99.89575 | 0.00000 | 0 |
| ETHUSDT_1h_h4 | 0.54425 | 0.69067 | 0.69051 | 71.64179 | -93.23333 | 0.00000 | 0 |
| ETHUSDT_4h_h1 | 0.54616 | 0.68997 | 0.69113 | 18.20714 | -90.80152 | 0.00000 | 0 |
| ETHUSDT_1d_h1 | 0.53069 | 0.69212 | 0.69959 | 34.60904 | -11.39622 | -3.24201 | 13 |

## Auditoría y relación con estudios anteriores

La evidencia anterior cubre miles de ajustes, algoritmos y features. Los problemas identificados siguen siendo sobreajuste, señales inestables, confianza excesiva, costos de rotación y repetición de selección sobre las mismas fechas. El informe financiero focalizado contiene indicios positivos (BTC diario logística +45,29% base y +28,24% doble costo; ETH mejora de AUC, Forex mejora relativa), pero la frase anterior de ‘mejora económica robusta’ sólo describe sensibilidad retrospectiva: falta confirmación independiente y control de toda la trayectoria de selección. No se alteró ese informe.

La logística regularizada se usa como referencia interpretable y parsimoniosa; XGBoost pequeño calibrado como contraste no lineal. No se repiten DMN, redes, barreras ni grandes grillas, ya explorados sin evidencia independiente suficiente. Las métricas de entrenamiento, validación y test, probabilidades crudas/calibradas y reliability bins se conservan en `diagnostics.json`.

Los modelos históricos conservan sus contratos y cutoffs. No se usaron sus pesos para puntuar fechas que podrían formar parte de su entrenamiento o selección. La nueva referencia de precio tiene 11 variables causales con ventanas físicas comparables; completa y ruido tienen 17. Es una comparación controlada de este protocolo, no una réplica del contrato original de 14 variables ni una comparación directa de rentabilidades obtenidas en períodos diferentes.

## Diseño efectivamente ejecutado

Dos activos spot; 1h→1h, 1h→4h, 4h→4h y 1d→24h; cuatro recetas: logística precio, logística precio+seis señales, logística precio+seis ruidos, XGBoost completo calibrado. Tres folds: enero–junio 2024, julio–diciembre 2024, enero–junio 2025. Entrenamiento expansivo desde 2022, validación de 90 días disjunta, purga por salida ejecutable y embargo de 48h. No hubo selección de hiperparámetros ni de umbrales en test.

Señales adicionales: ratio de volatilidad de Parkinson, volumen relativo, Amihud estandarizado, volumen firmado aproximado por forma de vela, ruptura de extremos previos y proporción de semivarianza bajista. El régimen causal combina tendencia EMA30d y volumen relativo mínimo. No hay libro de órdenes, profundidad ni flujo taker real en este contrato: son proxies OHLCV.

Todas las recetas comparten filas y costos dentro de cada tarea. H1→4h y H4→4h tienen igual horizonte físico, pero sus señales se emiten a distinta frecuencia; sus retornos agregados son comparación de políticas, no ablation pareada de frecuencias ni muestras independientes. Las ventanas de indicadores representan días físicos iguales. Los huecos invalidan etiquetas atravesadas y reinician warmup; no se rellenan precios.

| task | rows | eligible | gaps | first | last | latest_target |
| --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT_1h_h1 | 30646 | 29202 | 1 | 2022-01-01T00:00:00+00:00 | 2025-06-30T23:00:00+00:00 | 2025-06-30T23:00:00+00:00 |
| BTCUSDT_1h_h4 | 30646 | 29196 | 1 | 2022-01-01T00:00:00+00:00 | 2025-06-30T23:00:00+00:00 | 2025-06-30T23:00:00+00:00 |
| BTCUSDT_4h_h1 | 7661 | 7297 | 1 | 2022-01-01T00:00:00+00:00 | 2025-06-30T20:00:00+00:00 | 2025-06-30T20:00:00+00:00 |
| BTCUSDT_1d_h1 | 1276 | 1212 | 1 | 2022-01-01T00:00:00+00:00 | 2025-06-30T00:00:00+00:00 | 2025-06-30T00:00:00+00:00 |
| ETHUSDT_1h_h1 | 30646 | 29202 | 1 | 2022-01-01T00:00:00+00:00 | 2025-06-30T23:00:00+00:00 | 2025-06-30T23:00:00+00:00 |
| ETHUSDT_1h_h4 | 30646 | 29196 | 1 | 2022-01-01T00:00:00+00:00 | 2025-06-30T23:00:00+00:00 | 2025-06-30T23:00:00+00:00 |
| ETHUSDT_4h_h1 | 7661 | 7297 | 1 | 2022-01-01T00:00:00+00:00 | 2025-06-30T20:00:00+00:00 | 2025-06-30T20:00:00+00:00 |
| ETHUSDT_1d_h1 | 1276 | 1212 | 1 | 2022-01-01T00:00:00+00:00 | 2025-06-30T00:00:00+00:00 | 2025-06-30T00:00:00+00:00 |

## Resultados predictivos sobre las mismas filas por tarea

| task | recipe | rows | accuracy | roc_auc | log_loss | brier_score | ece_10_bins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT_1h_h1 | logistic_price | 13125 | 0.53280 | 0.54625 | 0.69030 | 0.24858 | 0.01412 |
| BTCUSDT_1h_h1 | logistic_full | 13125 | 0.53128 | 0.54699 | 0.69037 | 0.24861 | 0.01605 |
| BTCUSDT_1h_h1 | logistic_noise | 13125 | 0.52990 | 0.54487 | 0.69065 | 0.24875 | 0.01753 |
| BTCUSDT_1h_h1 | xgb_full | 13125 | 0.53897 | 0.54643 | 0.69024 | 0.24854 | 0.00815 |
| BTCUSDT_1h_h1 | constant_05 | 13125 | 0.50842 | 0.50000 | 0.69315 | 0.25000 | 0.00842 |
| BTCUSDT_1h_h1 | prevalence | 13125 | 0.50842 | 0.50068 | 0.69304 | 0.24995 | 0.00321 |
| BTCUSDT_1h_h1 | xgb_raw | 13125 | 0.53760 | 0.54796 | 0.68979 | 0.24832 | 0.00531 |
| BTCUSDT_1h_h4 | logistic_price | 13116 | 0.53172 | 0.54028 | 0.69038 | 0.24862 | 0.01201 |
| BTCUSDT_1h_h4 | logistic_full | 13116 | 0.53019 | 0.54124 | 0.69047 | 0.24866 | 0.01190 |
| BTCUSDT_1h_h4 | logistic_noise | 13116 | 0.52775 | 0.53671 | 0.69099 | 0.24892 | 0.01184 |
| BTCUSDT_1h_h4 | xgb_full | 13116 | 0.53652 | 0.54011 | 0.69043 | 0.24864 | 0.00568 |
| BTCUSDT_1h_h4 | constant_05 | 13116 | 0.51700 | 0.50000 | 0.69315 | 0.25000 | 0.01700 |
| BTCUSDT_1h_h4 | prevalence | 13116 | 0.51700 | 0.50175 | 0.69284 | 0.24984 | 0.01112 |
| BTCUSDT_1h_h4 | xgb_raw | 13116 | 0.53881 | 0.54696 | 0.68940 | 0.24813 | 0.01042 |
| BTCUSDT_4h_h1 | logistic_price | 3279 | 0.54224 | 0.56137 | 0.68773 | 0.24729 | 0.01569 |
| BTCUSDT_4h_h1 | logistic_full | 3279 | 0.53644 | 0.55811 | 0.68911 | 0.24793 | 0.02646 |
| BTCUSDT_4h_h1 | logistic_noise | 3279 | 0.54376 | 0.55829 | 0.68900 | 0.24790 | 0.01573 |
| BTCUSDT_4h_h1 | xgb_full | 3279 | 0.53065 | 0.55149 | 0.68937 | 0.24812 | 0.01630 |
| BTCUSDT_4h_h1 | constant_05 | 3279 | 0.51754 | 0.50000 | 0.69315 | 0.25000 | 0.01754 |
| BTCUSDT_4h_h1 | prevalence | 3279 | 0.51754 | 0.49311 | 0.69266 | 0.24976 | 0.00684 |
| BTCUSDT_4h_h1 | xgb_raw | 3279 | 0.54590 | 0.55509 | 0.68849 | 0.24767 | 0.01178 |
| BTCUSDT_1d_h1 | logistic_price | 544 | 0.50735 | 0.53858 | 0.69937 | 0.25291 | 0.05809 |
| BTCUSDT_1d_h1 | logistic_full | 544 | 0.50368 | 0.51528 | 0.70945 | 0.25769 | 0.06889 |
| BTCUSDT_1d_h1 | logistic_noise | 544 | 0.51287 | 0.52958 | 0.70480 | 0.25536 | 0.05894 |
| BTCUSDT_1d_h1 | xgb_full | 544 | 0.50919 | 0.50236 | 0.69639 | 0.25161 | 0.03989 |
| BTCUSDT_1d_h1 | constant_05 | 544 | 0.52022 | 0.50000 | 0.69315 | 0.25000 | 0.02022 |
| BTCUSDT_1d_h1 | prevalence | 544 | 0.47978 | 0.49636 | 0.69568 | 0.25126 | 0.03766 |
| BTCUSDT_1d_h1 | xgb_raw | 544 | 0.50184 | 0.51321 | 0.69785 | 0.25232 | 0.03812 |
| ETHUSDT_1h_h1 | logistic_price | 13125 | 0.53417 | 0.55042 | 0.68925 | 0.24806 | 0.00979 |
| ETHUSDT_1h_h1 | logistic_full | 13125 | 0.53349 | 0.55048 | 0.68932 | 0.24810 | 0.01089 |
| ETHUSDT_1h_h1 | logistic_noise | 13125 | 0.53524 | 0.54922 | 0.68959 | 0.24823 | 0.01083 |
| ETHUSDT_1h_h1 | xgb_full | 13125 | 0.54034 | 0.55434 | 0.68859 | 0.24773 | 0.00368 |
| ETHUSDT_1h_h1 | constant_05 | 13125 | 0.50910 | 0.50000 | 0.69315 | 0.25000 | 0.00910 |
| ETHUSDT_1h_h1 | prevalence | 13125 | 0.49897 | 0.49377 | 0.69320 | 0.25003 | 0.00893 |
| ETHUSDT_1h_h1 | xgb_raw | 13125 | 0.54057 | 0.55360 | 0.68887 | 0.24787 | 0.01018 |
| ETHUSDT_1h_h4 | logistic_price | 13116 | 0.53484 | 0.54273 | 0.69067 | 0.24876 | 0.00566 |
| ETHUSDT_1h_h4 | logistic_full | 13116 | 0.53507 | 0.54425 | 0.69051 | 0.24868 | 0.00516 |
| ETHUSDT_1h_h4 | logistic_noise | 13116 | 0.53195 | 0.54224 | 0.69079 | 0.24881 | 0.00270 |
| ETHUSDT_1h_h4 | xgb_full | 13116 | 0.52981 | 0.54634 | 0.69022 | 0.24854 | 0.01060 |
| ETHUSDT_1h_h4 | constant_05 | 13116 | 0.50694 | 0.50000 | 0.69315 | 0.25000 | 0.00694 |
| ETHUSDT_1h_h4 | prevalence | 13116 | 0.50694 | 0.49815 | 0.69308 | 0.24997 | 0.00141 |
| ETHUSDT_1h_h4 | xgb_raw | 13116 | 0.53683 | 0.54649 | 0.68958 | 0.24822 | 0.00528 |
| ETHUSDT_4h_h1 | logistic_price | 3279 | 0.53888 | 0.54866 | 0.68997 | 0.24841 | 0.00864 |
| ETHUSDT_4h_h1 | logistic_full | 3279 | 0.54071 | 0.54616 | 0.69113 | 0.24895 | 0.01042 |
| ETHUSDT_4h_h1 | logistic_noise | 3279 | 0.53522 | 0.54454 | 0.69061 | 0.24874 | 0.01403 |
| ETHUSDT_4h_h1 | xgb_full | 3279 | 0.52303 | 0.53959 | 0.69276 | 0.24980 | 0.02048 |
| ETHUSDT_4h_h1 | constant_05 | 3279 | 0.50168 | 0.50000 | 0.69315 | 0.25000 | 0.00168 |
| ETHUSDT_4h_h1 | prevalence | 3279 | 0.50168 | 0.50447 | 0.69312 | 0.24999 | 0.00124 |
| ETHUSDT_4h_h1 | xgb_raw | 3279 | 0.53370 | 0.55110 | 0.68914 | 0.24800 | 0.00997 |
| ETHUSDT_1d_h1 | logistic_price | 544 | 0.52574 | 0.54505 | 0.69212 | 0.24946 | 0.03504 |
| ETHUSDT_1d_h1 | logistic_full | 544 | 0.51287 | 0.53069 | 0.69959 | 0.25300 | 0.05973 |
| ETHUSDT_1d_h1 | logistic_noise | 544 | 0.51838 | 0.53904 | 0.69630 | 0.25148 | 0.05506 |
| ETHUSDT_1d_h1 | xgb_full | 544 | 0.50919 | 0.50256 | 0.69760 | 0.25219 | 0.04031 |
| ETHUSDT_1d_h1 | constant_05 | 544 | 0.50735 | 0.50000 | 0.69315 | 0.25000 | 0.00735 |
| ETHUSDT_1d_h1 | prevalence | 544 | 0.49265 | 0.47668 | 0.69447 | 0.25066 | 0.01719 |
| ETHUSDT_1d_h1 | xgb_raw | 544 | 0.52022 | 0.55218 | 0.68956 | 0.24821 | 0.04010 |

Los controles p=0,5 y prevalencia se evalúan sin reentrenamiento. La calibración sigmoide del árbol sólo usa validación anterior al test. La logística conserva probabilidades propias. Se dispone de una sola semilla fija; no se presenta como replicación por semillas.

### Diagnóstico nuevo de errores

La brecha train/OOS del árbol identifica sobreajuste; la comparación crudo/calibrado separa discriminación de probabilidades. El porcentaje de todas las filas cuya dirección alcista perdería tras fricción cuantifica el desajuste entre clasificación y beneficio. La cobertura p≥0,55 es descriptiva: no se modifican umbrales a partir de ella. AUC train es media por fold y AUC OOS se agrupa sobre todos los folds, de modo que la brecha es una señal diagnóstica, no un estimador pareado formal.

| task | direction_up_but_loses_after_cost_% | xgb_train_auc_mean | xgb_oos_raw_auc | xgb_raw_loss | xgb_calibrated_loss | logistic_p055_coverage_% | logistic_p055_hit_% | payoff_tau_min | payoff_tau_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT_1d_h1 | 8.63971 | 0.63334 | 0.51321 | 0.69785 | 0.69639 | 19.30147 | 49.52381 | 0.54012 | 0.63730 |
| BTCUSDT_1h_h1 | 32.48762 | 0.57546 | 0.54796 | 0.68979 | 0.69024 | 24.99810 | 55.28802 | 1.00000 | 1.00000 |
| BTCUSDT_1h_h4 | 19.80787 | 0.57501 | 0.54696 | 0.68940 | 0.69043 | 15.17993 | 56.75540 | 0.83961 | 0.90867 |
| BTCUSDT_4h_h1 | 19.18268 | 0.61583 | 0.55509 | 0.68849 | 0.68937 | 31.68649 | 58.61405 | 0.83612 | 0.90700 |
| ETHUSDT_1d_h1 | 4.77941 | 0.65171 | 0.55218 | 0.68956 | 0.69760 | 22.61029 | 52.84553 | 0.55814 | 0.62500 |
| ETHUSDT_1h_h1 | 26.70476 | 0.57038 | 0.55360 | 0.68887 | 0.68859 | 16.58667 | 57.37253 | 1.00000 | 1.00000 |
| ETHUSDT_1h_h4 | 14.43275 | 0.58004 | 0.54649 | 0.68958 | 0.69022 | 14.26502 | 58.47141 | 0.76047 | 0.85663 |
| ETHUSDT_4h_h1 | 13.63220 | 0.61022 | 0.55110 | 0.68914 | 0.69276 | 22.90332 | 55.65912 | 0.75844 | 0.86375 |

## Inferencia predictiva predeclarada

Ganancia positiva significa menor log-loss. La media de loss se calcula por día antes del remuestreo; bootstrap pareado con 2.000 réplicas, bloques 7/30 días dentro de segmentos continuos de cada fold, p unilateral para ganancia positiva centrado bajo H0 e intervalos puntuales de 95%. Holm controla la familia de 32 comparaciones de esta ronda; no corrige los miles de intentos previos ni convierte historia reutilizada en prueba ciega. Se usa el mayor p de ambas longitudes. Intervalos completos en CSV/JSON.

| task | recipe | comparison | mean_gain | p | p_holm |
| --- | --- | --- | --- | --- | --- |
| BTCUSDT_1h_h1 | logistic_full | logistic_price | -0.00007 | 0.67766 | 1.00000 |
| BTCUSDT_1h_h1 | logistic_full | logistic_noise | 0.00027 | 0.19890 | 1.00000 |
| BTCUSDT_1h_h1 | logistic_full | constant_05 | 0.00277 | 0.00100 | 0.02899 |
| BTCUSDT_1h_h1 | xgb_full | constant_05 | 0.00291 | 0.00050 | 0.01599 |
| BTCUSDT_1h_h4 | logistic_full | logistic_price | -0.00008 | 0.68016 | 1.00000 |
| BTCUSDT_1h_h4 | logistic_full | logistic_noise | 0.00052 | 0.11044 | 1.00000 |
| BTCUSDT_1h_h4 | logistic_full | constant_05 | 0.00267 | 0.00250 | 0.06747 |
| BTCUSDT_1h_h4 | xgb_full | constant_05 | 0.00270 | 0.00750 | 0.18741 |
| BTCUSDT_4h_h1 | logistic_full | logistic_price | -0.00137 | 0.88306 | 1.00000 |
| BTCUSDT_4h_h1 | logistic_full | logistic_noise | -0.00011 | 0.57221 | 1.00000 |
| BTCUSDT_4h_h1 | logistic_full | constant_05 | 0.00401 | 0.11394 | 1.00000 |
| BTCUSDT_4h_h1 | xgb_full | constant_05 | 0.00377 | 0.01749 | 0.41979 |
| BTCUSDT_1d_h1 | logistic_full | logistic_price | -0.01008 | 0.99850 | 1.00000 |
| BTCUSDT_1d_h1 | logistic_full | logistic_noise | -0.00465 | 0.93003 | 1.00000 |
| BTCUSDT_1d_h1 | logistic_full | constant_05 | -0.01630 | 0.99150 | 1.00000 |
| BTCUSDT_1d_h1 | xgb_full | constant_05 | -0.00324 | 0.78061 | 1.00000 |
| ETHUSDT_1h_h1 | logistic_full | logistic_price | -0.00007 | 0.76762 | 1.00000 |
| ETHUSDT_1h_h1 | logistic_full | logistic_noise | 0.00027 | 0.12994 | 1.00000 |
| ETHUSDT_1h_h1 | logistic_full | constant_05 | 0.00383 | 0.00050 | 0.01599 |
| ETHUSDT_1h_h1 | xgb_full | constant_05 | 0.00456 | 0.00050 | 0.01599 |
| ETHUSDT_1h_h4 | logistic_full | logistic_price | 0.00017 | 0.46627 | 1.00000 |
| ETHUSDT_1h_h4 | logistic_full | logistic_noise | 0.00028 | 0.33833 | 1.00000 |
| ETHUSDT_1h_h4 | logistic_full | constant_05 | 0.00265 | 0.00600 | 0.15592 |
| ETHUSDT_1h_h4 | xgb_full | constant_05 | 0.00294 | 0.00150 | 0.04198 |
| ETHUSDT_4h_h1 | logistic_full | logistic_price | -0.00114 | 0.80560 | 1.00000 |
| ETHUSDT_4h_h1 | logistic_full | logistic_noise | -0.00051 | 0.66267 | 1.00000 |
| ETHUSDT_4h_h1 | logistic_full | constant_05 | 0.00202 | 0.18791 | 1.00000 |
| ETHUSDT_4h_h1 | xgb_full | constant_05 | 0.00039 | 0.43878 | 1.00000 |
| ETHUSDT_1d_h1 | logistic_full | logistic_price | -0.00747 | 0.98001 | 1.00000 |
| ETHUSDT_1d_h1 | logistic_full | logistic_noise | -0.00329 | 0.79260 | 1.00000 |
| ETHUSDT_1d_h1 | logistic_full | constant_05 | -0.00644 | 0.89955 | 1.00000 |
| ETHUSDT_1d_h1 | xgb_full | constant_05 | -0.00445 | 0.89105 | 1.00000 |

## Resultados económicos y estrés

Todos los porcentajes siguientes son retornos acumulados, no anualizados. Sharpe usa retornos diarios con cash sin interés y 365 días/año. Cada fold empieza/termina flat, con costos de liquidación. Se incluyen días sin operaciones en la ventana de ejecución. Los pocos extremos omitidos por purga se identifican en los splits.

Costo por lado: comisión 10 bps + slippage 5 + medio spread 2; aproximadamente 34 bps ida y vuelta, aplicado multiplicativamente. No son fills reales ni una estimación histórica de profundidad. `double` duplica toda la fricción; `delay` demora una barra adicional sin extender el final del fold. Las operaciones cuyo precio de salida falta se descartan, sin inventar ejecuciones.

`fixed_055` opera full capital con p≥0,55. `payoff_risk` exige expectativa >1,5 veces costo, estimando retornos positivos/negativos sólo en validación, con tendencia EMA30d, volumen relativo≥0,25 y tamaño causal máximo 100% para objetivo anual 20% de volatilidad. `trend_risk` usa los mismos filtros/tamaño sin ML. Buy/hold opera exactamente entre los extremos de cada tarea/fold y paga sus costos. Cash retorna cero.

| task | recipe | policy | gross_% | net_% | double_% | delay_% | sharpe | drawdown_% | trades | positive_folds | buy_hold_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT_1h_h1 | logistic_price | fixed_055 | 220.01322 | -99.99602 | -100.00000 | -99.99806 | -19.73568 | -99.99602 | 3322 | 0 | 151.03046 |
| BTCUSDT_1h_h1 | logistic_price | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 151.03046 |
| BTCUSDT_1h_h1 | logistic_full | fixed_055 | 142.58680 | -99.99653 | -100.00000 | -99.99832 | -20.42737 | -99.99653 | 3281 | 0 | 151.03046 |
| BTCUSDT_1h_h1 | logistic_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 151.03046 |
| BTCUSDT_1h_h1 | logistic_noise | fixed_055 | 244.20489 | -99.99625 | -100.00000 | -99.99849 | -19.63511 | -99.99628 | 3361 | 0 | 151.03046 |
| BTCUSDT_1h_h1 | logistic_noise | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 151.03046 |
| BTCUSDT_1h_h1 | xgb_full | fixed_055 | 145.02365 | -99.99956 | -100.00000 | -99.99962 | -22.40756 | -99.99956 | 3888 | 0 | 151.03046 |
| BTCUSDT_1h_h1 | xgb_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 151.03046 |
| BTCUSDT_1h_h1 | trend | trend_risk | 34.11401 | -99.99908 | -100.00000 | -99.99905 | -21.35986 | -99.99908 | 7591 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | logistic_price | fixed_055 | 41.65891 | -93.35795 | -99.68857 | -92.96834 | -6.09671 | -93.39549 | 900 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | logistic_price | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | logistic_full | fixed_055 | 86.74454 | -96.96885 | -99.95080 | -96.09002 | -7.15147 | -96.98813 | 1212 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | logistic_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | logistic_noise | fixed_055 | 82.89068 | -93.30951 | -99.75525 | -93.70253 | -6.41738 | -93.30951 | 973 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | logistic_noise | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | xgb_full | fixed_055 | 113.94717 | -99.27070 | -99.99751 | -99.43646 | -8.40659 | -99.27856 | 1671 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | xgb_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 151.03046 |
| BTCUSDT_1h_h4 | trend | trend_risk | 33.79465 | -94.04950 | -99.73391 | -94.13350 | -10.57514 | -94.21123 | 1994 | 0 | 151.03046 |
| BTCUSDT_4h_h1 | logistic_price | fixed_055 | 189.42249 | -90.75866 | -99.70492 | -96.66871 | -5.23775 | -90.75866 | 1013 | 0 | 150.81109 |
| BTCUSDT_4h_h1 | logistic_price | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 150.81109 |
| BTCUSDT_4h_h1 | logistic_full | fixed_055 | 115.46236 | -93.70229 | -99.81593 | -96.52966 | -6.20358 | -93.70450 | 1039 | 0 | 150.81109 |
| BTCUSDT_4h_h1 | logistic_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 150.81109 |
| BTCUSDT_4h_h1 | logistic_noise | fixed_055 | 125.35641 | -93.50208 | -99.81264 | -96.00731 | -5.92151 | -93.50480 | 1043 | 0 | 150.81109 |
| BTCUSDT_4h_h1 | logistic_noise | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 150.81109 |
| BTCUSDT_4h_h1 | xgb_full | fixed_055 | 197.15023 | -91.94062 | -99.78141 | -97.68364 | -5.38016 | -92.32060 | 1061 | 0 | 150.81109 |
| BTCUSDT_4h_h1 | xgb_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 150.81109 |
| BTCUSDT_4h_h1 | trend | trend_risk | 38.13227 | -94.44945 | -99.77580 | -94.28696 | -9.80693 | -94.59634 | 1953 | 0 | 150.81109 |
| BTCUSDT_1d_h1 | logistic_price | fixed_055 | 57.88127 | 17.05492 | -13.21423 | -15.06461 | 0.53168 | -24.08771 | 88 | 2 | 144.38266 |
| BTCUSDT_1d_h1 | logistic_price | payoff_risk | -6.79679 | -10.43302 | -13.92362 | -5.15254 | -1.65696 | -10.60792 | 20 | 0 | 144.38266 |
| BTCUSDT_1d_h1 | logistic_full | fixed_055 | 4.73779 | -26.70738 | -48.71188 | -24.48210 | -0.70916 | -30.59232 | 105 | 0 | 144.38266 |
| BTCUSDT_1d_h1 | logistic_full | payoff_risk | -14.67031 | -19.25905 | -23.59647 | 0.62966 | -2.41584 | -19.33105 | 27 | 0 | 144.38266 |
| BTCUSDT_1d_h1 | logistic_noise | fixed_055 | 32.16072 | -10.91253 | -39.94757 | -26.43366 | -0.15584 | -27.64201 | 116 | 1 | 144.38266 |
| BTCUSDT_1d_h1 | logistic_noise | payoff_risk | -8.86417 | -14.69825 | -20.15249 | -7.43286 | -1.32917 | -14.86012 | 35 | 0 | 144.38266 |
| BTCUSDT_1d_h1 | xgb_full | fixed_055 | 99.03215 | -29.67931 | -75.15481 | -56.04708 | -0.41694 | -57.12775 | 306 | 0 | 144.38266 |
| BTCUSDT_1d_h1 | xgb_full | payoff_risk | 32.05303 | -10.95353 | -39.92541 | -5.40133 | -0.39468 | -29.50396 | 212 | 0 | 144.38266 |
| BTCUSDT_1d_h1 | trend | trend_risk | 39.72647 | -22.74623 | -57.25326 | -16.94970 | -0.76968 | -38.83996 | 331 | 0 | 144.38266 |
| ETHUSDT_1h_h1 | logistic_price | fixed_055 | 73.67948 | -99.84908 | -99.99987 | -99.85490 | -10.89145 | -99.84961 | 2073 | 0 | 8.49215 |
| ETHUSDT_1h_h1 | logistic_price | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 8.49215 |
| ETHUSDT_1h_h1 | logistic_full | fixed_055 | 70.86551 | -99.89575 | -99.99994 | -99.87677 | -11.75619 | -99.89587 | 2177 | 0 | 8.49215 |
| ETHUSDT_1h_h1 | logistic_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 8.49215 |
| ETHUSDT_1h_h1 | logistic_noise | fixed_055 | 62.53980 | -99.90083 | -99.99994 | -99.91342 | -10.99549 | -99.90117 | 2177 | 0 | 8.49215 |
| ETHUSDT_1h_h1 | logistic_noise | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 8.49215 |
| ETHUSDT_1h_h1 | xgb_full | fixed_055 | 92.48485 | -99.97996 | -100.00000 | -99.97686 | -14.00027 | -99.98027 | 2697 | 0 | 8.49215 |
| ETHUSDT_1h_h1 | xgb_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 8.49215 |
| ETHUSDT_1h_h1 | trend | trend_risk | 15.06764 | -99.85585 | -99.99982 | -99.85466 | -15.00029 | -99.85591 | 5654 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | logistic_price | fixed_055 | 26.33577 | -89.18707 | -99.07454 | -89.02982 | -3.81989 | -90.33331 | 723 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | logistic_price | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | logistic_full | fixed_055 | 71.64179 | -93.23333 | -99.73324 | -94.33467 | -4.00767 | -93.75739 | 951 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | logistic_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | logistic_noise | fixed_055 | 63.64945 | -89.85962 | -99.37166 | -90.75020 | -3.79384 | -90.82296 | 818 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | logistic_noise | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | xgb_full | fixed_055 | 39.93680 | -94.40770 | -99.77652 | -92.73019 | -4.49758 | -95.56503 | 947 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | xgb_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 8.49215 |
| ETHUSDT_1h_h4 | trend | trend_risk | 13.94525 | -79.90137 | -96.44170 | -79.74168 | -6.40604 | -80.26699 | 1469 | 0 | 8.49215 |
| ETHUSDT_4h_h1 | logistic_price | fixed_055 | 61.66012 | -85.87854 | -98.76646 | -92.88326 | -3.47271 | -86.38807 | 717 | 0 | 9.39261 |
| ETHUSDT_4h_h1 | logistic_price | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 9.39261 |
| ETHUSDT_4h_h1 | logistic_full | fixed_055 | 18.20714 | -90.80152 | -99.28421 | -94.54040 | -3.91462 | -90.92082 | 751 | 0 | 9.39261 |
| ETHUSDT_4h_h1 | logistic_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 9.39261 |
| ETHUSDT_4h_h1 | logistic_noise | fixed_055 | -0.21798 | -91.91195 | -99.34441 | -91.02774 | -4.34873 | -92.22960 | 739 | 0 | 9.39261 |
| ETHUSDT_4h_h1 | logistic_noise | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 9.39261 |
| ETHUSDT_4h_h1 | xgb_full | fixed_055 | 6.77803 | -93.75519 | -99.63478 | -95.24299 | -4.43591 | -94.14132 | 835 | 0 | 9.39261 |
| ETHUSDT_4h_h1 | xgb_full | payoff_risk | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 0 | 9.39261 |
| ETHUSDT_4h_h1 | trend | trend_risk | 22.86491 | -79.36146 | -96.52048 | -81.78785 | -6.11652 | -79.64588 | 1437 | 0 | 9.39261 |
| ETHUSDT_1d_h1 | logistic_price | fixed_055 | 156.31793 | 82.43948 | 29.85493 | -37.59958 | 1.24583 | -24.30213 | 100 | 2 | 7.34687 |
| ETHUSDT_1d_h1 | logistic_price | payoff_risk | -2.71247 | -3.45131 | -4.18356 | 2.95882 | -0.79290 | -3.73999 | 4 | 1 | 7.34687 |
| ETHUSDT_1d_h1 | logistic_full | fixed_055 | 34.60904 | -11.39622 | -41.67833 | -21.89969 | -0.02087 | -37.13405 | 123 | 1 | 7.34687 |
| ETHUSDT_1d_h1 | logistic_full | payoff_risk | -1.03280 | -3.24201 | -5.39879 | -3.48683 | -0.51752 | -4.99760 | 13 | 1 | 7.34687 |
| ETHUSDT_1d_h1 | logistic_noise | fixed_055 | 105.44674 | 34.31485 | -12.18907 | -20.67558 | 0.66777 | -34.76305 | 125 | 2 | 7.34687 |
| ETHUSDT_1d_h1 | logistic_noise | payoff_risk | 18.51056 | 15.30326 | 12.18723 | 7.93663 | 0.64253 | -9.38651 | 16 | 2 | 7.34687 |
| ETHUSDT_1d_h1 | xgb_full | fixed_055 | 19.32421 | -52.99547 | -81.48387 | -43.27698 | -0.74469 | -63.87222 | 274 | 0 | 7.34687 |
| ETHUSDT_1d_h1 | xgb_full | payoff_risk | -5.09807 | -8.40332 | -11.58801 | -11.68404 | -1.27306 | -8.69722 | 24 | 0 | 7.34687 |
| ETHUSDT_1d_h1 | trend | trend_risk | 16.06286 | -16.07567 | -39.27994 | -19.61407 | -0.42542 | -38.25057 | 241 | 1 | 7.34687 |

## Comparación económica primaria

Ganancia diaria media de payoff_risk menos buy/hold, misma ventana, con bootstrap por días y Holm separado en 16 contrastes. Esto prueba superioridad histórica frente al benchmark; no demuestra alpha ajustado por beta, ni rentabilidad futura. El control trend_risk ayuda a distinguir reducción de exposición de señal incremental. El drawdown sólo se observa en aperturas, no dentro de cada vela.

| task | recipe | comparison | mean_gain | p | p_holm |
| --- | --- | --- | --- | --- | --- |
| BTCUSDT_1h_h1 | logistic_full | buy_hold | -0.00202 | 0.97401 | 1.00000 |
| BTCUSDT_1h_h1 | xgb_full | buy_hold | -0.00202 | 0.97401 | 1.00000 |
| BTCUSDT_1h_h4 | logistic_full | buy_hold | -0.00202 | 0.97401 | 1.00000 |
| BTCUSDT_1h_h4 | xgb_full | buy_hold | -0.00202 | 0.97401 | 1.00000 |
| BTCUSDT_4h_h1 | logistic_full | buy_hold | -0.00203 | 0.97651 | 1.00000 |
| BTCUSDT_4h_h1 | xgb_full | buy_hold | -0.00203 | 0.97651 | 1.00000 |
| BTCUSDT_1d_h1 | logistic_full | buy_hold | -0.00237 | 0.98351 | 1.00000 |
| BTCUSDT_1d_h1 | xgb_full | buy_hold | -0.00216 | 0.98801 | 1.00000 |
| ETHUSDT_1h_h1 | logistic_full | buy_hold | -0.00082 | 0.69765 | 1.00000 |
| ETHUSDT_1h_h1 | xgb_full | buy_hold | -0.00082 | 0.69765 | 1.00000 |
| ETHUSDT_1h_h4 | logistic_full | buy_hold | -0.00082 | 0.69765 | 1.00000 |
| ETHUSDT_1h_h4 | xgb_full | buy_hold | -0.00082 | 0.69765 | 1.00000 |
| ETHUSDT_4h_h1 | logistic_full | buy_hold | -0.00081 | 0.69215 | 1.00000 |
| ETHUSDT_4h_h1 | xgb_full | buy_hold | -0.00081 | 0.69215 | 1.00000 |
| ETHUSDT_1d_h1 | logistic_full | buy_hold | -0.00086 | 0.70865 | 1.00000 |
| ETHUSDT_1d_h1 | xgb_full | buy_hold | -0.00096 | 0.72814 | 1.00000 |

## Preparación para seguimiento en dashboard

0 candidatos cumplen todos los filtros. Cada modelo queda serializado con cutoff, schema, parámetros y hashes; `shadow_manifest.json` conserva todas las aceptaciones y rechazos. El dashboard muestra estas decisiones en ‘Investigación de nuevas señales y temporalidades’. Esta integración es de consulta: no activa modelos ni registra predicciones históricas como prospectivas. El próximo seguimiento requiere congelar un período futuro distinto de todas las reservas existentes, contrato de ejecución open/open y contabilidad con los costos de esta receta; el log legado del dashboard usa otro costo y close/close, y no debe utilizarse como validación económica de estos candidatos.

## Integridad, límites y conclusión

Se verificaron 75,136 archivos previos por SHA-256 al cierre. Modelos históricos, reservas y resultados anteriores conservaron sus bytes. Ninguna fecha >=2025-07-01 entró en features, selección o evaluación; sólo se calcularon hashes de archivos completos que contienen períodos en cuarentena. No se abrió una reserva final independiente.

Verificación adicional: 96 modelos recargados, 60128 filas de predicción verificadas y 288 carteras reconstruidas con flujos de caja/acciones independientes desde aperturas originales. Diferencia máxima al leer probabilidades CSV como float64: 2.9800415024539006e-08; error en precisión nativa: 8.326672684688674e-17. Las probabilidades float32 se recuperan exactamente al convertir el decimal CSV a su precisión nativa; no se relajó la igualdad de esos modelos. No hubo reentrenamiento en esta verificación. Recibo: `artifacts\edge_research\verification\20261018T015816_ca7033e2_20261018T023343.json`.

Software: suite completa 321 pruebas aprobadas; posteriormente, 10 pruebas dirigidas de esta investigación aprobadas (incluyen las iniciales y las nuevas de cuarentena, bootstrap y contabilidad independiente, con capital casi agotado). Lint global correcto y formato de archivos de esta tarea correcto. El chequeo global de formato detectó 9 archivos preexistentes pendientes; se dejaron intactos. `pip check` sin incompatibilidades. Evidencia inicial en `artifacts/edge_research/verification/software_checks.json`, chequeo general en `software_final_checks.json` y prueba final de precisión en `numerical_verifier_checks.json`.

La selección ex post de dos activos supervivientes, las condiciones de un solo exchange, pocos ciclos, correlación BTC/ETH, una semilla y la historia reutilizada limitan la generalización. La fórmula de payoff supone magnitudes condicionales constantes en validación; p(up) no es p(beneficio neto). Calibración y filtros no crean información si no existe señal. Los resultados positivos aislados conservan ese carácter. No se presupone impacto cero al aumentar capital, ni que una AUC ligeramente mayor que 0,5 pague las comisiones. No se modelan restricciones de lotes ni órdenes mínimas; una estrategia con pérdida extrema dejaría de ser operable antes que en esta contabilidad fraccional.

XGBoost funciona aquí como contraste diagnóstico: no cuenta con versiones no lineales de precio y ruido de igual dimensión y, por esa carencia de controles, no puede superar el filtro de promoción de esta ronda. Un rechazo por ese requisito no equivale a demostrar que todo modelo de árboles carezca de señal. Los criterios económicos exigen superar buy/hold; una estrategia con menor retorno y riesgo podría ser útil, pero necesitaría un contraste predeclarado con exposición equivalente antes de atribuirle ventaja.

El dashboard previo calcula P&L con 14 bps aproximados y close/close y permite `seed_recent_closed_predictions`: cargar retrospectivamente predicciones posteriores al entrenamiento no las vuelve observaciones emitidas antes del resultado. Esas filas requieren identificación y separación antes de cualquier evaluación prospectiva. Se conservaron intactos sus logs; esta ronda no los usa como evidencia.

Conclusión: esta ronda no descubre una ventaja predictiva y económica robusta que justifique promover un modelo. Se conserva el sistema previo y se documentan también los resultados negativos.

## Próximos pasos recomendados

1. Evitar nuevas búsquedas sobre estos folds. Mantener congelados estos modelos y los indicios de BTC diario del estudio financiero anterior como hipótesis separadas.
2. Registrar un seguimiento futuro disjunto de todas las reservas, con criterio de parada por calendario, comparación simultánea contra cash, precio y tendencia/riesgo y multiplicidad explícita. No evaluarlo hasta completar su ventana.
3. Medir spreads, slippage y latencia reales para el tamaño de capital previsto mediante cotizaciones registradas; usar paper trading y ledger ejecutable, sin órdenes reales.
4. Priorizar una hipótesis acotada de magnitud condicional: estimar E[retorno neto | señales, volatilidad], contrastada contra retorno cero y un modelo simple de volatilidad, para detectar cuándo un movimiento esperado alcanza la fricción. El payoff global usado aquí no distingue magnitudes según las features; su abstención no prueba que todo filtro posible sea inviable. Registrar criterios antes de experimentar y confirmar en datos nuevos, sin volver a elegir parámetros con estos resultados. Una reducción de riesgo debe compararse contra un baseline con exposición equivalente.

## Reproducción

```powershell
.\.venv\Scripts\python.exe -m trading_ml.edge_research.study run --protocol config/edge_research.toml
# Reanudar una corrida existente sin repetir modelos completados:
.\.venv\Scripts\python.exe -m trading_ml.edge_research.study run --study artifacts/edge_research/runs/20261018T015816_ca7033e2
.\.venv\Scripts\python.exe scripts/verify_edge_research.py artifacts/edge_research/runs/20261018T015816_ca7033e2
```

Una corrida nueva agrega otro experimento; no repetirla para escoger mejores métricas. `source.zip` conserva el código previo a los ajustes. SHA-256 acredita integridad local, no autenticación frente a adversarios.

Referencias primarias: [validación temporal](https://scikit-learn.org/stable/modules/cross_validation.html), [calibración](https://scikit-learn.org/stable/modules/calibration.html), [sesgo de selección](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [procedencia Binance](https://github.com/binance/binance-public-data).
