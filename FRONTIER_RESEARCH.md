# Investigación predictiva multi-activo

Toda la evidencia histórica es de desarrollo. El test final independiente queda sellado hasta octubre de 2027; no existe confirmación futura todavía.

Corrida: `20261014T134646_e17d06e3`. Ajustes: 453 / 500. Registros: 892.

## Auditoría previa

ETH no tenía historial registrado: se incorporó en una ruta separada, con archivos públicos Binance y SHA256. BTC, modelos originales, punteros activos y reservas anteriores permanecen intactos.

Momentum, Mean Reversion y SMC son módulos existentes. Breakout era una familia de variables; MIXTO era una clasificación de resultados de riesgo. No se presupone una estrategia independiente MIXTO.

El Profit Factor previo de Mean Reversion sumaba retornos unitarios; esta fase utiliza ganancias/pérdidas monetarias. Exposición de capital y tiempo en mercado se reportan por separado. El objetivo close/close original no equivale al retorno ejecutable open/open.

## Protocolo

Tres folds walk-forward, purga por disponibilidad exacta y embargo de seis días entre etiquetas de train, calibración y evaluación. Escalado sólo en train. Calibración en cola temporal independiente de 120 días. Stacking sobre probabilidades de esa cola, fuera del ajuste de cada componente. Cada fold conserva recetas, hashes, modelos, métricas y operaciones.

Ocho recetas tabulares comparan features locales contra features cruzadas BTC/ETH, regularización, desbalance y ventana móvil. Logistic, XGBoost, Random Forest, LightGBM y CatBoost; ensamble equiponderado y stacking. Calibración sin invertir rankings. Redes profundas excluidas por pocos miles de observaciones diarias correlacionadas y escasos ciclos independientes.

Cross-asset usa la misma referencia de cierre, sin rellenar precios faltantes. Correlación y beta móviles no prueban cointegración. Los regímenes se calculan causalmente y sus métricas son descriptivas, sin optimizar cortes después de observar resultados.

Dirección h1, retorno neto h3, barreras de ±2 ATR supervisadas al cierre con vencimiento a cinco velas. TP/SL se ejecutan en la apertura siguiente: no se estima orden intravela. Timeout significa TP no alcanzado, no pérdida obligatoria. Magnitud y retorno/volatilidad se evalúan como regresiones con MAE/MSE frente al promedio de train.

Todas las probabilidades se prueban con la misma política defensiva y umbrales .55/.60; selección sólo en folds anteriores. El overlay mantiene hasta cinco velas y no utiliza el TP del etiquetado: sus retornos miden utilidad de señal bajo una política común, no reproducción exacta del evento objetivo.

Costos por lado: comisión 10 bps, slippage 5 bps, medio spread 1 bp; spread completo 2 bps. Supuestos reproducibles, no mediciones del libro. Cash sin interés; buy-and-hold con fricción. Ninguna receta se activa para trading real.

Criterios: mejora de loss > .003 sobre baselines, AUC > .505, mejora en al menos dos de tres folds, neto positivo, ≥30 operaciones, PF>1 y superioridad de loss mediante intervalos simultáneos por bloques 7/30. Bootstrap condicional por objetivo/activo, no evidencia independiente ni corrección global entre todas las fases investigadas.

## Selección anterior al período descriptivo

| Activo / objetivo | Receta | Accuracy | Bal. accuracy | AUC | Loss | Neto selección | Operaciones | Aceptada |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BTCUSDT__direction_h1 | random_forest_cross | 0.511 | 0.512 | 0.526 | 0.6932 | 0.00% | 0 | False |
| BTCUSDT__net_h3 | logistic_local | 0.529 | 0.528 | 0.544 | 0.6968 | 4.78% | 68 | False |
| BTCUSDT__barrier_close_h5 | logistic_local | 0.869 | 0.500 | 0.509 | 0.3983 | 0.00% | 0 | False |
| ETHUSDT__direction_h1 | random_forest_cross | 0.514 | 0.514 | 0.524 | 0.6971 | 2.63% | 87 | False |
| ETHUSDT__net_h3 | stacking | 0.495 | 0.499 | 0.496 | 0.6938 | 0.00% | 0 | False |
| ETHUSDT__barrier_close_h5 | logistic_local | 0.881 | 0.500 | 0.632 | 0.3594 | 0.00% | 0 | False |

## Comparación descriptiva posterior

No es test independiente: fechas de BTC ya fueron observadas. Se conservaron las selecciones anteriores sin cambiarlas por estos resultados.

| Activo / objetivo | Sistema | Acc. | AUC | Loss | Neto | DD | PF | Trades | Capital medio | Costos |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT__direction_h1 | xgboost_local | 0.502 | 0.495 | 0.6942 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| BTCUSDT__direction_h1 | random_forest_cross | 0.492 | 0.492 | 0.6964 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| BTCUSDT__net_h3 | xgboost_local | 0.492 | 0.574 | 0.7006 | 5.28% | -1.70% | 1.82 | 41 | 4.66% | 154.48 |
| BTCUSDT__net_h3 | logistic_local | 0.467 | 0.500 | 0.7047 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| BTCUSDT__barrier_close_h5 | xgboost_local | 0.904 | 0.707 | 0.3279 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| BTCUSDT__barrier_close_h5 | logistic_local | 0.904 | 0.675 | 0.3298 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| ETHUSDT__direction_h1 | xgboost_local | 0.488 | 0.500 | 0.6952 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| ETHUSDT__direction_h1 | random_forest_cross | 0.492 | 0.462 | 0.6973 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| ETHUSDT__net_h3 | xgboost_local | 0.540 | 0.505 | 0.6946 | 1.06% | -0.16% | n/a | 4 | 0.09% | 7.35 |
| ETHUSDT__net_h3 | stacking | 0.531 | 0.468 | 0.6924 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| ETHUSDT__barrier_close_h5 | xgboost_local | 0.862 | 0.637 | 0.3970 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |
| ETHUSDT__barrier_close_h5 | logistic_local | 0.849 | 0.601 | 0.4186 | -0.11% | -0.11% | 0.00 | 1 | 0.02% | 3.47 |

## Evidencia y límites

`summary.json` conserva métricas de cada modelo/activo/objetivo, precision/recall/F1/Brier, métricas por régimen y estabilidad por fold. `experiments.jsonl` incluye variantes perdedoras y regresiones. `uncertainty.json` registra todos los contrastes emparejados. CSV de cada corrida conserva curvas y operaciones, con costos brutos/netos sobre decisiones idénticas.

Una pérdida menor con capital medio inferior no demuestra mejor predicción. Los distintos objetivos no se comparan por accuracy o loss entre sí. BTC y ETH comparten shocks: no duplican el número de observaciones independientes. Los intervalos históricos no eliminan sesgo por investigaciones anteriores.

## Test final reservado

Ventana nueva 2027-04-13T00:00:00+00:00 → 2027-10-13T00:00:00+00:00; evita reemplazar o modificar reservas anteriores. `prospective.json` congela recetas, umbrales, modelos y calibraciones ajustados sólo con datos anteriores al 1/10/2026. Evaluación de una sola apertura con control de fecha y hashes; ningún dato futuro disponible fue leído.

Conclusión: INCONCLUSO hasta validación prospectiva. Conservar modelos actuales; no hay autorización automática de promoción.

Fuentes técnicas: [gap temporal de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html), [CatBoost](https://catboost.ai/docs/en/concepts/python-installation), [archivos Binance](https://github.com/binance/binance-public-data).


## Resultados completos por variante

Los siguientes números son exclusivamente de validación anterior al período descriptivo. Ninguna fila ganadora cambia la selección registrada.

| Activo / objetivo | Modelo | Acc. | Bal. acc. | Prec. | Recall | F1 | AUC | Loss | Brier | Neto .55 | Trades | PF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT__direction_h1 | logistic_local | 0.506 | 0.507 | 0.503 | 0.542 | 0.522 | 0.518 | 0.6933 | 0.2500 | 0.31% | 14 | 1.11 |
| BTCUSDT__direction_h1 | logistic_cross | 0.502 | 0.504 | 0.499 | 0.679 | 0.576 | 0.524 | 0.6945 | 0.2506 | 2.55% | 39 | 1.45 |
| BTCUSDT__direction_h1 | xgboost_local | 0.497 | 0.498 | 0.495 | 0.658 | 0.565 | 0.514 | 0.6950 | 0.2509 | 0.00% | 0 | n/a |
| BTCUSDT__direction_h1 | xgboost_cross | 0.506 | 0.507 | 0.503 | 0.600 | 0.547 | 0.515 | 0.6947 | 0.2508 | -1.98% | 22 | 0.47 |
| BTCUSDT__direction_h1 | xgboost_balanced_rolling | 0.502 | 0.502 | 0.499 | 0.602 | 0.545 | 0.508 | 0.6977 | 0.2523 | -6.46% | 34 | 0.15 |
| BTCUSDT__direction_h1 | random_forest_cross | 0.511 | 0.512 | 0.506 | 0.660 | 0.573 | 0.526 | 0.6932 | 0.2500 | -3.47% | 26 | 0.34 |
| BTCUSDT__direction_h1 | lightgbm_cross | 0.498 | 0.498 | 0.495 | 0.613 | 0.548 | 0.512 | 0.6959 | 0.2514 | -4.04% | 31 | 0.31 |
| BTCUSDT__direction_h1 | catboost_cross | 0.512 | 0.513 | 0.506 | 0.695 | 0.586 | 0.513 | 0.6953 | 0.2511 | -1.78% | 28 | 0.68 |
| BTCUSDT__direction_h1 | stacking | 0.495 | 0.497 | 0.494 | 0.703 | 0.581 | 0.518 | 0.6948 | 0.2508 | 0.00% | 0 | n/a |
| BTCUSDT__direction_h1 | equal_ensemble | 0.501 | 0.502 | 0.498 | 0.645 | 0.562 | 0.524 | 0.6939 | 0.2504 | -1.18% | 32 | 0.80 |
| BTCUSDT__direction_h1 | training_prior | 0.497 | 0.500 | 0.497 | 1.000 | 0.664 | 0.474 | 0.6969 | 0.2519 | 0.00% | 0 | n/a |
| BTCUSDT__direction_h1 | constant_half | 0.497 | 0.500 | 0.497 | 1.000 | 0.664 | 0.500 | 0.6931 | 0.2500 | 0.00% | 0 | n/a |
| BTCUSDT__direction_h1 | regime_prior | 0.484 | 0.486 | 0.489 | 0.824 | 0.613 | 0.468 | 0.6983 | 0.2526 | -3.68% | 31 | 0.45 |
| BTCUSDT__net_h3 | logistic_local | 0.529 | 0.528 | 0.518 | 0.484 | 0.500 | 0.544 | 0.6968 | 0.2517 | 4.78% | 68 | 1.40 |
| BTCUSDT__net_h3 | logistic_cross | 0.539 | 0.540 | 0.526 | 0.554 | 0.539 | 0.535 | 0.7053 | 0.2553 | 5.73% | 72 | 1.48 |
| BTCUSDT__net_h3 | xgboost_local | 0.524 | 0.522 | 0.513 | 0.456 | 0.483 | 0.513 | 0.7014 | 0.2540 | 4.74% | 65 | 1.41 |
| BTCUSDT__net_h3 | xgboost_cross | 0.517 | 0.517 | 0.504 | 0.524 | 0.514 | 0.524 | 0.7051 | 0.2557 | 0.49% | 67 | 1.04 |
| BTCUSDT__net_h3 | xgboost_balanced_rolling | 0.523 | 0.524 | 0.509 | 0.562 | 0.534 | 0.520 | 0.7033 | 0.2549 | 3.36% | 78 | 1.24 |
| BTCUSDT__net_h3 | random_forest_cross | 0.534 | 0.534 | 0.520 | 0.554 | 0.536 | 0.528 | 0.7005 | 0.2535 | 5.07% | 67 | 1.43 |
| BTCUSDT__net_h3 | lightgbm_cross | 0.501 | 0.502 | 0.489 | 0.537 | 0.512 | 0.521 | 0.7059 | 0.2561 | 8.23% | 68 | 1.77 |
| BTCUSDT__net_h3 | catboost_cross | 0.512 | 0.513 | 0.499 | 0.529 | 0.514 | 0.527 | 0.7051 | 0.2557 | 2.12% | 73 | 1.15 |
| BTCUSDT__net_h3 | stacking | 0.523 | 0.520 | 0.512 | 0.438 | 0.472 | 0.519 | 0.7000 | 0.2533 | 4.74% | 65 | 1.41 |
| BTCUSDT__net_h3 | equal_ensemble | 0.527 | 0.527 | 0.514 | 0.524 | 0.519 | 0.528 | 0.7011 | 0.2539 | 4.41% | 68 | 1.36 |
| BTCUSDT__net_h3 | training_prior | 0.477 | 0.480 | 0.470 | 0.562 | 0.512 | 0.473 | 0.6984 | 0.2526 | -2.87% | 47 | 0.71 |
| BTCUSDT__net_h3 | constant_half | 0.487 | 0.500 | 0.487 | 1.000 | 0.655 | 0.500 | 0.6931 | 0.2500 | 0.00% | 0 | n/a |
| BTCUSDT__net_h3 | regime_prior | 0.494 | 0.496 | 0.484 | 0.603 | 0.537 | 0.475 | 0.7043 | 0.2555 | -3.46% | 32 | 0.47 |
| BTCUSDT__barrier_close_h5 | logistic_local | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.509 | 0.3983 | 0.1156 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | logistic_cross | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.511 | 0.4020 | 0.1163 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | xgboost_local | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.463 | 0.4080 | 0.1180 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | xgboost_cross | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.457 | 0.4100 | 0.1187 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | xgboost_balanced_rolling | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.464 | 0.4170 | 0.1200 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | random_forest_cross | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.470 | 0.4081 | 0.1179 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | lightgbm_cross | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.474 | 0.4150 | 0.1190 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | catboost_cross | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.490 | 0.4030 | 0.1161 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | stacking | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.478 | 0.4041 | 0.1168 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | equal_ensemble | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.475 | 0.4053 | 0.1172 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | training_prior | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.402 | 0.3941 | 0.1153 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | constant_half | 0.131 | 0.500 | 0.131 | 1.000 | 0.232 | 0.500 | 0.6931 | 0.2500 | 0.00% | 0 | n/a |
| BTCUSDT__barrier_close_h5 | regime_prior | 0.869 | 0.500 | 0.000 | 0.000 | 0.000 | 0.443 | 0.3961 | 0.1158 | 0.00% | 0 | n/a |
| ETHUSDT__direction_h1 | logistic_local | 0.509 | 0.509 | 0.507 | 0.676 | 0.579 | 0.510 | 0.7035 | 0.2550 | -2.07% | 85 | 0.88 |
| ETHUSDT__direction_h1 | logistic_cross | 0.510 | 0.510 | 0.507 | 0.657 | 0.573 | 0.504 | 0.7039 | 0.2552 | 0.77% | 77 | 1.05 |
| ETHUSDT__direction_h1 | xgboost_local | 0.510 | 0.510 | 0.509 | 0.612 | 0.556 | 0.512 | 0.6983 | 0.2525 | 2.02% | 81 | 1.13 |
| ETHUSDT__direction_h1 | xgboost_cross | 0.504 | 0.504 | 0.503 | 0.599 | 0.547 | 0.508 | 0.7002 | 0.2535 | -5.55% | 68 | 0.59 |
| ETHUSDT__direction_h1 | xgboost_balanced_rolling | 0.506 | 0.506 | 0.505 | 0.593 | 0.545 | 0.504 | 0.6973 | 0.2521 | 2.26% | 72 | 1.15 |
| ETHUSDT__direction_h1 | random_forest_cross | 0.514 | 0.514 | 0.512 | 0.603 | 0.554 | 0.524 | 0.6971 | 0.2519 | 2.63% | 87 | 1.15 |
| ETHUSDT__direction_h1 | lightgbm_cross | 0.508 | 0.508 | 0.506 | 0.686 | 0.582 | 0.511 | 0.6971 | 0.2519 | 3.42% | 66 | 1.25 |
| ETHUSDT__direction_h1 | catboost_cross | 0.506 | 0.506 | 0.505 | 0.587 | 0.543 | 0.511 | 0.6981 | 0.2524 | 2.94% | 73 | 1.21 |
| ETHUSDT__direction_h1 | stacking | 0.497 | 0.497 | 0.498 | 0.705 | 0.584 | 0.510 | 0.6979 | 0.2523 | 3.42% | 66 | 1.25 |
| ETHUSDT__direction_h1 | equal_ensemble | 0.507 | 0.507 | 0.506 | 0.619 | 0.557 | 0.511 | 0.6986 | 0.2527 | 2.14% | 75 | 1.14 |
| ETHUSDT__direction_h1 | training_prior | 0.500 | 0.500 | 0.500 | 1.000 | 0.667 | 0.488 | 0.6988 | 0.2528 | -2.07% | 46 | 0.80 |
| ETHUSDT__direction_h1 | constant_half | 0.500 | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 | 0.6931 | 0.2500 | 0.00% | 0 | n/a |
| ETHUSDT__direction_h1 | regime_prior | 0.491 | 0.491 | 0.495 | 0.865 | 0.630 | 0.483 | 0.6994 | 0.2531 | -1.86% | 65 | 0.87 |
| ETHUSDT__net_h3 | logistic_local | 0.511 | 0.513 | 0.491 | 0.550 | 0.519 | 0.508 | 0.6970 | 0.2519 | 2.41% | 25 | 1.99 |
| ETHUSDT__net_h3 | logistic_cross | 0.505 | 0.505 | 0.484 | 0.519 | 0.501 | 0.500 | 0.7003 | 0.2535 | -1.38% | 31 | 0.69 |
| ETHUSDT__net_h3 | xgboost_local | 0.504 | 0.503 | 0.483 | 0.487 | 0.485 | 0.507 | 0.7019 | 0.2541 | -3.30% | 49 | 0.61 |
| ETHUSDT__net_h3 | xgboost_cross | 0.514 | 0.509 | 0.491 | 0.392 | 0.436 | 0.500 | 0.7112 | 0.2580 | -0.49% | 49 | 0.93 |
| ETHUSDT__net_h3 | xgboost_balanced_rolling | 0.493 | 0.488 | 0.464 | 0.380 | 0.418 | 0.495 | 0.7012 | 0.2540 | -0.83% | 49 | 0.90 |
| ETHUSDT__net_h3 | random_forest_cross | 0.514 | 0.512 | 0.493 | 0.450 | 0.471 | 0.506 | 0.7007 | 0.2536 | -4.16% | 48 | 0.53 |
| ETHUSDT__net_h3 | lightgbm_cross | 0.509 | 0.503 | 0.484 | 0.371 | 0.420 | 0.505 | 0.6994 | 0.2530 | 0.90% | 38 | 1.15 |
| ETHUSDT__net_h3 | catboost_cross | 0.511 | 0.508 | 0.489 | 0.442 | 0.464 | 0.515 | 0.7030 | 0.2545 | -1.97% | 50 | 0.75 |
| ETHUSDT__net_h3 | stacking | 0.495 | 0.499 | 0.478 | 0.598 | 0.532 | 0.496 | 0.6938 | 0.2503 | 0.00% | 0 | n/a |
| ETHUSDT__net_h3 | equal_ensemble | 0.512 | 0.509 | 0.489 | 0.424 | 0.454 | 0.498 | 0.6994 | 0.2530 | 0.75% | 45 | 1.12 |
| ETHUSDT__net_h3 | training_prior | 0.479 | 0.500 | 0.479 | 1.000 | 0.648 | 0.485 | 0.7030 | 0.2549 | -2.07% | 46 | 0.80 |
| ETHUSDT__net_h3 | constant_half | 0.479 | 0.500 | 0.479 | 1.000 | 0.648 | 0.500 | 0.6931 | 0.2500 | 0.00% | 0 | n/a |
| ETHUSDT__net_h3 | regime_prior | 0.475 | 0.491 | 0.474 | 0.866 | 0.612 | 0.476 | 0.7043 | 0.2555 | -1.86% | 65 | 0.87 |
| ETHUSDT__barrier_close_h5 | logistic_local | 0.881 | 0.500 | 0.000 | 0.000 | 0.000 | 0.632 | 0.3594 | 0.1030 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | logistic_cross | 0.879 | 0.499 | 0.000 | 0.000 | 0.000 | 0.599 | 0.3674 | 0.1047 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | xgboost_local | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.597 | 0.3631 | 0.1038 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | xgboost_cross | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.550 | 0.3708 | 0.1052 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | xgboost_balanced_rolling | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.574 | 0.3746 | 0.1058 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | random_forest_cross | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.584 | 0.3618 | 0.1028 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | lightgbm_cross | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.564 | 0.3689 | 0.1048 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | catboost_cross | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.631 | 0.3596 | 0.1018 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | stacking | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.570 | 0.3784 | 0.1073 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | equal_ensemble | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.610 | 0.3618 | 0.1030 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | training_prior | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.462 | 0.3772 | 0.1080 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | constant_half | 0.118 | 0.500 | 0.118 | 1.000 | 0.211 | 0.500 | 0.6931 | 0.2500 | 0.00% | 0 | n/a |
| ETHUSDT__barrier_close_h5 | regime_prior | 0.882 | 0.500 | 0.000 | 0.000 | 0.000 | 0.440 | 0.3782 | 0.1082 | 0.00% | 0 | n/a |

## Ablaciones de datos cruzados y calibración

| Activo / objetivo | Δloss logística cross−local | Δloss XGBoost cross−local | Mejor receta (selección) |
|---|---:|---:|---|
| BTCUSDT__direction_h1 | 0.00121 | -0.00027 | random_forest_cross |
| BTCUSDT__net_h3 | 0.00860 | 0.00368 | logistic_local |
| BTCUSDT__barrier_close_h5 | 0.00365 | 0.00205 | logistic_local |
| ETHUSDT__direction_h1 | 0.00034 | 0.00196 | random_forest_cross |
| ETHUSDT__net_h3 | 0.00333 | 0.00935 | stacking |
| ETHUSDT__barrier_close_h5 | 0.00794 | 0.00771 | logistic_local |

Las métricas train/raw/calibradas por fold figuran en cada `fit.json`. El registro conserva las regresiones de magnitud y retorno/volatilidad con controles de MAE/MSE, sin convertir esas magnitudes en probabilidades inventadas.

## Regímenes y estabilidad

| Activo / objetivo | Régimen | Filas | AUC receta seleccionada | Loss | Loss XGBoost local |
|---|---|---:|---:|---:|---:|
| BTCUSDT__direction_h1 | bear | 178 | 0.525 | 0.6931 | 0.6973 |
| BTCUSDT__direction_h1 | bull | 357 | 0.535 | 0.6908 | 0.6923 |
| BTCUSDT__direction_h1 | sideways | 713 | 0.519 | 0.6944 | 0.6958 |
| BTCUSDT__net_h3 | bear | 176 | 0.544 | 0.6953 | 0.7025 |
| BTCUSDT__net_h3 | bull | 353 | 0.551 | 0.6947 | 0.6988 |
| BTCUSDT__net_h3 | sideways | 713 | 0.544 | 0.6981 | 0.7024 |
| BTCUSDT__barrier_close_h5 | bear | 174 | 0.401 | 0.4296 | 0.4437 |
| BTCUSDT__barrier_close_h5 | bull | 349 | 0.510 | 0.4311 | 0.4356 |
| BTCUSDT__barrier_close_h5 | sideways | 713 | 0.539 | 0.3747 | 0.3857 |
| ETHUSDT__direction_h1 | bear | 273 | 0.472 | 0.7094 | 0.7084 |
| ETHUSDT__direction_h1 | bull | 227 | 0.542 | 0.6928 | 0.6964 |
| ETHUSDT__direction_h1 | sideways | 748 | 0.535 | 0.6939 | 0.6951 |
| ETHUSDT__net_h3 | bear | 271 | 0.509 | 0.6964 | 0.7064 |
| ETHUSDT__net_h3 | bull | 223 | 0.480 | 0.6913 | 0.7148 |
| ETHUSDT__net_h3 | sideways | 748 | 0.498 | 0.6935 | 0.6964 |
| ETHUSDT__barrier_close_h5 | bear | 270 | 0.542 | 0.3365 | 0.3528 |
| ETHUSDT__barrier_close_h5 | bull | 219 | 0.745 | 0.3865 | 0.3895 |
| ETHUSDT__barrier_close_h5 | sideways | 747 | 0.622 | 0.3598 | 0.3591 |

## Presupuesto, fallos e integridad

La corrida anterior `20261014T134018_d1e1df02` fue abortada por una máscara de ejecución dependiente del desenlace futuro. Se conservan todos sus archivos y se marcan inválidos sus retornos. Registró 70 ajustes completados y se cargaron 72 al presupuesto, incluyendo dos posibles ajustes interrumpidos. La recuperación reutilizó 32 modelos y calibraciones validados, efectuó 381 ajustes nuevos y cargó 453 en total, con límite 500. El contador total es una cota presupuestaria conservadora, no un número exacto de ajustes completados de la corrida abortada.

Las curvas de ejecución usan todas las referencias con features disponibles, sin consultar etiquetas o tiempo de finalización. Sólo las métricas predictivas excluyen objetivos no observables antes del cierre del fold. La corrección no alteró recetas ni amplió la búsqueda.

## Comparación con presupuesto de riesgo fijado anteriormente

| Activo / objetivo | Sistema | Capital medio anterior | Capital medio realizado | Neto posterior |
|---|---|---:|---:|---:|
| BTCUSDT__net_h3 | xgboost_local | 2.30% | 4.66% | 5.28% |
| BTCUSDT__net_h3 | logistic_local | 2.36% | 0.00% | 0.00% |
| ETHUSDT__direction_h1 | xgboost_local | 1.89% | 0.00% | 0.00% |
| ETHUSDT__direction_h1 | random_forest_cross | 1.90% | 0.00% | 0.00% |

## Modelos registrados: comparación sin nuevos ajustes

Se reconstruye el contrato de features de cada modelo BTC ya registrado y se mantiene su umbral original. Todos reciben costos de esta fase y la misma política defensiva. El período es histórico observado, no test independiente.

| Modelo | Accuracy | Bal. acc. | AUC | Loss | Neto | DD | PF | Trades | Capital medio | Costos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v20261012T130540_d2bb6ac0 | 0.492 | 0.492 | 0.478 | 0.7075 | -1.55% | -2.38% | 0.55 | 19 | 1.55% | 63.64 |
| v20261012T131650_5106daf4 | 0.492 | 0.493 | 0.477 | 0.6967 | -0.45% | -0.92% | 0.00 | 2 | 0.15% | 5.00 |
| v20261012T134917_f88d6f93 | 0.496 | 0.500 | 0.491 | 0.6951 | 0.00% | 0.00% | n/a | 0 | 0.00% | 0.00 |

## Interpretación estadística

Ninguno de los seis sistemas seleccionados por activo/objetivo superó todos los criterios de aceptación. Se conserva el original. Los candidatos de dirección quedaron en efectivo durante el período posterior; retorno y drawdown cero no acreditan selección rentable. El score de barreras BTC obtuvo accuracy 90,38% pero balanced accuracy 50%: el desbalance explica gran parte de ese porcentaje. Su AUC posterior es descriptivo y no justifica reabrir selección.

La mejora de log-loss por calibración puede representar menor confianza, sin mayor capacidad de ordenar operaciones. Resultados positivos en algunos folds o con un objetivo específico no permiten escoger después un nuevo ganador. Los regímenes y activos están correlacionados; estos contrastes históricos no prueban una ventaja prospectiva.


Los presupuestos se escalan con exposición media de validación, no con el período posterior. No producen igualdad exacta de exposición realizada. `stress_historical` registra fricción duplicada. No se atribuye una mejora predictiva a invertir menos capital.

## Gráficos exportables

![Diferencias predictivas](artifacts/frontier_analyses/20261014T134646_e17d06e3_final/predictive_comparison.png)

![Curvas netas](artifacts/frontier_analyses/20261014T134646_e17d06e3_final/historical_equity.png)

## Reproducción

Instalar extras con `python -m pip install -r requirements-research.txt`. Ejecutar `python -m trading_ml frontier-study --protocol config/frontier_research.toml`. Recuperación técnica: agregar `--cache-from artifacts/frontier_studies/20261014T134018_d1e1df02`; no vuelve a ajustar modelos cuya procedencia y contrato coinciden. Documentar sin entrenar: `python scripts/record_frontier_study.py <study>`; la carpeta de análisis se crea una sola vez.

Test futuro: `python -m trading_ml frontier-future-evaluate --study <study> --btc-raw <BTC_csv_verificado> --eth-raw <ETH_csv_verificado>`. Antes del 13/10/2027 UTC rechaza la apertura sin leer los CSV. Se requieren datos desde el mismo comienzo histórico, archivos públicos verificados y cobertura completa. No entrena, recalibra, optimiza ni activa modelos.

La fase anterior queda documentada en [ADVANCED_RESEARCH.md](ADVANCED_RESEARCH.md); Momentum, Mean Reversion y gestión de riesgo conservan sus informes y resultados negativos.
