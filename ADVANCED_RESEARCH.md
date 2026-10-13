# Investigación avanzada de predicción y selección

Run: `20261013T224826_25c6cac4`.

Candidato examinado: **calibrated**; aceptación de desarrollo: **False**. Recomendación conservada: **defensive_reference**. No se promovió ningún modelo.

Todo el historial es desarrollo observado. La reserva prospectiva existente se preservó y no se evaluó. Las 19 operaciones se auditaron para formular hipótesis, no para probar filtros diseñados con sus propios resultados. La selección se congeló antes de reejecutar ese período y usa únicamente los siete folds anteriores.

## Auditoría individual

Ganadoras: 7; perdedoras: 12. Cada operación, su retorno y PnL, RSI, ATR, volatilidad, momentum, pendiente, volumen y señales SMC al cierre anterior están en nineteen_trade_audit.csv/json. Las diferencias de media/mediana son descriptivas, condicionadas a 19 entradas seleccionadas. No se presentan p-values iid ni se concluye que un filtro hubiera evitado pérdidas de forma independiente. Una característica común sólo origina una hipótesis que requiere ventanas distintas y finalmente datos futuros.

## Datos e hipótesis

El lector antiguo conserva sólo OHLCV. La extensión opcional recupera quote_volume, trades, taker_base y taker_quote de ZIP existentes verificados por SHA-256. No se inventan datos de flujo ni se descargan archivos nuevos; se comprueba que esos mismos ZIP reproduzcan los precios/volúmenes del CSV original. Ese CSV y todos los archivos de modelos/reservas permanecen intactos.

Familias independientes: core sin MACD/signal y distancias redundantes; momentum 20/60, pendiente/eficiencia; z-score/reversión; ruptura de extremos anteriores y rango/volatilidad; flujo taker, quote volume, intensidad y tamaño medio de operaciones; SMC causal con pivotes confirmados. Sólo la combinación flow+momentum se fija a priori, por su hipótesis de persistencia de presión compradora; no se ensamblan todas las estrategias automáticamente.

Flow es volumen taker agregado al cierre, no order book ni número de compras individuales. Los ratios evitan niveles de precio/volumen y escalas monetarias dominantes. Ablaciones usan las mismas filas de train/evaluación y etiquetas. Se guardan entrenamiento, brecha de overfitting, bins de calibración y PSI con bins de train; detectar drift no cambia retrospectivamente un parámetro.

Optimización moderada: profundidad 2/3 y dos grados de regularización, ventana expansiva/móvil de 730 días, calibración sigmoide en una cola purgada de 90 días. Calibrated y su control raw comparten exactamente el mismo fit. Los scalers de logística se ajustan sólo en train. No hay redes neuronales ni búsqueda adicional tras observar los resultados.

Direction_h1 conserva el objetivo original close/close. Net_h3 y net_h5 usan próxima apertura/exit open con comisiones y slippage. Return_magnitude_h3 regresa el retorno neto y calibra el score con etiquetas de la cola disjunta. Cada objetivo tiene baselines propios: no se comparan accuracy/AUC de etiquetas u horizontes distintos como si fueran equivalentes.

## Reglas de evaluación

Train se purga por fecha real de target. Predicciones de la evaluación no participan de ningún scaler, calibraje, payoff medio o threshold. Se compara con XGBoost original refit sobre las mismas filas (control de información/entrenamiento), además del modelo defensivo registrado. La comparación contra el registrado por sí sola no identifica el efecto de features, pues su fecha de ajuste es anterior; el control refit lo documenta.

Bootstrap pareado por bloques de 7/30 velas dentro de cada fold. Los intervalos simultáneos usan max-|t| bootstrap centrado sobre todos los pares, métricas y bloques de cada objetivo. Son aproximados, condicionales a este historial y estas predicciones, y no garantizan control post-selección sobre todas las decisiones anteriores del proyecto ni confirmación futura.

Aceptación predefinida exige ganancia de log-loss y AUC, evidencia simultánea contra original/logística/prevalencia/p=0,5, estabilidad entre folds, retorno neto positivo, al menos 30 operaciones y cobertura suficiente frente a la referencia. Una cartera en efectivo o con pocas entradas no se acepta por aparentar un drawdown o una rentabilidad mejores.

## Probabilidades y filtros

Thresholds de entrada 0,55/0,60 fijados, control económico con payoffs de train y abstención. Se mantienen las salidas, sizing, costos, límites y drawdown defensivos. Los net-horizon models declaran su plazo diferente y no participan de la selección principal direction_h1.

Meta logística/XGBoost entrenan exclusivamente con oportunidades de predicciones base OOS anteriores, incluidas dos ventanas semilla de 2021. Las etiquetas evalúan retorno unitario al salir por ATR/probabilidad/máximo cinco velas; se censuran salidas que cruzan una ventana sin forecasts. No se etiqueta con una probabilidad in-sample del modelo base. El label aislado no promete reproducir el drawdown de una cartera ni trims parciales.

El filtro modifica únicamente la entrada. Su p estima rentabilidad neta de la oportunidad; el control económico combina medias/se de ganancias y pérdidas de train. El límite inferior de expectativa es un escenario aproximado y no una garantía: los trades hipotéticos solapados no son iid. Sin muestra suficiente, se registra abstención, no una probabilidad cero interpretada como certeza. Cada aceptación/rechazo y su motivo conocido al cierre se conserva en *_decisions.csv, diferenciando oportunidades hipotéticas de las entradas realmente ejecutadas por la referencia.

## Comparaciones

### selection_development

| Experimento | Neto | DD | Trades | PF | Capital medio | Tiempo invertido |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defensive_reference | -5.91% | -6.90% | 74 | 0.553 | 1.59% | 16.88% |
| cash | 0.00% | 0.00% | 0 | N/D | 0.00% | 0.00% |
| buy_hold | 117.00% | -66.73% | 7 | 3.057 | 99.52% | 99.52% |
| original_refit | -5.62% | -7.53% | 82 | 0.622 | 1.77% | 18.96% |
| core_deduplicated | -7.13% | -7.69% | 76 | 0.528 | 1.70% | 18.40% |
| momentum | -4.72% | -7.08% | 78 | 0.671 | 1.86% | 19.68% |
| mean_reversion | -4.89% | -6.78% | 75 | 0.607 | 1.59% | 17.52% |
| volatility_breakout | -5.74% | -6.03% | 72 | 0.535 | 1.62% | 17.76% |
| order_flow | -5.29% | -7.17% | 74 | 0.574 | 1.64% | 17.44% |
| smc | -7.99% | -8.40% | 79 | 0.528 | 1.82% | 19.52% |
| flow_momentum | -3.36% | -6.08% | 74 | 0.687 | 1.59% | 17.36% |
| direction_economic_gate | -3.36% | -6.08% | 74 | 0.687 | 1.59% | 17.36% |
| confidence_060 | -2.20% | -4.26% | 35 | 0.615 | 0.91% | 9.12% |
| regularized | -1.33% | -5.04% | 74 | 0.862 | 1.41% | 15.84% |
| rolling_730 | -3.15% | -5.99% | 59 | 0.712 | 1.12% | 14.08% |
| calibration_raw_control | -4.11% | -5.79% | 78 | 0.655 | 1.55% | 16.56% |
| calibrated | 1.70% | -1.88% | 33 | 1.435 | 0.98% | 10.56% |
| depth_3 | -1.12% | -4.68% | 72 | 0.882 | 1.38% | 15.36% |
| net_h3_reference | 4.90% | -4.99% | 148 | 1.225 | 3.21% | 35.12% |
| net_h3_flow | 0.03% | -5.83% | 103 | 1.015 | 2.26% | 24.64% |
| net_h5_reference | 4.63% | -6.40% | 117 | 1.220 | 4.37% | 45.36% |
| net_h5_flow | 3.18% | -6.53% | 107 | 1.169 | 3.66% | 41.60% |
| return_magnitude_h3 | 1.67% | -5.48% | 132 | 1.094 | 3.02% | 31.36% |
| logistic_direction_h1 | -1.36% | -2.80% | 74 | 0.873 | 1.65% | 17.68% |
| logistic_net_h3 | 6.36% | -5.09% | 161 | 1.298 | 3.69% | 38.64% |
| logistic_net_h5 | 12.56% | -6.74% | 139 | 1.558 | 5.06% | 54.48% |
| meta_logistic | -4.47% | -5.66% | 56 | 0.519 | 1.04% | 12.08% |
| meta_logistic_economic | -2.87% | -4.04% | 50 | 0.596 | 0.91% | 10.80% |
| meta_xgb | -2.76% | -4.88% | 61 | 0.730 | 1.26% | 13.92% |
| meta_xgb_economic | -1.75% | -2.25% | 26 | 0.529 | 0.41% | 4.56% |
| trend_gate | -2.35% | -4.21% | 35 | 0.656 | 0.90% | 8.56% |
| volatility_gate | -5.93% | -6.73% | 63 | 0.508 | 1.45% | 14.24% |

Objetivo **direction_h1**; comparaciones válidas dentro de esta tabla.

| Modelo | Accuracy | AUC | Log-loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| calibrated | 50.32% | 0.5282 | 0.69329 | 0.25008 |
| calibration_raw_control | 51.05% | 0.5185 | 0.69994 | 0.25328 |
| constant_direction_h1 | 49.60% | 0.5000 | 0.69315 | 0.25000 |
| core_deduplicated | 51.93% | 0.5153 | 0.70352 | 0.25495 |
| defensive_reference | 52.89% | 0.5235 | 0.70127 | 0.25382 |
| depth_3 | 51.05% | 0.5196 | 0.70009 | 0.25338 |
| flow_momentum | 50.96% | 0.5152 | 0.70324 | 0.25480 |
| logistic_direction_h1 | 51.45% | 0.5278 | 0.70058 | 0.25344 |
| mean_reversion | 51.93% | 0.5132 | 0.70347 | 0.25493 |
| momentum | 52.01% | 0.5107 | 0.70381 | 0.25513 |
| momentum_sign_direction_h1 | 50.72% | 0.5073 | 0.69672 | 0.25178 |
| order_flow | 50.32% | 0.5139 | 0.70301 | 0.25470 |
| original_refit | 52.09% | 0.5233 | 0.70113 | 0.25378 |
| prevalence_direction_h1 | 49.60% | 0.4778 | 0.69513 | 0.25099 |
| regularized | 50.56% | 0.5144 | 0.69905 | 0.25289 |
| rolling_730 | 51.37% | 0.5082 | 0.70286 | 0.25471 |
| smc | 51.85% | 0.5185 | 0.70206 | 0.25423 |
| volatility_breakout | 51.45% | 0.5166 | 0.70248 | 0.25447 |

Objetivo **net_h3**; comparaciones válidas dentro de esta tabla.

| Modelo | Accuracy | AUC | Log-loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| constant_net_h3 | 48.78% | 0.5000 | 0.69315 | 0.25000 |
| logistic_net_h3 | 52.44% | 0.5413 | 0.70585 | 0.25534 |
| momentum_sign_net_h3 | 48.29% | 0.4832 | 0.70160 | 0.25421 |
| net_h3_flow | 51.14% | 0.4994 | 0.70372 | 0.25513 |
| net_h3_reference | 49.84% | 0.4985 | 0.70710 | 0.25672 |
| prevalence_net_h3 | 48.78% | 0.4699 | 0.69552 | 0.25119 |
| return_magnitude_h3 | 49.76% | 0.4905 | 0.71192 | 0.25908 |

Objetivo **net_h5**; comparaciones válidas dentro de esta tabla.

| Modelo | Accuracy | AUC | Log-loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| constant_net_h5 | 49.18% | 0.5000 | 0.69315 | 0.25000 |
| logistic_net_h5 | 54.36% | 0.5540 | 0.70782 | 0.25590 |
| momentum_sign_net_h5 | 47.04% | 0.4706 | 0.70411 | 0.25546 |
| net_h5_flow | 52.30% | 0.5121 | 0.70803 | 0.25700 |
| net_h5_reference | 51.23% | 0.5165 | 0.71033 | 0.25793 |
| prevalence_net_h5 | 49.18% | 0.4591 | 0.69971 | 0.25326 |

### audited_observed_development

| Experimento | Neto | DD | Trades | PF | Capital medio | Tiempo invertido |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defensive_reference | -1.51% | -2.37% | 19 | 0.555 | 1.55% | 13.49% |
| cash | 0.00% | 0.00% | 0 | N/D | 0.00% | 0.00% |
| buy_hold | -20.33% | -52.97% | 1 | 0.000 | 100.00% | 100.00% |
| original_refit | 0.57% | -1.69% | 16 | 1.247 | 0.84% | 9.13% |
| core_deduplicated | -2.27% | -3.74% | 15 | 0.482 | 1.27% | 11.41% |
| momentum | -2.48% | -4.05% | 15 | 0.498 | 1.09% | 9.13% |
| mean_reversion | -2.00% | -3.02% | 16 | 0.442 | 1.08% | 9.75% |
| volatility_breakout | -2.89% | -4.27% | 17 | 0.419 | 1.27% | 10.37% |
| order_flow | -0.90% | -2.54% | 23 | 0.760 | 1.42% | 14.73% |
| smc | 0.00% | -2.31% | 19 | 1.001 | 1.19% | 10.79% |
| flow_momentum | -0.50% | -2.27% | 21 | 0.852 | 1.30% | 13.49% |
| direction_economic_gate | -0.50% | -2.27% | 21 | 0.852 | 1.30% | 13.49% |
| confidence_060 | -1.46% | -2.41% | 9 | 0.384 | 0.44% | 5.39% |
| regularized | -0.30% | -2.46% | 17 | 0.895 | 0.88% | 10.37% |
| rolling_730 | 0.02% | -1.07% | 20 | 1.012 | 0.91% | 11.20% |
| calibration_raw_control | -0.55% | -2.47% | 17 | 0.801 | 0.88% | 10.37% |
| calibrated | -1.18% | -2.70% | 12 | 0.615 | 0.91% | 11.41% |
| depth_3 | -2.12% | -3.28% | 20 | 0.478 | 0.95% | 10.79% |
| net_h3_reference | 0.67% | -2.48% | 28 | 1.151 | 1.55% | 17.43% |
| net_h3_flow | -4.14% | -5.56% | 48 | 0.455 | 2.51% | 29.25% |
| net_h5_reference | -1.68% | -4.41% | 29 | 0.775 | 2.82% | 29.25% |
| net_h5_flow | -1.26% | -3.14% | 35 | 0.817 | 3.59% | 35.68% |
| return_magnitude_h3 | -4.09% | -7.88% | 162 | 0.796 | 8.60% | 100.00% |
| logistic_direction_h1 | -0.32% | -1.88% | 23 | 0.915 | 1.66% | 14.73% |
| logistic_net_h3 | -1.31% | -3.42% | 67 | 0.871 | 4.57% | 41.29% |
| logistic_net_h5 | -1.39% | -5.74% | 61 | 0.860 | 5.70% | 62.24% |
| meta_logistic | -1.93% | -2.31% | 14 | 0.369 | 1.27% | 10.79% |
| meta_logistic_economic | -0.96% | -1.40% | 11 | 0.576 | 1.01% | 8.30% |
| meta_xgb | -0.27% | -1.20% | 16 | 0.887 | 1.25% | 11.20% |
| meta_xgb_economic | -0.16% | -1.53% | 12 | 0.904 | 0.56% | 6.02% |
| trend_gate | -0.03% | -1.05% | 9 | 0.977 | 0.91% | 7.26% |
| volatility_gate | -1.65% | -2.19% | 16 | 0.474 | 1.40% | 11.20% |
| candidate_matched_exposure | -1.18% | -2.70% | 12 | 0.615 | 0.91% | 11.41% |
| reference_matched_exposure | -0.93% | -1.46% | 19 | 0.556 | 0.96% | 13.49% |

Objetivo **direction_h1**; comparaciones válidas dentro de esta tabla.

| Modelo | Accuracy | AUC | Log-loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| calibrated | 48.13% | 0.4849 | 0.69760 | 0.25221 |
| calibration_raw_control | 49.79% | 0.4849 | 0.70402 | 0.25532 |
| constant_direction_h1 | 49.59% | 0.5000 | 0.69315 | 0.25000 |
| core_deduplicated | 49.59% | 0.4815 | 0.70617 | 0.25635 |
| defensive_reference | 49.17% | 0.4782 | 0.70747 | 0.25702 |
| depth_3 | 49.38% | 0.4938 | 0.70541 | 0.25595 |
| flow_momentum | 53.73% | 0.5081 | 0.70272 | 0.25461 |
| logistic_direction_h1 | 51.04% | 0.5021 | 0.70401 | 0.25521 |
| mean_reversion | 47.72% | 0.4726 | 0.70694 | 0.25674 |
| momentum | 49.17% | 0.4838 | 0.70601 | 0.25620 |
| momentum_sign_direction_h1 | 52.07% | 0.5206 | 0.69401 | 0.25043 |
| order_flow | 51.45% | 0.5084 | 0.70074 | 0.25370 |
| original_refit | 47.93% | 0.4905 | 0.70200 | 0.25431 |
| prevalence_direction_h1 | 49.59% | 0.5000 | 0.69361 | 0.25023 |
| regularized | 50.41% | 0.4839 | 0.70443 | 0.25552 |
| rolling_730 | 48.96% | 0.4995 | 0.71028 | 0.25823 |
| smc | 48.55% | 0.4979 | 0.70122 | 0.25394 |
| volatility_breakout | 48.55% | 0.4816 | 0.70576 | 0.25616 |

Objetivo **net_h3**; comparaciones válidas dentro de esta tabla.

| Modelo | Accuracy | AUC | Log-loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| constant_net_h3 | 46.88% | 0.5000 | 0.69315 | 0.25000 |
| logistic_net_h3 | 51.67% | 0.5321 | 0.69549 | 0.25122 |
| momentum_sign_net_h3 | 47.29% | 0.4712 | 0.70361 | 0.25521 |
| net_h3_flow | 50.42% | 0.5089 | 0.69624 | 0.25153 |
| net_h3_reference | 50.00% | 0.4967 | 0.69966 | 0.25312 |
| prevalence_net_h3 | 46.88% | 0.5000 | 0.69402 | 0.25044 |
| return_magnitude_h3 | 46.88% | 0.5000 | 0.72498 | 0.26565 |

Objetivo **net_h5**; comparaciones válidas dentro de esta tabla.

| Modelo | Accuracy | AUC | Log-loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| constant_net_h5 | 44.98% | 0.5000 | 0.69315 | 0.25000 |
| logistic_net_h5 | 49.79% | 0.5231 | 0.71377 | 0.25949 |
| momentum_sign_net_h5 | 47.07% | 0.4677 | 0.70405 | 0.25543 |
| net_h5_flow | 50.42% | 0.5270 | 0.69450 | 0.25067 |
| net_h5_reference | 53.14% | 0.5190 | 0.69550 | 0.25113 |
| prevalence_net_h5 | 44.98% | 0.5000 | 0.69798 | 0.25241 |

## Decisiones de selección anteriores al período auditado

- core_deduplicated: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_logistic_direction_h1, fails_point_log_loss_vs_original_refit, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_exposure_normalized_development_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- momentum: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_logistic_direction_h1, fails_point_log_loss_vs_original_refit, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- mean_reversion: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_logistic_direction_h1, fails_point_log_loss_vs_original_refit, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- volatility_breakout: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_logistic_direction_h1, fails_point_log_loss_vs_original_refit, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- order_flow: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_logistic_direction_h1, fails_point_log_loss_vs_original_refit, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- smc: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_logistic_direction_h1, fails_point_log_loss_vs_original_refit, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_exposure_normalized_development_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- flow_momentum: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_logistic_direction_h1, fails_point_log_loss_vs_original_refit, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- regularized: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- rolling_730: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_logistic_direction_h1, fails_point_log_loss_vs_original_refit, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- calibration_raw_control: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.
- calibrated: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, insufficient_auc_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7.
- depth_3: aceptación=False; fallas: fails_point_log_loss_vs_constant_direction_h1, fails_point_log_loss_vs_prevalence_direction_h1, insufficient_auc_gain, insufficient_log_loss_gain, no_joint_economic_gain_at_shared_risk_budget, no_joint_loss_evidence_vs_constant_direction_h1_block_30, no_joint_loss_evidence_vs_constant_direction_h1_block_7, no_joint_loss_evidence_vs_logistic_direction_h1_block_30, no_joint_loss_evidence_vs_logistic_direction_h1_block_7, no_joint_loss_evidence_vs_original_refit_block_30, no_joint_loss_evidence_vs_original_refit_block_7, no_joint_loss_evidence_vs_prevalence_direction_h1_block_30, no_joint_loss_evidence_vs_prevalence_direction_h1_block_7, nonpositive_development_net_return, unstable_predictive_folds.

El control de exposición equivalente del período auditado usa un multiplicador fijado con exposición media de desarrollo, aplicado a cap, riesgo ATR y objetivo de volatilidad. Es aproximado; la exposición realizada puede diferir por régimen y frecuencia. La comparación principal conserva el mismo presupuesto por operación y reporta capital/tiempo, sin llamar capacidad predictiva a una pérdida menor por invertir menos.

Más detalles: selection_economic_uncertainty.json, selection_frozen.json, summary.json, archivos predictive por fold, modelos UBJSON/coeficientes JSON, auditoría de las 19 operaciones y decisions/trades/equity/events de cada variante. Se preservan todas las variantes rechazadas para auditar el presupuesto de búsqueda; no se convierten en modelos activos.

Ajustes realizados: 198/210; registros: 524. Todos los ajustes usan un hilo CPU y hasta 160 árboles. No se repite la búsqueda hasta encontrar ganancias.

Fuentes primarias: [Binance, datos públicos](https://github.com/binance/binance-public-data), [Spot API: campos de klines](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data), [scikit-learn: calibración disjunta](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).

No se evaluó la reserva prospectiva, no hubo promoción y no hay operaciones reales.

## Análisis complementario, sin ajustes adicionales

Se evaluaron 214 oportunidades OOS anteriores al período de las 19 operaciones, incluyendo oportunidades solapadas. Bootstrap sobre calendario diario de cada fold, enmascarado a etiquetas observadas, para evitar tratar siete oportunidades espaciadas como si fueran siete días consecutivos.

| Meta predictor | Accuracy | AUC | Log-loss |
| --- | ---: | ---: | ---: |
| logistic | 54.21% | 0.5492 | 0.86321 |
| xgb | 54.67% | 0.5088 | 0.69332 |
| training_intercept | 57.01% | 0.4188 | 0.68699 |
| constant_05 | 57.01% | 0.5000 | 0.69315 |

El intercepto se lee del XGBoost congelado de cada fold: es un prior ajustado únicamente en train, sin usar features de la oportunidad. No se estima con las 19 operaciones. Véase [documentación oficial de base_score](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters). Estos resultados no son comparables con la tabla direction_h1: aquí el objetivo es rentabilidad de una oportunidad, no dirección de la próxima vela. Sus intervalos/criterios están en meta_ranking.json y meta_acceptance_diagnostic.json.

- meta_logistic: aceptación=False; fallas: nonpositive_selection_development_return, no_joint_incremental_log_loss_gain_vs_train_intercept.
- meta_logistic_economic: aceptación=False; fallas: nonpositive_selection_development_return, no_joint_incremental_log_loss_gain_vs_train_intercept.
- meta_xgb: aceptación=False; fallas: nonpositive_selection_development_return, no_joint_incremental_log_loss_gain_vs_train_intercept.
- meta_xgb_economic: aceptación=False; fallas: nonpositive_selection_development_return, too_few_executed_trades, no_joint_incremental_log_loss_gain_vs_train_intercept.

Rechazos de las 19 entradas originales (sólo descriptivos):

| Filtro | Entradas originales admitidas | Ganadoras rechazadas | Perdedoras rechazadas |
| --- | ---: | ---: | ---: |
| meta_logistic | 13 | 4 | 2 |
| meta_logistic_economic | 7 | 4 | 8 |
| meta_xgb | 13 | 1 | 5 |
| meta_xgb_economic | 9 | 2 | 8 |
| trend_gate | 9 | 4 | 6 |
| volatility_gate | 16 | 2 | 1 |

Las decisiones se generaron antes de conocer los resultados; cruzarlas ahora con ganadoras/perdedoras describe el resultado, no ajusta un nuevo filtro ni lo valida de forma independiente. Cambiar entradas cambia las posiciones/cooldowns posteriores, de modo que los trades de una cartera filtrada no tienen que ser un subconjunto de las 19.

## Control de exposición fijado con desarrollo

Referencia ajustada: capital medio 0.96%, neto -0.93%, DD -1.46%; candidato ajustado: capital medio 0.91%, neto -1.18%, DD -2.70%. Los factores fueron congelados antes del período auditado. No es una igualdad exacta de exposición realizada, pero evita atribuir la pérdida menor frente a la referencia íntegra automáticamente a mejor selección.

## Presupuesto y reproducibilidad

198 ajustes computados contra límite 210, incluidos 26 intentos de la corrida técnica fallida. 172 ajustes nuevos y 21 modelos recuperados verificando train, parámetros, corte y cola de calibración. El fallo fue de dtype del pool meta, no un resultado que causara cambio de hipótesis. La corrida fallida se conserva.

La búsqueda quedó cerrada: no se agregaron recetas ni se eligieron umbrales para mejorar el período auditado. El candidato seleccionado por log-loss anterior fue rechazado por los criterios conjuntos. La referencia defensiva permanece vigente y ninguna variante se promovió.

```powershell
.\.venv\Scripts\python.exe -m trading_ml advanced-study
# Recuperar modelos de una corrida técnica fallida del mismo protocolo:
.\.venv\Scripts\python.exe -m trading_ml advanced-study --cache-from artifacts/advanced_studies/20261013T224439_cdea5362
# Analizar una ejecución completada, sin ajustes:
.\.venv\Scripts\python.exe scripts/analyze_advanced_results.py artifacts/advanced_studies/20261013T224826_25c6cac4
.\.venv\Scripts\python.exe scripts/record_advanced_study.py artifacts/advanced_studies/20261013T224826_25c6cac4
```

Cada ejecución y análisis tienen directorios nuevos, sin sobrescribir artifacts/models, datos originales o reservas. El análisis complementario rechaza reemplazar su carpeta existente. Las probabilidades/modelos se guardan en JSON/UBJSON, sin pickle. Los módulos advanced_research separan datos, features, targets, modelos, meta-labels, incertidumbre, auditoría y orquestación. run_overlay admite entry_gate opcional para filtrar entradas conservando las probabilidades y reglas de salida de la base.
