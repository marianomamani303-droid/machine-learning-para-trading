# FINAL MODEL IMPROVEMENT REPORT

Fecha: 17 de octubre de 2026 (America/Buenos_Aires). Evaluación retrospectiva walk-forward con purga, embargo y reservas prospectivas selladas intactas.

---

## RESULTADO GENERAL: NO MEJORÓ (SIN MEJORA ESTADÍSTICAMENTE SIGNIFICATIVA NI VENTAJA PREDICTABLE ROBUSTA)

- **Evidencia global verificada:** **657 ajustes reales evaluados** sin repetir entrenamientos ya completados (**441** en la comparación controlada de extensión histórica `20261017_controlled`, **72** en la expansión multimercado/derivados/flujo `20261016T211432_bcec9ebe`, **24** en el estudio Forex de nuevos pares `20261016T210703_c9107a5c` y **120** en los dos estudios DMN previos `20261016T204450_9f760c70` y `20261016T204723_6818bead`).
- **Contrastes estadísticos primarios superados:** **0 / 26** en la extensión histórica (`p_Holm = 1,0000` en todos), **0 / 5** en la ablación de nuevos activos/flujo/perpetuos/contexto (`p_Holm >= 0,6897`) y **0 / 4** en los nuevos pares Forex (`p_Holm = 1,0000`).
- **Superioridad frente a controles simples (`constant_05`, prevalencia de entrenamiento, `long_scaled`, `tsmom`):** **0 modelos** logran una reducción de log-loss estadísticamente significativa frente a `p = 0,5` tras corrección de Holm, y **ningún modelo neuronal o árbol** supera el Sharpe ni el retorno neto del baseline pasivo escalado por volatilidad (`long_scaled`).

---

## 1. Estado recuperado y verificación de integridad

1. **Estudios y artefactos verificados sin reentrenar (`new_fits = 0` en esta etapa de cierre):**
   - `artifacts/retraining_comparison/20261017_controlled/`: **441 ajustes** (tabular, estructura/régimen, especialistas híbridos, Triple Barrier, meta-labeling OOF, LSTM/DMN y Forex `EURUSD` D1/H4), con **32.658 artefactos previos verificados por SHA-256 sin cambios**.
   - `artifacts/expansion/studies/20261016T211432_bcec9ebe/`: **72 ajustes completados** (3 folds × 6 escenarios × 2 arquitecturas [`xgboost`, `lstm`] × 2 semillas [`42`, `1729`]). `python -m trading_ml.expansion verify` confirmó **768 archivos intactos**, **234 carteras reconstruidas** y **468 escenarios de costos** exactos.
   - `artifacts/expansion/forex_studies/20261016T210703_c9107a5c/`: **24 ajustes completados** (`USDCAD` y `NZDUSD` D1 con ejecución H1 bid/ask, `price` vs `compact`, 3 folds × 2 semillas). `python -m trading_ml.expansion verify` confirmó **123 archivos intactos**, **24 modelos** y **24 carteras reproducidos**.
2. **Scripts de análisis verificados y ejecutados:**
   - `scripts/analyze_expansion_study.py`: recargó los 36 modelos `XGBClassifier` (`model.ubj`) y 36 señales `LSTM` (`signals.npy`), comprobó reproducción exacta de las probabilidades guardadas (error máximo `2,98e-8`), calculó las 9 métricas de clasificación por semilla/fold/activo y ejecutó bootstrap pareado por bloques móviles (7 y 30 días, 2.000 réplicas) con corrección de Holm. Salida inmutable en `artifacts/expansion_analysis/20261016T211432_bcec9ebe/20261017T221013/`.
   - `scripts/analyze_expansion_forex.py`: verificó hashes de `artifacts.json`, reconstruyó métricas completas (incluyendo `balanced_accuracy`, `precision`, `recall`, `f1`, `ece_10` y matrices de confusión) desde `predictions.csv` y ejecutó bootstrap pareado `compact` vs `price` y contra `constant_05`. Salida inmutable en `artifacts/expansion_analysis/20261016T210703_c9107a5c/20261017T221546/`.
   - Verificación estática y de pruebas: `ruff check` limpio (`All checks passed!`) y **13/13 pruebas** en `tests/test_expansion.py` y `tests/test_retraining_comparison.py`.

---

## 2. Tabla comparativa general (Antes vs. Después sobre las mismas observaciones)

Todas las filas comparan **el mismo estimador, hiperparámetros, objetivo y fechas de evaluación fuera de muestra**, sin solapamiento temporal, con purga por `target_time` y embargo explícito:

| Modelo | Mercado | Accuracy antes | Accuracy después | AUC antes | AUC después | Log-loss antes | Log-loss después | IC95 Δ Log-loss (A−B, bloque largo) | Conclusión |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| XGBoost (`original`) | BTC/USDT D1 | 0.54167 | 0.53824 | 0.54586 | 0.54814 | 0.69113 | 0.69148 | [-0.00357, +0.00300] | Mixto / No significativo (`p_Holm = 1,0`) |
| Random Forest (`random_forest_cross`) | BTC/USDT D1 | 0.53139 | 0.54395 | 0.54202 | 0.55319 | 0.68949 | 0.68788 | [-0.00340, +0.00429] | Mejoró numéricamente; no significativo (`p_Holm = 1,0`) |
| Regresión Logística (`logistic`) | BTC/USDT D1 | 0.53995 | 0.52055 | 0.54396 | 0.54001 | 0.68925 | 0.68955 | [-0.00257, +0.00203] | Empeoró predicción; mejoró backtest (+109,40%) |
| LightGBM (`lightgbm_cross`) | BTC/USDT D1 | 0.52626 | 0.52397 | 0.52032 | 0.53642 | 0.70129 | 0.69393 | [-0.00281, +0.01293] | Mejoró AUC/log-loss; bajó Acc (`p_Holm = 1,0`) |
| CatBoost (`catboost_cross`) | BTC/USDT D1 | 0.52169 | 0.53196 | 0.53132 | 0.54706 | 0.69195 | 0.68916 | [-0.00188, +0.00715] | Mejoró numéricamente; no significativo (`p_Holm = 1,0`) |
| XGBoost + SMC (`smc`) | BTC/USDT D1 | 0.54224 | 0.53653 | 0.55249 | 0.55310 | 0.68950 | 0.69021 | [-0.00494, +0.00396] | Sin cambio material (`p_Holm = 1,0`) |
| XGBoost + Estructura (`structure`) | BTC/USDT D1 | 0.53368 | 0.53938 | 0.55223 | 0.55313 | 0.68960 | 0.68991 | [-0.00470, +0.00493] | Mejora marginal Acc/AUC; peor log-loss (`p_Holm = 1,0`) |
| XGBoost + Régimen (`regime_rules`) | BTC/USDT D1 | 0.52397 | 0.53710 | 0.53937 | 0.54338 | 0.69200 | 0.69230 | [-0.00419, +0.00324] | Mejoró Acc/AUC; peor log-loss (`p_Holm = 1,0`) |
| Híbrido Especialistas (`specialists_rules`) | BTC/USDT D1 | 0.52511 | 0.52397 | 0.53703 | 0.52910 | 0.69462 | 0.69845 | [-0.01013, +0.00112] | Empeoró en todas las métricas |
| Triple Barrier (`tb_vol_h5_cal`, evento) | BTC/USDT D1 | 0.50913 | 0.51027 | 0.48456 | 0.49248 | 0.70093 | 0.70052 | [-0.00193, +0.00312] | Sigue bajo azar (`AUC < 0,50`) |
| Meta-labeling (`meta_net_h1_logistic_cal_55`) | BTC/USDT D1 | 0.58557 | 0.57426 | 0.53420 | 0.53269 | 0.68265 | 0.68183 | [-0.00092, +0.00372] | Empeoró Acc/AUC; 0 operaciones |
| XGBoost (`original`) | ETH/USDT D1 | 0.51313 | 0.51427 | 0.51628 | 0.50759 | 0.70053 | 0.70017 | [-0.00503, +0.00787] | Empeoró AUC; log-loss peor que 0,5 |
| Random Forest (`random_forest_cross`) | ETH/USDT D1 | 0.50799 | 0.50628 | 0.52185 | 0.51580 | 0.69372 | 0.69358 | [-0.00406, +0.00642] | Empeoró Acc y AUC (`p_Holm = 1,0`) |
| Regresión Logística (`logistic`) | ETH/USDT D1 | 0.51142 | 0.53311 | 0.51951 | 0.53518 | 0.69304 | 0.69156 | [-0.00275, +0.00821] | Mejoró numéricamente; no significativo (`p_Holm = 1,0`) |
| LightGBM (`lightgbm_cross`) | ETH/USDT D1 | 0.50685 | 0.49315 | 0.52630 | 0.49941 | 0.69795 | 0.70263 | [-0.01221, +0.00661] | Empeoró en todas las métricas |
| CatBoost (`catboost_cross`) | ETH/USDT D1 | 0.50228 | 0.52397 | 0.50756 | 0.52875 | 0.69665 | 0.69240 | [-0.00039, +0.00996] | Mejoró numéricamente (6/6 pares); no significativo (`p_Holm = 1,0`) |
| XGBoost + SMC (`smc`) | ETH/USDT D1 | 0.49201 | 0.51712 | 0.50633 | 0.51849 | 0.69977 | 0.69811 | [-0.00382, +0.01020] | Mejoró numéricamente; log-loss > 0,69315 |
| XGBoost + Estructura (`structure`) | ETH/USDT D1 | 0.50114 | 0.52169 | 0.50744 | 0.52976 | 0.69969 | 0.69548 | [-0.00075, +0.01234] | Mejoró numéricamente; log-loss > 0,69315 |
| XGBoost + Régimen (`regime_rules`) | ETH/USDT D1 | 0.51427 | 0.50457 | 0.50717 | 0.50401 | 0.70143 | 0.70019 | [-0.00358, +0.00840] | Empeoró Acc y AUC |
| Híbrido Especialistas (`specialists_rules`) | ETH/USDT D1 | 0.50114 | 0.49772 | 0.50500 | 0.50430 | 0.70314 | 0.70594 | [-0.00783, +0.00519] | Empeoró en todas las métricas |
| Triple Barrier (`tb_vol_h5_cal`, evento) | ETH/USDT D1 | 0.54338 | 0.53767 | 0.51487 | 0.52447 | 0.68499 | 0.68411 | [-0.00049, +0.00189] | Mixto; sólo 1 operación ejecutada |
| Meta-labeling (`meta_net_h1_logistic_cal_55`) | ETH/USDT D1 | 0.44970 | 0.55030 | 0.50000 | 0.57895 | 0.70290 | 0.68579 | [+0.00938, +0.03347] | Muestra degenerada (1 fold, 169 filas, recall 0%, 0 trades; `p_Holm = 1,0`) |
| LSTM / DMN (`lstm_sharpe_net`, historia) | BTC/USDT D1 | 0.49500 | 0.50220 | 0.51989 | 0.51961 | N/A | N/A | IC95 ΔRet: [-12,31, +5,28] pp | Sin mejora (retorno cartera `23,62% → 21,36%`) |
| LSTM / DMN (`lstm_sharpe_net`, historia) | ETH/USDT D1 | 0.48680 | 0.50440 | 0.52035 | 0.52825 | N/A | N/A | IC95 ΔRet: [-12,31, +5,28] pp | Leve suba de ranking; cartera conjunta baja retorno |
| XGBoost (`fx_compact`, historia 2012+) | EUR/USD D1 | 0.50934 | 0.51783 | 0.50993 | 0.51343 | 0.69388 | 0.69325 | [-0.00174, +0.00333] | Mejora numérica leve; no significativa (`p_Holm = 1,0`) |
| XGBoost (`fx_price`, historia 2012+) | EUR/USD D1 | 0.49745 | 0.51104 | 0.50324 | 0.50205 | 0.69470 | 0.69499 | [-0.00306, +0.00195] | Mixto; retorno cae de `0,00%` a `-2,31%` |
| XGBoost (`fx_compact`, historia 2012+) | EUR/USD H4 | 0.51036 | 0.50891 | 0.51566 | 0.51506 | 0.69279 | 0.69307 | [-0.00165, +0.00033] | Empeoró levemente (`p_Holm = 1,0`) |
| XGBoost (`fx_price`, historia 2012+) | EUR/USD H4 | 0.51533 | 0.50365 | 0.50949 | 0.51394 | 0.69323 | 0.69292 | [-0.00027, +0.00082] | Bajó Acc (−1,17 pp); 0 operaciones |
| XGBoost panel (`anchor` → `expanded_price`) | BTC+ETH D1 (5 cripto) | 0.49245 | 0.51202 | 0.50754 | 0.52508 | 0.69468 | 0.69732 | [-0.00675, +0.00117] | Mejoró AUC/Acc; empeoró log-loss (`p_Holm = 1,0`) |
| XGBoost panel (`anchor` → `expanded_flow`) | BTC+ETH D1 (+flujo) | 0.49245 | 0.51545 | 0.50754 | 0.51793 | 0.69468 | 0.69553 | [-0.00337, +0.00158] | Mejoró AUC/Acc; empeoró log-loss (`p_Holm = 1,0`) |
| XGBoost panel (`anchor` → `expanded_derivatives`) | BTC+ETH D1 (+perp/funding) | 0.49245 | 0.52163 | 0.50754 | 0.52370 | 0.69468 | 0.69477 | [-0.00227, +0.00199] | Mejoró AUC/Acc; no supera al control con ruido (`AUC 0,52754`) |
| XGBoost panel (`anchor` → `expanded_context`) | BTC+ETH D1 (+7 FX) | 0.49245 | 0.50240 | 0.50754 | 0.51358 | 0.69468 | 0.69401 | [-0.00161, +0.00293] | Mejora irrelevante; retorno neto `-6,52%` |
| XGBoost panel (`anchor` → `expanded_noise`) | BTC+ETH D1 (+ruido) | 0.49245 | 0.50412 | 0.50754 | 0.52754 | 0.69468 | 0.69356 | [-0.00062, +0.00285] | **Control negativo gana en AUC y log-loss** (`p_Holm = 0,6897`) |
| LSTM panel (`anchor` → `expanded_flow`) | BTC+ETH D1 (5 cripto + flujo) | 0.49657 | 0.50859 | 0.50021 | 0.49798 | N/A | N/A | IC95 ΔSharpe: [-0,394, +0,323] | Retorno `+13,70% → +15,60%`, pero Sharpe cae (`1,076 → 1,035`) |
| XGBoost (`price` → `compact`) | USD/CAD D1 | 0.51307 | 0.48148 | 0.49787 | 0.46852 | 0.71490 | 0.72740 | [-0.02425, -0.00070] | Empeoró significativamente en punto (`p_Holm = 1,0`) |
| XGBoost (`price` → `compact`) | NZD/USD D1 | 0.50655 | 0.50655 | 0.50641 | 0.49809 | 0.70407 | 0.71017 | [-0.01551, +0.00327] | Empeoró AUC, log-loss y retorno (`-10,32% → -14,92%`) |

*(Nota sobre `Random Forest` en `EUR/USD`: la canalización multimercado de Forex implementa el estimador predeclarado `XGBClassifier` con calibración temporal sobre recetas `fx_price` y `fx_compact`; no existe una receta Random Forest para Forex en el protocolo del repositorio y, siguiendo las reglas de no inventar arquitecturas ni mezclar pipelines incompatibles, en Forex se reportan las recetas reales implementadas).*

---

## 3. Métricas detalladas completas por modelo (Antes A vs. Después B)

### 3.1. Criptomonedas Spot — `BTCUSDT` D1 (876 fechas OOS comunes: 2023-01-01 a 2025-06-18)

| Modelo (`recipe`) | Brazo | Accuracy | Balanced Acc. | ROC-AUC | Log-loss | Brier | Precision | Recall | F1 | ECE | Retorno Neto | Sharpe | Max DD | PF | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `original` (XGBoost) | Antes A | 0.54167 | 0.53982 | 0.54586 | 0.69113 | 0.24896 | 0.55021 | 0.60200 | 0.57487 | 0.01549 | -6.66% | 0.021 | -37.26% | 0.968 | 219.5 |
| `original` (XGBoost) | Después B | 0.53824 | 0.53517 | 0.54814 | 0.69148 | 0.24913 | 0.54388 | 0.63858 | 0.58743 | 0.02258 | +9.05% | 0.248 | -35.17% | 1.040 | 247.5 |
| `random_forest_cross` | Antes A | 0.53139 | 0.53028 | 0.54202 | 0.68949 | 0.24820 | 0.54295 | 0.56763 | 0.55501 | 0.01201 | -10.82% | -0.074 | -36.02% | 0.938 | 190.5 |
| `random_forest_cross` | Después B | 0.54395 | 0.54143 | 0.55319 | 0.68788 | 0.24740 | 0.55015 | 0.62639 | 0.58580 | 0.00922 | -7.08% | 0.006 | -36.53% | 0.960 | 208.0 |
| `logistic` | Antes A | 0.53995 | 0.53612 | 0.54396 | 0.68925 | 0.24807 | 0.54348 | 0.66519 | 0.59821 | 0.01425 | +45.52% | 0.905 | -24.16% | 1.334 | 132.0 |
| `logistic` | Después B | 0.52055 | 0.51551 | 0.54001 | 0.68955 | 0.24822 | 0.52641 | 0.68514 | 0.59538 | 0.01237 | +109.40% | 1.491 | -12.08% | 1.649 | 146.0 |
| `lightgbm_cross` | Antes A | 0.52626 | 0.52329 | 0.52032 | 0.70129 | 0.25389 | 0.53422 | 0.62306 | 0.57523 | 0.04607 | -31.38% | -0.358 | -56.97% | 0.883 | 342.0 |
| `lightgbm_cross` | Después B | 0.52397 | 0.51992 | 0.53642 | 0.69393 | 0.25038 | 0.53047 | 0.65632 | 0.58672 | 0.04392 | -19.56% | -0.140 | -47.91% | 0.933 | 345.0 |
| `catboost_cross` | Antes A | 0.52169 | 0.51932 | 0.53132 | 0.69195 | 0.24940 | 0.53095 | 0.60976 | 0.56748 | 0.02446 | -1.89% | 0.086 | -34.23% | 0.986 | 212.5 |
| `catboost_cross` | Después B | 0.53196 | 0.52882 | 0.54706 | 0.68916 | 0.24802 | 0.53825 | 0.63969 | 0.58460 | 0.01391 | -22.70% | -0.270 | -43.80% | 0.878 | 238.5 |
| `smc` | Antes A | 0.54224 | 0.53943 | 0.55249 | 0.68950 | 0.24815 | 0.54790 | 0.63415 | 0.58786 | 0.01169 | -19.85% | -0.200 | -42.54% | 0.905 | 253.0 |
| `smc` | Después B | 0.53653 | 0.53287 | 0.55310 | 0.69021 | 0.24851 | 0.54114 | 0.65632 | 0.59318 | 0.02421 | -20.50% | -0.198 | -46.04% | 0.910 | 283.0 |
| `structure` | Antes A | 0.53368 | 0.53091 | 0.55223 | 0.68960 | 0.24820 | 0.54088 | 0.62417 | 0.57953 | 0.01996 | -21.30% | -0.226 | -44.97% | 0.899 | 250.5 |
| `structure` | Después B | 0.53938 | 0.53611 | 0.55313 | 0.68991 | 0.24836 | 0.54438 | 0.64634 | 0.59099 | 0.01908 | -31.03% | -0.397 | -50.36% | 0.863 | 279.5 |
| `regime_rules` | Antes A | 0.52397 | 0.52216 | 0.53937 | 0.69200 | 0.24939 | 0.53457 | 0.58315 | 0.55779 | 0.02685 | -10.19% | -0.063 | -41.91% | 0.953 | 228.5 |
| `regime_rules` | Después B | 0.53710 | 0.53366 | 0.54338 | 0.69230 | 0.24955 | 0.54210 | 0.64967 | 0.59101 | 0.01741 | +1.70% | 0.149 | -42.70% | 1.009 | 250.0 |
| `specialists_rules` | Antes A | 0.52511 | 0.52388 | 0.53703 | 0.69462 | 0.25063 | 0.53684 | 0.56541 | 0.55076 | 0.03422 | -12.55% | -0.076 | -42.63% | 0.955 | 287.0 |
| `specialists_rules` | Después B | 0.52397 | 0.52121 | 0.52910 | 0.69845 | 0.25247 | 0.53269 | 0.61419 | 0.57055 | 0.04345 | +0.38% | 0.155 | -38.52% | 1.001 | 315.0 |
| `tb_vol_h5_cal` (evento) | Antes A | 0.50913 | 0.49748 | 0.48456 | 0.70093 | 0.25384 | 0.43417 | 0.40470 | 0.41892 | 0.06297 | -30.34% | -0.476 | -48.32% | 0.858 | 173.0 |
| `tb_vol_h5_cal` (evento) | Después B | 0.51027 | 0.49820 | 0.49248 | 0.70052 | 0.25365 | 0.43503 | 0.40209 | 0.41791 | 0.05217 | -16.10% | -0.134 | -45.45% | 0.932 | 167.0 |
| `meta_net_h1_logistic_cal_55` | Antes A | 0.58557 | 0.51630 | 0.53420 | 0.68265 | 0.24477 | 0.68182 | 0.04983 | 0.09288 | 0.04554 | +0.00% | N/A | 0.00% | N/A | 0.0 |
| `meta_net_h1_logistic_cal_55` | Después B | 0.57426 | 0.50000 | 0.53269 | 0.68183 | 0.24437 | 0.00000 | 0.00000 | 0.00000 | 0.02648 | +0.00% | N/A | 0.00% | N/A | 0.0 |

### 3.2. Criptomonedas Spot — `ETHUSDT` D1 (876 fechas OOS comunes)

| Modelo (`recipe`) | Brazo | Accuracy | Balanced Acc. | ROC-AUC | Log-loss | Brier | Precision | Recall | F1 | ECE | Retorno Neto | Sharpe | Max DD | PF | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `original` (XGBoost) | Antes A | 0.51313 | 0.51453 | 0.51628 | 0.70053 | 0.25353 | 0.52561 | 0.42697 | 0.47118 | 0.04181 | -32.09% | -0.319 | -53.88% | 0.858 | 183.5 |
| `original` (XGBoost) | Después B | 0.51427 | 0.51706 | 0.50759 | 0.70017 | 0.25341 | 0.53415 | 0.34270 | 0.41751 | 0.04100 | -2.81% | 0.093 | -38.95% | 0.982 | 151.0 |
| `random_forest_cross` | Antes A | 0.50799 | 0.50693 | 0.52185 | 0.69372 | 0.25026 | 0.51412 | 0.57303 | 0.54192 | 0.03498 | -11.25% | 0.038 | -45.47% | 0.960 | 216.0 |
| `random_forest_cross` | Después B | 0.50628 | 0.50826 | 0.51580 | 0.69358 | 0.25021 | 0.51899 | 0.38427 | 0.44158 | 0.03657 | +2.96% | 0.172 | -37.24% | 1.009 | 144.0 |
| `logistic` | Antes A | 0.51142 | 0.50782 | 0.51951 | 0.69304 | 0.24995 | 0.51339 | 0.73258 | 0.60370 | 0.01628 | -11.94% | -0.057 | -39.42% | 0.939 | 156.0 |
| `logistic` | Después B | 0.53311 | 0.53370 | 0.53518 | 0.69156 | 0.24920 | 0.54433 | 0.49663 | 0.51939 | 0.01383 | +12.35% | 0.318 | -27.99% | 1.088 | 81.0 |
| `lightgbm_cross` | Antes A | 0.50685 | 0.50493 | 0.52630 | 0.69795 | 0.25224 | 0.51197 | 0.62472 | 0.56275 | 0.05720 | -44.36% | -0.402 | -55.43% | 0.858 | 316.0 |
| `lightgbm_cross` | Después B | 0.49315 | 0.49357 | 0.49941 | 0.70263 | 0.25465 | 0.50120 | 0.46742 | 0.48372 | 0.06692 | -7.01% | 0.066 | -28.08% | 0.974 | 203.0 |
| `catboost_cross` | Antes A | 0.50228 | 0.50060 | 0.50756 | 0.69665 | 0.25172 | 0.50886 | 0.60562 | 0.55266 | 0.04116 | -45.74% | -0.536 | -57.10% | 0.824 | 228.5 |
| `catboost_cross` | Después B | 0.52397 | 0.52527 | 0.52875 | 0.69240 | 0.24963 | 0.53806 | 0.44382 | 0.48633 | 0.01902 | -15.20% | -0.082 | -44.64% | 0.928 | 149.5 |
| `smc` | Antes A | 0.49201 | 0.49248 | 0.50633 | 0.69977 | 0.25323 | 0.50023 | 0.46292 | 0.48077 | 0.06181 | -3.01% | 0.111 | -44.91% | 0.990 | 205.5 |
| `smc` | Después B | 0.51712 | 0.51926 | 0.51849 | 0.69811 | 0.25239 | 0.53426 | 0.38539 | 0.44778 | 0.03625 | -14.37% | -0.046 | -40.61% | 0.938 | 154.5 |
| `structure` | Antes A | 0.50114 | 0.50171 | 0.50744 | 0.69969 | 0.25319 | 0.50976 | 0.46629 | 0.48701 | 0.05379 | -26.52% | -0.213 | -51.14% | 0.896 | 205.5 |
| `structure` | Después B | 0.52169 | 0.52365 | 0.52976 | 0.69548 | 0.25110 | 0.53916 | 0.40112 | 0.45996 | 0.03166 | -3.21% | 0.114 | -43.32% | 0.986 | 158.5 |
| `regime_rules` | Antes A | 0.51427 | 0.51550 | 0.50717 | 0.70143 | 0.25399 | 0.52630 | 0.43820 | 0.47821 | 0.04012 | -17.90% | -0.094 | -46.84% | 0.927 | 191.5 |
| `regime_rules` | Después B | 0.50457 | 0.50641 | 0.50401 | 0.70019 | 0.25343 | 0.51633 | 0.39101 | 0.44495 | 0.04916 | -14.70% | -0.050 | -35.44% | 0.936 | 161.0 |
| `specialists_rules` | Antes A | 0.50114 | 0.50103 | 0.50500 | 0.70314 | 0.25480 | 0.50901 | 0.50787 | 0.50844 | 0.05921 | -44.56% | -0.493 | -54.77% | 0.843 | 265.0 |
| `specialists_rules` | Después B | 0.49772 | 0.49857 | 0.50430 | 0.70594 | 0.25613 | 0.50639 | 0.44494 | 0.47368 | 0.07117 | -17.16% | -0.041 | -43.74% | 0.946 | 240.0 |
| `tb_vol_h5_cal` (evento) | Antes A | 0.54338 | 0.50407 | 0.51487 | 0.68499 | 0.24595 | 0.42667 | 0.26158 | 0.32432 | 0.04707 | -5.46% | -0.557 | -7.05% | 0.431 | 3.0 |
| `tb_vol_h5_cal` (evento) | Después B | 0.53767 | 0.50258 | 0.52447 | 0.68411 | 0.24551 | 0.42339 | 0.28610 | 0.34146 | 0.04644 | -4.07% | -0.633 | -4.07% | 0.000 | 1.0 |
| `meta_net_h1_logistic_cal_55` | Antes A | 0.44970 | 0.50000 | 0.50000 | 0.70290 | 0.25487 | 0.44970 | 1.00000 | 0.62041 | 0.08601 | +0.00% | N/A | 0.00% | N/A | 0.0 |
| `meta_net_h1_logistic_cal_55` | Después B | 0.55030 | 0.50000 | 0.57895 | 0.68579 | 0.24634 | 0.00000 | 0.00000 | 0.00000 | 0.00945 | +0.00% | N/A | 0.00% | N/A | 0.0 |

### 3.3. Forex — `EURUSD` D1 (589 fechas) y H4 (3.425 barras)

| Par / TF | Modelo | Brazo | Accuracy | Balanced Acc. | ROC-AUC | Log-loss | Brier | Precision | Recall | F1 | ECE | Retorno Neto | Sharpe | Max DD | PF | Trades |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EURUSD 1d | `fx_compact` | Antes A | 0.50934 | 0.50577 | 0.50993 | 0.69388 | 0.25036 | 0.49528 | 0.36585 | 0.42084 | 0.02533 | -2.28% | -0.386 | -2.81% | 0.742 | 46.0 |
| EURUSD 1d | `fx_compact` | Después B | 0.51783 | 0.51449 | 0.51343 | 0.69325 | 0.25005 | 0.50691 | 0.38328 | 0.43651 | 0.01308 | -0.80% | -0.161 | -1.85% | 0.898 | 44.0 |
| EURUSD 1d | `fx_price` | Antes A | 0.49745 | 0.49695 | 0.50324 | 0.69470 | 0.25077 | 0.48410 | 0.47735 | 0.48070 | 0.03082 | +0.00% | N/A | 0.00% | N/A | 0.0 |
| EURUSD 1d | `fx_price` | Después B | 0.51104 | 0.50830 | 0.50205 | 0.69499 | 0.25091 | 0.49784 | 0.40070 | 0.44402 | 0.01683 | -2.31% | -0.666 | -3.25% | 0.594 | 25.0 |
| EURUSD 4h | `fx_compact` | Antes A | 0.51036 | 0.50994 | 0.51566 | 0.69279 | 0.24982 | 0.52587 | 0.52319 | 0.52453 | 0.01668 | -0.01% | -0.070 | -0.05% | 0.902 | 2.0 |
| EURUSD 4h | `fx_compact` | Después B | 0.50891 | 0.51097 | 0.51506 | 0.69307 | 0.24996 | 0.52874 | 0.44740 | 0.48468 | 0.01851 | -0.13% | -0.484 | -0.21% | 0.488 | 5.0 |
| EURUSD 4h | `fx_price` | Antes A | 0.51533 | 0.50970 | 0.50949 | 0.69323 | 0.25004 | 0.52340 | 0.68326 | 0.59274 | 0.01736 | +0.00% | N/A | 0.00% | N/A | 0.0 |
| EURUSD 4h | `fx_price` | Después B | 0.50365 | 0.50436 | 0.51394 | 0.69292 | 0.24989 | 0.52076 | 0.48247 | 0.50088 | 0.01725 | +0.00% | N/A | 0.00% | N/A | 0.0 |

---

## 4. Análisis por mercado de la ampliación de datos

### 4.1. Criptomonedas Spot
- **Extensión temporal (2017–2019 añadido a 2020–2025 en `BTCUSDT` y `ETHUSDT`):**
  - *Mejor modelo anterior (BTC D1):* `smc` (`ROC-AUC = 0,55249`, `Accuracy = 54,22%`, `log-loss = 0,68950`).
  - *Mejor modelo reentrenado (BTC D1):* `random_forest_cross` (`ROC-AUC = 0,55319`, `Accuracy = 54,39%`, `log-loss = 0,68788`).
  - *Diferencias dentro de la misma receta `random_forest_cross` en BTC:* `Δ Accuracy = +1,26 pp` (`+2,36%` relativo), `Δ ROC-AUC = +0,01117` (`+2,06%` relativo), `Δ log-loss = −0,00161` (`−0,23%` relativo). Sin embargo, el IC95 de ganancia de log-loss es `[−0,00340, +0,00429]` (`p_Holm = 1,0`) y el backtest sigue perdiendo dinero (`−7,08%` neto, `−52,24%` bajo costos dobles).
  - *En ETH D1:* el mejor modelo reentrenado direccional es `logistic` (`ROC-AUC = 0,53518`, `Accuracy = 53,31%`, `log-loss = 0,69156`), con `Δ Accuracy = +2,17 pp`, `Δ AUC = +0,01567` y `Δ log-loss = −0,00148` (`IC95 [−0,00275, +0,00821]`, `p_Holm = 1,0`). En cambio, `lightgbm_cross` se deteriora (`Δ AUC = −0,02689`, `Δ Accuracy = −1,37 pp`).
- **Expansión transversal a 5 criptos (`BTC`, `ETH`, `SOL`, `BNB`, `XRP`) y flujo spot (`20261016T211432_bcec9ebe`):**
  - Pasar de `anchor_price` (2 criptos) a `expanded_price` (5 criptos) sube el AUC de XGBoost en BTC/ETH de `0,50754` a `0,52508`, pero **empeora el log-loss** (`0,69468 → 0,69732`) porque descalibra las probabilidades (`ECE 0,02705 → 0,03673`).

### 4.2. Futuros Perpetuos y Tasas de Funding
- Evaluados en `expanded_derivatives` (añadiendo prima perpetuo/spot negociada y `funding_rate` liquidado con disponibilidad causal a `BTC`, `ETH`, `SOL`, `BNB`, `XRP`):
  - En XGBoost sobre BTC+ETH, `expanded_derivatives` logra el mayor Accuracy de la ablación (`52,16%` vs `49,24%` en `anchor_price`, `Δ = +2,92 pp`) y `ROC-AUC = 0,52370` (`Δ = +0,01616`), con retorno neto `+3,49%` (Sharpe `0,262`), pero **empeora ligeramente el log-loss** (`0,69468 → 0,69477`) y cae a `−6,70%` bajo costos dobles.
  - En LSTM, añadir derivados **reduce** el retorno neto (`+13,70% → +12,06%`) y el Sharpe (`1,076 → 0,976`).
  - **Falsación por control negativo:** el escenario `expanded_noise` (ruido gaussiano causal) obtiene `ROC-AUC = 0,52754` y `log-loss = 0,69356`, superando en ambas métricas a `expanded_derivatives`. Por tanto, la mejora observada en árboles al añadir columnas de derivados no puede atribuirse a señal genuina de perpetuos/funding.

### 4.3. Forex (`EURUSD` D1/H4 y nuevos pares `USDCAD`, `NZDUSD` D1)
- En `EURUSD` D1 (extensión de historia desde 2012 con FXCM H1 bid/ask), `fx_compact` pasa de `Accuracy = 50,93%` a `51,78%` (`+0,85 pp`), `ROC-AUC = 0,50993 → 0,51343` (`+0,00350`) y `log-loss = 0,69388 → 0,69325` (`−0,00064`), pero sigue teniendo log-loss peor que `0,69315` (`constant_05`), retorno neto negativo (`−0,80%`) y `p_Holm = 1,0`. En `EURUSD` H4 (donde el entrenamiento crece de 3.534 a 11.999 barras), `fx_compact` empeora ligeramente (`AUC 0,51566 → 0,51506`, `log-loss 0,69279 → 0,69307`).
- En los nuevos pares `USDCAD` y `NZDUSD` D1 (`20261016T210703_c9107a5c`), ambos modelos tienen log-loss peor que la prevalencia de entrenamiento en **los 3 folds y ambas semillas** (`0,704–0,727` vs `0,693–0,695`), y la receta ampliada `compact` empeora respecto de `price` tanto en `USDCAD` (`AUC 0,49787 → 0,46852`, `Acc 51,31% → 48,15%`) como en `NZDUSD` (`AUC 0,50641 → 0,49809`, retorno `−10,32% → −14,92%`).

### 4.4. Futuros con Vencimiento (Trimestrales BTC/ETH COIN-M y Mensuales VIX Cboe)
- Se adquirieron y depuraron **50 contratos trimestrales individuales BTCUSD/ETHUSD** (1.126.654 filas M15–D1, con exclusión auditada de 27.707 velas M15 post-liquidación) y **161 contratos mensuales VX** (28.696 filas D1).
- **Límite metodológico estricto:** estos contratos individuales tienen vencimiento finito y estructura temporal de basis; **no existía un modelo previo entrenado sobre ellos ni una regla predeclarada de rollover/margen causal**. Concatenar contratos sin ajuste de roll o mezclarlos con spot violaría la validez del experimento. Quedan validados en el inventario para investigación futura específica de estructura de plazos/carry, sin fabricar comparaciones antes/después inexistentes.

### 4.5. Índices Bursátiles de Contado (8 índices diarios, 82.707 filas depuradas)
- Se incorporaron y depuraron `SP500`, `NASDAQ_COMPOSITE`, `DOW_JONES`, `RUSSELL_2000`, `VIX`, `DAX`, `FTSE_100` y `NIKKEI_225` (excluyendo 21.050 filas históricas tempranas con OHLC sintético idéntico del proveedor).
- **Límite metodológico:** los índices de contado no son activos directamente negociables y cierran en husos horarios locales distintos de UTC 00:00. No tenían modelos previos en el repositorio; usarlos sin un contrato explícito de reloj de publicación (`available_at` por bolsa) introduciría fuga temporal entre cierres de EE. UU./Asia y Europa.

---

## 5. Respuestas directas a las 6 preguntas obligatorias

### 1. ¿Los modelos mejoraron con los nuevos datos?
**No (evidencia insuficiente para declarar mejora real; 0 mejoras estadísticamente significativas).**
Aunque varios modelos muestran mejoras numéricas puntuales al añadir historia (2017–2019 en cripto) o más activos/derivados (2021–2025), **aproximadamente la mitad de las comparaciones empeoran o se mantienen neutras**, ningún contraste supera el criterio estadístico predeclarado (ganancia de log-loss con bootstrap por bloques de 7 y 30 días + corrección de Holm-Bonferroni, `0/26` en extensión histórica y `0/5` en expansión multimercado), y en la ablación de expansión el control negativo con **ruido gaussiano causal (`expanded_noise`)** superó en ROC-AUC y log-loss a todas las variantes con datos nuevos reales.

### 2. ¿Cuánto mejoraron? (Diferencias absolutas y relativas)
- **En `BTCUSDT` D1 (9 clasificadores direccionales sobre 876 días comunes):**
  - **Accuracy:** cambio medio `+0,02 pp` (rango: de `−1,94 pp` [`−3,59%` rel.] en `logistic` a `+1,31 pp` [`+2,51%` rel.] en `regime_rules` y `+1,26 pp` [`+2,36%` rel.] en `random_forest_cross`).
  - **ROC-AUC:** cambio medio `+0,00433` (rango: de `−0,00793` [`−1,48%` rel.] en `specialists_rules` a `+0,01611` [`+3,10%` rel.] en `lightgbm_cross`, `+0,01574` [`+2,96%` rel.] en `catboost_cross` y `+0,01117` [`+2,06%` rel.] en `random_forest_cross`).
  - **Log-loss:** cambio medio `−0,00075` (mejora máxima en `lightgbm_cross`: `−0,00736` [`−1,05%` rel.]; deterioro máximo en `specialists_rules`: `+0,00383` [`+0,55%` rel.]).
- **En `ETHUSDT` D1 (9 clasificadores direccionales sobre 876 días comunes):**
  - **Accuracy:** cambio medio `+0,75 pp` (rango: de `−1,37 pp` [`−2,70%` rel.] en `lightgbm_cross` a `+2,51 pp` [`+5,10%` rel.] en `smc`, `+2,17 pp` [`+4,32%` rel.] en `catboost_cross` y `+2,17 pp` [`+4,24%` rel.] en `logistic`).
  - **ROC-AUC:** cambio medio `+0,00291` (rango: de `−0,02689` [`−5,11%` rel.] en `lightgbm_cross` a `+0,02231` [`+4,40%` rel.] en `structure` y `+0,02119` [`+4,18%` rel.] en `catboost_cross`).
  - **Log-loss:** cambio medio `−0,00065` (mejora máxima en `catboost_cross`: `−0,00424` [`−0,61%` rel.]; deterioro en `lightgbm_cross`: `+0,00468` [`+0,67%` rel.]).
- **En `EURUSD` D1 y H4:** cambios entre `−1,17 pp` y `+1,36 pp` en Accuracy, `−0,00119` y `+0,00445` en ROC-AUC, y `−0,00064` a `+0,00029` en log-loss (`< 0,1%` relativo).
- **En expansión multimercado (`anchor_price` → `expanded_derivatives` en XGBoost sobre BTC+ETH):** Accuracy `+2,92 pp` (`+5,93%` rel.), ROC-AUC `+0,01616` (`+3,18%` rel.), pero log-loss empeora `+0,00009` (`+0,01%` rel.).

### 3. ¿Cuál es ahora el modelo más prometedor?
Separando estrictamente capacidad predictiva direccional y desempeño económico en backtest:

1. **En capacidad predictiva direccional (ranking y calibración probabilística):**
   - **Modelo:** `random_forest_cross` reentrenado (brazo B, con historia 2017–2025 y features locales + cross-asset + flujo real).
   - **Mercado y temporalidad:** `BTCUSDT` spot, temporalidad `1d` (`D1`).
   - **Período evaluado:** 3 folds walk-forward entre `2023-01-01` y `2025-06-18` (`876` observaciones diarias, media de semillas `42` y `1729`).
   - **Métricas antes → después:**
     - `Accuracy`: `53,14%` → **`54,39%`** (`+1,26 pp`, `+2,36%` rel.)
     - `Balanced Accuracy`: `53,03%` → **`54,14%`** (`+1,11 pp`)
     - `ROC-AUC`: `0,54202` → **`0,55319`** (`+0,01117`, `+2,06%` rel.)
     - `Log-loss`: `0,68949` → **`0,68788`** (`−0,00161`, mejor log-loss direccional de todos los modelos evaluados)
     - `Brier Score`: `0,24820` → **`0,24740`** (`−0,00080`)
     - `Precision / Recall / F1`: `54,29% / 56,76% / 0,55501` → **`55,02% / 62,64% / 0,58580`**
     - `ECE (10 bins)`: `0,01201` → **`0,00922`**
   - **Advertencia crítica:** pese a tener el mejor AUC y el menor log-loss direccional, con la regla operativa fija de umbral `0,55` y costos de `16 bps/lado` genera **retorno neto negativo (`−7,08%`, Sharpe `0,006`, Profit Factor `0,960`, 208 operaciones)** porque su exceso de rotación (`6.007,64 USD` en costos sobre `10.000 USD` iniciales) absorbe la débil ventaja direccional.
2. **En resultado económico retrospectivo (backtest neto después de costos):**
   - **Modelo:** `logistic` reentrenado (brazo B) en `BTCUSDT` `1d` (`2023-01-01..2025-06-18`): retorno neto **`+109,40%`** (vs `+45,52%` en A), Sharpe **`1,491`** (vs `0,905`), Max Drawdown **`−12,08%`**, Profit Factor **`1,649`**, `146` operaciones (`+31,24%` bajo costos dobles). Sin embargo, su predicción direccional en umbral `0,50` **empeoró** (`Accuracy 54,00% → 52,05%`, `AUC 0,54396 → 0,54001`, `log-loss 0,68925 → 0,68955`); el mayor retorno provino de operar con menor exposición (`23,77%`) en el umbral superior `0,55`, y el IC95 de la diferencia de retorno frente a A incluye cero.
   - **En cartera cripto de baja rotación:** `LSTM / DMN` (`lstm_sharpe_net` / `expanded_flow`) mantiene Sharpe neto **`1,035–1,433`** y drawdown contenido (`−6,6% a −7,1%`) gracias a un turnover más de 10 veces menor que los árboles, pero **no supera al baseline pasivo escalado por volatilidad (`long_scaled`: retorno `+38,05%` a `+63,51%`, Sharpe `1,149`)**.

### 4. ¿La mejora es estadísticamente significativa?
**No.**
- **Mejora numérica:** existe en ~13 de las 26 comparaciones de extensión histórica (por ejemplo, `catboost_cross` en ETH mejora en las 6/6 combinaciones fold×semilla con `Δ log-loss = −0,00424`, y `lightgbm_cross` en BTC mejora en 6/6 con `Δ log-loss = −0,00736`).
- **Mejora respaldada estadísticamente:** **ninguna (`0/26` en extensión histórica y `0/9` en estudios de expansión)**. Todos los intervalos de confianza al 95% de bootstrap por bloques móviles (bloques de 7 y 30 días) para los modelos direccionales cruzan cero, y todos los p-valores ajustados por Holm-Bonferroni son `p_Holm >= 0,6897` (`1,0000` en las 26 comparaciones principales).
- **Mejora robusta fuera de muestra:** **no existe**. La dirección del cambio se invierte entre activos (por ejemplo, `lightgbm_cross` mejora en BTC y se desploma en ETH; `logistic` mejora predicción en ETH y empeora en BTC) y ninguna reserva prospectiva sellada ha sido abierta.

### 5. ¿Hay alguna ventaja predictiva aprovechable?
**No hay evidencia validada de ventaja aprovechable en producción.**
- **Frente a controles predictivos (`constant_05`, `train_prevalence`, `logistic`):** en `BTCUSDT` D1, cinco recetas B (`random_forest_cross`, `catboost_cross`, `logistic`, `structure`, `smc`) logran un log-loss puntual ligeramente inferior a `0,69315` (`0,68788` a `0,69021`), pero la diferencia máxima frente a `constant_05` es de apenas `0,00527` nats y **ninguna es estadísticamente significativa** (`p_Holm = 1,0000`, todos los IC95 cruzan cero). En `ETHUSDT` D1 sólo 2 de 9 recetas direccionales bajan de `0,69315` (`logistic` `0,69156` y `catboost_cross` `0,69240`), y en `EURUSD`, `USDCAD` y `NZDUSD` **casi todas las recetas tienen log-loss peor que predecir siempre `p = 0,5`**.
- **En rentabilidad neta:** de las 26 configuraciones tabulares/eventos reentrenadas, **20 pierden dinero o quedan en 0% neto** después de costos estándar (`16 bps/lado` en spot) y **25 de 26 pierden o quedan en 0% bajo costos dobles** (la única excepción puntual es `logistic` en BTC D1). En DMN/LSTM, los retornos netos son positivos (`+10,66%` a `+21,36%`), pero inferiores a mantener una posición larga escalada por volatilidad (`long_scaled`) o una regla simple `TSMOM` sin red neuronal.

### 6. ¿Qué conviene hacer después?
1. **Detener la ampliación bruta de historia OHLCV para los mismos clasificadores diarios de dirección (`close-to-close` a 1 barra):** quedó demostrado empíricamente que triplicar la ventana histórica de entrenamiento (añadiendo 2012–2019 en Forex o 2017–2019 en cripto) o concatenar más pares al panel no reduce significativamente el log-loss y a menudo introduce regímenes antiguos no estacionarios que degradan la calibración en 2023–2025.
2. **No promover ningún modelo actual a producción ni abrir las reservas prospectivas selladas (`2025-07-01..2026-10-01` en cuarentena y reservas `2027`):** gastar el conjunto sellado cuando ningún endpoint superó Holm en desarrollo invalidaría la reserva sin justificación científica.
3. **Redirigir la investigación hacia tres cambios estructurales concretos (bajo un nuevo protocolo predeclarado):**
   - **Alinear el objetivo de entrenamiento con la fricción de ejecución:** los clasificadores tabulares pierden dinero incluso con `AUC ~ 0,55` porque rotan entre 150 y 345 veces en 2,5 años. Cualquier modelo futuro debe optimizar utilidad neta de costos o incorporar penalización explícita de cambio de posición (`turnover penalty` / umbral de no-operación con histéresis calibrado en validación).
   - **Explotar estructura de plazos y carry en los nuevos datasets de futuros con vencimiento (`BTCUSD/ETHUSD` trimestrales y `VX` mensuales):** en lugar de predecir dirección diaria del spot con OHLCV rezagado, construir un protocolo específico de **basis / roll yield anualizado y estructura temporal de volatilidad (`VIX` spot vs futuros `VX`)**, donde la señal económica es estructural (prima por plazo y carry) y varía más lentamente que el ruido diario.
   - **Horizontes multi-día (5 a 20 días) con regularización fuerte (`L2` / modelos lineales penalizados):** en todos los experimentos, modelos de baja capacidad (`logistic`, `long_scaled`, `tsmom`) sufrieron mucho menos sobreajuste y menor rotación que GBDTs profundos o mezclas de expertos.

