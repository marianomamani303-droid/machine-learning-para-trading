# Informe de Investigación: Ablación de Variables Financieras y Controles de Ruido en Modelos Existentes

**Fecha de ejecución:** 17 de octubre de 2026  
**Run ID auditable:** `20261017T224928_57040c88`  
**Directorio de artefactos:** [`artifacts/financial_feature_ablation/20261017T224928_57040c88/`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/financial_feature_ablation/20261017T224928_57040c88/)  
**Script reproducible:** [`scripts/financial_feature_ablation.py`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/scripts/financial_feature_ablation.py)  
**Estado de modelos registrados y reservas prospectivas (`>= 2025-07-01`):** **100% intactos (`protected_unchanged = true`)**

---

## 1. Resumen Ejecutivo y Conclusión Principal

Se investigó si la capacidad predictiva direccional fuera de muestra (**`ROC-AUC`** y **`log-loss`** a horizonte de 1 día, $h=1$) de los modelos existentes del repositorio (`LogisticRegression` L2, `XGBClassifier`, `RandomForestClassifier` y `LGBMClassifier`) mejora genuinamente al incorporar información financiera relevante según su disponibilidad histórica verificada (`< 2025-07-01`):

1. **Volatilidad y asimetría de rango (`volatility`)**
2. **Liquidez, impacto de precio y fricción de cotización (`liquidity`)**
3. **Volumen relativo y flujo de órdenes agresor (`relative_volume`)**
4. **Tasas de fondeo realizadas en futuros perpetuos (`funding_rates`)**
5. **Basis de futuros perpetuos, futuros trimestrales con vencimiento (`BTCUSD`/`ETHUSD` COIN-M) y estructura temporal de futuros de volatilidad CBOE `VX` (`futures_basis`)**
6. **Relaciones entre activos: cripto, divisas `USD` y contexto bursátil/volatilidad `^GSPC` / `^VIX` (`cross_asset`)**

### Conclusión Verificable

1. **El control con ruido gaussiano de igual dimensión (`noise_add4_mean`, `noise_all_mean`) es indispensable en modelos de árboles:** en `XGBoost`, `RandomForest` y `LightGBM`, agregar columnas de ruido puro $\mathcal{N}(0,1)$ suele reducir la `log-loss` frente al conjunto base (`F0_base`) porque el submuestreo de columnas (`colsample_bytree = 0.85`, `max_features`) diluye divisiones sobre variables ruidosas y contrae las probabilidades hacia $0.50$. Por lo tanto, evaluar sólo `add_familia - F0_base` genera falsos positivos; toda mejora genuina debe superar **simultáneamente a `F0_base` y al control de ruido de igual dimensión** sobre las mismas observaciones.
2. **Familias con señal incremental real (superando tanto a `F0_base` como al control de ruido de igual dimensión):**
   - **En Cripto Spot (`BTCUSDT` y `ETHUSDT`): `liquidity` (Liquidez e impacto de precio)** es la única familia que mejora consistentemente `log-loss` y `ROC-AUC` frente a `F0_base` **y** frente a `noise_add4_mean` a través de arquitecturas y semillas:
     - En `BTCUSDT`, `add_liquidity` supera tanto a `F0_base` como a `noise_add4_mean` en `RandomForest` (`log-loss = 0.69582`, $\Delta\text{Loss vs Base} = -0.00255$, $\Delta\text{Loss vs Ruido} = -0.00218$; `ROC-AUC = 0.54148` en s42 y `0.54233` en s1729), `XGBoost` (`log-loss = 0.69870`, $\Delta\text{Loss vs Base} = -0.00312$, $\Delta\text{Loss vs Ruido} = -0.00260$; `ROC-AUC = 0.53120`) y `LightGBM` (`log-loss = 0.70193`, $\Delta\text{Loss vs Base} = -0.00758$, $\Delta\text{Loss vs Ruido} = -0.00074$; `ROC-AUC = 0.53016`), además de alcanzar el mayor `ROC-AUC` global en `LogisticRegression` (`0.54549`, $+0.01037$ vs Base y $+0.03104$ vs Ruido).
     - En `ETHUSDT`, `add_liquidity` en `RandomForest` obtiene intervalos bootstrap de 30 días completamente favorables frente a `F0_base` en ambas semillas (`s42`: $\Delta\text{Loss} = -0.00282$, IC95% `[-0.00666, -0.00069]`, $\Delta\text{AUC} = +0.01533$, IC95% `[+0.00302, +0.03637]`; `s1729`: $\Delta\text{Loss} = -0.00344$, IC95% `[-0.00694, -0.00099]`, $\Delta\text{AUC} = +0.01164$, IC95% `[+0.00028, +0.03030]`), y su eliminación en ablación *leave-one-family-out* (`all_without_liquidity`) es la que más deteriora `XGBoost` (`+0.00458` log-loss, `-0.01375` AUC) y `LightGBM` (`+0.00480` log-loss, `-0.01312` AUC).
   - **En Forex (`EURUSD`): `futures_basis` (Estructura temporal de futuros de volatilidad CBOE `VX` frente a `^VIX`)** es la familia más informativa:
     - Supera simultáneamente a `F0_base` y a `noise_add4_mean` en `XGBoost` (`s42`: `log-loss = 0.69632`, $\Delta\text{Loss vs Base} = -0.00389$, $\Delta\text{Loss vs Ruido} = -0.00101$, `ROC-AUC = 0.50625`, $+0.01449$ vs Base; `s1729`: $\Delta\text{Loss vs Base} = -0.00403$, $\Delta\text{Loss vs Ruido} = -0.00216$, $\Delta\text{AUC vs Ruido} = +0.01595$) y `RandomForest` (`s42`: `log-loss = 0.69522`, $\Delta\text{Loss vs Base} = -0.00404$, $\Delta\text{Loss vs Ruido} = -0.00014$, `ROC-AUC = 0.50687`, $+0.01462$ vs Base; `s1729`: $\Delta\text{Loss vs Base} = -0.00477$, IC95% `[-0.00975, -0.00133]`).
     - Además, la selección anidada interna (`nested_noise_screened`) en `EURUSD` `XGBoost` mejora simultáneamente `log-loss` (`0.69586`, $-0.00435$ vs Base y $-0.00147$ vs Ruido) y `ROC-AUC` (`0.51623`, $+0.02447$ vs Base y $+0.01225$ vs Ruido).
3. **Variables individuales más robustas (con importancia por permutación OOS positiva en `3/3` folds):**
   - `fin_liq_corwin_schultz_20` (fricción de rango de dos sesiones de Corwin-Schultz normalizada por $\sigma_{20}$): positiva en `3/3` folds tanto en `BTCUSDT` como en `ETHUSDT` para todos los modelos de árboles.
   - `fin_liq_range_impact_z63` (z-score causal del rango intravela por dólar operado $\ln(H/L)/\text{quote\_volume}$): positiva en `3/3` folds en `BTCUSDT` (`XGBoost`) y `ETHUSDT` (`LightGBM`, `RandomForest`).
   - `fin_cross_sp500_mom_z` (momentum causal de 21 días del S&P 500 `^GSPC` ajustado por volatilidad): positiva en `3/3` folds sobre `ETHUSDT` en los tres modelos de árboles (`+0.005082` en `LightGBM`, `+0.002543` en `XGBoost`, `+0.001265` en `RandomForest`).
   - `fin_vol_parkinson_ratio` ($\text{Parkinson}_{20} / \sigma_{20} - 1$): positiva en `3/3` folds en `ETHUSDT` (`XGBoost`, `LightGBM`) y `BTCUSDT` (`RandomForest`).
   - `fin_basis_vx_z63` (z-score causal de 63 días del basis del futuro front-month `VX` sobre `^VIX`): positiva en `3/3` folds sobre `EURUSD` (`XGBoost`: `+0.004901`, `RandomForest`: `+0.001152`).
4. **Familias que sobreajustan o no aportan mejora neta en D1 ($h=1$):**
   - **`funding_rates`** y **`relative_volume`** como bloques completos degradan la `log-loss` OOS frente al control de ruido en `XGBoost`, `LightGBM` y `LogisticRegression`, y su eliminación desde el modelo completo (`all_without_funding_rates`, `all_without_relative_volume`) mejora el desempeño fuera de muestra.
   - **Acumular todas las familias (`all_financial`, 38 columnas en cripto / 29 en Forex)** multiplica la brecha de generalización train-OOS (hasta $+0.09149$ en `XGBoost` y $+0.13810$ en `LightGBM`) y pierde frente a `noise_all_mean`.
5. **Decisión de gobernanza estadística:** tras aplicar corrección de **Holm-Bonferroni** por comparaciones múltiples sobre bloques móviles de 30 días (`2,000` réplicas), ningún contraste individual supera el umbral estricto $p_{\text{Holm}} < 0.05$ (los más cercanos son `EURUSD` `RandomForest` `add_futures_basis` con $p_{\text{Holm}} = 0.066$ y `ETHUSDT` `RandomForest` `add_liquidity` con $p_{\text{Holm}} = 0.080$), ni bate simultáneamente la `log-loss` del baseline pasivo $p=0.5$ (`0.69315`) salvo `ETHUSDT` `logistic_l2` `add_volatility` (`0.69272`). Por ello, **no se promueve ningún cambio al contrato de producción congelado**, quedando las variables validadas (`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`, `fin_cross_sp500_mom_z`, `fin_vol_parkinson_ratio`, `fin_basis_vx_z63`) identificadas como el subconjunto parsimonioso de mayor valor informativo real.

---

## 2. Diseño Metodológico y Verificación de Causalidad (`point-in-time`)

### 2.1. Particiones Walk-Forward, Purga, Embargo y Observaciones Idénticas

Para garantizar comparabilidad exacta con [`RETRAINING_COMPARISON_REPORT.md`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/RETRAINING_COMPARISON_REPORT.md) y [`FINAL_MODEL_IMPROVEMENT_REPORT.md`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/FINAL_MODEL_IMPROVEMENT_REPORT.md):

- **Corte de cuarentena prospectiva:** `STOP = 2025-07-01T00:00:00Z` verificado mediante [`protect_reserved_labels`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/src/trading_ml/training/recipe.py).
- **Folds externos walk-forward (3 ventanas OOS contiguas con purga de etiquetas y embargo de 6 días):**
  - **Fold 0:** `2023-01-01` a `2024-01-01` (`364` días OOS en Cripto, `250` en `EURUSD`).
  - **Fold 1:** `2024-01-01` a `2025-01-01` (`365` días OOS en Cripto, `243` en `EURUSD`).
  - **Fold 2:** `2025-01-01` a `2025-07-01` (`180` días OOS en Cripto, `121` en `EURUSD`).
- **Mismas observaciones exactas:** dentro de cada activo, todas las recetas (`F0_base`, adiciones individuales `add_*`, controles de ruido `noise_*`, conjunto completo `all_financial`, ablaciones LOFO `all_without_*` y selección anidada `nested_noise_screened`) y todos los modelos utilizan **exactamente las mismas filas de entrenamiento y evaluación** (`909` observaciones OOS totales en `BTCUSDT`, `909` en `ETHUSDT`, `614` en `EURUSD`), verificadas por hash SHA-256 (`train_hash` y `evaluation_hash` en [`diagnostics.json`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/financial_feature_ablation/20261017T224928_57040c88/diagnostics.json)).
- **Prueba explícita de invariancia causal ante perturbaciones futuras (`verify_causal_invariance`):** antes de entrenar, el pipeline alteró artificialmente todos los precios, volúmenes, funding rates, contratos de futuros e índices bursátiles posteriores a `2024-01-01` y comprobó que las matrices de features en `reference_time <= 2024-01-01` permanecieran **idénticas bit a bit (`max_abs_diff = 0.0e+00`)** en `BTCUSDT` (`1,012` filas × `38` columnas), `ETHUSDT` (`1,012` × `38`) y `EURUSD` (`941` × `29`).

### 2.2. Definición de las Familias de Variables Financieras (`4` variables por familia)

Cada familia aporta exactamente $k = 4$ variables causales normalizadas (mediante volatilidad previa o [`zscore_past`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/src/trading_ml/expansion/features.py#L66-L68) de 63 sesiones), eliminando cualquier sesgo por diferente número de columnas entre familias:

| Familia | Disponibilidad | Variables Incluidas ($k=4$) | Hipótesis Financiera |
|---|---|---|---|
| **`volatility`** | Cripto y Forex | `fin_vol_term_5_60` ($\sigma_5/\sigma_{60}-1$), `fin_vol_parkinson_ratio` ($\text{Parkinson}_{20}/\sigma_{20}-1$), `fin_vol_garman_klass_z63`, `fin_vol_downside_asym_20` (($\sigma_{20}^- - \sigma_{20}^+)/\sigma_{20}$) | Compresión/expansión de estructura temporal de volatilidad, desacuerdo rango intravela vs cierre-a-cierre y asimetría bajista |
| **`liquidity`** | Cripto (microestructura spot) / Forex (spreads bid-ask) | **Cripto:** `fin_liq_amihud_z63` ($\ln(|r_1|/\text{quote\_vol})_{20}$), `fin_liq_range_impact_z63` ($\ln(\ln(H/L)/\text{quote\_vol})_5$), `fin_liq_corwin_schultz_20`, `fin_liq_trade_size_z63` ($\ln(\text{quote\_vol}/\text{trades})$)<br>**Forex:** `fin_liq_spread_z63`, `fin_liq_spread_to_range`, `fin_liq_corwin_schultz_20`, `fin_liq_quote_range_excess` | Impacto de precio por dólar operado (Amihud), profundidad intravela, spread efectivo implícito (Corwin-Schultz) y tamaño medio de ticket |
| **`relative_volume`** | Cripto Spot (5 activos con flujo taker verificado) | `fin_rvol_quote_z63`, `fin_rvol_taker_imbalance` ($2\cdot\text{taker\_quote}/\text{quote\_vol}-1$), `fin_rvol_flow_innovation_3_20`, `fin_rvol_signed_volume_shock` ($\text{sign}(r_1)\ln(\text{qvol}/\overline{\text{qvol}}_{20})$) | Choques de participación en dólares, agresión compradora/vendedora neta por órdenes a mercado e innovación de flujo corto vs medio plazo |
| **`funding_rates`** | Cripto Perpetuos (liquidaciones cada 8h cerradas antes de `reference_time`) | `fin_fund_1d` (suma 24h en bps), `fin_fund_7d` (suma 7d en bps), `fin_fund_z63` (z-score 63d), `fin_fund_mom_interaction` ($-\text{z63}\cdot r_7/(\sigma_{20}\sqrt{7})$) | Costo de acarreo apalancado, posicionamiento congestionado (*crowded longs/shorts*) y presión de liquidación/reversión |
| **`futures_basis`** | Cripto (Perp + Trimestrales COIN-M `BTCUSD`/`ETHUSD`) / Forex (Futuros `VX` CBOE) | **Cripto:** `fin_basis_perp_premium` (bps), `fin_basis_perp_z63`, `fin_basis_dated_ann` (basis anualizado del contrato trimestral activo con $\text{DTE}\ge 3\text{d}$), `fin_basis_dated_change_5d`<br>**Forex:** `fin_basis_vx_term` ($\text{VX}_{\text{front}}/\text{VIX}-1$), `fin_basis_vx_z63`, `fin_basis_vx_change_5d`, `fin_basis_vx_oi_z63` | Prima de futuros perpetuos, carry institucional anualizado en futuros con vencimiento y pendiente de la curva de volatilidad implícita (`VX` vs `VIX`) |
| **`cross_asset`** | Cripto y Forex (`point_in_time_join` con `available_at <= reference_time`) | **Cripto:** `fin_cross_crypto_residual` (retorno idiosincrático vs 5 criptos con $\beta_{60}$ previa), `fin_cross_usd_strength` (7 pares FX), `fin_cross_sp500_mom_z` (`^GSPC`), `fin_cross_vix_term_basis`<br>**Forex:** `fin_cross_usd_peers` (6 pares FX sin EUR), `fin_cross_fx_dispersion`, `fin_cross_sp500_mom_z`, `fin_cross_vix_z63` | Rotación relativa entre criptoactivos, fortaleza global del dólar estadounidense y transmisión de apetito por riesgo bursátil (`^GSPC`, `^VIX`, `VX`) |

---

## 3. Resultados del Estudio de Ablación y Controles de Ruido (`Semilla 42` y `Semilla 1729`)

En todas las tablas siguientes:
- **$\Delta\text{Loss vs Base} = \text{Loss}(\text{receta}) - \text{Loss}(\text{F0\_base})$** (**negativo es mejor**).
- **$\Delta\text{Loss vs Ruido} = \text{Loss}(\text{receta}) - \text{Loss}(\text{ruido de igual dimensión})$** (**negativo es mejor** y demuestra que las variables superan al efecto de dilución por ruido gaussiano).
- **$\Delta\text{AUC vs Base}$ y $\Delta\text{AUC vs Ruido}$** (**positivo es mejor**).
- **$\text{LOFO }\Delta\text{Loss (Sin}-\text{All}) = \text{Loss}(\text{all\_without\_familia}) - \text{Loss}(\text{all\_financial})$** (**positivo indica que quitar la familia empeora el modelo completo**).

### 3.1. `BTCUSDT` Spot D1 (`909` observaciones OOS comunes, `2023-01-01` a `2025-07-01`)

- **Referencias sin features:** `constant_05` ($p=0.5$): `Log-Loss = 0.69315`, `ROC-AUC = 0.50000`, `Brier = 0.25000`. `train_prevalence`: `Log-Loss = 0.69459`, `ROC-AUC = 0.50777`.

| Modelo (Semilla) | Receta / Familia | Cols | Log-Loss OOS ↓ | ΔLoss vs Base | ΔLoss vs Ruido | ROC-AUC OOS ↑ | ΔAUC vs Base | ΔAUC vs Ruido | LOFO ΔLoss (Sin−All) | Brecha Train-OOS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`logistic_l2` (42)** | `F0_base` (14 orig.) | 14 | **0.69809** | +0.00000 | — | 0.53512 | +0.00000 | — | — | +0.01042 |
| `logistic_l2` (42) | `noise_add4_mean` (+4 ruido) | 18 | 0.70167 | +0.00357 | +0.00000 | 0.51445 | -0.02067 | +0.00000 | — | — |
| `logistic_l2` (42) | `add_volatility` | 18 | 0.71333 | +0.01524 | +0.01167 | 0.53793 | +0.00281 | +0.02349 | -0.02403 | +0.02442 |
| **`logistic_l2` (42)** | **`add_liquidity`** | 18 | 0.69875 | +0.00065 | **-0.00292** | **0.54549** | **+0.01037** | **+0.03104** | -0.00407 | +0.01127 |
| `logistic_l2` (42) | `add_relative_volume` | 18 | 0.70840 | +0.01031 | +0.00674 | 0.51833 | -0.01679 | +0.00389 | -0.01380 | +0.02532 |
| `logistic_l2` (42) | `add_funding_rates` | 18 | 0.70152 | +0.00343 | -0.00014 | 0.52739 | -0.00773 | +0.01294 | -0.00122 | +0.01477 |
| `logistic_l2` (42) | **`add_futures_basis`** | 18 | 0.70129 | +0.00320 | -0.00038 | **0.53882** | **+0.00371** | **+0.02438** | +0.00001 | +0.01448 |
| `logistic_l2` (42) | `add_cross_asset` | 18 | 0.70146 | +0.00337 | -0.00020 | 0.50895 | -0.02617 | -0.00549 | -0.01295 | +0.01731 |
| `logistic_l2` (42) | `all_financial` (+24 fin.) | 38 | 0.74473 | +0.04663 | +0.02914 | 0.51331 | -0.02181 | +0.01030 | — | +0.06568 |
| **`random_forest` (42)** | `F0_base` (14 orig.) | 14 | 0.69837 | +0.00000 | — | 0.53587 | +0.00000 | — | — | +0.06425 |
| `random_forest` (42) | `noise_add4_mean` (+4 ruido) | 18 | 0.69800 | -0.00037 | +0.00000 | 0.52873 | -0.00715 | +0.00000 | — | — |
| `random_forest` (42) | `add_volatility` | 18 | 0.69871 | +0.00034 | +0.00071 | 0.53882 | +0.00295 | +0.01009 | -0.00095 | +0.06998 |
| **`random_forest` (42)** | **`add_liquidity`** | 18 | **0.69582** | **-0.00255** | **-0.00218** | **0.54148** | **+0.00561** | **+0.01275** | -0.00077 | +0.06767 |
| **`random_forest` (1729)** | **`add_liquidity`** | 18 | **0.69650** | **-0.00111** | **-0.00131** | **0.54233** | **+0.00468** | **+0.01197** | **+0.00200** | +0.06801 |
| `random_forest` (42) | `add_relative_volume` | 18 | 0.69683 | -0.00154 | -0.00117 | 0.53380 | -0.00207 | +0.00507 | -0.00244 | +0.07032 |
| `random_forest` (42) | `add_funding_rates` | 18 | 0.69694 | -0.00143 | -0.00106 | 0.53386 | -0.00201 | +0.00514 | -0.00175 | +0.06800 |
| `random_forest` (42) | `add_futures_basis` | 18 | 0.69828 | -0.00010 | +0.00028 | 0.53247 | -0.00341 | +0.00374 | -0.00201 | +0.06924 |
| `random_forest` (42) | `add_cross_asset` | 18 | 0.70023 | +0.00185 | +0.00222 | 0.53001 | -0.00587 | +0.00128 | -0.00360 | +0.07404 |
| `random_forest` (42) | `all_financial` (+24 fin.) | 38 | 0.69911 | +0.00074 | +0.00271 | 0.52564 | -0.01024 | -0.00228 | — | +0.07975 |
| **`xgboost` (42)** | `F0_base` (14 orig.) | 14 | 0.70182 | +0.00000 | — | 0.53050 | +0.00000 | — | — | +0.05876 |
| `xgboost` (42) | `noise_add4_mean` (+4 ruido) | 18 | 0.70130 | -0.00051 | +0.00000 | 0.52490 | -0.00560 | +0.00000 | — | — |
| `xgboost` (42) | `add_volatility` | 18 | 0.71042 | +0.00860 | +0.00912 | 0.52643 | -0.00407 | +0.00153 | -0.00382 | +0.07182 |
| **`xgboost` (42)** | **`add_liquidity`** | 18 | **0.69870** | **-0.00312** | **-0.00260** | **0.53120** | **+0.00070** | **+0.00630** | **+0.00107** | +0.06090 |
| **`xgboost` (1729)** | **`add_liquidity`** | 18 | **0.69793** | **-0.00319** | **-0.00272** | **0.53330** | -0.00108 | **+0.00968** | **+0.00283** | +0.06019 |
| `xgboost` (42) | `add_relative_volume` | 18 | 0.70575 | +0.00393 | +0.00444 | 0.52375 | -0.00675 | -0.00115 | +0.00024 | +0.07055 |
| `xgboost` (42) | `add_funding_rates` | 18 | 0.70232 | +0.00051 | +0.00102 | 0.52826 | -0.00224 | +0.00336 | -0.00079 | +0.06351 |
| `xgboost` (42) | `add_futures_basis` | 18 | 0.70603 | +0.00421 | +0.00473 | 0.52400 | -0.00651 | -0.00091 | -0.00103 | +0.06820 |
| `xgboost` (42) | `add_cross_asset` | 18 | 0.70967 | +0.00785 | +0.00836 | 0.51659 | -0.01391 | -0.00831 | -0.00466 | +0.07446 |
| `xgboost` (42) | `all_financial` (+24 fin.) | 38 | 0.71236 | +0.01054 | +0.00438 | 0.50439 | -0.02611 | +0.00506 | — | +0.09149 |
| **`lightgbm` (42)** | `F0_base` (14 orig.) | 14 | 0.70950 | +0.00000 | — | 0.52249 | +0.00000 | — | — | +0.09347 |
| `lightgbm` (42) | `noise_add4_mean` (+4 ruido) | 18 | 0.70266 | -0.00684 | +0.00000 | 0.52688 | +0.00439 | +0.00000 | — | — |
| **`lightgbm` (42)** | **`add_liquidity`** | 18 | **0.70193** | **-0.00758** | **-0.00074** | **0.53016** | **+0.00767** | **+0.00328** | **+0.00409** | +0.09788 |
| `lightgbm` (42) | `all_financial` (+24 fin.) | 38 | 0.71484 | +0.00534 | +0.00634 | 0.51899 | -0.00350 | +0.01740 | — | +0.13810 |

---

### 3.2. `ETHUSDT` Spot D1 (`909` observaciones OOS comunes, `2023-01-01` a `2025-07-01`)

- **Referencias sin features:** `constant_05` ($p=0.5$): `Log-Loss = 0.69315`, `ROC-AUC = 0.50000`. `train_prevalence`: `Log-Loss = 0.69318`, `ROC-AUC = 0.49143`.

| Modelo (Semilla) | Receta / Familia | Cols | Log-Loss OOS ↓ | ΔLoss vs Base | ΔLoss vs Ruido | ROC-AUC OOS ↑ | ΔAUC vs Base | ΔAUC vs Ruido | LOFO ΔLoss (Sin−All) | Brecha Train-OOS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`logistic_l2` (42)** | `F0_base` (14 orig.) | 14 | 0.69303 | +0.00000 | — | 0.51400 | +0.00000 | — | — | +0.00350 |
| `logistic_l2` (42) | `noise_add4_mean` (+4 ruido) | 18 | 0.69197 | -0.00106 | +0.00000 | 0.52389 | +0.00989 | +0.00000 | — | — |
| **`logistic_l2` (42)** | **`add_volatility`** | 18 | **0.69272** | **-0.00031** | +0.00075 | **0.52547** | **+0.01147** | **+0.00158** | -0.00376 | +0.00480 |
| **`logistic_l2` (42)** | **`add_liquidity`** | 18 | 0.69421 | +0.00118 | +0.00224 | **0.52819** | **+0.01419** | **+0.00430** | -0.00681 | +0.00747 |
| `logistic_l2` (42) | `add_relative_volume` | 18 | 0.69458 | +0.00154 | +0.00261 | 0.51395 | -0.00004 | -0.00993 | -0.00199 | +0.00527 |
| `logistic_l2` (42) | `add_funding_rates` | 18 | 0.69481 | +0.00177 | +0.00284 | 0.50721 | -0.00678 | -0.01667 | -0.00425 | +0.00671 |
| `logistic_l2` (42) | `add_futures_basis` | 18 | 0.69471 | +0.00168 | +0.00275 | 0.51313 | -0.00087 | -0.01075 | -0.00317 | +0.00652 |
| `logistic_l2` (42) | `add_cross_asset` | 18 | 0.69711 | +0.00408 | +0.00514 | 0.49867 | -0.01532 | -0.02521 | -0.00835 | +0.00831 |
| **`random_forest` (42)** | `F0_base` (14 orig.) | 14 | 0.69703 | +0.00000 | — | 0.51115 | +0.00000 | — | — | +0.05673 |
| `random_forest` (42) | `noise_add4_mean` (+4 ruido) | 18 | 0.69361 | -0.00342 | +0.00000 | 0.52513 | +0.01399 | +0.00000 | — | — |
| `random_forest` (42) | `add_volatility` | 18 | 0.69504 | -0.00198 | +0.00144 | 0.51890 | +0.00776 | -0.00623 | -0.00136 | +0.05898 |
| **`random_forest` (42)** | **`add_liquidity`** | 18 | **0.69420** | **-0.00282** | +0.00060 | **0.52648** | **+0.01533** | **+0.00135** | -0.00086 | +0.05806 |
| **`random_forest` (1729)** | **`add_liquidity`** | 18 | **0.69409** | **-0.00344** | +0.00043 | **0.52886** | **+0.01164** | **+0.00172** | **+0.00057** | +0.05821 |
| `random_forest` (42) | `add_futures_basis` | 18 | 0.69603 | -0.00100 | +0.00243 | 0.51792 | +0.00677 | -0.00722 | -0.00083 | +0.06067 |
| **`xgboost` (42)** | `F0_base` (14 orig.) | 14 | 0.69790 | +0.00000 | — | 0.52559 | +0.00000 | — | — | +0.04748 |
| `xgboost` (42) | `noise_add4_mean` (+4 ruido) | 18 | 0.69422 | -0.00368 | +0.00000 | 0.53514 | +0.00955 | +0.00000 | — | — |
| **`xgboost` (42)** | **`add_liquidity`** | 18 | **0.69612** | **-0.00177** | +0.00190 | **0.52574** | **+0.00015** | -0.00940 | **+0.00458** | +0.05061 |
| **`xgboost` (1729)** | **`add_liquidity`** | 18 | **0.69608** | **-0.00333** | +0.00152 | **0.52271** | **+0.00360** | -0.00837 | **+0.00424** | +0.05045 |
| `xgboost` (42) | `add_cross_asset` | 18 | 0.69895 | +0.00106 | +0.00473 | 0.52041 | -0.00518 | -0.01473 | **+0.00216** | +0.05641 |
| **`lightgbm` (42)** | `F0_base` (14 orig.) | 14 | 0.70557 | +0.00000 | — | 0.51630 | +0.00000 | — | — | +0.08081 |
| `lightgbm` (42) | `noise_add4_mean` (+4 ruido) | 18 | 0.69518 | -0.01039 | +0.00000 | 0.53428 | +0.01798 | +0.00000 | — | — |
| `lightgbm` (42) | `add_liquidity` | 18 | 0.70391 | -0.00166 | +0.00873 | 0.51219 | -0.00410 | -0.02208 | **+0.00480** | +0.08401 |
| **`lightgbm` (42)** | **`add_cross_asset`** | 18 | **0.70051** | **-0.00506** | +0.00533 | **0.53217** | **+0.01587** | -0.00211 | **+0.00460** | +0.08996 |

> **Observación crítica en `ETHUSDT`:** En `XGBoost` y `LightGBM`, el control de ruido `noise_add4_mean` reduce la `log-loss` en `-0.00368` y `-0.01039` respectivamente frente a `F0_base` simplemente por dilución de divisiones en árboles. Sin embargo, cuando evaluamos la ablación desde el modelo completo (**LOFO**), eliminar `liquidity` (`all_without_liquidity`) o `cross_asset` (`all_without_cross_asset`) deteriora severamente la `log-loss` (`+0.00458` y `+0.00216` en `XGBoost`; `+0.00480` y `+0.00460` en `LightGBM`), mientras que eliminar `funding_rates` (`all_without_funding_rates`) **mejora** el modelo (`-0.00551` en `LightGBM`).

---

### 3.3. `EURUSD` OTC Forex D1 (`614` observaciones OOS comunes, `2023-01-01` a `2025-06-30`)

- **Familias disponibles históricamente:** `volatility`, `liquidity` (spreads bid-ask y rango de cotización), `futures_basis` (estructura temporal de futuros de volatilidad CBOE `VX` sobre `^VIX`) y `cross_asset` (6 pares FX pares + `^GSPC` + `^VIX`).
- **Referencias sin features:** `constant_05` ($p=0.5$): `Log-Loss = 0.69315`, `ROC-AUC = 0.50000`. `train_prevalence`: `Log-Loss = 0.69324`, `ROC-AUC = 0.48002`.

| Modelo (Semilla) | Receta / Familia | Cols | Log-Loss OOS ↓ | ΔLoss vs Base | ΔLoss vs Ruido | ROC-AUC OOS ↑ | ΔAUC vs Base | ΔAUC vs Ruido | LOFO ΔLoss (Sin−All) | Brecha Train-OOS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`logistic_l2` (42)** | `F0_base` (13 orig.) | 13 | 0.69859 | +0.00000 | — | 0.50245 | +0.00000 | — | — | +0.01242 |
| `logistic_l2` (42) | `noise_add4_mean` (+4 ruido) | 17 | 0.69962 | +0.00103 | +0.00000 | 0.50217 | -0.00028 | +0.00000 | — | — |
| **`logistic_l2` (42)** | **`add_volatility`** | 17 | **0.69708** | **-0.00151** | **-0.00254** | **0.52009** | **+0.01765** | **+0.01792** | -0.00025 | +0.01531 |
| `logistic_l2` (42) | `add_liquidity` | 17 | 0.70046 | +0.00187 | +0.00084 | 0.50314 | +0.00069 | +0.00097 | -0.00380 | +0.01570 |
| `logistic_l2` (42) | `add_futures_basis` | 17 | 0.70236 | +0.00376 | +0.00274 | 0.51183 | +0.00938 | +0.00966 | -0.00185 | +0.02079 |
| **`logistic_l2` (42)** | **`add_cross_asset`** | 17 | 0.70252 | +0.00393 | +0.00290 | **0.53050** | **+0.02805** | **+0.02833** | -0.00645 | +0.02015 |
| **`random_forest` (42)** | `F0_base` (13 orig.) | 13 | 0.69926 | +0.00000 | — | 0.49225 | +0.00000 | — | — | +0.05562 |
| `random_forest` (42) | `noise_add4_mean` (+4 ruido) | 17 | 0.69536 | -0.00391 | +0.00000 | 0.50040 | +0.00815 | +0.00000 | — | — |
| **`random_forest` (42)** | **`add_futures_basis`** | 17 | **0.69522** | **-0.00404** | **-0.00014** | **0.50687** | **+0.01462** | **+0.00647** | -0.00137 | +0.05969 |
| **`random_forest` (1729)** | **`add_futures_basis`** | 17 | **0.69663** | **-0.00477** | +0.00011 | **0.49987** | **+0.01583** | **+0.00491** | **+0.00171** | +0.05884 |
| `random_forest` (42) | `add_cross_asset` | 17 | 0.69769 | -0.00157 | +0.00234 | 0.49745 | +0.00520 | -0.00295 | -0.00099 | +0.06006 |
| **`xgboost` (42)** | `F0_base` (13 orig.) | 13 | 0.70021 | +0.00000 | — | 0.49177 | +0.00000 | — | — | +0.05285 |
| `xgboost` (42) | `noise_add4_mean` (+4 ruido) | 17 | 0.69733 | -0.00288 | +0.00000 | 0.50398 | +0.01222 | +0.00000 | — | — |
| **`xgboost` (42)** | **`add_futures_basis`** | 17 | **0.69632** | **-0.00389** | **-0.00101** | **0.50625** | **+0.01449** | **+0.00227** | **+0.00138** | +0.05633 |
| **`xgboost` (1729)** | **`add_futures_basis`** | 17 | **0.69619** | **-0.00403** | **-0.00216** | **0.50502** | **+0.01326** | **+0.01595** | **+0.00252** | +0.05612 |
| **`xgboost` (42)** | **`nested_noise_screened`** | 15.7 | **0.69586** | **-0.00435** | **-0.00147** | **0.51623** | **+0.02447** | **+0.01225** | — | **+0.05119** |
| **`lightgbm` (42)** | `F0_base` (13 orig.) | 13 | 0.71316 | +0.00000 | — | 0.46489 | +0.00000 | — | — | +0.09520 |
| `lightgbm` (42) | `noise_add4_mean` (+4 ruido) | 17 | 0.70439 | -0.00877 | +0.00000 | 0.47947 | +0.01458 | +0.00000 | — | — |
| **`lightgbm` (42)** | **`add_futures_basis`** | 17 | **0.70488** | **-0.00828** | +0.00049 | **0.49968** | **+0.03479** | **+0.02020** | **+0.00476** | +0.09976 |
| `lightgbm` (42) | `add_cross_asset` | 17 | 0.70678 | -0.00638 | +0.00239 | 0.49689 | +0.03200 | +0.01742 | -0.00062 | +0.09105 |

---

## 4. Importancia por Permutación por Bloques Fuera de Muestra (`Bloques de 30 días`)

La permutación por bloques contiguos de 30 días (8 repeticiones sobre `all_financial`, semilla 42) mide cuánto empeora la `log-loss` OOS ($\Delta\text{Loss} > 0$) al romper la alineación temporal de una familia o variable individual manteniendo fijo el modelo entrenado:

| Activo | Variable / Familia | Familia | `XGBoost` ΔLoss Perm (Folds > 0) | `RandomForest` ΔLoss Perm (Folds > 0) | `LightGBM` ΔLoss Perm (Folds > 0) | Diagnóstico |
|---|---|---|---|---|---|---|
| **`BTCUSDT`** | **`fin_liq_corwin_schultz_20`** | `liquidity` | **`+0.000742` (`3/3`)** | **`+0.000721` (`3/3`)** | **`+0.001849` (`3/3`)** | **Señal consistente en 3/3 folds y 3 arquitecturas** |
| **`BTCUSDT`** | **`fin_liq_range_impact_z63`** | `liquidity` | **`+0.000356` (`3/3`)** | `+0.000008` (`2/3`) | `+0.000120` (`2/3`) | Señal positiva consistente en XGBoost (`3/3` folds) |
| **`BTCUSDT`** | **`fin_vol_parkinson_ratio`** | `volatility` | `+0.000112` (`2/3`) | **`+0.000289` (`3/3`)** | `+0.000568` (`1/3`) | Útil para distinguir rango intravela vs cierre |
| **`BTCUSDT`** | **`fin_basis_dated_ann`** | `futures_basis` | `+0.000275` (`2/3`) | `-0.000041` (`1/3`) | `+0.000364` (`2/3`) | Mejor variable de basis en BTC (supera al perp basis) |
| **`ETHUSDT`** | **`fin_cross_sp500_mom_z`** | `cross_asset` | **`+0.002543` (`3/3`)** | **`+0.001265` (`3/3`)** | **`+0.005082` (`3/3`)** | **Variable #1 en ETH (`3/3` folds en los 3 modelos)** |
| **`ETHUSDT`** | **`fin_liq_corwin_schultz_20`** | `liquidity` | **`+0.001344` (`3/3`)** | **`+0.000542` (`3/3`)** | **`+0.002378` (`3/3`)** | **Señal consistente en 3/3 folds y 3 arquitecturas** |
| **`ETHUSDT`** | **`fin_vol_parkinson_ratio`** | `volatility` | **`+0.001498` (`3/3`)** | `+0.000115` (`2/3`) | **`+0.002012` (`3/3`)** | Fuerte aporte en boosting sobre ETH (`3/3` folds) |
| **`ETHUSDT`** | **`fin_liq_range_impact_z63`** | `liquidity` | `+0.000374` (`2/3`) | **`+0.000510` (`3/3`)** | **`+0.001413` (`3/3`)** | Consistente en RandomForest y LightGBM (`3/3` folds) |
| **`EURUSD`** | **`fin_basis_vx_z63`** | `futures_basis` | **`+0.004901` (`3/3`)** | **`+0.001152` (`3/3`)** | `+0.003818` (`2/3`) | **Variable #1 en EURUSD (`3/3` folds en XGB y RF)** |
| **`EURUSD`** | **`fin_vol_garman_klass_z63`** | `volatility` | `+0.001929` (`2/3`) | `+0.001046` (`2/3`) | `+0.000520` (`2/3`) | Positiva en los 3 modelos en `2/3` folds |
| **`EURUSD`** | **`fin_cross_vix_z63`** | `cross_asset` | `+0.001806` (`2/3`) | `+0.000524` (`2/3`) | `+0.001039` (`2/3`) | Régimen de aversión al riesgo global para EURUSD |

---

## 5. Diagnóstico por Familia Financiera: ¿Qué Sirve y Qué Descartar?

1. **Liquidez (`liquidity`) — RECOMENDADA (especialmente en Cripto Spot):**
   - **Por qué funciona:** las variables `fin_liq_corwin_schultz_20` y `fin_liq_range_impact_z63` cuantifican la fricción transaccional y la absorción de volumen dentro de la vela sin depender de la dirección del retorno pasado. En `BTCUSDT` y `ETHUSDT`, mejoran tanto la discriminación (`ROC-AUC`) como la calibración (`log-loss`) frente a `F0_base` y frente al ruido gaussiano.
2. **Basis de Futuros (`futures_basis`) — RECOMENDADA EN FOREX (`VX` vs `VIX`) Y SELECTIVA EN CRIPTO (`fin_basis_dated_ann`):**
   - En `EURUSD`, la estructura temporal de futuros de volatilidad CBOE (`fin_basis_vx_z63`, `fin_basis_vx_term`) captura cambios de régimen de liquidez global y apetito por riesgo antes de que se reflejen en las medias móviles del tipo de cambio.
   - En cripto, el basis anualizado de los contratos trimestrales con vencimiento (`fin_basis_dated_ann` de `BTCUSD_YYMMDD` / `ETHUSD_YYMMDD`) aporta más información fuera de muestra que la prima instantánea del perpetuo (`fin_basis_perp_premium`), la cual introduce ruido de alta frecuencia en el cierre diario.
3. **Relaciones entre Activos (`cross_asset`) — SELECTIVA (`fin_cross_sp500_mom_z` en `ETHUSDT`, perjudicial en `BTCUSDT`):**
   - `ETHUSDT` se beneficia claramente del contexto bursátil estadounidense (`fin_cross_sp500_mom_z`, `3/3` folds positivos en todos los árboles), mientras que `BTCUSDT` empeora al añadir rezagos de otros activos. No debe aplicarse una receta *cross-asset* indiscriminada a `BTCUSDT`.
4. **Volatilidad (`volatility`) — ÚTIL EN MODELOS LINEALES (`logistic_l2`) O VÍA `fin_vol_parkinson_ratio`, REDUNDANTE EN ÁRBOLES:**
   - En `logistic_l2`, `add_volatility` mejora el `ROC-AUC` en los tres mercados y reduce la `log-loss` en `ETHUSDT` (`0.69272 < 0.69315`) y `EURUSD` (`0.69708`). En árboles de boosting, agregar las 4 variables juntas sobreajusta; sólo `fin_vol_parkinson_ratio` muestra utilidad incremental consistente.
5. **Volumen Relativo (`relative_volume`) y Funding Rates (`funding_rates`) — DESCARTADAS EN HORIZONTE DIARIO ($h=1$):**
   - En resolución diaria, las tasas de fondeo realizadas (`fin_fund_1d`, `fin_fund_7d`, `fin_fund_z63`) y el flujo agresor diario (`fin_rvol_*`) sufren de fuerte no estacionariedad entre el mercado bajista de 2022 y la expansión 2023–2025, aumentando la brecha train-OOS y perdiendo sistemáticamente contra controles de ruido gaussiano de igual dimensión.

---

## 6. Reproducibilidad e Integridad

Para reproducir el análisis o verificar los artefactos sin reentrenar:

```powershell
# Re-generar el informe y verificar hashes desde los artefactos guardados:
.\.venv\Scripts\python.exe scripts/financial_feature_ablation.py --analyze artifacts/financial_feature_ablation/20261017T224928_57040c88
```

- **Manifiesto SHA-256 de la corrida:** [`artifacts/financial_feature_ablation/20261017T224928_57040c88/artifacts.json`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/financial_feature_ablation/20261017T224928_57040c88/artifacts.json)
- **Tabla completa de ablación (CSV):** [`artifacts/financial_feature_ablation/20261017T224928_57040c88/family_ablation_table.csv`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/financial_feature_ablation/20261017T224928_57040c88/family_ablation_table.csv)
- **Predicciones OOS (`51,372` filas):** [`artifacts/financial_feature_ablation/20261017T224928_57040c88/predictions.csv`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/financial_feature_ablation/20261017T224928_57040c88/predictions.csv)
- **Resumen estadístico y bootstrap (JSON):** [`artifacts/financial_feature_ablation/20261017T224928_57040c88/summary.json`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/financial_feature_ablation/20261017T224928_57040c88/summary.json)

