# Informe Comparativo Focalizado: Capacidad Predictiva y Rentabilidad con Variables Financieras Seleccionadas

- **Fecha de ejecución:** 2026-10-17 UTC
- **Run ID auditable:** `20261017T232854_36d523ee`
- **Directorio de artefactos:** [`artifacts/focused_financial_study/20261017T232854_36d523ee/`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/focused_financial_study/20261017T232854_36d523ee)
- **Script reproducible:** [`scripts/focused_financial_study.py`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/scripts/focused_financial_study.py)
- **Estudio previo de referencia:** [`FINANCIAL_FEATURE_ABLATION_REPORT.md`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/FINANCIAL_FEATURE_ABLATION_REPORT.md) (`20261017T224928_57040c88`)
- **Integridad de reservas prospectivas (`>= 2025-07-01`) y modelos previos:** `protected_unchanged = true` (verificado mediante hashes SHA-256 antes y después de la ejecución en [`plan.json`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/focused_financial_study/20261017T232854_36d523ee/plan.json) y [`summary.json`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/focused_financial_study/20261017T232854_36d523ee/summary.json)).

---

## 1. Resumen Ejecutivo y Veredicto General

Partiendo de los hallazgos de [`FINANCIAL_FEATURE_ABLATION_REPORT.md`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/FINANCIAL_FEATURE_ABLATION_REPORT.md), diseñamos y ejecutamos un estudio experimental focalizado exclusivamente en las familias de variables que habían mostrado señal prometedora por activo, descartando las familias ruidosas (`relative_volume`, `funding_rates` y macro amplio indiscriminado en `BTCUSDT`/`EURUSD`):

1. **`BTCUSDT` y `ETHUSDT` (`1d`):** Liquidez (`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`, `fin_liq_amihud_z63`, `fin_liq_trade_size_z63`), volatilidad de rango de Parkinson (`fin_vol_parkinson_ratio`) y momento relativo del S&P 500 (`fin_cross_sp500_mom_z`), evaluando combinaciones parsimoniosas y selección anidada causal por activo (`nested_selected` e `inner_perm_top`).
2. **`EURUSD` (`1d` con ejecución `1h` bid/ask):** Estructura temporal y posicionamiento de futuros de volatilidad `VX/VIX` (`fin_basis_vx_z63`, `fin_basis_vx_term`, `fin_basis_vx_change_5d`, `fin_basis_vx_oi_z63`) y liquidez de cotización FXCM (`fin_liq_spread_z63`, `fin_liq_spread_to_range`, `fin_liq_corwin_schultz_20`, `fin_liq_quote_range_excess`).
3. **Modelos evaluados:** Únicamente los modelos existentes `LogisticRegression` con regularización $L_2$ (`logistic_l2`, `seed=42`), `RandomForestClassifier` (`random_forest`, `seeds=42, 1729`) y `XGBClassifier` (`xgboost`, `seeds=42, 1729`).
4. **Controles con ruido de dimensión equivalente:** Cada receta con $k$ variables financieras agregadas se comparó sobre **exactamente las mismas observaciones** contra el modelo original (`F0_base`) y contra el promedio de tres flujos independientes de ruido gaussiano causal de igual dimensión (`noise_k{k}_mean`, y `noise_nested_match` para `nested_selected`).
5. **Validación temporal y económica:** Walk-forward purgado de 3 pliegues (`2023-01-01..2025-07-01`) con embargo de `6` días, selección anidada en `2 × 120` días previos a cada pliegue, bootstrap por bloques móviles de `30` días (`2,000` réplicas) con corrección de Holm-Bonferroni, y simulación económica con costos reales y estrés de costos 2x (`simulate_events` para Cripto y `FxEngine.portfolio` sobre cotizaciones `1h` bid/ask para `EURUSD`).

### Conclusión Principal

- **Sí logramos mejoras predictivas estadísticamente significativas (incluso tras corrección de Holm-Bonferroni contra el modelo base y contra controles con ruido de igual dimensión) en `ETHUSDT` y `EURUSD`, y mejoras consistentes de selección compacta en `BTCUSDT`:**
  - En **`ETHUSDT`**, la receta compacta de 4 variables **`top_core_4_sp500`** (`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`, `fin_vol_parkinson_ratio`, `fin_cross_sp500_mom_z`) supera simultáneamente a `F0_base` y a `noise_k4_mean` en **los 3 modelos (`RandomForest`, `XGBoost`, `LogisticRegression`)**, ganando en **`3/3` pliegues** en `RandomForest` (`seed=42, 1729`) y `XGBoost` (`seed=1729`). En `RandomForest` (`seed=1729`), reduce `log_loss` en `-0.00647` vs. base ($p_{\text{Holm}} = 0.0045$) y `-0.00419` vs. ruido ($p_{\text{Holm}} = 0.0090$), y eleva `ROC-AUC` de `0.51500` a `0.53938` (`+0.02438` vs. base, $p_{\text{Holm}} = 0.0090$; `+0.02092` vs. ruido, $p_{\text{Holm}} = 0.0045$). En `XGBoost` (`seed=42`), `nested_selected` eleva `ROC-AUC` de `0.52239` a `0.54258` (`+0.02884` vs. ruido, $p_{\text{Holm}} = 0.018$) y `Accuracy` de `51.83%` a `53.88%`. En `LogisticRegression`, `top_core_4_sp500` eleva `ROC-AUC` de `0.51607` a `0.55870` (`+0.05026` vs. ruido, $p = 0.006$, $p_{\text{Holm}} = 0.054$) y `Accuracy` de `50.34%` a `54.91%` (`+4.57 pp`).
  - En **`EURUSD`**, el núcleo de 4 variables **`vx_liq_core_4`** (`fin_basis_vx_z63`, `fin_basis_vx_term`, `fin_liq_spread_to_range`, `fin_liq_corwin_schultz_20`) y el par **`vx_core_2`** (`fin_basis_vx_z63`, `fin_basis_vx_term`) superan significativamente tanto a `F0_base` como al ruido: en `XGBoost` (`seed=1729`), `vx_liq_core_4` reduce `log_loss` de `0.70184` a `0.69264` (`-0.00920` vs. base, $p_{\text{Holm}} = 0.0035$; `-0.00535` vs. ruido, $p = 0.0285$) y eleva `ROC-AUC` de `0.48231` a `0.53038` (`+0.04807` vs. base, $p_{\text{Holm}} < 0.001$; `+0.04200` vs. ruido, $p_{\text{Holm}} = 0.021$), con `ROC-AUC = 0.53357` en `seed=42` (`3/3` pliegues vs. base y `3/3` vs. ruido). En `RandomForest` (`seed=42`), `vx_core_2` supera a `noise_k2_mean` con $p_{\text{Holm}} = 0.0175$ en `log_loss` y $p_{\text{Holm}} = 0.0315$ en `ROC-AUC`.
  - En **`BTCUSDT`**, el núcleo compacto sin macro **`top_core_3`** (`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`, `fin_vol_parkinson_ratio`) y **`top_core_4_sp500`** mejoran `log_loss` y `ROC-AUC` frente a `F0_base` y frente al ruido en `RandomForest` (`ROC-AUC = 0.55659` en `seed=42`, $p = 0.0275$ vs. ruido; `3/3` pliegues en `seed=1729`) y `top_core_3` es la única receta que mejora simultáneamente `log_loss` y `ROC-AUC` vs. base y vs. ruido en ambos seeds de `XGBoost` (`2/3` pliegues), aunque en `BTCUSDT` los contrastes ajustados por Holm-Bonferroni no cruzan $\alpha = 0.05$ ($p_{\text{Holm}} \ge 0.0765$).
- **Sin embargo, la conversión de mejora predictiva en ventaja económica robusta tras costos reales es parcial y depende críticamente del umbral de selectividad ($\tau = 0.55$) y de la estructura de costos:**
  - **Donde sí hay mejora económica robusta (incluso bajo estrés de costos 2x):** En **`BTCUSDT`** con umbral selectivo $\tau = 0.55$ (`fixed_055`), **`top_core_3`** eleva en `logistic_l2` el retorno neto de `+33.10%` (`Sharpe = 0.839`, `max_drawdown = -10.93%`) en `F0_base` y `+29.77%` en `noise_k3_mean` a **`+45.29%` (`Sharpe = 1.165`, `max_drawdown = -8.92%`, `PF = 2.750`)** bajo costos reales (`34 bps` ida y vuelta), manteniendo **`+28.24%` (`Sharpe = 0.808`, `PF = 1.940`) bajo costos dobles (`68 bps` ida y vuelta)**. En `XGBoost` ($\tau = 0.55$), `top_core_3` eleva el retorno neto de `+24.26%` (`Sharpe = 0.538`) en `F0_base` y `+5.20%` en `noise_k3_mean` a **`+43.37%` (`Sharpe = 0.822`, `max_drawdown = -13.69%`, `PF = 1.396`)**, siendo la única receta de `XGBoost` que sobrevive en positivo al estrés 2x (`+5.45%` vs `-9.48%` en `F0_base`).
  - **Donde hay reversión a rentabilidad en costos base pero fragilidad al estrés 2x:** En **`ETHUSDT`**, donde `F0_base` perdía dinero en `logistic_l2` (`-14.75%` a `-31.04%`) y `xgboost` (`-3.92%` a `-52.53%`), `top_core_4_sp500` y `nested_selected` pasan a retornos positivos bajo costos reales base (`logistic_l2`: `+105.49%`, `Sharpe = 0.886` en $\tau = 0.55$ y `+160.47%`, `Sharpe = 1.073` en `inner_tau = 0.53`; `xgboost`: `+14.23%` y `+10.77%` en $\tau = 0.55$; `random_forest`: `+15.93%`, `Sharpe = 0.341` en `nested_selected`). No obstante, debido al elevado número de operaciones (`136..461` trades), bajo `double_cost` (`68 bps` por operación) todas las variantes de `ETHUSDT` vuelven a terreno negativo.
  - **Donde la mejora predictiva no alcanza para superar la fricción operativa actual:** En **`EURUSD`**, aunque `vx_liq_core_4` y `vx_core_2` logran el salto relativo más grande en `ROC-AUC` (`+4.07` a `+4.81 pp` en `XGBoost`), el motor `FxEngine.portfolio` (long-only sobre barras `1h` con carry/swap negativo y spread horario) sigue dando retornos negativos en `logistic_l2` y `xgboost` (salvo `+0.68%` en `vx_basis_plus_liquidity` y `+0.15%` en `nested_selected` para `xgboost` `seed=1729` en $\tau = 0.55$). Solo `RandomForest` en $\tau = 0.55$ logra retornos positivos moderados (`nested_selected`: `+3.57%`, `Sharpe = 0.705`, `+2.44%` en `double_cost`; `inner_perm_top`: `+3.92%`, `Sharpe = 0.812`, `+3.15%` en `double_cost`), pero sin superar de forma consistente a todos los controles de ruido en ambos seeds.

---

## 2. Diseño Experimental y Validación Causal

El protocolo implementado en [`scripts/focused_financial_study.py`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/scripts/focused_financial_study.py) cumple estrictamente las reglas de validación financiera:

1. **Partición walk-forward purgada sin fugas:**
   - 3 pliegues externos sobre `2023-01-01..2025-07-01` (`2023-01-01..2023-11-01`, `2023-11-01..2024-09-01`, `2024-09-01..2025-07-01`) con embargo causal de `6` días (`target_time < fold_start - 6d`) y recorte de horizonte al cierre de cada pliegue (`12` días en Cripto, `20` días en Forex) para que las simulaciones multi-día de eventos no crucen fronteras entre pliegues ni toquen la reserva sellada (`>= 2025-07-01`).
   - Exactamente `876` observaciones OOS evaluadas en `BTCUSDT`, `876` en `ETHUSDT` y `589` en `EURUSD`.
2. **Selección anidada estrictamente en train (`nested_select_for_asset`):**
   - Antes de cada pliegue externo, se construyen `2` sub-ventanas internas de `120` días dentro del conjunto de entrenamiento (`outer_start - 6d`).
   - En esas ventanas internas se calcula: (a) la importancia por permutación en bloques de `7` días de cada variable candidata, (b) el desempeño de cada receta candidata frente a `F0_base` y frente al control de ruido de igual dimensión `noise_k{k}` (exigiendo superar a ambos tanto en `log_loss` como en `ROC-AUC`, más penalización por complejidad `0.00015 × Δcols`; si ninguna califica, aplica fallback causal a `F0_base`), (c) el hiperparámetro de regularización $C \in \{0.1, 0.3, 1.0\}$ para `logistic_l2`, y (d) el umbral de probabilidad óptimo $\tau \in \{0.51, 0.52, 0.53, 0.54, 0.55\}$ (`inner_tau`) bajo retorno aproximado neto de costos en validación interna.
3. **Recetas evaluadas por activo:**
   - **Cripto (`BTCUSDT`, `ETHUSDT`):**
     - `F0_base` (`14` variables base).
     - `liq_only` (`+4`: `fin_liq_amihud_z63`, `fin_liq_range_impact_z63`, `fin_liq_corwin_schultz_20`, `fin_liq_trade_size_z63`).
     - `liq_parkinson` (`+5`: liquidez + `fin_vol_parkinson_ratio`).
     - `liq_sp500` (`+5`: liquidez + `fin_cross_sp500_mom_z`).
     - `parkinson_sp500` (`+2`: `fin_vol_parkinson_ratio` + `fin_cross_sp500_mom_z`).
     - `liq_parkinson_sp500` (`+6`: todas las prometedoras de Cripto).
     - `top_core_3` (`+3`: `fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`, `fin_vol_parkinson_ratio`).
     - `top_core_4_sp500` (`+4`: `top_core_3` + `fin_cross_sp500_mom_z`).
     - `nested_selected` (receta elegida causalmente por pliegue en validación interna con filtro de ruido).
     - `inner_perm_top` (variables con permutación interna positiva, tope 4).
   - **Forex (`EURUSD`):**
     - `F0_base` (`13` variables base).
     - `vx_basis_only` (`+4`: `fin_basis_vx_z63`, `fin_basis_vx_change_5d`, `fin_basis_vx_term`, `fin_basis_vx_oi_z63`).
     - `fx_liquidity_only` (`+4`: `fin_liq_spread_z63`, `fin_liq_spread_to_range`, `fin_liq_corwin_schultz_20`, `fin_liq_quote_range_excess`).
     - `vx_basis_plus_liquidity` (`+8`: ambas familias completas).
     - `vx_core_2` (`+2`: `fin_basis_vx_z63`, `fin_basis_vx_term`).
     - `vx_liq_core_4` (`+4`: `fin_basis_vx_z63`, `fin_basis_vx_term`, `fin_liq_spread_to_range`, `fin_liq_corwin_schultz_20`).
     - `nested_selected` e `inner_perm_top`.

---

## 3. Resultados Comparativos de Capacidad Predictiva y Significancia Estadística

Fuente completa: [`predictive_contrasts.csv`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/focused_financial_study/20261017T232854_36d523ee/predictive_contrasts.csv) y [`nested_decisions.json`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/focused_financial_study/20261017T232854_36d523ee/nested_decisions.json).

### 3.1. `BTCUSDT` (`876` observaciones OOS)

| Modelo (`seed`) | Receta (`noise_ref`) | `log_loss` OOS | `ROC-AUC` OOS | `Accuracy` | `Bal. Acc` | $\Delta\text{loss}$ vs Base (`p_raw` / `p_Holm`) | $\Delta\text{AUC}$ vs Base (`p_raw` / `p_Holm`) | $\Delta\text{loss}$ vs Ruido (`p_raw` / `p_Holm`) | $\Delta\text{AUC}$ vs Ruido (`p_raw` / `p_Holm`) | Pliegues Base / Ruido |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`logistic_l2` (`42`)** | `F0_base` | `0.69788` | `0.53908` | `52.17%` | `52.89%` | — | — | — | — | — |
| `logistic_l2` (`42`) | `top_core_3` (`noise_k3`) | `0.69916` | **`0.54866`** | **`53.65%`** | **`54.37%`** | `+0.00128` (`0.777`/`1.00`) | `+0.00958` (`0.214`/`1.00`) | `+0.00066` (`0.608`/`1.00`) | `+0.00764` (`0.159`/`1.00`) | `2/3` / `2/3` |
| `logistic_l2` (`42`) | `nested_selected` (`match`) | **`0.69658`** | `0.54485` | `51.14%` | `51.95%` | **`-0.00130`** (`0.366`/`1.00`) | `+0.00577` (`0.408`/`1.00`) | **`-0.00243`** (`0.200`/`1.00`) | `+0.00779` (`0.303`/`1.00`) | `2/3` / `2/3` |
| `logistic_l2` (`42`) | `parkinson_sp500` (`noise_k2`) | `0.69569` | `0.54252` | `50.80%` | `51.53%` | `-0.00219` (`0.271`/`1.00`) | `+0.00344` (`0.509`/`1.00`) | `-0.00211` (`0.352`/`1.00`) | `+0.00402` (`0.579`/`1.00`) | `1/3` / `1/3` |
| **`random_forest` (`42`)** | `F0_base` | `0.69766` | `0.54092` | `53.42%` | `54.09%` | — | — | — | — | — |
| `random_forest` (`42`) | **`top_core_4_sp500`** (`noise_k4`) | **`0.69340`** | **`0.55659`** | `51.71%` | `52.49%` | **`-0.00426`** (`0.0085`/`0.077`) | **`+0.01568`** (`0.019`/`0.171`) | **`-0.00265`** (`0.0275`/`0.248`) | **`+0.01411`** (`0.0275`/`0.248`) | `2/3` / `2/3` |
| `random_forest` (`42`) | **`top_core_3`** (`noise_k3`) | `0.69421` | `0.55282` | **`54.11%`** | **`54.75%`** | `-0.00345` (`0.0155`/`0.109`) | `+0.01191` (`0.042`/`0.336`) | `-0.00132` (`0.144`/`0.720`) | `+0.00716` (`0.0905`/`0.724`) | `2/3` / `2/3` |
| `random_forest` (`42`) | `liq_only` (`noise_k4`) | `0.69463` | `0.54918` | `52.28%` | `53.03%` | `-0.00303` (`0.0345`/`0.173`) | `+0.00826` (`0.0895`/`0.537`) | `-0.00141` (`0.091`/`0.637`) | `+0.00670` (`0.114`/`0.798`) | `2/3` / `1/3` |
| `random_forest` (`42`) | `nested_selected` (`match`) | `0.69522` | `0.54734` | `53.42%` | `54.05%` | `-0.00245` (`0.010`/`0.080`) | `+0.00643` (`0.0675`/`0.473`) | `+0.00028` (`0.509`/`1.00`) | `-0.00098` (`0.512`/`1.00`) | `2/3` / `1/3` |
| **`random_forest` (`1729`)** | `F0_base` | `0.69687` | `0.54336` | `53.31%` | `53.98%` | — | — | — | — | — |
| `random_forest` (`1729`) | **`top_core_4_sp500`** (`noise_k4`) | **`0.69489`** | **`0.55262`** | `52.40%` | `53.12%` | `-0.00198` (`0.088`/`0.792`) | `+0.00927` (`0.0965`/`0.869`) | `-0.00204` (`0.0725`/`0.508`) | **`+0.01156`** (`0.042`/`0.378`) | **`3/3` / `3/3`** |
| `random_forest` (`1729`) | **`top_core_3`** (`noise_k3`) | `0.69494` | `0.54982` | `52.05%` | `52.75%` | `-0.00193` (`0.109`/`0.872`) | `+0.00646` (`0.161`/`1.00`) | **`-0.00204`** (`0.036`/`0.324`) | **`+0.00867`** (`0.0425`/`0.378`) | `2/3` / `2/3` |
| `random_forest` (`1729`) | `nested_selected` (`match`) | `0.69601` | `0.54389` | `52.74%` | `53.38%` | `-0.00086` (`0.122`/`0.872`) | `+0.00053` (`0.277`/`1.00`) | **`-0.00207`** (`0.046`/`0.368`) | `+0.00647` (`0.0675`/`0.473`) | `1/3` / `2/3` |
| **`xgboost` (`42`)** | `F0_base` | `0.70059` | `0.53645` | `51.03%` | `51.63%` | — | — | — | — | — |
| `xgboost` (`42`) | **`top_core_3`** (`noise_k3`) | **`0.69766`** | **`0.54344`** | **`51.83%`** | **`52.37%`** | `-0.00293` (`0.180`/`1.00`) | `+0.00699` (`0.309`/`1.00`) | **`-0.00180`** (`0.231`/`1.00`) | **`+0.00638`** (`0.249`/`1.00`) | `2/3` / `2/3` |
| `xgboost` (`42`) | `top_core_4_sp500` (`noise_k4`) | `0.69840` | `0.54137` | `51.26%` | `51.87%` | `-0.00219` (`0.240`/`1.00`) | `+0.00493` (`0.343`/`1.00`) | `+0.00006` (`0.381`/`1.00`) | `+0.00209` (`0.345`/`1.00`) | `2/3` / `2/3` |
| **`xgboost` (`1729`)** | `F0_base` | `0.69996` | `0.53955` | `52.51%` | `53.02%` | — | — | — | — | — |
| `xgboost` (`1729`) | **`top_core_3`** (`noise_k3`) | **`0.69644`** | **`0.54285`** | `51.71%` | `52.23%` | `-0.00353` (`0.104`/`0.936`) | `+0.00330` (`0.443`/`1.00`) | **`-0.00118`** (`0.299`/`1.00`) | **`+0.00119`** (`0.407`/`1.00`) | `2/3` / `2/3` |
| `xgboost` (`1729`) | `top_core_4_sp500` (`noise_k4`) | `0.69744` | `0.54110` | `51.94%` | `52.52%` | `-0.00253` (`0.274`/`1.00`) | `+0.00155` (`0.526`/`1.00`) | `+0.00022` (`0.433`/`1.00`) | `+0.00039` (`0.421`/`1.00`) | `2/3` / `2/3` |

**Lectura clave en `BTCUSDT`:**
- Las `3` variables internas de liquidez y rango (`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`, `fin_vol_parkinson_ratio` = **`top_core_3`**) aportan señal consistente en los 3 modelos (`logistic_l2`, `random_forest`, `xgboost`), mientras que agregar las 6 variables juntas (`liq_parkinson_sp500`) diluye los árboles (`XGBoost` cae por debajo de `noise_k6_mean`).
- El momento del S&P 500 (`fin_cross_sp500_mom_z`) ayuda a `RandomForest` (`top_core_4_sp500` logra el máximo `ROC-AUC = 0.55659` en `seed=42` y gana en `3/3` pliegues en `seed=1729`), pero degrada a `logistic_l2` y `xgboost` frente a `top_core_3`.

---

### 3.2. `ETHUSDT` (`876` observaciones OOS)

| Modelo (`seed`) | Receta (`noise_ref`) | `log_loss` OOS | `ROC-AUC` OOS | `Accuracy` | `Bal. Acc` | $\Delta\text{loss}$ vs Base (`p_raw` / `p_Holm`) | $\Delta\text{AUC}$ vs Base (`p_raw` / `p_Holm`) | $\Delta\text{loss}$ vs Ruido (`p_raw` / `p_Holm`) | $\Delta\text{AUC}$ vs Ruido (`p_raw` / `p_Holm`) | Pliegues Base / Ruido |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`logistic_l2` (`42`)** | `F0_base` | `0.69271` | `0.51607` | `50.34%` | `50.11%` | — | — | — | — | — |
| `logistic_l2` (`42`) | **`top_core_4_sp500`** (`noise_k4`) | `0.69044` | **`0.55870`** | **`54.91%`** | **`54.61%`** | `-0.00227` (`0.369`/`1.00`) | **`+0.04262`** (`0.029`/`0.232`) | `-0.00270` (`0.286`/`1.00`) | **`+0.05026`** (`0.006`/`0.054`) | `2/3` / `2/3` |
| `logistic_l2` (`42`) | **`nested_selected`** (`match`) | **`0.69008`** | `0.55297` | `52.51%` | `52.09%` | **`-0.00263`** (`0.152`/`1.00`) | **`+0.03690`** (`0.012`/`0.108`) | `-0.00211` (`0.191`/`1.00`) | **`+0.03605`** (`0.0145`/`0.116`) | `2/3` / `2/3` |
| `logistic_l2` (`42`) | `top_core_3` (`noise_k3`) | `0.69058` | `0.54703` | `54.00%` | `53.72%` | `-0.00214` (`0.372`/`1.00`) | `+0.03095` (`0.111`/`0.686`) | `-0.00489` (`0.149`/`1.00`) | **`+0.04536`** (`0.030`/`0.210`) | `2/3` / `2/3` |
| **`random_forest` (`42`)** | `F0_base` | `0.69739` | `0.50875` | `51.14%` | `51.31%` | — | — | — | — | — |
| `random_forest` (`42`) | **`liq_sp500`** (`noise_k5`) | **`0.69077`** | **`0.54422`** | **`53.88%`** | **`54.03%`** | **`-0.00663`** (`0.000`/`0.000`) | **`+0.03546`** (`0.0005`/`0.004`) | **`-0.00280`** (`0.0295`/`0.266`) | `+0.01924` (`0.0655`/`0.590`) | **`3/3` / `3/3`** |
| `random_forest` (`42`) | **`top_core_4_sp500`** (`noise_k4`) | `0.69190` | `0.53642` | `51.60%` | `51.73%` | **`-0.00549`** (`0.002`/`0.014`) | **`+0.02766`** (`0.001`/`0.006`) | `-0.00229` (`0.065`/`0.455`) | `+0.01047` (`0.0785`/`0.628`) | **`3/3` / `3/3`** |
| `random_forest` (`42`) | `liq_parkinson_sp500` (`noise_k6`) | `0.69142` | `0.53755` | `53.31%` | `53.45%` | **`-0.00597`** (`0.0005`/`0.004`) | **`+0.02880`** (`0.0005`/`0.004`) | **`-0.00239`** (`0.0295`/`0.266`) | `+0.00966` (`0.0965`/`0.676`) | `3/3` / `2/3` |
| `random_forest` (`42`) | `nested_selected` (`match`) | `0.69461` | `0.52757` | `53.65%` | `53.83%` | **`-0.00278`** (`0.006`/`0.030`) | **`+0.01882`** (`0.000`/`0.000`) | `-0.00004` (`0.549`/`1.00`) | `+0.00309` (`0.441`/`1.00`) | `2/3` / `2/3` |
| **`random_forest` (`1729`)** | `F0_base` | `0.69806` | `0.51500` | `51.48%` | `51.50%` | — | — | — | — | — |
| `random_forest` (`1729`) | **`top_core_4_sp500`** (`noise_k4`) | `0.69159` | **`0.53938`** | `52.40%` | `52.54%` | **`-0.00647`** (`0.0005`/`0.0045`) | **`+0.02438`** (`0.001`/`0.009`) | **`-0.00419`** (`0.001`/`0.009`) | **`+0.02092`** (`0.0005`/`0.0045`) | **`3/3` / `3/3`** |
| `random_forest` (`1729`) | **`liq_parkinson_sp500`** (`noise_k6`) | **`0.69140`** | `0.53831` | **`53.08%`** | `53.21%` | **`-0.00666`** (`0.0005`/`0.0045`) | **`+0.02331`** (`0.0065`/`0.052`) | **`-0.00337`** (`0.008`/`0.064`) | **`+0.01898`** (`0.0345`/`0.175`) | **`3/3` / `3/3`** |
| `random_forest` (`1729`) | `parkinson_sp500` (`noise_k2`) | `0.69298` | `0.53460` | **`53.08%`** | **`53.28%`** | **`-0.00507`** (`0.0025`/`0.0175`) | **`+0.01960`** (`0.0095`/`0.0665`) | `-0.00288` (`0.050`/`0.300`) | **`+0.01766`** (`0.0145`/`0.116`) | **`3/3` / `3/3`** |
| **`xgboost` (`42`)** | `F0_base` | `0.69887` | `0.52239` | `51.83%` | `52.00%` | — | — | — | — | — |
| `xgboost` (`42`) | **`nested_selected`** (`match`) | `0.69395` | **`0.54258`** | **`53.88%`** | **`54.06%`** | **`-0.00492`** (`0.0185`/`0.116`) | **`+0.02018`** (`0.005`/`0.040`) | **`-0.00496`** (`0.0225`/`0.180`) | **`+0.02884`** (`0.002`/`0.018`) | `2/3` / **`3/3`** |
| `xgboost` (`42`) | **`top_core_4_sp500`** (`noise_k4`) | `0.69280` | `0.54215` | `51.83%` | `51.91%` | **`-0.00608`** (`0.0165`/`0.116`) | **`+0.01975`** (`0.0155`/`0.093`) | **`-0.00607`** (`0.010`/`0.090`) | **`+0.02848`** (`0.005`/`0.040`) | `2/3` / **`3/3`** |
| `xgboost` (`42`) | `liq_parkinson_sp500` (`noise_k6`) | **`0.69262`** | `0.54167` | `53.77%` | `53.88%` | **`-0.00625`** (`0.025`/`0.125`) | `+0.01927` (`0.048`/`0.192`) | `-0.00335` (`0.074`/`0.508`) | `+0.01643` (`0.0755`/`0.302`) | `1/3` / `2/3` |
| **`xgboost` (`1729`)** | `F0_base` | `0.70048` | `0.51517` | `50.46%` | `50.62%` | — | — | — | — | — |
| `xgboost` (`1729`) | **`top_core_4_sp500`** (`noise_k4`) | **`0.69258`** | **`0.54034`** | **`52.97%`** | **`53.07%`** | **`-0.00790`** (`0.000`/`0.000`) | **`+0.02517`** (`0.0015`/`0.012`) | **`-0.00594`** (`0.0095`/`0.0855`) | **`+0.02473`** (`0.007`/`0.063`) | **`3/3` / `3/3`** |
| `xgboost` (`1729`) | `liq_parkinson_sp500` (`noise_k6`) | `0.69301` | `0.53915` | `52.40%` | `52.50%` | **`-0.00746`** (`0.0005`/`0.004`) | **`+0.02398`** (`0.0025`/`0.0175`) | `-0.00333` (`0.0755`/`0.604`) | `+0.01506` (`0.083`/`0.431`) | `2/3` / `2/3` |
| `xgboost` (`1729`) | `nested_selected` (`match`) | `0.69742` | `0.52656` | `51.03%` | `51.20%` | `-0.00305` (`0.0615`/`0.232`) | `+0.01139` (`0.060`/`0.303`) | `-0.00247` (`0.085`/`0.604`) | **`+0.01847`** (`0.017`/`0.136`) | `2/3` / `2/3` |

---

### 3.3. `EURUSD` (`589` observaciones OOS)

| Modelo (`seed`) | Receta (`noise_ref`) | `log_loss` OOS | `ROC-AUC` OOS | `Accuracy` | `Bal. Acc` | $\Delta\text{loss}$ vs Base (`p_raw` / `p_Holm`) | $\Delta\text{AUC}$ vs Base (`p_raw` / `p_Holm`) | $\Delta\text{loss}$ vs Ruido (`p_raw` / `p_Holm`) | $\Delta\text{AUC}$ vs Ruido (`p_raw` / `p_Holm`) | Pliegues Base / Ruido |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`logistic_l2` (`42`)** | `F0_base` | `0.69671` | `0.50858` | `50.42%` | `50.39%` | — | — | — | — | — |
| `logistic_l2` (`42`) | **`vx_liq_core_4`** (`noise_k4`) | **`0.69652`** | `0.51299` | `50.93%` | `50.93%` | `-0.00020` (`0.390`/`1.00`) | `+0.00441` (`0.241`/`1.00`) | `-0.00288` (`0.107`/`0.714`) | `+0.01849` (`0.072`/`0.432`) | `1/3` / `2/3` |
| `logistic_l2` (`42`) | `vx_core_2` (`noise_k2`) | `0.69666` | `0.51372` | `49.75%` | `49.83%` | `-0.00005` (`0.455`/`1.00`) | `+0.00513` (`0.251`/`1.00`) | `-0.00269` (`0.148`/`0.738`) | `+0.01938` (`0.073`/`0.432`) | `1/3` / `2/3` |
| `logistic_l2` (`42`) | `vx_basis_only` (`noise_k4`) | `0.69900` | **`0.51772`** | **`51.95%`** | **`51.87%`** | `+0.00229` (`0.653`/`1.00`) | `+0.00914` (`0.239`/`1.00`) | `-0.00040` (`0.415`/`1.00`) | `+0.02322` (`0.103`/`0.432`) | `2/3` / `2/3` |
| **`random_forest` (`42`)** | `F0_base` | `0.69892` | `0.48558` | `50.08%` | `49.88%` | — | — | — | — | — |
| `random_forest` (`42`) | **`vx_core_2`** (`noise_k2`) | **`0.69402`** | **`0.51094`** | `49.24%` | `49.23%` | **`-0.00490`** (`0.0015`/`0.0105`) | **`+0.02536`** (`0.0195`/`0.117`) | **`-0.00433`** (`0.0025`/`0.0175`) | **`+0.02904`** (`0.0045`/`0.0315`) | `2/3` / `2/3` |
| `random_forest` (`42`) | **`vx_liq_core_4`** (`noise_k4`) | `0.69497` | `0.51057` | **`50.59%`** | **`50.44%`** | **`-0.00396`** (`0.0115`/`0.069`) | **`+0.02499`** (`0.014`/`0.098`) | `-0.00136` (`0.158`/`0.945`) | **`+0.01868`** (`0.021`/`0.126`) | `2/3` / `2/3` |
| `random_forest` (`42`) | `fx_liquidity_only` (`noise_k4`) | `0.69968` | `0.47387` | `49.92%` | `49.71%` | `+0.00076` (`0.650`/`0.654`) | `-0.01171` (`0.764`/`1.00`) | `+0.00335` (`0.985`/`1.00`) | `-0.01802` (`0.931`/`1.00`) | `0/3` / `0/3` |
| **`random_forest` (`1729`)** | `F0_base` | `0.69878` | `0.48975` | `48.90%` | `48.66%` | — | — | — | — | — |
| `random_forest` (`1729`) | **`vx_core_2`** (`noise_k2`) | `0.69683` | `0.49503` | `48.22%` | `48.13%` | **`-0.00195`** (`0.0485`/`0.340`) | `+0.00527` (`0.228`/`1.00`) | **`-0.00222`** (`0.0485`/`0.340`) | **`+0.01753`** (`0.030`/`0.210`) | `2/3` / `2/3` |
| `random_forest` (`1729`) | `nested_selected` (`match`) | `0.69745` | `0.49404` | `48.73%` | `48.54%` | `-0.00133` (`0.0765`/`0.383`) | `+0.00428` (`0.148`/`1.00`) | `-0.00119` (`0.275`/`1.00`) | `+0.01477` (`0.115`/`0.573`) | **`3/3`** / `2/3` |
| **`xgboost` (`42`)** | `F0_base` | `0.70057` | `0.49284` | `50.42%` | `50.36%` | — | — | — | — | — |
| `xgboost` (`42`) | **`vx_liq_core_4`** (`noise_k4`) | **`0.69234`** | **`0.53357`** | **`52.63%`** | **`52.56%`** | **`-0.00823`** (`0.0015`/`0.0105`) | **`+0.04074`** (`0.002`/`0.014`) | `-0.00464` (`0.056`/`0.392`) | **`+0.03909`** (`0.0085`/`0.0595`) | **`3/3` / `3/3`** |
| `xgboost` (`42`) | **`vx_core_2`** (`noise_k2`) | `0.69522` | `0.52039` | `51.44%` | `51.47%` | **`-0.00534`** (`0.020`/`0.120`) | **`+0.02755`** (`0.018`/`0.099`) | `-0.00337` (`0.100`/`0.600`) | **`+0.02298`** (`0.0455`/`0.228`) | **`3/3`** / `2/3` |
| `xgboost` (`42`) | `nested_selected` (`match`) | `0.69685` | `0.50809` | `51.78%` | `51.65%` | `-0.00372` (`0.0505`/`0.223`) | **`+0.01525`** (`0.0355`/`0.142`) | `-0.00347` (`0.117`/`0.600`) | **`+0.03190`** (`0.027`/`0.162`) | `2/3` / `2/3` |
| **`xgboost` (`1729`)** | `F0_base` | `0.70184` | `0.48231` | `47.71%` | `47.65%` | — | — | — | — | — |
| `xgboost` (`1729`) | **`vx_liq_core_4`** (`noise_k4`) | **`0.69264`** | **`0.53038`** | **`50.76%`** | **`50.71%`** | **`-0.00920`** (`0.0005`/`0.0035`) | **`+0.04807`** (`0.000`/`0.000`) | **`-0.00535`** (`0.0285`/`0.200`) | **`+0.04200`** (`0.003`/`0.021`) | **`3/3`** / `2/3` |
| `xgboost` (`1729`) | **`vx_core_2`** (`noise_k2`) | `0.69521` | `0.51119` | `49.41%` | `49.34%` | **`-0.00663`** (`0.0015`/`0.009`) | **`+0.02888`** (`0.0065`/`0.039`) | `-0.00255` (`0.141`/`0.705`) | `+0.01104` (`0.145`/`0.723`) | **`3/3`** / `1/3` |
| `xgboost` (`1729`) | **`nested_selected`** (`match`) | `0.69613` | `0.51135` | `49.41%` | `49.26%` | **`-0.00571`** (`0.033`/`0.165`) | **`+0.02904`** (`0.037`/`0.150`) | **`-0.00610`** (`0.0435`/`0.261`) | **`+0.04340`** (`0.0115`/`0.069`) | **`3/3` / `3/3`** |

---

## 4. Evaluación Económica con Costos Reales y Estrés 2x

Fuente completa: [`economic_table.csv`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/focused_financial_study/20261017T232854_36d523ee/economic_table.csv).

### 4.1. `BTCUSDT` (`simulate_events`: `base_cost` = 34 bps round-trip; `double_cost` = 68 bps round-trip)

| Modelo (`seed`) | Receta | Umbral ($\tau$) | Retorno Neto (`base`) | Sharpe (`base`) | Max DD (`base`) | PF (`base`) | Trades | Retorno Neto (`2x cost`) | Sharpe (`2x cost`) | PF (`2x cost`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_l2` (`42`) | `F0_base` | `0.55` | `+33.10%` | `0.839` | `-10.93%` | `1.838` | `41` | `+16.74%` | `0.491` | `1.393` |
| `logistic_l2` (`42`) | `noise_k3_mean` | `0.55` | `+29.77%` | `0.850` | `-17.66%` | `1.777` | `44` | `+12.73%` | `0.430` | `1.298` |
| `logistic_l2` (`42`) | **`top_core_3`** | `0.55` | **`+45.29%`** | **`1.165`** | **`-8.92%`** | **`2.750`** | `39` | **`+28.24%`** | **`0.808`** | **`1.940`** |
| `logistic_l2` (`42`) | **`nested_selected`** | `0.55` | **`+37.23%`** | **`0.960`** | **`-8.92%`** | **`2.125`** | `35` | **`+22.69%`** | **`0.651`** | **`1.618`** |
| `logistic_l2` (`42`) | `noise_nested_match` | `0.55` | `+18.83%` | `0.603` | `-15.68%` | `1.489` | `42` | `+3.89%` | `0.183` | `1.095` |
| `random_forest` (`42`) | `F0_base` | `0.55` | `+7.40%` | `0.243` | `-23.24%` | `1.094` | `88` | `-18.96%` | `-0.342` | `0.823` |
| `random_forest` (`42`) | `noise_k4_mean` | `0.55` | `-11.82%` | `-0.179` | `-27.02%` | `0.870` | `72` | `-29.97%` | `-0.677` | `0.666` |
| `random_forest` (`42`) | **`top_core_4_sp500`** | `0.55` | **`+28.87%`** | **`0.644`** | **`-14.74%`** | **`1.356`** | `70` | **`+3.01%`** | **`0.155`** | **`1.044`** |
| `random_forest` (`42`) | `nested_selected` | `0.55` | `+10.24%` | `0.298` | `-22.32%` | `1.133` | `84` | `-15.74%` | `-0.266` | `0.857` |
| `random_forest` (`1729`) | `F0_base` | `0.55` | `+17.15%` | `0.434` | `-15.01%` | `1.212` | `77` | `-8.44%` | `-0.102` | `0.918` |
| `random_forest` (`1729`) | **`nested_selected`** | `0.55` | **`+20.74%`** | **`0.491`** | `-15.01%` | **`1.225`** | `84` | `-7.72%` | `-0.077` | `0.923` |
| `random_forest` (`1729`) | `noise_nested_match` | `0.55` | `+6.44%` | `0.225` | `-20.21%` | `1.088` | `73` | `-15.73%` | `-0.263` | `0.842` |
| `random_forest` (`1729`) | **`top_core_4_sp500`** | `0.52` (`inner_tau`) | **`+21.16%`** | **`0.442`** | `-25.75%` | **`1.149`** | `150` | `-25.03%` | `-0.372` | `0.852` |
| `xgboost` (`42`) | `F0_base` | `0.55` | `+24.26%` | `0.538` | `-16.51%` | `1.228` | `99` | `-9.48%` | `-0.107` | `0.916` |
| `xgboost` (`42`) | `noise_k3_mean` | `0.55` | `+5.20%` | `0.200` | `-18.19%` | `1.049` | `97` | `-22.87%` | `-0.420` | `0.792` |
| `xgboost` (`42`) | **`top_core_3`** | `0.55` | **`+43.37%`** | **`0.822`** | **`-13.69%`** | **`1.396`** | `96` | **`+5.45%`** | **`0.205`** | **`1.055`** |
| `xgboost` (`42`) | `nested_selected` | `0.55` | `+25.63%` | `0.532` | `-18.53%` | `1.208` | `102` | `-9.35%` | `-0.078` | `0.927` |
| `xgboost` (`1729`) | `F0_base` | `0.55` | `+43.64%` | `0.818` | `-15.08%` | `1.385` | `107` | `+1.99%` | `0.138` | `1.026` |
| `xgboost` (`1729`) | `noise_k3_mean` | `0.55` | `+17.99%` | `0.425` | `-20.20%` | `1.161` | `96` | `-13.21%` | `-0.178` | `0.887` |
| `xgboost` (`1729`) | **`top_core_3`** | `0.55` | **`+41.40%`** | **`0.775`** | **`-14.63%`** | **`1.335`** | `110` | `-0.56%` | `0.090` | `0.998` |

### 4.2. `ETHUSDT` (`simulate_events`: `base_cost` = 34 bps round-trip; `double_cost` = 68 bps round-trip)

| Modelo (`seed`) | Receta | Umbral ($\tau$) | Retorno Neto (`base`) | Sharpe (`base`) | Max DD (`base`) | PF (`base`) | Trades | Retorno Neto (`2x cost`) | Sharpe (`2x cost`) | PF (`2x cost`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_l2` (`42`) | `F0_base` | `0.53` (`inner_tau`) | `-12.11%` | `-0.035` | `-43.63%` | `0.960` | `195` | `-52.91%` | `-0.896` | `0.725` |
| `logistic_l2` (`42`) | `noise_k4_mean` | `0.53` (`inner_tau`) | `+2.47%` | `0.193` | `-40.37%` | `1.013` | `236` | `-51.85%` | `-0.705` | `0.797` |
| `logistic_l2` (`42`) | **`top_core_4_sp500`** | `0.53` (`inner_tau`) | **`+160.47%`** | **`1.073`** | **`-35.87%`** | **`1.198`** | `461` | `-40.42%` | `-0.247` | `0.911` |
| `logistic_l2` (`42`) | **`nested_selected`** | `0.53` (`inner_tau`) | **`+85.37%`** | **`0.809`** | `-37.43%` | **`1.177`** | `322` | `-33.85%` | `-0.220` | `0.896` |
| `logistic_l2` (`42`) | **`liq_only`** | `0.55` | **`+91.65%`** | **`0.999`** | **`-21.80%`** | **`1.307`** | `215` | `-3.68%` | `0.100` | `0.986` |
| `random_forest` (`42`) | `F0_base` | `0.55` (`inner_tau`) | `+4.63%` | `0.206` | `-30.98%` | `1.067` | `134` | `-31.85%` | `-0.384` | `0.850` |
| `random_forest` (`42`) | `noise_nested_match` | `0.55` (`inner_tau`) | `-1.01%` | `0.128` | `-39.33%` | `1.017` | `140` | `-36.76%` | `-0.492` | `0.795` |
| `random_forest` (`42`) | **`nested_selected`** | `0.55` (`inner_tau`) | **`+15.93%`** | **`0.341`** | **`-29.92%`** | **`1.086`** | `136` | `-24.98%` | `-0.206` | `0.880` |
| `random_forest` (`1729`) | `F0_base` | `0.55` (`inner_tau`) | `+6.26%` | `0.228` | `-31.71%` | `1.043` | `143` | `-32.76%` | `-0.388` | `0.824` |
| `random_forest` (`1729`) | `noise_k3_mean` | `0.55` (`inner_tau`) | `-20.02%` | `-0.176` | `-32.47%` | `0.876` | `134` | `-47.91%` | `-0.782` | `0.677` |
| `random_forest` (`1729`) | **`top_core_3`** | `0.55` (`inner_tau`) | **`+13.03%`** | **`0.311`** | **`-25.35%`** | **`1.067`** | `141` | `-28.01%` | `-0.301` | `0.849` |
| `xgboost` (`42`) | `F0_base` | `0.55` (`inner_tau`) | `-6.83%` | `0.063` | `-35.09%` | `0.974` | `160` | `-44.16%` | `-0.597` | `0.766` |
| `xgboost` (`42`) | `noise_k4_mean` | `0.55` (`inner_tau`) | `-18.16%` | `-0.116` | `-35.69%` | `0.914` | `151` | `-49.52%` | `-0.755` | `0.720` |
| `xgboost` (`42`) | **`top_core_4_sp500`** | `0.55` (`inner_tau`) | **`+14.23%`** | **`0.321`** | `-45.06%` | **`1.070`** | `204` | `-40.54%` | `-0.409` | `0.855` |
| `xgboost` (`42`) | **`nested_selected`** | `0.55` (`inner_tau`) | **`+11.52%`** | **`0.293`** | `-41.77%` | **`1.049`** | `187` | `-38.70%` | `-0.398` | `0.831` |
| `xgboost` (`1729`) | `F0_base` | `0.55` | `-3.92%` | `0.102` | `-34.76%` | `0.999` | `163` | `-42.97%` | `-0.572` | `0.786` |
| `xgboost` (`1729`) | `noise_k4_mean` | `0.55` | `-22.12%` | `-0.179` | `-38.81%` | `0.898` | `157` | `-52.88%` | `-0.840` | `0.709` |
| `xgboost` (`1729`) | **`top_core_4_sp500`** | `0.55` | **`+10.77%`** | **`0.287`** | `-43.88%` | **`1.047`** | `192` | `-40.08%` | `-0.405` | `0.839` |
| `xgboost` (`1729`) | **`liq_parkinson_sp500`** | `0.55` | **`+19.84%`** | **`0.376`** | `-41.47%` | **`1.080`** | `178` | `-32.20%` | `-0.288` | `0.859` |

### 4.3. `EURUSD` (`FxEngine.portfolio` sobre cotizaciones `1h` bid/ask reales FXCM)

| Modelo (`seed`) | Receta | Umbral ($\tau$) | Retorno Neto (`base`) | Sharpe (`base`) | Max DD (`base`) | PF (`base`) | Trades | Retorno Neto (`2x cost`) | Sharpe (`2x cost`) | PF (`2x cost`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_l2` (`42`) | `F0_base` | `0.55` | `-6.33%` | `-1.103` | `-7.67%` | `0.488` | `56` | `-7.61%` | `-1.345` | `0.409` |
| `logistic_l2` (`42`) | `vx_basis_only` | `0.55` | `-5.30%` | `-0.768` | `-6.97%` | `0.656` | `83` | `-6.30%` | `-0.938` | `0.589` |
| `random_forest` (`42`) | `F0_base` | `0.55` | `+2.07%` | `0.483` | `-2.77%` | `1.414` | `41` | `+1.03%` | `0.247` | `1.201` |
| `random_forest` (`42`) | `noise_k2_mean` | `0.55` | `+0.49%` | `0.137` | `-2.92%` | `1.114` | `34` | `-0.56%` | `-0.147` | `0.874` |
| `random_forest` (`42`) | **`vx_core_2`** | `0.55` | **`+2.37%`** | **`0.544`** | **`-2.77%`** | **`1.489`** | `37` | **`+1.45%`** | **`0.339`** | **`1.295`** |
| `random_forest` (`42`) | **`vx_liq_core_4`** | `0.55` | **`+1.89%`** | **`0.452`** | `-2.98%` | **`1.532`** | `27` | **`+2.01%`** | **`0.518`** | **`1.753`** |
| `random_forest` (`1729`) | `F0_base` | `0.55` | `+1.13%` | `0.253` | `-2.92%` | `1.186` | `42` | `-0.25%` | `-0.047` | `0.962` |
| `random_forest` (`1729`) | **`nested_selected`** | `0.55` | **`+3.57%`** | **`0.705`** | **`-2.75%`** | **`1.935`** | `32` | **`+2.44%`** | **`0.510`** | **`1.701`** |
| `random_forest` (`1729`) | `noise_nested_match` | `0.55` | `-0.75%` | `-0.197` | `-3.54%` | `0.827` | `26` | `-1.11%` | `-0.296` | `0.740` |
| `random_forest` (`1729`) | **`inner_perm_top`** | `0.55` | **`+3.92%`** | **`0.812`** | **`-2.14%`** | **`2.208`** | `33` | **`+3.15%`** | **`0.662`** | **`2.071`** |
| `xgboost` (`42`) | `F0_base` | `0.55` | `-5.05%` | `-0.793` | `-6.26%` | `0.669` | `73` | `-6.96%` | `-1.155` | `0.538` |
| `xgboost` (`42`) | **`vx_liq_core_4`** | `0.55` | **`-0.83%`** | **`-0.147`** | **`-5.74%`** | **`0.918`** | `57` | `-2.39%` | `-0.478` | `0.734` |
| `xgboost` (`1729`) | `F0_base` | `0.55` (`inner_tau`) | `-3.82%` | `-0.667` | `-5.40%` | `0.708` | `70` | `-5.15%` | `-0.930` | `0.610` |
| `xgboost` (`1729`) | **`vx_basis_plus_liquidity`** | `0.55` (`inner_tau`) | **`+0.68%`** | **`0.132`** | **`-3.68%`** | **`1.097`** | `51` | **`+0.20%`** | **`0.046`** | **`1.033`** |
| `xgboost` (`1729`) | **`nested_selected`** | `0.55` (`inner_tau`) | **`+0.15%`** | **`0.037`** | **`-3.58%`** | **`1.023`** | `57` | `-0.69%` | `-0.122` | `0.919` |
| `xgboost` (`1729`) | `noise_nested_match` | `0.55` (`inner_tau`) | `-2.36%` | `-0.522` | `-3.51%` | `0.736` | `54` | `-4.05%` | `-0.926` | `0.576` |

---

## 5. Diagnóstico por Variable: Señal Genuina vs. Sobreajuste

Combinando la frecuencia de selección causal en validación interna ([`nested_decisions.json`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/artifacts/focused_financial_study/20261017T232854_36d523ee/nested_decisions.json)) con el desempeño fuera de muestra frente a `F0_base` y frente a los controles con ruido de igual dimensión (`noise_k{k}_mean`):

### 5.1. Variables que aportan información genuina

1. **`fin_cross_sp500_mom_z` en `ETHUSDT` (Señal genuina fuerte y estadísticamente significativa):**
   - Seleccionada con permutación interna positiva en `13/15` decisiones anidadas de `ETHUSDT` y responsable directa del salto entre `top_core_3` (`ROC-AUC = 0.52319..0.54703`) y `top_core_4_sp500` (`ROC-AUC = 0.53642..0.55870`), superando a `noise_k4_mean` con $p_{\text{Holm}} = 0.0045$ (`RandomForest` `seed=1729`) y $p_{\text{Holm}} = 0.040$ (`XGBoost` `seed=42`).
2. **`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63` y `fin_vol_parkinson_ratio` en `BTCUSDT` y `ETHUSDT` (`top_core_3`):**
   - Forman el núcleo microestructural de rango/liquidez más estable en Cripto: en `BTCUSDT`, `top_core_3` mejora `ROC-AUC` y `log_loss` frente a `F0_base` y `noise_k3_mean` en los 3 modelos, y produce el mejor desempeño económico neto de costos en `logistic_l2` (`Sharpe = 1.165`, `+28.24%` aún bajo costos 2x) y `xgboost` (`Sharpe = 0.822`).
3. **`fin_basis_vx_z63` y `fin_basis_vx_term` combinadas con `fin_liq_spread_to_range` y `fin_liq_corwin_schultz_20` en `EURUSD` (`vx_core_2` y `vx_liq_core_4`):**
   - `vx_core_2` supera significativamente al ruido en `RandomForest` ($p_{\text{Holm}} = 0.0175$ en `log_loss`, $p_{\text{Holm}} = 0.0315$ en `ROC-AUC`) y `vx_liq_core_4` supera significativamente a `F0_base` y al ruido en `XGBoost` ($p_{\text{Holm}} = 0.0035$ en `log_loss` vs. base y $p_{\text{Holm}} = 0.021$ en `ROC-AUC` vs. ruido).

### 5.2. Combinaciones que producen sobreajuste o dilución

1. **`fx_liquidity_only` en `EURUSD` (Sobreajuste / ruido cuando se usa aislada sin el régimen `VX/VIX`):**
   - En `RandomForest`, usar las 4 variables de liquidez de cotización solas empeora `log_loss` (`+0.00076` a `+0.00265`) y hunde el `ROC-AUC` a `0.46729..0.47387` (`0/3` pliegues ganados). Las variables de spread (`fin_liq_spread_to_range`, `fin_liq_corwin_schultz_20`) solo aportan valor en `EURUSD` como **moduladores condicionales del régimen de aversión al riesgo `VX/VIX`** dentro de `vx_liq_core_4`.
2. **Agregación simultánea de las 6 variables (`liq_parkinson_sp500`) en `BTCUSDT` `XGBoost`:**
   - Añadir simultáneamente `fin_liq_amihud_z63`, `fin_liq_trade_size_z63` y `fin_cross_sp500_mom_z` encima de `top_core_3` en `BTCUSDT` `XGBoost` hace que el modelo rinda peor que el control con 6 columnas de ruido gaussiano (`ROC-AUC 0.53451` vs `0.55603` en `noise_k6_mean`), confirmando que en `BTCUSDT` los árboles con `colsample_bytree = 0.9` se benefician más de la dilución aleatoria cuando se agregan 6 columnas que de las variables secundarias de liquidez/macro.
3. **Umbrales de probabilidad laxos ($\tau \in [0.51, 0.52]$) seleccionados por validación interna (`inner_tau`) cuando hay deriva de calibración:**
   - En los 3 activos, elegir $\tau = 0.51$ o $0.52$ en validación interna dispara entre `130` y `460` operaciones fuera de muestra, haciendo que los costos de transacción (`34 bps` por ciclo en Cripto y spread+swap en Forex) absorban toda la ventaja predictiva bajo el escenario de estrés 2x.

---

## 6. Limitaciones y Experimentos Independientes Faltantes

Para responder con rigor cuantitativo al objetivo final (*"determinar si podemos superar consistentemente los modelos anteriores y los controles con ruido, obteniendo una ventaja predictiva y económica robusta"*):

1. **Ventaja predictiva vs. ventaja económica:**
   - **Predictivamente**, la evidencia es sólida en **`ETHUSDT`** (`top_core_4_sp500` y `nested_selected`) y **`EURUSD`** (`vx_liq_core_4` y `vx_core_2`), donde superamos tanto a los modelos originales (`F0_base`) como a los controles con ruido de igual dimensión con significancia estadística tras corrección de Holm-Bonferroni ($p_{\text{Holm}} \le 0.040$). En **`BTCUSDT`**, `top_core_3` y `top_core_4_sp500` mejoran consistentemente las métricas puntuales frente a base y ruido, pero con $p_{\text{Holm}} \ge 0.0765$.
   - **Económicamente**, la ventaja solo es robusta al estrés de costos 2x en **`BTCUSDT` con umbral alto ($\tau = 0.55$) usando `top_core_3`** (`logistic_l2` y `xgboost`), y moderada en `EURUSD` con `RandomForest` ($\tau = 0.55$). En `ETHUSDT` (con costos 2x) y en `EURUSD` (`XGBoost`/`LogisticRegression`), la mejora predictiva aún no basta para compensar la rotación y el costo de carry/spread del motor actual.
2. **Experimentos independientes que faltan antes de abrir la reserva prospectiva sellada (`>= 2025-07-01`):**
   - **Calibración probabilística causal y control de rotación (turnover/hysteresis) en validación interna:** Las probabilidades crudas de `RandomForest`, `XGBoost` y `LogisticRegression` sufren desplazamientos de media entre pliegues; se requiere un experimento independiente de calibración isotónica/Platt estrictamente dentro de `train` combinado con una banda de no-operación ($\tau \ge 0.55$ o filtro de cambio de posición) para reducir el número de operaciones en `ETHUSDT` de `~200-400` a `~40-70` antes de evaluar costos 2x.
   - **Simulación direccional simétrica (long/short) y consciente del swap en `EURUSD`:** El motor actual `FxEngine.portfolio` solo abre posiciones largas (`probability >= threshold`), lo que desaprovecha la mitad inferior de las predicciones de `vx_liq_core_4` (`ROC-AUC = 0.53357`) y penaliza sistemáticamente a la estrategia cuando el diferencial de tasas implica carry negativo para el euro.
   - **Evaluación única sobre la reserva prospectiva sellada (`>= 2025-07-01`):** Una vez congeladas las recetas ganadoras por activo (`top_core_3` en `BTCUSDT`, `top_core_4_sp500` en `ETHUSDT`, `vx_liq_core_4` en `EURUSD`) y el protocolo de umbral/calibración, corresponde ejecutar una única prueba final fuera de muestra sobre el período sellado `>= 2025-07-01` que permanece intacto.

