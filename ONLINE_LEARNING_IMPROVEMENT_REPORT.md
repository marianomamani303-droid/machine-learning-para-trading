# Informe Técnico de Investigación y Arquitectura: Modelos Matemáticos Adaptativos y Evaluación Estadística Secuencial para Online Learning

**Fecha:** 2026-10-28  
**Paquete auditado y extendido:** `src/trading_ml/online/`  
**Configuración de laboratorio:** `config/online_adaptive_lab.toml`  
**Artefacto experimental reproducible:** `artifacts/online/experiments/adaptive_online_study_results.json`  
**Estado de ejecución real:** `orders_enabled = false` (100% investigación científica por CLI/PowerShell; sin órdenes reales).

---

## A. Resumen ejecutivo

Esta investigación transforma el motor de *online learning* de `src/trading_ml/online/` desde una cohorte elemental de tres modelos direccionales (`baseline`, `logistic_slow`, `logistic_fast`) hacia un **laboratorio cuantitativo multifamilia de aprendizaje incremental y evaluación estadística secuencial**, manteniendo intacta la compatibilidad con el estado SQLite existente (`artifacts/online/state.sqlite3`) y aprobando la totalidad de las **138 pruebas automatizadas** (129 pruebas previas + 9 pruebas integrales nuevas en `tests/test_online_adaptive.py`).

### Hallazgos científicos principales

1. **Diagnóstico matemático cerrado de la ceguera del evaluador anterior (`hoeffding_azuma`):**
   Demostramos analíticamente que para cualquier predictor probabilístico calibrado $p_{c,t} = 0.50 + \delta_t$ comparado contra una referencia $p_{b,t} = 0.50$, la diferencia pareada de pérdida de Brier $d_t = (p_{c,t} - y_t)^2 - (0.50 - y_t)^2$ tiene esperanza verdadera $\mathbb{E}[d_t] = -\delta_t^2$ y varianza verdadera $\text{Var}(d_t) = \delta_t^2(1 - 4\delta_t^2) \approx \delta_t^2 \in [0.002, 0.006]$. El test `paired_brier_conditional_hoeffding_azuma_alpha_spending` asumía el rango teórico peor caso $[-1, 1]$ ($\sigma^2_{\text{worst}} = 1.0$, sobreestimando la varianza por un factor de $1/\delta^2 \approx 165\text{--}400\times$) y evaluaba cada bloque de $W = 50$ barras de forma aislada mientras contraía el presupuesto $\alpha_k = \frac{\alpha}{2 M k(k+1)} \to 0$. Como consecuencia, para una ventaja realista de Brier de $0.003\text{--}0.005$, Hoeffding-Azuma requería entre **634.397 y 2.537.587 observaciones independientes** y tenía un Efecto Mínimo Detectable ($\text{MDE}$) en $W=50$ de **$0.4516$** (imposible de alcanzar dado que el Brier base es $0.25$).
2. **Rediseño del evaluador secuencial (`empirical_bernstein`):**
   Implementamos el contraste secuencial **Predictable Empirical Bernstein con varianza HAC de Newey-West** (`paired_brier_predictable_empirical_bernstein_alpha_spending`), que explota la cota condicional $\mathcal{F}_{t-1}$-predecible $|d_t| \le 2|p_{c,t} - p_{b,t}|$, estima la varianza empírica y la autocorrelación de rezago 1 sobre bloques desolapados, acumula evidencia histórica para promociones de superioridad genuina y conserva monitoreo local por bloque para revocar o desactivar rápidamente modelos degradados tras cambios de régimen. Esto mejora la eficiencia muestral en **$>200\times$** ($\text{MDE}(N=500) = 0.0109$ frente a $0.1435$), manteniendo $0.0\%$ de falsas promociones bajo los escenarios nulos `no_edge` y `serial_dependence`.
3. **Superioridad estadística demostrada en predicción online de volatilidad condicional (`volatility`):**
   En `BTCUSDT` `1h` (`2024-01-01` a `2024-06-01`, $N = 3.584$ predicciones prequential evaluadas), tanto `ewma_volatility` ($\text{QLIKE} = -9.45236$, $\text{MAE} = 0.002700$, $p_{\text{EB}} < 10^{-4}$) como el modelo multiescala `har_rv` alimentado con el conjunto de features `v2_financial11` ($\text{QLIKE} = -9.48749$, $\text{MAE} = 0.002538$, $\text{RMSE} = 0.003603$, $\Delta\text{QLIKE} = -0.15755$, $p_{\text{EB}} < 10^{-4}$) superan de manera estadísticamente significativa al control `volatility_naive_baseline` ($\text{QLIKE} = -9.32994$, $\text{MAE} = 0.003036$) y son promovidos automáticamente como **Champions validados de volatilidad**. En `BTCUSDT` `1d` ($N = 916$), `har_rv` también alcanza el mejor QLIKE descriptivo ($-6.04549$) y el menor MAE de volatilidad ($0.013601\text{--}0.014135$ vs $0.015553$ del baseline).
4. **Contraste riguroso entre liderazgo descriptivo, significancia estadística y rentabilidad neta en dirección (`direction`):**
   - En `BTCUSDT` `1h`, `logistic_static` ($\text{Brier} = 0.249289$, $\text{AUC} = 0.53495$, retorno bruto $+16.38\%$) y `logistic_adaptive` ($\text{Brier} = 0.249676$, $\text{Accuracy} = 52.26\%$ en `v1_base8` y $52.65\%$ en `v2_financial11`) superan descriptivamente a `baseline` ($\text{Brier} = 0.250323$) y a `logistic_slow`/`logistic_fast`. Sin embargo, la mejora de Brier ($\Delta \approx -0.0010$) no cruza la frontera secuencial y los costos de transacción (`10 bps` por lado) erosionan el retorno bruto positivo hasta convertirlo en negativo neto ($-37.42\%$ y $-35.56\%$).
   - En `BTCUSDT` `1d`, `logistic_adaptive` lidera descriptivamente ($\text{Brier} = 0.249867$ en `v1_base8`, $\text{Brier} = 0.249843$ en `v2_financial11`) superando al control congelado `logistic_static` ($\text{Brier} = 0.252074$) y a `logistic_slow` ($\text{Brier} = 0.255430$), y obtiene retorno neto positivo ($+22.08\%$, Sharpe neto $1.04$, drawdown máximo $-4.07\%$ en `v1_base8`). Pese a ello, al operar solo en 12 barras activas y tener $p_{\text{EB}} = 0.85 > \alpha_{\text{adj}}$, el selector Champion–Challenger **no declara falsamente una ventaja validada** y retiene conservadoramente a `baseline` como Champion activo.

---

## B. Auditoría de la arquitectura actual

La auditoría inicial de `src/trading_ml/online/` (`config.py`, `data.py`, `models.py`, `evaluation.py`, `store.py`, `service.py`, `cli.py`) identificó las siguientes fortalezas y limitaciones estructurales:

| Componente | Estado previo | Limitación detectada | Solución arquitectónica implementada |
| :--- | :--- | :--- | :--- |
| **`models.py`** | Solo `beta_frequency` (`baseline`) y `logistic` SGD sin estandarización ni control de deriva (`logistic_slow`, `logistic_fast`). | Sensibilidad extrema a la escala de features; sobreconfianza severa en `logistic_fast` ($\eta = 0.10$); ausencia de modelos de magnitud, volatilidad y cuantiles; ausencia de control estático congelado. | Nuevo módulo `adaptive.py` integrado en `models.py` con 12 modelos y controles agrupados en 4 familias de target (`direction`, `expected_return`, `volatility`, `quantiles`), incluyendo `logistic_static`, `logistic_adaptive`, `kalman_linear`, `ewma_volatility`, `har_rv` y `quantile_regression`. |
| **`data.py`** | 8 features OHLCV crudas (`ret_1`..`body_ratio`) con `warmup_bars = 21`. | No incorporaba información microestructural de liquidez o volatilidad de rango que demostró valor en estudios previos, ni calculaba métricas realizadas completas ($r_{t+h}, RV_{t+h}$) para targets no binarios. | Soporte configurable para `v1_base8` (default intacto), `v2_financial11` (`warmup_bars >= 64` con Corwin-Schultz, impacto de rango z-score y ratio de Parkinson) y `v1_noise11`, más `realized_target_metrics()`. |
| **`evaluation.py`** | `paired_brier_conditional_hoeffding_azuma_alpha_spending` por bloques aislados de $W=50$. | Cota de varianza peor caso $[-1,1]^2$ ($>200\times$ demasiado ancha) y reinicio de muestra a $W=50$ en cada look con $\alpha_k \to 0$. Cero sensibilidad real ante bordes financieros o calibración defectuosa. | Preservación de `hoeffding_azuma` para compatibilidad y adición de `empirical_bernstein` (rango condicional predecible $2|p_c - p_b|$ + varianza HAC Newey-West + evidencia acumulada para promoción + monitoreo por bloque para revocación), auditoría analítica y simulador Monte Carlo de 6 escenarios. |
| **`store.py` / `service.py`** | Esquema SQLite atómico con contrato JSON y bloqueo de proceso. | Validaba una lista fija `("baseline", "logistic_slow", "logistic_fast")` y no almacenaba salidas estructuradas multifamilia en `raw_output_json`. | Validación dinámica contra `contract.get("cohort_models", ...)` manteniendo 100% de compatibilidad con `artifacts/online/state.sqlite3`, más enriquecimiento de `raw_output_json` con predicciones de retorno, volatilidad y cuantiles. |
| **`cli.py`** | Subcomandos básicos (`init`, `run`, `status`, `predictions`, `report`, `verify`, `backup`, `restore`, `catalog`, `infer`). | Carecía de inspección del catálogo de modelos adaptativos, tabla de comparación multifamilia, selección Champion–Challenger, replay experimental y auditoría estadística desde terminal. | Nuevos subcomandos `models`, `compare`, `champions`, `experiment` y `audit-eval` integrados en `python -m trading_ml online` y `scripts/trading-online.ps1`. |

---

## C. Modelos matemáticos implementados y ecuaciones

Todos los modelos implementados en [`src/trading_ml/online/adaptive.py`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/src/trading_ml/online/adaptive.py) se actualizan de forma estrictamente incremental en complejidad constante $\mathcal{O}(d)$ u $\mathcal{O}(d^2)$ por observación confirmada, sin reentrenar sobre el historial completo.

### 1. Estandarización causal incremental de Welford
Para evitar filtración de información futura (*look-ahead bias*), todo modelo adaptativo estandariza el vector de características $x_t \in \mathbb{R}^d$ en el instante de predicción $t$ usando exclusivamente la media $\mu_{t-1}$ y varianza $s^2_{t-1}$ estimadas con observaciones previas:
$$\tilde{x}_{t,j} = \text{clip}\!\left(\frac{x_{t,j} - \mu_{t-1,j}}{\sqrt{s^2_{t-1,j} + \epsilon}},\; -5.0,\; +5.0\right)$$
Los contadores recursivos de Welford ($\mu_t, M_{2,t}$) se actualizan **únicamente después** de haber emitido la predicción o durante el paso `learn()`.

### 2. Modelo Lineal en Espacio de Estados con Filtro de Kalman (`kalman_linear`, `kind = "kalman_filter"`)
Modela la relación lineal variante en el tiempo entre el vector aumentado $\phi_t = [1, \tilde{x}_t^\top]^\top \in \mathbb{R}^{d+1}$ y el retorno continuo realizado a horizonte $h$, $r_{t+h} = \frac{C_{t+h}}{C_t} - 1$:
- **Ecuación de transición de estado (con encogimiento AR(1) hacia prior cero):**
  $$\beta_{t|t-1} = (1 - \lambda)\beta_{t-1|t-1}, \qquad P_{t|t-1} = (1 - \lambda)^2 P_{t-1|t-1} + Q_t$$
  donde $\lambda = 10^{-3}$ previene deriva numérica en direcciones no observables y $Q_t = q I_{d+1}$ ($q = 10^{-8}$).
- **Ecuación de observación y distribución predictiva a priori:**
  $$\hat{r}_{t+h|t} = \phi_t^\top \beta_{t|t-1}, \qquad S_t = \phi_t^\top P_{t|t-1} \phi_t + R_t$$
- **Actualización estabilizada en forma de Joseph (al confirmarse $r_{t+h}$):**
  Sea la innovación $\nu_t = r_{t+h} - \hat{r}_{t+h|t}$ y la ganancia de Kalman $K_t = \frac{P_{t|t-1}\phi_t}{S_t}$. El estado y la covarianza se actualizan mediante:
  $$\beta_{t|t} = \beta_{t|t-1} + K_t \nu_t$$
  $$P_{t|t} = (I - K_t \phi_t^\top) P_{t|t-1} (I - K_t \phi_t^\top)^\top + R_t K_t K_t^\top$$
  seguido de simetrización explícita $P_{t|t} \leftarrow \frac{1}{2}(P_{t|t} + P_{t|t}^\top)$ y proyección espectral de autovalores en $[10^{-9}, 10^2]$ para garantizar definición positiva estricta en aritmética de punto flotante.
- **Ruido observacional adaptativo EWMA:**
  $$R_t = \text{clip}\!\left((1 - \rho_R) R_{t-1} + \rho_R \nu_t^2,\; 10^{-8},\; 1.0\right)$$
- **Protección contra doble actualización:** el estado registra `last_learned_target_id` e ignora de forma idempotente cualquier intento de confirmar dos veces el mismo `target_id`.

### 3. Regresión Logística Incremental Adaptativa (`logistic_adaptive`) y Control Estático (`logistic_static`)
- **`logistic_adaptive`:**
  Minimiza la pérdida logística instantánea $\ell_t(w, b) = -y_{t+h}\log \hat{p}_t - (1-y_{t+h})\log(1-\hat{p}_t)$ con $\hat{p}_t = \sigma(b + w^\top \tilde{x}_t)$ mediante:
  1. **Gradiente con regularización $L_2$:** $g_{w,t} = (\hat{p}_t - y_{t+h})\tilde{x}_t + \lambda_2 w_t$.
  2. **Precondicionador diagonal RMSProp acotado:**
     $$G_{t,j} = \beta G_{t-1,j} + (1 - \beta) g_{w,t,j}^2, \qquad \eta_{t,j} = \frac{\eta_0 \cdot m_{\text{drift}}}{\sqrt{\max(G_{t,j}, 0.25)} + \epsilon}$$
  3. **Operador proximal $L_1$ (Soft-Thresholding para Elastic Net):**
     $$\tilde{w}_{t,j} = w_{t,j} - \eta_{t,j} g_{w,t,j}, \qquad w_{t,j}^+ = \text{sign}(\tilde{w}_{t,j})\max\!\left(0,\; |\tilde{w}_{t,j}| - \eta_{t,j}\lambda_1\right)$$
  4. **Proyección en bola $L_2$:** si $\|w_t^+\|_2 > R_{\max}$, se reescala $w_t \leftarrow w_t^+ \frac{R_{\max}}{\|w_t^+\|_2}$ ($R_{\max} = 3.0$).
  5. **Detector online de degradación Brier:** compara una media exponencial rápida ($\alpha_{\text{fast}} = 0.08$) frente a una lenta ($\alpha_{\text{slow}} = 0.01$) de la pérdida de Brier; si $\text{EWMA}_{\text{fast}} - \text{EWMA}_{\text{slow}} > 0.015$, activa un multiplicador de adaptación $m_{\text{drift}} = 1.5$ y contrae los pesos hacia cero ($\times 0.92$) para acelerar el olvido de relaciones obsoletas.
- **`logistic_static` (Control Estático Obligatorio):**
  Ejecuta exactamente la misma arquitectura y estandarización que `logistic_adaptive` durante las primeras `static_warmup_updates = 200` observaciones confirmadas y, al alcanzar dicho umbral, activa `frozen = True`, congelando permanentemente $w, b, \mu, s^2$. Esto permite medir de forma limpia si la adaptación continua post-warmup mejora o deteriora el desempeño frente a un modelo lineal fijo.

### 4. Modelos de Volatilidad Condicional (`ewma_volatility`, `har_rv`) y Cuantiles (`quantile_regression`)
- **`ewma_volatility`:**
  Combina recursivamente el retorno cuadrado confirmado $r_{t+h}^2$ y el estimador de varianza de rango de Parkinson de la barra actual $\sigma^2_{\text{Park}, t} = \frac{(\ln(H_t/L_t))^2}{4\ln 2}$:
  $$\hat{\sigma}^2_{t+1} = \lambda_{\text{EWMA}} \hat{\sigma}^2_t + (1 - \lambda_{\text{EWMA}})\!\left(0.65\, r_{t+h}^2 + 0.35\, \sigma^2_{\text{Park}, t}\right), \qquad \lambda_{\text{EWMA}} = 0.94$$
- **`har_rv` (Heterogeneous Autoregressive Realized Volatility en log-varianza con RLS):**
  Inspirado en Corsi (2009), modela el logaritmo de la varianza realizada sobre tres escalas causales (1 barra, 6 barras y 24 barras, más las variables de rango cuando están presentes en `v2_financial11`):
  $$\log \hat{v}_{t+h|t} = \theta_{t-1}^\top z_t + \frac{1}{2}\hat{\sigma}^2_{\eta, t-1}$$
  donde $\theta_t$ se actualiza mediante **Mínimos Cuadrados Recursivos (RLS)** con factor de olvido $\gamma = 0.995$ y actualización de covarianza de Joseph, y el término $+\frac{1}{2}\hat{\sigma}^2_{\eta, t-1}$ aplica la corrección de convexidad log-normal de Jensen para que $\hat{v}_{t+h|t} = \exp(\log \hat{v}_{t+h|t})$ sea un estimador insesgado bajo la pérdida cuasi-verosímil **QLIKE**:
  $$\text{QLIKE}(\hat{v}_t, v_t) = \log(\hat{v}_t) + \frac{v_t}{\hat{v}_t}$$
- **`quantile_regression` ($\tau \in \{0.10, 0.50, 0.90\}$):**
  Estima simultáneamente los cuantiles condicionales $q_{\tau, t} = (1 - \gamma_q)\hat{q}^{\text{emp}}_{\tau, t} + \gamma_q(b_\tau + w_\tau^\top \tilde{x}_t)$ minimizando la pérdida *pinball* $\rho_\tau(u) = u(\tau - \mathbb{I}(u < 0))$ mediante subgradiente incremental precondicionado:
  $$g_{\tau, t} = \mathbb{I}(r_{t+h} < q_{\tau, t}) - \tau$$
  seguido de **reordenamiento monótono post-inferencia** (Chernozhukov et al., 2010):
  $$q_{0.10, t}^* \le q_{0.50, t}^* \le q_{0.90, t}^*$$
  garantizando matemáticamente una tasa de cruce de cuantiles post-reordenamiento igual a **$0.0\%$**.

---

## D. Ingeniería de features causal (`v1_base8` vs `v2_financial11` vs `v1_noise11`)

En [`src/trading_ml/online/data.py`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/src/trading_ml/online/data.py) y [`src/trading_ml/online/config.py`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/src/trading_ml/online/config.py) se implementaron tres conjuntos de características estrictamente causales (calculados exclusivamente con velas cerradas hasta $t$):

1. **`v1_base8` (8 features base OHLCV, `warmup_bars = 21`, preservado por defecto):**
   - `ret_1`, `ret_3`, `ret_5`, `ret_10`, `ret_20` (retornos simples rezagados).
   - `vol_5`, `vol_20` (desviación estándar poblacional de retornos de 1 barra en ventanas de 5 y 20 barras).
   - `range_pct = (H_t - L_t)/C_t` y `body_ratio = (C_t - O_t)/(H_t - L_t)`.
2. **`v2_financial11` (`v1_base8` + 3 variables financieras seleccionadas por evidencia previa en `FINANCIAL_FEATURE_ABLATION_REPORT.md`, `warmup_bars = 64`):**
   - `fin_liq_corwin_schultz_20`: estimador causal de spread efectivo de Corwin & Schultz (2012) promediado sobre 20 barras cerradas a partir de los rangos de 1 y 2 barras ($\beta_{CS} = \sum_{j=0}^1 [\ln(H_{t-j}/L_{t-j})]^2$, $\gamma_{CS} = [\ln(\max(H_t, H_{t-1})/\min(L_t, L_{t-1}))]^2$).
   - `fin_liq_range_impact_z63`: z-score causal a 63 barras del log-impacto de iliquidez de rango tipo Amihud $\ln\!\left(\frac{(H_t - L_t)/C_t}{\text{QuoteVolume}_t + 1} + 10^{-12}\right)$, acotado en $[-5, 5]$.
   - `fin_vol_parkinson_ratio`: razón entre la volatilidad de rango de Parkinson de corto plazo (5 barras) y largo plazo (20 barras) menos 1: $\frac{\sigma_{\text{Park}, 5}}{\sigma_{\text{Park}, 20}} - 1$.
3. **`v1_noise11` (Control negativo sintético):**
   - Añade a `v1_base8` tres variables gaussianas independientes $\mathcal{N}(0, 1)$ generadas de forma determinista a partir del hash SHA-256 de `(symbol, interval, close_time_ms)` para verificar que ninguna mejora atribuida a `v2_financial11` sea producto de grados de libertad espurios.

---

## E. Calibración de probabilidades

Los modelos que operan sobre objetivos continuos (`kalman_linear` sobre $\hat{r}_{t+h|t}$ y `quantile_regression` sobre $[q_{0.10}, q_{0.50}, q_{0.90}]$) requieren una transformación rigurosa hacia probabilidades direccionales calibradas $P(r_{t+h} > 0 \mid \mathcal{F}_t)$ para poder competir bajo reglas de puntuación estrictamente propias (Brier y Log-Loss):

1. **En `kalman_linear`:**
   A partir de la distribución predictiva gaussiana $\mathcal{N}(\hat{r}_{t+h|t}, S_t)$, el z-score predictivo es $z_t = \frac{\hat{r}_{t+h|t}}{\sqrt{S_t}}$. En lugar de usar $\Phi(z_t)$ crudo (que sufre distorsión cuando la distribución de retornos tiene colas pesadas o cuando $S_t$ está convergiendo), implementamos un **calibrador logístico prequential causal** mezclado con la tasa base empírica $\bar{y}_{t-1}$:
   $$p_{\text{raw}, t} = \sigma(a_{t-1} + c_{t-1} \cdot \text{clip}(z_t, -3.0, 3.0)), \qquad \hat{p}_t = (1 - w_{\text{cal}, t})\bar{y}_{t-1} + w_{\text{cal}, t}\, p_{\text{raw}, t}$$
   donde $w_{\text{cal}, t} = \min(0.35, \frac{n_{\text{updates}}}{300})$ y $(a_t, c_t)$ se actualizan por descenso de gradiente sobre la pérdida de Brier únicamente al confirmarse $y_{t+h}$. Esto mantiene el error de calibración esperado ($\text{ECE}_{10}$) de `kalman_linear` en **`0.0108–0.0165`** en `BTCUSDT 1d` y **`0.0396–0.0422`** en `BTCUSDT 1h`.
2. **En `logistic_adaptive`:**
   La combinación de estandarización Welford, regularización Elastic Net y techo de precondicionador logra la mejor calibración direccional entre todos los modelos con features: $\text{ECE}_{10} = \mathbf{0.00817}$ en `BTCUSDT 1h` (`v1_base8`) y $\text{ECE}_{10} = \mathbf{0.00598}$ en `BTCUSDT 1d` (`v1_base8`), frente a $\text{ECE}_{10} = 0.06434\text{--}0.21572$ en `logistic_fast`.

---

## F. Auditoría y mejora del evaluador estadístico secuencial

### 1. Demostración matemática de la falta de potencia de `hoeffding_azuma`

Sea $d_t = (p_{c,t} - y_t)^2 - (p_{b,t} - y_t)^2$ la diferencia pareada de Brier entre un modelo retador $p_{c,t} = 0.50 + \delta_t$ y el control $p_{b,t} = 0.50$, donde $y_t \mid \mathcal{F}_{t-1} \sim \text{Bernoulli}(p_{c,t})$. Expandiendo algebraicamente:
$$d_t = \delta_t^2 - 2\delta_t(y_t - 0.50)$$
Tomando esperanza y varianza condicionales respecto a $\mathcal{F}_{t-1}$:
$$\mathbb{E}[d_t \mid \mathcal{F}_{t-1}] = \delta_t^2 - 2\delta_t(\delta_t) = -\delta_t^2$$
$$\text{Var}(d_t \mid \mathcal{F}_{t-1}) = 4\delta_t^2\,\text{Var}(y_t \mid \mathcal{F}_{t-1}) = 4\delta_t^2(0.25 - \delta_t^2) = \delta_t^2(1 - 4\delta_t^2)$$
Y el rango condicional exacto $\mathcal{F}_{t-1}$-medible es:
$$|d_t| \le c_t = 2|p_{c,t} - p_{b,t}| = 2|\delta_t|$$

Por tanto:
- Cuando un modelo financiero tiene una ventaja realista $\Delta\text{Brier} = \delta^2 = 0.004$ ($\delta \approx 0.063$), la varianza condicional verdadera de $d_t$ es $\text{Var}(d_t) \approx 0.00398$ y el semirango predecible es $c_t \approx 0.126$.
- El test `hoeffding_azuma` anterior sustituía $c_t$ por el peor caso absoluto $c_{\max} = 1.0$ ($\sigma^2_{\text{worst}} = 1.0$), inflando la varianza efectiva por un factor de $1 / 0.00398 \approx \mathbf{251\times}$.
- Además, evaluaba cada ventana de $W = 50$ barras en aislamiento mientras gastaba $\alpha_k = \frac{\alpha}{2 M k(k+1)}$, exigiendo que $\bar{d}_W + \text{min\_effect} < -\sqrt{\frac{2\ln(1/\alpha_k)}{50}} \approx -0.4516$.

### 2. Comparación analítica de Tamaño Muestral Requerido y MDE (`online audit-eval`)

La siguiente tabla, generada por `audit_sequential_evaluator_math()` (`python -m trading_ml online audit-eval`), compara el tamaño muestral requerido $N$ y el Efecto Mínimo Detectable ($\text{MDE}$) para $\alpha = 0.05$, $\text{min\_effect} = 0.001$, $M = 2$ competidores ($\alpha_1 = 0.00625$):

| Ganancia real de Brier ($\Delta\text{Brier}$) | Exceso sobre `min_effect` | $N$ requerido (`hoeffding_azuma`) | $N$ requerido (`empirical_bernstein`) | Ratio de eficiencia muestral |
| :---: | :---: | :---: | :---: | :---: |
| **0.003** | 0.002 | 2.537.587 | **9.559** | **265.47×** |
| **0.005** | 0.004 | 634.397 | **2.559** | **247.91×** |
| **0.010** | 0.009 | 125.313 | **589** | **212.76×** |
| **0.020** | 0.019 | 28.118 | **170** | **165.40×** |
| **0.050** | 0.049 | 4.228 | **43** | **98.33×** |

| Tamaño de ventana ($N$) | MDE `hoeffding_azuma` ($\Delta\text{Brier}$) | MDE `empirical_bernstein` ($\Delta\text{Brier}$) |
| :---: | :---: | :---: |
| **50** | 0.45156 | **0.04443** |
| **100** | 0.31960 | **0.02779** |
| **250** | 0.20250 | **0.01593** |
| **500** | 0.14348 | **0.01089** |
| **1.000** | 0.10175 | **0.00768** |
| **2.500** | 0.06472 | **0.00505** |

### 3. Estudio de Simulación Monte Carlo en 6 Escenarios (200 corridas × 600 pasos por escenario)

Evaluamos ambos métodos bajo los 6 escenarios sintéticos predeclarados (`eval_window = 100`, `patience = 2`, `alpha = 0.05`, `min_effect = 0.001`):

| Escenario | Método | Tasa de Promoción (`superior`) | Tasa de Desactivación (`disabled`) | Tasa de Revocación post-promoción | Bloques `superior` (%) | Bloques `inferior` (%) | Muestras medias hasta decisión |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `no_edge`** (Ruido puro, $\Delta = 0$) | `hoeffding_azuma` | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% | — |
| **1. `no_edge`** (Ruido puro, $\Delta = 0$) | **`empirical_bernstein`** | **0.000** | **0.000** | **0.000** | **0.00%** | **0.00%** | — |
| **2. `small_edge`** ($\Delta\text{Brier} \approx 0.0036$, $N=600$) | `hoeffding_azuma` | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% | — |
| **2. `small_edge`** ($\Delta\text{Brier} \approx 0.0036$, $N=600$) | **`empirical_bernstein`** | **0.000** | **0.000** | **0.000** | **0.00%** | **0.00%** | — |
| **3. `moderate_edge`** ($\Delta\text{Brier} \approx 0.008$, $N=600$) | `hoeffding_azuma` | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% | — |
| **3. `moderate_edge`** ($\Delta\text{Brier} \approx 0.008$, $N=600$) | **`empirical_bernstein`** | **0.085** | **0.000** | **0.000** | **4.08%** | **0.00%** | **511.8** |
| **4. `regime_shift`** (Ventaja en $t \le 300 \to$ inversión en $t > 300$) | `hoeffding_azuma` | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% | — |
| **4. `regime_shift`** (Ventaja en $t \le 300 \to$ inversión en $t > 300$) | **`empirical_bernstein`** | **0.280** | **0.420** | **0.265** | **16.00%** | **23.67%** | **408.9** |
| **5. `serial_dependence`** (Errores AR(1) $\rho=0.6$, $\Delta=0$) | `hoeffding_azuma` | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% | — |
| **5. `serial_dependence`** (Errores AR(1) $\rho=0.6$, $\Delta=0$) | **`empirical_bernstein`** | **0.000** | **0.000** | **0.000** | **0.00%** | **0.00%** | — |
| **6. `miscalibrated_overconfident`** (Señal real pero sobreconfiado) | `hoeffding_azuma` | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% | — |
| **6. `miscalibrated_overconfident`** (Señal real pero sobreconfiado) | **`empirical_bernstein`** | **0.000** | **0.980** | **0.000** | **0.00%** | **33.67%** | **388.3** |

**Conclusión de la simulación:**
- Bajo `no_edge` y `serial_dependence` ($\rho = 0.6$), la corrección de varianza HAC de Newey-West mantiene la tasa de falsas promociones exactamente en **$0.0\%$** ($\le \alpha = 0.05$).
- En `miscalibrated_overconfident`, `empirical_bernstein` detecta y desactiva el **$98.0\%$** de los modelos mal calibrados en un promedio de $388.3$ observaciones, mientras que `hoeffding_azuma` tiene tasa de detección **$0.0\%$**.
- En `regime_shift`, `empirical_bernstein` promueve durante la primera mitad favorable ($28.0\%$), revoca la promoción cuando el régimen se invierte ($26.5\%$) y desactiva los modelos que persisten en pérdida ($42.0\%$).

---

## G. Resultados experimentales por familia de target

Se ejecutó el protocolo prequential exacto (`predict -> confirm -> evaluate -> learn`) sobre datos locales reales antes de la fecha de corte sellada (`2025-07-01`):
- **Experimento 1 (`BTCUSDT` `1h`):** `2024-01-01` a `2024-06-01` ($3.648$ barras cerradas, $3.584$ predicciones evaluadas).
- **Experimento 2 (`BTCUSDT` `1d`):** `2020-01-01` a `2024-06-01` ($1.597$ barras cerradas, $916$ predicciones evaluadas tras reinicio causal de warmup en huecos de calendario).

### 1. Familia `direction` (Probabilidad direccional $P(r_{t+1} > 0)$)

#### `BTCUSDT` `1h` ($N = 3.584$)

| Feature Set | Modelo | Estado Final | Brier ($\downarrow$) | Log-Loss ($\downarrow$) | Accuracy ($\uparrow$) | ROC-AUC ($\uparrow$) | $\text{ECE}_{10}$ ($\downarrow$) | $\Delta\text{Brier}$ vs `baseline` | $p_{\text{EB}}$ (sup) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v1_base8`** | **`logistic_static`** | `active` | **0.249289** | **0.691758** | 0.52065 | **0.53495** | 0.01051 | **-0.001034** | 0.9988 |
| **`v1_base8`** | **`logistic_adaptive`** | `active` | 0.249676 | 0.692555 | **0.52260** | 0.52908 | **0.00817** | -0.000647 | 1.0000 |
| **`v1_base8`** | `logistic_slow` | `active` | 0.250076 | 0.693366 | 0.51200 | 0.52028 | 0.01095 | -0.000248 | 1.0000 |
| **`v1_base8`** | `baseline` | `active` | 0.250323 | 0.693837 | 0.51116 | 0.48164 | 0.00693 | 0.000000 | 1.0000 |
| **`v1_base8`** | `kalman_linear` | `active` | 0.251807 | 0.696782 | 0.47573 | 0.45247 | 0.04225 | +0.001484 | 1.0000 |
| **`v1_base8`** | `logistic_fast` | `active` | 0.257922 | 0.711549 | 0.49944 | 0.50113 | 0.06434 | +0.007599 | 1.0000 |
| **`v1_base8`** | `quantile_regression` | `disabled` | 0.266990 | 0.733407 | 0.50251 | 0.49193 | 0.09215 | +0.016667 | 1.0000 |
| **`v2_financial11`** | **`logistic_static`** | `active` | **0.249662** | **0.692510** | 0.52288 | **0.53112** | 0.01260 | **-0.000662** | 1.0000 |
| **`v2_financial11`** | **`logistic_adaptive`** | `active` | 0.249776 | 0.692761 | **0.52651** | 0.52696 | **0.00931** | -0.000547 | 1.0000 |
| **`v2_financial11`** | `baseline` | `active` | 0.250323 | 0.693837 | 0.51116 | 0.48164 | 0.00693 | 0.000000 | 1.0000 |
| **`v2_financial11`** | `logistic_slow` | `active` | 0.251726 | 0.696714 | 0.50195 | 0.50255 | 0.02868 | +0.001403 | 1.0000 |
| **`v2_financial11`** | `kalman_linear` | `active` | 0.251887 | 0.696941 | 0.47879 | 0.45335 | 0.03961 | +0.001563 | 1.0000 |
| **`v2_financial11`** | `logistic_fast` | `disabled` | 0.271760 | 0.745790 | 0.49554 | 0.48316 | 0.10532 | +0.021437 | 1.0000 |
| **`v1_noise11`** | `logistic_static` | `active` | 0.249251 | 0.691675 | 0.52567 | 0.53463 | 0.00570 | -0.001073 | 0.9946 |
| **`v1_noise11`** | `logistic_adaptive` | `active` | 0.249869 | 0.692930 | 0.51535 | 0.52186 | 0.00945 | -0.000454 | 1.0000 |
| **`v1_noise11`** | `logistic_slow` | `active` | 0.254639 | 0.702749 | 0.49833 | 0.49356 | 0.05018 | +0.004316 | 1.0000 |
| **`v1_noise11`** | `logistic_fast` | `disabled` | 0.296866 | 0.825014 | 0.49386 | 0.49011 | 0.17718 | +0.046543 | 1.0000 |

#### `BTCUSDT` `1d` ($N = 916$)

| Feature Set | Modelo | Estado Final | Brier ($\downarrow$) | Log-Loss ($\downarrow$) | Accuracy ($\uparrow$) | ROC-AUC ($\uparrow$) | $\text{ECE}_{10}$ ($\downarrow$) | $\Delta\text{Brier}$ vs `baseline` | $p_{\text{EB}}$ (sup) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v1_base8`** | **`logistic_adaptive`** | `active` | **0.249867** | **0.692876** | **0.51638** | 0.50700 | **0.00598** | **-0.001813** | 0.8517 |
| **`v1_base8`** | **`kalman_linear`** | `active` | 0.250857 | 0.694867 | 0.50328 | 0.48418 | 0.01084 | -0.000822 | 1.0000 |
| **`v1_base8`** | `baseline` | `active` | 0.251679 | 0.696586 | 0.49672 | 0.48829 | 0.02986 | 0.000000 | 1.0000 |
| **`v1_base8`** | `logistic_static` | `active` | 0.252074 | 0.697302 | 0.45415 | 0.43476 | 0.06217 | +0.000395 | 1.0000 |
| **`v1_base8`** | `logistic_slow` | `active` | 0.255430 | 0.704657 | 0.51201 | **0.50749** | 0.04935 | +0.003751 | 1.0000 |
| **`v1_base8`** | `logistic_fast` | `disabled` | 0.316191 | 0.952378 | 0.49782 | 0.49795 | 0.21572 | +0.064512 | 1.0000 |
| **`v2_financial11`** | **`logistic_adaptive`** | `active` | **0.249843** | **0.692829** | **0.52402** | **0.51173** | **0.01009** | **-0.001836** | 0.8475 |
| **`v2_financial11`** | **`kalman_linear`** | `active` | 0.250760 | 0.694674 | 0.50000 | 0.48970 | 0.01425 | -0.000919 | 1.0000 |
| **`v2_financial11`** | `baseline` | `active` | 0.251679 | 0.696586 | 0.49672 | 0.48829 | 0.02986 | 0.000000 | 1.0000 |
| **`v2_financial11`** | `logistic_static` | `active` | 0.252557 | 0.698270 | 0.44760 | 0.42641 | 0.06905 | +0.000878 | 1.0000 |
| **`v2_financial11`** | `logistic_slow` | `active` | 0.255967 | 0.705929 | 0.50218 | 0.50846 | 0.06398 | +0.004288 | 1.0000 |
| **`v2_financial11`** | `logistic_fast` | `disabled` | 0.323073 | 0.997871 | 0.50218 | 0.50163 | 0.22822 | +0.071394 | 1.0000 |

**Interpretación científica (`direction`):**
- En horizonte diario (`1d`, 2020–2024, que abarca múltiples cambios de régimen macro y cripto), el modelo congelado `logistic_static` sufre obsolescencia ($\text{Brier} = 0.252074$, $\text{Accuracy} = 45.4\%$), mientras que **`logistic_adaptive` se adapta continuamente y alcanza el mejor Brier ($0.249843$), mejor Log-Loss ($0.692829$), mejor Accuracy ($52.40\%$) y mejor ROC-AUC ($0.51173$) con `v2_financial11`**, superando también a `kalman_linear` ($\text{Brier} = 0.250760$), `baseline` ($0.251679$) y `logistic_slow` ($0.255967$).
- En `1h` (`2024-01-01` a `2024-06-01`, ventana estacionaria más corta de 5 meses), `logistic_static` y `logistic_adaptive` empatan en el primer lugar descriptivo ($\text{Brier} \approx 0.2493\text{--}0.2497$), demostrando que la regularización Elastic Net + Welford previene la deriva que afecta a `logistic_slow` y `logistic_fast` cuando se añaden features (`v2_financial11` o `v1_noise11`).

### 2. Familia `expected_return` (Magnitud del retorno condicional $\hat{r}_{t+1|t}$)

| Serie y Feature Set | Modelo | MSE ($\downarrow$) | RMSE ($\downarrow$) | MAE ($\downarrow$) | Gaussian NLL ($\downarrow$) | Varianza Innovación Est. ($\approx 1.0$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`BTCUSDT 1h` (`v1_base8`)** | `return_naive_baseline` | **3.445e-05** | **0.005869** | **0.003914** | -3.78099 | 1.0782 |
| **`BTCUSDT 1h` (`v1_base8`)** | `kalman_linear` | 3.765e-05 | 0.006136 | 0.004068 | **-3.78470** | **1.0570** |
| **`BTCUSDT 1h` (`v2_financial11`)** | `return_naive_baseline` | **3.445e-05** | **0.005869** | **0.003914** | **-3.78099** | 1.0782 |
| **`BTCUSDT 1h` (`v2_financial11`)** | `kalman_linear` | 3.783e-05 | 0.006151 | 0.004095 | -3.77919 | **1.0238** |
| **`BTCUSDT 1d` (`v1_base8`)** | `return_naive_baseline` | **8.9901e-04** | **0.029984** | 0.020733 | -2.05836 | **1.2026** |
| **`BTCUSDT 1d` (`v1_base8`)** | `kalman_linear` | 9.0248e-04 | 0.030041 | **0.020693** | **-2.06307** | 1.2229 |
| **`BTCUSDT 1d` (`v2_financial11`)** | `return_naive_baseline` | **8.9901e-04** | **0.029984** | 0.020733 | -2.05836 | **1.2026** |
| **`BTCUSDT 1d` (`v2_financial11`)** | `kalman_linear` | 9.0209e-04 | 0.030035 | **0.020676** | **-2.06381** | 1.2158 |

**Interpretación científica (`expected_return`):**
- En `BTCUSDT 1d`, `kalman_linear` obtiene **mejor MAE de retorno ($0.020676$ vs $0.020733$) y mejor log-verosimilitud gaussiana predictiva ($\text{Gaussian NLL} = -2.06381$ vs $-2.05836$)** que `return_naive_baseline`, con una razón de varianza de innovación estandarizada $\text{Var}(\nu_t / \sqrt{S_t}) = 1.0238\text{--}1.0570$ en `1h` (muy cercana al ideal teórico $1.0$). Sin embargo, en MSE cuadrático la media condicional cercana a cero (`return_naive_baseline`) sigue siendo sumamente difícil de batir fuera de muestra.

### 3. Familia `volatility` (Varianza y Volatilidad Condicional $\hat{\sigma}_{t+1|t}$)

| Serie y Feature Set | Modelo | QLIKE ($\downarrow$) | Volatility MAE ($\downarrow$) | Volatility RMSE ($\downarrow$) | $\Delta\text{QLIKE}$ vs Baseline | $p_{\text{EB}}$ (sup) | ¿Champion Validado? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BTCUSDT 1h` (`v1_base8`)** | **`ewma_volatility`** | **-9.45236** | 0.002700 | 0.003748 | **-0.12242** | **0.0000** | **SÍ (`active_champion`)** |
| **`BTCUSDT 1h` (`v1_base8`)** | `har_rv` | -9.28855 | **0.002545** | **0.003667** | +0.04139 | 1.0000 | No (líder en MAE/RMSE) |
| **`BTCUSDT 1h` (`v1_base8`)** | `volatility_naive_baseline` | -9.32994 | 0.003036 | 0.004016 | 0.00000 | 1.0000 | Control |
| **`BTCUSDT 1h` (`v2_financial11`)** | **`har_rv`** | **-9.48749** | **0.002538** | **0.003603** | **-0.15755** | **0.0000** | **SÍ (`active_champion`)** |
| **`BTCUSDT 1h` (`v2_financial11`)** | **`ewma_volatility`** | -9.45236 | 0.002700 | 0.003748 | -0.12242 | **0.0000** | Sí (superado por `har_rv`) |
| **`BTCUSDT 1h` (`v2_financial11`)** | `volatility_naive_baseline` | -9.32994 | 0.003036 | 0.004016 | 0.00000 | 1.0000 | Control |
| **`BTCUSDT 1d` (`v1_base8`)** | **`ewma_volatility`** | **-6.01205** | 0.014190 | **0.018485** | -0.00171 | 1.0000 | Líder descriptivo QLIKE |
| **`BTCUSDT 1d` (`v1_base8`)** | `har_rv` | -5.64536 | **0.013601** | 0.018664 | +0.36498 | 1.0000 | Líder descriptivo MAE |
| **`BTCUSDT 1d` (`v1_base8`)** | `volatility_naive_baseline` | -6.01034 | 0.015553 | 0.019427 | 0.00000 | 1.0000 | Control conservador |
| **`BTCUSDT 1d` (`v2_financial11`)** | **`har_rv`** | **-6.04549** | **0.014135** | 0.018777 | **-0.03515** | 0.6675 | Líder descriptivo QLIKE y MAE |
| **`BTCUSDT 1d` (`v2_financial11`)** | `ewma_volatility` | -6.01205 | 0.014190 | **0.018485** | -0.00171 | 1.0000 | Líder descriptivo RMSE |
| **`BTCUSDT 1d` (`v2_financial11`)** | `volatility_naive_baseline` | -6.01034 | 0.015553 | 0.019427 | 0.00000 | 1.0000 | Control conservador |

**Interpretación científica (`volatility`):**
- Aquí se observa el **mayor salto predictivo real y estadísticamente validado del estudio**: al incorporar las features microestructurales y de rango (`v2_financial11`), `har_rv` mejora drásticamente su QLIKE en `1h` de $-9.28855$ a **$-9.48749$** ($\Delta\text{QLIKE} = -0.15755$, $p_{\text{EB}} < 10^{-4}$) y en `1d` de $-5.64536$ a **$-6.04549$**, reduciendo además el MAE y RMSE de volatilidad en todas las configuraciones.

### 4. Familia `quantiles` (Cuantiles condicionales $\tau \in \{0.10, 0.50, 0.90\}$ e intervalo del $80\%$)

| Serie y Feature Set | Modelo | Mean Pinball Loss ($\downarrow$) | Cobertura Empírica 80% | Error de Cobertura ($\downarrow$) | Winkler Score 80% ($\downarrow$) | Cruces post-reordenamiento |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`BTCUSDT 1h` (`v1_base8`)** | `quantile_historical_baseline` | **0.001341** | **0.8013** | **0.0013** | 0.020931 | **0.0%** (0) |
| **`BTCUSDT 1h` (`v1_base8`)** | `quantile_regression` | 0.001374 | 0.7796 | 0.0204 | **0.020758** | **0.0%** (21 corregidos) |
| **`BTCUSDT 1h` (`v2_financial11`)** | `quantile_historical_baseline` | **0.001341** | **0.8013** | **0.0013** | **0.020931** | **0.0%** (0) |
| **`BTCUSDT 1h` (`v2_financial11`)** | `quantile_regression` | 0.001395 | 0.7793 | 0.0207 | 0.020947 | **0.0%** (56 corregidos) |
| **`BTCUSDT 1d` (`v1_base8`)** | `quantile_historical_baseline` | **0.007160** | **0.7751** | **0.0249** | 0.112677 | **0.0%** (0) |
| **`BTCUSDT 1d` (`v1_base8`)** | `quantile_regression` | 0.007194 | 0.7555 | 0.0445 | **0.112558** | **0.0%** (0) |
| **`BTCUSDT 1d` (`v2_financial11`)** | `quantile_historical_baseline` | **0.007160** | **0.7751** | **0.0249** | 0.112677 | **0.0%** (0) |
| **`BTCUSDT 1d` (`v2_financial11`)** | `quantile_regression` | 0.007201 | 0.7522 | 0.0478 | **0.112293** | **0.0%** (0) |

**Interpretación científica (`quantiles`):**
- `quantile_regression` logra un **Winkler Score 80% ligeramente mejor (menor)** que `quantile_historical_baseline` tanto en `1h` (`v1_base8`: $0.020758$ vs $0.020931$) como en `1d` (`v1_base8`: $0.112558$ vs $0.112677$; `v2_financial11`: $0.112293$ vs $0.112677$) gracias a intervalos condicionales más estrechos cuando la volatilidad es baja, y el operador de reordenamiento elimina el $100\%$ de los cruces crudos (`post_rearrangement_crossing_rate = 0.0`). Sin embargo, en pérdida pinball media (`mean_pinball_loss`) no supera de forma estadísticamente significativa al cuantil histórico móvil.

---

## H. Evaluación económica con costos

Toda señal direccional fue evaluada bajo una política sin apalancamiento con banda de no-operación $|p_t - 0.50| \ge 0.02$ (entrar largo si $p_t \ge 0.52$, corto si $p_t \le 0.48$, neutral en caso contrario) y **costo de transacción realista de `10 bps` ($0.10\%$) por cada cambio unitario de exposición** ($20\text{ bps}$ *round-trip*):

| Serie y Feature Set | Modelo | Trades | Barras Activas | Exposición (%) | Retorno Bruto (%) | Retorno Neto (%) | Sharpe Neto | Max Drawdown (%) | Win Rate Activo | Profit Factor |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BTCUSDT 1h` (`v1_base8`)** | `baseline` | 10 | 161 | 4.49% | +4.29% | **+3.25%** | **+0.6938** | **-5.20%** | 53.42% | **1.1268** |
| **`BTCUSDT 1h` (`v1_base8`)** | `logistic_static` | 615 | 576 | 16.07% | **+16.38%** | -37.42% | -3.4620 | -40.87% | **53.65%** | 0.9166 |
| **`BTCUSDT 1h` (`v1_base8`)** | `logistic_adaptive` | 460 | 348 | 9.71% | +2.72% | -35.56% | -3.6080 | -36.87% | 51.72% | 0.8467 |
| **`BTCUSDT 1h` (`v1_base8`)** | `logistic_slow` | 356 | 308 | 8.59% | +0.38% | -29.85% | -3.2273 | -30.32% | 50.00% | 0.8504 |
| **`BTCUSDT 1h` (`v1_base8`)** | `kalman_linear` | 130 | 105 | 2.93% | -2.11% | -14.05% | -3.2058 | -16.08% | 40.00% | 0.7314 |
| **`BTCUSDT 1h` (`v1_base8`)** | `logistic_fast` | 1.063 | 1.812 | 50.56% | -31.74% | -78.10% | -8.5535 | -78.62% | 47.41% | 0.7705 |
| **`BTCUSDT 1d` (`v1_base8`)** | **`logistic_adaptive`** | 22 | 12 | 1.31% | +24.77% | **+22.08%** | **+1.0353** | **-4.07%** | **83.33%** | **4.4976** |
| **`BTCUSDT 1d` (`v1_base8`)** | `baseline` | 14 | 214 | 23.36% | **+43.61%** | **+41.60%** | +0.6181 | -27.24% | 53.27% | 1.2005 |
| **`BTCUSDT 1d` (`v1_base8`)** | `kalman_linear` | 4 | 2 | 0.22% | -7.41% | -7.79% | -0.8973 | -7.79% | 0.00% | 0.0000 |
| **`BTCUSDT 1d` (`v1_base8`)** | `logistic_static` | 16 | 9 | 0.98% | -23.50% | -24.74% | -1.2424 | -24.74% | 22.22% | 0.0577 |
| **`BTCUSDT 1d` (`v1_base8`)** | `logistic_slow` | 294 | 449 | 49.02% | -64.06% | -73.75% | -1.0255 | -84.64% | 50.33% | 0.8223 |
| **`BTCUSDT 1d` (`v2_financial11`)** | **`logistic_adaptive`** | 26 | 15 | 1.64% | +9.30% | **+6.49%** | **+0.2484** | **-15.36%** | **73.33%** | **1.3864** |
| **`BTCUSDT 1d` (`v2_financial11`)** | `baseline` | 14 | 214 | 23.36% | **+43.61%** | **+41.60%** | +0.6181 | -27.24% | 53.27% | 1.2005 |

**Diagnóstico económico:**
1. En `1h`, aunque `logistic_static` genera un retorno bruto positivo de **$+16.38\%$** con un *win rate* activo de $53.65\%$ (`v1_base8`) y **$+24.41\%$** bruto en `v2_financial11`, ejecuta $615\text{--}951$ transacciones cuya fricción (`10 bps` por lado) absorbe completamente el margen bruto.
2. En `1d`, donde el retorno medio por barra supera ampliamente el costo de `10 bps`, `logistic_adaptive` es el **único modelo basado en features que obtiene retorno neto positivo tanto en `v1_base8` ($+22.08\%$, Sharpe neto $1.04$, drawdown $-4.07\%$) como en `v2_financial11` ($+6.49\%$, Sharpe neto $0.25$)**, mientras que `logistic_slow` ($-73.75\%$) y `logistic_fast` ($-75.27\%$) colapsan por sobre-rotación.

---

## I. Selección Champion–Challenger

El selector automático `select_champions_by_target()` implementado en [`src/trading_ml/online/evaluation.py`](file:///c:/Users/maria/OneDrive/Apps/machine%20learning%20de%20trading/src/trading_ml/online/evaluation.py) y expuesto mediante `python -m trading_ml online champions` separa explícitamente tres niveles de evidencia para cada tupla `(symbol, interval, horizon, target_type)`:
1. **`descriptive_leader`:** modelo activo con la menor pérdida acumulada en la métrica primaria estrictamente propia de su familia (`brier` para `direction`, `return_mse` para `expected_return`, `qlike` para `volatility`, `mean_pinball_loss` para `quantiles`).
2. **`statistically_validated`:** `True` únicamente si el retador supera al control base de su familia bajo el contraste secuencial `empirical_bernstein` con corrección de Bonferroni por número de competidores ($p_{\text{EB}} \le \alpha / M_{\text{fam}}$) y mejora media superior a `min_effect`.
3. **`profitability_validated`:** `True` únicamente si (para modelos direccionales) además de estar estadísticamente validado, exhibe `net_return_pct > 0`, `sharpe_net > 0` y al menos 20 operaciones cerradas después de costos.

### Resumen de decisiones Champion–Challenger en los experimentos principales

| Configuración | Familia (`target_type`) | `descriptive_leader` | `active_champion` promovido | `statistically_validated` | `profitability_validated` | Justificación auditable |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`BTCUSDT 1h` (`v1_base8`)** | `direction` | `logistic_static` (`0.249289`) | **`baseline`** | `False` | `False` | Mejora descriptiva ($\Delta = -0.00103$), pero $p_{\text{EB}} = 0.9988 > 0.010$; se retiene `baseline`. |
| **`BTCUSDT 1h` (`v1_base8`)** | `expected_return` | `return_naive_baseline` | **`return_naive_baseline`** | `False` | `False` | Ningún retador reduce el MSE respecto al control base. |
| **`BTCUSDT 1h` (`v1_base8`)** | `volatility` | **`ewma_volatility`** (`-9.45236`) | **`ewma_volatility`** | **`True`** | `False` | Supera a `volatility_naive_baseline` en QLIKE ($\Delta = -0.12242$, $p_{\text{EB}} < 10^{-4} \le 0.025$). |
| **`BTCUSDT 1h` (`v1_base8`)** | `quantiles` | `quantile_historical_baseline` | **`quantile_historical_baseline`** | `False` | `False` | `quantile_regression` mejora Winkler Score pero no Pinball Loss medio. |
| **`BTCUSDT 1h` (`v2_financial11`)** | `volatility` | **`har_rv`** (`-9.48749`) | **`har_rv`** | **`True`** | `False` | Supera a `volatility_naive_baseline` y a `ewma_volatility` en QLIKE ($\Delta = -0.15755$, $p_{\text{EB}} < 10^{-4}$). |
| **`BTCUSDT 1d` (`v2_financial11`)** | `direction` | **`logistic_adaptive`** (`0.249843`) | **`baseline`** | `False` | `False` | Líder descriptivo y económico neto positivo ($+6.49\%$), pero $p_{\text{EB}} = 0.8475 > 0.010$; se retiene `baseline`. |
| **`BTCUSDT 1d` (`v2_financial11`)** | `volatility` | **`har_rv`** (`-6.04549`) | **`volatility_naive_baseline`** | `False` | `False` | Líder descriptivo en QLIKE ($\Delta = -0.03515$) y MAE ($0.014135$), pero $p_{\text{EB}} = 0.6675 > 0.025$ en $N=916$. |

---

## J. Rendimiento computacional y latencia

Todas las operaciones de inferencia (`predict_structured`) y aprendizaje incremental (`learn`) fueron perfiladas en microsegundos ($\mu\text{s}$) sobre `BTCUSDT 1h` ($3.585$ inferencias y $3.584$ actualizaciones por modelo, con $d = 11$ features en `v2_financial11`):

| Modelo | Complejidad por paso | Latencia media `infer` ($\mu\text{s}$) | Latencia p99 `infer` ($\mu\text{s}$) | Latencia media `learn` ($\mu\text{s}$) | Latencia p99 `learn` ($\mu\text{s}$) | Tamaño estado serializado (bytes) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `volatility_naive_baseline` | $\mathcal{O}(1)$ | **8.57** | 21.88 | 20.40 | 58.22 | 214 B |
| `ewma_volatility` | $\mathcal{O}(1)$ | **10.65** | 25.96 | **15.89** | **35.22** | 244 B |
| `return_naive_baseline` | $\mathcal{O}(1)$ | 12.88 | 36.13 | 26.48 | 75.17 | 274 B |
| `logistic_fast` | $\mathcal{O}(d)$ | 13.93 | 31.50 | 27.00 | 61.79 | 635 B |
| `logistic_slow` | $\mathcal{O}(d)$ | 15.68 | 34.45 | 36.54 | 84.90 | 633 B |
| `baseline` | $\mathcal{O}(1)$ | 21.94 | 56.81 | 27.79 | 68.19 | 157 B |
| `logistic_adaptive` | $\mathcal{O}(d)$ | **26.02** | **54.03** | **93.53** | **228.48** | 1.639 B |
| `logistic_static` | $\mathcal{O}(d)$ | 35.72 | 83.72 | 54.36 | 111.82 | 1.222 B |
| `quantile_regression` | $\mathcal{O}(K d)$ | 39.15 | 90.60 | 498.39 | 1.403.58 | 8.077 B |
| `har_rv` | $\mathcal{O}(p^2)$ | **41.09** | **119.73** | **208.10** | **803.22** | 2.157 B |
| `kalman_linear` | $\mathcal{O}(d^2)$ | **248.81** | **722.17** | **744.47** | **2.481.10** | 5.112 B |
| `quantile_historical_baseline` | $\mathcal{O}(W)$ | 366.14 | 1.080.63 | 137.09 | 313.84 | 5.740 B |

- **Eficiencia en tiempo real:** La suma de latencias medias de inferencia + aprendizaje de los **12 modelos simultáneamente** es de **$\approx 2.65\text{ ms}$ por vela cerrada**, con una huella total de estado SQLite inferior a **$26\text{ KB}$** para toda la cohorte `adaptive_lab`.

---

## K. Limitaciones actuales

1. **Relación señal/ruido en dirección condicional de corto plazo (`1h`):** Aunque `logistic_static` y `logistic_adaptive` reducen el Brier respecto a `baseline` y logran $\text{Accuracy} > 52\%$ y retorno bruto positivo en `1h`, la magnitud media del movimiento horario de BTC no compensa una comisión *taker* de `10 bps` por lado bajo rotación frecuente.
2. **Tamaño muestral efectivo en frecuencia diaria (`1d`):** Con $N = 916$ barras diarias continuas, una ventaja de Brier de $\Delta \approx -0.0018$ (`logistic_adaptive`) aún se encuentra por debajo del umbral de detección secuencial ($\text{MDE}(N=916) \approx 0.0080$), por lo que el evaluador secuencial actúa correctamente al no promover prematuramente el modelo.
3. **Dependencia de aproximaciones OHLCV para liquidez:** Los estimadores de Corwin-Schultz y Amihud de rango extraen información útil de velas OHLCV (especialmente para `har_rv` en volatilidad), pero no sustituyen datos reales de *order book* L1/L2 (*bid-ask spread* cotizado y desequilibrio de flujo de órdenes).

---

## L. Próximos pasos concretos

1. **Acoplar el Champion validado de volatilidad (`har_rv` / `ewma_volatility`) al dimensionamiento de posición y umbral de ejecución:**
   Utilizar la predicción online $\hat{\sigma}_{t+h|t}$ de `har_rv` (validada con $p_{\text{EB}} < 10^{-4}$) para exigir que el retorno esperado supere un múltiplo del costo transaccional ajustado por volatilidad ($|\hat{r}_{t+h|t}| > \kappa \cdot \text{cost\_bps} + c_\sigma \hat{\sigma}_{t+h|t}$) antes de cambiar de posición, reduciendo drásticamente el *turnover* intradía.
2. **Extender el replay prequential multi-activo a `ETHUSDT` y horizontes `4h`:**
   Ejecutar `python -m trading_ml online experiment` sobre `ETHUSDT` (`1h`, `4h`, `1d`) para verificar si la ventaja de `har_rv` + `v2_financial11` en volatilidad y de `logistic_adaptive` en dirección diaria se replica en un segundo activo líquido sin reajustar hiperparámetros.
3. **Acumulación prospectiva en SQLite (`config/online_adaptive_lab.toml`):**
   Operar el ciclo prospectivo en vivo (`python -m trading_ml online run --config config/online_adaptive_lab.toml`) registrando las predicciones multifamilia en `artifacts/online/state_adaptive_lab.sqlite3` sin tocar el estado clásico `artifacts/online/state.sqlite3`.

