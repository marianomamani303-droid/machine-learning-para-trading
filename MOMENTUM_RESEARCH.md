# Investigación Momentum

Run: `20261013T215021_338a147b`.

**Conclusión: no hay ventaja confirmada en datos futuros independientes.** Las comparaciones históricas siguientes son descriptivas; todos los horizontes y umbrales se congelaron antes de ejecutar esta ronda. No se elige un ganador.

Ninguna variante cumple simultáneamente los criterios históricos prefijados. En el historial posterior a desarrollo las reglas retornan entre -20.77% y -6.92%, frente a buy and hold -20.34% y efectivo 0.00%. Menor drawdown que buy and hold puede ser útil como control de exposición, pero no demuestra exceso de rentabilidad ni capacidad predictiva adicional de XGBoost.

## Protocolo y causalidad

Momentum long/efectivo por instrumento. Entrada al cierre si retorno normalizado ≥0.5, precio sobre SMA rápida y lenta, pendiente lenta positiva, eficiencia ≥0.1, persistencia sobre SMA lenta ≥0.55 y cierre rompe el máximo HIGH de las L velas anteriores (shift de una vela). Ejecución en la próxima apertura.

SMA rápida=max(5,L//2); SMA lenta=2L; pendientes como cambio relativo de la media en 5 velas dividido por esa ventana. Persistencia: proporción de las últimas L velas sobre SMA lenta. Eficiencia: |log retorno L| dividido por suma de |log retornos de una vela|. Warmup puede alcanzar aproximadamente 3L velas para calcular persistencia; sin readiness no se opera.

Permanece mientras el retorno acumulado sea positivo, precio sobre SMA lenta y no rompa el mínimo LOW anterior; sale en próxima apertura ante cualquiera de esas invalidaciones o al cumplir 2L velas. No se reingresa en la misma apertura de salida. ML sólo filtra entradas con P(trade neto rentable) ≥0.55; mantiene las mismas salidas. La etiqueta replica un trade hipotético completo open/open con costos; etiquetas sin salida natural observada se excluyen.

Train se purga por la fecha real de salida de cada etiqueta (< inicio del fold). Las oportunidades de entrenamiento pueden solaparse; su cantidad no representa operaciones independientes. Modelo fijo de 120 árboles/profundidad 2, sin búsqueda, early stopping ni calibración posterior. Pocas oportunidades o una sola clase producen abstención registrada, no un ajuste con datos posteriores.

L=20/60/120/250 **velas**, no meses equivalentes entre mercados; en BTC diario son días de calendario. La literatura estudia horizontes de 1–12 meses, pero una cartera diversificada de futuros long/short no prueba una ventaja en BTC long/efectivo.

Fuentes primarias: [Moskowitz, Ooi y Pedersen (2012)](https://doi.org/10.1016/j.jfineco.2011.11.003), [Hurst, Ooi y Pedersen (2017)](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing).

## Features actuales

Schema original: return_1/3/7, SMA ratios 10/20/50, MACD/signal/hist divididos por close, volatilidad rolling20 y RSI. Schema ampliado: return_21 y momentum_vol_ratio = return_7/(volatilidad×√7), entre otras variables. Son causales al cierre y no cambian en esta implementación. Momentum añade retorno simple/log acumulado por L, volatilidad log, momentum normalizado, pendientes relativas SMA, eficiencia del recorrido, fracción de retornos positivos, persistencia/run sobre media y distancias/rupturas de extremos anteriores. Las variables de etiqueta nunca entran al allowlist de features.

## Rentabilidad neta y riesgo

Cada ventana inicia en efectivo y liquida al último open disponible dentro del intervalo. Todos los sistemas y benchmarks usan los mismos opens y costos. La concatenación walk-forward representa ventanas reiniciadas, con costos de liquidación entre folds; no es una única cartera continua sin reinicios.

### development

| Sistema | Neto | Drawdown | Turnover | Trades | Folds positivos |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold | 116.69% | -66.74% | 13.99 | 7 | 71.4% |
| cash | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L20_rules | 60.70% | -32.44% | 25.98 | 13 | 57.1% |
| momentum_L20_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L60_rules | 83.82% | -19.73% | 9.99 | 5 | 57.1% |
| momentum_L60_xgb | 83.82% | -19.73% | 9.99 | 5 | 57.1% |
| momentum_L120_rules | 103.41% | -20.13% | 7.99 | 4 | 57.1% |
| momentum_L120_xgb | 22.39% | -12.71% | 2.00 | 1 | 14.3% |
| momentum_L250_rules | 53.66% | -21.10% | 6.00 | 3 | 28.6% |
| momentum_L250_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| current_v20261012T130540_d2bb6ac0 | -71.90% | -78.83% | 842.98 | 422 | 42.9% |
| current_v20261012T131650_5106daf4 | -71.24% | -74.21% | 355.60 | 178 | 28.6% |
| current_v20261012T134917_f88d6f93 | -19.18% | -41.28% | 361.42 | 181 | 0.0% |

### Regímenes de esa ventana

Regímenes definidos causalmente con momentum de 60 velas normalizado: >1 alcista, <-1 bajista, resto lateral. Se atribuye cada retorno al régimen conocido al inicio de su intervalo; los costos se incluyen. La composición por régimen reúne intervalos discontinuos y no es una cartera negociable separada.

| Sistema | Alcista: neto (n) | Bajista: neto (n) | Lateral: neto (n) |
| --- | ---: | ---: | ---: |
| buy_hold | 88.05% (358) | 11.02% (180) | 3.79% (713) |
| cash | 0.00% (358) | 0.00% (180) | 0.00% (713) |
| momentum_L20_rules | 63.58% (358) | 0.00% (180) | -1.76% (713) |
| momentum_L20_xgb | 0.00% (358) | 0.00% (180) | 0.00% (713) |
| momentum_L60_rules | 30.50% (358) | 0.00% (180) | 40.86% (713) |
| momentum_L60_xgb | 30.50% (358) | 0.00% (180) | 40.86% (713) |
| momentum_L120_rules | 50.39% (358) | 0.00% (180) | 35.25% (713) |
| momentum_L120_xgb | 22.39% (358) | 0.00% (180) | 0.00% (713) |
| momentum_L250_rules | 30.91% (358) | 0.00% (180) | 17.38% (713) |
| momentum_L250_xgb | 0.00% (358) | 0.00% (180) | 0.00% (713) |
| current_v20261012T130540_d2bb6ac0 | -21.17% (358) | -20.52% (180) | -55.15% (713) |
| current_v20261012T131650_5106daf4 | -11.55% (358) | -26.43% (180) | -55.81% (713) |
| current_v20261012T134917_f88d6f93 | -4.87% (358) | 0.00% (180) | -15.04% (713) |

### Estabilidad y evidencia

- momentum_L20_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L20_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L60_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L60_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L120_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L120_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L250_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L250_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.

Intervalos pareados por bootstrap de bloques de [7, 30] velas, sin cruzar folds. Se comparan retornos log netos contra efectivo y buy and hold. Bonferroni cubre 8 sistemas × dos benchmarks × 2 bloques. No corrige todas las decisiones históricas del proyecto ni constituye evidencia prospectiva. La evaluación por trimestres y todos los folds están en calendar_quarters/folds de summary.json; no se elige horizonte usando esas tablas.

### development_stress

| Sistema | Neto | Drawdown | Turnover | Trades | Folds positivos |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold | 111.89% | -66.84% | 13.98 | 7 | 71.4% |
| cash | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L20_rules | 54.15% | -33.30% | 25.96 | 13 | 57.1% |
| momentum_L20_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L60_rules | 80.90% | -19.99% | 9.98 | 5 | 57.1% |
| momentum_L60_xgb | 80.90% | -19.99% | 9.98 | 5 | 57.1% |
| momentum_L120_rules | 100.82% | -20.13% | 7.99 | 4 | 57.1% |
| momentum_L120_xgb | 22.00% | -12.85% | 2.00 | 1 | 14.3% |
| momentum_L250_rules | 52.19% | -21.35% | 5.99 | 3 | 28.6% |
| momentum_L250_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| current_v20261012T130540_d2bb6ac0 | -92.72% | -94.16% | 841.96 | 422 | 14.3% |
| current_v20261012T131650_5106daf4 | -83.73% | -85.30% | 355.19 | 178 | 28.6% |
| current_v20261012T134917_f88d6f93 | -54.71% | -58.31% | 360.85 | 181 | 0.0% |

### Regímenes de esa ventana

Regímenes definidos causalmente con momentum de 60 velas normalizado: >1 alcista, <-1 bajista, resto lateral. Se atribuye cada retorno al régimen conocido al inicio de su intervalo; los costos se incluyen. La composición por régimen reúne intervalos discontinuos y no es una cartera negociable separada.

| Sistema | Alcista: neto (n) | Bajista: neto (n) | Lateral: neto (n) |
| --- | ---: | ---: | ---: |
| buy_hold | 86.55% (358) | 10.14% (180) | 3.13% (713) |
| cash | 0.00% (358) | 0.00% (180) | 0.00% (713) |
| momentum_L20_rules | 59.70% (358) | 0.00% (180) | -3.47% (713) |
| momentum_L20_xgb | 0.00% (358) | 0.00% (180) | 0.00% (713) |
| momentum_L60_rules | 29.46% (358) | 0.00% (180) | 39.74% (713) |
| momentum_L60_xgb | 29.46% (358) | 0.00% (180) | 39.74% (713) |
| momentum_L120_rules | 48.96% (358) | 0.00% (180) | 34.82% (713) |
| momentum_L120_xgb | 22.00% (358) | 0.00% (180) | 0.00% (713) |
| momentum_L250_rules | 29.87% (358) | 0.00% (180) | 17.19% (713) |
| momentum_L250_xgb | 0.00% (358) | 0.00% (180) | 0.00% (713) |
| current_v20261012T130540_d2bb6ac0 | -43.21% (358) | -38.08% (180) | -79.29% (713) |
| current_v20261012T131650_5106daf4 | -17.16% (358) | -38.40% (180) | -68.12% (713) |
| current_v20261012T134917_f88d6f93 | -26.24% (358) | 0.00% (180) | -38.60% (713) |

### Estabilidad y evidencia

- momentum_L20_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L20_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L60_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L60_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L120_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L120_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L250_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, too_few_trades, unstable_across_windows.
- momentum_L250_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.

Intervalos pareados por bootstrap de bloques de [7, 30] velas, sin cruzar folds. Se comparan retornos log netos contra efectivo y buy and hold. Bonferroni cubre 8 sistemas × dos benchmarks × 2 bloques. No corrige todas las decisiones históricas del proyecto ni constituye evidencia prospectiva. La evaluación por trimestres y todos los folds están en calendar_quarters/folds de summary.json; no se elige horizonte usando esas tablas.

### observed_historical

| Sistema | Neto | Drawdown | Turnover | Trades | Folds positivos |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold | -20.34% | -52.97% | 2.00 | 1 | 0.0% |
| cash | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L20_rules | -14.45% | -25.15% | 15.99 | 8 | 0.0% |
| momentum_L20_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L60_rules | -10.34% | -15.54% | 4.00 | 2 | 0.0% |
| momentum_L60_xgb | -10.34% | -15.54% | 4.00 | 2 | 0.0% |
| momentum_L120_rules | -6.92% | -14.62% | 2.00 | 1 | 0.0% |
| momentum_L120_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L250_rules | -20.77% | -26.14% | 2.00 | 1 | 0.0% |
| momentum_L250_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| current_v20261012T130540_d2bb6ac0 | -46.57% | -48.07% | 231.73 | 116 | 0.0% |
| current_v20261012T131650_5106daf4 | -13.97% | -22.01% | 33.97 | 17 | 0.0% |
| current_v20261012T134917_f88d6f93 | 0.00% | 0.00% | 0.00 | 0 | 0.0% |

### Regímenes de esa ventana

Regímenes definidos causalmente con momentum de 60 velas normalizado: >1 alcista, <-1 bajista, resto lateral. Se atribuye cada retorno al régimen conocido al inicio de su intervalo; los costos se incluyen. La composición por régimen reúne intervalos discontinuos y no es una cartera negociable separada.

| Sistema | Alcista: neto (n) | Bajista: neto (n) | Lateral: neto (n) |
| --- | ---: | ---: | ---: |
| buy_hold | -13.65% (94) | 2.33% (135) | -9.85% (254) |
| cash | 0.00% (94) | 0.00% (135) | 0.00% (254) |
| momentum_L20_rules | -17.75% (94) | -5.96% (135) | 10.60% (254) |
| momentum_L20_xgb | 0.00% (94) | 0.00% (135) | 0.00% (254) |
| momentum_L60_rules | -6.88% (94) | 0.00% (135) | -3.72% (254) |
| momentum_L60_xgb | -6.88% (94) | 0.00% (135) | -3.72% (254) |
| momentum_L120_rules | -3.28% (94) | 0.00% (135) | -3.77% (254) |
| momentum_L120_xgb | 0.00% (94) | 0.00% (135) | 0.00% (254) |
| momentum_L250_rules | -3.28% (94) | -2.67% (135) | -15.83% (254) |
| momentum_L250_xgb | 0.00% (94) | 0.00% (135) | 0.00% (254) |
| current_v20261012T130540_d2bb6ac0 | -13.20% (94) | -12.71% (135) | -29.48% (254) |
| current_v20261012T131650_5106daf4 | 2.68% (94) | -5.12% (135) | -11.69% (254) |
| current_v20261012T134917_f88d6f93 | 0.00% (94) | 0.00% (135) | 0.00% (254) |

### Estabilidad y evidencia

- momentum_L20_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L20_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L60_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L60_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L120_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L120_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L250_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L250_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.

Intervalos pareados por bootstrap de bloques de [7, 30] velas, sin cruzar folds. Se comparan retornos log netos contra efectivo y buy and hold. Bonferroni cubre 8 sistemas × dos benchmarks × 2 bloques. No corrige todas las decisiones históricas del proyecto ni constituye evidencia prospectiva. La evaluación por trimestres y todos los folds están en calendar_quarters/folds de summary.json; no se elige horizonte usando esas tablas.

### observed_historical_stress

| Sistema | Neto | Drawdown | Turnover | Trades | Folds positivos |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold | -20.60% | -52.97% | 2.00 | 1 | 0.0% |
| cash | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L20_rules | -16.62% | -26.58% | 15.97 | 8 | 0.0% |
| momentum_L20_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L60_rules | -10.92% | -15.92% | 3.99 | 2 | 0.0% |
| momentum_L60_xgb | -10.92% | -15.92% | 3.99 | 2 | 0.0% |
| momentum_L120_rules | -7.22% | -14.62% | 2.00 | 1 | 0.0% |
| momentum_L120_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| momentum_L250_rules | -21.02% | -26.26% | 2.00 | 1 | 0.0% |
| momentum_L250_xgb | 0.00% | 0.00% | 0.00 | 0 | 0.0% |
| current_v20261012T130540_d2bb6ac0 | -63.14% | -63.89% | 231.47 | 116 | 0.0% |
| current_v20261012T131650_5106daf4 | -18.53% | -23.80% | 33.94 | 17 | 0.0% |
| current_v20261012T134917_f88d6f93 | 0.00% | 0.00% | 0.00 | 0 | 0.0% |

### Regímenes de esa ventana

Regímenes definidos causalmente con momentum de 60 velas normalizado: >1 alcista, <-1 bajista, resto lateral. Se atribuye cada retorno al régimen conocido al inicio de su intervalo; los costos se incluyen. La composición por régimen reúne intervalos discontinuos y no es una cartera negociable separada.

| Sistema | Alcista: neto (n) | Bajista: neto (n) | Lateral: neto (n) |
| --- | ---: | ---: | ---: |
| buy_hold | -13.93% (94) | 2.33% (135) | -9.85% (254) |
| cash | 0.00% (94) | 0.00% (135) | 0.00% (254) |
| momentum_L20_rules | -18.40% (94) | -6.41% (135) | 9.20% (254) |
| momentum_L20_xgb | 0.00% (94) | 0.00% (135) | 0.00% (254) |
| momentum_L60_rules | -7.17% (94) | 0.00% (135) | -4.03% (254) |
| momentum_L60_xgb | -7.17% (94) | 0.00% (135) | -4.03% (254) |
| momentum_L120_rules | -3.28% (94) | 0.00% (135) | -4.08% (254) |
| momentum_L120_xgb | 0.00% (94) | 0.00% (135) | 0.00% (254) |
| momentum_L250_rules | -3.28% (94) | -2.83% (135) | -15.97% (254) |
| momentum_L250_xgb | 0.00% (94) | 0.00% (135) | 0.00% (254) |
| current_v20261012T130540_d2bb6ac0 | -20.00% (94) | -19.81% (135) | -42.54% (254) |
| current_v20261012T131650_5106daf4 | 1.86% (94) | -6.03% (135) | -14.88% (254) |
| current_v20261012T134917_f88d6f93 | 0.00% (94) | 0.00% (135) | 0.00% (254) |

### Estabilidad y evidencia

- momentum_L20_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L20_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L60_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L60_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L120_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L120_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L250_rules: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.
- momentum_L250_xgb: evidencia histórica consistente = False; condiciones fallidas: no_positive_adjusted_lower_bound_vs_buy_hold_block_30, no_positive_adjusted_lower_bound_vs_buy_hold_block_7, no_positive_adjusted_lower_bound_vs_cash_block_30, no_positive_adjusted_lower_bound_vs_cash_block_7, nonpositive_net_return, too_few_trades, unstable_across_windows.

Intervalos pareados por bootstrap de bloques de [7, 30] velas, sin cruzar folds. Se comparan retornos log netos contra efectivo y buy and hold. Bonferroni cubre 8 sistemas × dos benchmarks × 2 bloques. No corrige todas las decisiones históricas del proyecto ni constituye evidencia prospectiva. La evaluación por trimestres y todos los folds están en calendar_quarters/folds de summary.json; no se elige horizonte usando esas tablas.

## Costos y límites

Costos base: `{'commission_bps': 10.0, 'slippage_bps': 5.0, 'spread_bps': 2.0, 'annual_holding_bps': 0.0, 'annual_cash_rate': 0.0, 'initial_capital': 10000.0}`. Stress prefijado multiplica por 2.0 la comisión, slippage, spread y costo de tenencia; conserva modelos, señales y umbrales. Spread y slippage declarados son supuestos de investigación, no mediciones del libro ni garantía de fill. Costos por lado; benchmark también paga entrada/salida. Efectivo usa la tasa declarada (0 por defecto).

Drawdown marcado a cada apertura: no incluye mínimos intravela. Sin latencia, impacto dependiente del tamaño, impuestos, lotes, financiación variable ni riesgo de contraparte. El módulo permite spread y financiación configurables y gaps de sesiones; Forex/futuros necesitan adaptadores de calendario, rolls/contratos y costos adecuados. No se afirma validación en esos mercados.

Los modelos actuales originales se cargan sin modificarlos para el historial posterior a su selección. En desarrollo se ajusta una referencia con sus features y cantidad fija de árboles histórica, y recalibración temporal disjunta si corresponde; no se presenta como réplica exacta de la búsqueda histórica de cada receta. Sus parámetros heredados también tienen sesgo de selección histórica. No se usan modelos entrenados en el futuro de un fold.

## Reserva prospectiva

Inicio UTC: 2026-10-14T00:00:00+00:00; fin exclusivo: 2027-04-12T00:00:00+00:00. Estado: pending_future_data. Se guardan los cuatro modelos Momentum congelados y los hashes del protocolo, datos y código. No hay promoción ni operaciones reales.

Cada experimento, abstención, fallo y stress se conserva en experiments.jsonl y archivos individuales. Signals, labels, trades, equity y UBJSON permiten auditar resultados negativos y reproducir la ronda.

## Reproducir e integrar

Ejecutar desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m trading_ml momentum-study --protocol config/momentum_study.toml
.\.venv\Scripts\python.exe scripts/record_momentum_study.py artifacts/momentum_studies/20261013T215021_338a147b
```

Cada ejecución crea un directorio nuevo; nunca reemplaza experimentos, modelos
actuales ni punteros. La comparación de referencias requiere las tres versiones
registradas en el TOML; para otro proyecto indicar versiones existentes o una
lista vacía y registrar esa diferencia. Modificar el protocolo abre una hipótesis
nueva; observar repetidamente el mismo historial no crea un test independiente.

```python
import pandas as pd
from trading_ml.momentum import (
    MomentumConfig,
    ResearchCosts,
    build_momentum_features,
    momentum_signals,
)
from trading_ml.momentum.simulator import simulate_policy

# frame: OHLCV, timestamp UTC de apertura y reference_time UTC de cierre.
# El proveedor externo debe validar precios, cierres y calendario de sesiones.
config = MomentumConfig(lookback=60)
features = build_momentum_features(frame, config)
signals = momentum_signals(features, config)
simulation = simulate_policy(
    features,
    signals,
    pd.Timestamp("2025-06-05", tz="UTC"),
    pd.Timestamp("2026-10-01", tz="UTC"),
    ResearchCosts(),
    config.max_holding_bars,
)
```

El paquete `trading_ml.momentum` mantiene separados config, features, reglas,
simulador, estudio y reporte. `opportunity_labels` crea etiquetas investigativas;
`fit_momentum_filter` ajusta sólo las etiquetas con salida anterior al cutoff.
Los modelos de `frozen/L*/model.ubj` son XGBoost nativos; `fit.json` y `plan.json`
guardan contrato, hashes, corte y umbrales. Un modelo que se abstuvo no tiene UBJSON.

Evidencia de esta ejecución: `20261013T215021_338a147b`; 212
registros y 45 ajustes XGBoost. La ejecución técnica
anterior, si existe en artifacts/momentum_studies, también permanece conservada;
ninguna ejecución cambia los horizontes en respuesta a sus resultados.

La reserva futura también queda protegida por `protect_reserved_labels`, usada
por los comandos actuales de entrenamiento. No se ejecutó una evaluación futura
porque los datos todavía no existen. Reusar esos datos para entrenar invalidaría
su condición de test; el guard rechaza etiquetas que los invadan.
