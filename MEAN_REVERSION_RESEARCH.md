# Investigación estadística de Mean Reversion

Run: `20261013T220145_a6d4131b`.

**Ninguna variante supera todos los criterios históricos prefijados.**

No hay ventaja confirmada en un test futuro independiente. Todos los umbrales, ventanas y filtros de régimen se fijaron antes de ejecutar esta ronda; no se selecciona un ganador. El historial de junio de 2025 a septiembre de 2026 ya fue observado por estudios previos y se informa como descripción histórica, sin presentarlo como prueba ciega.

## Auditoría de variables existentes

`features/technical.py` calcula RSI con medias exponenciales de ganancias/pérdidas, normalizado a [0,1], y SMA ratios close/SMA−1 para 10/20/50 velas. Estas distancias no están expresadas en unidades de desviación estándar. `volatility` mide desviación de retornos simples de una vela, no de niveles del precio. La versión ampliada añade ATR relativo y momentum ajustado por volatilidad; ninguna equivale a un z-score del precio. Se conservan intactas esas features y los modelos que las usan.

## Definición matemática y riesgo

Para cierre Pₜ, μₜ=media(Pₜ₋₁,…,Pₜ₋L), sₜ=desviación muestral de esos L precios (ddof=1), zₜ=(Pₜ−μₜ)/sₜ. La distribución de referencia excluye el cierre actual. Se calculan también z-score logarítmico, distancia porcentual, s/μ, RSI, volatilidad de log retornos, ratio de volatilidad, ATR/precio, pendientes, autocorrelación de retornos y proporción de velas bajistas.

Entrada long si z≤-2.0 y RSI≤0.4, con features completas. Ventanas: [20, 60] velas. Filtros fijos: ['all', 'sideways']. La condición extrema es una distancia descriptiva: no se asume normalidad ni una probabilidad universal para z=−2. Tampoco se supone estacionariedad del precio.

Entrada en la apertura posterior al cierre de señal. La media μ de ese cierre queda fija como objetivo de la operación. Una media que se desplaza hacia el precio no basta para declarar reversión. Se descarta una apertura cuyo gap ya superó el objetivo o cuyo beneficio al objetivo no cubre costos.

Stop de referencia = apertura de entrada − 2.0×ATR conocido al cierre anterior. Fracción de cartera = min(0.5, 0.01/pérdida neta unitaria prevista al stop). El presupuesto incluye costos de entrada/salida; sin leverage, averaging down ni posiciones superpuestas. Al cierre, se evalúa primero stop, luego objetivo y luego timeout; se ejecuta en la próxima apertura. Plazo máximo: 10 velas. No se reingresa en la apertura de salida.

El stop es una señal al cierre, **no una garantía de pérdida máxima**: un gap o un movimiento intravela puede exceder el presupuesto. Se registra cada exceso realizado. El drawdown se marca a cada apertura y no incluye mínimos intravela. Al final de cada fold se liquida; estas salidas administrativas no se usan como etiquetas artificiales de reversión.

## Etiquetas y modelos

Y=1 si el cierre alcanza la media anclada antes del stop o timeout; Y=0 si sale por stop/timeout. Las etiquetas sin desenlace observado permanecen censuradas, y las oportunidades con entrada no ejecutable se registran por separado. Una reversión puede perder después de costos o del gap de salida; se informan por separado frecuencia de reversión, win rate y rentabilidad.

Se entrena sólo con oportunidades anteriores al fold cuyas aperturas reales de salida sean estrictamente anteriores a su inicio. Purga por target_time, incluyendo stops y timeouts. Las oportunidades hipotéticas pueden solaparse; su número no equivale al número de operaciones independientes. Folds cronológicos expansivos; nunca se ajusta scaler o modelo con datos de validación/test.

Logística L2: C=1.0, StandardScaler ajustado exclusivamente en train. XGBoost: 120 árboles, profundidad 2, parámetros fijos. Ambos estiman P(Y=1) y filtran la misma oportunidad con umbral 0.55; comparten salidas y sizing. No predicen una ganancia neta ni garantizan una expectativa positiva por superar ese umbral. Si falta muestra o hay una sola clase, se abstienen y no inventan probabilidades.

Se conservan log-loss, Brier, AUC y reliability bins por fold, junto a baselines p=0,5 y prevalencia de train. Los coeficientes/logística se guardan en JSON y XGBoost en UBJSON; no se usa pickle ni se promueve ningún modelo.

## Resultados netos, expectativa y estabilidad

Todos los sistemas comparten las mismas aperturas por fold. Buy and hold invierte 100%, mientras MR limita exposición y conserva efectivo: comparar drawdowns no equivale a demostrar alpha a riesgo igual. La concatenación walk-forward reinicia en efectivo cada fold y paga costos de liquidación.

### development

| Sistema | Neto cartera | Drawdown | Trades | Expectativa por trade¹ | Turnover | Folds positivos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold | 116.69% | -66.74% | 7 | — | 13.99 | 71.4% |
| cash | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_all_rules | -4.67% | -7.82% | 30 | -1.04% | 6.90 | 42.9% |
| mr_L20_all_logistic | -4.12% | -4.98% | 5 | -5.82% | 1.27 | 0.0% |
| mr_L20_all_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_sideways_rules | -1.39% | -5.36% | 23 | -0.38% | 5.22 | 42.9% |
| mr_L20_sideways_logistic | -0.42% | -0.45% | 1 | -2.27% | 0.36 | 0.0% |
| mr_L20_sideways_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_all_rules | 2.86% | -3.07% | 15 | 1.38% | 2.97 | 57.1% |
| mr_L60_all_logistic | 1.63% | -0.55% | 2 | 8.10% | 0.44 | 28.6% |
| mr_L60_all_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_sideways_rules | 3.06% | -2.03% | 11 | 2.52% | 2.42 | 57.1% |
| mr_L60_sideways_logistic | 0.82% | -0.55% | 1 | 6.23% | 0.27 | 14.3% |
| mr_L60_sideways_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |

¹Promedio del retorno neto sobre el capital asignado a cada trade, sin confundirlo con retorno de toda la cartera. Expectativa de contribución a cartera, profit factor, win rate, duración, salidas y excesos de riesgo están en summary.json. Pocas operaciones pueden producir promedios muy inestables.

### Mercado al momento de entrar

Alcista/bajista/lateral: momentum log de 60 velas dividido por volatilidad de 20×√60, bandas +1/−1. Es causal al cierre de señal y compartido por las variantes; el filtro lateral se fijó antes de la ronda. Las tablas no buscan un régimen ganador después de observar resultados.

| Sistema | Alcista: trades, expectativa | Bajista: trades, expectativa | Lateral: trades, expectativa |
| --- | ---: | ---: | ---: |
| mr_L20_all_rules | 2, -7.54% | 5, -1.52% | 23, -0.36% |
| mr_L20_all_logistic | 1, -7.63% | 3, -3.87% | 1, -9.84% |
| mr_L20_all_xgb | 0, — | 0, — | 0, — |
| mr_L20_sideways_rules | 0, — | 0, — | 23, -0.38% |
| mr_L20_sideways_logistic | 0, — | 0, — | 1, -2.27% |
| mr_L20_sideways_xgb | 0, — | 0, — | 0, — |
| mr_L60_all_rules | 0, — | 4, -1.75% | 11, 2.52% |
| mr_L60_all_logistic | 0, — | 1, 9.98% | 1, 6.23% |
| mr_L60_all_xgb | 0, — | 0, — | 0, — |
| mr_L60_sideways_rules | 0, — | 0, — | 11, 2.52% |
| mr_L60_sideways_logistic | 0, — | 0, — | 1, 6.23% |
| mr_L60_sideways_xgb | 0, — | 0, — | 0, — |

Equity por régimen conocido al inicio del intervalo, retornos por trimestre y métricas por fold también se conservan; los regímenes reúnen intervalos discontinuos y no representan carteras separadas.

### Evidencia estadística

- mr_L20_all_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_all_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_all_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.

Bootstrap pareado de retornos log netos, bloques dentro de folds de [7, 30] velas y 2000 remuestras. Comparaciones contra efectivo y buy and hold, y ML contra sus propias reglas. Bonferroni por período sobre 64 comparaciones×bloques. Los intervalos son aproximados y condicionales: las colas extremas tienen pocas remuestras y no se corrigen todas las decisiones históricas del proyecto. No constituyen una prueba futura independiente ni intervalos iid por trade.

### development_stress

| Sistema | Neto cartera | Drawdown | Trades | Expectativa por trade¹ | Turnover | Folds positivos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold | 111.89% | -66.84% | 7 | — | 13.98 | 71.4% |
| cash | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_all_rules | -5.41% | -7.83% | 30 | -1.35% | 6.63 | 42.9% |
| mr_L20_all_logistic | -4.08% | -4.84% | 5 | -6.12% | 1.21 | 0.0% |
| mr_L20_all_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_sideways_rules | -2.11% | -5.37% | 23 | -0.70% | 5.03 | 42.9% |
| mr_L20_sideways_logistic | -0.45% | -0.45% | 1 | -2.58% | 0.34 | 0.0% |
| mr_L20_sideways_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_all_rules | 2.29% | -3.09% | 15 | 1.06% | 2.88 | 57.1% |
| mr_L60_all_logistic | 1.51% | -0.52% | 2 | 7.76% | 0.42 | 28.6% |
| mr_L60_all_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_sideways_rules | 2.57% | -2.02% | 11 | 2.20% | 2.33 | 57.1% |
| mr_L60_sideways_logistic | 0.75% | -0.52% | 1 | 5.89% | 0.26 | 14.3% |
| mr_L60_sideways_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |

¹Promedio del retorno neto sobre el capital asignado a cada trade, sin confundirlo con retorno de toda la cartera. Expectativa de contribución a cartera, profit factor, win rate, duración, salidas y excesos de riesgo están en summary.json. Pocas operaciones pueden producir promedios muy inestables.

### Mercado al momento de entrar

Alcista/bajista/lateral: momentum log de 60 velas dividido por volatilidad de 20×√60, bandas +1/−1. Es causal al cierre de señal y compartido por las variantes; el filtro lateral se fijó antes de la ronda. Las tablas no buscan un régimen ganador después de observar resultados.

| Sistema | Alcista: trades, expectativa | Bajista: trades, expectativa | Lateral: trades, expectativa |
| --- | ---: | ---: | ---: |
| mr_L20_all_rules | 2, -7.84% | 5, -1.83% | 23, -0.68% |
| mr_L20_all_logistic | 1, -7.93% | 3, -4.18% | 1, -10.12% |
| mr_L20_all_xgb | 0, — | 0, — | 0, — |
| mr_L20_sideways_rules | 0, — | 0, — | 23, -0.70% |
| mr_L20_sideways_logistic | 0, — | 0, — | 1, -2.58% |
| mr_L20_sideways_xgb | 0, — | 0, — | 0, — |
| mr_L60_all_rules | 0, — | 4, -2.07% | 11, 2.20% |
| mr_L60_all_logistic | 0, — | 1, 9.63% | 1, 5.89% |
| mr_L60_all_xgb | 0, — | 0, — | 0, — |
| mr_L60_sideways_rules | 0, — | 0, — | 11, 2.20% |
| mr_L60_sideways_logistic | 0, — | 0, — | 1, 5.89% |
| mr_L60_sideways_xgb | 0, — | 0, — | 0, — |

Equity por régimen conocido al inicio del intervalo, retornos por trimestre y métricas por fold también se conservan; los regímenes reúnen intervalos discontinuos y no representan carteras separadas.

### Evidencia estadística

- mr_L20_all_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_all_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_all_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.

Bootstrap pareado de retornos log netos, bloques dentro de folds de [7, 30] velas y 2000 remuestras. Comparaciones contra efectivo y buy and hold, y ML contra sus propias reglas. Bonferroni por período sobre 64 comparaciones×bloques. Los intervalos son aproximados y condicionales: las colas extremas tienen pocas remuestras y no se corrigen todas las decisiones históricas del proyecto. No constituyen una prueba futura independiente ni intervalos iid por trade.

### observed_historical

| Sistema | Neto cartera | Drawdown | Trades | Expectativa por trade¹ | Turnover | Folds positivos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold | -20.34% | -52.97% | 1 | — | 2.00 | 0.0% |
| cash | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_all_rules | -3.64% | -8.38% | 12 | -2.83% | 3.58 | 0.0% |
| mr_L20_all_logistic | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_all_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_sideways_rules | -3.35% | -7.36% | 9 | -3.26% | 2.89 | 0.0% |
| mr_L20_sideways_logistic | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_sideways_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_all_rules | -5.63% | -7.02% | 8 | -5.69% | 1.83 | 0.0% |
| mr_L60_all_logistic | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_all_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_sideways_rules | -5.43% | -6.48% | 6 | -7.15% | 1.44 | 0.0% |
| mr_L60_sideways_logistic | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_sideways_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |

¹Promedio del retorno neto sobre el capital asignado a cada trade, sin confundirlo con retorno de toda la cartera. Expectativa de contribución a cartera, profit factor, win rate, duración, salidas y excesos de riesgo están en summary.json. Pocas operaciones pueden producir promedios muy inestables.

### Mercado al momento de entrar

Alcista/bajista/lateral: momentum log de 60 velas dividido por volatilidad de 20×√60, bandas +1/−1. Es causal al cierre de señal y compartido por las variantes; el filtro lateral se fijó antes de la ronda. Las tablas no buscan un régimen ganador después de observar resultados.

| Sistema | Alcista: trades, expectativa | Bajista: trades, expectativa | Lateral: trades, expectativa |
| --- | ---: | ---: | ---: |
| mr_L20_all_rules | 1, -10.61% | 3, -1.23% | 8, -2.46% |
| mr_L20_all_logistic | 0, — | 0, — | 0, — |
| mr_L20_all_xgb | 0, — | 0, — | 0, — |
| mr_L20_sideways_rules | 0, — | 0, — | 9, -3.26% |
| mr_L20_sideways_logistic | 0, — | 0, — | 0, — |
| mr_L20_sideways_xgb | 0, — | 0, — | 0, — |
| mr_L60_all_rules | 0, — | 3, -0.37% | 5, -8.89% |
| mr_L60_all_logistic | 0, — | 0, — | 0, — |
| mr_L60_all_xgb | 0, — | 0, — | 0, — |
| mr_L60_sideways_rules | 0, — | 0, — | 6, -7.15% |
| mr_L60_sideways_logistic | 0, — | 0, — | 0, — |
| mr_L60_sideways_xgb | 0, — | 0, — | 0, — |

Equity por régimen conocido al inicio del intervalo, retornos por trimestre y métricas por fold también se conservan; los regímenes reúnen intervalos discontinuos y no representan carteras separadas.

### Evidencia estadística

- mr_L20_all_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_all_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_all_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.

Bootstrap pareado de retornos log netos, bloques dentro de folds de [7, 30] velas y 2000 remuestras. Comparaciones contra efectivo y buy and hold, y ML contra sus propias reglas. Bonferroni por período sobre 64 comparaciones×bloques. Los intervalos son aproximados y condicionales: las colas extremas tienen pocas remuestras y no se corrigen todas las decisiones históricas del proyecto. No constituyen una prueba futura independiente ni intervalos iid por trade.

### observed_historical_stress

| Sistema | Neto cartera | Drawdown | Trades | Expectativa por trade¹ | Turnover | Folds positivos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_hold | -20.60% | -52.97% | 1 | — | 2.00 | 0.0% |
| cash | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_all_rules | -4.06% | -8.27% | 12 | -3.15% | 3.41 | 0.0% |
| mr_L20_all_logistic | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_all_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_sideways_rules | -3.67% | -7.22% | 9 | -3.57% | 2.75 | 0.0% |
| mr_L20_sideways_logistic | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L20_sideways_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_all_rules | -5.69% | -6.99% | 8 | -5.99% | 1.76 | 0.0% |
| mr_L60_all_logistic | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_all_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_sideways_rules | -5.43% | -6.40% | 6 | -7.45% | 1.38 | 0.0% |
| mr_L60_sideways_logistic | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |
| mr_L60_sideways_xgb | 0.00% | 0.00% | 0 | — | 0.00 | 0.0% |

¹Promedio del retorno neto sobre el capital asignado a cada trade, sin confundirlo con retorno de toda la cartera. Expectativa de contribución a cartera, profit factor, win rate, duración, salidas y excesos de riesgo están en summary.json. Pocas operaciones pueden producir promedios muy inestables.

### Mercado al momento de entrar

Alcista/bajista/lateral: momentum log de 60 velas dividido por volatilidad de 20×√60, bandas +1/−1. Es causal al cierre de señal y compartido por las variantes; el filtro lateral se fijó antes de la ronda. Las tablas no buscan un régimen ganador después de observar resultados.

| Sistema | Alcista: trades, expectativa | Bajista: trades, expectativa | Lateral: trades, expectativa |
| --- | ---: | ---: | ---: |
| mr_L20_all_rules | 1, -10.90% | 3, -1.54% | 8, -2.78% |
| mr_L20_all_logistic | 0, — | 0, — | 0, — |
| mr_L20_all_xgb | 0, — | 0, — | 0, — |
| mr_L20_sideways_rules | 0, — | 0, — | 9, -3.57% |
| mr_L20_sideways_logistic | 0, — | 0, — | 0, — |
| mr_L20_sideways_xgb | 0, — | 0, — | 0, — |
| mr_L60_all_rules | 0, — | 3, -0.69% | 5, -9.18% |
| mr_L60_all_logistic | 0, — | 0, — | 0, — |
| mr_L60_all_xgb | 0, — | 0, — | 0, — |
| mr_L60_sideways_rules | 0, — | 0, — | 6, -7.45% |
| mr_L60_sideways_logistic | 0, — | 0, — | 0, — |
| mr_L60_sideways_xgb | 0, — | 0, — | 0, — |

Equity por régimen conocido al inicio del intervalo, retornos por trimestre y métricas por fold también se conservan; los regímenes reúnen intervalos discontinuos y no representan carteras separadas.

### Evidencia estadística

- mr_L20_all_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_all_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_all_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L20_sideways_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_all_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_rules: criterio histórico=False; aporte del modelo frente a reglas=None; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_logistic: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.
- mr_L60_sideways_xgb: criterio histórico=False; aporte del modelo frente a reglas=False; fallas: too_few_trades, unstable_across_windows, nonpositive_return_or_expectancy, no_positive_adjusted_lower_bound_vs_cash, no_positive_adjusted_lower_bound_vs_buy_hold.

Bootstrap pareado de retornos log netos, bloques dentro de folds de [7, 30] velas y 2000 remuestras. Comparaciones contra efectivo y buy and hold, y ML contra sus propias reglas. Bonferroni por período sobre 64 comparaciones×bloques. Los intervalos son aproximados y condicionales: las colas extremas tienen pocas remuestras y no se corrigen todas las decisiones históricas del proyecto. No constituyen una prueba futura independiente ni intervalos iid por trade.

## Frecuencia de oportunidades

Se cuentan todas las señales extremas, episodios nuevos y etiquetas realmente observadas, incluidos los casos censurados. La tasa de reversión siguiente se refiere a trades hipotéticos solapados, no a operaciones ejecutadas independientes; no se interpreta como probabilidad iid.

### development

| Receta / fold | Oportunidades | Episodios | Etiquetas observadas | Reversión | Censuradas/cruzan límite |
| --- | ---: | ---: | ---: | ---: | ---: |
| mr_L20_all / 1 | 26 | 7 | 26 | 0.00% | 0 |
| mr_L20_all / 2 | 10 | 4 | 10 | 0.00% | 0 |
| mr_L20_all / 3 | 7 | 4 | 7 | 71.43% | 0 |
| mr_L20_all / 4 | 5 | 1 | 5 | 0.00% | 0 |
| mr_L20_all / 5 | 6 | 5 | 5 | 20.00% | 1 |
| mr_L20_all / 6 | 10 | 4 | 10 | 40.00% | 0 |
| mr_L20_all / 7 | 10 | 3 | 10 | 30.00% | 0 |
| mr_L20_sideways / 1 | 12 | 4 | 12 | 0.00% | 0 |
| mr_L20_sideways / 2 | 10 | 4 | 10 | 0.00% | 0 |
| mr_L20_sideways / 3 | 5 | 4 | 5 | 80.00% | 0 |
| mr_L20_sideways / 4 | 4 | 1 | 4 | 0.00% | 0 |
| mr_L20_sideways / 5 | 6 | 5 | 5 | 20.00% | 1 |
| mr_L20_sideways / 6 | 8 | 5 | 8 | 50.00% | 0 |
| mr_L20_sideways / 7 | 7 | 2 | 7 | 0.00% | 0 |
| mr_L60_all / 1 | 21 | 5 | 21 | 0.00% | 0 |
| mr_L60_all / 2 | 13 | 3 | 13 | 7.69% | 0 |
| mr_L60_all / 3 | 1 | 1 | 1 | 100.00% | 0 |
| mr_L60_all / 4 | 11 | 1 | 11 | 0.00% | 0 |
| mr_L60_all / 5 | 3 | 1 | 3 | 0.00% | 0 |
| mr_L60_all / 6 | 5 | 3 | 5 | 0.00% | 0 |
| mr_L60_all / 7 | 9 | 4 | 9 | 0.00% | 0 |
| mr_L60_sideways / 1 | 9 | 2 | 9 | 0.00% | 0 |
| mr_L60_sideways / 2 | 13 | 3 | 13 | 7.69% | 0 |
| mr_L60_sideways / 3 | 1 | 1 | 1 | 100.00% | 0 |
| mr_L60_sideways / 4 | 7 | 3 | 7 | 0.00% | 0 |
| mr_L60_sideways / 5 | 3 | 1 | 3 | 0.00% | 0 |
| mr_L60_sideways / 6 | 4 | 2 | 4 | 0.00% | 0 |
| mr_L60_sideways / 7 | 9 | 4 | 9 | 0.00% | 0 |
### observed_historical

| Receta / fold | Oportunidades | Episodios | Etiquetas observadas | Reversión | Censuradas/cruzan límite |
| --- | ---: | ---: | ---: | ---: | ---: |
| mr_L20_all / 1 | 33 | 18 | 33 | 12.12% | 0 |
| mr_L20_sideways / 1 | 19 | 12 | 19 | 15.79% | 0 |
| mr_L60_all / 1 | 39 | 6 | 39 | 0.00% | 0 |
| mr_L60_sideways / 1 | 26 | 8 | 26 | 0.00% | 0 |

Se guarda además frecuencia por año, proporción de velas elegibles y régimen. Diagnóstico AR(1) en log-precio usa únicamente el pasado de cada fold. Su φ y vida media condicional son descriptivos; no son un test ADF ni prueban estacionariedad, y no se usan para seleccionar señales o umbrales.

## Costos, alcance y reserva futura

Costos base: `{'commission_bps': 10.0, 'slippage_bps': 5.0, 'spread_bps': 2.0, 'annual_holding_bps': 0.0, 'annual_cash_rate': 0.0, 'initial_capital': 10000.0}`. Stress prefijado ×2.0 en comisión, slippage, spread y tenencia; no reentrena modelos ni cambia umbrales. Spread y slippage son supuestos explícitos, no mediciones del libro. Ambos benchmarks pagan sus costos; efectivo aplica la tasa declarada.

La literatura relaciona reversión corta en acciones con liquidez; no prueba el mismo efecto en BTC ni que un precio individual sea estacionario. Fuentes primarias consultadas: [Nagel, Evaporating Liquidity](https://www.nber.org/papers/w17653), [Dai et al., Reversals and the Returns to Liquidity Provision](https://www.nber.org/papers/w30917). Referencia sobre la hipótesis de raíz unitaria: [documentación de ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html).

OHLCV, UTC y gaps de sesiones pueden provenir de adaptadores externos. Forex y futuros requieren calendario, conversión monetaria, lotes, rolls y financiación reales; el prototipo es long/efectivo sin apalancamiento. No se afirma validación en esos mercados. Tampoco modela liquidez, impacto dependiente del tamaño o impuestos.

Reserva prospectiva UTC: 2026-10-14T00:00:00+00:00 a 2027-04-12T00:00:00+00:00 exclusivo. Estado pending_future_data. Se congelan todas las variantes, incluso las rechazadas; no se elige una usando el historial. El guard de entrenamiento protege las etiquetas reservadas. No se ejecutó una evaluación futura.

Se conservaron 232 registros y 68 ajustes en esta ejecución. Se preservaron hashes de modelos/punteros existentes. Cada fallo, pérdida y abstención queda en experiments.jsonl y archivos individuales; datos, etiquetas, modelos, trades, equity, plan y manifiesto permiten auditarlos. No hay activación ni operaciones reales.

## Reproducir e integrar

```powershell
.\.venv\Scripts\python.exe -m trading_ml mean-reversion-study --protocol config/mean_reversion_study.toml
# Registrar evidencia existente: no entrena ni vuelve a evaluar el test.
.\.venv\Scripts\python.exe scripts/record_mean_reversion_study.py artifacts/mean_reversion_studies/20261013T220145_a6d4131b
```

Cada ejecución crea un directorio nuevo en artifacts/mean_reversion_studies.
Se conservan todas las variantes, errores, pérdidas, abstenciones y casos censurados.
El protocolo ejecutado está copiado en protocol.toml y plan.json antes de ajustar
modelos; no se cambia en respuesta a los resultados.

```python
import pandas as pd
from trading_ml.mean_reversion import (
    ResearchCosts,
    ReversionConfig,
    build_reversion_features,
    reversion_opportunities,
)
from trading_ml.mean_reversion.simulator import simulate_reversion

# frame: OHLCV válido, timestamp UTC de apertura y reference_time UTC de cierre.
# Los adaptadores externos validan cierres, sesiones, lotes y contratos.
config = ReversionConfig(lookback=20, sideways_only=False)
features = build_reversion_features(frame, config)
opportunities = reversion_opportunities(features, config)
result = simulate_reversion(
    features,
    opportunities,
    pd.Timestamp("2025-06-05", tz="UTC"),
    pd.Timestamp("2026-10-01", tz="UTC"),
    config,
    ResearchCosts(),
)
```

`models.fit_filter` permite ajustar logística/XGBoost sobre etiquetas purgadas;
`models.load_filter` carga JSON/UBJSON y `filter.probabilities(features)` estima
la reversión. Pasar esas probabilidades a `simulate_reversion` sólo filtra entradas.
Para evaluación fuera de muestra, exigir referencias posteriores al corte guardado
y conservar los umbrales del plan. El runner aplica estos controles temporalmente;
el API de bajo nivel permite pruebas sintéticas y exige que el llamador mantenga
el contrato fuera de muestra.

Los módulos config, features, rules, simulator, models, statistics, study y report
son independientes del proveedor y no contienen envío de órdenes. Reutilizan costos,
validación de timeline y agregación de equity del motor de investigación existente;
no cambian las estrategias o features actuales. El runner de ingestión sigue usando
el calendario cripto actual; otros mercados necesitan su propio adaptador.

Las probabilidades se refieren a alcanzar la media anclada **dentro de 10 velas**,
no a cualquier rebote o ganancia parcial. Un timeout rentable sigue teniendo Y=0.
Cuando el objetivo es raro, especialmente con ventanas largas, los filtros pueden
permanecer en efectivo. Ese 0% no acredita capacidad predictiva ni ventaja frente
a efectivo. Las tablas por régimen no crean nuevos tests independientes.

Ejecución documentada: `20261013T220145_a6d4131b`. Los modelos actuales y sus punteros
se verificaron por hash antes y después del estudio. El historial reservado futuro
todavía no está disponible y queda protegido por el guard compartido de entrenamiento.
