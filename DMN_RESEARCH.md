# Deep Momentum Networks: diseño y protocolo

## Fuentes primarias consultadas

- Lim, Zohren y Roberts, [arXiv:1904.04912v3](https://arxiv.org/pdf/1904.04912), §§III–VI y apéndices A–C.
- Wood, Roberts y Zohren, [Slow Momentum with Fast Reversion](https://arxiv.org/abs/2105.13727): extensión con detección de cambios; candidata futura, sin incorporarla antes de establecer controles.
- [DeePM, repositorio de sus autores](https://github.com/kieranjwood/deepm): referencia complementaria de sistemas multi-activo, no código importado ni réplica del trabajo de 2019.
- [Documentación oficial PyTorch](https://pytorch.org/get-started/locally/) para compatibilidad e instalación CPU.

## Lectura del trabajo original

El paper combina posiciones continuas acotadas, escalado por volatilidad ex ante y entrenamiento directo por retorno o Sharpe. Compara modelos lineales, MLP, WaveNet y LSTM con clasificación/regresión y reglas tradicionales. Utiliza 88 futuros continuos ajustados de 1990–2015, recalibración expansiva cada cinco años, validación temporal final del 10%, Adam y búsqueda aleatoria de 50 configuraciones. La LSTM usa trayectorias de 63 pasos. Incluye retornos normalizados de 1/21/63/126/252 días y tres MACD normalizados. El escalado individual usa volatilidad exponencial de span 60 y objetivo anual 15%. Los resultados distinguen señal bruta, reescalado de cartera y costos; la ventaja de LSTM sin regularización es sensible al turnover. No constituyen evidencia de rentabilidad en otro mercado.

Referencias numéricas de sus tablas 2/3/8: Sharpe LSTM sin costos 2,804 frente a 1,192 de TSMOM; con escalado adicional, 2,907 frente a 1,392; con costos de 10 bps, LSTM sin regularización −5,313 y con regularización 0,912. Son cifras reportadas por los autores en futuros, no resultados obtenidos aquí.

## Auditoría del proyecto existente

- Hay descarga Binance con checksum, validación OHLCV estricta, contratos, walk-forward, baselines, bootstrap por bloques y protección prospectiva aprovechables.
- Los simuladores históricos son de operaciones discretas long/cash y sus pérdidas de clasificación no implementan la posición continua ni el objetivo DMN.
- BTC/ETH diarios 2020-01-01 a 2026-10-01 ya existen con manifiestos. Son datos observados: toda nueva evaluación sobre ellos será desarrollo histórico, aunque las predicciones de cada fold sean temporales fuera de muestra.
- Se conservarán modelos, punteros, estudios previos y cambios locales. No se abren reservas prospectivas ni se promueve un modelo automáticamente.

## Decisiones de adaptación (fijadas antes de entrenar)

1. Paquete `trading_ml.dmn`, CLI propia y dependencia PyTorch opcional. Los contratos anteriores siguen siendo utilizables.
2. Primer universo BTC/ETH spot diario, calendario UTC continuo, 365 períodos/año. Elección por cobertura pública verificable y datos locales; dos criptoactivos correlacionados no sustituyen la diversificación del paper. Futuros diversificados requieren precios continuos, roll y licencias verificables antes de afirmar una réplica. Forex existente requiere tratar huecos/calendario/spreads y no se mezclará artificialmente con cripto.
3. Ocho variables retrospectivas. MACD con `alpha=1/S`, no `span=S`; normalización de precio a 63 observaciones y del MACD a 252. EWM causal. Escalador de features aprendido exclusivamente en train, con clipping fijo a 5 desviaciones del train: desviación explícita respecto de la winsorización EWM del original.
4. Señal después de cierre de t, ejecución en open[t+1], retorno open[t+2]/open[t+1]-1. Volatilidad calculada con cierres conocidos hasta t. Se supone ejecución en la apertura siguiente sin latencia adicional; se evalúan costos adversos. No confundir este objetivo con close/close.
5. Posición por activo `w = clip(X * 0.15 / (sigma_daily * sqrt(365)), límites) / N`. Exposición bruta limitada por construcción. Modo principal long/cash spot; long/short sólo como sensibilidad de exposición sintética con costo de préstamo explícito, sin presentarlo como ejecución spot disponible.
6. Backtest diario de cartera con deriva de pesos entre aperturas, rebalanceo, comisión/slippage/spread por notional negociado, financiación, liquidación final y equity marcada cada día. El turnover de entrenamiento de tipo paper se distingue del turnover ejecutable con deriva de cartera.
7. Walk-forward expansivo; validación interior de 180 días; purga por disponibilidad de la etiqueta y embargo de un día. Contexto LSTM retrospectivo de 63 observaciones, estado reiniciado por secuencia. Escalador, checkpoint y early stopping nunca usan el fold exterior. Esta LSTM de ventanas solapadas difiere del entrenamiento de trayectorias del paper y no usa dropout recurrente variacional.
8. Presupuesto acotado: controles cash, long con y sin escalado, TSMOM anual y MACD; modelos lineal/MLP/LSTM, ablación de costos y objetivo supervisado XGBoost sobre las mismas fechas. Semillas fijadas y reportadas individualmente y como ensemble de posiciones; ninguna semilla se elige mirando resultados exteriores.
9. Las variantes se comparan como hipótesis predeclaradas. Bootstrap pareado por bloques de fechas, manteniendo juntos los activos y sin cruzar folds; intervalos descriptivos y sin corrección postselección. Costos base y dobles. Informar rentabilidad, volatilidad, Sharpe, drawdown, turnover y estabilidad por fold/activo. No declarar ventaja independiente a partir de este desarrollo.

## Estado

Diseño previo al entrenamiento guardado el 2026-10-16. Los dos protocolos se completaron con 120 ajustes; evidencia y conclusiones en `DMN_RESULTS.md`. Ver `CODEX_PROGRESS.md` para estado real, comandos y próximos pasos.

## Sensibilidad de dominio predeclarada

Antes de consultar el resultado agregado del primer estudio se fija `config/dmn_long_short.toml`: mismo universo, fechas, semillas, features, capacidad, pérdidas, costos y 60 ajustes; cambia únicamente el dominio permitido a long/short. Es una comparación con el dominio de posiciones del paper, necesaria para distinguir una falla de la red de la restricción long/cash. Exposiciones sintéticas sin apalancamiento, préstamo supuesto de 300 bps/año y su sensibilidad doble; sin afirmar disponibilidad de esos shorts en spot. Se publicarán ambos estudios completos aunque esta sensibilidad empeore los resultados. No se elegirán hiperparámetros ni semillas a partir de sus folds exteriores.
