# Expansión de mercados y datos — protocolo previo

Fecha: 2026-10-16. Toda evaluación sobre historia ya disponible es desarrollo retrospectivo. No se abrirán reservas existentes ni se activarán modelos.

## Fuentes primarias y decisiones

| Fuente | Evidencia | Decisión |
| --- | --- | --- |
| [Binance Public Data](https://github.com/binance/binance-public-data) y [condiciones](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md) | Archivos públicos spot/derivados con SHA256; prueba directa de funding y mark-price: HTTP 200 | Adquirir BTC, ETH, SOL, BNB y XRP spot; incorporar funding liquidado y prima perp/spot como información externa, sin simular operaciones perpetuas |
| [FXCM MarketData](https://github.com/fxcm/MarketData) | Uso personal sujeto a EULA; bid/ask indicativo, no volumen mundial FX; USDCAD/NZDUSD H1 2023 respondieron 200 | Completar siete pares; conservar NY17/DST, bid/ask y huecos reales. No redistribuir datos |
| [CME DataMine](https://www.cmegroup.com/datamine/datamine-api.html) | Datos contratados con autenticación; licencias y costos explícitos | Futuros ES/NQ/YM/RTY, GC/SI/CL/NG: catálogo y contrato de importación, sin inventar acceso ni mezclar CFDs/ETF con futuros |
| [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) | Prueba HTTPS de USA500IDXUSD falló por timeout | Cotizaciones/CFD de índices y materias primas pendientes; sin sustituirlas por futuros negociados |
| [ALFRED](https://alfred.stlouisfed.org/help) | Guarda versiones históricas y revisiones | Sólo admitir macro con vintage y disponibilidad de publicación. No usar CPI/empleo revisados de hoy en predicciones pasadas |

La prueba directa de SPX500 H1 en FXCM devolvió 404. No confundir una muestra de sentimiento de un mes con un histórico de precios disponible. No se compran licencias ni se contacta a proveedores.

## Hipótesis y presupuesto fijados antes de entrenar

Ventana común 2021-01-01 a 2025-07-01, elegida por cobertura de FXCM existente, sin consultar resultados nuevos. Universo ancla BTC/ETH; universo ampliado BTC/ETH/SOL/BNB/XRP. Evaluación sobre **los mismos BTC/ETH y las mismas fechas** para medir transferencia; las carteras de cinco activos se informan por separado. Fechas exteriores: 2023-07-01, 2024-01-01, 2025-01-01, 2025-07-01. Validación interior 180 días, embargo un día, secuencia 63. Se conservan capacidad/optimizador de la etapa DMN y costos spot 16 bps por notional y lado, más escenario doble.

Seis variantes: `anchor_price` (BTC/ETH), `expanded_price` (cinco cripto), `expanded_flow` (más volumen agresor/anomalías), `expanded_derivatives` (más funding/prima perp-spot), `expanded_context` (más relaciones cripto y siete pares FX conocidos), y `expanded_noise` (mismo número de entradas que contexto completo, con ruido causal independiente como control de capacidad). LSTM Sharpe neto y XGBoost binario; semillas 42 y 1729; tres folds: **72 ajustes**. No elegir una semilla ganadora ni ajustar umbrales por OOS. Una diferencia favorable en estas ablaciones sigue siendo exploratoria y requiere confirmación.

Se compararán pérdidas probabilísticas de XGBoost, retornos/Sharpe/drawdown/turnover de ambas arquitecturas, costos dobles, estabilidad por fold/activo y bootstrap pareado de bloques 7/30. Las diferencias se calcularán entre recetas sobre referencias y etiquetas idénticas. El control de ruido no contiene datos futuros.

Estudio Forex adicional de los dos pares nuevos: XGBoost precio frente a features de tendencia/volatilidad/spread, con ejecución y calendario propios. Se fijará el detalle antes del ajuste una vez validada su cobertura, sin seleccionar instrumentos por rentabilidad.

## Disponibilidad temporal y alcance

- Guardar `event_time`, `available_at`, UTC, frecuencia, timestamps originales, hash, procedencia y política de latencia. La hora de descarga de un archivo histórico no se confunde con el momento de observación pública del mercado.
- Joins externos por `available_at` con tolerancia y política explícita de revisiones; nunca por fecha de observación macro. Prohibir revisiones con disponibilidad retroactiva. Los registros originales se preservan.
- Funding: sumar sólo liquidaciones anteriores al cierre de señal. No usar tasas futuras anunciadas como si ya fueran realizadas. La prima de precio de marca del perpetuo no es basis de un futuro con vencimiento ni yield anualizado.
- El volumen taker mide agresión de transacciones, **no** desequilibrio del libro. [Cont, Kukanov y Stoikov](https://arxiv.org/abs/1011.6402) justifican investigar flujo, pero su evidencia de libro intradiario no valida automáticamente este proxy diario.
- [Crypto Carry, BIS WP 1087](https://www.bis.org/publ/work1087.pdf) motiva investigar derivados; [Liu y Tsyvinski](https://www.nber.org/papers/w24877) motivan momentum/atención. Son hipótesis, no garantías ni resultados replicados aquí.
- OI: implementar transformaciones y adquisición acotada con timestamp; no descargar miles de ZIP diarios sin comprobar cobertura/calidad primero. Liquidez/libro, liquidaciones, opciones, ETF/on-chain/noticias se inventarían si se dedujeran de OHLCV: quedan fuera del primer estudio hasta disponer de históricos adecuados.
- Universo de cripto elegido hoy con supervivencia conocida: documentar sesgo de supervivencia, correlación y dependencia de Binance. Incluir más activos no equivale automáticamente a diversificar mercados.

Estado y reanudación exacta en `CODEX_PROGRESS.md`. Resultados se publicarán completos, incluidos negativos y fuentes rechazadas.

### Auditoría anterior a ajustes: cambio de fuente de prima

La descarga completa BTC mark-price 2021-01 a 2025-06 mostró ocho días ausentes (incluye un bloque de cuatro días en julio de 2021). Se rechazó antes de entrenar, sin interpolar. Los archivos de precios negociados del perpetuo cubren completos los meses afectados. Se adquiere esa fuente para todos los activos y se redefine explícitamente `perp_premium = perp_traded_close / spot_close - 1`; no es precio de marca ni tasa anualizada de futuro con vencimiento. Evidencia en `artifacts/expansion/source_rejections/mark_BTCUSDT_20210101_20250701.json`.
