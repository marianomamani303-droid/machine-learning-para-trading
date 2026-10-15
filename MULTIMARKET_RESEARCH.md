# Motor multimercado: Forex y cripto

Implementación de investigación del 15 de octubre de 2026. No sustituye modelos activos ni envía órdenes. BTC/USDT y ETH/USDT conservan sus comandos, contratos y resultados anteriores. Los comandos `market-*` trabajan en un espacio separado: `artifacts/multimarket/`.

## Proveedor y cobertura

Se validó primero Dukascopy con EUR/USD. Después bloqueó la ampliación mediante HTTP429. El proveedor alternativo predeterminado es **FXCM**, con archivos públicos H1 bid/ask por HTTPS, sin conectar cuentas. Dukascopy se conserva como contraste; no se mezclan proveedores dentro de un dataset. H4 y D1 se construyen desde H1 del proveedor elegido.

**Período comparable:** `config/multimarket_verified_history.toml` utiliza datos 2020–junio de 2025 y seis folds comunes desde 2022 hasta el 1/7/2025. Se eligieron límites a partir de la disponibilidad de precios, antes de entrenar los otros cuatro pares. El protocolo completo hasta septiembre de 2026 (`config/multimarket.toml`) fue rechazado: faltan 2.559/42.049 horas nominales de EUR/USD. La evidencia está en `artifacts/multimarket/data_rejections/`. El archivo verificado no equivale a un historial sin huecos; la auditoría registra las operaciones que atraviesan horas ausentes. Un piloto anterior con corte 1/1/2025 se conserva separado y se excluye de la comparación final por los huecos de fin de 2024.

| Alternativa investigada | OHLC / bid-ask | Cobertura y límites | Acceso y licencia | Decisión |
|---|---|---|---|---|
| FXCM, archivos públicos H1 | OHLC bid/ask, UTC, sin volumen | Archivos semanales; cobertura verificada por timestamps reales. Límite propio de 2 conexiones y separación mínima de 0,3 s entre GETs; no es una cuota contractual | HTTPS sin clave. Uso personal bajo EULA. Precios **indicativos** de spreads Active Trader; no garantizan ejecución minorista | Predeterminado tras HTTP429 de Dukascopy; se registran archivos ausentes/vacíos y cotizaciones cruzadas |
| Dukascopy, BI5 histórico | OHLC H1 separado BID/ASK; también M1 y ticks | Cobertura realmente comprobada en este estudio: enero 2020–septiembre 2026. Un archivo por mes/lado; 162 solicitudes por par. No se encontró una cuota pública contractual ni SLA | Servicio público sin clave. Sin checksum publicado para estos archivos. Acceso gratuito no equivale a licencia abierta de redistribución | Elegido para investigación local; conservar procedencia y restricciones |
| OANDA v20 | Velas bid/ask/mid, granularidades y cierre diario configurables | Paginación: máximo 5.000 velas por solicitud. Cobertura efectiva y permisos dependen del instrumento/cuenta | Token y cuenta compatibles, incluso en práctica; condiciones del broker | Buena segunda fuente para contraste si se dispone de acceso de datos; no se conectó ninguna cuenta |
| HistData | M1 bid; ticks con bid/ask; el volumen de los ejemplos es cero | Descargas por archivos; calendario EST fijo **sin DST** | Descarga gratuita, formatos y condiciones propias; no asumir licencia abierta | Útil como contraste; menos cómodo para actualización incremental automatizada y spreads de velas |
| Alpha Vantage | FX intradía y diario OHLC; sin bid/ask histórico en esos endpoints | FX intradía figura como premium; requiere clave. No asumir que el tier gratuito permite intradía completo | API key y condiciones de su plan | No elegido para simular spreads históricos |

Fuentes consultadas: [datos históricos oficiales de Dukascopy](https://www.dukascopy.com/wiki/en/development/data-export/), [documentación del autor de dukascopy-node](https://www.dukascopy-node.app/), [código actual de su generador de URLs](https://github.com/Leo4815162342/dukascopy-node/blob/master/src/url-generator/index.ts), [OANDA desarrollo y autenticación](https://developer.oanda.com/rest-live-v20/development-guide/), [OANDA candles](https://developer.oanda.com/rest-live-v20/pricing-ep/), [formatos de HistData](https://www.histdata.com/f-a-q/data-files-detailed-specification/), [API oficial Alpha Vantage](https://www.alphavantage.co/documentation/).

Limitación observada: inicialmente el endpoint legado HTTPS de Dukascopy produjo timeouts; HTTP permitió descargar el piloto y más tarde redirigió a HTTPS con HTTP429. Su API HTTPS nueva devolvió 202 sin contenido en las rutas probadas. El protocolo original se conserva en `config/multimarket_dukascopy.toml`, con HTTP explícito; el adaptador exige HTTPS por defecto y nunca baja a HTTP silenciosamente. Los hashes no autentican la parte HTTP de ese piloto. El descargador limita conexiones, mantiene una cola acotada y corta al primer 429. La documentación oficial propone S3 Requester Pays, que requiere credenciales AWS y pagos; no había credenciales configuradas y no se activó.

El proveedor alternativo se investigó en el [repositorio oficial de FXCM](https://github.com/fxcm/MarketData/blob/master/CandleData/README.md) y su [EULA](https://www.fxcm.com/uk/forms/eula/). Su muestra está destinada a uso personal y sus spreads son indicativos de cuentas Active Trader. El contraste independiente de 120 horas con Dukascopy y las anomalías encontradas se guardan en `artifacts/multimarket/provider_research/fxcm_probe_20261015.json`.

No se encontró una licencia abierta general que autorice redistribuir BI5 ni su uso comercial sin condiciones. La [licencia XML publicada por Dukascopy](https://www.dukascopy.com/plugins/cont.php?ref_id=1273) corresponde a otro producto: no se presenta como autorización de estos archivos. Los históricos quedan locales e ignorados por Git. La licencia MIT del cliente dukascopy-node tampoco concede derechos sobre las cotizaciones.

Este proveedor no requiere credenciales. Si se implementa OANDA posteriormente, el token debe leerse desde una variable de entorno como `OANDA_API_TOKEN`, con host de práctica explícito; nunca incluirlo en TOML, logs, Git o manifiestos. No hay integración de cuentas ni rutas para cursar órdenes en esta ampliación.

## Arquitectura y compatibilidad

`multimarket/markets.py` declara mercado, proveedor, base, cotizada, pip, escala de precios, mínimo de unidades y paso. EURUSD, GBPUSD y AUDUSD tienen pip 0,0001; USDJPY 0,01; USDCHF 0,0001. Las precisiones BI5 son 1e-5 y 1e-3 para JPY. Especificaciones de ejecución de investigación, no contratos de un broker particular.

`multimarket/data.py` implementa snapshots, calendario, agregación y el adaptador BI5. `multimarket/fxcm.py` añade gzip CSV semanal por HTTPS con la misma salida bid/ask, sin inventar columnas de volumen. Duplicados idénticos se deduplican y conflictos detienen la descarga. No se corrigen precios ni se rellenan huecos. Los archivos cerrados se reutilizan por hash; `--refresh` revisa archivos anteriores y los períodos recientes se vuelven a consultar. Las descargas nuevas conservan una copia por hash en `cache/<proveedor>/<par>/versions/`. Archivos FXCM ausentes o vacíos no se interpretan automáticamente como cierres: se comparan con el calendario nominal y se rechaza un historial con más de 2% de horas nominales faltantes antes de entrenar.

Los fines de semana siguen Nueva York domingo 17:00–viernes 17:00, con DST mediante `zoneinfo`. El calendario nominal excluye 1 de enero y 25 de diciembre UTC; no pretende ser un calendario oficial exhaustivo del broker. Se registran cierres anticipados, horas nominales ausentes y cotizaciones observadas fuera de ese calendario. Los precios reales observados se conservan; el simulador restringe las **entradas** al calendario nominal. Los bloques festivos no se completan artificialmente.

H4 usa bloques UTC completos de cuatro horas. D1 usa sesiones completas con cierre a las 17:00 de `America/New_York`. Una agregación incompleta queda excluida de features y registrada en calidad. El historial H1 original **permanece como fuente de ejecución**, incluidos precios de bloques H4/D1 excluidos. Una señal entra en la siguiente cotización H1 observada después de su cierre, sin consultar si esa hora formará parte de una futura vela completa.

`multimarket/features.py` reutiliza los indicadores de precio existentes. `features/technical.py` agrega el parámetro opcional `include_volume`, por defecto **True**; las 14/22/46 features cripto anteriores se conservan. Forex lo llama con False. FXCM no aporta volumen; Dukascopy conserva su volumen de cotización/liquidez como metadato, sin usarlo como volumen global. El mercado FX es OTC y fragmentado, como documenta el [BIS](https://www.bis.org/publications/qr-202512/global-fx-markets-when-hedging-takes-centre-stage).

Una cotización indicativa cruzada conserva sus precios y flags `crossed_open`/`crossed_close`; no se elimina una hora futura completa. El modelo ampliado se abstiene cuando el spread de cierre observado es inválido; una entrada se rechaza si su ask de apertura es menor que su bid, en ese mismo instante. Estos controles no convierten una muestra indicativa en ticks ejecutables.

`multimarket/execution.py` reutiliza el kernel de barreras de `label_research/events.py` y separa barras de señales y ejecución. `study.py` reutiliza métricas, baselines logísticos, calibración, bootstrap temporal y protección de reservas. `service.py` permite entrenar, cargar, predecir y evaluar cada par por separado, en un registro distinto. No existe promoción automática al registro original.

## Hipótesis, targets y evaluación

El target original es dirección entre el cierre actual y un cierre futuro. Esto estima subida de precio, no probabilidad de una operación rentable: una subida menor que spread, slippage, comisiones y carry puede perder dinero. Se conserva como referencia explícita y se compara con:

- Dirección h=1 y h=3, con features de precio y un conjunto compacto ampliado.
- Beneficio neto positivo: entrada en ask siguiente, salida en bid, descontando costos.
- Triple Barrier: TP a 1,5 ATR, SL a 1 ATR y salida temporal; ATR calculado y congelado al cierre de la señal. Se conserva la etiqueta de evento -1/0/+1, pero XGBoost predice beneficio neto positivo, no simplemente qué barrera se toca.
- Meta-labeling long/cash: el padre debe dar p≥0,55 y el segundo modelo filtra la rentabilidad de esa operación. Su entrenamiento usa exclusivamente pronósticos trimestrales OOF del padre, nunca sus probabilidades de entrenamiento.
- Sigmoid sobre una cola temporal separada. Umbrales prefijados 0,50/0,55/0,60/0,65 para documentar cobertura; no se elige retrospectivamente el más rentable.

En Forex la duración de ejecución equivale a 1/4/24 horas de cotizaciones H1 observadas para H1/H4/D1, multiplicada por el horizonte. No son horas civiles: fines de semana, cierres y huecos pueden alargar la duración real, que se registra en cada operación. La dirección sigue h velas completas de features. Las dos definiciones no se confunden, especialmente alrededor de feriados. En cripto se conservan horizontes en velas diarias continuas.

Features: referencia común de 13 variables de precio del modelo original, sin volumen. La receta ampliada suma aceleración, eficiencia de tendencia, distancia estandarizada a media, ubicación en rango, volatilidad relativa, ATR, distancia a máximo **anterior**, hora/día y sesiones locales con DST. En FX agrega spread observado y su relación con la mediana pasada. No hay pivotes retrospectivos, máximos futuros, indicadores centrados ni normalización global. Las sesiones intradía tienen poca variación en D1; sus ablaciones permiten comprobar esa limitación. El comparador `direction_original14_h1` conserva además el volumen real de cada cripto.

Se ejecutan ablaciones quitando cada familia y permutation importance por familia mediante desplazamientos circulares prefijados dentro de cada fold. Se registra gain, pero no se utiliza como prueba de información incremental. Los experimentos de features comparten exactamente las filas de fit, calibración y OOS para el mismo target; no se atribuyen a una feature diferencias causadas por su warm-up. Con variables correlacionadas una permutación puede distribuir u ocultar importancia: ver [documentación de scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html). Estos diagnósticos OOS son exploratorios y no habilitan una selección seguida de una afirmación de test intacto.

El protocolo histórico común tiene cortes 1/1/2022, 1/7/2022, 1/1/2023, 1/7/2023, 1/1/2024, 1/7/2024 y 1/7/2025. El estudio extendido de Dukascopy/Binance conserva sus seis ventanas hasta septiembre de 2026, sin mezclar sus resultados con esta tabla. Fit expansivo, calibración de 180 días y embargo de siete días en ambos límites. La purga utiliza el final real de cada etiqueta. Se excluyen los últimos 20 días civiles de referencias mediante una regla conocida de antemano y se detiene un estudio si aún faltan outcomes dentro de OOS; no se esconden operaciones por censura futura. El simulador usa además un cutoff fijo de entradas. Los hiperparámetros son los del baseline, sin búsqueda en OOS.

Baselines: p=0,5, prevalencia de train, logística L2 con scaler ajustado sólo en train, probabilidades aleatorias con semilla fija, siempre long, momentum y reversión simples. Accuracy y precision clasifican con corte 0,5; las operaciones de referencia exigen 0,55 y su win rate descuenta costos. Los intervalos de diferencia de log-loss frente a prevalencia usan moving-block bootstrap dentro de folds, con bloques 7/30 y 500 remuestras. Son exploratorios sin corrección conjunta por todas las comparaciones; una significancia aislada no autoriza promoción. La [calibración](https://scikit-learn.org/stable/modules/calibration.html) y el [meta-labeling](https://mlfinpy.readthedocs.io/en/stable/Labelling.html) requieren datos ajenos al fit del modelo cuyo score utilizan.

Los períodos cripto anteriores ya habían sido estudiados. Esta ejecución es **walk-forward retrospectivo**, no una nueva prueba final intacta. Las reservas prospectivas preexistentes permanecen cerradas y su protección se aplica también a estos comandos. Para decidir una adopción será necesaria una nueva evaluación prospectiva registrada antes de observarla; no se crea evidencia final reutilizando estos folds.

## Ejecución y límites económicos

Cuenta USD, long/cash, una posición por activo, sin apalancamiento. Riesgo nominal 1% de equity por ancho ATR y notional máximo 100%, con redondeo de unidades hacia abajo. En estrategias de salida fija el ATR sólo determina tamaño: **no existe un stop garantizado de 1%**. En Triple Barrier sí se ejecuta el stop, sujeto a gaps y slippage. Cada activo se simula por separado; no se suma el capital como si hubiera diversificación gratuita entre siete cuentas.

Forex: spread bid/ask observado del proveedor identificado en el dataset, comisión supuesta 0,35 bps por lado y slippage supuesto 0,1 pip por lado. En FXCM el spread corresponde a precios indicativos Active Trader. Entrada si el spread actual está entre 0 y 5 pips y la sesión nominal está abierta. No se rellena una entrada rechazada con el spread o cierre futuro. TP limitado al precio de la orden, SL con gap al open real y SL primero si ambas barreras se tocan en una H1. USDJPY/USDCHF convierten ganancias comprando USD al ask y pérdidas al bid. Para un cruce intrahora sin timestamp exacto se utiliza el extremo menos favorable de esa H1 para la conversión.

En una coincidencia exacta entre vencimiento y apertura, el protocolo base da prioridad a la salida temporal a mercado. Por eso se evalúa también `expiry_policy="barrier_first"`: mantiene el límite de TP y el SL por gap en esa misma apertura. La auditoría vuelve a simular todas las políticas TB con las mismas predicciones para medir esta sensibilidad, sin reentrenar ni seleccionar una variante tras mirar OOS. Esta excepción de prioridad no debe confundirse con un TP ejecutado sin límite; su impacto se conserva en `diagnostics.json`.

Los swaps históricos por broker, dirección y feriado no están en estos datos. Se modela un **escenario de costo**, no una serie histórica inventada: 100 bps anual adversos sobre notional, rollover NY17 de lunes a viernes y miércoles triple; estrés a 300 bps y doble comisión/slippage. Los feriados pueden desplazar el rollover efectivo y ciertos pares pueden recibir carry positivo: esta aproximación no demuestra rentabilidad ejecutable. [OANDA explica el rollover y el triple cargo habitual del miércoles](https://www.oanda.com/us-en/trading/financing-fees/).

Cripto: se reutilizan los CSV verificados de Binance, 10 bps de comisión y 5 bps de slippage por lado. No se dispone de bid/ask histórico en esas velas; no se presenta su spread como medido. USDT se considera USD para esta comparación. Forex tiene marcas de equity H1 y cripto diarias; el máximo drawdown no captura el mismo detalle intradiario. No se iguala la volatilidad realizada a posteriori. Comparar retornos con cautela: cambian volatilidad, horarios, frecuencia, spread y calidad de ejecución.

## Comandos

Desde la raíz, usando `.venv\Scripts\python.exe -m trading_ml.cli` como prefijo:

```powershell
# Descarga inicial o incremental. Cambiar symbol para cada uno de los cinco pares.
.venv\Scripts\python.exe -m trading_ml.cli market-download --protocol config/multimarket_verified_history.toml --symbol EURUSD --interval 1d
.venv\Scripts\python.exe -m trading_ml.cli market-download --protocol config/multimarket_verified_history.toml --symbol USDJPY --interval 4h
# Configurar start/end (end exclusivo) y transporte en config/multimarket.toml.
# --refresh revisa también meses cerrados; nunca rellena precios faltantes.

# Primero el piloto completo. Su salida entrega el directorio que valida el pipeline.
.venv\Scripts\python.exe -m trading_ml.cli market-study --protocol config/multimarket_verified_history.toml --symbols EURUSD --intervals 1d 4h 1h

# Luego el resto, con los mismos períodos y protocolo.
.venv\Scripts\python.exe -m trading_ml.cli market-study --protocol config/multimarket_verified_history.toml --symbols USDJPY GBPUSD USDCHF AUDUSD BTCUSDT ETHUSDT --intervals 1d --pilot artifacts/multimarket/studies/ID_PILOTO

# Modelo independiente, sin evaluar un futuro ni sustituir producción.
.venv\Scripts\python.exe -m trading_ml.cli market-train --protocol config/multimarket_verified_history.toml --symbol EURUSD --interval 1d --cutoff 2024-01-01 --recipe price --target direction --horizon 1
# --target net_profit o triple_barrier; --horizon admite 1..10.

# DATA es el CSV versionado indicado por market-download; MODEL, el UBJ guardado.
.venv\Scripts\python.exe -m trading_ml.cli market-predict --model MODEL --data DATA --calibrated
.venv\Scripts\python.exe -m trading_ml.cli market-evaluate --model MODEL --data DATA --start INICIO_POSTERIOR_AL_FIT --end FIN_CUBIERTO
.venv\Scripts\python.exe -m trading_ml.cli market-backtest --model MODEL --data DATA --start INICIO_POSTERIOR_AL_FIT --end FIN_CUBIERTO --calibrated

# Reproduce pronósticos, contratos y resultados de trades guardados sin entrenar.
.venv\Scripts\python.exe scripts/record_multimarket_study.py --studies PILOTO COMPARACION --output DIRECTORIO_NUEVO
.venv\Scripts\python.exe -m pytest -q
```

Cada comando también acepta `--protocol` cuando corresponde. Las rutas de cripto conservadas son `artifacts/data/raw.csv` y `artifacts/research_data/ETHUSDT_20200101_20261001.csv`. Para un historial cripto distinto se conserva el descargador original; la comparación nueva exige cobertura diaria compatible.

`market-predict` recorta el historial a la referencia **antes** de calcular features y no construye targets. Devuelve probabilidad, target, calibración, fecha y antigüedad del último dato; no es una orden. Un historial hasta septiembre produce una predicción histórica, no una cotización actual. Rechaza referencias futuras, velas no cerradas y referencias anteriores al último target utilizado para fit/calibración. `market-evaluate`/`market-backtest` exigen un período posterior a ese límite y conservan predicciones, trades y equity en un directorio nuevo.

## Evidencia y resultados

Cada estudio guarda plan previo con protocolo, semilla, hashes del código y entorno; fuentes y versiones de datos; modelos UBJ y contratos de fit/calibración; pronósticos OOS; operaciones; equity; importancia; métricas y comparación CSV. Los planes y resultados se escriben sin sobrescribir. `record_multimarket_study.py` recompone los pronósticos desde modelos guardados (tolerancia 1e-7 por serialización de probabilidades float32) y los outcomes desde precios y costos, sin nuevos fits. También comprueba cronología, hashes y filas comunes de ablación.

La comprobación adicional `artifacts/multimarket/provider_research/eurusd_h1_vs_m1_20261015.json` compara bid/ask H1 con OHLC agregado desde M1 el 2/1/2020, 17/5/2023 y 15/9/2026: diferencia de precio cero en los seis archivos. Es contraste entre resoluciones del mismo proveedor, **no una verificación independiente del mercado**.

Los directorios de ejecución, la tabla real de siete activos y las conclusiones se registran al completar las verificaciones en el informe de resultados asociado. Ninguna mejora exploratoria se adopta automáticamente.
