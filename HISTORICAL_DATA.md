# Históricos ampliados — adquisición sin entrenamiento

La ampliación del 17 de octubre de 2026 agrega datos en `artifacts/historical/`.
Inventario activo en `artifacts/historical/latest.json` y
`artifacts/historical/validated_20261017.json`; totales en
[HISTORICAL_DATA_RESULTS.md](HISTORICAL_DATA_RESULTS.md).
Los modelos, checkpoints, resultados y configuraciones de experimentos anteriores
se conservan. Ninguno de los comandos de este documento ajusta modelos.

## Fuentes y contratos

| Mercado | Universo | Datos |
|---|---|---|
| Spot | BTC, ETH, SOL, BNB, XRP, ADA, DOGE, LINK, LTC, DOT, AVAX, BCH contra USDT | M15 oficial Binance; H1/H4/D1 agregados |
| Perpetuos | BTC, ETH, SOL, BNB, XRP, DOGE contra USDT | Precios negociados, no mark price; M15/H1/H4/D1 |
| Futuros cripto | Contratos trimestrales individuales BTCUSD y ETHUSD | COIN-M inversos; M15/H1/H4/D1; sin empalmes |
| Forex | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD, EURJPY, GBPJPY, EURGBP | M1 bid/ask FXCM desde 2012; M15/H1/H4/D1 completos |
| Índices | S&P 500, Nasdaq Composite, Dow Jones, Russell 2000, VIX, DAX, FTSE 100, Nikkei 225 | OHLC diario de Yahoo; contexto no negociable |
| Futuros de volatilidad | VX mensual de Cboe | Contratos individuales, OHLC negociado diario, settlement separado y open interest |

Se solicitan meses cerrados hasta `2026-10-01` exclusivo. La cobertura real puede
empezar después de la fecha solicitada o terminar antes: el inventario informa
`first_open` y `last_close` por instrumento y temporalidad. El intervalo de cada
contrato de futuros es su propia vida, no toda la historia del subyacente.

Fuentes: [Binance Public Data](https://github.com/binance/binance-public-data),
[FXCM Candle Data](https://github.com/fxcm/MarketData/tree/master/CandleData),
[Cboe Historical Data](https://www.cboe.com/markets/us/futures/market-statistics/historical-data/futures),
[Yahoo Finance](https://finance.yahoo.com/). Cboe publica su listado de contratos
en `https://www-api.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/`.
Uso de investigación personal sujeto a los términos de cada fuente; los datos
descargados están excluidos de Git y no se redistribuyen.

## Descargar, reanudar y verificar

```powershell
.\.venv\Scripts\python.exe -m trading_ml.data.history_cli acquire --config config/historical.toml
.\.venv\Scripts\python.exe -m trading_ml.data.history_cboe
.\.venv\Scripts\python.exe -m trading_ml.data.history_cli acquire --config config/historical_indices.toml --only indices
.\.venv\Scripts\python.exe -m trading_ml.data.history_fx_daily
.\.venv\Scripts\python.exe -m trading_ml.data.history_repair --receipt <comprobante_SOL_perpetuo.json>
.\.venv\Scripts\python.exe scripts/bound_history_futures.py --inventory <inventario_consolidado.json>
.\.venv\Scripts\python.exe -m trading_ml.data.history_cli verify --inventory <inventario.json>
```

La adquisición usa hasta cuatro conexiones por proveedor. FXCM tiene un límite
global de solicitudes; se conservan originales y comprobantes por URL/hash.
Repetir el comando reutiliza la caché verificada y los instrumentos completados.
`--only crypto`, `--only forex` o `--only indices` permite retomar un proveedor.
Los HTTP404 se registran como archivos ausentes; otros errores no se convierten
en cierres de mercado. Los fallos quedan identificados en los inventarios.

Cada CSV tiene un manifiesto JSON con checksum, procedencia, temporalidad,
disponibilidad y calidad. Binance se contrasta además con su checksum publicado;
en FXCM/Cboe/Yahoo el hash local detecta corrupción posterior, sin atribuirles un
checksum oficial inexistente.

## Usar los datos en los pipelines

`load_history(path, root=Path("artifacts"), start=..., end=..., horizon=1)` es el
adaptador general: verifica integridad y recorta **antes** de features o etiquetas.
Exige fin explícito, aplica las reservas prospectivas anteriores y bloquea para
desarrollo fechas posteriores a `2025-07-01`. Lo descargado después de ese límite
permanece en cuarentena para un protocolo futuro; no es evidencia independiente
nueva ni material habilitado para seleccionar modelos.

Para spot continuo, exportar un tramo completo a una ruta nueva compatible con
`prepare()` y DMN, sin cambiar sus entradas históricas:

```powershell
.\.venv\Scripts\python.exe -m trading_ml.data.history_cli export --source <spot.csv> --output artifacts/historical/adapters/spot_raw.csv --start 2021-01-01 --end 2025-07-01
```

El export rechaza huecos y otros mercados; elegir otro período completo o un
pipeline que maneje discontinuidades, nunca interpolar. Para Forex H1 bid/ask:

```powershell
.\.venv\Scripts\python.exe -m trading_ml.data.history_cli export --forex --source <forex_H1.csv> --start 2012-01-01 --end 2025-07-01
```

Ese adaptador conserva spreads/cotizaciones cruzadas y permite utilizar
`multimarket.features.features` y `aggregate_forex`. El D1 de esta última función
sigue cerrando a las 17 de Nueva York; los snapshots nuevos D1 UTC se identifican
por separado. EURJPY/GBPJPY/EURGBP se agregan al registro de instrumentos, sin
cambiar el universo de ningún protocolo anterior.

## Calidad y límites

- Fechas UTC únicas, ordenadas y alineadas; precios finitos/positivos, OHLC coherente,
  volumen negociado no negativo cuando la fuente lo suministra, y velas cerradas.
- Las velas Binance parciales/desalineadas de mantenimiento se excluyen con sus
  fechas y motivos registrados. Índices y Cboe excluyen OHLC nulos/no negociados;
  sus originales se conservan. No se reemplaza OHLC por settlement.
- La extensión completa de índices consulta la historia anterior a 1990, pero
  excluye OHLC diarios con rango cero: varias décadas tempranas repiten el cierre
  en los cuatro campos y no permiten verificar máximos/mínimos negociados.
  `historical_indices.toml` fija esta política; el inventario activo utiliza las
  versiones depuradas. El loader también rechaza esos rangos no verificables.
- Agregaciones exclusivamente a partir de todas las subvelas observadas de la
  ventana. Sin upsampling, interpolación ni relleno de feriados. La disponibilidad
  de una vela es su cierre; la de índices/Cboe es el siguiente día UTC conservador.
- `missing_grid_slots` incluye fines de semana y feriados en mercados con sesiones;
  no equivale a interrupciones del proveedor. Los huecos siguen visibles.
- La selección de cripto tiene sesgo de supervivencia. El volumen es específico del
  exchange; Forex e índices no reciben volumen inventado.
- Los perpetuos necesitan funding/costos y los futuros fechados necesitan margen,
  multiplicador, calendario y una política causal de roll antes de experimentar.
  No se los envía al simulador spot. No se adquirieron contratos licenciados de
  ES/NQ/GC/CL ni se los sustituyó por tickers continuos o CFD.
- En futuros COIN-M se excluye del uso la cola posterior a la liquidación
  programada a las 08:00 UTC. Algunos archivos incluyen precios congelados y
  otros registran volumen en esa cola: queda en los originales, sin asumir que
  son velas ordinarias negociables ni inventar una hora real de entrega diferida.
- Índices y VIX futuros son D1; no se fabrican M15/H1/H4 a partir de datos diarios.
  Las filas agregadas de distintas temporalidades reutilizan observaciones y no
  deben contarse como muestras independientes ni repartirse aleatoriamente.
- Forex `1d_ny` contiene D1 nativas NY17 con el cambio horario real, distintas de
  `1d` UTC. El archivo anual nativo alcanza 2021 y las ventanas UTC completas
  resultan mucho más escasas. Se preservan también las D1 NY17 de los experimentos
  anteriores. Las cotizaciones nativas fuera de la sesión semanal se excluyen.
- Conservar splits cronológicos, purga por `target_time` y ajuste de escaladores
  sólo en entrenamiento. Cruces entre mercados/temporalidades deben usar
  `available_at` y uniones hacia atrás, nunca la fecha de apertura de la vela.

Los totales finales y las exclusiones quedan en el informe generado de la
adquisición. No se ejecutaron entrenamientos, promociones ni evaluaciones nuevas.
