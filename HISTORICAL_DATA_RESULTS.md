# Ampliación histórica — 17 de octubre de 2026

**11.749.803 registros validados en 491 datasets**, con **8.983.894 observaciones
base** sin contar nuevamente las temporalidades derivadas. **Ningún modelo entrenado.**

| Mercado | Activos o contratos | Registros | Cobertura conjunta efectiva |
|---|---:|---:|---|
| Forex | 10 pares | 4.395.048 | 2012–septiembre de 2026 |
| Cripto spot | 12 pares | 4.301.790 | agosto de 2017–septiembre de 2026 |
| Perpetuos cripto | 6 | 1.814.908 | 2020–septiembre de 2026 |
| Futuros trimestrales BTC/ETH | 50 contratos individuales | 1.126.654 | junio de 2020–septiembre de 2026 |
| Índices | 8 | 82.707 | 1962–septiembre de 2026, según activo |
| Futuros VIX | 161 contratos mensuales | 28.696 | mayo de 2013–septiembre de 2026 |

Se agregaron EURJPY, GBPJPY y EURGBP; ADA, DOGE, LINK, LTC, DOT, AVAX y BCH contra
USDT; contratos trimestrales BTCUSD/ETHUSD y VX; S&P 500, Nasdaq Composite,
Dow Jones, Russell 2000, VIX, DAX, FTSE 100 y Nikkei 225. Se ampliaron también
los activos anteriores. M15/H1/H4/D1 donde existen datos reales; índices y VX sólo D1.
Forex incorpora además 25.928 D1 nativas NY17 de 2012–2021, identificadas aparte.

Se verificaron 9.741 originales y 234 agregaciones entre temporalidades. Se
recuperaron 480 M15 de SOL desde archivos diarios oficiales. Se excluyeron OHLC
inválidos/no verificables, sesiones incompletas y 27.707 M15 posteriores al horario
programado de liquidación de futuros, conservando los originales. **80 pruebas
aprobadas** y comprobación real de los adaptadores spot/Forex sin entrenar.

Los modelos, checkpoints y resultados anteriores permanecen intactos. El loader
aplica las reservas previas y limita el desarrollo a 2025-07-01; **1.552.659 filas
recientes permanecen en cuarentena**. Los splits deben seguir siendo cronológicos,
con purga de etiquetas y disponibilidad por cierre.

Límites: Forex indicativo con huecos, sin volumen global/swaps; D1 UTC estricto
escaso y D1 NY17 nativo publicado sólo hasta 2021. Los índices no son negociables;
perpetuos/futuros necesitan funding, costos y un protocolo de margen/roll antes
de experimentar. No se adquirieron ES/NQ/GC/CL licenciados.

Inventario activo: `artifacts/historical/validated_20261017.json`.
Verificación: `artifacts/historical/validated_verification_20261017.json`.
Cobertura individual, hashes y procedencia: manifiestos junto a cada CSV.
Uso y reproducción: [HISTORICAL_DATA.md](HISTORICAL_DATA.md).
