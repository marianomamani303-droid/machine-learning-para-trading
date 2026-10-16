# Resultados DMN — 2026-10-16

**No se encontró evidencia robusta para sustituir los controles tradicionales ni promover una red.** Se completaron 120 ajustes reproducibles, con resultados positivos en long/cash y negativos en long/short sintético. Son comparaciones de desarrollo sobre datos históricos ya observados, no una confirmación independiente ni una réplica exacta del paper.

## Experimentos y procedencia

| Dominio | Protocolo | Corrida | Ajustes |
| --- | --- | --- | ---: |
| Spot long/cash | `config/dmn.toml` | `20261016T204450_9f760c70` | 60 |
| Exposición sintética long/short | `config/dmn_long_short.toml` | `20261016T204723_6818bead` | 60 |

Datos oficiales Binance BTCUSDT y ETHUSDT diarios, 2020-01-01 a 2026-10-01 exclusivo. Fuentes originales con checksums conservadas; snapshots y procedencia en cada corrida. Retornos exteriores desde 2022-07-01, cinco folds con 183/364/365/364/272 observaciones negociadas. La curva conjunta tiene 1552 fechas, incluidos cuatro días de frontera en cash. La última apertura de salida es 2026-09-30; no se utiliza una salida en el límite exclusivo de octubre.

Dos semillas fijas (42/1729); lineal, MLP, LSTM con Sharpe bruto, LSTM con Sharpe neto, LSTM con MSE y control XGBoost binario. LSTM de 20 unidades, secuencia 63, Adam 0,001, dropout 0,2, máximo 30 épocas y paciencia 7. Validación interior de 180 días, purga y embargo; cada checkpoint utiliza únicamente pasado. Los resultados de receta promedian posiciones entre semillas antes del escalado de riesgo. Todas las semillas y todos los modelos rechazados se conservan.

Costos base: comisión 10 bps + slippage 5 bps + medio spread 1 bp por lado. En long/short se suma préstamo supuesto de 300 bps/año sobre el notional negativo. La sensibilidad duplica costos y financiación. El riesgo individual tiene objetivo ex ante 15% anual y exposición bruta máxima 1, dividido por número de activos; no hay igualación ex post de volatilidad. Contratos, fórmulas y límites en [DMN_GUIDE.md](DMN_GUIDE.md).

## Long/cash

| Estrategia | Retorno neto acumulado | Sharpe neto | Vol. anual | Drawdown | Sharpe con costos dobles |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long con volatilidad escalada | 59,49% | 0,812 | 14,89% | −24,91% | 0,777 |
| TSMOM anual | 30,48% | 0,613 | 11,23% | −14,13% | 0,572 |
| MACD | 19,96% | 0,668 | 6,75% | −9,25% | 0,621 |
| Lineal Sharpe neto | 32,98% | 0,793 | 8,96% | −14,77% | 0,740 |
| LSTM Sharpe bruto | 20,63% | 0,693 | 6,69% | −11,36% | 0,633 |
| **LSTM Sharpe neto** | **22,06%** | **0,717** | **6,87%** | **−11,63%** | **0,662** |
| XGBoost binario | 12,66% | 0,285 | 12,62% | −24,97% | −0,034 |

El control long con volatilidad escalada supera a la LSTM en Sharpe y rentabilidad, con más riesgo realizado. El control lineal supera a la LSTM en Sharpe; no se observa una prima clara por la memoria recurrente. Buy/hold a pesos iguales y rebalanceo diario obtuvo 240,44%, Sharpe 0,800 y drawdown −59,95%; tiene mucha más exposición y no se compara sólo por retorno. MLP Sharpe neto y LSTM MSE obtuvieron respectivamente 18,74%/0,642 y 17,51%/0,450. Cash rindió cero.

La LSTM neta tiene Sharpes por fold de **0,038 / 2,028 / 1,500 / −0,083 / −0,016**, y sólo dos folds de cinco con retorno neto positivo. Sus semillas tienen Sharpes globales 0,707 y 0,724; no se eligió la mejor. Los sleeves por separado, al ocupar capital completo, retornaron 23,07% en BTC y 20,85% en ETH.

Su exposición bruta media fue 0,129, frente a 0,291 del control long escalado. La correlación diaria de retornos con ese control fue 0,979. Esto es compatible con una participación menor en el mismo movimiento de mercado; por sí solo no demuestra una señal incremental ni identifica causalmente alfa/beta.

La variante neta redujo el turnover diario medio de la LSTM de aproximadamente 0,0069 a 0,0065 y mejoró el Sharpe agregado de 0,693 a 0,717. Es una diferencia pequeña en este desarrollo, no una confirmación de la regularización. Ver los valores exactos en `summary.json`.

## Long/short sintético

| Estrategia | Retorno neto acumulado | Sharpe neto | Drawdown | Sharpe con costos dobles |
| --- | ---: | ---: | ---: | ---: |
| TSMOM anual | 0,01% | 0,069 | −20,11% | −0,016 |
| MACD | 7,09% | 0,227 | −12,19% | 0,133 |
| Lineal Sharpe neto | 0,71% | 0,074 | −6,51% | −0,242 |
| MLP Sharpe neto | −17,62% | −1,196 | −18,59% | −1,444 |
| LSTM Sharpe bruto | −7,15% | −0,917 | −7,71% | −1,131 |
| **LSTM Sharpe neto** | **−10,46%** | **−0,977** | **−11,13%** | **−1,116** |
| LSTM MSE | −16,75% | −0,285 | −29,01% | −0,420 |
| XGBoost binario | −22,92% | −0,392 | −42,31% | −0,981 |

Los controles cash y long son iguales a los del estudio anterior. La LSTM neta no tuvo ningún fold positivo. Los Sharpes por semilla fueron −1,014 y −0,912. Los sleeves BTC/ETH perdieron 11,75% y 9,19%. Añadir la penalización de costos no solucionó el deterioro fuera de muestra; la variante neta fue peor que la bruta pese a menor turnover. La volatilidad realizada de la LSTM fue sólo 2,62% anual, por lo que un drawdown bajo no significa que haya alcanzado el objetivo de riesgo ni una ventaja económica.

## Incertidumbre

Bootstrap de 1000 remuestras pareadas por bloques de fechas dentro de cada fold, manteniendo ambos activos juntos. Intervalos marginales del 95% para diferencia de Sharpe **LSTM neta menos TSMOM**:

| Dominio | Bloques 7 fechas | Bloques 30 fechas |
| --- | --- | --- |
| Long/cash | [−0,609; 0,778] | [−0,755; 0,579] |
| Long/short sintético | [−2,202; 0,395] | [−2,468; 0,333] |

No se establece superioridad en ninguna de las comparaciones mostradas. Los intervalos no corrigen multiplicidad, selección ni cambios de régimen futuros. En long/cash, el intervalo de diferencia de retorno medio anualizado para bloques 30 es [−11,18%; 4,36%]. Un Sharpe puntual mayor que TSMOM coexiste con mucha incertidumbre y menor retorno acumulado.

## Diagnóstico y decisiones

1. **Conservar controles y resultados negativos.** No se reemplazó ni activó ningún modelo histórico. El sistema DMN es infraestructura de investigación, no un candidato demostrado para operar.
2. **La poca diversificación limita la comparación con el trabajo original.** Dos criptoactivos correlacionados y unos pocos ciclos no sustituyen décadas de múltiples futuros. Los controles long positivos tampoco demuestran momentum independiente de exposición de mercado.
3. **La convergencia no está demostrada en todos los ajustes.** En long/cash, tres de diez checkpoints LSTM netos seleccionaron la última época disponible; otros tres seleccionaron la primera. El presupuesto de 30 épocas fue fijado antes de entrenar, menor que el del paper. Estos resultados describen este protocolo, no descartan toda arquitectura LSTM ni DMN.
4. **No ampliar búsqueda sobre el mismo test observado.** La siguiente hipótesis justificada es si existe capacidad incremental en un universo diversificado de futuros, con procedencia, calendario y roll verificables. Primero debe resolverse el contrato de datos; los precios continuos sin auditoría de ajustes no son una sustitución válida. Alternativamente, un estudio Forex debe reutilizar bid/ask y calendario real del repositorio, con un protocolo independiente.
5. **Confirmación prospectiva pendiente.** Los períodos existentes reservados no se tocaron y se extienden hasta 2027-10-13. Una nueva confirmación debe congelar hipótesis/modelos/criterios y tener una ventana disjunta; no transformar estos resultados ya observados en una prueba ciega.

## Verificación y reproducción

La suite completa pasó **264 pruebas**. Ruff, formato y `pip check` correctos. Cada estudio conserva **511 archivos verificados**, 60 ajustes/60 intentos completados y 39 archivos históricos protegidos sin alteraciones. El verificador reconstruye **345 carteras por estudio** (23 variantes incluyendo semillas × 5 folds × 3 escenarios de costos) a partir de posiciones/retornos, sin volver a entrenar.

```powershell
.\.venv\Scripts\python.exe -m trading_ml.dmn verify --study artifacts/dmn_studies/20261016T204450_9f760c70
.\.venv\Scripts\python.exe -m trading_ml.dmn verify --study artifacts/dmn_studies/20261016T204723_6818bead
.\.venv\Scripts\python.exe -m trading_ml.dmn study --protocol config/dmn.toml --resume artifacts/dmn_studies/20261016T204450_9f760c70
.\.venv\Scripts\python.exe -m trading_ml.dmn study --protocol config/dmn_long_short.toml --resume artifacts/dmn_studies/20261016T204723_6818bead
```

Las dos últimas órdenes recuperan resultados si el código y entorno coinciden, sin consumir nuevos ajustes. Para repetir desde cero, omitir `--resume`; se generará otro ID. Los gráficos ya están en `artifacts/dmn_analyses/<run_id>/equity_drawdown.{png,svg}` y `fold_sharpes.{png,svg}`. Cada carpeta tiene manifiesto; los artefactos grandes permanecen excluidos de Git y deben respaldarse junto con el código y el lockfile. Diseño en [DMN_RESEARCH.md](DMN_RESEARCH.md), uso y ecuaciones en [DMN_GUIDE.md](DMN_GUIDE.md), continuidad en [CODEX_PROGRESS.md](CODEX_PROGRESS.md).
