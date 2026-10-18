# Auditoría del fracaso predictivo

**Fecha: 16 de octubre de 2026. Alcance: código, datos locales, modelos y experimentos existentes.** Esta auditoría no entrena modelos nuevos, no modifica contratos ni resultados anteriores y no consume reservas prospectivas. El protocolo de diagnóstico se fijó en [config/predictive_failure_audit.toml](config/predictive_failure_audit.toml), con presupuesto de **cero entrenamientos**.

## 1. Resumen ejecutivo

**No existe evidencia suficiente de una ventaja económica robusta, replicada e independiente con los sistemas evaluados. La evidencia tampoco permite afirmar que el mercado sea absolutamente impredecible o que los datos carezcan de toda información.** Son afirmaciones distintas.

El diagnóstico más probable combina **poca información incremental sobre dirección en las representaciones utilizadas, sobreajuste a muestras pequeñas y cambiantes, probabilidades demasiado dispersas para la señal que contienen y costos incompatibles con la rotación**. A esto se suma un problema de investigación: el historial se reutilizó en muchas rondas y ya no constituye un test final independiente. No aparece una explicación única del tipo «XGBoost está mal implementado».

Hallazgos principales, con evidencia a favor y límites:

| Hallazgo | Evidencia | Solidez y límite |
|---|---|---|
| Sobreajuste de varios estimadores | AUC de entrenamiento aproximadamente 0,74–0,84 en árboles frente a 0,51–0,52 OOS; brechas de loss positivas | **Alta** en las configuraciones medidas; no identifica por sí sola el mejor remedio |
| Los costos destruyen parte sustancial del resultado | BTC original: −10,01% bruto → −83,17% neto sobre exactamente las mismas 524 operaciones | **Alta**, identidad verificada del ledger; no son porcentajes que puedan restarse como contribuciones aditivas |
| También existe pérdida antes de costos | ETH original: −64,23% bruto; BTC estructura: −22,99% bruto | **Alta**; bajar comisiones no arregla todas las variantes |
| La dirección no es lo mismo que la rentabilidad | En h1, 6,92% de etiquetas BTC y 5,49% ETH cambian entre dirección y beneficio neto | **Alta** en el historial; la diferencia de signo close/open es 0% aquí: el desajuste observado viene principalmente del umbral de costos |
| Mayor confianza no implica mayor acierto | BTC original, confianza media 0,670: 46,27% de aciertos, n=67 | **Alta descriptiva**; el bin pequeño no justifica invertir señales ni crear filtros |
| La muestra efectiva depende de la pregunta | BTC/ETH tienen 2.465 velas cada uno, correlación de retornos 0,835 y ventanas superpuestas | **Alta** para dependencia; **no existe un único número demostrado de observaciones independientes** |
| No todo es ausencia de señal | EUR/USD H1 compacto: AUC 0,532; menor loss que prevalencia; AUC>0,5 en 6/6 folds | **Media** como indicio histórico: una temporalidad, un par, múltiples pruebas; pierde 17,75% neto |
| La magnitud es una pregunta diferente | Proxy causal de magnitud h1 reduce MSE agregado 5,74% BTC / 5,91% ETH; media móvil de retorno con signo empeora el benchmark cero | **Media-baja**: sólo mejora magnitud en 3/6 folds; no hay confirmación estadística ni alpha demostrado |
| Forex tiene problemas adicionales de cobertura | EUR/USD FXCM: 619 horas nominales ausentes, 474 cruces bid/ask H1 y horizontes que atraviesan huecos largos | **Alta** para los registros; **no explica** el fracaso en cripto, cuya continuidad se verifica |

**La prioridad científica no es añadir una red neuronal. Es separar qué información se puede pronosticar, con qué incertidumbre, qué parte llega a la operación y qué evidencia sobreviviría fuera del historial ya observado.**

## 2. Inventario del sistema y de la evidencia

### 2.1 Arquitectura y contratos

| Área | Código principal | Situación auditada |
|---|---|---|
| Datos cripto | `data/binance.py`, `preprocessing/quality.py` | ZIP mensuales oficiales, checksum, soporte de unidades temporales; validación OHLCV y continuidad |
| Datos FX | `multimarket/data.py`, `fxcm.py`, `markets.py` | FXCM y Dukascopy separados; bid/ask; calendario NY con DST; H4 UTC y D1 NY17 |
| Original y enhanced | `features/technical.py` | 14/22 features; retornos, indicadores, volatilidad y forma de vela |
| SMC | `smc/detectors.py`, `rules.py` | 32 variables; pivots confirmados, BOS, CHoCH, OB, FVG, sweeps y touches |
| Familias adicionales | `momentum/features.py`, `mean_reversion/features.py`, `advanced_research/features.py` | Momentum, eficiencia, reversión, autocorrelación, flujo taker, rupturas y riesgo |
| Features investigadas | `feature_research/features.py`, `frontier_research/features.py` | Ocho familias; semanas cerradas; alineación exacta BTC/ETH y siete variables cruzadas frontier |
| Estructura y regímenes | `structure_regime/` | Extensiones de retest/impulso; estados causales, KMeans sólo train, especialistas con fallback |
| Modelos | `models/`, `advanced_research/models.py`, `frontier_research/models.py` | Logística, XGBoost, Random Forest, LightGBM, CatBoost, regresores, calibración, ensamble y stacking |
| Etiquetado | `preprocessing/pipeline.py`, `advanced_research/targets.py`, `frontier_research/targets.py`, `label_research/events.py`, `multimarket/study.py` | Dirección, retorno neto, barreras y meta-labeling con definiciones diferentes y explícitas |
| Validación y ejecución | `training/`, `evaluation/`, `backtesting/`, `risk_overlay/`, `label_research/execution.py`, `multimarket/execution.py` | Separación temporal, purga, embargo, costos, abstenciones, modelos registrados y reservas |

Los contratos no son intercambiables. El original tiene 14 columnas; original+SMC tiene 46; enhanced+SMC, 54; estructura extendida, 57. Momentum declara 19 features y reversión 15; el conjunto local frontier tiene 28 y el cruzado 35. Forex usa una base de 13 variables de precio y su compacto añade familias temporales y spread; **no inventa volumen**. La auditoría cuantitativa nueva revisa una **unión de 105 columnas**, no un modelo único entrenado con 105 variables. Las otras rutas se revisaron en código, contratos y pruebas existentes.

### 2.2 Estudios que no deben volver a contarse como evidencia nueva

- [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md): 635 ajustes, ablaciones, selección temporal interna e importancia por bloques; ninguna receta aceptada.
- [TRIPLE_BARRIER_RESEARCH.md](TRIPLE_BARRIER_RESEARCH.md): 486 ajustes; objetivos netos, barreras y meta-labeling; ninguna mejora consistente aceptada.
- [FRONTIER_RESEARCH.md](FRONTIER_RESEARCH.md): 453 ajustes contabilizados, múltiples algoritmos y ensambles; sin aceptación independiente.
- [STRUCTURE_REGIME_RESEARCH.md](STRUCTURE_REGIME_RESEARCH.md): 360 ajustes, dos activos y dos semillas; sin variante aceptada ni combinación ejecutada.
- [ADVANCED_RESEARCH.md](ADVANCED_RESEARCH.md), [MOMENTUM_RESEARCH.md](MOMENTUM_RESEARCH.md), [MEAN_REVERSION_RESEARCH.md](MEAN_REVERSION_RESEARCH.md), [RISK_RESEARCH.md](RISK_RESEARCH.md): antecedentes, incluidos resultados positivos aislados y reducción de exposición.
- [MULTIMARKET_RESEARCH.md](MULTIMARKET_RESEARCH.md): piloto EUR/USD y comparación diaria de siete activos; los pilotos incompletos y proveedores distintos no se mezclan para elegir ganadores.

Sólo las primeras cuatro rondas suman **1.934 ajustes contabilizados**, incluyendo calibraciones y modelos de estado cuando corresponde. No equivalen a 1.934 hipótesis independientes ni a 1.934 tests ciegos. Los estudios restantes aumentan la exposición al mismo historial. Se conservan también corridas técnicas incompletas, sin tratarlas como confirmaciones científicas.

## 3. Diagnóstico de datos, temporalidad e independencia

### 3.1 Procedencia e integridad

Se verificaron **23 snapshots/datasets**, sus hashes y estadísticas de precios. BTC y ETH tienen 2.465 velas diarias reales desde enero de 2020 hasta septiembre de 2026: sin duplicados, celdas OHLC ausentes, precios no positivos, OHLC incoherentes ni discontinuidades diarias en las fuentes auditadas. Los movimientos extremos no se borraron por ser incómodos.

La recuperación de flujo en `advanced_research/data.py` exige que los ZIP y sus CHECKSUM coincidan y reproduzcan los precios y volúmenes originales. Los hashes acreditan integridad local y reproducibilidad; no demuestran que una muestra de un proveedor represente todo el mercado. Las rutas cripto se apoyan en HTTPS y archivos oficiales; el piloto legado Dukascopy por HTTP tiene una limitación adicional de autenticidad en tránsito, ya documentada.

En FX hay cinco pares: EUR/USD, USD/JPY, GBP/USD, USD/CHF y AUD/USD. FXCM proporciona cotizaciones indicativas bid/ask y no volumen negociado global. Dukascopy contiene otro tipo de volumen, que no se trata como volumen de bolsa. Los snapshots duplicados de un mismo historial, sus agregaciones y sus versiones no aumentan el número de experiencias de mercado.

| Fuente comparable | Filas | Volatilidad de retorno log por vela | ACF(1) retorno | ACF(1) magnitud |
|---|---:|---:|---:|---:|
| BTC D1 Binance | 2.465 | 3,21% | −0,072 | 0,162 |
| ETH D1 Binance | 2.465 | 4,29% | −0,071 | 0,180 |
| EUR/USD D1 FXCM | 1.390 | 0,48% | 0,019 | 0,142 |
| USD/JPY D1 FXCM | 1.390 | 0,62% | −0,033 | 0,196 |
| GBP/USD D1 FXCM | 1.390 | 0,58% | 0,062 | 0,263 |
| USD/CHF D1 FXCM | 1.381 | 0,51% | 0,030 | 0,107 |
| AUD/USD D1 FXCM | 1.390 | 0,70% | 0,018 | 0,074 |

FX en esta tabla termina en junio de 2025; cripto, en septiembre de 2026. Es una descripción de cada fuente, **no un contraste entre mercados con períodos idénticos**. Las comparaciones multimercado de modelos sí usan sus calendarios comunes documentados.

EUR/USD FXCM 2020–junio de 2025 tiene 33.683 horas observadas, 34.249 horas nominales esperadas y **619 ausencias nominales**; hay observaciones fuera del calendario nominal, por lo que no basta restar los dos totales. Parte de las ausencias coincide con cierres festivos: el calendario es aproximado, no una certificación de todas las sesiones del proveedor. También hay bloques largos que merecen investigación. Se registran **474 cruces bid/ask de apertura y 474 de cierre en H1**; tras agregar D1 queda uno de cada tipo. No deben trasladarse los conteos H1 directamente a D1.

El protocolo completo FXCM hasta septiembre de 2026 había sido rechazado por cobertura: no se reemplazó por datos sintéticos. El histórico comparable más corto conserva huecos. EUR/USD Dukascopy tiene 42.089 H1 y nueve horas nominales ausentes en su rango más largo, pero es otra fuente y no valida automáticamente cada precio FXCM. Los cotejos M1/H1 del mismo proveedor comprueban decodificación/agregación, no independencia de fuente.

### 3.2 H1, H4 y D1 no son tres muestras independientes

H4 y D1 de Forex se construyen desde H1: comparten shocks y precios. Se excluyen 594 bloques H4 incompletos y 38 bloques D1 incompletos en el snapshot EUR/USD FXCM comparable; el padre H1 se conserva para ejecutar. H4 usa anclas UTC, D1 cierra a NY17 y cripto D1 a UTC00. Unir por fecha sin respetar **reference_time** mezclaría información disponible en momentos distintos.

Más filas H1 no significan más ciclos independientes. La configuración también cambia el significado de las ventanas: veinte barras son veinte horas en H1, ochenta horas en H4 y veinte sesiones en D1. No es una ablación pura de frecuencia manteniendo constante el horizonte físico y el contenido informativo.

Las etiquetas avanzan por filas o por horas cotizadas observadas, no siempre por horas civiles. En las predicciones históricas EUR/USD:

| Temporalidad y etiqueta | Mediana civil | Máximo civil observado |
|---|---:|---:|
| H1, dirección h1 | 1 hora | 367 horas |
| H4, dirección h1 | 4 horas | 392 horas |
| D1, dirección h1 | 24 horas | 408 horas |
| D1, dirección h3 | 120 horas | 744 horas |

Los fines de semana explican parte de la variación; los extremos reflejan además huecos y barras excluidas. El motor registra estas duraciones y no inventa fills dentro del hueco. **El cálculo puede ser causal y, aun así, responder una pregunta temporal distinta de la que sugiere “próxima vela”.** No se demostró que esta heterogeneidad cause toda la pérdida; sí se demostró que existe.

### 3.3 Distribución, ruido e información independiente

BTC/ETH tienen excesos de curtosis de **27,39/18,55**, lejos de una distribución normal. La volatilidad y las colas importan; accuracy pondera igual un error de 0,1% y otro de 10%. La ACF negativa de retornos no es cero: es incorrecto declarar aleatoriedad perfecta. Pero autocorrelación histórica pequeña no implica que un clasificador pueda explotarla establemente después de costos.

La correlación BTC/ETH es **0,835 en retornos y 0,757 en magnitudes**, sobre referencias exactamente alineadas. No hay 4.930 observaciones independientes por sumar ambos activos. Los horizontes 3/5/10 comparten retornos por construcción; incluso sin autocorrelación económica, sus etiquetas se solapan. Un límite descriptivo de submuestra sin solapamiento es aproximadamente N/h, no una estimación definitiva de independencia.

Como referencia, un estimador HAC Bartlett con treinta retardos da tamaño efectivo aproximado para **la media de retornos absolutos** de 648 BTC y 605 ETH, frente a 2.464 retornos. Para la media de retornos con signo da alrededor de 2.298/2.307. Estos números no miden el tamaño efectivo para AUC, trades, parámetros, regímenes o rarezas SMC; dependen de supuestos de estacionariedad y del ancho de banda. Los estadísticos Ljung–Box almacenados son descriptivos, sin convertir sus p-values en una familia de descubrimientos.

La evaluación diaria habitual tiene 1.728 filas por activo. Bajo una referencia binomial **iid idealizada**, la semiamplitud de un intervalo 95% cerca de 50% es 2,36 puntos porcentuales; una prueba con 80% de potencia detectaría aproximadamente 3,37 puntos, no necesariamente una mejora de un punto. Para un punto harían falta unas 19.623 observaciones iid en ese cálculo simplificado. Dependencia, selección y múltiples comparaciones pueden empeorar la situación; las pruebas pareadas pueden comportarse de otra manera. No es un requisito universal de “19.623 velas”.

**Conclusión:** hay datos suficientes para reproducir procesos, detectar problemas grandes y explorar señales intradía débiles. No hay base para tratar unos pocos ciclos, escasas operaciones y miles de reutilizaciones como evidencia contundente de una pequeña ventaja generalizable.

## 4. Diagnóstico de features y causalidad

### 4.1 Disponibilidad al predecir

La ruta original usa rolling/EMA no centrados y listas explícitas de features. SMC confirma el pivot j en j+right; OB nace al confirmar la ruptura y no en su vela fuente. Los touches no aparecen en el nacimiento. Las nuevas extensiones consumen esos eventos existentes; no redetectan pivots con otra convención. Se distingue el estado de tendencia persistente de una predicción del siguiente retorno.

El régimen direccional usa una ventana histórica y bandas fijas. Los cuantiles de volatilidad usan datos anteriores; KMeans y su scaler se ajustan sólo en train. Las semanas se publican después de cerrarse. Las variables cruzadas usan referencias exactas o información pasada, sin completar un precio de otro activo con datos futuros. Los normalizadores entrenables de las logísticas se ajustan con train.

Se reconstruyeron **105 columnas desde prefijos reales** de 700 y 1.700 filas en ambos activos y se comprobaron extensiones por prefijo: ocho verificaciones, todas iguales al histórico correspondiente. La suite también prueba mutación futura de OHLCV, purga, inferencia, zonas y límites temporales. No se encontró leakage de features en estas rutas comprobadas. **Esto no es una prueba formal de ausencia de cualquier fuga en toda rama posible del repositorio.**

La independencia del diseño es otra cuestión: reutilizar en folds de 2022 una receta cuyos hiperparámetros se eligieron históricamente con 2024–2025 no introduce un futuro valor de precio como feature, pero impide presentar ese backtest como una simulación de investigación que realmente habría ocurrido en 2022. Los informes recientes lo reconocen como exploración retrospectiva.

### 4.2 Redundancia y variables sin variación

- `macd = macd_signal + macd_hist`: identidad verificada con error máximo del orden de 10⁻¹⁷. Son tres representaciones de dos cantidades.
- `body_ratio` y `return_1`: Spearman 0,999992 BTC y 0,999996 ETH en el último train. Las aperturas cripto casi coinciden con el cierre anterior; en estos datos son casi duplicados.
- RSI, distancias SMA, retornos de ventanas cercanas y algunas representaciones de volatilidad se correlacionan. No son fuentes independientes de información por recibir nombres distintos.
- En D1 Forex, los tres flags de sesión Londres/Nueva York/Tokio son constantes antes de 2022. En D1 cripto también lo son seno/coseno de hora. No aportan selección temporal dentro de esa frecuencia. En H1/H4 sí varían.
- Readiness y disponibilidad de niveles pueden hacerse constantes tras el calentamiento. Se conserva su utilidad contractual, sin confundirla con capacidad predictiva.

Eliminar correlaciones no garantiza una mejora: la ronda anterior redujo el original de catorce a nueve variables y empeoró ETH. Redundancia demostrada y conveniencia de eliminar una feature son conclusiones diferentes. No se modificó el contrato activo.

### 4.3 Ablaciones e importancia exclusivamente internas

Se reutilizaron las ablaciones y permutaciones de **las dos ventanas internas de cada train**, guardadas en `feature_studies/.../selections.json`. No se eligieron variables nuevas con el test exterior. Hay 12 ventanas internas por activo, agrupadas en seis folds; sus solapamientos no las vuelven independientes.

La importancia individual interna disponible cubre las **39 columnas** de esa ronda, no toda la unión de 105. Para las restantes, esta auditoría aporta causalidad, redundancia y antecedentes de sus familias/modelos; no inventa una importancia individual validada. Los controles SMC completos y las extensiones tienen comparaciones históricas exteriores ya publicadas, que aquí son evidencia descriptiva y no selección de features. Ejecutar un nuevo barrido de las 105 variables duplicaría búsquedas sobre el mismo historial sin resolver la falta de confirmación independiente.

| Familia añadida | Δloss interna media BTC | Folds BTC con mejora en ambas ventanas | Δloss interna media ETH | Folds ETH con mejora en ambas ventanas |
|---|---:|---:|---:|---:|
| Momentum | −0,001374 | 2/6 | −0,002060 | 2/6 |
| Volatilidad | +0,003408 | 0/6 | −0,000504 | 1/6 |
| Flujo | −0,001978 | 4/6 | −0,001338 | 2/6 |
| Reversión | −0,001684 | 4/6 | −0,000978 | 1/6 |
| Rupturas | −0,000300 | 2/6 | −0,003162 | 4/6 |
| Estructura | −0,000601 | 3/6 | −0,001865 | 3/6 |
| Semanal/diario | +0,002753 | 1/6 | −0,002426 | 3/6 |
| Cruce BTC/ETH | +0,000473 | 2/6 | −0,000766 | 1/6 |

Ninguna familia mejora ambas ventanas internas en todos los folds. Ninguna feature presenta importancia positiva en todas las ventanas y bloques de los seis folds: el máximo es cuatro folds BTC y tres ETH. Hay contribuciones locales, pero su estabilidad es insuficiente. Gain/SHAP o una permutación positiva de un modelo no demostrarían información incremental universal; features correlacionadas pueden ocultar o repartir importancia. La [documentación de permutación](https://scikit-learn.org/stable/modules/permutation_importance.html) respalda esa cautela.

### 4.4 Inestabilidad y drift

Los estudios previos documentan PSI con bins fijados en train y cambios fuertes en algunas representaciones MACD/volatilidad. El nuevo diagnóstico guarda, para las 105 columnas y cada fold, correlaciones, constantes y fracción fuera de los percentiles 1/99 de train. En el último fold, volumen relativo fuera de ese rango alcanza 7,2% en ETH frente a 3,3% BTC. En cambio, MACD puede cambiar de distribución dentro del rango antiguo sin generar nuevos extremos. **Drift no equivale sólo a outliers.**

Esto demuestra cambios de distribución y fragilidad de contribuciones. No demuestra que un detector de drift o un modelo por régimen vaya a corregirlos: los especialistas ya se probaron y empeoraron la loss agregada.

## 5. Etiquetas, horizontes y dificultad del objetivo

### 5.1 Qué se predice realmente

| Objetivo | Implementación y evidencia | Diagnóstico |
|---|---|---|
| Dirección h1 | Original; XGBoost, logística y varios controles | Etiqueta cercana al equilibrio de clases; no exige cubrir costos ni predice magnitud |
| Dirección de varias velas | `multimarket` h3; ventanas y etiquetas purgadas | Cambia plazo físico y solapamiento; no constituye sólo “menos ruido” |
| Movimiento mayor que umbral de volatilidad | Escalas causales y barreras disponibles; conteo diagnóstico de ±1σ | Falta un estudio específico y predefinido de probabilidad de excedencia, sin confundirlo con TP antes que SL |
| Magnitud absoluta | No es el objetivo del experimento llamado `return_magnitude_h3` | Ese regresor estima **retorno neto con signo**. Predecir \|r\| merece evaluación distinta |
| TP antes que SL | Frontier supervisa cierres; label research usa OHLC con resolución conservadora de ambigüedad | Son eventos distintos; un timeout rentable puede pertenecer a otra clase según la definición |
| Señal favorable después de costos | Net h1/h3/h5/h10, TB neta, meta-labeling OOF | Mejor alineación con utilidad, pero clases, cobertura y número de oportunidades cambian; no hubo aceptación consistente |

No se reetiquetó ni sobrescribió ningún dataset. Se preservan definición, horizonte, período, benchmark y unidades de cada objetivo. Comparar una accuracy de barreras con otra de dirección sería incorrecto.

### 5.2 El ruido no se reduce a velas casi planas

Se midió cercanía a cero como \|retorno log futuro\| ≤0,1σ20 actual×√h, con umbral fijado antes del diagnóstico. En h1 representa **12,11% BTC y 11,05% ETH**. Eso permite señalar ambigüedad práctica cerca de cero; **no estima ruido irreducible ni error de etiquetado**. Una etiqueta determinista no se vuelve incorrecta por ser difícil de aprender.

Los errores tampoco se concentran únicamente en movimientos pequeños: el original acierta sólo 48,53% BTC y 47,78% ETH cuando la magnitud futura supera 1σ, semilla 42. Filtrar retrospectivamente los movimientos pequeños no resolvería ese hallazgo y usar la magnitud futura como filtro sería leakage.

### 5.3 Dirección frente a beneficio neto

En el historial disponible, el signo close/close y el signo open/open correspondiente coinciden en todas las observaciones h1 comparables. Por tanto, **no atribuyo aquí el fracaso a un desfase de apertura demostrado**. La diferencia conceptual sigue existiendo y en FX/gaps debe revisarse, pero en cripto esta prueba permite descartar esa explicación concreta como causa principal del signo.

Los costos sí cambian el objetivo: 6,92% de filas BTC y 5,49% ETH pasan de dirección favorable a resultado neto desfavorable. Entre etiquetas alcistas, **13,61% BTC y 10,58% ETH** no ganan después de costos. La proporción neta favorable cae a 43,92%/46,38% en h1. Con horizontes 3/5/10 disminuye esta discrepancia, pero aparecen solapamiento, menor muestra no superpuesta y mayor riesgo de cambio de régimen.

Una probabilidad de subida de 0,55 no garantiza expectativa neta positiva. Si G es ganancia bruta media, L la pérdida bruta media en magnitud y c el costo, una aproximación es `EV = p·G − (1−p)·L − c`. El umbral de equilibrio depende de esas magnitudes y de su estabilidad, no sólo de superar 0,5. Los filtros económicos anteriores ya exploraron parte de esta cuestión y no validaron una solución consistente.

### 5.4 Diagnóstico barato: magnitud frente a retorno con signo

Sin ajustar un modelo ML nuevo se comparó, en los seis folds purgados, el proxy `σ20[t]·√h·√(2/π)` para magnitud contra la media absoluta de train. Para retorno con signo se comparó la media móvil de sesenta retornos, multiplicada por h, contra cero. El factor normal es una **aproximación predefinida**, no una afirmación de normalidad de estos retornos.

| Activo | Horizonte | Reducción MSE magnitud | Reducción MSE retorno con signo |
|---|---:|---:|---:|
| BTC | 1 | +5,74% | −1,53% |
| BTC | 3 | +1,94% | −4,95% |
| BTC | 5 | −1,25% | −8,44% |
| BTC | 10 | −6,35% | −14,75% |
| ETH | 1 | +5,91% | −1,76% |
| ETH | 3 | +0,20% | −5,53% |
| ETH | 5 | −3,00% | −9,59% |
| ETH | 10 | −6,34% | −18,13% |

Cada columna usa **su propio benchmark**: no son escalas para rankear dos problemas como si fueran el mismo. La mejora de magnitud h1 ocurre sólo en **3/6 folds en ambos activos**. Es un indicio para investigar, no un resultado aceptado: no se ejecutó un test de significancia simultáneo de estos pronósticos ni una estrategia financiera basada en ellos. El fracaso de esta media móvil de signo tampoco descarta todo modelo de retorno esperado.

![Magnitud y signo](artifacts/predictive_failure_analyses/20261016T155154_cd986e21/magnitude_vs_sign.png)

## 6. Diagnóstico de modelos y probabilidades

### 6.1 Sobreajuste observable

Se releyeron métricas de **192 contratos históricos frontier** de componentes supervisados y 24 contratos de stacking; no se cargaron modelos prospectivos para evaluarlos. Se recalcularon 252 combinaciones guardadas de modelo/objetivo/activo/período, incluidas probabilidades raw, calibradas y baselines.

Medias simples sobre folds/activos de los ajustes de **dirección h1**, no un ranking seleccionado ni un promedio de tests independientes:

| Modelo | AUC train | AUC OOS raw | Loss train | Loss OOS raw | Brecha loss |
|---|---:|---:|---:|---:|---:|
| Logística local | 0,600 | 0,517 | 0,6742 | 0,7150 | +0,0408 |
| Logística cruzada | 0,616 | 0,516 | 0,6672 | 0,7223 | +0,0551 |
| XGBoost local | 0,735 | 0,508 | 0,6379 | 0,7093 | +0,0714 |
| XGBoost cruzado | 0,748 | 0,506 | 0,6317 | 0,7094 | +0,0777 |
| XGBoost balanceado/móvil | 0,807 | 0,510 | 0,6111 | 0,7095 | +0,0984 |
| Random Forest cruzado | 0,825 | 0,521 | 0,6214 | 0,7019 | +0,0805 |
| LightGBM cruzado | 0,837 | 0,512 | 0,5892 | 0,7172 | +0,1280 |
| CatBoost cruzado | 0,786 | 0,513 | 0,6290 | 0,7041 | +0,0751 |

Las filas usan aproximadamente 1.110 ejemplos medios de train, salvo la receta móvil, con 583. Hay evidencia de capacidad para ajustar el pasado y de que más ajuste no se traduce en generalización. La brecha también puede incluir drift, no sólo complejidad. **No hay evidencia de que la causa principal sea falta de capacidad del estimador.** Tampoco se demostró ausencia de subajuste para toda representación; falta una curva de aprendizaje controlada.

En selección frontier, dirección BTC tiene loss 0,693203 con RF calibrado frente a 0,693147 de p=0,5; su AUC 0,526 no basta para ganar en calidad probabilística. Stacking BTC tiene AUC 0,518 y loss 0,694766; el ensamble, AUC 0,524 y loss 0,693927. No aparece una mejora automática por combinar estimadores. El stacking usa predicciones base raw de una cola temporal ajena al ajuste base; no se encontró entrenamiento del metamodelo sobre predicciones base in-sample en esa ruta.

### 6.2 Calibrar puede corregir exceso de confianza sin crear discriminación

En frontier, la calibración mejora la loss media de dirección aproximadamente **0,01386**, pero **11 de 64** calibradores de dirección colapsan el ranking a una prevalencia constante porque el ajuste intentaría invertirlo. En todos los objetivos, colapsan 44/192. Son resultados negativos conservados, no modelos capaces de reconocer un régimen con certeza.

La cola de calibración, típicamente unas 110 observaciones utilizables por fold, contiene poca evidencia para estimar frecuencias condicionales de eventos raros. Una salida constante o por debajo de 0,55 puede ser una consecuencia razonable de señal insuficiente; cero operaciones no prueba que el modelo esté “roto”. Una mejor Brier o log-loss tampoco identifica por sí sola calibración: ambas combinan discriminación y fiabilidad, como explica la [documentación de calibración](https://scikit-learn.org/stable/modules/calibration.html).

Se verificó además la identidad exacta:

`Brier = Var(y) + Var(p) + (media(p)−media(y))² − 2·Cov(p,y)`.

Para el original BTC, `Var(p)=0,005277`, sesgo²=0,000477 y `2Cov=0,001801`: la dispersión de probabilidades supera ampliamente la contribución útil de su covarianza con el resultado. En ETH, `2Cov=−0,000698` y `Var(p)=0,005975`. Esta es una explicación algebraica del mal Brier observado, **no una estimación del error de Bayes ni una nueva calibración aprendida con el test**.

### 6.3 Barreras y desbalance: un acierto alto puede ocultar ausencia de decisiones útiles

En el segmento descriptivo frontier de barreras al cierre h5, BTC tiene sólo **9,62%** de eventos positivos. El XGBoost obtiene **90,38% de accuracy**, exactamente lo que logra la prevalencia prediciendo siempre negativo. AUC alcanza 0,707, pero su loss 0,32785 no mejora a prevalencia 0,32495 y no genera entradas con la política de referencia. En ETH hay AUC 0,637 y una pequeña mejora de loss, con accuracy 86,19%, también igual a la clase mayoritaria.

Esto muestra **información de ranking posible en ciertos eventos**, no rentabilidad automática, y descarta interpretar accuracy elevada como ventaja. El umbral y la magnitud de los pagos importan; cambiarlo después de observar este segmento sería otra selección retrospectiva.

### 6.4 Excepción que debe conservarse: EUR/USD H1

En el piloto verificado 2020–junio de 2025, el compacto H1 tiene **19.146 predicciones OOS**, accuracy 52,25%, AUC 0,53196 y loss 0,690443 frente a 0,693318 de prevalencia. AUC supera 0,5 en los seis folds. El intervalo bootstrap puntual de Δloss frente a prevalencia, bloques de treinta filas, es **[−0,004296; −0,001511]**.

Es una señal favorable que contradice la frase absoluta «ningún modelo predice nada». Pero los intervalos son marginales, no corregidos por todos los pares, frecuencias y pilotos; treinta filas H1 son treinta horas observadas, no treinta días. Se necesita dependencia de mayor plazo, otra fuente, otro par y evaluación independiente. La estrategia ejecuta 1.844 operaciones y pierde **17,75%**, con PF 0,752 y **0/6 folds rentables**. El modelo neto H1 tiene AUC 0,546, pero pierde 3,99% en 182 trades. La evidencia favorable de ranking y la adversa económica se mantienen juntas.

Para el mismo piloto EUR/USD, el compacto H4 obtiene AUC 0,521 y loss 0,696187, peor que prevalencia; D1 tiene AUC 0,483 y loss 0,717530. No se extrapola la señal H1 a otros mercados ni se usa para recomendar operar.

## 7. Cuándo fallan: análisis de errores

Las categorías siguientes se fijaron como diagnóstico; **no se crearon reglas ni se volvió a seleccionar un modelo con esos resultados**. Los bins de magnitud usan el resultado futuro únicamente para describir errores.

### 7.1 Volatilidad, tendencia y tamaño del movimiento

| Condición, original semilla 42 | BTC AUC | ETH AUC | Lectura |
|---|---:|---:|---|
| Volatilidad baja causal | 0,472 | 0,478 | Ranking adverso en ambos activos; 432/367 filas |
| Volatilidad alta causal | 0,548 | 0,524 | Mejor ranking puntual; no es una regla validada de entrada |
| Régimen bull | 0,499 | 0,510 | No hay un beneficio uniforme de tendencia |
| Régimen bear | 0,502 | 0,481 | Depende del activo |
| Régimen lateral | 0,525 | 0,494 | No se replica el signo del efecto entre activos |
| Magnitud futura >1σ | 0,496 | 0,455 | Especialmente débil en movimientos grandes de ETH |

El mal ranking con volatilidad baja se repite con la segunda semilla: AUC 0,482 BTC / 0,487 ETH. El mal ranking de grandes movimientos ETH también se repite: 0,458. En BTC, en cambio, el grupo de movimientos grandes pasa de AUC 0,496 a 0,507 entre semillas. Es evidencia de **algunos patrones descriptivos repetidos y otros inestables**; falta demostrar qué explicaciones persisten en cada ventana independiente.

La media simple por fold de correlación Spearman entre probabilidad y retorno futuro es sólo 0,031 BTC y 0,011 ETH, semilla 42. La accuracy ponderada por magnitud promedia aproximadamente 49,91%/49,78% por fold. Aciertos pequeños y errores grandes pueden coexistir con una accuracy ordinaria ligeramente superior a 50%.

### 7.2 Confianza

Confianza significa aquí `max(p,1−p)`, no exclusivamente probabilidad de una compra rentable:

| Activo/bin | Confianza media | Aciertos | n |
|---|---:|---:|---:|
| BTC 0,55–0,60 | 57,23% | 50,48% | 523 |
| BTC 0,65–0,70 | 66,98% | 46,27% | 67 |
| BTC 0,70–0,80 | 71,79% | 50,00% | 16 |
| ETH 0,55–0,60 | 57,22% | 48,59% | 496 |
| ETH 0,60–0,65 | 62,14% | 46,61% | 236 |
| ETH 0,70–0,80 | 72,23% | 47,50% | 40 |

No hay una relación monotónica fiable que permita confiar más en los extremos. Los bins pequeños son ruidosos; invertirlos retrospectivamente no sería evidencia. Los resultados de calibradores y cambios de semilla son compatibles con sobreconfianza y poca estabilidad, sin probar que toda probabilidad sea inútil.

![Confianza y aciertos](artifacts/predictive_failure_analyses/20261016T155154_cd986e21/confidence.png)

### 7.3 Horizonte, temporalidad y activo

La dirección h3 en el estudio multimercado mejora puntualmente AUC de BTC/ETH a 0,529/0,543 y AUD/USD a 0,552, pero empeora USD/CHF a 0,420 y no mejora consistentemente la loss frente a prevalencia. H1 EUR/USD es una excepción local favorable; H4/D1 no la reproducen. Por tanto, **no hay un horizonte universal que solucione el problema**.

Los timeouts, el desbalance de barreras, gaps y la diferencia entre etiqueta y política de salida cambian qué significa un error. No se agregaron AUC de objetivos distintos como si midieran el mismo problema.

## 8. Validación estadística y económica

### 8.1 Qué está bien implementado

1. Separaciones cronológicas, purga por el momento de disponibilidad del target y embargo explícito en las rondas recientes. La purga evita usar una etiqueta aún no observada; el embargo no convierte las filas restantes en iid.
2. Escalado y modelos de estado ajustados con train; calibración temporal separada; metamodelos con oportunidades OOF de su padre en las rutas revisadas.
3. **192 contratos temporales frontier** y **672 contratos multimercado** comprobados: entrenamiento anterior a calibración y ésta anterior a OOS, con las brechas guardadas. Ninguno de esos contratos incumple el orden verificado.
4. Simuladores con siguiente precio disponible, costos, no solapamiento según política, gaps y marcas de capital. El labeler y los kernels de eventos comparten lógica; las pruebas distinguen nacimiento, confirmación, apertura y cierre.
5. Benchmarks p=0,5, prevalencia, logística, efectivo y buy-and-hold; resultados rechazados conservados. No hay promoción automática por optimizar.
6. Bootstrap por bloques dentro del fold y comparación pareada; corrección simultánea en varios estudios recientes; reservas futuras protegidas.

### 8.2 Qué no permite afirmar el procedimiento

- **Walk-forward retrospectivo no es un test prospectivo.** El diseño, las familias, los cortes y algunas recetas heredadas se escogieron después de observar partes del historial. No basta con purgar las filas para borrar esa información del investigador.
- La corrección max-|t| o Bonferroni de una ronda no corrige automáticamente todas las decisiones de las rondas anteriores. Los intervalos FX son marginales; algunos usan 500 remuestras y bloques 7/30 filas independientemente de temporalidad. Hay que conservar esa limitación al interpretar hallazgos H1.
- Los folds comparten entrenamientos y activos correlacionados. No corresponde calcular un test iid con seis folds como si fueran seis universos independientes o multiplicar evidencia por cada semilla.
- El bootstrap con predicciones fijas condiciona a los modelos ya ajustados; no incorpora completamente incertidumbre de reentrenamiento, selección de familias, proveedores, costos o futuros regímenes.
- Censurar etiquetas que no terminan antes de un límite es necesario; usar la duración futura para decidir si una operación habría sido ofrecida sería otro problema. Los motores recientes separan disponibilidad de forecast y disponibilidad de outcome y fijan cortes de entrada por calendario.
- Un intervalo que incluye cero no demuestra efecto exactamente nulo. Un intervalo favorable después de muchas inspecciones no demuestra una ventaja real sin replicación y control de selección.

La [investigación sobre backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) motiva esta separación entre resultado seleccionado y evidencia. No se calculó aquí una probabilidad global de sobreajuste del proyecto: faltan una familia completa predefinida de decisiones y replicaciones independientes para identificarla de forma defendible.

### 8.3 Costos: descomposición sobre las mismas operaciones

Se reconstruyó el retorno bruto eliminando únicamente comisión y fricción de los fills registrados, manteniendo entradas y salidas. Se verificó algebraicamente el arrastre logarítmico de costos en cada ledger. En cripto base son 10 bps de comisión y 6 bps de slippage+medio spread por lado; el costo redondo efectivo ronda **31,95 bps** con la convención multiplicativa.

| Variante, semilla 42 | Trades BTC | Bruto BTC | Neto BTC | Bruto ETH | Neto ETH |
|---|---:|---:|---:|---:|---:|
| Original | 524 | −10,01% | −83,17% | −64,23% | −93,71% |
| Estructura | 554 | −22,99% | −86,92% | −58,53% | −93,18% |
| Global + cluster | 513 | +12,81% | −78,15% | −64,48% | −93,79% |
| Especialistas por reglas | 591 | +0,79% | −84,79% | −59,63% | −94,76% |

Los conteos ETH correspondientes no son los BTC de la tabla: están guardados por activo en el JSON. Una compra y venta seguidas de una nueva entrada paga costos otra vez, como define el original. En BTC original, 524 rondas multiplican la riqueza por un factor de costo aproximado `exp(−1,6768)`, además de su resultado bruto. Eso explica una parte grande de la erosión sin necesitar especular sobre la red neuronal.

Hay dos problemas distintos: algunas señales sólo tienen una ventaja bruta pequeña que no cubre costos; otras pierden incluso sin ellos. Reducir costos, exposición o cantidad de entradas puede reducir pérdida y DD, pero no demuestra mejor predicción. En el overlay de riesgo anterior, las probabilidades no cambian; su mejora no debe contarse como aumento de AUC.

Los spreads cripto constantes y el slippage son supuestos, no observaciones del libro. FX usa bid/ask indicativos y escenarios de financiación/slippage, tampoco fills garantizados. No se midieron impacto, latencia o drawdown intravela completo en todas las rutas. Las nuevas métricas conservan exactamente el ámbito del simulador que las generó.

## 9. Causas probables y respuesta a las preguntas centrales

### 9.1 Matriz de conclusiones

| Conclusión | Estado | Evidencia y qué falta para atribuir causalidad |
|---|---|---|
| Los estimadores sobreajustan varias representaciones | **Demostrado en las configuraciones auditadas** | Brechas train/OOS, capacidad de ajustar el pasado y falta de generalización. Falta comparar curvas de aprendizaje controladas |
| La rotación y los costos son un obstáculo principal de rentabilidad | **Demostrado** | Replay exacto bruto/neto y pérdidas con costos dobles; no explica toda pérdida bruta |
| Las probabilidades contienen exceso de dispersión para su información observada | **Demostrado en originales y varias recetas** | Brier, covarianza y bins de confianza; calibración ayuda en loss sin crear ranking estable |
| Hay redundancia y familias inestables | **Demostrado** | Identidad MACD, casi duplicado cuerpo/retorno, constantes D1, ablaciones internas y selección variable |
| La representación OHLCV actual tiene poca señal incremental generalizable de dirección | **Explicación probable, evidencia media-alta** | Diversidad de modelos/familias y baselines difíciles de superar. No es una cota de Bayes ni una prueba sobre todos los datos posibles |
| La información faltante podría cambiar el resultado | **Hipótesis razonable** | El sistema omite fuentes potencialmente distintas de precio/volumen. Falta una ablación temporal de información nueva auténtica |
| El drift contribuye al fracaso | **Probable, evidencia media** | Distribuciones y desempeño cambian; no se aisló causalmente del ruido y del sobreajuste |
| Defectos Forex distorsionan horizontes y ejecución | **Demostrado como limitación de datos/interpretación** | Ausencias, cruces, duración variable. Falta un contraste amplio con proveedor independiente; no explica cripto |
| El pipeline contiene una fuga futura que explica todo | **No demostrado** | Los controles revisados pasan y los resultados son débiles, no sospechosamente extraordinarios. No se garantiza ausencia absoluta de bugs |
| Más complejidad o una red neuronal solucionaría el problema | **Sin respaldo** | Modelos más flexibles ajustan mejor train sin mejorar suficientemente OOS |
| No hay ninguna señal en ningún mercado | **Refutado como afirmación absoluta por indicios históricos** | H1 EUR/USD y ciertos rankings de barreras contienen señal observada; falta confirmación económica independiente |
| Ya existe una ventaja económica aprovechable | **No demostrado** | Los casos favorables no superan conjuntamente costos, estabilidad, selección múltiple y validación nueva |

### 9.2 Qué funciona y qué conviene conservar

Conservar infraestructura de procedencia, contratos, pruebas causales, separaciones temporales, registros de rechazo, benchmarks simples, simuladores y reservas. Conservar **como controles**, no como sistemas ya rentables, el original, logística, p=0,5 y prevalencia. Conservar el hallazgo H1 EUR/USD para replicación; conservar predicción de magnitud/volatilidad como posible línea de riesgo, sin atribuirle alpha direccional.

Archivar como hipótesis sin respaldo actual: más familias técnicas acumuladas, especialistas por régimen con esta muestra, cruce BTC/ETH como solución general, incremento de complejidad y filtros SMC muy raros. No borrar modelos, ni reescribir los resultados negativos, ni declarar inútil para siempre una técnica que sólo fue refutada bajo un protocolo concreto. La combinación estructura+régimen sigue correctamente sin ejecutar por falta de evidencia previa.

### 9.3 Qué información puede faltar

Libro y operaciones con timestamps utilizables, spread/depth/impacto verdaderamente ejecutables, funding y open interest point-in-time para derivados, variables macro con calendario de publicación y revisiones, diferenciales de tipos en FX y datos de eventos o flujos que no sean transformaciones del mismo cierre. También faltan más episodios independientes y una réplica fuera del historial explorado. Ninguna fuente se presume útil: coste, licencias, revisiones y disponibilidad deben verificarse antes de usarla.

### 9.4 ¿Es posible que no haya una ventaja explotable en estos datos?

**Sí. Es compatible con los resultados, especialmente para dirección diaria de los dos criptoactivos con estas features y costos.** Pero también son compatibles una señal demasiado pequeña para la potencia disponible, una señal local que no cubra costos o una representación que omita información relevante. La auditoría no identifica de manera única esas posibilidades.

Que modelos distintos fracasen después de probar muchas representaciones reduce la justificación de seguir buscando por complejidad. No demuestra una ley universal de imposibilidad. La literatura distingue persistencia de volatilidad, inestabilidad de relaciones predictivas y dificultad OOS; los resultados de otros mercados no sustituyen un test local. [Cont, hechos estilizados](https://rama.cont.perso.math.cnrs.fr/pdf/empirical.pdf), [Goyal y Welch, desempeño OOS](https://www.nber.org/papers/w10483) y [Campbell y Thompson, límites de interpretar el fracaso OOS](https://www.nber.org/papers/w11468) ofrecen perspectivas complementarias, no una certificación de eficiencia para BTC/FX.

## 10. Plan priorizado: máximo cinco experimentos

**No se ejecutaron los entrenamientos de este plan.** Los diagnósticos anteriores usan cálculos deterministas, pronósticos simples predefinidos y artefactos ya existentes. Cada experimento debe congelarse antes de ejecutarse; superar un filtro histórico sólo habilita investigación posterior. Los experimentos 2–4 se detienen si falla el contrato de datos del primero. El quinto requiere un candidato que supere criterios de desarrollo y respeta todas las reservas anteriores.

### 1. Contraste de reloj, proveedor y controles conocidos del pipeline

- **Valor informativo:** muy alto; permite refutar una explicación de ingeniería antes de invertir en señales. **Costo:** bajo, cero ajustes históricos nuevos; hasta doce ajustes pequeños sólo en fixtures de señal conocida/nula. **Riesgo de sobreajuste:** bajo.
- **H0:** las anomalías de sincronización/etiquetado o un defecto del pipeline explican el aparente desempeño. **H1:** el pipeline respeta la disponibilidad y preserva una señal artificial causal conocida, mientras la señal histórica no depende de esos defectos.
- **Datos:** mismos cierres/quotes de EUR/USD H1 en una segunda fuente; tres períodos fijos que incluyan un hueco, un cambio DST y un tramo normal; BTC/ETH locales; fixtures sintéticos separados de cualquier evidencia financiera.
- **Métricas:** violaciones de disponibilidad, duplicados/conflictos, horas faltantes, duración target y latencia señal→entrada; delta de etiquetas y loss al alinear horizontes físicos. Benchmarks: relojes calculados manualmente, cero/prevalencia, fixtures sin señal y con señal retardada conocida.
- **Aceptación antes de correr:** cero uso de datos no disponibles, cero discrepancias contra los casos temporales manuales; AUC≥0,70 en control positivo y ausencia de mejora confirmada en control nulo. Diferencias entre proveedores se registran con unidades de tick/pip, sin exigir igualdad de venues ni escoger el proveedor de mejor retorno. Una ventaja histórica sólo continúa si no depende de las anomalías en los tres tramos y conserva el signo del contraste pareado; si no, se refuta la interpretación previa, no se “repara” retrospectivamente su resultado.

### 2. Separar pronóstico de magnitud, excedencia y signo a horizonte físico fijo

- **Valor:** alto; evalúa qué parte del futuro es aprendible. **Costo:** bajo, máximo doce ajustes simples predefinidos. **Riesgo:** bajo-medio.
- **H0:** ninguna representación supera a su benchmark propio de magnitud/excedencia/signo. **H1:** existe pronosticabilidad de magnitud o excedencia que no se traduce necesariamente en dirección.
- **Datos:** BTC/ETH D1 actuales como desarrollo; una sola formulación principal h1, `|r[t+1]|` y evento `|r[t+1]|>σ20[t]`. Un único regressor simple regularizado y una logística de excedencia, sin buscar ventanas; cero/media train/EWMA como controles. Signo conserva la etiqueta original y sus predicciones ya guardadas.
- **Métricas:** MAE/MSE y skill frente a media/mediana histórica compatibles con cada loss; Brier/log-loss para excedencia; IC descriptivo. Sin convertir una mejora de riesgo en recomendación direccional.
- **Aceptación:** reducción relativa de MSE≥2% en ambos activos, mejora en ≥4/6 folds y límite superior de Δloss pareada <0 en bloques 7/30 con corrección conjunta por objetivos y activos; para excedencia, mejora log-loss≥0,003 y Brier no peor frente a prevalencia. El resultado sigue siendo exploratorio hasta réplica. Si sólo mejora magnitud, se conserva esa conclusión y se rechaza la extensión a alpha de signo.

### 3. Curvas de aprendizaje y estabilidad con una sola receta

- **Valor:** alto; distingue insuficiencia de muestra reciente, complejidad y drift mejor que otra grilla. **Costo:** moderado, máximo 36 ajustes. **Riesgo:** medio, contenido por diseño fijo.
- **H0:** cambiar cantidad/antigüedad de train no mejora de forma estable la generalización del original. **H1:** una política de datos fija reduce la brecha y mejora pérdida OOS, replicándose.
- **Datos:** BTC/ETH, tres cortes cronológicos predefinidos de desarrollo, ventanas de 365, 730 y todo el pasado, dos semillas; misma intersección de filas OOS, mismas catorce features y receta original. No seleccionar columnas o profundidad.
- **Métricas:** loss/Brier/AUC train y validación exterior, brecha, varianza entre semillas y tamaño de muestra; p=0,5, prevalencia y logística ya definida como benchmarks.
- **Aceptación:** una política preespecificada debe ganar loss≥0,003 y AUC≥0,005 en ambos activos y semillas, mejorar conjuntamente en ≥2/3 cortes y superar los baselines con intervalos simultáneos. Si sólo reduce train/OOS gap pero no loss exterior, no se adopta. No se elige el mejor tamaño después de mirar el fold final; se usa una regla interna anterior o se informa únicamente la curva.

### 4. Una fuente incremental point-in-time, con ablación emparejada

- **Valor:** potencialmente alto; pone a prueba la hipótesis de información faltante. **Costo:** medio-alto de adquisición/limpieza, máximo 24 ajustes. **Riesgo:** medio-alto por timestamps y selección de fuentes.
- **H0:** una familia realmente nueva no aporta información condicional sobre el baseline OHLCV. **H1:** mejora predicción y utilidad fuera de muestra después de su disponibilidad real.
- **Datos:** elegir y justificar **una sola familia antes de observar su relación con outcomes**; por ejemplo, funding/open interest de instrumentos concretos o diferenciales macro FX con fecha de publicación. No combinar fuentes hasta que una ablación justifique su contribución. Documentar venue, retrasos y revisiones; no asumir que datos retrospectivos se conocían entonces.
- **Métricas/benchmarks:** baseline fijo frente a baseline+familia, p=0,5/prevalencia/logística; loss, Brier, AUC, cobertura, retorno neto, DD y cantidad de operaciones; mismos períodos y ejecución observada o stress de costos.
- **Aceptación:** loss≥0,003 y AUC≥0,005 de mejora, estabilidad en ≥2/3 cortes y ambos activos elegibles; intervalos corregidos que excluyan cero, neto positivo y DD no peor con costos base y stress. Presupuesto y parada fijos. Si la fuente no permite réplica en dos activos comparables, debe declararse evidencia local y continuar únicamente a una réplica independiente específica, sin relajar los umbrales para aceptarla.

### 5. Réplica prospectiva única de un candidato congelado

- **Valor:** imprescindible para afirmar ventaja real; debe tener prioridad de diseño aunque su ejecución sea posterior. **Costo:** bajo de cómputo, alto de tiempo calendario. **Riesgo de sobreajuste:** bajo sólo si se evita inspección/tuning intermedio.
- **H0:** el candidato no supera al benchmark neto después de costos en datos no utilizados para diseño. **H1:** mantiene una ventaja predictiva y económica de magnitud predefinida.
- **Datos:** un período nuevo fijado después de registrar el protocolo, **sin apropiarse ni consumir ahora las reservas de otros estudios**. Mantener intactos sus modelos/sellos. Fijar universo, candidato, timestamp de congelación, política de datos, costos y fecha o tamaño de cierre antes de observar outcomes.
- **Métricas:** una primaria predictiva y una económica previamente declaradas; loss pareada contra baselines, riqueza log neta, DD, operaciones, estabilidad por activos/períodos. Costos base y adversos establecidos antes de comenzar.
- **Aceptación:** umbral económico mínimo que cubra costos e incertidumbre, IC simultáneos de ventaja predictiva y económica que excluyan cero, DD dentro del presupuesto y consistencia entre activos. Calcular potencia usando dependencia de desarrollo; treinta operaciones son apenas un piso operativo y no certifican potencia. Una única lectura al cierre; no ampliar el plazo o cambiar el candidato porque el resultado sea desfavorable. Si la muestra planeada no alcanza, registrar “inconcluso”, no “ventaja”.

## 11. Evidencia, reproducibilidad y límites de esta auditoría

Corrida completa: **`20261016T155154_cd986e21`**, en [artifacts/predictive_failure_audits/20261016T155154_cd986e21](artifacts/predictive_failure_audits/20261016T155154_cd986e21). Preservó byte por byte **34.423 archivos anteriores**, con hash conjunto `0b79e34ccb9c83d69939266116afcf3f719ba33066b06e68d3893929f6490177`. Cero ajustes nuevos, cero aperturas de reservas prospectivas y cero promociones.

| Archivo de evidencia | Contenido |
|---|---|
| [plan.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/plan.json) | Protocolo anterior a los cálculos, versiones de estudios, entorno, hash del script y hashes de archivos protegidos |
| [data.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/data.json) | 23 snapshots: procedencia, integridad, anomalías, distribución, dependencia y límites de ESS |
| [features.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/features.json) | Lista de 105 columnas, ablaciones e importancias internas, redundancia y controles causales |
| [targets.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/targets.json) | Horizontes, ruido cercano a cero, desacuerdo neto y pronósticos simples purgados |
| [models.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/models.json) | Raw/calibrado, generalización, calibradores colapsados y contratos de stacking |
| [errors.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/errors.json) | Errores por activo, semilla, volatilidad, régimen, magnitud futura, confianza y fold |
| [forex.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/forex.json) | Resultados H1/H4/D1 y activos, duraciones reales, constantes y contratos temporales |
| [economics.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/economics.json) | Replay bruto/neto sobre las mismas operaciones, sin elegir nuevas entradas |
| [summary.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/summary.json) y [artifacts.json](artifacts/predictive_failure_audits/20261016T155154_cd986e21/artifacts.json) | Resultado de preservación y hashes del paquete diagnóstico |
| [supplement.json](artifacts/predictive_failure_analyses/20261016T155154_cd986e21/supplement.json) | Identidad Brier exacta, referencia de potencia y verificación de hashes del diagnóstico |

Hubo una primera corrida técnica parcial, `20261016T154901_05736edf`, detenida al encontrar que el contrato de stacking no guarda las mismas métricas que los componentes. Se adaptó el lector y se ejecutó la corrida completa sin sobrescribir la parcial. No hubo entrenamiento en ninguna; no son dos replicaciones científicas. En stacking se registra explícitamente la ausencia de métricas in-sample en esos contratos, en lugar de inventarlas.

Validación local: **253 pruebas aprobadas**, incluyendo tres nuevas para ACF, interpretación del ESS y descomposición exacta de costos; lint y formato aprobados. Los tests sintéticos comprueban invariantes; las métricas del informe provienen de históricos reales. La ausencia de una prueba de aprendizaje sintética end-to-end específica permanece como experimento propuesto, no como resultado ya demostrado.

```powershell
# Diagnósticos sin entrenamiento, en un directorio nuevo:
.\.venv\Scripts\python.exe scripts/audit_predictive_failure.py
# Usar la corrida completa devuelta por el comando anterior:
.\.venv\Scripts\python.exe scripts/summarize_predictive_failure.py artifacts/predictive_failure_audits/<run>
.\.venv\Scripts\python.exe -m pytest -q
```

La auditoría no midió el error irreducible del mercado, no estimó una probabilidad global de que “no exista alpha” y no abrió un test nuevo. Su conclusión defendible es más concreta: **los sistemas actuales no han demostrado una ventaja económica robusta; existen causas observables de fracaso y algunas señales locales que merecen pruebas mejores, no una mejora artificial obtenida por seguir buscando en el mismo historial.**
