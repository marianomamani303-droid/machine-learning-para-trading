# Triple Barrier y Meta-Labeling: viabilidad técnica y económica

**Resultado al 15/10/2026: las técnicas son implementables, pero esta investigación no demuestra señales consistentemente rentables fuera de muestra. Se conserva el enfoque original.** Hay mejoras parciales y resultados positivos aislados; ninguno se sostiene en ambos activos, períodos y costos de stress. Los experimentos permanecen separados del entrenamiento y la predicción de producción.

Run `20261015T170214_699ffc3e`: **486 ajustes**, con cota fijada antes de empezar de 490 y límite de 600. Incluye 238 XGBoost, 36 logísticas y 212 calibraciones sigmoides. Se compararon **83 variantes de decisión**, con seis folds por activo sobre BTCUSDT/ETHUSDT. No se ampliaron hipótesis después de observar resultados.

## Qué predice y cómo opera el sistema existente

| Componente | Objetivo histórico | Significado de la probabilidad | Ejecución |
|---|---|---|---|
| Pipeline original / enhanced / SMC | `close[t+h] > close[t]`; h=1 por defecto | Subida del cierre, sin exigir superar costos | Long/cash, umbral 0,55; compra en apertura siguiente y venta h aperturas después |
| `advanced_research` direction_h1 | Misma dirección close/close | Subida del cierre | Overlay de riesgo independiente del significado del objetivo |
| `advanced_research` net_h3/net_h5 | Retorno neto entre próxima apertura y apertura h días después >0 | Rentabilidad neta de ese trade hipotético a horizonte fijo | La política efectiva puede usar otros stops/salidas; debe verificarse su alineación |
| `frontier_research` barrier_close_h5 | TP de 2 ATR antes de SL/timeout, vigilado al cierre; salida en apertura posterior | Probabilidad de evento TP, no de ganar dinero | La estrategia defensiva también tiene sus propias reglas; objetivo y trade no son equivalentes automáticamente |
| Meta-modelo avanzado anterior | Rentabilidad de oportunidades de pronósticos base OOS con salidas ATR/probabilidad/tiempo | Rentabilidad condicional de esa oportunidad | Filtra entradas del overlay; no predice el drawdown de la cartera |

Código auditado: `preprocessing/pipeline.py`, `backtesting/simulator.py`, `advanced_research/targets.py`, `advanced_research/meta.py`, `frontier_research/targets.py` y `risk_overlay/simulator.py`.

El original **no predice si una operación será rentable después de costos**. La etiqueta usa cierres y el backtest usa aperturas posteriores; la propia implementación original reconoce esa diferencia. Se conserva como baseline mediante reajustes temporales de la misma receta, sin aplicar retrospectivamente un modelo entrenado después del fold. Un test verifica que el nuevo motor, con salida fija h1 y las mismas fricciones, reproduzca exactamente el libro de operaciones del simulador original.

El objetivo anterior de barreras cuenta un timeout rentable como clase negativa si no hubo TP. Tampoco garantiza que un TP observado al cierre termine siendo rentable al ejecutar en otra apertura. Esta ronda separa evento de barrera, precio de ejecución y rentabilidad financiera.

## Triple Barrier implementado

La referencia de [Triple Barrier y Meta-Labeling](https://mlfinpy.readthedocs.io/en/stable/Labelling.html) describe barreras horizontales y una vertical, y el uso de un modelo secundario para aceptar o rechazar la oportunidad propuesta por un modelo primario. Se adapta esa idea a las aperturas, OHLC diario y costos de este proyecto; no se instala otra dependencia ni se importa su código.

En cada cierre de señal t se calcula σ20 con retornos logarítmicos históricos. Las distancias se congelan en ese momento. Al entrar en la apertura siguiente P, el bracket queda en `TP = P*(1+1,5*σ20[t])` y `SL = P*(1−σ20[t])`. El precio de apertura se usa cuando ocurre la ejecución, nunca como feature anticipada. No se actualizan las barreras con volatilidad posterior.

Se preservan dos etiquetas diferentes:

- `event_label = +1 / −1 / 0`: TP / SL / límite temporal.
- `target = 1[unit_net_return > 0]`: rentabilidad del trade modelado después de costos; los timeouts se clasifican por su resultado neto. Esta es la variable financiera binaria usada por XGBoost y por los meta-modelos long.

No se convierte un timeout en una pérdida automáticamente ni se descartan empates intravela después de conocer el futuro. Un evento TP de barrera estrecha puede tener `target=0` por costos; existe un test específico de ese caso. El experimento guarda también la etiqueta ternaria; no se entrenó otro clasificador multiclase ni se presenta un aumento de accuracy entre objetivos diferentes como evidencia de superioridad.

| Familia | Variantes fijadas |
|---|---|
| Horizonte fijo neto | 1, 3, 5 y 10 días entre aperturas, sin barreras horizontales |
| Triple Barrier con σ20 | TP 1,5σ / SL 1σ; horizontes 1, 3, 5 y 10 |
| Asimetría adicional | TP 2σ / SL 1σ; h5 |
| Barreras porcentuales | TP 3% / SL 2%; h5 |
| ATR | TP 1,5 / SL 1 veces ATR14 como fracción del cierre de señal; h5 |
| Supervisión al cierre | TP 1,5σ / SL 1σ vigilados al cierre y ejecutados en apertura posterior; h5 |

La σ diaria se mantiene con la misma definición al comparar horizontes: cambia el tiempo permitido, no se optimiza simultáneamente una regla de escalado `sqrt(h)`. Un control adicional pondera el entrenamiento de TB h5 por unicidad media de sus eventos; los pesos se calculan sólo con eventos de train. El [solapamiento de etiquetas](https://mlfinpy.readthedocs.io/en/latest/Sampling.html) motiva ese diagnóstico, sin tratar los trades hipotéticos superpuestos como observaciones independientes.

## Limitaciones de OHLC diario y ejecución causal

Una vela puede cruzar TP y SL sin revelar cuál ocurrió primero. El supuesto principal es **SL primero**. Se repite la ejecución con **TP primero**, manteniendo modelos y probabilidades, como sensibilidad al orden desconocido; no se eliminan esas velas ni se elige su orden por el cierre futuro. En los eventos históricos observados, el porcentaje ambiguo de TB σ h5 es 1,56% BTC / 1,19% ETH; con barreras fijas 3%/2% sube a 5,89% / 11,74%. Son estadísticas del etiquetado histórico completo, no métricas predictivas OOS.

Los gaps de SL ejecutan en la apertura peor disponible, sin simular un fill garantizado en el stop. Los TP se limitan conservadoramente al precio del bracket incluso con gap favorable. Se asumen órdenes bracket colocadas tras entrar, con la fricción modelada; las velas no acreditan liquidez ni fills reales. En intravela, la fecha de observación de la etiqueta es el cierre de la vela, porque no existe un timestamp más preciso. Esto evita permitir el uso de una etiqueta antes de que el OHLC la confirme.

El motor avanza cronológicamente. En la apertura sólo consulta apertura y cierre previo; el máximo/mínimo de esa vela no puede liberar una posición para comprar en esa misma apertura. Después se procesa lo que sucede durante la vela. No permite reentrada después de una salida intravela ese mismo día. El labeler y el motor comparten un paso de ejecución para evitar objetivos incompatibles con el simulador.

El límite temporal ejecuta h aperturas después de entrar. Todas las estrategias bloquean nuevas entradas en los últimos **once días** del fold, un corte calendario conocido que cubre el horizonte máximo de diez días. No se decide disponibilidad por `target`, `target_time`, duración futura o éxito del evento. Una etiqueta temprana ya observada se conserva aunque falte todo el horizonte restante del archivo; lo no observable permanece censurado.

## Meta-Labeling y calibración

El modelo base conserva las **14 features originales**, 160 árboles, profundidad 2, learning rate 0,03, lambda 5, min_child_weight 10 y semilla 42. No se mezclan cambios de features con cambios de objetivo.

Se generan **4.198 probabilidades base OOS** mediante 46 modelos trimestrales, desde enero de 2021 hasta septiembre de 2026. Cada modelo sólo usa etiquetas observadas antes de su cutoff y embargo. Una señal con `p_base >= 0,55` propone una oportunidad long; el meta-modelo aprende si su ejecución h1 o TB σ h5 fue rentable. No aprende de probabilidades base in-sample ni de las operaciones elegidas después de conocer resultados.

El meta-modelo utiliza las 14 features originales más `base_probability`. Se prueban logística L2 con C=0,1 y XGBoost de 80 árboles, profundidad 2, lambda 20 y min_child_weight 20. Su etiqueta es la rentabilidad neta de la oportunidad. La base mantiene la dirección long; meta sólo acepta/rechaza. Las oportunidades hipotéticas de train pueden superponerse, pero el portfolio mantiene como máximo una posición abierta.

La [calibración de probabilidades](https://scikit-learn.org/stable/modules/calibration.html) usa una cola disjunta de 120 días, purgada por final real del evento y embargo de seis días. Raw/cal comparten exactamente el modelo y el train, aislando la calibración. Se reutiliza la sigmoide regularizada del proyecto: si intenta invertir el ranking, se conserva una constante de prevalencia. Se registraron **83 colapsos de ranking en 212 calibraciones**. Una probabilidad constante puede reducir exceso de confianza sin aportar discriminación.

Las ventanas meta de calibración contienen entre 18 y 71 oportunidades. En el último fold BTC sólo hay 18: sus cuatro recetas meta calibradas se abstienen por no alcanzar el mínimo de 20; no se sustituyen silenciosamente por scores raw. Las métricas raw/cal de meta se comparan también sobre su intersección común, evitando atribuir una mejora a evaluar períodos distintos. No se añade isotonic sobre estas colas pequeñas ni una nueva búsqueda de calibradores.

## Validación y criterios

Datos Binance spot reales, SHA-256 verificados, 2.465 velas por activo de 2020-01-01 a 2026-09-30. Seis folds externos con límites **2022-01-01, 2022-07-01, 2023-01-01, 2023-07-01, 2024-01-01, 2025-01-01, 2026-10-01**. Mismos días y condiciones para todos los candidatos. Hay 1.734 días de ejecución y **1.668 referencias puntuables por activo**, excluyendo por calendario la cola común de entradas. Los 14 inputs son causales y el contrato activo no cambia.

Train/calibración se purgan por `target_time` real y se añade embargo de seis días. Para meta, tanto la propuesta OOS como la etiqueta y su calibración deben ser anteriores al fold. Los tests modifican etiquetas y features futuras para comprobar que el ajuste anterior no cambie, alteran columnas de resultados para verificar que el motor las ignore y prueban gaps, empates, censura, solapamiento y equivalencia con el baseline.

Se reportan accuracy, balanced accuracy, AUC, Brier, log-loss y bins de fiabilidad **dentro de cada objetivo**. Los baselines de probabilidad son prevalencia de train; también hay logística de dirección y efectivo como controles. Las estrategias originales operan a 0,50/0,55/0,60/0,65; las nuevas a 0,55/0,60 o mediante `EV>0`, usando payoffs netos de train:

`EV = p * mean(net_return | target=1, train) + (1−p) * mean(net_return | target=0, train)`.

La fórmula no vuelve a restar los mismos costos. P(win)>0,5 no garantiza EV positivo cuando los payoffs son asimétricos, y EV estimado con el pasado tampoco garantiza rentabilidad futura.

Tres comparaciones principales se fijaron antes de entrenar: TB σ h5 calibrado con EV, meta-logística calibrada h1 y meta-logística calibrada TB h5, umbral 0,55. Se exige retorno neto y stress positivos, ≥30 operaciones, ≥4/6 folds positivos, ≥25% de la cantidad de trades del padre, mejora de precisión y media neta/trade, habilidad probabilística frente a prevalencia e intervalos de crecimiento favorables contra original/padre/efectivo. Debe sostenerse en ambos activos.

La inferencia económica usa bootstrap pareado no circular, 2.000 remuestras en bloques 14/30 días dentro del fold, intervalos conjuntos max-|t| sobre esos contrastes y alpha/2 por activo. Los intervalos se expresan en **diferencia de crecimiento logarítmico anualizado**, no en puntos de rentabilidad acumulada. El historial ya había sido observado en rondas anteriores: sigue siendo desarrollo retrospectivo, sin confirmación prospectiva nueva ni corrección de todas las búsquedas anteriores del proyecto.

## Costos y resultados económicos

Se usa 100% de capital fijo por posición, long/cash, sin leverage ni posiciones simultáneas. Comisión **10 bps/lado**, slippage **5 bps/lado** y spread completo **2 bps**, equivalente a 1 bp por lado. Se aplican las mismas condiciones al original; respecto de su configuración histórica de 15 bps/lado se incorpora este spread para todos. Aproximadamente 32 bps de ida y vuelta antes de movimiento de precios.

`annual_holding_bps` y tasa de efectivo son configurables y quedan en cero: spot sin financiación/deuda. No se inventan funding, gastos de infraestructura desconocidos o costos de préstamo. Stress duplica comisión, slippage y spread sin cambiar señales ni reentrenar. La curva bruta repite las decisiones netas; capital, PnL y costos se componen y escalan entre folds. DD y exposición se miden al cierre; no estiman drawdown ni horas efectivas intravela.

| Estrategia | Neto BTC | Trades BTC | PF BTC | Neto ETH | Trades ETH | PF ETH |
|---|---:|---:|---:|---:|---:|---:|
| Original, h1, p≥0,55 | −82,50% | 506 | 0,68 | −93,31% | 521 | 0,67 |
| Misma base, sólo cambiar salida a TB σ h5 | −68,06% | 347 | 0,78 | −94,14% | 400 | 0,71 |
| XGBoost TB σ h5, calibrado, p≥0,55 | −32,28% | 148 | 0,83 | −18,44% | 20 | 0,52 |
| XGBoost TB σ h5, calibrado, EV>0 | −80,36% | 530 | 0,82 | −89,32% | 471 | 0,80 |
| Meta-logística h1 calibrada, p_meta≥0,55 | −4,04% | 16 | 0,78 | −37,59% | 89 | 0,70 |
| Meta-logística TB h5 calibrada, p_meta≥0,55 | −8,58% | 11 | 0,64 | −34,72% | 63 | 0,64 |
| Efectivo | 0% | 0 | — | 0% | 0 | — |

Estos retornos corresponden al historial completo de esta ronda y a refits temporales. **No son el backtest estático de 2025–2026 publicado anteriormente**; no deben compararse directamente con aquel −45,32%. El control de salida permite separar el efecto del bracket del efecto de cambiar etiquetas, y el padre de cada meta-filtro comparte sus reglas de salida.

TB con EV sólo tiene 1/6 folds netos positivos BTC y 0/6 ETH. Los dos meta-filtros principales tienen 0/6 en ambos activos. El filtro meta h1 acepta apenas 3,16% de la cantidad de trades del padre BTC y 17,08% ETH; el meta TB, 3,17% y 15,75%. Perder menos con tan pocas operaciones no acredita una señal rentable.

Con fricción duplicada, los resultados son aún negativos:

| Principal | Neto BTC stress | Neto ETH stress |
|---|---:|---:|
| Original | −96,53% | −98,74% |
| TB σ h5 calibrado, EV>0 | −96,40% | −97,63% |
| Meta-logística h1 calibrada | −8,83% | −53,06% |
| Meta-logística TB h5 calibrada | −11,74% | −46,64% |

La sensibilidad al orden intravela importa: el meta TB BTC pasa de **−8,58% a +4,96%** al suponer TP primero, con las mismas señales. No se acepta esa rentabilidad favorable, porque el orden no está identificado. Para TB EV, incluso TP primero conserva pérdidas de −70,99% BTC / −83,38% ETH. La ambigüedad no explica por sí sola el resultado negativo de los principales.

![Curvas netas comunes](artifacts/label_analyses/20261015T170214_699ffc3e/net_equity.png)

## Fiabilidad de las probabilidades y efecto de la confianza

| Activo / modelo | Accuracy | Balanced accuracy | AUC | Log-loss | Loss prevalencia train |
|---|---:|---:|---:|---:|---:|
| BTC TB σ h5 raw | 55,70% | 48,35% | 0,4826 | 0,69520 | 0,67967 |
| BTC TB σ h5 calibrado | 54,68% | 50,25% | 0,5007 | 0,69217 | 0,67967 |
| ETH TB σ h5 raw | 55,10% | 48,52% | 0,4659 | 0,69759 | 0,67790 |
| ETH TB σ h5 calibrado | 52,76% | 50,38% | 0,5110 | 0,68667 | 0,67790 |

La calibración mejora loss/Brier de TB, pero sigue peor que prevalencia. Para meta h1, la calibración empeora la loss sobre filas comunes: BTC **0,68660 → 0,69488** (379 oportunidades), ETH **0,70059 → 0,70487** (521). No se presume que calibrar siempre beneficie. El meta TB raw ETH obtiene accuracy **61,61%** y balanced accuracy **49,49%**: es un ejemplo de accuracy aparente sin capacidad predictiva útil.

Subir el umbral original de 0,55 a 0,65 reduce trades **506→89 BTC / 521→91 ETH**, pero la media neta por operación empeora **−0,297%→−0,335% / −0,430%→−0,821%**. En TB h5 calibrado, p≥0,60 no produce ninguna oportunidad en ninguno de los activos. Efectivo y menos rotación no se confunden con mayor fiabilidad.

Las curvas de calibración son descriptivas, con bins fijos. En meta BTC raw hay 506 oportunidades y calibradas 379 por la abstención del último fold; las comparaciones numéricas de calibración usan explícitamente la intersección. Oportunidades superpuestas y operaciones ejecutadas son poblaciones diferentes, ambas guardadas.

![Calibración fuera de muestra](artifacts/label_analyses/20261015T170214_699ffc3e/calibration.png)

## Resultados positivos aislados que tampoco se conservan

- Horizonte fijo neto h10 calibrado p≥0,55: **+90,59% BTC / −68,87% ETH**, 112/129 trades. No generaliza entre activos.
- Meta-logística h1 raw p≥0,55: **+13,32% BTC con 22 trades / −37,62% ETH con 131**. El positivo BTC tampoco alcanza el mínimo de 30 trades.
- La única variante con neto positivo en ambos activos es supervisión **al cierre** h5 calibrada p≥0,60: **+12,22% BTC / +2,27% ETH**, 84/33 trades. Gana en sólo **1/6 folds en cada activo**, y con costos duplicados pasa a **−14,23% / −7,98%**. Es exploratoria, distinta de las tres comparaciones principales, y no se selecciona retrospectivamente como nuevo ganador.

Ninguna de las 83 variantes conserva rentabilidad positiva en ambos activos bajo stress. Los resultados negativos, ceros y positivos parciales quedan en el mismo informe; no se elimina evidencia desfavorable.

## Verificación, artefactos y reproducción

Implementación: `src/trading_ml/label_research/`; protocolo fijo: [config/label_study.toml](config/label_study.toml). El entrenamiento, los targets y la ejecución de producción conservan su comportamiento. La CLI incorpora el comando de investigación independiente `label-study`.

```powershell
.\.venv\Scripts\python.exe -m trading_ml label-study --protocol config/label_study.toml
# Verificación y gráficos de esta corrida, sin entrenar:
.\.venv\Scripts\python.exe scripts/record_label_study.py artifacts/label_studies/20261015T170214_699ffc3e --output artifacts/label_analyses/revision_nueva
```

El comando de entrenamiento requiere los CSV y ZIP BTC/ETH originales con sus checksums, crea una carpeta nueva y no sobrescribe experimentos. Los gráficos usan Matplotlib, ya utilizado por informes anteriores; el experimento sólo utiliza dependencias existentes del motor.

La verificación reconstruyó features y etiquetas desde los archivos verificados, validó los contratos de **274 modelos**, reprodujo **468 series externas**, incluidas sus abstenciones declaradas, y **4.198 pronósticos OOF**, y cotejó **80.302 retornos de operaciones** con su etiqueta financiera y regla de salida, incluyendo stress y sensibilidad. Comprobó **9.141 artefactos**, sin nuevos ajustes. Las etiquetas censuradas permanecen vacías; el verificador compara correctamente sus fechas `NaT` sin convertirlas en eventos observados.

Informe completo y todas las variantes: [report.md](artifacts/label_studies/20261015T170214_699ffc3e/report.md). Métricas, intervalos conjuntos y razones de rechazo: [summary.json](artifacts/label_studies/20261015T170214_699ffc3e/summary.json). Auditoría: [verification.json](artifacts/label_analyses/20261015T170214_699ffc3e/verification.json).

`plan.json` registra protocolo, entorno y hashes antes del primer ajuste. Los CSV `causal_features`, `labels_*`, `base_oof` y `meta_pool_*` distinguen inputs, futuros de investigación y ejemplos meta. Se guardan modelos/calibraciones, train/calibration hashes y fechas, forecasts, libros de trades, decisiones causales y curvas. `artifacts.json` verifica el conjunto de evidencia. SHA-256 de ese manifiesto: `13e9c6240ee9d46b0eae111944260b2c09e5d2a28facf840c9423e8098f157b0`; plan: `b5a77a41eff8bfd98b6524c81b41e3307e3a4bfc6bf7f672484acaedd2efe432`.

Modelos originales, punteros y reservas prospectivas existentes se preservaron por hash. No se abrió un test futuro, no se activó ningún candidato y no se enviaron operaciones reales. Lo que queda implementado es un laboratorio reproducible para contrastar mejores objetivos, sin presentar una mejora económica no demostrada.

Validación final: **211 tests aprobados**, incluidos 22 nuevos de etiquetas, gaps, empates, calibración temporal, propuestas OOF, costos, composición entre folds y censura. Ruff check, formato de los 135 archivos Python y `pip check` también pasaron. Comandos, salidas, entorno y hashes: [checks.json](artifacts/label_checks/20261015T170214_699ffc3e/checks.json).
