# Protocolo previo: misma señal, reglas frente a DQN

Registrado el 2026-10-27 (America/Buenos_Aires), antes de entrenar o evaluar
el DQN. Configuración ejecutable: `config/rl_signal.toml`. Una corrida nueva
copia ambos archivos y sus hashes antes del primer ajuste. No se amplía el
presupuesto por malos resultados. No se buscan otras redes, mercados o señales.

## Pregunta y evidencia anterior

¿Una política DQN usa mejor la misma probabilidad después de fricciones plausibles?
Cambiar la política NO cambia AUC, log-loss ni Brier del predictor compartido.
No se presupone ventaja económica. Se leyeron código e informes
`PREDICTIVE_FAILURE_AUDIT.md`, `DMN_RESULTS.md`,
`FOCUSED_FINANCIAL_FEATURES_REPORT.md`, `EDGE_RESEARCH_RESULTS.md` y
`CODEX_PROGRESS.md`. La auditoría previa identifica sobreajuste, costos y
reutilización histórica. DMN completó 120 ajustes sin ventaja robusta; edge
completó 120 y aceptó 0/16 candidatos.

Antes de esta implementación se verificaron los 8 archivos del manifiesto
financiero `20261017T232854_36d523ee`. Sus 876 predicciones de BTC logística
top_core_3 reproducen AUC 0.54866310, log-loss 0.69915901, Brier 0.25260769.
El ledger publicado muestra +45.2857% base y +28.2398% doble con fixed_055,
39 operaciones de eventos; NO demuestra superioridad estadística independiente.
No es comparable directamente con exposición máxima 50%, mantenimiento de
unidades y nueva gestión de riesgo de este protocolo.

Discrepancias documentales: los artefactos y BOUNDS del código usan folds
2023/2024/primer semestre 2025, no noviembre/septiembre como dice parte del
informe. El motor anterior usa `spread_bps/2`: 10+5+2/2=16 bps por lado,
aproximadamente 32 ida y vuelta; las menciones a 34 bps y a costos reales en
ese informe no se toman como datos de fills. No se modifica el estudio previo.

## Señal congelada y calendario

BTC/USDT spot diario UTC, snapshot oficial existente desde 2021-01-01,
fin exclusivo 2025-07-01. Se verifica SHA256 y continuidad; sin interpolar.
14 variables originales de `feature_columns(Features())`, más exactamente
`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`,
`fin_vol_parkinson_ratio`: 17 columnas. Fórmulas idénticas al estudio financiero,
sin exigir variables macro ajenas a esta receta para elegir filas.
StandardScaler(train), logística L2, C=1, lbfgs, max_iter=2000, seed=42,
sin calibración ni selección de C. Objetivo original: close siguiente > close
actual; etiqueta disponible al siguiente cierre. No es retorno neto ejecutable.
No había pesos serializados de esta receta en el estudio focalizado (el runner
descarta el estimador); se reconstruye la receta, no se inventa un modelo guardado.

El predictor se ajusta una vez al comienzo de cada bloque temporal y se congela
dentro del bloque. Bloques internos consecutivos de 90 días desde 2022-01-01
producen TODAS las señales de entrenamiento RL con walk-forward expansivo;
train supervisado >=180 filas, etiquetas estrictamente anteriores al inicio
menos 6 días. Una receta con un único predictor entrenado con todo el train RL
para puntuar retroactivamente su historia queda expresamente prohibida.
Validación: un bloque separado de 120 días antes de cada inicio exterior.
Su predictor sólo usa etiquetas anteriores al inicio de validación menos 6 días.
Exterior: tres bloques [2023-01-01,2024-01-01), [2024-01-01,2025-01-01),
[2025-01-01,2025-07-01); un predictor congelado por bloque, ajustado con etiquetas
anteriores al inicio menos 6 días. Reentrenamiento supervisado programado al
inicio exterior, incluyendo el pasado ya disponible de validación, es parte de
la receta; el DQN NO se reajusta con validación. Los distintos largos de bloque
interior/exterior son una limitación de distribución, no una búsqueda.

Train RL termina con salida ejecutable < inicio validación -6 días.
Validación termina con salida ejecutable < inicio exterior -6 días.
Test termina con salida ejecutable < fin exterior. Se purga por máximo entre
disponibilidad de etiqueta y salida con demora; embargo adicional conservador
de seis días heredado del estudio financiero (no se afirma independencia por
ese embargo). Cada episodio empieza cash y liquida al final con fricción;
residuos no vendibles permanecen valorados y se reportan.

## Información, ejecución y riesgo comunes

Estado de cinco componentes: p(up), volatilidad log causal rolling20, costo
estimado por lado, exposición marcada al último cierre disponible y edad de
posición. StandardScaler de los tres primeros sólo con train RL, conservado
durante validación y test; exposición /0.5 y edad truncada /365 son escalas
fijas. No se incluyen features originales, retornos futuros ni precios futuros
en el estado. Reglas y DQN reciben exactamente esas mismas observaciones;
las reglas pueden ignorar componentes. Acciones objetivo 0, .25, .50, sin deuda.
Máscara común: w*sigma*sqrt(365)<=20%, y w<=.50. El long escalado elige el mayor
escalón permitido. No se iguala volatilidad realizada usando futuro del test.
La deriva entre decisiones puede superar el peso objetivo; se registra.

Señal de vela cerrada t disponible en t+1 UTC00. Precio base: apertura de la
primera vela con timestamp >= disponibilidad de señal (t+1); no se opera al
cierre usado para calcularla. Simultaneidad límite presupone cómputo/latencia
cero; es una aproximación optimista OHLC. Estrés demora: acción decidida con
el estado de t+1 se encola y ejecuta recién a apertura t+2, sin actualizarla
con información posterior. Primera fecha cash, último encargo sin fill si
no cabe en el fold; misma ventana y restricciones para todas las políticas.

Contabilidad persistente de unidades y cash. Se negocia sólo la diferencia
respecto al objetivo; una banda común de rebalanceo de 2.5 puntos porcentuales
evita microajustes, pero nunca impide reducir un exceso del límite causal.
No se cierra/reabre al repetir acción. El objetivo se calcula sobre equity
después de costos. Compra redondeada hacia abajo; venta nunca mayor a unidades.
Precio adverso de referencia: apertura*(1 +/- (halfspread+slippage)); comisión
10 bps sobre notional ejecutado. Halfspread=2 y slippage=5 bps: ~17 bps/lado.
Costos dobles duplican los tres componentes; demora conserva costos base.
Mínimo notional 10 USDT, cantidad mínima y paso 0.00001 BTC son SUPUESTOS
conservadores, no reglas históricas certificadas. No se encontraron filtros
históricos por día en estos snapshots. No se confunden filtros vigentes con
filtros de 2021–2025. Sin profundidad, impacto, quotes ni fills verificables;
capital inicial 10000 USDT, cash sin interés, sin funding spot. El costo de
spread/slippage se contabiliza además de comisiones, sólo por unidades negociadas.

Recompensa = log(E_next/E_before) - 0.5*(w_post*sigma_causal)^2;
incluye fricción de liquidación terminal. DQN recibe 100*recompensa por escala
numérica; la métrica económica usa retorno log neto sin penalización.

## Entrenamiento y controles fijados

Tres semillas 42,1729,2027; una MLP 5→32 ReLU→32 ReLU→3. DQN con red target,
Huber, Adam .001, gamma .99, batch64, buffer local 10000, warmup256,
actualización cada4 pasos, target cada250, clip gradiente10; epsilon lineal
1→.05 hasta paso4000. Exactamente6000 pasos/fold/seed, máximo54000 pasos,
9 ajustes DQN. Repetir episodios sobre la misma historia NO agrega evidencia.
Buffer, optimizador y redes nuevos en cada fold/semilla; nada de validación
o test entra al replay. Seis checkpoints (1000..6000), selección por suma de
recompensa en validación base; empate favorece el más temprano. No se elige
semilla con test, no se entrena bajo escenarios de estrés. Se conservan todos
los checkpoints, normalizadores, modelos supervisados y logs de entrenamiento.

Controles: fixed compra el mayor escalón permitido si p>=.55, en otro caso cash.
Band entra igual a .55, sale con p<.50 y mantiene su objetivo anterior en medio,
recortado por la misma máscara causal. Long escalado no usa p y toma el máximo
permitido; cash cero. Todos tienen idéntica banda de rebalanceo, capital, precios,
costos, lotes, liquidación y límites de riesgo. No hay tuning de umbrales.

## Evaluación y aceptación

Primaria: suma del diferencial diario log neto DQN menos band; agregado central
= media de retornos log diarios de TODAS las semillas (no un ensemble negociado,
no tres muestras temporales independientes). Se informa cada semilla/fold además.
Secundarias: retorno acumulado, Sharpe aritmético 365d, DD incluyendo capital
inicial, exposición media/máxima, turnover notional/equity previo, costos USDT,
órdenes ejecutadas, entradas/salidas completas, residuos; AUC/loss/Brier comunes.

Bootstrap pareado de bloques móviles no circulares DENTRO de fold, 4000 réplicas,
seed99173, bloques7 y30. Mismas fechas y mismos índices para políticas y semillas;
no remuestrear semillas como fechas nuevas. Diferencia de sumas log. P unilateral
con bootstrap centrado bajo H0 y corrección finita (1+k)/(B+1).
Familia predeclarada: 4 referencias (band/fixed/long_scaled/cash) ×3 escenarios
×2 bloques=24 contrastes, Holm global al5%; IC95% marginales y simultáneos
Bonferroni para24, sin presentarlos como corrección de selección histórica.
IC por semilla vs band descriptivos, no decisiones adicionales.

Aceptación dentro de este protocolo requiere ganancia primaria >=.02 log,
ambos IC simultáneos inferiores>0 y Holm<.05 vs band en base; ganancia positiva
en TODAS las semillas agregadas, >=2/3 folds por semilla y al menos2/3 semillas
positivas en cada fold; ganancia positiva y contrastes vs band favorables bajo
ambos estreses; no empeorar DD base más de5pp vs band. Se requiere además
retorno neto DQN positivo y superar cash con inferencia ajustada en base.
No se acepta por un fold/semilla, retorno bruto o quedarse cash. Si la estimación
primaria<=0 o falla estabilidad: «sin mejora dentro del protocolo»; si es positiva
pero no supera incertidumbre: «inconcluso»; sólo todos los criterios permiten
«mejora demostrada dentro del protocolo exploratorio», nunca confirmación nueva.

## Historia reutilizada, reservas y futuro

2020–septiembre2026 ya fue inspeccionado por DMN y otras rondas; 2023–junio2025
se usó en financiero/edge. Un archivo sellado no vuelve desconocida esa historia.
Este experimento sólo usa hasta junio2025 y etiqueta todos sus resultados como
exploratorios. No abre datos>=julio2025, ni reservas de SMC, momentum, mean
reversion o frontier. Se auditan metadatos de reservas y hashes de artefactos
anteriores, sin evaluar sus outcomes. Los estudios futuros existentes llegan
al menos hasta2027-10-13. Seguimiento propuesto: [2027-11-01,2028-11-01),
única evaluación desde2028-11-02, sin extender ante resultados desfavorables.
Si aparecen reservas posteriores incompatibles, bloquear antes del inicio;
no desplazar fechas ni usar ventanas retrospectivas sin otro registro previo.

Se fijan las tres políticas DQN del último fold, cada una con su checkpoint
elegido internamente, y el predictor de ese fold, sin elegir ganador con test.
Seguimiento sin reentrenamiento ni actualización de scaler: mismas reglas,
riesgo y escenarios. Antes de comenzar registrar hashes de todos los modelos,
datos, código, protocolo y costos. Registrar append-only cada predicción,
hora UTC de emisión real, cierre fuente, disponibilidad, pesos/estado/acción,
hash del registro anterior; emisión anterior al precio ejecutable, sin backfill.
Con datos en vivo, registrar la primera cotización posterior a emisión y latencia,
sin fingir que una apertura ya ocurrida es ejecutable. Congelar feed de quotes
y reglas operativas en una enmienda previa al inicio; si faltan, bloquear
evaluación económica prospectiva, conservando predicciones para diagnóstico.
Al cierre comparar todas las semillas, los cuatro controles y los24 contrastes;
>=30 round trips por semilla como piso operativo, sin afirmación de potencia.
Falta de cobertura/registro previo o muestra suficiente => inconcluso.
No hay órdenes reales, activación de dashboard ni resultados futuros inventados.

Referencias primarias consultadas: [DQN PyTorch](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
y [filtros Spot Binance](https://github.com/binance/binance-spot-api-docs/blob/master/filters.md).
La documentación identifica filtros de lotes/notional; no aporta una historia
diaria de esos valores ni demuestra que los costos supuestos sean fills reales.
