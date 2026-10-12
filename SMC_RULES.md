# Definiciones operativas SMC v1

Estas son reglas reproducibles elegidas para este motor, no una definición universal ni una identificación de órdenes institucionales. OHLCV no permite verificar quién operó. Toda señal aparece **al cierre de su vela de confirmación**, sin modificar filas anteriores.

Notación: O/H/L/C/V son apertura, máximo, mínimo, cierre y volumen. t es la vela cerrada actual. Los parámetros de detección se guardan en el plan y contrato. Defaults: left=3, right=2, liquidez=20, OB=10, ATR=14, edad máxima=50 y hasta 20 zonas por tipo/dirección.

## Swings, estructura y tendencia

Un swing high en j exige `H[j] > max(H[j-left:j])` y `H[j] > max(H[j+1:j+right+1])`; low usa mínimos y desigualdad estricta inversa. Se confirma sólo en `t=j+right`. Los empates no forman pivots. No se escribe señal en j ni se aplica desplazamiento gráfico al pasado.

HH/LH comparan los dos últimos swing highs **ya confirmados**; HL/LL comparan los dos lows. `structure=+1` para HH y HL, `-1` para LH y LL; en mezclas o falta de pares, 0. La tendencia inicia en 0; se inicializa con estructura no ambigua y luego cambia al confirmar rupturas direccionales. Una tendencia persistente describe la última dirección estructural, no garantiza la dirección de la próxima vela. Se devuelve la estructura mixta/no confirmada por separado.

Orden dentro de t: confirmar los swings disponibles, calcular estructura, inicializar tendencia sólo si aún es 0, y después evaluar ruptura. La dirección previa a la ruptura se refiere a ese estado disponible al cierre; una inicialización con los swings recién confirmados puede ocurrir en el mismo t. No se utilizan pivots todavía sin confirmar.

## BOS y CHoCH

Ruptura alcista del último swing high s: `C[t-1] <= H[s] < C[t]`. Ruptura bajista del último swing low q: `C[t] < L[q] <= C[t-1]`. La vela debe cerrar más allá del nivel; una mecha no alcanza.

Si la ruptura coincide con la tendencia disponible antes de evaluar la ruptura, o todavía no había dirección definida, se llama BOS. Si rompe en dirección contraria a esa tendencia definida, se llama CHoCH. Cada nivel confirmado sólo puede producir una ruptura; volver a cruzarlo no crea eventos repetidos. Un nuevo pivot crea un nivel distinto.

## Liquidity sweeps

Con `U=max(H[t-N:t])` y `D=min(L[t-N:t])`, excluyendo la vela actual:

- Barrido de máximos (evidencia bajista): `H[t] > U` y `C[t] < U`.
- Barrido de mínimos (evidencia alcista): `L[t] < D` y `C[t] > D`.

Un cierre que continúa más allá del extremo no se llama sweep. Una vela puede barrer ambos extremos; las reglas direccionales se abstienen ante esa contradicción. Son proxies de rechazo de precio, no una medición de liquidez del libro de órdenes.

## FVG

`TR[t]=max(H[t]-L[t], abs(H[t]-C[t-1]), abs(L[t]-C[t-1]))`. El ATR de referencia en t es la media de TR de las **14 velas previas**; no usa la vela actual. No se generan FVG/OB antes de disponer de ese ATR.

FVG alcista: `L[t]-H[t-2] > 0.1*ATR_past[t]`, vela central alcista (`C[t-1]>O[t-1]`) y cuerpo central `>=0.5*ATR_past[t]`. Zona `[H[t-2], L[t]]`. Bajista: `L[t-2]-H[t] > 0.1*ATR_past[t]`, cuerpo central bajista con la misma magnitud; zona `[H[t], L[t-2]]`.

La zona nace en t; la aparición de tres velas no certifica un desequilibrio institucional. Un retorno parcial posterior al rango de la zona produce touch. FVG alcista se elimina si un low alcanza o cruza el borde inferior; bajista, si un high alcanza o cruza el superior. Es una regla conservadora de cobertura completa de la zona con datos agregados, no un modelo intravela.

## Order blocks

Se requiere BOS/CHoCH y una vela de ruptura con cuerpo en la misma dirección `>=0.5*ATR_past[t]`. Buscar retrospectivamente la última vela de cuerpo opuesto entre `t-10` y `t-1`; su rango completo `[L[j], H[j]]` define el OB. El OB **nace en t**, no en j. Se conservan por separado la vela fuente y el momento de disponibilidad.

OB alcista se invalida con `C[t]<lower`; bajista con `C[t]>upper`. No se crea una zona ya inválida al nacer. El solapamiento de una vela posterior con una zona vigente produce touch; el nacimiento nunca cuenta también como retest. Los patrones se derivan de precio y no prueban acumulación o distribución institucional.

## Estado de zonas y variables

Zonas expiradas cuando `t-born > 50`; las restantes se limitan a las 20 más recientes por tipo/dirección. La actualización de invalidación/touch se hace antes de crear las zonas de t. No se reescribe el histórico al saber que una zona fue mitigada después.

Se agregan 32 features numéricas: disponibilidad, tendencia/estructura, HH/HL/LH/LL, confirmaciones de pivots, BOS/CHoCH, sweeps, nacimientos FVG/OB, cantidades de zonas vigentes, touches, presencia y distancia a swings y posición en rango estructural. Distancias usan ATR pasado y clipping [-10,10]; sin nivel disponible se utiliza 0 más un flag de ausencia. Posición estructural = `clip((C-low_swing)/(high_swing-low_swing),0,1)` si el rango es válido; en otro caso 0.5. No hay escala ajustada con el dataset completo.

## Reglas sin ML y estado neutral

Sumar evidencia **actual** por lado: BOS y CHoCH pesan 2, sweep 1.5, nacimiento FVG/OB 1 y touch FVG/OB 1. Añadir 0.5 por dirección de tendencia. Sólo tendencia, falta de calentamiento, conflicto entre lados o magnitud final <1 produce neutral y score probabilístico 0.5. En caso elegible: `p_up=0.5+0.1*clip(score,-3,3)`; `p_down=1-p_up`.

Este número es **confianza heurística no calibrada**, no una frecuencia empírica aprendida. Se evalúa con ROC-AUC/log-loss/Brier como score probabilístico declarado, sin presentarlo como una probabilidad calibrada.

Para XGBoost, p es la salida del clasificador o de su calibrador guardado. Estado alcista si `p>=0.55`, bajista si `p<=0.45` y neutral entre ambos (se tolera un ULP en el extremo inferior). Sin calentamiento suficiente, neutral. Neutral no es una tercera etiqueta: el modelo sigue estimando subida frente a bajada **o precio sin cambio**, y el estado representa abstención.

La calibración sigmoide usa una cola de train disjunta y purgada. Ajusta regresión logística regularizada sobre logit(p), con C=1 prefijado. Si el coeficiente sería negativo se conserva una probabilidad constante igual a la prevalencia de la cola, en vez de invertir el ranking. Se informa ese colapso explícitamente. No se reentrena XGBoost después de calibrar; su calibrador permanece asociado al mismo modelo. Un calibrador ajustado no garantiza calibración futura bajo drift.
