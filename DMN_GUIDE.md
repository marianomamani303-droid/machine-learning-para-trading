# Uso del sistema de investigación DMN

## Instalación reproducible

Entorno de esta investigación: Windows, Python 3.14.3, PyTorch 2.14.1+cpu. El entorno original y `requirements-lock.txt` se conservan. `requirements-dmn-lock.txt` captura el entorno ampliado, sin rutas editables locales.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dmn-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

En otro entorno compatible, el extra opcional instala versiones admitidas: `python -m pip install -e '.[dmn,dev]'`. Los comandos anteriores deben ejecutarse desde la raíz del repositorio. El lock fija versiones, pero no promete igualdad bit a bit entre sistemas/hardware.

## Ejecutar, continuar y verificar

```powershell
# Usa los históricos locales con sus manifiestos SHA256.
.\.venv\Scripts\python.exe -m trading_ml.dmn study --protocol config/dmn.toml

# Obtiene fuentes faltantes mediante los archivos públicos mensuales de Binance.
.\.venv\Scripts\python.exe -m trading_ml.dmn study --protocol config/dmn.toml --download

# Continuar una ejecución interrumpida; sustituir <run_id> por el ID impreso al comenzar.
.\.venv\Scripts\python.exe -m trading_ml.dmn study --protocol config/dmn.toml --resume artifacts/dmn_studies/<run_id>

# Comprueba hashes, preservación de modelos/reservas y reconstruye la contabilidad.
.\.venv\Scripts\python.exe -m trading_ml.dmn verify --study artifacts/dmn_studies/<run_id>

# Genera gráficos exportables en otro directorio, sin entrenar ni seleccionar modelos.
.\.venv\Scripts\python.exe scripts/plot_dmn_study.py --study artifacts/dmn_studies/<run_id>

.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

También está disponible `trading-dmn` al instalar el paquete. La CLI histórica `trading-ml` conserva sus funciones. Los logs se escriben en stderr y la respuesta JSON en stdout. Una ejecución nueva siempre crea un ID nuevo; para recuperar evidencia existente debe usarse `--resume`.

Reanudar exige protocolo, hash del código Python y versiones de entorno idénticos. No editar `src` mientras se ejecuta un estudio. El directorio guarda `source.zip`, el protocolo, snapshots propios y hashes: si otra sesión cambió el código, recuperar esa versión en un checkout separado para continuar. No sobrescribir el árbol con cambios del usuario. Los ajustes completados se cargan desde `completed.json`; un ajuste incompleto se archiva con sufijo `_interrupted_...` antes del reintento. Los registros `attempts/*_started.json` y `*_finished.json` permiten distinguir intentos interrumpidos de ajustes terminados.

## Contrato matemático y temporal implementado

Para una vela que comienza en el día `t`, todas las entradas se conocen al cierre, con referencia `t+1`. La señal usa exclusivamente las últimas 63 observaciones hasta ese cierre. La apertura de ejecución es `open[t+1]`, la de salida/marcación es `open[t+2]`. La etiqueta económica es:

```text
u[t+1] = open[t+2] / open[t+1] - 1
sigma[t] = EWM_std(close.pct_change(), span=60, adjust=False)
feature_return[h,t] = (close[t]/close[t-h] - 1) / (sigma[t] * sqrt(h))
q[S,L,t] = (EWM_alpha(1/S)(close)[t] - EWM_alpha(1/L)(close)[t]) / rolling_std_63(close)[t]
feature_macd[S,L,t] = q[S,L,t] / rolling_std_252(q[S,L])[t]
```

Los horizontes son 1/21/63/126/252 y los pares MACD (8,24)/(16,48)/(32,96). Son ocho entradas. `asset_return`, fechas futuras y etiquetas nunca entran al tensor de features. Un calentamiento insuficiente o un hueco posterior causa error, sin interpolación ni sustitución por precios ficticios.

El escalador ajusta media/desviación exclusivamente sobre ventanas de entrenamiento, agrupando activos, y limita a ±5 desviaciones. Las ventanas solapadas ponderan más las observaciones interiores al ajustar el escalador; esto se conserva como parte del contrato reproducible. Lineal y MLP usan el vector más reciente; la LSTM usa las 63 observaciones. Este control lineal/MLP simplifica la concatenación temporal del paper.

```text
direct_long_short = tanh(network(features))
direct_long_only = (tanh(network(features)) + 1) / 2
w[asset,t] = clip(signal[asset,t] * 0.15 / (sigma[asset,t]*sqrt(365)), -1, 1) / N
```

El modo long/cash limita a cero las señales negativas de los controles tradicionales. Las salidas directas de redes se transforman suavemente a [0,1] para conservar gradientes. La exposición bruta objetivo total no supera 1. El objetivo de volatilidad es individual y ex ante: la volatilidad realizada de cartera puede ser menor; no se iguala usando el período evaluado. Cash rinde cero. Los controles buy/hold a pesos iguales se rebalancean diariamente y pagan costos.

La pérdida Sharpe agrupa retornos individuales por activo/fecha y usa desviación poblacional con estabilizador `epsilon=1e-8`. La variante neta resta cambios absolutos de exposición y préstamo sintético. Los minibatches conservan fechas contiguas; cada bloque incluye su predecesor real para calcular turnover, y los activos permanecen separados. Se mezclan bloques completos, sin calcular diferencias entre activos ni entre fechas aleatorias. Esa pérdida aproxima el turnover del paper; la contabilidad final utiliza la deriva real de posiciones y capital.

El entrenador aplica Adam, clipping de gradiente, dropout de entrada/salida y early stopping según Sharpe neto de cartera en validación interior. No hay dropout recurrente variacional. MSE usa retornos ejecutables normalizados por volatilidad y toma el signo como posición; el control XGBoost clasifica esos mismos retornos y selecciona árboles por log-loss temporal. Los distintos criterios de selección se registran explícitamente.

La LSTM se reinicia en cada ventana. Puede consultar contexto histórico anterior al inicio del fold porque estaba disponible entonces; no arrastra estado futuro ni aprende con las etiquetas de ese contexto fuera de train. Los splits exigen `target_time < siguiente_frente - embargo`, con embargo de un día. El extremo exterior es exclusivo. Los días de frontera purgados quedan en cash en la curva agregada y cuentan para la anualización.

## Contabilidad y alcance económico

El simulador conserva holdings, deja que sus valores deriven con los retornos y rebalancea hacia los pesos objetivo. Si `h` es el vector de holdings dividido por el patrimonio anterior, `w` el peso objetivo después de comisiones y `c` el costo proporcional por notional, resuelve:

```text
q + c * sum(abs(w*q - h)) = 1
patrimonio_tras_rebalanceo = q * patrimonio_anterior
patrimonio_fin_día = patrimonio_tras_rebalanceo * (1 + dot(w, asset_returns) - funding)
```

Cada fold comienza en efectivo y termina con liquidación costeada. El costo base supuesto es 10 bps de comisión + 5 de slippage + 1 de medio spread por lado. La sensibilidad duplica esos costos y financiación. Son supuestos de investigación, no cotizaciones garantizadas. Se contabilizan también costos con señales constantes cuando la deriva obliga a rebalancear.

El modo sintético long/short admite exposiciones negativas con costo anual de préstamo; no representa por sí solo disponibilidad de préstamo spot ni un contrato perpetuo. No incluye liquidaciones de margen, funding observado de perpetuos, impuestos, lotes, profundidad, impacto dependiente del tamaño ni spreads históricos. El sistema no tiene ejecución de órdenes. Para estudiar futuros continuos se necesita un proveedor con rolls/ajustes verificables y su calendario; no sirve renombrar las series de cripto.

## Artefactos y significado de los resultados

```text
artifacts/dmn_studies/<id>/
  plan.json, source.zip, inputs.json, sources.json
  BTCUSDT_raw.csv, ETHUSDT_raw.csv
  fold_00/...fold_04/
    attempts/                         intentos de ajustes, incluso interrumpidos
    <receta>__seed_<semilla>/          pesos, scaler, historia, metadatos, señales y hashes
    <estrategia>_positions.csv         pesos, retornos, costos y equity diaria
    report.json                       splits, métricas brutas/netas/dobles y por activo
  <estrategia>_equity.csv              curva concatenada con cash en fronteras
  summary.json, REPORT.md, artifacts.json
artifacts/dmn_analyses/<id>/            gráficos PNG/SVG con manifiesto
```

Las filas de receta promedian las posiciones de las dos semillas antes de aplicar el riesgo. También se publican todas las semillas por separado; no se elige una por sus métricas exteriores. El bootstrap pareado remuestrea bloques de fechas dentro del fold y mantiene juntos ambos activos. Los intervalos son descriptivos, marginales y condicionados a estos históricos; no corrigen la selección de modelos ni la multiplicidad.

Una mejora de Sharpe de desarrollo, incluso con un intervalo favorable, no confirma una ventaja independiente. Este universo ya fue observado en estudios anteriores y tiene poca diversificación. El código protege las reservas previas mediante las reglas existentes y restringe el desarrollo al 2026-10-01. Cambiar ese límite requiere un protocolo nuevo que respete los períodos reservados.

Para decidir una confirmación futura: fijar hipótesis/costos/universo antes de abrir datos nuevos, congelar modelo y criterios de aceptación, reservar una ventana completamente disjunta de las reservas existentes, y publicar todos los resultados. Las reservas previas se extienden hasta 2027-10-13; no utilizar sus observaciones como validación DMN. Ver `CODEX_PROGRESS.md` para el punto exacto de continuación.
