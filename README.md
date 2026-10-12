# Motor de Machine Learning para trading

Motor Python independiente para investigación: descarga OHLCV reales, valida datos, construye indicadores causales, entrena XGBoost, evalúa fuera de muestra y simula una estrategia con costos. No requiere frontend, backend propio ni credenciales. No coloca órdenes.

Se ejecutó con BTCUSDT diario, datos de 2020-01-01 a 2026-09-30. El primer modelo obtuvo ROC-AUC 0.4787 y el backtest perdió 45.32% después de costos. Estos resultados no justifican usar esta estrategia para operar. Ver [EXPERIMENTS.md](EXPERIMENTS.md) para la evidencia completa.

## Instalación

Entorno verificado: Windows 11, Python 3.14.3. `.venv` ya está creada en esta carpeta y el paquete está instalado. Desde PowerShell, no hace falta activar el entorno:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

`requirements.txt` fija dependencias directas; `requirements-lock.txt` fija el entorno completo, incluidas herramientas de desarrollo. Para otro sistema o Python compatible (>=3.11), usar `python -m pip install -e '.[dev]'` en su entorno virtual: el resolver elegirá versiones compatibles dentro de los rangos de `pyproject.toml`. La reproducibilidad exacta se verificó en el entorno fijado, no entre plataformas.

## Configuración

Editar [config/default.toml](config/default.toml). Los comandos admiten `--config ruta.toml` **antes** del subcomando. `paths.root` se resuelve desde el directorio actual; ejecutar los ejemplos desde la raíz del proyecto.

| Sección | Configuración |
| --- | --- |
| market | Símbolo spot Binance, intervalo, start inclusivo y end exclusivo en UTC |
| features | RSI, MACD, ventanas SMA y volatilidad |
| training | Horizonte en velas, proporciones temporales, mínimo de filas y semilla |
| model | Hiperparámetros XGBoost |
| search | Grilla de profundidad y learning rate; selección por log-loss de validación |
| backtest | Umbral long, comisión y slippage en puntos básicos **por lado**, capital |
| paths | Directorio de artefactos |

Se admiten `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`. Cripto se eligió por su calendario continuo y OHLCV público sin claves. El proveedor usa [archivos públicos oficiales de Binance](https://github.com/binance/binance-public-data): ZIP mensual y checksum SHA-256. Las fechas desde 2025 usan microsegundos; el lector soporta también milisegundos.

La descarga cubre meses cerrados. `end` no puede superar el primer día del mes actual UTC; la publicación mensual puede demorarse hasta el primer lunes. Un archivo faltante provoca un error explícito: no se reemplaza por datos ficticios. La configuración inicial termina en `2026-10-01`; una predicción de ese archivo es histórica y avisa su antigüedad. Para datos dentro del mes corriente se necesita agregar un proveedor diario/incremental, pendiente en el roadmap. Con intervalos de minutos, la descarga y memoria requeridas aumentan mucho: comenzar con un rango corto.

## Comandos

```powershell
# 1. Descargar históricos reales (reutiliza caché verificada)
.\.venv\Scripts\python.exe -m trading_ml download
# --refresh vuelve a descargar ZIP y CHECKSUM; guardar los originales si se quiere reproducir una revisión anterior.

# 2. Limpiar, validar y construir features/etiquetas
.\.venv\Scripts\python.exe -m trading_ml process

# 3. Entrenar, seleccionar con validación y evaluar test
.\.venv\Scripts\python.exe -m trading_ml train

# 4. Cargar y evaluar sólo observaciones posteriores a entrenamiento/selección
.\.venv\Scripts\python.exe -m trading_ml evaluate --version latest --output artifacts/reports/evaluation.json

# 5. Predecir la última vela cerrada del dataset
.\.venv\Scripts\python.exe -m trading_ml predict --version latest --output artifacts/reports/prediction.json
# También acepta --reference 2026-10-01T00:00:00Z

# 6. Backtest long/cash; se guardan resumen, trades y equity
.\.venv\Scripts\python.exe -m trading_ml backtest --version latest

# 7. Reentrenar con un dataset ampliado, luego de extender market.end, descargar y procesar
.\.venv\Scripts\python.exe -m trading_ml retrain --version latest --data artifacts/data/prepared.csv
```

El último comando rechaza un dataset sin nuevas velas. Si el último mes disponible ya está descargado, no hay datos más nuevos que incorporar aún. Para verificar el mecanismo hoy con dos cortes históricos reales:

```powershell
.\.venv\Scripts\python.exe scripts/run_retraining_demo.py
$retrainingDemo = Get-Content artifacts/reports/retraining_demo.json -Raw -Encoding UTF8 | ConvertFrom-Json
.\.venv\Scripts\python.exe -m trading_ml compare --versions $retrainingDemo.parent_version $retrainingDemo.retrained_version --output artifacts/reports/comparison.json
```

La demostración usa un corte anterior a julio de 2026 y luego incorpora julio-septiembre. Crea nuevas versiones y actualiza `latest`; consultar `artifacts/reports/retraining_demo.json` para los IDs de cada ejecución. `compare` usa las mismas filas posteriores a la selección de **todos** los modelos. Comparar tests distintos produciría una comparación engañosa.

Pipeline completo, con un manifiesto JSON de resultados, entorno y procedencia:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

El stdout de la CLI es JSON y el logging va a stderr; código de salida 0 indica éxito y 1 un error esperado. Usar `--verbose` antes del comando para diagnóstico. Los comandos pueden recibir `--data`, `--input` o `--output` según su ayuda: `python -m trading_ml process --help`.

## Integración desde Python

```python
from pathlib import Path
from trading_ml.models import load_model
from trading_ml.prediction import predict
from trading_ml.evaluation import evaluate
from trading_ml.backtesting import backtest

bundle = load_model(Path("artifacts"), "latest")
dataset = Path("artifacts/data/prepared.csv")
prediction = predict(bundle, dataset)
metrics = evaluate(bundle, dataset)
simulation = backtest(bundle, dataset)
print(prediction["probability_up"])
```

Una predicción incluye activo, referencia UTC de cierre, intervalo, horizonte en velas/segundos, probabilidades, versión, estado y advertencias. `probability_down = 1 - probability_up` representa **bajada o precio sin cambio**. No estima magnitud del movimiento, ganancia neta ni probabilidad de una operación rentable. Las probabilidades se evalúan con Brier, log-loss y reliability bins. El pipeline básico conserva probabilidades sin calibración posterior; la etapa SMC incorpora una variante con calibración sigmoide temporal disjunta y declara `calibration_status` en la respuesta.

Los módulos usan `TradingMLError`, `DataError` y `LeakageError` para errores de dominio. El dataset CSV preparado y su `.csv.json` son un contrato de intercambio explícito. Contienen etiquetas futuras para investigación, pero la inferencia selecciona únicamente las columnas causales registradas (14 originales, 22 mejoradas o 46 en esta ronda SMC). Las últimas `horizon` filas no tienen etiqueta y sirven para predicción. Al integrar otro proveedor, normalizar OHLCV y crear un manifiesto de procedencia veraz compatible; nunca cambiar el hash para ocultar una alteración.

## Evaluación y backtesting

La partición es cronológica 60%/20%/20%, con purga de etiquetas que alcanzan la siguiente ventana. XGBoost entrena sólo con train. La validación selecciona parámetros; no se reajusta el modelo con validación ni test. Tanto predicción como evaluación y backtest rechazan referencias utilizadas para entrenar o seleccionar. Los cambios posteriores al test no cambian la selección: esto está cubierto por una prueba adversarial.

El backtest compra en la apertura posterior a la vela de señal si `P(up) >= 0.55`, mantiene `horizon` velas y vende en una apertura posterior. Opera long/cash, sin short ni posiciones superpuestas. Comisión 10 bps + slippage 5 bps por lado; buy and hold usa los mismos extremos y costos. La equity se valúa a cada apertura, no sólo cuando termina una operación; drawdown incluye el capital inicial. Las señales sin precio futuro de salida se excluyen y cuentan.

Esto supone ejecución en la siguiente apertura sin latencia adicional, capital completamente reinvertido, sin intereses sobre efectivo ni restricciones de lotes. No modela spreads variables, profundidad, impacto, impuestos ni pérdidas intravela. El horizonte de la etiqueta close/close y el retorno ejecutable open/open son distintos; una clasificación correcta puede perder dinero después de costos.

## Artefactos y pruebas

```text
artifacts/cache/                 ZIP originales + checksums
artifacts/data/                  CSV crudo/preparado y manifiestos
artifacts/models/<version>/      model.ubj, metadata.json, test_metrics.json, test_predictions.csv
artifacts/models/latest.json     puntero a la versión más reciente
artifacts/reports/<version>/     evaluación, predicción y backtest
```

Los artefactos están excluidos de Git para evitar incorporar grandes datasets; permanecen en esta carpeta. Conservarlos junto con el lockfile para reproducir los resultados. [Persistencia nativa de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html) en UBJSON: no se usa pickle. Los hashes detectan corrupción accidental, no autentican archivos maliciosos.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
# Conserva la salida de todos los controles en artifacts/verification/checks.json:
.\.venv\Scripts\python.exe scripts/verify.py
```

Las pruebas usan datos sintéticos únicamente para verificar invariantes y errores; las métricas experimentales se obtuvieron de históricos reales. Detalles del diseño en [ARCHITECTURE.md](ARCHITECTURE.md), estado y próximas etapas en [ROADMAP.md](ROADMAP.md).

## Optimización moderada y walk-forward

La segunda etapa agregó `training/walk_forward.py`, `training/optimization.py` y `evaluation/selection.py`. La ronda real ejecutó cuatro recetas (referencia más tres variantes), con siete ventanas purgadas, y se detuvo según el protocolo. Ninguna variante se aceptó: el campeón anterior sigue en `latest`; el candidato reentrenado quedó guardado para investigación. Los resultados completos están al final de [EXPERIMENTS.md](EXPERIMENTS.md).

```powershell
.\.venv\Scripts\python.exe -m trading_ml optimize --protocol config/optimization.toml --version v20261012T130540_d2bb6ac0
# Registrar el informe ya existente; este script no entrena:
.\.venv\Scripts\python.exe scripts/record_optimization.py
```

Si se invoca nuevamente con el mismo modelo anterior y el mismo protocolo ya completado, se devuelve el informe guardado sin entrenar ni consumir experimentos adicionales. El presupuesto incluye la referencia y como máximo cuatro variantes; no se agrega una grilla dentro de cada receta. La cola interior de train selecciona cantidad de árboles; el fold exterior evalúa probabilidades y ejecución con costos. La parada requiere tres variantes seguidas sin cumplir la regla completa, no sólo sin mejorar accuracy.

Los períodos walk-forward usados para elegir son **desarrollo**. El test original se consultó una sola vez después de congelar el candidato; como ya había sido publicado en la etapa anterior, no se presenta como evaluación nueva y ciega. No conviene cambiar repetidamente el protocolo para buscar mejores números en ese mismo test.

`features.enhanced = true` activa ocho variables adicionales y schema 2; la configuración por defecto y los modelos anteriores conservan schema 1 y sus 14 variables. El modelo auditado necesita su dataset compatible:

```powershell
.\.venv\Scripts\python.exe -m trading_ml predict --version v20261012T131650_5106daf4 --data artifacts/optimization/20261012T131625_fe8294f6/E2/prepared.csv
```

Esto permite inspeccionar el candidato sin convertirlo en campeón. El `compare` general exige contratos idénticos; la ronda de optimización compara recetas con distintas features sobre referencias/etiquetas idénticas dentro de su informe propio. La optimización actual está definida para el modelo original diario y snapshot exacto: otros horizontes, contratos o períodos requieren definir un protocolo nuevo antes de entrenar.

## Smart Money Concepts

Se incorporaron BOS, CHoCH, liquidity sweeps, FVG, order blocks y estructura/tendencia como detectores causales. Sus reglas matemáticas, confirmaciones y limitaciones están en [SMC_RULES.md](SMC_RULES.md). No identifican órdenes institucionales ni envían operaciones. `Features.smc=true` activa el contrato 3; desactivado conserva los contratos anteriores.

La comparación real evaluó cuatro recetas en las mismas siete ventanas purgadas y se detuvo: SMC no mostró aporte predictivo consistente. El original sigue en `latest`; el candidato combinado seleccionado, con calibración temporal, está guardado como investigación. El test observado anteriormente no se volvió a evaluar. Los resultados de los tres sistemas y del control con sólo features SMC están en [EXPERIMENTS.md](EXPERIMENTS.md).

```powershell
# Recupera la ronda ya completada sin consumir nuevos experimentos:
.\.venv\Scripts\python.exe -m trading_ml smc-study --protocol config/smc_study.toml --version v20261012T130540_d2bb6ac0

# Análisis usando el candidato de investigación SMC, no el puntero global latest:
.\.venv\Scripts\python.exe -m trading_ml smc-analyze --output artifacts/smc_studies/20261012T134906_05753740/latest_analysis.json

# Se puede especificar modelo, dataset compatible y referencia:
.\.venv\Scripts\python.exe -m trading_ml smc-analyze --version v20261012T134917_f88d6f93 --data artifacts/smc_studies/20261012T134906_05753740/prepared_smc.csv --reference 2026-10-01T00:00:00Z

# Documenta una ronda existente; no entrena ni evalúa el test:
.\.venv\Scripts\python.exe scripts/record_smc_study.py
```

La respuesta incluye `trend`, `market_structure`, `patterns` con referencia de confirmación, `active_zones`, `probability_up/down`, `signal_state`, horizonte y versión. Los estados son `bullish`, `bearish` y `neutral`. Neutral expresa abstención entre p=0.45 y p=0.55 o datos de calentamiento insuficientes; no es una tercera clase de XGBoost.

`calibration_status` distingue ausencia de calibración y sigmoide aprendida en pasado disjunto. La evaluación de calibración figura en Brier/log-loss/ECE y reliability bins; puede degradarse bajo drift. `smc_rules` muestra por separado el score heurístico y su estado `heuristic_uncalibrated`: no se presenta como una frecuencia empírica aprendida. Una tendencia estructural alcista puede coexistir con estado neutral a un día.

```python
from pathlib import Path
from trading_ml.models import load_model
from trading_ml.prediction.smc_analysis import analyze_smc
from trading_ml.smc import detect_smc, rule_probabilities

bundle = load_model(Path("artifacts"), "v20261012T134917_f88d6f93")
analysis = analyze_smc(bundle, Path(bundle.metadata["dataset_path"]))
print(analysis["trend"], analysis["signal_state"], analysis["probability_up"])

# Sólo reglas, sin entrenar ningún modelo; ohlcv debe ser un DataFrame validado:
# patterns = detect_smc(ohlcv)
# probability, state = rule_probabilities(patterns)
```

El ejemplo actual usa la vela cerrada con referencia 2026-10-01: es histórico y avisa su antigüedad. No evalúa si esa predicción acertó ni constituye una decisión actual.

### Evaluación futura independiente

Se congelaron 90 referencias UTC nuevas: **2026-10-13 inclusive a 2027-01-11 exclusivo**. La última apertura de salida debe existir en el CSV como vela cerrada, además de las etiquetas. Con los datos actuales la evaluación final permanece pendiente; ningún resultado futuro se inventó.

Cuando los archivos mensuales necesarios estén publicados, copiar `config/default.toml` a una configuración nueva, mantener activo/intervalo/start y extender `market.end`, por ejemplo a `2027-02-01`. Descargar hacia **otro archivo** para preservar el snapshot y ejecutar:

```powershell
.\.venv\Scripts\python.exe -m trading_ml --config config/future.toml download --output artifacts/data/future_raw.csv
.\.venv\Scripts\python.exe -m trading_ml smc-future-evaluate --raw artifacts/data/future_raw.csv
```

Ese comando verifica integridad, conservación exacta del histórico, ventana completa y etiquetas/aperturas de salida; evalúa original guardado, reglas fijas y candidato congelado. No entrena, recalibra, elige candidatos ni promueve. Con una ventana parcial falla explícitamente y no produce métricas finales. Sus fechas de ajuste pueden diferir entre versiones; el control S2 de desarrollo, con train idéntico, sirve para estudiar el aporte específico de las features.
