# Motor de Machine Learning para trading

Motor Python independiente para investigación: descarga OHLCV reales, valida datos, construye indicadores causales, entrena XGBoost, evalúa fuera de muestra y simula una estrategia con costos. No requiere frontend, backend propio ni credenciales. No coloca órdenes.

La evaluación predictiva incorpora p=0,5, prevalencia histórica y logística L2, bootstrap por bloques y ablaciones fijas. `train` informa y conserva los candidatos rechazados por baselines; su promoción queda bloqueada. `predictive-study` genera comparaciones reproducibles sin optimizar el test observado ni promover modelos. Protocolo, resultados positivos/negativos y límites en [PREDICTIVE_EVALUATION.md](PREDICTIVE_EVALUATION.md).

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

# 3. Entrenar/seleccionar en pasado y registrar un candidato inactivo
$trainingResult = .\.venv\Scripts\python.exe -m trading_ml train | ConvertFrom-Json
$candidateVersion = $trainingResult.model_version

# 4. Cargar y evaluar sólo observaciones posteriores a entrenamiento/selección
.\.venv\Scripts\python.exe -m trading_ml evaluate --version $candidateVersion --output "artifacts/reports/$candidateVersion/evaluation.json"

# 5. Predecir la última vela cerrada del dataset
.\.venv\Scripts\python.exe -m trading_ml predict --version $candidateVersion --output "artifacts/reports/$candidateVersion/prediction.json"
# También acepta --reference 2026-10-01T00:00:00Z

# 6. Backtest long/cash; se guardan resumen, trades y equity
.\.venv\Scripts\python.exe -m trading_ml backtest --version $candidateVersion

# 7. Ampliar el histórico hacia archivos nuevos, preservando el snapshot del padre
# config/expanded.toml debe conservar contrato/start y extender market.end
.\.venv\Scripts\python.exe -m trading_ml --config config/expanded.toml download --output artifacts/data/expanded_raw.csv
.\.venv\Scripts\python.exe -m trading_ml --config config/expanded.toml process --input artifacts/data/expanded_raw.csv --output artifacts/data/expanded_prepared.csv
.\.venv\Scripts\python.exe -m trading_ml retrain --version latest --data artifacts/data/expanded_prepared.csv
```

`train` y `retrain` conservan el modelo activo. Para inspeccionar un candidato usar su ID explícito. `latest`, `active` y `champion` resuelven `models/active.json` cuando existe; los punteros históricos siguen disponibles como compatibilidad. La activación exige una decisión explícita y un informe íntegro generado con `evaluate --output`; ver [MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md). No se activa ningún candidato por ejecutar el pipeline u optimizar.

El último comando rechaza un dataset sin nuevas velas y exige conservar el snapshot del padre. Si el último mes disponible ya está descargado, no hay datos más nuevos que incorporar aún. Para verificar el mecanismo hoy con dos cortes históricos reales:

```powershell
.\.venv\Scripts\python.exe scripts/run_retraining_demo.py
$retrainingDemo = Get-Content artifacts/reports/retraining_demo.json -Raw -Encoding UTF8 | ConvertFrom-Json
.\.venv\Scripts\python.exe -m trading_ml compare --versions $retrainingDemo.parent_version $retrainingDemo.retrained_version --output artifacts/reports/comparison.json
```

La demostración usa un corte anterior a julio de 2026 y luego incorpora julio-septiembre. Crea candidatos sin activarlos; consultar `artifacts/reports/retraining_demo.json` para los IDs de cada ejecución. `compare` usa las mismas filas posteriores a la selección de **todos** los modelos. Comparar tests distintos produciría una comparación engañosa.

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
artifacts/models/<version>/      model.ubj, metadata.json, test_metrics.json, data/prepared.csv y manifiesto
artifacts/models/active.json     puntero único de activación explícita
artifacts/models/latest.json     puntero histórico de compatibilidad
artifacts/evaluations/           recibos de evaluaciones registrados sin sobrescritura
artifacts/promotions/            decisiones explícitas de activación
artifacts/reports/<version>/     evaluación, predicción y backtest
```

Los artefactos están excluidos de Git para evitar incorporar grandes datasets; permanecen en esta carpeta. Conservarlos junto con el lockfile para reproducir los resultados. [Persistencia nativa de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html) en UBJSON: no se usa pickle. Los hashes detectan corrupción accidental, no autentican archivos maliciosos.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
# Conserva cada ejecución nueva en artifacts/verification/runs/<id>.json:
.\.venv\Scripts\python.exe scripts/verify.py
# Comprueba también la ronda histórica sin reescribir datasets/metadatos:
.\.venv\Scripts\python.exe scripts/verify.py --study artifacts/recent_retraining/20261012_short
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

## Momentum experimental independiente

El paquete `trading_ml.momentum` agrega features causales, una estrategia long/efectivo con permanencia según tendencia y una variante donde XGBoost filtra las mismas oportunidades. Conserva las features, reglas, modelos y punteros actuales. El protocolo fija cuatro horizontes (20/60/120/250 velas), sin grilla ni elección del mejor test. Incluye costos por lado, spread, stress con costos duplicados, siete ventanas temporales, comparación con tres modelos actuales, buy and hold y efectivo, análisis por régimen y trimestre, y bootstrap pareado por bloques con corrección de comparaciones múltiples.

```powershell
.\.venv\Scripts\python.exe -m trading_ml momentum-study --protocol config/momentum_study.toml
# Registrar un informe existente; no vuelve a entrenar:
.\.venv\Scripts\python.exe scripts/record_momentum_study.py artifacts/momentum_studies/20261013T215021_338a147b
```

El estudio real conserva **212 registros y 45 ajustes XGBoost por ejecución** en `artifacts/momentum_studies/<run_id>/`, incluyendo abstenciones y pérdidas. Ninguna variante demostró ventaja bajo los criterios fijados. En el historial posterior a desarrollo, las reglas perdieron entre 6,92% y 20,77% neto; el filtro ML no demostró una mejora consistente. Los modelos Momentum para una reserva futura de 180 días quedaron congelados sin activación; el guard de entrenamiento protege sus etiquetas. Ese test todavía está pendiente.

Protocolo, resultados completos, costos, limitaciones y API reutilizable para adaptadores de Forex/cripto/futuros en [MOMENTUM_RESEARCH.md](MOMENTUM_RESEARCH.md). Las fechas de esa reserva y las de SMC son distintas y están registradas en sus respectivos planes. Todo el módulo trabaja offline y no envía órdenes.

## Mean Reversion experimental

`trading_ml.mean_reversion` incorpora z-score respecto de la media y desviación de las velas anteriores, RSI, volatilidad, ATR y estadísticas de retornos. Compara reglas, regresión logística L2 con scaler temporal y XGBoost sobre las mismas oportunidades. El protocolo fija ventanas de 20/60 velas y filtros de todos los regímenes/sólo lateral, sin búsqueda de umbrales en test.

El objetivo de reversión queda anclado a la media conocida en la señal. Una operación sale por objetivo, stop al cierre o timeout de 10 velas; las etiquetas distinguen estos desenlaces y conservan los casos censurados. La asignación máxima es 50% y el sizing usa un presupuesto previsto de pérdida de 1%, incluidos costos; los gaps pueden superar ese presupuesto y se registran. Se ejecuta en la próxima apertura, sin apalancamiento ni promediar pérdidas. El filtro ML estima alcanzar el objetivo dentro del plazo, no una ganancia neta garantizada.

```powershell
.\.venv\Scripts\python.exe -m trading_ml mean-reversion-study --protocol config/mean_reversion_study.toml
# Publicar documentación de la ejecución existente, sin entrenar:
.\.venv\Scripts\python.exe scripts/record_mean_reversion_study.py artifacts/mean_reversion_studies/20261013T220145_a6d4131b
```

La ejecución real conserva **232 registros y 68 ajustes** en `artifacts/mean_reversion_studies/<run_id>/`. Ninguna variante cumple todos los criterios estadísticos. En el historial posterior a desarrollo las reglas pierden entre 3,35% y 5,63% neto; los filtros permanecen en efectivo. El 0% de esos filtros no demuestra una ventaja frente a efectivo. No se modificaron modelos ni punteros existentes, verificados por hash.

Frecuencia de oportunidades, expectativa, drawdown, turnover, tasas de reversión, métricas probabilísticas, regímenes, estabilidad trimestral y bootstrap temporal corregido por comparaciones múltiples en [MEAN_REVERSION_RESEARCH.md](MEAN_REVERSION_RESEARCH.md). Se congelan todas las variantes para una reserva futura protegida; la evaluación independiente permanece pendiente. Los módulos se pueden reutilizar con adaptadores de otros mercados y no contienen envío de órdenes.

## Políticas experimentales de gestión de riesgo

`trading_ml.risk_overlay` compara diez políticas fijas sobre los tres modelos registrados: original, caps de 10/25/50%, sizing por volatilidad y ATR, stop de ATR al cierre, reducción/pausa por drawdown, persistencia/cooldown y una combinación. Mantiene las mismas probabilidades en todas las políticas. No cambia estrategias, modelos ni reservas existentes.

```powershell
.\.venv\Scripts\python.exe -m trading_ml risk-study --protocol config/risk_study.toml
# Documentar resultados existentes; no vuelve a evaluar ni entrenar:
.\.venv\Scripts\python.exe scripts/record_risk_study.py artifacts/risk_studies/20261013T221816_a8b7d443
```

La comparación principal fue fijada como `combined_25` sobre el modelo original antes de evaluar el historial final. En BTC diario, 2025-06-05 a 2026-09-30, el resultado es **MIXTO**: neto −45,32% → −1,51%, drawdown −46,89% → −2,37%, exposición media del capital 24,07% → 1,55%, operaciones 116 → 19 y costos de ejecución 2.745,29 → 59,68 sobre capital inicial de 10.000. La mejora proviene principalmente de menos exposición y rotación; efectivo sigue siendo superior y las probabilidades originales no mejoraron.

Los controles estáticos 10/25/50% retornaron −5,51%/−13,33%/−25,27% neto en el mismo historial, sin cambiar entradas o salidas. El control adicional de exposición usa exclusivamente desarrollo para fijar su cap. La combinación pierde 5,91% en desarrollo, con sólo dos de siete folds positivos; no se presenta como una estrategia rentable validada.

Se conservan **518 registros y 21 refits temporales de XGBoost**, más la calibración temporal de la referencia SMC, en `artifacts/risk_studies/<run_id>/`. El libro bruto repite las decisiones y fracciones del neto, eliminando únicamente fricción de ejecución. Los stops y límites de drawdown usan cierres conocidos y ejecutan en la apertura siguiente; no prometen fills intravela ni límites garantizados ante gaps. Las políticas se guardan sin activación. Modelos, punteros y reservas prospectivas se verificaron por hash y permanecen intactos.

Auditoría, Profit Factor, bruto/neto, exposición temporal, costos, estabilidad y API en [RISK_RESEARCH.md](RISK_RESEARCH.md). El estudio no evalúa ni modifica el conjunto prospectivo reservado, y no envía órdenes.
