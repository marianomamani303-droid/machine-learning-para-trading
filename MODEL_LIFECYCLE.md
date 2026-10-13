# Correcciones del ciclo de modelos — 2026-10-13

El entrenamiento registra candidatos inactivos; la evaluación y la promoción son acciones independientes. Estas correcciones mantienen XGBoost, etiquetas, indicadores, reglas SMC y estrategia long/cash. No agregan ejecución de órdenes.

## Entrenamiento, registro, evaluación y activación

`train(settings, dataset)` ajusta y selecciona con train/validación. Guarda un candidato con `test_metrics.status=not_evaluated`, sin inferir sobre el test ni escribir predicciones de test. `publish_candidate` registra el UBJSON y evidencia en una versión nueva. Cada versión nueva conserva una copia propia del CSV preparado y su manifiesto, con origen y digest; procesar el archivo de entrada nuevamente no sobrescribe esa copia.

`evaluate(bundle, dataset, output)` mide únicamente filas posteriores a ajuste/selección. Con `output`, crea un informe nuevo y un recibo en `artifacts/evaluations`. Rechaza sobrescribir informes existentes. Los metadatos del modelo permanecen inmutables; las métricas nuevas viven fuera de su directorio. Una ventana prospectiva con límites congelados debe estar completa y se evalúa dentro de esos límites.

`promote(root, version, evaluation, reason)` exige ID concreto, motivo y un informe registrado sin alteraciones. Comprueba hashes de reporte, modelo, metadatos, configuración y dataset, además de contrato y límites temporales. La promoción es una decisión explícita de investigación: la integridad de la evidencia no constituye un criterio de rentabilidad. Los criterios cuantitativos de aceptación de los estudios permanecen separados y no se reemplazan por esta operación administrativa.

La etapa predictiva posterior agrega un filtro de baselines: `train`/`retrain` registran decisiones calculadas solo en validación. `promote` bloquea nuevos candidatos rechazados o sin la evaluación requerida de p=0,5, prevalencia y logística. Los modelos históricos conservan sus metadatos originales; ver [PREDICTIVE_EVALUATION.md](PREDICTIVE_EVALUATION.md).

La activación utiliza un solo reemplazo atómico de `models/active.json`. Los alias `latest`, `active` y `champion` consultan ese puntero. Antes de la primera promoción explícita utilizan los punteros históricos. Los archivos anteriores `latest.json` y `champion.json` se conservan como evidencia histórica y ya no deben consultarse directamente para conocer una activación nueva. Los eventos quedan registrados en `artifacts/promotions`; el puntero determina qué versión está activa.

```python
from pathlib import Path
from trading_ml.config import load_settings
from trading_ml.training import train, retrain
from trading_ml.models import load_model, promote
from trading_ml.evaluation import evaluate
from trading_ml.backtesting import backtest

settings = load_settings()
version = train(settings)  # candidato, conserva el activo
bundle = load_model(settings.root, version)
report = settings.root / "reports" / version / "evaluation.json"
evaluate(bundle, Path(bundle.metadata["dataset_path"]), report)
simulation = backtest(bundle, Path(bundle.metadata["dataset_path"]))
# Tras una decisión explícita basada en la evidencia:
# promote(settings.root, version, report, "Motivo de la decisión revisada")
```

```powershell
# Usar un ID concreto; no equivale a una recomendación de activar el histórico actual.
.\.venv\Scripts\python.exe -m trading_ml evaluate --version <ID> --data <CSV> --output <INFORME_NUEVO>
.\.venv\Scripts\python.exe -m trading_ml promote --version <ID> --evaluation <INFORME_NUEVO> --reason "Motivo explícito de la decisión"
```

## Recetas y compatibilidad histórica

`training_recipe` diferencia máximos de ajuste y parámetros efectivos. Guarda búsqueda, semilla/horizonte/fracciones, método temporal, early stopping, configuración walk-forward, calibración y umbral. `retrain` reutiliza la receta completa, reajusta los árboles con ES cuando corresponde y ajusta un calibrador nuevo en pasado disjunto. Nunca reutiliza el calibrador del padre sobre un XGBoost nuevo.

Para versiones anteriores, la receta se reconstruye desde sus planes y resúmenes con digest verificado. Los límites de 400 árboles en optimización/SMC y 80 en la ronda corta se recuperan de esos planes, aunque el ajuste final tuviera dos árboles. Si falta el plan, se produce un error de datos explícito; no se inventa una receta a partir del número efectivo de árboles.

El loader normaliza metadatos en memoria. En los candidatos cortos recupera `last_reference` y `last_target_time` del período futuro declarado. `dataset_reference_end` identifica por separado el fin real del snapshot y gobierna la comprobación de nuevas velas. Un nuevo candidato temporal con evaluación futura abierta contiene `last_reference` y `last_target_time` nulos: no se inventan fechas observadas todavía desconocidas.

Los bloques de fit, ES y calibración tienen hashes separados. Se registran configuración, receta, entorno, commit Git y hashes de los archivos Python; estos últimos identifican también cambios todavía sin commit. Los snapshots antiguos deben seguir disponibles para comprobar el período compartido al reentrenar. Los candidatos nuevos preservan ese snapshot dentro de su versión.

La purga continúa utilizando fechas de disponibilidad de etiquetas. Los datos de cualquier reserva SMC congelada están bloqueados para ajuste y selección. Reentrenar una receta temporal crea un candidato posterior con su propio cutoff; no sustituye los modelos, fechas o plan del test prospectivo vigente.

## Umbrales y costos

`rule_probabilities` recibe el umbral explícito. Estudio, análisis, métricas selectivas y evaluador prospectivo utilizan el umbral registrado; se preservan los valores del protocolo congelado. Las probabilidades heurísticas y el algoritmo detector no cambian.

La CLI y `backtest(bundle, dataset)` usan los costos, capital y umbral del modelo. La API rechaza overrides incompatibles. Para análisis de escenarios diferentes se puede llamar al simulador independiente `simulate` y registrar un experimento distinto. Cambiar `config/default.toml` no altera el backtest de una versión registrada.

## Verificación y evidencia preservada

El verificador mantenido es `scripts/verify.py`. Ejecuta realmente pytest, Ruff lint/formato y pip check; conserva comandos, stdout, stderr, códigos de salida, entorno y código. Genera un informe nuevo por ejecución. Un control fallido produce `passed=false` y salida distinta de cero.

```powershell
.\.venv\Scripts\python.exe scripts/verify.py --study artifacts/recent_retraining/20261012_short
```

`--study` comprueba integridad, contratos, coincidencia de datasets, carga nativa y artefactos protegidos de la ronda corta sin reconstruir datasets ni reescribir metadatos. Los checks de código actuales se ejecutan; no se confunden con hashes del código anterior al cambio.

La entrada `artifacts/recent_retraining/20261012_short/verify.py` ahora delega en el verificador mantenido y ejecuta controles reales. Su código original se preservó byte por byte en `verify_20261012_record.py`, como registro histórico que no debe ejecutarse: contenía valores declarados de checks y mutaciones de metadatos. Los modelos, datasets, informes originales y reserva prospectiva se mantienen intactos; la implementación mantenida está en `src/trading_ml/verification.py`, bajo Git.

Los tests nuevos cubren ausencia de activación/test durante train, promoción explícita e integridad, snapshots, recetas temporales calibradas/no calibradas, adaptación de metadatos cortos, historia revisada, protección prospectiva, ventanas incompletas, umbrales, equivalencia CLI/API de costos y propagación de fallos reales de verificadores. Una prueba fuerza aceptación completa en optimización y comprueba que aun así no hay activación automática.

## Validación de cierre

El verificador corregido se ejecutó también desde la ruta antigua, con `--output artifacts/verification/runs/lifecycle-final-20261013.json`: **91 pruebas aprobadas en 57,86 segundos**, Ruff lint correcto, formato de 58 archivos correcto y `pip check` sin dependencias rotas. El informe guarda las salidas reales y la comprobación de la ronda histórica. Son pruebas de software con entrenamientos sintéticos pequeños; no se entrenaron ni promovieron modelos reales.

Se verificaron por SHA-256 los 575 archivos presentes en artifacts al comenzar: 574 conservan exactamente sus bytes; únicamente cambió la entrada ejecutable del verificador. Su fuente original se conserva con el mismo digest en `verify_20261012_record.py`. Todos los modelos, metadatos históricos, datasets, resultados y archivos de la reserva prospectiva permanecen intactos. Se cargaron las ocho versiones y se recuperaron sus recetas; el activo sigue siendo `v20261012T130540_d2bb6ac0` y la reserva sigue `pending_future_data`.

## Límites que permanecen

La rentabilidad, señal predictiva y calibración futura no se mejoran mediante estas correcciones de infraestructura. La evaluación prospectiva continúa pendiente. El entrenamiento básico conserva su partición 60/20/20; las recetas temporales conservan su ajuste causal con ES/calibración. No se rediseñaron estrategia, modelos ni proveedor mensual. Las rutas de artefactos históricos pueden requerir adaptación explícita si se mueve el proyecto a otra máquina.
