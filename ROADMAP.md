# Roadmap

Fecha del análisis inicial: 2026-10-12 (America/Buenos_Aires).

## Estado inicial

La carpeta de trabajo está vacía. No hay código, datos, modelos, pruebas ni arquitectura previa que verificar. Python 3.14.3 está instalado. No se encontró AGENTS.md en el proyecto ni en los directorios superiores inspeccionados.

Las tareas se marcan terminadas únicamente después de su ejecución y verificación. No se construirá una aplicación web ni un sistema de ejecución de órdenes.

## Etapa 1 — P0: base reproducible

- [x] Paquete Python modular, configuración validada, CLI y logging.
- [x] Entorno virtual, dependencias instaladas y registro de versiones.
- [x] Descarga de OHLCV reales, procedencia y validación de integridad.
- [x] Limpieza, duplicados, faltantes y reporte de calidad.
- [x] Variables causales y etiquetas con horizonte configurable.

## Etapa 2 — P0: entrenamiento y evaluación válida

- [x] Separación cronológica train/validation/test con purga por horizonte.
- [x] XGBoost reproducible y búsqueda básica de hiperparámetros usando sólo validación.
- [x] Probabilidades, métricas de clasificación y calibración evaluada.
- [x] Persistencia nativa, versiones, comparación y reentrenamiento.
- [x] Controles de procedencia para impedir evaluar/backtestear sobre entrenamiento.

## Etapa 3 — P0: uso independiente

- [x] API Python y comandos download/process/train/evaluate/predict/backtest/retrain/compare.
- [x] Predicciones con activo, referencia UTC, horizonte, versión y advertencias.
- [x] Backtesting causal con entrada en apertura posterior, costos y buy and hold comparable.

## Etapa 4 — P0: ejecución y evidencia

- [x] Pruebas unitarias y de integración ejecutadas.
- [x] Pipeline ejecutado con históricos reales y primer modelo guardado.
- [x] Predicción y backtest reales registrados sin ajustes guiados por test.
- [x] README.md, ARCHITECTURE.md y EXPERIMENTS.md con resultados y límites.

Evidencia de cierre: 37 tests aprobados sin warnings, Ruff lint/formato y pip check correctos; salida guardada en `artifacts/verification/checks.json`. 2,465 velas BTCUSDT de 81 meses con checksums verificados. Pipeline completo, reentrenamiento retrospectivo con datos reales y comparación ejecutados. La repetición del corte completo produjo el mismo hash de modelo. Detalles y resultados negativos reales en [EXPERIMENTS.md](EXPERIMENTS.md).

## Etapa 5 — P1/P2: mejoras futuras

- [x] Validación walk-forward de múltiples ventanas con purga (etapa 6, siete períodos reales).
- [x] Calibración entrenada en ventana disjunta del fit y early stopping (etapa 7); generalización prospectiva pendiente.
- [ ] Más proveedores, activos y validación de cambios de régimen.
- [ ] Archivos diarios/incrementales para inferencia reciente dentro del mes corriente.
- [ ] Monitoreo de drift, alertas y política de reentrenamiento.
- [ ] Modelos alternativos y baselines adicionales.
- [ ] Simulación de liquidez, spreads y costos variables.
- [ ] Integración futura mediante la API Python con un backend externo.

## Etapa 6 — Optimización moderada (2026-10-12)

Protocolo fijado en `config/optimization.toml`: máximo cinco recetas incluyendo referencia, siete ventanas de desarrollo, purga exterior/interior y parada tras tres variantes consecutivas sin aceptación. El período final original se excluye de selección; ya fue observado en la etapa anterior y no se presenta como test nuevo ciego.

- [x] Walk-forward probado y ejecutado con históricos reales.
- [x] Regularización, early stopping interior y ocho features causales adicionales evaluados.
- [x] Historia desde 2017 evaluada frente al mismo período fuera de muestra.
- [x] Parada anticipada y conservación del campeón verificadas.
- [x] Comparación final única, artefactos y resultados registrados.

Se ejecutaron E0/E1/E2/E3: cuatro recetas de un máximo de cinco. Ninguna variante cumplió la regla fijada; parada tras tres rechazos. E4 (train móvil de 730 días) quedó sin ejecutar. La historia extendida tiene 3,332 velas reales y no aportó mejora suficiente. Candidato auditado: `v20261012T131650_5106daf4`, elegido por desarrollo, con mejora descriptiva de log-loss/Brier y menor pérdida final, pero accuracy/AUC peores y falta de consistencia económica. `latest` y `champion.json` conservan `v20261012T130540_d2bb6ac0`.

Evidencia: `artifacts/optimization/20261012T131625_fe8294f6/summary.json`, plan y selección congelados, 28 modelos por fold, predicciones OOF y equity/trades de todas las recetas. Las ventanas E0 se reentrenan antes de cada evaluación; no se usa el modelo guardado para simular fechas anteriores a su entrenamiento. Pruebas adicionales: purga interior/exterior, causalidad de 22 features, compatibilidad de modelos viejos, test final adversarial, presupuesto/parada, cache de ronda completada y conservación del campeón.

Verificación final: 45 tests aprobados sin warnings, Ruff lint/formato y pip check correctos; evidencia en `artifacts/verification/checks.json`. Candidato cargado y predicción histórica ejecutada desde CLI. La recuperación de la ronda completada se verificó sin crear nuevos experimentos.

### Pendientes tras la parada

- [ ] Nueva ventana prospectiva no observada, sin seguir ajustando el mismo test.
- [ ] Confirmación prospectiva de la calibración (ajuste en ventana separada implementado en etapa 7).
- [ ] Evaluar ventana móvil de train en una futura ronda autorizada (implementada y probada como mecanismo, receta E4 no ejecutada).
- [ ] Señales distintas a los indicadores OHLCV y monitoreo de drift/regímenes.
- [ ] Atribuir las mejoras de pérdidas entre exposición, costos, datos recientes y features con un protocolo independiente.

## Etapa 7 — SMC (2026-10-12)

Reglas matemáticas en `SMC_RULES.md`, configuración previa en `config/smc_study.toml`. Se excluye completamente el test histórico observado de entrenamiento/selección/evaluación de esta ronda. La confirmación final se reserva a velas nuevas posteriores al día de ejecución; su estado debe permanecer pendiente hasta obtenerlas. No hay promoción automática.

- [x] Detectores causales, ciclo de vida de zonas y contrato de features probado.
- [x] Comparación real de original, SMC reglas y XGBoost+SMC en las mismas ventanas.
- [x] Calibración temporal disjunta y señal neutral verificadas.
- [x] Candidato SMC reentrenado, guardado y análisis de BTC/USDT ejecutado.
- [x] Presupuesto/parada cumplidos y campeón anterior conservado.
- [x] Resultados, frecuencia/calidad de señales y limitaciones documentados.
- [ ] Evaluación final independiente con datos futuros (pendiente por diseño).

Ronda `20261012T134906_05753740`: S0 original, S1 reglas SMC, S2 original+32 features SMC y S3 regularizado/calibrado. Cuatro de hasta cinco recetas; parada tras dos variantes ML no aceptadas. S4 de confirmación más lenta no se ejecutó. 1,244 referencias OOF en siete ventanas, 21 modelos por fold y un candidato reentrenado `v20261012T134917_f88d6f93`. `latest` y `champion.json` siguen en `v20261012T130540_d2bb6ac0`.

No se demostró aporte consistente atribuible a SMC: S2 mejoró log-loss sólo 0.000908 y bajó AUC media por fold 0.000240. S3 mejoró Brier/log-loss con regularización y calibración, pero tuvo 135 trades concentrados en una ventana y no pasó consistencia económica ni retorno positivo. Las reglas sin ML tuvieron AUC OOF 0.482711. Ninguna promoción automática.

Reserva final congelada: referencias UTC 2026-10-13 inclusive a 2027-01-11 exclusivo. `smc-future-evaluate` verifica ventana completa, etiquetas, apertura de salida, modelos congelados y snapshot histórico preservado; hoy devuelve pendiente y no produce métricas. La ruta de evaluación completa se probó con datos sintéticos y reloj simulado en tests temporales, no con datos futuros reales.

Evidencia: `artifacts/smc_studies/20261012T134906_05753740` y sección nueva en [EXPERIMENTS.md](EXPERIMENTS.md). Detectores, prefijos causales, zonas, calibración disjunta, contrato legado, señal neutral, presupuesto/parada, conservación, inferencia y flujo prospectivo incluidos en pruebas. Funcionalidad de datos diarios incrementales y validación independiente siguen pendientes; no se seguirá ajustando el test histórico observado.

Verificación de cierre: 65 tests pasaron sin warnings, Ruff lint/formato y pip check correctos. `smc-analyze` se ejecutó y guardó `latest_analysis.json`; `smc-future-evaluate` rechazó correctamente los históricos actuales y no escribió métricas finales. Snapshot y plan íntegros, 21 modelos de folds, candidato cargable, S4 ausente y campeón anterior intacto.

## Reentrenamiento corto posterior — 2026-10-12

- [x] Redescarga moderada y verificada: 1,004 velas BTC/USDT diarias de enero 2024 a septiembre 2026; sin fechas adicionales respecto del snapshot disponible.
- [x] Dos recetas cortas fijas, máximo 80 árboles y un hilo; ES interior, SMC con calibración separada y seis modelos de tres folds comunes purgados.
- [x] Comparación retrospectiva anterior/nuevo de 480 referencias, con costos y frecuencia de señales; sin buscar variantes con el antiguo test observado.
- [x] Candidatos completos guardados sin reemplazo: original `v20261012T143622_ff1c464c`, SMC `v20261012T143623_0d9b01f0`. Modelos previos, activo, arquitectura y configuración intactos.
- [x] Parada: no hay mejora consistente. Original mejora pérdida agregada pero empeora AUC/accuracy; SMC empeora calidad probabilística y no opera. 65 pruebas y Ruff aprobados; procedencia, contratos e integridad verificados.
- [ ] Obtener datos posteriores al cutoff para predicciones nuevas de los candidatos completos, conservando la guardia de inferencia.
- [ ] Completar la evaluación prospectiva independiente congelada; no reemplazar sus modelos/plan con estos candidatos ni afirmar generalización con los resultados retrospectivos.

Detalle y métricas por fold en [RESULTS.md](artifacts/recent_retraining/20261012_short/RESULTS.md) y registro en [EXPERIMENTS.md](EXPERIMENTS.md). No se implementó frontend, trading real ni nueva funcionalidad.
