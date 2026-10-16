# Progreso persistente — Deep Momentum Networks

Actualizado: 2026-10-16 (America/Buenos_Aires), adaptación implementada y 120 ajustes terminados.

## Objetivo
Transformar este repositorio en un sistema reproducible de investigación cuantitativa basado en *Deep Momentum Networks — Enhancing Time Series Momentum Strategies Using Deep Neural Networks* (Lim, Zohren y Roberts, arXiv:1904.04912). Evaluar ventajas económicas fuera de muestra después de costos sin operaciones con dinero real ni manipulación de resultados.

## Estado verificado
- Paper investigado, diseño predeclarado e implementación DMN terminada. **Ambos estudios reales completados y verificados**, sin ventaja robusta demostrada; conclusiones completas en `DMN_RESULTS.md`.
- Repositorio Python existente con `src`, `tests`, `config`, `scripts` y artefactos históricos.
- HEAD inicial: `5ee3b85` (historial anterior `dee134a`, `454ede8`).
- Hay cambios previos del usuario: `README.md`, `src/trading_ml/cli.py`; archivos nuevos de auditoría predictiva y `structure_regime`. Preservarlos y no atribuirlos a esta adaptación.
- No se encontraron instrucciones AGENTS.md en el repositorio ni en sus ancestros.

## Etapas completadas
1. Lectura del pedido completo y revisión inicial de estado Git.
2. Creación de este registro antes de investigación o cambios de código.
3. Auditoría de módulos de datos, configuración, CLI, contratos temporales y protección de reservas; consulta del paper completo por secciones y apéndices, investigación complementaria y compatibilidad PyTorch.
4. Diseño previo al entrenamiento en `DMN_RESEARCH.md`.
5. Línea base: **253 pruebas existentes aprobadas** (145,59 s). PyTorch **2.14.1+cpu** instalado y operativo; `pip check` sin incompatibilidades.
6. Paquete `trading_ml.dmn`: ocho features causales, secuencias, split purgado, lineal/MLP/LSTM/XGBoost, pérdidas directas, simulador con deriva y costos, bootstrap, estudios append-only y CLI separada. Dependencia opcional `dmn`.
7. **264 pruebas completas aprobadas** en 120,82 s (253 previas + 11 DMN). Incluye interrupción después del ajuste, recuperación sin fuente original, ausencia de nuevo entrenamiento y reconstrucción de contabilidad. Ruff y formato correctos; `pip check` correcto. Evidencia en `artifacts/verification/dmn_full_tests.txt`.
8. 120 ajustes reales terminados (long/cash + sensibilidad long/short), 1022 archivos verificados y 690 carteras reconstruidas. Ninguna orden real, promoción ni reserva previa evaluada.
9. Gráficos PNG/SVG generados e inspeccionados; guía de instalación/reanudación/fórmulas y análisis de resultados positivos/negativos terminados. README y arquitectura enlazan el nuevo núcleo.
10. Recuperación real de ambas corridas completadas comprobada: ningún archivo cambió, ningún ajuste adicional. Lockfile verificado offline con pip, entrypoint `trading-dmn` operativo. Recibo en `artifacts/verification/dmn_final_verification.json`.
11. Implementación/documentación guardadas en commit **`a496295`**. Se incorporó sólo el cambio DMN de README mediante staging parcial: los cambios iniciales de README/CLI y los archivos de `structure_regime`/auditoría permanecen intactos y sin incluirse en ese commit.

## Etapas pendientes
1. Investigación posterior: auditar disponibilidad/licencia y rolls de datos diarios de futuros diversificados o preparar un contrato Forex con bid/ask y calendario real. No sustituirlo por series continuas sin procedencia.
2. Fijar un nuevo protocolo antes de nuevos ajustes; no extender búsquedas arbitrarias para mejorar los folds ya observados.
3. Confirmación prospectiva independiente sólo después de congelar hipótesis/modelos/criterios y definir una reserva disjunta de las existentes (que llegan hasta 2027-10-13). Pendiente por tiempo/datos futuros; no hay evidencia actual que justifique activar DMN.

## Decisiones técnicas
- Conservar código y resultados históricos; añadir la investigación DMN sin sobrescribir cambios previos ni abrir conjuntos reservados para seleccionar modelos.
- No ejecutar órdenes reales ni imprimir credenciales.
- Contrastar el registro con Git, código y artefactos al retomar.

## Archivos modificados por esta tarea
- `CODEX_PROGRESS.md` (nuevo).
- `DMN_RESEARCH.md` (nuevo, metodología y desviaciones explícitas).
- `src/trading_ml/dmn/{__init__,__main__,config,data,models,portfolio,report,study}.py` (nuevos).
- `config/dmn.toml`, `tests/test_dmn.py`, `requirements-dmn-lock.txt` (nuevos).
- `scripts/plot_dmn_study.py`: gráficos PNG/SVG desde artefactos verificados, sin entrenar.
- `DMN_GUIDE.md`: instalación, fórmulas, contratos, ejecución/reanudación y límites económicos.
- `config/dmn_long_short.toml`: sensibilidad de dominio predeclarada antes de resultados agregados; 60 ajustes adicionales con exposiciones sintéticas y préstamo explícito.
- `DMN_RESULTS.md`: evidencia, resultados positivos/negativos, incertidumbre y diagnóstico.
- `README.md`, `ARCHITECTURE.md`: entrada principal y arquitectura DMN. En README se preservó la sección previa del usuario sobre estructura/regímenes.
- `pyproject.toml`: descripción, extra opcional `dmn` y entrypoint `trading-dmn`.

## Experimentos y resultados
- Principal long/cash: `20261016T204450_9f760c70`, **60 ajustes terminados**, 1552 fechas exteriores con cuatro días de frontera en cash. LSTM neta: retorno neto 22,06%, Sharpe 0,7172, 2/5 folds positivos; TSMOM: 30,48%, Sharpe 0,6135; long escalado: 59,49%, Sharpe 0,8119. El bootstrap de diferencia Sharpe vs TSMOM incluye cero: bloque 7 [-0,6086; 0,7784], bloque 30 [-0,7552; 0,5795]. **No hay ventaja robusta demostrada.**
- Principal verificado con reconstrucción contable y gráficos en `artifacts/dmn_analyses/20261016T204450_9f760c70` (PNG/SVG y manifiesto). 39 archivos de modelos/reservas protegidos, sin alteraciones.
- Sensibilidad long/short sintética: `20261016T204723_6818bead`, **60 ajustes terminados**, LSTM neta −10,46%, Sharpe −0,9775 y 0/5 folds positivos; costo doble Sharpe −1,116. Intervalos de mejora vs TSMOM también incluyen cero. El modelo neto fue peor que el bruto. No se eligió esta sensibilidad por resultados exteriores.
- Ambas usan BTC/ETH diarios, cinco folds, dos semillas, seis recetas y baselines cash/long/TSMOM/MACD. Cada una: 60 intentos/60 ajustes, 511 archivos verificados, 345 carteras reconstruidas y 39 archivos históricos protegidos intactos.
- No hay reemplazo/promoción de modelos, prueba independiente abierta ni ventaja robusta confirmada. La correlación de retornos LSTM long/cash con long escalado es 0,979: no atribuir su retorno positivo automáticamente a señal incremental.

## Problemas conocidos
- Árbol de trabajo inicialmente sucio; evitar commits que incluyan cambios ajenos inadvertidamente.
- El texto adjunto debe leerse como UTF-8 (PowerShell mostró mojibake con la codificación predeterminada).
- Resueltos en pruebas: pandas usa timestamps en microsegundos en este entorno (comparar Timedelta, no enteros ns); dropout exigió forzar entradas contiguas para reproducibilidad independiente del layout de memoria.
- LSTM de ventanas solapadas con dropout de entrada/salida, no recurrente variacional; mercado cripto long/cash y 365 días/año: adaptación explícita, no réplica exacta de los 88 futuros originales.
- Las evidencias incluyen rutas absolutas a archivos históricos protegidos. Conservar ubicación al verificar/reanudar estas corridas, o registrar explícitamente una migración; una corrida nueva puede reproducirse en otro checkout con sus propias rutas. Resguardar `artifacts` aparte: está excluido de Git.

## Comandos
```powershell
git status --short
git log -4 --oneline
rg --files src tests config scripts
.\.venv\Scripts\python.exe -m pytest tests/test_dmn.py -q
.\.venv\Scripts\python.exe -m trading_ml.dmn study --protocol config/dmn.toml
.\.venv\Scripts\python.exe -m trading_ml.dmn study --protocol config/dmn.toml --resume artifacts/dmn_studies/<run_id>
.\.venv\Scripts\python.exe -m trading_ml.dmn verify --study artifacts/dmn_studies/<run_id>
```

## Próxima tarea exacta
Al retomar, ejecutar `git status --short` y verificar ambos estudios con `python -m trading_ml.dmn verify --study artifacts/dmn_studies/<id>`; contrastar con este registro. **No repetir los 120 ajustes ni modificar los protocolos terminados para buscar mejores métricas.** La próxima investigación exacta es inventariar los manifiestos/cobertura/calendario de los datos Forex existentes en `artifacts/multimarket` y las condiciones de datos de futuros diversificados, sin abrir reservas ni sus métricas; documentar cuál permite una comparación DMN válida. Después diseñar un adaptador de panel/calendario/ejecución apropiado y predeclarar el siguiente estudio. La evaluación independiente permanece pendiente, y no se justifica activar un modelo con los resultados actuales. Los snapshots `source.zip` permiten continuar una corrida con código original si una sesión posterior cambia `src`.
