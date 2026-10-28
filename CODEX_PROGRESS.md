# Memoria persistente del proyecto de machine learning de trading

Actualizado: 2026-10-28 (America/Buenos_Aires). **Implementación online/CLI terminada y verificada; siguiente etapa: acumular observaciones prospectivas con el proceso activo.** Esta cabecera es el estado vigente; el registro anterior se conserva debajo como historial, no como instrucciones activas.

## 1. Objetivo general del proyecto

Construir un laboratorio local que analice mercado continuamente, emita predicciones sin operar, confirme resultados y aprenda incrementalmente. Comparar modelos sobre observaciones futuras comunes, conservar los validados y desactivar los persistentemente inferiores. Usar CLI en PowerShell, pruebas automatizadas y recuperación de estado. El código local es la fuente de verdad; no consultar ni descargar GitHub, reemplazar cambios locales ni escribir fuera del workspace. No abrir reservas históricas para seleccionar modelos online.

## 2. Arquitectura actual

- `src/trading_ml/`: motor Python; `data`, `features`, `training`, `prediction`, `evaluation`, `backtesting`, `models` y módulos de investigación DMN, RL y multimercado.
- `config/`: protocolos TOML. `scripts/`: análisis/verificación. `tests/`: pruebas pytest. `artifacts/`: datos, modelos y resultados locales, excluidos de Git.
- Arquitectura previa: modelos congelados XGBoost, logística, árboles, DMN y RL; datos CSV/manifiestos SHA-256; catálogo/inferencia/feed/log legado en `dashboard/`.
- Nueva arquitectura implementada: `online/{config,data,store,models,evaluation,service,cli}.py`, SQLite esquema 2, pesos JSON, checksums y snapshots inmutables de estados validados. Flujo: REST/CSV → velas cerradas → resolver etiquetas ya vencidas disponibles → aprender una vez → comparar en ventanas comunes → emitir nuevas probabilidades → commit atómico. La aplicación Streamlit se retiró; se conservan auxiliares compartidos en `dashboard/`.
- Almacenamiento: `bars` conserva OHLCV inmutables; `predictions` conserva probabilidades/features/etiquetas y si hubo aprendizaje; `models` guarda parámetros JSON; `meta` fija protocolo, proveedor, modo y cursor; `events` registra ciclos/decisiones; `validated_snapshots` conserva pesos y evidencia al validar. SQLite WAL, `synchronous=FULL`, `BEGIN IMMEDIATE` y claves únicas impiden aprendizaje duplicado entre procesos. No se usa pickle ni se reemplazan bases ajenas.
- Protocolo inicial `config/online.toml`: BTCUSDT spot H1, horizonte 1, inicio 2026-10-28 UTC, warmup 32, polling 30 s, retraso máximo 120 s; ventanas de evaluación 50, mínimo 100 objetivos comunes no solapados, paciencia 3, alpha 0,05. Ventanas cortas no garantizan evidencia: el contraste conservador puede exigir mejoras fuertes o una predeclaración distinta en una base nueva.
- Entorno local: `.venv/Scripts/python.exe`; comandos existentes `python -m trading_ml --help`, `python -m trading_ml.dmn --help`, `python -m pytest` y `python -m ruff`.

## 3. Funcionalidades completadas

- Memoria preexistente revisada y conservada; inventariados raíz, dependencias y modificaciones locales. Los resultados históricos debajo son registros previos y no se han reproducido en esta sesión.
- Se identificaron cambios anteriores del usuario en `README.md`, `rl_signal/`, su configuración, guías, script y prueba. Deben preservarse.
- Implementados baseline de frecuencia Beta y dos logísticas incrementales con tasas de aprendizaje fijas. Solo aprenden de predicciones propias confirmadas. Nuevas cohortes independientes de DMN/RL/XGBoost históricos.
- Implementada comparación pareada de Brier con intervalos no solapados, cota Hoeffding-Azuma bajo nulo condicional y alpha spending para dos direcciones/cohorte/miradas. Inferioridad significativa persistente desactiva; mejora persistente valida el proceso predictivo. Deterioro significativo revoca validación; los snapshots previos se conservan.
- CLI `online init/run/status/predictions/report/verify/backup/restore/catalog/infer`, wrapper PowerShell, salida JSON por línea y reintentos de red/locks. `run` devuelve las nuevas probabilidades; `predictions` consulta pendientes/evaluadas y sus resultados con filtros. Eliminados `dashboard/app.py`, comando web y dependencias web opcionales; auxiliares compartidos y sus pruebas permanecen.
- Suite completa: **455 pruebas aprobadas** en 176,70 s, recibo `artifacts/online_smoke/final_pytest.xml`. Tras las últimas ampliaciones de consulta y preservación SQLite: **53 pruebas de integración aprobadas** en 13,04 s (servicio 17, CLI 23, regresiones 13), recibo `artifacts/online_smoke/final_online_integration.xml`. Estos conteos se superponen y no deben sumarse. Ruff global limpio, formato de los archivos nuevos limpio y `pip check` correcto.
- Smoke real del wrapper PowerShell: `artifacts/online_smoke/cli_20261028_v2/`; 96 velas sintéticas, 195 predicciones, 192 confirmadas, 3 pendientes, reinicio idempotente sin nuevos cambios, backup/restauración verificados y consultas de probabilidades pendientes/evaluadas correctas. Esto verifica software, no calidad predictiva real.
- Proveedor real Binance comprobado. Estado principal `artifacts/online/state.sqlite3`: 999 velas H1, cero predicciones porque el cierre estaba fuera de tolerancia; integridad verificada. Smoke live de un minuto aislado en `artifacts/online_smoke/live_20261028_1m/`: **12 predicciones reales, 9 evaluadas sobre 3 objetivos compartidos y 3 pendientes**, cada modelo actualizado 3 veces, 1.004 velas almacenadas. Reinicio tras interrupción confirmó pendientes sin inventar predicciones omitidas. Evidencia exacta: `final_evidence.json`; ninguna operación, modelo validado ni rentabilidad demostrada.

## 4. Trabajo en progreso

- No queda una implementación obligatoria pendiente de esta sesión. La actividad pendiente es operativa y estadística: mantener el proceso abierto y reunir una muestra prospectiva suficiente; no hay modelos online validados todavía.
- Decisiones: SQLite con transacción por ciclo; modelos incrementales separados de artefactos históricos; modos live/replay incompatibles en una misma base; velas cerradas, horizontes por timestamp exacto y sin cruzar huecos; no simular prospectividad mediante backfill.
- Comparación de Brier pareada, intervalos no solapados, control por multiplicidad y miradas sucesivas, persistencia de inferioridad antes de desactivar. Conservar modelos y evidencia, sin borrado automático.
- Comprobados y corregidos errores durante revisión: fuente retrasada no invalida una etiqueta antes de alcanzar su objetivo; emisión live se comprueba contra tiempo real después de datos/inferencia para no registrar predicciones vencidas; inicialización exige esquema y versión reconocidos antes de escribir en una SQLite existente. Reloj monotónico respecto de último poll y cursor; revisiones de velas cerradas ya persistidas se rechazan. Pruebas específicas confirman rollback total si falla el aprendizaje y exclusión de escrituras fuera del workspace.

## 5. Tareas pendientes

1. Mantener `online run` activo con el protocolo H1 existente y comprobar las siguientes predicciones/confirmaciones mediante `predictions`, `status`, `report` y `verify`. Si se retoma tarde, se resuelven pendientes disponibles y se espera un cierre fresco; no se rellenan predicciones faltantes.
2. Acumular evidencia prospectiva suficiente antes de considerar validado un modelo. Conservar protocolo, ventanas y cohorte; cualquier experimento distinto necesita otra base. No tratar los tres objetivos reales de smoke como validación.
3. Añadir, si se solicita, proveedores live de otros mercados y comparadores adicionales con contratos temporales propios; no asumir que los modelos históricos se volvieron incrementales.
4. Medir rendimiento/almacenamiento en seguimiento extenso y diseñar compactación auditada antes de usar años de datos de un minuto. Automatización como servicio de Windows todavía no instalada.
5. La calidad económica requiere un protocolo independiente de costos, latencia y ejecución hipotética; Brier y precisión no demuestran rentabilidad.

## 6. Historial de desarrollo

### 2026-10-28 — Inicio de aprendizaje en línea y CLI

- Objetivo: implementar la misión solicitada usando exclusivamente el workspace local.
- Inspección: `Get-Location`, `Get-ChildItem -Force`, `rg --files`, `git status --short`, lecturas UTF-8 de memoria, configuración, CLI y módulos compartidos.
- Estado inicial: README modificado y archivos RL sin seguimiento, conservados. `CODEX_PROGRESS.md` existía con historial hasta investigaciones del 17/18 de octubre; se añade esta estructura sin eliminarlo.
- Problema detectado: log legado permite backfill y usa semánticas de costos close/close; no se reutilizará como evidencia prospectiva del nuevo servicio.
- Archivos de implementación nuevos: `config/online.toml`, `src/trading_ml/online/`, `scripts/trading-online.ps1`, `tests/test_online_{models,data,service,cli,regressions}.py`. Modificados CLI, README, pyproject y este documento; eliminado solo `src/trading_ml/dashboard/app.py` tras verificar dependencias.
- Errores de infraestructura: pytest requirió crear el padre de `--basetemp`. La primera regresión global dio 450 aprobadas/2 fallidas porque dos cachés históricas excedían MAX_PATH de Windows (274/280 caracteres) con un directorio temporal largo. Ambas pasaron al usar `artifacts/t`; la repetición global con ruta corta `artifacts/f` dio 455 aprobadas (se habían agregado tres pruebas de latencia). No se modificaron los módulos históricos para sortear ese problema.
- Hallazgo final: una SQLite ajena con tablas de nombres similares podía recibir metadatos antes de fallar. Se reforzó el reconocimiento de esquema/versión antes de escribir; pruebas con y sin tablas similares confirman preservación byte por byte. También se verificaron destinos externos rechazados antes de crear carpetas.
- Comandos principales ejecutados: `python -m pytest -q --basetemp artifacts/f --junitxml artifacts/online_smoke/final_pytest.xml`; `python -m pytest tests/test_online_regressions.py tests/test_online_cli.py tests/test_online_service.py -q --basetemp artifacts/of --junitxml artifacts/online_smoke/final_online_integration.xml`; `python -m ruff check src tests scripts`; `python -m ruff format --check src/trading_ml/online tests/test_online*`; `python -m pip check`; `git diff --check`. Todos con `.venv/Scripts/python.exe`.
- Ejecución real H1: `online run --once --state artifacts/online/live_20261028.sqlite3`, luego copia verificada a `artifacts/online/state.sqlite3` para que funcione el comando por defecto. Ejecución live 1m: protocolo aislado en `artifacts/online_smoke/live_20261028_1m/protocol.toml`, cuatro ciclos a 25 s y un ciclo posterior de recuperación. El proceso de smoke terminó; no hay daemon instalado ni seguimiento que continúe tras esta sesión.
- Los cambios previos del usuario en RL y README se preservaron. No hubo pull/clone/checkout/reset, investigación del remoto, promoción de modelos históricos ni apertura de reservas. Los ajustes de pruebas fueron sintéticos/temporales dentro del proyecto; la cohorte live aprendió solo de sus tres objetivos confirmados.

## 7. Cómo continuar el proyecto

Leer primero esta cabecera, después `git status --short` y los archivos locales. Contrastar la memoria con `config/online.toml`, `src/trading_ml/online/{service,store,evaluation,models,cli}.py`, `scripts/trading-online.ps1` y `tests/test_online_*`. El código local prevalece. La implementación está terminada; **la siguiente acción concreta es arrancar el seguimiento H1 y verificar el primer cierre nuevo de esa base**. La base H1 principal está preparada con 999 velas y ningún pronóstico; el smoke 1m está separado y no debe fusionarse con ella.

```powershell
# Desde la raíz, usando el entorno ya existente:
.\.venv\Scripts\python.exe -m trading_ml online verify
.\scripts\trading-online.ps1 run
# Dejar el proceso activo; Ctrl+C termina limpiamente y el próximo run retoma.
# En otra terminal del mismo proyecto:
.\scripts\trading-online.ps1 status
.\scripts\trading-online.ps1 predictions --limit 20
.\scripts\trading-online.ps1 report
.\scripts\trading-online.ps1 verify
# Copia coherente: elegir SIEMPRE un nombre de destino nuevo.
.\scripts\trading-online.ps1 backup --output artifacts/online/backup_nuevo.sqlite3
.\scripts\trading-online.ps1 restore --backup artifacts/online/backup_nuevo.sqlite3 --state artifacts/online/recuperado_nuevo.sqlite3
# Regresión con ruta corta (los temporales permanecen dentro del proyecto):
.\.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/t
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

El estado por defecto se toma de `online.state_path`. `--state` permite otra base. El wrapper resuelve la raíz aunque se invoque desde otro directorio. Para replay: copiar el protocolo, fijar intervalo/start_at correspondientes al CSV de desarrollo y usar `--source <csv> --replay --state <base_distinta> --once`. Nunca apuntar a reservas selladas. Un cambio de contrato en la misma base se rechaza; no editar metadatos/checksums para forzarlo. No repetir investigaciones históricas, tocar modelos previos, activar órdenes ni reemplazar cambios del usuario. Actualizar este archivo después de cambios, pruebas o problemas importantes y al cerrar la próxima sesión.

## 8. Limitaciones y riesgos

- No hay evidencia nueva de rentabilidad ni modelos online validados con observaciones reales. Los tests de validación/snapshots usan escenarios sintéticos identificados explícitamente.
- Evitar data leakage: no aprender de velas abiertas, etiquetas sin horizonte completo, backfill marcado prospectivo ni resultados de reservas ya examinadas.
- Los resultados históricos son experimentales; selección repetida, sobreajuste, costos omitidos y deriva pueden invalidar ventajas aparentes.
- Forex y modelos históricos conservan sus limitaciones originales. El proveedor online es spot público Binance por polling REST; sin Forex live ni adaptación incremental de los modelos congelados.
- Validación Brier corresponde al proceso incremental, no valida independientemente los pesos finales de un snapshot. Si se congela ese estado para otra estrategia requiere evaluación futura propia. El criterio es conservador y puede permanecer sin evidencia durante mucho tiempo.
- El proceso debe mantenerse activo; cerrar la terminal interrumpe el seguimiento y reanudar no inventa predicciones omitidas. SQLite ofrece recuperación de transacciones y copias verificadas a rutas nuevas; no reconstruye datos corruptos sin copia válida.
- El estado crece sin poda; auditoría y carga del historial por ciclo escalan con su tamaño. H1 es el valor inicial, no se ha validado rendimiento de años de datos de un minuto.
- Se usa reloj UTC local; no hay cotejo automático con `serverTime`. Mantener sincronizado el reloj del equipo. Las copias SQLite deben hacerse con el comando `backup`, que incluye el estado coherente de WAL; no asumir que copiar únicamente el archivo abierto conserva las últimas transacciones.

---

# Registro histórico conservado (anterior a la misión online)

Actualizado: 2026-10-17 (America/Buenos_Aires). Etapa actual: **estudio focalizado de capacidad predictiva y rentabilidad con variables financieras seleccionadas (`20261017T232854_36d523ee`) completado**, superando significativamente (`p_Holm <= 0,040`) tanto a `F0_base` como a controles de ruido de igual dimensión en capacidad predictiva para `ETHUSDT` (`top_core_4_sp500` y `nested_selected`) y `EURUSD` (`vx_liq_core_4` y `vx_core_2`), y mejorando la rentabilidad neta bajo costos reales y estrés 2x en `BTCUSDT` ($\tau = 0,55$, `top_core_3`).

## Etapa actual: estudio focalizado de capacidad predictiva y rentabilidad económica con variables seleccionadas (2026-10-17)
- Implementado y ejecutado `scripts/focused_financial_study.py` → `artifacts/focused_financial_study/20261017T232854_36d523ee/` (`ruff check` limpio, modelos registrados y reservas prospectivas selladas `>= 2025-07-01` intactas `protected_unchanged = true`). Informe comparativo completo en `FOCUSED_FINANCIAL_FEATURES_REPORT.md`.
- Evaluados exclusivamente los subconjuntos más prometedores por activo sobre walk-forward purgado (`2023-01-01..2025-07-01`, 3 folds con embargo de 6 días y recorte de horizonte al cierre de fold: `876` filas OOS en `BTCUSDT`/`ETHUSDT` y `589` en `EURUSD`), con selección anidada interna (`2 × 120` días en `train`), controles con ruido gaussiano causal de igual dimensión (`noise_k{k}_mean` y `noise_nested_match`), bootstrap por bloques de 30 días (`2.000` réplicas, corrección Holm-Bonferroni) y simulación económica con costos reales y estrés 2x (`simulate_events` en Cripto: 34 bps base / 68 bps doble round-trip; `FxEngine.portfolio` sobre barras H1 bid/ask reales FXCM en `EURUSD`):
  1. **`ETHUSDT`:** `top_core_4_sp500` (`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`, `fin_vol_parkinson_ratio`, `fin_cross_sp500_mom_z`) supera simultáneamente a `F0_base` y a `noise_k4_mean` en los 3 modelos, ganando en `3/3` folds en `RandomForest` (`s42, s1729`) y `XGBoost` (`s1729`). En `RandomForest` (`s1729`), `log-loss = 0,69159` (`ΔLoss vs Base = -0,00647`, `p_Holm = 0,0045`; `ΔLoss vs Ruido = -0,00419`, `p_Holm = 0,0090`) y `ROC-AUC = 0,53938` (`ΔAUC vs Base = +0,02438`, `p_Holm = 0,0090`; `ΔAUC vs Ruido = +0,02092`, `p_Holm = 0,0045`). En `XGBoost` (`s42`), `nested_selected` alcanza `ROC-AUC = 0,54258` (`+0,02884` vs Ruido, `p_Holm = 0,018`) y `Accuracy = 53,88%`. En `LogisticRegression`, `top_core_4_sp500` alcanza `ROC-AUC = 0,55870` (`+0,05026` vs Ruido, `p = 0,006`, `p_Holm = 0,054`) y `Accuracy = 54,91%` (`+4,57 pp`). Económicamente (`base_cost`), revierte las pérdidas de `F0_base` a retornos positivos (`logistic_l2`: `+160,47%`, `Sharpe = 1,073` en `inner_tau = 0,53`; `xgboost`: `+14,23%` y `+10,77%` en $\tau = 0,55$; `random_forest` `nested_selected`: `+15,93%`, `Sharpe = 0,341`), pero sigue siendo sensible al estrés `double_cost` por exceso de rotación (`136..461` trades).
  2. **`BTCUSDT`:** `top_core_3` (`fin_liq_corwin_schultz_20`, `fin_liq_range_impact_z63`, `fin_vol_parkinson_ratio`) y `top_core_4_sp500` mejoran `log-loss` y `ROC-AUC` frente a base y ruido en `RandomForest` (`ROC-AUC = 0,55659` en `s42`, `p = 0,0275` vs Ruido; `3/3` folds en `s1729`, aunque `p_Holm >= 0,0765`), y `top_core_3` es la única receta que mejora en ambos seeds de `XGBoost` frente a base y ruido. En simulación económica ($\tau = 0,55$), `top_core_3` logra ventaja robusta incluso bajo costos dobles: en `logistic_l2` sube de `+33,10%` (`Sharpe = 0,839`, `MaxDD = -10,93%`) en `F0_base` a **`+45,29%` (`Sharpe = 1,165`, `MaxDD = -8,92%`, `PF = 2,750`)** en `base_cost` y **`+28,24%` (`Sharpe = 0,808`, `PF = 1,940`)** en `double_cost`; en `xgboost` (`s42`) sube de `+24,26%` (`Sharpe = 0,538`) en `F0_base` y `+5,20%` en `noise_k3_mean` a **`+43,37%` (`Sharpe = 0,822`, `MaxDD = -13,69%`, `PF = 1,396`)**, manteniéndose positivo en `double_cost` (`+5,45%`).
  3. **`EURUSD`:** `vx_liq_core_4` (`fin_basis_vx_z63`, `fin_basis_vx_term`, `fin_liq_spread_to_range`, `fin_liq_corwin_schultz_20`) en `XGBoost` (`s1729`) reduce `log-loss` a `0,69264` (`-0,00920` vs Base, `p_Holm = 0,0035`; `-0,00535` vs Ruido, `p = 0,0285`) y eleva `ROC-AUC` a `0,53038` (`+0,04807` vs Base, `p_Holm < 0,001`; `+0,04200` vs Ruido, `p_Holm = 0,021`; `0,53357` en `s42` con `3/3` folds). En `RandomForest` (`s42`), `vx_core_2` supera al ruido con `p_Holm = 0,0175` (loss) y `p_Holm = 0,0315` (AUC). En cambio, `fx_liquidity_only` sola sobreajusta (`ROC-AUC 0,46729..0,47387` en `RandomForest`). En `FxEngine.portfolio` (long-only con swap/spread horario), la mejora predictiva reduce pérdidas en `XGBoost` y logra retornos positivos moderados en `RandomForest` ($\tau = 0,55$, `nested_selected` `+3,57%`, `Sharpe = 0,705`, `+2,44%` en `double_cost`), pero requiere ejecución simétrica long/short y control de carry antes de abrir la reserva sellada.

## Etapa previa del 2026-10-17: ablación controlada de variables financieras y controles de ruido (conservada)
- Implementado y ejecutado `scripts/financial_feature_ablation.py` → `artifacts/financial_feature_ablation/20261017T224928_57040c88/` (`ruff check` limpio, modelos registrados y reservas prospectivas intactas `protected_unchanged = true`, test explícito de invariancia causal ante perturbaciones futuras `max_abs_diff = 0,0` en los tres activos).
- Evaluadas **6 familias financieras** ($k=4$ variables por familia) según disponibilidad histórica (`< 2025-07-01`) sobre las mismas observaciones walk-forward (`2023-01-01` a `2025-07-01`, 3 folds con purga y embargo de 6 días: `909` filas OOS en `BTCUSDT`/`ETHUSDT` y `614` en `EURUSD`) y **4 modelos existentes** (`logistic_l2`, `xgboost`, `random_forest`, `lightgbm`, semillas `42` y `1729`), incluyendo adición individual (`add_*`), controles con ruido gaussiano causal de igual dimensión (`noise_add4_mean`, `noise_all_mean`, `noise_lofo_s1`), ablación *leave-one-family-out* (`all_without_*`), selección anidada interna (`nested_noise_screened`) y permutación por bloques OOS (7 y 30 días):
  1. **`liquidity` (Liquidez e impacto de precio):** única familia que mejora simultáneamente `log-loss` y `ROC-AUC` en `BTCUSDT` frente a `F0_base` **y** frente al control de ruido `noise_add4_mean` en `RandomForest` (`log-loss = 0,69582`, `ΔLoss vs Base = -0,00255`, `ΔLoss vs Ruido = -0,00218`; `ROC-AUC = 0,54148` en s42 y `0,54233` en s1729), `XGBoost` (`log-loss = 0,69870`, `ΔLoss vs Base = -0,00312`, `ΔLoss vs Ruido = -0,00260`; `ROC-AUC = 0,53120`) y `LightGBM` (`log-loss = 0,70193`, `ROC-AUC = 0,53016`), además del mayor `ROC-AUC` en `LogisticRegression` (`0,54549`). En `ETHUSDT`, `add_liquidity` en `RandomForest` logra IC95% bootstrap (bloque 30d) completamente favorables frente a base en ambas semillas (`s42`: `ΔLoss = -0,00282 [-0,00666; -0,00069]`, `ΔAUC = +0,01533 [+0,00302; +0,03637]`), y quitarla en LOFO es la ablación más dañina en `XGBoost` (`+0,00458` log-loss) y `LightGBM` (`+0,00480`).
  2. **`futures_basis` (Basis perpetuo, trimestral COIN-M `BTCUSD`/`ETHUSD` y futuros `VX` sobre `^VIX`):** en `EURUSD`, el basis de futuros de volatilidad CBOE `VX` frente a `^VIX` supera tanto a `F0_base` como a `noise_add4_mean` en `XGBoost` (`s42`: `log-loss = 0,69632`, `ΔLoss vs Base = -0,00389`, `ΔLoss vs Ruido = -0,00101`, `ROC-AUC = 0,50625`; `s1729`: `ΔLoss vs Ruido = -0,00216`, `ΔAUC vs Ruido = +0,01595`) y `RandomForest` (`s42`: `log-loss = 0,69522`, `ROC-AUC = 0,50687`; `s1729`: `ΔLoss vs Base = -0,00477 [-0,00975; -0,00133]`). La selección interna `nested_noise_screened` en `EURUSD` `XGBoost` logra `log-loss = 0,69586` (`-0,00435` vs Base, `-0,00147` vs Ruido) y `ROC-AUC = 0,51623` (`+0,02447` vs Base, `+0,01225` vs Ruido).
  3. **Variables individuales con permutación OOS positiva en `3/3` folds:** `fin_liq_corwin_schultz_20` (BTC y ETH en los 3 modelos de árboles), `fin_liq_range_impact_z63` (BTC y ETH), `fin_cross_sp500_mom_z` (ETH en los 3 modelos de árboles), `fin_vol_parkinson_ratio` (ETH y BTC) y `fin_basis_vx_z63` (EURUSD).
  4. **Familias descartadas en D1 ($h=1$):** `funding_rates`, `relative_volume` y la acumulación completa `all_financial` aumentan la brecha de generalización train-OOS y pierden contra controles de ruido de igual dimensión. Documentado en `FINANCIAL_FEATURE_ABLATION_REPORT.md`.

## Etapa previa del 2026-10-17: cierre de análisis de reentrenamiento y expansión de mercados (conservada)
- Recuperados y verificados sin repetir entrenamientos (`new_fits = 0`):
  1. Comparación controlada de extensión histórica `artifacts/retraining_comparison/20261017_controlled` (**441 ajustes**, **32.658 archivos previos intactos por SHA-256**).
  2. Estudio de expansión multimercado/derivados/flujo `artifacts/expansion/studies/20261016T211432_bcec9ebe` (**72 ajustes completados**, **768 archivos verificados**, **234 carteras reconstruidas**).
  3. Estudio Forex en nuevos pares `artifacts/expansion/forex_studies/20261016T210703_c9107a5c` (**24 ajustes completados**, **123 archivos verificados**, **24 modelos y 24 carteras reproducidos**).
- Scripts de análisis verificados (`ruff check` limpio) y ejecutados sobre artefactos congelados:
  - `scripts/analyze_expansion_study.py` → `artifacts/expansion_analysis/20261016T211432_bcec9ebe/20261017T221013/` (error máx. de reproducción de probabilidades XGBoost `2,98e-8`).
  - `scripts/analyze_expansion_forex.py` → `artifacts/expansion_analysis/20261016T210703_c9107a5c/20261017T221546/`.
- Informes actualizados y generados: `RETRAINING_COMPARISON_REPORT.md` (sección I añadida con ablaciones de 72 y 24 ajustes y tabla consolidada) y `FINAL_MODEL_IMPROVEMENT_REPORT.md` (síntesis ejecutiva y respuestas a las 6 preguntas).
- Conclusión verificable: **no se demostró mejora estadísticamente significativa ni ventaja predictiva robusta**. En BTC D1 el mejor modelo direccional nuevo descriptivo es `random_forest_cross` (`AUC 0,54202 → 0,55319`, `Accuracy 53,14% → 54,39%`, `log-loss 0,68949 → 0,68788`, `p_Holm = 1,0`, retorno neto `−7,08%`); en el estudio de expansión multimercado, el control con ruido gaussiano causal (`expanded_noise`) obtuvo mayor AUC (`0,52754`) y menor log-loss (`0,69356`) que todas las variantes con datos reales nuevos. Reservas prospectivas selladas intactas.

## Etapa previa del 2026-10-17: adquisición histórica, sin entrenamiento (conservada)
- Este pedido reemplaza la siguiente acción de entrenamiento de la etapa anterior: **no ejecutar los 72 ajustes pendientes ni nuevas corridas**. Modelos, checkpoints, protocolos y resultados anteriores se conservan.
- Implementados `config/historical.toml` y `trading_ml.data.history{,_sources,_cli}`: adquisición reanudable, caché de originales con SHA256, snapshots inmutables, inventario y verificación reproducible.
- Universo solicitado: 12 spot, 6 perpetuos, contratos trimestrales individuales BTCUSD/ETHUSD, 10 pares Forex y 8 índices diarios. Cripto desde el primer archivo real publicado desde 2017; Forex desde 2012; índices desde 1990, hasta el último mes cerrado solicitado (fin exclusivo 2026-10-01). La cobertura efectiva se informa por dataset, no se presume igual a la solicitada.
- Forex M15 se agrega desde M1 oficial FXCM; H1/H4/D1 usan exclusivamente ventanas UTC completas. No se interpolan precios, volúmenes ni sesiones parciales. D1 UTC se distingue del D1 NY17 de los experimentos anteriores.
- Se excluyen y registran observaciones Binance parciales/desalineadas durante mantenimiento; originales preservados y huecos explícitos. Índices no negociables se identifican como contexto diario; nunca se convierten en futuros o velas intradiarias.
- `load_history` exige fin explícito y bloquea datos posteriores a 2025-07-01 para desarrollo, además de aplicar las reservas prospectivas existentes. Exportación opt-in compatible con el pipeline spot y adaptador H1 bid/ask para el multimercado; sin modificar fuentes históricas ni configuraciones de estudios anteriores.
- **Adquisición y validación terminadas, sin entrenamientos.** Inventario activo: `artifacts/historical/validated_20261017.json`; comprobante: `artifacts/historical/validated_verification_20261017.json`; puntero: `artifacts/historical/latest.json`. Resumen legible: `HISTORICAL_DATA_RESULTS.md`; comandos/contratos: `HISTORICAL_DATA.md`.
- **491 datasets, 11.749.803 registros**, de los cuales **8.983.894 son observaciones base** sin volver a contar las temporalidades derivadas. 10.197.144 filas de desarrollo antes de purgar etiquetas y **1.552.659 filas posteriores a 2025-07-01 en cuarentena**. No seleccionar modelos con esos datos ni habilitar las reservas anteriores.
- Cobertura efectiva: 10 pares Forex desde 2012 (nuevos EURJPY/GBPJPY/EURGBP), 12 spot desde sus altas desde 2017 (nuevos ADA/DOGE/LINK/LTC/DOT/AVAX/BCH), 6 perpetuos desde 2020, **50 contratos trimestrales BTCUSD/ETHUSD** desde 2020, 8 índices diarios con OHLC verificable desde 1962 según activo y 161 futuros VX desde 2013. Fin solicitado 2026-10-01 exclusivo; cada manifiesto conserva extremos/huecos reales.
- SOL perpetuo reparado con **480 M15 reales de cinco archivos diarios oficiales** (2022-02-26/27/28 y 2022-04-01/02). Snapshot nuevo sin huecos desde 2020-09-14 hasta 2026-10-01; originales y snapshots previos intactos. Comprobantes en `artifacts/historical/repairs/` y `sol_repair_verification.json`.
- Exclusiones trazables: 444 observaciones Binance parciales/inválidas; 21.050 OHLC de índices inválidos o con rango diario no verificable; 1.232 filas Cboe no negociadas/inválidas. Además, **27.707 M15 de la cola posterior a la liquidación programada** de futuros COIN-M quedaron fuera del uso (36.629 filas contando temporalidades derivadas). Incluye precios congelados y registros con volumen de settlement/horario ambiguo: no asumir entrega diferida real ni operaciones ordinarias; originales preservados. Control y reparación: `scripts/bound_history_futures.py` y `artifacts/historical/delivery_bounds/`.
- Forex D1 UTC desde M1 estricto resultó escaso: se incorporaron **25.928 D1 nativas NY17, 2012–2021**, como intervalo distinto `1d_ny`, sin cambiar relojes ni mezclarlas con UTC. Los archivos anuales 2022–2026 devolvieron 404. Las D1 NY17 de experimentos anteriores también permanecen intactas.
- Verificación final: **9.741 originales con checksum comprobado**, **234 agregaciones entre temporalidades reconstruidas**, 291 datasets sin cambios revalidados por SHA256 de CSV/manifiesto y 200 datasets de futuros revalidados tras depurar vencimientos. **40.469 artefactos anteriores conservados por tamaño/mtime**, incluyendo modelos/checkpoints/resultados; se excluye únicamente el log mutable creado por esta adquisición. Sin promociones ni evaluaciones independientes.
- Futuros VIX reales incorporados: **161 contratos mensuales individuales, 28.696 observaciones diarias**, 2013-05-20 a 2026-09-16 (disponibilidad hasta 2026-09-17 UTC), obtenidos del listado/archivos oficiales Cboe. Sin series continuas ni sustitución de OHLC no negociados por settlement.
- Extensión completa de índices con `config/historical_indices.toml`: **82.707 filas OHLC diarias depuradas**. Se detectaron décadas de OHLC idénticos en historia temprana del vendor; sus rangos no verificables se excluyen y el loader impide utilizarlos. El inventario completo reemplaza las versiones iniciales de índices al consolidar, verificando el tramo válido común.
- **80 pruebas de datos, proveedores y compatibilidad aprobadas**, sin ajustes de modelos; Ruff/formato correctos en los archivos de esta ampliación. Evidencia `artifacts/historical/data_tests_final.txt`. Integración real comprobada: BTC H1 2024-01 a 2025-07 produjo 13.079 filas preparadas; EURUSD H1 2012-01 a 2025-07 produjo 74.434 filas bid/ask compatibles. Evidencia `artifacts/historical/adapters/data_adapter_smoke.json`; sólo se escribieron entradas nuevas.
- Limitaciones: Forex indicativo con huecos/cotizaciones cruzadas diagnosticadas, sin volumen global ni swaps históricos; D1 UTC estricto escaso y nativo NY17 hasta 2021; índices/VX sólo diarios. Perpetuos requieren funding/costos y futuros individuales requieren un protocolo causal de roll/margen antes de experimentar. No se adquirieron ES/NQ/GC/CL licenciados ni se fabricaron sustitutos.
- Próxima acción: revisar el inventario/definir un protocolo temporal si el usuario lo solicita. **No entrenar ni reanudar los ajustes pendientes sin un nuevo pedido explícito.** Los snapshots `source.zip` y protocolos históricos permiten reproducir los experimentos anteriores con su código original.

## Registro de la etapa anterior (conservado)

## Etapa activa: expansión de mercados, datos y features
- Pedido nuevo: ampliar diversidad de instrumentos y fuentes; comparar aporte incremental para DMN/XGBoost, con disponibilidad temporal verificable, costos y reservas protegidas.
- Estado inicial comprobado: HEAD `ded6660`, mismos cambios previos ajenos en README/CLI y estructura/auditoría. No se repetirán las 120 pruebas históricas de modelos como si fueran evidencia nueva.
- Antes de investigación/cambios se abre este checkpoint. Próxima acción: inventariar contratos/cobertura de datos existentes sin abrir resultados reservados, investigar fuentes oficiales y fijar protocolo de expansión/ablaciones antes de entrenar.
- Plan de trabajo: catálogo explícito spot/FX/CFD/ETF/futuros; adquisición versionada con checksums; features de mercado y datos externos con `available_at`; evaluación temporal sobre filas comparables, controles por activo/semilla y resultados positivos/negativos.
- El mensaje recibido termina en «Desequilibrios de mercado»; el alcance implementable ya está claro y se continúa con lo disponible.
- Avance: fuentes oficiales investigadas, catálogo/tipos y protocolo en `MARKET_EXPANSION.md`; `config/expansion.toml` fija 72 ajustes (6 variantes × LSTM/XGB × 2 semillas × 3 folds). `config/expansion_forex.toml` fija 24 ajustes adicionales de los dos pares nuevos.
- Se adquirieron USDCAD/NZDUSD H1 y D1 2020-01 a 2025-07 con FXCM, completando siete pares. Índice en `artifacts/expansion/forex_acquisition.json`.
- Forex **24 ajustes completados**: `artifacts/expansion/forex_studies/20261016T210703_c9107a5c`; todavía pendiente verificación y análisis agregado. Sin promoción ni prueba independiente.
- Cripto en adquisición: cinco spot + perp negociado + funding, 2021-01 a 2025-07. BTC/ETH completos; SOL perp detectó huecos. Caché content-addressed permite continuar sin perder descargas. Se investiga recuperación desde archivos diarios oficiales, sin interpolar.
- Mark-price BTC rechazado antes del ajuste: 8 días ausentes; evidencia en `artifacts/expansion/source_rejections/mark_BTCUSDT_20210101_20250701.json`. Se usa prima de cierre **negociado** perp/spot, con cambio documentado antes del entrenamiento cripto.
- Piloto OI de un día adquirido y validado (`artifacts/expansion/data/BTCUSDT/oi_pilot_*.csv`); no entra a modelos ni se extiende ficticiamente.
- Software nuevo: `src/trading_ml/expansion/{data,features,catalog,study,forex,__main__,__init__}.py`; features por disponibilidad, revisiones macro seguras, flujo/funding/prima/contexto. DMN admite contratos dinámicos explícitos preservando schema anterior. Registro de dos nuevos pares en `multimarket/markets.py`.
- Línea base real: 264 pruebas previas aprobadas (100,04 s). Pruebas dirigidas nuevas + DMN: **18 aprobadas**. Falta test integrado del estudio ampliado y suite completa final.
- Próxima tarea inmediata de esta etapa: resolver huecos SOL con fuentes diarias oficiales o documentar rechazo; completar adquisición de BNB/XRP, probar runner integrado, ejecutar 72 ajustes predeclarados, verificar contabilidad y comparar con control ancla/ruido. No modificar código durante corridas: el resume exige hash idéntico.

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

## Dashboard Web Minimalista de Inferencia en Tiempo Real (2026-10-17)
- Implementado en `src/trading_ml/dashboard/` (`models_catalog.py`, `live_feed.py`, `inference_engine.py`, `prospective_log.py`, `app.py`) usando Streamlit + Plotly y WebSocket/REST en vivo de Binance.
- Reutiliza el registro de modelos (`artifacts/models/v*`, modelos XGBoost `BTCUSDT` `1d` con verificación SHA-256) y los modelos direccionales congelados (`artifacts/frontier_studies/20261014T134646_e17d06e3/frozen/` para `BTCUSDT` y `ETHUSDT` `1d`), sin modificar ningún modelo entrenado, dataset histórico ni abrir tests sellados.
- Separa estrictamente las **velas cerradas** (`timestamp + delta <= now_utc`) de la **vela en formación** (`is_closed=False`), garantizando que las features (`build_features` / `multi_asset_features`) y predicciones definitivas (`SUBE`, `BAJA`, `SIN SEÑAL`) solo utilicen velas cerradas posteriores a `selection_label_end`.
- Registra cada predicción prospectiva de manera idempotente en `artifacts/live_dashboard/prospective_predictions.jsonl` (separado del backtesting histórico) y evalúa automáticamente el resultado (`CORRECTO` / `INCORRECTO` / `SIN SEÑAL`) una vez que cierra la vela objetivo.
- Comandos de ejecución y pruebas:
  - `.\.venv\Scripts\python.exe -m trading_ml dashboard` (o `.\.venv\Scripts\streamlit.exe run src/trading_ml/dashboard/app.py`)
  - `.\.venv\Scripts\pytest.exe tests/test_dashboard.py -v`

## Próxima tarea exacta
Al retomar, ejecutar `git status --short` y verificar ambos estudios con `python -m trading_ml.dmn verify --study artifacts/dmn_studies/<id>`; contrastar con este registro. **No repetir los 120 ajustes ni modificar los protocolos terminados para buscar mejores métricas.** La próxima investigación exacta es inventariar los manifiestos/cobertura/calendario de los datos Forex existentes en `artifacts/multimarket` y las condiciones de datos de futuros diversificados, sin abrir reservas ni sus métricas; documentar cuál permite una comparación DMN válida. Después diseñar un adaptador de panel/calendario/ejecución apropiado y predeclarar el siguiente estudio. La evaluación independiente permanece pendiente, y no se justifica activar un modelo con los resultados actuales. Los snapshots `source.zip` permiten continuar una corrida con código original si una sesión posterior cambia `src`.



## Investigación de ventaja predictiva y económica — cierre 2026-10-17

- Pedido atendido: auditoría de modelos/errores, nuevas temporalidades 1h/4h además de diario, señales causales, filtros/probabilidades/riesgo y validación económica rigurosa. Investigación aislada en `src/trading_ml/edge_research/`; no modifica contratos ni modelos históricos.
- Protocolo previo en `config/edge_research.toml` y `EDGE_RESEARCH.md`. Datos locales oficiales BTC/ETH desde 2022 hasta junio de 2025, sin usar señales/etiquetas desde julio de 2025. Huecos excluidos sin imputación; reinicio de warmup. Tres folds exteriores, 90 días de validación disjunta y embargo de 48h por fecha de salida ejecutable.
- Corrida terminada: `artifacts/edge_research/runs/20261018T015816_ca7033e2`. **120 ajustes reales: 96 estimadores y 24 calibradores**, sin búsquedas de hiperparámetros ni repeticiones de entrenamiento por los problemas de verificación. Cuatro recetas: logística precio, completa, ruido y XGBoost completo calibrado. Tareas: 1h→1h, 1h→4h, 4h→4h, diario→24h.
- Resultado: **5/32 contrastes predictivos pasan Holm**, todos contra p=0,5; **0/16 contrastes incrementales frente a precio/ruido**, **0/16 económicos** y **0/16 candidatos aceptados**. Hay señal direccional retrospectiva débil, sin ventaja económica robusta. BTC 4h logística completa: AUC 0,55811, +115,46% bruto → −93,70% neto a aproximadamente 34 bps por round trip. El filtro de expectativa queda en cash en intradía; cash no demuestra rentabilidad.
- Evidencia final: `EDGE_RESEARCH_RESULTS.md`; análisis CSV y gráficos PNG/SVG inspeccionados en `artifacts/edge_research/analyses/20261018T015816_ca7033e2_20261018T023347`.
- Auditoría independiente aprobada: `artifacts/edge_research/verification/20261018T015816_ca7033e2_20261018T023343.json`. **750 archivos nuevos verificados, 75.136 archivos anteriores intactos, 60.128 filas de predicción, 96 modelos recargados y 288 carteras reconstruidas desde precios con contabilidad independiente**. Cero nuevos ajustes en la auditoría; ninguna reserva consumida ni modelo promovido.
- Precisión: CSV float32 de XGBoost difiere hasta 2,98e-8 al leerlo como float64, pero recupera exactamente la probabilidad nativa. Error máximo nativo global 8,33e-17. El verificador compara dtype nativo y comprueba identidad de cash/acciones por bloques, sin cancelación numérica cuando el capital se aproxima a cero. Recibos de las pruebas en `numerical_verifier_checks.json` y `float32_probability_checks.json`, sin alterar resultados originales.
- Software: suite completa 321 pruebas aprobadas; 10 pruebas dirigidas finales aprobadas, incluyendo causalidad, cuarentena, purga, costos, demora, bootstrap, contabilidad independiente y precisión nativa. Los conteos se superponen y no deben sumarse. Ruff lint global y formato de los archivos de esta tarea correctos; formato global pendiente en nueve archivos preexistentes, conservados. Pip check correcto.
- Dashboard: vista de investigación de solo lectura en `dashboard/edge_candidates.py` y `app.py`, con hashes y 16 decisiones comprobadas sobre datos reales (`dashboard_catalog_check.json`). No incorpora estos modelos a inferencia activa ni a un log prospectivo. `shadow_manifest.json` conserva estados, modelos y contratos para revisión posterior; todos quedan sin rentabilidad confirmada.
- Hallazgo de auditoría: el log legado del dashboard usa 14 bps y close/close y permite backfill con `seed_recent_closed_predictions`. No se usó como prueba prospectiva ni como ledger económico de las nuevas recetas; los logs previos permanecen intactos. Corregir/separar esas semánticas antes de una confirmación prospectiva.
- Próximos pasos: no repetir estos 120 ajustes ni elegir nuevos filtros sobre estos folds. Medir costos/latencia mediante quotes registradas; predeclarar una hipótesis de magnitud condicional/retorno neto y confirmarla en una ventana futura disjunta de todas las reservas. El indicio anterior de BTC diario del estudio financiero se conserva como hipótesis separada, sin declararlo rentable.
- Verificar sin entrenar: `.venv/Scripts/python.exe scripts/verify_edge_research.py artifacts/edge_research/runs/20261018T015816_ca7033e2`. Los snapshots originales y el paquete final de herramientas conservan versiones y hashes; no sobrescribirlos.
