# Experimento acotado: reglas y DQN con la misma señal

Diseño previo: [RL_SIGNAL_PROTOCOL.md](RL_SIGNAL_PROTOCOL.md).
Configuración: [config/rl_signal.toml](config/rl_signal.toml).
Implementación aislada: `src/trading_ml/rl_signal/`; no modifica modelos anteriores
ni integra el agente a operación real o al dashboard.

Usa el entorno Python existente y su dependencia opcional `dmn` (PyTorch CPU),
además de numpy, pandas, scikit-learn y joblib. Cada corrida conserva versiones
exactas en `requirements-freeze.txt` y `plan.json`, así como una copia del código
en `source.zip`. No descargar otros mercados para este experimento.

```powershell
# Pruebas de causalidad, ejecución, cuenta persistente y DQN:
.venv/Scripts/python.exe -m pytest tests/test_rl_signal.py -q

# Evaluación histórica autorizada; crea otra corrida, nunca sobrescribe:
.venv/Scripts/python.exe -m trading_ml.rl_signal.study run --protocol config/rl_signal.toml

# Verificación sin volver a ajustar modelos (reemplazar ID):
.venv/Scripts/python.exe -m trading_ml.rl_signal.study verify --study artifacts/rl_signal/ID
```

No repetir el entrenamiento para seleccionar una semilla o resultado mejor.
El comando `verify` comprueba hashes, recarga los modelos, verifica normalizadores
contra train, la elección interna de checkpoints y las fechas del replay, vuelve
a inferir acciones sin aprender, reconstruye el efectivo y las unidades desde
flujos firmados y repite bootstrap/decisión sin ajustes. La verificación tampoco
modifica la corrida. Guardar su salida en un recibo separado.

Los datos se localizan mediante el índice de expansión ya existente. Se exige
el SHA256 del snapshot BTC spot y sus metadatos; falta de datos o integridad
inválida bloquea la corrida. En otro equipo debe restaurarse el snapshot y su
índice con rutas válidas, manteniendo el contenido y hashes de fuentes. No
crear precios artificiales ni habilitar el período en cuarentena.

Una corrida conserva:

- Protocolo fechado, fuente y hashes, reservas anteriores y entorno.
- `fold_N/{train,validation,test}_predictions.csv`: variables, probabilidades,
  etiquetas, precios y tiempos; sólo cinco componentes permitidos llegan al agente.
- Modelos supervisados por bloque y contratos con último cierre/etiqueta de train.
- `fold_N/seed_S/`: seis checkpoints, elegido, scaler, replay train y validaciones.
- Carteras y órdenes por escenario, fold y semilla; status de mínimos y polvo final.
- CSV de métricas agregadas, por fold, predictor, contrastes y CI por semilla.
- `decision.json`, `report.md`, `preservation.json`, `prospective.json`, manifiesto.

Los artefactos se excluyen de Git por la política existente del repositorio:
respaldar `artifacts/rl_signal/` junto con el código. El informe principal enlaza
la corrida efectiva. Un hash no demuestra autenticidad ni independencia del
mercado; conserva la trazabilidad local.

El protocolo prospectivo propone noviembre de 2027 a octubre de 2028 y una
sola evaluación al cierre. Sus pesos se fijan sin elegir semilla con test.
No hay scheduler, órdenes reales ni resultados futuros. Antes del inicio falta
registrar un feed de cotizaciones y el contrato de emisión/ejecución real:
el precio de apertura idealizado del backtest no puede atribuirse a una señal
que se emitió después de esa apertura. Sin ese registro no se evaluará ventaja
económica prospectiva. Ver los criterios completos en el protocolo.
