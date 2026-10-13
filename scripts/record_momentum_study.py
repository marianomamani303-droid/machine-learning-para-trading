"""Publish an existing immutable study report to project documentation; never trains."""

from __future__ import annotations

import argparse
from pathlib import Path

from trading_ml.common import read_json, sha256
from trading_ml.errors import DataError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("MOMENTUM_RESEARCH.md"))
    args = parser.parse_args()
    manifest = read_json(args.study / "artifacts.json")
    for relative, expected in manifest.items():
        if sha256(args.study / relative) != expected:
            raise DataError(f"Artefacto Momentum modificado: {relative}")
    summary = read_json(args.study / "summary.json")
    report = (args.study / "REPORT.md").read_text(encoding="utf-8")
    usage = f"""
## Reproducir e integrar

Ejecutar desde la raíz del proyecto:

```powershell
.\\.venv\\Scripts\\python.exe -m trading_ml momentum-study --protocol config/momentum_study.toml
.\\.venv\\Scripts\\python.exe scripts/record_momentum_study.py {args.study.as_posix()}
```

Cada ejecución crea un directorio nuevo; nunca reemplaza experimentos, modelos
actuales ni punteros. La comparación de referencias requiere las tres versiones
registradas en el TOML; para otro proyecto indicar versiones existentes o una
lista vacía y registrar esa diferencia. Modificar el protocolo abre una hipótesis
nueva; observar repetidamente el mismo historial no crea un test independiente.

```python
import pandas as pd
from trading_ml.momentum import (
    MomentumConfig,
    ResearchCosts,
    build_momentum_features,
    momentum_signals,
)
from trading_ml.momentum.simulator import simulate_policy

# frame: OHLCV, timestamp UTC de apertura y reference_time UTC de cierre.
# El proveedor externo debe validar precios, cierres y calendario de sesiones.
config = MomentumConfig(lookback=60)
features = build_momentum_features(frame, config)
signals = momentum_signals(features, config)
simulation = simulate_policy(
    features,
    signals,
    pd.Timestamp("2025-06-05", tz="UTC"),
    pd.Timestamp("2026-10-01", tz="UTC"),
    ResearchCosts(),
    config.max_holding_bars,
)
```

El paquete `trading_ml.momentum` mantiene separados config, features, reglas,
simulador, estudio y reporte. `opportunity_labels` crea etiquetas investigativas;
`fit_momentum_filter` ajusta sólo las etiquetas con salida anterior al cutoff.
Los modelos de `frozen/L*/model.ubj` son XGBoost nativos; `fit.json` y `plan.json`
guardan contrato, hashes, corte y umbrales. Un modelo que se abstuvo no tiene UBJSON.

Evidencia de esta ejecución: `{summary["run_id"]}`; {summary["experiment_records"]}
registros y {summary["actual_xgboost_fits"]} ajustes XGBoost. La ejecución técnica
anterior, si existe en artifacts/momentum_studies, también permanece conservada;
ninguna ejecución cambia los horizontes en respuesta a sus resultados.

La reserva futura también queda protegida por `protect_reserved_labels`, usada
por los comandos actuales de entrenamiento. No se ejecutó una evaluación futura
porque los datos todavía no existen. Reusar esos datos para entrenar invalidaría
su condición de test; el guard rechaza etiquetas que los invadan.
"""
    args.output.write_text(report + usage, encoding="utf-8")
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
