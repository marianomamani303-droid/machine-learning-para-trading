"""Record an existing immutable Mean Reversion study in project documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from trading_ml.common import read_json, sha256
from trading_ml.errors import DataError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("MEAN_REVERSION_RESEARCH.md"))
    args = parser.parse_args()
    for relative, digest in read_json(args.study / "artifacts.json").items():
        if sha256(args.study / relative) != digest:
            raise DataError(f"Artefacto de reversión modificado: {relative}")
    summary = read_json(args.study / "summary.json")
    usage = f"""
## Reproducir e integrar

```powershell
.\\.venv\\Scripts\\python.exe -m trading_ml mean-reversion-study --protocol config/mean_reversion_study.toml
# Registrar evidencia existente: no entrena ni vuelve a evaluar el test.
.\\.venv\\Scripts\\python.exe scripts/record_mean_reversion_study.py {args.study.as_posix()}
```

Cada ejecución crea un directorio nuevo en artifacts/mean_reversion_studies.
Se conservan todas las variantes, errores, pérdidas, abstenciones y casos censurados.
El protocolo ejecutado está copiado en protocol.toml y plan.json antes de ajustar
modelos; no se cambia en respuesta a los resultados.

```python
import pandas as pd
from trading_ml.mean_reversion import (
    ResearchCosts,
    ReversionConfig,
    build_reversion_features,
    reversion_opportunities,
)
from trading_ml.mean_reversion.simulator import simulate_reversion

# frame: OHLCV válido, timestamp UTC de apertura y reference_time UTC de cierre.
# Los adaptadores externos validan cierres, sesiones, lotes y contratos.
config = ReversionConfig(lookback=20, sideways_only=False)
features = build_reversion_features(frame, config)
opportunities = reversion_opportunities(features, config)
result = simulate_reversion(
    features,
    opportunities,
    pd.Timestamp("2025-06-05", tz="UTC"),
    pd.Timestamp("2026-10-01", tz="UTC"),
    config,
    ResearchCosts(),
)
```

`models.fit_filter` permite ajustar logística/XGBoost sobre etiquetas purgadas;
`models.load_filter` carga JSON/UBJSON y `filter.probabilities(features)` estima
la reversión. Pasar esas probabilidades a `simulate_reversion` sólo filtra entradas.
Para evaluación fuera de muestra, exigir referencias posteriores al corte guardado
y conservar los umbrales del plan. El runner aplica estos controles temporalmente;
el API de bajo nivel permite pruebas sintéticas y exige que el llamador mantenga
el contrato fuera de muestra.

Los módulos config, features, rules, simulator, models, statistics, study y report
son independientes del proveedor y no contienen envío de órdenes. Reutilizan costos,
validación de timeline y agregación de equity del motor de investigación existente;
no cambian las estrategias o features actuales. El runner de ingestión sigue usando
el calendario cripto actual; otros mercados necesitan su propio adaptador.

Las probabilidades se refieren a alcanzar la media anclada **dentro de 10 velas**,
no a cualquier rebote o ganancia parcial. Un timeout rentable sigue teniendo Y=0.
Cuando el objetivo es raro, especialmente con ventanas largas, los filtros pueden
permanecer en efectivo. Ese 0% no acredita capacidad predictiva ni ventaja frente
a efectivo. Las tablas por régimen no crean nuevos tests independientes.

Ejecución documentada: `{summary["run_id"]}`. Los modelos actuales y sus punteros
se verificaron por hash antes y después del estudio. El historial reservado futuro
todavía no está disponible y queda protegido por el guard compartido de entrenamiento.
"""
    args.output.write_text(
        (args.study / "REPORT.md").read_text(encoding="utf-8") + usage, encoding="utf-8"
    )
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
