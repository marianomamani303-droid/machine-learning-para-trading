"""Document a completed risk study without refitting models or re-evaluating data."""

from __future__ import annotations

import argparse
from pathlib import Path

from trading_ml.common import read_json, sha256
from trading_ml.errors import DataError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("RISK_RESEARCH.md"))
    args = parser.parse_args()
    for relative, expected in read_json(args.study / "artifacts.json").items():
        if sha256(args.study / relative) != expected:
            raise DataError(f"Artefacto de riesgo alterado: {relative}")
    summary = read_json(args.study / "summary.json")
    usage = f"""
## Ejecutar e integrar

```powershell
.\\.venv\\Scripts\\python.exe -m trading_ml risk-study --protocol config/risk_study.toml
# Documentar resultados existentes, sin repetir el test:
.\\.venv\\Scripts\\python.exe scripts/record_risk_study.py {args.study.as_posix()}
```

Cada ejecución crea un directorio nuevo y conserva resultados negativos/nulos.
El comando compara 10 políticas prefijadas sobre tres modelos registrados,
con costos base y duplicados. No realiza búsqueda, promoción ni trading real.

El módulo `trading_ml.risk_overlay` separa configuración, simulador, estudio y
reporte. `RiskPolicy` y `run_overlay` permiten experimentar con probabilidades
ya calculadas; no modifican un modelo ni un archivo de estrategia original.

```python
import pandas as pd
from trading_ml.momentum.config import ResearchCosts
from trading_ml.risk_overlay.config import RiskPolicy
from trading_ml.risk_overlay.simulator import risk_features, run_overlay

# frame: OHLCV, timestamp y reference_time UTC; p es un array alineado.
# El llamador del API asegura que p sea una predicción causal fuera de muestra.
result = run_overlay(
    risk_features(frame),
    p,
    pd.Timestamp("2025-06-05", tz="UTC"),
    pd.Timestamp("2026-10-01", tz="UTC"),
    ResearchCosts(spread_bps=0),
    RiskPolicy(name="cap_25", capital_cap=0.25),
    horizon=1,
)
```

El simulador comprueba cierre de señal anterior a ejecución, probabilidades
finitas y caps en las aperturas. El runner valida procedencia, período anterior
a las reservas y entrenamiento temporal con purga. El API de bajo nivel exige
que el llamador mantenga el contrato fuera de muestra.

`events.csv` registra compra, venta y cada reducción parcial con sus costos;
`trades.csv` agrupa esos eventos por operación y contiene PnL bruto/neto.
`equity.csv` conserva exposición, drawdown usado para decidir, circuito de
pausa y las dos curvas. `matched_caps.json` se fija con desarrollo únicamente.

El caso principal fue fijado como `{summary["diagnosis"]["policy"]}` sobre
`{summary["diagnosis"]["reference"]}` antes de evaluar el período final.
La reserva prospectiva y los modelos registrados se preservaron por hash;
esa reserva no se evaluó ni se incorporó al entrenamiento. Los resultados
históricos siguen siendo descriptivos porque ese historial ya se observó.
"""
    args.output.write_text(
        (args.study / "REPORT.md").read_text(encoding="utf-8") + usage, encoding="utf-8"
    )
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
