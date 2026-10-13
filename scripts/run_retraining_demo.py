"""Verifica reentrenamiento retrospectivo con datos reales ya descargados.

Ejecutar desde la raíz: python scripts/run_retraining_demo.py.
No interpreta la comparación como evidencia prospectiva ni elige un ganador.
"""

from dataclasses import replace

from trading_ml.common import write_json
from trading_ml.config import load_settings
from trading_ml.data import download
from trading_ml.evaluation.comparison import compare
from trading_ml.models import load_model
from trading_ml.preprocessing import prepare
from trading_ml.training import retrain, train


def main() -> None:
    settings = load_settings()
    initial_version = load_model(settings.root).metadata["version"]
    earlier = replace(settings, market=replace(settings.market, end="2026-07-01"))
    raw = download(earlier, settings.root / "data" / "snapshot_raw.csv")
    dataset = prepare(earlier, raw, settings.root / "data" / "snapshot_prepared.csv")
    parent = train(earlier, dataset)
    updated = retrain(settings.root, settings.root / "data" / "prepared.csv", parent)
    comparison = compare(settings.root, settings.root / "data" / "prepared.csv", [parent, updated])
    write_json(
        settings.root / "reports" / "retraining_demo.json",
        {
            "initial_version": initial_version,
            "parent_version": parent,
            "retrained_version": updated,
            "parent_snapshot_end": earlier.market.end,
            "new_snapshot_end": settings.market.end,
            "same_model_hash_as_initial": load_model(settings.root, initial_version).metadata[
                "model_sha256"
            ]
            == load_model(settings.root, updated).metadata["model_sha256"],
            "comparison": comparison,
            "note": "Demostración retrospectiva de funcionamiento; períodos de test comunes, sin seleccionar modelos con esta comparación.",
        },
    )
    print(f"Reentrenamiento verificado: {parent} -> {updated}")


if __name__ == "__main__":
    main()
