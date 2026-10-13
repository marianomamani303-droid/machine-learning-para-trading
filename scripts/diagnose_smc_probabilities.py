"""Read-only probability diagnostics on labels already used by archived SMC models."""

from __future__ import annotations

import argparse
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_ml.common import frame_hash, sha256, write_new_json
from trading_ml.evaluation.diagnostics import probability_diagnostics
from trading_ml.models import load_model
from trading_ml.preprocessing import load_prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--versions", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "predictive_diagnostics" / (uuid.uuid4().hex + ".json")
    if output.exists():
        parser.error("El informe existe; elegí otra ruta")
    results = {}
    for version in args.versions:
        bundle = load_model(args.root, version)
        if not bundle.metadata["contract"]["features"].get("smc"):
            parser.error("El diagnóstico requiere versiones con features SMC")
        dataset = Path(bundle.metadata["dataset_path"])
        frame, metadata = load_prepared(dataset, bundle.metadata["contract"])
        cutoff = pd.Timestamp(bundle.metadata["selection_label_end"])
        known = frame.loc[frame.target_time <= cutoff].dropna(subset=["target", "target_time"])
        raw = bundle.model.predict_proba(known[bundle.metadata["features"]])[:, 1]
        calibrated = bundle.probabilities(known)
        threshold = bundle.metadata["config"]["backtest"]["threshold"]
        calibration = bundle.metadata.get("calibration")
        results[version] = {
            "scope": "previously_used_fit_and_calibration_labels_only",
            "independent_evaluation": False,
            "model_sha256": sha256(bundle.directory / "model.ubj"),
            "metadata_sha256": sha256(bundle.directory / "metadata.json"),
            "dataset_sha256": metadata["sha256"],
            "known_labels_sha256": frame_hash(known),
            "last_known_label": known.target_time.max().isoformat(),
            "trees": bundle.model.get_booster().num_boosted_rounds(),
            "calibration": calibration,
            "raw": probability_diagnostics(known.target.to_numpy(), raw, threshold),
            "calibrated": probability_diagnostics(known.target.to_numpy(), calibrated, threshold),
            "warning": "Diagnostic of fitted score dispersion; in-sample metrics do not establish predictive skill or choose a threshold.",
        }
    write_new_json(
        output,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "script_sha256": sha256(Path(__file__)),
            "models": results,
            "new_fits": 0,
            "promoted": False,
        },
    )
    print(f"Diagnóstico sin ajustes nuevos: {output}")


if __name__ == "__main__":
    main()
