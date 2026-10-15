from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import read_json, sha256
from trading_ml.config import Settings
from trading_ml.errors import DataError
from trading_ml.feature_research.study import (
    _frozen_hashes,
    _load_panel,
    choose_inner,
    fixed_recipes,
    temporal_score,
    temporal_train,
)


def verify_feature_study(settings: Settings, directory: Path) -> dict:
    """Rebuild features/labels and reproduce saved OOS probabilities without fitting."""
    manifest = read_json(directory / "artifacts.json")
    for relative, digest in manifest.items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise DataError(f"Artefacto de features alterado: {relative}")
    summary = read_json(directory / "summary.json")
    plan = read_json(directory / "plan.json")
    if plan["protected_before"] != _frozen_hashes(settings):
        raise DataError("Modelos o reservas históricos alterados")
    for origin in plan["data"].values():
        if sha256(Path(origin["source"])) != origin["source_sha256"]:
            raise DataError("Datos originales alterados")
    r = plan["protocol"]["research"]
    dates = pd.to_datetime(r["boundaries"], utc=True)
    panel, _ = _load_panel(settings, dates[-1])
    predictions = pd.read_csv(
        directory / "predictions.csv", parse_dates=["reference_time", "target_time"]
    )
    checked = 0
    for symbol, frame in panel.items():
        common = plan["common_columns"]
        snapshot = pd.read_csv(directory / f"{symbol}_features.csv")
        np.testing.assert_allclose(
            snapshot[common], frame[common], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        causal = frame.loc[frame[common].notna().all(axis=1) & frame.target.notna()]
        for fold, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True)):
            train = temporal_train(causal, start, r["embargo_days"], r["minimum_train_rows"])
            evaluation = temporal_score(causal, start, end)
            fold_dir = directory / symbol / f"fold_{fold}"
            selection = read_json(fold_dir / "selection.json")
            repeated = choose_inner(
                selection["inner_log_losses"], selection["individual_importance"], train, r
            )
            for name, value in repeated.items():
                if selection[name] != value:
                    raise DataError("La selección interna no coincide con su evidencia")
            if any(
                pd.Timestamp(w["last_validation_target"])
                >= start - pd.Timedelta(days=r["embargo_days"])
                or pd.Timestamp(w["last_train_target"])
                >= pd.Timestamp(w["start"]) - pd.Timedelta(days=r["embargo_days"])
                for w in selection["windows"]
            ):
                raise DataError("La selección invade la ventana externa")
            all_saved = predictions.loc[(predictions.symbol == symbol) & (predictions.fold == fold)]
            models = {}
            fits = read_json(fold_dir / "fits.json")
            fixed = fixed_recipes()
            for record in fits:
                columns = record["columns"]
                if record["recipe"] in fixed and columns != fixed[record["recipe"]]:
                    raise DataError("Contrato de features alterado")
                filename = record["model_file"]
                if filename not in models:
                    model_path = directory / filename
                    if not model_path.resolve().is_relative_to(directory.resolve()):
                        raise DataError("Ruta de modelo inválida")
                    model = XGBClassifier()
                    model.load_model(model_path)
                    models[filename] = model.predict_proba(
                        evaluation[columns].to_numpy(dtype=float)
                    )[:, 1]
                saved = all_saved.loc[
                    (all_saved.recipe == record["recipe"]) & (all_saved.seed == record["seed"])
                ]
                np.testing.assert_allclose(
                    saved.probability, models[filename], rtol=1e-7, atol=1e-8
                )
                np.testing.assert_array_equal(saved.reference_time, evaluation.reference_time)
                np.testing.assert_array_equal(saved.target_time, evaluation.target_time)
                np.testing.assert_array_equal(saved.target, evaluation.target)
                checked += 1
            baseline = read_json(fold_dir / "baseline.json")
            x = evaluation[baseline["features"]].to_numpy(dtype=float)
            z = (x - baseline["scaler_mean"]) / baseline["scaler_scale"]
            score = z @ np.asarray(baseline["coefficients"]) + baseline["intercept"]
            probabilities = {
                "constant_05": np.full(len(x), 0.5),
                "train_prevalence": np.full(len(x), baseline["train_prevalence"]),
                "logistic_l2": 1 / (1 + np.exp(-score)),
            }
            for name, p in probabilities.items():
                saved = all_saved.loc[all_saved.recipe == name]
                np.testing.assert_allclose(saved.probability, p, rtol=1e-7, atol=1e-8)
                np.testing.assert_array_equal(saved.target, evaluation.target)
                checked += 1
    return {
        "passed": True,
        "run_id": summary["run_id"],
        "artifacts_verified": len(manifest),
        "prediction_series_reproduced": checked,
        "new_fits": 0,
        "features_and_targets_rebuilt_from_verified_archives": True,
        "existing_models_and_reservations_preserved": True,
    }
