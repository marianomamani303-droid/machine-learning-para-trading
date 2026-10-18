"""Replay saved predictions and verify economic metrics; never fits or overwrites a study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.structure_regime.features import CLUSTER_COLUMNS, CLUSTER_INPUT


def predict(path, frame, columns):
    model = XGBClassifier()
    model.load_model(path)
    return model.predict_proba(frame[columns].to_numpy(dtype=float))[:, 1]


def verify(directory):
    plan, summary = read_json(directory / "plan.json"), read_json(directory / "summary.json")
    manifest = read_json(directory / "artifacts.json")
    for relative, digest in manifest.items():
        if sha256(directory / relative) != digest:
            raise ValueError(f"Artifact mismatch: {relative}")
    combination = (directory / "predictions_with_combination.csv").exists()
    suffix = "_with_combination" if combination else ""
    predictions = pd.read_csv(
        directory / f"predictions{suffix}.csv", parse_dates=["reference_time", "target_time"]
    )
    curves = pd.read_csv(directory / f"curves{suffix}.csv")
    trades = pd.read_csv(directory / f"trades{suffix}.csv")
    fit_index = {
        (f["symbol"], f["fold"], f["seed"], f["recipe"]): f
        for f in read_json(directory / "fits.json")
    }
    replayed = 0
    worst = 0.0
    for symbol, data in predictions.groupby("symbol"):
        panel = pd.read_csv(directory / f"{symbol}_features.csv", parse_dates=["reference_time"])
        for (fold, seed), rows in data.groupby(["fold", "seed"]):
            fold_dir = directory / symbol / f"fold_{fold}" / f"seed_{seed}"
            clusters = read_json(fold_dir / "clusters.json")
            start, end = pd.to_datetime(
                plan["protocol"]["research"]["boundaries"][fold : fold + 2], utc=True
            )
            if pd.Timestamp(clusters["train_last_target"]) >= start - pd.Timedelta(days=6):
                raise ValueError("Training invades embargo")
            baseline = read_json(fold_dir / "baselines.json")
            reference_rows = rows.loc[rows.recipe == "original"].sort_values("reference_time")
            if reference_rows.target_time.max() >= end:
                raise ValueError("Evaluation target crosses fold")
            frame = reference_rows[["reference_time"]].merge(
                panel, on="reference_time", validate="one_to_one"
            )
            standardized = (frame[CLUSTER_INPUT].to_numpy() - clusters["mean"]) / clusters["scale"]
            states = (
                ((standardized[:, None, :] - np.asarray(clusters["centers"])[None, :, :]) ** 2)
                .sum(axis=2)
                .argmin(axis=1)
            )
            frame["sr_cluster_state"] = states.astype(str)
            for i, name in enumerate(CLUSTER_COLUMNS):
                frame[name] = (states == i).astype(float)
            probabilities = {}
            for recipe, columns in plan["recipes"].items():
                probabilities[recipe] = predict(fold_dir / f"{recipe}.ubj", frame, columns)
                audit = fit_index[(symbol, fold, seed, recipe)]
                if pd.Timestamp(audit["last_train_target"]) >= start - pd.Timedelta(days=6):
                    raise ValueError("Global training invades embargo")
            for method, state in [("rules", "sr_state"), ("cluster", "sr_cluster_state")]:
                recipe = f"specialists_{method}"
                p = probabilities["original"].copy().astype(float)
                audit = fit_index[(symbol, fold, seed, recipe)]
                for expert in audit["experts"]:
                    mask = frame[state].to_numpy() == expert["state"]
                    if expert["last_target"] and pd.Timestamp(
                        expert["last_target"]
                    ) >= start - pd.Timedelta(days=6):
                        raise ValueError("Expert training invades embargo")
                    if not expert["fallback_global"] and mask.any():
                        raw = predict(
                            fold_dir / recipe / f"expert_{expert['state']}.ubj",
                            frame.loc[mask],
                            audit["columns"],
                        )
                        weight = expert["expert_weight"]
                        p[mask] = weight * raw + (1 - weight) * p[mask]
                probabilities[recipe] = p
            columns = baseline["features"]
            values = (frame[columns].to_numpy() - baseline["scaler_mean"]) / baseline[
                "scaler_scale"
            ]
            score = values @ np.asarray(baseline["coefficients"]) + baseline["intercept"]
            probabilities["logistic_l2"] = 1 / (1 + np.exp(-np.clip(score, -700, 700)))
            probabilities["constant_05"] = np.full(len(frame), 0.5)
            probabilities["train_prevalence"] = np.full(len(frame), baseline["train_prevalence"])
            if combination:
                probabilities["combined"] = predict(
                    fold_dir / "combined.ubj", frame, summary["combination"]["columns"]
                )
            for recipe, p in probabilities.items():
                saved = rows.loc[rows.recipe == recipe].sort_values("reference_time")
                if not np.array_equal(
                    saved.reference_time.to_numpy(), reference_rows.reference_time.to_numpy()
                ):
                    raise ValueError("Different OOS timestamps")
                delta = float(np.abs(p - saved.probability.to_numpy()).max())
                worst = max(worst, delta)
                np.testing.assert_allclose(p, saved.probability, rtol=0, atol=2e-7)
                replayed += 1
    for key, inference in summary["inference"].items():
        symbol, seed = key.split("_seed")
        for recipe, expected in inference["predictive"]["metrics"].items():
            rows = predictions.loc[
                (predictions.symbol == symbol)
                & (predictions.seed == int(seed))
                & (predictions.recipe == recipe)
            ]
            measured = classification_metrics(rows.target, rows.probability)
            for name in ["accuracy", "roc_auc", "log_loss", "brier_score"]:
                np.testing.assert_allclose(measured[name], expected[name], atol=1e-12)
    for expected in summary["economic_metrics"]:
        mask = (
            (curves.symbol == expected["symbol"])
            & (curves.seed == expected["seed"])
            & (curves.recipe == expected["recipe"])
            & (curves.scenario == expected["scenario"])
        )
        part = curves.loc[mask].sort_values(["fold", "step"])
        returns = part.groupby("fold").equity.pct_change().fillna(0).to_numpy()
        equity = np.cumprod(np.r_[1.0, 1 + returns])
        np.testing.assert_allclose(equity[-1] - 1, expected["net_return"], atol=1e-12)
        np.testing.assert_allclose(
            (equity / np.maximum.accumulate(equity) - 1).min(), expected["max_drawdown"], atol=1e-12
        )
        ledger = trades.loc[
            (trades.symbol == expected["symbol"])
            & (trades.seed == expected["seed"])
            & (trades.recipe == expected["recipe"])
            & (trades.scenario == expected["scenario"])
        ]
        assert len(ledger) == expected["trade_count"]
        if len(ledger):
            assert (pd.to_datetime(ledger.entry_time) >= pd.to_datetime(ledger.signal_time)).all()
    # Full preservation comparison was performed by the run; verify its receipt and
    # recheck production pointers/models here without rehashing 34k old files.
    checked = 0
    for absolute, digest in plan["protected_files"].items():
        path = Path(absolute)
        if "models" in path.parts or path.name == "prospective.json":
            if sha256(path) != digest:
                raise ValueError(f"Protected file changed: {path}")
            checked += 1
    return {
        "run_id": summary["run_id"],
        "artifact_hashes_verified": len(manifest),
        "prediction_series_replayed": replayed,
        "maximum_probability_error": worst,
        "economic_series_verified": len(summary["economic_metrics"]),
        "production_and_reservation_files_rechecked": checked,
        "full_previous_preservation_receipt": summary["previous_artifacts_preserved"],
        "new_fits": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    result = verify(args.study)
    target = (
        args.study.parent.parent / "structure_regime_verifications" / f"{result['run_id']}.json"
    )
    write_new_json(target, result)
    print(result)
