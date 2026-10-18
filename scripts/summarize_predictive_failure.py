"""Produce exact Brier moment diagnostics, power references and audit figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from trading_ml.common import read_json, sha256, write_new_json


def brier_moments(target, probability):
    y, p = np.asarray(target, float), np.asarray(probability, float)
    uncertainty = float(y.mean() * (1 - y.mean()))
    variance = float(np.var(p))
    bias = float((p.mean() - y.mean()) ** 2)
    covariance = float(np.mean((p - p.mean()) * (y - y.mean())))
    measured = float(np.mean((y - p) ** 2))
    np.testing.assert_allclose(uncertainty + variance + bias - 2 * covariance, measured, atol=1e-12)
    return {
        "rows": len(y),
        "brier": measured,
        "outcome_uncertainty": uncertainty,
        "forecast_variance": variance,
        "mean_bias_squared": bias,
        "twice_forecast_outcome_covariance": 2 * covariance,
        "note": "Exact moment identity, not a binned Murphy decomposition and not a fitted recalibrator",
    }


def summarize(directory):
    summary = read_json(directory / "summary.json")
    if not summary["previous_artifacts_preserved"]:
        raise ValueError("Missing preservation receipt")
    for name, digest in read_json(directory / "artifacts.json").items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Diagnostic checksum changed: {name}")
    root = directory.parent.parent
    output = root / "predictive_failure_analyses" / summary["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    source = root / "structure_regime_studies/20261016T152046_8f7d1886/predictions.csv"
    predictions = pd.read_csv(source)
    moments = [
        {
            "symbol": symbol,
            "seed": int(seed),
            "recipe": recipe,
            **brier_moments(g.target, g.probability),
        }
        for (symbol, seed, recipe), g in predictions.groupby(["symbol", "seed", "recipe"])
    ]
    n = 1728
    power = {
        "assumptions": "IID Bernoulli reference only; no multiplicity, serial dependence or model-selection adjustment",
        "n_reference": n,
        "95pct_halfwidth_at_50pct": float(norm.ppf(0.975) * np.sqrt(0.25 / n)),
        "80pct_power_mde": float((norm.ppf(0.975) + norm.ppf(0.8)) * np.sqrt(0.25 / n)),
        "approx_n_for_one_percentage_point_at_80pct": int(
            np.ceil(0.25 * (norm.ppf(0.975) + norm.ppf(0.8)) ** 2 / 0.01**2)
        ),
    }
    write_new_json(
        output / "supplement.json",
        {
            "brier_moments": moments,
            "power_reference": power,
            "audit_artifact_hashes_verified": len(read_json(directory / "artifacts.json")),
            "new_model_fits": 0,
            "script_sha256": sha256(Path(__file__)),
        },
    )
    errors = pd.DataFrame(read_json(directory / "errors.json")["confidence_errors"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, symbol in zip(axes, ["BTCUSDT", "ETHUSDT"], strict=True):
        for recipe, label in [
            ("original", "Original"),
            ("structure", "Estructura"),
            ("logistic_l2", "Logística"),
        ]:
            sample = errors.loc[
                (errors.symbol == symbol) & (errors.seed == 42) & (errors.recipe == recipe)
            ]
            ax.plot(sample.predicted_confidence, sample.hit_rate, marker="o", label=label)
            for _, row in sample.loc[sample.recipe == "original"].iterrows():
                ax.annotate(
                    str(row.rows),
                    (row.predicted_confidence, row.hit_rate),
                    xytext=(3, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.plot([0.5, 0.85], [0.5, 0.85], ls="--", color="black", alpha=0.5)
        ax.set(
            title=symbol,
            xlabel="Confianza media max(p,1−p)",
            ylabel="Fracción correcta observada",
            ylim=(0.3, 0.9),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Confianza histórica OOS · etiquetas junto al original = observaciones; bins pequeños son inestables"
    )
    fig.savefig(output / "confidence.png", dpi=160)
    plt.close(fig)
    targets = read_json(directory / "targets.json")
    values = pd.DataFrame(targets["magnitude_vs_signed_no_training"])
    rows = []
    for (symbol, horizon), group in values.groupby(["symbol", "horizon"]):
        rows.append(
            {
                "symbol": symbol,
                "horizon": int(horizon),
                "absolute_mse_skill": 1
                - float(
                    (group.rows * group.magnitude_mse_vol20).sum()
                    / (group.rows * group.magnitude_mse_train_mean).sum()
                ),
                "signed_mse_skill": 1
                - float(
                    (group.rows * group.signed_mse_trailing_mean60).sum()
                    / (group.rows * group.signed_mse_zero).sum()
                ),
                "folds_absolute_better": int(
                    (group.magnitude_mse_vol20 < group.magnitude_mse_train_mean).sum()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output / "magnitude_and_sign.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, symbol in zip(axes, ["BTCUSDT", "ETHUSDT"], strict=True):
        selected = [r for r in rows if r["symbol"] == symbol]
        ax.plot(
            [r["horizon"] for r in selected],
            [r["absolute_mse_skill"] for r in selected],
            marker="o",
            label="Magnitud / media absoluta train",
        )
        ax.plot(
            [r["horizon"] for r in selected],
            [r["signed_mse_skill"] for r in selected],
            marker="o",
            label="Retorno con signo / cero",
        )
        ax.axhline(0, color="black", ls="--", alpha=0.5)
        ax.set(
            title=symbol, xlabel="Horizonte (velas diarias)", ylabel="1 − MSE modelo / MSE baseline"
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Dos problemas distintos · pronósticos simples causales, sin entrenamiento ML nuevo"
    )
    fig.savefig(output / "magnitude_vs_sign.png", dpi=160)
    plt.close(fig)
    return {"output": str(output.resolve()), "power": power, "magnitude_comparison": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    print(summarize(parser.parse_args().audit))
