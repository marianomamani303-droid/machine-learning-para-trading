"""Verify a saved feature study and export figures without fitting new models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trading_ml.common import read_json, write_new_json  # noqa: E402
from trading_ml.config import load_settings  # noqa: E402
from trading_ml.feature_research.verify import verify_feature_study  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    args = parser.parse_args()
    settings = load_settings(args.config)
    result = verify_feature_study(settings, args.study)
    summary = read_json(args.study / "summary.json")
    output = settings.root / "feature_analyses" / args.study.name
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "verification.json", result)
    seed = read_json(args.study / "plan.json")["protocol"]["research"]["seed"]
    fits = pd.DataFrame(
        [
            {
                "symbol": row["symbol"],
                "recipe": row["recipe"],
                "fold": row["fold"],
                "loss": row["oos"]["log_loss"],
            }
            for row in summary["fold_metrics"]
            if row["seed"] == seed
        ]
    )
    names = [
        "original_pruned",
        "add_momentum",
        "add_volatility",
        "add_flow",
        "add_reversion",
        "add_breakout",
        "add_structure",
        "add_timeframes",
        "add_cross_asset",
        "all",
        "nested_choice",
        "compact_inner",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for ax, (symbol, group) in zip(axes, fits.groupby("symbol"), strict=True):
        wide = group.pivot(index="recipe", columns="fold", values="loss")
        delta = wide.loc[names].subtract(wide.loc["original"], axis=1)
        for j, name in enumerate(names):
            ax.scatter(delta.loc[name], np.full(len(delta.columns), j), alpha=0.6, s=22)
            ax.scatter([delta.loc[name].mean()], [j], color="black", marker="D", s=32)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlabel("Δ log-loss vs original (cada fold; media = rombo)")
        ax.set_title(symbol)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Generalización temporal histórica — menor Δloss es mejor")
    fig.savefig(output / "fold_comparison.png", dpi=160)
    plt.close(fig)

    table = pd.DataFrame(summary["importance_summary"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for ax, symbol in zip(axes, sorted(table.symbol.unique()), strict=True):
        subset = table.loc[
            (table.symbol == symbol) & (table.recipe == "original") & (table.block == 30)
        ]
        subset = subset.sort_values("mean_delta_loss")
        colors = ["#21856b" if x > 0 else "#b24d4d" for x in subset.mean_delta_loss]
        ax.barh(subset.name, subset.mean_delta_loss, color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Δ log-loss al permutar, media de folds (bloques 30 días)")
        ax.set_title(symbol)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Importancia fuera de muestra — positivo indica utilidad al modelo")
    fig.savefig(output / "original_permutation.png", dpi=160)
    plt.close(fig)
    print(f"Verificado: {result['prediction_series_reproduced']} series; informe gráfico: {output}")


if __name__ == "__main__":
    main()
