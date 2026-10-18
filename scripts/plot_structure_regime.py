"""Standalone historical comparisons; no selection or fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_ml.common import read_json


def plot(directory):
    summary = read_json(directory / "summary.json")
    output = directory.parent.parent / "structure_regime_analyses" / summary["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    names = [
        "original",
        "structure",
        "regime_rules",
        "specialists_rules",
        "regime_cluster",
        "specialists_cluster",
    ]
    labels = [
        "Original",
        "Estructura",
        "Régimen reglas",
        "Expertos reglas",
        "Régimen clusters",
        "Expertos clusters",
    ]
    colors = ["#64748b", "#0284c7", "#d97706", "#ea580c", "#7c3aed", "#be185d"]
    seed = 42
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, symbol in enumerate(["BTCUSDT", "ETHUSDT"]):
        data = summary["inference"][f"{symbol}_seed{seed}"]["predictive"]["metrics"]
        axes[0, column].barh(
            labels[::-1], [data[n]["log_loss"] for n in names][::-1], color=colors[::-1]
        )
        axes[0, column].axvline(np.log(2), color="#16a34a", ls="--", label="p=0,5")
        axes[0, column].set(
            xlabel="Log-loss OOS (menor es mejor)",
            title=symbol,
            xlim=(0.68, max(data[n]["log_loss"] for n in names) + 0.004),
        )
        axes[0, column].legend()
        curves = pd.read_csv(directory / "curves.csv", parse_dates=["timestamp"])
        for name, label, color in zip(names, labels, colors, strict=True):
            rows = curves.loc[
                (curves.symbol == symbol)
                & (curves.seed == seed)
                & (curves.scenario == "base")
                & (curves.recipe == name)
            ]
            rows = rows.sort_values(["fold", "step"])
            returns = rows.groupby("fold").equity.pct_change().fillna(0)
            axes[1, column].plot(
                rows.timestamp, 100 * (1 + returns).cumprod(), label=label, color=color, lw=1.4
            )
        axes[1, column].axhline(100, color="#16a34a", ls="--", lw=1)
        axes[1, column].set(xlabel="Fecha UTC", ylabel="Capital inicial = 100; costos base")
        axes[1, column].legend(fontsize=8)
        for ax in axes[:, column]:
            ax.grid(alpha=0.2)
    figure.suptitle(
        "Walk-forward histórico 2022–2026 · semilla 42 · períodos ya observados", fontsize=14
    )
    figure.savefig(output / "comparison.png", dpi=160)
    figure.savefig(output / "comparison.svg")
    plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    print(plot(parser.parse_args().study))
