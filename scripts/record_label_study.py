"""Verify a frozen label study and export scientific plots without new fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trading_ml.common import read_json, write_new_json  # noqa: E402
from trading_ml.config import load_settings  # noqa: E402
from trading_ml.label_research.verify import verify_label_study  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = load_settings(args.config)
    result = verify_label_study(settings, args.study)
    summary = read_json(args.study / "summary.json")
    output = args.output or settings.root / "label_analyses" / args.study.name
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "verification.json", result)
    curves = pd.read_csv(args.study / "curves.csv", parse_dates=["timestamp"])
    names = [
        "original",
        "control_tb_vol_h5",
        "tb_vol_h5_cal_ev",
        "meta_net_h1_logistic_cal_55",
        "meta_tb_vol_h5_logistic_cal_55",
        "cash",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, symbol in zip(axes, sorted(curves.symbol.unique()), strict=True):
        for name in names:
            rows = curves.loc[
                (curves.symbol == symbol) & (curves.scenario == "main") & (curves.strategy == name)
            ].sort_values(["fold", "timestamp"])
            wealth = 10000 * (1 + rows.net_return).cumprod()
            ax.plot(rows.timestamp, wealth, label=name, linewidth=1.4)
        ax.set_title(symbol)
        ax.set_ylabel("Capital neto, inicio 10.000")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Comparación temporal con costos — candidatos fijados antes de evaluar")
    fig.savefig(output / "net_equity.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for col, symbol in enumerate(sorted(curves.symbol.unique())):
        for row, prefix in enumerate(["tb_vol_h5", "meta_tb_vol_h5_logistic"]):
            ax = axes[row, col]
            for variant in ["raw", "cal"]:
                record = next(
                    p
                    for p in summary["predictive"]
                    if p["symbol"] == symbol and p["forecast_id"] == prefix + "_" + variant
                )
                metric = record["metrics"]
                if metric["rows"]:
                    bins = metric["reliability_bins"]
                    ax.plot(
                        [b["mean_probability"] for b in bins],
                        [b["observed_up_fraction"] for b in bins],
                        "o-",
                        label=f"{variant}, n={metric['rows']}",
                    )
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Probabilidad media de rentabilidad neta",
                ylabel="Fracción observada rentable",
                title=f"{symbol}: {prefix}",
            )
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle("Calibración OOS: todos los eventos TB / ofertas del padre para meta")
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)
    print(
        f"Verificado: {result['prediction_series_reproduced']} series, {result['executed_trade_payoffs_verified']} trades; {output}"
    )


if __name__ == "__main__":
    main()
