"""Render saved, hash-verified research results without training/selecting models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.dmn.study import verify_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verified = verify_study(args.study)
    summary = read_json(args.study / "summary.json")
    plan = read_json(args.study / "plan.json")
    domain = plan["protocol"]["risk"]["position_mode"]
    domain_label = "long/cash" if domain == "long_only" else "synthetic long/short"
    output = args.output or Path("artifacts/dmn_analyses") / summary["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    chosen = [
        "long_scaled",
        "tsmom",
        "macd",
        "mlp_sharpe_net",
        "lstm_sharpe_gross",
        "lstm_sharpe_net",
        "lstm_mse",
        "xgboost_binary",
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for name in chosen:
        if name not in summary["models"]:
            continue
        frame = pd.read_csv(args.study / f"{name}_equity.csv")
        times = pd.to_datetime(frame.reference_time, utc=True)
        equity = frame.equity.to_numpy()
        axes[0].plot(times, equity, label=name, linewidth=1.2)
        drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1
        axes[1].plot(times, 100 * drawdown, linewidth=1)
    axes[0].axhline(1, color="gray", linewidth=0.5)
    axes[0].set(
        ylabel="Equity / initial capital",
        yscale="log",
        title=f"DMN — historical development, BTC/ETH {domain_label}; costs included",
    )
    axes[1].set(ylabel="Drawdown (%)", xlabel="Signal / entry date (UTC)")
    axes[0].legend(fontsize=8, ncols=2, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.text(
        0.5,
        0.005,
        "Fixed recipes, mean positions across seeds. No independent holdout or realized-volatility rescaling.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"equity_drawdown.{suffix}", dpi=170)
    plt.close(fig)

    names = [name for name in summary["models"] if "__seed_" not in name and name != "cash"]
    values = np.array(
        [
            [np.nan if x is None else x for x in summary["models"][name]["fold_sharpes"]]
            for name in names
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(names)), names)
    labels = [f"{fold['start'][:10]}\n{fold['end_exclusive'][:10]}" for fold in summary["folds"]]
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(names)):
        for j in range(len(labels)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Net Sharpe per outer fold — descriptive historical stability")
    fig.colorbar(image, ax=ax, label="Sharpe (color clipped at ±2)")
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"fold_sharpes.{suffix}", dpi=170)
    plt.close(fig)
    write_new_json(
        output / "manifest.json",
        {
            "study": str(args.study.resolve()),
            "summary_sha256": sha256(args.study / "summary.json"),
            "verified": verified,
            "matplotlib": matplotlib.__version__,
            "plot_script_sha256": sha256(Path(__file__)),
            "plots": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
