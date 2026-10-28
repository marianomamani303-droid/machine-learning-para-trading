"""Plot saved, verified RL study curves without fitting or selecting models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_ml.common import sha256, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("artifacts/rl_signal/analyses") / args.study.name
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    variants = [
        ("band", 0, "#111111"),
        ("fixed", 0, "#888888"),
        ("long_scaled", 0, "#c67d12"),
        ("cash", 0, "#769982"),
        ("dqn", 42, "#1468b3"),
        ("dqn", 1729, "#c94760"),
        ("dqn", 2027, "#7452aa"),
    ]
    for policy, seed, color in variants:
        frames = [
            pd.read_csv(p)
            for p in sorted(args.study.glob(f"fold_*/base_{policy}_{seed}_curve.csv"))
        ]
        curve = pd.concat(frames, ignore_index=True)
        dates = pd.to_datetime(curve.exit_time, utc=True)
        equity = np.exp(curve.log_return.cumsum())
        dd = equity / np.maximum.accumulate(np.r_[1, equity])[1:] - 1
        label = f"DQN {seed}" if seed else policy
        axes[0].plot(dates, equity, label=label, color=color, linewidth=1.4)
        axes[1].plot(dates, 100 * dd, color=color, linewidth=1.1)
        axes[2].plot(
            dates,
            100 * curve.exposure.rolling(30, min_periods=1).mean(),
            color=color,
            linewidth=1.1,
        )
    axes[0].set_title("BTC/USDT diario: misma señal, distintas políticas — historia reutilizada")
    axes[0].set_ylabel("Equity normalizada")
    axes[0].legend(ncol=4, fontsize=9)
    axes[1].set_ylabel("Drawdown (%)")
    axes[2].set_ylabel("Exposición media 30d (%)")
    axes[2].set_xlabel("Fecha UTC · costos supuestos · folds reinician en 10000 USDT")
    for ax in axes:
        ax.grid(alpha=0.2)
    for extension in ("png", "svg"):
        fig.savefig(output / f"equity_risk.{extension}", dpi=160)
    plt.close(fig)
    write_new_json(
        output / "manifest.json", {p.name: sha256(p) for p in sorted(output.glob("equity_risk.*"))}
    )
    print(output)


if __name__ == "__main__":
    main()
