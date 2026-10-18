"""Inference-free analysis of the completed 24-fit Forex expansion study.

Reads the verified predictions.csv and summary.json only (no fitting).
Computes full classification metrics (including balanced_accuracy, precision,
recall, F1, ECE and confusion matrix), paired moving-block bootstrap between
`compact` and `price` and vs. train prevalence / 0.5 controls, plus compounded
economic metrics across folds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.dmn.study import _verify_files

STUDY = Path("artifacts/expansion/forex_studies/20261016T210703_c9107a5c")
EPS = 1e-15


def logloss_vec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def classification(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    d = (p >= 0.5).astype(int)
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(
        abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
        for b in range(10)
        if (bins == b).any()
    )
    return {
        "rows": int(len(y)),
        "up_fraction": float(y.mean()),
        "accuracy": float((d == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, d)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "log_loss": float(logloss_vec(y, p).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "precision": float(precision_score(y, d, zero_division=0)),
        "recall": float(recall_score(y, d, zero_division=0)),
        "f1": float(f1_score(y, d, zero_division=0)),
        "ece_10": float(ece),
        "confusion_tn_fp_fn_tp": confusion_matrix(y, d, labels=[0, 1]).ravel().tolist(),
    }


def block_indices(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n


def paired_bootstrap(diff_by_fold, reps=2000, blocks=(7, 30), seed=20261017):
    observed = float(np.mean(np.concatenate(diff_by_fold)))
    out = {"mean_gain": observed}
    pvals = []
    for block in blocks:
        rng = np.random.default_rng(seed + block)
        draws = np.empty(reps)
        for r in range(reps):
            parts = [d[block_indices(len(d), block, rng)] for d in diff_by_fold]
            draws[r] = np.concatenate(parts).mean()
        centered = draws - draws.mean()
        p = float((1 + (centered >= observed).sum()) / (reps + 1))
        out[f"block_{block}"] = {
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "p_one_sided": p,
        }
        pvals.append(p)
    out["p_max_blocks"] = max(pvals)
    return out


def holm(pvalues: dict) -> dict:
    keys = sorted(pvalues, key=pvalues.get)
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def main():
    _verify_files({str(STUDY / k): v for k, v in read_json(STUDY / "artifacts.json").items()})
    s = read_json(STUDY / "summary.json")
    preds = {}
    rows = []
    for r in s["results"]:
        key = (r["symbol"], r["recipe"], r["fold"], r["seed"])
        folder = STUDY / f"{r['symbol']}_{r['recipe']}_fold{r['fold']}_seed{r['seed']}"
        df = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        preds[key] = df
        c = classification(df.target.to_numpy(int), df.probability.to_numpy(float))
        m, b = r["metrics"], r["baseline_prevalence"]
        if not np.isclose(c["log_loss"], m["log_loss"], atol=1e-6):
            raise ValueError(f"Log-loss mismatch {key}")
        rows.append({
            "symbol": r["symbol"], "recipe": r["recipe"], "fold": r["fold"], "seed": r["seed"],
            "rows": c["rows"], "accuracy": c["accuracy"],
            "balanced_accuracy": c["balanced_accuracy"], "roc_auc": c["roc_auc"],
            "log_loss": c["log_loss"], "brier": c["brier"], "precision": c["precision"],
            "recall": c["recall"], "f1": c["f1"], "ece_10": c["ece_10"],
            "baseline_log_loss": b["log_loss"], "baseline_accuracy": b["accuracy"],
            "net_return": r["net"]["net_return"], "trades": r["net"]["trades"],
            "win_rate": r["net"]["win_rate"], "profit_factor": r["net"]["profit_factor"],
            "max_drawdown": r["net"]["max_drawdown"], "stress_return": r["stress"]["net_return"],
        })
    f = pd.DataFrame(rows)

    # Full-OOS pooled predictions per (symbol, recipe, seed), then averaged across seeds
    per_seed = []
    for (sym, rec, seed), g in f.groupby(["symbol", "recipe", "seed"]):
        cat = pd.concat([preds[(sym, rec, fold, seed)] for fold in sorted(g.fold)], ignore_index=True)
        c = classification(cat.target.to_numpy(int), cat.probability.to_numpy(float))
        w = g.rows / g.rows.sum()
        per_seed.append({
            "symbol": sym, "recipe": rec, "seed": seed, **c,
            "baseline_log_loss": float((g.baseline_log_loss * w).sum()),
            "baseline_accuracy": float((g.baseline_accuracy * w).sum()),
            "compounded_net_return": float(np.prod(1 + g.net_return) - 1),
            "compounded_stress_return": float(np.prod(1 + g.stress_return) - 1),
            "trades": int(g.trades.sum()),
            "mean_profit_factor": float(g.profit_factor.dropna().mean()),
            "worst_fold_drawdown": float(g.max_drawdown.min()),
            "folds_logloss_better_than_prevalence": int((g.log_loss < g.baseline_log_loss).sum()),
        })
    per_seed_df = pd.DataFrame(per_seed)
    num_cols = [
        "rows", "up_fraction", "accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier",
        "precision", "recall", "f1", "ece_10", "baseline_log_loss", "baseline_accuracy",
        "compounded_net_return", "compounded_stress_return", "trades", "mean_profit_factor",
        "worst_fold_drawdown", "folds_logloss_better_than_prevalence",
    ]
    table = per_seed_df.groupby(["symbol", "recipe"])[num_cols].mean()

    # Paired bootstrap on common dates per symbol: compact vs price, and both vs constant 0.5
    paired_compact_vs_price, paired_vs_05, p_comp, p_05 = {}, {}, {}, {}
    for sym in ["USDCAD", "NZDUSD"]:
        diffs_cp = []
        for fold in [0, 1, 2]:
            y = preds[(sym, "price", fold, 42)].target.to_numpy(int)
            loss_p = np.mean([
                logloss_vec(y, preds[(sym, "price", fold, sd)].probability.to_numpy(float))
                for sd in [42, 1729]
            ], axis=0)
            loss_c = np.mean([
                logloss_vec(y, preds[(sym, "compact", fold, sd)].probability.to_numpy(float))
                for sd in [42, 1729]
            ], axis=0)
            diffs_cp.append(loss_p - loss_c)
        paired_compact_vs_price[sym] = paired_bootstrap(diffs_cp)
        p_comp[sym] = paired_compact_vs_price[sym]["p_max_blocks"]
        for rec in ["price", "compact"]:
            diffs_05 = []
            for fold in [0, 1, 2]:
                y = preds[(sym, rec, fold, 42)].target.to_numpy(int)
                loss_r = np.mean([
                    logloss_vec(y, preds[(sym, rec, fold, sd)].probability.to_numpy(float))
                    for sd in [42, 1729]
                ], axis=0)
                diffs_05.append(logloss_vec(y, np.full_like(y, 0.5, float)) - loss_r)
            k = f"{sym}_{rec}"
            paired_vs_05[k] = paired_bootstrap(diffs_05)
            p_05[k] = paired_vs_05[k]["p_max_blocks"]
    for k, v in holm(p_comp).items():
        paired_compact_vs_price[k]["p_holm"] = v
    for k, v in holm(p_05).items():
        paired_vs_05[k]["p_holm"] = v

    out = Path("artifacts/expansion_analysis") / STUDY.name / datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in {"fold_seed_metrics.csv": f, "per_seed.csv": per_seed_df}.items():
        with (out / name).open("x", newline="", encoding="utf-8") as stream:
            frame.to_csv(stream, index=False)
    write_new_json(out / "summary.json", {
        "study": str(STUDY.resolve()), "new_fits": 0,
        "table": table.reset_index().to_dict(orient="records"),
        "paired_logloss_compact_vs_price": paired_compact_vs_price,
        "paired_logloss_vs_constant_05": paired_vs_05,
    })
    write_new_json(out / "outputs.json", {p.name: sha256(p) for p in sorted(out.iterdir())})
    print(json.dumps({"out": str(out)}))
    print(table.round(5).to_string())
    print("compact_vs_price:", json.dumps(paired_compact_vs_price))
    print("vs_05:", json.dumps(paired_vs_05))


if __name__ == "__main__":
    main()
