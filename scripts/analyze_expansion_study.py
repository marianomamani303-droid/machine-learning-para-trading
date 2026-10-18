"""Inference-only analysis of the completed 72-fit market-expansion study.

No model is fitted. Stored XGBoost checkpoints are reloaded and their seed-mean
probabilities must reproduce the probabilities saved by the study before any
metric is computed. LSTM signals are read from verified `signals.npy`.
Comparisons are paired on identical BTC/ETH test dates and labels.
Output is append-only under artifacts/expansion_analysis/<run_id>/<stamp>.
"""

from __future__ import annotations

import argparse
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
from xgboost import XGBClassifier

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.dmn.data import temporal_split
from trading_ml.dmn.study import _verify_files
from trading_ml.expansion.study import SCENARIOS, prepare

EPS = 1e-15
REFERENCE = "anchor_price"
CANDIDATES = [s for s in SCENARIOS if s != REFERENCE]


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


def paired_bootstrap(diff_by_fold, reps, blocks, seed=20261017):
    """Circular moving blocks inside each fold; one-sided centered p for mean>0."""
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


def auc_bootstrap(y_by_fold, a_by_fold, b_by_fold, reps, block, seed=7):
    """CI for AUC(b) - AUC(a) on date blocks (rows are date x asset)."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        ys, pa, pb = [], [], []
        for y, a, b in zip(y_by_fold, a_by_fold, b_by_fold, strict=True):
            idx = block_indices(len(y), block, rng)
            ys.append(y[idx].ravel())
            pa.append(a[idx].ravel())
            pb.append(b[idx].ravel())
        ys = np.concatenate(ys)
        draws.append(
            roc_auc_score(ys, np.concatenate(pb)) - roc_auc_score(ys, np.concatenate(pa))
        )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def holm(pvalues: dict) -> dict:
    keys = sorted(pvalues, key=pvalues.get)
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", type=Path, default=Path("artifacts/expansion/studies/20261016T211432_bcec9ebe")
    )
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args()
    study = args.study.resolve()
    plan = read_json(study / "plan.json")
    summary = read_json(study / "summary.json")
    _verify_files({str(study / k): v for k, v in read_json(study / "artifacts.json").items()})
    _verify_files(plan["inputs"])
    p = plan["protocol"]
    index = next(Path(k) for k in plan["inputs"] if "datasets" in k and k.endswith(".json"))
    panels, _ = prepare(Path("artifacts"), p, read_json(index))
    anchors = p["data"]["anchors"]
    seeds = p["research"]["seeds"]
    blocks = p["bootstrap"]["block_lengths"]
    bounds = pd.to_datetime(p["research"]["boundaries"], utc=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = Path("artifacts/expansion_analysis") / plan["run_id"] / stamp
    out.mkdir(parents=True, exist_ok=False)

    # probabilities[scenario][seed] -> list per fold of (dates x 2) arrays; labels per fold
    prob = {s: {seed: [] for seed in seeds} for s in SCENARIOS}
    lstm = {s: {seed: [] for seed in seeds} for s in SCENARIOS}
    labels, prevalence, fold_dates, reproduction = [], [], [], {}
    for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        fold_dir = study / f"fold_{fold:02d}"
        y_ref = None
        for scenario in SCENARIOS:
            panel = panels[scenario]
            train, _, test = temporal_split(panel, start, end, p["training"])
            ids = [panel.symbols.index(a) for a in anchors]
            y = (panel.returns[test][:, ids] > 0).astype(int)
            if y_ref is None:
                y_ref, times_ref = y, panel.times[test]
                prevalence.append(float((panel.returns[train][:, ids] > 0).mean()))
                labels.append(y)
                fold_dates.append([str(times_ref[0]), str(times_ref[-1]), len(test)])
            elif not (np.array_equal(y, y_ref) and panel.times[test].equals(times_ref)):
                raise ValueError("Test labels/dates differ between scenarios")
            x = panel.x[test, :, -1, :].reshape(-1, len(panel.feature_names))
            for seed in seeds:
                model = XGBClassifier()
                model.load_model(fold_dir / f"{scenario}__xgboost__seed_{seed}" / "model.ubj")
                prob[scenario][seed].append(
                    model.predict_proba(x)[:, 1].reshape(len(test), -1)[:, ids]
                )
                sig = np.load(
                    fold_dir / f"{scenario}__lstm__seed_{seed}" / "signals.npy", allow_pickle=False
                )
                lstm[scenario][seed].append(sig[:, ids])
            stored = pd.read_csv(fold_dir / f"{scenario}__xgboost_probabilities.csv")
            mean = np.mean([prob[scenario][s][-1] for s in seeds], axis=0).ravel()
            err = float(np.max(np.abs(stored.probability.to_numpy() - mean)))
            reproduction[f"fold_{fold}/{scenario}"] = err
            if err > 1e-6 or not np.array_equal(stored.target.to_numpy(), y.ravel()):
                raise ValueError(f"Stored probabilities not reproduced: {fold}/{scenario} {err}")

    y_all = np.concatenate(labels)
    rows, per_fold, per_asset = [], [], []
    for scenario in SCENARIOS:
        for seed in seeds:
            pr = np.concatenate(prob[scenario][seed])
            rows.append({"scenario": scenario, "seed": seed, **classification(y_all.ravel(), pr.ravel())})
            for j, a in enumerate(anchors):
                per_asset.append(
                    {"scenario": scenario, "seed": seed, "asset": a,
                     **classification(y_all[:, j], pr[:, j])}
                )
            for fold in range(len(labels)):
                per_fold.append(
                    {"scenario": scenario, "seed": seed, "fold": fold,
                     **classification(labels[fold].ravel(), prob[scenario][seed][fold].ravel())}
                )
    seed_table = pd.DataFrame(rows)
    fold_table = pd.DataFrame(per_fold)
    asset_table = pd.DataFrame(per_asset)
    keys = ["accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier",
            "precision", "recall", "f1", "ece_10"]
    mean_table = seed_table.groupby("scenario")[keys].mean().reindex(list(SCENARIOS))
    std_table = seed_table.groupby("scenario")[keys].std(ddof=0).reindex(list(SCENARIOS))

    # Baselines on identical rows (train-only prevalence per fold)
    base = {
        "constant_05": classification(y_all.ravel(), np.full(y_all.size, 0.5)),
        "train_prevalence": classification(
            y_all.ravel(),
            np.concatenate([np.full(y_f.size, prevalence[i]) for i, y_f in enumerate(labels)]),
        ),
    }

    def date_loss(scenario, fold):
        return np.mean(
            [logloss_vec(labels[fold], prob[scenario][s][fold]).mean(axis=1) for s in seeds], axis=0
        )

    inference, p_data, p_ctrl, ctrl = {}, {}, {}, {}
    for scenario in CANDIDATES:
        diffs = [date_loss(REFERENCE, f) - date_loss(scenario, f) for f in range(len(labels))]
        res = paired_bootstrap(diffs, args.reps, blocks)
        res["folds_with_gain"] = int(sum(d.mean() > 0 for d in diffs))
        mean_a = [np.mean([prob[REFERENCE][s][f] for s in seeds], axis=0) for f in range(len(labels))]
        mean_b = [np.mean([prob[scenario][s][f] for s in seeds], axis=0) for f in range(len(labels))]
        res["auc_diff_ci95_block30"] = auc_bootstrap(labels, mean_a, mean_b, args.reps // 4, 30)
        inference[scenario] = res
        p_data[scenario] = res["p_max_blocks"]
    for scenario in SCENARIOS:
        diffs = [logloss_vec(labels[f], np.full(labels[f].shape, 0.5)).mean(axis=1)
                 - date_loss(scenario, f) for f in range(len(labels))]
        ctrl[scenario] = paired_bootstrap(diffs, args.reps, blocks)
        p_ctrl[scenario] = ctrl[scenario]["p_max_blocks"]
    for k, v in holm(p_data).items():
        inference[k]["p_holm"] = v
    for k, v in holm(p_ctrl).items():
        ctrl[k]["p_holm"] = v

    # LSTM: allocation ranking vs next-return sign (descriptive only; no log-loss)
    lstm_rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            s = np.concatenate(lstm[scenario][seed])
            d = (s >= 0.5).astype(int)
            lstm_rows.append({
                "scenario": scenario, "seed": seed,
                "auc_score": float(roc_auc_score(y_all.ravel(), s.ravel())),
                "accuracy_ge_05": float((d.ravel() == y_all.ravel()).mean()),
                "balanced_accuracy": float(balanced_accuracy_score(y_all.ravel(), d.ravel())),
                "mean_allocation": float(s.mean()),
            })
    lstm_table = pd.DataFrame(lstm_rows)

    econ = []
    for scenario in SCENARIOS:
        for arch in ["xgboost", "lstm"]:
            for scope in ["anchor", "universe"]:
                m = summary["models"][f"{scenario}__{arch}__{scope}"]
                econ.append({
                    "scenario": scenario, "architecture": arch, "scope": scope,
                    **{f"net_{k}": m["net"][k] for k in
                       ["total_return", "sharpe", "max_drawdown", "total_turnover",
                        "summed_cost_fractions", "mean_gross_exposure"]},
                    "double_cost_total_return": m["double_cost"]["total_return"],
                    "double_cost_sharpe": m["double_cost"]["sharpe"],
                })
    for b in ["cash", "long_scaled", "tsmom"]:
        m = summary["models"][f"{b}__anchor"]
        econ.append({"scenario": b, "architecture": "baseline", "scope": "anchor",
                     **{f"net_{k}": m["net"][k] for k in
                        ["total_return", "sharpe", "max_drawdown", "total_turnover",
                         "summed_cost_fractions", "mean_gross_exposure"]},
                     "double_cost_total_return": m["double_cost"]["total_return"],
                     "double_cost_sharpe": m["double_cost"]["sharpe"]})
    econ_table = pd.DataFrame(econ)

    for name, frame in {
        "xgb_seed_metrics.csv": seed_table, "xgb_fold_metrics.csv": fold_table,
        "xgb_asset_metrics.csv": asset_table, "lstm_direction.csv": lstm_table,
        "economic.csv": econ_table,
    }.items():
        with (out / name).open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
    result = {
        "study": str(study), "run_id": plan["run_id"], "fits_reused": summary["actual_fits"],
        "new_fits": 0, "fold_test_dates": fold_dates, "train_prevalence_by_fold": prevalence,
        "probability_reproduction_max_abs_error": reproduction,
        "xgb_mean_over_seeds": mean_table.to_dict(orient="index"),
        "xgb_seed_std": std_table.to_dict(orient="index"),
        "baselines": base, "paired_logloss_vs_anchor": inference,
        "logloss_gain_vs_constant_05": ctrl,
        "paired_sharpe_intervals_from_study": summary["paired_uncertainty"],
        "reps": args.reps, "blocks": blocks,
        "note": "Retrospective development folds; not an independent test.",
    }
    write_new_json(out / "analysis.json", result)
    write_new_json(out / "outputs.json", {f.name: sha256(f) for f in sorted(out.iterdir())})
    print(json.dumps({"out": str(out), "reproduction_max": max(reproduction.values())}))
    print(mean_table.round(5).to_string())
    print(std_table[["accuracy", "roc_auc", "log_loss"]].round(5).to_string())
    print(json.dumps(base, default=str)[:400])
    print(pd.DataFrame({k: {"gain": v["mean_gain"], "p_max": v["p_max_blocks"], "p_holm": v["p_holm"],
                            "folds": v["folds_with_gain"], "auc_ci": v["auc_diff_ci95_block30"],
                            "ci30": v["block_30"]["ci95"]} for k, v in inference.items()}).T.to_string())
    print(pd.DataFrame({k: {"gain_vs_05": v["mean_gain"], "p_holm": v["p_holm"]} for k, v in ctrl.items()}).T.to_string())
    print(lstm_table.groupby("scenario")[["auc_score", "accuracy_ge_05", "mean_allocation"]].mean().to_string())
    print(econ_table.query("scope=='anchor'")[["scenario", "architecture", "net_total_return", "net_sharpe", "net_max_drawdown", "double_cost_total_return"]].to_string())


if __name__ == "__main__":
    main()

