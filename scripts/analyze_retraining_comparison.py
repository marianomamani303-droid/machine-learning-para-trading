"""Paired temporal inference and artifact verification; never fits models."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.evaluation.uncertainty import _auc, moving_block_indices

OUT = Path("artifacts/retraining_comparison/20261017_controlled")


def losses(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def holm(pvalues):
    names = sorted(pvalues, key=pvalues.get)
    result = {}
    running = 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * pvalues[name]))
        result[name] = running
    return result


def paired_inference(a, b, groups, samples=2000, blocks=(7, 30)):
    """Arrays date x seed. Resample dates jointly across seeds, never seed rows.

    Confidence intervals are conditional on the fitted models. Fold boundaries
    are respected. Secondary AUC/accuracy tests do not authorize significance.
    """
    y = a["target"].to_numpy(int)
    acols = [c for c in a if c.startswith("p_")]
    ap = a[acols].to_numpy(float)
    bp = b[acols].to_numpy(float)
    delta_loss = (losses(y[:, None], ap) - losses(y[:, None], bp)).mean(axis=1)
    delta_acc = (
        ((bp >= 0.5) == y[:, None]).astype(float) - ((ap >= 0.5) == y[:, None]).astype(float)
    ).mean(axis=1)
    delta_brier = ((ap - y[:, None]) ** 2 - (bp - y[:, None]) ** 2).mean(axis=1)
    result = {}
    for block in blocks:
        idx = moving_block_indices(groups, samples, block, 42)
        draws = delta_loss[idx].mean(axis=1)
        point = float(delta_loss.mean())
        # Centered block bootstrap null; one-sided improvement was predeclared.
        p = float((1 + np.sum(draws - point >= point)) / (samples + 1)) if point > 0 else 1.0
        auc = []
        for indices in idx:
            auc.append(
                np.mean(
                    [
                        _auc(y[indices], bp[indices, j]) - _auc(y[indices], ap[indices, j])
                        for j in range(ap.shape[1])
                    ]
                )
            )

        def interval(values):
            values = np.asarray(values)
            return [float(x) for x in np.nanquantile(values, [0.025, 0.975])]

        result[str(block)] = {
            "log_loss_gain": point,
            "log_loss_gain_ci95": interval(draws),
            "primary_p": p,
            "accuracy_delta": float(delta_acc.mean()),
            "accuracy_delta_ci95": interval(delta_acc[idx].mean(axis=1)),
            "brier_gain": float(delta_brier.mean()),
            "brier_gain_ci95": interval(delta_brier[idx].mean(axis=1)),
            "auc_delta": float(
                np.mean([_auc(y, bp[:, j]) - _auc(y, ap[:, j]) for j in range(ap.shape[1])])
            ),
            "auc_delta_ci95": interval(auc),
            "dates": len(y),
            "seeds": len(acols),
        }
    return result


def collect():
    groups = defaultdict(list)
    for directory in ["fits", "events", "forex"]:
        for path in (OUT / directory).rglob("complete.json"):
            parent = path.parent
            if not (parent / "predictions.csv").exists():
                continue
            fit = read_json(parent / "fit.json")
            key = (fit["symbol"], fit["interval"], fit["recipe"], fit["target_kind"])
            groups[key].append((parent, fit))
    return groups


def table_metrics(y, p):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    from trading_ml.evaluation.metrics import classification_metrics

    m = classification_metrics(y, p)
    m["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= 0.5))
    m["confusion_matrix"] = confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist()
    return m


def loss_skill(y, baseline, probability, groups, blocks):
    delta = (losses(y[:, None], baseline) - losses(y[:, None], probability)).mean(axis=1)
    result = {}
    for block in blocks:
        indices = moving_block_indices(groups, 2000, block, 42)
        draws = delta[indices].mean(axis=1)
        point = float(delta.mean())
        result[str(block)] = {
            "gain": point,
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "p": float((1 + np.sum(draws - point >= point)) / 2001) if point > 0 else 1.0,
        }
    return result


def analyze():
    groups = collect()
    records, details, economics, folds, baselines = [], {}, [], [], []
    paired = {}
    skill = {}
    availability = {}
    for key, items in groups.items():
        symbol, interval, recipe, target = key
        name = "|".join(key)
        by_arm = {
            arm: {(m["fold"], m["seed"]) for _, m in items if m["arm"] == arm} for arm in ["A", "B"]
        }
        common = by_arm["A"] & by_arm["B"]
        availability[name] = {
            "paired_fold_seed": sorted(common),
            "excluded_unpaired_fold_seed": sorted(by_arm["A"] ^ by_arm["B"]),
            "reason": "Pair only policies available from past training/calibration; never filter by test outcomes",
        }
        items = [(p, m) for p, m in items if (m["fold"], m["seed"]) in common]
        if not items:
            continue
        frames = {arm: [] for arm in ["A", "B"]}
        metrics_by_arm = {arm: [] for arm in ["A", "B"]}
        seed_economics = defaultdict(list)
        for parent, fit in items:
            arm, seed, fold = fit["arm"], fit["seed"], fit["fold"]
            p = pd.read_csv(parent / "predictions.csv")
            p["seed"] = seed
            p["fold"] = fold
            frames[arm].append(p)
            m = read_json(parent / "predictive.json")
            folds.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "recipe": recipe,
                    "target_kind": target,
                    "arm": arm,
                    "seed": seed,
                    "fold": fold,
                    **{
                        k: m[k]
                        for k in [
                            "rows",
                            "roc_auc",
                            "accuracy",
                            "log_loss",
                            "brier_score",
                            "balanced_accuracy",
                        ]
                    },
                }
            )
            for scenario in ["base", "double"]:
                curve = pd.read_csv(parent / f"{scenario}_curve.csv")
                summary = read_json(parent / f"{scenario}_economic.json")
                trpath = parent / f"{scenario}_trades.csv"
                try:
                    trades = pd.read_csv(trpath)
                except pd.errors.EmptyDataError:
                    trades = pd.DataFrame()
                seed_economics[(arm, seed, scenario)].append((fold, curve, summary, trades))
        for (arm, seed, scenario), parts in seed_economics.items():
            returns = []
            pnls = []
            trades_count = cost = exposure = 0.0
            capital_scale = 1.0
            exposed_seconds = duration_seconds = 0.0
            fx_quotes = None
            if recipe.startswith("fx_"):
                source = next(m["source_original"] for _, m in items)
                fx_quotes = pd.read_csv(source).set_index("timestamp")
                fx_quotes.index = pd.to_datetime(fx_quotes.index, utc=True)
            for _fold, curve, summary, trades in sorted(parts, key=lambda p: p[0]):
                equity = curve.equity.to_numpy()
                # Forex engine omits its initial cash row, unlike event engine.
                if recipe.startswith("fx_"):
                    equity = np.r_[10000.0, equity]
                if recipe.startswith("fx_"):
                    from trading_ml.multimarket.markets import session_end

                    clock = pd.to_datetime(curve.timestamp, utc=True)
                    sessions = session_end(pd.DatetimeIndex(clock) - pd.Timedelta(nanoseconds=1))
                    daily = (
                        curve.assign(session=sessions).groupby("session").equity.last().to_numpy()
                    )
                    returns.extend(daily / np.r_[10000.0, daily[:-1]] - 1)
                else:
                    returns.extend(equity[1:] / equity[:-1] - 1)
                trades_count += summary.get("trade_count", summary.get("trades", 0))
                cost += summary.get("execution_cost_cash", 0.0) * capital_scale
                exposure += summary.get("fraction_bars_exposed", 0.0) / len(parts)
                if len(trades):
                    col = "profit_usd" if "profit_usd" in trades else "pnl_cash"
                    pnls.extend(trades[col].to_numpy() * capital_scale)
                    if recipe.startswith("fx_"):
                        entry_times = pd.to_datetime(trades.entry_time, utc=True)
                        exit_times = pd.to_datetime(trades.target_time, utc=True)
                        bid_entry = fx_quotes.open_bid.reindex(entry_times).to_numpy()
                        bid_exit = fx_quotes.open_bid.reindex(exit_times).to_numpy()
                        assert np.isfinite(bid_entry).all() and np.isfinite(bid_exit).all()
                        execution = (
                            (trades.entry_price.to_numpy() - bid_entry)
                            + (bid_exit - trades.exit_price.to_numpy())
                            + trades.fees_per_unit.to_numpy()
                        ) * trades.units.to_numpy()
                        assert (execution >= -1e-8).all()
                        cost += float(execution.sum()) * capital_scale
                        exposed_seconds += float(
                            (exit_times - entry_times).dt.total_seconds().sum()
                        )
                if recipe.startswith("fx_"):
                    duration_seconds += float(
                        (
                            pd.to_datetime(curve.timestamp.iloc[-1], utc=True)
                            - pd.to_datetime(curve.timestamp.iloc[0], utc=True)
                        ).total_seconds()
                    )
                capital_scale *= float(equity[-1] / equity[0])
            r = np.array(returns)
            equity = np.r_[1.0, np.cumprod(1 + r)]
            pnl = np.asarray(pnls)
            negative = -pnl[pnl < 0].sum()
            periods = 252 if recipe.startswith("fx_") else 365
            economics.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "recipe": recipe,
                    "target_kind": target,
                    "arm": arm,
                    "seed": seed,
                    "scenario": scenario,
                    "net_return": float(equity[-1] - 1),
                    "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(periods))
                    if r.std(ddof=1) > 1e-12
                    else None,
                    "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
                    "trades": int(trades_count),
                    "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
                    "profit_factor": float(pnl[pnl > 0].sum() / negative) if negative > 0 else None,
                    "execution_cost_cash_compounded": cost,
                    "exposure": exposure
                    if not recipe.startswith("fx_")
                    else exposed_seconds / duration_seconds,
                }
            )
        for arm in ["A", "B"]:
            frame = pd.concat(frames[arm], ignore_index=True)
            frames[arm] = frame
            for seed, part in frame.groupby("seed"):
                part = part.sort_values("reference_time")
                m = table_metrics(part.target.to_numpy(int), part.probability.to_numpy())
                details[f"{name}|{arm}|{seed}"] = m
                metrics_by_arm[arm].append(m)
        # Keyed alignment is mandatory, including labels and fold IDs.
        wide = {}
        for arm in ["A", "B"]:
            frame = frames[arm]
            assert not frame.duplicated(["reference_time", "seed"]).any()
            w = frame.pivot(
                index=["reference_time", "target_time", "target", "fold"],
                columns="seed",
                values="probability",
            ).sort_index()
            assert not w.isna().any().any()
            w.columns = [f"p_{s}" for s in w.columns]
            wide[arm] = w.reset_index()
        identity = ["reference_time", "target_time", "target", "fold"]
        assert wide["A"][identity].equals(wide["B"][identity]), f"Nonpaired: {name}"
        # Bar blocks converted to day blocks for H4; NY17 uses 5 days/week, conservative.
        blocks = (42, 180) if interval == "4h" else (7, 30)
        inference = paired_inference(wide["A"], wide["B"], wide["A"].fold.to_numpy(), blocks=blocks)
        paired[name] = {
            "inference": inference,
            "primary_p_max_blocks": max(v["primary_p"] for v in inference.values()),
            "dates": len(wide["A"]),
            "seeds": len(metrics_by_arm["A"]),
        }
        if target == "close_direction":
            y = wide["B"].target.to_numpy(int)
            pcols = [c for c in wide["B"] if c.startswith("p_")]
            probs = wide["B"][pcols].to_numpy(float)
            base_probs = {"constant_05": np.full_like(probs, 0.5)}
            prior_by_fold_seed = {
                (fit["fold"], fit["seed"]): fit.get("prior")
                for _, fit in items
                if fit["arm"] == "B"
            }
            if all(v is not None for v in prior_by_fold_seed.values()):
                base_probs["train_prevalence"] = np.column_stack(
                    [
                        [prior_by_fold_seed[(fold, int(c[2:]))] for fold in wide["B"].fold]
                        for c in pcols
                    ]
                )
            if interval == "1d" and symbol in ["BTCUSDT", "ETHUSDT"]:
                logistic_parts = [
                    pd.read_csv(
                        OUT / "fits" / f"{symbol}_1d_f{fold}_s42_B/logistic/predictions.csv"
                    )
                    for fold in range(3)
                ]
                logistic = pd.concat(logistic_parts).set_index("reference_time").probability
                logp = logistic.reindex(wide["B"].reference_time).to_numpy()
                assert np.isfinite(logp).all()
                base_probs["logistic_l2"] = np.repeat(logp[:, None], len(pcols), axis=1)
            for baseline, basep in base_probs.items():
                skill[name + "|" + baseline] = loss_skill(
                    y, basep, probs, wide["B"].fold.to_numpy(), blocks
                )
        metric_names = [
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "brier_score",
            "precision",
            "recall",
            "f1",
            "ece_10_bins",
        ]
        for metric in metric_names:
            before = float(np.mean([m[metric] for m in metrics_by_arm["A"]]))
            after = float(np.mean([m[metric] for m in metrics_by_arm["B"]]))
            records.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "recipe": recipe,
                    "target_kind": target,
                    "metric": metric,
                    "before": before,
                    "after": after,
                    "difference": after - before,
                    "relative_difference": (after - before) / abs(before) if before else None,
                    "dates": len(wide["A"]),
                    "seeds": len(metrics_by_arm["A"]),
                }
            )
        if target == "close_direction":
            y = wide["A"].target.to_numpy(int)
            rng = np.random.default_rng(42)
            for base, prob in [
                ("constant_05", np.full(len(y), 0.5)),
                ("random_coin_seed42", rng.integers(0, 2, len(y)).astype(float)),
            ]:
                bm = table_metrics(y, prob)
                if base == "random_coin_seed42":
                    # Random decisions are not deterministic 0/1 probability forecasts.
                    for score in ["roc_auc", "log_loss", "brier_score", "ece_10_bins"]:
                        bm[score] = None
                baselines.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "recipe": recipe,
                        "baseline": base,
                        **{k: bm[k] for k in metric_names},
                    }
                )
            # Historical fit prevalence per fold/seed, never test prevalence.
            for arm in ["A", "B"]:
                for seed in frames[arm].seed.unique():
                    f = (
                        frames[arm]
                        .loc[frames[arm].seed == seed]
                        .sort_values(["fold", "reference_time"])
                    )
                    priors = {
                        fit["fold"]: fit.get("prior", fit.get("audit", {}).get("train_prior"))
                        for _, fit in items
                        if fit["arm"] == arm and fit["seed"] == seed
                    }
                    if any(v is None for v in priors.values()):
                        continue
                    prior = f.fold.map(priors).to_numpy(float)
                    for bn, prob in [
                        ("train_prevalence", prior),
                        ("majority_class", (prior >= 0.5).astype(float)),
                    ]:
                        bm = table_metrics(f.target.to_numpy(int), prob)
                        baselines.append(
                            {
                                "symbol": symbol,
                                "interval": interval,
                                "recipe": recipe,
                                "arm": arm,
                                "seed": int(seed),
                                "baseline": bn,
                                **{k: bm[k] for k in metric_names},
                            }
                        )
        print(
            "ANALYZED",
            name,
            "logloss gain",
            round(next(iter(inference.values()))["log_loss_gain"], 6),
            flush=True,
        )
    adjusted = holm({name: v["primary_p_max_blocks"] for name, v in paired.items()})
    skill_adjusted = holm(
        {name: max(v["p"] for v in result.values()) for name, result in skill.items()}
    )
    for name, p in adjusted.items():
        paired[name]["holm_p"] = p
        paired[name]["statistically_supported_improvement"] = p < 0.05 and all(
            v["log_loss_gain_ci95"][0] > 0 for v in paired[name]["inference"].values()
        )
    pd.DataFrame(records).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(economics).to_csv(OUT / "economic_comparison.csv", index=False)
    pd.DataFrame(baselines).to_csv(OUT / "baselines.csv", index=False)
    write_new_json(OUT / "all_predictive_metrics.json", details)
    write_new_json(OUT / "paired_inference.json", paired)
    write_new_json(OUT / "paired_availability.json", availability)
    write_new_json(
        OUT / "baseline_skill_inference.json",
        {name: {"blocks": v, "holm_p": skill_adjusted[name]} for name, v in skill.items()},
    )
    plot(pd.DataFrame(records), details, paired)
    verify()


def plot(records, details, paired):
    fig, axes = plt.subplots(1, 3, figsize=(17, 9))
    for ax, metric, title in zip(
        axes,
        ["roc_auc", "accuracy", "log_loss"],
        ["ROC-AUC (B − A)", "Accuracy (B − A)", "Log-loss (B − A; negative improves)"],
        strict=True,
    ):
        rows = records.loc[records.metric == metric].sort_values(
            ["target_kind", "symbol", "interval", "recipe"]
        )
        labels = rows.symbol + " " + rows.interval + " " + rows.recipe
        ax.barh(
            labels,
            rows.difference,
            color=np.where((rows.difference > 0) ^ (metric == "log_loss"), "#197b62", "#b94c4c"),
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Controlled historical comparison — same evaluation observations; seed means")
    fig.tight_layout()
    fig.savefig(OUT / "predictive_differences.png", dpi=160)
    fig.savefig(OUT / "predictive_differences.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, d in paired.items():
        v = list(d["inference"].values())[-1]
        low, high = v["log_loss_gain_ci95"]
        point = v["log_loss_gain"]
        ax.errorbar(
            point,
            name,
            xerr=[[max(point - low, 0)], [max(high - point, 0)]],
            fmt="o",
            color="#256c9d",
            markersize=3,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-loss improvement A − B, pointwise 95% temporal block CI")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "logloss_intervals.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, symbol in zip(axes, ["BTCUSDT", "ETHUSDT"], strict=True):
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
        for recipe, color in [("original", "#196b9b"), ("logistic", "#a05b22")]:
            for arm, style in [("A", "--"), ("B", "-")]:
                m = details[f"{symbol}|1d|{recipe}|close_direction|{arm}|42"]
                bins = m["reliability_bins"]
                ax.plot(
                    [b["mean_probability"] for b in bins],
                    [b["observed_up_fraction"] for b in bins],
                    style,
                    color=color,
                    marker="o",
                    label=recipe + " " + arm,
                )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean forecast probability",
            ylabel="Observed up fraction",
            title=symbol + " seed 42",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=160)
    plt.close(fig)


def verify():
    previous = read_json(OUT / "protected_before.json")
    changed = []

    def check_old(item):
        path, digest = item
        return path if not Path(path).exists() or sha256(Path(path)) != digest else None

    with ThreadPoolExecutor(max_workers=4) as pool:
        changed = [path for path in pool.map(check_old, previous.items()) if path]
    fits = []
    for p in OUT.rglob("fit.json"):
        m = read_json(p)
        if "actual_fits" in m:
            fits.append(m["actual_fits"])
    oof = len(list((OUT / "events").glob("*_*/oof/*/fit.json")))
    unavailable = [read_json(p) for p in (OUT / "events").rglob("unavailable.json")]
    unavailable_fits = sum(int(m.get("status") == "fitted") for m in unavailable)
    result = {
        "old_artifact_files_verified": len(previous),
        "changed_old_artifacts": changed,
        "actual_estimator_and_calibration_fits": sum(fits) + oof + unavailable_fits,
        "unavailable_calibrated_event_policies": len(unavailable),
        "raw_fits_without_sufficient_calibration": unavailable_fits,
        "oof_fits": oof,
        "comparison_endpoints": len(collect()),
        "quarantine_used": False,
        "promoted": False,
    }
    write_new_json(OUT / "verification.json", result)
    assert not changed
    assert (
        result["actual_estimator_and_calibration_fits"]
        <= read_json(OUT / "protocol_amendment_before_fitting.json")["maximum_fits"]
    )
    manifests = {
        str(p.relative_to(OUT)): sha256(p)
        for p in OUT.rglob("*")
        if p.is_file() and p.name != "artifacts.json"
    }
    write_new_json(OUT / "artifacts.json", manifests)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    analyze()
