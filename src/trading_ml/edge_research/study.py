from __future__ import annotations

import argparse
import shutil
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from trading_ml.common import (
    code_provenance,
    environment,
    read_json,
    sha256,
    sidecar,
    write_new_json,
)
from trading_ml.dmn.study import _source_archive, _verify_files
from trading_ml.edge_research.execution import (
    daily_returns,
    economic_metrics,
    execute,
    payoff_threshold,
)
from trading_ml.edge_research.features import DELTAS, load_development, partitions, supervised
from trading_ml.edge_research.models import fit_or_load, predict
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import moving_block_indices
from trading_ml.training.recipe import protect_reserved_labels


def verify_preserved(manifest):
    paths = [Path(p) for p in manifest]
    with ThreadPoolExecutor(max_workers=12) as pool:
        digests = list(pool.map(sha256, paths))
    if any(digest != manifest[str(path)] for path, digest in zip(paths, digests, strict=True)):
        raise DataError("Previous artifact changed")


def loss(y, p):
    p = np.clip(np.asarray(p), 1e-8, 1 - 1e-8)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def paired_test(frame, value, protocol):
    values = frame[value].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(frame["day"])
    split = (dates.to_series(index=range(len(dates))).diff().ne(pd.Timedelta(days=1))).to_numpy()
    split = split | frame.fold.ne(frame.fold.shift()).to_numpy()
    groups = np.cumsum(split)
    intervals = []
    for block in protocol["bootstrap_days"]:
        try:
            indices = moving_block_indices(
                groups, protocol["bootstrap_samples"], block, protocol["seed"]
            )
        except DataError:
            intervals.append(
                {
                    "block_days": block,
                    "lower": None,
                    "upper": None,
                    "p": 1.0,
                    "status": "insufficient_contiguous_days",
                }
            )
            continue
        sampled = values[indices].mean(axis=1)
        centered = sampled - values.mean()
        p = (1 + np.sum(centered >= values.mean())) / (len(sampled) + 1)
        intervals.append(
            {
                "block_days": block,
                "lower": float(np.quantile(sampled, 0.025)),
                "upper": float(np.quantile(sampled, 0.975)),
                "p": float(p),
                "status": "exploratory_reused_history",
            }
        )
    return {
        "mean_gain": float(values.mean()),
        "p": max(v["p"] for v in intervals),
        "intervals": intervals,
    }


def holm(tests):
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(tests) - rank) * tests[i]["p"]))
        tests[i]["p_holm"] = previous


def append_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def concatenate_curves(parts):
    capital, hold = 1.0, 1.0
    output = []
    for curve in parts:
        f = curve.copy()
        f["equity"] *= capital
        f["buy_hold_equity"] *= hold
        capital, hold = float(f.equity.iloc[-1]), float(f.buy_hold_equity.iloc[-1])
        output.append(f)
    return pd.concat(output, ignore_index=True)


def run(protocol_path: Path, resume: Path | None = None):
    with protocol_path.open("rb") as stream:
        protocol = tomllib.load(stream)
    if (
        protocol["maximum_fits"] != 120
        or protocol["end_exclusive"] != "2025-07-01"
        or protocol["recipes"] != ["logistic_price", "logistic_full", "logistic_noise", "xgb_full"]
    ):
        raise DataError("This runner requires the frozen 120-fit protocol")
    root = Path("artifacts")
    preserved = root / "edge_research" / "preservation_before.json"
    if not preserved.exists():
        raise DataError("Run snapshot_edge_research.py before research")
    source_inputs = {}
    for symbol in protocol["symbols"]:
        for interval in DELTAS:
            candidates = list(
                (root / "historical" / "datasets" / "binance_spot" / symbol).glob(
                    f"{interval}_*.csv"
                )
            )
            if len(candidates) != 1:
                raise DataError("Ambiguous historical input; explicit protocol revision required")
            path = candidates[0]
            source_inputs[f"{symbol}:{interval}"] = {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "metadata_sha256": sha256(sidecar(path)),
            }
    code = code_provenance()
    if resume:
        directory = resume
        plan = read_json(directory / "plan.json")
        if plan["protocol"] != protocol or plan["code"] != code or plan["inputs"] != source_inputs:
            raise DataError("Resume requires identical protocol, source and code")
        if (directory / "summary.json").exists():
            _verify_files(
                {str(directory / p): h for p, h in read_json(directory / "artifacts.json").items()}
            )
            return read_json(directory / "summary.json")
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        directory = root / "edge_research" / "runs" / run_id
        directory.mkdir(parents=True)
        plan = {
            "protocol": protocol,
            "code": code,
            "environment": environment(),
            "inputs": source_inputs,
            "preservation_manifest": str(preserved.resolve()),
            "preservation_sha256": sha256(preserved),
            "status": "exploratory_not_confirmation",
            "created_at": datetime.now(UTC).isoformat(),
        }
        write_new_json(directory / "plan.json", plan)
        _source_archive(directory, protocol_path)
        shutil.copyfile("EDGE_RESEARCH.md", directory / "preregistered_protocol.md")
    boundaries = pd.to_datetime(protocol["boundaries"], utc=True)
    predictive, economic, tests, economic_tests, quality, diagnostics = [], [], [], [], [], []
    total_fits = 0
    scenarios = {"gross": (0, 0), "base": (1, 0), "double": (2, 0), "delay": (1, 1)}
    for symbol in protocol["symbols"]:
        raw_by_interval = {
            interval: load_development(
                Path(source_inputs[f"{symbol}:{interval}"]["path"]),
                protocol["start"],
                protocol["end_exclusive"],
            )
            for interval in DELTAS
        }
        for task in protocol["tasks"]:
            interval, horizon_text = task.split(":")
            horizon = int(horizon_text)
            frame = supervised(
                raw_by_interval[interval], interval, horizon, protocol["end_exclusive"]
            )
            protect_reserved_labels(root, frame.loc[frame.eligible])
            task_id = f"{symbol}_{interval}_h{horizon}"
            quality.append(
                {
                    "task": task_id,
                    "rows": len(frame),
                    "eligible": int(frame.eligible.sum()),
                    "gaps": int(frame.timestamp.diff().iloc[1:].ne(DELTAS[interval]).sum()),
                    "first": frame.timestamp.min().isoformat(),
                    "last": frame.timestamp.max().isoformat(),
                    "latest_target": frame.loc[frame.eligible, "target_time"].max().isoformat(),
                }
            )
            prediction_parts = []
            curve_parts, trade_parts = {}, {}
            fold_summaries = {}
            for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
                fit, validation, test = partitions(
                    frame, start, end, protocol["validation_days"], protocol["embargo_hours"]
                )
                tau = payoff_threshold(validation, protocol)
                folder = directory / task_id / f"fold_{fold}"
                folder.mkdir(parents=True, exist_ok=True)
                split_record = {
                    "fit_rows": len(fit),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "fit_target_end": fit.target_time.max().isoformat(),
                    "validation_start": validation.reference_time.min().isoformat(),
                    "selection_end": validation.target_time.max().isoformat(),
                    "test_start": test.reference_time.min().isoformat(),
                    "test_target_end": test.target_time.max().isoformat(),
                    "payoff_tau": tau,
                }
                if not (folder / "split.json").exists():
                    write_new_json(folder / "split.json", split_record)
                predictions = test[
                    [
                        "timestamp",
                        "reference_time",
                        "entry_time",
                        "target_time",
                        "target",
                        "gross_return",
                        "trend_long",
                        "volatility",
                        "relative_volume",
                    ]
                ].copy()
                predictions["fold"] = fold
                predictions["constant_05"] = 0.5
                predictions["prevalence"] = float(fit.target.mean())
                for recipe in protocol["recipes"]:
                    model_directory = folder / recipe
                    state, _ = fit_or_load(model_directory, fit, validation, recipe, protocol)
                    total_fits += 2 if recipe == "xgb_full" else 1
                    if total_fits > protocol["maximum_fits"]:
                        raise DataError("Fit budget exceeded")
                    prob = predict(model_directory, state, test)
                    predictions[recipe] = prob
                    raw_prob = predict(model_directory, state, test, raw=True)
                    if recipe == "xgb_full":
                        predictions["xgb_raw"] = raw_prob
                    diagnostics.append(
                        {
                            "task": task_id,
                            "fold": fold,
                            "recipe": recipe,
                            "train_raw": classification_metrics(
                                fit.target, predict(model_directory, state, fit, raw=True)
                            ),
                            "validation_raw": classification_metrics(
                                validation.target,
                                predict(model_directory, state, validation, raw=True),
                            ),
                            "test_raw": classification_metrics(test.target, raw_prob),
                            "test": classification_metrics(test.target, prob),
                        }
                    )
                    for policy in ["fixed_055", "payoff_risk"]:
                        for scenario, (multiplier, delay) in scenarios.items():
                            key = (recipe, policy, scenario)
                            curve, trades = execute(
                                frame,
                                test,
                                prob,
                                horizon,
                                protocol,
                                policy=policy,
                                tau=tau,
                                multiplier=multiplier,
                                delay=delay,
                            )
                            curve["fold"] = fold
                            trades["fold"] = fold
                            curve_parts.setdefault(key, []).append(curve)
                            trade_parts.setdefault(key, []).append(trades)
                            fold_summaries.setdefault(key, []).append(
                                economic_metrics(curve, trades)
                            )
                for scenario, (multiplier, delay) in scenarios.items():
                    key = ("trend", "trend_risk", scenario)
                    curve, trades = execute(
                        frame,
                        test,
                        np.ones(len(test)),
                        horizon,
                        protocol,
                        policy="trend_risk",
                        tau=0.5,
                        multiplier=multiplier,
                        delay=delay,
                    )
                    curve["fold"], trades["fold"] = fold, fold
                    curve_parts.setdefault(key, []).append(curve)
                    trade_parts.setdefault(key, []).append(trades)
                    fold_summaries.setdefault(key, []).append(economic_metrics(curve, trades))
                if not (folder / "predictions.csv").exists():
                    append_csv(folder / "predictions.csv", predictions)
                else:
                    saved = pd.read_csv(folder / "predictions.csv")
                    for name in protocol["recipes"]:
                        if not np.allclose(saved[name], predictions[name], rtol=0, atol=1e-12):
                            raise DataError("Stored probabilities not reproduced")
                prediction_parts.append(predictions)
                print(f"Completed {task_id} fold {fold}; {total_fits}/120 fits", flush=True)
            pred = pd.concat(prediction_parts, ignore_index=True)
            y = pred.target.to_numpy()
            for recipe in [*protocol["recipes"], "constant_05", "prevalence", "xgb_raw"]:
                per_fold = [
                    classification_metrics(g.target, g[recipe]) for _, g in pred.groupby("fold")
                ]
                predictive.append(
                    {
                        "task": task_id,
                        "recipe": recipe,
                        "rows": len(pred),
                        **classification_metrics(y, pred[recipe]),
                        "folds": per_fold,
                    }
                )
            for recipe, comparison in [
                ("logistic_full", "logistic_price"),
                ("logistic_full", "logistic_noise"),
                ("logistic_full", "constant_05"),
                ("xgb_full", "constant_05"),
            ]:
                differences = (
                    pd.DataFrame(
                        {
                            "day": pred.reference_time.dt.floor("D"),
                            "fold": pred.fold,
                            "gain": loss(y, pred[comparison]) - loss(y, pred[recipe]),
                        }
                    )
                    .groupby(["fold", "day"], as_index=False)
                    .gain.mean()
                )
                tests.append(
                    {
                        "task": task_id,
                        "recipe": recipe,
                        "comparison": comparison,
                        **paired_test(differences, "gain", protocol),
                    }
                )
            for key, curves in curve_parts.items():
                recipe, policy, scenario = key
                curve = concatenate_curves(curves)
                trades = pd.concat(trade_parts[key], ignore_index=True)
                stats = {
                    "task": task_id,
                    "recipe": recipe,
                    "policy": policy,
                    "scenario": scenario,
                    **economic_metrics(curve, trades),
                    "folds": fold_summaries[key],
                    "positive_folds": sum(s["net_return"] > 0 for s in fold_summaries[key]),
                }
                economic.append(stats)
                output = directory / task_id / "portfolios" / f"{recipe}_{policy}_{scenario}"
                if not output.with_suffix(".csv").exists():
                    append_csv(output.with_suffix(".csv"), curve)
                    append_csv(output.with_suffix(".trades.csv"), trades)
                if (
                    recipe in {"logistic_full", "xgb_full"}
                    and policy == "payoff_risk"
                    and scenario == "base"
                ):
                    daily_parts = []
                    for fold, part in enumerate(curves):
                        d = daily_returns(part).reset_index().rename(columns={"timestamp": "day"})
                        d["fold"] = fold
                        d["gain"] = d.net_return - d.buy_hold_return
                        daily_parts.append(d)
                    economic_tests.append(
                        {
                            "task": task_id,
                            "recipe": recipe,
                            "comparison": "buy_hold",
                            **paired_test(
                                pd.concat(daily_parts, ignore_index=True), "gain", protocol
                            ),
                        }
                    )
    holm(tests)
    holm(economic_tests)
    candidates = []
    for task_id in sorted({p["task"] for p in predictive}):
        for recipe in ["logistic_full", "xgb_full"]:
            ptests = [t for t in tests if t["task"] == task_id and t["recipe"] == recipe]
            # XGB has no matched nonlinear price/noise control: never eligible by default.
            predictive_pass = (
                recipe == "logistic_full"
                and len(ptests) == 3
                and all(t["mean_gain"] > 0 and t["p_holm"] < protocol["alpha"] for t in ptests)
            )
            etest = next(
                t for t in economic_tests if t["task"] == task_id and t["recipe"] == recipe
            )
            stats = {
                v["scenario"]: v
                for v in economic
                if v["task"] == task_id and v["recipe"] == recipe and v["policy"] == "payoff_risk"
            }
            economic_pass = (
                etest["mean_gain"] > 0
                and etest["p_holm"] < protocol["alpha"]
                and stats["base"]["trades"] >= protocol["minimum_trades"]
                and stats["base"]["positive_folds"] >= protocol["minimum_positive_folds"]
                and stats["double"]["net_return"] > 0
                and stats["delay"]["net_return"] > 0
            )
            candidates.append(
                {
                    "task": task_id,
                    "recipe": recipe,
                    "predictive_pass": predictive_pass,
                    "economic_pass": economic_pass,
                    "eligible_for_shadow": bool(predictive_pass and economic_pass),
                    "profitability_status": "unconfirmed",
                    "automatic_activation": False,
                    "model_directory": str((directory / task_id / "fold_2" / recipe).resolve()),
                    "state_sha256": sha256(directory / task_id / "fold_2" / recipe / "state.json"),
                    "feature_schema": "edge_causal_ohlcv_v1",
                    "execution_label": "next_open_to_open",
                    "monitoring_ready": False,
                    "monitoring_pending": "Distinct future protocol, executable feed and cost-aware ledger; no historical backfill",
                }
            )
    print("Verifying all pre-existing artifacts before closing study", flush=True)
    verify_preserved(read_json(preserved)["files"])
    summary = {
        "directory": str(directory.resolve()),
        "fits": total_fits,
        "predictive": predictive,
        "economic": economic,
        "predictive_tests": tests,
        "economic_tests": economic_tests,
        "data_quality": quality,
        "candidates": candidates,
        "prior_artifacts_verified": read_json(preserved)["count"],
        "protected_unchanged": True,
        "sealed_data_used": False,
        "independent_confirmation": False,
        "promoted_models": 0,
    }
    write_new_json(directory / "diagnostics.json", diagnostics)
    write_new_json(directory / "summary.json", summary)
    write_new_json(
        directory / "shadow_manifest.json",
        {
            "schema": 1,
            "candidates": candidates,
            "prospective_start_rule": "At or after preregistration of a distinct window disjoint from every existing reservation; do not backfill historical predictions",
            "status": "research_only",
            "automatic_activation": False,
            "cost_contract": {
                k: protocol[k] for k in ["commission_bps", "slippage_bps", "half_spread_bps"]
            },
        },
    )
    artifact_hashes = {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    write_new_json(directory / "artifacts.json", artifact_hashes)
    return summary


def verify(directory):
    plan = read_json(directory / "plan.json")
    _verify_files(
        {str(directory / p): h for p, h in read_json(directory / "artifacts.json").items()}
    )
    verify_preserved(read_json(Path(plan["preservation_manifest"]))["files"])
    if sha256(Path(plan["preservation_manifest"])) != plan["preservation_sha256"]:
        raise DataError("Preservation evidence changed")
    predictions, ledgers = 0, 0
    for path in directory.glob("*/fold_*/predictions.csv"):
        rows = pd.read_csv(
            path, parse_dates=["timestamp", "reference_time", "entry_time", "target_time"]
        )
        split = read_json(path.parent / "split.json")
        if not (
            pd.Timestamp(split["fit_target_end"]) < pd.Timestamp(split["validation_start"])
            and pd.Timestamp(split["selection_end"]) < rows.reference_time.min()
        ):
            raise DataError("Invalid stored split")
        for recipe in plan["protocol"]["recipes"]:
            state = read_json(path.parent / recipe / "state.json")
            # Full feature rebuild verified below in the independent reconstruction script.
            if state["selection_end"] != split["selection_end"]:
                raise DataError("Mismatched model cutoff")
        predictions += len(rows)
    summary = read_json(directory / "summary.json")
    for stat in summary["economic"]:
        prefix = (
            directory
            / stat["task"]
            / "portfolios"
            / f"{stat['recipe']}_{stat['policy']}_{stat['scenario']}"
        )
        curve = pd.read_csv(prefix.with_suffix(".csv"), parse_dates=["timestamp"])
        trades = pd.read_csv(prefix.with_suffix(".trades.csv"))
        computed = economic_metrics(curve, trades)
        if any(
            not np.isclose(computed[k], stat[k], rtol=0, atol=1e-10)
            for k in ["net_return", "sharpe", "max_drawdown", "buy_hold_return", "trades"]
        ):
            raise DataError("Economic summary differs from ledger")
        ledgers += 1
    return {
        "files_verified": len(read_json(directory / "artifacts.json")),
        "prior_artifacts_verified": summary["prior_artifacts_verified"],
        "prediction_rows": predictions,
        "portfolios_recomputed": ledgers,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "verify"])
    parser.add_argument("--protocol", type=Path, default=Path("config/edge_research.toml"))
    parser.add_argument("--study", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        print(verify(args.study))
    else:
        result = run(args.protocol, args.study)
        print(
            {k: result[k] for k in ["directory", "fits", "protected_unchanged", "promoted_models"]}
        )


if __name__ == "__main__":
    main()
