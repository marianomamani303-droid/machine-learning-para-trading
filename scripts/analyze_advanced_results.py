"""Read-only analysis of completed research: meta ranking and 19-trade rejections.

No fitting, threshold selection or modification of original study artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import read_json, sha256, write_new_json
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import _auc, moving_block_indices


def calendar_bootstrap(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    active = data.target.notna().to_numpy()
    target = data.target.fillna(0).to_numpy()
    names = ["logistic", "xgb", "training_intercept", "constant_05"]
    metrics = {
        name: classification_metrics(target[active], data.loc[active, name].to_numpy())
        for name in names
    }
    pairs = [("logistic", "training_intercept"), ("xgb", "training_intercept"), ("xgb", "logistic")]
    intervals = []
    statistics = []
    output = {}
    for block in config["block_lengths"]:
        indices = moving_block_indices(
            data.fold.to_numpy(), config["samples"], block, config["seed"]
        )
        sampled_active = active[indices]
        counts = sampled_active.sum(axis=1)
        draws = {}
        for name in names:
            raw = data[name].fillna(0.5).to_numpy()
            p = np.clip(raw, 1e-12, 1 - 1e-12)
            loss = -(target * np.log(p) + (1 - target) * np.log1p(-p)) * active
            draws[name] = {
                "log_loss": loss[indices].sum(axis=1) / np.maximum(counts, 1),
                "roc_auc": np.array(
                    [
                        _auc(target[row[mask]], raw[row[mask]]) if mask.any() else np.nan
                        for row, mask in zip(indices, sampled_active, strict=True)
                    ]
                ),
            }
        output[str(block)] = {}
        for a, b in pairs:
            pair = {}
            for metric in ["log_loss", "roc_auc"]:
                values = draws[a][metric] - draws[b][metric]
                point = metrics[a][metric] - metrics[b][metric]
                se = float(np.nanstd(values, ddof=1))
                interval = {
                    "estimate": point,
                    "bootstrap_se": se,
                    "valid_draws": int(np.isfinite(values).sum()),
                }
                if np.isfinite(values).all():
                    statistics.append(
                        np.abs((values - point) / se) if se > 1e-12 else np.zeros(len(values))
                    )
                    intervals.append(interval)
                pair[metric] = interval
            output[str(block)][a + "__minus__" + b] = pair
    critical = (
        float(np.quantile(np.max(statistics, axis=0), config["confidence"])) if statistics else None
    )
    for interval in intervals:
        interval["lower"] = interval["estimate"] - critical * interval["bootstrap_se"]
        interval["upper"] = interval["estimate"] + critical * interval["bootstrap_se"]
    return {
        "opportunities": int(active.sum()),
        "metrics": metrics,
        "blocks": output,
        "method": "calendar_blocks_within_fold_masked_to_observed_opportunities_joint_max_t",
        "critical": critical,
        "opportunities_overlap": True,
        "independent_confirmation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    root = args.study
    for relative, expected in read_json(root / "artifacts.json").items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Artefacto de investigación alterado: {relative}")
    summary = read_json(root / "summary.json")
    destination = root.parent.parent / "advanced_analyses" / summary["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    phase = root / "selection_development"
    pieces = []
    for fold_folder in sorted(phase.glob("fold_*")):
        number = int(fold_folder.name.split("_")[-1])
        state = read_json(fold_folder / "models" / "meta_xgb" / "fit.json")
        if state["audit"]["status"] != "fitted":
            continue
        logistic = read_json(fold_folder / "models" / "meta_logistic" / "fit.json")
        if logistic["audit"]["status"] != "fitted":
            continue
        model = XGBClassifier()
        model.load_model(fold_folder / "models" / "meta_xgb" / "model.ubj")
        encoded = json.loads(model.get_booster().save_config())["learner"]["learner_model_param"][
            "base_score"
        ]
        score = json.loads(encoded)
        prior = float(score[0] if isinstance(score, list) else score)
        decisions = {}
        for kind in ["logistic", "xgb"]:
            table = pd.read_csv(fold_folder / ("meta_" + kind + "_decisions.csv"))
            table["reference_time"] = pd.to_datetime(table.reference_time, utc=True)
            decisions[kind] = table
        calendar = decisions["xgb"][["reference_time"]].copy()
        offers = pd.read_csv(fold_folder / "oos_meta_opportunities.csv")
        offers["reference_time"] = pd.to_datetime(offers.reference_time, utc=True)
        offers = offers.loc[offers.status == "observed", ["reference_time", "target"]]
        calendar = calendar.merge(offers, on="reference_time", how="left", validate="one_to_one")
        for name, table in decisions.items():
            calendar[name] = table.filter_probability.to_numpy()
        calendar["training_intercept"] = prior
        calendar["constant_05"] = 0.5
        calendar["fold"] = number
        pieces.append(calendar)
    all_data = pd.concat(pieces, ignore_index=True).sort_values(["fold", "reference_time"])
    all_data.to_csv(destination / "oos_meta_ranking_calendar.csv", index=False)
    evidence = calendar_bootstrap(all_data, summary["protocol"]["bootstrap"])
    write_new_json(destination / "meta_ranking.json", evidence)
    audit = pd.read_csv(root / "nineteen_trade_audit.csv")
    audit["signal_reference"] = pd.to_datetime(audit.signal_reference, utc=True)
    rejection_summary = {}
    for name in [
        "meta_logistic",
        "meta_logistic_economic",
        "meta_xgb",
        "meta_xgb_economic",
        "trend_gate",
        "volatility_gate",
    ]:
        decisions = pd.read_csv(
            root / "audited_observed_development" / "fold_1" / (name + "_decisions.csv")
        )
        decisions["reference_time"] = pd.to_datetime(decisions.reference_time, utc=True)
        comparison = audit.merge(
            decisions[["reference_time", "filter_probability", "accepted", "rejection_reason"]],
            left_on="signal_reference",
            right_on="reference_time",
            how="left",
            validate="one_to_one",
            suffixes=("", "_decision"),
        )
        comparison.to_csv(destination / (name + "_nineteen_trade_rejections.csv"), index=False)
        rejection_summary[name] = {
            "accepted_reference_entries": int(comparison.accepted.sum()),
            "rejected_winners": int(
                ((~comparison.accepted) & (comparison.outcome == "winner")).sum()
            ),
            "rejected_losers": int(
                ((~comparison.accepted) & (comparison.outcome == "loser")).sum()
            ),
            "independent_evidence": False,
            "interpretation": "Rejecting an old entry changes future portfolio trajectory; executed new trades need not be a subset of these 19",
        }
    write_new_json(destination / "nineteen_rejection_summary.json", rejection_summary)
    acceptance = summary["protocol"]["acceptance"]
    decisions = {}
    economic = summary["periods"]["selection_development"]["economic"]
    for kind in ["logistic", "xgb"]:
        for variant in ["meta_" + kind, "meta_" + kind + "_economic"]:
            metrics = economic[variant]["metrics"]
            failed = []
            if metrics["net_return"] <= 0:
                failed.append("nonpositive_selection_development_return")
            if metrics["trade_count"] < acceptance["minimum_trades"]:
                failed.append("too_few_executed_trades")
            if any(
                block[kind + "__minus__training_intercept"]["log_loss"].get("upper", 1) >= 0
                for block in evidence["blocks"].values()
            ):
                failed.append("no_joint_incremental_log_loss_gain_vs_train_intercept")
            decisions[variant] = {
                "accepted": not failed,
                "failed_checks": failed,
                "audited_period_used_for_acceptance": False,
            }
    write_new_json(destination / "meta_acceptance_diagnostic.json", decisions)
    write_new_json(
        destination / "manifest.json",
        {
            "source_summary_sha256": sha256(root / "summary.json"),
            "source_artifacts_manifest_sha256": sha256(root / "artifacts.json"),
            "source_study_modified": False,
            "additional_fits": 0,
            "script_sha256": sha256(Path(__file__)),
            "baseline": "training-only XGBoost global intercept read from each frozen model; no fitted features",
            "baseline_documentation": "https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters",
        },
    )
    print(
        json.dumps(
            {
                "directory": str(destination.resolve()),
                "opportunities": evidence["opportunities"],
                "ranking_metrics": evidence["metrics"],
                "decisions": decisions,
                "nineteen_rejections": rejection_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
