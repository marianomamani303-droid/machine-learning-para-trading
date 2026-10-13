from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_ml.evaluation.uncertainty import moving_block_indices
from trading_ml.mean_reversion.simulator import trade_statistics
from trading_ml.momentum.simulator import Simulation
from trading_ml.momentum.study import _aggregate


def aggregate(results: list[tuple[int, Simulation]]) -> dict[str, Any]:
    """Reuse verified equity aggregation, adding statistics of executed MR trades."""
    data = _aggregate(results)
    trades = pd.concat([r.trades for _, r in results], ignore_index=True)
    if "portfolio_contribution" in trades:
        data["metrics"].update(trade_statistics(trades))
    if "opportunities" in results[0][1].metrics:
        count_keys = [
            "opportunities",
            "rejected_by_probability",
            "skipped_while_in_position",
            "nonexecutable_at_open",
            "skipped_on_exit_open",
            "terminal_opportunities",
        ]
        data["metrics"].update({key: sum(r.metrics[key] for _, r in results) for key in count_keys})
    return data


def ar1_diagnostic(close: pd.Series) -> dict[str, Any]:
    """Descriptive train-only AR(1) fit; no unit-root claim or signal selection."""
    values = np.log(close.to_numpy())
    x, y = values[:-1], values[1:]
    if len(x) < 30 or x.std() == 0:
        return {"status": "insufficient_or_constant", "rows": len(values)}
    intercept, phi = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    return {
        "status": "descriptive_only",
        "rows": len(values),
        "phi": float(phi),
        "intercept": float(intercept),
        "conditional_half_life_bars": float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else None,
        "interpretation": "AR1 point estimate on log price; not ADF, not proof of stationarity, not a trading filter",
    }


def paired_evidence(systems: dict[str, Any], protocol: dict[str, Any]) -> tuple[dict, dict]:
    names = [n for n in systems if n.startswith("mr_")]
    pairs = [(n, reference) for n in names for reference in ["cash", "buy_hold"]]
    pairs += [(n, n.rsplit("_", 1)[0] + "_rules") for n in names if not n.endswith("_rules")]
    boot = protocol["bootstrap"]
    family = len(pairs) * len(boot["block_lengths"])
    tail = (1 - boot["confidence"]) / (2 * family)
    groups = systems["cash"]["_curve"].fold.to_numpy()
    indices_by_block = {
        block: moving_block_indices(groups, boot["samples"], block, protocol["research"]["seed"])
        for block in boot["block_lengths"]
        if all((groups == g).sum() >= block for g in np.unique(groups))
    }
    comparisons = {}
    for candidate, reference in pairs:
        a, b = systems[candidate]["_curve"], systems[reference]["_curve"]
        if not a.timestamp.equals(b.timestamp):
            raise ValueError("Comparaciones Mean Reversion desalineadas")
        difference = np.log1p(a.net_return.to_numpy()) - np.log1p(b.net_return.to_numpy())
        blocks = {}
        for block in boot["block_lengths"]:
            if block not in indices_by_block:
                blocks[str(block)] = {"status": "insufficient_window"}
                continue
            draws = difference[indices_by_block[block]].mean(axis=1)
            lo, hi = np.quantile(draws, [tail, 1 - tail])
            blocks[str(block)] = {
                "status": "estimated",
                "mean_excess_log_return": float(difference.mean()),
                "lower": float(lo),
                "upper": float(hi),
                "simultaneous_family_size": family,
                "adjusted_confidence": 1 - 2 * tail,
                "samples": boot["samples"],
            }
        comparisons[f"{candidate}__minus__{reference}"] = blocks
    decisions = {}
    acceptance = protocol["acceptance"]
    for name in names:
        metrics = systems[name]["metrics"]
        failed = []
        if metrics["trade_count"] < acceptance["minimum_trades"]:
            failed.append("too_few_trades")
        if systems[name]["positive_fold_fraction"] < acceptance["required_positive_fold_fraction"]:
            failed.append("unstable_across_windows")
        if metrics["net_return"] <= 0 or (metrics.get("expectancy_net_per_trade") or 0) <= 0:
            failed.append("nonpositive_return_or_expectancy")
        for reference in ["cash", "buy_hold"]:
            intervals = comparisons[f"{name}__minus__{reference}"]
            if any(item.get("lower", -1) <= 0 for item in intervals.values()):
                failed.append(f"no_positive_adjusted_lower_bound_vs_{reference}")
        incremental = None
        if not name.endswith("_rules"):
            reference = name.rsplit("_", 1)[0] + "_rules"
            incremental = all(
                item.get("lower", -1) > 0
                for item in comparisons[f"{name}__minus__{reference}"].values()
            )
        decisions[name] = {
            "historical_evidence_passed": not failed,
            "failed_checks": failed,
            "incremental_model_evidence_vs_rules": incremental,
            "independent_advantage_confirmed": False,
            "promoted": False,
        }
    return {
        "method": "paired_non_circular_blocks_within_fold",
        "family_per_period": family,
        "limitations": "Conditional on fixed historical predictions; finite bootstrap tails, overlapping trades and past project selection remain limitations; regime tables descriptive",
        "comparisons": comparisons,
    }, decisions
