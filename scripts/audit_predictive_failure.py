"""Read-only scientific diagnostics of existing artifacts; zero model training."""

from __future__ import annotations

import argparse
import logging
import math
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats

from trading_ml.common import environment, object_hash, read_json, sha256, sidecar, write_new_json
from trading_ml.config import Settings
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.feature_research.study import _load_panel, temporal_score, temporal_train
from trading_ml.multimarket.data import load_snapshot
from trading_ml.multimarket.features import features as market_features
from trading_ml.structure_regime.features import extend_structure
from trading_ml.structure_regime.study import snapshot
from trading_ml.training.recipe import protect_reserved_labels

LOG = logging.getLogger(__name__)
RUNS = {
    "features": "feature_studies/20261015T162842_a5404a4b",
    "structure": "structure_regime_studies/20261016T152046_8f7d1886",
    "frontier": "frontier_studies/20261014T134646_e17d06e3",
    "labels": "label_studies/20261015T170214_699ffc3e",
    "advanced": "advanced_studies/20261013T224826_25c6cac4",
    "fx_daily": "multimarket/studies/20261015T222542_98a869d8",
    "fx_pilot": "multimarket/studies/20261015T222034_4db5f88c",
}


def acf(values, lags):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    variance = x @ x
    return np.array(
        [
            float(x[k:] @ x[:-k] / variance) if variance > 0 else 0
            for k in range(1, min(lags, len(x) - 1) + 1)
        ]
    )


def dependence(values, lags=30):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    rho = acf(x, lags)
    k = np.arange(1, len(rho) + 1)
    # Bartlett HAC variance of a sample mean, NOT an estimate of independent regimes.
    tau = float(1 + 2 * np.sum((1 - k / (len(rho) + 1)) * rho))
    q = float(len(x) * (len(x) + 2) * np.sum(rho**2 / (len(x) - k)))
    return {
        "rows": len(x),
        "acf_1": float(rho[0]),
        "acf_5": float(rho[4]),
        "acf_20": float(rho[19]),
        "ljung_box_q": q,
        "ljung_box_p_descriptive": float(stats.chi2.sf(q, len(rho))),
        "hac_bandwidth": len(rho),
        "hac_mean_variance_inflation": tau,
        "mean_ess_uncapped": float(len(x) / tau) if tau > 0 else None,
        "caveat": "Statistic-specific, covariance-stationarity approximation; not independent trading opportunities",
    }


def data_diagnostic(frame, metadata, path):
    frame = frame.sort_values("timestamp").copy()
    returns = np.log(frame.close).diff().dropna()
    gaps = frame.timestamp.diff().dt.total_seconds().div(3600).dropna()
    ohlc = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    incoherent = (
        (frame.high < frame[["open", "close"]].max(axis=1))
        | (frame.low > frame[["open", "close"]].min(axis=1))
        | (frame.high < frame.low)
    )
    quality = metadata.get("quality", {})
    info = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": len(frame),
        "provider": metadata.get("instrument", {}).get("provider", "binance"),
        "symbol": metadata.get("instrument", metadata.get("market", {})).get("symbol"),
        "interval": metadata.get("interval", metadata.get("market", {}).get("interval")),
        "first_open": frame.timestamp.min().isoformat(),
        "last_reference": frame.reference_time.max().isoformat(),
        "duplicated_timestamps": int(frame.timestamp.duplicated().sum()),
        "missing_ohlc_cells": int(frame[["open", "high", "low", "close"]].isna().sum().sum()),
        "nonfinite_ohlc": int((~np.isfinite(ohlc)).sum()),
        "nonpositive_ohlc": int((ohlc <= 0).sum()),
        "incoherent_ohlc": int(incoherent.sum()),
        "zero_return_fraction": float((returns == 0).mean()),
        "gap_hours_median": float(gaps.median()),
        "gap_hours_max": float(gaps.max()),
        "gap_hours_above_72": int((gaps > 72).sum()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_skew": float(returns.skew()),
        "return_excess_kurtosis": float(returns.kurt()),
        "return_quantiles": {
            str(q): float(returns.quantile(q)) for q in [0.001, 0.01, 0.5, 0.99, 0.999]
        },
        "largest_moves": [
            {"open_time": frame.timestamp.loc[i].isoformat(), "log_return": float(v)}
            for i, v in returns.loc[returns.abs().nlargest(5).index].items()
        ],
        "return_dependence": dependence(returns),
        "absolute_return_dependence": dependence(returns.abs()),
        "quality_counts": {
            k: v for k, v in quality.items() if isinstance(v, (int, float, str, bool))
        },
        "incomplete_buckets_excluded": len(quality.get("incomplete_buckets_excluded", [])),
        "parent_csv": metadata.get("parent_csv"),
    }
    if "open_bid" in frame:
        info["crossed_open"] = int((frame.open_ask < frame.open_bid).sum())
        info["crossed_close"] = int((frame.close_ask < frame.close_bid).sum())
        spread = (frame.close_ask - frame.close_bid) / frame.close_bid * 10000
        info["close_spread_bps_quantiles"] = {
            str(q): float(spread.quantile(q)) for q in [0, 0.5, 0.95, 1]
        }
    return info


def inventory_data(root):
    records, sources = [], {}
    for symbol, path in {
        "BTCUSDT": root / "data/raw.csv",
        "ETHUSDT": root / "research_data/ETHUSDT_20200101_20261001.csv",
    }.items():
        metadata = read_json(sidecar(path))
        if metadata["sha256"] != sha256(path):
            raise ValueError(f"Invalid source checksum: {path}")
        f = pd.read_csv(path, parse_dates=["timestamp"])
        f["reference_time"] = f.timestamp + pd.Timedelta(days=1)
        records.append(data_diagnostic(f, metadata, path))
        sources[symbol] = f
    for manifest in sorted((root / "multimarket/data").glob("*/*.csv.json")):
        path = manifest.with_suffix("")
        f, metadata = load_snapshot(path)
        records.append(data_diagnostic(f, metadata, path))
    return records, sources


def target_diagnostics(raw, protocol):
    output, forecasts, cross = [], [], {}
    bounds = pd.to_datetime(protocol["boundaries"], utc=True)
    for symbol, f in raw.items():
        price = np.log(f.close)
        r = price.diff()
        sigma = r.rolling(20).std()
        cross[symbol] = pd.DataFrame(
            {"reference_time": f.reference_time, "return": r, "abs_return": r.abs()}
        )
        for horizon in protocol["horizons"]:
            future = price.shift(-horizon) - price
            target_time = f.reference_time.shift(-horizon)
            valid = future.notna() & sigma.notna()
            z = future / (sigma * np.sqrt(horizon))
            open_return = f.open.shift(-(horizon + 1)) / f.open.shift(-1) - 1
            # Cost threshold from current research: 10 bps commission + 6 bps friction per side.
            factor = (1 - 0.001) * (1 - 0.0006) / ((1 + 0.001) * (1 + 0.0006))
            up = future > 0
            executable = open_return * factor + factor - 1
            usable = valid & open_return.notna()
            output.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "rows": int(valid.sum()),
                    "up_fraction": float(up[valid].mean()),
                    "near_zero_fraction": float(
                        (z[valid].abs() <= protocol["near_zero_sigma_fraction"]).mean()
                    ),
                    "large_positive_fraction": float((z[valid] > 1).mean()),
                    "large_negative_fraction": float((z[valid] < -1).mean()),
                    "positive_net_fraction": float((executable[usable] > 0).mean()),
                    "direction_net_disagreement": float(
                        (up[usable] != (executable[usable] > 0)).mean()
                    ),
                    "close_open_sign_disagreement": float(
                        (up[usable] != (open_return[usable] > 0)).mean()
                    ),
                    "up_but_net_loss_fraction_of_up": float((executable[usable & up] <= 0).mean()),
                    "nonoverlapping_sample_upper_bound": int(valid.sum() // horizon),
                    "overlapping_target_dependence": dependence(future[valid]),
                }
            )
            for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
                # Purge actual outcome horizon, including for unconditional baselines.
                train = valid & (target_time < start - pd.Timedelta(days=6))
                score = valid & (f.reference_time >= start) & (target_time < end)
                y = future[score].to_numpy()
                magnitude_prediction = (
                    sigma[score].to_numpy() * np.sqrt(horizon) * np.sqrt(2 / np.pi)
                )
                unconditional = float(future[train].abs().mean())
                mean_signed = float(future[train].mean())
                signed_prediction = r.rolling(60).mean()[score].to_numpy() * horizon
                forecasts.append(
                    {
                        "symbol": symbol,
                        "horizon": horizon,
                        "fold": fold,
                        "rows": len(y),
                        "train_last_target": target_time[train].max().isoformat(),
                        "start": start.isoformat(),
                        "magnitude_mse_vol20": float(
                            np.mean((np.abs(y) - magnitude_prediction) ** 2)
                        ),
                        "magnitude_mse_train_mean": float(
                            np.mean((np.abs(y) - unconditional) ** 2)
                        ),
                        "magnitude_mae_vol20": float(
                            np.mean(np.abs(np.abs(y) - magnitude_prediction))
                        ),
                        "magnitude_mae_train_mean": float(
                            np.mean(np.abs(np.abs(y) - unconditional))
                        ),
                        "signed_mse_zero": float(np.mean(y**2)),
                        "signed_mse_train_mean": float(np.mean((y - mean_signed) ** 2)),
                        "signed_mse_trailing_mean60": float(np.mean((y - signed_prediction) ** 2)),
                    }
                )
    merged = cross["BTCUSDT"].merge(
        cross["ETHUSDT"], on="reference_time", suffixes=("_btc", "_eth")
    )
    correlation = merged[["return_btc", "return_eth"]].corr().iloc[0, 1]
    return {
        "horizons": output,
        "magnitude_vs_signed_no_training": forecasts,
        "btc_eth_return_correlation": float(correlation),
        "btc_eth_abs_return_correlation": float(
            merged[["abs_return_btc", "abs_return_eth"]].corr().iloc[0, 1]
        ),
        "exactly_aligned_crypto_references": len(merged),
    }


def feature_diagnostics(root, protocol):
    study = root / RUNS["features"]
    selections = read_json(study / "selections.json")
    inner = []
    for s in selections:
        original = np.asarray(s["inner_log_losses"]["original"])
        for name, losses in s["inner_log_losses"].items():
            inner.append(
                {
                    "symbol": s["symbol"],
                    "fold": s["fold"],
                    "family": name,
                    "delta_loss": float(np.mean(losses - original)),
                    "both_inner_windows_improved": bool(np.all(np.asarray(losses) < original)),
                    "selected": s["chosen"] == name,
                }
            )
    importance = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        local = [s for s in selections if s["symbol"] == symbol]
        for name in local[0]["individual_importance"]:
            means = [float(np.mean(s["individual_importance"][name])) for s in local]
            importance.append(
                {
                    "symbol": symbol,
                    "feature": name,
                    "positive_all_inner_windows_and_blocks_folds": sum(
                        name in s["positive_inner_importance"] for s in local
                    ),
                    "mean_inner_permutation_delta_loss": float(np.mean(means)),
                    "selected_compact_folds": sum(name in s["compact_columns"] for s in local),
                }
            )
    panel, _ = _load_panel(
        Settings(root=root), pd.Timestamp(protocol["historical_end_exclusive"], tz="UTC")
    )
    config = tomllib.loads(Path("config/structure_regime.toml").read_text(encoding="utf-8"))[
        "structure"
    ]
    panel = {symbol: extend_structure(frame, config) for symbol, frame in panel.items()}
    diagnostics, causal = [], []
    from trading_ml.feature_research.study import fixed_recipes
    from trading_ml.structure_regime.features import CLUSTER_INPUT, RULE_COLUMNS, STRUCTURE

    columns = list(
        dict.fromkeys(
            [c for recipe in fixed_recipes().values() for c in recipe]
            + STRUCTURE
            + CLUSTER_INPUT
            + RULE_COLUMNS
        )
    )
    for symbol, frame in panel.items():
        ready = frame[columns].notna().all(axis=1) & frame.sr_ready
        for fold, (start, end) in enumerate(
            zip(
                pd.to_datetime(protocol["boundaries"][:-1], utc=True),
                pd.to_datetime(protocol["boundaries"][1:], utc=True),
                strict=True,
            )
        ):
            train = temporal_train(frame.loc[ready], start)
            score = temporal_score(frame.loc[ready], start, end)
            corr = train[columns].corr(method="spearman")
            constants = [name for name in columns if train[name].nunique() < 2]
            pairs = [
                (a, b, float(corr.loc[a, b]))
                for i, a in enumerate(columns)
                for b in columns[i + 1 :]
                if abs(corr.loc[a, b]) > 0.95
            ]
            outside = {
                c: float(
                    (
                        (score[c] < train[c].quantile(0.01)) | (score[c] > train[c].quantile(0.99))
                    ).mean()
                )
                for c in columns
            }
            diagnostics.append(
                {
                    "symbol": symbol,
                    "fold": fold,
                    "train_rows": len(train),
                    "oos_rows": len(score),
                    "constants_train": constants,
                    "correlations_above_095_train": pairs,
                    "outside_train_1_99_quantiles": outside,
                    "macd_identity_error": float(
                        (train.macd - train.macd_signal - train.macd_hist).abs().max()
                    ),
                    "body_return_spearman": float(corr.loc["body_ratio", "return_1"]),
                }
            )
        # Real-data prefix check; functions receive only their prefix, not future rows.
        for cut in [700, 1700]:
            prefix = extend_structure(frame.iloc[:cut], config)
            pd.testing.assert_frame_equal(
                prefix[STRUCTURE + RULE_COLUMNS], frame.iloc[:cut][STRUCTURE + RULE_COLUMNS]
            )
            causal.append({"symbol": symbol, "cut": cut, "structure_extension_prefix_equal": True})
    from trading_ml.feature_research.features import build_feature_panel

    raw_columns = [
        "timestamp",
        "reference_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_quote",
        "taker_base",
    ]
    for cut in [700, 1700]:
        rebuilt = build_feature_panel(
            {s: f[raw_columns].iloc[:cut].copy() for s, f in panel.items()}
        )
        for symbol, prefix in rebuilt.items():
            pd.testing.assert_frame_equal(
                extend_structure(prefix, config)[columns], panel[symbol].iloc[:cut][columns]
            )
            causal.append(
                {
                    "symbol": symbol,
                    "cut": cut,
                    "all_research_features_rebuilt_from_raw_prefix": True,
                }
            )
    return {
        "all_feature_columns": columns,
        "count": len(columns),
        "inner_ablations": inner,
        "inner_importance": importance,
        "train_only_diagnostics": diagnostics,
        "real_prefix_checks": causal,
    }, panel


def compact_metrics(y, p):
    m = classification_metrics(np.asarray(y).astype(int), np.asarray(p))
    return {
        k: m[k]
        for k in [
            "rows",
            "up_fraction",
            "accuracy",
            "roc_auc",
            "log_loss",
            "brier_score",
            "ece_10_bins",
        ]
    }


def prediction_errors(root, panel, protocol):
    source = root / RUNS["structure"]
    predictions = pd.read_csv(
        source / "predictions.csv", parse_dates=["reference_time", "target_time"]
    )
    output, confidence, fold_metrics = [], [], []
    for symbol, frame in panel.items():
        frame = frame.copy()
        future = frame.close.shift(-1) / frame.close - 1
        frame["audit_future_return"] = future
        sigma = np.log(frame.close).diff().rolling(20).std()
        z = np.log1p(future).abs() / sigma
        frame["audit_future_magnitude"] = np.select(
            [z <= 0.1, z <= 1], ["small_01sigma", "middle_01_1sigma"], default="large_1sigma"
        )
        joined = predictions.loc[predictions.symbol == symbol].merge(
            frame[
                [
                    "reference_time",
                    "audit_future_return",
                    "audit_future_magnitude",
                    "sr_vol",
                    "sr_direction",
                ]
            ],
            on="reference_time",
            validate="many_to_one",
        )
        for (seed, recipe), group in joined.groupby(["seed", "recipe"]):
            if recipe in ["constant_05", "train_prevalence"]:
                continue
            for dimension in ["sr_vol", "sr_direction", "audit_future_magnitude"]:
                for category, sample in group.groupby(dimension):
                    m = compact_metrics(sample.target, sample.probability)
                    weighted = np.average(
                        (sample.probability >= 0.5) == sample.target,
                        weights=sample.audit_future_return.abs(),
                    )
                    output.append(
                        {
                            "symbol": symbol,
                            "seed": int(seed),
                            "recipe": recipe,
                            "dimension": dimension,
                            "category": category,
                            **m,
                            "magnitude_weighted_accuracy": float(weighted),
                            "mean_realized_return_if_long": float(
                                sample.audit_future_return.mean()
                            ),
                        }
                    )
            conf = np.maximum(group.probability, 1 - group.probability)
            bins = pd.cut(conf, [0.499999, 0.55, 0.60, 0.65, 0.70, 0.80, 1], include_lowest=True)
            for interval, sample in group.groupby(bins, observed=True):
                confidence.append(
                    {
                        "symbol": symbol,
                        "seed": int(seed),
                        "recipe": recipe,
                        "confidence_bin": str(interval),
                        "rows": len(sample),
                        "predicted_confidence": float(
                            np.maximum(sample.probability, 1 - sample.probability).mean()
                        ),
                        "hit_rate": float(((sample.probability >= 0.5) == sample.target).mean()),
                        "mean_close_return_descriptive": float(sample.audit_future_return.mean()),
                        "log_loss": compact_metrics(sample.target, sample.probability)["log_loss"],
                    }
                )
            for fold, sample in group.groupby("fold"):
                corr = stats.spearmanr(sample.probability, sample.audit_future_return).statistic
                fold_metrics.append(
                    {
                        "symbol": symbol,
                        "seed": int(seed),
                        "recipe": recipe,
                        "fold": int(fold),
                        **compact_metrics(sample.target, sample.probability),
                        "spearman_probability_future_signed_return": float(corr),
                        "magnitude_weighted_accuracy": float(
                            np.average(
                                (sample.probability >= 0.5) == sample.target,
                                weights=sample.audit_future_return.abs(),
                            )
                        ),
                    }
                )
    return {
        "conditional_errors": output,
        "confidence_errors": confidence,
        "fold_errors": fold_metrics,
        "posthoc_rules_created": False,
    }


def model_inventory(root):
    base = root / RUNS["frontier"]
    models, gaps, calibration, stacking_contracts = [], [], [], []
    for path in sorted(base.glob("*__predictions.csv")):
        phase, symbol, objective, _ = path.name.split("__")
        frame = pd.read_csv(path)
        excluded = {"reference_time", "target_time", "target", "regime", "fold"}
        for name in frame.columns.difference(list(excluded)):
            p = frame[name]
            m = compact_metrics(frame.target, p)
            models.append(
                {
                    "phase": phase,
                    "symbol": symbol,
                    "objective": objective,
                    "recipe": name,
                    **m,
                    "probability_std": float(p.std()),
                    "entry_055_fraction": float((p >= 0.55).mean()),
                    "majority_accuracy_descriptive": float(
                        max(frame.target.mean(), 1 - frame.target.mean())
                    ),
                }
            )
    # Explicitly exclude frozen prospective components: inspect historical fits only.
    for phase in ["selection", "historical_description"]:
        for path in sorted((base / phase).rglob("fit.json")):
            state = read_json(path)
            if "train_metrics" not in state:
                if path.parent.name != "stacking":
                    raise ValueError(f"Unknown historical fit schema: {path}")
                stacking_contracts.append(
                    {"path": str(path.resolve()), **state, "in_sample_metrics_available": False}
                )
                continue
            train, score, raw = (
                state["train_metrics"],
                state["oos_metrics"],
                state["raw_oos_metrics"],
            )
            gaps.append(
                {
                    "phase": phase,
                    "symbol": state["asset"],
                    "objective": state["objective"],
                    "recipe": state["recipe"]["id"],
                    "fold": state["fold"],
                    "train_rows": state["train_rows"],
                    "calibration_rows": state["calibration_rows"],
                    "train_loss": train["log_loss"],
                    "oos_loss_raw": raw["log_loss"],
                    "oos_loss_calibrated": score["log_loss"],
                    "train_auc": train["roc_auc"],
                    "oos_auc": raw["roc_auc"],
                    "gap_raw": raw["log_loss"] - train["log_loss"],
                }
            )
            cal = state["calibration"]
            calibration.append(
                {
                    "phase": phase,
                    "symbol": state["asset"],
                    "objective": state["objective"],
                    "recipe": state["recipe"]["id"],
                    "fold": state["fold"],
                    **cal,
                    "delta_oos_loss_cal_minus_raw": score["log_loss"] - raw["log_loss"],
                    "train_last_target": state["train_last_target"],
                    "calibration_first_reference": state["calibration_first_reference"],
                    "calibration_last_target": state["calibration_last_target"],
                    "evaluation_first_reference": state["evaluation_first_reference"],
                }
            )
            assert pd.Timestamp(state["train_last_target"]) < pd.Timestamp(
                state["calibration_first_reference"]
            ) - pd.Timedelta(days=6)
            assert pd.Timestamp(state["calibration_last_target"]) < pd.Timestamp(
                state["evaluation_first_reference"]
            ) - pd.Timedelta(days=6)
    return {
        "classification": models,
        "generalization_gaps": gaps,
        "calibration": calibration,
        "historical_temporal_contracts_checked": len(gaps),
        "stacking_contracts": stacking_contracts,
    }


def forex_audit(root):
    results, durations, feature_checks, contracts = [], [], [], []
    bases = [
        root / RUNS["fx_daily"],
        root / RUNS["fx_pilot"],
        root / "multimarket/studies/20261015T221605_65cd699d",
    ]
    for base in bases:
        for path in sorted(base.glob("*/summary.json")):
            state = read_json(path)
            symbol, interval = state["symbol"], state["interval"]
            for name, record in state["experiments"].items():
                m = record["classification"]
                results.append(
                    {
                        "run": base.name,
                        "symbol": symbol,
                        "interval": interval,
                        "experiment": name,
                        **{
                            k: m.get(k)
                            for k in [
                                "rows",
                                "up_fraction",
                                "accuracy",
                                "roc_auc",
                                "log_loss",
                                "brier_score",
                            ]
                        },
                        "prevalence_loss": m.get("train_prevalence", {}).get("log_loss"),
                        "economic": {k: v for k, v in record.items() if k != "classification"},
                    }
                )
            frame, meta = load_snapshot(Path(state["data"]))
            f = market_features(frame, symbol)
            recipe = __import__("trading_ml.multimarket.features", fromlist=["recipes"]).recipes(
                symbol
            )["compact"]
            training = f.loc[f.reference_time < pd.Timestamp("2022-01-01", tz="UTC")]
            feature_checks.append(
                {
                    "run": base.name,
                    "symbol": symbol,
                    "interval": interval,
                    "constant_features_before_2022": [
                        c for c in recipe if training[c].nunique() <= 1
                    ],
                    "feature_rows": len(f),
                    "ready_rows": int(f[recipe].notna().all(axis=1).sum()),
                }
            )
            for name in [
                "direction_price_h1",
                "direction_compact_h3",
                "net_price_h1",
                "tb_price_h3",
            ]:
                pred_file = path.parent / f"{name}_predictions.csv"
                if not pred_file.exists():
                    continue
                pred = pd.read_csv(pred_file, parse_dates=["reference_time", "target_time"])
                hours = (pred.target_time - pred.reference_time).dt.total_seconds() / 3600
                durations.append(
                    {
                        "run": base.name,
                        "symbol": symbol,
                        "interval": interval,
                        "experiment": name,
                        "rows": len(pred),
                        "duration_hours_min": float(hours.min()),
                        "duration_hours_median": float(hours.median()),
                        "duration_hours_max": float(hours.max()),
                        "duration_hours_p95": float(hours.quantile(0.95)),
                    }
                )
            for contract in path.parent.glob("*_fold*.json"):
                state = read_json(contract)
                if "last_train_target" not in state or "first_calibration_reference" not in state:
                    continue
                train_end, cal_start = (
                    pd.Timestamp(state["last_train_target"]),
                    pd.Timestamp(state["first_calibration_reference"]),
                )
                cal_end, oos_start = (
                    pd.Timestamp(state["last_calibration_target"]),
                    pd.Timestamp(state["first_oos_reference"]),
                )
                contracts.append(
                    {
                        "path": str(contract.resolve()),
                        "train_before_calibration": bool(train_end < cal_start),
                        "calibration_before_oos": bool(cal_end < oos_start),
                        "train_cal_gap_days": (cal_start - train_end).total_seconds() / 86400,
                        "cal_oos_gap_days": (oos_start - cal_end).total_seconds() / 86400,
                    }
                )
    return {
        "comparisons": results,
        "outcome_durations": durations,
        "feature_checks": feature_checks,
        "temporal_contracts": contracts,
    }


def economic_decomposition(root):
    path = root / RUNS["structure"] / "trades.csv"
    frame = pd.read_csv(path)
    output = []
    for (symbol, seed, recipe), rows in frame.loc[frame.scenario == "base"].groupby(
        ["symbol", "seed", "recipe"]
    ):
        gross_factors = rows.exit_fill / rows.entry_fill * (1 + 0.0006) / (1 - 0.0006)
        net = float(np.prod(1 + rows.net_return) - 1)
        gross = float(np.prod(gross_factors) - 1)
        expected_log_drag = len(rows) * math.log(
            ((1 - 0.001) * (1 - 0.0006)) / ((1 + 0.001) * (1 + 0.0006))
        )
        realized_log_drag = math.log1p(net) - math.log1p(gross)
        assert abs(expected_log_drag - realized_log_drag) < 1e-8
        output.append(
            {
                "symbol": symbol,
                "seed": int(seed),
                "recipe": recipe,
                "trade_count": len(rows),
                "net_compound_return": net,
                "gross_same_trades_return": gross,
                "round_trip_cost_bps": float((1 - math.exp(expected_log_drag / len(rows))) * 10000),
                "log_wealth_cost_drag": realized_log_drag,
                "profitable_gross_trades": float((gross_factors > 1).mean()),
                "profitable_net_trades": float((rows.net_return > 0).mean()),
            }
        )
    return output


def run(root, protocol_path):
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["new_model_fit_budget"] != 0:
        raise ValueError("This audit does not fit new models")
    first = pd.Timestamp(protocol["boundaries"][0], tz="UTC")
    end = pd.Timestamp(protocol["historical_end_exclusive"], tz="UTC")
    protect_reserved_labels(
        root,
        pd.DataFrame(
            {"reference_time": [first], "target_time": [end - pd.Timedelta(nanoseconds=1)]}
        ),
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    directory = root / "predictive_failure_audits" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    LOG.warning("Registrando integridad previa de todos los artefactos")
    before = snapshot(root, excluded=directory)
    write_new_json(
        directory / "plan.json",
        {
            "protocol": protocol,
            "existing_runs": RUNS,
            "protected_files": before,
            "script_sha256": sha256(Path(__file__)),
            "environment": environment(),
            "claim_scope": "descriptive historical diagnosis; no selection, no fresh independent test",
        },
    )
    LOG.warning("Auditoría de datos y dependencia; cero entrenamientos")
    datasets, raw = inventory_data(root)
    write_new_json(directory / "data.json", datasets)
    write_new_json(directory / "targets.json", target_diagnostics(raw, protocol))
    LOG.warning("Reutilizando ablaciones e importancias de validaciones internas")
    features, panel = feature_diagnostics(root, protocol)
    write_new_json(directory / "features.json", features)
    write_new_json(directory / "errors.json", prediction_errors(root, panel, protocol))
    LOG.warning("Auditando modelos guardados, calibración, Forex y economía")
    write_new_json(directory / "models.json", model_inventory(root))
    write_new_json(directory / "forex.json", forex_audit(root))
    write_new_json(directory / "economics.json", economic_decomposition(root))
    LOG.warning("Verificando preservación final de %s archivos", len(before))
    after = snapshot(root, excluded=directory)
    if before != after:
        raise ValueError("Previous artifacts changed during audit")
    summary = {
        "run_id": run_id,
        "directory": str(directory.resolve()),
        "new_model_fits": 0,
        "previous_artifact_count": len(before),
        "previous_artifacts_preserved": True,
        "previous_hash": object_hash(before),
        "prospective_evaluated": False,
        "datasets_checked": len(datasets),
        "feature_columns_checked": features["count"],
        "plan_sha256": sha256(directory / "plan.json"),
    }
    write_new_json(directory / "summary.json", summary)
    write_new_json(
        directory / "artifacts.json", {p.name: sha256(p) for p in sorted(directory.glob("*.json"))}
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("config/predictive_failure_audit.toml")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    print(run(args.root, args.protocol))
