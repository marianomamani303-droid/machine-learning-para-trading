"""Append-only, controlled data-extension experiment using existing estimators.

Run audit, then pilot, then full; analysis is a separate command. No promotion.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from trading_ml.advanced_research.data import recover_microstructure
from trading_ml.common import (
    code_provenance,
    environment,
    frame_hash,
    read_json,
    sha256,
    sidecar,
    write_new_json,
)
from trading_ml.config import Settings
from trading_ml.config.settings import Market
from trading_ml.data.binance import parse_archive
from trading_ml.data.history import load_history
from trading_ml.data.history_sources import fetch
from trading_ml.dmn.data import build_panel, temporal_split
from trading_ml.dmn.portfolio import metrics as portfolio_metrics
from trading_ml.dmn.portfolio import simulate as portfolio_simulate
from trading_ml.dmn.portfolio import target_weights
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.feature_research.study import FitBudget
from trading_ml.frontier_research.features import CROSS, LOCAL, multi_asset_features
from trading_ml.frontier_research.models import create_estimator
from trading_ml.label_research.events import ExitRule, causal_scales, label_events
from trading_ml.label_research.execution import simulate_events
from trading_ml.label_research.models import Budget, fit_event_model, meta_pool
from trading_ml.models.baselines import fit_reference_models
from trading_ml.momentum.config import ResearchCosts
from trading_ml.multimarket.data import aggregate_forex, load_snapshot
from trading_ml.multimarket.execution import Costs, Engine
from trading_ml.multimarket.features import features as fx_features
from trading_ml.multimarket.features import recipes as fx_recipes
from trading_ml.multimarket.study import _fit, labelled
from trading_ml.smc.detectors import smc_columns
from trading_ml.structure_regime.features import RULE_COLUMNS, STRUCTURE, extend_structure
from trading_ml.structure_regime.study import fit_experts
from trading_ml.training.recipe import protect_reserved_labels

ROOT = Path("artifacts")
OUT = ROOT / "retraining_comparison" / "20261017_controlled"
STOP = pd.Timestamp("2025-07-01", tz="UTC")
OLD_START = pd.Timestamp("2020-01-01", tz="UTC")
NEW_START = pd.Timestamp("2017-08-18", tz="UTC")
BOUNDS = pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01", "2025-07-01"], utc=True)
ORIGINAL = read_json(ROOT / "models/v20261012T130540_d2bb6ac0/metadata.json")["features"]
PARAMETERS = read_json(ROOT / "models/v20261012T130540_d2bb6ac0/metadata.json")[
    "selected_parameters"
]
COSTS = ResearchCosts(commission_bps=10, slippage_bps=5, spread_bps=2)
SR = tomllib.loads(Path("config/structure_regime.toml").read_text(encoding="utf-8"))
LOG = logging.getLogger(__name__)


def save_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def read_frame(path):
    f = pd.read_csv(path, float_precision="round_trip")
    for c in ["timestamp", "reference_time", "target_time"]:
        if c in f:
            f[c] = pd.to_datetime(f[c], utc=True)
    return f


def metrics(y, p):
    m = classification_metrics(np.asarray(y), np.asarray(p))
    m["balanced_accuracy"] = float(balanced_accuracy_score(y, np.asarray(p) >= 0.5))
    m["confusion_matrix_tn_fp_fn_tp"] = (
        confusion_matrix(y, np.asarray(p) >= 0.5, labels=[0, 1]).ravel().tolist()
    )
    return m


def audit():
    OUT.mkdir(parents=True, exist_ok=False)
    previous = {}
    # Preserve results, models, reservations and original snapshots. Exclude only caches.
    paths = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and "cache" not in p.parts
        and not p.is_relative_to(OUT)
        and p.suffix not in {".png", ".svg", ".zip"}
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p, digest in zip(paths, pool.map(sha256, paths), strict=True):
            previous[str(p.resolve())] = digest
    write_new_json(OUT / "protected_before.json", previous)
    index = read_json(ROOT / "historical/validated_20261017.json")
    inventory = []
    for d in index["datasets"]:
        path = Path(d["path"])
        if sha256(path) != d["sha256"] or sha256(sidecar(path)) != d["manifest_sha256"]:
            raise ValueError(f"Historical integrity: {path}")
        inventory.append(d)
    write_new_json(OUT / "inventory.json", inventory)
    old, expanded, sources = {}, {}, {}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        path = ROOT / (
            "data/raw.csv" if symbol == "BTCUSDT" else "research_data/ETHUSDT_20200101_20261001.csv"
        )
        settings = replace(
            Settings(), root=ROOT, market=Market(symbol, "1d", "2020-01-01", "2026-10-01")
        )
        # Recover real quote/taker volumes from the original verified daily archives.
        frame, provenance = recover_microstructure(settings, path)
        frame = frame.loc[frame.reference_time <= STOP].reset_index(drop=True)
        protect_reserved_labels(
            ROOT, frame.assign(target_time=frame.reference_time + pd.Timedelta(days=1))
        )
        old[symbol] = frame
        d = next(
            d
            for d in inventory
            if d["symbol"] == symbol and d["interval"] == "1d" and d["market_type"] == "crypto_spot"
        )
        hist, _ = load_history(Path(d["path"]), root=ROOT, end=str(STOP.date()))
        # Native daily candles repair incomplete intraday aggregations; never fill prices.
        entries, parts = [], []
        for month in pd.period_range("2017-08", "2019-12", freq="M"):
            url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{month}.zip"
            payload, receipt = fetch(ROOT, url, upstream=True)
            if payload is None:
                raise ValueError(f"Missing native daily repair: {url}")
            parts.append(parse_archive(payload, include_microstructure=True))
            entries.append(receipt)
        extra = pd.concat(parts, ignore_index=True)
        extra = extra.loc[(extra.timestamp >= NEW_START) & (extra.timestamp < OLD_START)].copy()
        extra["reference_time"] = extra.timestamp + pd.Timedelta(days=1)
        joined = hist.loc[hist.timestamp < OLD_START].merge(
            extra, on="timestamp", suffixes=("_hist", "_native"), validate="one_to_one"
        )
        mismatch = {}
        for c in ["open", "high", "low", "close", "volume"]:
            mismatch[c] = int(
                (
                    ~np.isclose(joined[c + "_hist"], joined[c + "_native"], rtol=1e-9, atol=1e-8)
                ).sum()
            )
        combined = pd.concat([extra, frame], ignore_index=True)
        assert combined.timestamp.diff().iloc[1:].eq(pd.Timedelta(days=1)).all()
        assert not combined.timestamp.duplicated().any()
        assert (
            np.isfinite(
                combined[
                    [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "quote_volume",
                        "trades",
                        "taker_base",
                        "taker_quote",
                    ]
                ]
            )
            .all()
            .all()
        )
        expanded[symbol] = combined
        sources[symbol] = {
            "old": provenance,
            "new_native_daily": entries,
            "incorporated_daily": d,
            "common_native_aggregation_mismatch": mismatch,
            "extra_rows": len(extra),
            "extra_rows_missing_in_incorporated": len(extra) - len(joined),
            "old_rows": len(frame),
            "expanded_rows": len(combined),
        }
        for arm, raw in [("A", frame), ("B", combined)]:
            save_frame(OUT / "data" / f"{symbol}_{arm}_raw.csv", raw)
    # Build features independently, then freeze ALL original rows for both arms.
    # Only the additional pre-2020 fitting observations can change A -> B.
    for arm, panel in [("A", old), ("B", expanded)]:
        built = multi_asset_features(panel)
        for symbol, f in built.items():
            f = extend_structure(f, SR["structure"])
            f["target"] = (
                (f.close.shift(-1) > f.close).astype(float).where(f.close.shift(-1).notna())
            )
            f["target_time"] = f.reference_time.shift(-1)
            f = causal_scales(f)
            if arm == "B":
                f = pd.concat(
                    [
                        f.loc[f.timestamp < OLD_START],
                        read_frame(OUT / "data" / f"{symbol}_A_features.csv"),
                    ],
                    ignore_index=True,
                )
            save_frame(OUT / "data" / f"{symbol}_{arm}_features.csv", f)
    models = []
    for p in (ROOT / "models").glob("*/metadata.json"):
        m = read_json(p)
        models.append({"path": str(p), "sha256": sha256(p), "metadata": m})
    # Recover summaries verbatim, never use previous OOS for tuning.
    history = {}
    for folder in [
        "frontier_studies",
        "structure_regime_studies",
        "label_studies",
        "dmn_studies",
        "expansion/studies",
        "expansion/forex_studies",
    ]:
        for p in (ROOT / folder).glob("*/summary.json"):
            history[str(p)] = read_json(p)
    write_new_json(
        OUT / "historical_results.json", {"registered_models": models, "summaries": history}
    )
    write_new_json(OUT / "sources.json", sources)
    write_new_json(
        OUT / "plan.json",
        {
            "created_before_fitting": True,
            "hypothesis": "Effect of adding earlier observations with original common features fixed",
            "A_start": str(OLD_START),
            "B_start": str(NEW_START),
            "development_end": str(STOP),
            "boundaries": [str(x) for x in BOUNDS],
            "seeds": [42, 1729],
            "embargo_days": 6,
            "tabular": [
                "original",
                "logistic",
                "random_forest_cross",
                "lightgbm_cross",
                "catboost_cross",
                "structure",
                "regime_rules",
                "specialists_rules",
                "smc",
            ],
            "event": ["tb_vol_h5_cal", "meta_net_h1_logistic_cal_55"],
            "neural": ["lstm_sharpe_net"],
            "forex": ["EURUSD"],
            "forex_intervals": ["1d", "4h"],
            "pilot": "BTC original XGBoost, first fold, seed42, both arms; integrity/runtime gate only",
            "maximum_fits": 500,
            "specialists_seeds": [42],
            "bootstrap_samples": 2000,
            "blocks_days": [7, 30],
            "primary_endpoint": "paired mean log-loss improvement B vs A",
            "multiplicity": "Holm across all model-market-timeframe endpoints; max p across blocks",
            "minimum_material_log_loss_gain": 0.003,
            "minimum_material_auc_gain": 0.005,
            "independent_prospective": False,
            "historical_already_explored": True,
            "original_common_features_and_labels_identical": True,
            "native_daily_repair_deviation": "Needed to repair incomplete M15 daily windows and recover actual flow; original common history retained verbatim",
            "environment": environment(),
            "code": code_provenance(),
            "script_sha256": sha256(Path(__file__)),
            "all_old_artifacts_sha256": sha256(OUT / "protected_before.json"),
        },
    )
    print(
        "AUDIT COMPLETE",
        json.dumps(
            {
                s: {k: v for k, v in d.items() if k.endswith("rows") or k.startswith("common_")}
                for s, d in sources.items()
            }
        ),
        flush=True,
    )


def record(directory, rows, probabilities, *, full=None, strategy="fixed", rule=None, gate=None):
    pred = rows[["timestamp", "reference_time", "target_time", "target"]].copy()
    pred["probability"] = probabilities
    save_frame(directory / "predictions.csv", pred)
    write_new_json(directory / "predictive.json", metrics(rows.target.astype(int), probabilities))
    if full is not None:
        # Event engine consumes forward prices only, not labels; same entry cutoff in both arms.
        p = np.zeros(len(full))
        p[rows.index] = probabilities
        for scenario, multiplier in [("base", 1), ("double", 2)]:
            costs = replace(
                COSTS,
                commission_bps=10 * multiplier,
                slippage_bps=5 * multiplier,
                spread_bps=2 * multiplier,
            )
            start, end = pd.to_datetime(
                read_json(directory / "fit.json")["evaluation_window"], utc=True
            )
            result = simulate_events(
                full,
                p,
                rule or ExitRule("net_h1", 1),
                start,
                end,
                costs,
                threshold=0.55,
                gate=gate,
                maximum_horizon=10,
            )
            summary = result.metrics
            returns = result.curve.net_return.iloc[1:].to_numpy()
            summary["sharpe"] = (
                float(returns.mean() / returns.std(ddof=1) * np.sqrt(365))
                if returns.std(ddof=1) > 1e-12
                else None
            )
            save_frame(directory / f"{scenario}_curve.csv", result.curve)
            save_frame(directory / f"{scenario}_trades.csv", result.trades)
            write_new_json(directory / f"{scenario}_economic.json", summary)


def tabular(pilot=False):
    recipes = ["original"] if pilot else read_json(OUT / "plan.json")["tabular"]
    for symbol in ["BTCUSDT"] if pilot else ["BTCUSDT", "ETHUSDT"]:
        for arm in ["A", "B"]:
            full = read_frame(OUT / "data" / f"{symbol}_{arm}_features.csv")
            for fold, (start, end) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:], strict=True)):
                if pilot and fold != 0:
                    continue
                for seed in [42] if pilot else [42, 1729]:
                    common_cols = list(
                        dict.fromkeys(ORIGINAL + LOCAL + CROSS + STRUCTURE + RULE_COLUMNS)
                    )
                    # All tabular recipes share exactly the same observations.
                    eligible = (
                        full[common_cols].notna().all(axis=1) & full.sr_ready & full.research_ready
                    )
                    train = full.loc[
                        eligible & (full.target_time < start - pd.Timedelta(days=6))
                    ].copy()
                    out = full.loc[
                        eligible
                        & (full.reference_time >= start)
                        & (full.reference_time < end - pd.Timedelta(days=12))
                    ].copy()
                    assert out.target.notna().all() and out.target_time.max() < end
                    protect_reserved_labels(ROOT, train, out)
                    assert len(train) > 180 and train.target_time.max() < out.reference_time.min()
                    base_directory = OUT / "fits" / f"{symbol}_1d_f{fold}_s{seed}_{arm}"
                    base_directory.mkdir(parents=True, exist_ok=True)
                    for name in recipes:
                        if name in {"logistic", "specialists_rules"} and seed != 42:
                            continue
                        directory = base_directory / name
                        if (directory / "complete.json").exists():
                            continue
                        directory.mkdir(exist_ok=True)
                        columns = {
                            "original": ORIGINAL,
                            "logistic": ORIGINAL,
                            "smc": ORIGINAL + smc_columns(),
                            "structure": ORIGINAL + STRUCTURE,
                            "regime_rules": ORIGINAL + RULE_COLUMNS,
                            "specialists_rules": ORIGINAL,
                        }.get(name, LOCAL + CROSS)
                        t0 = time.perf_counter()
                        expert_fits = 0
                        if name == "logistic":
                            reference = fit_reference_models(train, columns, seed=seed)
                            model = reference
                            probability = reference.probabilities(out)["logistic_l2"]
                        elif name.endswith("_cross"):
                            kind = name.removesuffix("_cross")
                            model = create_estimator({"kind": kind})
                            # Secondary seed is the only deviation from the fixed frontier factory.
                            model.set_params(
                                **(
                                    {"random_seed": seed}
                                    if kind == "catboost"
                                    else {"random_state": seed}
                                )
                            )
                            model.fit(train[columns], train.target.astype(int))
                            probability = model.predict_proba(out[columns])[:, 1]
                        elif name == "specialists_rules":
                            from xgboost import XGBClassifier

                            parent = joblib.load(base_directory / "original/model.joblib")
                            budget = FitBudget(40, PARAMETERS)
                            probability, expert_audit = fit_experts(
                                train,
                                out,
                                "sr_state",
                                columns,
                                parent.predict_proba(out[columns].to_numpy(float))[:, 1],
                                budget,
                                seed,
                                SR["research"],
                                directory,
                            )
                            expert_fits = budget.actual
                            model = {"parent": parent, "experts": expert_audit}
                            write_new_json(directory / "experts.json", expert_audit)
                        else:
                            from xgboost import XGBClassifier

                            model = XGBClassifier(
                                **PARAMETERS,
                                random_state=seed,
                                n_jobs=1,
                                tree_method="hist",
                                device="cpu",
                                eval_metric="logloss",
                            )
                            model.fit(train[columns].to_numpy(float), train.target.astype(int))
                            probability = model.predict_proba(out[columns].to_numpy(float))[:, 1]
                        joblib.dump(model, directory / "model.joblib")
                        fit = {
                            "symbol": symbol,
                            "market": "spot",
                            "interval": "1d",
                            "fold": fold,
                            "seed": seed,
                            "arm": arm,
                            "recipe": name,
                            "target_kind": "close_direction",
                            "columns": columns,
                            "parameters": PARAMETERS
                            if name
                            not in [
                                "logistic",
                                "random_forest_cross",
                                "lightgbm_cross",
                                "catboost_cross",
                            ]
                            else getattr(model, "get_params", lambda: {})(),
                            "train_rows": len(train),
                            "train_first": str(train.reference_time.min()),
                            "train_last_target": str(train.target_time.max()),
                            "train_hash": frame_hash(train[columns + ["target", "target_time"]]),
                            "evaluation_window": [str(start), str(end)],
                            "test_rows": len(out),
                            "test_hash": frame_hash(
                                out[["reference_time", "target_time", "target"]]
                            ),
                            "fit_seconds": time.perf_counter() - t0,
                            "actual_fits": expert_fits if name == "specialists_rules" else 1,
                            "prior": float(train.target.mean()),
                            "probability_calibration": "raw",
                            "train_metrics": metrics(
                                train.target,
                                reference.probabilities(train)["logistic_l2"]
                                if name == "logistic"
                                else model.predict_proba(
                                    train[columns]
                                    if name.endswith("_cross")
                                    else train[columns].to_numpy(float)
                                )[:, 1],
                            )
                            if name != "specialists_rules"
                            else None,
                        }
                        write_new_json(directory / "fit.json", fit)
                        record(directory, out, probability, full=full)
                        write_new_json(directory / "complete.json", {"completed": True})
                        print(
                            "FIT",
                            directory,
                            "rows",
                            len(train),
                            "test",
                            len(out),
                            "seconds",
                            round(fit["fit_seconds"], 2),
                            flush=True,
                        )
    if pilot:
        a = read_frame(OUT / "fits/BTCUSDT_1d_f0_s42_A/original/predictions.csv")
        b = read_frame(OUT / "fits/BTCUSDT_1d_f0_s42_B/original/predictions.csv")
        assert a.drop(columns="probability").equals(b.drop(columns="probability"))
        write_new_json(
            OUT / "pilot.json",
            {
                "passed": True,
                "test_rows_identical": len(a),
                "prediction_finite": True,
                "full_gate": "integrity and comparable observations passed; outcomes not used to select expansion",
            },
        )


def neural():
    from trading_ml.dmn.models import apply_scaler, fit_network, predict_network

    protocol = tomllib.loads(Path("config/dmn.toml").read_text(encoding="utf-8"))
    recipe = next(r for r in protocol["recipes"] if r["id"] == "lstm_sharpe_net")
    panels = {}
    for arm in ["A", "B"]:
        raw = {s: read_frame(OUT / "data" / f"{s}_{arm}_raw.csv") for s in ["BTCUSDT", "ETHUSDT"]}
        panels[arm] = build_panel(raw, 63, 0.0001)
    a, b = panels["A"], panels["B"]
    locations = b.times.get_indexer(a.times)
    assert (locations >= 0).all()
    # Freeze original sequences and risk scaling exactly, as for tabular features.
    b.x = np.array(b.x, copy=True)
    b.x[locations] = a.x
    b.returns[locations] = a.returns
    b.volatility[locations] = a.volatility
    for arm, panel in panels.items():
        for fold, (start, end) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:], strict=True)):
            train, val, test = temporal_split(panel, start, end, protocol["training"])
            for seed in [42, 1729]:
                directory = OUT / "neural" / f"panel_1d_f{fold}_s{seed}_{arm}"
                if (directory / "complete.json").exists():
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                model, scaler, audit = fit_network(
                    panel, train, val, recipe, protocol, seed, directory / "network"
                )
                signals = predict_network(
                    model, apply_scaler(panel.x[test], scaler), recipe, protocol["risk"]
                )
                for scenario, multiplier in [("base", 1), ("double", 2)]:
                    result = portfolio_simulate(
                        target_weights(signals, panel.volatility[test], protocol["risk"]),
                        panel.returns[test],
                        panel.times[test],
                        protocol["costs"],
                        multiplier=multiplier,
                    )
                    save_frame(directory / f"{scenario}_curve.csv", result)
                    write_new_json(
                        directory / f"{scenario}_economic.json", portfolio_metrics(result)
                    )
                # Sharpe-trained weights are not calibrated directional probabilities.
                save_frame(
                    directory / "signals.csv",
                    pd.DataFrame(
                        {
                            "reference_time": np.repeat(panel.times[test], 2),
                            "symbol": np.tile(panel.symbols, len(test)),
                            "signal": signals.ravel(),
                            "asset_return": panel.returns[test].ravel(),
                            "target": (panel.returns[test] > 0).astype(int).ravel(),
                        }
                    ),
                )
                write_new_json(
                    directory / "fit.json",
                    {
                        "recipe": "lstm_sharpe_net",
                        "arm": arm,
                        "fold": fold,
                        "seed": seed,
                        "actual_fits": 1,
                        "train_dates": len(train),
                        "validation_dates": len(val),
                        "test_dates": len(test),
                        "audit": audit,
                        "probabilistic_direction_metrics": "not applicable: optimized allocation, not direction probability",
                    },
                )
                write_new_json(directory / "complete.json", {"completed": True})
                print("NEURAL", directory, "epoch", audit["selected_epoch"], flush=True)


def event_models():
    from trading_ml.label_research.events import attach_labels

    protocol = tomllib.loads(Path("config/label_study.toml").read_text(encoding="utf-8"))
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for arm in ["A", "B"]:
            frame = read_frame(OUT / "data" / f"{symbol}_{arm}_features.csv")
            frame["research_ready"] = (
                frame[ORIGINAL + ["label_sigma", "label_atr_fraction"]].notna().all(axis=1)
            )
            events = {
                n: label_events(frame, r, COSTS)
                for n, r in [
                    ("net_h1", ExitRule("net_h1", 1)),
                    ("tb_vol_h5", ExitRule("tb_vol_h5", 5, "vol")),
                ]
            }
            directory = OUT / "events" / f"{symbol}_{arm}"
            directory.mkdir(parents=True, exist_ok=True)
            oof_parts = []
            # Original quarterly OOF protocol, saved and reused for meta-labeling.
            quarters = pd.date_range("2021-01-01", "2025-07-01", freq="QS", tz="UTC")
            budget = Budget(200)
            for start, end in zip(quarters[:-1], quarters[1:], strict=True):
                destination = directory / "oof" / start.strftime("%Y%m%d")
                if (destination / "oof.csv").exists():
                    oof_parts.append(read_frame(destination / "oof.csv"))
                    continue
                model, audit = fit_event_model(
                    frame.assign(unit_net_return=events["net_h1"].unit_net_return),
                    start,
                    ORIGINAL,
                    protocol,
                    PARAMETERS,
                    budget,
                    ROOT,
                    reserve=False,
                )
                model.save(destination)
                out = frame.loc[(frame.reference_time >= start) & (frame.reference_time < end)]
                pred = pd.DataFrame(
                    {
                        "source_index": out.index,
                        "reference_time": out.reference_time,
                        "probability": model.probabilities(out),
                        "model_file": str(destination),
                        "model_cutoff": str(start),
                        "last_train_target": audit["train_target_end"],
                    }
                )
                save_frame(destination / "oof.csv", pred)
                oof_parts.append(pred)
            oof = pd.concat(oof_parts, ignore_index=True)
            for fold, (start, end) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:], strict=True)):
                # Meta protocol has one fixed seed, not a seed search.
                base = read_json(directory / "oof" / start.strftime("%Y%m%d") / "fit.json")
                from xgboost import XGBClassifier

                parent = XGBClassifier()
                parent.load_model(directory / "oof" / start.strftime("%Y%m%d") / "model.ubj")
                base_p = parent.predict_proba(frame[ORIGINAL].fillna(0))[:, 1]
                for name in ["tb_vol_h5_cal", "meta_net_h1_logistic_cal_55"]:
                    destination = directory / f"f{fold}" / name
                    if (destination / "complete.json").exists():
                        continue
                    if name.startswith("tb"):
                        data = attach_labels(frame, events["tb_vol_h5"])
                        columns = ORIGINAL
                        rule = ExitRule("tb_vol_h5", 5, "vol")
                        model, audit = fit_event_model(
                            data, start, columns, protocol, PARAMETERS, budget, ROOT
                        )
                    else:
                        data = meta_pool(frame, oof, events["net_h1"], start)
                        columns = ORIGINAL + ["base_probability"]
                        rule = ExitRule("net_h1", 1)
                        model, audit = fit_event_model(
                            data,
                            start,
                            columns,
                            protocol,
                            protocol["meta_xgboost"],
                            budget,
                            ROOT,
                            kind="logistic",
                            meta=True,
                        )
                    destination.mkdir(parents=True, exist_ok=True)
                    if model is None or model.calibration is None:
                        write_new_json(destination / "unavailable.json", audit)
                        print("EVENT ABSTAIN", destination, flush=True)
                        continue
                    model.save(destination / "model")
                    scoring = frame.assign(base_probability=base_p)
                    target_events = events["tb_vol_h5" if name.startswith("tb") else "net_h1"]
                    scoring["target"] = target_events.target
                    scoring["target_time"] = target_events.target_time
                    out = scoring.loc[
                        (scoring.reference_time >= start)
                        & (scoring.reference_time < end - pd.Timedelta(days=12))
                    ].copy()
                    assert out.target.notna().all() and out.target_time.max() < end
                    protect_reserved_labels(ROOT, out)
                    probability = model.probabilities(out)
                    gate = base_p >= 0.55 if name.startswith("meta") else None
                    write_new_json(
                        destination / "fit.json",
                        {
                            "symbol": symbol,
                            "market": "spot",
                            "interval": "1d",
                            "fold": fold,
                            "seed": 42,
                            "arm": arm,
                            "recipe": name,
                            "target_kind": "net_event_profit",
                            "evaluation_window": [str(start), str(end)],
                            "actual_fits": 2,
                            "audit": audit,
                            "test_rows": len(out),
                            "test_hash": frame_hash(
                                out[["reference_time", "target_time", "target"]]
                            ),
                            "base_fit": base["audit"],
                        },
                    )
                    record(destination, out, probability, full=scoring, rule=rule, gate=gate)
                    write_new_json(destination / "complete.json", {"completed": True})
                    print("EVENT", destination, flush=True)


def forex():
    from trading_ml.models.calibration import apply_calibration, fit_sigmoid
    from trading_ml.multimarket.data import _snapshot

    p = tomllib.loads(Path("config/multimarket_verified_history.toml").read_text(encoding="utf-8"))
    inventory = read_json(OUT / "inventory.json")
    for symbol in ["EURUSD"]:
        original_path = next(
            path
            for path in (ROOT / "multimarket/data" / symbol).glob("1h_*.csv")
            if read_json(sidecar(path))["instrument"]["provider"] == "fxcm"
            and read_json(sidecar(path))["end_exclusive"].startswith("2025-07-01")
        )
        old, meta = load_snapshot(original_path)
        d = next(
            d
            for d in inventory
            if d["symbol"] == symbol and d["interval"] == "1h" and d["provider"] == "fxcm"
        )
        hist, hmeta = load_history(Path(d["path"]), root=ROOT, end="2025-07-01")
        hist = hist.rename(columns={"available_at": "reference_time"})
        hist["crossed_open"] = hist.open_ask < hist.open_bid
        hist["crossed_close"] = hist.close_ask < hist.close_bid
        expanded = pd.concat([hist.loc[hist.timestamp < OLD_START], old], ignore_index=True)
        for interval in ["1d", "4h"]:
            built = {}
            for arm, raw in [("A", old), ("B", expanded)]:
                adapter = OUT / "forex_sources" / f"{symbol}_{arm}"
                source = _snapshot(
                    adapter, symbol, raw, {**meta, "interval": "1h", "quality": meta["quality"]}
                )
                aggregated = aggregate_forex(adapter, source, interval)
                candles, _ = load_snapshot(aggregated)
                frame = fx_features(candles, symbol)
                if arm == "B":
                    frame = pd.concat(
                        [
                            frame.loc[frame.reference_time < built["A"].reference_time.min()],
                            built["A"],
                        ],
                        ignore_index=True,
                    )
                    frame["source_index"] = np.arange(len(frame))
                built[arm] = frame
                engine = Engine(
                    frame,
                    symbol,
                    Costs(**p["costs"]),
                    execution_frame=raw,
                    bars_per_horizon=24 if interval == "1d" else 4,
                )
                data = labelled(frame, engine, 1, "direction", False)
                for fold, (start, end) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:], strict=True)):
                    boundary = start - pd.Timedelta(days=187)
                    common = fx_recipes(symbol)["compact"]
                    train = data.loc[data.target_time < boundary - pd.Timedelta(days=7)].dropna(
                        subset=common + ["target"]
                    )
                    val = data.loc[
                        (data.reference_time >= boundary)
                        & (data.target_time < start - pd.Timedelta(days=7))
                    ].dropna(subset=common + ["target"])
                    out = data.loc[
                        (data.reference_time >= start)
                        & (data.reference_time < end - pd.Timedelta(days=20))
                    ].dropna(subset=common)
                    assert out.target.notna().all() and out.target_time.max() < end
                    protect_reserved_labels(ROOT, train, val, out)
                    for recipe, cols in fx_recipes(symbol).items():
                        destination = (
                            OUT / "forex" / f"{symbol}_{interval}_f{fold}_s42_{arm}" / recipe
                        )
                        if (destination / "complete.json").exists():
                            continue
                        destination.mkdir(parents=True, exist_ok=True)
                        model = _fit(train, cols, p)
                        calibration = fit_sigmoid(
                            model.predict_proba(val[cols])[:, 1], val.target.to_numpy(), 42
                        )
                        probability = apply_calibration(
                            model.predict_proba(out[cols])[:, 1], calibration
                        )
                        joblib.dump(model, destination / "model.joblib")
                        fit = {
                            "symbol": symbol,
                            "market": "forex",
                            "interval": interval,
                            "fold": fold,
                            "seed": 42,
                            "arm": arm,
                            "recipe": "fx_" + recipe,
                            "target_kind": "close_direction",
                            "train_rows": len(train),
                            "calibration_rows": len(val),
                            "train_last_target": str(train.target_time.max()),
                            "calibration_last_target": str(val.target_time.max()),
                            "test_rows": len(out),
                            "test_hash": frame_hash(
                                out[["reference_time", "target_time", "target"]]
                            ),
                            "evaluation_window": [str(start), str(end)],
                            "actual_fits": 2,
                            "columns": cols,
                            "parameters": p["model"],
                            "calibration": calibration,
                            "prior": float(train.target.mean()),
                            "train_hash": frame_hash(train[cols + ["target", "target_time"]]),
                            "source_historical": d["path"],
                            "source_original": str(original_path),
                        }
                        write_new_json(destination / "fit.json", fit)
                        record(destination, out, probability)
                        for scenario, multiplier in [("base", 1), ("double", 2)]:
                            cost = replace(
                                Costs(**p["costs"]),
                                forex_commission_bps=0.35 * multiplier,
                                forex_slippage_pips=0.1 * multiplier,
                                forex_swap_annual_bps=100 * multiplier,
                                forex_additional_spread_pips=0 if multiplier == 1 else 1,
                            )
                            execution = Engine(
                                frame,
                                symbol,
                                cost,
                                execution_frame=raw,
                                bars_per_horizon=24 if interval == "1d" else 4,
                            )
                            summary, trades, curve = execution.portfolio(
                                out.assign(probability=probability), 1, False, 0.55, end
                            )
                            save_frame(destination / f"{scenario}_curve.csv", curve)
                            save_frame(destination / f"{scenario}_trades.csv", trades)
                            write_new_json(destination / f"{scenario}_economic.json", summary)
                        write_new_json(destination / "complete.json", {"completed": True})
                        print(
                            "FOREX", destination, "train", len(train), "test", len(out), flush=True
                        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "pilot", "full", "neural", "events", "forex"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.stage == "audit":
        audit()
    elif args.stage in ["pilot", "full"]:
        tabular(args.stage == "pilot")
    elif args.stage == "neural":
        neural()
    elif args.stage == "events":
        event_models()
    elif args.stage == "forex":
        forex()


if __name__ == "__main__":
    main()
