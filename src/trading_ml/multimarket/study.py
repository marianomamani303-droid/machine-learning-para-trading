from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import (
    code_provenance,
    environment,
    frame_hash,
    read_json,
    sha256,
    sidecar,
    write_new_json,
)
from trading_ml.config.settings import Settings
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import moving_block_indices
from trading_ml.feature_research.study import _frozen_hashes
from trading_ml.models.baselines import fit_reference_models
from trading_ml.models.calibration import apply_calibration, fit_sigmoid
from trading_ml.multimarket.data import (
    _snapshot,
    aggregate_forex,
    download_forex,
    execution_source,
    load_snapshot,
)
from trading_ml.multimarket.execution import Costs, Engine
from trading_ml.multimarket.features import GROUPS, PRICE, features, recipes
from trading_ml.multimarket.markets import instrument
from trading_ml.multimarket.protocol import load_protocol
from trading_ml.preprocessing.quality import clean_ohlcv
from trading_ml.training.recipe import protect_reserved_labels

LOG = logging.getLogger(__name__)


def dataset(settings, protocol, symbol, interval, refresh=False):
    symbol = instrument(symbol).symbol
    if instrument(symbol).market == "forex":
        if protocol.get("forex_provider", "dukascopy") == "fxcm":
            from trading_ml.multimarket.fxcm import download_fxcm

            source = download_fxcm(
                settings.root, symbol, protocol["start"], protocol["end"], refresh
            )
        else:
            source = download_forex(
                settings.root,
                symbol,
                protocol["start"],
                protocol["end"],
                refresh,
                transport=protocol.get("dukascopy_transport", "https"),
            )
        return aggregate_forex(settings.root, source, interval)
    if interval != "1d":
        raise DataError(
            "La comparación cripto reutiliza los historiales diarios; use el descargador original para ampliar intervalos"
        )
    path = settings.root / (
        "data/raw.csv" if symbol == "BTCUSDT" else "research_data/ETHUSDT_20200101_20261001.csv"
    )
    record = read_json(sidecar(path))
    if (
        record.get("sha256") != sha256(path)
        or record["market"]["symbol"] != symbol
        or record["market"]["interval"] != "1d"
    ):
        raise DataError("Historial cripto sin procedencia compatible")
    frame, quality = clean_ohlcv(pd.read_csv(path), pd.Timedelta(days=1))
    frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
    first, last = (
        pd.to_datetime(protocol["start"], utc=True),
        pd.to_datetime(protocol["end"], utc=True),
    )
    frame = frame.loc[(frame.timestamp >= first) & (frame.reference_time <= last)]
    if frame.empty or frame.timestamp.min() > first or frame.reference_time.max() < last:
        raise DataError("Cobertura cripto insuficiente")
    return _snapshot(
        settings.root,
        symbol,
        frame,
        {
            "instrument": instrument(symbol).contract(),
            "start": first.isoformat(),
            "end_exclusive": last.isoformat(),
            "interval": interval,
            "quality": quality,
            "sources": [
                {
                    "source": str(path.resolve()),
                    "sha256": sha256(path),
                    "manifest_sha256": sha256(sidecar(path)),
                }
            ],
            "schema": "multimarket_crypto_v1",
        },
    )


def labelled(frame, engine, horizon, target, barrier):
    result = frame.copy()
    if target == "direction":
        future = result.close.shift(-horizon)
        result["target"] = (future > result.close).astype(float).where(future.notna())
        result["target_time"] = result.reference_time.shift(-horizon)
    else:
        rows = [engine.event(i, horizon, barrier) for i in range(len(frame))]
        result["target"] = [r["target"] if r else np.nan for r in rows]
        result["target_time"] = pd.to_datetime(
            [r["target_time"] if r else pd.NaT for r in rows], utc=True
        )
    return result


def experiments(symbol, interval):
    sets = recipes(symbol)
    result = [
        {
            "name": "direction_price_h1",
            "columns": sets["price"],
            "horizon": 1,
            "target": "direction",
            "barrier": False,
        },
        {
            "name": "direction_compact_h1",
            "columns": sets["compact"],
            "horizon": 1,
            "target": "direction",
            "barrier": False,
        },
        {
            "name": "direction_compact_h3",
            "columns": sets["compact"],
            "horizon": 3,
            "target": "direction",
            "barrier": False,
        },
        {
            "name": "net_price_h1",
            "columns": sets["price"],
            "horizon": 1,
            "target": "net_profit",
            "barrier": False,
        },
        {
            "name": "tb_price_h3",
            "columns": sets["price"],
            "horizon": 3,
            "target": "net_profit",
            "barrier": True,
        },
        {
            "name": "tb_compact_h3",
            "columns": sets["compact"],
            "horizon": 3,
            "target": "net_profit",
            "barrier": True,
        },
    ]
    if interval == "1d":
        for name, columns in GROUPS.items():
            if set(columns) & set(sets["compact"]):
                result.append(
                    {
                        **result[1],
                        "name": f"direction_without_{name}",
                        "columns": [c for c in sets["compact"] if c not in columns],
                    }
                )
        if instrument(symbol).market == "crypto":
            result.append(
                {
                    **result[0],
                    "name": "direction_original14_h1",
                    "columns": PRICE + ["volume_ratio"],
                }
            )
    return result


def require_outcomes(rows, engine, target):
    """Missing future outcomes may stop a study, never silently select its trades."""
    for missing in rows.loc[rows.target.isna()].itertuples():
        entry = int(engine.entries[int(missing.source_index)])
        crossed_entry = (
            target != "direction"
            and engine.fx
            and entry < len(engine.execution)
            and engine.ask["open"][entry] < engine.bid["open"][entry]
        )
        if not crossed_entry:
            raise DataError(
                "Outcomes incompletos dentro de OOS: se detiene el experimento, no se filtran operaciones por etiquetas futuras censuradas"
            )


def _fit(train, columns, protocol):
    model = XGBClassifier(
        **protocol["model"],
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=protocol["seed"],
    )
    model.fit(train[columns], train.target.astype(int))
    return model


def _loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def bootstrap_skill(predictions, samples, blocks, seed):
    y = predictions.target.to_numpy(int)
    delta = _loss(y, predictions.probability.to_numpy()) - _loss(
        y, predictions.train_prevalence.to_numpy()
    )
    output = {}
    for block in blocks:
        indices = moving_block_indices(predictions.fold.to_numpy(), samples, block, seed)
        means = delta[indices].mean(axis=1)
        # Inference remains exploratory: no multiplicity-adjusted promotion.
        output[str(block)] = {
            "mean_log_loss_difference_vs_train_prior": float(delta.mean()),
            "lower_95": float(np.quantile(means, 0.025)),
            "upper_95": float(np.quantile(means, 0.975)),
            "better_all_ci": bool(np.quantile(means, 0.975) < 0),
        }
    return output


def _summary(predictions, folds, trades, curves):
    metrics = classification_metrics(
        predictions.target.to_numpy(int), predictions.probability.to_numpy()
    )
    for baseline in ["constant_05", "train_prevalence", "logistic_l2", "random_probability"]:
        metrics[baseline] = classification_metrics(
            predictions.target.to_numpy(int), predictions[baseline].to_numpy()
        )
    profits = (
        np.concatenate([t.profit_usd.to_numpy() for t in trades if len(t)])
        if any(len(t) for t in trades)
        else np.array([])
    )
    equity = [10000.0]
    cumulative = 1.0
    for fold, curve in zip(folds, curves, strict=True):
        if len(curve):
            # Each fold is an independent initial-capital account; join its
            # already net equity as a scaled return sequence for aggregate DD.
            values = curve.equity.to_numpy() / fold["initial_capital"]
            equity.extend((values * cumulative * 10000).tolist())
        cumulative *= 1 + fold["net_return"]
    equity = np.asarray(equity)
    losses = -profits[profits < 0].sum()
    return {
        "classification": metrics,
        "trades": len(profits),
        "win_rate": float((profits > 0).mean()) if len(profits) else None,
        "profit_factor": float(profits[profits > 0].sum() / losses) if losses else None,
        "net_return": float(cumulative - 1),
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "positive_folds": int(sum(f["net_return"] > 0 for f in folds)),
        "total_folds": len(folds),
        "fold_returns": [f["net_return"] for f in folds],
        "auc_above_half_folds": int(sum(f["auc"] is not None and f["auc"] > 0.5 for f in folds)),
        "costs_usd": float(
            sum(
                (t.fees_per_unit * t.units).sum() + (t.swap_per_unit * t.units).sum()
                for t in trades
                if len(t)
            )
        ),
        "profit_factor_aggregation": "sum of independent fold USD profits / losses; equal starting capital each fold",
    }


def _parent_oof(frame, protocol, root, destination):
    """Quarterly refits; every meta-training parent probability was out of sample."""
    first = pd.to_datetime(protocol["start"], utc=True) + pd.DateOffset(years=1)
    last = pd.to_datetime(protocol["end"], utc=True)
    bounds = list(pd.date_range(first, last, freq="QS"))
    if bounds[-1] < last:
        bounds.append(last)
    rows, contracts = [], []
    for number, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        cutoff = start - pd.Timedelta(days=protocol["embargo_days"])
        train = frame.loc[(frame.target_time < cutoff)].dropna(subset=PRICE + ["target"])
        validation = frame.loc[
            (frame.reference_time >= start) & (frame.reference_time < end)
        ].dropna(subset=PRICE)
        if len(train) < protocol["min_train_rows"] or validation.empty:
            continue
        protect_reserved_labels(root, train)
        model = _fit(train, PRICE, protocol)
        part = validation[["source_index", "reference_time"]].copy()
        part["parent_probability"] = model.predict_proba(validation[PRICE])[:, 1]
        rows.append(part)
        path = destination / f"parent_oof_{number}.ubj"
        model.save_model(path)
        contracts.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "fit_rows": len(train),
                "fit_sha256": frame_hash(
                    train[PRICE + ["target", "target_time", "reference_time"]]
                ),
                "last_target": train.target_time.max().isoformat(),
                "first_prediction": validation.reference_time.min().isoformat(),
            }
        )
    if not rows:
        raise DataError("No hay predicciones OOF para investigar meta-labeling")
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(destination / "parent_oof.csv", index=False)
    write_new_json(destination / "parent_oof_contracts.json", {"models": contracts})
    return combined


def run_asset(settings, protocol, symbol, interval, destination, source):
    destination.mkdir(parents=True, exist_ok=False)
    raw, metadata = load_snapshot(source)
    frame = features(raw, symbol)
    if instrument(symbol).market == "crypto":
        # Preserve the old volume feature for an additional explicit comparator.
        frame["volume_ratio"] = raw.volume / raw.volume.rolling(20).mean().replace(0, np.nan)
    costs = Costs(**protocol["costs"])
    execution, multiplier = execution_source(raw, metadata)
    engine = Engine(frame, symbol, costs, execution_frame=execution, bars_per_horizon=multiplier)
    compact = recipes(symbol)["compact"]
    common_columns = compact + (["volume_ratio"] if instrument(symbol).market == "crypto" else [])
    frames = {
        "direction_1": labelled(frame, engine, 1, "direction", False),
        "direction_3": labelled(frame, engine, 3, "direction", False),
        "net_1": labelled(frame, engine, 1, "net_profit", False),
        "tb_3": labelled(frame, engine, 3, "net_profit", True),
    }
    write_new_json(
        destination / "data_contract.json",
        {
            "path": str(source.resolve()),
            "sha256": sha256(source),
            "manifest_sha256": sha256(sidecar(source)),
            "metadata": metadata,
            "feature_schema": "multimarket_price_v1",
            "columns": compact,
        },
    )
    oof = _parent_oof(frames["direction_1"], protocol, settings.root, destination)
    meta = frames["net_1"].merge(
        oof[["source_index", "parent_probability"]],
        on="source_index",
        how="left",
        validate="one_to_one",
    )
    meta = meta.loc[meta.parent_probability >= 0.55].copy()
    configurations = experiments(symbol, interval)
    bounds = pd.to_datetime(protocol["boundaries"], utc=True)
    all_predictions, all_folds, all_trades, all_curves = {}, {}, {}, {}
    fit_count = len(read_json(destination / "parent_oof_contracts.json")["models"])
    feature_importance = []

    def record(name, prediction, fold, trades, curve):
        all_predictions.setdefault(name, []).append(prediction)
        all_folds.setdefault(name, []).append(fold)
        all_trades.setdefault(name, []).append(trades)
        all_curves.setdefault(name, []).append(curve)

    def execute(name, prediction, fold_index, start, end, horizon, barrier, threshold=0.55):
        summary, trades, curve = engine.portfolio(prediction, horizon, barrier, threshold, end)
        trades.to_csv(destination / f"{name}_fold{fold_index}_trades.csv", index=False)
        curve.to_csv(destination / f"{name}_fold{fold_index}_equity.csv", index=False)
        summary.update(
            {
                "fold": fold_index,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "auc": classification_metrics(
                    prediction.target.to_numpy(int), prediction.probability.to_numpy()
                )["roc_auc"],
                "initial_capital": costs.initial_capital,
            }
        )
        record(name, prediction, summary, trades, curve)

    for fold_index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True), 1):
        cutoff = start - pd.Timedelta(days=protocol["embargo_days"])
        calibration_start = cutoff - pd.Timedelta(days=protocol["calibration_days"])
        LOG.info("%s %s fold %d/%d", symbol, interval, fold_index, len(bounds) - 1)
        parent_model = None
        for spec in configurations:
            key = (
                f"direction_{spec['horizon']}"
                if spec["target"] == "direction"
                else ("tb_3" if spec["barrier"] else "net_1")
            )
            data = frames[key]
            cols = spec["columns"]
            train = data.loc[
                data.target_time < calibration_start - pd.Timedelta(days=protocol["embargo_days"])
            ].dropna(subset=common_columns + ["target"])
            cal = data.loc[
                (data.reference_time >= calibration_start) & (data.target_time < cutoff)
            ].dropna(subset=common_columns + ["target"])
            # Same ex-ante calendar cutoff for all targets and horizons, not the realized label exit.
            candidate_rows = data.loc[
                (data.reference_time >= start) & (data.reference_time < end - pd.Timedelta(days=20))
            ].dropna(subset=common_columns)
            require_outcomes(candidate_rows, engine, spec["target"])
            observed = candidate_rows.dropna(subset=["target"])
            if (
                len(train) < protocol["min_train_rows"]
                or len(cal) < 20
                or len(observed) < 40
                or train.target.nunique() != 2
            ):
                raise DataError(
                    f"Partición insuficiente {symbol}/{interval}/{spec['name']}/fold{fold_index}"
                )
            if observed.target_time.max() >= end:
                raise DataError(
                    "El buffer fijo no alcanza para cerrar las etiquetas; aumentar protocolo, no filtrar por resultados"
                )
            protect_reserved_labels(settings.root, train, cal, observed)
            model = _fit(train, cols, protocol)
            fit_count += 1
            ref = fit_reference_models(train, cols, seed=protocol["seed"])
            fit_count += 1
            p = model.predict_proba(observed[cols])[:, 1]
            calibration = (
                fit_sigmoid(
                    model.predict_proba(cal[cols])[:, 1], cal.target.to_numpy(int), protocol["seed"]
                )
                if cal.target.nunique() == 2
                else None
            )
            fit_count += int(calibration is not None)
            prediction = observed[
                ["source_index", "reference_time", "target_time", "target"]
            ].copy()
            prediction["probability"] = p
            for name, values in ref.probabilities(observed).items():
                prediction[name] = values
            rng = np.random.default_rng(protocol["seed"] + fold_index)
            prediction["random_probability"] = np.clip(
                rng.beta(25, 25, len(prediction)), 1e-6, 1 - 1e-6
            )
            prediction["fold"] = fold_index
            model_path = destination / f"{spec['name']}_fold{fold_index}.ubj"
            model.save_model(model_path)
            contract = {
                "symbol": symbol,
                "interval": interval,
                "instrument": metadata["instrument"],
                "specification": spec,
                "feature_schema": "multimarket_price_v1",
                "model_sha256": sha256(model_path),
                "source": str(source.resolve()),
                "data_sha256": sha256(source),
                "fit_rows": len(train),
                "fit_sha256": frame_hash(train[cols + ["reference_time", "target", "target_time"]]),
                "last_train_target": train.target_time.max().isoformat(),
                "first_calibration_reference": cal.reference_time.min().isoformat(),
                "last_calibration_target": cal.target_time.max().isoformat(),
                "first_oos_reference": observed.reference_time.min().isoformat(),
                "evaluation_start": start.isoformat(),
                "evaluation_end": end.isoformat(),
                "calibration": calibration,
                "reference_models": ref.to_dict(),
                "seed": protocol["seed"],
                "model_parameters": protocol["model"],
                "costs": asdict(costs),
                "production": False,
            }
            write_new_json(model_path.with_suffix(".json"), contract)
            execute(
                spec["name"], prediction, fold_index, start, end, spec["horizon"], spec["barrier"]
            )
            if spec["name"] in {"direction_price_h1", "tb_compact_h3"}:
                cp = prediction.copy()
                cp["probability"] = apply_calibration(p, calibration)
                execute(
                    spec["name"] + "_cal",
                    cp,
                    fold_index,
                    start,
                    end,
                    spec["horizon"],
                    spec["barrier"],
                )
                for threshold in [0.50, 0.60, 0.65]:
                    execute(
                        spec["name"] + f"_cal_t{int(threshold * 100)}",
                        cp.copy(),
                        fold_index,
                        start,
                        end,
                        spec["horizon"],
                        spec["barrier"],
                        threshold,
                    )
            if spec["name"] == "direction_price_h1":
                parent_model = model
                for baseline in ["logistic_l2", "train_prevalence", "random_probability"]:
                    bp = prediction.copy()
                    bp["probability"] = prediction[baseline]
                    execute(baseline, bp, fold_index, start, end, 1, False)
                for baseline, values in [
                    ("always_long", np.ones(len(prediction))),
                    ("momentum_rule", (observed.return_7.to_numpy() > 0).astype(float)),
                    ("mean_reversion_rule", (observed.zscore_20.to_numpy() < -1).astype(float)),
                ]:
                    bp = prediction.copy()
                    bp["probability"] = values
                    execute(baseline, bp, fold_index, start, end, 1, False)
            if spec["name"] == "direction_compact_h1":
                base_loss = _loss(observed.target.to_numpy(), p).mean()
                for group_name, group_columns in GROUPS.items():
                    selected = [c for c in group_columns if c in cols]
                    if not selected:
                        continue
                    differences = []
                    for shift in [7, 19, 37]:
                        shuffled = observed[cols].copy()
                        shuffled[selected] = np.roll(shuffled[selected].to_numpy(), shift, axis=0)
                        differences.append(
                            float(
                                _loss(
                                    observed.target.to_numpy(), model.predict_proba(shuffled)[:, 1]
                                ).mean()
                                - base_loss
                            )
                        )
                    feature_importance.append(
                        {
                            "fold": fold_index,
                            "group": group_name,
                            "permutation_loss_increase": differences,
                            "gain": {
                                c: float(v)
                                for c, v in zip(cols, model.feature_importances_, strict=True)
                            },
                        }
                    )
        if parent_model is None:
            raise DataError("Falta modelo padre")
        meta_cols = compact + ["parent_probability"]
        mt = meta.loc[
            meta.target_time < calibration_start - pd.Timedelta(days=protocol["embargo_days"])
        ].dropna(subset=meta_cols + ["target"])
        mc = meta.loc[
            (meta.reference_time >= calibration_start) & (meta.target_time < cutoff)
        ].dropna(subset=meta_cols + ["target"])
        base = frames["net_1"]
        mo = (
            base.loc[
                (base.reference_time >= start) & (base.reference_time < end - pd.Timedelta(days=20))
            ]
            .dropna(subset=compact + ["target"])
            .copy()
        )
        mo["parent_probability"] = parent_model.predict_proba(mo[PRICE])[:, 1]
        # Keep all prediction rows: parent abstention is a signal filter, not missing observations.
        if len(mt) >= 40 and mt.target.nunique() == 2:
            protect_reserved_labels(settings.root, mt, mc, mo)
            mm = _fit(mt, meta_cols, protocol)
            fit_count += 1
            calibration = (
                fit_sigmoid(mm.predict_proba(mc[meta_cols])[:, 1], mc.target.to_numpy(int))
                if len(mc) >= 20 and mc.target.nunique() == 2
                else None
            )
            fit_count += int(calibration is not None)
            mp = apply_calibration(mm.predict_proba(mo[meta_cols])[:, 1], calibration)
            mr = fit_reference_models(mt, meta_cols, seed=protocol["seed"])
            fit_count += 1
            prediction = mo[
                ["source_index", "reference_time", "target_time", "target", "parent_probability"]
            ].copy()
            prediction["meta_probability"] = mp
            prediction["probability"] = np.where(mo.parent_probability >= 0.55, mp, 0.0)
            for name, values in mr.probabilities(mo).items():
                prediction[name] = values
            prediction["random_probability"] = np.full(len(mo), 0.5)
            prediction["fold"] = fold_index
            name = "meta_net_h1"
            path = destination / f"{name}_fold{fold_index}.ubj"
            mm.save_model(path)
            write_new_json(
                path.with_suffix(".json"),
                {
                    "symbol": symbol,
                    "interval": interval,
                    "specification": {
                        "name": name,
                        "columns": meta_cols,
                        "horizon": 1,
                        "target": "net_profit_given_parent_signal",
                        "barrier": False,
                    },
                    "model_sha256": sha256(path),
                    "parent_model": f"direction_price_h1_fold{fold_index}.ubj",
                    "calibration": calibration,
                    "last_train_target": mt.target_time.max().isoformat(),
                    "last_calibration_target": mc.target_time.max().isoformat()
                    if len(mc)
                    else None,
                    "first_oos_reference": mo.reference_time.min().isoformat(),
                    "fit_rows": len(mt),
                    "calibration_rows": len(mc),
                    "source": str(source.resolve()),
                    "data_sha256": sha256(source),
                    "costs": asdict(costs),
                    "production": False,
                },
            )
            execute(name, prediction, fold_index, start, end, 1, False)
        else:
            write_new_json(
                destination / f"meta_fold{fold_index}_insufficient.json",
                {
                    "rows": len(mt),
                    "reason": "insufficient causal parent OOF signals; no in-sample substitution",
                },
            )
    reports = {}
    for name, parts in all_predictions.items():
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(destination / f"{name}_predictions.csv", index=False)
        summary = _summary(combined, all_folds[name], all_trades[name], all_curves[name])
        summary["folds"] = all_folds[name]
        # Meta's combined policy score is not a calibrated probability. Report
        # conditional calibration on the population on which the filter was fitted.
        if name == "meta_net_h1":
            subset = combined.loc[combined.parent_probability >= 0.55]
            summary["conditional_meta_classification"] = (
                classification_metrics(
                    subset.target.to_numpy(int), subset.meta_probability.to_numpy()
                )
                if len(subset)
                else None
            )
            summary["classification_score_semantics"] = (
                "policy score zero on parent abstentions; use conditional_meta_classification for meta probability quality"
            )
        if not name.startswith("meta"):
            summary["bootstrap_skill"] = bootstrap_skill(
                combined,
                protocol["bootstrap_samples"],
                protocol["bootstrap_blocks"],
                protocol["seed"],
            )
        reports[name] = summary
    stress_costs = replace(
        costs,
        forex_commission_bps=costs.forex_commission_bps * 2,
        forex_slippage_pips=costs.forex_slippage_pips * 2,
        forex_swap_annual_bps=300.0,
        crypto_commission_bps=costs.crypto_commission_bps * 2,
        crypto_slippage_bps=costs.crypto_slippage_bps * 2,
    )
    stressed = Engine(
        frame, symbol, stress_costs, execution_frame=execution, bars_per_horizon=multiplier
    )
    for name in ["direction_price_h1", "tb_compact_h3_cal", "meta_net_h1"]:
        if name not in all_predictions:
            continue
        values = []
        for part, fold in zip(all_predictions[name], all_folds[name], strict=True):
            end = pd.to_datetime(fold["end"], utc=True)
            summary, _, _ = stressed.portfolio(
                part, 3 if name.startswith("tb") else 1, name.startswith("tb"), 0.55, end
            )
            values.append(summary["net_return"])
        reports[name]["stress_net_return"] = float(np.prod(1 + np.asarray(values)) - 1)
    write_new_json(
        destination / "feature_importance.json",
        {
            "diagnostic_only": True,
            "method": "group circular-shift permutation, fixed offsets 7/19/37 per OOS fold",
            "records": feature_importance,
        },
    )
    result = {
        "symbol": symbol,
        "interval": interval,
        "actual_fits": fit_count,
        "data": str(source.resolve()),
        "data_sha256": sha256(source),
        "quality": metadata["quality"],
        "observed_close_volatility": float(frame.return_1.std()),
        "experiments": reports,
        "adopted": False,
        "decision": "research only; no production promotion; historical OOS is not a pristine final test",
    }
    write_new_json(destination / "summary.json", result)
    return result


def run_study(
    settings: Settings, protocol_path: Path, symbols=None, intervals=None, pilot=None, refresh=False
):
    protocol = load_protocol(protocol_path)
    chosen = [instrument(s).symbol for s in (symbols or protocol["symbols"])]
    timeframes = intervals or protocol["intervals"]
    if any(s not in protocol["symbols"] for s in chosen) or not set(timeframes) <= {
        "1d",
        "4h",
        "1h",
    }:
        raise DataError("Activos o intervalos fuera del protocolo")
    if any(instrument(s).market == "forex" and s != "EURUSD" for s in chosen):
        if pilot is None:
            raise DataError(
                "Primero ejecute el piloto EURUSD y pase --pilot <directorio>; el piloto valida el pipeline, no exige rentabilidad"
            )
        report = read_json(pilot / "summary.json")
        if not report.get("pilot_pipeline_validated") or report["protocol_sha256"] != sha256(
            protocol_path
        ):
            raise DataError("Piloto no validado o de otro protocolo")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = settings.root / "multimarket" / "studies" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    frozen = _frozen_hashes(settings)
    write_new_json(
        destination / "plan.json",
        {
            "run_id": run_id,
            "protocol": protocol,
            "protocol_sha256": sha256(protocol_path),
            "chosen_symbols": chosen,
            "chosen_intervals": timeframes,
            "code": code_provenance(),
            "environment": environment(),
            "frozen_before": frozen,
            "pilot": str(pilot.resolve()) if pilot else None,
            "selection": "fixed hypotheses; no optimization on evaluation folds; no automatic adoption",
            "split": "expanding fit / disjoint 180d calibration / 7d embargo / OOS; label-time purging at every boundary",
            "data_integrity_limit": "HTTP legacy transport if explicitly configured; local hashes are not provider signatures",
        },
    )
    reports = {}
    for symbol in chosen:
        for interval in timeframes:
            path = dataset(settings, protocol, symbol, interval, refresh)
            reports[f"{symbol}_{interval}"] = run_asset(
                settings, protocol, symbol, interval, destination / f"{symbol}_{interval}", path
            )
    after = _frozen_hashes(settings)
    if frozen != after:
        raise DataError("Se alteraron modelos o reservas previos")
    report = {
        "run_id": run_id,
        "directory": str(destination.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "pilot_pipeline_validated": chosen == ["EURUSD"]
        and all(r["actual_fits"] > 0 for r in reports.values()),
        "results": reports,
        "actual_fits": sum(r["actual_fits"] for r in reports.values()),
        "preserved_frozen_files": len(frozen),
        "frozen_unchanged": True,
        "adopted": False,
    }
    write_new_json(destination / "summary.json", report)
    rows = []
    for key, value in reports.items():
        for name, experiment in value["experiments"].items():
            metrics = experiment["classification"]
            rows.append(
                {
                    "asset_interval": key,
                    "experiment": name,
                    **{
                        c: metrics[c]
                        for c in ["accuracy", "precision", "roc_auc", "log_loss", "brier_score"]
                    },
                    **{
                        c: experiment[c]
                        for c in [
                            "trades",
                            "win_rate",
                            "profit_factor",
                            "net_return",
                            "max_drawdown",
                            "positive_folds",
                            "total_folds",
                        ]
                    },
                }
            )
    pd.DataFrame(rows).to_csv(destination / "comparison.csv", index=False)
    return report
