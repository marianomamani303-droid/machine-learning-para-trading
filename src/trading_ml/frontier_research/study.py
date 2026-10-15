from __future__ import annotations

import importlib.metadata
import json
import tomllib
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error

from trading_ml.advanced_research.data import recover_microstructure
from trading_ml.advanced_research.statistics import paired_predictive
from trading_ml.common import (
    code_provenance,
    environment,
    frame_hash,
    read_json,
    sha256,
    write_new_json,
)
from trading_ml.config import Settings
from trading_ml.data.binance import download
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.frontier_research.features import CROSS, LOCAL, multi_asset_features
from trading_ml.frontier_research.models import fit_model, predict_probability, predict_value
from trading_ml.frontier_research.targets import research_targets
from trading_ml.frontier_research.validation import (
    evaluation_rows,
    prediction_rows,
    temporal_partitions,
)
from trading_ml.mean_reversion.study import _protected_hashes
from trading_ml.models.calibration import apply_calibration, fit_sigmoid
from trading_ml.momentum.config import ResearchCosts
from trading_ml.risk_overlay.config import RiskPolicy
from trading_ml.risk_overlay.simulator import run_overlay
from trading_ml.risk_overlay.study import _check_reserved_end, aggregate, reservation_hashes


def metrics(y, p):
    return {
        **classification_metrics(np.asarray(y), np.asarray(p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, np.asarray(p) >= 0.5)),
    }


def policy():
    return RiskPolicy(
        name="frontier_fixed_25",
        capital_cap=0.25,
        sizing="combined",
        atr_stop=True,
        soft_drawdown=0.05,
        hard_drawdown=0.10,
        confirmations=2,
        cooldown_bars=2,
        persistent=True,
        max_holding_bars=5,
    )


def economic(frame, p, start, end, costs, threshold):
    # Missing closed-candle features abstain; no future label determines execution.
    return run_overlay(
        frame, np.nan_to_num(p, nan=0.5), start, end, costs, policy(), threshold=threshold
    ).simulation


def _save_sim(directory, simulation):
    directory.mkdir(parents=True, exist_ok=False)
    simulation.equity.to_csv(directory / "equity.csv", index=False)
    simulation.trades.to_csv(directory / "trades.csv", index=False)
    write_new_json(directory / "metrics.json", simulation.metrics)


def _aggregate_sims(items):
    result = aggregate(items)
    result.pop("_curve")
    return result


def run_frontier_study(settings: Settings, protocol_path: Path, cache_from: Path | None = None):
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    r = protocol["research"]
    dates = pd.to_datetime(r["boundaries"], utc=True)
    end = pd.Timestamp(r["historical_end"], tz="UTC")
    future_start = pd.Timestamp(r["prospective_start"], tz="UTC")
    future_end = pd.Timestamp(r["prospective_end"], tz="UTC")
    if (
        not dates.is_monotonic_increasing
        or dates.duplicated().any()
        or len(dates) != 4
        or not dates[-1] < end < pd.Timestamp.now(tz="UTC") < future_start < future_end
        or r["embargo_days"] < 6
        or len(protocol["recipes"]) > 8
        or r["maximum_fits"] > 500
        or r["symbols"] != ["BTCUSDT", "ETHUSDT"]
        or r["thresholds"] != [0.55, 0.60]
    ):
        raise DataError("Protocolo frontier incompatible con búsqueda acotada")
    _check_reserved_end(settings.root, end)
    costs = ResearchCosts(**protocol["costs"])
    costs.validate()
    protected, reserved = _protected_hashes(settings), reservation_hashes(settings.root)
    prior_charge = 0
    cache_hashes = {}
    if cache_from is not None:
        if read_json(cache_from / "plan.json")["protocol_sha256"] != sha256(protocol_path):
            raise DataError("Caché de otro protocolo")
        prior_charge = read_json(cache_from / "failure.json")["budget_charge"]
        cache_hashes = {str(p.resolve()): sha256(p) for p in cache_from.rglob("*") if p.is_file()}
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = settings.root / "frontier_studies" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "protocol.toml").write_bytes(protocol_path.read_bytes())
    write_new_json(
        directory / "plan.json",
        {
            "protocol": protocol,
            "protocol_sha256": sha256(protocol_path),
            "protected_models": protected,
            "reserved": reserved,
            "code": code_provenance(),
            "environment": {
                **environment(),
                **{p: importlib.metadata.version(p) for p in ["lightgbm", "catboost", "joblib"]},
            },
            "historical_is_development": True,
            "independent_final_test": "future_pending",
            "prospective_start": future_start.isoformat(),
            "prospective_end": future_end.isoformat(),
            "selection": "3 chronological outer folds; train plus disjoint temporal calibration; 6-day embargo on target end at each boundary",
            "deep_learning": "excluded: 2465 daily rows per strongly correlated asset, few market cycles; no justified sequence sample size",
            "hypotheses": [
                "Cross-asset closed-candle features add predictive information",
                "Alternative regularized trees outperform XGBoost and logistic",
                "Calibration improves loss without establishing economic edge",
                "Net and close-barrier labels better approximate opportunities",
            ],
            "live_trading": False,
            "automatic_promotion": False,
            "prior_failed_budget_charge": prior_charge,
            "cache_hashes": cache_hashes,
        },
    )
    fits, records, reused_models = prior_charge, 0, 0

    def consume():
        nonlocal fits
        if fits >= r["maximum_fits"]:
            raise DataError("Presupuesto de ajustes agotado")
        fits += 1
        with (directory / "fit_budget.jsonl").open("a", encoding="utf-8") as ledger:
            ledger.write(
                json.dumps({"charged_fits": fits, "at": datetime.now(UTC).isoformat()}) + "\n"
            )

    def record(value):
        nonlocal records
        with (directory / "experiments.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        records += 1

    try:
        raw, sources = {}, {}
        for symbol in r["symbols"]:
            asset_settings = replace(
                settings, market=replace(settings.market, symbol=symbol, end=r["historical_end"])
            )
            path = (
                settings.root / "data" / "raw.csv"
                if symbol == "BTCUSDT"
                else (settings.root / "research_data" / "ETHUSDT_20200101_20261001.csv")
            )
            if not path.exists():
                download(asset_settings, output=path)
            raw[symbol], sources[symbol] = recover_microstructure(asset_settings, path)
            if (raw[symbol].reference_time > end).any():
                raise LeakageError("Datos fuera del intervalo histórico permitido")
        frames = multi_asset_features(raw)
        if cache_from is not None:
            prior_sources = read_json(cache_from / "data_provenance.json")
            if any(
                sources[s]["source_sha256"] != prior_sources[s]["source_sha256"] for s in sources
            ):
                raise DataError("Caché de otros datos")
        targets = {s: research_targets(f, costs) for s, f in frames.items()}
        write_new_json(directory / "data_provenance.json", sources)
        for symbol, frame in frames.items():
            frame.to_csv(directory / f"{symbol}_causal_features.csv", index=False)
        predictions, execution_predictions, sims, frozen, rankings = {}, {}, {}, {}, {}
        # Each later fold may learn past folds; the final descriptive period is not
        # opened until the recipe and threshold have been committed to selection.json.
        windows = [(dates[i], dates[i + 1], "selection", i + 1) for i in range(3)]
        windows.append((dates[-1], end, "historical_description", 4))
        for start, stop, phase, fold in windows:
            print(f"frontier {phase} fold={fold} fits={fits}", flush=True)
            for symbol, objectives in targets.items():
                for objective, frame in objectives.items():
                    key = symbol + "__" + objective
                    observed = evaluation_rows(frame, start, stop)
                    forecast = prediction_rows(frame, start, stop)
                    scoring_positions = forecast.index.get_indexer(observed.index)
                    execution = forecast[["reference_time", "regime"]].copy()
                    execution["fold"] = fold
                    train, cal = temporal_partitions(
                        frame,
                        start,
                        embargo_days=r["embargo_days"],
                        calibration_days=r["calibration_days"],
                    )
                    base = observed[["reference_time", "target_time", "target", "regime"]].copy()
                    base["fold"] = fold
                    raw_cal, raw_eval, calibrated = {}, {}, {}
                    for recipe in protocol["recipes"]:
                        fitting, calibration = temporal_partitions(
                            frame,
                            start,
                            embargo_days=r["embargo_days"],
                            calibration_days=r["calibration_days"],
                            rolling_days=recipe.get("rolling_days"),
                        )
                        columns = LOCAL + CROSS if recipe["features"] == "cross" else LOCAL
                        adjusted = {**recipe, "parameters": recipe.get("parameters", {}).copy()}
                        if recipe.get("balance"):
                            adjusted["parameters"]["scale_pos_weight"] = float(
                                (fitting.target == 0).sum() / (fitting.target == 1).sum()
                            )
                        cached = (
                            cache_from / phase / str(fold) / key / recipe["id"]
                            if cache_from
                            else None
                        )
                        previous = (
                            read_json(cached / "fit.json")
                            if cached and (cached / "fit.json").exists()
                            else None
                        )
                        if previous is not None:
                            if (
                                previous["fit_sha256"]
                                != frame_hash(fitting[columns + ["target", "target_time"]])
                                or previous["recipe"] != adjusted
                                or previous["features"] != columns
                                or previous["calibration_rows"] != len(calibration)
                                or previous["calibration_last_target"]
                                != calibration.target_time.max().isoformat()
                                or previous["model_sha256"] != sha256(cached / "model.joblib")
                            ):
                                raise DataError("Contrato de ajuste cacheado incompatible")
                            estimator = joblib.load(cached / "model.joblib")
                            reused_models += 1
                        else:
                            consume()
                            estimator = fit_model(fitting, columns, "target", adjusted)
                        raw_cal[recipe["id"]] = predict_probability(estimator, calibration, columns)
                        raw_eval[recipe["id"]] = predict_probability(estimator, forecast, columns)
                        if previous is not None:
                            calibration_state = previous["calibration"]
                        else:
                            consume()
                            calibration_state = fit_sigmoid(
                                raw_cal[recipe["id"]], calibration.target.to_numpy()
                            )
                        p = apply_calibration(raw_eval[recipe["id"]], calibration_state)
                        calibrated[recipe["id"]] = p
                        base[recipe["id"]] = p[scoring_positions]
                        execution[recipe["id"]] = p
                        base[recipe["id"] + "_raw"] = raw_eval[recipe["id"]][scoring_positions]
                        model_dir = directory / phase / str(fold) / key / recipe["id"]
                        model_dir.mkdir(parents=True, exist_ok=False)
                        joblib.dump(estimator, model_dir / "model.joblib")
                        audit = {
                            "phase": phase,
                            "fold": fold,
                            "asset": symbol,
                            "objective": objective,
                            "recipe": adjusted,
                            "features": columns,
                            "train_rows": len(fitting),
                            "calibration_rows": len(calibration),
                            "train_last_target": fitting.target_time.max().isoformat(),
                            "calibration_first_reference": calibration.reference_time.min().isoformat(),
                            "calibration_last_target": calibration.target_time.max().isoformat(),
                            "evaluation_first_reference": start.isoformat(),
                            "embargo_days": r["embargo_days"],
                            "fit_sha256": frame_hash(fitting[columns + ["target", "target_time"]]),
                            "model_sha256": sha256(model_dir / "model.joblib"),
                            "calibration": calibration_state,
                            "train_metrics": metrics(
                                fitting.target, predict_probability(estimator, fitting, columns)
                            ),
                            "calibration_sha256": frame_hash(
                                calibration[columns + ["target", "target_time"]]
                            ),
                            "reused": previous is not None,
                            "oos_metrics": metrics(observed.target, p[scoring_positions]),
                            "raw_oos_metrics": metrics(
                                observed.target, raw_eval[recipe["id"]][scoring_positions]
                            ),
                        }
                        write_new_json(model_dir / "fit.json", audit)
                        record(audit)
                    ensemble_ids = [
                        "logistic_cross",
                        "xgboost_cross",
                        "random_forest_cross",
                        "lightgbm_cross",
                        "catboost_cross",
                    ]
                    # The disjoint calibration tail is genuine OOS relative to every
                    # component fit. Stacker learns only this tail, before outer OOS.
                    zcal = np.column_stack([raw_cal[name] for name in ensemble_ids])
                    zeval = np.column_stack([raw_eval[name] for name in ensemble_ids])
                    consume()
                    stack = LogisticRegression(C=0.1, random_state=42).fit(zcal, cal.target)
                    execution["stacking"] = stack.predict_proba(zeval)[:, 1]
                    execution["equal_ensemble"] = np.mean(
                        [calibrated[name] for name in ensemble_ids], axis=0
                    )
                    for name in ["stacking", "equal_ensemble"]:
                        base[name] = execution[name].to_numpy()[scoring_positions]
                    for table in [base, execution]:
                        table["training_prior"] = float(train.target.mean())
                        table["constant_half"] = 0.5
                    priors = train.groupby("regime").target.agg(["sum", "count"])
                    priors["probability"] = (priors["sum"] + 20 * train.target.mean()) / (
                        priors["count"] + 20
                    )
                    base["regime_prior"] = observed.regime.map(priors.probability).fillna(
                        train.target.mean()
                    )
                    execution["regime_prior"] = forecast.regime.map(priors.probability).fillna(
                        train.target.mean()
                    )
                    stack_dir = directory / phase / str(fold) / key / "stacking"
                    stack_dir.mkdir(parents=True, exist_ok=False)
                    joblib.dump(stack, stack_dir / "model.joblib")
                    write_new_json(
                        stack_dir / "fit.json",
                        {
                            "components": ensemble_ids,
                            "training_first": cal.reference_time.min().isoformat(),
                            "training_last_target": cal.target_time.max().isoformat(),
                            "training_sha256": frame_hash(cal[["target", "target_time"]]),
                            "model_sha256": sha256(stack_dir / "model.joblib"),
                        },
                    )
                    predictions.setdefault(phase, {}).setdefault(key, []).append(base)
                    execution_predictions.setdefault(phase, {}).setdefault(key, []).append(
                        execution
                    )
                    full_mask = frame.index.isin(forecast.index)
                    for name in [rec["id"] for rec in protocol["recipes"]] + [
                        "stacking",
                        "equal_ensemble",
                        "training_prior",
                        "constant_half",
                        "regime_prior",
                    ]:
                        full_p = np.full(len(frame), 0.5)
                        full_p[full_mask] = execution[name]
                        for threshold in r["thresholds"]:
                            simulation = economic(frame, full_p, start, stop, costs, threshold)
                            sim_name = f"{name}__{threshold}"
                            sims.setdefault(phase, {}).setdefault(key, {}).setdefault(
                                sim_name, []
                            ).append((fold, simulation))
                            _save_sim(directory / phase / str(fold) / key / sim_name, simulation)
                            record(
                                {
                                    "phase": phase,
                                    "fold": fold,
                                    "asset": symbol,
                                    "objective": objective,
                                    "system": sim_name,
                                    "economic": simulation.metrics,
                                }
                            )
                    for benchmark in ["cash", "buy_and_hold"]:
                        benchmark_policy = RiskPolicy(name=benchmark, max_holding_bars=100000)
                        simulation = run_overlay(
                            frame,
                            np.full(len(frame), 1.0 if benchmark == "buy_and_hold" else 0.5),
                            start,
                            stop,
                            costs,
                            benchmark_policy,
                        ).simulation
                        sims[phase][key].setdefault(benchmark, []).append((fold, simulation))
                        _save_sim(directory / phase / str(fold) / key / benchmark, simulation)
                        record(
                            {
                                "phase": phase,
                                "fold": fold,
                                "asset": symbol,
                                "objective": objective,
                                "system": benchmark,
                                "economic": simulation.metrics,
                            }
                        )
                    # Magnitude and risk-adjusted return are regressions, not fake binary scores.
                    if objective == "net_h3":
                        risk_scale = train.risk_volatility * np.sqrt(3)
                        for regression_target in ["magnitude", "risk_adjusted"]:
                            regression_train = train.copy()
                            regression_train["regression_target"] = (
                                train.net_return_target
                                if regression_target == "magnitude"
                                else train.net_return_target / risk_scale
                            )
                            consume()
                            model = fit_model(
                                regression_train,
                                LOCAL + CROSS,
                                "regression_target",
                                {"kind": "xgboost"},
                                regression=True,
                            )
                            prediction = predict_value(model, observed, LOCAL + CROSS)
                            truth = (
                                observed.net_return_target
                                if regression_target == "magnitude"
                                else observed.net_return_target
                                / (observed.risk_volatility * np.sqrt(3))
                            )
                            regression_dir = directory / phase / str(fold) / key / regression_target
                            regression_dir.mkdir(parents=True, exist_ok=False)
                            joblib.dump(model, regression_dir / "model.joblib")
                            record(
                                {
                                    "phase": phase,
                                    "fold": fold,
                                    "asset": symbol,
                                    "objective": regression_target,
                                    "rows": len(observed),
                                    "mae": float(mean_absolute_error(truth, prediction)),
                                    "mse": float(mean_squared_error(truth, prediction)),
                                    "constant_train_mean_mae": float(
                                        mean_absolute_error(
                                            truth,
                                            np.repeat(
                                                regression_train.regression_target.mean(),
                                                len(truth),
                                            ),
                                        )
                                    ),
                                    "fit_sha256": frame_hash(
                                        regression_train[
                                            LOCAL + CROSS + ["regression_target", "target_time"]
                                        ]
                                    ),
                                    "train_last_target": train.target_time.max().isoformat(),
                                }
                            )
                            pd.DataFrame(
                                {
                                    "reference_time": observed.reference_time,
                                    "truth": truth,
                                    "prediction": prediction,
                                }
                            ).to_csv(regression_dir / "predictions.csv", index=False)
            if phase == "selection" and fold == 3:
                for key, pieces in predictions["selection"].items():
                    joined = pd.concat(pieces, ignore_index=True)
                    names = [rec["id"] for rec in protocol["recipes"]] + [
                        "stacking",
                        "equal_ensemble",
                    ]
                    ranking = {
                        name: metrics(joined.target, joined[name])
                        for name in names + ["training_prior", "constant_half"]
                    }
                    best = min(names, key=lambda name: ranking[name]["log_loss"])
                    alternatives = {
                        threshold: _aggregate_sims(sims["selection"][key][f"{best}__{threshold}"])
                        for threshold in r["thresholds"]
                    }
                    threshold = max(
                        alternatives, key=lambda t: alternatives[t]["metrics"]["net_return"]
                    )
                    econ = alternatives[threshold]["metrics"]
                    baseline_loss = min(
                        ranking[name]["log_loss"]
                        for name in [
                            "training_prior",
                            "constant_half",
                            "logistic_local",
                            "xgboost_local",
                        ]
                    )
                    fold_wins = sum(
                        metrics(part.target, part[best])["log_loss"]
                        < metrics(part.target, part.xgboost_local)["log_loss"]
                        for part in pieces
                    )
                    accepted = (
                        ranking[best]["log_loss"] < baseline_loss - 0.003
                        and (ranking[best]["roc_auc"] or 0) > 0.505
                        and fold_wins >= 2
                        and econ["net_return"] > 0
                        and econ["trade_count"] >= 30
                        and (econ["profit_factor"] or 0) > 1
                    )
                    frozen[key] = {
                        "recipe": best,
                        "threshold": threshold,
                        "eligible_before_joint_inference": accepted,
                        "selection_metrics": ranking[best],
                        "selection_economic": econ,
                        "fold_loss_wins_vs_xgboost": fold_wins,
                        "baseline_best_loss": baseline_loss,
                    }
                    rankings[key] = ranking
                write_new_json(
                    directory / "selection.json",
                    {
                        "systems": frozen,
                        "rankings": rankings,
                        "historical_description_opened": False,
                        "independent_future_opened": False,
                    },
                )
        inference, result = {}, {}
        for phase, by_key in predictions.items():
            result[phase] = {}
            for key, pieces in by_key.items():
                joined = pd.concat(pieces, ignore_index=True)
                joined.to_csv(directory / f"{phase}__{key}__predictions.csv", index=False)
                pd.concat(execution_predictions[phase][key], ignore_index=True).to_csv(
                    directory / f"{phase}__{key}__execution_forecasts.csv", index=False
                )
                names = [rec["id"] for rec in protocol["recipes"]] + [
                    "stacking",
                    "equal_ensemble",
                    "training_prior",
                    "constant_half",
                    "regime_prior",
                ]
                by_model = {name: metrics(joined.target, joined[name]) for name in names}
                regime = {
                    str(regime_name): {
                        "rows": len(group),
                        "models": {name: metrics(group.target, group[name]) for name in names},
                    }
                    for regime_name, group in joined.groupby("regime")
                }
                economies = {
                    name: _aggregate_sims(items) for name, items in sims[phase][key].items()
                }
                result[phase][key] = {
                    "predictive": by_model,
                    "by_regime": regime,
                    "economic": economies,
                }
                if phase == "selection":
                    probs = {name: joined[name].to_numpy() for name in names}
                    pairs = [
                        (name, baseline)
                        for name in names
                        if name not in ["training_prior", "constant_half"]
                        for baseline in [
                            "training_prior",
                            "constant_half",
                            "xgboost_local",
                            "logistic_local",
                        ]
                        if name != baseline
                    ]
                    inference[key] = paired_predictive(
                        joined.target.to_numpy(),
                        probs,
                        joined.fold.to_numpy(),
                        pairs,
                        {"block_lengths": [7, 30], "samples": 2000, "confidence": 0.95, "seed": 42},
                    )
                    best = frozen[key]["recipe"]
                    joint_pass = all(
                        inference[key]["blocks"][str(block)][best + "__minus__" + baseline][
                            "log_loss"
                        ]["upper"]
                        < 0
                        for block in [7, 30]
                        for baseline in [
                            "training_prior",
                            "constant_half",
                            "xgboost_local",
                            "logistic_local",
                        ]
                        if best != baseline
                    )
                    frozen[key]["joint_loss_pass"] = joint_pass
                    frozen[key]["accepted"] = (
                        frozen[key]["eligible_before_joint_inference"] and joint_pass
                    )
        write_new_json(directory / "uncertainty.json", inference)
        matched = {}
        for key, selected in frozen.items():
            symbol, objective = key.split("__")
            frame = targets[symbol][objective]
            names = ["xgboost_local", selected["recipe"]]
            allocations = {
                name: result["selection"][key]["economic"][f"{name}__{selected['threshold']}"][
                    "metrics"
                ]["mean_capital_exposure"]
                for name in names
            }
            common = min(allocations.values())
            joined = execution_predictions["historical_description"][key][0]
            comparisons = {}
            for name in dict.fromkeys(names):
                full = np.full(len(frame), 0.5)
                eligible = prediction_rows(frame, dates[-1], end)
                full[eligible.index] = joined[name].to_numpy()
                if common > 0:
                    scale = common / allocations[name]
                    scaled = replace(
                        policy(),
                        name="matched_development_budget",
                        capital_cap=0.25 * scale,
                        volatility_target_per_bar=0.01 * scale,
                        risk_budget=0.01 * scale,
                    )
                    matched_sim = run_overlay(
                        frame, full, dates[-1], end, costs, scaled, threshold=selected["threshold"]
                    ).simulation
                    _save_sim(directory / "matched_historical" / key / name, matched_sim)
                    comparisons[name] = matched_sim.metrics
                stress = replace(
                    costs,
                    commission_bps=costs.commission_bps * 2,
                    slippage_bps=costs.slippage_bps * 2,
                    spread_bps=costs.spread_bps * 2,
                )
                stress_sim = economic(frame, full, dates[-1], end, stress, selected["threshold"])
                _save_sim(directory / "stress_historical" / key / name, stress_sim)
                record(
                    {
                        "phase": "historical_description_stress",
                        "asset": symbol,
                        "objective": objective,
                        "system": name,
                        "economic": stress_sim.metrics,
                    }
                )
            matched[key] = {
                "train_only_target_mean_exposure": common,
                "selection_mean_exposures": allocations,
                "historical_actual": comparisons,
                "exact_equal_historical_exposure": False,
                "reason": "Risk budgets scaled using earlier folds only; realized exposure depends on opportunities",
            }
        # Future model artifacts are frozen only for the diagnostic selected recipe;
        # rejected recipes remain research-only and cannot be promoted by this module.
        from trading_ml.frontier_research.prospective import freeze_systems

        consume_count = freeze_systems(directory, targets, frozen, protocol, consume)
        if (
            _protected_hashes(settings) != protected
            or reservation_hashes(settings.root) != reserved
            or any(sha256(Path(p)) != value for p, value in cache_hashes.items())
            or any(
                sha256(Path(p)) != value["source_sha256"]
                for p, value in [
                    (str(settings.root / "data" / "raw.csv"), sources["BTCUSDT"]),
                    (
                        str(settings.root / "research_data" / "ETHUSDT_20200101_20261001.csv"),
                        sources["ETHUSDT"],
                    ),
                ]
            )
        ):
            raise DataError("Originales o reservas alterados")
        summary = {
            "run_id": run_id,
            "directory": str(directory.resolve()),
            "status": "completed",
            "actual_fits": fits,
            "new_fits": fits - prior_charge,
            "fit_count_is_budget_charge": True,
            "reused_models": reused_models,
            "records": records,
            "future_fits": consume_count,
            "selection": frozen,
            "periods": result,
            "costs": asdict(costs),
            "policy": asdict(policy()),
            "matched_risk_budget": matched,
            "historical_is_development": True,
            "prospective_evaluated": False,
            "live_trading": False,
            "promoted": False,
            "independent_advantage_confirmed": False,
            "conclusion": "INCONCLUSO; mantener referencias hasta test futuro independiente",
        }
        write_new_json(directory / "summary.json", summary)
        from trading_ml.frontier_research.report import write_report

        write_report(directory, summary)
        write_new_json(
            directory / "artifacts.json",
            {
                str(p.relative_to(directory)): sha256(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            },
        )
        return summary
    except Exception as exc:
        write_new_json(
            directory / "failure.json",
            {
                "exception": type(exc).__name__,
                "reason": str(exc),
                "actual_fits": fits,
                "budget_charge": fits,
                "records": records,
            },
        )
        raise
