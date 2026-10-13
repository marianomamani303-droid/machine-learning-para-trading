from __future__ import annotations

import json
import tomllib
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from trading_ml.config import Settings
from trading_ml.config.settings import Features
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import moving_block_indices
from trading_ml.features import build_features
from trading_ml.models import ModelBundle, load_model
from trading_ml.models.calibration import apply_calibration, fit_sigmoid
from trading_ml.momentum.config import MomentumConfig, ResearchCosts
from trading_ml.momentum.features import build_momentum_features, causal_regimes, momentum_columns
from trading_ml.momentum.report import write_report
from trading_ml.momentum.rules import momentum_signals
from trading_ml.momentum.simulator import (
    Simulation,
    opportunity_labels,
    performance,
    simulate_policy,
)
from trading_ml.preprocessing.quality import clean_ohlcv
from trading_ml.training.recipe import protect_reserved_labels


def _classifier(parameters: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        **parameters,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        n_jobs=1,
        random_state=seed,
    )


def fit_momentum_filter(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    cutoff: pd.Timestamp,
    parameters: dict[str, Any],
    seed: int,
    minimum_rows: int,
) -> tuple[XGBClassifier | None, dict[str, Any]]:
    """No validation/test statistic participates in fitting or threshold selection."""
    usable = labels.loc[(labels.target_time < cutoff) & (labels.reference_time < cutoff)]
    if len(usable):
        usable = usable.loc[frame.loc[usable.source_index, "mom_ready"].to_numpy()]
    training = frame.loc[usable.source_index, momentum_columns()].copy()
    audit = {
        "rows": len(usable),
        "positive_labels": int(usable.target.sum()),
        "last_target_time": usable.target_time.max().isoformat() if len(usable) else None,
        "cutoff": cutoff.isoformat(),
        "training_features_sha256": frame_hash(training),
        "training_labels_sha256": frame_hash(usable),
        "parameters": parameters,
        "threshold_selected": False,
        "future_or_test_used": False,
    }
    if len(usable) < minimum_rows or usable.target.nunique() != 2:
        return None, {**audit, "status": "abstain_insufficient_opportunities_or_single_class"}
    model = _classifier(parameters, seed)
    model.fit(training, usable.target.astype(int))
    return model, {**audit, "status": "fitted"}


def _probabilities(model: XGBClassifier | None, frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(frame))
    ready = frame.mom_ready.to_numpy()
    if model is not None and ready.any():
        result[ready] = model.predict_proba(frame.loc[ready, momentum_columns()])[:, 1]
    return result


def _reference_frame(raw: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    result = build_features(raw, Features(**bundle.metadata["config"]["features"]))
    result["reference_time"] = raw.reference_time
    result["regime"] = raw.regime
    horizon = bundle.metadata["contract"]["horizon"]
    future = result.close.shift(-horizon)
    result["target"] = (future > result.close).astype(float).where(future.notna())
    result["target_time"] = result.reference_time.shift(-horizon)
    return result


def _reference_probabilities(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    start: pd.Timestamp,
    seed: int,
    historical: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    ready = frame[bundle.metadata["features"]].notna().all(axis=1)
    p = np.zeros(len(frame))
    if historical:
        if start <= pd.Timestamp(bundle.metadata["selection_label_end"]):
            raise LeakageError("La comparación histórica invade selección del modelo actual")
        evaluation = ready & (frame.reference_time >= start)
        p[evaluation] = bundle.probabilities(frame.loc[evaluation])
        return p, {"status": "loaded_frozen_current_model", "version": bundle.metadata["version"]}
    train = frame.loc[ready & (frame.target_time < start)].copy()
    protect_reserved_labels(bundle.directory.parent.parent, train)
    calibration = None
    fitting = train
    if bundle.metadata.get("calibration"):
        boundary = start - pd.Timedelta(days=90)
        fitting = train.loc[train.target_time < boundary]
        calibration = train.loc[train.reference_time >= boundary]
    if len(fitting) < 120 or fitting.target.nunique() != 2:
        raise DataError("Referencia temporal insuficiente")
    model = _classifier(bundle.metadata["selected_parameters"], seed)
    columns = bundle.metadata["features"]
    model.fit(fitting[columns], fitting.target.astype(int))
    raw_p = model.predict_proba(frame.loc[ready, columns])[:, 1]
    calibration_audit = None
    if calibration is not None:
        calibration_audit = fit_sigmoid(
            model.predict_proba(calibration[columns])[:, 1], calibration.target.to_numpy(), seed
        )
        raw_p = apply_calibration(raw_p, calibration_audit)
    p[ready] = raw_p
    return p, {
        "status": "fixed_historical_parameters_temporal_refit",
        "rows": len(fitting),
        "last_target_time": fitting.target_time.max().isoformat(),
        "training_sha256": frame_hash(fitting),
        "calibration": calibration_audit,
        "calibration_last_target": calibration.target_time.max().isoformat()
        if calibration is not None
        else None,
        "parameters": bundle.metadata["selected_parameters"],
    }


def _aggregate(results: list[tuple[int, Simulation]]) -> dict[str, Any]:
    curves = []
    trades = []
    folds = []
    for number, result in results:
        curve = result.equity.iloc[1:].copy()
        curve["fold"] = number
        curves.append(curve)
        trades.append(result.trades)
        folds.append({"fold": number, **result.metrics})
    curve = pd.concat(curves, ignore_index=True)
    initial = results[0][1].equity.equity.iloc[0]
    curve["equity"] = initial * (1 + curve.net_return).cumprod()
    first = curve.iloc[[0]].copy()
    first["equity"] = initial
    first["net_return"] = 0.0
    first["turnover"] = 0.0
    first["regime"] = "initial"
    full = pd.concat([first, curve], ignore_index=True)
    full.attrs = results[0][1].equity.attrs.copy()
    trade_frame = pd.concat(trades, ignore_index=True)
    regimes = {}
    for regime in ["bull", "bear", "sideways", "unknown"]:
        part = curve.loc[curve.regime == regime]
        regimes[regime] = {
            "observations": len(part),
            "compounded_return": float((1 + part.net_return).prod() - 1),
            "mean_net_return": float(part.net_return.mean()) if len(part) else None,
            "turnover": float(part.turnover.sum()),
            "exposure_fraction": float(part.exposed.mean()) if len(part) else None,
        }
    quarters = {}
    for quarter, part in curve.groupby(
        curve.timestamp.dt.strftime("%Y") + "Q" + curve.timestamp.dt.quarter.astype(str)
    ):
        quarters[quarter] = {
            "observations": len(part),
            "net_return": float((1 + part.net_return).prod() - 1),
            "turnover": float(part.turnover.sum()),
        }
    return {
        "metrics": performance(full, trade_frame),
        "folds": folds,
        "positive_fold_fraction": float(np.mean([x["net_return"] > 0 for x in folds])),
        "regimes": regimes,
        "calendar_quarters": quarters,
        "_curve": curve,
    }


def _evidence(systems: dict[str, Any], protocol: dict[str, Any]) -> tuple[dict, dict]:
    candidates = [name for name in systems if name.startswith("momentum_")]
    boot = protocol["bootstrap"]
    confidence = boot["confidence"]
    family = len(candidates) * 2 * len(boot["block_lengths"])
    tail = (1 - confidence) / (2 * family)
    groups = systems["cash"]["_curve"].fold.to_numpy()
    evidence: dict[str, Any] = {}
    decisions = {}
    for name in candidates:
        evidence[name] = {}
        candidate = systems[name]["_curve"]
        failed = []
        if systems[name]["metrics"]["trade_count"] < protocol["acceptance"]["minimum_trades"]:
            failed.append("too_few_trades")
        if (
            systems[name]["positive_fold_fraction"]
            < protocol["acceptance"]["required_positive_fold_fraction"]
        ):
            failed.append("unstable_across_windows")
        if systems[name]["metrics"]["net_return"] <= 0:
            failed.append("nonpositive_net_return")
        for reference in ["cash", "buy_hold"]:
            base = systems[reference]["_curve"]
            if not candidate.timestamp.equals(base.timestamp):
                raise DataError("Comparaciones de retorno desalineadas")
            excess = np.log1p(candidate.net_return.to_numpy()) - np.log1p(
                base.net_return.to_numpy()
            )
            evidence[name][reference] = {}
            for block in boot["block_lengths"]:
                if any((groups == g).sum() < block for g in np.unique(groups)):
                    evidence[name][reference][str(block)] = {
                        "status": "insufficient_window_for_block"
                    }
                    failed.append(f"insufficient_block_{block}")
                    continue
                indices = moving_block_indices(
                    groups, boot["samples"], block, protocol["research"]["seed"]
                )
                draws = excess[indices].mean(axis=1)
                lower, upper = np.quantile(draws, [tail, 1 - tail])
                evidence[name][reference][str(block)] = {
                    "mean_excess_log_return": float(excess.mean()),
                    "lower": float(lower),
                    "upper": float(upper),
                    "confidence_after_bonferroni": 1 - 2 * tail,
                    "samples": boot["samples"],
                    "family_comparisons": family,
                }
                if lower <= 0:
                    failed.append(f"no_positive_adjusted_lower_bound_vs_{reference}_block_{block}")
        decisions[name] = {
            "historical_evidence_passed": not failed,
            "failed_checks": sorted(set(failed)),
            "independent_advantage_confirmed": False,
            "selection_or_promotion": False,
        }
    return evidence, decisions


def _validate_protocol(protocol: dict[str, Any]) -> None:
    research = protocol["research"]
    horizons = research["lookbacks"]
    dates = pd.to_datetime(research["boundaries"], utc=True)
    if not 1 <= len(horizons) <= 4 or len(set(horizons)) != len(horizons):
        raise DataError("Momentum admite hasta cuatro horizontes fijos distintos")
    if (
        len(dates) < 3
        or len(dates) > 9
        or not dates.is_monotonic_increasing
        or dates.duplicated().any()
    ):
        raise DataError("Se requieren 2–8 folds consecutivos")
    if pd.Timestamp(research["historical_end"], tz="UTC") <= dates[-1]:
        raise DataError("Reserva histórica inválida")
    if not 0.5 <= research["ml_threshold"] <= 1 or research["min_train_opportunities"] < 2:
        raise DataError("Umbral o muestra de entrenamiento inválidos")
    refs = research["reference_versions"]
    if len(refs) > 3 or len(refs) != len(set(refs)):
        raise DataError("Hasta tres referencias distintas")
    expected_fits = len(horizons) * (len(dates) + 1) + len(refs) * (len(dates) - 1)
    if expected_fits > research["max_fits"] or research["max_fits"] > 80:
        raise DataError("Presupuesto de ajustes excedido")
    if (
        not 1 <= protocol["xgboost"]["n_estimators"] <= 160
        or not 1 <= protocol["xgboost"]["max_depth"] <= 4
    ):
        raise DataError("XGBoost Momentum fuera del presupuesto fijo")
    if not 1 <= research["prospective_days"] <= 730 or protocol["stress"]["cost_multiplier"] < 1:
        raise DataError("Reserva prospectiva o stress inválidos")
    acceptance = protocol["acceptance"]
    if (
        acceptance["minimum_trades"] < 1
        or not 0 <= acceptance["required_positive_fold_fraction"] <= 1
    ):
        raise DataError("Criterios de aceptación inválidos")
    boot = protocol["bootstrap"]
    if (
        not 100 <= boot["samples"] <= 5000
        or not 0 < boot["confidence"] < 1
        or not boot["block_lengths"]
    ):
        raise DataError("Bootstrap inválido")
    if any(not isinstance(x, int) or x < 1 for x in boot["block_lengths"]):
        raise DataError("Bloques inválidos")
    for horizon in horizons:
        MomentumConfig(lookback=horizon, **protocol["rules"]).validate()
    ResearchCosts(**protocol["costs"]).validate()


def run_momentum_study(
    settings: Settings, protocol_path: Path, source: Path | None = None
) -> dict[str, Any]:
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    research = protocol["research"]
    source = source or settings.root / "data" / "raw.csv"
    manifest = read_json(sidecar(source))
    if manifest.get("kind") != "raw" or manifest.get("sha256") != sha256(source):
        raise DataError("OHLCV crudo no coincide con su manifiesto")
    if any(manifest["market"][k] != getattr(settings.market, k) for k in ["symbol", "interval"]):
        raise DataError("Mercado/intervalo incompatible")
    raw, quality = clean_ohlcv(pd.read_csv(source), settings.delta)
    historical_end = pd.Timestamp(research["historical_end"], tz="UTC")
    raw["reference_time"] = raw.timestamp + settings.delta
    if (
        raw.reference_time.iloc[-1] < historical_end
        or (raw.reference_time > pd.Timestamp.now(tz="UTC")).any()
    ):
        raise DataError("Cobertura histórica incompleta o velas futuras/abiertas")
    # Extra future rows cannot enter any decision or fit in this research run.
    raw = raw.loc[raw.reference_time <= historical_end].reset_index(drop=True)
    raw["regime"] = causal_regimes(raw)
    bundles = [load_model(settings.root, version) for version in research["reference_versions"]]
    for bundle in bundles:
        if any(
            bundle.metadata["contract"][k] != getattr(settings.market, k)
            for k in ["symbol", "interval"]
        ):
            raise DataError("Referencia de otro mercado o intervalo")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = settings.root / "momentum_studies" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "protocol.toml").write_bytes(protocol_path.read_bytes())
    raw.to_csv(destination / "ohlcv.csv", index=False)
    # The first future UTC close after execution date is unobserved by construction.
    future_start = max(
        pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1),
        historical_end + settings.delta,
    )
    prospective = {
        "status": "pending_future_data",
        "start": future_start.isoformat(),
        "end": (future_start + pd.Timedelta(days=research["prospective_days"])).isoformat(),
        "no_training_or_selection": True,
        "historical_test_already_observed": True,
    }
    costs = ResearchCosts(**protocol["costs"])
    multiplier = protocol["stress"]["cost_multiplier"]
    stress = replace(
        costs,
        commission_bps=costs.commission_bps * multiplier,
        slippage_bps=costs.slippage_bps * multiplier,
        spread_bps=costs.spread_bps * multiplier,
        annual_holding_bps=costs.annual_holding_bps * multiplier,
    )
    stress.validate()
    plan = {
        "run_id": run_id,
        "protocol": protocol,
        "protocol_sha256": sha256(protocol_path),
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "manifest": manifest,
        "snapshot_sha256": sha256(destination / "ohlcv.csv"),
        "quality": quality,
        "code_provenance": code_provenance(),
        "environment": environment(),
        "references": [
            {
                "version": b.metadata["version"],
                "model_sha256": b.metadata["model_sha256"],
                "selection_label_end": b.metadata["selection_label_end"],
            }
            for b in bundles
        ],
        "prospective": prospective,
        "selection": "none_all_fixed_horizons_reported",
        "live_trading": False,
        "promotion": False,
    }
    write_new_json(destination / "plan.json", plan)

    def record(value: dict[str, Any]) -> None:
        with (destination / "experiments.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    fits = 0
    try:
        horizon_data = {}
        for horizon in research["lookbacks"]:
            cfg = MomentumConfig(lookback=horizon, **protocol["rules"])
            frame = build_momentum_features(raw, cfg)
            signals = momentum_signals(frame, cfg)
            labels = opportunity_labels(frame, signals, cfg.max_holding_bars, costs)
            frame.to_csv(destination / f"features_L{horizon}.csv", index=False)
            signals.to_csv(destination / f"signals_L{horizon}.csv", index=False)
            labels.to_csv(destination / f"labels_L{horizon}.csv", index=False)
            horizon_data[horizon] = (cfg, frame, signals, labels)
        reference_data = {b.metadata["version"]: _reference_frame(raw, b) for b in bundles}
        dates = pd.to_datetime(research["boundaries"], utc=True)
        windows = [
            ("development", i + 1, a, b, False)
            for i, (a, b) in enumerate(zip(dates[:-1], dates[1:], strict=True))
        ]
        windows.append(("observed_historical", 1, dates[-1], historical_end, True))
        results: dict[str, dict[str, list[tuple[int, Simulation]]]] = {}

        def save(
            period: str, fold: int, name: str, result: Simulation, audit: dict[str, Any]
        ) -> None:
            folder = destination / period / f"fold_{fold}" / name
            folder.mkdir(parents=True, exist_ok=False)
            result.equity.to_csv(folder / "equity.csv", index=False)
            result.trades.to_csv(folder / "trades.csv", index=False)
            experiment = {
                "period": period,
                "fold": fold,
                "system": name,
                "metrics": result.metrics,
                "training": audit,
                "status": "completed",
                "negative_net_return": result.metrics["net_return"] < 0,
            }
            write_new_json(folder / "experiment.json", experiment)
            record(experiment)
            results.setdefault(period, {}).setdefault(name, []).append((fold, result))

        for period, fold, start, end, historical in windows:
            baseline_signal = pd.DataFrame({"entry": True, "exit": False}, index=raw.index)
            for scenario, scenario_costs in [("", costs), ("_stress", stress)]:
                for name, signals in [
                    ("buy_hold", baseline_signal),
                    ("cash", baseline_signal.assign(entry=False)),
                ]:
                    result = simulate_policy(raw, signals, start, end, scenario_costs, len(raw) + 1)
                    save(period + scenario, fold, name, result, {"status": "no_model"})
            for horizon, (cfg, frame, signals, labels) in horizon_data.items():
                train_labels = labels.loc[labels.target_time < start]
                protect_reserved_labels(settings.root, train_labels)
                model, audit = fit_momentum_filter(
                    frame,
                    labels,
                    start,
                    protocol["xgboost"],
                    research["seed"],
                    research["min_train_opportunities"],
                )
                fits += model is not None
                p = _probabilities(model, frame)
                opportunities = (
                    signals.entry & (frame.reference_time >= start) & (frame.reference_time < end)
                )
                audit["evaluation_opportunities"] = int(opportunities.sum())
                audit["accepted_opportunities"] = int(
                    (opportunities & (p >= research["ml_threshold"])).sum()
                )
                observed_labels = labels.loc[
                    (labels.reference_time >= start) & (labels.target_time < end)
                ]
                audit["evaluation_label_metrics"] = (
                    classification_metrics(
                        observed_labels.target.to_numpy(),
                        p[observed_labels.source_index.to_numpy()],
                    )
                    if model is not None and len(observed_labels)
                    else None
                )
                audit["evaluation_metrics_used_for_selection"] = False
                if model is not None:
                    folder = destination / period / f"fold_{fold}" / f"model_L{horizon}"
                    folder.mkdir(parents=True, exist_ok=False)
                    model.save_model(folder / "model.ubj")
                    write_new_json(
                        folder / "fit.json", {**audit, "model_sha256": sha256(folder / "model.ubj")}
                    )
                predictions = frame[["timestamp", "reference_time", "mom_ready"]].copy()
                predictions["probability_profitable_trade"] = p
                predictions = predictions.loc[
                    (predictions.reference_time >= start) & (predictions.reference_time < end)
                ]
                prediction_path = (
                    destination / period / f"fold_{fold}" / f"predictions_L{horizon}.csv"
                )
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(prediction_path, index=False)
                for scenario, scenario_costs in [("", costs), ("_stress", stress)]:
                    for variant, probability, details in [
                        ("rules", None, {"status": "no_model"}),
                        ("xgb", p, audit),
                    ]:
                        result = simulate_policy(
                            frame,
                            signals,
                            start,
                            end,
                            scenario_costs,
                            cfg.max_holding_bars,
                            probabilities=probability,
                            threshold=research["ml_threshold"],
                        )
                        save(
                            period + scenario,
                            fold,
                            f"momentum_L{horizon}_{variant}",
                            result,
                            details,
                        )
            for bundle in bundles:
                frame = reference_data[bundle.metadata["version"]]
                p, audit = _reference_probabilities(
                    frame, bundle, start, research["seed"], historical
                )
                fits += not historical
                signals = pd.DataFrame({"entry": True, "exit": False}, index=frame.index)
                for scenario, scenario_costs in [("", costs), ("_stress", stress)]:
                    result = simulate_policy(
                        frame,
                        signals,
                        start,
                        end,
                        scenario_costs,
                        bundle.metadata["contract"]["horizon"],
                        probabilities=p,
                        threshold=bundle.metadata["config"]["backtest"]["threshold"],
                        reenter_on_exit=True,
                    )
                    save(
                        period + scenario,
                        fold,
                        "current_" + bundle.metadata["version"],
                        result,
                        audit,
                    )
        # All recipes are frozen for a genuinely future evaluation, irrespective of performance.
        frozen = {}
        for horizon, (cfg, frame, _signals, labels) in horizon_data.items():
            protect_reserved_labels(settings.root, labels.loc[labels.target_time < historical_end])
            model, audit = fit_momentum_filter(
                frame,
                labels,
                historical_end,
                protocol["xgboost"],
                research["seed"],
                research["min_train_opportunities"],
            )
            fits += model is not None
            folder = destination / "frozen" / f"L{horizon}"
            folder.mkdir(parents=True, exist_ok=False)
            if model is not None:
                model.save_model(folder / "model.ubj")
                audit["model_sha256"] = sha256(folder / "model.ubj")
            write_new_json(folder / "fit.json", audit)
            frozen[str(horizon)] = {"config": asdict(cfg), "fit": audit}
            record(
                {
                    "period": "freeze_for_future",
                    "system": f"momentum_L{horizon}_xgb",
                    "training": audit,
                }
            )
        if fits > research["max_fits"]:
            raise DataError("Presupuesto de ajustes excedido")
        periods = {}
        for period, systems in results.items():
            aggregate = {name: _aggregate(items) for name, items in systems.items()}
            evidence, decisions = _evidence(aggregate, protocol)
            for name, item in aggregate.items():
                item.pop("_curve").to_csv(
                    destination / f"{period}__{name}__returns.csv", index=False
                )
            periods[period] = {
                "systems": aggregate,
                "bootstrap_excess_log_returns": evidence,
                "decisions": decisions,
            }
        summary = {
            "run_id": run_id,
            "directory": str(destination.resolve()),
            "status": "completed",
            "plan_sha256": sha256(destination / "plan.json"),
            "actual_xgboost_fits": fits,
            "experiment_records": sum(
                1 for _ in (destination / "experiments.jsonl").open(encoding="utf-8")
            ),
            "costs": asdict(costs),
            "protocol": protocol,
            "stress_costs": asdict(stress),
            "periods": periods,
            "prospective": prospective,
            "frozen": frozen,
            "independent_advantage_confirmed": False,
            "promoted": False,
            "live_trading": False,
        }
        write_new_json(destination / "summary.json", summary)
        write_report(destination / "REPORT.md", summary)
        write_new_json(
            destination / "artifacts.json",
            {
                str(p.relative_to(destination)): sha256(p)
                for p in sorted(destination.rglob("*"))
                if p.is_file()
            },
        )
        return summary
    except Exception as exc:
        failure = {
            "status": "failed",
            "exception": type(exc).__name__,
            "reason": str(exc),
            "actual_xgboost_fits": fits,
            "run_id": run_id,
        }
        record(failure)
        write_new_json(destination / "failure.json", failure)
        raise
