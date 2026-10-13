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

from trading_ml.advanced_research.audit import audit_defensive_trades
from trading_ml.advanced_research.data import recover_microstructure
from trading_ml.advanced_research.features import (
    META,
    build_research_features,
    family_columns,
)
from trading_ml.advanced_research.meta import causal_opportunity_labels, fit_meta, gates
from trading_ml.advanced_research.models import Fitted, fit_recipe, load_fitted
from trading_ml.advanced_research.report import write_report
from trading_ml.advanced_research.statistics import drift_diagnostic, paired_predictive
from trading_ml.advanced_research.targets import build_targets
from trading_ml.common import code_provenance, environment, read_json, sha256, write_new_json
from trading_ml.config import Settings
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.mean_reversion.study import _protected_hashes
from trading_ml.models import load_model
from trading_ml.momentum.config import ResearchCosts
from trading_ml.momentum.study import _reference_frame, _reference_probabilities
from trading_ml.risk_overlay.config import RiskPolicy
from trading_ml.risk_overlay.simulator import RiskSimulation, run_overlay
from trading_ml.risk_overlay.study import _check_reserved_end, aggregate, reservation_hashes
from trading_ml.training.recipe import protect_reserved_labels


def _validate(protocol: dict[str, Any]) -> None:
    research, budget = protocol["research"], protocol["budget"]
    dates = pd.to_datetime(research["boundaries"], utc=True)
    if not 3 <= len(dates) <= 9 or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise DataError("Ventanas de investigación avanzada inválidas")
    recipes = protocol["recipes"]
    ids = [r["id"] for r in recipes]
    if not 2 <= len(recipes) <= budget["max_recipes"] <= 24 or len(set(ids)) != len(ids):
        raise DataError("Presupuesto o IDs de recetas inválidos")
    if "original_refit" not in ids:
        raise DataError("Falta el control original con train equivalente")
    if not 1 <= budget["max_fits"] <= 240 or budget["cpu_threads"] != 1:
        raise DataError("Presupuesto avanzado inválido")
    if budget["max_trees"] > 200:
        raise DataError("Presupuesto de árboles inválido")
    for kind in ["xgboost_reference", "xgboost_regularized"]:
        if protocol[kind]["n_estimators"] > budget["max_trees"]:
            raise DataError("Receta excede árboles permitidos")
    for recipe in recipes:
        family_columns(recipe["family"])
        if recipe["target"] not in {"direction_h1", "net_h3", "net_h5"}:
            raise DataError("Objetivo avanzado inválido")
        if recipe.get("reuse_model") and ids.index(recipe["reuse_model"]) >= ids.index(
            recipe["id"]
        ):
            raise DataError("Dependencia de receta fuera de orden")
    if pd.Timestamp(research["observed_end"], tz="UTC") <= dates[-1]:
        raise DataError("Período auditado inválido")
    if research["minimum_train_rows"] < 120 or research["calibration_days"] < 30:
        raise DataError("Train/calibración insuficientes")
    RiskPolicy(**protocol["policy"]).validate()
    ResearchCosts(**protocol["costs"]).validate()


def _economic_stats(result: dict[str, Any]) -> dict[str, Any]:
    return result["metrics"]


def run_advanced_study(
    settings: Settings,
    protocol_path: Path,
    source: Path | None = None,
    cache_from: Path | None = None,
) -> dict[str, Any]:
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    _validate(protocol)
    research = protocol["research"]
    costs = ResearchCosts(**protocol["costs"])
    policy = RiskPolicy(**protocol["policy"])
    dates = pd.to_datetime(research["boundaries"], utc=True)
    observed_end = pd.Timestamp(research["observed_end"], tz="UTC")
    _check_reserved_end(settings.root, observed_end)
    source = source or settings.root / "data" / "raw.csv"
    expanded, provenance = recover_microstructure(settings, source)
    if expanded.reference_time.iloc[-1] < observed_end:
        raise DataError("No está completo el intervalo observado")
    expanded = expanded.loc[expanded.reference_time <= observed_end].reset_index(drop=True)
    frame = build_research_features(expanded)
    targets = build_targets(frame, costs)
    risk_directory = Path(research["reference_risk_study"])
    risk_summary = read_json(risk_directory / "summary.json")
    risk_manifest = read_json(risk_directory / "artifacts.json")
    if risk_summary["diagnosis"]["reference"] != research["reference_version"] or risk_summary[
        "costs"
    ] != asdict(costs):
        raise DataError("La comparación defensiva no coincide en modelo o costos")
    reference_policy = next(
        p for p in risk_summary["protocol"]["policies"] if p["name"] == policy.name
    )
    if RiskPolicy(**reference_policy) != policy:
        raise DataError("Cambió la política defensiva de la comparación")
    reference = load_model(settings.root, research["reference_version"])
    if dates[-1] <= pd.Timestamp(reference.metadata["selection_label_end"]):
        raise DataError("La referencia registrada no es fuera de muestra en el período auditado")
    protected = _protected_hashes(settings)
    reserved = reservation_hashes(settings.root)
    previous_fits = 0
    cache_hashes = {}
    if cache_from is not None:
        old_plan = read_json(cache_from / "plan.json")
        if (
            old_plan["protocol_sha256"] != sha256(protocol_path)
            or old_plan["provenance"]["source_sha256"] != provenance["source_sha256"]
        ):
            raise DataError("La caché no corresponde al mismo protocolo y dataset")
        previous_fits = read_json(cache_from / "failure.json")["actual_fits"]
        cache_hashes = {str(p.resolve()): sha256(p) for p in cache_from.rglob("*") if p.is_file()}
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = settings.root / "advanced_studies" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    expanded.to_csv(directory / "extended_ohlcv.csv", index=False)
    frame.to_csv(directory / "causal_features.csv", index=False)
    (directory / "protocol.toml").write_bytes(protocol_path.read_bytes())
    audit = audit_defensive_trades(
        frame,
        risk_directory
        / "observed_historical"
        / "fold_1"
        / (research["reference_version"] + "__combined_25")
        / "trades.csv",
        directory,
    )
    plan = {
        "run_id": run_id,
        "protocol": protocol,
        "protocol_sha256": sha256(protocol_path),
        "provenance": provenance,
        "snapshot_sha256": sha256(directory / "extended_ohlcv.csv"),
        "reference_risk_summary_sha256": sha256(risk_directory / "summary.json"),
        "protected_models": protected,
        "reserved_hashes": reserved,
        "code_provenance": code_provenance(),
        "environment": environment(),
        "audited_19_are_hypothesis_generation_only": True,
        "selection_end": dates[-1].isoformat(),
        "observed_history_is_development": True,
        "prospective_evaluated": False,
        "automatic_promotion": False,
        "cache_from": str(cache_from.resolve()) if cache_from is not None else None,
        "prior_fit_budget_charged": previous_fits,
        "cache_hashes": cache_hashes,
        "hypotheses": [
            {
                "id": r["id"],
                "family": r["family"],
                "target": r["target"],
                "purpose": "independent family/regularization/window/target ablation; no audited-trade threshold fitting",
            }
            for r in protocol["recipes"]
        ],
    }
    write_new_json(directory / "plan.json", plan)
    fits = previous_fits
    reused_models = 0
    records = 0

    def consume_fit() -> None:
        nonlocal fits
        if fits >= protocol["budget"]["max_fits"]:
            raise DataError("Presupuesto de ajustes avanzado agotado")
        fits += 1

    def record(value: dict[str, Any]) -> None:
        nonlocal records
        with (directory / "experiments.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        records += 1

    simulations: dict[str, dict[str, list[tuple[int, Any]]]] = {}
    scores: dict[str, dict[str, list[pd.DataFrame]]] = {}
    folds: dict[str, list[dict[str, Any]]] = {}
    meta_pool = pd.DataFrame(
        columns=[
            "source_index",
            "reference_time",
            "target_time",
            "target",
            "unit_net_return",
            "base_probability",
            "status",
        ]
    )
    for col in ["reference_time", "target_time"]:
        meta_pool[col] = pd.to_datetime(meta_pool[col], utc=True)

    def save_sim(
        phase: str, fold: int, name: str, result: RiskSimulation, metadata: dict[str, Any]
    ) -> None:
        destination = directory / phase / f"fold_{fold}" / name
        destination.mkdir(parents=True, exist_ok=False)
        result.simulation.equity.to_csv(destination / "equity.csv", index=False)
        result.simulation.trades.to_csv(destination / "trades.csv", index=False)
        result.events.to_csv(destination / "events.csv", index=False)
        data = {
            "phase": phase,
            "fold": fold,
            "experiment": name,
            "economic_metrics": result.simulation.metrics,
            "metadata": metadata,
            "negative_net_return": result.simulation.metrics["net_return"] < 0,
        }
        write_new_json(destination / "experiment.json", data)
        record(data)
        simulations.setdefault(phase, {}).setdefault(name, []).append((fold, result.simulation))

    def score_model(
        phase: str,
        fold: int,
        name: str,
        target_key: str,
        probability: np.ndarray,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fitted: Fitted | None,
    ) -> None:
        dataset = targets[target_key]
        evaluation = dataset.loc[
            dataset.research_ready
            & (dataset.reference_time >= start)
            & (dataset.reference_time < end)
            & (dataset.target_time < end)
            & dataset.target.notna()
        ]
        output = evaluation[
            ["timestamp", "reference_time", "target", "target_time", "regime"]
        ].copy()
        output["probability"] = probability[evaluation.index]
        output["fold"] = fold
        metrics = classification_metrics(output.target.to_numpy(), output.probability.to_numpy())
        magnitude = None
        if fitted is not None and fitted.regression:
            predicted = fitted.expected_return(evaluation)
            actual = evaluation.net_return_target.to_numpy()
            output["predicted_net_return"] = predicted
            output["actual_net_return"] = actual
            magnitude = {
                "mae": float(np.abs(predicted - actual).mean()),
                "rmse": float(np.sqrt(((predicted - actual) ** 2).mean())),
                "mean_predicted": float(predicted.mean()),
                "mean_actual": float(actual.mean()),
            }
        scores.setdefault(phase, {}).setdefault(target_key, []).append(output.assign(model=name))
        if phase == "selection_development":
            folds.setdefault(name, []).append(
                {"fold": fold, "target": target_key, "metrics": metrics}
            )
        destination = directory / phase / f"fold_{fold}" / "predictive"
        destination.mkdir(parents=True, exist_ok=True)
        output.to_csv(destination / (name + ".csv"), index=False)
        audit_data = (
            fitted.audit if fitted is not None else {"statistical_or_frozen_baseline": True}
        )
        if fitted is not None:
            used = dataset.loc[dataset.research_ready & (dataset.target_time < start)]
            parameters = fitted.audit["parameters"]
            if parameters.get("rolling"):
                used = used.loc[
                    used.reference_time >= start - pd.Timedelta(days=research["rolling_days"])
                ]
            if parameters.get("reserve_calibration"):
                used = used.loc[
                    used.target_time < start - pd.Timedelta(days=research["calibration_days"])
                ]
            audit_data = {
                **audit_data,
                "drift": drift_diagnostic(
                    used, evaluation, ["volatility", "flow_buy_share", "adv_momentum_60"]
                ),
            }
        record(
            {
                "phase": phase,
                "fold": fold,
                "experiment": name,
                "target": target_key,
                "predictive_metrics": metrics,
                "magnitude_metrics": magnitude,
                "fit": audit_data,
            }
        )

    def old_probabilities(
        phase: str, number: int, start: pd.Timestamp, stop: pd.Timestamp
    ) -> np.ndarray:
        p = np.zeros(len(frame))
        if phase == "selection_development":
            relative = f"development/fold_{number}/{research['reference_version']}_forecasts.csv"
            path = risk_directory / relative
            if sha256(path) != risk_manifest[str(Path(relative))]:
                raise DataError("Predicciones OOS históricas de referencia alteradas")
            old = pd.read_csv(path)
            old["reference_time"] = pd.to_datetime(old.reference_time, utc=True)
            indexed = old.set_index("reference_time").probability_up
            mask = (frame.reference_time >= start) & (frame.reference_time < stop)
            p[mask] = frame.loc[mask, "reference_time"].map(indexed).to_numpy()
            if not np.isfinite(p).all():
                raise DataError("Predicciones defensivas desalineadas")
        else:
            mask = (frame.reference_time >= start) & (frame.reference_time < stop)
            p[mask] = reference.probabilities(frame.loc[mask])
        return p

    try:
        # Two seed windows provide genuinely prequential past meta opportunities.
        seed_dates = pd.to_datetime(research["meta_seed_boundaries"], utc=True)
        original_frame = _reference_frame(frame, reference)
        for i, (start, stop) in enumerate(zip(seed_dates[:-1], seed_dates[1:], strict=True), 1):
            consume_fit()
            p, fit_audit = _reference_probabilities(
                original_frame, reference, start, research["seed"], False
            )
            p[(frame.reference_time < start) | (frame.reference_time >= stop)] = 0.0
            labels = causal_opportunity_labels(frame, p, start, stop, costs, policy)
            meta_pool = pd.concat([meta_pool, labels], ignore_index=True)
            record(
                {
                    "phase": "prequential_meta_seed",
                    "fold": i,
                    "fit": fit_audit,
                    "opportunities": len(labels),
                }
            )
        selected = None
        selection = None
        selection_predictive = None
        selection_economic = None
        matched_policies = None
        windows = [
            ("selection_development", i + 1, a, b)
            for i, (a, b) in enumerate(zip(dates[:-1], dates[1:], strict=True))
        ]
        windows.append(("audited_observed_development", 1, dates[-1], observed_end))
        for phase, number, start, stop in windows:
            base = old_probabilities(phase, number, start, stop)
            base_sim = run_overlay(frame, base, start, stop, costs, policy)
            if phase == "audited_observed_development":
                actual = base_sim.simulation.metrics
                recorded = risk_summary["diagnosis"]["after"]
                if actual["trade_count"] != 19 or not all(
                    np.isclose(actual[key], recorded[key], rtol=1e-8, atol=1e-10)
                    for key in ["net_return", "max_drawdown", "mean_capital_exposure"]
                ):
                    raise DataError(
                        "La referencia no reproduce las 19 operaciones defensivas registradas"
                    )
            save_sim(
                phase,
                number,
                "defensive_reference",
                base_sim,
                {
                    "predictions": "frozen_model_or_previously_recorded_oos",
                    "risk_policy": asdict(policy),
                },
            )
            score_model(
                phase, number, "defensive_reference", "direction_h1", base, start, stop, None
            )
            for name, probability in [
                ("cash", np.zeros(len(frame))),
                ("buy_hold", np.ones(len(frame))),
            ]:
                save_sim(
                    phase,
                    number,
                    name,
                    run_overlay(
                        frame, probability, start, stop, costs, RiskPolicy(), len(frame) + 1
                    ),
                    {"baseline": True},
                )
            fitted_models = {}
            recipe_probabilities = {}
            for recipe in protocol["recipes"]:
                name, key = recipe["id"], recipe["target"]
                dataset = targets[key]
                training = dataset.loc[dataset.research_ready & (dataset.target_time < start)]
                protect_reserved_labels(settings.root, training)
                cached = (
                    cache_from / phase / f"fold_{number}" / "models" / name
                    if cache_from is not None
                    else None
                )
                if cached is not None and (cached / "fit.json").exists():
                    fitted = load_fitted(cached, dataset, start, recipe, protocol)
                    reused_models += 1
                else:
                    fitted = fit_recipe(
                        dataset,
                        family_columns(recipe["family"]),
                        start,
                        recipe,
                        protocol,
                        consume_fit,
                        fitted_models.get(recipe.get("reuse_model")),
                    )
                fitted_models[name] = fitted
                model_destination = directory / phase / f"fold_{number}" / "models" / name
                fitted.save(model_destination)
                mask = (
                    frame.research_ready
                    & (frame.reference_time >= start)
                    & (frame.reference_time < stop)
                )
                probability = np.zeros(len(frame))
                probability[mask] = fitted.probabilities(frame.loc[mask])
                recipe_probabilities[name] = probability
                score_model(phase, number, name, key, probability, start, stop, fitted)
                # Directional models share the original defensive execution/risk budget.
                # Net H3/H5 get their own same-horizon controls, never a cross-target AUC claim.
                strategy_policy = (
                    policy
                    if key == "direction_h1"
                    else replace(
                        policy,
                        confirmations=1,
                        persistent=False,
                        cooldown_bars=0,
                        max_holding_bars=int(key[-1]),
                    )
                )
                save_sim(
                    phase,
                    number,
                    name,
                    run_overlay(frame, probability, start, stop, costs, strategy_policy),
                    {"target": key, "risk_policy": asdict(strategy_policy)},
                )
                if name == "flow_momentum":
                    expected = np.zeros(len(frame))
                    expected[mask] = fitted.expected_return(frame.loc[mask])
                    save_sim(
                        phase,
                        number,
                        "direction_economic_gate",
                        run_overlay(
                            frame, probability, start, stop, costs, policy, entry_gate=expected > 0
                        ),
                        {"expected_payoff_estimated_in_train_only": True},
                    )
                    save_sim(
                        phase,
                        number,
                        "confidence_060",
                        run_overlay(frame, probability, start, stop, costs, policy, threshold=0.60),
                        {"threshold_fixed": 0.60},
                    )
            # Logistic, prevalence, constant 0.5, momentum-sign baselines per target.
            for key, dataset in targets.items():
                logistic_recipe = {
                    "id": "logistic_" + key,
                    "kind": "logistic",
                    "family": "flow_momentum",
                    "target": key,
                }
                cached = (
                    cache_from / phase / f"fold_{number}" / "models" / logistic_recipe["id"]
                    if cache_from is not None
                    else None
                )
                if cached is not None and (cached / "fit.json").exists():
                    fitted = load_fitted(cached, dataset, start, logistic_recipe, protocol)
                    reused_models += 1
                else:
                    fitted = fit_recipe(
                        dataset,
                        family_columns("flow_momentum"),
                        start,
                        logistic_recipe,
                        protocol,
                        consume_fit,
                    )
                fitted.save(directory / phase / f"fold_{number}" / "models" / logistic_recipe["id"])
                mask = (
                    frame.research_ready
                    & (frame.reference_time >= start)
                    & (frame.reference_time < stop)
                )
                values = {
                    logistic_recipe["id"]: fitted.probabilities(frame.loc[mask]),
                    "prevalence_" + key: np.full(int(mask.sum()), fitted.audit["train_prevalence"]),
                    "constant_" + key: np.full(int(mask.sum()), 0.5),
                    "momentum_sign_" + key: np.where(
                        frame.loc[mask, "adv_momentum_60"] > 0, 0.55, 0.45
                    ),
                }
                for name, val in values.items():
                    p = np.zeros(len(frame))
                    p[mask] = val
                    score_model(
                        phase,
                        number,
                        name,
                        key,
                        p,
                        start,
                        stop,
                        fitted if name.startswith("logistic_") else None,
                    )
                    if name.startswith("logistic_"):
                        execution_policy = (
                            policy
                            if key == "direction_h1"
                            else replace(
                                policy,
                                confirmations=1,
                                persistent=False,
                                cooldown_bars=0,
                                max_holding_bars=int(key[-1]),
                            )
                        )
                        save_sim(
                            phase,
                            number,
                            name,
                            run_overlay(frame, p, start, stop, costs, execution_policy),
                            {"target": key, "baseline": "logistic_l2"},
                        )
            # Meta models train on OLD OOS opportunities; current-fold labels not appended yet.
            protect_reserved_labels(
                settings.root,
                meta_pool.loc[(meta_pool.status == "observed") & (meta_pool.target_time < start)],
            )
            meta_models = {
                kind: fit_meta(frame, meta_pool, start, kind, protocol, consume_fit)
                for kind in ["logistic", "xgb"]
            }
            for kind, fitted_meta in meta_models.items():
                destination = directory / phase / f"fold_{number}" / "models" / ("meta_" + kind)
                if fitted_meta.model is not None:
                    Fitted(fitted_meta.model, META, fitted_meta.audit).save(destination)
                else:
                    write_new_json(destination / "fit.json", fitted_meta.audit)
            offers = causal_opportunity_labels(frame, base, start, stop, costs, policy)
            for name, (gate, p, detail) in gates(frame, base, meta_models, protocol).items():
                result = run_overlay(frame, base, start, stop, costs, policy, entry_gate=gate)
                kind = "logistic" if "logistic" in name else "xgb"
                save_sim(
                    phase,
                    number,
                    name,
                    result,
                    {
                        "meta_training": meta_models[kind].audit
                        if name.startswith("meta_")
                        else None,
                        "entry_only_filter": True,
                        "exit_probabilities_unchanged": True,
                    },
                )
                decisions = frame.loc[
                    (frame.reference_time >= start) & (frame.reference_time < stop),
                    ["timestamp", "reference_time", "regime", "adv_regime_score", "volatility"],
                ].copy()
                loc = decisions.index
                decisions["base_probability"] = base[loc]
                decisions["filter_probability"] = p[loc]
                decisions["accepted"] = gate[loc]
                decisions["base_opportunity"] = decisions.reference_time.isin(offers.reference_time)
                decisions["actual_reference_entry"] = decisions.reference_time.isin(
                    base_sim.simulation.trades.signal_reference
                )
                if detail:
                    decisions["expected_net"] = detail["expected_net"][loc]
                    decisions["expectation_lower"] = detail["expectation_lower"][loc]
                if name.startswith("meta_"):
                    rejected = np.where(
                        p[loc] < protocol["filters"]["meta_threshold"],
                        "low_meta_confidence",
                        "expectation_lower_not_positive",
                    )
                    if meta_models[kind].model is None:
                        rejected = np.full(len(loc), "insufficient_past_meta_data")
                else:
                    rejected = np.full(
                        len(loc),
                        "negative_trend"
                        if name == "trend_gate"
                        else "volatility_above_past_quantile",
                    )
                decisions["rejection_reason"] = np.where(gate[loc], "accepted", rejected)
                decisions.to_csv(
                    directory / phase / f"fold_{number}" / (name + "_decisions.csv"), index=False
                )
                observed = offers.loc[offers.status == "observed"]
                if (
                    name in {"meta_logistic", "meta_xgb"}
                    and meta_models[kind].model is not None
                    and len(observed)
                ):
                    record(
                        {
                            "phase": phase,
                            "fold": number,
                            "experiment": name + "_ranking",
                            "target": "isolated_opportunity_net_profit",
                            "metrics": classification_metrics(
                                observed.target.to_numpy(), p[observed.source_index.to_numpy()]
                            ),
                            "same_trades_not_independent": True,
                        }
                    )
            offers["forecast_fold"] = number
            offers.to_csv(
                directory / phase / f"fold_{number}" / "oos_meta_opportunities.csv", index=False
            )
            meta_pool = pd.concat(
                [meta_pool, offers.drop(columns=["forecast_fold"])], ignore_index=True
            )
            if phase == "selection_development" and number == len(dates) - 1:
                # Freeze all selection BEFORE accessing/replaying the audited period.
                selection_predictive = _predictive_results(scores[phase], protocol)
                selection_economic = {
                    name: aggregate(items) for name, items in simulations[phase].items()
                }
                economic_intervals = _economic_uncertainty(selection_economic, protocol)
                write_new_json(
                    directory / "selection_economic_uncertainty.json", economic_intervals
                )
                selection, selected = _select(
                    selection_predictive, selection_economic, folds, protocol, economic_intervals
                )
                ref_exposure = selection_economic["defensive_reference"]["metrics"][
                    "mean_capital_exposure"
                ]
                new_exposure = selection_economic[selected]["metrics"]["mean_capital_exposure"]
                candidate_scale = min(1.0, ref_exposure / new_exposure) if new_exposure > 0 else 1.0
                reference_scale = min(1.0, new_exposure / ref_exposure) if ref_exposure > 0 else 1.0
                matched_policies = {
                    "candidate": _scaled_policy(policy, candidate_scale),
                    "reference": _scaled_policy(policy, reference_scale)
                    if reference_scale > 0
                    else None,
                }
                write_new_json(
                    directory / "selection_frozen.json",
                    {
                        "selected_for_diagnostic": selected,
                        "decisions": selection,
                        "candidate_scale": candidate_scale,
                        "reference_scale": reference_scale,
                        "audited_period_used": False,
                        "matching_is_development_only_approximation": True,
                    },
                )
            if phase == "audited_observed_development":
                p = recipe_probabilities[selected]
                save_sim(
                    phase,
                    number,
                    "candidate_matched_exposure",
                    run_overlay(frame, p, start, stop, costs, matched_policies["candidate"]),
                    {"scaling_frozen_in_development": True},
                )
                if matched_policies["reference"] is not None:
                    save_sim(
                        phase,
                        number,
                        "reference_matched_exposure",
                        run_overlay(frame, base, start, stop, costs, matched_policies["reference"]),
                        {"scaling_frozen_in_development": True},
                    )
        if (
            _protected_hashes(settings) != protected
            or reservation_hashes(settings.root) != reserved
            or sha256(source) != provenance["source_sha256"]
            or any(sha256(Path(p)) != expected for p, expected in cache_hashes.items())
        ):
            raise DataError("Se alteró un modelo, dato o reserva protegida")
        periods = {}
        for phase, systems in simulations.items():
            economic = {name: aggregate(items) for name, items in systems.items()}
            for name, item in economic.items():
                item.pop("_curve").to_csv(
                    directory / (phase + "__" + name + "__returns.csv"), index=False
                )
            periods[phase] = {
                "economic": economic,
                "predictive": selection_predictive
                if phase == "selection_development"
                else _predictive_results(scores[phase], protocol, bootstrap=False),
            }
        observed = periods["audited_observed_development"]
        target_metrics = observed["predictive"]["direction_h1"]["metrics"]
        accepted = selection[selected]["accepted"]
        comparison = {
            "candidate": selected,
            "candidate_accepted": accepted,
            "before_predictive": target_metrics["defensive_reference"],
            "after_predictive": target_metrics[selected],
            "before_economic": observed["economic"]["defensive_reference"]["metrics"],
            "after_economic": observed["economic"][selected]["metrics"],
            "equal_data_control": target_metrics["original_refit"],
            "same_labels_dates_costs_and_risk_budget": True,
            "historical_independent_confirmation": False,
            "retained_recommendation": selected if accepted else "defensive_reference",
            "conclusion": "MIXTO" if accepted else "INCONCLUSO",
            "note": "Experimental candidate; predictive improvement requires paired earlier-fold evidence and economic coverage. The audited period is descriptive only; reference retained unless all acceptance checks passed.",
        }
        summary = {
            "run_id": run_id,
            "directory": str(directory.resolve()),
            "status": "completed",
            "protocol": protocol,
            "plan_sha256": sha256(directory / "plan.json"),
            "audit": {"winners": audit["winners"], "losers": audit["losers"]},
            "periods": periods,
            "selection": selection,
            "comparison": comparison,
            "actual_fits": fits,
            "fits_in_this_run": fits - previous_fits,
            "reused_models": reused_models,
            "experiment_records": records,
            "existing_models_preserved": True,
            "prospective_unchanged": True,
            "prospective_evaluated": False,
            "promoted": False,
            "live_trading": False,
        }
        write_new_json(directory / "summary.json", summary)
        write_report(directory / "REPORT.md", summary)
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
        failure = {
            "status": "failed",
            "exception": type(exc).__name__,
            "reason": str(exc),
            "actual_fits": fits,
        }
        record(failure)
        write_new_json(directory / "failure.json", failure)
        raise


def _scaled_policy(policy: RiskPolicy, factor: float) -> RiskPolicy:
    return replace(
        policy,
        capital_cap=policy.capital_cap * factor,
        risk_budget=policy.risk_budget * factor,
        volatility_target_per_bar=policy.volatility_target_per_bar * factor,
    )


def _predictive_results(
    scores: dict[str, list[pd.DataFrame]], protocol: dict[str, Any], bootstrap: bool = True
) -> dict[str, Any]:
    results = {}
    for key, pieces in scores.items():
        data = pd.concat(pieces, ignore_index=True)
        pivots = data.pivot(
            index=["fold", "reference_time"], columns="model", values="probability"
        ).sort_index()
        target = (
            data.drop_duplicates(["fold", "reference_time"])
            .set_index(["fold", "reference_time"])
            .loc[pivots.index, "target"]
            .to_numpy()
        )
        probabilities = {name: pivots[name].to_numpy() for name in pivots}
        references = (
            [
                "original_refit",
                "logistic_direction_h1",
                "prevalence_direction_h1",
                "constant_direction_h1",
            ]
            if key == "direction_h1"
            else [f"{key}_reference", "logistic_" + key, "prevalence_" + key]
        )
        candidates = [
            r["id"] for r in protocol["recipes"] if r["target"] == key and r["id"] not in references
        ]
        pairs = [(candidate, reference) for candidate in candidates for reference in references]
        if key == "direction_h1":
            pairs.append(("calibrated", "calibration_raw_control"))
            pairs += [
                (name, "core_deduplicated")
                for name in [
                    "momentum",
                    "mean_reversion",
                    "volatility_breakout",
                    "order_flow",
                    "smc",
                ]
            ]
            pairs += [
                ("flow_momentum", "order_flow"),
                ("flow_momentum", "momentum"),
                ("regularized", "flow_momentum"),
                ("rolling_730", "regularized"),
                ("depth_3", "regularized"),
            ]
        pairs = [(a, b) for a, b in pairs if a in probabilities and b in probabilities]
        if bootstrap:
            results[key] = paired_predictive(
                target,
                probabilities,
                pivots.index.get_level_values("fold").to_numpy(),
                pairs,
                protocol["bootstrap"],
            )
        else:
            results[key] = {
                "metrics": {
                    name: classification_metrics(target, p) for name, p in probabilities.items()
                },
                "independent_confirmation": False,
            }
    return results


def _select(
    predictive: dict[str, Any],
    economics: dict[str, Any],
    folds: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
    economic_intervals: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    group = predictive["direction_h1"]
    references = [
        "original_refit",
        "logistic_direction_h1",
        "prevalence_direction_h1",
        "constant_direction_h1",
    ]
    candidates = [
        r["id"]
        for r in protocol["recipes"]
        if r["target"] == "direction_h1" and r["id"] != "original_refit"
    ]
    acceptance = protocol["acceptance"]
    decisions = {}
    for name in candidates:
        point = group["metrics"][name]
        baseline = group["metrics"]["original_refit"]
        failed = []
        if baseline["log_loss"] - point["log_loss"] < acceptance["minimum_log_loss_gain"]:
            failed.append("insufficient_log_loss_gain")
        if (point["roc_auc"] or 0) - (baseline["roc_auc"] or 0) < acceptance["minimum_auc_gain"]:
            failed.append("insufficient_auc_gain")
        for ref in references:
            if point["log_loss"] >= group["metrics"][ref]["log_loss"]:
                failed.append("fails_point_log_loss_vs_" + ref)
            for block, pairs in group["blocks"].items():
                upper = pairs[name + "__minus__" + ref]["log_loss"]["upper"]
                if upper is None or upper >= 0:
                    failed.append("no_joint_loss_evidence_vs_" + ref + "_block_" + block)
        stable = np.mean(
            [
                a["metrics"]["log_loss"] < b["metrics"]["log_loss"]
                for a, b in zip(folds[name], folds["original_refit"], strict=True)
            ]
        )
        if stable < acceptance["required_fold_fraction"]:
            failed.append("unstable_predictive_folds")
        metrics = economics[name]["metrics"]
        if metrics["trade_count"] < acceptance["minimum_trades"]:
            failed.append("insufficient_economic_operations")
        ref_metrics = economics["defensive_reference"]["metrics"]
        if (
            metrics["trade_count"]
            < ref_metrics["trade_count"] * acceptance["minimum_coverage_vs_reference"]
        ):
            failed.append("abstention_without_sufficient_coverage")
        if metrics["net_return"] <= 0:
            failed.append("nonpositive_development_net_return")
        if acceptance["require_positive_risk_budget_gain"] and any(
            row[name]["lower"] <= 0 for row in economic_intervals["blocks"].values()
        ):
            failed.append("no_joint_economic_gain_at_shared_risk_budget")
        if metrics["mean_capital_exposure"] <= 0:
            failed.append("no_exposure_for_matched_comparison")
        else:
            # Exposure-unit return is descriptive. Requires positive mean net return,
            # never treats reduced loss from smaller capital as alpha.
            if metrics["net_return"] / metrics["mean_capital_exposure"] <= ref_metrics[
                "net_return"
            ] / max(ref_metrics["mean_capital_exposure"], 1e-12):
                failed.append("no_exposure_normalized_development_gain")
        decisions[name] = {
            "accepted": not failed,
            "failed_checks": sorted(set(failed)),
            "positive_loss_fold_fraction": float(stable),
            "audited_period_used": False,
            "independent_confirmation": False,
        }
    eligible = [name for name, result in decisions.items() if result["accepted"]]
    chosen = min(eligible or candidates, key=lambda name: group["metrics"][name]["log_loss"])
    return decisions, chosen


def _economic_uncertainty(economics: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    from trading_ml.evaluation.uncertainty import moving_block_indices

    base = economics["defensive_reference"]["_curve"]
    names = [name for name in economics if name not in {"defensive_reference", "cash", "buy_hold"}]
    config = protocol["bootstrap"]
    estimates = {}
    standardized = []
    intervals = []
    for block in config["block_lengths"]:
        indices = moving_block_indices(
            base.fold.to_numpy(), config["samples"], block, config["seed"]
        )
        estimates[str(block)] = {}
        for name in names:
            candidate = economics[name]["_curve"]
            if not candidate.timestamp.equals(base.timestamp):
                raise DataError("Comparación económica desalineada")
            difference = np.log1p(candidate.net_return.to_numpy()) - np.log1p(
                base.net_return.to_numpy()
            )
            draws = difference[indices].mean(axis=1)
            point, se = float(difference.mean()), float(draws.std(ddof=1))
            item = {"estimate_mean_log_gain": point, "se": se}
            standardized.append(
                np.abs((draws - point) / se) if se > 1e-12 else np.zeros(len(draws))
            )
            intervals.append(item)
            estimates[str(block)][name] = item
    critical = float(np.quantile(np.max(standardized, axis=0), config["confidence"]))
    for item in intervals:
        item["lower"] = item["estimate_mean_log_gain"] - critical * item["se"]
        item["upper"] = item["estimate_mean_log_gain"] + critical * item["se"]
    return {
        "blocks": estimates,
        "method": "paired_return_blocks_simultaneous_max_t",
        "critical": critical,
        "exposure_differences_not_removed": True,
        "interpretation": "Return differences alone do not demonstrate prediction improvements; original risk budget shared, exposure/trade-count reported separately",
    }
