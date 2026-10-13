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

from trading_ml.common import (
    code_provenance,
    environment,
    read_json,
    sha256,
    sidecar,
    write_new_json,
)
from trading_ml.config import Settings
from trading_ml.errors import DataError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.mean_reversion.config import ResearchCosts, ReversionConfig
from trading_ml.mean_reversion.features import build_reversion_features
from trading_ml.mean_reversion.models import fit_filter
from trading_ml.mean_reversion.report import write_report
from trading_ml.mean_reversion.rules import reversion_opportunities
from trading_ml.mean_reversion.simulator import opportunity_labels, simulate_reversion
from trading_ml.mean_reversion.statistics import aggregate, ar1_diagnostic, paired_evidence
from trading_ml.momentum.features import causal_regimes
from trading_ml.momentum.simulator import YEAR_SECONDS, Simulation, simulate_policy
from trading_ml.preprocessing.quality import clean_ohlcv
from trading_ml.training.recipe import protect_reserved_labels


def validate_protocol(protocol: dict[str, Any]) -> None:
    research = protocol["research"]
    dates = pd.to_datetime(research["boundaries"], utc=True)
    windows = research["lookbacks"]
    filters = research["regime_filters"]
    if not 1 <= len(windows) <= 3 or len(set(windows)) != len(windows):
        raise DataError("Hasta tres ventanas Mean Reversion fijas")
    if not filters or len(set(filters)) != len(filters) or not set(filters) <= {"all", "sideways"}:
        raise DataError("Filtros de régimen inválidos")
    if not 3 <= len(dates) <= 9 or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise DataError("Folds Mean Reversion inválidos")
    if pd.Timestamp(research["historical_end"], tz="UTC") <= dates[-1]:
        raise DataError("Historial posterior a desarrollo inválido")
    expected_fits = len(windows) * len(filters) * 2 * (len(dates) + 1)
    if not expected_fits <= research["max_fits"] <= 120:
        raise DataError("Presupuesto de ajustes Mean Reversion excedido")
    if (
        not 0.5 <= research["probability_threshold"] <= 1
        or research["minimum_train_opportunities"] < 2
    ):
        raise DataError("Umbral probabilístico o muestra inválidos")
    if not 1 <= research["prospective_days"] <= 730:
        raise DataError("Reserva prospectiva inválida")
    if (
        not 1 <= protocol["xgboost"]["n_estimators"] <= 160
        or not 1 <= protocol["xgboost"]["max_depth"] <= 4
    ):
        raise DataError("Presupuesto XGBoost inválido")
    if protocol["logistic"]["C"] <= 0 or not 100 <= protocol["logistic"]["max_iter"] <= 5000:
        raise DataError("Regresión logística inválida")
    boot = protocol["bootstrap"]
    if (
        not 100 <= boot["samples"] <= 5000
        or not 0 < boot["confidence"] < 1
        or not boot["block_lengths"]
    ):
        raise DataError("Bootstrap Mean Reversion inválido")
    if any(not isinstance(x, int) or x < 1 for x in boot["block_lengths"]):
        raise DataError("Bloques Mean Reversion inválidos")
    if protocol["stress"]["cost_multiplier"] < 1:
        raise DataError("Stress inválido")
    if (
        protocol["acceptance"]["minimum_trades"] < 1
        or not 0 < protocol["acceptance"]["required_positive_fold_fraction"] <= 1
    ):
        raise DataError("Aceptación Mean Reversion inválida")
    for window in windows:
        ReversionConfig(lookback=window, **protocol["rules"]).validate()
    ResearchCosts(**protocol["costs"]).validate()


def _protected_hashes(settings: Settings) -> dict[str, str]:
    return {
        str(p.resolve()): sha256(p)
        for p in sorted((settings.root / "models").rglob("*"))
        if p.is_file()
        and (
            p.suffix == ".ubj"
            or p.name in {"metadata.json", "active.json", "latest.json", "champion.json"}
        )
    }


def run_reversion_study(
    settings: Settings, protocol_path: Path, source: Path | None = None
) -> dict[str, Any]:
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    research = protocol["research"]
    source = source or settings.root / "data" / "raw.csv"
    manifest = read_json(sidecar(source))
    if manifest.get("kind") != "raw" or manifest.get("sha256") != sha256(source):
        raise DataError("Datos crudos Mean Reversion sin procedencia íntegra")
    if any(manifest["market"][k] != getattr(settings.market, k) for k in ["symbol", "interval"]):
        raise DataError("Mercado Mean Reversion incompatible")
    raw, quality = clean_ohlcv(pd.read_csv(source), settings.delta)
    raw["reference_time"] = raw.timestamp + settings.delta
    historical_end = pd.Timestamp(research["historical_end"], tz="UTC")
    if (
        raw.reference_time.iloc[-1] < historical_end
        or (raw.reference_time > pd.Timestamp.now(tz="UTC")).any()
    ):
        raise DataError("Historial Mean Reversion incompleto o con velas abiertas")
    raw = raw.loc[raw.reference_time <= historical_end].reset_index(drop=True)
    raw["regime"] = causal_regimes(raw)
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
    protected = _protected_hashes(settings)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = settings.root / "mean_reversion_studies" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "protocol.toml").write_bytes(protocol_path.read_bytes())
    raw.to_csv(destination / "ohlcv.csv", index=False)
    future_start = max(
        pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1),
        historical_end + settings.delta,
    )
    prospective = {
        "status": "pending_future_data",
        "start": future_start.isoformat(),
        "end": (future_start + pd.Timedelta(days=research["prospective_days"])).isoformat(),
        "no_training_or_selection": True,
    }
    plan = {
        "run_id": run_id,
        "protocol": protocol,
        "protocol_sha256": sha256(protocol_path),
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "source_manifest": manifest,
        "snapshot_sha256": sha256(destination / "ohlcv.csv"),
        "quality": quality,
        "code_provenance": code_provenance(),
        "environment": environment(),
        "protected_models": protected,
        "prospective": prospective,
        "historical_test_already_observed": True,
        "selection": "none_all_fixed_hypotheses_reported",
        "live_trading": False,
        "promoted": False,
    }
    write_new_json(destination / "plan.json", plan)
    records = 0

    def record(value: dict[str, Any]) -> None:
        nonlocal records
        with (destination / "experiments.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        records += 1

    actual_fits = 0
    try:
        recipes = {}
        for window in research["lookbacks"]:
            for regime_filter in research["regime_filters"]:
                name = f"mr_L{window}_{regime_filter}"
                config = ReversionConfig(
                    lookback=window, sideways_only=regime_filter == "sideways", **protocol["rules"]
                )
                frame = build_reversion_features(raw, config)
                opportunities = reversion_opportunities(frame, config)
                labels = opportunity_labels(frame, opportunities, config, costs)
                folder = destination / "datasets" / name
                folder.mkdir(parents=True, exist_ok=False)
                frame.to_csv(folder / "features.csv", index=False)
                opportunities.to_csv(folder / "opportunities.csv", index=False)
                labels.to_csv(folder / "labels.csv", index=False)
                recipes[name] = (config, frame, opportunities, labels)
        dates = pd.to_datetime(research["boundaries"], utc=True)
        windows = [
            ("development", i + 1, a, b)
            for i, (a, b) in enumerate(zip(dates[:-1], dates[1:], strict=True))
        ]
        windows.append(("observed_historical", 1, dates[-1], historical_end))
        results: dict[str, dict[str, list[tuple[int, Simulation]]]] = {}
        frequency: dict[str, dict[str, list[dict[str, Any]]]] = {}

        def save(
            period: str, fold: int, name: str, simulation: Simulation, audit: dict[str, Any]
        ) -> None:
            folder = destination / period / f"fold_{fold}" / name
            folder.mkdir(parents=True, exist_ok=False)
            simulation.equity.to_csv(folder / "equity.csv", index=False)
            simulation.trades.to_csv(folder / "trades.csv", index=False)
            experiment = {
                "period": period,
                "fold": fold,
                "system": name,
                "metrics": simulation.metrics,
                "audit": audit,
                "status": "completed",
                "negative_net_return": simulation.metrics["net_return"] < 0,
            }
            write_new_json(folder / "experiment.json", experiment)
            record(experiment)
            results.setdefault(period, {}).setdefault(name, []).append((fold, simulation))

        for period, fold, start, end in windows:
            benchmark_signals = pd.DataFrame({"entry": True, "exit": False}, index=raw.index)
            for suffix, scenario_costs in [("", costs), ("_stress", stress)]:
                for name, signals in [
                    ("buy_hold", benchmark_signals),
                    ("cash", benchmark_signals.assign(entry=False)),
                ]:
                    save(
                        period + suffix,
                        fold,
                        name,
                        simulate_policy(raw, signals, start, end, scenario_costs, len(raw) + 1),
                        {"status": "no_model"},
                    )
            for recipe, (config, frame, opportunities, labels) in recipes.items():
                fitting_labels = labels.loc[
                    (labels.label_status == "observed") & (labels.target_time < start)
                ]
                protect_reserved_labels(settings.root, fitting_labels)
                eligible = (
                    (frame.reference_time >= start) & (frame.reference_time < end) & frame.mr_ready
                )
                selected = opportunities & eligible
                label_eval = labels.loc[
                    (labels.reference_time >= start) & (labels.reference_time < end)
                ]
                observed_eval = label_eval.loc[
                    (label_eval.label_status == "observed") & (label_eval.target_time < end)
                ]
                years = (end - start).total_seconds() / YEAR_SECONDS
                stats = {
                    "fold": fold,
                    "first_reference": start.isoformat(),
                    "end_exclusive": end.isoformat(),
                    "ready_bars": int(eligible.sum()),
                    "opportunities": int(selected.sum()),
                    "opportunity_fraction": float(selected.sum() / eligible.sum())
                    if eligible.any()
                    else None,
                    "opportunities_per_year": float(selected.sum() / years),
                    "episodes": int((selected & ~opportunities.shift(1, fill_value=False)).sum()),
                    "observed_labels_within_window": len(observed_eval),
                    "right_censored_or_cross_boundary": int(
                        (label_eval.label_status == "right_censored").sum()
                        + (
                            (label_eval.label_status == "observed")
                            & (label_eval.target_time >= end)
                        ).sum()
                    ),
                    "not_executable": int(
                        (label_eval.label_status == "entry_not_executable").sum()
                    ),
                    "empirical_reversion_fraction": float(observed_eval.target.mean())
                    if len(observed_eval)
                    else None,
                    "mean_hypothetical_trade_net_return": float(
                        observed_eval.trade_net_return.mean()
                    )
                    if len(observed_eval)
                    else None,
                    "regimes": {
                        r: int((selected & (frame.regime == r)).sum())
                        for r in ["bull", "bear", "sideways"]
                    },
                    "train_log_price_ar1": ar1_diagnostic(
                        frame.loc[frame.reference_time < start, "close"]
                    ),
                }
                frequency.setdefault(period, {}).setdefault(recipe, []).append(stats)
                write_new_json(
                    destination / period / f"fold_{fold}" / f"{recipe}_frequency.json", stats
                )
                for variant in ["rules", "logistic", "xgb"]:
                    audit: dict[str, Any] = {
                        "status": "no_model",
                        "probability_target": "not_applicable",
                    }
                    p = None
                    if variant != "rules":
                        model = fit_filter(
                            frame,
                            labels,
                            start,
                            variant,
                            protocol["logistic" if variant == "logistic" else "xgboost"],
                            research["minimum_train_opportunities"],
                            research["seed"],
                        )
                        actual_fits += model.estimator is not None
                        p = model.probabilities(frame)
                        audit = model.audit.copy()
                        audit["evaluation_opportunities"] = int(selected.sum())
                        audit["accepted_opportunities"] = int(
                            (selected & (p >= research["probability_threshold"])).sum()
                        )
                        audit["evaluation_label_metrics"] = None
                        if model.estimator is not None and len(observed_eval):
                            target = observed_eval.target.to_numpy()
                            audit["evaluation_label_metrics"] = {
                                "model": classification_metrics(
                                    target, p[observed_eval.source_index.to_numpy()]
                                ),
                                "train_prevalence": classification_metrics(
                                    target, np.full(len(target), audit["train_prevalence"])
                                ),
                                "constant_05": classification_metrics(
                                    target, np.full(len(target), 0.5)
                                ),
                            }
                        audit["evaluation_used_for_selection"] = False
                        folder = (
                            destination / period / f"fold_{fold}" / f"filter_{recipe}_{variant}"
                        )
                        model.save(folder)
                        predictions = frame.loc[
                            eligible, ["timestamp", "reference_time", "mr_z", "mr_rsi", "regime"]
                        ].copy()
                        predictions["opportunity"] = opportunities.loc[eligible]
                        predictions["probability_reversion"] = (
                            p[eligible.to_numpy()] if model.estimator is not None else np.nan
                        )
                        predictions.to_csv(folder / "predictions.csv", index=False)
                    for suffix, scenario_costs in [("", costs), ("_stress", stress)]:
                        simulation = simulate_reversion(
                            frame,
                            opportunities,
                            start,
                            end,
                            config,
                            scenario_costs,
                            p,
                            research["probability_threshold"],
                        )
                        save(period + suffix, fold, recipe + "_" + variant, simulation, audit)
        frozen = {}
        for recipe, (config, frame, _opportunities, labels) in recipes.items():
            protect_reserved_labels(
                settings.root,
                labels.loc[
                    (labels.label_status == "observed") & (labels.target_time < historical_end)
                ],
            )
            frozen[recipe] = {"config": asdict(config), "models": {}}
            for kind in ["logistic", "xgb"]:
                model = fit_filter(
                    frame,
                    labels,
                    historical_end,
                    kind,
                    protocol["logistic" if kind == "logistic" else "xgboost"],
                    research["minimum_train_opportunities"],
                    research["seed"],
                )
                actual_fits += model.estimator is not None
                model.save(destination / "frozen" / recipe / kind)
                frozen[recipe]["models"][kind] = model.audit
                record(
                    {
                        "period": "freeze_for_future",
                        "system": recipe + "_" + kind,
                        "audit": model.audit,
                    }
                )
        if actual_fits > research["max_fits"] or _protected_hashes(settings) != protected:
            raise DataError(
                "Presupuesto excedido o modelos existentes alterados durante investigación"
            )
        periods = {}
        for period, systems in results.items():
            aggregated = {name: aggregate(items) for name, items in systems.items()}
            evidence, decisions = paired_evidence(aggregated, protocol)
            for name, item in aggregated.items():
                item.pop("_curve").to_csv(
                    destination / f"{period}__{name}__returns.csv", index=False
                )
            periods[period] = {
                "systems": aggregated,
                "statistical_evidence": evidence,
                "decisions": decisions,
            }
        summary = {
            "run_id": run_id,
            "directory": str(destination.resolve()),
            "status": "completed",
            "protocol": protocol,
            "plan_sha256": sha256(destination / "plan.json"),
            "actual_fits": actual_fits,
            "experiment_records": records,
            "periods": periods,
            "opportunity_frequency": frequency,
            "frozen": frozen,
            "prospective": prospective,
            "costs": asdict(costs),
            "stress_costs": asdict(stress),
            "existing_models_preserved": True,
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
            "run_id": run_id,
            "status": "failed",
            "exception": type(exc).__name__,
            "reason": str(exc),
            "actual_fits": actual_fits,
        }
        record(failure)
        write_new_json(destination / "failure.json", failure)
        raise
