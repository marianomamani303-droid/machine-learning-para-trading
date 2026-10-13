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
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.mean_reversion.study import _protected_hashes
from trading_ml.models import load_model
from trading_ml.momentum.config import ResearchCosts
from trading_ml.momentum.features import causal_regimes
from trading_ml.momentum.simulator import Simulation, performance
from trading_ml.momentum.study import _aggregate, _reference_frame, _reference_probabilities
from trading_ml.preprocessing.quality import clean_ohlcv
from trading_ml.risk_overlay.config import RiskPolicy
from trading_ml.risk_overlay.report import write_report
from trading_ml.risk_overlay.simulator import (
    RiskSimulation,
    risk_features,
    risk_statistics,
    run_overlay,
)


def reservation_hashes(root: Path) -> dict[str, str]:
    paths = list((root / "smc_studies").glob("*/prospective.json"))
    for name in ["smc_studies", "momentum_studies", "mean_reversion_studies"]:
        paths.extend((root / name).glob("*/plan.json"))
        paths.extend(p for p in (root / name).glob("*/frozen/**/*") if p.is_file())
    return {str(p.resolve()): sha256(p) for p in sorted(set(paths))}


def _check_reserved_end(root: Path, end: pd.Timestamp) -> None:
    future = [
        read_json(p)["first_reference"] for p in (root / "smc_studies").glob("*/prospective.json")
    ]
    for name in ["momentum_studies", "mean_reversion_studies"]:
        future.extend(
            read_json(p)["prospective"]["start"]
            for p in (root / name).glob("*/plan.json")
            if read_json(p).get("prospective")
        )
    if future and end > min(pd.Timestamp(t) for t in future):
        raise LeakageError("El estudio de riesgo no puede evaluar datos de la reserva prospectiva")


def validate_protocol(protocol: dict[str, Any]) -> list[RiskPolicy]:
    research = protocol["research"]
    dates = pd.to_datetime(research["boundaries"], utc=True)
    versions = research["reference_versions"]
    if not 1 <= len(versions) <= 3 or len(set(versions)) != len(versions):
        raise DataError("Hasta tres referencias de riesgo distintas")
    if not 3 <= len(dates) <= 9 or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise DataError("Folds de riesgo inválidos")
    if (
        pd.Timestamp(research["historical_end"], tz="UTC") <= dates[-1]
        or len(versions) * (len(dates) - 1) > research["max_fits"]
        or research["max_fits"] > 30
    ):
        raise DataError("Período o presupuesto de riesgo inválidos")
    policies = [RiskPolicy(**item) for item in protocol["policies"]]
    names = [p.name for p in policies]
    if not 2 <= len(policies) <= research["max_policies"] <= 12 or len(set(names)) != len(names):
        raise DataError("Presupuesto o nombres de políticas inválidos")
    for policy in policies:
        policy.validate()
    original = next((p for p in policies if p.name == "original"), None)
    if (
        original != RiskPolicy()
        or research["primary_reference"] not in versions
        or research["primary_policy"] not in names
    ):
        raise DataError("Referencia original o comparación prefijada inválida")
    if protocol["stress"]["cost_multiplier"] < 1:
        raise DataError("Stress de riesgo inválido")
    ResearchCosts(**protocol["costs"]).validate()
    return policies


def aggregate(results: list[tuple[int, Simulation]]) -> dict[str, Any]:
    data = _aggregate(results)
    curve = data["_curve"].copy()
    initial = results[0][1].equity.equity.iloc[0]
    curve["gross_equity"] = initial * (1 + curve.gross_return).cumprod()
    first = curve.iloc[[0]].copy()
    for field in ["equity", "gross_equity"]:
        first[field] = initial
    for field in [
        "net_return",
        "gross_return",
        "turnover",
        "execution_cost_cash",
        "commission_cash",
        "slippage_cash",
        "capital_exposure",
    ]:
        first[field] = 0.0
    full = pd.concat([first, curve], ignore_index=True)
    trades = pd.concat([r.trades for _, r in results], ignore_index=True)
    data["metrics"] = {**performance(full, trades), **risk_statistics(full, trades)}
    for key in [
        "eligible_signals",
        "confirmation_rejections",
        "cooldown_rejections",
        "drawdown_rejections",
        "signals_while_in_position",
        "invalid_sizing_rejections",
        "terminal_signal_exclusions",
        "cap_reductions",
        "drawdown_halts",
    ]:
        data["metrics"][key] = sum(r.metrics[key] for _, r in results)
    return data


def run_risk_study(
    settings: Settings, protocol_path: Path, source: Path | None = None
) -> dict[str, Any]:
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    policies = validate_protocol(protocol)
    research = protocol["research"]
    dates = pd.to_datetime(research["boundaries"], utc=True)
    end = pd.Timestamp(research["historical_end"], tz="UTC")
    _check_reserved_end(settings.root, end)
    source = source or settings.root / "data" / "raw.csv"
    manifest = read_json(sidecar(source))
    if manifest.get("kind") != "raw" or manifest.get("sha256") != sha256(source):
        raise DataError("Datos crudos de riesgo sin procedencia íntegra")
    if any(manifest["market"][k] != getattr(settings.market, k) for k in ["symbol", "interval"]):
        raise DataError("Mercado de riesgo incompatible")
    raw, quality = clean_ohlcv(pd.read_csv(source), settings.delta)
    raw["reference_time"] = raw.timestamp + settings.delta
    if raw.reference_time.iloc[-1] < end or (raw.reference_time > pd.Timestamp.now(tz="UTC")).any():
        raise DataError("Cobertura de riesgo incompleta o velas abiertas")
    raw = raw.loc[raw.reference_time <= end].reset_index(drop=True)
    raw["regime"] = causal_regimes(raw)
    bundles = [load_model(settings.root, version) for version in research["reference_versions"]]
    for b in bundles:
        if any(
            b.metadata["contract"][k] != getattr(settings.market, k) for k in ["symbol", "interval"]
        ):
            raise DataError("Modelo de otro mercado en estudio de riesgo")
    costs = ResearchCosts(**protocol["costs"])
    stress = replace(
        costs,
        commission_bps=costs.commission_bps * protocol["stress"]["cost_multiplier"],
        slippage_bps=costs.slippage_bps * protocol["stress"]["cost_multiplier"],
        spread_bps=costs.spread_bps * protocol["stress"]["cost_multiplier"],
    )
    stress.validate()
    protected = _protected_hashes(settings)
    reserved = reservation_hashes(settings.root)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = settings.root / "risk_studies" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "protocol.toml").write_bytes(protocol_path.read_bytes())
    raw.to_csv(directory / "ohlcv.csv", index=False)
    prior_studies = {
        str(p): {
            "sha256": sha256(p),
            "independent_advantage_confirmed": read_json(p).get("independent_advantage_confirmed"),
        }
        for name in ["momentum_studies", "mean_reversion_studies"]
        for p in (settings.root / name).glob("*/summary.json")
    }
    plan = {
        "run_id": run_id,
        "protocol": protocol,
        "protocol_sha256": sha256(protocol_path),
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "manifest": manifest,
        "snapshot_sha256": sha256(directory / "ohlcv.csv"),
        "quality": quality,
        "code_provenance": code_provenance(),
        "environment": environment(),
        "protected_models": protected,
        "reserved_hashes": reserved,
        "prior_studies": prior_studies,
        "selection": "primary_fixed_before_evaluation; matched_caps_from_development_only",
        "historical_test_already_observed": True,
        "prospective_evaluated": False,
        "live_trading": False,
        "promoted": False,
    }
    write_new_json(directory / "plan.json", plan)
    records = 0
    fits = 0

    def record(value: dict[str, Any]) -> None:
        nonlocal records
        with (directory / "experiments.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        records += 1

    results: dict[str, dict[str, list[tuple[int, Simulation]]]] = {}
    audits = {}

    def save(
        period: str, fold: int, system: str, result: RiskSimulation, audit: dict[str, Any]
    ) -> None:
        folder = directory / period / f"fold_{fold}" / system
        folder.mkdir(parents=True, exist_ok=False)
        result.simulation.equity.to_csv(folder / "equity.csv", index=False)
        result.simulation.trades.to_csv(folder / "trades.csv", index=False)
        result.events.to_csv(folder / "events.csv", index=False)
        report = {
            "period": period,
            "fold": fold,
            "system": system,
            "metrics": result.simulation.metrics,
            "audit": audit,
            "status": "completed",
            "negative_net_return": result.simulation.metrics["net_return"] < 0,
        }
        write_new_json(folder / "experiment.json", report)
        record(report)
        results.setdefault(period, {}).setdefault(system, []).append((fold, result.simulation))

    try:
        reference_frames = {
            b.metadata["version"]: risk_features(_reference_frame(raw, b)) for b in bundles
        }
        matched_caps = {}
        windows = [
            ("development", i + 1, a, z, False)
            for i, (a, z) in enumerate(zip(dates[:-1], dates[1:], strict=True))
        ]
        windows.append(("observed_historical", 1, dates[-1], end, True))
        for period, fold, start, stop, historical in windows:
            if historical:
                for bundle in bundles:
                    version = bundle.metadata["version"]
                    original = aggregate(results["development"][version + "__original"])["metrics"]
                    primary = aggregate(
                        results["development"][version + "__" + research["primary_policy"]]
                    )["metrics"]
                    ratio = (
                        primary["mean_capital_exposure"] / original["mean_capital_exposure"]
                        if original["mean_capital_exposure"] > 0
                        else 0.0
                    )
                    matched_caps[version] = min(1.0, ratio)
                write_new_json(
                    directory / "matched_caps.json",
                    {
                        "method": "development_mean_exposure_ratio_vs_original",
                        "caps": matched_caps,
                        "test_used": False,
                    },
                )
            for suffix, scenario_costs in [("", costs), ("_stress", stress)]:
                benchmark = risk_features(raw)
                for name, p in [("buy_hold", np.ones(len(raw))), ("cash", np.zeros(len(raw)))]:
                    result = run_overlay(
                        benchmark, p, start, stop, scenario_costs, RiskPolicy(), len(raw) + 1
                    )
                    save(period + suffix, fold, name, result, {"status": "no_model"})
            for bundle in bundles:
                version = bundle.metadata["version"]
                frame = reference_frames[version]
                p, audit = _reference_probabilities(
                    frame, bundle, start, research["seed"], historical
                )
                fits += not historical
                p[frame.reference_time < start] = 0.0
                labelled = (
                    (frame.reference_time >= start)
                    & (frame.reference_time < stop)
                    & (frame.target_time < stop)
                )
                metrics = classification_metrics(
                    frame.loc[labelled, "target"].to_numpy(), p[labelled.to_numpy()]
                )
                audit = {
                    **audit,
                    "prediction_metrics": metrics,
                    "probabilities_unchanged_across_policies": True,
                }
                audits.setdefault(period, {}).setdefault(version, []).append(audit)
                forecasts = frame.loc[
                    (frame.reference_time >= start) & (frame.reference_time < stop),
                    ["timestamp", "reference_time", "risk_volatility", "risk_atr"],
                ].copy()
                forecasts["probability_up"] = p[
                    (frame.reference_time >= start) & (frame.reference_time < stop)
                ]
                output = directory / period / f"fold_{fold}" / f"{version}_forecasts.csv"
                output.parent.mkdir(parents=True, exist_ok=True)
                forecasts.to_csv(output, index=False)
                candidates = policies.copy()
                if historical and matched_caps[version] > 0:
                    candidates.append(
                        RiskPolicy(name="matched_original", capital_cap=matched_caps[version])
                    )
                for policy in candidates:
                    for suffix, scenario_costs in [("", costs), ("_stress", stress)]:
                        result = run_overlay(
                            frame,
                            p,
                            start,
                            stop,
                            scenario_costs,
                            policy,
                            bundle.metadata["contract"]["horizon"],
                            bundle.metadata["config"]["backtest"]["threshold"],
                        )
                        save(
                            period + suffix,
                            fold,
                            version + "__" + policy.name,
                            result,
                            {**audit, "policy": asdict(policy)},
                        )
        if (
            _protected_hashes(settings) != protected
            or reservation_hashes(settings.root) != reserved
            or sha256(source) != plan["source_sha256"]
        ):
            raise DataError("Cambió un modelo, dato o reserva protegida durante el estudio")
        periods = {}
        for period, systems in results.items():
            combined = {name: aggregate(items) for name, items in systems.items()}
            for name, item in combined.items():
                item.pop("_curve").to_csv(directory / f"{period}__{name}__returns.csv", index=False)
            periods[period] = {"systems": combined}
        primary_version = research["primary_reference"]
        historical_systems = periods["observed_historical"]["systems"]
        before = historical_systems[primary_version + "__original"]["metrics"]
        after = historical_systems[primary_version + "__" + research["primary_policy"]]["metrics"]
        baseline_name = primary_version + "__matched_original"
        matched = historical_systems.get(baseline_name, {}).get("metrics")
        reduction = (
            after["max_drawdown"] > before["max_drawdown"]
            and after["net_return"] > before["net_return"]
        )
        development_primary = periods["development"]["systems"][
            primary_version + "__" + research["primary_policy"]
        ]
        favorable = (
            reduction
            and after["net_return"] > 0
            and (after["profit_factor"] or 0) > 1
            and development_primary["positive_fold_fraction"] >= 0.6
        )
        diagnosis = {
            "classification": "FAVORABLE"
            if favorable
            else "MIXTO"
            if reduction
            else "DESFAVORABLE",
            "reference": primary_version,
            "policy": research["primary_policy"],
            "before": before,
            "after": after,
            "matched_original": matched,
            "predictive_capacity_changed": False,
            "independent_advantage_confirmed": False,
            "reason": "Exposure, allocation and execution decisions change; model probabilities are identical; lower loss is not predictive improvement",
        }
        summary = {
            "run_id": run_id,
            "directory": str(directory.resolve()),
            "status": "completed",
            "protocol": protocol,
            "plan_sha256": sha256(directory / "plan.json"),
            "periods": periods,
            "prediction_audits": audits,
            "matched_caps_from_development": matched_caps,
            "diagnosis": diagnosis,
            "costs": asdict(costs),
            "stress_costs": asdict(stress),
            "actual_fits": fits,
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
            "run_id": run_id,
            "status": "failed",
            "exception": type(exc).__name__,
            "reason": str(exc),
            "actual_fits": fits,
        }
        record(failure)
        write_new_json(directory / "failure.json", failure)
        raise
