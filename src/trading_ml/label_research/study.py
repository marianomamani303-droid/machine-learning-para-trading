from __future__ import annotations

import logging
import tomllib
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from trading_ml.advanced_research.features import ORIGINAL
from trading_ml.common import code_provenance, environment, sha256, write_new_json
from trading_ml.config import Settings
from trading_ml.errors import DataError, LeakageError
from trading_ml.feature_research.study import _frozen_hashes, _load_panel
from trading_ml.label_research.events import (
    attach_labels,
    causal_scales,
    exit_rules,
    label_events,
)
from trading_ml.label_research.execution import simulate_events
from trading_ml.label_research.models import Budget, fit_event_model, forecasts, meta_pool
from trading_ml.models import load_model
from trading_ml.momentum.config import ResearchCosts
from trading_ml.training.recipe import protect_reserved_labels

LOG = logging.getLogger(__name__)
META_COLUMNS = ORIGINAL + ["base_probability"]


def validate_protocol(protocol):
    r = protocol["research"]
    dates = pd.to_datetime(r["boundaries"], utc=True)
    quarters = pd.date_range(pd.Timestamp(r["oof_start"], tz="UTC"), dates[-1], freq="QS")
    if (
        not 4 <= len(dates) <= 9
        or not dates.is_monotonic_increasing
        or dates.duplicated().any()
        or dates[-1] > pd.Timestamp.now(tz="UTC")
        or not set(dates).issubset(quarters)
        or quarters[0] >= dates[0]
        or r["embargo_days"] < 6
        or r["calibration_days"] < 120
        or r["minimum_train_rows"] < 120
        or r["minimum_meta_train_rows"] < 40
        or r["minimum_meta_calibration_rows"] < 20
        or r["maximum_horizon"] != 10
        or r["thresholds"] != [0.50, 0.55, 0.60, 0.65]
    ):
        raise DataError("Protocolo temporal de etiquetas inválido")
    folds = 2 * (len(dates) - 1)
    # One raw OOF direction model/quarter; outer reserved direction, alternatives,
    # uniqueness control, logistic direction; calibrations and four meta fits.
    planned = 2 * (len(quarters) - 1) + folds * (2 * (len(exit_rules()) + 2) + 1 + 8)
    if not planned <= r["maximum_fits"] <= 650:
        raise DataError(f"Presupuesto insuficiente para etiquetas: {planned}")
    ResearchCosts(**protocol["costs"]).validate()
    b = protocol["bootstrap"]
    if (
        not 100 <= b["samples"] <= 5000
        or b["block_lengths"] != [14, 30]
        or b["confidence"] != 0.95
        or protocol["stress"]["friction_multiplier"] < 1
    ):
        raise DataError("Bootstrap/stress incompatibles")
    expected = ["tb_vol_h5_cal_ev", "meta_net_h1_logistic_cal_55", "meta_tb_vol_h5_logistic_cal_55"]
    if r["primary_candidates"] != expected:
        raise DataError("Comparaciones principales no preregistradas")
    a = protocol["acceptance"]
    if (
        a["minimum_trades"] < 30
        or not 0 < a["minimum_positive_fold_fraction"] <= 1
        or not 0 < a["minimum_coverage_vs_parent"] <= 1
        or a["minimum_average_net_trade"] < 0
        or not a["require_both_assets"]
        or not a["require_positive_stressed_return"]
    ):
        raise DataError("Aceptación económica inválida")
    if not all(
        a[k]
        for k in [
            "require_better_trade_precision",
            "require_better_mean_trade_vs_parent",
            "require_predictive_skill_vs_prior",
        ]
    ):
        raise DataError("Se requieren controles de precisión y calidad probabilística")
    return dates, quarters, planned


def _specifications(probabilities, fitted, frame, full_base, rule_map):
    """Thresholds are fixed before evaluation; nothing here reads a label."""
    n = len(frame)
    output = {}
    fixed = rule_map["net_h1"]
    output["original"] = (full_base, fixed, 0.55, None, "original")
    for threshold in [0.50, 0.60, 0.65]:
        output[f"original_th{int(threshold * 100)}"] = (
            full_base,
            fixed,
            threshold,
            None,
            "original",
        )
    for variant in ["raw", "cal"]:
        p = probabilities[f"direction_reserved_{variant}"]
        output[f"direction_reserved_{variant}_55"] = (p, fixed, 0.55, None, "original")
    output["direction_reserved_cal_60"] = (
        probabilities["direction_reserved_cal"],
        fixed,
        0.60,
        None,
        "original",
    )
    output["logistic_direction_55"] = (
        probabilities["logistic_direction_raw"],
        fixed,
        0.55,
        None,
        "original",
    )
    for name, rule in rule_map.items():
        parent = "original" if name == "net_h1" else f"control_{name}"
        if parent != "original":
            output[parent] = (full_base, rule, 0.55, None, "original")
        for variant in ["raw", "cal"]:
            output[f"{name}_{variant}_55"] = (
                probabilities[f"{name}_{variant}"],
                rule,
                0.55,
                None,
                parent,
            )
        output[f"{name}_cal_60"] = (probabilities[f"{name}_cal"], rule, 0.60, None, parent)
        model = fitted[name]
        p = probabilities[f"{name}_cal"]
        expected = (
            p * model.audit["positive_payoff_mean"] + (1 - p) * model.audit["negative_payoff_mean"]
        )
        output[f"{name}_cal_ev"] = (p, rule, 0.0, np.isfinite(expected) & (expected > 0), parent)
    for variant in ["raw", "cal"]:
        output[f"tb_vol_h5_unique_{variant}_55"] = (
            probabilities[f"tb_vol_h5_unique_{variant}"],
            rule_map["tb_vol_h5"],
            0.55,
            None,
            "control_tb_vol_h5",
        )
    for target in ["net_h1", "tb_vol_h5"]:
        parent = "original" if target == "net_h1" else "control_tb_vol_h5"
        for kind in ["logistic", "xgboost"]:
            for variant, threshold in [("raw", 0.55), ("cal", 0.55), ("cal", 0.60)]:
                meta_p = probabilities[f"meta_{target}_{kind}_{variant}"]
                gate = np.isfinite(meta_p) & (meta_p >= threshold)
                # Only the original directional parent proposes trades. Meta p is
                # a filter, never a replacement direction or a probability of up.
                output[f"meta_{target}_{kind}_{variant}_{int(threshold * 100)}"] = (
                    full_base,
                    rule_map[target],
                    0.55,
                    gate,
                    parent,
                )
    output["cash"] = (np.zeros(n), fixed, 1.0, None, "original")
    output["always_tb_vol_h5"] = (
        np.ones(n),
        rule_map["tb_vol_h5"],
        0.55,
        None,
        "control_tb_vol_h5",
    )
    return output


def run_label_study(settings: Settings, protocol_path: Path):
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    dates, quarters, planned = validate_protocol(protocol)
    r = protocol["research"]
    # Protect every requested interval before reading prices, including OOF seeds.
    guard = pd.DataFrame(
        {"reference_time": [quarters[0]], "target_time": [dates[-1] - pd.Timedelta(nanoseconds=1)]}
    )
    protect_reserved_labels(settings.root, guard)
    if settings.market.interval != "1d" or settings.training.horizon != 1:
        raise DataError("El estudio compara el sistema diario original h=1")
    bundle = load_model(settings.root, r["reference_version"])
    if bundle.metadata["features"] != ORIGINAL:
        raise DataError("El control requiere las 14 features originales")
    parameters = bundle.metadata["selected_parameters"].copy()
    if parameters["n_estimators"] > 160 or protocol["meta_xgboost"]["n_estimators"] > 80:
        raise DataError("Presupuesto de árboles excedido")
    protected = _frozen_hashes(settings)
    panel, origin = _load_panel(settings, dates[-1])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    directory = settings.root / "label_studies" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    costs = ResearchCosts(**protocol["costs"])
    factor = protocol["stress"]["friction_multiplier"]
    stress = replace(
        costs,
        commission_bps=costs.commission_bps * factor,
        slippage_bps=costs.slippage_bps * factor,
        spread_bps=costs.spread_bps * factor,
    )
    rules = {rule.name: rule for rule in exit_rules()}
    plan = {
        "run_id": run_id,
        "protocol": protocol,
        "parameters": parameters,
        "features": ORIGINAL,
        "meta_features": META_COLUMNS,
        "rules": [asdict(rule) for rule in rules.values()],
        "data": origin,
        "planned_maximum_fits": planned,
        "code": code_provenance(),
        "environment": environment(),
        "protected_before": protected,
        "historical_data_previously_observed": True,
        "primary_tie_policy": "stop_first",
        "entry_tail_days": r["maximum_horizon"] + 1,
        "operating_costs": "Spot: no assumed borrowing/funding; annual holding configurable, default zero",
        "future_prices_used_for_labels_only": True,
    }
    write_new_json(directory / "plan.json", plan)
    budget = Budget(r["maximum_fits"])
    forecast_rows, records, fold_results, all_curves, all_trades, pools = [], [], [], [], [], []
    for symbol, raw in panel.items():
        frame = causal_scales(raw)
        frame["research_ready"] = (
            frame[ORIGINAL + ["label_sigma", "label_atr_fraction"]].notna().all(axis=1)
        )
        output = directory / symbol
        output.mkdir()
        snapshot_columns = [
            "timestamp",
            "reference_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label_sigma",
            "label_atr_fraction",
            "fe_regime",
            "research_ready",
            *ORIGINAL,
        ]
        frame[snapshot_columns].to_csv(output / "causal_features.csv", index=False)
        events, targets = {}, {}
        for name, rule in rules.items():
            events[name] = label_events(frame, rule, costs)
            events[name].to_csv(output / f"labels_{name}.csv", index=False)
            targets[name] = attach_labels(frame, events[name])
        direction = targets["net_h1"].copy()
        direction["target"] = frame.target
        direction["target_time"] = frame.target_time
        targets["direction"] = direction
        base_models, oof_parts = {}, []
        for start, end in zip(quarters[:-1], quarters[1:], strict=True):
            model, audit = fit_event_model(
                direction,
                start,
                ORIGINAL,
                protocol,
                parameters,
                budget,
                settings.root,
                reserve=False,
            )
            model_dir = output / "base_oof" / start.strftime("%Y%m%d")
            model.save(model_dir)
            base_models[start] = model
            selected = frame.loc[(frame.reference_time >= start) & (frame.reference_time < end)]
            p, _ = forecasts(model, selected)
            oof_parts.append(
                pd.DataFrame(
                    {
                        "source_index": selected.index,
                        "reference_time": selected.reference_time,
                        "probability": p,
                        "model_file": str(model_dir.relative_to(directory)),
                        "model_cutoff": start.isoformat(),
                        "last_train_target": audit["train_target_end"],
                    }
                )
            )
        oof = pd.concat(oof_parts, ignore_index=True)
        oof.to_csv(output / "base_oof.csv", index=False)
        for fold, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True)):
            fold_dir = output / f"fold_{fold}"
            fold_dir.mkdir()
            evaluation = frame.loc[(frame.reference_time >= start) & (frame.reference_time < end)]
            raw_full, _ = forecasts(base_models[start], frame)
            probabilities, fitted, model_targets = {}, {}, {}
            probabilities["original"] = raw_full
            fitted["original"] = base_models[start]
            model_targets["original"] = "direction"
            fitted_paths = {
                "original": str(
                    (output / "base_oof" / start.strftime("%Y%m%d")).relative_to(directory)
                )
            }
            models = [
                ("direction_reserved", "direction", False, "xgboost"),
                ("logistic_direction", "direction", False, "logistic"),
            ]
            models += [(name, name, False, "xgboost") for name in rules]
            models += [("tb_vol_h5_unique", "tb_vol_h5", True, "xgboost")]
            for name, target, weighted, kind in models:
                reserve = kind != "logistic"
                model, audit = fit_event_model(
                    targets[target],
                    start,
                    ORIGINAL,
                    protocol,
                    parameters,
                    budget,
                    settings.root,
                    reserve=reserve,
                    weighted=weighted,
                    kind=kind,
                )
                model.save(fold_dir / name)
                fitted[name], model_targets[name] = model, target
                fitted_paths[name] = str((fold_dir / name).relative_to(directory))
                raw_p, cal_p = forecasts(model, frame)
                probabilities[f"{name}_raw"], probabilities[f"{name}_cal"] = raw_p, cal_p
            for target in ["net_h1", "tb_vol_h5"]:
                pool = meta_pool(frame, oof, events[target], start)
                pool[
                    [
                        "reference_time",
                        "target_time",
                        "target",
                        "unit_net_return",
                        "base_probability",
                        "oof_model_file",
                    ]
                ].to_csv(
                    fold_dir / f"meta_pool_{target}.csv", index=True, index_label="source_index"
                )
                meta_frame = frame.assign(base_probability=raw_full)
                for kind in ["logistic", "xgboost"]:
                    name = f"meta_{target}_{kind}"
                    model, audit = fit_event_model(
                        pool,
                        start,
                        META_COLUMNS,
                        protocol,
                        protocol["meta_xgboost"],
                        budget,
                        settings.root,
                        kind=kind,
                        meta=True,
                    )
                    if model is not None:
                        model.save(fold_dir / name)
                        fitted_paths[name] = str((fold_dir / name).relative_to(directory))
                    raw_p, cal_p = forecasts(model, meta_frame)
                    probabilities[name + "_raw"], probabilities[name + "_cal"] = raw_p, cal_p
                    fitted[name], model_targets[name] = model, target
                    pools.append(
                        {
                            "symbol": symbol,
                            "fold": fold,
                            "name": name,
                            "pool_rows": len(pool),
                            "audit": audit,
                        }
                    )
            for key, p in probabilities.items():
                suffix = "_cal" if key.endswith("_cal") else "_raw" if key.endswith("_raw") else ""
                name = key.removesuffix(suffix) if suffix else key
                model = fitted[name]
                target = model_targets[name]
                labeled = targets[target]
                rows = labeled.loc[
                    evaluation.index, ["reference_time", "target_time", "target", "unit_net_return"]
                ].copy()
                rows["source_index"] = evaluation.index
                rows["probability"] = p[evaluation.index]
                rows["symbol"], rows["fold"], rows["forecast_id"], rows["target_name"] = (
                    symbol,
                    fold,
                    key,
                    target,
                )
                rows["base_offer"] = raw_full[evaluation.index] >= 0.55
                rows["calendar_eligible"] = rows.reference_time < end - pd.Timedelta(
                    days=r["maximum_horizon"] + 1
                )
                rows["train_prior"] = model.audit["train_prior"] if model else np.nan
                forecast_rows.append(rows)
                records.append(
                    {
                        "symbol": symbol,
                        "fold": fold,
                        "forecast_id": key,
                        "target_name": target,
                        "model_path": fitted_paths.get(name),
                        "calibrated": suffix == "_cal",
                        "meta": name.startswith("meta_"),
                        "audit": model.audit
                        if model
                        else next(
                            x["audit"]
                            for x in pools
                            if x["symbol"] == symbol and x["fold"] == fold and x["name"] == name
                        ),
                    }
                )
            specs = _specifications(probabilities, fitted, frame, raw_full, rules)
            for name, (p, rule, threshold, gate, parent) in specs.items():
                for scenario, current_costs, tie in [
                    ("main", costs, "stop_first"),
                    ("stress", stress, "stop_first"),
                    ("take_first", costs, "take_first"),
                ]:
                    # Fixed horizon/close rules are tie invariant; reuse is safe.
                    if scenario == "take_first" and rule.mode in {"fixed", "close"}:
                        continue
                    simulation = simulate_events(
                        frame,
                        p,
                        rule,
                        start,
                        end,
                        current_costs,
                        threshold=threshold,
                        gate=gate,
                        maximum_horizon=r["maximum_horizon"],
                        tie=tie,
                    )
                    path = fold_dir / "simulations" / scenario / name
                    path.mkdir(parents=True)
                    simulation.curve.to_csv(path / "equity.csv", index=False)
                    simulation.trades.to_csv(path / "trades.csv", index=False)
                    if scenario == "main":
                        simulation.decisions.to_csv(path / "decisions.csv", index=False)
                    write_new_json(path / "metrics.json", simulation.metrics)
                    fold_results.append(
                        {
                            "symbol": symbol,
                            "fold": fold,
                            "strategy": name,
                            "scenario": scenario,
                            "parent": parent,
                            "rule": rule.name,
                            "threshold": threshold,
                            "metrics": simulation.metrics,
                        }
                    )
                    curve = simulation.curve.iloc[1:].copy()
                    curve["symbol"], curve["fold"], curve["strategy"], curve["scenario"] = (
                        symbol,
                        fold,
                        name,
                        scenario,
                    )
                    all_curves.append(curve)
                    trades = simulation.trades.copy()
                    trades["symbol"], trades["fold"], trades["strategy"], trades["scenario"] = (
                        symbol,
                        fold,
                        name,
                        scenario,
                    )
                    all_trades.append(trades)
            write_new_json(
                fold_dir / "model_records.json",
                [x for x in records if x["symbol"] == symbol and x["fold"] == fold],
            )
            LOG.warning(
                "Label study %s fold %s listo: %s ajustes/%s",
                symbol,
                fold,
                budget.actual,
                budget.maximum,
            )
    predictions = pd.concat(forecast_rows, ignore_index=True)
    curves = pd.concat(all_curves, ignore_index=True)
    trades = pd.concat(all_trades, ignore_index=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    curves.to_csv(directory / "curves.csv", index=False)
    trades.to_csv(directory / "trades.csv", index=False)
    write_new_json(directory / "fold_results.json", fold_results)
    write_new_json(directory / "model_records.json", records)
    write_new_json(directory / "meta_audit.json", pools)
    from trading_ml.label_research.report import analyze, write_report

    analysis = analyze(predictions, curves, trades, fold_results, protocol)
    if protected != _frozen_hashes(settings):
        raise LeakageError("Se alteraron modelos o reservas históricos")
    summary = {
        "run_id": run_id,
        "directory": str(directory.resolve()),
        "actual_fits": budget.actual,
        "planned_maximum_fits": planned,
        "plan_sha256": sha256(directory / "plan.json"),
        "costs": asdict(costs),
        "existing_models_preserved": True,
        "prospective_unchanged": True,
        "prospective_evaluated": False,
        "promoted": False,
        "independent_advantage_confirmed": False,
        "historical_data_previously_observed": True,
        **analysis,
    }
    write_new_json(directory / "summary.json", summary)
    write_report(directory / "report.md", summary, protocol)
    write_new_json(
        directory / "artifacts.json",
        {
            str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        },
    )
    return summary
