from __future__ import annotations

import logging
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from trading_ml.common import (
    code_provenance,
    environment,
    frame_hash,
    object_hash,
    sha256,
    write_new_json,
)
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import BootstrapConfig
from trading_ml.feature_research.study import FitBudget, _load_panel, temporal_score, temporal_train
from trading_ml.models import load_model
from trading_ml.structure_regime.execution import execute
from trading_ml.structure_regime.features import (
    CLUSTER_COLUMNS,
    CLUSTER_INPUT,
    RULE_COLUMNS,
    STRUCTURE,
    apply_clusters,
    extend_structure,
    fit_clusters,
    structure_gate,
)
from trading_ml.structure_regime.report import REGIME_RECIPES, analyze, markdown
from trading_ml.training.recipe import protect_reserved_labels

LOG = logging.getLogger(__name__)


def validate_protocol(protocol):
    r, a = protocol["research"], protocol["acceptance"]
    dates = pd.to_datetime(r["boundaries"], utc=True)
    if (
        len(dates) != 7
        or not dates.is_monotonic_increasing
        or dates.duplicated().any()
        or dates[-1] > pd.Timestamp.now(tz="UTC")
        or r["embargo_days"] < 6
        or r["minimum_train_rows"] < 240
        or len(r["seeds"]) != 2
        or len(set(r["seeds"])) != 2
        or not 480 <= r["maximum_fits"] <= 500
        or r["minimum_expert_rows"] < 80
        or r["minimum_expert_class"] < 20
        or r["shrinkage_rows"] < 200
        or a["minimum_log_loss_gain"] < 0.003
        or a["minimum_auc_gain"] < 0.005
        or a["required_fold_fraction"] < 2 / 3
        or a["required_fold_fraction"] > 1
        or a["minimum_trades"] < 30
    ):
        raise DataError("Protocolo de estructura/regímenes inválido")
    BootstrapConfig(
        **(protocol["bootstrap"] | {"block_lengths": tuple(protocol["bootstrap"]["block_lengths"])})
    ).validate()
    costs = protocol["costs"]
    if (
        not np.isfinite(list(costs.values())).all()
        or not 0.5 <= costs["threshold"] <= 1
        or min(costs[k] for k in ["commission_bps", "slippage_bps", "spread_bps"]) < 0
        or costs["initial_capital"] <= 0
        or 2 * (costs["slippage_bps"] + costs["spread_bps"] / 2) >= 10000
        or 2 * costs["commission_bps"] >= 10000
    ):
        raise DataError("Costos inválidos")
    return dates


def snapshot(root, excluded=None):
    root = root.resolve()
    excluded = excluded.resolve() if excluded is not None else None
    paths = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and (excluded is None or not p.is_relative_to(excluded))
    ]
    # Many small OneDrive files: parallel reads avoid serial per-file I/O latency.
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(map(str, paths), pool.map(sha256, paths), strict=True))


def fit_experts(
    train, evaluation, state, columns, global_probability, budget, seed, config, directory
):
    p = np.asarray(global_probability, dtype=float).copy()
    audits = []
    for value in sorted(set(train[state]) | set(evaluation[state])):
        subset = train.loc[train[state] == value]
        selected = evaluation[state].to_numpy() == value
        counts = subset.target.value_counts()
        sufficient = (
            len(subset) >= config["minimum_expert_rows"]
            and len(counts) == 2
            and counts.min() >= config["minimum_expert_class"]
        )
        weight = len(subset) / (len(subset) + config["shrinkage_rows"]) if sufficient else 0.0
        audit = {
            "state": str(value),
            "rows": len(subset),
            "class_counts": counts.to_dict(),
            "evaluation_rows": int(selected.sum()),
            "fallback_global": not sufficient,
            "expert_weight": weight,
            "train_hash": frame_hash(subset),
            "last_target": subset.target_time.max().isoformat() if len(subset) else None,
        }
        if sufficient:
            model = budget.fit(subset, columns, seed)
            model.save_model(directory / f"expert_{value}.ubj")
            train_p = model.predict_proba(subset[columns].to_numpy(dtype=float))[:, 1]
            audit["train_metrics"] = classification_metrics(subset.target, train_p)
            if selected.any():
                expert = model.predict_proba(
                    evaluation.loc[selected, columns].to_numpy(dtype=float)
                )[:, 1]
                p[selected] = weight * expert + (1 - weight) * p[selected]
                audit["oos_metrics"] = classification_metrics(
                    evaluation.loc[selected].target, expert
                )
                audit["loss_gap"] = (
                    audit["oos_metrics"]["log_loss"] - audit["train_metrics"]["log_loss"]
                )
        audits.append(audit)
    return p, audits


def run_structure_regime_study(settings, protocol_path):
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    dates = validate_protocol(protocol)
    r = protocol["research"]
    bundle = load_model(settings.root, r["reference_version"])
    original = bundle.metadata["features"]
    if len(original) != 14 or settings.market.interval != "1d" or settings.training.horizon != 1:
        raise DataError("La investigación requiere original diario h=1")
    guard = pd.DataFrame(
        {"reference_time": [dates[0]], "target_time": [dates[-1] - pd.Timedelta(nanoseconds=1)]}
    )
    protect_reserved_labels(settings.root, guard)
    LOG.warning("Verificando hashes de todos los artefactos previos")
    protected = snapshot(settings.root)
    LOG.warning("Patrimonio registrado: %s archivos; recuperando datos verificados", len(protected))
    panel, provenance = _load_panel(settings, dates[-1])
    panel = {
        symbol: extend_structure(frame, protocol["structure"]) for symbol, frame in panel.items()
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    directory = settings.root / "structure_regime_studies" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    parameters = bundle.metadata["selected_parameters"]
    common = list(dict.fromkeys(original + STRUCTURE + CLUSTER_INPUT + RULE_COLUMNS))
    from trading_ml.smc.detectors import smc_columns

    recipes = {
        "original": original,
        "smc_control": original + smc_columns(),
        "structure": original + STRUCTURE,
        "regime_rules": original + RULE_COLUMNS,
        "regime_cluster": original + CLUSTER_COLUMNS,
    }
    plan = {
        "run_id": run_id,
        "protocol": protocol,
        "parameters": parameters,
        "recipes": recipes,
        "data": provenance,
        "code": code_provenance(),
        "environment": environment(),
        "common_columns": common,
        "historical_data_previously_observed": True,
        "maximum_planned_fits": 480,
        "protected_files": protected,
        "registered_model_comparison_after": bundle.metadata["selection_label_end"],
        "research_plan_sha256": sha256(Path("STRUCTURE_REGIME_PLAN.md")),
    }
    write_new_json(directory / "plan.json", plan)
    budget = FitBudget(r["maximum_fits"], parameters)
    predictions, curves, trades, fits, cached, quality = [], [], [], [], [], {}

    def record(symbol, fold, seed, name, full, evaluation, p, *, gate=None, scores=True):
        if scores:
            rows = evaluation[
                ["timestamp", "reference_time", "target_time", "target", "sr_state"]
            ].copy()
            rows = rows.assign(symbol=symbol, fold=fold, seed=seed, recipe=name, probability=p)
            predictions.append(rows)
        for scenario, multiplier, spread in [
            ("base", 1, True),
            ("double_cost", 2, True),
            ("legacy_no_spread", 1, False),
        ]:
            simulation = execute(
                full,
                evaluation,
                p,
                protocol["costs"],
                multiplier=multiplier,
                gate=gate,
                spread=spread,
            )
            curve = simulation.equity.assign(
                symbol=symbol,
                fold=fold,
                seed=seed,
                recipe=name,
                scenario=scenario,
                step=np.arange(len(simulation.equity)),
            )
            ledger = simulation.trades.assign(
                symbol=symbol, fold=fold, seed=seed, recipe=name, scenario=scenario
            )
            curves.append(curve)
            trades.append(ledger)

    for symbol, full in panel.items():
        ready = full.sr_ready & full[common].notna().all(axis=1)
        frame = full.loc[ready & full.target.notna()].copy()
        quality[symbol] = {
            "raw_rows": len(full),
            "common_ready_rows": int(ready.sum()),
            "events": full.loc[ready, STRUCTURE].sum().to_dict(),
            "same_rows_for_all_recipes": True,
        }
        full.loc[ready, ["timestamp", "reference_time", *common, "sr_state"]].to_csv(
            directory / f"{symbol}_features.csv", index=False
        )
        for fold, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True)):
            train = temporal_train(frame, start, r["embargo_days"], r["minimum_train_rows"])
            evaluation = temporal_score(frame, start, end)
            if len(evaluation) < 2 * max(protocol["bootstrap"]["block_lengths"]):
                raise DataError("Fold insuficiente")
            protect_reserved_labels(settings.root, train, evaluation)
            for seed in r["seeds"]:
                fold_dir = directory / symbol / f"fold_{fold}" / f"seed_{seed}"
                fold_dir.mkdir(parents=True)
                budget.actual += 1  # KMeans is an estimated model too.
                scaler, cluster = fit_clusters(train, seed)
                training, scoring = (
                    apply_clusters(train, scaler, cluster),
                    apply_clusters(evaluation, scaler, cluster),
                )
                write_new_json(
                    fold_dir / "clusters.json",
                    {
                        "input_columns": CLUSTER_INPUT,
                        "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(),
                        "centers": cluster.cluster_centers_.tolist(),
                        "centers_original_scale": scaler.inverse_transform(
                            cluster.cluster_centers_
                        ).tolist(),
                        "train_hash": frame_hash(train),
                        "train_last_target": train.target_time.max().isoformat(),
                        "evaluation_start": start.isoformat(),
                        "train_counts": training.sr_cluster_state.value_counts().to_dict(),
                        "evaluation_counts": scoring.sr_cluster_state.value_counts().to_dict(),
                        "rule_transitions": float(
                            scoring.sr_state.ne(scoring.sr_state.shift()).iloc[1:].mean()
                        ),
                        "cluster_transitions": float(
                            scoring.sr_cluster_state.ne(scoring.sr_cluster_state.shift())
                            .iloc[1:]
                            .mean()
                        ),
                    },
                )
                global_p = None
                for name, columns in recipes.items():
                    model = budget.fit(training, columns, seed)
                    model.save_model(fold_dir / f"{name}.ubj")
                    p = model.predict_proba(scoring[columns].to_numpy(dtype=float))[:, 1]
                    training_metrics = classification_metrics(
                        training.target,
                        model.predict_proba(training[columns].to_numpy(dtype=float))[:, 1],
                    )
                    oos = classification_metrics(scoring.target, p)
                    fits.append(
                        {
                            "symbol": symbol,
                            "fold": fold,
                            "seed": seed,
                            "recipe": name,
                            "columns": columns,
                            "train_hash": frame_hash(training),
                            "last_train_target": train.target_time.max().isoformat(),
                            "evaluation_start": start.isoformat(),
                            "train": training_metrics,
                            "oos": oos,
                            "loss_gap": oos["log_loss"] - training_metrics["log_loss"],
                        }
                    )
                    record(symbol, fold, seed, name, full, scoring, p)
                    if name == "original":
                        global_p = p
                for method, state in [("rules", "sr_state"), ("cluster", "sr_cluster_state")]:
                    expert_dir = fold_dir / f"specialists_{method}"
                    expert_dir.mkdir()
                    p, audit = fit_experts(
                        training, scoring, state, original, global_p, budget, seed, r, expert_dir
                    )
                    fits.append(
                        {
                            "symbol": symbol,
                            "fold": fold,
                            "seed": seed,
                            "recipe": f"specialists_{method}",
                            "experts": audit,
                            "columns": original,
                            "state_column": state,
                            "global_fallback_model": "original.ubj",
                        }
                    )
                    record(symbol, fold, seed, f"specialists_{method}", full, scoring, p)
                baseline = budget.baseline(training, original, seed)
                write_new_json(fold_dir / "baselines.json", baseline.to_dict())
                for name, p in baseline.probabilities(scoring).items():
                    record(symbol, fold, seed, name, full, scoring, p)
                record(
                    symbol,
                    fold,
                    seed,
                    "structure_gate",
                    full,
                    scoring,
                    global_p,
                    gate=structure_gate(scoring),
                    scores=False,
                )
                cached.append((symbol, fold, seed, full, training, scoring, fold_dir))
            LOG.warning(
                "Estructura/regímenes %s fold %s: train=%s OOS=%s; ajustes=%s",
                symbol,
                fold,
                len(train),
                len(evaluation),
                budget.actual,
            )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    curve_frame, trade_frame = (
        pd.concat(curves, ignore_index=True),
        pd.concat(trades, ignore_index=True),
    )
    # Save expensive work before bootstrap; a report failure must not lose the fits.
    prediction_frame.to_csv(directory / "predictions.csv", index=False)
    curve_frame.to_csv(directory / "curves.csv", index=False)
    trade_frame.to_csv(directory / "trades.csv", index=False)
    write_new_json(directory / "fits.json", fits)
    LOG.warning("Ajustes completos; calculando intervalos y criterios predefinidos")
    analysis = analyze(prediction_frame, curve_frame, trade_frame, protocol)
    eligible = [name for name in REGIME_RECIPES if name in analysis["qualified"]]
    combination = {
        "status": "no ejecutado por falta de evidencia en experimentos 1 y 2",
        "structure_qualified": "structure" in analysis["qualified"],
        "eligible_regimes": eligible,
    }
    if "structure" in analysis["qualified"] and eligible:
        # Deterministic priority declared in REGIME_RECIPES; no best outer-score selection.
        chosen = eligible[0]
        columns = (
            original + STRUCTURE + (RULE_COLUMNS if chosen.endswith("rules") else CLUSTER_COLUMNS)
        )
        combination = {
            "status": "ejecutado como exploración condicionada",
            "regime": chosen,
            "columns": columns,
        }
        for symbol, fold, seed, full, training, scoring, fold_dir in cached:
            model = budget.fit(training, columns, seed)
            model.save_model(fold_dir / "combined.ubj")
            p = model.predict_proba(scoring[columns].to_numpy(dtype=float))[:, 1]
            record(symbol, fold, seed, "combined", full, scoring, p)
        prediction_frame = pd.concat(predictions, ignore_index=True)
        curve_frame, trade_frame = (
            pd.concat(curves, ignore_index=True),
            pd.concat(trades, ignore_index=True),
        )
        prediction_frame.to_csv(directory / "predictions_with_combination.csv", index=False)
        curve_frame.to_csv(directory / "curves_with_combination.csv", index=False)
        trade_frame.to_csv(directory / "trades_with_combination.csv", index=False)
        analysis = analyze(prediction_frame, curve_frame, trade_frame, protocol)
    # Registered BTC model is valid only after its selection information cutoff.
    frozen_rows = prediction_frame.loc[
        (prediction_frame.symbol == "BTCUSDT")
        & (prediction_frame.seed == r["seeds"][0])
        & (prediction_frame.recipe == "original")
        & (prediction_frame.reference_time > pd.Timestamp(bundle.metadata["selection_label_end"]))
    ].copy()
    frozen = panel["BTCUSDT"].merge(
        frozen_rows[["reference_time", "fold"]], on="reference_time", validate="one_to_one"
    )
    frozen_p = bundle.model.predict_proba(frozen[original])[:, 1]
    frozen_rows["probability"] = frozen_p
    frozen_rows.to_csv(directory / "registered_predictions.csv", index=False)
    registered = {
        "version": r["reference_version"],
        "rows": len(frozen),
        "start": frozen.reference_time.min().isoformat(),
        "end": frozen.reference_time.max().isoformat(),
        "metrics": classification_metrics(frozen.target, frozen_p),
        "comparison": {},
        "previously_observed": True,
    }
    for name, part in prediction_frame.loc[
        (prediction_frame.symbol == "BTCUSDT")
        & (prediction_frame.seed == r["seeds"][0])
        & prediction_frame.reference_time.isin(frozen.reference_time)
    ].groupby("recipe"):
        aligned = frozen[["reference_time"]].merge(part, on="reference_time", validate="one_to_one")
        simulated = execute(panel["BTCUSDT"], frozen, aligned.probability, protocol["costs"])
        registered["comparison"][name] = {
            "metrics": classification_metrics(aligned.target, aligned.probability),
            "execution": simulated.summary,
        }
    frozen_execution = execute(panel["BTCUSDT"], frozen, frozen_p, protocol["costs"])
    registered["execution"] = frozen_execution.summary
    frozen_execution.equity.to_csv(directory / "registered_curve.csv", index=False)
    frozen_execution.trades.to_csv(directory / "registered_trades.csv", index=False)
    LOG.warning("Verificando preservación de %s archivos previos", len(protected))
    after = snapshot(settings.root, excluded=directory)
    if protected != after:
        changed = sorted(
            k for k in protected.keys() | after.keys() if protected.get(k) != after.get(k)
        )
        raise LeakageError(f"Cambió el patrimonio previo: {changed[:5]}")
    summary = {
        "run_id": run_id,
        "directory": str(directory.resolve()),
        "actual_fits": budget.actual,
        "maximum_planned_fits": 480,
        "combination": combination,
        "quality": quality,
        "registered_comparison": registered,
        "existing_models_preserved": True,
        "previous_artifacts_preserved": True,
        "protected_file_count": len(protected),
        "protected_hash": object_hash(protected),
        "prospective_unchanged": True,
        "prospective_evaluated": False,
        "promoted": False,
        "independent_advantage_confirmed": False,
        **analysis,
    }
    write_new_json(directory / "summary.json", summary)
    (directory / "report.md").write_text(markdown(summary), encoding="utf-8")
    write_new_json(
        directory / "artifacts.json",
        {
            str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        },
    )
    return summary
