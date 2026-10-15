from __future__ import annotations

import logging
import tomllib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.advanced_research.data import recover_microstructure
from trading_ml.advanced_research.features import CORE, FLOW, MOMENTUM, ORIGINAL
from trading_ml.advanced_research.statistics import drift_diagnostic
from trading_ml.common import (
    code_provenance,
    environment,
    frame_hash,
    sha256,
    sidecar,
    write_new_json,
)
from trading_ml.config import Settings
from trading_ml.config.settings import Features
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import BootstrapConfig
from trading_ml.feature_research.diagnostics import (
    binary_loss,
    correlation_pairs,
    permutation_diagnostic,
    prune_correlated,
)
from trading_ml.feature_research.features import FAMILIES, ORIGINAL_GROUPS, build_feature_panel
from trading_ml.features import feature_columns
from trading_ml.mean_reversion.study import _protected_hashes
from trading_ml.models import load_model
from trading_ml.models.baselines import fit_reference_models
from trading_ml.training.recipe import protect_reserved_labels

LOG = logging.getLogger(__name__)
ADDITIONS = list(dict.fromkeys(c for group in FAMILIES.values() for c in group))
ALL = ORIGINAL + ADDITIONS
SENSITIVITY = ["original", "all", "nested_choice", "compact_inner"]
BASELINES = ["constant_05", "train_prevalence", "logistic_l2"]


def fixed_recipes() -> dict[str, list[str]]:
    return {
        "original": ORIGINAL.copy(),
        **{f"add_{name}": ORIGINAL + members for name, members in FAMILIES.items()},
        "all": ALL.copy(),
        **{
            f"all_without_{name}": [c for c in ALL if c not in members]
            for name, members in FAMILIES.items()
        },
        **{
            f"original_without_{name}": [c for c in ORIGINAL if c not in members]
            for name, members in ORIGINAL_GROUPS.items()
        },
        "enhanced_control": feature_columns(Features(enhanced=True)),
        "smc_existing_control": feature_columns(Features(smc=True)),
        "frontier_local_control": list(dict.fromkeys(CORE + FLOW + MOMENTUM)),
    }


def temporal_train(frame, start, embargo_days=6, minimum=240):
    boundary = start - pd.Timedelta(days=embargo_days)
    train = frame.loc[(frame.reference_time < start) & (frame.target_time < boundary)].copy()
    if len(train) < minimum or train.target.nunique() != 2:
        raise DataError("Entrenamiento temporal insuficiente")
    return train


def temporal_score(frame, start, end):
    return frame.loc[
        (frame.reference_time >= start) & (frame.reference_time < end) & (frame.target_time < end)
    ].copy()


def choose_inner(scores, importance, train, research):
    """Selection consumes ONLY inner OOS scores; no outer predictions/labels."""
    original_losses = np.asarray(scores["original"])
    qualifying = [
        name
        for name, losses in scores.items()
        if name != "original"
        and np.all(np.asarray(losses) < original_losses)
        and float(np.mean(original_losses - losses)) >= research["inner_minimum_gain"]
    ]
    chosen = min(qualifying, key=lambda n: (np.mean(scores[n]), n)) if qualifying else "original"
    # Retain individually useful features in BOTH inner windows and both block
    # lengths. This is deliberately conservative; permutation std is not a CI.
    ranking = {
        c: float(np.mean(importance[c])) for c in ALL if np.all(np.asarray(importance[c]) > 0)
    }
    proposed = sorted(ranking, key=lambda c: (-ranking[c], c))
    if len(proposed) >= 2:
        compact, removed = prune_correlated(
            train, proposed, research["correlation_threshold"], ranking
        )
        compact = compact[: research["compact_max_features"]]
    else:
        compact, removed = [], []
    fallback = len(compact) < 2
    if fallback:
        compact = ORIGINAL.copy()
    return {
        "chosen": chosen,
        "qualifying": qualifying,
        "inner_log_losses": scores,
        "compact_columns": compact,
        "compact_fallback_original": fallback,
        "positive_inner_importance": ranking,
        "compact_removed_correlated": removed,
    }


def validate_protocol(protocol):
    r = protocol["research"]
    dates = pd.to_datetime(r["boundaries"], utc=True)
    if (
        not 4 <= len(dates) <= 9
        or not dates.is_monotonic_increasing
        or dates.duplicated().any()
        or dates[-1] > pd.Timestamp.now(tz="UTC")
        or r["embargo_days"] < 6
        or r["inner_days"] < 90
        or r["minimum_train_rows"] < 120
        or not 0 < r["correlation_threshold"] <= 1
        or not 2 <= r["compact_max_features"] <= 30
        or r["inner_minimum_gain"] < 0
        or r["seed"] == r["secondary_seed"]
    ):
        raise DataError("Protocolo temporal de features inválido")
    fits = 2 * (len(dates) - 1) * (20 + len(fixed_recipes()) + 3 + 4 + 1)
    if not fits <= r["maximum_fits"] <= 850:
        raise DataError(f"Presupuesto insuficiente: cota {fits} ajustes")
    BootstrapConfig(
        **{**protocol["bootstrap"], "block_lengths": tuple(protocol["bootstrap"]["block_lengths"])}
    ).validate()
    importance = protocol["importance"]
    if (
        not importance["block_lengths"]
        or any(b < 1 for b in importance["block_lengths"])
        or not 2 <= importance["repeats"] <= 30
        or not 2 <= importance["inner_repeats"] <= 20
        or r["inner_days"] <= 2 * max(importance["block_lengths"])
    ):
        raise DataError("Configuración de permutación inválida")
    a = protocol["acceptance"]
    if (
        not 0 < a["required_fold_fraction"] <= 1
        or a["minimum_log_loss_gain"] <= 0
        or a["minimum_auc_gain"] <= 0
        or not a["require_both_assets"]
        or not a["require_secondary_seed"]
    ):
        raise DataError("Criterios de aceptación inválidos")
    return dates, fits


def _frozen_hashes(settings):
    result = _protected_hashes(settings)
    for directory in [
        "smc_studies",
        "momentum_studies",
        "mean_reversion_studies",
        "advanced_studies",
        "frontier_studies",
    ]:
        for path in (settings.root / directory).rglob("*"):
            if path.is_file() and (
                path.name in {"plan.json", "prospective.json"} or "frozen" in path.parts
            ):
                result[str(path.resolve())] = sha256(path)
    return result


def _load_panel(settings, end):
    if settings.market.interval != "1d" or settings.training.horizon != 1:
        raise DataError("Este protocolo aísla features para dirección diaria h=1")
    raw, provenance = {}, {}
    sources = {
        "BTCUSDT": settings.root / "data" / "raw.csv",
        "ETHUSDT": settings.root / "research_data" / "ETHUSDT_20200101_20261001.csv",
    }
    for symbol, path in sources.items():
        if not path.exists():
            raise DataError(f"Falta el historial verificado: {path}")
        local = replace(settings, market=replace(settings.market, symbol=symbol))
        data, origin = recover_microstructure(local, path)
        data = data.loc[data.reference_time <= end].reset_index(drop=True)
        if data.empty or data.reference_time.iloc[-1] < end:
            raise DataError("Cobertura histórica incompleta")
        raw[symbol], provenance[symbol] = (
            data,
            origin | {"source": str(path.resolve()), "manifest_sha256": sha256(sidecar(path))},
        )
    return build_feature_panel(raw), provenance


class FitBudget:
    def __init__(self, maximum, parameters):
        self.maximum, self.parameters, self.actual = maximum, parameters, 0

    def fit(self, train, columns, seed):
        if self.actual >= self.maximum:
            raise DataError("Presupuesto de ajustes agotado")
        self.actual += 1
        model = XGBClassifier(
            **(
                self.parameters
                | {
                    "random_state": seed,
                    "n_jobs": 1,
                    "tree_method": "hist",
                    "device": "cpu",
                    "eval_metric": "logloss",
                }
            )
        )
        model.fit(train[columns].to_numpy(dtype=float), train.target.to_numpy(dtype=int))
        return model

    def baseline(self, train, columns, seed):
        if self.actual >= self.maximum:
            raise DataError("Presupuesto de ajustes agotado")
        self.actual += 1
        return fit_reference_models(train, columns, seed=seed)


def _inner_selection(frame, outer_start, train, budget, r, importance):
    end = outer_start - pd.Timedelta(days=r["embargo_days"])
    bounds = [end - pd.Timedelta(days=i * r["inner_days"]) for i in [2, 1, 0]]
    candidates = {
        name: columns
        for name, columns in fixed_recipes().items()
        if name == "original" or name == "all" or name.startswith("add_")
    }
    losses = {name: [] for name in candidates}
    individual = {c: [] for c in ALL}
    windows = []
    for start, finish in zip(bounds[:-1], bounds[1:], strict=True):
        inner_train = temporal_train(frame, start, r["embargo_days"], r["minimum_train_rows"])
        val = temporal_score(frame, start, finish)
        if len(val) < 2 * max(importance["block_lengths"]):
            raise DataError("Validación interna insuficiente")
        for name, columns in candidates.items():
            model = budget.fit(inner_train, columns, r["seed"])
            p = model.predict_proba(val[columns].to_numpy(dtype=float))[:, 1]
            losses[name].append(float(binary_loss(val.target, p).mean()))
            if name == "all":
                rows = permutation_diagnostic(
                    model,
                    val,
                    columns,
                    {c: [c] for c in ALL},
                    blocks=importance["block_lengths"],
                    repeats=importance["inner_repeats"],
                    seed=r["seed"],
                )
                for row in rows:
                    individual[row["name"]].append(row["delta_loss_mean"])
        windows.append(
            {
                "start": start.isoformat(),
                "end": finish.isoformat(),
                "train_rows": len(inner_train),
                "validation_rows": len(val),
                "last_train_target": inner_train.target_time.max().isoformat(),
                "last_validation_target": val.target_time.max().isoformat(),
            }
        )
    return choose_inner(losses, individual, train, r) | {
        "windows": windows,
        "individual_importance": individual,
    }


def run_feature_study(settings: Settings, protocol_path: Path) -> dict[str, Any]:
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    dates, maximum_planned = validate_protocol(protocol)
    r, imp = protocol["research"], protocol["importance"]
    bundle = load_model(settings.root, r["reference_version"])
    if bundle.metadata["features"] != ORIGINAL:
        raise DataError("La referencia debe conservar las catorce features originales")
    parameters = bundle.metadata["selected_parameters"].copy()
    if parameters["n_estimators"] > 160:
        raise DataError("La receta excede el presupuesto de árboles")
    # Reject any evaluation in a reserved future window BEFORE reading market data.
    guard = pd.DataFrame(
        {"reference_time": [dates[0]], "target_time": [dates[-1] - pd.Timedelta(nanoseconds=1)]}
    )
    protect_reserved_labels(settings.root, guard)
    protected = _frozen_hashes(settings)
    panel, provenance = _load_panel(settings, dates[-1])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    directory = settings.root / "feature_studies" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    common = list(dict.fromkeys(c for columns in fixed_recipes().values() for c in columns))
    plan = {
        "run_id": run_id,
        "protocol": protocol,
        "parameters": parameters,
        "reference_version": r["reference_version"],
        "families": FAMILIES,
        "fixed_recipes": fixed_recipes(),
        "common_columns": common,
        "maximum_planned_fits": maximum_planned,
        "data": provenance,
        "environment": environment(),
        "code": code_provenance(),
        "historical_data_previously_observed": True,
        "independent_test_claimed": False,
        "protected_before": protected,
    }
    write_new_json(directory / "plan.json", plan)
    budget = FitBudget(r["maximum_fits"], parameters)
    predictions, fits, permutations, diagnostics, selections = [], [], [], [], []
    quality = {}
    for symbol, full in panel.items():
        ready = full[common].notna().all(axis=1)
        quality[symbol] = {
            "raw_rows": len(full),
            "common_ready_rows": int(ready.sum()),
            "missing_by_column": full[common].isna().sum().to_dict(),
            "same_rows_for_every_recipe": True,
        }
        # Feature availability is causal and independent of target presence.
        full[["timestamp", "reference_time", "weekly_available_time", *common]].to_csv(
            directory / f"{symbol}_features.csv", index=False
        )
        frame = full.loc[ready & full.target.notna()].copy()
        for fold, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True)):
            train = temporal_train(frame, start, r["embargo_days"], r["minimum_train_rows"])
            evaluation = temporal_score(frame, start, end)
            if len(evaluation) < 2 * max(
                imp["block_lengths"] + protocol["bootstrap"]["block_lengths"]
            ):
                raise DataError("Fold externo insuficiente")
            protect_reserved_labels(settings.root, train, evaluation)
            selection = _inner_selection(frame, start, train, budget, r, imp)
            pruned, removed = prune_correlated(train, ORIGINAL, r["correlation_threshold"])
            recipes = fixed_recipes() | {
                "original_pruned": pruned,
                "nested_choice": fixed_recipes()[selection["chosen"]],
                "compact_inner": selection["compact_columns"],
            }
            selections.append(
                {
                    "symbol": symbol,
                    "fold": fold,
                    **selection,
                    "original_pruned_columns": pruned,
                    "original_pruned_removed": removed,
                }
            )
            fold_dir = directory / symbol / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            write_new_json(fold_dir / "selection.json", selections[-1])
            diagnostics.append(
                {
                    "symbol": symbol,
                    "fold": fold,
                    "correlated_training_pairs": correlation_pairs(train, common),
                    "drift": drift_diagnostic(train, evaluation, common),
                    "train_target_max": train.target_time.max().isoformat(),
                    "evaluation_start": start.isoformat(),
                    "evaluation_end": end.isoformat(),
                    "train_frame_sha256": frame_hash(train),
                    "evaluation_frame_sha256": frame_hash(evaluation),
                }
            )
            cached = {}
            for seed in [r["seed"], r["secondary_seed"]]:
                for name, columns in recipes.items():
                    if seed != r["seed"] and name not in SENSITIVITY:
                        continue
                    key = (seed, tuple(columns))
                    if key not in cached:
                        model = budget.fit(train, columns, seed)
                        p = model.predict_proba(evaluation[columns].to_numpy(dtype=float))[:, 1]
                        train_p = model.predict_proba(train[columns].to_numpy(dtype=float))[:, 1]
                        model_file = fold_dir / f"{name}_seed{seed}.ubj"
                        model.save_model(model_file)
                        cached[key] = model, p, train_p, str(model_file.relative_to(directory))
                    model, p, train_p, filename = cached[key]
                    train_metrics = classification_metrics(train.target, train_p)
                    test_metrics = classification_metrics(evaluation.target, p)
                    fits.append(
                        {
                            "symbol": symbol,
                            "fold": fold,
                            "recipe": name,
                            "seed": seed,
                            "columns": columns,
                            "model_file": filename,
                            "train": train_metrics,
                            "oos": test_metrics,
                            "loss_generalization_gap": test_metrics["log_loss"]
                            - train_metrics["log_loss"],
                            "gain": dict(
                                zip(columns, map(float, model.feature_importances_), strict=True)
                            ),
                        }
                    )
                    rows = evaluation[
                        ["reference_time", "target_time", "target", "fe_regime", "fe_vol_regime"]
                    ].copy()
                    rows["symbol"], rows["fold"], rows["recipe"], rows["seed"] = (
                        symbol,
                        fold,
                        name,
                        seed,
                    )
                    rows["probability"] = p
                    predictions.append(rows)
                    if seed == r["seed"] and name in {"original", "all", "compact_inner"}:
                        groups = {c: [c] for c in columns}
                        groups |= {
                            f"family_{k}": v
                            for k, v in (
                                ORIGINAL_GROUPS if name == "original" else FAMILIES
                            ).items()
                        }
                        for row in permutation_diagnostic(
                            model,
                            evaluation,
                            columns,
                            groups,
                            blocks=imp["block_lengths"],
                            repeats=imp["repeats"],
                            seed=seed,
                        ):
                            permutations.append(
                                {
                                    "symbol": symbol,
                                    "fold": fold,
                                    "recipe": name,
                                    "scope": "outer_oos_descriptive_only",
                                    **row,
                                }
                            )
            reference = budget.baseline(train, ORIGINAL, r["seed"])
            for name, p in reference.probabilities(evaluation).items():
                rows = evaluation[
                    ["reference_time", "target_time", "target", "fe_regime", "fe_vol_regime"]
                ].copy()
                rows["symbol"], rows["fold"], rows["recipe"], rows["seed"] = (
                    symbol,
                    fold,
                    name,
                    r["seed"],
                )
                rows["probability"] = p
                predictions.append(rows)
            write_new_json(fold_dir / "baseline.json", reference.to_dict())
            write_new_json(
                fold_dir / "fits.json",
                [f for f in fits if f["symbol"] == symbol and f["fold"] == fold],
            )
            LOG.warning(
                "Features %s fold %s listo: train=%s OOS=%s, ajustes=%s/%s",
                symbol,
                fold,
                len(train),
                len(evaluation),
                budget.actual,
                budget.maximum,
            )
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    write_new_json(directory / "quality.json", quality)
    write_new_json(directory / "importance.json", permutations)
    write_new_json(directory / "diagnostics.json", diagnostics)
    write_new_json(directory / "selections.json", selections)
    from trading_ml.feature_research.report import analyze, write_report

    analysis = analyze(predictions, fits, permutations, protocol)
    if protected != _frozen_hashes(settings):
        raise LeakageError("Se alteraron modelos o reservas durante la investigación")
    summary = {
        "run_id": run_id,
        "directory": str(directory.resolve()),
        "actual_fits": budget.actual,
        "maximum_planned_fits": maximum_planned,
        "parameters": parameters,
        "data_quality": quality,
        "existing_models_preserved": True,
        "prospective_unchanged": True,
        "prospective_evaluated": False,
        "promoted": False,
        "historical_data_previously_observed": True,
        "independent_advantage_confirmed": False,
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
