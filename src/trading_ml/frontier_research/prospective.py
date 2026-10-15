from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from trading_ml.advanced_research.data import recover_microstructure
from trading_ml.common import frame_hash, read_json, sha256, write_new_json
from trading_ml.errors import DataError, LeakageError
from trading_ml.frontier_research.features import CROSS, LOCAL, multi_asset_features
from trading_ml.frontier_research.models import fit_model, predict_probability
from trading_ml.frontier_research.targets import research_targets
from trading_ml.frontier_research.validation import (
    evaluation_rows,
    prediction_rows,
    temporal_partitions,
)
from trading_ml.models.calibration import apply_calibration, fit_sigmoid
from trading_ml.momentum.config import ResearchCosts

COMPONENTS = [
    "logistic_cross",
    "xgboost_cross",
    "random_forest_cross",
    "lightgbm_cross",
    "catboost_cross",
]


def freeze_systems(directory, targets, selection, protocol, consume):
    cutoff = pd.Timestamp(protocol["research"]["historical_end"], tz="UTC")
    count, systems = 0, {}
    for key, chosen in selection.items():
        symbol, objective = key.split("__")
        frame = targets[symbol][objective]
        ids = (
            COMPONENTS if chosen["recipe"] in {"equal_ensemble", "stacking"} else [chosen["recipe"]]
        )
        states, cal_predictions = {}, {}
        for name in ids:
            recipe = next(rec for rec in protocol["recipes"] if rec["id"] == name)
            train, cal = temporal_partitions(
                frame,
                cutoff,
                embargo_days=protocol["research"]["embargo_days"],
                calibration_days=protocol["research"]["calibration_days"],
                rolling_days=recipe.get("rolling_days"),
            )
            columns = LOCAL + CROSS if recipe["features"] == "cross" else LOCAL
            adjusted = {**recipe, "parameters": recipe.get("parameters", {}).copy()}
            if recipe.get("balance"):
                adjusted["parameters"]["scale_pos_weight"] = float(
                    (train.target == 0).sum() / (train.target == 1).sum()
                )
            consume()
            count += 1
            model = fit_model(train, columns, "target", adjusted)
            raw = predict_probability(model, cal, columns)
            cal_predictions[name] = raw
            consume()
            count += 1
            calibration = fit_sigmoid(raw, cal.target.to_numpy())
            model_dir = directory / "frozen" / key / name
            model_dir.mkdir(parents=True, exist_ok=False)
            joblib.dump(model, model_dir / "model.joblib")
            states[name] = {
                "path": str((model_dir / "model.joblib").relative_to(directory)),
                "sha256": sha256(model_dir / "model.joblib"),
                "columns": columns,
                "parameters": adjusted,
                "calibration": calibration,
                "train_sha256": frame_hash(train[columns + ["target", "target_time"]]),
                "train_last_target": train.target_time.max().isoformat(),
                "calibration_last_target": cal.target_time.max().isoformat(),
            }
        state = {"selected": chosen, "components": states}
        if chosen["recipe"] == "stacking":
            consume()
            count += 1
            stack = LogisticRegression(C=0.1, random_state=42).fit(
                np.column_stack([cal_predictions[n] for n in COMPONENTS]), cal.target
            )
            path = directory / "frozen" / key / "stack.joblib"
            joblib.dump(stack, path)
            state["stack"] = {"path": str(path.relative_to(directory)), "sha256": sha256(path)}
        systems[key] = state
    write_new_json(
        directory / "prospective.json",
        {
            "kind": "frontier_research_only",
            "start": protocol["research"]["prospective_start"],
            "end": protocol["research"]["prospective_end"],
            "frozen_at": datetime.now(UTC).isoformat(),
            "cutoff": cutoff.isoformat(),
            "systems": systems,
            "protocol": protocol,
            "independent_final_test_opened": False,
            "live_trading": False,
            "automatic_promotion": False,
        },
    )
    return count


def require_future_available(plan, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    end = pd.Timestamp(plan["end"], tz="UTC")
    if now < end:
        raise LeakageError(
            f"Test final sellado hasta {end.isoformat()}; no se abre ni sustituye por histórico"
        )


def frozen_probabilities(directory: Path, state: dict, frame: pd.DataFrame):
    raw, calibrated = {}, {}
    for name, component in state["components"].items():
        path = directory / component["path"]
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or sha256(path) != component["sha256"]
        ):
            raise DataError("Artefacto futuro alterado")
        # Load only our locally generated, hash-verified experiment artifacts.
        model = joblib.load(path)
        raw[name] = predict_probability(model, frame, component["columns"])
        calibrated[name] = apply_calibration(raw[name], component["calibration"])
    recipe = state["selected"]["recipe"]
    if recipe == "equal_ensemble":
        return np.mean([calibrated[n] for n in COMPONENTS], axis=0)
    if recipe == "stacking":
        path = directory / state["stack"]["path"]
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or sha256(path) != state["stack"]["sha256"]
        ):
            raise DataError("Stack futuro alterado")
        return joblib.load(path).predict_proba(np.column_stack([raw[n] for n in COMPONENTS]))[:, 1]
    return calibrated[recipe]


def validate_future_history(raw, snapshot, start, end):
    """Require full reserved window and reproduce the frozen historical prefix."""
    first = pd.to_datetime(snapshot.timestamp.iloc[0], utc=True)
    last = pd.to_datetime(snapshot.timestamp.iloc[-1], utc=True)
    if raw.timestamp.min() > first or raw.reference_time.max() < end:
        raise DataError("Faltan datos cerrados del período completo o warmup congelado")
    raw = raw.loc[raw.timestamp >= first].reset_index(drop=True)
    shared = raw.loc[raw.timestamp <= last]
    timestamps = pd.to_datetime(snapshot.timestamp, utc=True)
    columns = [
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
    if (
        len(shared) != len(snapshot)
        or not (shared.timestamp.reset_index(drop=True) == timestamps.reset_index(drop=True)).all()
        or not np.allclose(shared[columns], snapshot[columns], rtol=1e-11, atol=1e-8)
    ):
        raise DataError("El pasado compartido no reproduce el snapshot congelado")
    if (
        not (raw.reference_time <= start).any()
        or not ((raw.reference_time >= start) & (raw.reference_time < end)).any()
    ):
        raise DataError("Test futuro incompleto")
    return raw


def evaluate_frontier_future(settings, directory: Path, btc_raw: Path, eth_raw: Path):
    plan = read_json(directory / "prospective.json")
    require_future_available(plan)
    for relative, digest in read_json(directory / "artifacts.json").items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise DataError("El experimento congelado no está íntegro")
    output = settings.root / "frontier_future_evaluations" / directory.name
    if output.exists():
        raise DataError("El test final ya fue abierto; no permite reintentos selectivos")
    raw = {}
    for symbol, source in [("BTCUSDT", btc_raw), ("ETHUSDT", eth_raw)]:
        asset_settings = replace(settings, market=replace(settings.market, symbol=symbol))
        raw[symbol], _ = recover_microstructure(asset_settings, source)
        snapshot = pd.read_csv(directory / f"{symbol}_causal_features.csv")
        raw[symbol] = validate_future_history(
            raw[symbol],
            snapshot,
            pd.Timestamp(plan["start"], tz="UTC"),
            pd.Timestamp(plan["end"], tz="UTC"),
        )
    frames = multi_asset_features(raw)
    costs = ResearchCosts(**plan["protocol"]["costs"])
    targets = {s: research_targets(f, costs) for s, f in frames.items()}
    start, end = pd.Timestamp(plan["start"], tz="UTC"), pd.Timestamp(plan["end"], tz="UTC")
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(
        output / "opened.json",
        {
            "opened_at": datetime.now(UTC).isoformat(),
            "plan_sha256": sha256(directory / "prospective.json"),
        },
    )
    from trading_ml.frontier_research.study import _save_sim, economic, metrics

    results = {}
    for key, state in plan["systems"].items():
        symbol, objective = key.split("__")
        frame = targets[symbol][objective]
        observed = evaluation_rows(frame, start, end)
        forecast = prediction_rows(frame, start, end)
        p = frozen_probabilities(directory, state, forecast)
        full = np.full(len(frame), 0.5)
        full[forecast.index] = p
        simulation = economic(frame, full, start, end, costs, state["selected"]["threshold"])
        _save_sim(output / key, simulation)
        results[key] = {
            "predictive": metrics(observed.target, p[forecast.index.get_indexer(observed.index)]),
            "economic": simulation.metrics,
            "eligible_before_final_test": state["selected"]["accepted"],
        }
    summary = {"systems": results, "retrained": False, "live_trading": False, "promoted": False}
    write_new_json(output / "summary.json", summary)
    return summary
