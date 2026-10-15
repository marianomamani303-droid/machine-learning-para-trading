from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trading_ml.common import (
    code_provenance,
    environment,
    frame_hash,
    read_json,
    sha256,
    write_new_json,
)
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.models.calibration import apply_calibration, fit_sigmoid
from trading_ml.multimarket.data import execution_source, load_snapshot
from trading_ml.multimarket.execution import Costs, Engine
from trading_ml.multimarket.features import features, recipes
from trading_ml.multimarket.markets import instrument
from trading_ml.multimarket.protocol import load_protocol
from trading_ml.multimarket.study import _fit, dataset, labelled, require_outcomes
from trading_ml.training.recipe import protect_reserved_labels


def train_market(
    settings,
    protocol_path,
    symbol,
    interval,
    cutoff,
    recipe="price",
    target="direction",
    horizon=1,
    data=None,
):
    protocol = load_protocol(protocol_path)
    symbol = instrument(symbol).symbol
    if (
        recipe not in recipes(symbol)
        or target not in {"direction", "net_profit", "triple_barrier"}
        or not 1 <= horizon <= 10
    ):
        raise DataError("Receta/target/horizonte no soportados")
    cutoff = pd.to_datetime(cutoff, utc=True)
    if cutoff > pd.Timestamp.now(tz="UTC") or cutoff > pd.to_datetime(protocol["end"], utc=True):
        raise DataError("Cutoff fuera del rango cerrado del protocolo")
    source = data or dataset(settings, protocol, symbol, interval)
    raw, meta = load_snapshot(source)
    _compatible(meta, symbol, interval)
    frame = features(raw.loc[raw.reference_time <= cutoff].reset_index(drop=True), symbol)
    costs = Costs(**protocol["costs"])
    barrier = target == "triple_barrier"
    execution, multiplier = execution_source(raw, meta)
    execution = execution.loc[execution.reference_time <= cutoff].reset_index(drop=True)
    labelled_frame = labelled(
        frame,
        Engine(frame, symbol, costs, execution_frame=execution, bars_per_horizon=multiplier),
        horizon,
        "direction" if target == "direction" else "net_profit",
        barrier,
    )
    columns = recipes(symbol)[recipe]
    end = cutoff - pd.Timedelta(days=protocol["embargo_days"])
    cal_start = end - pd.Timedelta(days=protocol["calibration_days"])
    train = labelled_frame.loc[
        labelled_frame.target_time < cal_start - pd.Timedelta(days=protocol["embargo_days"])
    ].dropna(subset=columns + ["target"])
    calibration = labelled_frame.loc[
        (labelled_frame.reference_time >= cal_start) & (labelled_frame.target_time < end)
    ].dropna(subset=columns + ["target"])
    if len(train) < protocol["min_train_rows"] or len(calibration) < 20:
        raise DataError("Historial insuficiente")
    protect_reserved_labels(settings.root, train, calibration)
    model = _fit(train, columns, protocol)
    state = fit_sigmoid(
        model.predict_proba(calibration[columns])[:, 1],
        calibration.target.to_numpy(int),
        protocol["seed"],
    )
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = settings.root / "multimarket" / "models" / symbol / interval / version
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "model.ubj"
    model.save_model(path)
    metadata = {
        "symbol": symbol,
        "interval": interval,
        "feature_schema": "multimarket_price_v1",
        "instrument": meta["instrument"],
        "specification": {
            "name": target + "_" + recipe,
            "columns": columns,
            "horizon": horizon,
            "target": "direction" if target == "direction" else "net_profit",
            "barrier": barrier,
        },
        "model_sha256": sha256(path),
        "source": str(source.resolve()),
        "data_sha256": sha256(source),
        "last_train_target": train.target_time.max().isoformat(),
        "last_calibration_target": calibration.target_time.max().isoformat(),
        "first_calibration_reference": calibration.reference_time.min().isoformat(),
        "fit_rows": len(train),
        "fit_sha256": frame_hash(train),
        "calibration_rows": len(calibration),
        "calibration": state,
        "costs": asdict(costs),
        "cutoff": cutoff.isoformat(),
        "code": code_provenance(),
        "environment": environment(),
        "production": False,
        "auto_promoted": False,
    }
    write_new_json(path.with_suffix(".json"), metadata)
    return {
        "model": str(path.resolve()),
        "fit_rows": len(train),
        "calibration_rows": len(calibration),
        "production": False,
    }


def _compatible(metadata, symbol, interval, provider=None):
    if metadata["instrument"]["symbol"] != symbol or metadata["interval"] != interval:
        raise DataError("Mercado/intervalo del dataset incompatible con el modelo")
    if provider is not None and metadata["instrument"]["provider"] != provider:
        raise DataError("Proveedor del dataset incompatible con el modelo")


def load_candidate(path):
    metadata = read_json(path.with_suffix(".json"))
    if (
        sha256(path) != metadata["model_sha256"]
        or metadata.get("feature_schema", "multimarket_price_v1") != "multimarket_price_v1"
    ):
        raise DataError("Artefacto de modelo alterado o incompatible")
    model = XGBClassifier()
    model.load_model(path)
    return model, metadata


def probabilities(path, frame, calibrated=False):
    model, meta = load_candidate(path)
    cols = meta["specification"]["columns"]
    local = frame.copy()
    parent = None
    if "parent_probability" in cols:
        parent_path = path.parent / meta["parent_model"]
        parent, _ = probabilities(parent_path, frame, False)
        local["parent_probability"] = parent
    values = local[cols].to_numpy(float)
    if not np.isfinite(values).all():
        raise DataError("Features incompletas para predecir")
    raw = model.predict_proba(local[cols])[:, 1]
    p = apply_calibration(raw, meta.get("calibration")) if calibrated else raw
    return p, parent


def predict_market(path, data, reference=None, calibrated=False):
    _, meta = load_candidate(path)
    raw, origin = load_snapshot(data)
    _compatible(
        origin, meta["symbol"], meta["interval"], meta.get("instrument", {}).get("provider")
    )
    now = pd.Timestamp.now(tz="UTC")
    when = pd.to_datetime(reference, utc=True) if reference else min(raw.reference_time.max(), now)
    if when > now:
        raise LeakageError("No se puede predecir con una referencia futura")
    # Cut input BEFORE building features; this operation never constructs labels.
    frame = features(raw.loc[raw.reference_time <= when].reset_index(drop=True), meta["symbol"])
    if frame.empty or frame.reference_time.iloc[-1] != when:
        raise DataError("No hay una vela cerrada exactamente en la referencia pedida")
    last_fit = pd.to_datetime(meta["last_train_target"], utc=True)
    if meta.get("last_calibration_target"):
        last_fit = max(last_fit, pd.to_datetime(meta["last_calibration_target"], utc=True))
    if when <= last_fit:
        raise LeakageError(
            "Predicción anterior o igual al último resultado usado para ajustar/calibrar"
        )
    row = frame.iloc[[-1]]
    probability, parent = probabilities(path, row, calibrated)
    allow = bool(probability[0] >= 0.55 and (parent is None or parent[0] >= 0.55))
    return {
        "symbol": meta["symbol"],
        "interval": meta["interval"],
        "reference_time": when.isoformat(),
        "probability": float(probability[0]),
        "probability_target": meta["specification"]["target"],
        "calibrated": bool(calibrated and meta.get("calibration") is not None),
        "parent_probability": float(parent[0]) if parent is not None else None,
        "research_long_signal": allow,
        "staleness_hours": float((now - when).total_seconds() / 3600),
        "execution": "research only; next executable ask, subject to spread and risk limits",
        "orders_submitted": 0,
        "model_sha256": sha256(path),
        "data_sha256": sha256(data),
    }


def evaluate_market(settings, path, data, start, end, calibrated=False, output=None):
    _, meta = load_candidate(path)
    raw, origin = load_snapshot(data)
    _compatible(
        origin, meta["symbol"], meta["interval"], meta.get("instrument", {}).get("provider")
    )
    frame = features(raw, meta["symbol"])
    execution, multiplier = execution_source(raw, origin)
    first, last = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    last_fit = pd.to_datetime(meta["last_train_target"], utc=True)
    if meta.get("last_calibration_target"):
        last_fit = max(last_fit, pd.to_datetime(meta["last_calibration_target"], utc=True))
    if first <= last_fit or first >= last or last > execution.reference_time.max():
        raise LeakageError(
            "Período de evaluación no es posterior al ajuste/calibración o no tiene cobertura"
        )
    spec = meta["specification"]
    engine = Engine(
        frame,
        meta["symbol"],
        Costs(**meta["costs"]),
        execution_frame=execution,
        bars_per_horizon=multiplier,
    )
    labels = labelled(
        frame,
        engine,
        spec["horizon"],
        "direction" if spec["target"] == "direction" else "net_profit",
        spec["barrier"],
    )
    needed = [c for c in spec["columns"] if c != "parent_probability"]
    candidate_rows = labels.loc[
        (labels.reference_time >= first) & (labels.reference_time < last - pd.Timedelta(days=20))
    ].dropna(subset=needed)
    require_outcomes(candidate_rows, engine, spec["target"])
    observed = candidate_rows.dropna(subset=["target"])
    if observed.empty or observed.target_time.max() >= last:
        raise DataError("Evaluación vacía o buffer de etiquetas insuficiente")
    protect_reserved_labels(settings.root, observed)
    p, parent = probabilities(path, observed, calibrated)
    predictions = observed[["source_index", "reference_time", "target_time", "target"]].copy()
    predictions["probability"] = p if parent is None else np.where(parent >= 0.55, p, 0.0)
    summary, trades, curve = engine.portfolio(
        predictions, spec["horizon"], spec["barrier"], 0.55, last
    )
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = output or settings.root / "multimarket" / "evaluations" / version
    destination.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    trades.to_csv(destination / "trades.csv", index=False)
    curve.to_csv(destination / "equity.csv", index=False)
    summary.update(
        {
            "model": str(path.resolve()),
            "model_sha256": sha256(path),
            "data": str(data.resolve()),
            "data_sha256": sha256(data),
            "start": first.isoformat(),
            "end": last.isoformat(),
            "calibrated": bool(calibrated and meta.get("calibration") is not None),
            "classification": classification_metrics(
                predictions.target.to_numpy(int), predictions.probability.to_numpy()
            ),
            "directory": str(destination.resolve()),
            "adopted": False,
        }
    )
    write_new_json(destination / "summary.json", summary)
    return summary
