from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from trading_ml.config.settings import INTERVALS
from trading_ml.errors import DataError
from trading_ml.online import store
from trading_ml.online.adaptive import COHORT_DEFINITIONS, MODEL_SPECS
from trading_ml.online.config import (
    FEATURE_NAMES,
    cohort_for_config,
    feature_names_for_config,
    load_config,
    protocol,
    utc,
)
from trading_ml.online.data import (
    COLUMNS,
    features,
    fetch_candles,
    realized_target_metrics,
    validate_candles,
)
from trading_ml.online.evaluation import (
    evaluate_models,
    evaluate_target_specific_records,
)
from trading_ml.online.models import learn, new_models, predict, predict_structured


def _state(config: dict, state_path: Path | None) -> Path:
    return Path(state_path or config["online"]["state_path"])


def _contract(connection, config: dict) -> None:
    expected = protocol(config)
    actual = store.get_meta(connection, "protocol")
    if actual != expected or store.get_meta(connection, "protocol_hash") != store.digest(expected):
        raise DataError(
            "La configuración no coincide con el protocolo persistido; usar un estado nuevo"
        )
    if store.get_meta(connection, "schema_version") != store.SCHEMA_VERSION:
        raise DataError("Versión del estado incompatible")


def init_state(config_path: Path, state_path: Path | None = None) -> dict:
    config = load_config(config_path)
    path = _state(config, state_path)
    connection = store.connect(path, create=True)
    try:
        connection.execute("BEGIN IMMEDIATE")
        if store.get_meta(connection, "schema_version") is None:
            if connection.execute("SELECT count(*) FROM models").fetchone()[0]:
                raise DataError(
                    "Estado parcial/incompatible, no se inicializará sobre modelos existentes"
                )
            proto = protocol(config)
            fnames = feature_names_for_config(config)
            cohort = cohort_for_config(config)
            options = {
                "static_warmup_updates": config["online"].get("static_warmup_updates", 120),
            }
            store.set_meta(connection, "schema_version", store.SCHEMA_VERSION)
            store.set_meta(connection, "protocol", proto)
            store.set_meta(connection, "protocol_hash", store.digest(proto))
            store.set_meta(connection, "mode", None)
            store.set_meta(connection, "cursor", None)
            store.save_models(
                connection,
                new_models(
                    len(fnames),
                    config["online"]["seed"],
                    cohort=cohort,
                    options=options,
                ),
            )
            store.event(
                connection, pd.Timestamp.now(tz="UTC").isoformat(), "initialized", proto
            )
        _contract(connection, config)
        store.load_models(connection)
        connection.commit()
    finally:
        connection.close()
    return status(path)


def _insert_bars(connection, frame: pd.DataFrame) -> int:
    added = 0
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(COLUMNS, values, strict=True))
        row["timestamp"] = pd.Timestamp(row["timestamp"]).isoformat()
        row.update({key: float(row[key]) for key in COLUMNS[1:]})
        old = connection.execute(
            "SELECT digest FROM bars WHERE timestamp=?", (row["timestamp"],)
        ).fetchone()
        checksum = store.row_digest(row)
        if old is not None:
            if old[0] != checksum:
                raise DataError(f"Revisión de vela cerrada ya persistida: {row['timestamp']}")
            continue
        connection.execute("INSERT INTO bars VALUES(?,?,?,?,?,?,?)", (*row.values(), checksum))
        added += 1
    return added


def _save_prediction(connection, row: dict) -> None:
    row["digest"] = store.row_digest(row)
    columns = list(row)
    connection.execute(
        f"INSERT INTO predictions({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
        tuple(row.values()),
    )


def _resolve(
    connection, models: dict, bars: pd.DataFrame, at: pd.Timestamp, delta: pd.Timedelta
) -> int:
    confirmed = 0
    if bars.empty:
        return 0
    # A delayed response is not proof of a gap: wait until the source reaches the target.
    ready_through = min(at, bars.timestamp.iloc[-1] + delta)
    by_open = bars.set_index("timestamp")
    pending = connection.execute(
        "SELECT * FROM predictions WHERE status='pending' AND target_time<=? ORDER BY target_time,model_id",
        (ready_through.isoformat(),),
    ).fetchall()
    for raw in pending:
        row = dict(raw)
        reference = utc(row["reference_time"])
        target = utc(row["target_time"])
        required = pd.date_range(reference, target - delta, freq=delta)
        if not required.isin(by_open.index).all():
            row["status"] = "invalid_gap"
        else:
            target_bars = by_open.loc[required]
            target_close = float(by_open.loc[target - delta, "close"])
            realized_ret, realized_var = realized_target_metrics(
                float(row["reference_close"]), target_bars
            )
            row["target_close"] = target_close
            row["label"] = int(target_close > row["reference_close"])
            row["status"] = "evaluated"
            row["confirmed_at"] = at.isoformat()
            row["learned"] = int(models[row["model_id"]]["status"] == "active")
            model_obj = models[row["model_id"]]
            if model_obj["kind"] in ("beta_frequency", "logistic"):
                learn(model_obj, json.loads(row["features"]), row["label"])
            else:
                learn(
                    model_obj,
                    json.loads(row["features"]),
                    row["label"],
                    realized_return=realized_ret,
                    realized_variance=realized_var,
                )
            confirmed += 1
        row["digest"] = store.row_digest(row)
        connection.execute(
            "UPDATE predictions SET target_close=?,label=?,status=?,confirmed_at=?,learned=?,digest=? WHERE model_id=? AND reference_time=?",
            (
                row["target_close"],
                row["label"],
                row["status"],
                row["confirmed_at"],
                row["learned"],
                row["digest"],
                row["model_id"],
                row["reference_time"],
            ),
        )
    return confirmed


def _evaluate(connection, models: dict, config: dict, at: str) -> list[dict]:
    # Keep only a bounded unevaluated suffix per candidate; past decisions live in events.
    last_targets = [
        model.get("evaluation", {}).get("last_target")
        for model in models.values()
        if model["kind"] != "beta_frequency" and model["status"] == "active"
    ]
    cutoff = min(last_targets) if last_targets and all(last_targets) else ""
    rows = [
        dict(row)
        for row in connection.execute(
            "SELECT model_id,reference_time,target_time,probability,label,status,horizon FROM predictions WHERE status='evaluated' AND reference_time>=? ORDER BY reference_time,model_id",
            (cutoff,),
        )
    ]
    decisions = evaluate_models(
        rows, models, {**config["evaluation"], "horizon": config["market"]["horizon"]}
    )
    for decision in decisions:
        store.event(connection, at, "comparison", decision)
        if decision["decision"] == "validated_predictive":
            snapshot = {
                "id": connection.execute(
                    "SELECT coalesce(max(id),0)+1 FROM validated_snapshots"
                ).fetchone()[0],
                "timestamp": at,
                "model_id": decision["candidate"],
                "payload": store.encode(models[decision["candidate"]]),
                "decision": store.encode(decision),
            }
            connection.execute(
                "INSERT INTO validated_snapshots VALUES(?,?,?,?,?,?)",
                (*snapshot.values(), store.row_digest(snapshot)),
            )
    return decisions


def run_once(
    config_path: Path,
    state_path: Path | None = None,
    *,
    source: Path | None = None,
    replay: bool = False,
    now: str | None = None,
) -> dict:
    config = load_config(config_path)
    path = _state(config, state_path)
    if replay and source is None:
        raise DataError("Replay requiere una fuente CSV explícita")
    at = utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    delta = pd.Timedelta(INTERVALS[config["market"]["interval"]])
    init_state(config_path, path)
    connection = store.connect(path)
    try:
        _contract(connection, config)
        old_mode = store.get_meta(connection, "mode")
        mode = "replay" if replay else "live"
        if old_mode is not None and old_mode != mode:
            raise DataError("No se pueden mezclar replay y live en un mismo estado")
        # Fetch outside the write transaction; concurrent writers are checked again below.
        latest = connection.execute("SELECT max(timestamp) FROM bars").fetchone()[0]
        frame = (
            validate_candles(pd.read_csv(source), config["market"]["interval"], at)
            if source is not None
            else fetch_candles(config, at, latest)
        )
        if now is None:
            # Record issuance after I/O, so network latency cannot backdate a forecast.
            at = pd.Timestamp.now(tz="UTC")
        connection.execute("BEGIN IMMEDIATE")
        _contract(connection, config)
        mode_now = store.get_meta(connection, "mode")
        if mode_now is not None and mode_now != mode:
            raise DataError("No se pueden mezclar replay y live en un mismo estado")
        provider = "csv" if source is not None else "binance_spot_public"
        if store.get_meta(connection, "provider", provider) != provider:
            raise DataError("No se pueden mezclar proveedores en un mismo estado")
        cursor = store.get_meta(connection, "cursor")
        last_poll = store.get_meta(connection, "last_poll")
        if (cursor and at < utc(cursor)) or (last_poll and at < utc(last_poll)):
            raise DataError("El reloj no puede retroceder respecto del estado")
        models = store.load_models(connection)
        # Detect logical corruption before consuming labels, including modified SQLite rows.
        store.check_records(connection)
        added = _insert_bars(connection, frame)
        bars = pd.read_sql_query(
            "SELECT timestamp,open,high,low,close,volume FROM bars ORDER BY timestamp", connection
        )
        bars["timestamp"] = pd.to_datetime(bars.timestamp, utc=True)
        eligible = bars.loc[bars.timestamp + delta <= at]
        candidates = eligible.loc[eligible.timestamp + delta > utc(cursor)] if cursor else eligible
        emitted = confirmed = 0
        forecasts = []
        decisions = []
        start_at = utc(config["online"]["start_at"])
        warmup = config["online"]["warmup_bars"]
        fset = config["online"].get("feature_set", "v1_base8")
        cohort = cohort_for_config(config)
        if replay:
            references = (candidates.timestamp + delta).tolist()
        else:
            # Catch up existing forecasts after downtime, never invent missed live forecasts.
            references = [(eligible.timestamp.iloc[-1] + delta)] if not eligible.empty else []
        for reference in references:
            confirmation_time = reference if replay else at
            confirmed += _resolve(connection, models, eligible, confirmation_time, delta)
            decisions.extend(_evaluate(connection, models, config, confirmation_time.isoformat()))
            if reference < start_at:
                continue
            if (
                not replay
                and (at - reference).total_seconds()
                > config["online"]["max_prediction_delay_seconds"]
            ):
                continue
            if not replay and at >= reference + config["market"]["horizon"] * delta:
                # A stale source cannot make an expired forecast prospective,
                # even if the operational delay tolerance is longer than its horizon.
                continue
            window = eligible.loc[eligible.timestamp < reference].tail(warmup)
            if len(window) != warmup or not (window.timestamp.diff().dropna() == delta).all():
                continue
            values = features(window, feature_set=fset)
            for name, model in models.items():
                if (
                    model["status"] != "active"
                    or connection.execute(
                        "SELECT 1 FROM predictions WHERE model_id=? AND reference_time=?",
                        (name, reference.isoformat()),
                    ).fetchone()
                ):
                    continue
                if cohort == "classic_3":
                    probability = predict(model, values)
                    structured = None
                else:
                    structured = predict_structured(model, values)
                    probability = float(structured["probability"])
                issuance = (
                    reference if replay else (at if now is not None else pd.Timestamp.now(tz="UTC"))
                )
                if not replay and (
                    issuance >= reference + config["market"]["horizon"] * delta
                    or (issuance - reference).total_seconds()
                    > config["online"]["max_prediction_delay_seconds"]
                ):
                    continue
                target_iso = (reference + config["market"]["horizon"] * delta).isoformat()
                _save_prediction(
                    connection,
                    {
                        "model_id": name,
                        "reference_time": reference.isoformat(),
                        "target_time": target_iso,
                        "probability": probability,
                        "features": store.encode(values),
                        "reference_close": float(window.close.iloc[-1]),
                        "target_close": None,
                        "label": None,
                        "status": "pending",
                        "horizon": config["market"]["horizon"],
                        "mode": mode,
                        "created_at": issuance.isoformat(),
                        "confirmed_at": None,
                        "learned": 0,
                    },
                )
                if structured is not None:
                    store.event(
                        connection,
                        issuance.isoformat(),
                        "structured_forecast",
                        {
                            "model_id": name,
                            "reference_time": reference.isoformat(),
                            "target_time": target_iso,
                            **structured,
                        },
                    )
                forecasts.append(
                    {
                        "model_id": name,
                        "reference_time": reference.isoformat(),
                        "target_time": target_iso,
                        "probability_up": probability,
                        "probability_down_or_flat": 1.0 - probability,
                        "created_at": issuance.isoformat(),
                        "mode": mode,
                    }
                )
                emitted += 1
        if not replay:
            confirmed += _resolve(connection, models, eligible, at, delta)
            decisions.extend(_evaluate(connection, models, config, at.isoformat()))
        store.save_models(connection, models)
        store.set_meta(connection, "mode", mode)
        store.set_meta(connection, "provider", provider)
        if not eligible.empty:
            # Cursor is a closed reference, never wall time, so the next close is not skipped.
            store.set_meta(connection, "cursor", (eligible.timestamp.iloc[-1] + delta).isoformat())
        store.set_meta(connection, "last_poll", at.isoformat())
        summary = {
            "state_path": str(path),
            "mode": mode,
            "new_bars": added,
            "new_predictions": emitted,
            "confirmed_predictions": confirmed,
            "comparisons": decisions,
            "last_reference": store.get_meta(connection, "cursor"),
            "observed_at": at.isoformat(),
            "orders_enabled": False,
            "forecasts": forecasts,
        }
        if added or emitted or confirmed or decisions:
            store.event(connection, at.isoformat(), "cycle", summary)
        connection.commit()
        return summary
    finally:
        connection.close()


def status(state_path: Path) -> dict:
    connection = store.connect(state_path)
    try:
        connection.execute("BEGIN")
        models = store.load_models(connection)
        counts = dict(
            connection.execute("SELECT status,count(*) FROM predictions GROUP BY status").fetchall()
        )
        return {
            "state_path": str(state_path),
            "schema_version": store.get_meta(connection, "schema_version"),
            "mode": store.get_meta(connection, "mode"),
            "protocol": store.get_meta(connection, "protocol"),
            "last_reference": store.get_meta(connection, "cursor"),
            "last_poll": store.get_meta(connection, "last_poll"),
            "bars": connection.execute("SELECT count(*) FROM bars").fetchone()[0],
            "predictions": counts,
            "models": models,
            "orders_enabled": False,
        }
    finally:
        connection.close()


def predictions(
    state_path: Path,
    *,
    limit: int = 20,
    status_filter: str | None = None,
    model_id: str | None = None,
) -> dict:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise DataError("limit debe ser entero entre 1 y 1000")
    if status_filter not in (None, "pending", "evaluated", "invalid_gap"):
        raise DataError("Filtro de estado inválido")
    connection = store.connect(state_path)
    try:
        connection.execute("BEGIN")
        if model_id is not None and model_id not in store.load_models(connection):
            raise DataError("Modelo online desconocido")
        clauses, parameters = [], []
        for column, value in (("status", status_filter), ("model_id", model_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(limit)
        rows = connection.execute(
            "SELECT * FROM predictions" + where + " ORDER BY reference_time DESC,model_id LIMIT ?",
            parameters,
        )
        forecasts = []
        for raw in rows:
            row = dict(raw)
            if row["digest"] != store.row_digest(row):
                raise DataError("Checksum de predicción inválido")
            row.pop("features")
            row.pop("digest")
            row["probability_up"] = row.pop("probability")
            row["probability_down_or_flat"] = 1.0 - row["probability_up"]
            row["realized_return"] = (
                row["target_close"] / row["reference_close"] - 1 if row["target_close"] else None
            )
            forecasts.append(row)
        contract = store.get_meta(connection, "protocol")
        return {
            "state_path": str(state_path),
            "mode": store.get_meta(connection, "mode"),
            "market": contract["market"],
            "forecasts": forecasts,
            "orders_enabled": False,
        }
    finally:
        connection.close()


def report(state_path: Path) -> dict:
    connection = store.connect(state_path)
    try:
        connection.execute("BEGIN")
        metrics = []
        for name, model in store.load_models(connection).items():
            rows = connection.execute(
                "SELECT probability,label FROM predictions WHERE model_id=? AND status='evaluated'",
                (name,),
            ).fetchall()
            n = len(rows)
            metrics.append(
                {
                    "model_id": name,
                    "status": model["status"],
                    "evaluated": n,
                    "brier": sum((p - y) ** 2 for p, y in rows) / n if n else None,
                    "log_loss": -sum(y * math.log(p) + (1 - y) * math.log1p(-p) for p, y in rows)
                    / n
                    if n
                    else None,
                    "accuracy": sum(int(p >= 0.5) == y for p, y in rows) / n if n else None,
                    "evaluation": model.get("evaluation", {}),
                }
            )
        comparisons = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload FROM events WHERE kind='comparison' ORDER BY id DESC LIMIT 100"
            )
        ]
        validated = [
            row["model_id"]
            for row in metrics
            if row["status"] == "active"
            and row["evaluation"].get("validation_state") == "validated_predictive"
        ]
        snapshots = [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "model_id": row["model_id"],
                "digest": row["digest"],
                "decision": json.loads(row["decision"]),
            }
            for row in connection.execute("SELECT * FROM validated_snapshots ORDER BY id")
        ]
        return {
            "mode": store.get_meta(connection, "mode"),
            "metrics": metrics,
            "comparisons": comparisons,
            "validated_models": validated,
            "retained_validated_snapshots": snapshots,
            "profitability_validated": False,
            "orders_enabled": False,
            "note": "Métricas globales descriptivas; decisiones solo en ventanas pareadas comunes. Replay no es evidencia prospectiva.",
        }
    finally:
        connection.close()


def verify(state_path: Path) -> dict:
    connection = store.connect(state_path)
    try:
        connection.execute("BEGIN")
        if [row[0] for row in connection.execute("PRAGMA integrity_check")] != ["ok"]:
            raise DataError("Integridad SQLite inválida")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise DataError("Referencias de modelos inválidas")
        contract = store.get_meta(connection, "protocol")
        if not contract or store.get_meta(connection, "protocol_hash") != store.digest(contract):
            raise DataError("Checksum del protocolo inválido")
        if store.get_meta(connection, "schema_version") != store.SCHEMA_VERSION:
            raise DataError("Versión del estado incompatible")
        store.check_records(connection)
        fnames = contract.get("feature_names", FEATURE_NAMES)
        models = store.load_models(connection)
        for name, model in models.items():
            predict(model, [0.0] * len(fnames))
            updates = connection.execute(
                "SELECT coalesce(sum(learned),0) FROM predictions WHERE model_id=?", (name,)
            ).fetchone()[0]
            if updates != model["updates"]:
                raise DataError(
                    "Contador de aprendizaje no coincide con confirmaciones persistidas"
                )
        delta = pd.Timedelta(INTERVALS[contract["market"]["interval"]])
        bars = {
            row["timestamp"]: row["close"]
            for row in connection.execute("SELECT timestamp,close FROM bars")
        }
        mode = store.get_meta(connection, "mode")
        if mode not in (None, "live", "replay"):
            raise DataError("Modo del estado inválido")
        cursor = store.get_meta(connection, "cursor")
        if bars and cursor != (utc(max(bars)) + delta).isoformat():
            raise DataError("Cursor no coincide con última vela cerrada")
        validate_candles(
            pd.read_sql_query(
                "SELECT timestamp,open,high,low,close,volume FROM bars ORDER BY timestamp",
                connection,
            ),
            contract["market"]["interval"],
            utc(cursor) if cursor else pd.Timestamp.now(tz="UTC"),
        )
        for raw in connection.execute("SELECT * FROM predictions"):
            row = dict(raw)
            reference, target = utc(row["reference_time"]), utc(row["target_time"])
            if (
                target != reference + contract["market"]["horizon"] * delta
                or row["horizon"] != contract["market"]["horizon"]
                or row["mode"] != mode
            ):
                raise DataError("Contrato de predicción inválido")
            if row["reference_close"] != bars.get((reference - delta).isoformat()):
                raise DataError("Precio de referencia inválido")
            if reference < utc(contract["start_at"]):
                raise DataError("Predicción anterior al inicio del protocolo")
            if not 0 < row["probability"] < 1 or not math.isfinite(row["probability"]):
                raise DataError("Probabilidad inválida")
            if len(json.loads(row["features"])) != len(fnames) or not all(
                math.isfinite(v) for v in json.loads(row["features"])
            ):
                raise DataError("Features inválidas")
            if row["status"] == "evaluated":
                target_close = bars.get((target - delta).isoformat())
                required = pd.date_range(reference, target - delta, freq=delta)
                if (
                    target_close is None
                    or row["target_close"] != target_close
                    or not all(t.isoformat() in bars for t in required)
                    or row["label"] != int(target_close > row["reference_close"])
                    or not row["confirmed_at"]
                    or utc(row["confirmed_at"]) < target
                ):
                    raise DataError("Etiqueta/confirmación inválida")
            elif (
                row["status"] not in ("pending", "invalid_gap")
                or row["label"] is not None
                or row["learned"]
            ):
                raise DataError("Estado de predicción inválido")
            created = utc(row["created_at"])
            if created < reference or (
                mode == "live"
                and (
                    created >= target
                    or (created - reference).total_seconds()
                    > contract["max_prediction_delay_seconds"]
                )
            ):
                raise DataError("Predicción no causal")
        return {
            "state_path": str(state_path),
            "ok": True,
            "integrity": "ok",
            "mode": mode,
            "bars": len(bars),
            "predictions": connection.execute("SELECT count(*) FROM predictions").fetchone()[0],
        }
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataError(f"Estado corrupto: {exc}") from exc
    finally:
        connection.close()


def backup(state_path: Path, output: Path) -> dict:
    output = store.local_path(output)
    if Path(state_path).resolve() == output.resolve() or output.exists():
        raise DataError("La copia necesita un destino nuevo")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a backup from accidental replacement, including races.
    with output.open("xb"):
        pass
    try:
        source = store.connect(state_path)
        try:
            with sqlite3.connect(output) as destination:
                source.backup(destination)
        finally:
            source.close()
        checked = verify(output)
        return {"backup_path": str(output), "verified": checked}
    except BaseException:
        with contextlib.suppress(OSError):
            output.unlink()
        raise


def restore(backup_path: Path, state_path: Path) -> dict:
    verify(backup_path)
    result = backup(backup_path, state_path)
    return {
        "state_path": str(state_path),
        "restored_from": str(backup_path),
        "verified": result["verified"],
    }


def models_info(state_path: Path | None = None) -> dict:
    """List registered mathematical models, cohorts, and persisted model state summaries."""
    catalog = [spec.to_dict() for spec in MODEL_SPECS.values()]
    cohorts = {k: list(v) for k, v in COHORT_DEFINITIONS.items()}
    persisted: dict[str, dict] = {}
    if state_path is not None and Path(state_path).is_file():
        connection = store.connect(state_path)
        try:
            connection.execute("BEGIN")
            loaded = store.load_models(connection)
            for mid, m in loaded.items():
                spec = MODEL_SPECS.get(mid)
                params = m.get("params", {})
                summary: dict = {
                    "model_id": mid,
                    "kind": m["kind"],
                    "target_type": m.get(
                        "target_type", spec.target_type if spec else "direction"
                    ),
                    "status": m["status"],
                    "updates": m["updates"],
                    "n_features": m["n_features"],
                    "validation_state": m.get("evaluation", {}).get(
                        "validation_state", "unvalidated"
                    ),
                }
                if m["kind"] in ("logistic", "logistic_static", "logistic_adaptive"):
                    w = params.get("weights", [])
                    summary["weight_l2_norm"] = round(
                        math.sqrt(sum(float(x) * float(x) for x in w)), 6
                    )
                    if m["kind"] == "logistic_static":
                        summary["frozen"] = bool(params.get("frozen", False))
                        summary["weight_updates"] = int(params.get("weight_updates", 0))
                    if m["kind"] == "logistic_adaptive":
                        summary["drift_monitor"] = params.get("drift_monitor", {})
                elif m["kind"] == "kalman_filter":
                    summary["diagnostics"] = params.get("diagnostics", {})
                    summary["obs_noise"] = params.get("obs_noise")
                    summary["calibrator"] = params.get("calibrator", {})
                elif m["kind"] == "ewma_volatility":
                    summary["current_volatility"] = round(
                        math.sqrt(max(1e-10, float(params.get("variance", 1e-4)))), 6
                    )
                elif m["kind"] == "har_rv":
                    summary["theta"] = [round(float(x), 5) for x in params.get("theta", [])]
                elif m["kind"] == "quantile_regression":
                    summary["raw_crossings"] = int(params.get("raw_crossings", 0))
                persisted[mid] = summary
        finally:
            connection.close()
    return {
        "registered_models": catalog,
        "cohorts": cohorts,
        "persisted_state_path": str(state_path) if state_path else None,
        "persisted_models": persisted,
        "orders_enabled": False,
    }


def compare_models(state_path: Path) -> dict:
    """Perform target-separated statistical and economic comparison on confirmed SQLite predictions."""
    connection = store.connect(state_path)
    try:
        connection.execute("BEGIN")
        contract = store.get_meta(connection, "protocol") or {}
        market = contract.get("market", {"symbol": "BTCUSDT", "interval": "1h", "horizon": 1})
        eval_cfg = contract.get("evaluation", {})
        models = store.load_models(connection)
        delta = pd.Timedelta(INTERVALS[market["interval"]])

        bars_df = pd.read_sql_query(
            "SELECT timestamp,open,high,low,close,volume FROM bars ORDER BY timestamp",
            connection,
        )
        if not bars_df.empty:
            bars_df["timestamp"] = pd.to_datetime(bars_df.timestamp, utc=True)
            by_open = bars_df.set_index("timestamp")
        else:
            by_open = pd.DataFrame()

        structured_map: dict[tuple[str, str], dict] = {}
        for row in connection.execute(
            "SELECT payload FROM events WHERE kind='structured_forecast'"
        ):
            payload = json.loads(row[0])
            structured_map[(payload["model_id"], payload["reference_time"])] = payload

        records = []
        for raw in connection.execute(
            "SELECT model_id,reference_time,target_time,probability,reference_close,target_close,label,status,horizon FROM predictions WHERE status='evaluated' ORDER BY reference_time,model_id"
        ):
            rec = dict(raw)
            ref_ts = utc(rec["reference_time"])
            tgt_ts = utc(rec["target_time"])
            req = pd.date_range(ref_ts, tgt_ts - delta, freq=delta)
            if not by_open.empty and req.isin(by_open.index).all():
                r_ret, r_var = realized_target_metrics(
                    float(rec["reference_close"]), by_open.loc[req]
                )
            else:
                r_ret = (
                    math.log(float(rec["target_close"]) / float(rec["reference_close"]))
                    if rec.get("target_close") and rec.get("reference_close")
                    else 0.0
                )
                r_var = max(1e-8, r_ret * r_ret)
            rec["realized_return"] = r_ret
            rec["realized_variance"] = r_var
            extra = structured_map.get((rec["model_id"], rec["reference_time"]))
            if extra:
                for k in (
                    "expected_return",
                    "predictive_std",
                    "pred_variance",
                    "pred_volatility",
                    "q10",
                    "q50",
                    "q90",
                ):
                    if k in extra:
                        rec[k] = extra[k]
            records.append(rec)

        result = evaluate_target_specific_records(
            records,
            models,
            symbol=market["symbol"],
            interval=market["interval"],
            horizon=int(market["horizon"]),
            alpha=float(eval_cfg.get("alpha", 0.05)),
            min_effect=float(eval_cfg.get("min_effect", 0.001)),
        )
        result["state_path"] = str(state_path)
        result["mode"] = store.get_meta(connection, "mode")
        result["orders_enabled"] = False
        return result
    finally:
        connection.close()


def champions(state_path: Path) -> dict:
    """Return active Champion vs Challenger status per (symbol, interval, horizon, target_type)."""
    comp = compare_models(state_path)
    return {
        "state_path": str(state_path),
        "mode": comp.get("mode"),
        "symbol": comp["symbol"],
        "interval": comp["interval"],
        "horizon": comp["horizon"],
        "champions": comp["champions"],
        "orders_enabled": False,
    }
