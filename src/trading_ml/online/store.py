from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from trading_ml.errors import DataError
from trading_ml.online.models import validate_model

WORKSPACE = Path(__file__).resolve().parents[3]

SCHEMA_VERSION = "2"
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bars(
 timestamp TEXT PRIMARY KEY, open REAL NOT NULL, high REAL NOT NULL,
 low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL, digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS models(
 model_id TEXT PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions(
 model_id TEXT NOT NULL REFERENCES models(model_id), reference_time TEXT NOT NULL,
 target_time TEXT NOT NULL, probability REAL NOT NULL CHECK(probability >= 0 AND probability <= 1),
 features TEXT NOT NULL, reference_close REAL NOT NULL, target_close REAL,
 label INTEGER CHECK(label IN (0,1)), status TEXT NOT NULL, horizon INTEGER NOT NULL,
 mode TEXT NOT NULL, created_at TEXT NOT NULL, confirmed_at TEXT,
 learned INTEGER NOT NULL CHECK(learned IN (0,1)), digest TEXT NOT NULL,
 PRIMARY KEY(model_id, reference_time));
CREATE INDEX IF NOT EXISTS prediction_pending ON predictions(status, target_time);
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS validated_snapshots(
 id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, model_id TEXT NOT NULL REFERENCES models(model_id),
 payload TEXT NOT NULL, decision TEXT NOT NULL, digest TEXT NOT NULL);
"""


def encode(value) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def digest(value) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def row_digest(row: dict) -> str:
    return digest({key: value for key, value in row.items() if key != "digest"})


def local_path(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise DataError("El estado y las copias deben permanecer dentro del proyecto local")
    return path


def connect(path: Path, *, create: bool = False) -> sqlite3.Connection:
    path = local_path(path)
    existed = path.exists()
    if not create and not path.is_file():
        raise DataError(f"No existe el estado: {path}")
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=15000")
    connection.execute("PRAGMA synchronous=FULL")
    if create:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        expected_tables = {"meta", "models", "bars", "predictions", "events", "validated_snapshots"}
        if existed:
            try:
                version = (
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()
                    if expected_tables <= tables
                    else None
                )
                recognized = version is not None and json.loads(version[0]) == SCHEMA_VERSION
            except (ValueError, sqlite3.Error):
                recognized = False
            if not recognized:
                connection.close()
                raise DataError(
                    "El archivo existente no es un estado online compatible; no se reemplazará"
                )
        if not existed:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
    return connection


def get_meta(connection, key: str, default=None):
    row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_meta(connection, key: str, value) -> None:
    connection.execute(
        "INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, encode(value)),
    )


def load_models(connection) -> dict[str, dict]:
    models = {}
    for row in connection.execute("SELECT * FROM models ORDER BY model_id"):
        payload = json.loads(row["payload"])
        if row["digest"] != digest(payload):
            raise DataError(f"Checksum del modelo inválido: {row['model_id']}")
        try:
            validate_model(payload)
        except (ValueError, TypeError, KeyError) as exc:
            raise DataError(f"Estado inválido del modelo {row['model_id']}: {exc}") from exc
        models[row["model_id"]] = payload
    if set(models) != {"baseline", "logistic_slow", "logistic_fast"}:
        raise DataError("Cohorte online incompleta o desconocida")
    return models


def save_models(connection, models: dict) -> None:
    for name, payload in models.items():
        connection.execute(
            "INSERT INTO models VALUES(?,?,?) ON CONFLICT(model_id) DO UPDATE SET payload=excluded.payload,digest=excluded.digest",
            (name, encode(payload), digest(payload)),
        )


def event(connection, timestamp: str, kind: str, payload) -> None:
    connection.execute(
        "INSERT INTO events(timestamp,kind,payload) VALUES(?,?,?)",
        (timestamp, kind, encode(payload)),
    )


def check_records(connection) -> None:
    for table in ("bars", "predictions"):
        for raw in connection.execute(f"SELECT * FROM {table}"):
            row = dict(raw)
            if row["digest"] != row_digest(row):
                raise DataError(f"Checksum inválido en {table}")
    load_models(connection)
    for raw in connection.execute("SELECT * FROM validated_snapshots"):
        row = dict(raw)
        if row["digest"] != row_digest(row):
            raise DataError("Checksum inválido de snapshot validado")
