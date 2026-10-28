from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path

import pandas as pd

from trading_ml.config.settings import INTERVALS
from trading_ml.errors import DataError
from trading_ml.online.adaptive import COHORT_DEFINITIONS

FEATURE_NAMES = [
    "return_1",
    "return_3",
    "return_7",
    "momentum",
    "volatility",
    "range",
    "body",
    "volume_ratio",
]

FINANCIAL_FEATURE_NAMES = [
    "fin_liq_corwin_schultz_20",
    "fin_liq_range_impact_z63",
    "fin_vol_parkinson_ratio",
]

FEATURE_SETS: dict[str, list[str]] = {
    "v1_base8": FEATURE_NAMES,
    "v2_financial11": [*FEATURE_NAMES, *FINANCIAL_FEATURE_NAMES],
}

EVALUATION_METHODS = ("hoeffding_azuma", "empirical_bernstein")


def utc(value: str | pd.Timestamp) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
        if result.tzinfo is None or pd.isna(result):
            raise ValueError("Se requiere zona horaria explícita")
        return result.tz_convert("UTC")
    except (ValueError, TypeError) as exc:
        raise DataError(f"Timestamp UTC inválido: {value!r}") from exc


def feature_names_for_config(config: dict) -> list[str]:
    fset = config.get("online", {}).get("feature_set", "v1_base8")
    return list(FEATURE_SETS[fset])


def cohort_for_config(config: dict) -> str:
    return str(config.get("online", {}).get("cohort", "classic_3"))


def evaluator_for_config(config: dict) -> str:
    return str(config.get("evaluation", {}).get("method", "hoeffding_azuma"))


def load_config(path: Path) -> dict:
    try:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        allowed = {
            "market": {"symbol", "interval", "horizon"},
            "online": {
                "state_path",
                "poll_seconds",
                "warmup_bars",
                "max_prediction_delay_seconds",
                "max_fetch_pages",
                "seed",
                "start_at",
                "cohort",
                "feature_set",
                "static_warmup_updates",
            },
            "evaluation": {
                "eval_window",
                "min_samples",
                "patience",
                "alpha",
                "min_effect",
                "method",
            },
        }
        if set(raw) - set(allowed):
            raise ValueError(f"Secciones desconocidas: {sorted(set(raw) - set(allowed))}")
        for section, names in allowed.items():
            values = raw.get(section, {})
            if not isinstance(values, dict) or set(values) - names:
                raise ValueError(f"Sección {section} inválida o con parámetros desconocidos")
        market = {"symbol": "BTCUSDT", "interval": "1h", "horizon": 1, **raw.get("market", {})}
        online = {
            "state_path": "artifacts/online/state.sqlite3",
            "poll_seconds": 30,
            "warmup_bars": 32,
            "max_prediction_delay_seconds": 120,
            "max_fetch_pages": 20,
            "seed": 42,
            **raw.get("online", {}),
        }
        evaluation = {
            "eval_window": 50,
            "min_samples": 100,
            "patience": 3,
            "alpha": 0.05,
            "min_effect": 0.001,
            **raw.get("evaluation", {}),
        }
        if not isinstance(market["symbol"], str) or not re.fullmatch(
            r"[A-Z0-9]{3,24}", market["symbol"]
        ):
            raise ValueError("Símbolo inválido")
        if not isinstance(market["interval"], str) or market["interval"] not in INTERVALS:
            raise ValueError("Intervalo inválido")
        for value, minimum in [
            (market["horizon"], 1),
            (online["warmup_bars"], 8),
            (online["max_fetch_pages"], 1),
            (online["seed"], 0),
            (evaluation["eval_window"], 2),
            (evaluation["min_samples"], 2),
            (evaluation["patience"], 1),
        ]:
            if type(value) is not int or value < minimum:
                raise ValueError("Parámetro entero fuera de rango")
        if "cohort" in online and online["cohort"] not in COHORT_DEFINITIONS:
            raise ValueError(f"Cohorte online desconocida: {online['cohort']}")
        if "feature_set" in online:
            if online["feature_set"] not in FEATURE_SETS:
                raise ValueError(f"feature_set desconocido: {online['feature_set']}")
            if online["feature_set"] == "v2_financial11" and online["warmup_bars"] < 64:
                raise ValueError("feature_set v2_financial11 requiere warmup_bars >= 64")
        if "static_warmup_updates" in online and (
            type(online["static_warmup_updates"]) is not int or online["static_warmup_updates"] < 0
        ):
            raise ValueError("static_warmup_updates debe ser entero >= 0")
        if "method" in evaluation and evaluation["method"] not in EVALUATION_METHODS:
            raise ValueError(f"Método de evaluación desconocido: {evaluation['method']}")
        for value in [online["poll_seconds"], online["max_prediction_delay_seconds"]]:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Intervalos operativos deben ser positivos y finitos")
        if (
            any(
                type(evaluation[name]) not in (int, float) or not math.isfinite(evaluation[name])
                for name in ("alpha", "min_effect")
            )
            or not 0 < evaluation["alpha"] < 1
            or not 0 <= evaluation["min_effect"] < 1
        ):
            raise ValueError("alpha/min_effect fuera de rango")
        if online["seed"] >= 2**32:
            raise ValueError("seed debe ser menor que 2**32")
        if not isinstance(online["state_path"], str) or not online["state_path"].strip():
            raise ValueError("state_path debe ser una ruta no vacía")
        online["start_at"] = utc(online["start_at"]).isoformat()
        return {"market": market, "online": online, "evaluation": evaluation}
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise DataError(f"Configuración online inválida en {path}: {exc}") from exc


def protocol(config: dict) -> dict:
    fnames = feature_names_for_config(config)
    result = {
        "schema": 2,
        "feature_names": fnames,
        "market": config["market"],
        "start_at": config["online"]["start_at"],
        "warmup_bars": config["online"]["warmup_bars"],
        "max_prediction_delay_seconds": config["online"]["max_prediction_delay_seconds"],
        "seed": config["online"]["seed"],
        "evaluation": config["evaluation"],
    }
    cohort = cohort_for_config(config)
    if cohort != "classic_3":
        result["cohort"] = cohort
        result["cohort_models"] = list(COHORT_DEFINITIONS[cohort])
    if "feature_set" in config.get("online", {}):
        result["feature_set"] = config["online"]["feature_set"]
    if "static_warmup_updates" in config.get("online", {}):
        result["static_warmup_updates"] = config["online"]["static_warmup_updates"]
    return result
