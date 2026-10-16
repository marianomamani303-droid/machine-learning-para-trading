from __future__ import annotations

import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.errors import DataError


def load_protocol(path: Path) -> dict:
    with path.open("rb") as stream:
        protocol = tomllib.load(stream)
    validate_protocol(protocol)
    return protocol


def validate_protocol(p: dict) -> None:
    try:
        d, r, t, risk, costs = (p[k] for k in ["data", "research", "training", "risk", "costs"])
        dates = pd.to_datetime(r["boundaries"], utc=True)
        start, end = pd.to_datetime([d["start"], d["end"]], utc=True)
        integer_fields = {
            "sequence_length": (2, 252),
            "validation_days": (30, 730),
            "embargo_days": (1, 30),
            "minimum_train_dates": (30, 5000),
            "epochs": (1, 100),
            "patience": (1, 30),
            "batch_dates": (8, 1024),
            "hidden_size": (2, 128),
            "threads": (1, 8),
        }
        for key, (low, high) in integer_fields.items():
            if type(t[key]) is not int or not low <= t[key] <= high:
                raise ValueError(key)
        if (
            len(dates) < 2
            or not dates.is_monotonic_increasing
            or dates.duplicated().any()
            or start >= dates[0]
            or dates[-1] > end
            or end > pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
            or start != start.normalize()
            or end != end.normalize()
            or any(x != x.normalize() for x in dates)
            or end > pd.Timestamp("2026-10-01", tz="UTC")
        ):
            raise ValueError("dates: historical development is capped at 2026-10-01")
        symbols = d["symbols"]
        if (
            not symbols
            or len(set(symbols)) != len(symbols)
            or any(not re.fullmatch(r"[A-Z0-9]{3,24}", s) for s in symbols)
            or len(symbols) > 20
            or d["periods_per_year"] != 365
        ):
            raise ValueError("daily crypto universe")
        seeds = r["seeds"]
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        ):
            raise ValueError("seeds")
        if not np.isfinite([t[k] for k in ["learning_rate", "dropout", "gradient_clip"]]).all():
            raise ValueError("nonfinite training")
        if (
            not 0 < t["learning_rate"] <= 0.1
            or not 0 <= t["dropout"] < 1
            or t["gradient_clip"] <= 0
        ):
            raise ValueError("optimizer")
        if risk["position_mode"] not in {"long_only", "long_short"}:
            raise ValueError("position_mode")
        if not (0 < risk["target_volatility"] <= 1 and 0 < risk["maximum_gross"] <= 1):
            raise ValueError("risk limits")
        if not 1e-8 <= risk["daily_volatility_floor"] <= 0.1:
            raise ValueError("volatility floor")
        if not np.isfinite(list(costs.values())).all() or min(costs.values()) < 0:
            raise ValueError("costs")
        if sum(costs[k] for k in ["commission_bps", "slippage_bps", "half_spread_bps"]) >= 500:
            raise ValueError("transaction costs")
        ids = [rec["id"] for rec in p["recipes"]]
        if (
            not ids
            or len(ids) != len(set(ids))
            or any(not re.fullmatch(r"[a-z0-9_]+", s) for s in ids)
        ):
            raise ValueError("recipe ids")
        for rec in p["recipes"]:
            if rec["architecture"] not in {"linear", "mlp", "lstm", "xgboost"}:
                raise ValueError("architecture")
            if rec["objective"] not in {"sharpe", "mean_return", "mse", "binary"}:
                raise ValueError("objective")
            if type(rec["net_training"]) is not bool:
                raise ValueError("net_training")
            if rec["architecture"] == "xgboost" and rec["objective"] != "binary":
                raise ValueError("xgboost baseline is binary")
        required = (len(dates) - 1) * len(seeds) * len(ids)
        if not required <= r["maximum_fits"] <= 250:
            raise ValueError("fit budget")
        b = p["bootstrap"]
        if (
            not 100 <= b["samples"] <= 5000
            or not b["block_lengths"]
            or any(type(x) is not int or x < 1 for x in b["block_lengths"])
        ):
            raise ValueError("bootstrap")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataError(f"Protocolo DMN inválido: {exc}") from exc
