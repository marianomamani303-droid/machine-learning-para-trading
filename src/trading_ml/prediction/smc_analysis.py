from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.config.settings import INTERVALS
from trading_ml.errors import DataError
from trading_ml.models import ModelBundle
from trading_ml.prediction.service import predict
from trading_ml.preprocessing import load_prepared
from trading_ml.smc.rules import directional_states, rule_probabilities


def analyze_smc(
    bundle: ModelBundle,
    dataset: Path,
    reference: str | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Información de mercado y probabilidades; no produce instrucciones de órdenes."""
    if not bundle.metadata["contract"]["features"].get("smc", False):
        raise DataError("El análisis combinado requiere un modelo y dataset con features SMC")
    prediction = predict(bundle, dataset, reference, now)
    frame, _ = load_prepared(dataset, bundle.metadata["contract"])
    row = frame.loc[frame.reference_time == pd.Timestamp(prediction["reference_time"])]
    values = row.iloc[0]
    threshold = bundle.metadata["config"]["backtest"]["threshold"]
    state = str(
        directional_states([prediction["probability_up"]], threshold, [values.smc_ready > 0])[0]
    )
    rule_probability, rule_state = rule_probabilities(row, threshold)
    names = (
        values.smc_pattern_names.split("|")
        if isinstance(values.smc_pattern_names, str) and values.smc_pattern_names
        else []
    )
    delta = pd.Timedelta(INTERVALS[prediction["interval"]])
    zones = json.loads(values.smc_active_zones)
    for zone in zones:
        if "created_open_time" in zone:
            zone["available_reference_time"] = (
                pd.to_datetime(zone["created_open_time"], utc=True) + delta
            ).isoformat()
    trend = {1: "bullish", -1: "bearish", 0: "range_or_unconfirmed"}[int(values.smc_trend)]
    structure = {
        1: "higher_highs_higher_lows",
        -1: "lower_highs_lower_lows",
        0: "mixed_or_unconfirmed",
    }[int(values.smc_structure)]
    return {
        **prediction,
        "market_pair": "BTC/USDT" if prediction["asset"] == "BTCUSDT" else prediction["asset"],
        "trend": trend,
        "market_structure": structure,
        "patterns": [
            {"name": name, "available_at_reference": prediction["reference_time"]} for name in names
        ],
        "active_zones": zones,
        "signal_state": state,
        "neutral_band": {
            "bearish_at_or_below": round(1 - threshold, 12),
            "bullish_at_or_above": threshold,
        },
        "evidence_ready": bool(values.smc_ready),
        "smc_rules": {
            "version": "smc_rules_v1",
            "probability_up": float(rule_probability[0]),
            "probability_down": float(1 - rule_probability[0]),
            "signal_state": str(rule_state[0]),
            "calibration_status": "heuristic_uncalibrated",
            "probability_semantics": "fixed confidence score; not an empirically calibrated event frequency",
        },
        "research_status": (
            "active_research_model"
            if bundle.metadata.get("activation_status") == "active"
            else "not_promoted_pending_independent_future_evaluation"
        ),
        "warnings": [
            *prediction["warnings"],
            "Estos patrones OHLCV no verifican actividad institucional ni órdenes reales.",
        ],
    }
