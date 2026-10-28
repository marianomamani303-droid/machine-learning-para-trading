"""Reproducible prequential online learning laboratory & ablation benchmark runner.

Strictly respects historical quarantine (< 2025-07-01) and reserved prospective labels via load_history().
Never mutates artifacts/online/state.sqlite3.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.config.settings import INTERVALS
from trading_ml.data.history import load_history
from trading_ml.errors import DataError
from trading_ml.online import store
from trading_ml.online.config import FEATURE_SETS
from trading_ml.online.data import features, realized_target_metrics, validate_candles
from trading_ml.online.evaluation import (
    evaluate_models,
    evaluate_target_specific_records,
    run_evaluator_simulation_study,
)
from trading_ml.online.models import learn, new_models, predict_structured

DEFAULT_HISTORICAL_PATHS: dict[tuple[str, str], str] = {
    ("BTCUSDT", "1h"): "artifacts/historical/datasets/binance_spot/BTCUSDT/1h_5b9cf016d8ef39526dc6.csv",
    ("BTCUSDT", "1d"): "artifacts/historical/datasets/binance_spot/BTCUSDT/1d_1e3c9ab913aa78e6127b.csv",
    ("ETHUSDT", "1h"): "artifacts/historical/datasets/binance_spot/ETHUSDT/1h_97a73f87b406ee07d6f5.csv",
    ("ETHUSDT", "1d"): "artifacts/historical/datasets/binance_spot/ETHUSDT/1d_95e51561c3f42edf65bb.csv",
}


def load_experiment_candles(
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "2023-06-01",
    end: str = "2024-06-01",
    horizon: int = 1,
    source: Path | None = None,
    root: Path = Path("artifacts"),
) -> pd.DataFrame:
    """Load validated historical candles strictly before 2025-07-01 without touching reserved labels."""
    stop_ts = pd.to_datetime(end, utc=True)
    if stop_ts > pd.Timestamp("2025-07-01T00:00:00Z"):
        raise DataError("Los experimentos históricos online no pueden superar 2025-07-01T00:00:00Z")
    if source is not None:
        sidecar_path = source.with_suffix(source.suffix + ".json")
        if sidecar_path.exists():
            frame, _ = load_history(source, root=root, start=start, end=end, horizon=horizon)
            return validate_candles(frame, interval, stop_ts)
        raw = pd.read_csv(source)
        frame = validate_candles(raw, interval, stop_ts)
        start_ts = pd.to_datetime(start, utc=True)
        return frame.loc[(frame.timestamp >= start_ts) & (frame.timestamp < stop_ts)].reset_index(
            drop=True
        )
    rel = DEFAULT_HISTORICAL_PATHS.get((symbol.upper(), interval))
    if not rel:
        raise DataError(f"No hay ruta histórica predeterminada para {symbol} {interval}")
    path = Path(rel)
    if not path.is_file():
        raise DataError(f"No existe el archivo histórico: {path}")
    frame, _ = load_history(path, root=root, start=start, end=end, horizon=horizon)
    return validate_candles(frame, interval, stop_ts)


def run_prequential_replay(
    candles: pd.DataFrame,
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    horizon: int = 1,
    feature_mode: str = "v1_base8",
    cohort: str = "adaptive_lab",
    evaluator_method: str = "empirical_bernstein",
    warmup_bars: int = 64,
    static_warmup_updates: int = 120,
    eval_window: int = 50,
    min_samples: int = 100,
    patience: int = 3,
    alpha: float = 0.05,
    min_effect: float = 0.001,
    cost_bps: float = 10.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run an isolated causal prequential walk-forward experiment on closed candles."""
    if feature_mode == "v1_noise11":
        n_features = 11
        base_fset = "v1_base8"
    elif feature_mode in FEATURE_SETS:
        n_features = len(FEATURE_SETS[feature_mode])
        base_fset = feature_mode
    else:
        raise DataError(f"Modo de features desconocido: {feature_mode}")

    rng = np.random.default_rng(seed)
    delta = pd.Timedelta(INTERVALS[interval])
    models = new_models(
        n_features,
        seed=seed,
        cohort=cohort,
        options={"static_warmup_updates": static_warmup_updates},
    )
    # Keep a shadow copy of models for Hoeffding-Azuma comparison if primary is Empirical Bernstein
    models_hoeffding = copy.deepcopy(models)

    by_open = candles.set_index("timestamp")
    timestamps = candles["timestamp"].tolist()
    ts_series = candles["timestamp"]
    step_ok = (ts_series.diff() == delta).to_numpy(dtype=bool, copy=True)
    step_ok[0] = True
    contig_run = np.zeros(len(timestamps), dtype=int)
    run_len = 0
    for i, ok in enumerate(step_ok):
        run_len = (run_len + 1) if ok else 1
        contig_run[i] = run_len
    pending_by_target: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    evaluated_records: list[dict[str, Any]] = []
    decisions_primary: list[dict[str, Any]] = []
    decisions_hoeffding: list[dict[str, Any]] = []

    infer_times_ns: dict[str, list[int]] = {mid: [] for mid in models}
    learn_times_ns: dict[str, list[int]] = {mid: [] for mid in models}

    policy_primary = {
        "eval_window": eval_window,
        "min_samples": min_samples,
        "patience": patience,
        "horizon": horizon,
        "alpha": alpha,
        "min_effect": min_effect,
        "method": evaluator_method,
    }
    policy_hoeff = {**policy_primary, "method": "hoeffding_azuma"}

    for idx in range(len(timestamps)):
        reference = timestamps[idx] + delta
        # 1. Resolve all pending forecasts whose target_time <= reference
        due_targets = [t for t in list(pending_by_target.keys()) if t <= reference]
        new_confirmations = False
        for tgt in sorted(due_targets):
            items = pending_by_target.pop(tgt)
            if not items:
                continue
            ref_ts = items[0]["ref_ts"]
            ref_close_0 = items[0]["reference_close"]
            req = pd.date_range(ref_ts, tgt - delta, freq=delta)
            if not req.isin(by_open.index).all():
                for item in items:
                    item["status"] = "invalid_gap"
                continue
            target_bars = by_open.loc[req]
            target_close = float(by_open.loc[tgt - delta, "close"])
            r_ret, r_var = realized_target_metrics(ref_close_0, target_bars)
            label = int(target_close > ref_close_0)
            confirmed_iso = reference.isoformat()
            for item in items:
                item["target_close"] = target_close
                item["label"] = label
                item["realized_return"] = r_ret
                item["realized_variance"] = r_var
                item["status"] = "evaluated"
                item["confirmed_at"] = confirmed_iso

                mid = item["model_id"]
                t0 = time.perf_counter_ns()
                learn(
                    models[mid],
                    item["features_vec"],
                    label,
                    realized_return=r_ret,
                    realized_variance=r_var,
                )
                learn_times_ns[mid].append(time.perf_counter_ns() - t0)
                if models_hoeffding[mid]["status"] == "active":
                    learn(
                        models_hoeffding[mid],
                        item["features_vec"],
                        label,
                        realized_return=r_ret,
                        realized_variance=r_var,
                    )
                evaluated_records.append(item)
                new_confirmations = True

        # 2. Run sequential statistical evaluation at the end of the replay stream
        if new_confirmations and idx == len(timestamps) - 1 and len(evaluated_records) >= eval_window:
            decs = evaluate_models(evaluated_records, models, policy_primary)
            if decs:
                decisions_primary.extend(decs)
            decs_h = evaluate_models(evaluated_records, models_hoeffding, policy_hoeff)
            if decs_h:
                decisions_hoeffding.extend(decs_h)

        # 3. Check if we have a full contiguous warmup window ending at idx
        if contig_run[idx] < warmup_bars:
            continue
        window = candles.iloc[idx + 1 - warmup_bars : idx + 1]

        vals = features(window, feature_set=base_fset)
        if feature_mode == "v1_noise11":
            noise = np.clip(rng.normal(0.0, 2.0, size=3), -10.0, 10.0).tolist()
            vals = [*vals, *[float(v) for v in noise]]

        target_ts = reference + horizon * delta
        ref_close = float(window.close.iloc[-1])

        for mid, model in models.items():
            if model["status"] != "active":
                continue
            t0 = time.perf_counter_ns()
            structured = predict_structured(model, vals)
            infer_times_ns[mid].append(time.perf_counter_ns() - t0)
            rec = {
                "symbol": symbol,
                "interval": interval,
                "model_id": mid,
                "reference_time": reference.isoformat(),
                "target_time": target_ts.isoformat(),
                "ref_ts": reference,
                "probability": float(structured["probability"]),
                "raw_probability": float(structured.get("raw_probability", structured["probability"])),
                "expected_return": float(structured.get("expected_return", 0.0)),
                "predictive_std": float(structured.get("predictive_std", 0.01)),
                "pred_variance": float(structured.get("pred_variance", 1e-4)),
                "pred_volatility": float(structured.get("pred_volatility", 0.01)),
                "q10": float(structured.get("q10", -0.01)),
                "q50": float(structured.get("q50", 0.0)),
                "q90": float(structured.get("q90", 0.01)),
                "reference_close": ref_close,
                "target_close": None,
                "label": None,
                "status": "pending",
                "horizon": horizon,
                "features_vec": vals,
            }
            pending_by_target.setdefault(target_ts, []).append(rec)

    evaluation_summary = evaluate_target_specific_records(
        evaluated_records,
        models,
        symbol=symbol,
        interval=interval,
        horizon=horizon,
        alpha=alpha,
        min_effect=min_effect,
        cost_bps=cost_bps,
    )

    performance_profile: dict[str, dict[str, Any]] = {}
    for mid, model in models.items():
        inf_arr = np.array(infer_times_ns[mid], dtype=float) / 1000.0  # microseconds
        lrn_arr = np.array(learn_times_ns[mid], dtype=float) / 1000.0  # microseconds
        serialized_bytes = len(store.encode(model).encode("utf-8"))
        performance_profile[mid] = {
            "inferences": len(inf_arr),
            "updates": model["updates"],
            "infer_mean_us": round(float(np.mean(inf_arr)), 2) if len(inf_arr) else 0.0,
            "infer_p99_us": round(float(np.percentile(inf_arr, 99)), 2) if len(inf_arr) else 0.0,
            "learn_mean_us": round(float(np.mean(lrn_arr)), 2) if len(lrn_arr) else 0.0,
            "learn_p99_us": round(float(np.percentile(lrn_arr, 99)), 2) if len(lrn_arr) else 0.0,
            "state_bytes": serialized_bytes,
            "final_status": model["status"],
            "validation_state_eb": model.get("evaluation", {}).get("validation_state", "unvalidated"),
            "validation_state_hoeffding": models_hoeffding[mid]
            .get("evaluation", {})
            .get("validation_state", "unvalidated"),
        }

    return {
        "symbol": symbol,
        "interval": interval,
        "horizon": horizon,
        "feature_mode": feature_mode,
        "n_features": n_features,
        "cohort": cohort,
        "evaluator_method": evaluator_method,
        "bars_total": len(candles),
        "evaluated_predictions_total": len(evaluated_records),
        "targets": evaluation_summary["targets"],
        "champions": evaluation_summary["champions"],
        "sequential_decisions_primary": decisions_primary,
        "sequential_decisions_hoeffding": decisions_hoeffding,
        "performance_profile": performance_profile,
    }


def run_comprehensive_laboratory_study(
    output_dir: Path = Path("artifacts/online/experiments"),
) -> dict[str, Any]:
    """Execute the full suite of prequential replay, feature ablation, and evaluator benchmarks."""
    output_dir = store.local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. BTCUSDT 1h Replay & Feature Ablation (2024-01-01 to 2024-06-01: ~3,648 H1 bars)
    btc_h1 = load_experiment_candles(
        symbol="BTCUSDT",
        interval="1h",
        start="2024-01-01",
        end="2024-06-01",
        horizon=1,
    )
    h1_v1 = run_prequential_replay(
        btc_h1,
        symbol="BTCUSDT",
        interval="1h",
        horizon=1,
        feature_mode="v1_base8",
        warmup_bars=64,
        static_warmup_updates=150,
        eval_window=100,
        min_samples=100,
        patience=2,
    )
    h1_v2 = run_prequential_replay(
        btc_h1,
        symbol="BTCUSDT",
        interval="1h",
        horizon=1,
        feature_mode="v2_financial11",
        warmup_bars=64,
        static_warmup_updates=150,
        eval_window=100,
        min_samples=100,
        patience=2,
    )
    h1_noise = run_prequential_replay(
        btc_h1,
        symbol="BTCUSDT",
        interval="1h",
        horizon=1,
        feature_mode="v1_noise11",
        warmup_bars=64,
        static_warmup_updates=150,
        eval_window=100,
        min_samples=100,
        patience=2,
    )

    # 2. BTCUSDT 1d Replay & Feature Ablation (2020-01-01 to 2024-06-01: ~1,597 D1 bars)
    btc_d1 = load_experiment_candles(
        symbol="BTCUSDT",
        interval="1d",
        start="2020-01-01",
        end="2024-06-01",
        horizon=1,
    )
    d1_v1 = run_prequential_replay(
        btc_d1,
        symbol="BTCUSDT",
        interval="1d",
        horizon=1,
        feature_mode="v1_base8",
        warmup_bars=64,
        static_warmup_updates=120,
        eval_window=80,
        min_samples=80,
        patience=2,
    )
    d1_v2 = run_prequential_replay(
        btc_d1,
        symbol="BTCUSDT",
        interval="1d",
        horizon=1,
        feature_mode="v2_financial11",
        warmup_bars=64,
        static_warmup_updates=120,
        eval_window=80,
        min_samples=80,
        patience=2,
    )
    d1_noise = run_prequential_replay(
        btc_d1,
        symbol="BTCUSDT",
        interval="1d",
        horizon=1,
        feature_mode="v1_noise11",
        warmup_bars=64,
        static_warmup_updates=120,
        eval_window=80,
        min_samples=80,
        patience=2,
    )

    # 3. Sequential Evaluator Simulation Study (6 scenarios)
    sim_study = run_evaluator_simulation_study(
        n_trials=200,
        n_steps=600,
        eval_window=100,
        min_samples=100,
        patience=2,
        seed=42,
    )

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "quarantine_respected": True,
        "max_historical_timestamp": "2024-06-01T00:00:00+00:00",
        "experiments": {
            "BTCUSDT_1h_v1_base8": h1_v1,
            "BTCUSDT_1h_v2_financial11": h1_v2,
            "BTCUSDT_1h_v1_noise11": h1_noise,
            "BTCUSDT_1d_v1_base8": d1_v1,
            "BTCUSDT_1d_v2_financial11": d1_v2,
            "BTCUSDT_1d_v1_noise11": d1_noise,
        },
        "evaluator_audit": sim_study,
    }
    out_file = output_dir / "adaptive_online_study_results.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_path": str(out_file),
        "summary": {
            exp_name: {
                "bars": exp_data["bars_total"],
                "champions": exp_data["champions"],
            }
            for exp_name, exp_data in payload["experiments"].items()
        },
    }
