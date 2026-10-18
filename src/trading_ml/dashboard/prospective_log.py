from __future__ import annotations

import json
import math
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from trading_ml.config.settings import INTERVALS
from trading_ml.dashboard.inference_engine import LiveInferenceResult, run_causal_inference
from trading_ml.dashboard.models_catalog import ModelDescriptor

_LOG_THREAD_LOCK = threading.Lock()
DEFAULT_ROUNDTRIP_COST_RATE = 0.0014  # 14 bps round-trip (comisión + slippage + spread realista)


def default_log_path(root: Path) -> Path:
    return root / "live_dashboard" / "prospective_predictions.jsonl"


def _prediction_key(model_version: str, asset: str, interval: str, reference_time: str) -> str:
    ref_norm = pd.Timestamp(reference_time).tz_convert("UTC").isoformat()
    return f"{model_version}__{asset.upper()}__{interval}__{ref_norm}"


def _read_records_unlocked(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            records.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return records


def _write_records_unlocked(log_path: Path, records: list[dict[str, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = log_path.with_suffix(".jsonl.tmp")
    payload = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)
    if payload:
        payload += "\n"
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(log_path)


def record_prospective_prediction(
    result: LiveInferenceResult,
    log_path: Path | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Registra una predicción prospectiva de forma idempotente (sin duplicar cierres)."""
    target_path = log_path or default_log_path(root or Path("artifacts"))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_path.with_suffix(".jsonl.lock")

    key = _prediction_key(
        result.model_version, result.asset, result.interval, result.reference_time
    )

    with _LOG_THREAD_LOCK, FileLock(str(lock_path), timeout=10):
        records = _read_records_unlocked(target_path)
        for existing in records:
            if existing.get("prediction_id") == key:
                return existing

        entry: dict[str, Any] = {
            "prediction_id": key,
            "log_schema": "prospective_live_v2",
            "separated_from_backtest": True,
            "model_id": result.model_id,
            "display_name": result.display_name,
            "model_version": result.model_version,
            "model_type": result.model_type,
            "category": result.category,
            "output_type": result.output_type,
            "objective_description": result.objective_description,
            "asset": result.asset,
            "interval": result.interval,
            "horizon": result.horizon,
            "horizon_label": result.horizon_label,
            "candle_open_time": result.candle_open_time,
            "reference_time": pd.Timestamp(result.reference_time).tz_convert("UTC").isoformat(),
            "target_reference_time": pd.Timestamp(result.target_reference_time)
            .tz_convert("UTC")
            .isoformat(),
            "logged_at_utc": result.generated_at_utc,
            "input_data_sha256": result.input_data_sha256,
            "feature_schema_version": result.feature_schema_version,
            "reference_close": result.reference_close,
            "prediction": result.prediction,
            "state_code": result.state_code,
            "probability_up": result.probability_up,
            "probability_down": result.probability_down,
            "raw_probability_up": result.raw_probability_up,
            "primary_metric_label": result.primary_metric_label,
            "signal_value": result.signal_value,
            "target_weight": result.target_weight,
            "opportunity_active": result.opportunity_active,
            "threshold": result.threshold,
            "is_calibrated": result.is_calibrated,
            "calibration_summary": result.calibration_summary,
            "status": "PENDIENTE",
            "target_close": None,
            "realized_return": None,
            "realized_direction": None,
            "direction_correct": None,
            "hypothetical_net_return": None,
            "outcome_label": "PENDIENTE (esperando cierre de vela objetivo)",
            "evaluated_at_utc": None,
        }
        records.append(entry)
        _write_records_unlocked(target_path, records)
        return entry


def evaluate_pending_predictions(
    closed_candles: pd.DataFrame,
    asset: str,
    interval: str,
    log_path: Path | None = None,
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Evalúa predicciones pendientes una vez que la vela del horizonte objetivo cerró."""
    target_path = log_path or default_log_path(root or Path("artifacts"))
    if not target_path.exists() or closed_candles.empty:
        return load_prospective_predictions(target_path, asset=asset, interval=interval)

    delta = pd.Timedelta(INTERVALS[interval])
    df = closed_candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["reference_time"] = df["timestamp"] + delta
    close_by_ref: dict[pd.Timestamp, float] = {
        pd.Timestamp(row.reference_time).tz_convert("UTC"): float(row.close)
        for row in df.itertuples(index=False)
    }

    lock_path = target_path.with_suffix(".jsonl.lock")
    changed = False

    with _LOG_THREAD_LOCK, FileLock(str(lock_path), timeout=10):
        records = _read_records_unlocked(target_path)
        for rec in records:
            if rec.get("asset", "").upper() != asset.upper() or rec.get("interval") != interval:
                continue
            if rec.get("status") == "EVALUADA":
                continue

            target_ref = pd.Timestamp(rec["target_reference_time"]).tz_convert("UTC")
            ref_ts = pd.Timestamp(rec["reference_time"]).tz_convert("UTC")
            if target_ref not in close_by_ref:
                continue

            target_close = close_by_ref[target_ref]
            ref_close = float(rec.get("reference_close") or close_by_ref.get(ref_ts, 0.0))
            if ref_close <= 0:
                continue

            realized_return = (target_close / ref_close) - 1.0
            if target_close > ref_close:
                realized_dir = "SUBE"
            elif target_close < ref_close:
                realized_dir = "BAJA"
            else:
                realized_dir = "SIN CAMBIO"

            output_type = str(rec.get("output_type", "directional_probability"))
            state_code = str(rec.get("state_code", "neutral"))
            pred = str(rec.get("prediction", "SIN SEÑAL"))
            target_weight = rec.get("target_weight")

            if output_type == "continuous_position" and isinstance(target_weight, (int, float)):
                w = float(target_weight)
                net_ret = w * realized_return - abs(w) * (DEFAULT_ROUNDTRIP_COST_RATE / 2.0)
                if abs(w) <= 0.05:
                    is_correct = None
                    outcome_label = f"NEUTRAL / EFECTIVO (mercado: {realized_dir} {realized_return:+.2%})"
                else:
                    is_correct = bool((w > 0 and realized_return > 0) or (w < 0 and realized_return < 0))
                    outcome_label = (
                        f"DIRECCIÓN FAVORABLE ({net_ret:+.2%} neto)"
                        if is_correct
                        else f"DIRECCIÓN ADVERSA ({net_ret:+.2%} neto)"
                    )
            elif state_code == "bullish" or pred == "SUBE":
                net_ret = realized_return - DEFAULT_ROUNDTRIP_COST_RATE
                if output_type == "net_return_probability":
                    is_correct = bool(net_ret > 0)
                else:
                    is_correct = bool(realized_dir == "SUBE")
                outcome_label = "CORRECTO" if is_correct else "INCORRECTO"
            elif state_code == "bearish" or pred == "BAJA":
                net_ret = 0.0  # Ejecución spot long/cash: señal bajista permanece en efectivo
                is_correct = bool(realized_dir == "BAJA")
                outcome_label = "CORRECTO" if is_correct else "INCORRECTO"
            else:
                net_ret = 0.0
                is_correct = None
                outcome_label = f"SIN SEÑAL (mercado: {realized_dir})"

            rec["status"] = "EVALUADA"
            rec["target_close"] = target_close
            rec["realized_return"] = realized_return
            rec["realized_direction"] = realized_dir
            rec["direction_correct"] = is_correct
            rec["hypothetical_net_return"] = net_ret
            rec["outcome_label"] = outcome_label
            rec["evaluated_at_utc"] = datetime.now(UTC).isoformat()
            changed = True

        if changed:
            _write_records_unlocked(target_path, records)

    return load_prospective_predictions(target_path, asset=asset, interval=interval)


def seed_recent_closed_predictions(
    descriptor: ModelDescriptor,
    closed_candles: pd.DataFrame,
    *,
    peer_closed_candles: pd.DataFrame | None = None,
    max_recent: int = 8,
    log_path: Path | None = None,
    root: Path | None = None,
    now_utc: pd.Timestamp | None = None,
) -> None:
    """Registra causalmente las últimas velas cerradas fuera de muestra si aún no están en el registro."""
    if not descriptor.is_deployable or closed_candles.empty or max_recent <= 0:
        return
    delta = pd.Timedelta(INTERVALS[descriptor.interval])
    df = closed_candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["reference_time"] = df["timestamp"] + delta

    cutoff = (
        pd.Timestamp(descriptor.selection_label_end).tz_convert("UTC")
        if descriptor.selection_label_end
        else pd.Timestamp("1970-01-01", tz="UTC")
    )
    eligible = df.loc[df["reference_time"] > cutoff]
    if eligible.empty:
        return

    target_path = log_path or default_log_path(root or Path("artifacts"))
    existing_keys: set[str] = set()
    if target_path.exists():
        for rec in load_prospective_predictions(
            log_path=target_path,
            asset=descriptor.symbol,
            interval=descriptor.interval,
            model_version=descriptor.version,
            limit=200,
        ):
            existing_keys.add(str(rec.get("prediction_id", "")))

    recent_refs = list(eligible["reference_time"].tail(max_recent))
    for ref_ts in recent_refs:
        key = _prediction_key(
            descriptor.version, descriptor.symbol, descriptor.interval, pd.Timestamp(ref_ts).isoformat()
        )
        if key in existing_keys:
            continue
        try:
            res = run_causal_inference(
                descriptor,
                df,
                peer_closed_candles=peer_closed_candles,
                reference_ts=pd.Timestamp(ref_ts),
                now_utc=now_utc,
            )
            record_prospective_prediction(res, log_path=target_path, root=root)
        except Exception:
            continue

    evaluate_pending_predictions(
        df,
        asset=descriptor.symbol,
        interval=descriptor.interval,
        log_path=target_path,
        root=root,
    )


def load_prospective_predictions(
    log_path: Path | None = None,
    *,
    root: Path | None = None,
    asset: str | None = None,
    interval: str | None = None,
    model_version: str | None = None,
    model_versions: list[str] | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Devuelve las predicciones prospectivas ordenadas de más reciente a más antigua."""
    target_path = log_path or default_log_path(root or Path("artifacts"))
    if not target_path.exists():
        return []
    lock_path = target_path.with_suffix(".jsonl.lock")
    with _LOG_THREAD_LOCK, FileLock(str(lock_path), timeout=10):
        records = _read_records_unlocked(target_path)

    version_set = set(model_versions) if model_versions is not None else None
    filtered: list[dict[str, Any]] = []
    for rec in records:
        if asset and rec.get("asset", "").upper() != asset.upper():
            continue
        if interval and rec.get("interval") != interval:
            continue
        if model_version and rec.get("model_version") != model_version:
            continue
        if version_set is not None and rec.get("model_version") not in version_set:
            continue
        filtered.append(rec)

    filtered.sort(
        key=lambda r: (str(r.get("reference_time", "")), str(r.get("model_version", ""))),
        reverse=True,
    )
    return filtered[:limit]


def _wilson_interval(successes: int, total: int, z: float = 1.95996) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def compute_model_prospective_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calcula métricas prospectivas rigurosas por modelo informando siempre el tamaño muestral."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        ver = str(rec.get("model_version", ""))
        if ver:
            by_model.setdefault(ver, []).append(rec)

    summaries: list[dict[str, Any]] = []
    for ver, rows in by_model.items():
        first = rows[0]
        display_name = str(first.get("display_name") or ver)
        output_type = str(first.get("output_type", "directional_probability"))
        horizon = int(first.get("horizon", 1))
        category = str(first.get("category", "ML_MODEL"))

        total_n = len(rows)
        evaluated = [r for r in rows if r.get("status") == "EVALUADA"]
        eval_n = len(evaluated)
        pending_n = total_n - eval_n

        active_eval = [r for r in evaluated if r.get("direction_correct") is not None]
        active_n = len(active_eval)
        coverage = (active_n / eval_n) if eval_n > 0 else 0.0

        # Accuracy sobre señales activas
        correct_active = sum(1 for r in active_eval if r.get("direction_correct") is True)
        signal_acc = (correct_active / active_n) if active_n > 0 else None
        ci_low, ci_high = _wilson_interval(correct_active, active_n)

        # Matriz de confusión, Precision, Recall, Brier y Log-Loss para modelos probabilísticos
        probs: list[float] = []
        y_true: list[int] = []
        tp = fp = tn = fn = 0

        for r in evaluated:
            p_up = r.get("probability_up")
            real_ret = r.get("realized_return")
            if not isinstance(p_up, (int, float)) or not isinstance(real_ret, (int, float)):
                continue
            p_clamped = float(np.clip(p_up, 1e-6, 1.0 - 1e-6))
            if output_type == "net_return_probability":
                y = 1 if (real_ret - DEFAULT_ROUNDTRIP_COST_RATE) > 0 else 0
            else:
                y = 1 if real_ret > 0 else 0
            probs.append(p_clamped)
            y_true.append(y)
            pred_bin = 1 if p_clamped >= 0.5 else 0
            if pred_bin == 1 and y == 1:
                tp += 1
            elif pred_bin == 1 and y == 0:
                fp += 1
            elif pred_bin == 0 and y == 0:
                tn += 1
            else:
                fn += 1

        prob_n = len(probs)
        if prob_n > 0:
            p_arr = np.asarray(probs, dtype=float)
            y_arr = np.asarray(y_true, dtype=float)
            overall_acc_50 = float(((p_arr >= 0.5).astype(float) == y_arr).mean())
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
            brier = float(np.mean((p_arr - y_arr) ** 2))
            log_loss = float(-np.mean(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr)))
            cal_gap = float(abs(np.mean(p_arr) - np.mean(y_arr)))
        else:
            overall_acc_50 = precision = recall = brier = log_loss = cal_gap = None

        net_rets = [
            float(r["hypothetical_net_return"])
            for r in evaluated
            if isinstance(r.get("hypothetical_net_return"), (int, float))
        ]
        cum_net_return = (
            float(np.prod([1.0 + x for x in net_rets]) - 1.0) if net_rets else 0.0
        )

        if eval_n < 30:
            evidence_note = f"Muestra pequeña (n={eval_n} < 30): sin evidencia estadística suficiente"
        else:
            evidence_note = f"Muestra evaluada n={eval_n} (IC 95% Wilson [{ci_low:.1%}, {ci_high:.1%}])" if ci_low is not None else f"Muestra evaluada n={eval_n}"

        summaries.append(
            {
                "model_version": ver,
                "display_name": display_name,
                "category": category,
                "output_type": output_type,
                "horizon": horizon,
                "total_predictions": total_n,
                "evaluated_predictions": eval_n,
                "pending_predictions": pending_n,
                "active_signals": active_n,
                "signal_coverage": coverage,
                "signal_accuracy": signal_acc,
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "overall_accuracy_50": overall_acc_50,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "brier_score": brier,
                "log_loss": log_loss,
                "calibration_gap": cal_gap,
                "hypothetical_net_return": cum_net_return,
                "evidence_note": evidence_note,
            }
        )

    return sorted(
        summaries,
        key=lambda s: (
            -int(s["evaluated_predictions"]),
            s["brier_score"] if s["brier_score"] is not None else 9.0,
        ),
    )


def compute_common_window_comparison(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compara modelos exclusivamente sobre ventanas temporales comunes y objetivos equivalentes."""
    evaluated = [r for r in records if r.get("status") == "EVALUADA"]
    if not evaluated:
        return []

    # Agrupar por (asset, interval, horizon, output_type) para prohibir comparaciones inválidas
    groups: dict[tuple[str, str, int, str], dict[str, dict[str, dict[str, Any]]]] = {}
    for rec in evaluated:
        g_key = (
            str(rec.get("asset", "")),
            str(rec.get("interval", "")),
            int(rec.get("horizon", 1)),
            str(rec.get("output_type", "directional_probability")),
        )
        ver = str(rec.get("model_version", ""))
        ref = str(rec.get("reference_time", ""))
        groups.setdefault(g_key, {}).setdefault(ver, {})[ref] = rec

    comparisons: list[dict[str, Any]] = []
    for (asset, interval, horizon, output_type), model_map in groups.items():
        if len(model_map) < 1:
            continue
        common_refs = set.intersection(*(set(ref_dict.keys()) for ref_dict in model_map.values()))
        if not common_refs:
            continue
        sorted_refs = sorted(common_refs)
        common_n = len(sorted_refs)
        window_str = f"{sorted_refs[0][:10]} → {sorted_refs[-1][:10]} (n={common_n} cierres comunes)"

        for _ver, ref_dict in model_map.items():
            sub_rows = [ref_dict[ref] for ref in sorted_refs]
            single_metrics = compute_model_prospective_metrics(sub_rows)[0]
            comparisons.append(
                {
                    "group_label": f"{asset} · {interval} · h={horizon} · {output_type}",
                    "common_window": window_str,
                    "common_n": common_n,
                    **single_metrics,
                }
            )

    return comparisons


def compute_live_agreement_summary(
    outcomes: list[LiveInferenceResult],
) -> list[dict[str, Any]]:
    """Calcula nivel de acuerdo y desacuerdo entre modelos compatibles sobre la vela cerrada actual."""
    groups: dict[tuple[str, int, str], list[LiveInferenceResult]] = {}
    for res in outcomes:
        key = (res.asset, res.horizon, res.output_type)
        groups.setdefault(key, []).append(res)

    summaries: list[dict[str, Any]] = []
    for (asset, horizon, out_type), items in groups.items():
        total = len(items)
        bull = sum(1 for x in items if x.state_code == "bullish")
        bear = sum(1 for x in items if x.state_code == "bearish")
        neut = sum(1 for x in items if x.state_code == "neutral")
        probs = [x.probability_up for x in items if x.probability_up is not None]
        mean_p = float(np.mean(probs)) if probs else None
        std_p = float(np.std(probs)) if len(probs) > 1 else 0.0

        majority_count = max(bull, bear, neut)
        agreement_rate = majority_count / total if total > 0 else 0.0
        if bull > bear and bull >= neut:
            consensus = "ALCISTA / SUBE"
        elif bear > bull and bear >= neut:
            consensus = "BAJISTA / BAJA"
        else:
            consensus = "ABSTENCIÓN / NEUTRAL"

        summaries.append(
            {
                "Grupo Compatible": f"{asset} · h={horizon}d · {out_type}",
                "Modelos Comparados": total,
                "Alcistas (SUBE / Long)": bull,
                "Bajistas (BAJA / Short)": bear,
                "Neutrales (SIN SEÑAL)": neut,
                "Consenso Mayoritario": consensus,
                "Grado de Acuerdo": f"{agreement_rate:.1%}",
                "P(subida) Media": f"{mean_p:.2%}" if mean_p is not None else "N/A (Posición)",
                "Dispersión (Desv. Est.)": f"{std_p:.2%}" if mean_p is not None else "N/A",
            }
        )
    return summaries
