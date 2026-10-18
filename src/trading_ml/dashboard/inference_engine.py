from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from trading_ml.advanced_research.features import build_research_features
from trading_ml.common import frame_hash, read_json
from trading_ml.config.settings import INTERVALS, Features
from trading_ml.dashboard.models_catalog import ModelDescriptor, verify_model_integrity
from trading_ml.dmn.data import SCALES
from trading_ml.dmn.data import causal_features as dmn_causal_features
from trading_ml.dmn.models import apply_scaler, load_network, signal_from_output
from trading_ml.errors import DataError, LeakageError
from trading_ml.features.technical import build_features
from trading_ml.frontier_research.features import multi_asset_features
from trading_ml.frontier_research.models import predict_probability
from trading_ml.mean_reversion.config import ReversionConfig
from trading_ml.mean_reversion.features import build_reversion_features
from trading_ml.mean_reversion.models import load_filter
from trading_ml.mean_reversion.rules import reversion_opportunities
from trading_ml.models.calibration import apply_calibration
from trading_ml.momentum.config import MomentumConfig
from trading_ml.momentum.features import build_momentum_features, causal_regimes
from trading_ml.momentum.rules import momentum_signals
from trading_ml.multimarket.features import features as multimarket_features
from trading_ml.multimarket.service import load_candidate
from trading_ml.smc.rules import directional_states, rule_probabilities
from trading_ml.structure_regime.features import CLUSTER_COLUMNS, extend_structure

_CACHE_LOCK = threading.Lock()
_MODEL_OBJECT_CACHE: dict[tuple[str, str], Any] = {}
_FEATURE_FRAME_CACHE: dict[tuple[str, str, str, int], Any] = {}
_INFERENCE_RESULT_CACHE: dict[tuple[str, str, str, str], LiveInferenceResult] = {}


@dataclass(frozen=True)
class LiveInferenceResult:
    """Resultado causal de un modelo o estrategia sobre una vela estrictamente cerrada."""

    model_id: str
    display_name: str
    model_version: str
    model_type: str
    category: str  # "ML_MODEL" | "DETERMINISTIC_RULE"
    output_type: str  # "directional_probability" | "net_return_probability" | "barrier_probability" | "opportunity_filter_probability" | "continuous_position"
    objective_description: str
    asset: str
    interval: str
    horizon: int
    horizon_label: str
    candle_open_time: str
    reference_time: str
    target_reference_time: str
    generated_at_utc: str
    reference_close: float
    prediction: str  # "SUBE", "BAJA", "SIN SEÑAL", "FAVORABLE (Long)", "POSICIÓN LONG", etc.
    state_code: str  # "bullish" | "bearish" | "neutral"
    probability_up: float | None
    probability_down: float | None
    raw_probability_up: float | None
    primary_metric_label: str  # Ej: "P(subida): 54.20%" o "Peso objetivo: +42.1% (señal 0.65)"
    threshold: float
    is_calibrated: bool
    calibration_summary: str
    calibration_warning: str | None
    selection_label_end: str
    input_data_sha256: str
    feature_schema_version: str
    signal_value: float | None = None
    target_weight: float | None = None
    opportunity_active: bool | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelInferenceOutcome:
    """Envoltorio con aislamiento de errores para ejecución simultánea de múltiples modelos."""

    descriptor: ModelDescriptor
    result: LiveInferenceResult | None
    error: str | None
    from_cache: bool = False

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.error is None


def clear_inference_caches() -> None:
    with _CACHE_LOCK:
        _MODEL_OBJECT_CACHE.clear()
        _FEATURE_FRAME_CACHE.clear()
        _INFERENCE_RESULT_CACHE.clear()


def map_state_to_spanish(state_code: str) -> str:
    return {
        "bullish": "SUBE",
        "bearish": "BAJA",
        "neutral": "SIN SEÑAL",
    }.get(state_code, "SIN SEÑAL")


def validate_strictly_closed_candles(
    closed_candles: pd.DataFrame,
    interval: str,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Garantiza que ninguna vela en formación se filtre al cálculo de features o inferencia."""
    if closed_candles.empty:
        raise DataError("No hay velas cerradas disponibles para ejecutar inferencia")
    if interval not in INTERVALS:
        raise DataError(f"Intervalo inválido: {interval}")
    delta = pd.Timedelta(INTERVALS[interval])
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    df = closed_candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "is_closed" in df.columns and not df["is_closed"].astype(bool).all():
        raise LeakageError("Se detectó una vela marcada como abierta (is_closed=False) en inferencia")
    if ((df["timestamp"] + delta) > now).any():
        raise LeakageError("Fuga temporal: una vela cuyo cierre aún no ocurrió ingresó a inferencia")
    if df["timestamp"].duplicated().any():
        raise DataError("Timestamps duplicados en el historial de velas cerradas")
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)
    df["reference_time"] = df["timestamp"] + delta
    return df


def _crop_to_reference(df: pd.DataFrame, reference_ts: pd.Timestamp | None) -> pd.DataFrame:
    if reference_ts is None:
        return df
    ref = pd.Timestamp(reference_ts)
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    else:
        ref = ref.tz_convert("UTC")
    cropped = df.loc[df["reference_time"] <= ref].reset_index(drop=True)
    if cropped.empty or cropped.iloc[-1]["reference_time"] != ref:
        raise DataError(f"No existe vela cerrada con reference_time={ref.isoformat()}")
    return cropped


def _check_post_cutoff(last_ref: pd.Timestamp, selection_label_end: str) -> None:
    if selection_label_end:
        cutoff = pd.Timestamp(selection_label_end)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        if last_ref <= cutoff:
            raise LeakageError(
                f"La vela cerrada ({last_ref.isoformat()}) pertenece al período de entrenamiento/selección (<= {cutoff.isoformat()})"
            )


def _cached_model(key: tuple[str, str], loader: Any) -> Any:
    with _CACHE_LOCK:
        if key in _MODEL_OBJECT_CACHE:
            return _MODEL_OBJECT_CACHE[key]
    loaded = loader()
    with _CACHE_LOCK:
        _MODEL_OBJECT_CACHE[key] = loaded
    return loaded


def _cached_features(
    pipeline_id: str,
    symbol: str,
    last_ref_iso: str,
    rows_len: int,
    builder: Any,
) -> Any:
    key = (pipeline_id, symbol.upper(), last_ref_iso, rows_len)
    with _CACHE_LOCK:
        if key in _FEATURE_FRAME_CACHE:
            return _FEATURE_FRAME_CACHE[key]
    built = builder()
    with _CACHE_LOCK:
        if len(_FEATURE_FRAME_CACHE) > 64:
            _FEATURE_FRAME_CACHE.clear()
        _FEATURE_FRAME_CACHE[key] = built
    return built


def _input_digest(df: pd.DataFrame) -> str:
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in df.columns]
    return frame_hash(df.tail(15)[cols].reset_index(drop=True))


def _infer_registry_model(
    descriptor: ModelDescriptor,
    df: pd.DataFrame,
    delta: pd.Timedelta,
) -> LiveInferenceResult:
    bundle = descriptor.bundle
    if bundle is None:
        raise DataError("ModelBundle no disponible")
    verify_model_integrity(descriptor)

    feat_cfg = Features(**bundle.metadata["contract"]["features"])
    pipe_key = f"tech_enh{int(feat_cfg.enhanced)}_smc{int(feat_cfg.smc)}"
    last_ref = pd.Timestamp(df["reference_time"].iloc[-1])
    _check_post_cutoff(last_ref, descriptor.selection_label_end)

    def _build() -> pd.DataFrame:
        f = build_features(df, feat_cfg)
        f["reference_time"] = df["reference_time"]
        return f

    featured = _cached_features(pipe_key, descriptor.symbol, last_ref.isoformat(), len(df), _build)
    row = featured.iloc[[-1]]
    last_open = pd.Timestamp(row["timestamp"].iloc[0])
    target_ref = last_ref + descriptor.horizon * delta

    if row[descriptor.features].isna().any(axis=1).iloc[0]:
        raise DataError(
            "Historial insuficiente de velas cerradas para completar el calentamiento (warmup) de los indicadores"
        )

    raw_prob = float(bundle.model.predict_proba(row[descriptor.features])[:, 1][0])
    final_prob = float(bundle.probabilities(row)[0])

    smc_ready = True
    if "smc_ready" in row.columns:
        smc_ready = bool(row["smc_ready"].iloc[0] > 0)

    state_code = str(
        directional_states(
            np.asarray([final_prob], dtype=float),
            threshold=descriptor.threshold,
            ready=np.asarray([smc_ready], dtype=bool),
        )[0]
    )
    prediction_label = map_state_to_spanish(state_code)

    warnings: list[str] = []
    if descriptor.calibration_warning:
        warnings.append(descriptor.calibration_warning)
    if not smc_ready:
        warnings.append("Estructura SMC aún en calentamiento; abstiene (SIN SEÑAL).")

    return LiveInferenceResult(
        model_id=descriptor.model_id,
        display_name=descriptor.display_name,
        model_version=descriptor.version,
        model_type=descriptor.architecture,
        category=descriptor.category,
        output_type=descriptor.output_type,
        objective_description=descriptor.objective_description,
        asset=descriptor.symbol,
        interval=descriptor.interval,
        horizon=descriptor.horizon,
        horizon_label=descriptor.horizon_label,
        candle_open_time=last_open.isoformat(),
        reference_time=last_ref.isoformat(),
        target_reference_time=target_ref.isoformat(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        reference_close=float(row["close"].iloc[0]),
        prediction=prediction_label,
        state_code=state_code,
        probability_up=final_prob,
        probability_down=float(1.0 - final_prob),
        raw_probability_up=raw_prob,
        primary_metric_label=f"P(subida): {final_prob:.2%}",
        threshold=descriptor.threshold,
        is_calibrated=descriptor.is_calibrated,
        calibration_summary=descriptor.calibration_summary,
        calibration_warning=descriptor.calibration_warning,
        selection_label_end=descriptor.selection_label_end,
        input_data_sha256=_input_digest(df),
        feature_schema_version=pipe_key,
        warnings=warnings,
    )


def _infer_frontier_model(
    descriptor: ModelDescriptor,
    own_df: pd.DataFrame,
    peer_df: pd.DataFrame | None,
    delta: pd.Timedelta,
) -> LiveInferenceResult:
    verify_model_integrity(descriptor)
    last_ref = pd.Timestamp(own_df["reference_time"].iloc[-1])
    _check_post_cutoff(last_ref, descriptor.selection_label_end)

    if peer_df is None or peer_df.empty:
        raise DataError("Este modelo requiere velas cerradas alineadas de BTCUSDT y ETHUSDT")

    peer_symbol = "ETHUSDT" if descriptor.symbol == "BTCUSDT" else "BTCUSDT"
    frames = _cached_features(
        "multi_asset_features",
        f"{descriptor.symbol}_{peer_symbol}",
        last_ref.isoformat(),
        min(len(own_df), len(peer_df)),
        lambda: multi_asset_features({descriptor.symbol: own_df, peer_symbol: peer_df}),
    )
    featured = frames[descriptor.symbol]
    row = featured.iloc[[-1]]
    last_open = pd.Timestamp(row["timestamp"].iloc[0])
    target_ref = last_ref + descriptor.horizon * delta

    if row[descriptor.features].isna().any(axis=1).iloc[0]:
        raise DataError("Warmup insuficiente o datos cruzados faltantes en la última vela cerrada")

    frontier_dir = Path(descriptor.extra_meta["frontier_dir"])
    recipe = str(descriptor.extra_meta["recipe"])

    if recipe == "stacking":
        state = descriptor.extra_meta["state"]
        comp_order = [
            "logistic_cross",
            "xgboost_cross",
            "random_forest_cross",
            "lightgbm_cross",
            "catboost_cross",
        ]
        raw_list = []
        for c_name in comp_order:
            comp = state["components"][c_name]
            c_path = frontier_dir / Path(comp["path"])
            c_model = _cached_model((f"{descriptor.model_id}:{c_name}", comp["sha256"]), lambda p=c_path: joblib.load(p))
            raw_list.append(float(predict_probability(c_model, row, comp["columns"])[0]))
        stack_path = frontier_dir / Path(state["stack"]["path"])
        stack_model = _cached_model(
            (f"{descriptor.model_id}:stack", state["stack"]["sha256"]),
            lambda: joblib.load(stack_path),
        )
        raw_prob = float(np.mean(raw_list))
        final_prob = float(stack_model.predict_proba(np.asarray([raw_list]))[:, 1][0])
    else:
        comp = descriptor.extra_meta["component"]
        c_path = frontier_dir / Path(comp["path"])
        model = _cached_model(
            (descriptor.model_id, comp["sha256"]),
            lambda: joblib.load(c_path),
        )
        raw_arr = predict_probability(model, row, descriptor.features)
        cal_arr = apply_calibration(raw_arr, comp.get("calibration"))
        raw_prob = float(raw_arr[0])
        final_prob = float(cal_arr[0])

    state_code = str(
        directional_states(np.asarray([final_prob], dtype=float), threshold=descriptor.threshold)[0]
    )
    if descriptor.output_type == "directional_probability":
        prediction_label = map_state_to_spanish(state_code)
        metric_label = f"P(subida 1d): {final_prob:.2%}"
    elif descriptor.output_type == "net_return_probability":
        prediction_label = (
            "FAVORABLE (Long 3d)"
            if state_code == "bullish"
            else ("DESFAVORABLE (3d)" if state_code == "bearish" else "SIN SEÑAL")
        )
        metric_label = f"P(retorno neto > 0 a 3d): {final_prob:.2%}"
    else:
        prediction_label = (
            "FAVORABLE (TP 2xATR 5d)"
            if state_code == "bullish"
            else ("DESFAVORABLE (SL/Timeout 5d)" if state_code == "bearish" else "SIN SEÑAL")
        )
        metric_label = f"P(barrera TP a 5d): {final_prob:.2%}"

    warnings = [descriptor.calibration_warning] if descriptor.calibration_warning else []
    return LiveInferenceResult(
        model_id=descriptor.model_id,
        display_name=descriptor.display_name,
        model_version=descriptor.version,
        model_type=descriptor.architecture,
        category=descriptor.category,
        output_type=descriptor.output_type,
        objective_description=descriptor.objective_description,
        asset=descriptor.symbol,
        interval=descriptor.interval,
        horizon=descriptor.horizon,
        horizon_label=descriptor.horizon_label,
        candle_open_time=last_open.isoformat(),
        reference_time=last_ref.isoformat(),
        target_reference_time=target_ref.isoformat(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        reference_close=float(row["close"].iloc[0]),
        prediction=prediction_label,
        state_code=state_code,
        probability_up=final_prob,
        probability_down=float(1.0 - final_prob),
        raw_probability_up=raw_prob,
        primary_metric_label=metric_label,
        threshold=descriptor.threshold,
        is_calibrated=descriptor.is_calibrated,
        calibration_summary=descriptor.calibration_summary,
        calibration_warning=descriptor.calibration_warning,
        selection_label_end=descriptor.selection_label_end,
        input_data_sha256=_input_digest(own_df),
        feature_schema_version="frontier_multi_asset_v1",
        warnings=warnings,
    )


def _infer_retraining_or_cluster_model(
    descriptor: ModelDescriptor,
    own_df: pd.DataFrame,
    peer_df: pd.DataFrame | None,
    delta: pd.Timedelta,
) -> LiveInferenceResult:
    verify_model_integrity(descriptor)
    last_ref = pd.Timestamp(own_df["reference_time"].iloc[-1])
    _check_post_cutoff(last_ref, descriptor.selection_label_end)

    needs_cross = any(c.startswith("peer_") or c.startswith("cross_") or c == "spread_z_60" for c in descriptor.features)
    if needs_cross:
        if peer_df is None or peer_df.empty:
            raise DataError("Este modelo requiere velas cerradas alineadas de BTCUSDT y ETHUSDT")
        peer_symbol = "ETHUSDT" if descriptor.symbol == "BTCUSDT" else "BTCUSDT"
        frames = _cached_features(
            "multi_asset_features",
            f"{descriptor.symbol}_{peer_symbol}",
            last_ref.isoformat(),
            min(len(own_df), len(peer_df)),
            lambda: multi_asset_features({descriptor.symbol: own_df, peer_symbol: peer_df}),
        )
        base_frame = frames[descriptor.symbol]
    else:
        base_frame = _cached_features(
            "research_features_single",
            descriptor.symbol,
            last_ref.isoformat(),
            len(own_df),
            lambda: build_research_features(own_df),
        )

    structure_cfg = descriptor.extra_meta.get(
        "structure_config",
        {
            "retest_bars": 5,
            "tolerance_atr": 0.35,
            "impulse_atr": 0.9,
            "impulse_close_location": 0.65,
        },
    )
    extended = _cached_features(
        f"extended_structure_{needs_cross}",
        descriptor.symbol,
        last_ref.isoformat(),
        len(base_frame),
        lambda: extend_structure(base_frame, structure_cfg),
    )

    if descriptor.source_kind == "structure_cluster":
        clusters_info = read_json(Path(descriptor.extra_meta["clusters_path"]))
        in_cols = list(clusters_info["input_columns"])
        row = extended.iloc[[-1]].copy()
        if row[in_cols].isna().any(axis=1).iloc[0]:
            raise DataError("Warmup insuficiente para calcular clusters de régimen")
        x_val = row[in_cols].to_numpy(dtype=float)
        mean = np.asarray(clusters_info["mean"], dtype=float)
        scale = np.asarray(clusters_info["scale"], dtype=float)
        centers = np.asarray(clusters_info["centers"], dtype=float)
        z = (x_val - mean) / scale
        dists = np.linalg.norm(centers - z, axis=1)
        cluster_idx = int(np.argmin(dists))
        for i, c_col in enumerate(CLUSTER_COLUMNS):
            row[c_col] = float(i == cluster_idx)

        ubj_path = Path(descriptor.extra_meta["ubj_path"])

        def _load_cluster_xgb() -> XGBClassifier:
            m = XGBClassifier()
            m.load_model(ubj_path)
            return m

        model = _cached_model((descriptor.model_id, descriptor.artifact_sha256 or ""), _load_cluster_xgb)
        raw_prob = float(model.predict_proba(row[descriptor.features].to_numpy(dtype=float))[:, 1][0])
        final_prob = raw_prob
        state_note = f"Cluster K-Means activo: C{cluster_idx}"
    else:
        row = extended.iloc[[-1]].copy()
        if row[descriptor.features].isna().any(axis=1).iloc[0]:
            raise DataError("Warmup insuficiente para las features de estructura/régimen")
        rec_dir = Path(descriptor.extra_meta["rec_dir"])
        recipe_key = str(descriptor.extra_meta["recipe_key"])
        model_path = rec_dir / "model.joblib"
        loaded_obj = _cached_model(
            (descriptor.model_id, descriptor.artifact_sha256 or ""),
            lambda: joblib.load(model_path),
        )
        if hasattr(loaded_obj, "probabilities"):
            global_prob = float(loaded_obj.probabilities(row)["logistic_l2"][0])
        elif isinstance(loaded_obj, dict) and "parent" in loaded_obj:
            global_prob = float(
                predict_probability(loaded_obj["parent"], row, descriptor.features)[0]
            )
        else:
            global_prob = float(predict_probability(loaded_obj, row, descriptor.features)[0])

        raw_prob = global_prob
        final_prob = global_prob
        state_note = f"Régimen actual: {row['sr_state'].iloc[0]}" if "sr_state" in row.columns else ""

        if recipe_key == "specialists_rules":
            sr_state = str(row["sr_state"].iloc[0])
            experts_list = (
                loaded_obj.get("experts")
                if isinstance(loaded_obj, dict) and "experts" in loaded_obj
                else read_json(rec_dir / "experts.json")
            )
            exp_entry = next((e for e in experts_list if e.get("state") == sr_state), None)
            exp_ubj = rec_dir / f"expert_{sr_state}.ubj"
            if (
                exp_entry is not None
                and not exp_entry.get("fallback_global", True)
                and exp_ubj.exists()
            ):
                weight = float(exp_entry.get("expert_weight", 0.0))

                def _load_exp(p: Path = exp_ubj) -> XGBClassifier:
                    em = XGBClassifier()
                    em.load_model(p)
                    return em

                exp_model = _cached_model((f"{descriptor.model_id}:{sr_state}", str(exp_ubj)), _load_exp)
                exp_prob = float(
                    exp_model.predict_proba(row[descriptor.features].to_numpy(dtype=float))[:, 1][0]
                )
                final_prob = float(weight * exp_prob + (1.0 - weight) * global_prob)
                state_note = (
                    f"Régimen '{sr_state}': peso experto={weight:.2f} (P_exp={exp_prob:.2%}, P_glob={global_prob:.2%})"
                )

    last_open = pd.Timestamp(row["timestamp"].iloc[0])
    target_ref = last_ref + descriptor.horizon * delta
    state_code = str(
        directional_states(np.asarray([final_prob], dtype=float), threshold=descriptor.threshold)[0]
    )
    prediction_label = map_state_to_spanish(state_code)
    warnings = [w for w in [descriptor.calibration_warning, state_note] if w]

    return LiveInferenceResult(
        model_id=descriptor.model_id,
        display_name=descriptor.display_name,
        model_version=descriptor.version,
        model_type=descriptor.architecture,
        category=descriptor.category,
        output_type=descriptor.output_type,
        objective_description=descriptor.objective_description,
        asset=descriptor.symbol,
        interval=descriptor.interval,
        horizon=descriptor.horizon,
        horizon_label=descriptor.horizon_label,
        candle_open_time=last_open.isoformat(),
        reference_time=last_ref.isoformat(),
        target_reference_time=target_ref.isoformat(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        reference_close=float(row["close"].iloc[0]),
        prediction=prediction_label,
        state_code=state_code,
        probability_up=final_prob,
        probability_down=float(1.0 - final_prob),
        raw_probability_up=raw_prob,
        primary_metric_label=f"P(subida): {final_prob:.2%}",
        threshold=descriptor.threshold,
        is_calibrated=descriptor.is_calibrated,
        calibration_summary=descriptor.calibration_summary,
        calibration_warning=descriptor.calibration_warning,
        selection_label_end=descriptor.selection_label_end,
        input_data_sha256=_input_digest(own_df),
        feature_schema_version=f"{descriptor.source_kind}_v1",
        warnings=warnings,
    )


def _infer_dmn_checkpoint(
    descriptor: ModelDescriptor,
    df: pd.DataFrame,
    delta: pd.Timedelta,
) -> LiveInferenceResult:
    verify_model_integrity(descriptor)
    last_ref = pd.Timestamp(df["reference_time"].iloc[-1])
    _check_post_cutoff(last_ref, descriptor.selection_label_end)

    risk_cfg = descriptor.extra_meta["risk"]
    seq_len = int(descriptor.extra_meta["sequence_length"])
    floor = float(risk_cfg.get("daily_volatility_floor", 1e-4))

    feat_df = _cached_features(
        "dmn_causal_features",
        descriptor.symbol,
        last_ref.isoformat(),
        len(df),
        lambda: dmn_causal_features(df[["timestamp", "open", "high", "low", "close", "volume"]], floor),
    )
    if len(feat_df) < seq_len:
        raise DataError(
            f"Historia insuficiente tras calentamiento DMN: se tienen {len(feat_df)} filas válidas y la secuencia exige {seq_len} (se requieren ~380 velas cerradas)"
        )

    last_row = feat_df.iloc[-1]
    last_open = pd.Timestamp(last_row["timestamp"])
    target_ref = last_ref + descriptor.horizon * delta
    vol_now = max(float(last_row["daily_volatility"]), floor)
    scale_factor = float(risk_cfg["target_volatility"] / (vol_now * np.sqrt(365.0)))
    max_gross = float(risk_cfg["maximum_gross"])

    run_dir = Path(descriptor.extra_meta["run_dir"])
    is_xgb = bool(descriptor.extra_meta.get("is_xgb", False))
    pos_mode = str(descriptor.extra_meta.get("position_mode", "long_only"))

    if is_xgb:
        ubj_path = run_dir / "model.ubj"

        def _load_dmn_xgb() -> XGBClassifier:
            m = XGBClassifier()
            m.load_model(ubj_path)
            return m

        xgb_model = _cached_model((descriptor.model_id, descriptor.artifact_sha256 or ""), _load_dmn_xgb)
        x_last = feat_df.iloc[[-1]][descriptor.features].to_numpy(dtype=np.float32)
        prob_up = float(xgb_model.predict_proba(x_last)[:, 1][0])
        raw_sig = float(np.sign(prob_up - 0.5))
        signal_val = max(0.0, raw_sig) if pos_mode == "long_only" else raw_sig
    else:
        meta = descriptor.extra_meta["metadata"]
        scaler = meta["scaler"]
        recipe = meta["recipe"]
        net_model, _ = _cached_model(
            (descriptor.model_id, descriptor.artifact_sha256 or ""),
            lambda: load_network(run_dir),
        )
        seq_values = feat_df.iloc[-seq_len:][descriptor.features].to_numpy(dtype=np.float32)
        scaled_seq = apply_scaler(seq_values[np.newaxis, :, :], scaler)
        with torch.no_grad():
            out_t = net_model(torch.from_numpy(scaled_seq))
            sig_t = signal_from_output(out_t, recipe["objective"], pos_mode)
            signal_val = float(sig_t.item())
        prob_up = None

    weight_val = float(np.clip(signal_val * scale_factor, -max_gross, max_gross))
    if weight_val > 0.05:
        state_code = "bullish"
        pred_label = f"POSICIÓN LONG ({weight_val:+.1%})"
    elif weight_val < -0.05:
        state_code = "bearish"
        pred_label = f"POSICIÓN SHORT ({weight_val:+.1%})"
    else:
        state_code = "neutral"
        pred_label = f"EN EFECTIVO / NEUTRAL ({weight_val:+.1%})"

    return LiveInferenceResult(
        model_id=descriptor.model_id,
        display_name=descriptor.display_name,
        model_version=descriptor.version,
        model_type=descriptor.architecture,
        category=descriptor.category,
        output_type="continuous_position",
        objective_description=descriptor.objective_description,
        asset=descriptor.symbol,
        interval=descriptor.interval,
        horizon=descriptor.horizon,
        horizon_label=descriptor.horizon_label,
        candle_open_time=last_open.isoformat(),
        reference_time=last_ref.isoformat(),
        target_reference_time=target_ref.isoformat(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        reference_close=float(last_row["close"]),
        prediction=pred_label,
        state_code=state_code,
        probability_up=prob_up,
        probability_down=(1.0 - prob_up) if prob_up is not None else None,
        raw_probability_up=prob_up,
        primary_metric_label=f"Peso objetivo: {weight_val:+.1%} (señal={signal_val:+.3f})",
        threshold=descriptor.threshold,
        is_calibrated=False,
        calibration_summary=descriptor.calibration_summary,
        calibration_warning=descriptor.calibration_warning,
        selection_label_end=descriptor.selection_label_end,
        input_data_sha256=_input_digest(df),
        feature_schema_version=f"dmn_seq{seq_len}_v1",
        signal_value=signal_val,
        target_weight=weight_val,
        warnings=[descriptor.calibration_warning] if descriptor.calibration_warning else [],
    )


def _infer_hybrid_filter_model(
    descriptor: ModelDescriptor,
    df: pd.DataFrame,
    delta: pd.Timedelta,
) -> LiveInferenceResult:
    verify_model_integrity(descriptor)
    last_ref = pd.Timestamp(df["reference_time"].iloc[-1])
    _check_post_cutoff(last_ref, descriptor.selection_label_end)

    if descriptor.source_kind == "momentum_frozen":
        lookback = int(descriptor.extra_meta["lookback"])
        rules_cfg = descriptor.extra_meta["rules_config"]
        cfg = MomentumConfig(lookback=lookback, **rules_cfg)
        feat_df = _cached_features(
            f"momentum_L{lookback}",
            descriptor.symbol,
            last_ref.isoformat(),
            len(df),
            lambda: build_momentum_features(df, cfg),
        )
        sigs = momentum_signals(feat_df, cfg)
        opp_active = bool(sigs["entry"].iloc[-1])
        row = feat_df.iloc[[-1]]
        if not bool(row["mom_ready"].iloc[0]):
            raise DataError(f"Warmup insuficiente para filtro de Momentum L{lookback}")
        ubj_path = Path(descriptor.extra_meta["ubj_path"])

        def _load_mom_xgb() -> XGBClassifier:
            m = XGBClassifier()
            m.load_model(ubj_path)
            return m

        model = _cached_model((descriptor.model_id, descriptor.artifact_sha256 or ""), _load_mom_xgb)
        prob_val = float(model.predict_proba(row[descriptor.features])[:, 1][0])
        rule_name = f"Ruptura Momentum L{lookback}d"
    else:
        lookback = int(descriptor.extra_meta["lookback"])
        rules_cfg = descriptor.extra_meta["rules_config"]
        sideways_only = bool(descriptor.extra_meta["sideways_only"])
        cfg = ReversionConfig(lookback=lookback, sideways_only=sideways_only, **rules_cfg)

        def _build_mr() -> pd.DataFrame:
            f = build_reversion_features(df, cfg)
            f["regime"] = causal_regimes(df)
            return f

        feat_df = _cached_features(
            f"mr_L{lookback}_sw{int(sideways_only)}",
            descriptor.symbol,
            last_ref.isoformat(),
            len(df),
            _build_mr,
        )
        opps = reversion_opportunities(feat_df, cfg)
        opp_active = bool(opps.iloc[-1])
        row = feat_df.iloc[[-1]]
        if not bool(row["mr_ready"].iloc[0]):
            raise DataError(f"Warmup insuficiente para filtro de Mean Reversion L{lookback}")
        filter_dir = Path(descriptor.extra_meta["filter_dir"])
        rev_filter = _cached_model(
            (descriptor.model_id, descriptor.artifact_sha256 or ""),
            lambda: load_filter(filter_dir),
        )
        prob_val = float(rev_filter.probabilities(row)[0])
        rule_name = f"Sobreventa Mean Reversion L{lookback}d"

    last_open = pd.Timestamp(row["timestamp"].iloc[0])
    target_ref = last_ref + descriptor.horizon * delta

    if opp_active and prob_val >= descriptor.threshold:
        state_code = "bullish"
        pred_label = "ENTRADA APROBADA (Regla + ML)"
    elif opp_active:
        state_code = "neutral"
        pred_label = "OPORTUNIDAD RECHAZADA POR FILTRO ML"
    else:
        state_code = "neutral"
        pred_label = "SIN GATILLO DE REGLA (En espera)"

    opp_str = "ACTIVA" if opp_active else "Inactiva en esta vela"
    return LiveInferenceResult(
        model_id=descriptor.model_id,
        display_name=descriptor.display_name,
        model_version=descriptor.version,
        model_type=descriptor.architecture,
        category=descriptor.category,
        output_type="opportunity_filter_probability",
        objective_description=descriptor.objective_description,
        asset=descriptor.symbol,
        interval=descriptor.interval,
        horizon=descriptor.horizon,
        horizon_label=descriptor.horizon_label,
        candle_open_time=last_open.isoformat(),
        reference_time=last_ref.isoformat(),
        target_reference_time=target_ref.isoformat(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        reference_close=float(row["close"].iloc[0]),
        prediction=pred_label,
        state_code=state_code,
        probability_up=prob_val,
        probability_down=float(1.0 - prob_val),
        raw_probability_up=prob_val,
        primary_metric_label=f"P(éxito filtro): {prob_val:.2%} · Regla ({rule_name}): {opp_str}",
        threshold=descriptor.threshold,
        is_calibrated=False,
        calibration_summary=descriptor.calibration_summary,
        calibration_warning=descriptor.calibration_warning,
        selection_label_end=descriptor.selection_label_end,
        input_data_sha256=_input_digest(df),
        feature_schema_version=f"{descriptor.source_kind}_v1",
        opportunity_active=opp_active,
        warnings=[ f"Estado del gatillo determinista ({rule_name}): {opp_str}." ],
    )


def _infer_multimarket_model(
    descriptor: ModelDescriptor,
    df: pd.DataFrame,
    delta: pd.Timedelta,
) -> LiveInferenceResult:
    verify_model_integrity(descriptor)
    last_ref = pd.Timestamp(df["reference_time"].iloc[-1])
    _check_post_cutoff(last_ref, descriptor.selection_label_end)

    def _build_mm() -> pd.DataFrame:
        mm_in = df.copy()
        if "close_bid" not in mm_in.columns or "close_ask" not in mm_in.columns:
            mm_in["close_bid"] = mm_in["close"] * (1.0 - 0.00005)
            mm_in["close_ask"] = mm_in["close"] * (1.0 + 0.00005)
        return multimarket_features(mm_in, descriptor.symbol)

    feat_df = _cached_features(
        "multimarket_price_v1",
        descriptor.symbol,
        last_ref.isoformat(),
        len(df),
        _build_mm,
    )
    row = feat_df.iloc[[-1]]
    if row[descriptor.features].isna().any(axis=1).iloc[0]:
        raise DataError("Warmup insuficiente para las 13 features de Multimarket Price v1")

    model_path = Path(descriptor.extra_meta["model_path"])
    model, meta = _cached_model(
        (descriptor.model_id, descriptor.artifact_sha256 or ""),
        lambda: load_candidate(model_path),
    )
    raw_arr = model.predict_proba(row[descriptor.features])[:, 1]
    cal_arr = apply_calibration(raw_arr, meta.get("calibration"))
    raw_prob = float(raw_arr[0])
    final_prob = float(cal_arr[0])

    last_open = pd.Timestamp(row["timestamp"].iloc[0])
    target_ref = last_ref + descriptor.horizon * delta
    state_code = str(
        directional_states(np.asarray([final_prob], dtype=float), threshold=descriptor.threshold)[0]
    )
    prediction_label = map_state_to_spanish(state_code)

    return LiveInferenceResult(
        model_id=descriptor.model_id,
        display_name=descriptor.display_name,
        model_version=descriptor.version,
        model_type=descriptor.architecture,
        category=descriptor.category,
        output_type="directional_probability",
        objective_description=descriptor.objective_description,
        asset=descriptor.symbol,
        interval=descriptor.interval,
        horizon=descriptor.horizon,
        horizon_label=descriptor.horizon_label,
        candle_open_time=last_open.isoformat(),
        reference_time=last_ref.isoformat(),
        target_reference_time=target_ref.isoformat(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        reference_close=float(row["close"].iloc[0]),
        prediction=prediction_label,
        state_code=state_code,
        probability_up=final_prob,
        probability_down=float(1.0 - final_prob),
        raw_probability_up=raw_prob,
        primary_metric_label=f"P(subida): {final_prob:.2%}",
        threshold=descriptor.threshold,
        is_calibrated=descriptor.is_calibrated,
        calibration_summary=descriptor.calibration_summary,
        calibration_warning=descriptor.calibration_warning,
        selection_label_end=descriptor.selection_label_end,
        input_data_sha256=_input_digest(df),
        feature_schema_version="multimarket_price_v1",
        warnings=[descriptor.calibration_warning] if descriptor.calibration_warning else [],
    )


def _infer_deterministic_rule(
    descriptor: ModelDescriptor,
    df: pd.DataFrame,
    delta: pd.Timedelta,
) -> LiveInferenceResult:
    rule_kind = str(descriptor.extra_meta.get("rule_kind", ""))
    last_ref = pd.Timestamp(df["reference_time"].iloc[-1])

    if rule_kind == "smc_rules":
        feat_cfg = Features(enhanced=False, smc=True)
        featured = _cached_features(
            "tech_enh0_smc1",
            descriptor.symbol,
            last_ref.isoformat(),
            len(df),
            lambda: build_features(df, feat_cfg),
        )
        row = featured.iloc[[-1]]
        last_open = pd.Timestamp(row["timestamp"].iloc[0])
        target_ref = last_ref + descriptor.horizon * delta
        prob_arr, state_arr = rule_probabilities(row, threshold=descriptor.threshold)
        prob_val = float(prob_arr[0])
        state_code = str(state_arr[0])
        pred_label = map_state_to_spanish(state_code)
        return LiveInferenceResult(
            model_id=descriptor.model_id,
            display_name=descriptor.display_name,
            model_version=descriptor.version,
            model_type=descriptor.architecture,
            category="DETERMINISTIC_RULE",
            output_type="directional_probability",
            objective_description=descriptor.objective_description,
            asset=descriptor.symbol,
            interval=descriptor.interval,
            horizon=descriptor.horizon,
            horizon_label=descriptor.horizon_label,
            candle_open_time=last_open.isoformat(),
            reference_time=last_ref.isoformat(),
            target_reference_time=target_ref.isoformat(),
            generated_at_utc=datetime.now(UTC).isoformat(),
            reference_close=float(row["close"].iloc[0]),
            prediction=pred_label,
            state_code=state_code,
            probability_up=prob_val,
            probability_down=float(1.0 - prob_val),
            raw_probability_up=prob_val,
            primary_metric_label=f"Score heurístico SMC: {prob_val:.2%}",
            threshold=descriptor.threshold,
            is_calibrated=False,
            calibration_summary=descriptor.calibration_summary,
            calibration_warning=descriptor.calibration_warning,
            selection_label_end=descriptor.selection_label_end,
            input_data_sha256=_input_digest(df),
            feature_schema_version="smc_rules_v1",
            warnings=[descriptor.calibration_warning] if descriptor.calibration_warning else [],
        )

    # TSMOM o MACD multiescala
    floor = 1e-4
    feat_df = _cached_features(
        "dmn_causal_features",
        descriptor.symbol,
        last_ref.isoformat(),
        len(df),
        lambda: dmn_causal_features(df[["timestamp", "open", "high", "low", "close", "volume"]], floor),
    )
    last_row = feat_df.iloc[-1]
    last_open = pd.Timestamp(last_row["timestamp"])
    target_ref = last_ref + descriptor.horizon * delta
    vol_now = max(float(last_row["daily_volatility"]), floor)
    scale_factor = float(0.15 / (vol_now * np.sqrt(365.0)))

    if rule_kind == "tsmom_252":
        raw_sig = float(np.sign(float(last_row["return_252"])))
        signal_val = max(0.0, raw_sig)
    else:
        macd_vals = np.asarray(
            [float(last_row[f"macd_{short_w}_{long_w}"]) for short_w, long_w in SCALES], dtype=float
        )
        phi = np.clip((macd_vals * np.exp(-(macd_vals**2) / 4.0) / 0.89).mean(), -1.0, 1.0)
        signal_val = float(max(0.0, phi))

    weight_val = float(np.clip(signal_val * scale_factor, 0.0, 1.0))
    state_code = "bullish" if weight_val > 0.05 else "neutral"
    pred_label = (
        f"POSICIÓN LONG ({weight_val:+.1%})"
        if weight_val > 0.05
        else f"EN EFECTIVO / NEUTRAL ({weight_val:+.1%})"
    )

    return LiveInferenceResult(
        model_id=descriptor.model_id,
        display_name=descriptor.display_name,
        model_version=descriptor.version,
        model_type=descriptor.architecture,
        category="DETERMINISTIC_RULE",
        output_type="continuous_position",
        objective_description=descriptor.objective_description,
        asset=descriptor.symbol,
        interval=descriptor.interval,
        horizon=descriptor.horizon,
        horizon_label=descriptor.horizon_label,
        candle_open_time=last_open.isoformat(),
        reference_time=last_ref.isoformat(),
        target_reference_time=target_ref.isoformat(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        reference_close=float(last_row["close"]),
        prediction=pred_label,
        state_code=state_code,
        probability_up=None,
        probability_down=None,
        raw_probability_up=None,
        primary_metric_label=f"Peso objetivo: {weight_val:+.1%} (señal={signal_val:+.3f})",
        threshold=descriptor.threshold,
        is_calibrated=False,
        calibration_summary=descriptor.calibration_summary,
        calibration_warning=descriptor.calibration_warning,
        selection_label_end=descriptor.selection_label_end,
        input_data_sha256=_input_digest(df),
        feature_schema_version=f"rule_{rule_kind}_v1",
        signal_value=signal_val,
        target_weight=weight_val,
        warnings=[descriptor.calibration_warning] if descriptor.calibration_warning else [],
    )


def run_causal_inference(
    descriptor: ModelDescriptor,
    closed_candles: pd.DataFrame,
    *,
    peer_closed_candles: pd.DataFrame | None = None,
    reference_ts: pd.Timestamp | None = None,
    now_utc: pd.Timestamp | None = None,
    use_cache: bool = True,
) -> LiveInferenceResult:
    """Ejecuta inferencia real exclusivamente sobre velas cerradas respetando el pipeline original."""
    if not descriptor.is_deployable:
        raise DataError(
            f"El modelo {descriptor.version} no es desplegable: {descriptor.unavailable_reason}"
        )

    df = validate_strictly_closed_candles(closed_candles, descriptor.interval, now_utc=now_utc)
    df = _crop_to_reference(df, reference_ts)
    delta = pd.Timedelta(INTERVALS[descriptor.interval])
    last_ref_iso = pd.Timestamp(df["reference_time"].iloc[-1]).tz_convert("UTC").isoformat()

    cache_key = (descriptor.model_id, descriptor.symbol.upper(), descriptor.interval, last_ref_iso)
    if use_cache and now_utc is None:
        with _CACHE_LOCK:
            cached = _INFERENCE_RESULT_CACHE.get(cache_key)
            if cached is not None:
                return cached

    peer_cropped: pd.DataFrame | None = None
    if peer_closed_candles is not None and not peer_closed_candles.empty:
        peer_valid = validate_strictly_closed_candles(
            peer_closed_candles, descriptor.interval, now_utc=now_utc
        )
        peer_cropped = peer_valid.loc[
            peer_valid["reference_time"] <= pd.Timestamp(last_ref_iso)
        ].reset_index(drop=True)

    kind = descriptor.source_kind
    if kind == "registry":
        result = _infer_registry_model(descriptor, df, delta)
    elif kind == "frontier_frozen":
        result = _infer_frontier_model(descriptor, df, peer_cropped, delta)
    elif kind in {"retraining_b", "structure_cluster"}:
        result = _infer_retraining_or_cluster_model(descriptor, df, peer_cropped, delta)
    elif kind == "dmn_checkpoint":
        result = _infer_dmn_checkpoint(descriptor, df, delta)
    elif kind in {"momentum_frozen", "mr_frozen"}:
        result = _infer_hybrid_filter_model(descriptor, df, delta)
    elif kind == "multimarket":
        result = _infer_multimarket_model(descriptor, df, delta)
    elif kind == "deterministic_rule":
        result = _infer_deterministic_rule(descriptor, df, delta)
    else:
        raise DataError(f"Tipo de fuente de modelo desconocido: {kind}")

    if use_cache and now_utc is None:
        with _CACHE_LOCK:
            _INFERENCE_RESULT_CACHE[cache_key] = result
    return result


def run_batch_inference(
    descriptors: list[ModelDescriptor],
    closed_candles: pd.DataFrame,
    *,
    peer_closed_candles: pd.DataFrame | None = None,
    reference_ts: pd.Timestamp | None = None,
    now_utc: pd.Timestamp | None = None,
    use_cache: bool = True,
) -> list[ModelInferenceOutcome]:
    """Ejecuta múltiples modelos aislando errores para que un modelo fallido nunca detenga a los demás."""
    outcomes: list[ModelInferenceOutcome] = []
    last_ref_iso = ""
    if not closed_candles.empty and "reference_time" in closed_candles.columns:
        last_ref_iso = pd.Timestamp(closed_candles["reference_time"].iloc[-1]).tz_convert("UTC").isoformat()

    for desc in descriptors:
        cache_key = (desc.model_id, desc.symbol.upper(), desc.interval, last_ref_iso)
        was_cached = False
        if use_cache and now_utc is None and last_ref_iso:
            with _CACHE_LOCK:
                was_cached = cache_key in _INFERENCE_RESULT_CACHE
        try:
            res = run_causal_inference(
                desc,
                closed_candles,
                peer_closed_candles=peer_closed_candles,
                reference_ts=reference_ts,
                now_utc=now_utc,
                use_cache=use_cache,
            )
            outcomes.append(
                ModelInferenceOutcome(
                    descriptor=desc,
                    result=res,
                    error=None,
                    from_cache=was_cached,
                )
            )
        except Exception as exc:
            outcomes.append(
                ModelInferenceOutcome(
                    descriptor=desc,
                    result=None,
                    error=str(exc),
                    from_cache=False,
                )
            )
    return outcomes
