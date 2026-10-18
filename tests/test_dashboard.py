from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.config.settings import Features
from trading_ml.dashboard import (
    LiveStreamWorker,
    assess_data_quality,
    clear_inference_caches,
    compute_common_window_comparison,
    compute_live_agreement_summary,
    compute_model_prospective_metrics,
    discover_all_models,
    discover_deterministic_strategies,
    discover_dmn_models,
    discover_frontier_models,
    discover_hybrid_filter_models,
    discover_multimarket_models,
    discover_non_deployable_models,
    discover_registry_models,
    discover_retraining_and_regime_models,
    evaluate_pending_predictions,
    get_models_for_selection,
    load_prospective_predictions,
    parse_rest_klines,
    record_prospective_prediction,
    run_batch_inference,
    run_causal_inference,
    validate_strictly_closed_candles,
    verify_model_integrity,
)
from trading_ml.errors import DataError, LeakageError
from trading_ml.features.technical import build_features


def _make_raw_klines(
    start: str,
    count: int,
    interval_hours: int = 24,
    base_price: float = 60000.0,
    seed: int = 42,
) -> list[list]:
    base_ts = pd.Timestamp(start, tz="UTC")
    step = pd.Timedelta(hours=interval_hours)
    rows = []
    rng = np.random.default_rng(seed)
    price = base_price
    for i in range(count):
        open_ts = base_ts + i * step
        close_ts = open_ts + step - pd.Timedelta(milliseconds=1)
        ret = float(rng.normal(0.001, 0.015))
        open_p = price
        close_p = max(0.2, open_p * (1.0 + ret))
        high_p = max(open_p, close_p) * 1.008
        low_p = min(open_p, close_p) * 0.992
        vol = float(1000.0 + rng.uniform(0, 500))
        q_vol = vol * close_p
        rows.append(
            [
                int(open_ts.timestamp() * 1000),
                f"{open_p:.4f}",
                f"{high_p:.4f}",
                f"{low_p:.4f}",
                f"{close_p:.4f}",
                f"{vol:.4f}",
                int(close_ts.timestamp() * 1000),
                f"{q_vol:.2f}",
                15000,
                f"{vol * 0.52:.4f}",
                f"{q_vol * 0.52:.2f}",
                "0",
            ]
        )
        price = close_p
    return rows


def test_discover_all_models_catalog_completeness() -> None:
    """1. Descubrimiento exhaustivo de modelos ML, reglas deterministas y estudios no desplegables."""
    root = Path("artifacts")
    catalog = discover_all_models(root, include_non_deployable=True)
    assert len(catalog) >= 100

    deployable = [m for m in catalog if m.is_deployable]
    non_deployable = [m for m in catalog if not m.is_deployable]
    deterministic = [m for m in deployable if m.category == "DETERMINISTIC_RULE"]
    ml_models = [m for m in deployable if m.category == "ML_MODEL"]

    assert len(deployable) >= 95
    assert len(ml_models) >= 89
    assert len(deterministic) == 6
    assert len(non_deployable) >= 5

    families = {m.source_kind for m in catalog}
    expected_families = {
        "registry",
        "frontier_frozen",
        "retraining_b",
        "structure_cluster",
        "dmn_checkpoint",
        "momentum_frozen",
        "mr_frozen",
        "multimarket",
        "deterministic_rule",
        "non_deployable",
    }
    assert expected_families.issubset(families)

    # Verificar que todos tienen nombre descriptivo claro y versión original intacta
    for item in catalog:
        assert item.display_name.strip() != ""
        assert item.version.strip() != ""
        assert item.objective_description.strip() != ""
        if not item.is_deployable:
            assert item.unavailable_reason is not None and len(item.unavailable_reason) > 15


def test_verify_model_integrity_passes_and_detects_tampering() -> None:
    """2. Verificación criptográfica SHA-256 y detección de artefactos faltantes."""
    root = Path("artifacts")
    reg_models = discover_registry_models(root)
    dmn_models = discover_dmn_models(root)
    assert len(reg_models) >= 8
    assert len(dmn_models) >= 30

    # Integridad válida sobre modelos reales
    verify_model_integrity(reg_models[0])
    verify_model_integrity(dmn_models[0])

    # Si falta el archivo físico del checkpoint, debe lanzar DataError
    broken_dmn = replace(dmn_models[0], artifact_path=root / "nonexistent_checkpoint.pt")
    with pytest.raises(DataError, match="Checksum inválido o archivo faltante"):
        verify_model_integrity(broken_dmn)


def test_parse_rest_klines_separates_closed_and_forming_candle() -> None:
    klines = _make_raw_klines("2026-10-01T00:00:00Z", count=5, interval_hours=24)
    now_utc = pd.Timestamp("2026-10-05T12:00:00Z")
    closed, forming = parse_rest_klines(klines, interval="1d", now_utc=now_utc)

    assert len(closed) == 4
    assert closed["is_closed"].all()
    assert closed.iloc[-1]["timestamp"] == pd.Timestamp("2026-10-04T00:00:00Z")
    assert closed.iloc[-1]["reference_time"] == pd.Timestamp("2026-10-05T00:00:00Z")

    assert forming is not None
    assert forming["is_closed"] is False
    assert forming["timestamp"] == "2026-10-05T00:00:00+00:00"
    assert forming["expected_close_time"] == "2026-10-06T00:00:00+00:00"


def test_websocket_forming_vs_closed_candle_and_data_quality(tmp_path: Path) -> None:
    """11. Reconexión, deduplicación WebSocket y reporte de calidad de datos."""
    worker = LiveStreamWorker(root=tmp_path, symbol="BTCUSDT", interval="1d")
    klines = _make_raw_klines("2026-10-01T00:00:00Z", count=3, interval_hours=24)
    closed, _ = parse_rest_klines(
        klines, interval="1d", now_utc=pd.Timestamp("2026-10-04T00:00:00Z")
    )
    worker._closed_candles = closed

    open_ms = int(pd.Timestamp("2026-10-04T00:00:00Z").timestamp() * 1000)
    forming_msg = {
        "e": "kline",
        "k": {
            "t": open_ms,
            "o": "61000.0",
            "h": "62000.0",
            "l": "60500.0",
            "c": "61800.0",
            "v": "1200.0",
            "q": "74160000.0",
            "n": 1000,
            "V": "600.0",
            "Q": "37080000.0",
            "x": False,
        },
    }
    promoted = worker.handle_ws_kline_message(
        forming_msg, now_utc=pd.Timestamp("2026-10-04T15:00:00Z")
    )
    assert promoted is False
    assert len(worker._closed_candles) == 3
    assert worker._forming_candle is not None
    assert worker._live_price == 61800.0

    closed_msg = {
        "e": "kline",
        "k": {**forming_msg["k"], "c": "61950.0", "x": True},
    }
    promoted_first = worker.handle_ws_kline_message(
        closed_msg, now_utc=pd.Timestamp("2026-10-05T00:00:01Z")
    )
    assert promoted_first is True
    assert len(worker._closed_candles) == 4
    assert worker._forming_candle is None

    # Evento duplicado del mismo cierre: incrementa duplicates_removed sin duplicar fila
    promoted_dup = worker.handle_ws_kline_message(
        closed_msg, now_utc=pd.Timestamp("2026-10-05T00:00:05Z")
    )
    assert promoted_dup is False
    assert len(worker._closed_candles) == 4
    assert worker._duplicates_removed == 1

    # Verificar assess_data_quality con huecos (missing bars)
    q_ok = assess_data_quality(
        worker._closed_candles,
        "BTCUSDT",
        "1d",
        duplicates_removed=worker._duplicates_removed,
        now_utc=pd.Timestamp("2026-10-05T02:00:00Z"),
    )
    assert q_ok.missing_bars_count == 0
    assert q_ok.duplicates_removed == 1
    assert q_ok.clock_sync_ok is True

    gapped = worker._closed_candles.drop(index=[1]).reset_index(drop=True)
    q_gap = assess_data_quality(
        gapped,
        "BTCUSDT",
        "1d",
        now_utc=pd.Timestamp("2026-10-05T02:00:00Z"),
    )
    assert q_gap.missing_bars_count == 1
    assert any("hueco" in w.lower() for w in q_gap.notes)


def test_validate_strictly_closed_candles_rejects_open_candle() -> None:
    """6. Inferencia estricta sobre velas cerradas."""
    klines = _make_raw_klines("2026-10-01T00:00:00Z", count=5, interval_hours=24)
    closed, _ = parse_rest_klines(
        klines, interval="1d", now_utc=pd.Timestamp("2026-10-06T00:00:00Z")
    )
    with pytest.raises(LeakageError):
        validate_strictly_closed_candles(
            closed, interval="1d", now_utc=pd.Timestamp("2026-10-05T12:00:00Z")
        )

    tampered = closed.copy()
    tampered.loc[tampered.index[-1], "is_closed"] = False
    with pytest.raises(LeakageError):
        validate_strictly_closed_candles(
            tampered, interval="1d", now_utc=pd.Timestamp("2026-10-06T00:00:00Z")
        )


def test_all_model_families_load_and_infer_independently() -> None:
    """3, 4 y 5. Carga de modelos/checkpoints, compatibilidad de features y predicciones independientes."""
    clear_inference_caches()
    root = Path("artifacts")
    now_utc = pd.Timestamp("2026-10-18T12:00:00Z")

    # 820 velas cerradas hasta 2026-10-18T00:00:00Z para cubrir cualquier warmup (incl. L250 = 750 velas)
    btc_klines = _make_raw_klines("2024-07-20T00:00:00Z", count=820, interval_hours=24, base_price=65000.0, seed=11)
    eth_klines = _make_raw_klines("2024-07-20T00:00:00Z", count=820, interval_hours=24, base_price=2600.0, seed=22)
    eur_klines = _make_raw_klines("2024-07-20T00:00:00Z", count=820, interval_hours=24, base_price=1.09, seed=33)

    btc_closed, _ = parse_rest_klines(btc_klines, interval="1d", now_utc=now_utc)
    eth_closed, _ = parse_rest_klines(eth_klines, interval="1d", now_utc=now_utc)
    eur_closed, _ = parse_rest_klines(eur_klines, interval="1d", now_utc=now_utc)

    # 1) Registry XGBoost (calibrado y no calibrado)
    reg_models = discover_registry_models(root)
    base_reg = next(m for m in reg_models if m.version == "v20261012T143622_ff1c464c")
    smc_reg = next(m for m in reg_models if m.version == "v20261012T143623_0d9b01f0")

    expected_features = build_features(btc_closed, Features(**base_reg.bundle.metadata["contract"]["features"]))
    res_base = run_causal_inference(base_reg, btc_closed, now_utc=now_utc)
    res_smc = run_causal_inference(smc_reg, btc_closed, now_utc=now_utc)
    assert np.isclose(res_base.reference_close, float(expected_features.iloc[-1]["close"]))
    assert res_base.is_calibrated is False
    assert res_smc.is_calibrated is True
    assert not np.isclose(res_smc.probability_up, res_base.probability_up)

    # 2) Frontier Frozen (direction_h1, net_h3, barrier_close_h5, stacking)
    frontier_models = discover_frontier_models(root)
    eth_h1 = next(m for m in frontier_models if m.symbol == "ETHUSDT" and m.horizon == 1 and "random_forest" in m.version)
    btc_h3 = next(m for m in frontier_models if m.symbol == "BTCUSDT" and m.horizon == 3)
    btc_h5 = next(m for m in frontier_models if m.symbol == "BTCUSDT" and m.horizon == 5)
    stack_m = next(m for m in frontier_models if "stacking" in m.version)

    res_eth_h1 = run_causal_inference(eth_h1, eth_closed, peer_closed_candles=btc_closed, now_utc=now_utc)
    res_btc_h3 = run_causal_inference(btc_h3, btc_closed, peer_closed_candles=eth_closed, now_utc=now_utc)
    res_btc_h5 = run_causal_inference(btc_h5, btc_closed, peer_closed_candles=eth_closed, now_utc=now_utc)
    res_stack = run_causal_inference(stack_m, btc_closed, peer_closed_candles=eth_closed, now_utc=now_utc)

    assert res_eth_h1.horizon == 1 and 0.0 <= float(res_eth_h1.probability_up) <= 1.0
    assert res_btc_h3.horizon == 3 and res_btc_h3.output_type == "net_return_probability"
    assert res_btc_h5.horizon == 5 and res_btc_h5.output_type == "barrier_probability"
    assert 0.0 <= float(res_stack.probability_up) <= 1.0

    # 3) Retraining Phase B + Structure K-Means Cluster
    ret_models = discover_retraining_and_regime_models(root)
    spec_m = next(m for m in ret_models if "specialists_rules" in m.version and m.symbol == "BTCUSDT")
    clus_m = next(m for m in ret_models if m.source_kind == "structure_cluster" and m.symbol == "BTCUSDT")
    res_spec = run_causal_inference(spec_m, btc_closed, peer_closed_candles=eth_closed, now_utc=now_utc)
    res_clus = run_causal_inference(clus_m, btc_closed, peer_closed_candles=eth_closed, now_utc=now_utc)
    assert 0.0 <= float(res_spec.probability_up) <= 1.0
    assert 0.0 <= float(res_clus.probability_up) <= 1.0

    # 4) Deep Momentum Networks (LSTM / MLP / Linear con sizing continuo)
    dmn_models = discover_dmn_models(root)
    lstm_lo = next(
        m
        for m in dmn_models
        if "lstm_sharpe_net" in m.version
        and m.symbol == "BTCUSDT"
        and m.extra_meta.get("position_mode") == "long_only"
    )
    lstm_ls = next(
        m
        for m in dmn_models
        if "lstm_sharpe_net" in m.version
        and m.symbol == "BTCUSDT"
        and m.extra_meta.get("position_mode") == "long_short"
    )
    res_dmn_lo = run_causal_inference(lstm_lo, btc_closed, now_utc=now_utc)
    res_dmn_ls = run_causal_inference(lstm_ls, btc_closed, now_utc=now_utc)
    assert res_dmn_lo.output_type == "continuous_position"
    assert res_dmn_lo.target_weight is not None and 0.0 <= res_dmn_lo.target_weight <= 1.0
    assert res_dmn_ls.target_weight is not None and -1.0 <= res_dmn_ls.target_weight <= 1.0

    # 5) Hybrid Filters (Momentum L250 + Mean Reversion)
    hyb_models = discover_hybrid_filter_models(root)
    mom_250 = next(m for m in hyb_models if "L250" in m.version)
    mr_side = next(m for m in hyb_models if "mr_L20_sideways_xgb" in m.version)
    res_mom = run_causal_inference(mom_250, btc_closed, now_utc=now_utc)
    res_mr = run_causal_inference(mr_side, btc_closed, now_utc=now_utc)
    assert res_mom.output_type == "opportunity_filter_probability"
    assert res_mr.output_type == "opportunity_filter_probability"
    assert 0.0 <= float(res_mom.probability_up) <= 1.0

    # 6) Multimarket Forex (EURUSD)
    mm_models = discover_multimarket_models(root)
    eur_m = next(m for m in mm_models if m.symbol == "EURUSD")
    res_eur = run_causal_inference(eur_m, eur_closed, now_utc=now_utc)
    assert res_eur.asset == "EURUSD"
    assert 0.0 <= float(res_eur.probability_up) <= 1.0

    # 7) Reglas deterministas (SMC, TSMOM, MACD)
    det_models = discover_deterministic_strategies(root)
    for dm in det_models[:3]:
        res_det = run_causal_inference(dm, btc_closed, now_utc=now_utc)
        assert res_det.category == "DETERMINISTIC_RULE"


def test_selection_cutoff_guardrail_blocks_in_sample_inference() -> None:
    root = Path("artifacts")
    base_model = next(
        m for m in discover_registry_models(root) if m.version == "v20261012T143622_ff1c464c"
    )
    klines = _make_raw_klines("2026-01-01T00:00:00Z", count=260, interval_hours=24)
    now_utc = pd.Timestamp("2026-09-25T00:00:00Z")
    closed, _ = parse_rest_klines(klines, interval="1d", now_utc=now_utc)
    with pytest.raises(LeakageError):
        run_causal_inference(base_model, closed, now_utc=now_utc)


def test_batch_inference_error_isolation() -> None:
    """7. Aislamiento de errores: si un modelo falla, los demás continúan sin interrupción."""
    root = Path("artifacts")
    reg_models = discover_registry_models(root)[:2]
    non_dep = discover_non_deployable_models(root)[0]
    broken_model = replace(reg_models[0], model_id="broken_test_id", bundle=None)

    klines = _make_raw_klines("2025-10-01T00:00:00Z", count=380, interval_hours=24)
    now_utc = pd.Timestamp("2026-10-18T12:00:00Z")
    closed, _ = parse_rest_klines(klines, interval="1d", now_utc=now_utc)

    outcomes = run_batch_inference(
        [reg_models[0], non_dep, broken_model, reg_models[1]],
        closed,
        now_utc=now_utc,
    )
    assert len(outcomes) == 4
    assert outcomes[0].succeeded is True
    assert outcomes[1].succeeded is False and outcomes[1].error is not None
    assert outcomes[2].succeeded is False and outcomes[2].error is not None
    assert outcomes[3].succeeded is True


def test_prospective_log_persistence_metrics_and_scientific_comparison(tmp_path: Path) -> None:
    """8, 9 y 10. Registro prospectivo, persistencia tras reinicios, métricas y ventana común."""
    root = Path("artifacts")
    reg_models = discover_registry_models(root)
    model_a = next(m for m in reg_models if m.version == "v20261012T143622_ff1c464c")
    model_b = next(m for m in reg_models if m.version == "v20261012T143623_0d9b01f0")

    klines = _make_raw_klines("2026-01-01T00:00:00Z", count=292, interval_hours=24)
    log_file = tmp_path / "prospective_predictions.jsonl"

    # Día 1: predicción al cierre del 2026-10-16T00:00:00Z (vela del 15 de oct)
    now_day1 = pd.Timestamp("2026-10-16T12:00:00Z")
    closed_day1, _ = parse_rest_klines(klines[:-3], interval="1d", now_utc=now_day1)

    res_a1 = run_causal_inference(model_a, closed_day1, now_utc=now_day1)
    res_b1 = run_causal_inference(model_b, closed_day1, now_utc=now_day1)

    # Idempotencia al registrar dos veces
    e1 = record_prospective_prediction(res_a1, log_path=log_file)
    e2 = record_prospective_prediction(res_a1, log_path=log_file)
    record_prospective_prediction(res_b1, log_path=log_file)
    assert e1["prediction_id"] == e2["prediction_id"]
    assert e1["input_data_sha256"] != ""

    # Día 2: predicción al cierre del 2026-10-17T00:00:00Z + evaluación automática del Día 1
    now_day2 = pd.Timestamp("2026-10-17T12:00:00Z")
    closed_day2, _ = parse_rest_klines(klines[:-2], interval="1d", now_utc=now_day2)
    evaluate_pending_predictions(closed_day2, asset="BTCUSDT", interval="1d", log_path=log_file)

    res_a2 = run_causal_inference(model_a, closed_day2, now_utc=now_day2)
    res_b2 = run_causal_inference(model_b, closed_day2, now_utc=now_day2)
    record_prospective_prediction(res_a2, log_path=log_file)
    record_prospective_prediction(res_b2, log_path=log_file)

    # Día 3: resolución del Día 2
    now_day3 = pd.Timestamp("2026-10-18T12:00:00Z")
    closed_day3, _ = parse_rest_klines(klines[:-1], interval="1d", now_utc=now_day3)
    evaluate_pending_predictions(closed_day3, asset="BTCUSDT", interval="1d", log_path=log_file)

    # Simular reinicio recargando desde disco
    reloaded = load_prospective_predictions(log_path=log_file)
    assert len(reloaded) == 4
    assert all(r["status"] == "EVALUADA" for r in reloaded)

    # Calcular métricas prospectivas por modelo
    metrics_list = compute_model_prospective_metrics(reloaded)
    assert len(metrics_list) == 2
    for row in metrics_list:
        assert row["evaluated_predictions"] == 2
        assert row["brier_score"] is not None
        assert row["log_loss"] is not None
        assert row["calibration_gap"] is not None
        assert 0.0 <= row["signal_coverage"] <= 1.0

    # Comparación científica sobre ventana temporal común
    common_rows = compute_common_window_comparison(reloaded)
    assert len(common_rows) == 2
    assert all(r["common_n"] == 2 for r in common_rows)

    # Resumen de acuerdo/desacuerdo en vivo
    agreement = compute_live_agreement_summary([res_a2, res_b2])
    assert len(agreement) == 1
    assert agreement[0]["Modelos Comparados"] == 2


def test_unsupported_timeframe_returns_explicit_unavailable_state() -> None:
    root = Path("artifacts")
    models, reason = get_models_for_selection(root, symbol="BTCUSDT", interval="15m")
    assert models == []
    assert reason is not None
    assert "No existe ningún modelo entrenado" in reason
