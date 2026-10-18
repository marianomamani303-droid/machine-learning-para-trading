from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_ml.dashboard.edge_candidates import research_candidates
from trading_ml.dashboard.inference_engine import (
    LiveInferenceResult,
    ModelInferenceOutcome,
    run_batch_inference,
)
from trading_ml.dashboard.live_feed import MarketSnapshot, get_feed_manager
from trading_ml.dashboard.models_catalog import (
    ModelDescriptor,
    discover_all_models,
    get_models_for_selection,
)
from trading_ml.dashboard.prospective_log import (
    compute_common_window_comparison,
    compute_live_agreement_summary,
    compute_model_prospective_metrics,
    default_log_path,
    evaluate_pending_predictions,
    load_prospective_predictions,
    record_prospective_prediction,
    seed_recent_closed_predictions,
)
from trading_ml.errors import DataError

SUPPORTED_ASSETS = [
    "BTCUSDT",
    "ETHUSDT",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
]


def _resolve_root() -> Path:
    env_root = os.environ.get("TRADING_ML_ROOT")
    if env_root:
        return Path(env_root)
    return Path("artifacts")


def _build_candlestick_figure(
    snapshot: MarketSnapshot,
    reference_close: float | None = None,
    bars_to_show: int = 75,
) -> go.Figure:
    closed = snapshot.closed_candles.tail(bars_to_show).copy()
    fig = go.Figure()

    if not closed.empty:
        fig.add_trace(
            go.Candlestick(
                x=closed["timestamp"],
                open=closed["open"],
                high=closed["high"],
                low=closed["low"],
                close=closed["close"],
                name="Velas cerradas (aptas para inferencia)",
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
            )
        )

    if snapshot.forming_candle is not None:
        fc = snapshot.forming_candle
        open_ts = pd.Timestamp(fc["timestamp"])
        fig.add_trace(
            go.Candlestick(
                x=[open_ts],
                open=[fc["open"]],
                high=[fc["high"]],
                low=[fc["low"]],
                close=[fc["close"]],
                name="Vela en formación (NO usada en predicción)",
                increasing_line_color="#f59e0b",
                decreasing_line_color="#f59e0b",
                opacity=0.45,
            )
        )

    if reference_close is not None:
        fig.add_hline(
            y=reference_close,
            line_dash="dot",
            line_color="#64748b",
            annotation_text=f"Cierre vela base ({reference_close:,.4f})",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"{snapshot.symbol} · Temporalidad {snapshot.interval} (UTC)",
        xaxis_title="Fecha apertura de vela (UTC)",
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        height=400,
        margin={"l": 16, "r": 16, "t": 40, "b": 16},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def _build_probability_comparison_chart(results: list[LiveInferenceResult]) -> go.Figure | None:
    prob_items = [r for r in results if r.probability_up is not None]
    if not prob_items:
        return None
    prob_items = sorted(prob_items, key=lambda r: float(r.probability_up or 0.5))
    labels = [f"{r.display_name[:48]} ({r.horizon_label[:6]})" for r in prob_items]
    values = [float(r.probability_up or 0.5) for r in prob_items]
    colors = [
        "#16a34a"
        if r.state_code == "bullish"
        else ("#dc2626" if r.state_code == "bearish" else "#64748b")
        for r in prob_items
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2%} ({r.prediction})" for v, r in zip(values, prob_items, strict=False)],
            textposition="outside",
        )
    )
    fig.add_vline(x=0.50, line_dash="dash", line_color="#94a3b8", annotation_text="0.50")
    fig.add_vline(
        x=0.55, line_dash="dot", line_color="#16a34a", annotation_text="Umbral Long (0.55)"
    )
    fig.add_vline(
        x=0.45, line_dash="dot", line_color="#dc2626", annotation_text="Umbral Short (0.45)"
    )
    fig.update_layout(
        title="Comparación de Probabilidades Estimadas P(evento positivo) sobre la Vela Cerrada Actual",
        xaxis_title="Probabilidad estimada [0, 1]",
        xaxis={"range": [0.0, 1.08]},
        height=max(280, 34 * len(prob_items) + 90),
        margin={"l": 16, "r": 40, "t": 44, "b": 16},
    )
    return fig


def _build_position_comparison_chart(results: list[LiveInferenceResult]) -> go.Figure | None:
    pos_items = [
        r for r in results if r.output_type == "continuous_position" and r.target_weight is not None
    ]
    if not pos_items:
        return None
    pos_items = sorted(pos_items, key=lambda r: float(r.target_weight or 0.0))
    labels = [r.display_name[:52] for r in pos_items]
    weights = [float(r.target_weight or 0.0) for r in pos_items]
    colors = ["#16a34a" if w > 0.05 else ("#dc2626" if w < -0.05 else "#64748b") for w in weights]

    fig = go.Figure(
        go.Bar(
            x=weights,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[
                f"{w:+.1%} (señal={float(r.signal_value or 0.0):+.2f})"
                for w, r in zip(weights, pos_items, strict=False)
            ],
            textposition="outside",
        )
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        title="Exposición / Peso Objetivo (Modelos de Gestión Dinámica de Posición DMN y Baselines)",
        xaxis_title="Peso objetivo escalado por volatilidad [-1, +1]",
        xaxis={"range": [-1.15, 1.15]},
        height=max(260, 32 * len(pos_items) + 90),
        margin={"l": 16, "r": 40, "t": 44, "b": 16},
    )
    return fig


def _catalog_dataframe(models: list[ModelDescriptor]) -> pd.DataFrame:
    rows = []
    for m in models:
        rows.append(
            {
                "Nombre Descriptivo": m.display_name,
                "Categoría": "Modelo ML" if m.category == "ML_MODEL" else "Regla Determinista",
                "Arquitectura": m.architecture,
                "Versión / ID Original": m.version,
                "Activos": ", ".join(m.compatible_assets),
                "Temp.": m.interval,
                "Horizonte": m.horizon_label,
                "Objetivo": m.objective_description,
                "Features": f"{m.feature_count} ({m.feature_family})",
                "Entrenamiento": m.training_period,
                "Calibración": "Calibrada" if m.is_calibrated else m.calibration_summary[:38],
                "Estado": m.availability_status,
                "Limitaciones / Diagnóstico": m.unavailable_reason or m.known_limitations,
            }
        )
    return pd.DataFrame(rows)


def _comparison_dataframe(outcomes: list[ModelInferenceOutcome]) -> pd.DataFrame:
    rows = []
    for out in outcomes:
        d = out.descriptor
        if out.result is not None:
            r = out.result
            p_str = (
                f"{r.probability_up:.2%}"
                if r.probability_up is not None
                else "N/A (Posición continua)"
            )
            raw_str = f"{r.raw_probability_up:.2%}" if r.raw_probability_up is not None else "-"
            rows.append(
                {
                    "Modelo": r.display_name,
                    "Versión": r.model_version,
                    "Tipo": "ML" if r.category == "ML_MODEL" else "Regla",
                    "Arquitectura": d.architecture,
                    "Predicción": r.prediction,
                    "P(subida)": p_str,
                    "Salida Principal": r.primary_metric_label,
                    "Score Crudo": raw_str,
                    "Calibrada": "Sí" if r.is_calibrated else "No",
                    "Horizonte": r.horizon_label,
                    "Cierre Base (UTC)": r.reference_time[:19].replace("T", " "),
                    "Cierre Objetivo (UTC)": r.target_reference_time[:19].replace("T", " "),
                    "Estado": "Disponible (Cache)" if out.from_cache else "Disponible",
                    "_prob_sort": float(r.probability_up) if r.probability_up is not None else -1.0,
                }
            )
        else:
            rows.append(
                {
                    "Modelo": d.display_name,
                    "Versión": d.version,
                    "Tipo": "ML" if d.category == "ML_MODEL" else "Regla",
                    "Arquitectura": d.architecture,
                    "Predicción": "ERROR / EN ESPERA",
                    "P(subida)": "-",
                    "Salida Principal": f"Error aislado: {out.error}",
                    "Score Crudo": "-",
                    "Calibrada": "Sí" if d.is_calibrated else "No",
                    "Horizonte": d.horizon_label,
                    "Cierre Base (UTC)": "-",
                    "Cierre Objetivo (UTC)": "-",
                    "Estado": f"Error ({out.error[:40]})" if out.error else "No disponible",
                    "_prob_sort": -2.0,
                }
            )
    return pd.DataFrame(rows)


def _metrics_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    rows = []
    for m in metrics_list:
        sig_acc = m.get("signal_accuracy")
        ci_l, ci_h = m.get("accuracy_ci_low"), m.get("accuracy_ci_high")
        ci_str = (
            f"[{ci_l:.1%}, {ci_h:.1%}]"
            if ci_l is not None and ci_h is not None
            else "Sin muestra activa"
        )
        prec = m.get("precision")
        rec = m.get("recall")
        brier = m.get("brier_score")
        ll = m.get("log_loss")
        c_gap = m.get("calibration_gap")
        net_r = m.get("hypothetical_net_return", 0.0)

        rows.append(
            {
                "Modelo": m["display_name"],
                "Versión": m["model_version"],
                "Tipo": "ML" if m["category"] == "ML_MODEL" else "Regla",
                "Horizonte": f"{m['horizon']}d ({m['output_type']})",
                "Predicciones (Total / Eval. / Pend.)": f"{m['total_predictions']} / {m['evaluated_predictions']} / {m['pending_predictions']}",
                "Cobertura Señal": f"{m['signal_coverage']:.1%} ({m['active_signals']}/{m['evaluated_predictions']})",
                "Accuracy Señales": f"{sig_acc:.2%}" if sig_acc is not None else "N/A",
                "IC 95% (Wilson)": ci_str,
                "Precision / Recall": (
                    f"{prec:.2%} / {rec:.2%}" if prec is not None and rec is not None else "N/A"
                ),
                "Matriz Confusión (TP/FP/TN/FN)": f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}",
                "Brier Score": f"{brier:.4f}" if brier is not None else "N/A",
                "Log-Loss": f"{ll:.4f}" if ll is not None else "N/A",
                "Gap Calibración": f"{c_gap:.4f}" if c_gap is not None else "N/A",
                "Retorno Hipotético Neto (14 bps)": f"{net_r:+.2%}",
                "Diagnóstico Estadístico": m["evidence_note"],
            }
        )
    return pd.DataFrame(rows)


def _format_predictions_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        p_up = rec.get("probability_up")
        ref_close = rec.get("reference_close")
        tgt_close = rec.get("target_close")
        ret = rec.get("realized_return")
        net_ret = rec.get("hypothetical_net_return")

        rows.append(
            {
                "Cierre Vela Base (UTC)": str(rec.get("reference_time", ""))[:19].replace("T", " "),
                "Cierre Objetivo (UTC)": str(rec.get("target_reference_time", ""))[:19].replace(
                    "T", " "
                ),
                "Modelo": str(rec.get("display_name") or rec.get("model_version", "")),
                "Versión": str(rec.get("model_version", "")),
                "Horizonte": str(rec.get("horizon_label") or f"{rec.get('horizon', 1)}d"),
                "Predicción / Señal": str(rec.get("prediction", "")),
                "Salida / Probabilidad": (
                    str(rec.get("primary_metric_label"))
                    if rec.get("primary_metric_label")
                    else (
                        f"P(subida): {float(p_up):.2%}" if isinstance(p_up, (int, float)) else "-"
                    )
                ),
                "Calibrada": "Sí" if rec.get("is_calibrated") else "No",
                "Precio Base": f"{ref_close:,.4f}" if isinstance(ref_close, (int, float)) else "-",
                "Precio Objetivo": (
                    f"{tgt_close:,.4f}" if isinstance(tgt_close, (int, float)) else "Pendiente"
                ),
                "Retorno Mercado": f"{ret:+.2%}" if isinstance(ret, (int, float)) else "Pendiente",
                "Retorno Neto Hipotético": (
                    f"{net_ret:+.2%}" if isinstance(net_ret, (int, float)) else "Pendiente"
                ),
                "Estado": str(rec.get("status", "PENDIENTE")),
                "Resultado": str(rec.get("outcome_label", "PENDIENTE")),
            }
        )
    return pd.DataFrame(rows)


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Laboratorio de Modelos IA en Tiempo Real",
        layout="wide",
    )

    root = _resolve_root()
    all_catalog = discover_all_models(root, include_non_deployable=True)
    deployable_catalog = [m for m in all_catalog if m.is_deployable]
    non_deployable_catalog = [m for m in all_catalog if not m.is_deployable]

    st.title("Laboratorio de Investigación y Comparación de Modelos en Tiempo Real")
    st.caption(
        "Evaluación prospectiva sobre datos nuevos del mercado (exclusivamente velas cerradas). "
        "Sin ejecución de órdenes reales ni alteración de modelos entrenados."
    )

    with st.expander("Investigación de nuevas señales y temporalidades"):
        try:
            research_rows = research_candidates(root)
            if research_rows:
                st.dataframe(pd.DataFrame(research_rows), hide_index=True, use_container_width=True)
                st.caption(
                    "Resultados históricos exploratorios. La rentabilidad sigue sin confirmar; "
                    "el seguimiento requiere un período futuro registrado y separado de las reservas existentes."
                )
            else:
                st.info("Todavía no hay una investigación de nuevas señales terminada.")
        except (OSError, ValueError, KeyError, DataError) as exc:
            st.warning(f"No se pudo verificar la evidencia de investigación: {exc}")

    with st.sidebar:
        st.header("Mercado y Selección de Modelos")
        symbol = st.selectbox(
            "Activo",
            options=SUPPORTED_ASSETS,
            index=0,
            help="BTCUSDT y ETHUSDT operan con WebSocket + REST en tiempo real de Binance; pares Forex operan sobre el repositorio diario verificado.",
        )
        interval = st.selectbox(
            "Temporalidad",
            options=["1d", "4h", "1h", "15m"],
            index=0,
            help="Todos los modelos entrenados y auditados en el repositorio tienen contrato diario (1d).",
        )
        include_rules = st.checkbox(
            "Incluir Estrategias Deterministas (Reglas baseline)",
            value=True,
            help="Incluye reglas SMC, TSMOM 252d y MACD multiescala claramente diferenciadas de los modelos de ML.",
        )

        compatible_models, no_model_reason = get_models_for_selection(
            root, symbol, interval, include_rules=include_rules
        )

        selected_descriptors: list[ModelDescriptor] = []
        if compatible_models:
            arch_groups = sorted({m.source_kind for m in compatible_models})
            selection_mode = st.radio(
                "Modo de selección",
                options=[
                    "Todos los modelos compatibles",
                    "Selección personalizada / individual",
                    "Filtrar por familia / arquitectura",
                ],
                index=0,
            )

            if selection_mode == "Todos los modelos compatibles":
                selected_descriptors = compatible_models
            elif selection_mode == "Filtrar por familia / arquitectura":
                chosen_kinds = st.multiselect(
                    "Familias de modelos",
                    options=arch_groups,
                    default=arch_groups,
                )
                selected_descriptors = [
                    m for m in compatible_models if m.source_kind in chosen_kinds
                ]
            else:
                label_map = {f"{m.display_name} [{m.version}]": m for m in compatible_models}
                default_keys = list(label_map.keys())[: min(6, len(label_map))]
                chosen_labels = st.multiselect(
                    "Modelos a evaluar simultáneamente",
                    options=list(label_map.keys()),
                    default=default_keys,
                )
                selected_descriptors = [label_map[k] for k in chosen_labels]
        else:
            st.warning("Sin modelos compatibles para esta temporalidad/activo.")

        st.divider()
        seed_history_bars = st.slider(
            "Velas cerradas recientes a evaluar prospectivamente",
            min_value=1,
            max_value=15,
            value=5,
            help="Evalúa causalmente las últimas N velas cerradas posteriores al entrenamiento para poblar el seguimiento prospectivo.",
        )
        auto_refresh = st.checkbox("Actualización automática en vivo", value=True)
        refresh_seconds = st.slider(
            "Intervalo de refresco (segundos)",
            min_value=3,
            max_value=60,
            value=8,
            disabled=not auto_refresh,
        )
        force_sync = st.button("Sincronizar datos ahora (REST + WS)")

    run_every = f"{refresh_seconds}s" if auto_refresh else None

    @st.fragment(run_every=run_every)
    def _live_laboratory() -> None:
        manager = get_feed_manager()
        snapshot = manager.get_snapshot(
            root=root,
            symbol=symbol,
            interval=interval,
            force_rest=force_sync,
            ensure_ws=True,
        )

        peer_closed: pd.DataFrame | None = None
        if symbol in {"BTCUSDT", "ETHUSDT"} and interval == "1d":
            peer_symbol = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
            peer_snap = manager.get_snapshot(
                root=root,
                symbol=peer_symbol,
                interval=interval,
                force_rest=force_sync,
                ensure_ws=False,
            )
            peer_closed = peer_snap.closed_candles

        log_file = default_log_path(root)
        outcomes: list[ModelInferenceOutcome] = []
        valid_results: list[LiveInferenceResult] = []

        if selected_descriptors and not snapshot.closed_candles.empty:
            for desc in selected_descriptors:
                seed_recent_closed_predictions(
                    desc,
                    snapshot.closed_candles,
                    peer_closed_candles=peer_closed,
                    max_recent=seed_history_bars,
                    log_path=log_file,
                    root=root,
                )
            outcomes = run_batch_inference(
                selected_descriptors,
                snapshot.closed_candles,
                peer_closed_candles=peer_closed,
            )
            for out in outcomes:
                if out.result is not None:
                    valid_results.append(out.result)
                    record_prospective_prediction(out.result, log_path=log_file, root=root)
            evaluate_pending_predictions(
                snapshot.closed_candles,
                asset=symbol,
                interval=interval,
                log_path=log_file,
                root=root,
            )

        tab_overview, tab_models, tab_comparison, tab_prospective = st.tabs(
            [
                "1. Vista general",
                f"2. Modelos ({len(all_catalog)} en catálogo)",
                f"3. Comparación ({len(valid_results)} activos)",
                "4. Resultados reales (Prospectivo)",
            ]
        )

        # =====================================================================
        # SECCIÓN 1: VISTA GENERAL
        # =====================================================================
        with tab_overview:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Activo y Temporalidad", f"{symbol} · {interval}")
            ref_close_val = (
                float(snapshot.closed_candles.iloc[-1]["close"])
                if not snapshot.closed_candles.empty
                else None
            )
            live_p = snapshot.live_price or ref_close_val
            delta_pct = (
                ((live_p / ref_close_val) - 1.0) * 100.0
                if (live_p and ref_close_val and ref_close_val > 0)
                else 0.0
            )
            k2.metric(
                "Precio en Vivo",
                f"{live_p:,.4f}"
                if live_p and live_p < 100
                else (f"{live_p:,.2f}" if live_p else "N/D"),
                f"{delta_pct:+.2f}% vs último cierre",
            )
            k3.metric(
                "Modelos Compatibles / Seleccionados",
                f"{len(compatible_models)} / {len(selected_descriptors)}",
                f"{len(deployable_catalog)} desplegables en repo",
            )
            last_closed_str = (
                str(snapshot.closed_candles.iloc[-1]["reference_time"])[:19].replace("T", " ")
                if not snapshot.closed_candles.empty
                else "N/D"
            )
            k4.metric("Última Vela Cerrada (UTC)", last_closed_str)
            k5.metric(
                "Última Actualización (UTC)",
                snapshot.last_update_utc[:19].replace("T", " "),
                f"WS msgs: {snapshot.ws_messages_received}",
            )

            q = snapshot.quality
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                st.markdown(
                    f"**Estado de conexión:** `{snapshot.connection_status}`  \n"
                    f"**Fuente de datos:** `{snapshot.data_source}`"
                )
            with q_col2:
                if q is not None:
                    st.markdown(
                        f"**Calidad de datos:** `{q.quality_status}`  \n"
                        f"**Velas cerradas en memoria:** `{q.closed_bars_count}` · "
                        f"**Huecos detectados:** `{q.missing_bars_count}` · "
                        f"**Duplicados filtrados:** `{q.duplicates_removed}` · "
                        f"**Antigüedad último cierre:** `{q.staleness_hours:.1f} h`"
                    )

            if snapshot.forming_candle is not None:
                fc = snapshot.forming_candle
                st.caption(
                    f"Vela en formación actual: apertura `{fc['timestamp'][:19].replace('T', ' ')} UTC` → "
                    f"cierre esperado `{fc['expected_close_time'][:19].replace('T', ' ')} UTC`. "
                    "Se muestra en ámbar traslúcido en el gráfico, pero NUNCA se utiliza como entrada de features ni como predicción definitiva."
                )

            if snapshot.errors:
                with st.expander("Avisos / Registro de reconexiones recientes", expanded=False):
                    for err in snapshot.errors:
                        st.text(err)

            if not snapshot.closed_candles.empty:
                st.plotly_chart(
                    _build_candlestick_figure(snapshot, ref_close_val),
                    width="stretch",
                )
            if not compatible_models and no_model_reason:
                st.error(f"**SIN MODELO VÁLIDO DISPONIBLE:** {no_model_reason}")

        # =====================================================================
        # SECCIÓN 2: MODELOS (CATÁLOGO COMPLETO)
        # =====================================================================
        with tab_models:
            st.subheader("Catálogo Auditado de Modelos y Estrategias del Repositorio")
            cat_filter = st.radio(
                "Filtrar catálogo por categoría y estado",
                options=[
                    f"Compatibles con {symbol} ({interval}) ({len(compatible_models)})",
                    f"Todos los modelos desplegables del repositorio ({len(deployable_catalog)})",
                    f"Solo Modelos de Machine Learning ({sum(1 for m in deployable_catalog if m.category == 'ML_MODEL')})",
                    f"Solo Estrategias Deterministas / Reglas ({sum(1 for m in deployable_catalog if m.category == 'DETERMINISTIC_RULE')})",
                    f"No desplegables / Incompatibles con motivo técnico ({len(non_deployable_catalog)})",
                ],
                horizontal=True,
            )

            if cat_filter.startswith("Compatibles"):
                shown_models = compatible_models
            elif cat_filter.startswith("Todos"):
                shown_models = deployable_catalog
            elif cat_filter.startswith("Solo Modelos"):
                shown_models = [m for m in deployable_catalog if m.category == "ML_MODEL"]
            elif cat_filter.startswith("Solo Estrategias"):
                shown_models = [m for m in deployable_catalog if m.category == "DETERMINISTIC_RULE"]
            else:
                shown_models = non_deployable_catalog

            if shown_models:
                st.dataframe(
                    _catalog_dataframe(shown_models),
                    width="stretch",
                    hide_index=True,
                )

                with st.expander("Inspeccionar ficha técnica completa y features de un modelo"):
                    inspect_map = {f"{m.display_name} ({m.version})": m for m in shown_models}
                    inspected_key = st.selectbox(
                        "Seleccionar modelo para ver contrato completo",
                        options=list(inspect_map.keys()),
                    )
                    im = inspect_map[inspected_key]
                    st.markdown(
                        f"- **Nombre descriptivo:** {im.display_name}\n"
                        f"- **Identificador / Versión original:** `{im.version}` (`{im.model_id}`)\n"
                        f"- **Categoría:** `{im.category}` · **Etapa:** {im.stage}\n"
                        f"- **Arquitectura:** {im.architecture}\n"
                        f"- **Activos compatibles:** `{', '.join(im.compatible_assets)}` · **Temporalidad:** `{im.interval}`\n"
                        f"- **Horizonte de predicción:** {im.horizon_label} (`horizon={im.horizon}`)\n"
                        f"- **Objetivo de optimización:** {im.objective_description}\n"
                        f"- **Período de entrenamiento:** `{im.training_period}` (fin de selección: `{im.selection_label_end}`)\n"
                        f"- **Estado de calibración:** {im.calibration_summary}\n"
                        f"- **Estado de disponibilidad:** `{im.availability_status}`\n"
                        f"- **Ruta del artefacto:** `{im.artifact_path}`"
                        + (
                            f" (SHA-256: `{im.artifact_sha256[:16]}...`)"
                            if im.artifact_sha256
                            else ""
                        )
                        + f"\n- **Limitaciones conocidas / Diagnóstico:** {im.unavailable_reason or im.known_limitations}\n"
                        f"- **Features utilizadas ({im.feature_count}):** `{', '.join(im.features) if im.features else 'N/A'}`"
                    )
            else:
                st.info("No hay modelos en esta vista para la selección actual.")

        # =====================================================================
        # SECCIÓN 3: COMPARACIÓN SIMULTÁNEA DE MODELOS
        # =====================================================================
        with tab_comparison:
            st.subheader(f"Comparación Simultánea en Vivo — {symbol} ({interval})")
            if not outcomes:
                st.info(
                    "Seleccioná al menos un activo, temporalidad y modelo compatible en la barra lateral para comparar inferencias en tiempo real."
                )
            else:
                comp_df = _comparison_dataframe(outcomes)
                f_col1, f_col2, f_col3 = st.columns(3)
                pred_filter = f_col1.multiselect(
                    "Filtrar por predicción / señal",
                    options=sorted(comp_df["Predicción"].unique().tolist()),
                    default=sorted(comp_df["Predicción"].unique().tolist()),
                )
                type_filter = f_col2.multiselect(
                    "Filtrar por tipo (ML vs Regla)",
                    options=sorted(comp_df["Tipo"].unique().tolist()),
                    default=sorted(comp_df["Tipo"].unique().tolist()),
                )
                sort_col = f_col3.selectbox(
                    "Ordenar tabla por",
                    options=[
                        "Probabilidad P(subida) (Mayor a menor)",
                        "Modelo (A-Z)",
                        "Horizonte",
                        "Predicción",
                    ],
                    index=0,
                )

                filtered_comp = comp_df.loc[
                    comp_df["Predicción"].isin(pred_filter) & comp_df["Tipo"].isin(type_filter)
                ].copy()

                if sort_col.startswith("Probabilidad"):
                    filtered_comp = filtered_comp.sort_values("_prob_sort", ascending=False)
                elif sort_col == "Modelo (A-Z)":
                    filtered_comp = filtered_comp.sort_values("Modelo", ascending=True)
                elif sort_col == "Horizonte":
                    filtered_comp = filtered_comp.sort_values("Horizonte", ascending=True)
                else:
                    filtered_comp = filtered_comp.sort_values("Predicción", ascending=True)

                display_cols = [c for c in filtered_comp.columns if c != "_prob_sort"]
                st.dataframe(filtered_comp[display_cols], width="stretch", hide_index=True)

                # Acuerdos y desacuerdos entre modelos científicamente compatibles
                if valid_results:
                    st.markdown("#### Consenso, Acuerdos y Desacuerdos por Grupo Compatible")
                    st.caption(
                        "Los modelos se agrupan estrictamente por (Activo, Horizonte, Tipo de Objetivo) "
                        "para no mezclar probabilidades direccionales de 1 día con objetivos a 3/5 días ni con políticas de posición continua."
                    )
                    agreement_rows = compute_live_agreement_summary(valid_results)
                    st.dataframe(pd.DataFrame(agreement_rows), width="stretch", hide_index=True)

                    prob_fig = _build_probability_comparison_chart(valid_results)
                    if prob_fig is not None:
                        st.plotly_chart(prob_fig, width="stretch")

                    pos_fig = _build_position_comparison_chart(valid_results)
                    if pos_fig is not None:
                        st.plotly_chart(pos_fig, width="stretch")

        # =====================================================================
        # SECCIÓN 4: RESULTADOS REALES (SEGUIMIENTO PROSPECTIVO)
        # =====================================================================
        with tab_prospective:
            st.subheader("Seguimiento Prospectivo y Evaluación en Datos Nuevos")
            st.caption(
                f"Archivo prospectivo persistente: `{log_file.as_posix()}` (completamente separado de los backtests históricos). "
                "Cada predicción queda registrada al cierre de su vela base y se evalúa automáticamente cuando cierra la vela de su horizonte objetivo."
            )

            selected_versions = [d.version for d in selected_descriptors]
            prospective_records = load_prospective_predictions(
                log_path=log_file,
                asset=symbol,
                interval=interval,
                model_versions=selected_versions if selected_versions else None,
                limit=500,
            )

            if not prospective_records:
                st.info(
                    "Aún no hay predicciones prospectivas registradas para la selección actual."
                )
            else:
                st.markdown(
                    "#### Métricas Prospectivas por Modelo (Separadas por Horizonte y Objetivo)"
                )
                model_metrics = compute_model_prospective_metrics(prospective_records)
                st.dataframe(_metrics_dataframe(model_metrics), width="stretch", hide_index=True)

                st.markdown(
                    "#### Comparación Científicamente Válida sobre Ventanas Temporales Comunes"
                )
                st.caption(
                    "Solo compara modelos que comparten exactamente el mismo activo, temporalidad, horizonte y tipo de salida, "
                    "restringiendo el cálculo a la intersección exacta de fechas ya evaluadas (`common_n`)."
                )
                common_comp = compute_common_window_comparison(prospective_records)
                if common_comp:
                    common_df = _metrics_dataframe(common_comp)
                    common_df.insert(0, "Ventana Común", [c["common_window"] for c in common_comp])
                    common_df.insert(0, "Grupo Compatible", [c["group_label"] for c in common_comp])
                    st.dataframe(common_df, width="stretch", hide_index=True)
                else:
                    st.info(
                        "Aún no hay suficientes predicciones evaluadas en fechas comunes para comparar grupos."
                    )

                st.markdown(
                    "#### Historial Detallado de Predicciones Prospectivas (Pendientes y Evaluadas)"
                )
                status_filter = st.multiselect(
                    "Filtrar historial por estado",
                    options=["PENDIENTE", "EVALUADA"],
                    default=["PENDIENTE", "EVALUADA"],
                )
                filtered_hist = [r for r in prospective_records if r.get("status") in status_filter]
                st.dataframe(
                    _format_predictions_table(filtered_hist[:150]),
                    width="stretch",
                    hide_index=True,
                )

    _live_laboratory()


if __name__ == "__main__":
    render_dashboard()
