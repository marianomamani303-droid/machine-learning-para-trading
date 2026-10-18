"""Focused study of the most promising financial variables for predictive and economic performance.

Evaluates:
1. BTCUSDT & ETHUSDT (D1): Liquidity (Corwin-Schultz, range impact, Amihud, trade size),
   Parkinson volatility ratio, and S&P 500 causal momentum, selected per asset in
   nested inner validation without touching outer test folds.
2. EURUSD (D1 with H1 bid/ask execution): CBOE VX/VIX futures term basis and
   OTC quote liquidity (spread z-score, spread-to-range, Corwin-Schultz, quote range excess).
3. Models: Regularized Logistic Regression (L2), Random Forest, and XGBoost.
4. Controls: Original baseline features (F0_base) and matched-dimension causal
   Gaussian noise controls on the exact same observations.
5. Economic Simulation: Real transaction costs (and 2x cost stress) via the repository's
   causal event simulators (`simulate_events` for Crypto, `Engine.portfolio` for Forex).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from trading_ml.common import (
    code_provenance,
    environment,
    read_json,
    sha256,
    sidecar,
    write_new_json,
)
from trading_ml.config.settings import Features
from trading_ml.errors import DataError, LeakageError
from trading_ml.evaluation.metrics import classification_metrics
from trading_ml.evaluation.uncertainty import _auc, moving_block_indices
from trading_ml.expansion.data import load_snapshot as load_crypto_snapshot
from trading_ml.expansion.features import point_in_time_join, zscore_past
from trading_ml.feature_research.diagnostics import (
    binary_loss,
    permutation_diagnostic,
    prune_correlated,
)
from trading_ml.features.technical import build_features, feature_columns
from trading_ml.frontier_research.models import create_estimator
from trading_ml.label_research.events import ExitRule, causal_scales
from trading_ml.label_research.execution import simulate_events
from trading_ml.momentum.config import ResearchCosts
from trading_ml.multimarket.data import load_snapshot as load_fx_snapshot
from trading_ml.multimarket.execution import Costs as FxCosts
from trading_ml.multimarket.execution import Engine as FxEngine
from trading_ml.multimarket.features import PRICE as FX_BASE_COLUMNS
from trading_ml.multimarket.features import features as build_fx_features
from trading_ml.training.recipe import protect_reserved_labels

ROOT = Path("artifacts")
STOP = pd.Timestamp("2025-07-01", tz="UTC")
BOUNDS = pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01", "2025-07-01"], utc=True)
EMBARGO_DAYS = 6
INNER_DAYS = 120
WARMUP_Z = 63

CRYPTO_BASE_COLUMNS: list[str] = feature_columns(Features())
XGB_PARAMETERS: dict[str, Any] = read_json(
    ROOT / "models/v20261012T130540_d2bb6ac0/metadata.json"
)["selected_parameters"]
CRYPTO_COSTS = ResearchCosts(commission_bps=10, slippage_bps=5, spread_bps=2)
FX_COSTS = FxCosts(
    forex_commission_bps=0.35,
    forex_slippage_pips=0.1,
    forex_swap_annual_bps=100.0,
    forex_additional_spread_pips=0.0,
    risk_fraction=0.01,
    maximum_notional_fraction=1.0,
    maximum_entry_spread_pips=5.0,
    initial_capital=10000.0,
)

# Promising financial variables identified in FINANCIAL_FEATURE_ABLATION_REPORT.md
CRYPTO_LIQUIDITY_COLS = [
    "fin_liq_corwin_schultz_20",
    "fin_liq_range_impact_z63",
    "fin_liq_amihud_z63",
    "fin_liq_trade_size_z63",
]
CRYPTO_PARKINSON_COLS = ["fin_vol_parkinson_ratio"]
CRYPTO_SP500_COLS = ["fin_cross_sp500_mom_z"]
CRYPTO_PROMISING_ALL = CRYPTO_LIQUIDITY_COLS + CRYPTO_PARKINSON_COLS + CRYPTO_SP500_COLS

FOREX_VX_BASIS_COLS = [
    "fin_basis_vx_z63",
    "fin_basis_vx_term",
    "fin_basis_vx_change_5d",
    "fin_basis_vx_oi_z63",
]
FOREX_LIQUIDITY_COLS = [
    "fin_liq_spread_z63",
    "fin_liq_spread_to_range",
    "fin_liq_corwin_schultz_20",
    "fin_liq_quote_range_excess",
]
FOREX_PROMISING_ALL = FOREX_VX_BASIS_COLS + FOREX_LIQUIDITY_COLS


def _protected_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted((ROOT / "models").glob("*/metadata.json")):
        snapshot[str(path.resolve())] = sha256(path)
    for path in sorted(ROOT.glob("*_studies/*/prospective.json")):
        snapshot[str(path.resolve())] = sha256(path)
    for path in sorted((ROOT / "expansion").glob("*/*/summary.json")):
        snapshot[str(path.resolve())] = sha256(path)
    for path in sorted((ROOT / "financial_feature_ablation").glob("*/summary.json")):
        snapshot[str(path.resolve())] = sha256(path)
    return snapshot


def _corwin_schultz_proxy(high: pd.Series, low: pd.Series, vol_20: pd.Series) -> pd.Series:
    hl_1 = np.log((high / low).clip(lower=1.0))
    beta = hl_1.pow(2) + hl_1.shift(1).pow(2)
    h_2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    l_2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log((h_2 / l_2).clip(lower=1.0)).pow(2)
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / k) - np.sqrt((gamma / k).clip(lower=0.0))
    spread_est = 2.0 * (np.exp(alpha.clip(lower=0.0)) - 1.0) / (1.0 + np.exp(alpha.clip(lower=0.0)))
    return spread_est.rolling(20, min_periods=20).mean() / vol_20.clip(lower=1e-6)


def _load_tradfi_context(
    hist_index: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    files: dict[str, str] = {}
    by_symbol = {
        d["symbol"]: d
        for d in hist_index["datasets"]
        if d["market_type"] == "cash_index_context" and d["interval"] == "1d"
    }
    sp_meta = by_symbol["^GSPC"]
    vix_meta = by_symbol["^VIX"]
    for meta in [sp_meta, vix_meta]:
        p = Path(meta["path"])
        files[str(p.resolve())] = sha256(p)
        files[str(sidecar(p).resolve())] = sha256(sidecar(p))

    sp = pd.read_csv(sp_meta["path"])
    sp["event_time"] = pd.to_datetime(sp["timestamp"], utc=True)
    sp["available_at"] = pd.to_datetime(sp["available_at"], utc=True)
    sp = sp.loc[sp.available_at <= STOP].sort_values("available_at").reset_index(drop=True)
    sp_ret = sp.close.pct_change(fill_method=None)
    sp_vol = sp_ret.rolling(60, min_periods=60).std().clip(lower=1e-6)
    sp["sp500_mom_z"] = sp.close.pct_change(21, fill_method=None) / (sp_vol * np.sqrt(21.0))
    sp = sp.dropna(subset=["sp500_mom_z"]).reset_index(drop=True)

    vix = pd.read_csv(vix_meta["path"])
    vix["event_time"] = pd.to_datetime(vix["timestamp"], utc=True)
    vix["available_at"] = pd.to_datetime(vix["available_at"], utc=True)
    vix = vix.loc[vix.available_at <= STOP].sort_values("available_at").reset_index(drop=True)
    vix["vix_close"] = vix.close

    vx_items = [
        d
        for d in hist_index["datasets"]
        if d["market_type"] == "exchange_dated_futures" and d["interval"] == "1d"
    ]
    vx_frames = []
    for item in vx_items:
        p = Path(item["path"])
        df = pd.read_csv(p)
        df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
        df = df.loc[df.available_at <= STOP].copy()
        if df.empty:
            continue
        files[str(p.resolve())] = sha256(p)
        exp_dt = pd.to_datetime(
            item["symbol"].split("_")[1], format="%Y%m%d", utc=True
        ) + pd.Timedelta(hours=14)
        df["dte"] = (exp_dt - df["available_at"]).dt.total_seconds() / 86400.0
        df["vx_price"] = df["settlement"].where(df["settlement"] > 0, df["close"])
        df["vx_oi"] = df["open_interest"].clip(lower=1.0)
        vx_frames.append(df[["timestamp", "available_at", "dte", "vx_price", "vx_oi"]])

    vx_all = pd.concat(vx_frames, ignore_index=True)
    vx_valid = vx_all.loc[vx_all.dte >= 3.0].sort_values(["available_at", "dte"])
    vx_front = vx_valid.groupby("available_at", as_index=False).first()
    vx_front["event_time"] = pd.to_datetime(vx_front["timestamp"], utc=True)

    merged_vix = pd.merge(
        vix[["event_time", "available_at", "vix_close"]],
        vx_front[["available_at", "vx_price", "vx_oi", "dte"]],
        on="available_at",
        how="inner",
    ).sort_values("available_at")
    merged_vix["vx_term_basis"] = merged_vix["vx_price"] / merged_vix["vix_close"] - 1.0
    merged_vix["vx_basis_z63"] = zscore_past(merged_vix["vx_term_basis"], window=WARMUP_Z)
    merged_vix["vx_basis_change_5d"] = merged_vix["vx_term_basis"].diff(5)
    merged_vix["vx_oi_z63"] = zscore_past(np.log(merged_vix["vx_oi"]), window=WARMUP_Z)
    merged_vix = merged_vix.dropna(
        subset=["vx_term_basis", "vx_basis_z63", "vx_basis_change_5d", "vx_oi_z63"]
    ).reset_index(drop=True)
    return sp, merged_vix, files


def build_focused_panels() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Build BTCUSDT, ETHUSDT, and EURUSD panels plus EURUSD 1H execution frame."""
    hist_index = read_json(ROOT / "historical/validated_20261017.json")
    exp_index = read_json(ROOT / "expansion/datasets/crypto_d4dca621c1adedb8.json")
    sp_df, vix_df, files = _load_tradfi_context(hist_index)
    files[str((ROOT / "historical/validated_20261017.json").resolve())] = sha256(
        ROOT / "historical/validated_20261017.json"
    )
    files[str((ROOT / "expansion/datasets/crypto_d4dca621c1adedb8.json").resolve())] = sha256(
        ROOT / "expansion/datasets/crypto_d4dca621c1adedb8.json"
    )

    panels: dict[str, pd.DataFrame] = {}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        spot_path = Path(exp_index["assets"][sym]["spot"])
        spot, _ = load_crypto_snapshot(spot_path)
        spot = spot.loc[spot.available_at <= STOP].reset_index(drop=True)
        spot["reference_time"] = spot["available_at"]
        files[str(spot_path.resolve())] = sha256(spot_path)
        files[str(sidecar(spot_path).resolve())] = sha256(sidecar(spot_path))

        base = build_features(spot, Features(), include_volume=True)
        base = causal_scales(base)

        log_ret = np.log(base.close / base.close.shift(1))
        vol_20 = log_ret.rolling(20, min_periods=20).std().clip(lower=1e-8)
        log_hl = np.log((base.high / base.low).clip(lower=1.0))
        parkinson_20 = np.sqrt(
            log_hl.pow(2).rolling(20, min_periods=20).mean() / (4.0 * np.log(2.0))
        )
        base["fin_vol_parkinson_ratio"] = parkinson_20 / vol_20 - 1.0

        abs_ret = base.return_1.abs()
        amihud_20 = (abs_ret / base.quote_volume.clip(lower=1.0)).rolling(
            20, min_periods=20
        ).mean()
        base["fin_liq_amihud_z63"] = zscore_past(
            np.log(amihud_20.clip(lower=1e-18)), window=WARMUP_Z
        )
        range_imp_5 = (
            np.log((base.high / base.low).clip(lower=1e-8)) / base.quote_volume.clip(lower=1.0)
        ).rolling(5, min_periods=5).mean()
        base["fin_liq_range_impact_z63"] = zscore_past(
            np.log(range_imp_5.clip(lower=1e-18)), window=WARMUP_Z
        )
        base["fin_liq_corwin_schultz_20"] = _corwin_schultz_proxy(base.high, base.low, vol_20)
        base["fin_liq_trade_size_z63"] = zscore_past(
            np.log((base.quote_volume / base.trades.clip(lower=1.0)).clip(lower=1e-8)),
            window=WARMUP_Z,
        )

        ref_times = pd.DatetimeIndex(base.reference_time)
        sp_join = point_in_time_join(ref_times, sp_df, ["sp500_mom_z"], pd.Timedelta(days=5))
        base["fin_cross_sp500_mom_z"] = sp_join["sp500_mom_z"].ffill().to_numpy()

        base["target"] = (
            (base.close.shift(-1) > base.close).astype(float).where(base.close.shift(-1).notna())
        )
        base["target_time"] = base.reference_time.shift(-1)
        panels[sym] = base

    # EURUSD D1 and H1 execution frame
    eur_d1_path = next(
        path
        for path in sorted((ROOT / "multimarket/data/EURUSD").glob("1d_*.csv"))
        if read_json(sidecar(path))["instrument"]["provider"] == "fxcm"
        and read_json(sidecar(path))["end_exclusive"].startswith("2025-07-01")
    )
    eur_h1_path = next(
        path
        for path in sorted((ROOT / "multimarket/data/EURUSD").glob("1h_*.csv"))
        if read_json(sidecar(path))["instrument"]["provider"] == "fxcm"
        and read_json(sidecar(path))["end_exclusive"].startswith("2025-07-01")
    )
    for p in [eur_d1_path, eur_h1_path]:
        files[str(p.resolve())] = sha256(p)
        files[str(sidecar(p).resolve())] = sha256(sidecar(p))

    eur_d1_raw, _ = load_fx_snapshot(eur_d1_path)
    eur_h1_raw, _ = load_fx_snapshot(eur_h1_path)
    eur_d1_raw = (
        eur_d1_raw.loc[eur_d1_raw.reference_time <= STOP]
        .sort_values("reference_time")
        .reset_index(drop=True)
    )
    eur_h1_raw = (
        eur_h1_raw.loc[eur_h1_raw.reference_time <= STOP]
        .sort_values("reference_time")
        .reset_index(drop=True)
    )

    eur = build_fx_features(eur_d1_raw, "EURUSD")
    vol_20_fx = eur.return_1.rolling(20, min_periods=20).std().clip(lower=1e-8)

    spread_frac = ((eur.close_ask - eur.close_bid) / eur.close_bid.clip(lower=1e-6)).clip(
        lower=1e-7
    )
    if "crossed_close" in eur:
        spread_frac = spread_frac.where(~eur.crossed_close, np.nan).ffill()
    eur["fin_liq_spread_z63"] = zscore_past(np.log(spread_frac), window=WARMUP_Z)
    hl_range = ((eur.high - eur.low) / eur.close.clip(lower=1e-6)).clip(lower=1e-6)
    eur["fin_liq_spread_to_range"] = spread_frac / hl_range
    eur["fin_liq_corwin_schultz_20"] = _corwin_schultz_proxy(eur.high, eur.low, vol_20_fx)
    outer_q_range = (eur.high_ask - eur.low_bid) / eur.close_bid.clip(lower=1e-6)
    inner_q_range = (eur.high_bid - eur.low_ask).clip(lower=0.0) / eur.close_bid.clip(lower=1e-6)
    eur["fin_liq_quote_range_excess"] = zscore_past(
        (outer_q_range - inner_q_range).clip(lower=1e-7), window=WARMUP_Z
    )

    eur_refs = pd.DatetimeIndex(eur.reference_time)
    vix_fx = point_in_time_join(
        eur_refs,
        vix_df,
        ["vx_term_basis", "vx_basis_z63", "vx_basis_change_5d", "vx_oi_z63"],
        pd.Timedelta(days=5),
    )
    eur["fin_basis_vx_term"] = vix_fx["vx_term_basis"].ffill().to_numpy()
    eur["fin_basis_vx_z63"] = vix_fx["vx_basis_z63"].ffill().to_numpy()
    eur["fin_basis_vx_change_5d"] = vix_fx["vx_basis_change_5d"].ffill().to_numpy()
    eur["fin_basis_vx_oi_z63"] = vix_fx["vx_oi_z63"].ffill().to_numpy()

    eur["target"] = (
        (eur.close.shift(-1) > eur.close).astype(float).where(eur.close.shift(-1).notna())
    )
    eur["target_time"] = eur.reference_time.shift(-1)
    eur["source_index"] = np.arange(len(eur))
    panels["EURUSD"] = eur

    return panels, eur_h1_raw, files


def attach_matched_noise(panels: dict[str, pd.DataFrame]) -> dict[str, dict[int, list[list[str]]]]:
    """Attach 3 independent causal Gaussian noise streams for each needed dimensionality k."""
    noise_bank: dict[str, dict[int, list[list[str]]]] = {}
    for asset_idx, (symbol, frame) in enumerate(panels.items()):
        noise_bank[symbol] = {}
        new_cols: dict[str, np.ndarray] = {}
        for k in range(1, 9):
            streams: list[list[str]] = []
            for s_idx in [1, 2, 3]:
                rng = np.random.default_rng(20265000 + asset_idx * 1000 + k * 10 + s_idx)
                cols = [f"noise_k{k}_s{s_idx}_{j}" for j in range(k)]
                vals = rng.normal(0.0, 1.0, size=(len(frame), k))
                for j, c in enumerate(cols):
                    new_cols[c] = vals[:, j]
                streams.append(cols)
            noise_bank[symbol][k] = streams
        panels[symbol] = pd.concat([frame, pd.DataFrame(new_cols, index=frame.index)], axis=1)
    return noise_bank


def candidate_recipes(symbol: str) -> dict[str, list[str]]:
    if symbol in {"BTCUSDT", "ETHUSDT"}:
        base = CRYPTO_BASE_COLUMNS.copy()
        return {
            "F0_base": base,
            "liq_only": base + CRYPTO_LIQUIDITY_COLS,
            "liq_parkinson": base + CRYPTO_LIQUIDITY_COLS + CRYPTO_PARKINSON_COLS,
            "liq_sp500": base + CRYPTO_LIQUIDITY_COLS + CRYPTO_SP500_COLS,
            "parkinson_sp500": base + CRYPTO_PARKINSON_COLS + CRYPTO_SP500_COLS,
            "liq_parkinson_sp500": base + CRYPTO_PROMISING_ALL,
            "top_core_3": base
            + ["fin_liq_corwin_schultz_20", "fin_liq_range_impact_z63", "fin_vol_parkinson_ratio"],
            "top_core_4_sp500": base
            + [
                "fin_liq_corwin_schultz_20",
                "fin_liq_range_impact_z63",
                "fin_vol_parkinson_ratio",
                "fin_cross_sp500_mom_z",
            ],
        }
    base = FX_BASE_COLUMNS.copy()
    return {
        "F0_base": base,
        "vx_basis_only": base + FOREX_VX_BASIS_COLS,
        "fx_liquidity_only": base + FOREX_LIQUIDITY_COLS,
        "vx_basis_plus_liquidity": base + FOREX_PROMISING_ALL,
        "vx_core_2": base + ["fin_basis_vx_z63", "fin_basis_vx_term"],
        "vx_liq_core_4": base
        + [
            "fin_basis_vx_z63",
            "fin_basis_vx_term",
            "fin_liq_spread_to_range",
            "fin_liq_corwin_schultz_20",
        ],
    }


def fit_and_predict(
    model_kind: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    seed: int,
    *,
    logistic_c: float = 1.0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    x_tr = train[columns]
    y_tr = train.target.to_numpy(dtype=int)
    x_ev = evaluation[columns]

    if model_kind == "xgboost":
        model = XGBClassifier(
            **XGB_PARAMETERS,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            device="cpu",
            eval_metric="logloss",
        )
        model.fit(x_tr.to_numpy(dtype=float), y_tr)
        p_tr = model.predict_proba(x_tr.to_numpy(dtype=float))[:, 1]
        p_ev = model.predict_proba(x_ev.to_numpy(dtype=float))[:, 1]
        return model, p_tr, p_ev

    if model_kind == "logistic_l2":
        scaler = StandardScaler().fit(x_tr.to_numpy(dtype=float))
        clf = LogisticRegression(C=logistic_c, solver="lbfgs", max_iter=2000, random_state=seed)
        clf.fit(scaler.transform(x_tr.to_numpy(dtype=float)), y_tr)
        p_tr = clf.predict_proba(scaler.transform(x_tr.to_numpy(dtype=float)))[:, 1]
        p_ev = clf.predict_proba(scaler.transform(x_ev.to_numpy(dtype=float)))[:, 1]
        return (scaler, clf), p_tr, p_ev

    if model_kind == "random_forest":
        model = create_estimator({"kind": "random_forest"})
        model.set_params(random_state=seed)
        model.fit(x_tr, y_tr)
        p_tr = model.predict_proba(x_tr)[:, 1]
        p_ev = model.predict_proba(x_ev)[:, 1]
        return model, p_tr, p_ev

    raise DataError(f"Unsupported model_kind: {model_kind}")


class _Adapter:
    def __init__(self, estimator: Any, columns: list[str], model_kind: str) -> None:
        self.estimator = estimator
        self.columns = columns
        self.model_kind = model_kind

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_kind == "xgboost":
            return self.estimator.predict_proba(x)
        if self.model_kind == "logistic_l2":
            scaler, clf = self.estimator
            return clf.predict_proba(scaler.transform(x))
        return self.estimator.predict_proba(pd.DataFrame(x, columns=self.columns))


def nested_select_for_asset(
    frame: pd.DataFrame,
    outer_start: pd.Timestamp,
    symbol: str,
    model_kind: str,
    seed: int,
    noise_bank: dict[int, list[list[str]]],
) -> dict[str, Any]:
    """Select asset-specific promising subset and C/threshold strictly on inner validation."""
    end = outer_start - pd.Timedelta(days=EMBARGO_DAYS)
    bounds = [end - pd.Timedelta(days=i * INNER_DAYS) for i in [2, 1, 0]]
    base_cols = CRYPTO_BASE_COLUMNS.copy() if symbol != "EURUSD" else FX_BASE_COLUMNS.copy()
    fin_pool = CRYPTO_PROMISING_ALL if symbol != "EURUSD" else FOREX_PROMISING_ALL
    recipes = candidate_recipes(symbol)

    # Also build a correlation-pruned base + permutation-filtered financial subset on inner windows
    inner_splits = []
    for start_w, end_w in zip(bounds[:-1], bounds[1:], strict=True):
        in_tr = frame.loc[
            (frame.reference_time < start_w)
            & (frame.target_time < start_w - pd.Timedelta(days=EMBARGO_DAYS))
        ]
        in_val = frame.loc[
            (frame.reference_time >= start_w)
            & (frame.reference_time < end_w)
            & (frame.target_time < end_w)
        ]
        if len(in_tr) < 120 or len(in_val) < 30:
            return {
                "chosen_recipe": "F0_base",
                "columns": base_cols,
                "added_k": 0,
                "logistic_c": 1.0,
                "selected_threshold": 0.53,
                "fallback": True,
            }
        inner_splits.append((in_tr, in_val))

    # 1. Evaluate individual financial variable permutation importance on inner validation
    perm_scores: dict[str, list[float]] = {c: [] for c in fin_pool}
    full_cols = base_cols + fin_pool
    for in_tr, in_val in inner_splits:
        est, _, _ = fit_and_predict(model_kind, in_tr, in_val, full_cols, seed)
        rows = permutation_diagnostic(
            _Adapter(est, full_cols, model_kind),
            in_val,
            full_cols,
            {c: [c] for c in fin_pool},
            blocks=(7, 14),
            repeats=6,
            seed=seed,
        )
        for r in rows:
            perm_scores[r["name"]].append(r["delta_loss_mean"])

    # Variables with positive mean inner permutation importance across inner windows & blocks
    pos_fin = [
        c for c in fin_pool if float(np.mean(perm_scores[c])) > 0 and sum(x > 0 for x in perm_scores[c]) >= 2
    ]
    pos_fin.sort(key=lambda c: -float(np.mean(perm_scores[c])))

    # Pruned non-collinear base fitted on outer train prior to outer_start
    outer_train = frame.loc[
        (frame.reference_time < outer_start)
        & (frame.target_time < outer_start - pd.Timedelta(days=EMBARGO_DAYS))
    ]
    pruned_base, _ = prune_correlated(outer_train, base_cols, threshold=0.90)

    dynamic_candidates = dict(recipes)
    if pos_fin:
        dynamic_candidates["inner_perm_selected"] = base_cols + pos_fin[:4]
        dynamic_candidates["pruned_base_plus_perm"] = pruned_base + pos_fin[:4]
    dynamic_candidates["pruned_base_plus_top"] = (
        pruned_base
        + (
            ["fin_liq_corwin_schultz_20", "fin_liq_range_impact_z63", "fin_vol_parkinson_ratio"]
            if symbol != "EURUSD"
            else ["fin_basis_vx_z63", "fin_basis_vx_term", "fin_liq_spread_to_range"]
        )
    )

    # Evaluate all candidate recipes and matched noise on the two inner windows
    cand_losses: dict[str, list[float]] = {k: [] for k in dynamic_candidates}
    cand_aucs: dict[str, list[float]] = {k: [] for k in dynamic_candidates}
    noise_losses_by_k: dict[int, list[float]] = {}
    noise_aucs_by_k: dict[int, list[float]] = {}

    for k in range(1, 9):
        n_l, n_a = [], []
        for in_tr, in_val in inner_splits:
            y_v = in_val.target.to_numpy(dtype=int)
            p_streams = []
            for s_cols in noise_bank[k]:
                _, _, p_v = fit_and_predict(model_kind, in_tr, in_val, base_cols + s_cols, seed)
                p_streams.append(p_v)
            p_mean = np.mean(p_streams, axis=0)
            n_l.append(float(binary_loss(y_v, p_mean).mean()))
            n_a.append(float(_auc(y_v, p_mean)))
        noise_losses_by_k[k] = n_l
        noise_aucs_by_k[k] = n_a

    for name, cols in dynamic_candidates.items():
        for in_tr, in_val in inner_splits:
            y_v = in_val.target.to_numpy(dtype=int)
            _, _, p_v = fit_and_predict(model_kind, in_tr, in_val, cols, seed)
            cand_losses[name].append(float(binary_loss(y_v, p_v).mean()))
            cand_aucs[name].append(float(_auc(y_v, p_v)))

    base_l = np.asarray(cand_losses["F0_base"])
    base_a = np.asarray(cand_aucs["F0_base"])

    qualifying: list[str] = []
    for name, cols in dynamic_candidates.items():
        if name == "F0_base":
            continue
        fin_k = max(1, min(8, len([c for c in cols if c.startswith("fin_")])))
        ref_nl = np.asarray(noise_losses_by_k[fin_k])
        ref_na = np.asarray(noise_aucs_by_k[fin_k])
        l_arr = np.asarray(cand_losses[name])
        a_arr = np.asarray(cand_aucs[name])

        # Require lower mean log-loss AND higher mean ROC-AUC than BOTH F0_base and matched noise
        if float(np.mean(l_arr)) < min(float(np.mean(base_l)), float(np.mean(ref_nl))) and float(
            np.mean(a_arr)
        ) > max(float(np.mean(base_a)), float(np.mean(ref_na))):
            qualifying.append(name)

    if qualifying:
        # Score by standardized improvement in both log-loss and ROC-AUC
        chosen = min(
            qualifying,
            key=lambda n: float(np.mean(cand_losses[n])) - 0.15 * float(np.mean(cand_aucs[n])),
        )
        chosen_cols = dynamic_candidates[chosen]
        fallback = False
    else:
        chosen = "F0_base"
        chosen_cols = base_cols
        fallback = True

    # For logistic_l2, also select regularization C in {0.1, 0.3, 1.0} on inner validation
    best_c = 1.0
    if model_kind == "logistic_l2":
        c_scores = {}
        for c_val in [0.1, 0.3, 1.0]:
            losses_c = []
            for in_tr, in_val in inner_splits:
                y_v = in_val.target.to_numpy(dtype=int)
                _, _, p_v = fit_and_predict(
                    model_kind, in_tr, in_val, chosen_cols, seed, logistic_c=c_val
                )
                losses_c.append(float(binary_loss(y_v, p_v).mean()))
            c_scores[c_val] = float(np.mean(losses_c))
        best_c = min(c_scores, key=lambda k: c_scores[k])

    # Select probability entry threshold in {0.51, 0.52, 0.53, 0.55} on inner validation
    best_tau = 0.53
    best_tau_score = -np.inf
    for tau in [0.51, 0.52, 0.53, 0.55]:
        tau_rets = []
        for in_tr, in_val in inner_splits:
            y_v = in_val.target.to_numpy(dtype=int)
            _, _, p_v = fit_and_predict(
                model_kind, in_tr, in_val, chosen_cols, seed, logistic_c=best_c
            )
            r_next = in_val.close.shift(-1) / in_val.close - 1.0
            r_arr = r_next.fillna(0.0).to_numpy(dtype=float)
            active_m = p_v >= tau
            cost_rate = 0.0017 if symbol != "EURUSD" else 0.0002
            switches = np.abs(np.diff(np.r_[0.0, active_m.astype(float)]))
            net_r = active_m * r_arr - switches * cost_rate
            if active_m.sum() >= 5 and np.std(net_r) > 1e-8:
                tau_rets.append(float(np.mean(net_r) / np.std(net_r) * np.sqrt(252.0)))
            else:
                tau_rets.append(-1.0)
        mean_score = float(np.mean(tau_rets))
        if mean_score > best_tau_score:
            best_tau_score = mean_score
            best_tau = tau

    added_k = max(1, min(8, len([c for c in chosen_cols if c.startswith("fin_")])))
    return {
        "chosen_recipe": chosen,
        "columns": chosen_cols,
        "added_k": added_k,
        "logistic_c": best_c,
        "selected_threshold": best_tau,
        "fallback": fallback,
        "positive_inner_perm_vars": pos_fin,
        "qualifying_candidates": qualifying,
    }


def simulate_crypto_multi_fold(
    full_frame: pd.DataFrame,
    fold_predictions: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, np.ndarray]],
    threshold: float,
    cost_multiplier: int = 1,
) -> dict[str, Any]:
    """Run causal `simulate_events` across all 3 outer folds and aggregate portfolio metrics."""
    costs = replace(
        CRYPTO_COSTS,
        commission_bps=10 * cost_multiplier,
        slippage_bps=5 * cost_multiplier,
        spread_bps=2 * cost_multiplier,
    )
    rule = ExitRule("net_h1", 1)
    daily_returns: list[np.ndarray] = []
    daily_times: list[pd.Series] = []
    all_trades: list[pd.DataFrame] = []
    equity = 1.0
    equity_path = [1.0]

    for start, end, out_rows, prob in fold_predictions:
        p_full = np.zeros(len(full_frame), dtype=float)
        p_full[out_rows.index] = prob
        sim = simulate_events(
            full_frame,
            p_full,
            rule,
            start,
            end,
            costs,
            threshold=threshold,
            maximum_horizon=10,
        )
        r_fold = sim.curve.net_return.iloc[1:].to_numpy(dtype=float)
        t_fold = sim.curve.timestamp.iloc[1:]
        daily_returns.append(r_fold)
        daily_times.append(t_fold)
        if len(sim.trades):
            all_trades.append(sim.trades)
        for r in r_fold:
            equity *= 1.0 + float(r)
            equity_path.append(equity)

    ret_vec = np.concatenate(daily_returns) if daily_returns else np.zeros(0)
    eq_arr = np.asarray(equity_path, dtype=float)
    max_dd = float((eq_arr / np.maximum.accumulate(eq_arr) - 1.0).min()) if len(eq_arr) else 0.0
    sharpe = (
        float(ret_vec.mean() / ret_vec.std(ddof=1) * np.sqrt(365.0))
        if len(ret_vec) > 1 and ret_vec.std(ddof=1) > 1e-12
        else 0.0
    )
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    pos_pnl = trades_df.pnl_cash[trades_df.pnl_cash > 0].sum() if len(trades_df) else 0.0
    neg_pnl = -trades_df.pnl_cash[trades_df.pnl_cash < 0].sum() if len(trades_df) else 0.0
    pf = float(pos_pnl / neg_pnl) if neg_pnl > 0 else None
    win_rate = float((trades_df.net_return > 0).mean()) if len(trades_df) else None

    return {
        "net_return": float(equity - 1.0),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trade_count": int(len(trades_df)),
        "win_rate": win_rate,
        "profit_factor": pf,
        "daily_returns": ret_vec,
    }


def simulate_forex_multi_fold(
    full_d1: pd.DataFrame,
    raw_h1: pd.DataFrame,
    fold_predictions: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, np.ndarray]],
    threshold: float,
    cost_multiplier: int = 1,
) -> dict[str, Any]:
    """Run causal `FxEngine.portfolio` on H1 bid/ask quotes across all 3 outer folds."""
    cost = replace(
        FX_COSTS,
        forex_commission_bps=0.35 * cost_multiplier,
        forex_slippage_pips=0.1 * cost_multiplier,
        forex_swap_annual_bps=100.0 * cost_multiplier,
        forex_additional_spread_pips=0.0 if cost_multiplier == 1 else 1.0,
    )
    engine = FxEngine(
        full_d1,
        "EURUSD",
        cost,
        execution_frame=raw_h1,
        bars_per_horizon=24,
    )
    all_trades: list[pd.DataFrame] = []
    step_returns: list[np.ndarray] = []
    equity = 1.0
    equity_path = [1.0]

    for _, end, out_rows, prob in fold_predictions:
        signals = out_rows.assign(probability=prob)
        _, trades, curve = engine.portfolio(signals, 1, False, threshold, end)
        if len(trades):
            all_trades.append(trades)
        if not curve.empty:
            eq_f = np.r_[cost.initial_capital, curve.equity.to_numpy(dtype=float)]
            r_f = np.diff(eq_f) / eq_f[:-1]
            # Aggregate H1 equity steps to daily buckets for Sharpe comparability
            c_df = pd.DataFrame({"ts": pd.to_datetime(curve.timestamp, utc=True), "r": r_f})
            daily_r = c_df.groupby(c_df.ts.dt.floor("D"))["r"].apply(
                lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0)
            )
            d_arr = daily_r.to_numpy(dtype=float)
            step_returns.append(d_arr)
            for r in d_arr:
                equity *= 1.0 + float(r)
                equity_path.append(equity)

    ret_vec = np.concatenate(step_returns) if step_returns else np.zeros(0)
    eq_arr = np.asarray(equity_path, dtype=float)
    max_dd = float((eq_arr / np.maximum.accumulate(eq_arr) - 1.0).min()) if len(eq_arr) else 0.0
    sharpe = (
        float(ret_vec.mean() / ret_vec.std(ddof=1) * np.sqrt(252.0))
        if len(ret_vec) > 1 and ret_vec.std(ddof=1) > 1e-12
        else 0.0
    )
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    profits = trades_df.profit_usd.to_numpy(dtype=float) if len(trades_df) else np.zeros(0)
    pos_pnl = profits[profits > 0].sum() if len(profits) else 0.0
    neg_pnl = -profits[profits < 0].sum() if len(profits) else 0.0
    pf = float(pos_pnl / neg_pnl) if neg_pnl > 0 else None
    win_rate = float((profits > 0).mean()) if len(profits) else None

    return {
        "net_return": float(equity - 1.0),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trade_count": int(len(trades_df)),
        "win_rate": win_rate,
        "profit_factor": pf,
        "daily_returns": ret_vec,
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for idx, (key, p) in enumerate(items):
        adj = min(1.0, p * (m - idx))
        running_max = max(running_max, adj)
        adjusted[key] = running_max
    return adjusted


def run_focused_study(output_root: Path | None = None) -> dict[str, Any]:
    protected_before = _protected_snapshot()
    panels, eur_h1_raw, input_files = build_focused_panels()
    noise_bank = attach_matched_noise(panels)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    out_dir = (
        output_root.resolve()
        if output_root is not None
        else (ROOT / "focused_financial_study" / run_id).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    models_to_eval = [
        ("logistic_l2", [42]),
        ("random_forest", [42, 1729]),
        ("xgboost", [42, 1729]),
    ]

    plan = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "stop_quarantine": STOP.isoformat(),
        "boundaries": [b.isoformat() for b in BOUNDS],
        "embargo_days": EMBARGO_DAYS,
        "inner_days": INNER_DAYS,
        "crypto_promising_variables": CRYPTO_PROMISING_ALL,
        "forex_promising_variables": FOREX_PROMISING_ALL,
        "models": [{"kind": m, "seeds": s} for m, s in models_to_eval],
        "inputs": input_files,
        "protected_before": protected_before,
        "environment": environment(),
        "code": code_provenance(),
    }
    write_new_json(out_dir / "plan.json", plan)

    all_predictions: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    nested_decisions: list[dict[str, Any]] = []
    economic_records: list[dict[str, Any]] = []

    for symbol, full in panels.items():
        base_cols = CRYPTO_BASE_COLUMNS.copy() if symbol != "EURUSD" else FX_BASE_COLUMNS.copy()
        fin_cols = CRYPTO_PROMISING_ALL if symbol != "EURUSD" else FOREX_PROMISING_ALL
        recipes = candidate_recipes(symbol)

        # Common eligibility across base, promising financial variables, and noise columns
        noise_cols_all = [
            c for k_streams in noise_bank[symbol].values() for s in k_streams for c in s
        ]
        required_cols = list(dict.fromkeys(base_cols + fin_cols + noise_cols_all))
        if symbol != "EURUSD":
            required_cols += ["label_sigma", "label_atr_fraction"]
        else:
            required_cols += ["atr_ratio"]

        eligible = full[required_cols].notna().all(axis=1) & full.target.notna()
        frame = full.loc[eligible].copy()
        protect_reserved_labels(ROOT, frame)

        for model_kind, seeds in models_to_eval:
            for seed in seeds:
                fold_store: dict[
                    str, list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, np.ndarray]]
                ] = {}
                tau_by_fold: list[float] = []

                for fold, (start, end) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:], strict=True)):
                    train = frame.loc[
                        (frame.reference_time < start)
                        & (frame.target_time < start - pd.Timedelta(days=EMBARGO_DAYS))
                    ].copy()
                    # Respect calendar buffer at end of fold for event simulation comparability
                    eval_cutoff = end - pd.Timedelta(days=12 if symbol != "EURUSD" else 20)
                    evaluation = frame.loc[
                        (frame.reference_time >= start) & (frame.reference_time < eval_cutoff)
                    ].copy()
                    protect_reserved_labels(ROOT, train, evaluation)

                    nested = nested_select_for_asset(
                        frame, start, symbol, model_kind, seed, noise_bank[symbol]
                    )
                    tau_by_fold.append(nested["selected_threshold"])
                    nested_decisions.append(
                        {
                            "symbol": symbol,
                            "fold": fold,
                            "model": model_kind,
                            "seed": seed,
                            "chosen_recipe": nested["chosen_recipe"],
                            "column_count": len(nested["columns"]),
                            "added_k": nested["added_k"],
                            "logistic_c": nested["logistic_c"],
                            "selected_threshold": nested["selected_threshold"],
                            "fallback": nested["fallback"],
                            "positive_inner_perm_vars": nested["positive_inner_perm_vars"],
                            "qualifying_candidates": nested["qualifying_candidates"],
                        }
                    )

                    # Evaluate fixed candidate recipes + nested_selected + matched noise controls
                    fold_recipes = dict(recipes)
                    fold_recipes["nested_selected"] = nested["columns"]
                    if nested["positive_inner_perm_vars"]:
                        fold_recipes["inner_perm_top"] = (
                            base_cols + nested["positive_inner_perm_vars"][:4]
                        )
                    else:
                        fold_recipes["inner_perm_top"] = base_cols

                    # Add noise controls for k in {3, 4, 5, 6, 8} as applicable
                    needed_ks = {
                        max(1, min(8, len([c for c in cols if c.startswith("fin_")])))
                        for r_name, cols in fold_recipes.items()
                        if r_name != "F0_base"
                    }
                    needed_ks.add(nested["added_k"])
                    for k in sorted(needed_ks):
                        for s_idx, s_cols in enumerate(noise_bank[symbol][k], start=1):
                            fold_recipes[f"noise_k{k}_s{s_idx}"] = base_cols + s_cols

                    # Fit and predict all recipes on the exact same train/evaluation observations
                    p_by_recipe: dict[str, np.ndarray] = {}
                    tr_by_recipe: dict[str, np.ndarray] = {}
                    for r_name, cols in fold_recipes.items():
                        c_param = (
                            nested["logistic_c"]
                            if (model_kind == "logistic_l2" and r_name == "nested_selected")
                            else 1.0
                        )
                        _, p_tr, p_ev = fit_and_predict(
                            model_kind, train, evaluation, cols, seed, logistic_c=c_param
                        )
                        p_by_recipe[r_name] = p_ev
                        tr_by_recipe[r_name] = p_tr

                    # Also build pooled noise_k{k}_mean for each k
                    for k in sorted(needed_ks):
                        p_by_recipe[f"noise_k{k}_mean"] = np.mean(
                            [p_by_recipe[f"noise_k{k}_s{s_idx}"] for s_idx in [1, 2, 3]], axis=0
                        )
                        tr_by_recipe[f"noise_k{k}_mean"] = np.mean(
                            [tr_by_recipe[f"noise_k{k}_s{s_idx}"] for s_idx in [1, 2, 3]], axis=0
                        )

                    # Alias matched noise for nested_selected
                    p_by_recipe["noise_nested_match"] = p_by_recipe[
                        f"noise_k{nested['added_k']}_mean"
                    ]
                    tr_by_recipe["noise_nested_match"] = tr_by_recipe[
                        f"noise_k{nested['added_k']}_mean"
                    ]

                    for r_name, p_ev in p_by_recipe.items():
                        if r_name.endswith(("_s1", "_s2", "_s3")):
                            continue
                        p_tr = tr_by_recipe[r_name]
                        tr_m = classification_metrics(train.target.to_numpy(dtype=int), p_tr)
                        ev_m = classification_metrics(evaluation.target.to_numpy(dtype=int), p_ev)
                        ev_m["balanced_accuracy"] = float(
                            balanced_accuracy_score(
                                evaluation.target.to_numpy(dtype=int), p_ev >= 0.5
                            )
                        )
                        cols_used = fold_recipes.get(r_name, base_cols)
                        fit_records.append(
                            {
                                "symbol": symbol,
                                "fold": fold,
                                "model": model_kind,
                                "seed": seed,
                                "recipe": r_name,
                                "feature_count": len(cols_used),
                                "train_rows": int(len(train)),
                                "oos_rows": int(len(evaluation)),
                                "train_log_loss": tr_m["log_loss"],
                                "train_roc_auc": tr_m["roc_auc"],
                                "oos_log_loss": ev_m["log_loss"],
                                "oos_roc_auc": ev_m["roc_auc"],
                                "oos_brier": ev_m["brier_score"],
                                "oos_accuracy": ev_m["accuracy"],
                                "oos_balanced_accuracy": ev_m["balanced_accuracy"],
                                "generalization_gap_loss": ev_m["log_loss"] - tr_m["log_loss"],
                            }
                        )
                        fold_store.setdefault(r_name, []).append((start, end, evaluation, p_ev))
                        rows = evaluation[["reference_time", "target_time", "target"]].copy()
                        rows["symbol"] = symbol
                        rows["fold"] = fold
                        rows["model"] = model_kind
                        rows["seed"] = seed
                        rows["recipe"] = r_name
                        rows["probability"] = p_ev
                        all_predictions.append(rows)

                # Run economic simulations across the 3 outer folds for each main recipe
                mean_inner_tau = float(np.median(tau_by_fold))
                for r_name, fold_list in fold_store.items():
                    for tau_label, tau_val in [
                        ("fixed_055", 0.55),
                        ("fixed_052", 0.52),
                        ("inner_tau", mean_inner_tau),
                    ]:
                        for cost_label, mult in [("base_cost", 1), ("double_cost", 2)]:
                            if symbol != "EURUSD":
                                econ = simulate_crypto_multi_fold(
                                    full, fold_list, threshold=tau_val, cost_multiplier=mult
                                )
                            else:
                                econ = simulate_forex_multi_fold(
                                    full,
                                    eur_h1_raw,
                                    fold_list,
                                    threshold=tau_val,
                                    cost_multiplier=mult,
                                )
                            economic_records.append(
                                {
                                    "symbol": symbol,
                                    "model": model_kind,
                                    "seed": seed,
                                    "recipe": r_name,
                                    "threshold_mode": tau_label,
                                    "threshold_value": tau_val,
                                    "cost_scenario": cost_label,
                                    "net_return": econ["net_return"],
                                    "sharpe": econ["sharpe"],
                                    "max_drawdown": econ["max_drawdown"],
                                    "trade_count": econ["trade_count"],
                                    "win_rate": econ["win_rate"],
                                    "profit_factor": econ["profit_factor"],
                                }
                            )
                print(f"COMPLETED {symbol} {model_kind} seed={seed}", flush=True)

    pred_df = pd.concat(all_predictions, ignore_index=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    write_new_json(out_dir / "fits.json", fit_records)
    write_new_json(out_dir / "nested_decisions.json", nested_decisions)
    write_new_json(out_dir / "economics.json", economic_records)
    pd.DataFrame(economic_records).to_csv(out_dir / "economic_table.csv", index=False)

    if protected_before != _protected_snapshot():
        raise LeakageError("Protected models or prospective reserves altered during focused study")

    return postprocess_study(out_dir)


def postprocess_study(out_dir: Path) -> dict[str, Any]:
    pred_df = pd.read_csv(out_dir / "predictions.csv")
    nested_decisions = read_json(out_dir / "nested_decisions.json")
    statistical_contrasts: list[dict[str, Any]] = []

    # Compute paired moving-block bootstrap (blocks 30 days, 2000 samples)
    for (symbol, model_kind, seed), grp in pred_df.groupby(["symbol", "model", "seed"]):
        wide = (
            grp.pivot(
                index=["fold", "reference_time", "target_time", "target"],
                columns="recipe",
                values="probability",
            )
            .sort_index()
            .dropna(axis=1)
        )
        y = wide.index.get_level_values("target").to_numpy(dtype=int)
        folds = wide.index.get_level_values("fold").to_numpy(dtype=int)
        probs = {col: wide[col].to_numpy(dtype=float) for col in wide.columns}
        losses = {col: binary_loss(y, probs[col]) for col in wide.columns}

        point_m = {}
        for col, p in probs.items():
            m = classification_metrics(y, p)
            m["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= 0.5))
            point_m[col] = m

        recipes = candidate_recipes(symbol)
        eval_candidates = [r for r in recipes if r != "F0_base"] + [
            "nested_selected",
            "inner_perm_top",
        ]

        idx_30 = moving_block_indices(folds, 2000, 30, int(seed) + 30)
        loss_draws = {col: losses[col][idx_30].mean(axis=1) for col in probs}
        auc_draws = {
            col: np.array([_auc(y[row], probs[col][row]) for row in idx_30]) for col in probs
        }

        raw_p_loss_base: dict[str, float] = {}
        raw_p_auc_base: dict[str, float] = {}
        raw_p_loss_noise: dict[str, float] = {}
        raw_p_auc_noise: dict[str, float] = {}
        temp_rows: list[dict[str, Any]] = []

        for cand in eval_candidates:
            if cand not in probs:
                continue
            if cand == "nested_selected":
                noise_ref = "noise_nested_match"
            elif cand == "inner_perm_top":
                noise_ref = "noise_k2_mean" if "noise_k2_mean" in probs else "F0_base"
            else:
                cols_c = recipes.get(cand, [])
                k_c = max(1, min(8, len([c for c in cols_c if c.startswith("fin_")])))
                noise_ref = f"noise_k{k_c}_mean" if f"noise_k{k_c}_mean" in probs else "F0_base"

            dl_base = loss_draws[cand] - loss_draws["F0_base"]
            da_base = auc_draws[cand] - auc_draws["F0_base"]
            dl_noise = loss_draws[cand] - loss_draws[noise_ref]
            da_noise = auc_draws[cand] - auc_draws[noise_ref]

            p_l_base = float(np.mean(dl_base >= 0.0))
            p_a_base = float(np.mean(da_base <= 0.0))
            p_l_noise = float(np.mean(dl_noise >= 0.0))
            p_a_noise = float(np.mean(da_noise <= 0.0))

            raw_p_loss_base[cand] = p_l_base
            raw_p_auc_base[cand] = p_a_base
            raw_p_loss_noise[cand] = p_l_noise
            raw_p_auc_noise[cand] = p_a_noise

            # Count folds where candidate simultaneously improves loss and AUC
            folds_beat_base = 0
            folds_beat_noise = 0
            for f_id in np.unique(folds):
                m_f = folds == f_id
                y_f = y[m_f]
                if float(losses[cand][m_f].mean()) < float(losses["F0_base"][m_f].mean()) and float(
                    _auc(y_f, probs[cand][m_f])
                ) > float(_auc(y_f, probs["F0_base"][m_f])):
                    folds_beat_base += 1
                if float(losses[cand][m_f].mean()) < float(losses[noise_ref][m_f].mean()) and float(
                    _auc(y_f, probs[cand][m_f])
                ) > float(_auc(y_f, probs[noise_ref][m_f])):
                    folds_beat_noise += 1

            temp_rows.append(
                {
                    "symbol": symbol,
                    "model": model_kind,
                    "seed": int(seed),
                    "recipe": cand,
                    "noise_reference": noise_ref,
                    "oos_log_loss": point_m[cand]["log_loss"],
                    "oos_roc_auc": point_m[cand]["roc_auc"],
                    "oos_accuracy": point_m[cand]["accuracy"],
                    "oos_balanced_accuracy": point_m[cand]["balanced_accuracy"],
                    "oos_brier": point_m[cand]["brier_score"],
                    "base_log_loss": point_m["F0_base"]["log_loss"],
                    "base_roc_auc": point_m["F0_base"]["roc_auc"],
                    "base_accuracy": point_m["F0_base"]["accuracy"],
                    "noise_log_loss": point_m[noise_ref]["log_loss"],
                    "noise_roc_auc": point_m[noise_ref]["roc_auc"],
                    "noise_accuracy": point_m[noise_ref]["accuracy"],
                    "delta_loss_vs_base": point_m[cand]["log_loss"] - point_m["F0_base"]["log_loss"],
                    "delta_loss_vs_base_ci95": [
                        float(np.quantile(dl_base, 0.025)),
                        float(np.quantile(dl_base, 0.975)),
                    ],
                    "p_one_sided_loss_vs_base": p_l_base,
                    "delta_auc_vs_base": point_m[cand]["roc_auc"] - point_m["F0_base"]["roc_auc"],
                    "delta_auc_vs_base_ci95": [
                        float(np.quantile(da_base, 0.025)),
                        float(np.quantile(da_base, 0.975)),
                    ],
                    "p_one_sided_auc_vs_base": p_a_base,
                    "delta_loss_vs_noise": point_m[cand]["log_loss"]
                    - point_m[noise_ref]["log_loss"],
                    "delta_loss_vs_noise_ci95": [
                        float(np.quantile(dl_noise, 0.025)),
                        float(np.quantile(dl_noise, 0.975)),
                    ],
                    "p_one_sided_loss_vs_noise": p_l_noise,
                    "delta_auc_vs_noise": point_m[cand]["roc_auc"] - point_m[noise_ref]["roc_auc"],
                    "delta_auc_vs_noise_ci95": [
                        float(np.quantile(da_noise, 0.025)),
                        float(np.quantile(da_noise, 0.975)),
                    ],
                    "p_one_sided_auc_vs_noise": p_a_noise,
                    "folds_beat_base_both": folds_beat_base,
                    "folds_beat_noise_both": folds_beat_noise,
                    "total_folds": int(len(np.unique(folds))),
                }
            )

        hl_base = holm_bonferroni(raw_p_loss_base)
        ha_base = holm_bonferroni(raw_p_auc_base)
        hl_noise = holm_bonferroni(raw_p_loss_noise)
        ha_noise = holm_bonferroni(raw_p_auc_noise)
        for r in temp_rows:
            c = r["recipe"]
            r["p_holm_loss_vs_base"] = hl_base[c]
            r["p_holm_auc_vs_base"] = ha_base[c]
            r["p_holm_loss_vs_noise"] = hl_noise[c]
            r["p_holm_auc_vs_noise"] = ha_noise[c]
            statistical_contrasts.append(r)

    summary = {
        "run_id": out_dir.name,
        "directory": str(out_dir),
        "protected_unchanged": True,
        "statistical_contrasts": statistical_contrasts,
        "nested_decisions": nested_decisions,
    }
    if not (out_dir / "summary.json").exists():
        write_new_json(out_dir / "summary.json", summary)
    pd.DataFrame(statistical_contrasts).to_csv(out_dir / "predictive_contrasts.csv", index=False)

    manifest = {
        str(p.relative_to(out_dir)): sha256(p)
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.name != "artifacts.json"
    }
    if not (out_dir / "artifacts.json").exists():
        write_new_json(out_dir / "artifacts.json", manifest)
    print(f"FOCUSED FINANCIAL STUDY COMPLETED: {out_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Optional output directory")
    parser.add_argument(
        "--postprocess-dir",
        type=Path,
        default=None,
        help="Postprocess an existing run directory without repeating model fits",
    )
    args = parser.parse_args()
    if args.postprocess_dir is not None:
        postprocess_study(args.postprocess_dir)
        return
    run_focused_study(args.out)


if __name__ == "__main__":
    main()

