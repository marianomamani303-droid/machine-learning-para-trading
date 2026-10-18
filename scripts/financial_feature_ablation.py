"""Controlled point-in-time ablation study of financial feature families.

Evaluates volatility, liquidity, relative volume, funding rates, futures basis
(perpetual and dated quarterly contracts, plus CBOE VX VIX futures), and
cross-asset relationships across existing model architectures (XGBoost,
Logistic Regression L2, Random Forest, LightGBM) on identical out-of-sample
observations, including matched-dimension Gaussian noise controls and
leave-one-family-out (LOFO) ablations.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

from trading_ml.common import (
    code_provenance,
    environment,
    frame_hash,
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
    correlation_pairs,
    permutation_diagnostic,
)
from trading_ml.features.technical import build_features, feature_columns
from trading_ml.frontier_research.models import create_estimator
from trading_ml.models.baselines import fit_reference_models
from trading_ml.multimarket.data import load_snapshot as load_fx_snapshot
from trading_ml.multimarket.features import PRICE as FX_BASE_COLUMNS
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

CRYPTO_FAMILIES: dict[str, list[str]] = {
    "volatility": [
        "fin_vol_term_5_60",
        "fin_vol_parkinson_ratio",
        "fin_vol_garman_klass_z63",
        "fin_vol_downside_asym_20",
    ],
    "liquidity": [
        "fin_liq_amihud_z63",
        "fin_liq_range_impact_z63",
        "fin_liq_corwin_schultz_20",
        "fin_liq_trade_size_z63",
    ],
    "relative_volume": [
        "fin_rvol_quote_z63",
        "fin_rvol_taker_imbalance",
        "fin_rvol_flow_innovation_3_20",
        "fin_rvol_signed_volume_shock",
    ],
    "funding_rates": [
        "fin_fund_1d",
        "fin_fund_7d",
        "fin_fund_z63",
        "fin_fund_mom_interaction",
    ],
    "futures_basis": [
        "fin_basis_perp_premium",
        "fin_basis_perp_z63",
        "fin_basis_dated_ann",
        "fin_basis_dated_change_5d",
    ],
    "cross_asset": [
        "fin_cross_crypto_residual",
        "fin_cross_usd_strength",
        "fin_cross_sp500_mom_z",
        "fin_cross_vix_term_basis",
    ],
}

FOREX_FAMILIES: dict[str, list[str]] = {
    "volatility": [
        "fin_vol_term_5_60",
        "fin_vol_parkinson_ratio",
        "fin_vol_garman_klass_z63",
        "fin_vol_downside_asym_20",
    ],
    "liquidity": [
        "fin_liq_spread_z63",
        "fin_liq_spread_to_range",
        "fin_liq_corwin_schultz_20",
        "fin_liq_quote_range_excess",
    ],
    "futures_basis": [
        "fin_basis_vx_term",
        "fin_basis_vx_z63",
        "fin_basis_vx_change_5d",
        "fin_basis_vx_oi_z63",
    ],
    "cross_asset": [
        "fin_cross_usd_peers",
        "fin_cross_fx_dispersion",
        "fin_cross_sp500_mom_z",
        "fin_cross_vix_z63",
    ],
}


def _protected_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted((ROOT / "models").glob("*/metadata.json")):
        snapshot[str(path.resolve())] = sha256(path)
    for path in sorted(ROOT.glob("*_studies/*/prospective.json")):
        snapshot[str(path.resolve())] = sha256(path)
    for path in sorted((ROOT / "expansion").glob("*/*/summary.json")):
        snapshot[str(path.resolve())] = sha256(path)
    return snapshot


def _corwin_schultz_proxy(high: pd.Series, low: pd.Series, vol_20: pd.Series) -> pd.Series:
    """Causal two-period high-low range friction proxy normalized by 20d volatility."""
    hl_1 = np.log((high / low).clip(lower=1.0))
    beta = hl_1.pow(2) + hl_1.shift(1).pow(2)
    h_2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    l_2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log((h_2 / l_2).clip(lower=1.0)).pow(2)
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / k) - np.sqrt((gamma / k).clip(lower=0.0))
    spread_est = 2.0 * (np.exp(alpha.clip(lower=0.0)) - 1.0) / (1.0 + np.exp(alpha.clip(lower=0.0)))
    return spread_est.rolling(20, min_periods=20).mean() / vol_20.clip(lower=1e-6)


def _volatility_family(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    log_ret = np.log(out.close / out.close.shift(1))
    vol_5 = log_ret.rolling(5, min_periods=5).std()
    vol_20 = log_ret.rolling(20, min_periods=20).std().clip(lower=1e-8)
    vol_60 = log_ret.rolling(60, min_periods=60).std().clip(lower=1e-8)

    out["fin_vol_term_5_60"] = vol_5 / vol_60 - 1.0

    log_hl = np.log((out.high / out.low).clip(lower=1.0))
    parkinson_20 = np.sqrt(log_hl.pow(2).rolling(20, min_periods=20).mean() / (4.0 * np.log(2.0)))
    out["fin_vol_parkinson_ratio"] = parkinson_20 / vol_20 - 1.0

    log_co = np.log(out.close / out.open)
    gk_var = (0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)).clip(lower=1e-12)
    gk_vol_20 = np.sqrt(gk_var.rolling(20, min_periods=20).mean())
    out["fin_vol_garman_klass_z63"] = zscore_past(np.log(gk_vol_20), window=WARMUP_Z)

    down_sq = log_ret.clip(upper=0.0).pow(2).rolling(20, min_periods=20).mean()
    up_sq = log_ret.clip(lower=0.0).pow(2).rolling(20, min_periods=20).mean()
    out["fin_vol_downside_asym_20"] = (np.sqrt(down_sq) - np.sqrt(up_sq)) / vol_20
    return out


def _load_tradfi_context(
    hist_index: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load point-in-time ^GSPC, ^VIX, and CBOE VX front-month futures (< STOP)."""
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
    vix["vix_z63"] = zscore_past(np.log(vix.close.clip(lower=1.0)), window=WARMUP_Z)

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
        vix[["event_time", "available_at", "vix_close", "vix_z63"]],
        vx_front[["available_at", "vx_price", "vx_oi", "dte"]],
        on="available_at",
        how="inner",
    ).sort_values("available_at")
    merged_vix["vx_term_basis"] = merged_vix["vx_price"] / merged_vix["vix_close"] - 1.0
    merged_vix["vx_basis_z63"] = zscore_past(merged_vix["vx_term_basis"], window=WARMUP_Z)
    merged_vix["vx_basis_change_5d"] = merged_vix["vx_term_basis"].diff(5)
    merged_vix["vx_oi_z63"] = zscore_past(np.log(merged_vix["vx_oi"]), window=WARMUP_Z)
    merged_vix = merged_vix.dropna(
        subset=["vix_z63", "vx_term_basis", "vx_basis_z63", "vx_basis_change_5d", "vx_oi_z63"]
    ).reset_index(drop=True)
    return sp, merged_vix, files


def _load_dated_crypto_basis(
    hist_index: dict[str, Any], root_prefix: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Extract point-in-time active front quarterly COIN-M futures (< STOP)."""
    files: dict[str, str] = {}
    items = [
        d
        for d in hist_index["datasets"]
        if d["market_type"] == "crypto_dated_futures"
        and d["interval"] == "1d"
        and d["symbol"].startswith(root_prefix)
    ]
    frames = []
    for item in items:
        p = Path(item["path"])
        df = pd.read_csv(p)
        df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
        df = df.loc[df.available_at <= STOP].copy()
        if df.empty:
            continue
        files[str(p.resolve())] = sha256(p)
        exp_str = item["symbol"].split("_")[1]
        exp_dt = pd.to_datetime("20" + exp_str, format="%Y%m%d", utc=True) + pd.Timedelta(hours=8)
        df["dte"] = (exp_dt - df["available_at"]).dt.total_seconds() / 86400.0
        df["dated_close"] = df["close"]
        frames.append(df[["timestamp", "available_at", "dte", "dated_close"]])

    all_df = pd.concat(frames, ignore_index=True)
    valid = all_df.loc[all_df.dte >= 3.0].sort_values(["available_at", "dte"])
    front = valid.groupby("available_at", as_index=False).first()
    front["event_time"] = pd.to_datetime(front["timestamp"], utc=True)
    return front[["event_time", "available_at", "dte", "dated_close"]], files


def _forex_sources(symbols: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for symbol in symbols:
        candidates = []
        for path in (ROOT / "multimarket" / "data" / symbol).glob("1d_*.csv.json"):
            meta = read_json(path)
            if (
                meta["instrument"]["provider"] == "fxcm"
                and meta.get("start", "")[:10] == "2020-01-01"
                and meta.get("end_exclusive", "")[:10] == "2025-07-01"
            ):
                candidates.append(path.with_suffix(""))
        if not candidates:
            raise DataError(f"Missing FXCM audited snapshot for {symbol}")
        sources[symbol] = sorted(candidates)[0]
    return sources


def build_datasets(
    *, perturb_after: pd.Timestamp | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Construct causal feature panels for BTCUSDT, ETHUSDT, and EURUSD."""
    hist_index = read_json(ROOT / "historical/validated_20261017.json")
    exp_index = read_json(ROOT / "expansion/datasets/crypto_d4dca621c1adedb8.json")
    sp_df, vix_df, files = _load_tradfi_context(hist_index)
    files[str((ROOT / "historical/validated_20261017.json").resolve())] = sha256(
        ROOT / "historical/validated_20261017.json"
    )
    files[str((ROOT / "expansion/datasets/crypto_d4dca621c1adedb8.json").resolve())] = sha256(
        ROOT / "expansion/datasets/crypto_d4dca621c1adedb8.json"
    )

    if perturb_after is not None:
        sp_df = sp_df.copy()
        vix_df = vix_df.copy()
        sp_df.loc[sp_df.available_at > perturb_after, "sp500_mom_z"] += 50.0
        for col in ["vix_z63", "vx_term_basis", "vx_basis_z63", "vx_basis_change_5d", "vx_oi_z63"]:
            vix_df.loc[vix_df.available_at > perturb_after, col] += 50.0

    fx_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    fx_raw: dict[str, pd.DataFrame] = {}
    for sym, path in _forex_sources(fx_symbols).items():
        df, _ = load_fx_snapshot(path)
        df = df.loc[df.reference_time <= STOP].sort_values("reference_time").reset_index(drop=True)
        if perturb_after is not None:
            mask = df.reference_time > perturb_after
            for col in ["open", "high", "low", "close", "close_bid", "close_ask"]:
                if col in df:
                    df.loc[mask, col] = df.loc[mask, col] * 1.5
        fx_raw[sym] = df
        files[str(path.resolve())] = sha256(path)
        files[str(sidecar(path).resolve())] = sha256(sidecar(path))

    # Build FX momentum event streams for point-in-time USD strength proxy
    fx_events: dict[str, pd.DataFrame] = {}
    for sym, df in fx_raw.items():
        f = df.copy()
        ret = f.close.pct_change(fill_method=None)
        vol = ret.ewm(span=60, adjust=False, min_periods=60).std().clip(lower=1e-5)
        f["fx_mom"] = f.close.pct_change(21, fill_method=None) / (vol * np.sqrt(21.0))
        f["event_time"] = f.reference_time
        f["available_at"] = f.reference_time + pd.Timedelta(seconds=1)
        fx_events[sym] = f.dropna(subset=["fx_mom"])[
            ["event_time", "available_at", "fx_mom"]
        ].reset_index(drop=True)

    # Load 5 crypto assets for broad market breadth/residual and full features on BTC/ETH
    crypto_spots: dict[str, pd.DataFrame] = {}
    crypto_perps: dict[str, pd.DataFrame] = {}
    crypto_fundings: dict[str, pd.DataFrame] = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]:
        sources = exp_index["assets"][sym]
        for kind, store in [
            ("spot", crypto_spots),
            ("perp", crypto_perps),
            ("funding", crypto_fundings),
        ]:
            p = Path(sources[kind])
            df, _ = load_crypto_snapshot(p)
            tcol = "reference_time" if "reference_time" in df else "available_at"
            df = df.loc[df[tcol] <= STOP].reset_index(drop=True)
            if kind == "spot" and "reference_time" not in df:
                df["reference_time"] = df["available_at"]
            if perturb_after is not None:
                mask = df[tcol] > perturb_after
                for col in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "quote_volume",
                    "taker_quote",
                    "funding_rate",
                ]:
                    if col in df:
                        df.loc[mask, col] = df.loc[mask, col] * 1.3 + 0.01
            store[sym] = df
            files[str(p.resolve())] = sha256(p)
            files[str(sidecar(p).resolve())] = sha256(sidecar(p))

    # Align 5-crypto market return across common reference_time
    ref_times = pd.DatetimeIndex(crypto_spots["BTCUSDT"].reference_time)
    ret_1_mat = np.stack(
        [
            crypto_spots[s].close.pct_change(fill_method=None).to_numpy()
            for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        ],
        axis=1,
    )
    market_ret_1 = pd.Series(np.nanmean(ret_1_mat, axis=1), index=ref_times)

    # Pre-join FX USD strength and TradFi context onto crypto reference_time
    usd_components = []
    for sym, ev in sorted(fx_events.items()):
        j = point_in_time_join(ref_times, ev, ["fx_mom"], pd.Timedelta(days=5))
        sign = 1.0 if sym.startswith("USD") else -1.0
        usd_components.append(j["fx_mom"].ffill().fillna(0.0).to_numpy() * sign)
    crypto_usd_strength = np.mean(usd_components, axis=0)

    sp_join_crypto = point_in_time_join(ref_times, sp_df, ["sp500_mom_z"], pd.Timedelta(days=5))
    vix_join_crypto = point_in_time_join(
        ref_times, vix_df, ["vx_term_basis"], pd.Timedelta(days=5)
    )

    panels: dict[str, pd.DataFrame] = {}
    for sym, dated_prefix in [("BTCUSDT", "BTCUSD_"), ("ETHUSDT", "ETHUSD_")]:
        spot = crypto_spots[sym].copy()
        perp = crypto_perps[sym].copy()
        funding = crypto_fundings[sym].copy()

        base = build_features(spot, Features(), include_volume=True)
        base = _volatility_family(base)

        # Family 2: Liquidity
        abs_ret = base.return_1.abs()
        vol_20 = base.return_1.rolling(20, min_periods=20).std().clip(lower=1e-8)
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

        # Family 3: Relative Volume & Order Flow
        base["fin_rvol_quote_z63"] = zscore_past(
            np.log(base.quote_volume.clip(lower=1.0)), window=WARMUP_Z
        )
        imb = 2.0 * base.taker_quote / base.quote_volume.clip(lower=1.0) - 1.0
        base["fin_rvol_taker_imbalance"] = imb
        base["fin_rvol_flow_innovation_3_20"] = (
            imb.rolling(3, min_periods=3).mean()
            - imb.shift(3).rolling(20, min_periods=20).mean()
        )
        prev_qvol_20 = base.quote_volume.shift(1).rolling(20, min_periods=20).mean().clip(lower=1.0)
        base["fin_rvol_signed_volume_shock"] = np.sign(base.return_1) * np.log(
            base.quote_volume.clip(lower=1.0) / prev_qvol_20
        )

        # Family 4: Realized Funding Rates
        settlements = funding.assign(day=funding.available_at.dt.floor("D"))
        daily_f = settlements.groupby("day").funding_rate.agg(["sum", "count"])
        daily_f["fin_fund_1d"] = daily_f["sum"] * 1e4  # in bps for numerical conditioning
        daily_f["fin_fund_7d"] = daily_f["sum"].rolling(7, min_periods=7).sum() * 1e4
        daily_f["fin_fund_z63"] = zscore_past(daily_f["sum"], window=WARMUP_Z)
        daily_f["available_at"] = daily_f.index + pd.Timedelta(days=1)
        daily_f["event_time"] = daily_f["available_at"]
        f_join = point_in_time_join(
            ref_times,
            daily_f.reset_index(drop=True).dropna(
                subset=["fin_fund_1d", "fin_fund_7d", "fin_fund_z63"]
            ),
            ["fin_fund_1d", "fin_fund_7d", "fin_fund_z63"],
            pd.Timedelta(days=1),
        )
        for col in ["fin_fund_1d", "fin_fund_7d", "fin_fund_z63"]:
            base[col] = f_join[col].to_numpy()
        norm_ret_7 = base.return_7 / (vol_20 * np.sqrt(7.0))
        base["fin_fund_mom_interaction"] = -base["fin_fund_z63"] * norm_ret_7

        # Family 5: Futures Basis (Perpetual + Dated Quarterly COIN-M)
        marks = perp.rename(columns={"timestamp": "event_time", "close": "mark_close"})
        m_join = point_in_time_join(
            ref_times,
            marks[["event_time", "available_at", "mark_close"]],
            ["mark_close"],
            pd.Timedelta(seconds=1),
        )
        base["fin_basis_perp_premium"] = (m_join["mark_close"].to_numpy() / base.close - 1.0) * 1e4
        base["fin_basis_perp_z63"] = zscore_past(base["fin_basis_perp_premium"], window=WARMUP_Z)

        dated_ev, dated_files = _load_dated_crypto_basis(hist_index, dated_prefix)
        files.update(dated_files)
        if perturb_after is not None:
            dated_ev = dated_ev.copy()
            dated_ev.loc[dated_ev.available_at > perturb_after, "dated_close"] *= 1.4
        d_join = point_in_time_join(
            ref_times,
            dated_ev,
            ["dated_close", "dte"],
            pd.Timedelta(days=5),
        )
        raw_dated_basis = d_join["dated_close"].to_numpy() / base.close - 1.0
        dte_arr = d_join["dte"].clip(lower=3.0).to_numpy()
        base["fin_basis_dated_ann"] = raw_dated_basis * (365.0 / dte_arr)
        base["fin_basis_dated_change_5d"] = base["fin_basis_dated_ann"].diff(5)

        # Family 6: Cross-Asset Relationships (Crypto market residual + FX USD + SP500 + VIX/VX)
        mkt_r = pd.Series(market_ret_1.to_numpy(), index=base.index)
        cov_60 = base.return_1.shift(1).rolling(60, min_periods=60).cov(mkt_r.shift(1))
        var_60 = mkt_r.shift(1).rolling(60, min_periods=60).var().clip(lower=1e-12)
        beta_prev = cov_60 / var_60
        resid_1 = (base.return_1 - beta_prev * mkt_r) / vol_20
        base["fin_cross_crypto_residual"] = resid_1.rolling(5, min_periods=5).mean()
        base["fin_cross_usd_strength"] = crypto_usd_strength
        base["fin_cross_sp500_mom_z"] = sp_join_crypto["sp500_mom_z"].ffill().to_numpy()
        base["fin_cross_vix_term_basis"] = vix_join_crypto["vx_term_basis"].ffill().to_numpy()

        # Target (1-step ahead close direction)
        base["target"] = (
            (base.close.shift(-1) > base.close).astype(float).where(base.close.shift(-1).notna())
        )
        base["target_time"] = base.reference_time.shift(-1)
        panels[sym] = base

    # Build EURUSD D1 panel with historically available financial families
    eur_raw = fx_raw["EURUSD"].copy()
    eur = build_features(eur_raw, Features(), include_volume=False)
    for col in [
        "open_bid",
        "high_bid",
        "low_bid",
        "close_bid",
        "open_ask",
        "high_ask",
        "low_ask",
        "close_ask",
        "crossed_close",
    ]:
        eur[col] = eur_raw[col].to_numpy()
    eur = _volatility_family(eur)
    vol_20_fx = eur.return_1.rolling(20, min_periods=20).std().clip(lower=1e-8)

    # FX Liquidity family
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

    # FX Futures Basis family (CBOE VX VIX futures term structure at point-in-time)
    eur_refs = pd.DatetimeIndex(eur.reference_time)
    vix_fx = point_in_time_join(
        eur_refs,
        vix_df,
        ["vix_z63", "vx_term_basis", "vx_basis_z63", "vx_basis_change_5d", "vx_oi_z63"],
        pd.Timedelta(days=5),
    )
    eur["fin_basis_vx_term"] = vix_fx["vx_term_basis"].ffill().to_numpy()
    eur["fin_basis_vx_z63"] = vix_fx["vx_basis_z63"].ffill().to_numpy()
    eur["fin_basis_vx_change_5d"] = vix_fx["vx_basis_change_5d"].ffill().to_numpy()
    eur["fin_basis_vx_oi_z63"] = vix_fx["vx_oi_z63"].ffill().to_numpy()

    # FX Cross-Asset family (Peer USD strength excluding EURUSD, FX dispersion, SP500, VIX)
    peer_strengths = []
    for sym, ev in sorted(fx_events.items()):
        if sym == "EURUSD":
            continue
        j = point_in_time_join(eur_refs, ev, ["fx_mom"], pd.Timedelta(days=5))
        sign = 1.0 if sym.startswith("USD") else -1.0
        peer_strengths.append(j["fx_mom"].ffill().fillna(0.0).to_numpy() * sign)
    peer_mat = np.stack(peer_strengths, axis=1)
    eur["fin_cross_usd_peers"] = peer_mat.mean(axis=1)
    eur["fin_cross_fx_dispersion"] = peer_mat.std(axis=1)
    sp_fx = point_in_time_join(eur_refs, sp_df, ["sp500_mom_z"], pd.Timedelta(days=5))
    eur["fin_cross_sp500_mom_z"] = sp_fx["sp500_mom_z"].ffill().to_numpy()
    eur["fin_cross_vix_z63"] = vix_fx["vix_z63"].ffill().to_numpy()

    eur["target"] = (
        (eur.close.shift(-1) > eur.close).astype(float).where(eur.close.shift(-1).notna())
    )
    eur["target_time"] = eur.reference_time.shift(-1)
    panels["EURUSD"] = eur

    return panels, files


def attach_noise_columns(
    panels: dict[str, pd.DataFrame], families_by_asset: dict[str, dict[str, list[str]]]
) -> dict[str, dict[str, list[str]]]:
    """Attach deterministic causal Gaussian noise columns matched to family dimensions."""
    noise_map: dict[str, dict[str, list[str]]] = {}
    for asset_idx, (symbol, frame) in enumerate(panels.items()):
        families = families_by_asset[symbol]
        n_fam = len(families)
        k_single = 4
        k_all = n_fam * k_single
        k_lofo = (n_fam - 1) * k_single
        asset_noise: dict[str, list[str]] = {}

        for stream_idx in [1, 2, 3]:
            rng = np.random.default_rng(20261000 + asset_idx * 100 + stream_idx)
            cols = [f"noise_add4_s{stream_idx}_{j}" for j in range(k_single)]
            vals = rng.normal(0.0, 1.0, size=(len(frame), k_single))
            for j, c in enumerate(cols):
                frame[c] = vals[:, j]
            asset_noise[f"noise_add4_s{stream_idx}"] = cols

        for stream_idx in [1, 2]:
            rng = np.random.default_rng(20262000 + asset_idx * 100 + stream_idx)
            cols = [f"noise_all_s{stream_idx}_{j}" for j in range(k_all)]
            vals = rng.normal(0.0, 1.0, size=(len(frame), k_all))
            for j, c in enumerate(cols):
                frame[c] = vals[:, j]
            asset_noise[f"noise_all_s{stream_idx}"] = cols

        rng_lofo = np.random.default_rng(20263000 + asset_idx * 100 + 1)
        cols_lofo = [f"noise_lofo_s1_{j}" for j in range(k_lofo)]
        vals_lofo = rng_lofo.normal(0.0, 1.0, size=(len(frame), k_lofo))
        for j, c in enumerate(cols_lofo):
            frame[c] = vals_lofo[:, j]
        asset_noise["noise_lofo_s1"] = cols_lofo

        noise_map[symbol] = asset_noise
    return noise_map


def verify_causal_invariance() -> dict[str, Any]:
    """Verify that perturbing future observations (> 2024-01-01) leaves past features unchanged."""
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    clean_panels, _ = build_datasets(perturb_after=None)
    pert_panels, _ = build_datasets(perturb_after=cutoff)
    report: dict[str, Any] = {"cutoff": cutoff.isoformat(), "assets": {}}
    for symbol in ["BTCUSDT", "ETHUSDT", "EURUSD"]:
        fams = CRYPTO_FAMILIES if symbol != "EURUSD" else FOREX_FAMILIES
        base_cols = CRYPTO_BASE_COLUMNS if symbol != "EURUSD" else FX_BASE_COLUMNS
        cols = base_cols + [c for g in fams.values() for c in g]
        c_past = clean_panels[symbol].loc[clean_panels[symbol].reference_time <= cutoff, cols]
        p_past = pert_panels[symbol].loc[pert_panels[symbol].reference_time <= cutoff, cols]
        diff = (c_past.to_numpy(dtype=float) - p_past.to_numpy(dtype=float))
        finite_mask = np.isfinite(c_past.to_numpy(dtype=float))
        max_diff = float(np.max(np.abs(diff[finite_mask]))) if finite_mask.any() else 0.0
        if max_diff > 1e-12:
            raise LeakageError(f"Causal invariance violated in {symbol}: max_diff={max_diff}")
        report["assets"][symbol] = {
            "checked_rows": int(len(c_past)),
            "checked_columns": len(cols),
            "max_abs_diff": max_diff,
            "passed": True,
        }
    return report


def build_recipes_for_asset(
    symbol: str, noise_cols: dict[str, list[str]]
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    base = CRYPTO_BASE_COLUMNS.copy() if symbol != "EURUSD" else FX_BASE_COLUMNS.copy()
    families = CRYPTO_FAMILIES if symbol != "EURUSD" else FOREX_FAMILIES
    all_fin = [c for g in families.values() for c in g]
    recipes: dict[str, list[str]] = {
        "F0_base": base,
        **{f"add_{fam}": base + cols for fam, cols in families.items()},
        "noise_add4_s1": base + noise_cols["noise_add4_s1"],
        "noise_add4_s2": base + noise_cols["noise_add4_s2"],
        "noise_add4_s3": base + noise_cols["noise_add4_s3"],
        "all_financial": base + all_fin,
        "noise_all_s1": base + noise_cols["noise_all_s1"],
        "noise_all_s2": base + noise_cols["noise_all_s2"],
        **{
            f"all_without_{fam}": base + [c for c in all_fin if c not in cols]
            for fam, cols in families.items()
        },
        "noise_lofo_s1": base + noise_cols["noise_lofo_s1"],
    }
    return base, families, recipes


def fit_and_predict(
    model_kind: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    seed: int,
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
        ref = fit_reference_models(train, columns, seed=seed, C=1.0)
        p_tr = ref.probabilities(train)["logistic_l2"]
        p_ev = ref.probabilities(evaluation)["logistic_l2"]
        return ref, p_tr, p_ev

    if model_kind in {"random_forest", "lightgbm"}:
        model = create_estimator({"kind": model_kind})
        model.set_params(random_state=seed)
        model.fit(x_tr, y_tr)
        p_tr = model.predict_proba(x_tr)[:, 1]
        p_ev = model.predict_proba(x_ev)[:, 1]
        return model, p_tr, p_ev

    raise DataError(f"Unsupported model_kind: {model_kind}")


def select_nested_noise_screened(
    frame: pd.DataFrame,
    outer_start: pd.Timestamp,
    base_cols: list[str],
    families: dict[str, list[str]],
    noise_cols: dict[str, list[str]],
    model_kind: str,
    seed: int,
) -> dict[str, Any]:
    """Select families that beat BOTH F0_base and matched noise_add4 in inner validation."""
    end = outer_start - pd.Timedelta(days=EMBARGO_DAYS)
    bounds = [end - pd.Timedelta(days=i * INNER_DAYS) for i in [2, 1, 0]]
    candidates = {
        "F0_base": base_cols,
        **{f"add_{fam}": base_cols + cols for fam, cols in families.items()},
        "noise_add4_s1": base_cols + noise_cols["noise_add4_s1"],
        "noise_add4_s2": base_cols + noise_cols["noise_add4_s2"],
        "noise_add4_s3": base_cols + noise_cols["noise_add4_s3"],
    }
    losses: dict[str, list[float]] = {k: [] for k in candidates}
    aucs: dict[str, list[float]] = {k: [] for k in candidates}

    for start_w, end_w in zip(bounds[:-1], bounds[1:], strict=True):
        in_train = frame.loc[
            (frame.reference_time < start_w)
            & (frame.target_time < start_w - pd.Timedelta(days=EMBARGO_DAYS))
        ]
        in_val = frame.loc[
            (frame.reference_time >= start_w)
            & (frame.reference_time < end_w)
            & (frame.target_time < end_w)
        ]
        if len(in_train) < 120 or len(in_val) < 30:
            return {
                "selected_families": [],
                "columns": base_cols,
                "fallback_base": True,
                "reason": "insufficient_inner_rows",
            }
        for name, cols in candidates.items():
            _, _, p_val = fit_and_predict(model_kind, in_train, in_val, cols, seed)
            y_val = in_val.target.to_numpy(dtype=int)
            losses[name].append(float(binary_loss(y_val, p_val).mean()))
            aucs[name].append(float(_auc(y_val, p_val)))

    base_loss = np.asarray(losses["F0_base"])
    noise_loss = np.mean(
        [losses["noise_add4_s1"], losses["noise_add4_s2"], losses["noise_add4_s3"]], axis=0
    )
    base_auc = np.asarray(aucs["F0_base"])
    noise_auc = np.mean(
        [aucs["noise_add4_s1"], aucs["noise_add4_s2"], aucs["noise_add4_s3"]], axis=0
    )

    qualifying_fams: list[str] = []
    for fam in families:
        key = f"add_{fam}"
        l_arr = np.asarray(losses[key])
        a_arr = np.asarray(aucs[key])
        beats_loss = bool(np.all(l_arr < np.minimum(base_loss, noise_loss)))
        beats_auc = bool(np.mean(a_arr) > max(float(np.mean(base_auc)), float(np.mean(noise_auc))))
        if beats_loss and beats_auc:
            qualifying_fams.append(fam)

    qualifying_fams.sort(key=lambda f: float(np.mean(losses[f"add_{f}"])))
    selected = qualifying_fams[:2]
    if not selected:
        return {
            "selected_families": [],
            "columns": base_cols,
            "fallback_base": True,
            "inner_losses": losses,
            "inner_aucs": aucs,
        }
    chosen_cols = base_cols + [c for fam in selected for c in families[fam]]
    return {
        "selected_families": selected,
        "columns": chosen_cols,
        "fallback_base": False,
        "inner_losses": losses,
        "inner_aucs": aucs,
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


def bootstrap_ablation_contrasts(
    target: np.ndarray,
    folds: np.ndarray,
    probs: dict[str, np.ndarray],
    families: list[str],
    *,
    samples: int = 2000,
    blocks: tuple[int, ...] = (7, 30),
    seed: int = 42,
) -> dict[str, Any]:
    """Paired moving-block bootstrap against F0_base AND matched-dimension noise."""
    y = np.asarray(target, dtype=int)
    p_dict = {k: np.asarray(v, dtype=float) for k, v in probs.items()}

    # Construct pooled noise probability averages for exact matched-dimension reference
    p_dict["noise_add4_mean"] = np.mean(
        [p_dict["noise_add4_s1"], p_dict["noise_add4_s2"], p_dict["noise_add4_s3"]], axis=0
    )
    p_dict["noise_all_mean"] = np.mean(
        [p_dict["noise_all_s1"], p_dict["noise_all_s2"]], axis=0
    )

    point_metrics = {}
    point_losses = {}
    for name, p in p_dict.items():
        m = classification_metrics(y, p)
        m["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= 0.5))
        point_metrics[name] = m
        point_losses[name] = binary_loss(y, p)

    # Contrasts of interest: (candidate, reference)
    contrasts: list[tuple[str, str]] = []
    for fam in families:
        contrasts.append((f"add_{fam}", "F0_base"))
        contrasts.append((f"add_{fam}", "noise_add4_mean"))
        contrasts.append((f"all_without_{fam}", "all_financial"))
    contrasts.extend(
        [
            ("all_financial", "F0_base"),
            ("all_financial", "noise_all_mean"),
            ("nested_noise_screened", "F0_base"),
            ("nested_noise_screened", "noise_add4_mean"),
            ("noise_add4_mean", "F0_base"),
            ("noise_all_mean", "F0_base"),
        ]
    )

    by_block: dict[str, Any] = {}
    for block in blocks:
        idx = moving_block_indices(folds, samples, block, seed + block)
        loss_draws = {name: point_losses[name][idx].mean(axis=1) for name in p_dict}
        auc_draws = {
            name: np.array([_auc(y[row], p_dict[name][row]) for row in idx]) for name in p_dict
        }
        block_contrasts: dict[str, Any] = {}
        raw_p_loss: dict[str, float] = {}
        raw_p_auc: dict[str, float] = {}
        for cand, ref in contrasts:
            key = f"{cand}__vs__{ref}"
            d_loss = loss_draws[cand] - loss_draws[ref]
            d_auc = auc_draws[cand] - auc_draws[ref]
            pt_loss = point_metrics[cand]["log_loss"] - point_metrics[ref]["log_loss"]
            pt_auc = point_metrics[cand]["roc_auc"] - point_metrics[ref]["roc_auc"]

            # Two-sided bootstrap p-value centered at 0
            p_loss = float(2.0 * min(np.mean(d_loss >= 0.0), np.mean(d_loss <= 0.0)))
            p_auc = float(2.0 * min(np.mean(d_auc >= 0.0), np.mean(d_auc <= 0.0)))
            # One-sided p-value for genuine improvement (d_loss < 0, d_auc > 0)
            p_loss_improve = float(np.mean(d_loss >= 0.0))
            p_auc_improve = float(np.mean(d_auc <= 0.0))

            if cand.startswith("add_") or cand in {"all_financial", "nested_noise_screened"}:
                raw_p_loss[key] = p_loss_improve
                raw_p_auc[key] = p_auc_improve

            block_contrasts[key] = {
                "delta_log_loss": float(pt_loss),
                "delta_log_loss_ci95": [
                    float(np.quantile(d_loss, 0.025)),
                    float(np.quantile(d_loss, 0.975)),
                ],
                "p_two_sided_loss": min(1.0, p_loss),
                "p_one_sided_loss_improve": p_loss_improve,
                "delta_roc_auc": float(pt_auc),
                "delta_roc_auc_ci95": [
                    float(np.quantile(d_auc, 0.025)),
                    float(np.quantile(d_auc, 0.975)),
                ],
                "p_two_sided_auc": min(1.0, p_auc),
                "p_one_sided_auc_improve": p_auc_improve,
            }

        holm_loss = holm_bonferroni(raw_p_loss)
        holm_auc = holm_bonferroni(raw_p_auc)
        for key, val in holm_loss.items():
            block_contrasts[key]["p_holm_loss_improve"] = val
        for key, val in holm_auc.items():
            block_contrasts[key]["p_holm_auc_improve"] = val

        by_block[str(block)] = block_contrasts

    return {
        "metrics": point_metrics,
        "blocks": by_block,
    }


def run_ablation_study(output_root: Path | None = None) -> dict[str, Any]:
    protected_before = _protected_snapshot()
    causality_check = verify_causal_invariance()
    panels, input_files = build_datasets(perturb_after=None)
    noise_map = attach_noise_columns(
        panels,
        {
            "BTCUSDT": CRYPTO_FAMILIES,
            "ETHUSDT": CRYPTO_FAMILIES,
            "EURUSD": FOREX_FAMILIES,
        },
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    out_dir = (
        output_root.resolve()
        if output_root is not None
        else (ROOT / "financial_feature_ablation" / run_id).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    models_to_eval = [
        ("xgboost", [42, 1729]),
        ("logistic_l2", [42]),
        ("random_forest", [42, 1729]),
        ("lightgbm", [42]),
    ]

    plan = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "stop_quarantine": STOP.isoformat(),
        "boundaries": [b.isoformat() for b in BOUNDS],
        "embargo_days": EMBARGO_DAYS,
        "inner_days": INNER_DAYS,
        "warmup_z": WARMUP_Z,
        "crypto_families": CRYPTO_FAMILIES,
        "forex_families": FOREX_FAMILIES,
        "models": [{"kind": m, "seeds": s} for m, s in models_to_eval],
        "causality_invariance_check": causality_check,
        "inputs": input_files,
        "protected_before": protected_before,
        "environment": environment(),
        "code": code_provenance(),
    }
    write_new_json(out_dir / "plan.json", plan)

    all_predictions: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    inner_selections: list[dict[str, Any]] = []
    permutation_records: list[dict[str, Any]] = []
    diagnostics_records: list[dict[str, Any]] = []
    data_quality: dict[str, Any] = {}

    for symbol, full in panels.items():
        base_cols, families, fixed_recipes = build_recipes_for_asset(symbol, noise_map[symbol])
        all_cols = list(
            dict.fromkeys(c for cols in fixed_recipes.values() for c in cols)
        )
        eligible = full[all_cols].notna().all(axis=1) & full.target.notna()
        frame = full.loc[eligible].reset_index(drop=True)
        protect_reserved_labels(ROOT, frame)

        data_quality[symbol] = {
            "total_rows_before_stop": int(len(full)),
            "eligible_common_rows": int(len(frame)),
            "first_reference_time": frame.reference_time.min().isoformat(),
            "last_reference_time": frame.reference_time.max().isoformat(),
            "last_target_time": frame.target_time.max().isoformat(),
            "base_feature_count": len(base_cols),
            "financial_families": {k: len(v) for k, v in families.items()},
            "same_observations_across_all_recipes_and_models": True,
        }

        for fold, (start, end) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:], strict=True)):
            train = frame.loc[
                (frame.reference_time < start)
                & (frame.target_time < start - pd.Timedelta(days=EMBARGO_DAYS))
            ].copy()
            evaluation = frame.loc[
                (frame.reference_time >= start)
                & (frame.reference_time < end)
                & (frame.target_time < end)
            ].copy()
            protect_reserved_labels(ROOT, train, evaluation)
            if len(train) < 180 or len(evaluation) < 60:
                raise DataError(f"Insufficient split for {symbol} fold {fold}")

            fin_cols = [c for g in families.values() for c in g]
            diagnostics_records.append(
                {
                    "symbol": symbol,
                    "fold": fold,
                    "train_rows": int(len(train)),
                    "evaluation_rows": int(len(evaluation)),
                    "train_start": train.reference_time.min().isoformat(),
                    "train_last_target": train.target_time.max().isoformat(),
                    "evaluation_start": evaluation.reference_time.min().isoformat(),
                    "evaluation_last_target": evaluation.target_time.max().isoformat(),
                    "train_hash": frame_hash(train[["reference_time", "target_time", "target"]]),
                    "evaluation_hash": frame_hash(
                        evaluation[["reference_time", "target_time", "target"]]
                    ),
                    "high_correlations_train_85": correlation_pairs(
                        train, base_cols + fin_cols, threshold=0.85
                    ),
                }
            )

            # Reference constant/prevalence probabilities
            for ref_name, ref_p in [
                ("constant_05", np.full(len(evaluation), 0.5)),
                ("train_prevalence", np.full(len(evaluation), float(train.target.mean()))),
            ]:
                rows = evaluation[["reference_time", "target_time", "target"]].copy()
                rows["symbol"] = symbol
                rows["fold"] = fold
                rows["model"] = "baseline"
                rows["seed"] = 42
                rows["recipe"] = ref_name
                rows["probability"] = ref_p
                all_predictions.append(rows)

            for model_kind, seeds in models_to_eval:
                for seed in seeds:
                    sel = select_nested_noise_screened(
                        frame,
                        start,
                        base_cols,
                        families,
                        noise_map[symbol],
                        model_kind,
                        seed,
                    )
                    inner_selections.append(
                        {
                            "symbol": symbol,
                            "fold": fold,
                            "model": model_kind,
                            "seed": seed,
                            "selected_families": sel["selected_families"],
                            "fallback_base": sel["fallback_base"],
                            "column_count": len(sel["columns"]),
                        }
                    )
                    recipes = fixed_recipes | {"nested_noise_screened": sel["columns"]}

                    for recipe_name, cols in recipes.items():
                        fitted, p_tr, p_ev = fit_and_predict(
                            model_kind, train, evaluation, cols, seed
                        )
                        tr_m = classification_metrics(train.target.to_numpy(dtype=int), p_tr)
                        ev_m = classification_metrics(evaluation.target.to_numpy(dtype=int), p_ev)
                        ev_m["balanced_accuracy"] = float(
                            balanced_accuracy_score(
                                evaluation.target.to_numpy(dtype=int), p_ev >= 0.5
                            )
                        )
                        fit_records.append(
                            {
                                "symbol": symbol,
                                "fold": fold,
                                "model": model_kind,
                                "seed": seed,
                                "recipe": recipe_name,
                                "feature_count": len(cols),
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
                        rows = evaluation[["reference_time", "target_time", "target"]].copy()
                        rows["symbol"] = symbol
                        rows["fold"] = fold
                        rows["model"] = model_kind
                        rows["seed"] = seed
                        rows["recipe"] = recipe_name
                        rows["probability"] = p_ev
                        all_predictions.append(rows)

                        # Block permutation diagnostic on all_financial for primary seed 42
                        if (
                            seed == 42
                            and recipe_name == "all_financial"
                            and model_kind in {"xgboost", "random_forest", "lightgbm"}
                        ):
                            groups = {f"family_{k}": v for k, v in families.items()} | {
                                c: [c] for c in fin_cols
                            }
                            # Wrap dataframe-expecting estimators for permutation_diagnostic
                            for perm_row in permutation_diagnostic(
                                _NumpyModelAdapter(fitted, cols),
                                evaluation,
                                cols,
                                groups,
                                blocks=(7, 30),
                                repeats=8,
                                seed=42,
                            ):
                                permutation_records.append(
                                    {
                                        "symbol": symbol,
                                        "fold": fold,
                                        "model": model_kind,
                                        "recipe": recipe_name,
                                        "name": perm_row["name"],
                                        "block": perm_row["block"],
                                        "delta_loss_mean": perm_row["delta_loss_mean"],
                                        "permutation_std": perm_row["permutation_std"],
                                    }
                                )
            print(
                f"COMPLETED {symbol} fold={fold} train={len(train)} oos={len(evaluation)}",
                flush=True,
            )

    pred_df = pd.concat(all_predictions, ignore_index=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    write_new_json(out_dir / "fits.json", fit_records)
    write_new_json(out_dir / "inner_selections.json", inner_selections)
    write_new_json(out_dir / "permutations.json", permutation_records)
    write_new_json(out_dir / "diagnostics.json", diagnostics_records)

    # Compute bootstrap ablation contrasts per (symbol, model, seed)
    statistical_results: dict[str, Any] = {}
    family_summary_rows: list[dict[str, Any]] = []

    for (symbol, model_kind, seed), grp in pred_df.groupby(["symbol", "model", "seed"]):
        if model_kind == "baseline":
            continue
        fams = list((CRYPTO_FAMILIES if symbol != "EURUSD" else FOREX_FAMILIES).keys())
        wide = grp.pivot(
            index=["fold", "reference_time", "target_time", "target"],
            columns="recipe",
            values="probability",
        ).sort_index()
        y = wide.index.get_level_values("target").to_numpy(dtype=int)
        folds = wide.index.get_level_values("fold").to_numpy(dtype=int)
        probs = {col: wide[col].to_numpy(dtype=float) for col in wide.columns}

        stats = bootstrap_ablation_contrasts(
            y, folds, probs, fams, samples=2000, blocks=(7, 30), seed=int(seed)
        )
        statistical_results.setdefault(symbol, {}).setdefault(model_kind, {})[str(seed)] = stats

        # Fold-level consistency for each family
        for fam in fams:
            add_key = f"add_{fam}"
            lofo_key = f"all_without_{fam}"
            noise4_p = np.mean(
                [probs["noise_add4_s1"], probs["noise_add4_s2"], probs["noise_add4_s3"]], axis=0
            )
            folds_beat_base_both = 0
            folds_beat_noise_both = 0
            folds_lofo_hurt_both = 0
            for f_id in np.unique(folds):
                m_f = folds == f_id
                y_f = y[m_f]
                l_base = float(binary_loss(y_f, probs["F0_base"][m_f]).mean())
                l_add = float(binary_loss(y_f, probs[add_key][m_f]).mean())
                l_noise = float(binary_loss(y_f, noise4_p[m_f]).mean())
                l_all = float(binary_loss(y_f, probs["all_financial"][m_f]).mean())
                l_lofo = float(binary_loss(y_f, probs[lofo_key][m_f]).mean())

                a_base = float(_auc(y_f, probs["F0_base"][m_f]))
                a_add = float(_auc(y_f, probs[add_key][m_f]))
                a_noise = float(_auc(y_f, noise4_p[m_f]))
                a_all = float(_auc(y_f, probs["all_financial"][m_f]))
                a_lofo = float(_auc(y_f, probs[lofo_key][m_f]))

                if l_add < l_base and a_add > a_base:
                    folds_beat_base_both += 1
                if l_add < l_noise and a_add > a_noise:
                    folds_beat_noise_both += 1
                if l_lofo > l_all and a_lofo < a_all:
                    folds_lofo_hurt_both += 1

            b30 = stats["blocks"]["30"]
            vs_base = b30[f"{add_key}__vs__F0_base"]
            vs_noise = b30[f"{add_key}__vs__noise_add4_mean"]
            vs_lofo = b30[f"{lofo_key}__vs__all_financial"]

            family_summary_rows.append(
                {
                    "symbol": symbol,
                    "model": model_kind,
                    "seed": int(seed),
                    "family": fam,
                    "oos_log_loss": stats["metrics"][add_key]["log_loss"],
                    "oos_roc_auc": stats["metrics"][add_key]["roc_auc"],
                    "base_log_loss": stats["metrics"]["F0_base"]["log_loss"],
                    "base_roc_auc": stats["metrics"]["F0_base"]["roc_auc"],
                    "noise4_log_loss": stats["metrics"]["noise_add4_mean"]["log_loss"],
                    "noise4_roc_auc": stats["metrics"]["noise_add4_mean"]["roc_auc"],
                    "delta_loss_vs_base": vs_base["delta_log_loss"],
                    "delta_loss_vs_base_ci95_b30": vs_base["delta_log_loss_ci95"],
                    "p_holm_loss_vs_base_b30": vs_base.get("p_holm_loss_improve"),
                    "delta_auc_vs_base": vs_base["delta_roc_auc"],
                    "delta_auc_vs_base_ci95_b30": vs_base["delta_roc_auc_ci95"],
                    "p_holm_auc_vs_base_b30": vs_base.get("p_holm_auc_improve"),
                    "delta_loss_vs_noise4": vs_noise["delta_log_loss"],
                    "delta_loss_vs_noise4_ci95_b30": vs_noise["delta_log_loss_ci95"],
                    "p_holm_loss_vs_noise4_b30": vs_noise.get("p_holm_loss_improve"),
                    "delta_auc_vs_noise4": vs_noise["delta_roc_auc"],
                    "delta_auc_vs_noise4_ci95_b30": vs_noise["delta_roc_auc_ci95"],
                    "p_holm_auc_vs_noise4_b30": vs_noise.get("p_holm_auc_improve"),
                    "lofo_remove_minus_all_loss": vs_lofo["delta_log_loss"],
                    "lofo_remove_minus_all_auc": vs_lofo["delta_roc_auc"],
                    "folds_beat_base_loss_and_auc": folds_beat_base_both,
                    "folds_beat_noise_loss_and_auc": folds_beat_noise_both,
                    "folds_lofo_removal_hurts_both": folds_lofo_hurt_both,
                    "total_folds": int(len(np.unique(folds))),
                }
            )

    if protected_before != _protected_snapshot():
        raise LeakageError("Protected models or prospective reserves modified during ablation run")

    summary = {
        "run_id": run_id,
        "directory": str(out_dir),
        "data_quality": data_quality,
        "causality_invariance": causality_check,
        "protected_unchanged": True,
        "family_ablation_summary": family_summary_rows,
        "statistical_results": statistical_results,
    }
    write_new_json(out_dir / "summary.json", summary)
    pd.DataFrame(family_summary_rows).to_csv(out_dir / "family_ablation_table.csv", index=False)
    print(f"ABLATION STUDY COMPLETED: {out_dir}", flush=True)
    return summary


class _NumpyModelAdapter:
    """Adapter so permutation_diagnostic (which passes numpy arrays) works with DataFrame estimators."""

    def __init__(self, estimator: Any, columns: list[str]) -> None:
        self.estimator = estimator
        self.columns = columns

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if isinstance(self.estimator, XGBClassifier):
            return self.estimator.predict_proba(x)
        df = pd.DataFrame(x, columns=self.columns)
        return self.estimator.predict_proba(df)


def analyze_and_write_report(run_dir: Path) -> None:
    """Generate markdown report and SHA-256 manifest from a completed ablation study."""
    run_dir = run_dir.resolve()
    summary = read_json(run_dir / "summary.json")
    fits = read_json(run_dir / "fits.json")
    perms = read_json(run_dir / "permutations.json")
    selections = read_json(run_dir / "inner_selections.json")
    preds = pd.read_csv(run_dir / "predictions.csv")

    # Compute baseline constant_05 and train_prevalence OOS metrics per symbol
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for sym, grp in preds.loc[preds.model == "baseline"].groupby("symbol"):
        baseline_metrics[sym] = {}
        for r_name, r_grp in grp.groupby("recipe"):
            y = r_grp.target.to_numpy(dtype=int)
            p = r_grp.probability.to_numpy(dtype=float)
            m = classification_metrics(y, p)
            m["balanced_accuracy"] = float(balanced_accuracy_score(y, p >= 0.5))
            baseline_metrics[sym][r_name] = m

    # Mean generalization gap and feature count per (symbol, model, seed, recipe)
    fits_df = pd.DataFrame(fits)
    gap_map = (
        fits_df.groupby(["symbol", "model", "seed", "recipe"])[
            ["feature_count", "generalization_gap_loss", "train_log_loss", "train_roc_auc"]
        ]
        .mean()
        .to_dict(orient="index")
    )

    lines = [
        "# Estudio Controlado de Ablación de Variables Financieras",
        "",
        f"- **Run ID:** `{summary['run_id']}`",
        f"- **Directorio:** `{run_dir}`",
        f"- **Modelos protegidos y reservas intactas:** `{summary['protected_unchanged']}`",
        "",
        "## 1. Verificación de Causalidad e Integridad de Observaciones",
        "",
        "| Activo | Filas Totales (<2025-07-01) | Filas Elegibles Comunes | Inicio OOS | Fin OOS | Folds | Test Invariancia Causal (max diff) |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for sym, dq in summary["data_quality"].items():
        ci = summary["causality_invariance"]["assets"][sym]
        lines.append(
            f"| {sym} | {dq['total_rows_before_stop']} | {dq['eligible_common_rows']} | "
            f"`2023-01-01` | `{dq['last_target_time'][:10]}` | 3 | `{ci['max_abs_diff']:.1e}` (PASS) |"
        )

    lines += [
        "",
        "## 2. Comparación Global de Recetas y Controles de Ruido (Semilla Principal 42)",
        "",
        "| Activo | Modelo | Receta | Cols | Log-Loss OOS ↓ | ΔLoss vs Base | ΔLoss vs Ruido | ROC-AUC OOS ↑ | ΔAUC vs Base | ΔAUC vs Ruido | Brier ↓ | Acc ↑ | Bal.Acc ↑ | Brecha Train-OOS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sym in ["BTCUSDT", "ETHUSDT", "EURUSD"]:
        for r_name, bm in baseline_metrics[sym].items():
            lines.append(
                f"| {sym} | baseline | `{r_name}` | 0 | {bm['log_loss']:.5f} | — | — | "
                f"{bm['roc_auc']:.5f} | — | — | {bm['brier_score']:.5f} | "
                f"{bm['accuracy']:.4f} | {bm['balanced_accuracy']:.4f} | — |"
            )
        for mod in ["logistic_l2", "random_forest", "xgboost", "lightgbm"]:
            st = summary["statistical_results"][sym][mod]["42"]
            metrics = st["metrics"]
            base_l = metrics["F0_base"]["log_loss"]
            base_a = metrics["F0_base"]["roc_auc"]
            n4_l = metrics["noise_add4_mean"]["log_loss"]
            n4_a = metrics["noise_add4_mean"]["roc_auc"]
            nall_l = metrics["noise_all_mean"]["log_loss"]
            nall_a = metrics["noise_all_mean"]["roc_auc"]
            nlofo_l = metrics["noise_lofo_s1"]["log_loss"]
            nlofo_a = metrics["noise_lofo_s1"]["roc_auc"]

            fams = list((CRYPTO_FAMILIES if sym != "EURUSD" else FOREX_FAMILIES).keys())
            order = (
                ["F0_base", "noise_add4_mean"]
                + [f"add_{f}" for f in fams]
                + ["all_financial", "noise_all_mean"]
                + [f"all_without_{f}" for f in fams]
                + ["noise_lofo_s1", "nested_noise_screened"]
            )
            for r_name in order:
                m = metrics[r_name]
                if r_name.startswith("add_") or r_name == "nested_noise_screened":
                    ref_nl, ref_na = n4_l, n4_a
                elif r_name == "all_financial":
                    ref_nl, ref_na = nall_l, nall_a
                elif r_name.startswith("all_without_"):
                    ref_nl, ref_na = nlofo_l, nlofo_a
                else:
                    ref_nl, ref_na = None, None

                g_info = gap_map.get((sym, mod, 42, r_name), {})
                cols_str = f"{g_info['feature_count']:.1f}" if "feature_count" in g_info else "—"
                gap_str = (
                    f"{g_info['generalization_gap_loss']:+.5f}"
                    if "generalization_gap_loss" in g_info
                    else "—"
                )
                dl_noise = f"{m['log_loss'] - ref_nl:+.5f}" if ref_nl is not None else "—"
                da_noise = f"{m['roc_auc'] - ref_na:+.5f}" if ref_na is not None else "—"
                lines.append(
                    f"| {sym} | {mod} | `{r_name}` | {cols_str} | {m['log_loss']:.5f} | "
                    f"{m['log_loss'] - base_l:+.5f} | {dl_noise} | {m['roc_auc']:.5f} | "
                    f"{m['roc_auc'] - base_a:+.5f} | {da_noise} | {m['brier_score']:.5f} | "
                    f"{m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {gap_str} |"
                )

    lines += [
        "",
        "## 3. Ablación por Familia: Adición Individual (Forward) y Eliminación desde Completo (LOFO)",
        "",
        "| Activo | Modelo | Semilla | Familia | ΔLoss (Add−Base) | IC95% ΔLoss (B30) | ΔLoss (Add−Ruido4) | ΔAUC (Add−Base) | ΔAUC (Add−Ruido4) | LOFO ΔLoss (Sin−All) | LOFO ΔAUC (Sin−All) | Folds Mejora Base | Folds Mejora Ruido |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["family_ablation_summary"]:
        ci_l = row["delta_loss_vs_base_ci95_b30"]
        lines.append(
            f"| {row['symbol']} | {row['model']} | {row['seed']} | `{row['family']}` | "
            f"{row['delta_loss_vs_base']:+.5f} | [{ci_l[0]:+.5f}, {ci_l[1]:+.5f}] | "
            f"{row['delta_loss_vs_noise4']:+.5f} | {row['delta_auc_vs_base']:+.5f} | "
            f"{row['delta_auc_vs_noise4']:+.5f} | {row['lofo_remove_minus_all_loss']:+.5f} | "
            f"{row['lofo_remove_minus_all_auc']:+.5f} | "
            f"{row['folds_beat_base_loss_and_auc']}/{row['total_folds']} | "
            f"{row['folds_beat_noise_loss_and_auc']}/{row['total_folds']} |"
        )

    # Permutation importance summary (block=30)
    perm_df = pd.DataFrame(perms)
    perm_b30 = (
        perm_df.loc[perm_df.block == 30]
        .groupby(["symbol", "model", "name"])
        .agg(
            mean_delta_loss=("delta_loss_mean", "mean"),
            positive_folds=("delta_loss_mean", lambda s: int((s > 0).sum())),
            total_folds=("delta_loss_mean", "count"),
        )
        .reset_index()
        .sort_values(["symbol", "model", "mean_delta_loss"], ascending=[True, True, False])
    )
    lines += [
        "",
        "## 4. Importancia por Permutación por Bloques Fuera de Muestra (Bloque 30 días, `all_financial`, Semilla 42)",
        "",
        "| Activo | Modelo | Grupo / Variable | ΔLoss Permutación Media (↑ útil) | Folds con ΔLoss > 0 |",
        "|---|---|---|---:|---:|",
    ]
    for (sym, mod), g in perm_b30.groupby(["symbol", "model"]):
        fams_g = g.loc[g.name.str.startswith("family_")]
        top_vars = g.loc[~g.name.str.startswith("family_")].head(6)
        for _, r in pd.concat([fams_g, top_vars]).iterrows():
            lines.append(
                f"| {sym} | {mod} | `{r['name']}` | {r['mean_delta_loss']:+.6f} | "
                f"{int(r['positive_folds'])}/{int(r['total_folds'])} |"
            )

    lines += [
        "",
        "## 5. Decisiones de Selección Anidada Interna (`nested_noise_screened`)",
        "",
        "| Activo | Fold | Modelo | Semilla | Familias Elegidas en Validación Interna | Fallback a Base | Columnas |",
        "|---|---:|---|---:|---|---|---:|",
    ]
    for sel in selections:
        f_str = ", ".join(sel["selected_families"]) if sel["selected_families"] else "ninguna"
        lines.append(
            f"| {sel['symbol']} | {sel['fold']} | {sel['model']} | {sel['seed']} | "
            f"`{f_str}` | `{sel['fallback_base']}` | {sel['column_count']} |"
        )

    report_path = run_dir / "report.md"
    if not report_path.exists():
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifacts_path = run_dir / "artifacts.json"
    if not artifacts_path.exists():
        manifest = {
            str(p.relative_to(run_dir)): sha256(p)
            for p in sorted(run_dir.rglob("*"))
            if p.is_file() and p.name != "artifacts.json"
        }
        write_new_json(artifacts_path, manifest)

    print("REPORT GENERATED:", report_path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Optional output directory")
    parser.add_argument(
        "--analyze",
        type=Path,
        default=None,
        help="Generate report.md and artifacts.json for an existing run directory",
    )
    args = parser.parse_args()
    if args.analyze is not None:
        analyze_and_write_report(args.analyze)
    else:
        summary = run_ablation_study(args.out)
        analyze_and_write_report(Path(summary["directory"]))


if __name__ == "__main__":
    main()
