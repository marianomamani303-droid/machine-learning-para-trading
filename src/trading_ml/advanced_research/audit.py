from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_ml.common import sha256, write_new_json
from trading_ml.errors import DataError

AUDIT_COLUMNS = [
    "rsi",
    "volatility",
    "risk_atr",
    "atr_ratio",
    "return_1",
    "return_7",
    "adv_return_20",
    "adv_return_60",
    "adv_momentum_60",
    "adv_slope_20",
    "sma_ratio_20",
    "volume_ratio",
    "flow_buy_share",
    "flow_pressure_10",
    "flow_trades_ratio",
    "mr_z",
    "smc_trend",
    "smc_bos_up",
    "smc_bos_down",
    "smc_sweep_low",
    "smc_choch_down",
]


def audit_defensive_trades(
    frame: pd.DataFrame, trades_path: Path, destination: Path
) -> dict[str, Any]:
    trades = pd.read_csv(trades_path)
    trades["signal_reference"] = pd.to_datetime(trades.signal_reference, utc=True)
    trades["entry_time"] = pd.to_datetime(trades.entry_time, utc=True)
    causal = frame[["reference_time", "timestamp", "close", "regime", *AUDIT_COLUMNS]].copy()
    audit = trades.merge(
        causal,
        left_on="signal_reference",
        right_on="reference_time",
        how="left",
        validate="one_to_one",
    )
    if (
        len(audit) != 19
        or audit.reference_time.isna().any()
        or (audit.reference_time > audit.entry_time).any()
    ):
        raise DataError("No se pudo auditar causalmente las 19 operaciones defensivas")
    audit["outcome"] = np.where(audit.net_return > 0, "winner", "loser")
    audit["hypothesis_only"] = True
    audit.to_csv(destination / "nineteen_trade_audit.csv", index=False)
    rows = []
    for row in audit.itertuples():
        rows.append(
            {
                "entry": row.entry_time.isoformat(),
                "outcome": row.outcome,
                "net_unit_return": float(row.net_return),
                "pnl_cash": float(row.pnl_cash),
                "regime_before_entry": row.regime,
                "pre_entry_features": {name: float(getattr(row, name)) for name in AUDIT_COLUMNS},
            }
        )
    differences = {
        name: {
            outcome: {"mean": float(group[name].mean()), "median": float(group[name].median())}
            for outcome, group in audit.groupby("outcome")
        }
        for name in AUDIT_COLUMNS
    }
    report = {
        "trades": len(audit),
        "winners": int((audit.net_return > 0).sum()),
        "losers": int((audit.net_return <= 0).sum()),
        "source_trades_sha256": sha256(trades_path),
        "rows": rows,
        "descriptive_differences": differences,
        "purpose": "hypothesis_generation_only",
        "independent_evidence": False,
        "no_filters_fitted_on_these_19": True,
        "limitations": "Small selected sample; outcomes used only for description; common features do not establish patterns or counterfactual filters",
    }
    write_new_json(destination / "nineteen_trade_audit.json", report)
    return report
