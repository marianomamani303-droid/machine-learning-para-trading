from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from trading_ml.advanced_research.features import META
from trading_ml.common import frame_hash
from trading_ml.momentum.config import ResearchCosts
from trading_ml.risk_overlay.config import RiskPolicy


def causal_opportunity_labels(
    frame: pd.DataFrame,
    base_probability: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    costs: ResearchCosts,
    policy: RiskPolicy,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """Meta training comes ONLY from earlier OOS base forecasts, never in-sample scores.

    Unit trade labels share the base ATR/holding/probability exits. Censor labels
    crossing the forecast window rather than use unforecasted p=0 as a fake exit.
    Portfolio DD and partial cap trims are evaluated in the actual overlay, not
    asserted to be predictable labels of an isolated opportunity.
    """
    rows = []
    p = base_probability
    for source in np.flatnonzero(
        (p >= threshold)
        & (frame.reference_time >= start)
        & (frame.reference_time < end)
        & frame.research_ready
    ):
        source = int(source)
        if source < 1 or frame.reference_time.iloc[source - 1] < start or p[source - 1] < threshold:
            continue
        entry = source + 1
        if entry >= len(frame):
            continue
        price = float(frame.open.iloc[entry])
        stop = price - policy.atr_multiple * float(frame.risk_atr.iloc[source])
        if stop <= 0:
            continue
        row = {
            "source_index": source,
            "reference_time": frame.reference_time.iloc[source],
            "target_time": pd.NaT,
            "target": np.nan,
            "unit_net_return": np.nan,
            "base_probability": float(p[source]),
            "status": "censored",
        }
        maximum = policy.max_holding_bars or 5
        for i in range(entry + 1, min(entry + maximum + 1, len(frame))):
            if frame.reference_time.iloc[i - 1] >= end or frame.timestamp.iloc[i] >= end:
                break
            if (
                frame.close.iloc[i - 1] <= stop
                or p[i - 1] < policy.exit_threshold
                or i - entry >= maximum
            ):
                net = frame.open.iloc[i] / price * costs.sell_factor / costs.buy_factor - 1
                row.update(
                    {
                        "target_time": frame.timestamp.iloc[i],
                        "target": int(net > 0),
                        "unit_net_return": float(net),
                        "status": "observed",
                    }
                )
                break
        rows.append(row)
    result = pd.DataFrame(
        rows,
        columns=[
            "source_index",
            "reference_time",
            "target_time",
            "target",
            "unit_net_return",
            "base_probability",
            "status",
        ],
    )
    for column in ["reference_time", "target_time"]:
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


@dataclass
class MetaFilter:
    model: Any
    audit: dict[str, Any]

    def probabilities(self, frame: pd.DataFrame, p: np.ndarray) -> np.ndarray:
        output = np.zeros(len(frame))
        if self.model is None:
            return output
        features = frame.copy()
        features["base_probability"] = p
        ready = features.research_ready
        output[ready] = self.model.predict_proba(features.loc[ready, META])[:, 1]
        return output


def fit_meta(
    frame: pd.DataFrame,
    pool: pd.DataFrame,
    cutoff: pd.Timestamp,
    kind: str,
    protocol: dict[str, Any],
    consume_fit: Callable[[], None],
) -> MetaFilter:
    used = pool.loc[
        (pool.status == "observed") & (pool.target_time < cutoff) & (pool.reference_time < cutoff)
    ].copy()
    features = frame.loc[used.source_index].copy()
    features["base_probability"] = used.base_probability.to_numpy(dtype=float)
    wins = used.loc[used.target == 1, "unit_net_return"]
    losses = used.loc[used.target == 0, "unit_net_return"]
    audit = {
        "rows": len(used),
        "last_target": used.target_time.max().isoformat() if len(used) else None,
        "train_sha256": frame_hash(features[META]),
        "label_sha256": frame_hash(used),
        "win_mean": float(wins.mean()) if len(wins) else 0.0,
        "loss_mean": float(losses.mean()) if len(losses) else 0.0,
        "win_se": float(wins.std() / np.sqrt(len(wins))) if len(wins) > 1 else 1.0,
        "loss_se": float(losses.std() / np.sqrt(len(losses))) if len(losses) > 1 else 1.0,
        "past_oos_forecasts_only": True,
        "status": "abstain_insufficient_meta_data",
    }
    if len(used) < protocol["research"]["minimum_meta_rows"] or used.target.nunique() != 2:
        return MetaFilter(None, audit)
    if kind == "logistic":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        **protocol["logistic"], random_state=protocol["research"]["seed"]
                    ),
                ),
            ]
        )
    else:
        estimator = XGBClassifier(
            **protocol["xgboost_regularized"],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=protocol["research"]["seed"],
        )
    consume_fit()
    estimator.fit(features[META], used.target.astype(int))
    audit["status"] = "fitted"
    return MetaFilter(estimator, audit)


def gates(
    frame: pd.DataFrame, base: np.ndarray, meta: dict[str, MetaFilter], protocol: dict[str, Any]
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    output = {}
    for kind, model in meta.items():
        p = model.probabilities(frame, base)
        expected = p * model.audit["win_mean"] + (1 - p) * model.audit["loss_mean"]
        se = np.sqrt((p * model.audit["win_se"]) ** 2 + ((1 - p) * model.audit["loss_se"]) ** 2)
        lower = expected - protocol["filters"]["expectation_se_multiplier"] * se
        confidence = p >= protocol["filters"]["meta_threshold"]
        output["meta_" + kind] = (
            confidence,
            p,
            {"expected_net": expected, "expectation_lower": lower},
        )
        output["meta_" + kind + "_economic"] = (
            confidence & (lower > protocol["filters"]["minimum_expected_net_return"]),
            p,
            {"expected_net": expected, "expectation_lower": lower},
        )
    output["trend_gate"] = ((frame.adv_regime_score >= 0).to_numpy(), base, {})
    output["volatility_gate"] = ((frame.volatility <= frame.adv_vol_limit).to_numpy(), base, {})
    return output
