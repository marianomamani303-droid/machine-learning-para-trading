from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trading_ml.errors import DataError
from trading_ml.momentum.config import ResearchCosts


@dataclass(frozen=True)
class ExitRule:
    name: str
    horizon: int
    mode: str = "fixed"  # fixed horizon, volatility, ATR fraction, percent, close monitoring
    take: float = 1.5
    stop: float = 1.0

    def validate(self):
        if (
            not self.name
            or not isinstance(self.horizon, int)
            or isinstance(self.horizon, bool)
            or not 1 <= self.horizon <= 10
            or self.mode not in {"fixed", "vol", "atr", "percent", "close"}
            or not np.isfinite([self.take, self.stop]).all()
            or min(self.take, self.stop) <= 0
        ):
            raise DataError("Regla Triple Barrier inválida")
        if self.mode == "percent" and self.stop >= 1:
            raise DataError("Stop porcentual inválido")


def exit_rules():
    return [
        *[ExitRule(f"net_h{h}", h) for h in [1, 3, 5, 10]],
        *[ExitRule(f"tb_vol_h{h}", h, "vol") for h in [1, 3, 5, 10]],
        ExitRule("tb_asym_h5", 5, "vol", 2.0, 1.0),
        ExitRule("tb_percent_h5", 5, "percent", 0.03, 0.02),
        ExitRule("tb_atr_h5", 5, "atr"),
        ExitRule("tb_close_h5", 5, "close"),
    ]


def causal_scales(frame):
    f = frame.copy()
    f["label_sigma"] = np.log(f.close).diff().rolling(20, min_periods=20).std()
    tr = pd.concat(
        [f.high - f.low, (f.high - f.close.shift(1)).abs(), (f.low - f.close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    f["label_atr_fraction"] = tr.rolling(14, min_periods=14).mean() / f.close
    return f


def widths(rule, sigma, atr_fraction):
    if rule.mode == "fixed":
        return 0.0, 0.0
    scale = 1.0 if rule.mode == "percent" else (atr_fraction if rule.mode == "atr" else sigma)
    up, down = rule.take * scale, rule.stop * scale
    if not np.isfinite([up, down]).all() or not 0 < down < 1 or up <= 0:
        return np.nan, np.nan
    return float(up), float(down)


def observed_exit(rule, *, age, opening, high, low, previous_close, upper, lower, tie="stop_first"):
    """One forward execution step, using prices observed in this bar only.

    TP fills are capped at the limit even on favorable gaps; SL gaps fill at open.
    Intrabar timestamps are conservatively observable at bar close, not invented.
    The same kernel is used by historical labels and the forward portfolio engine.
    """
    if tie not in {"stop_first", "take_first"}:
        raise DataError("Orden intravela inválido")
    if age >= rule.horizon:
        return float(opening), "timeout", "open", False
    if rule.mode == "fixed":
        return None
    if rule.mode == "close":
        if age > 0 and previous_close <= lower:
            return float(opening), "sl_close", "open", False
        if age > 0 and previous_close >= upper:
            return float(opening), "tp_close", "open", False
        return None
    if opening <= lower:
        return float(opening), "sl_gap", "open", False
    if opening >= upper:
        return float(upper), "tp_gap", "open", False
    hit_stop, hit_take = low <= lower, high >= upper
    both = bool(hit_stop and hit_take)
    if hit_stop and (not hit_take or tie == "stop_first"):
        return float(lower), "sl", "intrabar", both
    if hit_take:
        return float(upper), "tp", "intrabar", both
    return None


def unit_return(entry, exit_price, years, costs):
    return float(
        exit_price
        / entry
        * costs.sell_factor
        / costs.buy_factor
        * np.exp(-costs.annual_holding_bps / 10000 * years)
        - 1
    )


def label_events(frame, rule: ExitRule, costs: ResearchCosts, *, tie="stop_first"):
    """Future prices are outcomes only. No full-horizon requirement for early hits."""
    rule.validate()
    costs.validate()
    opening, high, low, close = (
        frame[c].to_numpy(dtype=float) for c in ["open", "high", "low", "close"]
    )
    sigma, atr = frame.label_sigma.to_numpy(), frame.label_atr_fraction.to_numpy()
    timestamps, references = list(frame.timestamp), list(frame.reference_time)
    records = []
    for source in range(len(frame)):
        up, down = widths(rule, sigma[source], atr[source])
        row = {
            "source_index": source,
            "reference_time": references[source],
            "entry_index": source + 1,
            "exit_index": np.nan,
            "target_time": pd.NaT,
            "target": np.nan,
            "event_label": np.nan,
            "unit_net_return": np.nan,
            "up_fraction": up,
            "down_fraction": down,
            "exit_price": np.nan,
            "reason": "censored",
            "phase": "unknown",
            "ambiguous": False,
        }
        entry = source + 1
        if entry >= len(frame) or not np.isfinite([up, down]).all():
            records.append(row)
            continue
        price = opening[entry]
        upper, lower = price * (1 + up), price * (1 - down)
        for j in range(entry, min(entry + rule.horizon + 1, len(frame))):
            result = observed_exit(
                rule,
                age=j - entry,
                opening=opening[j],
                high=high[j],
                low=low[j],
                previous_close=close[j - 1],
                upper=upper,
                lower=lower,
                tie=tie,
            )
            if result is None:
                continue
            exit_price, reason, phase, ambiguous = result
            time = timestamps[j] if phase == "open" else references[j]
            # Daily intrabar holding time charged conservatively as the whole bar.
            years = (time - timestamps[entry]).total_seconds() / (365.25 * 86400)
            net = unit_return(price, exit_price, years, costs)
            row.update(
                exit_index=j,
                target_time=time,
                target=int(net > 0),
                event_label=1
                if reason.startswith("tp")
                else (-1 if reason.startswith("sl") else 0),
                unit_net_return=net,
                exit_price=exit_price,
                reason=reason,
                phase=phase,
                ambiguous=ambiguous,
            )
            break
        records.append(row)
    output = pd.DataFrame(records)
    output.target_time = pd.to_datetime(output.target_time, utc=True)
    output.attrs["rule"] = asdict(rule)
    return output


def attach_labels(frame, events):
    result = frame.copy()
    for name in ["target", "target_time", "unit_net_return", "event_label", "reason", "ambiguous"]:
        result[name] = events[name].to_numpy()
    result.target_time = pd.to_datetime(result.target_time, utc=True)
    return result


def average_uniqueness(events):
    """Training-only overlap weights. Intervals are [signal, observed label end]."""
    if events.empty or events.target_time.isna().any():
        raise DataError("Unicidad requiere eventos observados de train")
    grid = pd.date_range(events.reference_time.min(), events.target_time.max(), freq="1D")
    starts = np.searchsorted(grid.asi8, pd.DatetimeIndex(events.reference_time).asi8)
    ends = np.searchsorted(grid.asi8, pd.DatetimeIndex(events.target_time).asi8)
    difference = np.zeros(len(grid) + 1)
    np.add.at(difference, starts, 1)
    np.add.at(difference, ends + 1, -1)
    concurrency = np.cumsum(difference[:-1])
    reciprocal = np.divide(1.0, concurrency, out=np.zeros_like(concurrency), where=concurrency > 0)
    cumulative = np.r_[0.0, np.cumsum(reciprocal)]
    weights = (cumulative[ends + 1] - cumulative[starts]) / (ends - starts + 1)
    return weights / weights.mean()
