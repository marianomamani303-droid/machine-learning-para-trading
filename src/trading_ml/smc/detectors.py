from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trading_ml.errors import DataError

if TYPE_CHECKING:
    from trading_ml.config.settings import Features


@dataclass(frozen=True)
class SMCConfig:
    swing_left: int = 3
    swing_right: int = 2
    liquidity_window: int = 20
    ob_window: int = 10
    zone_age: int = 50
    atr_window: int = 14
    min_gap_atr: float = 0.1
    displacement_atr: float = 0.5
    max_zones_per_type: int = 20

    @classmethod
    def from_features(cls, features: Features) -> SMCConfig:
        return cls(
            swing_left=features.smc_swing_left,
            swing_right=features.smc_swing_right,
            liquidity_window=features.smc_liquidity_window,
            ob_window=features.smc_ob_window,
            zone_age=features.smc_zone_age,
        )

    def validate(self) -> None:
        sizes = [
            self.swing_left,
            self.swing_right,
            self.liquidity_window,
            self.ob_window,
            self.zone_age,
            self.atr_window,
            self.max_zones_per_type,
        ]
        if (
            not all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 1 for item in sizes
            )
            or self.liquidity_window < 2
            or not all(
                np.isfinite(item) and item >= 0
                for item in [self.min_gap_atr, self.displacement_atr]
            )
        ):
            raise DataError("Configuración SMC inválida")


SMC_COLUMNS = [
    "smc_ready",
    "smc_trend",
    "smc_structure",
    "smc_higher_high",
    "smc_higher_low",
    "smc_lower_high",
    "smc_lower_low",
    "smc_pivot_high_confirmed",
    "smc_pivot_low_confirmed",
    "smc_bos_up",
    "smc_bos_down",
    "smc_choch_up",
    "smc_choch_down",
    "smc_sweep_low",
    "smc_sweep_high",
    "smc_fvg_up",
    "smc_fvg_down",
    "smc_ob_up",
    "smc_ob_down",
    "smc_fvg_bull_active",
    "smc_fvg_bear_active",
    "smc_ob_bull_active",
    "smc_ob_bear_active",
    "smc_fvg_bull_touch",
    "smc_fvg_bear_touch",
    "smc_ob_bull_touch",
    "smc_ob_bear_touch",
    "smc_has_swing_high",
    "smc_has_swing_low",
    "smc_swing_high_distance",
    "smc_swing_low_distance",
    "smc_structure_location",
]


def smc_columns() -> list[str]:
    return SMC_COLUMNS.copy()


@dataclass(frozen=True)
class Zone:
    kind: str
    direction: int
    lower: float
    upper: float
    born: int
    source: int


def detect_smc(frame: pd.DataFrame, config: SMCConfig | None = None) -> pd.DataFrame:
    """Reconoce eventos al cierre de t; nunca escribe en filas anteriores.

    Un pivot j se confirma en j+right. Las zonas nacen en confirmación, no en
    la vela fuente. Todas las actualizaciones de estado son hacia adelante.
    """
    config = config or SMCConfig()
    config.validate()
    fields = ["open", "high", "low", "close", "volume"]
    if frame.empty or not set(fields).issubset(frame.columns):
        raise DataError("SMC requiere OHLCV no vacío")
    values = frame[fields].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise DataError("OHLCV no finito o inválido para SMC")
    opening, high, low, close, _ = values.T
    if (
        (high < np.maximum(opening, close)).any()
        or (low > np.minimum(opening, close)).any()
        or (high < low).any()
    ):
        raise DataError("OHLC incoherente para SMC")
    if "timestamp" in frame:
        dates = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            raise DataError("SMC requiere timestamps ordenados y únicos")
    previous_close = np.r_[close[0], close[:-1]]
    tr = np.maximum.reduce(
        [high - low, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    atr = (
        pd.Series(tr)
        .shift(1)
        .rolling(config.atr_window, min_periods=config.atr_window)
        .mean()
        .to_numpy()
    )
    hi: tuple[int, float] | None = None
    lo: tuple[int, float] | None = None
    previous_hi: float | None = None
    previous_lo: float | None = None
    broken_hi = broken_lo = -1
    trend = 0
    zones: list[Zone] = []
    records: list[dict[str, Any]] = []
    for i in range(len(frame)):
        row: dict[str, Any] = dict.fromkeys(SMC_COLUMNS, 0.0)
        row["smc_structure_location"] = 0.5
        row["smc_ready"] = float(
            i
            >= max(
                config.liquidity_window, config.swing_left + config.swing_right, config.atr_window
            )
        )
        # j sólo se puede confirmar ahora, al cierre de i = j + right.
        j = i - config.swing_right
        if j >= config.swing_left:
            before = slice(j - config.swing_left, j)
            after = slice(j + 1, i + 1)
            if high[j] > high[before].max() and high[j] > high[after].max():
                previous_hi = hi[1] if hi is not None else None
                hi = (j, high[j])
                row["smc_pivot_high_confirmed"] = 1.0
            if low[j] < low[before].min() and low[j] < low[after].min():
                previous_lo = lo[1] if lo is not None else None
                lo = (j, low[j])
                row["smc_pivot_low_confirmed"] = 1.0
        if hi is not None and previous_hi is not None:
            row["smc_higher_high"] = float(hi[1] > previous_hi)
            row["smc_lower_high"] = float(hi[1] < previous_hi)
        if lo is not None and previous_lo is not None:
            row["smc_higher_low"] = float(lo[1] > previous_lo)
            row["smc_lower_low"] = float(lo[1] < previous_lo)
        structure = int(row["smc_higher_high"] and row["smc_higher_low"]) - int(
            row["smc_lower_high"] and row["smc_lower_low"]
        )
        row["smc_structure"] = float(structure)
        if trend == 0 and structure != 0:
            trend = structure
        if i > 0 and hi is not None and broken_hi != hi[0] and close[i - 1] <= hi[1] < close[i]:
            row["smc_choch_up" if trend < 0 else "smc_bos_up"] = 1.0
            trend = 1
            broken_hi = hi[0]
        if i > 0 and lo is not None and broken_lo != lo[0] and close[i] < lo[1] <= close[i - 1]:
            row["smc_choch_down" if trend > 0 else "smc_bos_down"] = 1.0
            trend = -1
            broken_lo = lo[0]
        row["smc_trend"] = float(trend)
        if i >= config.liquidity_window:
            prior_high = high[i - config.liquidity_window : i].max()
            prior_low = low[i - config.liquidity_window : i].min()
            row["smc_sweep_high"] = float(high[i] > prior_high and close[i] < prior_high)
            row["smc_sweep_low"] = float(low[i] < prior_low and close[i] > prior_low)
        retained: list[Zone] = []
        for zone in zones:
            if i - zone.born > config.zone_age:
                continue
            if zone.kind == "fvg":
                invalid = low[i] <= zone.lower if zone.direction > 0 else high[i] >= zone.upper
            else:
                invalid = close[i] < zone.lower if zone.direction > 0 else close[i] > zone.upper
            if invalid:
                continue
            if low[i] <= zone.upper and high[i] >= zone.lower:
                row[f"smc_{zone.kind}_{'bull' if zone.direction > 0 else 'bear'}_touch"] = 1.0
            retained.append(zone)
        zones = retained
        if i >= 2 and np.isfinite(atr[i]) and atr[i] > 0:
            middle_body = close[i - 1] - opening[i - 1]
            impulse = abs(middle_body) >= config.displacement_atr * atr[i]
            if impulse and middle_body > 0 and low[i] - high[i - 2] > config.min_gap_atr * atr[i]:
                row["smc_fvg_up"] = 1.0
                zones.append(Zone("fvg", 1, high[i - 2], low[i], i, i - 2))
            elif impulse and middle_body < 0 and low[i - 2] - high[i] > config.min_gap_atr * atr[i]:
                row["smc_fvg_down"] = 1.0
                zones.append(Zone("fvg", -1, high[i], low[i - 2], i, i - 2))
            direction = int(row["smc_bos_up"] or row["smc_choch_up"]) - int(
                row["smc_bos_down"] or row["smc_choch_down"]
            )
            body = close[i] - opening[i]
            if direction != 0 and direction * body >= config.displacement_atr * atr[i]:
                for source in range(i - 1, max(-1, i - config.ob_window - 1), -1):
                    if direction * (close[source] - opening[source]) < 0:
                        if (direction > 0 and close[i] >= low[source]) or (
                            direction < 0 and close[i] <= high[source]
                        ):
                            row["smc_ob_up" if direction > 0 else "smc_ob_down"] = 1.0
                            zones.append(
                                Zone("ob", direction, low[source], high[source], i, source)
                            )
                        break
        # Límite explícito de memoria por tipo/dirección, conserva zonas más recientes.
        zones = [
            zone
            for kind in ["fvg", "ob"]
            for direction in [1, -1]
            for zone in [z for z in zones if z.kind == kind and z.direction == direction][
                -config.max_zones_per_type :
            ]
        ]
        for kind in ["fvg", "ob"]:
            for direction, name in [(1, "bull"), (-1, "bear")]:
                row[f"smc_{kind}_{name}_active"] = float(
                    sum(z.kind == kind and z.direction == direction for z in zones)
                )
        scale = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else max(close[i] * 0.0001, 1e-12)
        if hi is not None:
            row["smc_has_swing_high"] = 1.0
            row["smc_swing_high_distance"] = float(np.clip((hi[1] - close[i]) / scale, -10, 10))
        if lo is not None:
            row["smc_has_swing_low"] = 1.0
            row["smc_swing_low_distance"] = float(np.clip((close[i] - lo[1]) / scale, -10, 10))
        if hi is not None and lo is not None and hi[1] > lo[1]:
            row["smc_structure_location"] = float(
                np.clip((close[i] - lo[1]) / (hi[1] - lo[1]), 0, 1)
            )
        event_names = [
            key.removeprefix("smc_")
            for key in [
                "smc_pivot_high_confirmed",
                "smc_pivot_low_confirmed",
                "smc_bos_up",
                "smc_bos_down",
                "smc_choch_up",
                "smc_choch_down",
                "smc_sweep_low",
                "smc_sweep_high",
                "smc_fvg_up",
                "smc_fvg_down",
                "smc_ob_up",
                "smc_ob_down",
                "smc_fvg_bull_touch",
                "smc_fvg_bear_touch",
                "smc_ob_bull_touch",
                "smc_ob_bear_touch",
            ]
            if row[key] == 1
        ]
        row["smc_pattern_names"] = "|".join(event_names)
        descriptions = []
        for zone in zones:
            item: dict[str, Any] = {
                "kind": zone.kind,
                "direction": "bullish" if zone.direction > 0 else "bearish",
                "lower": float(zone.lower),
                "upper": float(zone.upper),
                "age_bars": i - zone.born,
                "created_position": zone.born,
                "source_position": zone.source,
            }
            if "timestamp" in frame:
                item.update(
                    {
                        "created_open_time": pd.Timestamp(
                            frame.timestamp.iloc[zone.born]
                        ).isoformat(),
                        "source_open_time": pd.Timestamp(
                            frame.timestamp.iloc[zone.source]
                        ).isoformat(),
                    }
                )
            descriptions.append(item)
        row["smc_active_zones"] = json.dumps(descriptions, allow_nan=False)
        records.append(row)
    return pd.DataFrame(records, index=frame.index)
