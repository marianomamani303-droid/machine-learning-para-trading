from __future__ import annotations

import math
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.errors import TradingMLError

INTERVALS = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


@dataclass(frozen=True)
class Market:
    symbol: str = "BTCUSDT"
    interval: str = "1d"
    start: str = "2020-01-01"
    end: str = "2026-10-01"


@dataclass(frozen=True)
class Features:
    rsi_window: int = 14
    sma_windows: list[int] = field(default_factory=lambda: [10, 20, 50])
    volatility_window: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    enhanced: bool = False
    smc: bool = False
    smc_swing_left: int = 3
    smc_swing_right: int = 2
    smc_liquidity_window: int = 20
    smc_ob_window: int = 10
    smc_zone_age: int = 50


@dataclass(frozen=True)
class Training:
    horizon: int = 1
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    min_partition_rows: int = 60
    seed: int = 42


@dataclass(frozen=True)
class Backtest:
    threshold: float = 0.55
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_capital: float = 10000.0


@dataclass(frozen=True)
class Settings:
    market: Market = field(default_factory=Market)
    features: Features = field(default_factory=Features)
    training: Training = field(default_factory=Training)
    backtest: Backtest = field(default_factory=Backtest)
    model: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 160,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 5.0,
            "min_child_weight": 10,
        }
    )
    search: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.07],
        }
    )
    root: Path = Path("artifacts")

    @property
    def delta(self) -> pd.Timedelta:
        return pd.Timedelta(INTERVALS[self.market.interval])

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["root"] = str(self.root)
        return result

    def contract(self) -> dict[str, Any]:
        features = asdict(self.features)
        if not self.features.enhanced:
            features.pop("enhanced")
        if not self.features.smc:
            features = {key: value for key, value in features.items() if not key.startswith("smc")}
        return {
            "symbol": self.market.symbol,
            "interval": self.market.interval,
            "features": features,
            "horizon": self.training.horizon,
            "feature_schema": 3 if self.features.smc else (2 if self.features.enhanced else 1),
        }

    def validate(self) -> None:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise TradingMLError(message)

        require(bool(re.fullmatch(r"[A-Z0-9]{3,24}", self.market.symbol)), "Símbolo inválido")
        require(self.market.interval in INTERVALS, f"Intervalos admitidos: {list(INTERVALS)}")
        require(
            bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.market.start))
            and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.market.end)),
            "start y end deben usar YYYY-MM-DD (UTC)",
        )
        require(
            pd.Timestamp(self.market.start) < pd.Timestamp(self.market.end),
            "start debe ser anterior a end",
        )
        require(
            isinstance(self.training.horizon, int) and self.training.horizon > 0,
            "horizon debe ser entero positivo",
        )
        fractions = [self.training.train_fraction, self.training.validation_fraction]
        require(
            all(math.isfinite(x) and 0 < x < 1 for x in fractions) and sum(fractions) < 1,
            "Fracciones cronológicas inválidas",
        )
        require(
            isinstance(self.training.min_partition_rows, int)
            and self.training.min_partition_rows >= 2,
            "min_partition_rows debe ser entero >= 2",
        )
        require(
            isinstance(self.training.seed, int) and 0 <= self.training.seed < 2**32,
            "seed debe ser entero entre 0 y 2**32-1",
        )
        windows = [
            *self.features.sma_windows,
            self.features.rsi_window,
            self.features.volatility_window,
            self.features.macd_fast,
            self.features.macd_slow,
            self.features.macd_signal,
        ]
        require(
            bool(self.features.sma_windows) and all(isinstance(x, int) and x >= 2 for x in windows),
            "Ventanas de indicadores inválidas",
        )
        require(
            len(set(self.features.sma_windows)) == len(self.features.sma_windows),
            "Ventanas SMA duplicadas",
        )
        require(
            self.features.macd_fast < self.features.macd_slow, "MACD fast debe ser menor que slow"
        )
        require(isinstance(self.features.enhanced, bool), "features.enhanced debe ser booleano")
        require(isinstance(self.features.smc, bool), "features.smc debe ser booleano")
        require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
                for value in [
                    self.features.smc_swing_left,
                    self.features.smc_swing_right,
                    self.features.smc_liquidity_window,
                    self.features.smc_ob_window,
                    self.features.smc_zone_age,
                ]
            ),
            "Ventanas SMC inválidas",
        )
        require(self.features.smc_liquidity_window >= 2, "Ventana de liquidez debe ser >= 2")
        require(
            math.isfinite(self.backtest.threshold) and 0.5 <= self.backtest.threshold <= 1,
            "threshold debe estar entre 0.5 y 1",
        )
        require(
            math.isfinite(self.backtest.initial_capital) and self.backtest.initial_capital > 0,
            "Capital inválido",
        )
        require(
            all(
                math.isfinite(x) and 0 <= x < 10000
                for x in [self.backtest.commission_bps, self.backtest.slippage_bps]
            ),
            "Costos inválidos",
        )
        require(
            not (
                {"objective", "random_state", "n_jobs", "tree_method", "device", "eval_metric"}
                & self.model.keys()
            ),
            "Parámetros reservados de reproducibilidad",
        )
        domains = {
            "n_estimators": lambda x: isinstance(x, int) and x >= 1,
            "max_depth": lambda x: isinstance(x, int) and x >= 1,
            "learning_rate": lambda x: (
                isinstance(x, (float, int)) and math.isfinite(x) and 0 < x <= 1
            ),
            "subsample": lambda x: isinstance(x, (float, int)) and math.isfinite(x) and 0 < x <= 1,
            "colsample_bytree": lambda x: (
                isinstance(x, (float, int)) and math.isfinite(x) and 0 < x <= 1
            ),
            "reg_lambda": lambda x: isinstance(x, (float, int)) and math.isfinite(x) and x >= 0,
            "reg_alpha": lambda x: isinstance(x, (float, int)) and math.isfinite(x) and x >= 0,
            "min_child_weight": lambda x: (
                isinstance(x, (float, int)) and math.isfinite(x) and x >= 0
            ),
            "gamma": lambda x: isinstance(x, (float, int)) and math.isfinite(x) and x >= 0,
            "max_bin": lambda x: isinstance(x, int) and x >= 2,
        }
        require(set(self.model) <= set(domains), "Hiperparámetro XGBoost desconocido")
        require(
            {"max_depth", "learning_rate"} <= set(self.model), "Faltan max_depth o learning_rate"
        )
        require(
            all(domains[key](item) for key, item in self.model.items()),
            "Hiperparámetros XGBoost inválidos",
        )
        require(
            set(self.search) <= {"enabled", "max_depth", "learning_rate"},
            "Parámetros de búsqueda desconocidos",
        )
        require(
            isinstance(self.search.get("enabled", False), bool), "search.enabled debe ser booleano"
        )
        require(
            all(
                isinstance(self.search.get(key, [self.model[key]]), list)
                and self.search.get(key, [self.model[key]])
                for key in ["max_depth", "learning_rate"]
            ),
            "Grilla de búsqueda vacía o inválida",
        )
        require(
            all(
                domains[key](item)
                for key in ["max_depth", "learning_rate"]
                for item in self.search.get(key, [self.model[key]])
            ),
            "Valores de búsqueda inválidos",
        )


def settings_from_dict(value: dict[str, Any]) -> Settings:
    settings = Settings(
        market=Market(**value.get("market", {})),
        features=Features(**value.get("features", {})),
        training=Training(**value.get("training", {})),
        backtest=Backtest(**value.get("backtest", {})),
        model={**Settings().model, **value.get("model", {})},
        search=value.get("search", Settings().search),
        root=Path(value.get("root", value.get("paths", {}).get("root", "artifacts"))),
    )
    settings.validate()
    return settings


def load_settings(path: Path | str = "config/default.toml") -> Settings:
    with Path(path).open("rb") as stream:
        return settings_from_dict(tomllib.load(stream))
