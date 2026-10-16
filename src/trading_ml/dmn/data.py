from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ml.common import read_json, sha256, sidecar
from trading_ml.config import Settings
from trading_ml.config.settings import Market
from trading_ml.data import download
from trading_ml.errors import DataError, LeakageError
from trading_ml.preprocessing.quality import clean_ohlcv
from trading_ml.training.recipe import protect_reserved_labels

HORIZONS = (1, 21, 63, 126, 252)
SCALES = ((8, 24), (16, 48), (32, 96))
FEATURES = tuple(
    [f"return_{h}" for h in HORIZONS] + [f"macd_{short}_{long}" for short, long in SCALES]
)


def causal_features(raw: pd.DataFrame, volatility_floor: float = 1e-4) -> pd.DataFrame:
    """Feature values at close t; execution label is deliberately a separate column."""
    frame, _ = clean_ohlcv(raw, pd.Timedelta(days=1))
    close = frame.close
    sigma = close.pct_change(fill_method=None).ewm(span=60, adjust=False, min_periods=60).std()
    frame["daily_volatility"] = sigma.clip(lower=volatility_floor)
    for h in HORIZONS:
        frame[f"return_{h}"] = close.pct_change(h, fill_method=None) / (
            frame.daily_volatility * np.sqrt(h)
        )
    price_std = close.rolling(63, min_periods=63).std()
    for short, long in SCALES:
        # Equation 6 uses time scale S (alpha=1/S), not pandas span=S.
        macd = (
            close.ewm(alpha=1 / short, adjust=False).mean()
            - close.ewm(alpha=1 / long, adjust=False).mean()
        )
        q = macd / price_std.replace(0, np.nan)
        frame[f"macd_{short}_{long}"] = q / q.rolling(252, min_periods=252).std().replace(0, np.nan)
    frame["reference_time"] = frame.timestamp + pd.Timedelta(days=1)
    frame["target_time"] = frame.timestamp + pd.Timedelta(days=2)
    frame["asset_return"] = frame.open.shift(-2) / frame.open.shift(-1) - 1
    values = frame[list(FEATURES)].to_numpy()
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        raise DataError("Historia insuficiente o sin variación para features DMN")
    first = int(np.flatnonzero(valid)[0])
    if not valid[first:].all():
        raise DataError("Features DMN inválidas después del calentamiento; no se rellenan")
    return frame.iloc[first:].reset_index(drop=True)


@dataclass
class Panel:
    times: pd.DatetimeIndex
    target_times: pd.DatetimeIndex
    symbols: tuple[str, ...]
    x: np.ndarray  # date, asset, lookback, feature; each window ends at signal close
    returns: np.ndarray
    volatility: np.ndarray
    baselines: dict[str, np.ndarray]

    def eligible(self, start: pd.Timestamp | None, end: pd.Timestamp) -> np.ndarray:
        mask = (self.target_times < end) & np.isfinite(self.returns).all(axis=1)
        if start is not None:
            mask &= self.times >= start
        return np.flatnonzero(mask)


def build_panel(raw: dict[str, pd.DataFrame], sequence_length: int, floor: float) -> Panel:
    if not raw or sequence_length < 2:
        raise DataError("Universo o secuencia DMN inválida")
    frames = {s: causal_features(f, floor) for s, f in raw.items()}
    first = next(iter(frames.values()))
    for f in frames.values():
        if not f.reference_time.equals(first.reference_time):
            raise DataError("Universos desalineados: exigir cobertura común, sin imputar activos")
    if len(first) < sequence_length + 2:
        raise DataError("Historia insuficiente para las secuencias DMN")
    values = np.stack(
        [f[list(FEATURES)].to_numpy(dtype=np.float32) for f in frames.values()], axis=1
    )
    x = np.lib.stride_tricks.sliding_window_view(values, sequence_length, axis=0).transpose(
        0, 1, 3, 2
    )
    tail = sequence_length - 1
    returns = np.stack([f.asset_return.to_numpy()[tail:] for f in frames.values()], axis=1)
    vol = np.stack([f.daily_volatility.to_numpy()[tail:] for f in frames.values()], axis=1)
    macds = np.stack(
        [
            f[[f"macd_{short}_{long}" for short, long in SCALES]].to_numpy()[tail:]
            for f in frames.values()
        ],
        axis=1,
    )
    # phi is applied per time scale and then averaged. Clamp the rounded 0.89 bound.
    macd_position = np.clip((macds * np.exp(-(macds**2) / 4) / 0.89).mean(axis=2), -1, 1)
    return Panel(
        pd.DatetimeIndex(first.reference_time.iloc[tail:]),
        pd.DatetimeIndex(first.target_time.iloc[tail:]),
        tuple(frames),
        x,
        returns,
        vol,
        {
            "cash": np.zeros_like(vol),
            "long_scaled": np.ones_like(vol),
            "buy_hold_rebalanced": np.ones_like(vol),
            "tsmom": np.stack(
                [np.sign(f.return_252.to_numpy()[tail:]) for f in frames.values()], axis=1
            ),
            "macd": macd_position,
        },
    )


def load_raw(settings: Settings, protocol: dict, output: Path, *, fetch: bool) -> tuple[dict, dict]:
    """Verify source manifests; crop before features/labels, never load model results."""
    data = protocol["data"]
    raw, sources = {}, {}
    start, end = pd.to_datetime([data["start"], data["end"]], utc=True)
    output.mkdir(parents=True, exist_ok=True)
    for symbol in data["symbols"]:
        source = Path(data["sources"][symbol])
        if not source.exists():
            if not fetch:
                raise DataError(
                    f"Falta {source}; usar --download para obtener históricos verificables"
                )
            asset = replace(
                settings, root=output, market=Market(symbol, "1d", data["start"], data["end"])
            )
            source = download(asset, output / f"{symbol}_raw.csv")
        meta = read_json(sidecar(source))
        if sha256(source) != meta["sha256"]:
            raise DataError(f"SHA256 del histórico DMN inválido: {source}")
        if meta["market"]["symbol"] != symbol or meta["market"]["interval"] != "1d":
            raise DataError("Fuente de otro activo/intervalo")
        f = pd.read_csv(source)
        f.timestamp = pd.to_datetime(f.timestamp, utc=True)
        f = f.loc[(f.timestamp >= start) & (f.timestamp < end)].copy()
        f, quality = clean_ohlcv(f, pd.Timedelta(days=1))
        if f.timestamp.iloc[0] != start or f.timestamp.iloc[-1] + pd.Timedelta(days=1) != end:
            raise DataError("Cobertura DMN incompleta")
        guard = pd.DataFrame(
            {
                "reference_time": f.timestamp + pd.Timedelta(days=1),
                "target_time": f.timestamp + pd.Timedelta(days=2),
            }
        )
        # All phases, including evaluation, must avoid the existing reserved intervals.
        protect_reserved_labels(settings.root, guard)
        raw[symbol] = f
        sources[symbol] = {
            "path": str(source.resolve()),
            "sha256": sha256(source),
            "upstream": meta,
            "quality": quality,
        }
    return raw, sources


def temporal_split(panel: Panel, start: pd.Timestamp, end: pd.Timestamp, training: dict) -> tuple:
    validation_start = start - pd.Timedelta(days=training["validation_days"])
    embargo = pd.Timedelta(days=training["embargo_days"])
    train = panel.eligible(None, validation_start - embargo)
    validation = panel.eligible(validation_start, start - embargo)
    test = panel.eligible(start, end)
    if len(train) < training["minimum_train_dates"] or len(validation) < 20 or len(test) < 20:
        raise DataError("Fold DMN insuficiente después de contexto, purga y embargo")
    if not (
        panel.target_times[train].max() < panel.times[validation].min()
        and panel.target_times[validation].max() < panel.times[test].min()
    ):
        raise LeakageError("Particiones temporales DMN superpuestas")
    return train, validation, test
