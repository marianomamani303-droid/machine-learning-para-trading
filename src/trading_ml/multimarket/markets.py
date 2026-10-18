from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from trading_ml.errors import DataError


@dataclass(frozen=True)
class Instrument:
    symbol: str
    market: str
    base: str
    quote: str
    pip: float
    price_scale: int
    minimum_units: float
    unit_step: float
    provider: str

    def contract(self) -> dict:
        return asdict(self)


INSTRUMENTS = {
    symbol: Instrument(symbol, "forex", symbol[:3], symbol[3:], pip, scale, 1, 1, "dukascopy")
    for symbol, pip, scale in [
        ("EURUSD", 0.0001, 100000),
        ("USDJPY", 0.01, 1000),
        ("GBPUSD", 0.0001, 100000),
        ("USDCHF", 0.0001, 100000),
        ("AUDUSD", 0.0001, 100000),
        ("USDCAD", 0.0001, 100000),
        ("NZDUSD", 0.0001, 100000),
        ("EURJPY", 0.01, 1000),
        ("GBPJPY", 0.01, 1000),
        ("EURGBP", 0.0001, 100000),
    ]
}
INSTRUMENTS.update(
    {
        symbol: Instrument(
            symbol, "crypto", symbol[:-4], "USDT", 0.01, 100, 0.00001, 0.00001, "binance"
        )
        for symbol in ["BTCUSDT", "ETHUSDT"]
    }
)


def instrument(symbol: str) -> Instrument:
    key = symbol.upper().replace("/", "").replace("_", "")
    if key not in INSTRUMENTS:
        raise DataError(f"Instrumento no configurado: {symbol}")
    return INSTRUMENTS[key]


def fx_open(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Nominal NY Sunday 17:00 to Friday 17:00, including DST.

    Jan 1 and Dec 25 are treated as closures. Early holiday closures remain
    explicit missing-hour diagnostics; no invented calendar exceptions.
    """
    local = timestamps.tz_convert("America/New_York")
    weekday = local.dayofweek
    active = (
        (weekday < 4) | ((weekday == 4) & (local.hour < 17)) | ((weekday == 6) & (local.hour >= 17))
    )
    holiday = ((timestamps.month == 1) & (timestamps.day == 1)) | (
        (timestamps.month == 12) & (timestamps.day == 25)
    )
    return pd.Series(active & ~holiday, index=timestamps)


def session_end(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    local = timestamps.tz_convert("America/New_York")
    dates = local.tz_localize(None).normalize()
    dates += pd.to_timedelta((local.hour >= 17).astype(int), unit="D")
    return (dates + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("UTC")
