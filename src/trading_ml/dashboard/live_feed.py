from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from trading_ml.config.settings import INTERVALS
from trading_ml.errors import DataError
from trading_ml.preprocessing.pipeline import clean_ohlcv

LOGGER = logging.getLogger(__name__)

REST_ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
]
WS_ENDPOINTS = [
    "wss://stream.binance.com:9443/ws",
    "wss://data-stream.binance.vision/ws",
]

FOREX_SYMBOLS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"}

KLINE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
]


@dataclass(frozen=True)
class DataQualityReport:
    """Diagnóstico de integridad, huecos, duplicados y sincronización temporal de las velas."""

    closed_bars_count: int
    duplicates_removed: int
    missing_bars_count: int
    missing_timestamps_sample: list[str]
    first_closed_open_utc: str
    last_closed_open_utc: str
    last_closed_reference_utc: str
    expected_next_close_utc: str
    staleness_hours: float
    clock_sync_ok: bool
    quality_status: str
    notes: list[str] = field(default_factory=list)


@dataclass
class MarketSnapshot:
    """Instantánea separando estrictamente velas cerradas de la vela en formación."""

    symbol: str
    interval: str
    closed_candles: pd.DataFrame
    forming_candle: dict[str, Any] | None
    live_price: float | None
    last_update_utc: str
    connection_status: str
    ws_connected: bool
    ws_messages_received: int
    data_source: str
    quality: DataQualityReport | None = None
    errors: list[str] = field(default_factory=list)


def assess_data_quality(
    closed_candles: pd.DataFrame,
    symbol: str,
    interval: str,
    *,
    duplicates_removed: int = 0,
    now_utc: pd.Timestamp | None = None,
) -> DataQualityReport:
    """Evalúa huecos temporales, duplicados y sincronización del historial de velas cerradas."""
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    if closed_candles.empty or interval not in INTERVALS:
        return DataQualityReport(
            closed_bars_count=0,
            duplicates_removed=duplicates_removed,
            missing_bars_count=0,
            missing_timestamps_sample=[],
            first_closed_open_utc="",
            last_closed_open_utc="",
            last_closed_reference_utc="",
            expected_next_close_utc="",
            staleness_hours=0.0,
            clock_sync_ok=False,
            quality_status="SIN DATOS",
            notes=["No hay velas cerradas disponibles."],
        )

    delta = pd.Timedelta(INTERVALS[interval])
    ts = pd.to_datetime(closed_candles["timestamp"], utc=True).sort_values().reset_index(drop=True)
    first_open = pd.Timestamp(ts.iloc[0])
    last_open = pd.Timestamp(ts.iloc[-1])
    last_ref = last_open + delta
    next_close = last_ref + delta

    is_forex = symbol.upper() in FOREX_SYMBOLS
    missing_sample: list[str] = []
    missing_count = 0

    if not is_forex:
        expected_grid = pd.date_range(first_open, last_open, freq=delta, tz="UTC")
        missing_idx = expected_grid.difference(pd.DatetimeIndex(ts))
        missing_count = len(missing_idx)
        missing_sample = [t.isoformat() for t in missing_idx[:5]]
    else:
        diffs = ts.diff().dropna()
        # En Forex diario, saltos <= 3 días corresponden al fin de semana; > 4 días son huecos
        large_gaps = ts.iloc[1:].loc[diffs > pd.Timedelta(days=4)]
        missing_count = len(large_gaps)
        missing_sample = [t.isoformat() for t in large_gaps[:5]]

    staleness_hours = max(0.0, float((now - last_ref).total_seconds() / 3600.0))
    max_allowed_hours = (delta.total_seconds() / 3600.0) * (3.5 if is_forex else 1.25)
    clock_sync_ok = staleness_hours <= max_allowed_hours

    notes: list[str] = []
    if duplicates_removed > 0:
        notes.append(f"Se deduplicaron {duplicates_removed} registros repetidos.")
    if missing_count > 0:
        notes.append(f"Se detectaron {missing_count} huecos en la grilla temporal.")
    if not clock_sync_ok:
        notes.append(
            f"Última vela cerrada hace {staleness_hours:.1f} h (posible retraso o mercado cerrado)."
        )

    if missing_count == 0 and clock_sync_ok:
        status = "ÓPTIMA (Sin huecos ni duplicados; sincronizada)"
    elif clock_sync_ok:
        status = f"ACEPTABLE ({missing_count} hueco(s) detectado(s))"
    else:
        status = "HISTÓRICA / DESFASADA (Última vela cerrada no es la actual)"

    return DataQualityReport(
        closed_bars_count=len(ts),
        duplicates_removed=duplicates_removed,
        missing_bars_count=missing_count,
        missing_timestamps_sample=missing_sample,
        first_closed_open_utc=first_open.isoformat(),
        last_closed_open_utc=last_open.isoformat(),
        last_closed_reference_utc=last_ref.isoformat(),
        expected_next_close_utc=next_close.isoformat(),
        staleness_hours=round(staleness_hours, 2),
        clock_sync_ok=clock_sync_ok,
        quality_status=status,
        notes=notes,
    )


def parse_rest_klines(
    raw_klines: list[list[Any]],
    interval: str,
    now_utc: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Separa velas cerradas (`timestamp + delta <= now_utc`) de la vela en formación."""
    if interval not in INTERVALS:
        raise DataError(f"Intervalo no soportado: {interval}")
    delta = pd.Timedelta(INTERVALS[interval])
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    if not raw_klines:
        raise DataError("Respuesta vacía de klines")

    frame = pd.DataFrame(raw_klines, columns=KLINE_COLUMNS)
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(pd.to_numeric(frame["close_time"]), unit="ms", utc=True)
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)

    raw_len = len(frame)
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    duplicates_removed = raw_len - len(frame)

    is_closed_mask = (frame["timestamp"] + delta) <= now
    closed_raw = frame.loc[is_closed_mask].reset_index(drop=True)
    forming_raw = frame.loc[~is_closed_mask].reset_index(drop=True)

    if closed_raw.empty:
        raise DataError("No hay velas cerradas disponibles")

    cleaned, _ = clean_ohlcv(
        closed_raw[["timestamp", "open", "high", "low", "close", "volume"]],
        delta,
    )
    extra_cols = ["quote_volume", "trades", "taker_base", "taker_quote"]
    closed_merged = cleaned.merge(
        closed_raw[["timestamp", *extra_cols]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    closed_merged["reference_time"] = closed_merged["timestamp"] + delta
    closed_merged["is_closed"] = True
    closed_merged.attrs["duplicates_removed"] = duplicates_removed

    forming_dict: dict[str, Any] | None = None
    if not forming_raw.empty:
        last_open = forming_raw.iloc[-1]
        forming_dict = {
            "timestamp": pd.Timestamp(last_open["timestamp"]).isoformat(),
            "expected_close_time": (pd.Timestamp(last_open["timestamp"]) + delta).isoformat(),
            "open": float(last_open["open"]),
            "high": float(last_open["high"]),
            "low": float(last_open["low"]),
            "close": float(last_open["close"]),
            "volume": float(last_open["volume"]),
            "quote_volume": float(last_open["quote_volume"]),
            "trades": float(last_open["trades"]),
            "taker_base": float(last_open["taker_base"]),
            "taker_quote": float(last_open["taker_quote"]),
            "is_closed": False,
        }

    return closed_merged, forming_dict


def load_local_fallback_candles(
    root: Path,
    symbol: str,
    interval: str,
    limit: int = 850,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Carga datos históricos locales cerrados para Cripto o Forex."""
    if interval not in INTERVALS:
        raise DataError(f"Intervalo no soportado: {interval}")
    delta = pd.Timedelta(INTERVALS[interval])
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)

    sym = symbol.upper()
    candidates = [
        root / "historical" / "datasets" / "crypto" / sym / f"{interval}.csv",
        root / "historical" / "datasets" / "forex" / sym / f"{interval}.csv",
        root / "historical" / "fxcm_native_daily" / f"{sym}_1d.csv",
        root / "frontier_studies" / "20261014T134646_e17d06e3" / f"{sym}_causal_features.csv",
        root / "research_data" / "ETHUSDT_20200101_20261001.csv" if sym == "ETHUSDT" else Path("__none__"),
        root / "data" / "raw.csv" if sym == "BTCUSDT" else Path("__none__"),
    ]
    mm_data_dir = root / "multimarket" / "data" / sym
    if mm_data_dir.exists():
        candidates.extend(sorted(mm_data_dir.glob(f"{interval}_*.csv"), reverse=True))
        if (mm_data_dir / interval).exists():
            candidates.extend(sorted((mm_data_dir / interval).glob("*.csv"), reverse=True))

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if "volume" not in df.columns:
                df["volume"] = 1.0
            df = df.loc[df["timestamp"] + delta <= now].sort_values("timestamp").tail(limit)
            if df.empty:
                continue
            if sym in FOREX_SYMBOLS:
                ohlcv_cols = ["open", "high", "low", "close", "volume"]
                cleaned = df[["timestamp", *ohlcv_cols]].copy()
                cleaned[ohlcv_cols] = cleaned[ohlcv_cols].apply(pd.to_numeric, errors="coerce")
                cleaned = (
                    cleaned.dropna(subset=["timestamp", *ohlcv_cols])
                    .drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                if len(cleaned) < 2:
                    continue
            else:
                cleaned, _ = clean_ohlcv(
                    df[["timestamp", "open", "high", "low", "close", "volume"]], delta
                )
            for col in ["quote_volume", "trades", "taker_base", "taker_quote"]:
                if col in df.columns:
                    cleaned = cleaned.merge(
                        df[["timestamp", col]].drop_duplicates("timestamp", keep="last"),
                        on="timestamp",
                        how="left",
                    )
                else:
                    cleaned[col] = cleaned["volume"] * cleaned["close"] if "quote" in col else 1.0
            for fx_col in ["close_bid", "close_ask", "crossed_close"]:
                if fx_col in df.columns:
                    cleaned = cleaned.merge(
                        df[["timestamp", fx_col]].drop_duplicates("timestamp", keep="last"),
                        on="timestamp",
                        how="left",
                    )
            cleaned["reference_time"] = cleaned["timestamp"] + delta
            cleaned["is_closed"] = True
            return cleaned.reset_index(drop=True)
        except Exception:
            continue
    raise DataError(f"Sin datos locales de respaldo para {symbol} ({interval})")


def fetch_binance_klines(
    symbol: str,
    interval: str,
    limit: int = 850,
    timeout: float = 8.0,
    now_utc: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Obtiene klines desde Binance REST API y separa velas cerradas de la vela en formación."""
    if symbol.upper() in FOREX_SYMBOLS:
        raise DataError(f"{symbol} es un par Forex; no pertenece a Binance Spot.")
    last_exc: Exception | None = None
    for endpoint in REST_ENDPOINTS:
        try:
            resp = requests.get(
                endpoint,
                params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            closed, forming = parse_rest_klines(payload, interval=interval, now_utc=now_utc)
            return closed, forming, endpoint
        except Exception as exc:
            last_exc = exc
    raise DataError(f"Error consultando REST de Binance ({symbol} {interval}): {last_exc}")


class LiveStreamWorker:
    """Trabajador por (symbol, interval) con bootstrap REST + stream WebSocket en segundo plano."""

    def __init__(self, root: Path, symbol: str, interval: str, limit: int = 850) -> None:
        self.root = root
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit
        self.delta = pd.Timedelta(INTERVALS[interval])

        self._lock = threading.Lock()
        self._closed_candles = pd.DataFrame()
        self._forming_candle: dict[str, Any] | None = None
        self._live_price: float | None = None
        self._last_update_utc: str = datetime.now(UTC).isoformat()
        self._connection_status: str = "INICIALIZANDO"
        self._ws_connected: bool = False
        self._ws_messages: int = 0
        self._data_source: str = "Pendiente"
        self._duplicates_removed: int = 0
        self._errors: list[str] = []
        self._stop_event = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._ws_app: Any = None
        self._last_rest_sync: float = 0.0

    def _record_error(self, message: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S UTC")
        entry = f"[{stamp}] {message}"
        if not self._errors or self._errors[-1] != entry:
            self._errors.append(entry)
            if len(self._errors) > 10:
                self._errors = self._errors[-10:]

    def sync_rest(self, force: bool = False) -> None:
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_rest_sync) < 15.0 and not self._closed_candles.empty:
            return
        if self.symbol in FOREX_SYMBOLS:
            try:
                fallback = load_local_fallback_candles(
                    self.root, self.symbol, self.interval, limit=self.limit
                )
                with self._lock:
                    self._closed_candles = fallback
                    self._forming_candle = None
                    self._live_price = float(fallback.iloc[-1]["close"])
                    self._last_update_utc = datetime.now(UTC).isoformat()
                    self._last_rest_sync = now_mono
                    self._connection_status = "DATASET VERIFICADO FOREX (Snapshot local; sin WebSocket Binance)"
                    self._data_source = "Repositorio Histórico Forex Verificado (Dukascopy/FXCM)"
            except Exception as exc:
                with self._lock:
                    self._connection_status = "SIN DATOS FOREX"
                    self._record_error(str(exc))
            return

        try:
            closed, forming, endpoint = fetch_binance_klines(
                self.symbol, self.interval, limit=self.limit
            )
            with self._lock:
                self._closed_candles = closed
                self._duplicates_removed += int(closed.attrs.get("duplicates_removed", 0))
                if forming is not None:
                    self._forming_candle = forming
                    self._live_price = float(forming["close"])
                elif not closed.empty:
                    self._live_price = float(closed.iloc[-1]["close"])
                self._last_update_utc = datetime.now(UTC).isoformat()
                self._last_rest_sync = now_mono
                if self._ws_connected:
                    self._connection_status = "CONECTADO (WebSocket en tiempo real)"
                else:
                    self._connection_status = "CONECTADO (REST API en vivo; iniciando WebSocket)"
                self._data_source = f"Binance Live ({endpoint.split('//')[1].split('/')[0]})"
        except Exception as exc:
            with self._lock:
                self._record_error(str(exc))
                if self._closed_candles.empty:
                    try:
                        fallback = load_local_fallback_candles(
                            self.root, self.symbol, self.interval, limit=self.limit
                        )
                        self._closed_candles = fallback
                        self._forming_candle = None
                        self._live_price = float(fallback.iloc[-1]["close"])
                        self._connection_status = (
                            "MODO LOCAL (Sin conexión a Binance; usando histórico local cerrado)"
                        )
                        self._data_source = "Respaldo histórico local (artifacts/)"
                    except Exception as fb_exc:
                        self._connection_status = "ERROR DE CONEXIÓN"
                        self._record_error(str(fb_exc))

    def handle_ws_kline_message(
        self, payload: dict[str, Any], now_utc: pd.Timestamp | None = None
    ) -> bool:
        """Procesa un evento kline de WebSocket. Devuelve True si se cerró una vela nueva."""
        k = payload.get("k")
        if not isinstance(k, dict):
            return False

        open_ts = pd.to_datetime(int(k["t"]), unit="ms", utc=True)
        expected_close = open_ts + self.delta
        now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
        is_closed = bool(k.get("x", False)) and (expected_close <= now + pd.Timedelta(seconds=2))

        candle_row = {
            "timestamp": open_ts,
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "quote_volume": float(k.get("q", 0.0)),
            "trades": float(k.get("n", 0.0)),
            "taker_base": float(k.get("V", 0.0)),
            "taker_quote": float(k.get("Q", 0.0)),
            "reference_time": expected_close,
            "is_closed": is_closed,
        }

        closed_promoted = False
        with self._lock:
            self._ws_messages += 1
            self._ws_connected = True
            self._connection_status = "CONECTADO (WebSocket en tiempo real)"
            self._live_price = candle_row["close"]
            self._last_update_utc = datetime.now(UTC).isoformat()

            if is_closed:
                new_df = pd.DataFrame([candle_row])
                if self._closed_candles.empty:
                    self._closed_candles = new_df
                    closed_promoted = True
                else:
                    already_exists = (self._closed_candles["timestamp"] == open_ts).any()
                    if already_exists:
                        self._duplicates_removed += 1
                    combined = pd.concat([self._closed_candles, new_df], ignore_index=True)
                    combined = (
                        combined.sort_values("timestamp")
                        .drop_duplicates(subset=["timestamp"], keep="last")
                        .tail(self.limit)
                        .reset_index(drop=True)
                    )
                    self._closed_candles = combined
                    closed_promoted = not bool(already_exists)
                self._forming_candle = None
            else:
                self._forming_candle = {
                    "timestamp": open_ts.isoformat(),
                    "expected_close_time": expected_close.isoformat(),
                    "open": candle_row["open"],
                    "high": candle_row["high"],
                    "low": candle_row["low"],
                    "close": candle_row["close"],
                    "volume": candle_row["volume"],
                    "quote_volume": candle_row["quote_volume"],
                    "trades": candle_row["trades"],
                    "taker_base": candle_row["taker_base"],
                    "taker_quote": candle_row["taker_quote"],
                    "is_closed": False,
                }
        return closed_promoted

    def start_background_ws(self) -> None:
        if self.symbol in FOREX_SYMBOLS:
            return
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        self._stop_event.clear()
        self._ws_thread = threading.Thread(
            target=self._ws_loop,
            name=f"binance-ws-{self.symbol}-{self.interval}",
            daemon=True,
        )
        self._ws_thread.start()

    def _ws_loop(self) -> None:
        try:
            import websocket  # type: ignore[import-untyped]
        except Exception as exc:
            with self._lock:
                self._record_error(f"websocket-client no disponible: {exc}")
            return

        stream_name = f"{self.symbol.lower()}@kline_{self.interval}"
        endpoint_idx = 0
        backoff = 2.0

        while not self._stop_event.is_set():
            base_ws = WS_ENDPOINTS[endpoint_idx % len(WS_ENDPOINTS)]
            ws_url = f"{base_ws}/{stream_name}"
            host_label = base_ws.split("//")[1].split("/")[0]

            def _on_open(_ws: Any, host: str = host_label) -> None:
                nonlocal backoff
                backoff = 2.0
                with self._lock:
                    self._ws_connected = True
                    self._connection_status = "CONECTADO (WebSocket en tiempo real)"
                    self._data_source = f"Binance WebSocket ({host})"

            def _on_message(_ws: Any, raw_msg: str) -> None:
                try:
                    data = json.loads(raw_msg)
                    self.handle_ws_kline_message(data)
                except Exception as msg_exc:
                    with self._lock:
                        self._record_error(f"Error procesando mensaje WS: {msg_exc}")

            def _on_error(_ws: Any, err: Any) -> None:
                if self._stop_event.is_set():
                    return
                with self._lock:
                    self._ws_connected = False
                    if not self._closed_candles.empty:
                        self._connection_status = "RECONECTANDO WS (Respaldo REST activo)"
                    self._record_error(f"WebSocket aviso: {err}")

            def _on_close(_ws: Any, _code: Any, _msg: Any) -> None:
                with self._lock:
                    self._ws_connected = False

            try:
                self._ws_app = websocket.WebSocketApp(
                    ws_url,
                    on_open=_on_open,
                    on_message=_on_message,
                    on_error=_on_error,
                    on_close=_on_close,
                )
                self._ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as loop_exc:
                with self._lock:
                    self._ws_connected = False
                    self._record_error(f"Excepción en ciclo WS: {loop_exc}")

            if self._stop_event.is_set():
                break
            endpoint_idx += 1
            time.sleep(min(backoff, 15.0))
            backoff = min(backoff * 1.5, 15.0)

    def snapshot(self, ensure_ws: bool = True) -> MarketSnapshot:
        if self._closed_candles.empty or (self.symbol not in FOREX_SYMBOLS and not self._ws_connected):
            self.sync_rest(force=self._closed_candles.empty)
        if ensure_ws and self.symbol not in FOREX_SYMBOLS:
            self.start_background_ws()
        with self._lock:
            closed_copy = self._closed_candles.copy()
            dups = self._duplicates_removed
            quality = assess_data_quality(
                closed_copy,
                self.symbol,
                self.interval,
                duplicates_removed=dups,
            )
            return MarketSnapshot(
                symbol=self.symbol,
                interval=self.interval,
                closed_candles=closed_copy,
                forming_candle=dict(self._forming_candle) if self._forming_candle else None,
                live_price=self._live_price,
                last_update_utc=self._last_update_utc,
                connection_status=self._connection_status,
                ws_connected=self._ws_connected,
                ws_messages_received=self._ws_messages,
                data_source=self._data_source,
                quality=quality,
                errors=list(self._errors),
            )


class LiveFeedManager:
    """Gestor singleton thread-safe para evitar múltiples conexiones duplicadas."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[tuple[str, str], LiveStreamWorker] = {}

    def get_snapshot(
        self,
        root: Path,
        symbol: str,
        interval: str,
        *,
        force_rest: bool = False,
        ensure_ws: bool = True,
    ) -> MarketSnapshot:
        key = (symbol.upper(), interval)
        with self._lock:
            worker = self._workers.get(key)
            if worker is None:
                worker = LiveStreamWorker(root=root, symbol=symbol, interval=interval)
                self._workers[key] = worker
        if force_rest:
            worker.sync_rest(force=True)
        return worker.snapshot(ensure_ws=ensure_ws)


_GLOBAL_MANAGER = LiveFeedManager()


def get_feed_manager() -> LiveFeedManager:
    return _GLOBAL_MANAGER
