from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.errors import TradingMLError
from trading_ml.online import service


def _candles(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(212)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, rows)))
    opening = np.r_[100.0, close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC"),
            "open": opening,
            "high": np.maximum(opening, close) * 1.002,
            "low": np.minimum(opening, close) * 0.998,
            "close": close,
            "volume": rng.uniform(10, 20, rows),
        }
    )


def _config(tmp_path: Path, *, horizon: int = 1, warmup: int = 32) -> Path:
    path = tmp_path / "online.toml"
    path.write_text(
        f"""[market]
symbol = "BTCUSDT"
interval = "1h"
horizon = {horizon}

[online]
state_path = "artifacts/online/state.sqlite3"
start_at = "2024-01-01T00:00:00Z"
poll_seconds = 30
warmup_bars = {warmup}
max_prediction_delay_seconds = 120

[evaluation]
eval_window = 50
min_samples = 100
patience = 3
alpha = 0.05
min_effect = 0.001
""",
        encoding="utf-8",
    )
    return path


def _write_source(tmp_path: Path, frame: pd.DataFrame, name: str = "candles.csv") -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _at(hours: int, seconds: int = 0) -> str:
    return (
        pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(hours=hours, seconds=seconds)
    ).isoformat()


def _rows(state: Path, table: str) -> list[dict]:
    assert table in {"bars", "predictions", "models", "meta", "events", "validated_snapshots"}
    with sqlite3.connect(state) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def _payloads(state: Path) -> dict[str, dict]:
    return {row["model_id"]: json.loads(row["payload"]) for row in _rows(state, "models")}


def _probabilities(state: Path) -> dict[tuple[str, str], float]:
    return {
        (row["model_id"], row["reference_time"]): row["probability"]
        for row in _rows(state, "predictions")
    }


def test_replay_restart_is_idempotent_and_learns_each_confirmation_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles())
    state = tmp_path / "state.sqlite3"
    service.init_state(config, state)
    service.run_once(config, state, source=source, replay=True, now=_at(48))

    before_predictions = _rows(state, "predictions")
    before_models = _payloads(state)
    assert before_predictions
    assert {row["mode"] for row in before_predictions} == {"replay"}
    assert any(row["label"] is not None for row in before_predictions)
    for model_id, model in before_models.items():
        confirmed = sum(
            row["model_id"] == model_id and row["status"] == "evaluated"
            for row in before_predictions
        )
        assert model["updates"] == confirmed

    service.run_once(config, state, source=source, replay=True, now=_at(48))
    assert _rows(state, "predictions") == before_predictions
    assert _payloads(state) == before_models

    service.run_once(config, state, source=source, replay=True, now=_at(49))
    after = _rows(state, "predictions")
    probabilities = _probabilities(state)
    assert len(after) > len(before_predictions)
    assert all(
        probabilities[(row["model_id"], row["reference_time"])] == row["probability"]
        for row in before_predictions
    )
    assert len(probabilities) == len(after)
    for model_id, model in _payloads(state).items():
        assert model["updates"] == sum(
            row["model_id"] == model_id and row["status"] == "evaluated" for row in after
        )


def test_open_candles_and_future_values_cannot_affect_current_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = _candles()
    altered = original.copy()
    altered.loc[40:, ["open", "high", "low", "close"]] *= 7.0
    source = _write_source(tmp_path, original)
    future_source = _write_source(tmp_path, altered, "future.csv")
    left, right = tmp_path / "left.sqlite3", tmp_path / "right.sqlite3"

    service.run_once(config, left, source=source, replay=True, now=_at(40, 30))
    service.run_once(config, right, source=future_source, replay=True, now=_at(40, 30))
    assert _probabilities(left) == _probabilities(right)
    assert _payloads(left) == _payloads(right)
    bars = _rows(left, "bars")
    assert len(bars) == 40
    assert max(pd.Timestamp(row["timestamp"]) for row in bars) == pd.Timestamp(_at(39))
    predictions = _rows(left, "predictions")
    assert predictions
    assert all(pd.Timestamp(row["reference_time"]) <= pd.Timestamp(_at(40)) for row in predictions)
    assert all(
        row["label"] is None
        for row in predictions
        if pd.Timestamp(row["target_time"]) > pd.Timestamp(_at(40, 30))
    )


def test_replay_full_history_matches_forecasts_from_shorter_causal_prefix(tmp_path: Path) -> None:
    config = _config(tmp_path, horizon=3)
    source = _write_source(tmp_path, _candles())
    prefix_state = tmp_path / "prefix.sqlite3"
    full_state = tmp_path / "full.sqlite3"
    service.run_once(config, prefix_state, source=source, replay=True, now=_at(45))
    service.run_once(config, full_state, source=source, replay=True, now=_at(80))
    prefix = _probabilities(prefix_state)
    full = _probabilities(full_state)
    assert prefix
    assert all(full[key] == probability for key, probability in prefix.items())


def test_target_requires_exact_closed_bar_and_never_uses_next_available_price(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, horizon=3)
    frame = _candles()
    source = _write_source(tmp_path, frame)
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, replay=True, now=_at(40))
    target_rows = [
        row
        for row in _rows(state, "predictions")
        if pd.Timestamp(row["reference_time"]) == pd.Timestamp(_at(40))
    ]
    assert target_rows
    assert all(row["label"] is None for row in target_rows)

    # The target close is hour 43, so its candle must open at hour 42.
    source = _write_source(tmp_path, frame.drop(index=42))
    service.run_once(config, state, source=source, replay=True, now=_at(48))
    unresolved = [
        row
        for row in _rows(state, "predictions")
        if pd.Timestamp(row["reference_time"]) == pd.Timestamp(_at(40))
    ]
    assert all(row["label"] is None for row in unresolved)
    assert all(row["status"] != "evaluated" for row in unresolved)
    assert all(
        row["label"] is None
        for row in _rows(state, "predictions")
        if row["status"] == "invalid_gap"
    )


def test_gap_invalidates_crossing_horizons_and_restarts_feature_warmup(tmp_path: Path) -> None:
    config = _config(tmp_path, horizon=3, warmup=32)
    frame = _candles(100).drop(index=40)
    source = _write_source(tmp_path, frame)
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, replay=True, now=_at(90))
    predictions = _rows(state, "predictions")
    assert predictions
    assert any(row["status"] == "invalid_gap" for row in predictions)
    assert all(row["label"] is None for row in predictions if row["status"] == "invalid_gap")
    # Hour 41 is the last close before the missing bar; fresh features need
    # 32 consecutive candles starting at open 41 (first possible close 73).
    assert not any(
        pd.Timestamp(_at(41)) < pd.Timestamp(row["reference_time"]) < pd.Timestamp(_at(73))
        for row in predictions
    )
    assert any(pd.Timestamp(row["reference_time"]) >= pd.Timestamp(_at(73)) for row in predictions)


def test_replay_state_cannot_be_reused_as_live(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles())
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, replay=True, now=_at(40))
    before = _rows(state, "predictions"), _payloads(state)
    with pytest.raises(TradingMLError):
        service.run_once(config, state, source=source, replay=False, now=_at(41))
    assert (_rows(state, "predictions"), _payloads(state)) == before


def test_live_predictions_are_observed_before_learning_and_outages_do_not_backfill(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles())
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, now=_at(40, 30))
    first = _rows(state, "predictions")
    assert first
    assert {row["mode"] for row in first} == {"live"}
    assert {pd.Timestamp(row["reference_time"]) for row in first} == {pd.Timestamp(_at(40))}
    assert all(row["label"] is None for row in first)
    assert all(model["updates"] == 0 for model in _payloads(state).values())

    service.run_once(config, state, source=source, now=_at(41, 30))
    second = _rows(state, "predictions")
    assert all(
        row["status"] == "evaluated"
        for row in second
        if pd.Timestamp(row["reference_time"]) == pd.Timestamp(_at(40))
    )
    assert all(model["updates"] == 1 for model in _payloads(state).values())

    # Restarting after four hours evaluates pending predictions, but cannot
    # invent predictions for the three closes nobody actually observed.
    service.run_once(config, state, source=source, now=_at(45, 30))
    assert {pd.Timestamp(row["reference_time"]) for row in _rows(state, "predictions")} == {
        pd.Timestamp(_at(40)),
        pd.Timestamp(_at(41)),
        pd.Timestamp(_at(45)),
    }
    assert all(model["updates"] == 2 for model in _payloads(state).values())


def test_stale_live_snapshot_produces_no_prediction_or_training(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles(40))
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, now=_at(40, 180))
    assert _rows(state, "predictions") == []
    assert all(model["updates"] == 0 for model in _payloads(state).values())


def test_live_cannot_emit_after_target_deadline_even_inside_delay_tolerance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace('interval = "1h"', 'interval = "1m"'),
        encoding="utf-8",
    )
    frame = _candles(40)
    frame["timestamp"] = pd.date_range("2024-01-01", periods=40, freq="1min", tz="UTC")
    source = _write_source(tmp_path, frame)
    state = tmp_path / "state.sqlite3"
    # The latest known close is 00:40. Its 1m target has already ended at
    # 00:41, although a 90s delay is inside the configured tolerance of 120s.
    service.run_once(config, state, source=source, now="2024-01-01T00:41:30Z")
    assert _rows(state, "predictions") == []
    service.verify(state)


def test_backup_restore_recovers_models_and_pending_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles())
    state = tmp_path / "state.sqlite3"
    saved = tmp_path / "backup.sqlite3"
    recovered = tmp_path / "recovered.sqlite3"
    service.run_once(config, state, source=source, replay=True, now=_at(45))
    service.backup(state, saved)
    service.restore(saved, recovered)
    assert _rows(recovered, "predictions") == _rows(state, "predictions")
    assert _payloads(recovered) == _payloads(state)
    service.verify(recovered)

    with pytest.raises((TradingMLError, FileExistsError)):
        service.restore(saved, recovered)
    service.run_once(config, recovered, source=source, replay=True, now=_at(46))
    service.run_once(config, state, source=source, replay=True, now=_at(46))
    assert _rows(recovered, "predictions") == _rows(state, "predictions")
    assert _payloads(recovered) == _payloads(state)


def test_malformed_closed_candle_rejects_cycle_without_partial_learning(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = _candles()
    source = _write_source(tmp_path, frame)
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, replay=True, now=_at(40))
    before = _rows(state, "predictions"), _payloads(state), _rows(state, "bars")
    frame.loc[41, "close"] = -1
    source = _write_source(tmp_path, frame)
    with pytest.raises(TradingMLError):
        service.run_once(config, state, source=source, replay=True, now=_at(45))
    assert (_rows(state, "predictions"), _payloads(state), _rows(state, "bars")) == before


def test_changed_experiment_contract_cannot_silently_reuse_weights(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles())
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, replay=True, now=_at(40))
    _config(tmp_path, horizon=2)
    before = _rows(state, "predictions"), _payloads(state)
    with pytest.raises(TradingMLError):
        service.run_once(config, state, source=source, replay=True, now=_at(41))
    assert (_rows(state, "predictions"), _payloads(state)) == before


@pytest.mark.parametrize("table", ["predictions", "models", "bars"])
def test_verify_detects_modified_prediction_model_or_market_state(
    tmp_path: Path, table: str
) -> None:
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles())
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, replay=True, now=_at(40))
    with sqlite3.connect(state) as connection:
        if table == "predictions":
            connection.execute("UPDATE predictions SET probability=0.123 WHERE rowid=1")
        elif table == "bars":
            connection.execute("UPDATE bars SET close=close*1.001 WHERE rowid=1")
        else:
            payload = json.loads(
                connection.execute("SELECT payload FROM models LIMIT 1").fetchone()[0]
            )
            payload["updates"] += 1
            connection.execute("UPDATE models SET payload=? WHERE rowid=1", (json.dumps(payload),))
    with pytest.raises(TradingMLError):
        service.verify(state)
    before = _rows(state, "predictions"), _rows(state, "models"), _rows(state, "bars")
    with pytest.raises(TradingMLError):
        service.run_once(config, state, source=source, replay=True, now=_at(41))
    assert (_rows(state, "predictions"), _rows(state, "models"), _rows(state, "bars")) == before


def test_training_failure_rolls_back_whole_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles())
    state = tmp_path / "state.sqlite3"
    service.run_once(config, state, source=source, replay=True, now=_at(40))
    before = {
        table: _rows(state, table) for table in ("predictions", "models", "bars", "meta", "events")
    }
    original_learn = service.learn
    learned = []

    def fail_after_first_update(model: dict, values: list[float], label: int) -> None:
        original_learn(model, values, label)
        learned.append(True)
        raise RuntimeError("injected failure after updating an in-memory model")

    monkeypatch.setattr(service, "learn", fail_after_first_update)
    with pytest.raises(RuntimeError, match="injected failure"):
        service.run_once(config, state, source=source, replay=True, now=_at(43))
    assert learned == [True]
    assert {table: _rows(state, table) for table in before} == before
    monkeypatch.setattr(service, "learn", original_learn)
    service.run_once(config, state, source=source, replay=True, now=_at(43))
    service.verify(state)


def test_synthetic_validation_decision_preserves_exact_snapshot_through_later_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise storage after an injected decision, without claiming real validation."""
    config = _config(tmp_path)
    source = _write_source(tmp_path, _candles())
    state = tmp_path / "state.sqlite3"
    emitted = []

    def synthetic_validation(predictions: list[dict], models: dict, policy: dict) -> list[dict]:
        if emitted or not predictions:
            return []
        emitted.append(True)
        candidate = "logistic_slow"
        models[candidate]["evaluation"] = {
            "validation_state": "validated_predictive",
            "last_target": predictions[-1]["target_time"],
            "look": 1,
            "worse_streak": 0,
            "better_streak": 3,
            "total_pairs": 100,
        }
        return [
            {
                "candidate": candidate,
                "baseline": "baseline",
                "decision": "validated_predictive",
                "validation_state": "validated_predictive",
                "profitability_validated": False,
                "test_only_injected_decision": True,
            }
        ]

    monkeypatch.setattr(service, "evaluate_models", synthetic_validation)
    service.run_once(config, state, source=source, replay=True, now=_at(40))
    snapshots = _rows(state, "validated_snapshots")
    assert len(snapshots) == 1
    frozen = json.loads(snapshots[0]["payload"])
    assert frozen["updates"] < _payloads(state)["logistic_slow"]["updates"]
    assert json.loads(snapshots[0]["decision"])["test_only_injected_decision"]

    service.run_once(config, state, source=source, replay=True, now=_at(45))
    assert _rows(state, "validated_snapshots") == snapshots
    assert frozen["params"] != _payloads(state)["logistic_slow"]["params"]
    service.verify(state)

    saved, restored = tmp_path / "saved.sqlite3", tmp_path / "restored.sqlite3"
    service.backup(state, saved)
    service.restore(saved, restored)
    assert _rows(restored, "validated_snapshots") == snapshots
    with sqlite3.connect(restored) as connection:
        connection.execute("UPDATE validated_snapshots SET payload='{}' WHERE id=1")
    with pytest.raises(TradingMLError):
        service.verify(restored)
