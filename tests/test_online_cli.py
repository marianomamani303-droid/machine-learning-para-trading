from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import requests

from trading_ml.cli import main, parser
from trading_ml.online import cli


@pytest.fixture
def fake_service(monkeypatch):
    module = ModuleType("trading_ml.online.service")
    monkeypatch.setitem(sys.modules, "trading_ml.online.service", module)
    import trading_ml.online

    monkeypatch.setattr(trading_ml.online, "service", module, raising=False)
    return module


def _lines(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_online_uses_own_config_and_never_loads_historical_settings(
    monkeypatch, fake_service, capsys, tmp_path
):
    config = tmp_path / "online.toml"
    state = tmp_path / "state.sqlite3"
    calls = []

    def init_state(config_path, state_path):
        calls.append((config_path, state_path))
        return {"event": "initialized", "state": str(state_path)}

    fake_service.init_state = init_state

    def reject_load(*_args):
        pytest.fail("online no debe leer config/default.toml")

    monkeypatch.setattr("trading_ml.cli.load_settings", reject_load)
    assert main(["online", "init", "--config", str(config), "--state", str(state)]) == 0
    assert calls == [(config, state)]
    assert _lines(capsys)[0]["event"] == "initialized"


def test_common_options_accept_both_positions_without_losing_values():
    args = parser().parse_args(
        ["online", "--config", "custom.toml", "--state", "custom.db", "status"]
    )
    assert args.online_config == Path("custom.toml")
    assert args.state == Path("custom.db")
    args = parser().parse_args(["online", "status", "--state", "another.db"])
    assert args.state == Path("another.db")


def test_state_defaults_to_path_declared_by_online_protocol(fake_service, capsys, tmp_path):
    protocol = tmp_path / "custom.toml"
    state = tmp_path / "custom.sqlite3"
    protocol.write_text('[online]\nstate_path = "' + state.as_posix() + '"\n', encoding="utf-8")
    fake_service.status = lambda path: {"state": str(path)}
    assert main(["online", "status", "--config", str(protocol)]) == 0
    assert _lines(capsys)[0]["state"] == str(state)


def test_predictions_exposes_stored_probabilities_and_passes_filters(
    fake_service, capsys, tmp_path
):
    state = tmp_path / "predictions.sqlite3"
    calls = []
    forecast = {
        "model_id": "logistic_slow",
        "reference_time": "2026-10-28T01:00:00Z",
        "target_time": "2026-10-28T02:00:00Z",
        "probability_up": 0.63,
        "probability_down_or_flat": 0.37,
        "status": "evaluated",
        "label": 1,
    }

    def predictions(path, **kwargs):
        calls.append((path, kwargs))
        return {"mode": "live", "forecasts": [forecast]}

    fake_service.predictions = predictions
    assert (
        main(
            [
                "online",
                "predictions",
                "--state",
                str(state),
                "--limit",
                "2",
                "--status",
                "evaluated",
                "--model-id",
                "logistic_slow",
            ]
        )
        == 0
    )
    assert calls == [
        (state, {"limit": 2, "status_filter": "evaluated", "model_id": "logistic_slow"})
    ]
    assert _lines(capsys)[0] == {"mode": "live", "forecasts": [forecast]}


def test_predictions_defaults_to_twenty_without_filters(fake_service, capsys):
    calls = []

    def predictions(_path, **kwargs):
        calls.append(kwargs)
        return {"forecasts": [{"probability_up": 0.5, "label": None, "status": "pending"}]}

    fake_service.predictions = predictions
    assert main(["online", "predictions"]) == 0
    assert calls == [{"limit": 20, "status_filter": None, "model_id": None}]
    assert _lines(capsys)[0]["forecasts"][0]["label"] is None


@pytest.mark.parametrize("value", ["0", "-1", "1001"])
def test_predictions_rejects_out_of_range_limit(value):
    with pytest.raises(SystemExit) as failure:
        parser().parse_args(["online", "predictions", "--limit", value])
    assert failure.value.code == 2


def test_predictions_rejects_unknown_status():
    with pytest.raises(SystemExit) as failure:
        parser().parse_args(["online", "predictions", "--status", "unconfirmed"])
    assert failure.value.code == 2


def test_run_once_reports_network_failure_without_hiding_it(fake_service, capsys):
    def run_once(*_args, **_kwargs):
        raise requests.ConnectionError("provider unavailable")

    fake_service.run_once = run_once
    assert main(["online", "run", "--once", "--poll-seconds", "1"]) == 1
    result = _lines(capsys)
    assert result[0]["event"] == "cycle_error"
    assert result[0]["recoverable"] is True


def test_continuous_run_retries_transient_errors_and_outputs_json_per_cycle(
    fake_service, monkeypatch, capsys
):
    calls = []
    sleeps = []

    def run_once(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise requests.Timeout("request timed out")
        return {"event": "cycle", "predictions": len(calls)}

    fake_service.run_once = run_once
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    assert main(["online", "run", "--cycles", "3", "--poll-seconds", "2"]) == 0
    result = _lines(capsys)
    assert [item["event"] for item in result] == ["cycle_error", "cycle", "cycle"]
    assert sleeps == [4.0, 2.0]


def test_invalid_state_error_stops_without_retry(fake_service, monkeypatch, capsys):
    def run_once(*_args, **_kwargs):
        raise ValueError("state contract mismatch")

    fake_service.run_once = run_once
    monkeypatch.setattr(cli.time, "sleep", lambda _: pytest.fail("invalid state was retried"))
    assert main(["online", "run", "--cycles", "3", "--poll-seconds", "1"]) == 1
    assert _lines(capsys)[0]["recoverable"] is False


def test_ctrl_c_during_polling_exits_cleanly(fake_service, monkeypatch, capsys):
    fake_service.run_once = lambda *_args, **_kwargs: {"event": "cycle"}

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt)
    assert main(["online", "run", "--poll-seconds", "1"]) == 0
    assert _lines(capsys)[-1] == {"event": "stopped", "reason": "keyboard_interrupt"}


@pytest.mark.parametrize("arguments", [["--replay"], ["--now", "2026-10-28T00:00:00Z"]])
def test_replay_requires_explicit_source_and_live_disallows_simulated_clock(
    arguments, fake_service, capsys
):
    fake_service.run_once = lambda *_args, **_kwargs: pytest.fail("invalid flags reached engine")
    assert main(["online", "run", "--once", *arguments]) == 1
    assert _lines(capsys)[0]["event"] == "error"


def test_verification_failure_produces_nonzero_exit(fake_service, capsys):
    fake_service.verify = lambda _path: {"valid": False, "errors": ["invalid state"]}
    assert main(["online", "verify"]) == 1
    assert _lines(capsys)[0]["valid"] is False


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_polling_rejects_nonpositive_or_nonfinite_values(value):
    with pytest.raises(SystemExit) as failure:
        parser().parse_args(["online", "run", "--poll-seconds", value])
    assert failure.value.code == 2


def test_web_application_is_not_exposed_by_cli():
    with pytest.raises(SystemExit) as failure:
        parser().parse_args(["dashboard"])
    assert failure.value.code == 2


def test_catalog_filters_local_models_without_requiring_online_state(
    monkeypatch, fake_service, capsys, tmp_path
):
    from trading_ml.dashboard.models_catalog import ModelDescriptor

    values = {name: "metadata" for name in ModelDescriptor.__dataclass_fields__}
    values.update(model_id="historical-1", symbol="BTCUSDT", interval="1d", is_deployable=True)
    values["bundle"] = object()  # A loaded model must never be sent to JSON output.
    descriptor = SimpleNamespace(**values)
    monkeypatch.setattr(
        "trading_ml.dashboard.models_catalog.discover_all_models",
        lambda root, **_kwargs: [descriptor],
    )
    assert (
        main(
            ["online", "catalog", "--config", str(tmp_path / "absent.toml"), "--symbol", "BTCUSDT"]
        )
        == 0
    )
    result = _lines(capsys)[0]
    assert result["models"][0]["model_id"] == "historical-1"
    assert "bundle" not in result["models"][0]


def test_historical_inference_is_read_only_and_exposes_individual_model_errors(
    monkeypatch, fake_service, capsys, tmp_path
):
    descriptors = [
        SimpleNamespace(model_id=name, symbol="BTCUSDT", interval="1d")
        for name in ["first", "second"]
    ]
    monkeypatch.setattr(
        "trading_ml.dashboard.models_catalog.discover_all_models",
        lambda *_args, **_kwargs: descriptors,
    )
    calls = []

    def infer(models, candles, **kwargs):
        calls.append((models, candles, kwargs))
        return [
            SimpleNamespace(
                descriptor=descriptors[0], result=None, error="cutoff guard", succeeded=False
            ),
            SimpleNamespace(
                descriptor=descriptors[1], result=None, error="missing peer", succeeded=False
            ),
        ]

    monkeypatch.setattr("trading_ml.dashboard.inference_engine.run_batch_inference", infer)
    source = tmp_path / "candles.csv"
    source.write_text(
        "timestamp,open,high,low,close,volume\n2020-01-01,1,2,1,2,4\n", encoding="utf-8"
    )
    assert (
        main(
            [
                "online",
                "infer",
                "--model-id",
                "first",
                "--model-id",
                "second",
                "--source",
                str(source),
            ]
        )
        == 1
    )
    assert calls[0][2]["use_cache"] is False
    result = _lines(capsys)[0]
    assert result["evidence"] == "read_only_snapshot_not_prospective_validation"
    assert [item["error"] for item in result["results"]] == ["cutoff guard", "missing peer"]
