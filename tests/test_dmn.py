from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.common import sha256, sidecar, write_json
from trading_ml.config import Settings
from trading_ml.dmn.config import load_protocol, validate_protocol
from trading_ml.dmn.data import FEATURES, build_panel, causal_features, temporal_split
from trading_ml.dmn.portfolio import metrics, simulate, target_weights
from trading_ml.errors import DataError


@pytest.fixture
def protocol():
    return load_protocol(Path("config/dmn.toml"))


def raw_frame(rows=1096):
    rng = np.random.default_rng(91)
    opening = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, rows)))
    closing = opening * np.exp(rng.normal(0, 0.009, rows))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC"),
            "open": opening,
            "close": closing,
            "high": np.maximum(opening, closing) * 1.01,
            "low": np.minimum(opening, closing) * 0.99,
            "volume": 100.0,
        }
    )


def test_features_are_causal_and_return_uses_next_open():
    raw = raw_frame()
    before = causal_features(raw)
    changed = raw.copy()
    changed.loc[800:, ["open", "high", "low", "close"]] *= 7
    after = causal_features(changed)
    mask = before.timestamp < raw.timestamp.iloc[800]
    pd.testing.assert_frame_equal(
        before.loc[mask, [*FEATURES, "daily_volatility"]],
        after.loc[mask, [*FEATURES, "daily_volatility"]],
    )
    row = before.iloc[10]
    i = raw.index[raw.timestamp == row.timestamp][0]
    assert row.asset_return == pytest.approx(raw.open.iloc[i + 2] / raw.open.iloc[i + 1] - 1)
    assert row.target_time == raw.timestamp.iloc[i + 2]
    assert before.asset_return.iloc[-2:].isna().all()


def test_macd_uses_paper_time_scale():
    raw = raw_frame()
    close = raw.close
    q = (
        close.ewm(alpha=1 / 8, adjust=False).mean() - close.ewm(alpha=1 / 24, adjust=False).mean()
    ) / close.rolling(63).std()
    expected = q / q.rolling(252).std()
    actual = causal_features(raw)
    np.testing.assert_allclose(actual.macd_8_24, expected.iloc[-len(actual) :])


def test_panel_shapes_and_temporal_purge(protocol):
    panel = build_panel({"A": raw_frame(), "B": raw_frame()}, 63, 1e-4)
    assert panel.x.shape[1:] == (2, 63, 8)
    train, val, test = temporal_split(
        panel,
        pd.Timestamp("2022-07-01", tz="UTC"),
        pd.Timestamp("2023-01-01", tz="UTC"),
        protocol["training"],
    )
    assert panel.target_times[train[-1]] < panel.times[val[0]]
    assert panel.target_times[val[-1]] < panel.times[test[0]]
    assert panel.target_times[test[-1]] < pd.Timestamp("2023-01-01", tz="UTC")
    changed = raw_frame().iloc[:-1]
    with pytest.raises(DataError, match="desalineados"):
        build_panel({"A": raw_frame(), "B": changed}, 63, 1e-4)


def test_weights_respect_gross_and_position_mode(protocol):
    signal = np.array([[1.0, -1.0], [0.5, 0.2]])
    vol = np.full_like(signal, 1e-7)
    w = target_weights(signal, vol, protocol["risk"])
    assert (w >= 0).all()
    assert (np.abs(w).sum(axis=1) <= 1).all()
    risk = {**protocol["risk"], "position_mode": "long_short"}
    assert target_weights(signal, vol, risk)[0, 1] == -0.5


def test_costs_entry_liquidation_and_initial_drawdown(protocol):
    costs = {k: 0.0 for k in protocol["costs"]}
    costs["commission_bps"] = 100
    curve = simulate(
        np.ones((1, 1)), np.array([[0.1]]), pd.date_range("2024-01-01", periods=1, tz="UTC"), costs
    )
    assert curve.equity.iloc[-1] == pytest.approx(1.1 / 1.01 * 0.99)
    assert curve.transaction_cost.iloc[0] == pytest.approx(1 - 1 / 1.01 + 1.1 / 1.01 * 0.01)
    flat = simulate(np.zeros((1, 1)), np.zeros((1, 1)), curve.reference_time, costs)
    assert metrics(flat)["sharpe"] is None
    assert metrics(flat)["total_return"] == 0
    loss = simulate(
        np.ones((1, 1)), np.array([[-0.2]]), curve.reference_time, {k: 0.0 for k in costs}
    )
    assert metrics(loss)["max_drawdown"] == pytest.approx(-0.2)


def test_turnover_includes_drift_when_signal_weights_unchanged(protocol):
    costs = {k: 0.0 for k in protocol["costs"]}
    curve = simulate(
        np.full((2, 2), 0.5),
        np.array([[0.2, -0.2], [0.0, 0.0]]),
        pd.date_range("2024-01-01", periods=2, tz="UTC"),
        costs,
    )
    assert curve.turnover.iloc[0] == pytest.approx(1)
    assert curve.turnover.iloc[1] == pytest.approx(0.2 + 1)  # rebalance plus liquidation
    assert curve.equity.iloc[-1] == pytest.approx(1)


def test_borrow_and_higher_costs_reduce_equity(protocol):
    times = pd.date_range("2024-01-01", periods=5, tz="UTC")
    w = np.full((5, 1), -0.5)
    r = np.full((5, 1), -0.01)
    base = simulate(w, r, times, protocol["costs"])
    stress = simulate(w, r, times, protocol["costs"], multiplier=2)
    assert (base.funding_cost > 0).all()
    assert stress.equity.iloc[-1] < base.equity.iloc[-1]
    with pytest.raises(DataError, match="huecos"):
        simulate(w[:2], r[:2], times[[0, 2]], protocol["costs"])


def test_sharpe_loss_and_cost_neighbors(protocol):
    torch = pytest.importorskip("torch")
    from trading_ml.dmn.models import differentiable_returns, sharpe_loss

    r = torch.tensor([[0.01, -0.02], [0.03, 0.02]], requires_grad=True)
    loss = sharpe_loss(r)
    expected = -r.detach().numpy().mean() / r.detach().numpy().std() * np.sqrt(365)
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert torch.isfinite(r.grad).all()
    constant = torch.zeros(10, requires_grad=True)
    sharpe_loss(constant).backward()
    assert torch.isfinite(constant.grad).all()
    signal = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    vol = torch.full_like(signal, 0.02)
    ret = torch.zeros_like(signal)
    pnl = differentiable_returns(
        signal, ret, vol, protocol["risk"], protocol["costs"], net=True, first_flat=True
    )
    chunk = differentiable_returns(
        signal[1:],
        ret[1:],
        vol[1:],
        protocol["risk"],
        protocol["costs"],
        net=True,
        first_flat=False,
    )
    torch.testing.assert_close(chunk, pnl[2:])


def test_checkpoint_and_no_test_influence(tmp_path, protocol):
    pytest.importorskip("torch")
    from trading_ml.dmn.models import apply_scaler, fit_network, load_network, predict_network

    protocol["training"].update(epochs=2, patience=1, hidden_size=4, batch_dates=128)
    panel = build_panel({"A": raw_frame()}, 8, 1e-4)
    train, val, test = temporal_split(
        panel,
        pd.Timestamp("2022-07-01", tz="UTC"),
        pd.Timestamp("2023-01-01", tz="UTC"),
        protocol["training"],
    )
    rec = next(x for x in protocol["recipes"] if x["id"] == "lstm_sharpe_net")
    model, scaler, metadata = fit_network(panel, train, val, rec, protocol, 42, tmp_path / "a")
    expected = predict_network(model, apply_scaler(panel.x[test], scaler), rec, protocol["risk"])
    loaded, meta = load_network(tmp_path / "a")
    np.testing.assert_array_equal(
        predict_network(loaded, apply_scaler(panel.x[test], meta["scaler"]), rec, protocol["risk"]),
        expected,
    )
    changed = copy.deepcopy(panel)
    changed.x = panel.x.copy()
    changed.x[test] *= -10
    changed.returns[test] = 0.9
    _, scaler2, metadata2 = fit_network(changed, train, val, rec, protocol, 42, tmp_path / "b")
    assert scaler == scaler2
    assert metadata["history"] == metadata2["history"]
    for key, value in model.state_dict().items():
        torch = pytest.importorskip("torch")
        state = torch.load(tmp_path / "b" / "weights.pt", weights_only=True)
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    weights = tmp_path / "a" / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(DataError, match="alterado"):
        load_network(tmp_path / "a")


def test_invalid_protocols_are_rejected(protocol):
    bad = copy.deepcopy(protocol)
    bad["data"]["end"] = "2027-11-01"
    with pytest.raises(DataError):
        validate_protocol(bad)
    bad = copy.deepcopy(protocol)
    bad["research"]["maximum_fits"] = 1
    with pytest.raises(DataError):
        validate_protocol(bad)


def test_study_resume_and_protected_artifacts(tmp_path, protocol, monkeypatch):
    pytest.importorskip("torch")
    import tomllib

    from trading_ml.dmn import study as study_module
    from trading_ml.dmn.study import run_study, verify_study

    raw = raw_frame()
    source = tmp_path / "raw.csv"
    raw.to_csv(source, index=False)
    write_json(
        sidecar(source),
        {"sha256": sha256(source), "market": {"symbol": "BTCUSDT", "interval": "1d"}},
    )
    # Reuse checked-in TOML to exercise the real CLI protocol, with a tiny fit budget.
    text = Path("config/dmn.toml").read_text(encoding="utf-8")
    text = text.replace('symbols = ["BTCUSDT", "ETHUSDT"]', 'symbols = ["BTCUSDT"]')
    text = text.replace('end = "2026-10-01"', 'end = "2023-01-01"')
    text = text.replace('"artifacts/data/raw.csv"', '"' + source.as_posix() + '"')
    text = text.replace(
        'boundaries = ["2022-07-01", "2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01", "2026-10-01"]',
        'boundaries = ["2022-07-01", "2023-01-01"]',
    )
    text = text.replace("seeds = [42, 1729]", "seeds = [42]")
    text = text.replace("epochs = 30", "epochs = 1").replace("hidden_size = 20", "hidden_size = 4")
    text = (
        text.split("[[recipes]]")[0]
        + '\n[[recipes]]\nid="tiny"\narchitecture="linear"\nobjective="sharpe"\nnet_training=true\n'
    )
    tomllib.loads(text)
    config = tmp_path / "protocol.toml"
    config.write_text(text, encoding="utf-8")
    root = tmp_path / "artifacts"
    protected = root / "models" / "active.json"
    write_json(protected, {"version": "preserved"})
    digest = sha256(protected)
    settings = Settings(root=root)
    original = study_module.summarize

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption after fitting")

    monkeypatch.setattr(study_module, "summarize", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        run_study(settings, config)
    directory = next((root / "dmn_studies").iterdir())
    checkpoint = directory / "fold_00" / "tiny__seed_42" / "weights.pt"
    checkpoint_hash = sha256(checkpoint)
    monkeypatch.setattr(study_module, "summarize", original)
    source.rename(tmp_path / "removed_source.csv")  # Resumes from its own verified snapshot.
    report = run_study(settings, config, resume=directory)
    directory = Path(report["directory"])
    assert report["actual_fits"] == 1
    assert report["fit_attempts"] == 1
    assert sha256(checkpoint) == checkpoint_hash
    assert sha256(protected) == digest
    assert verify_study(directory)["intact"]
    assert run_study(settings, config, resume=directory) == report
    assert not report["independent_advantage_confirmed"]
    data = directory / "tiny_equity.csv"
    data.write_bytes(data.read_bytes() + b"tamper")
    with pytest.raises(DataError, match="alterado"):
        verify_study(directory)
