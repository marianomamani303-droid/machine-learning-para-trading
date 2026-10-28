"""Tests exhaustivos para los modelos adaptativos, filtro de Kalman, evaluador secuencial y CLI."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_ml.cli import main as cli_main
from trading_ml.errors import DataError
from trading_ml.online import service
from trading_ml.online.adaptive import (
    COHORT_DEFINITIONS,
    AdaptiveOnlineModel,
    new_adaptive_model,
)
from trading_ml.online.config import FEATURE_SETS
from trading_ml.online.data import features
from trading_ml.online.evaluation import (
    audit_sequential_evaluator_math,
    minimum_detectable_effect_empirical_bernstein,
    minimum_detectable_effect_hoeffding,
    required_sample_size_empirical_bernstein,
    required_sample_size_hoeffding,
    run_evaluator_simulation_study,
)
from trading_ml.online.experiment import run_prequential_replay
from trading_ml.online.models import validate_model


def _synthetic_candles(n: int = 220, seed: int = 123, interval: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    delta = pd.Timedelta("1h" if interval == "1h" else "1D")
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    timestamps = [start + i * delta for i in range(n)]
    price = 40000.0
    rows = []
    for ts in timestamps:
        ret = float(rng.normal(0.0002, 0.008))
        open_p = price
        close_p = price * math.exp(ret)
        spread = abs(float(rng.normal(0.005, 0.002)))
        high_p = max(open_p, close_p) * (1.0 + spread)
        low_p = min(open_p, close_p) * (1.0 - spread)
        vol = float(rng.uniform(50.0, 500.0))
        rows.append(
            {
                "timestamp": ts,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol,
            }
        )
        price = close_p
    return pd.DataFrame(rows)


def test_causal_features_v1_and_v2_invariance_to_future_bars() -> None:
    candles = _synthetic_candles(120, seed=7)
    window_64 = candles.iloc[:64].copy()
    f_v1_orig = features(window_64, feature_set="v1_base8")
    f_v2_orig = features(window_64, feature_set="v2_financial11")

    assert len(f_v1_orig) == len(FEATURE_SETS["v1_base8"]) == 8
    assert len(f_v2_orig) == len(FEATURE_SETS["v2_financial11"]) == 11
    assert all(math.isfinite(v) and -10.0 <= v <= 10.0 for v in f_v2_orig)
    assert f_v2_orig[:8] == f_v1_orig

    # Mutate future bars strictly after index 63 and verify causal features at 64 do not change
    tampered = candles.copy()
    tampered.loc[64:, "close"] *= 5.0
    tampered.loc[64:, "high"] *= 6.0
    tampered.loc[64:, "volume"] *= 100.0
    assert features(tampered.iloc[:64], feature_set="v1_base8") == f_v1_orig
    assert features(tampered.iloc[:64], feature_set="v2_financial11") == f_v2_orig


def test_kalman_filter_numerical_stability_and_synthetic_tracking() -> None:
    rng = np.random.default_rng(99)
    n_features = 4
    kf = AdaptiveOnlineModel.create(
        "kalman_linear",
        n_features=n_features,
        seed=42,
        options={
            "kalman_init_variance": 0.10,
            "kalman_process_noise": 1e-4,
            "kalman_obs_noise": 1e-4,
            "max_weight_norm": 0.25,
        },
    )

    true_beta = np.array([0.001, 0.012, -0.009, 0.005, 0.0])
    errors_early: list[float] = []
    errors_late: list[float] = []

    for step in range(180):
        # Slowly drifting state in second half
        if step == 90:
            true_beta = np.array([0.001, 0.015, -0.011, 0.004, 0.0])
        x_raw = rng.normal(0.0, 1.0, size=n_features).tolist()
        out = kf.predict_structured(x_raw)
        assert 0.0 < out["probability"] < 1.0
        assert out["predictive_std"] > 0.0
        assert out["q10"] < out["q50"] < out["q90"]

        # Realized return from true linear state-space process
        scaled_approx = np.array([1.0] + x_raw, dtype=float)
        r_true = float(np.dot(scaled_approx, true_beta) + rng.normal(0.0, 0.002))
        label = int(r_true > 0.0)
        sq_err = (out["expected_return"] - r_true) ** 2
        if 10 <= step < 40:
            errors_early.append(sq_err)
        elif step >= 120:
            errors_late.append(sq_err)

        kf.update(x_raw, label, realized_return=r_true, target_id=f"tgt_{step}")
        kf.validate_state()

    cov = np.array(kf.state["params"]["state_cov"], dtype=float)
    assert np.allclose(cov, cov.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0.0)
    assert np.mean(errors_late) < np.mean(errors_early)


def test_kalman_persistence_and_duplicate_target_guard() -> None:
    kf = AdaptiveOnlineModel.create("kalman_linear", n_features=8, seed=42)
    x = [0.2, -0.1, 0.05, 0.3, 0.1, 0.2, -0.05, 0.15]
    kf.update(x, 1, realized_return=0.012, target_id="bar_2024_01_01T01")
    updates_after_first = kf.state["updates"]
    mean_after_first = list(kf.state["params"]["state_mean"])
    assert updates_after_first == 1

    # Duplicate confirmation of the exact same target_id must be ignored
    kf.update(x, 1, realized_return=0.012, target_id="bar_2024_01_01T01")
    assert kf.state["updates"] == 1
    assert kf.state["params"]["state_mean"] == mean_after_first

    # Serialize and restore
    saved_json = json.loads(json.dumps(kf.save()))
    restored = AdaptiveOnlineModel.load("kalman_linear", saved_json)
    assert restored.predict_structured(x) == kf.predict_structured(x)


def test_logistic_static_freezes_after_warmup_and_logistic_adaptive_bounds_weights() -> None:
    static_m = AdaptiveOnlineModel.create(
        "logistic_static",
        n_features=4,
        seed=42,
        options={"static_warmup_updates": 10},
    )
    adaptive_m = AdaptiveOnlineModel.create(
        "logistic_adaptive",
        n_features=4,
        seed=42,
        options={"max_weight_norm": 1.5, "l1": 0.005, "l2": 0.01},
    )
    rng = np.random.default_rng(11)
    weights_at_10 = None
    for i in range(25):
        x = rng.normal(0.0, 2.0, size=4).tolist()
        y = int(x[0] - 0.5 * x[1] > 0.0)
        static_m.update(x, y, target_id=f"t_{i}")
        adaptive_m.update(x, y, target_id=f"t_{i}")
        if i == 9:
            weights_at_10 = list(static_m.state["params"]["weights"])
            assert static_m.state["params"]["frozen"] is True

    assert static_m.state["updates"] == 25
    assert static_m.state["params"]["weight_updates"] == 10
    assert static_m.state["params"]["weights"] == weights_at_10

    adapt_w = adaptive_m.state["params"]["weights"]
    norm = math.sqrt(sum(w * w for w in adapt_w))
    assert norm <= 1.5 + 1e-6
    assert 0.0 < adaptive_m.predict([1.0, -1.0, 0.2, 0.0]) < 1.0


def test_volatility_and_quantile_models_respect_contracts() -> None:
    ewma = AdaptiveOnlineModel.create("ewma_volatility", n_features=8)
    har = AdaptiveOnlineModel.create("har_rv", n_features=8)
    qr = AdaptiveOnlineModel.create("quantile_regression", n_features=8)
    rng = np.random.default_rng(21)

    for step in range(60):
        x = rng.normal(0.0, 1.0, size=8).tolist()
        ret = float(rng.normal(0.0, 0.015))
        rv = max(1e-7, ret * ret * 1.2)
        y = int(ret > 0)

        p_ewma = ewma.predict_structured(x)
        p_har = har.predict_structured(x)
        p_qr = qr.predict_structured(x)

        assert p_ewma["pred_volatility"] > 0.0
        assert p_har["pred_volatility"] > 0.0
        assert p_qr["q10"] <= p_qr["q50"] <= p_qr["q90"]
        assert p_qr["interval_width"] > 0.0

        ewma.update(x, y, realized_return=ret, realized_variance=rv, target_id=f"v_{step}")
        har.update(x, y, realized_return=ret, realized_variance=rv, target_id=f"v_{step}")
        qr.update(x, y, realized_return=ret, realized_variance=rv, target_id=f"v_{step}")

    ewma.validate_state()
    har.validate_state()
    qr.validate_state()


def test_sequential_evaluator_math_and_simulation_scenarios() -> None:
    audit = audit_sequential_evaluator_math()
    n_hoeff = required_sample_size_hoeffding(0.005, min_effect=0.001, alpha_spent=0.00625)
    n_eb = required_sample_size_empirical_bernstein(
        0.005, empirical_variance=0.0035, scale_bound=0.20, min_effect=0.001, alpha_spent=0.00625
    )
    assert n_eb < n_hoeff / 20
    assert minimum_detectable_effect_empirical_bernstein(
        100, 0.0035, 0.20, 0.001, 0.00625
    ) < minimum_detectable_effect_hoeffding(100, 0.001, 0.00625)
    assert len(audit["sample_size_by_effect"]) == 5

    sim = run_evaluator_simulation_study(
        n_trials=40, n_steps=450, eval_window=75, min_samples=75, patience=2, seed=42
    )["simulation_results"]
    # Under null hypothesis (no_edge), false promotion rate must remain strictly controlled
    assert sim["no_edge__empirical_bernstein"]["promotion_rate"] <= 0.05
    assert sim["no_edge__hoeffding_azuma"]["promotion_rate"] == 0.0
    # Under serial dependence with no edge, false promotion rate remains controlled
    assert sim["serial_dependence__empirical_bernstein"]["promotion_rate"] <= 0.05
    # Under moderate edge, Empirical Bernstein has substantially higher power than Hoeffding
    assert (
        sim["moderate_edge__empirical_bernstein"]["promotion_rate"]
        > sim["moderate_edge__hoeffding_azuma"]["promotion_rate"]
    )
    # Under miscalibrated overconfident predictions, models are disabled rather than promoted
    assert sim["miscalibrated_overconfident__empirical_bernstein"]["promotion_rate"] == 0.0
    assert sim["miscalibrated_overconfident__empirical_bernstein"]["disabling_rate"] > 0.50


def test_corrupted_kalman_covariance_and_quarantine_guard_rejected() -> None:
    from trading_ml.online.experiment import load_experiment_candles

    kf = new_adaptive_model("kalman_linear", n_features=4, seed=42)
    validate_model(kf)

    # Corrupt symmetry of state_cov
    bad_sym = copy.deepcopy(kf)
    bad_sym["params"]["state_cov"][0][1] = 0.5
    bad_sym["params"]["state_cov"][1][0] = -0.2
    with pytest.raises(ValueError, match="no simétrica"):
        validate_model(bad_sym)

    # Corrupt diagonal of state_cov
    bad_diag = copy.deepcopy(kf)
    bad_diag["params"]["state_cov"][2][2] = 0.0
    with pytest.raises(ValueError, match="positiva"):
        validate_model(bad_diag)

    # Attempting to load historical experiment data beyond 2025-07-01 must fail
    with pytest.raises(DataError, match="2025-07-01"):
        load_experiment_candles(
            symbol="BTCUSDT",
            interval="1h",
            start="2025-01-01",
            end="2025-08-01",
        )


def test_prequential_replay_feature_ablation_and_target_separation() -> None:
    candles = _synthetic_candles(180, seed=77)
    res_v1 = run_prequential_replay(
        candles,
        symbol="BTCUSDT",
        interval="1h",
        horizon=1,
        feature_mode="v1_base8",
        warmup_bars=64,
        static_warmup_updates=30,
        eval_window=40,
        min_samples=40,
        patience=2,
    )
    res_v2 = run_prequential_replay(
        candles,
        symbol="BTCUSDT",
        interval="1h",
        horizon=1,
        feature_mode="v2_financial11",
        warmup_bars=64,
        static_warmup_updates=30,
        eval_window=40,
        min_samples=40,
        patience=2,
    )
    assert res_v1["n_features"] == 8
    assert res_v2["n_features"] == 11
    assert set(res_v2["targets"]) == {"direction", "expected_return", "volatility", "quantiles"}
    assert set(res_v2["champions"]) == {"direction", "expected_return", "volatility", "quantiles"}
    assert len(res_v2["performance_profile"]) == len(COHORT_DEFINITIONS["adaptive_lab"])


def test_adaptive_lab_sqlite_lifecycle_and_cli_subcommands(tmp_path: Path, capsys) -> None:
    candles = _synthetic_candles(160, seed=55)
    csv_path = tmp_path / "candles.csv"
    candles.to_csv(csv_path, index=False)

    state_path = Path("artifacts/t/test_adaptive_lab_state.sqlite3")
    if state_path.exists():
        state_path.unlink()

    cfg_path = tmp_path / "online_adaptive.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[market]",
                'symbol = "BTCUSDT"',
                'interval = "1h"',
                "horizon = 1",
                "",
                "[online]",
                f'state_path = "{state_path.as_posix()}"',
                "poll_seconds = 30",
                "warmup_bars = 64",
                "max_prediction_delay_seconds = 120",
                "max_fetch_pages = 5",
                "seed = 42",
                'start_at = "2024-01-03T16:00:00Z"',
                'cohort = "adaptive_lab"',
                'feature_set = "v2_financial11"',
                "static_warmup_updates = 20",
                "",
                "[evaluation]",
                "eval_window = 25",
                "min_samples = 25",
                "patience = 2",
                "alpha = 0.05",
                "min_effect = 0.001",
                'method = "empirical_bernstein"',
            ]
        ),
        encoding="utf-8",
    )

    try:
        res = service.run_once(
            cfg_path,
            state_path,
            source=csv_path,
            replay=True,
            now="2024-01-08T00:00:00Z",
        )
        assert res["new_bars"] == 160
        assert res["new_predictions"] > 0
        assert res["confirmed_predictions"] > 0

        ver = service.verify(state_path)
        assert ver["ok"] is True

        comp = service.compare_models(state_path)
        assert set(comp["targets"]) == {"direction", "expected_return", "volatility", "quantiles"}
        assert set(comp["champions"]) == {"direction", "expected_return", "volatility", "quantiles"}

        # Test CLI subcommands: models, compare, champions
        for subcmd in ("models", "compare", "champions"):
            rc = cli_main(
                [
                    "online",
                    subcmd,
                    "--config",
                    str(cfg_path),
                    "--state",
                    str(state_path),
                ]
            )
            assert rc == 0
            out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
            assert out["orders_enabled"] is False
    finally:
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(state_path) + suffix)
            if p.exists():
                p.unlink()
