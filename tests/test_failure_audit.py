from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "failure_audit", Path("scripts/audit_predictive_failure.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_acf_matches_direct_biased_covariance():
    values = np.array([1, 2, 0, -1, 3, 0, -2], dtype=float)
    centered = values - values.mean()
    direct = np.correlate(centered, centered, "full")[len(values) - 1 :]
    np.testing.assert_allclose(audit.acf(values, 3), direct[1:4] / direct[0])


def test_hac_effective_size_is_statistic_specific():
    rng = np.random.default_rng(42)
    innovations = rng.normal(size=6000)
    persistent = pd.Series(innovations).rolling(20).sum().dropna().to_numpy()
    independent = audit.dependence(innovations)
    correlated = audit.dependence(persistent)
    assert correlated["mean_ess_uncapped"] < independent["mean_ess_uncapped"] / 5
    assert correlated["hac_mean_variance_inflation"] > 5


def test_cost_decomposition_replays_same_trades(tmp_path):
    directory = tmp_path / audit.RUNS["structure"]
    directory.mkdir(parents=True)
    fee, slip = 0.001, 0.0006
    gross_factors = np.array([1.01, 0.99, 1.02])
    net_factors = gross_factors * (1 - fee) * (1 - slip) / ((1 + fee) * (1 + slip))
    pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "seed": 42,
            "recipe": "original",
            "scenario": "base",
            "entry_fill": 100 * (1 + slip),
            "exit_fill": 100 * gross_factors * (1 - slip),
            "net_return": net_factors - 1,
        }
    ).to_csv(directory / "trades.csv", index=False)
    row = audit.economic_decomposition(tmp_path)[0]
    np.testing.assert_allclose(row["gross_same_trades_return"], gross_factors.prod() - 1)
    np.testing.assert_allclose(row["net_compound_return"], net_factors.prod() - 1)
    assert row["round_trip_cost_bps"] > 31
