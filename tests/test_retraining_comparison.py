"""Guard against seed pseudoreplication and broken multiple-testing correction."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "retraining_analysis", Path(__file__).parents[1] / "scripts/analyze_retraining_comparison.py"
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_identical_arms_have_no_significant_improvement():
    y = np.tile([0, 1], 40)
    a = pd.DataFrame({"target": y, "p_42": np.linspace(0.2, 0.8, 80)})
    result = analysis.paired_inference(a, a.copy(), np.repeat([0, 1], 40), samples=60)
    for block in result.values():
        assert block["dates"] == 80
        assert block["seeds"] == 1
        assert block["primary_p"] == 1
        assert block["log_loss_gain_ci95"] == [0, 0]
        assert block["auc_delta_ci95"] == [0, 0]


def test_duplicate_seed_does_not_increase_effective_sample_or_shrink_ci():
    y = np.tile([0, 1], 40)
    a = pd.DataFrame({"target": y, "p_42": np.linspace(0.2, 0.8, 80)})
    b = pd.DataFrame({"target": y, "p_42": np.full(80, 0.5)})
    groups = np.repeat([0, 1], 40)
    original = analysis.paired_inference(a, b, groups, samples=60)
    a["p_1729"] = a.p_42
    b["p_1729"] = b.p_42
    duplicate = analysis.paired_inference(a, b, groups, samples=60)
    for block in original:
        assert duplicate[block]["dates"] == original[block]["dates"]
        assert duplicate[block]["log_loss_gain_ci95"] == original[block]["log_loss_gain_ci95"]
        assert duplicate[block]["primary_p"] == original[block]["primary_p"]


def test_holm_accounts_for_the_entire_family_and_preserves_order():
    assert analysis.holm({"first": 0.01, "second": 0.03, "third": 0.9}) == {
        "first": 0.03,
        "second": 0.06,
        "third": 0.9,
    }


def test_daily_equity_aggregation_preserves_initial_execution_cost(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "retraining_economics",
        Path(__file__).parents[1] / "scripts/retraining_economic_inference.py",
    )
    economics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(economics)
    path = tmp_path / "curve.csv"
    pd.DataFrame(
        {
            "timestamp": ["2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"],
            "equity": [10000.0, 9990.0, 10000.0],
        }
    ).to_csv(path, index=False)
    daily = economics.daily_curve(path)
    assert daily.net_return.iloc[0] == -0.0010000000000000009
    assert np.isclose(np.prod(1 + daily.net_return), 1)
