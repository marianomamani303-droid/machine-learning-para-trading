"""Record an externally interrupted technical run without interpreting its metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from trading_ml.common import read_json, sha256, write_new_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    completed = 0
    known_models = set()
    for path in args.study.rglob("fit.json"):
        state = read_json(path)
        completed += 1 if "components" in state else 2
        known_models.add(path.parent / "model.joblib")
    for path in args.study.rglob("model.joblib"):
        if path not in known_models:
            completed += 1
    charge = completed + 2
    write_new_json(
        args.study / "failure.json",
        {
            "status": "technical_abort",
            "completed_fits": completed,
            "budget_charge": charge,
            "actual_fit_count_range": [completed, charge],
            "reason": "Future-dependent barrier label availability incorrectly used to mask economic entries; corrected before selection or any prospective evaluation",
            "completed_models_reusable": True,
            "economic_results_valid": False,
            "extra_two_fits": "Conservative charge for interrupted estimator/calibration; first version had no per-fit ledger",
            "hypotheses_changed": False,
            "additional_fits": 0,
        },
    )
    write_new_json(
        args.study / "aborted_artifacts.json",
        {
            str(p.relative_to(args.study)): sha256(p)
            for p in sorted(args.study.rglob("*"))
            if p.is_file()
        },
    )
    print({"completed_fits": completed, "budget_charge": charge})


if __name__ == "__main__":
    main()
