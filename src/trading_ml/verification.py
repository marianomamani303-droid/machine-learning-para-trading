from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ml.common import code_provenance, environment, read_json, sha256
from trading_ml.models import load_model
from trading_ml.preprocessing import load_prepared


def run_checks(project: Path, *, python: str | None = None) -> dict[str, Any]:
    """Execute checks and retain actual stdout, stderr and return codes."""
    commands = [
        ["-m", "pytest", "-q"],
        ["-m", "ruff", "check", "src", "scripts", "tests"],
        ["-m", "ruff", "format", "--check", "src", "scripts", "tests"],
        ["-m", "pip", "check"],
    ]
    results = []
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for arguments in commands:
        process = subprocess.run(
            [python or sys.executable, *arguments],
            cwd=project,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=child_env,
            check=False,
        )
        results.append(
            {
                "arguments": arguments,
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
    return {
        "executed_at": datetime.now(UTC).isoformat(),
        "environment": environment(),
        "code_provenance": code_provenance(),
        "checks": results,
        "passed": all(item["exit_code"] == 0 for item in results),
    }


def verify_recent_study(root: Path, directory: Path) -> dict[str, Any]:
    """Read-only replacement for the archived verifier that rewrote metadata."""
    summary = read_json(directory / "summary.json")
    plan = read_json(directory / "plan.json")
    if sha256(directory / "plan.json") != summary["plan_sha256"]:
        raise ValueError("El plan de la ronda no coincide con su digest")
    original, _ = load_prepared(directory / "prepared_original.csv")
    combined, _ = load_prepared(directory / "prepared_smc.csv")
    pd.testing.assert_frame_equal(combined[original.columns], original, check_exact=True)
    old = pd.read_csv(root / "data" / "raw.csv")
    fresh = pd.read_csv(directory / "raw.csv")
    pd.testing.assert_frame_equal(
        old.loc[old.timestamp.isin(fresh.timestamp)].reset_index(drop=True), fresh, check_exact=True
    )
    checks: dict[str, Any] = {
        "matching_features_and_labels": True,
        "raw_matches_previous_snapshot": True,
        "models": {},
    }
    for name, version in summary["versions"].items():
        bundle = load_model(root, version)
        dataset = directory / (
            "prepared_original.csv" if name == "original_new" else "prepared_smc.csv"
        )
        load_prepared(dataset, bundle.metadata["contract"])
        checks["models"][name] = {
            "version": version,
            "trees": bundle.model.get_booster().num_boosted_rounds(),
            "model_sha256": sha256(bundle.directory / "model.ubj"),
        }
    artifact_checks = {}
    for path, digest in plan["protected_sha256"].items():
        candidate = Path(path)
        if candidate.parts[0] == "artifacts":
            artifact_checks[path] = sha256(root.joinpath(*candidate.parts[1:])) == digest
    if not all(artifact_checks.values()):
        raise ValueError("Un artefacto histórico protegido cambió")
    checks["protected_artifacts"] = artifact_checks
    return checks
