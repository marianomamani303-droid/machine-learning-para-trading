"""Ejecuta controles de calidad y conserva su salida sin requerir una shell."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from trading_ml.common import environment, write_json


def main() -> int:
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    checks = [
        ["-m", "pytest", "-q"],
        ["-m", "ruff", "check", "."],
        ["-m", "ruff", "format", "--check", "."],
        ["-m", "pip", "check"],
    ]
    results = []
    for arguments in checks:
        process = subprocess.run(
            [sys.executable, *arguments],
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
        print(process.stdout, end="")
        if process.stderr:
            print(process.stderr, file=sys.stderr, end="")
    passed = all(result["exit_code"] == 0 for result in results)
    write_json(
        Path("artifacts/verification/checks.json"),
        {
            "executed_at": datetime.now(UTC).isoformat(),
            "environment": environment(),
            "passed": passed,
            "checks": results,
        },
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
