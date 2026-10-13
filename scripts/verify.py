"""Execute real checks; preserve previous verification reports."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from trading_ml.common import write_new_json
from trading_ml.verification import run_checks, verify_recent_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--study", type=Path, help="Verify a historical recent-retraining study read-only"
    )
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    output = args.output or project / "artifacts" / "verification" / "runs" / (
        uuid.uuid4().hex + ".json"
    )
    if output.exists():
        parser.error("El informe ya existe; elegí otra ruta para conservar la evidencia")
    study = verify_recent_study(project / "artifacts", args.study) if args.study else None
    result = run_checks(project)
    if study is not None:
        result["historical_study"] = study
    for check in result["checks"]:
        print(check["stdout"], end="")
        if check["stderr"]:
            print(check["stderr"], end="")
    write_new_json(output, result)
    print(f"Informe de verificación: {output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
