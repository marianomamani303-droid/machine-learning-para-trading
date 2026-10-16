from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from trading_ml.config import load_settings
from trading_ml.errors import TradingMLError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Deep Momentum Networks — investigación histórica reproducible"
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    study = commands.add_parser("study")
    study.add_argument("--protocol", type=Path, default=Path("config/dmn.toml"))
    study.add_argument(
        "--download", action="store_true", help="Descargar fuentes faltantes con checksum"
    )
    study.add_argument(
        "--resume", type=Path, help="Continuar un plan del mismo código/entorno/protocolo"
    )
    verify = commands.add_parser("verify")
    verify.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        from trading_ml.dmn.study import run_study, verify_study

        if args.command == "verify":
            result = verify_study(args.study)
        else:
            result = run_study(
                load_settings(args.config), args.protocol, resume=args.resume, fetch=args.download
            )
            result = {
                k: result[k]
                for k in [
                    "run_id",
                    "directory",
                    "actual_fits",
                    "models",
                    "conclusion",
                    "protected_artifacts_unchanged",
                    "independent_advantage_confirmed",
                ]
            }
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (TradingMLError, OSError, ImportError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
