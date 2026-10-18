from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from trading_ml.config import Settings
from trading_ml.errors import TradingMLError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Expansión de mercados y datos point-in-time")
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("acquire")
    fetch.add_argument("--start", default="2021-01-01")
    fetch.add_argument("--end", default="2025-07-01")
    study = sub.add_parser("study")
    study.add_argument("--protocol", type=Path, default=Path("config/expansion.toml"))
    study.add_argument("--dataset", type=Path, required=True)
    study.add_argument("--resume", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--study", type=Path, required=True)
    cat = sub.add_parser("catalogue")
    cat.add_argument("--output", type=Path, required=True)
    sub.add_parser("oi-pilot")
    fx = sub.add_parser("forex-study")
    fx.add_argument("--protocol", type=Path, default=Path("config/expansion_forex.toml"))
    fx.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "acquire":
            from trading_ml.expansion.data import acquire

            result = {"dataset_index": str(acquire(args.root, args.start, args.end).resolve())}
        elif args.command == "forex-study":
            from trading_ml.expansion.forex import run_forex

            report = run_forex(Settings(root=args.root), args.protocol, resume=args.resume)
            result = {
                k: report[k]
                for k in ["directory", "actual_fits", "independent_advantage_confirmed"]
            }
        elif args.command == "study":
            from trading_ml.expansion.study import run_study

            report = run_study(
                Settings(root=args.root), args.protocol, args.dataset, resume=args.resume
            )
            result = {
                k: report[k]
                for k in [
                    "run_id",
                    "directory",
                    "actual_fits",
                    "independent_advantage_confirmed",
                    "protected_artifacts_unchanged",
                ]
            }
        elif args.command == "verify":
            from trading_ml.common import read_json

            if "data" in read_json(args.study / "plan.json")["protocol"]:
                from trading_ml.expansion.study import verify

                result = verify(args.study)
            else:
                from trading_ml.expansion.forex import verify_forex

                result = verify_forex(args.study)
        elif args.command == "catalogue":
            from trading_ml.expansion.catalog import save_catalogue

            result = save_catalogue(args.root, args.output)
        else:
            from trading_ml.expansion.catalog import acquire_oi_pilot

            result = {"pilot": str(acquire_oi_pilot(args.root).resolve())}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (TradingMLError, OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
