from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scikit-learn", "xgboost", "requests"]
        },
    }


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.12g").encode()).hexdigest()
