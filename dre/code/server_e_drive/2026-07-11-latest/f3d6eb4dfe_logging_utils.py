from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "PyYAML", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": commit,
        "dependencies": packages,
    }


def write_environment_manifest(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(environment_manifest(), indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["environment_manifest", "write_environment_manifest"]
