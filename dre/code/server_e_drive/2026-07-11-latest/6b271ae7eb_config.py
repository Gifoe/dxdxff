"""Configuration loading with no required YAML dependency."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Use JSON-compatible YAML or install PyYAML.") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Confirmatory config must contain a mapping")
    payload["_config_path"] = str(source.resolve())
    return payload


def resolve_path(config: dict[str, Any], key: str, *, required: bool = True) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing required config key: {key}")
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(config["_config_path"]).parent.parent / path
    return path
