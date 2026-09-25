from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .constants import PATH_ENV_KEYS


class ConfigError(ValueError):
    """Raised when a resolved experiment configuration is unsafe or incomplete."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError("Configuration top level must be a mapping.")
    return copy.deepcopy(payload)


def _expand_environment(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {str(key): _expand_environment(child, environ) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand_environment(child, environ) for child in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: str(environ.get(match.group(1), match.group(0))), value)
    return value


def _apply_cli_overrides(config: dict[str, Any], cli_overrides: Mapping[str, Any]) -> None:
    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ConfigError("config.paths must be a mapping.")
    for key in PATH_ENV_KEYS:
        value = cli_overrides.get(key)
        if value is not None:
            paths[key] = str(value)
    for key, value in cli_overrides.items():
        if key in PATH_ENV_KEYS or value is None:
            continue
        if "." in key:
            section, child_key = key.split(".", 1)
            section_value = config.setdefault(section, {})
            if not isinstance(section_value, dict):
                raise ConfigError(f"config.{section} must be a mapping to override {key!r}.")
            section_value[child_key] = value
        else:
            config[key] = value


def validate_config(config: Mapping[str, Any], *, require_paths: Sequence[str] = ()) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    paths = resolved.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ConfigError("config.paths must be a mapping.")
    missing = [key for key in require_paths if not str(paths.get(key, "")).strip()]
    unresolved = [
        key
        for key in require_paths
        if isinstance(paths.get(key), str) and _ENV_PATTERN.search(str(paths.get(key)))
    ]
    if missing or unresolved:
        details = sorted(set(missing + unresolved))
        raise ConfigError(f"Missing required path configuration: {', '.join(details)}")
    return resolved


def resolve_config(
    yaml_path: str | Path | None,
    cli_overrides: Mapping[str, Any] | None,
    environ: Mapping[str, str] | None,
    *,
    require_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve YAML, environment, and CLI values with CLI taking precedence."""

    env = dict(environ or {})
    config = _expand_environment(_load_yaml(yaml_path), env)
    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ConfigError("config.paths must be a mapping.")
    for key, env_key in PATH_ENV_KEYS.items():
        value = env.get(env_key)
        if value:
            paths[key] = value
    _apply_cli_overrides(config, dict(cli_overrides or {}))
    return validate_config(config, require_paths=require_paths)


__all__ = ["ConfigError", "resolve_config", "validate_config"]
