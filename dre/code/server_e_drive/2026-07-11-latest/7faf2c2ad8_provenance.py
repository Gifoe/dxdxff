from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import sha256_file, stable_hash


def git_state(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    diff = run("diff", "--no-ext-diff") or ""
    cached = run("diff", "--cached", "--no-ext-diff") or ""
    untracked = (run("ls-files", "--others", "--exclude-standard", "--", "a12_vcsn", "scripts/*a12*", "tests/test_a12_*.py") or "").splitlines()
    untracked_payload = []
    for relative in sorted(untracked):
        path = root / relative
        if path.is_file():
            untracked_payload.append({"path": relative.replace("\\", "/"), "sha256": sha256_file(path)})
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")), "diff_hash": stable_hash(diff),
            "cached_diff_hash": stable_hash(cached), "a12_untracked_source_hash": stable_hash(untracked_payload)}


def execution_fingerprint(*, ledger: pd.DataFrame, cache_audit: dict[str, Any] | None,
                          hnc_audit: dict[str, Any] | None = None, resolved_schema: dict[str, Any] | None = None,
                          feature_registry: dict[str, Any] | None = None, config: dict[str, Any] | None = None,
                          variant: str | None = None, outer_fold: int | None = None,
                          old_v3_summary_hash: str | None = None, root: str | Path = ".") -> dict[str, Any]:
    cache = cache_audit or {}
    filter_audit = cache.get("filter_audit", {})
    return {"old_v3_input_hash": stable_hash(ledger.sort_values([name for name in ("subject_id", "channel_name_norm", "channel_name_original") if name in ledger]).to_dict("records")),
            "source_cache_sha256": cache.get("cache_sha256"),
            "cache_invalid_record_policy": (config or {}).get("cache_invalid_record_policy"),
            "excluded_record_manifest_hash": stable_hash(filter_audit),
            "filtered_cache_schema_hash": stable_hash({key: cache.get(key) for key in ("feature_dim", "feature_names", "n_run_records", "n_subjects_with_features")}),
            "hnc_hash": (hnc_audit or {}).get("sha256"), "old_v3_summary_hash": old_v3_summary_hash,
            "feature_registry_hash": stable_hash(feature_registry or {}), "resolved_schema_hash": stable_hash(resolved_schema or {}),
            "config_hash": stable_hash(config or {}), "git": git_state(root), "variant": variant, "outer_fold": outer_fold,
            "code_version": "a12-vcsn-final-repair-2026-07-11"}


def fingerprint_hash(payload: dict[str, Any]) -> str:
    return stable_hash(payload)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value}


def differing_fingerprint_fields(expected: dict[str, Any], current: dict[str, Any]) -> list[str]:
    left, right = _flatten(expected), _flatten(current)
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def assert_resume_compatible(expected: dict[str, Any], current: dict[str, Any]) -> None:
    from .protocol import ProtocolError
    differing = differing_fingerprint_fields(expected, current)
    if differing:
        raise ProtocolError(f"resume refused; differing fields: {differing}")
