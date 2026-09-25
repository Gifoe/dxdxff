from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constants import FORBIDDEN_EXACT_KEYS, FORBIDDEN_KEY_FRAGMENTS, MODEL_INPUT_WHITELIST


class LeakageError(ValueError):
    """Raised when outcome or channel-label information enters a model-facing tree."""


@dataclass(frozen=True)
class LeakageScanResult:
    stage: str
    scanned_key_count: int
    forbidden_paths: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.forbidden_paths


def _normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _is_forbidden_key(key: str) -> bool:
    normalized = _normalized_key(key)
    return normalized in FORBIDDEN_EXACT_KEYS or any(fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS)


def scan_label_blind_tree(value: Any, *, stage: str) -> LeakageScanResult:
    forbidden: list[str] = []
    scanned = 0

    def visit(node: Any, path: str) -> None:
        nonlocal scanned
        if isinstance(node, Mapping):
            for key, child in node.items():
                scanned += 1
                child_path = f"{path}.{key}" if path else str(key)
                if _is_forbidden_key(str(key)):
                    forbidden.append(child_path)
                visit(child, child_path)
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return LeakageScanResult(stage=str(stage), scanned_key_count=scanned, forbidden_paths=tuple(sorted(set(forbidden))))


def assert_label_blind_tree(value: Any, *, stage: str) -> LeakageScanResult:
    result = scan_label_blind_tree(value, stage=stage)
    if not result.passed:
        raise LeakageError(f"Forbidden outcome/channel-label field(s) at {stage}: {', '.join(result.forbidden_paths)}")
    return result


def select_model_inputs(mapping: Mapping[str, Any]) -> dict[str, Any]:
    selected = {key: mapping[key] for key in MODEL_INPUT_WHITELIST if key in mapping}
    assert_label_blind_tree(selected, stage="model_input_whitelist")
    return selected


def write_leakage_audit(results: Iterable[LeakageScanResult], path: str | Path) -> None:
    scans = list(results)
    payload = {
        "passed": all(result.passed for result in scans),
        "scans": [asdict(result) for result in scans],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)


__all__ = [
    "LeakageError",
    "LeakageScanResult",
    "assert_label_blind_tree",
    "scan_label_blind_tree",
    "select_model_inputs",
    "write_leakage_audit",
]
