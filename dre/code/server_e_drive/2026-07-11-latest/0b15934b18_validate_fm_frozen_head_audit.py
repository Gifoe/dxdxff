from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_frozen_head_audit(audit: Mapping[str, Any], *, expected_n_folds: int) -> None:
    incomplete = audit.get("incomplete_methods") or []
    _require(not incomplete, f"incomplete_methods must be empty, got {incomplete}")
    completed = audit.get("completed_folds_by_method") or {}
    expected = set(range(1, int(expected_n_folds) + 1))
    for method, folds in completed.items():
        observed = {int(item) for item in folds}
        _require(observed == expected, f"{method} must have folds {sorted(expected)}, got {sorted(observed)}")
    _require(bool(audit.get("comparison_only_baseline")), "comparison_only_baseline must be true")
    _require(audit.get("frozen_or_finetuned") == "frozen", "frozen_or_finetuned must be frozen")
    _require(audit.get("true_pretrained_fm") is True, "true_pretrained_fm must be true for paper FM models")
    _require(int(audit.get("n_folds", 0) or 0) == int(expected_n_folds), "n_folds must equal expected_n_folds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_path", required=True)
    parser.add_argument("--expected_n_folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.audit_path)
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
        validate_frozen_head_audit(audit, expected_n_folds=args.expected_n_folds)
    except Exception as exc:
        print(f"FAIL {path}: {exc}")
        return 1
    print(f"PASS {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
