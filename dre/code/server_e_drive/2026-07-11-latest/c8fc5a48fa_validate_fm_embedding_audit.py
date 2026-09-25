from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TRUE_FM_MODELS = {"biot", "cbramod", "labram"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_embedding_audit(
    audit: Mapping[str, Any],
    *,
    model_name: str,
    require_true_pretrained_fm: bool = False,
    require_no_skipped_windows: bool = False,
    require_full_manifest_coverage: bool = False,
) -> None:
    model = str(model_name).lower()
    n_manifest = int(audit.get("n_manifest_rows", 0) or 0)
    n_embeddings = int(audit.get("n_embeddings_written", 0) or 0)
    _require(n_manifest > 0, "n_manifest_rows must be > 0")
    _require(n_embeddings > 0, "n_embeddings_written must be > 0")
    _require(bool((audit.get("metadata_embedding_alignment_check") or {}).get("pass")), "metadata_embedding_alignment_check.pass must be true")
    if require_full_manifest_coverage:
        _require(n_embeddings == n_manifest, "n_embeddings_written must equal n_manifest_rows")
    if require_no_skipped_windows:
        _require(int(audit.get("skipped_windows_count", 0) or 0) == 0, "skipped_windows_count must be 0")
    _require(bool(audit.get("comparison_only_baseline")), "comparison_only_baseline must be true")
    _require(bool(audit.get("frozen_encoder")), "frozen_encoder must be true")
    _require(audit.get("fine_tuned") is False, "fine_tuned must be false")
    _require(audit.get("adapter_tuned") is False, "adapter_tuned must be false")
    _require(int(audit.get("backbone_trainable_params", -1)) == 0, "backbone_trainable_params must be 0")
    if require_true_pretrained_fm or model in TRUE_FM_MODELS:
        _require(audit.get("true_pretrained_fm") is True, "true_pretrained_fm must be true")
        _require(audit.get("debug_only") is False, "debug_only must be false")
        _require(audit.get("paper_baseline") is True, "paper_baseline must be true")
        _require(audit.get("adapter_status") == "implemented", "adapter_status must be implemented")
        _require(bool(str(audit.get("checkpoint_path", "")).strip()), "checkpoint_path must not be empty")
        _require(bool(str(audit.get("external_repo_path", "")).strip()), "external_repo_path must not be empty")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--require_true_pretrained_fm", action="store_true")
    parser.add_argument("--require_no_skipped_windows", action="store_true")
    parser.add_argument("--require_full_manifest_coverage", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.audit_path)
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
        validate_embedding_audit(
            audit,
            model_name=args.model_name,
            require_true_pretrained_fm=args.require_true_pretrained_fm,
            require_no_skipped_windows=args.require_no_skipped_windows,
            require_full_manifest_coverage=args.require_full_manifest_coverage,
        )
    except Exception as exc:
        print(f"FAIL {path}: {exc}")
        return 1
    print(f"PASS {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
