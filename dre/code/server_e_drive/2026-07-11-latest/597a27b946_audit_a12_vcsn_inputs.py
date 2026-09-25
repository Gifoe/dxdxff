"""Audit a frozen V3 ledger and read-only A12 window cache before any training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a12_vcsn.audit import audit_anchor
from a12_vcsn.io import load_v3_ledger, load_window_feature_store
from a12_vcsn.protocol import audit_cache_subject_coverage
from a12_vcsn.utils import environment_snapshot, write_json


def _summary_metric(path: str | None) -> float | None:
    if not path: return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"patient_macro_f1", "old_v3_patient_macro_f1"} and isinstance(item, (int, float)): return float(item)
                found = walk(item)
                if found is not None: return found
        return None
    metric = walk(data)
    if metric is None: raise RuntimeError("old-v3-summary lacks patient_macro_f1")
    return metric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-v3-ledger", required=True)
    parser.add_argument("--window-cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-old-v3-macro-f1", type=float, default=None)
    parser.add_argument("--old-v3-summary", default=None)
    parser.add_argument("--anchor-parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-invalid-record-policy", choices=["fail", "drop"], default="drop")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ledger, resolved = load_v3_ledger(args.old_v3_ledger, strict=args.strict)
    channel_features, _, cache = load_window_feature_store(args.window_cache_path, strict=args.strict, invalid_record_policy=args.cache_invalid_record_policy)
    coverage = audit_cache_subject_coverage(ledger, channel_features, strict=args.strict)
    expected = args.expected_old_v3_macro_f1 if args.expected_old_v3_macro_f1 is not None else _summary_metric(args.old_v3_summary)
    if args.strict and expected is None:
        raise RuntimeError("strict metric-parity audit requires --expected-old-v3-macro-f1 from an independent old-V3 summary")
    audit = audit_anchor(ledger, expected_macro_f1=expected, expected_patients=90, expected_folds=5, tolerance=args.anchor_parity_tolerance)
    if args.strict and not audit["passed"]:
        raise RuntimeError(f"frozen V3 anchor parity failed: {audit}")
    root = Path(args.output_dir) / "audit"
    write_json(root / "resolved_schema.json", resolved)
    write_json(root / "anchor_contract.json", audit["contract"])
    write_json(root / "anchor_metric_parity.json", audit.get("metric_parity", audit))
    write_json(root / "anchor_parity.json", audit)
    write_json(root / "cache_subject_coverage.json", coverage)
    write_json(Path(args.output_dir) / "environment.json", environment_snapshot())
    print(json.dumps({"anchor_parity": audit["passed"], "cache_feature_dim": cache["feature_dim"], "cache_missing_subjects": len(coverage["missing_subjects"]), "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
