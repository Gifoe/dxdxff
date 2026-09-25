from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ez_dataset import build_or_load_run_records


OUTCOME_CHOICES = ["all", "success", "failure", "success_failure", "unknown"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _reader_args(cache_path: Path, outcome_subset: str, output_dir: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        window_cache_path=str(cache_path),
        sample_cache_path=None,
        outcome_subset=outcome_subset,
        drop_high_ez_fraction_lzu=False,
        output_dir=str(output_dir) if output_dir is not None else None,
    )


def _records_by_subject(run_records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        grouped[str(record.get("subject_id"))].append(record)
    return dict(grouped)


def _feature_names_for_subject(records: Sequence[dict[str, Any]]) -> list[str] | None:
    for record in records:
        sample = record.get("sample", {})
        names = sample.get("window_feature_names", record.get("window_feature_names"))
        if names is not None:
            return [str(name) for name in names]
    return None


def _patient_signature(
    subject_id: str,
    patient_index: dict[str, dict[str, Any]],
    records_by_subject: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    meta = patient_index.get(subject_id, {})
    records = records_by_subject.get(subject_id, [])
    labels = meta.get("labels")
    if labels is None and records:
        labels = records[0].get("labels")
    labels_arr = np.asarray(labels if labels is not None else [], dtype=np.float32)
    canonical_channels = meta.get("canonical_channels")
    if canonical_channels is None and records:
        canonical_channels = records[0].get("channel_names_norm", [])
    valid = labels_arr >= 0.0
    return {
        "subject_id": subject_id,
        "num_channels": int(len(canonical_channels or [])),
        "num_ez": int(np.sum(labels_arr[valid] > 0.5)) if labels_arr.size else 0,
        "canonical_channels": [str(name) for name in (canonical_channels or [])],
        "run_ids": sorted(str(record.get("run_id")) for record in records),
        "window_feature_names": _feature_names_for_subject(records),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def compare_caches(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    old_output = args.output_dir / "old_success_reader" if args.output_dir is not None else None
    new_output = args.output_dir / "new_superset_reader" if args.output_dir is not None else None
    old_records, old_index = build_or_load_run_records(_reader_args(args.old_success_cache, "all", old_output))
    new_records, new_index = build_or_load_run_records(_reader_args(args.new_superset_cache, args.outcome_subset, new_output))

    old_grouped = _records_by_subject(old_records)
    new_grouped = _records_by_subject(new_records)
    old_subjects = sorted(old_index)
    new_subjects = sorted(new_index)
    old_signatures = {subject_id: _patient_signature(subject_id, old_index, old_grouped) for subject_id in old_subjects}
    new_signatures = {subject_id: _patient_signature(subject_id, new_index, new_grouped) for subject_id in new_subjects}

    patient_diff_rows: list[dict[str, Any]] = []
    for subject_id in sorted(set(old_subjects) | set(new_subjects)):
        old_sig = old_signatures.get(subject_id)
        new_sig = new_signatures.get(subject_id)
        if old_sig == new_sig:
            continue
        diff_fields = []
        for field in ("num_channels", "num_ez", "canonical_channels", "window_feature_names"):
            if (old_sig or {}).get(field) != (new_sig or {}).get(field):
                diff_fields.append(field)
        patient_diff_rows.append(
            {
                "subject_id": subject_id,
                "status": "missing_old" if old_sig is None else "missing_new" if new_sig is None else "different",
                "diff_fields": ",".join(diff_fields),
                "old_num_channels": (old_sig or {}).get("num_channels"),
                "new_num_channels": (new_sig or {}).get("num_channels"),
                "old_num_ez": (old_sig or {}).get("num_ez"),
                "new_num_ez": (new_sig or {}).get("num_ez"),
            }
        )

    run_diff_rows: list[dict[str, Any]] = []
    for subject_id in sorted(set(old_subjects) | set(new_subjects)):
        old_runs = set((old_signatures.get(subject_id) or {}).get("run_ids", []))
        new_runs = set((new_signatures.get(subject_id) or {}).get("run_ids", []))
        for run_id in sorted(old_runs - new_runs):
            run_diff_rows.append({"subject_id": subject_id, "run_id": run_id, "status": "missing_new"})
        for run_id in sorted(new_runs - old_runs):
            run_diff_rows.append({"subject_id": subject_id, "run_id": run_id, "status": "missing_old"})

    mismatch_types = []
    if old_subjects != new_subjects:
        mismatch_types.append("patient_ids")
    if len(old_records) != len(new_records):
        mismatch_types.append("num_run_records")
    if patient_diff_rows:
        mismatch_types.append("patient_schema")
    if run_diff_rows:
        mismatch_types.append("run_ids")
    match = not mismatch_types
    summary = {
        "old_success_cache": str(args.old_success_cache),
        "new_superset_cache": str(args.new_superset_cache),
        "outcome_subset": args.outcome_subset,
        "match": match,
        "mismatch_types": mismatch_types,
        "old_num_patients": len(old_index),
        "new_num_patients": len(new_index),
        "old_num_run_records": len(old_records),
        "new_num_run_records": len(new_records),
        "old_patient_ids_preview": old_subjects[:20],
        "new_patient_ids_preview": new_subjects[:20],
    }
    return summary, patient_diff_rows, run_diff_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare old success-only PKL with superset PKL filtered to success.")
    parser.add_argument("--old-success-cache", type=Path, required=True)
    parser.add_argument("--new-superset-cache", type=Path, required=True)
    parser.add_argument("--outcome-subset", choices=OUTCOME_CHOICES, default="success")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary, patient_rows, run_rows = compare_caches(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "compare_success_filter_summary.json", "w", encoding="utf-8") as fout:
            json.dump(_json_safe(summary), fout, indent=2, ensure_ascii=False, sort_keys=True)
        _write_csv(args.output_dir / "compare_success_filter_patient_diff.csv", patient_rows)
        _write_csv(args.output_dir / "compare_success_filter_run_diff.csv", run_rows)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("MATCH" if summary["match"] else "MISMATCH", flush=True)
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    raise SystemExit(0 if summary["match"] else 1)


if __name__ == "__main__":
    main()
