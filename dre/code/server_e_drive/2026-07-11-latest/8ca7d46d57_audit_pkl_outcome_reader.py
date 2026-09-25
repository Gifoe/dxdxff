from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ez_dataset import _channel_meta_outcome_details, build_or_load_run_records, flatten_window_samples


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


def _center_for_patient(subject_id: str, meta: dict[str, Any]) -> str:
    for key in ("source_center", "center", "source_dataset"):
        value = str(meta.get(key, "")).strip().lower()
        if value:
            return value
    if ":" in subject_id:
        return subject_id.split(":", 1)[0].strip().lower()
    return "unknown"


def _center_for_record(record: dict[str, Any], patient_index: dict[str, dict[str, Any]]) -> str:
    for source in (record, record.get("sample", {}), patient_index.get(str(record.get("subject_id")), {})):
        if not isinstance(source, dict):
            continue
        for key in ("source_center", "center", "source_dataset"):
            value = str(source.get(key, "")).strip().lower()
            if value:
                return value
    subject_id = str(record.get("subject_id", ""))
    if ":" in subject_id:
        return subject_id.split(":", 1)[0].strip().lower()
    return "unknown"


def _labels_for_patient(subject_id: str, meta: dict[str, Any], records_by_subject: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    labels = meta.get("labels")
    if labels is None and records_by_subject.get(subject_id):
        labels = records_by_subject[subject_id][0].get("labels")
    return np.asarray(labels if labels is not None else [], dtype=np.float32)


def _feature_shape(sample: dict[str, Any]) -> str:
    arr = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
    return "x".join(str(dim) for dim in arr.shape)


def _raw_shape(sample: dict[str, Any]) -> str:
    if "raw_waveform" not in sample:
        return "missing"
    arr = np.asarray(sample["raw_waveform"])
    return "x".join(str(dim) for dim in arr.shape)


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


def _channel_meta_sources_for_subject(
    subject_id: str,
    patient_index: dict[str, dict[str, Any]],
    records_by_subject: dict[str, list[dict[str, Any]]],
) -> list[Any]:
    sources = [patient_index.get(subject_id, {}).get("channel_meta")]
    for record in records_by_subject.get(subject_id, []):
        sources.append(record.get("channel_meta"))
        sample = record.get("sample", {})
        if isinstance(sample, dict):
            sources.append(sample.get("channel_meta"))
    return sources


def _channel_meta_keys_preview(channel_sources: Sequence[Any]) -> list[str]:
    keys: set[str] = set()
    for source in channel_sources:
        if not isinstance(source, list):
            continue
        for item in source[:5]:
            if isinstance(item, dict):
                keys.update(str(key) for key in item.keys())
    return sorted(keys)[:30]


def audit_cache(args: argparse.Namespace) -> dict[str, Any]:
    reader_args = argparse.Namespace(
        window_cache_path=str(args.cache_path),
        sample_cache_path=None,
        outcome_subset=args.outcome_subset,
        include_raw_waveform=bool(args.include_raw_waveform),
        drop_high_ez_fraction_lzu=False,
        output_dir=str(args.output_dir) if args.output_dir is not None else None,
    )
    run_records, patient_index = build_or_load_run_records(reader_args)
    reader_summary: dict[str, Any] = {}
    if args.output_dir is not None:
        reader_summary_path = args.output_dir / "data_reader_outcome_subset_summary.json"
        if reader_summary_path.exists():
            with open(reader_summary_path, "r", encoding="utf-8") as fin:
                reader_summary = json.load(fin)
    flattened = flatten_window_samples(run_records, include_raw_waveform=bool(args.include_raw_waveform))

    records_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        records_by_subject[str(record.get("subject_id"))].append(record)

    patient_rows: list[dict[str, Any]] = []
    patient_outcome_counts: Counter[str] = Counter()
    patient_center_counts: Counter[str] = Counter()
    channel_meta_outcome_detected_count = 0
    channel_meta_source_fields: Counter[str] = Counter()
    unknown_channel_meta_previews: list[dict[str, Any]] = []
    for subject_id in sorted(patient_index):
        meta = patient_index[subject_id]
        labels = _labels_for_patient(subject_id, meta, records_by_subject)
        valid = labels >= 0.0
        num_channels = int(labels.size)
        num_ez = int(np.sum(labels[valid] > 0.5)) if labels.size else 0
        valid_count = int(np.sum(valid)) if labels.size else 0
        ez_fraction = float(num_ez / valid_count) if valid_count else 0.0
        outcome_group = str(meta.get("outcome_group", "unknown"))
        center = _center_for_patient(subject_id, meta)
        detected_from_channel_meta = False
        channel_sources = _channel_meta_sources_for_subject(subject_id, patient_index, records_by_subject)
        for channel_source in channel_sources:
            channel_group, _, source_fields = _channel_meta_outcome_details(channel_source)
            channel_meta_source_fields.update(source_fields)
            if channel_group != "unknown":
                detected_from_channel_meta = True
        if detected_from_channel_meta:
            channel_meta_outcome_detected_count += 1
        if outcome_group == "unknown":
            unknown_channel_meta_previews.append(
                {
                    "subject_id": subject_id,
                    "available_channel_meta_keys": _channel_meta_keys_preview(channel_sources),
                }
            )
        patient_outcome_counts[outcome_group] += 1
        patient_center_counts[center] += 1
        patient_rows.append(
            {
                "subject_id": subject_id,
                "outcome_group": outcome_group,
                "center": center,
                "num_channels": num_channels,
                "num_ez": num_ez,
                "ez_fraction": ez_fraction,
                "num_run_records": len(records_by_subject.get(subject_id, [])),
            }
        )

    run_outcome_counts: Counter[str] = Counter()
    run_center_counts: Counter[str] = Counter()
    for record in run_records:
        subject_id = str(record.get("subject_id"))
        run_outcome_counts[str(patient_index.get(subject_id, {}).get("outcome_group", "unknown"))] += 1
        run_center_counts[_center_for_record(record, patient_index)] += 1

    feature_dim_values = Counter()
    feature_name_len_values = Counter()
    raw_shape_counts = Counter()
    raw_sfreq_values = Counter()
    raw_present = 0
    raw_missing = 0
    for sample in flattened:
        features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
        if features.ndim >= 1:
            feature_dim_values[int(features.shape[-1])] += 1
        names = sample.get("window_feature_names")
        feature_name_len_values[len(names) if names is not None else -1] += 1
        if sample.get("raw_waveform_missing"):
            raw_missing += 1
            raw_shape_counts["missing"] += 1
        elif "raw_waveform" in sample:
            raw_present += 1
            raw_shape_counts[_raw_shape(sample)] += 1
            if "raw_temporal_sfreq" in sample:
                raw_sfreq_values[str(sample["raw_temporal_sfreq"])] += 1
        else:
            raw_missing += 1
            raw_shape_counts["not_requested_or_missing"] += 1

    run_subjects = {str(record.get("subject_id")) for record in run_records}
    index_subjects = set(patient_index)
    missing_from_index = sorted(run_subjects - index_subjects)
    patients_without_runs = sorted(index_subjects - run_subjects)

    by_center_rows = []
    for center in sorted(set(patient_center_counts) | set(run_center_counts)):
        by_center_rows.append(
            {
                "center": center,
                "patient_count": int(patient_center_counts.get(center, 0)),
                "run_count": int(run_center_counts.get(center, 0)),
            }
        )
    raw_shape_rows = [{"raw_waveform_shape": key, "count": int(value)} for key, value in sorted(raw_shape_counts.items())]

    summary = {
        "cache_path": str(args.cache_path),
        "outcome_subset": args.outcome_subset,
        "num_patients": len(patient_index),
        "num_run_records": len(run_records),
        "num_flattened_samples": len(flattened),
        "patient_counts_by_outcome_group": dict(sorted(patient_outcome_counts.items())),
        "run_counts_by_outcome_group": dict(sorted(run_outcome_counts.items())),
        "centers_patient_counts": dict(sorted(patient_center_counts.items())),
        "centers_run_counts": dict(sorted(run_center_counts.items())),
        "run_subjects_missing_from_patient_index": missing_from_index,
        "patient_index_subjects_without_run_records": patients_without_runs,
        "window_feature_dim_values": {str(key): int(value) for key, value in sorted(feature_dim_values.items())},
        "window_feature_names_length_values": {
            str(key): int(value) for key, value in sorted(feature_name_len_values.items())
        },
        "label_stats": {
            "patients": patient_rows,
            "total_ez_channels": int(sum(row["num_ez"] for row in patient_rows)),
        },
        "raw_waveform_present_count": raw_present,
        "raw_waveform_missing_count": raw_missing,
        "raw_waveform_shapes_top_counts": dict(raw_shape_counts.most_common(20)),
        "raw_temporal_sfreq_values": dict(sorted(raw_sfreq_values.items())),
        "channel_meta_outcome_detected_count": channel_meta_outcome_detected_count,
        "top_outcome_source_fields": dict(channel_meta_source_fields.most_common(20)),
        "unknown_patients_channel_meta_keys_preview": unknown_channel_meta_previews[:20],
        "reader_outcome_subset_summary": reader_summary,
        "warnings": [],
    }

    reader_unknown_count = int(
        (reader_summary.get("patient_counts_by_outcome_group") or {}).get("unknown", patient_outcome_counts.get("unknown", 0))
    )
    if args.outcome_subset == "success_failure" and reader_unknown_count > 0:
        summary["warnings"].append(
            f"success_failure subset has {reader_unknown_count} unknown-outcome patient(s) before filtering."
        )

    if args.require_success_failure:
        if patient_outcome_counts.get("success", 0) <= 0 or patient_outcome_counts.get("failure", 0) <= 0:
            raise ValueError("--require-success-failure needs at least one success and one failure patient.")
    if args.require_raw and raw_missing > 0:
        raise ValueError(f"--require-raw failed: {raw_missing} flattened sample(s) lack raw_waveform.")

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "pkl_outcome_reader_audit.json", "w", encoding="utf-8") as fout:
            json.dump(_json_safe(summary), fout, indent=2, ensure_ascii=False, sort_keys=True)
        _write_csv(output_dir / "pkl_outcome_reader_patients.csv", patient_rows)
        _write_csv(output_dir / "pkl_outcome_reader_by_center.csv", by_center_rows)
        _write_csv(output_dir / "pkl_outcome_reader_raw_shapes.csv", raw_shape_rows)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit NeuroEZ-C superset PKL outcome reader behavior.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--outcome-subset", choices=OUTCOME_CHOICES, default="all")
    parser.add_argument("--require-success-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-raw-waveform", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-raw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = audit_cache(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
