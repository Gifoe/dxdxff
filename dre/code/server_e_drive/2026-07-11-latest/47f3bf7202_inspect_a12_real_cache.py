"""Read-only structural audit for an A12 all-window cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a12_vcsn.utils import sha256_file, write_json
from a12_vcsn.cache_filtering import filter_cache_records


def _hash_subject(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _channel_source(record: dict, sample: dict, meta: dict, width: int) -> str | None:
    for key, holder in (("run_record.canonical_channels", record), ("sample.canonical_channels", sample), ("run_record.channel_names", record), ("sample.channel_names", sample), ("patient_index.canonical_channels", meta)):
        value = holder.get(key.split(".")[-1])
        if isinstance(value, (list, tuple)) and len(value) == width:
            return key
    return None


def inspect_cache(path: str | Path, *, max_preview_records: int = 20) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    source = Path(path)
    with source.open("rb") as handle: cache = pickle.load(handle)
    if not isinstance(cache, dict): raise ValueError("cache top-level object must be a mapping")
    records, patient_index = cache.get("run_records"), cache.get("patient_index")
    if not isinstance(records, list) or not isinstance(patient_index, dict): raise ValueError("cache requires run_records list and patient_index mapping")
    rows, subject_rows, sources = [], {}, {}
    seen_runs, duplicate_runs = set(), []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            rows.append({"record_index": index, "valid": False, "reason": "not_mapping"}); continue
        subject = str(record.get("subject_id", "")); sample = record.get("sample") or {}; values = np.asarray(sample.get("window_features"))
        run_id = str(record.get("run_id", record.get("record_id", f"index:{index}")))
        run_id_hash = _hash_subject(run_id)
        subject_hash = _hash_subject(subject)
        key = (subject_hash, run_id)
        if key in seen_runs: duplicate_runs.append(key)
        seen_runs.add(key)
        shape = list(values.shape) if values.ndim == 3 else None
        channel_source = _channel_source(record, sample, patient_index.get(subject, {}) or {}, values.shape[1] if values.ndim == 3 else 0)
        mask = sample.get("window_mask", record.get("window_mask")); mask_shape = list(np.asarray(mask).shape) if mask is not None else None
        finite = np.isfinite(values) if values.ndim == 3 and np.issubdtype(values.dtype, np.number) else np.zeros(0, dtype=bool)
        rows.append({"record_index": index, "subject_id_hash": subject_hash, "run_id_hash": run_id_hash, "valid": values.ndim == 3 and bool(subject), "window_features_shape": json.dumps(shape), "n_windows": int(values.shape[0]) if values.ndim == 3 else 0, "n_channels": int(values.shape[1]) if values.ndim == 3 else 0, "feature_dim": int(values.shape[2]) if values.ndim == 3 else 0, "channel_name_source": channel_source, "window_mask_shape": json.dumps(mask_shape), "nan_count": int(np.isnan(values).sum()) if values.ndim == 3 else 0, "inf_count": int(np.isinf(values).sum()) if values.ndim == 3 else 0})
        sources[channel_source or "missing"] = sources.get(channel_source or "missing", 0) + 1
        if subject:
            entry = subject_rows.setdefault(subject_hash, {"subject_id_hash": subject_hash, "n_runs": 0, "n_channels_min": None, "n_channels_max": None, "n_windows_min": None, "n_windows_max": None})
            entry["n_runs"] += 1
            if values.ndim == 3:
                for name, value in (("n_channels", values.shape[1]), ("n_windows", values.shape[0])):
                    entry[f"{name}_min"] = value if entry[f"{name}_min"] is None else min(entry[f"{name}_min"], value); entry[f"{name}_max"] = value if entry[f"{name}_max"] is None else max(entry[f"{name}_max"], value)
    report = {"cache_path": str(source.resolve()), "cache_sha256": sha256_file(source), "top_level_type": type(cache).__name__, "top_level_keys": sorted(map(str, cache.keys())), "run_records_type": type(records).__name__, "n_run_records": len(records), "patient_index_type": type(patient_index).__name__, "n_patient_index": len(patient_index), "feature_names_key": "feature_names" if "feature_names" in cache else "window_feature_names" if "window_feature_names" in cache else None, "feature_names_length": len(cache.get("feature_names") or cache.get("window_feature_names") or []), "duplicate_subject_runs": len(duplicate_runs), "channel_name_sources": sources, "axis_interpretation": "window_features=[window,channel,feature]"}
    return report, pd.DataFrame(rows), pd.DataFrame(subject_rows.values())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--cache-path", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True); parser.add_argument("--invalid-record-policy", choices=["fail", "drop"], default="fail"); parser.add_argument("--max-preview-records", type=int, default=20)
    args = parser.parse_args(argv); report, runs, subjects = inspect_cache(args.cache_path, max_preview_records=args.max_preview_records)
    with Path(args.cache_path).open("rb") as handle: cache = pickle.load(handle)
    source_hash_before = sha256_file(args.cache_path)
    _, filter_audit = filter_cache_records(cache["run_records"], cache["patient_index"], policy=args.invalid_record_policy)
    source_hash_after = sha256_file(args.cache_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("read-only source cache changed during audit")
    report["filter_audit"] = filter_audit
    report["source_cache_unchanged"] = True
    if args.strict and args.invalid_record_policy == "fail" and (runs["channel_name_source"].isna().any() or report["duplicate_subject_runs"]): raise RuntimeError("cache contract failed: missing channel source or duplicate subject-run")
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    write_json(root / "cache_schema.json", report); write_json(root / "cache_contract_report.json", {"passed": True, **report}); write_json(root / "cache_channel_name_sources.json", report["channel_name_sources"])
    subjects.to_csv(root / "cache_subject_coverage.csv", index=False); runs.to_csv(root / "cache_run_summary.csv", index=False)
    excluded = pd.DataFrame(filter_audit.get("excluded_records_detail", []))
    retained = pd.DataFrame(filter_audit.get("retained_records_detail", []))
    coverage = pd.DataFrame(filter_audit.get("patient_coverage", []))
    excluded.to_csv(root / "excluded_cache_records.csv", index=False)
    retained.to_csv(root / "retained_cache_records.csv", index=False)
    coverage.to_csv(root / "patient_cache_coverage_after_filter.csv", index=False)
    pd.concat([retained.assign(status="retained"), excluded.assign(status="excluded")], ignore_index=True).to_csv(root / "cache_record_filter_audit.csv", index=False)
    write_json(root / "cache_record_filter_summary.json", {key: value for key, value in filter_audit.items() if not key.endswith("_detail") and key != "patient_coverage"})
    write_json(root / "cache_invalid_value_summary.json", {"nan_count": int(runs["nan_count"].sum()), "inf_count": int(runs["inf_count"].sum()), "n_runs_with_invalid": int(((runs["nan_count"] + runs["inf_count"]) > 0).sum())})
    print(json.dumps({"output_dir": str(root.resolve()), "n_runs": len(runs), "n_subjects": len(subjects)}, ensure_ascii=False))


if __name__ == "__main__": main()
