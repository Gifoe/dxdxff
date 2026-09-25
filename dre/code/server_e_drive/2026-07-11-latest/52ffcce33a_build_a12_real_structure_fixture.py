"""Create a tiny privacy-preserving cache fixture from a read-only real cache."""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a12_vcsn.cache_filtering import filter_cache_records, resolve_channel_names


def _anon(value: object) -> str: return "fixture:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--cache-path", required=True); parser.add_argument("--output-path", required=True); parser.add_argument("--invalid-record-policy", choices=["fail", "drop"], default="drop"); parser.add_argument("--max-subjects", type=int, default=4); parser.add_argument("--max-runs-per-subject", type=int, default=2); parser.add_argument("--max-channels", type=int, default=8); parser.add_argument("--max-windows", type=int, default=16)
    args = parser.parse_args(argv)
    with Path(args.cache_path).open("rb") as handle: cache = pickle.load(handle)
    patient_index, records = cache["patient_index"], cache["run_records"]
    records, filter_audit = filter_cache_records(records, patient_index, policy=args.invalid_record_policy)
    selected, new_index, per_subject = [], {}, {}
    for record in records:
        subject = str(record.get("subject_id", "")); sample = record.get("sample") or {}; values = np.asarray(sample.get("window_features"))
        names = resolve_channel_names(record, patient_index, values.shape[1] if values.ndim == 3 else 0)
        if values.ndim != 3 or names is None: continue
        anon = _anon(subject)
        if anon not in per_subject and len(per_subject) >= args.max_subjects: continue
        if per_subject.get(anon, 0) >= args.max_runs_per_subject: continue
        per_subject[anon] = per_subject.get(anon, 0) + 1
        channels, windows = min(args.max_channels, values.shape[1]), min(args.max_windows, values.shape[0])
        selected.append({"subject_id": anon, "run_id": f"run-{per_subject[anon]}", "canonical_channels": [str(x) for x in names[:channels]], "sample": {"window_features": values[:windows, :channels, :].copy()}})
        new_index[anon] = {"canonical_channels": [str(x) for x in names[:channels]]}
    fixture = {"cache_version": "a12_real_structure_fixture", "run_records": selected, "patient_index": new_index, "window_feature_names": list(cache.get("window_feature_names") or cache.get("feature_names") or [])}
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_path).open("wb") as handle: pickle.dump(fixture, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print({"n_subjects": len(new_index), "n_runs": len(selected), "filter_summary": {key: filter_audit[key] for key in ("input_records", "retained_records", "excluded_records", "retained_patients", "zero_valid_run_patients")}, "output_path": str(Path(args.output_path).resolve())})


if __name__ == "__main__": main()
