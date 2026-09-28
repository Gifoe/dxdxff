"""Run the repository's validated compound-key raw alignment on A1 metadata."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.source_root))
    from neuroez_c.dual_view_data import RawAlignmentStore

    with a.manifest.open(newline="", encoding="utf-8-sig") as h:
        ids = {row["subject_id"] for row in csv.DictReader(h)}
    if len(ids) != 80:
        raise RuntimeError(f"Frozen A1 manifest must contain 80 patients, found {len(ids)}")
    with a.feature_cache.open("rb") as h:
        payload = pickle.load(h)
    records = [r for r in payload["run_records"] if str(r["subject_id"]) in ids]
    del payload
    if len(records) != 256:
        raise RuntimeError(f"A1 feature cohort must have 256 records, found {len(records)}")
    store = RawAlignmentStore(
        records,
        feature_cache_path=a.feature_cache,
        raw_cache_path=a.raw_cache,
        raw_target_samples=500,
        raw_target_sampling_rate=250.0,
    )
    store.assert_formal_coverage(min_channel_match_rate=1.0, min_window_match_rate=1.0, expected_patients=80)
    audit = store.audit
    safe_keys = (
        "n_feature_patients", "n_raw_patients", "n_matched_patients",
        "n_feature_records", "n_raw_records", "n_matched_records",
        "n_feature_channels", "n_raw_channels", "n_matched_channels",
        "n_feature_windows", "n_raw_windows", "n_matched_windows",
        "patient_match_rate", "record_match_rate", "channel_match_rate", "window_match_rate",
        "raw_nonfinite_count", "raw_target_samples", "raw_target_sampling_rate", "status",
    )
    safe = {key: audit[key] for key in safe_keys}
    safe["a1_manifest_patients"] = len(ids)
    safe["a1_feature_cache"] = str(a.feature_cache)
    safe["raw_cache"] = str(a.raw_cache)
    safe["method"] = "neuroez_c.dual_view_data.RawAlignmentStore compound-key alignment"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    print(json.dumps(safe, indent=2), flush=True)


if __name__ == "__main__":
    main()
