"""Phase-0 metadata-only inspection of the frozen ictal raw/feature caches.

This emits no patient, seizure, channel, label, or waveform records.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np


def summarize(path: Path, raw: bool) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    records = payload.get("run_records")
    if not isinstance(records, list):
        raise RuntimeError("Expected run_records list")
    fields = collections.Counter()
    sample_fields = collections.Counter()
    lengths = collections.Counter()
    sample_rates = collections.Counter()
    durations = collections.Counter()
    valid_seconds = collections.Counter()
    valid_starts = collections.Counter()
    channel_counts = collections.Counter()
    subjects = set()
    missing_waveform = 0
    for record in records:
        fields.update(record.keys())
        sample = record.get("sample", {})
        sample_fields.update(sample.keys())
        subjects.add(str(record.get("subject_id", "")))
        channel_counts[len(sample.get("channel_names_norm", record.get("channel_names_norm", [])))] += 1
        if raw:
            waveform = sample.get("raw_waveform")
            if waveform is None:
                missing_waveform += 1
                continue
            shape = np.shape(waveform)
            if len(shape) != 2:
                raise RuntimeError("Raw waveform not channel-by-time")
            lengths[int(shape[-1])] += 1
            sample_rates[str(sample.get("raw_temporal_sfreq", "missing"))] += 1
            durations[str(sample.get("raw_temporal_duration_sec", "missing"))] += 1
            rate = float(sample.get("raw_temporal_sfreq", 0) or 0)
            if rate > 0:
                valid_seconds[str(round(float(sample.get("raw_valid_samples", 0)) / rate, 3))] += 1
            valid_starts[str(sample.get("raw_valid_start_sample", "missing"))] += 1
    return {
        "cache": "raw" if raw else "feature",
        "run_records": len(records),
        "subjects": len(subjects),
        "record_fields": sorted(fields),
        "sample_fields": sorted(sample_fields),
        "channel_count_min": min(channel_counts, default=None),
        "channel_count_max": max(channel_counts, default=None),
        "raw_waveform_missing_records": missing_waveform if raw else None,
        "raw_length_samples": dict(sorted(lengths.items())),
        "raw_sampling_rates": dict(sorted(sample_rates.items())),
        "raw_duration_seconds": dict(sorted(durations.items())),
        "raw_valid_seconds": dict(sorted(valid_seconds.items(), key=lambda item: float(item[0]))),
        "raw_valid_start_samples": dict(sorted(valid_starts.items())),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = {"feature": summarize(args.feature_cache, False),
              "raw": summarize(args.raw_cache, True)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"run_records": v["run_records"], "subjects": v["subjects"],
                          "raw_length_samples": v["raw_length_samples"],
                          "raw_sampling_rates": v["raw_sampling_rates"],
                          "raw_duration_seconds": v["raw_duration_seconds"]}
                      for k, v in output.items()}), flush=True)


if __name__ == "__main__":
    main()
