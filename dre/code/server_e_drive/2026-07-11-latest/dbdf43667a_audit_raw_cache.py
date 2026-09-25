from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allowed = set(pd.read_csv(args.subjects)["subject_id"].astype(str))
    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    records = [record for record in payload.get("run_records", []) if str(record.get("subject_id")) in allowed]
    rates = []
    missing = 0
    for record in records:
        sample = record.get("sample", {}) if isinstance(record.get("sample"), dict) else record
        rate = sample.get("raw_temporal_sfreq", record.get("raw_temporal_sfreq"))
        waveform = sample.get("raw_waveform", record.get("raw_waveform"))
        if rate is not None:
            rates.append(float(rate))
        if waveform is None:
            missing += 1
    report = {
        "cache": str(args.cache),
        "n_patients": len({str(record.get("subject_id")) for record in records}),
        "n_runs": len(records),
        "sampling_rates": {str(key): value for key, value in sorted(Counter(rates).items())},
        "missing_raw_waveform": missing,
        "hfo_recovery_supported": bool(rates and min(rates) >= 320.0),
        "hfo_recovery_reason": "raw_temporal_sfreq=250Hz, Nyquist=125Hz" if rates and max(rates) == 250.0 else "see sampling_rates",
    }
    if report["n_patients"] != 90 or report["n_runs"] != 281 or missing:
        raise RuntimeError(f"Raw cache all90 audit failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
