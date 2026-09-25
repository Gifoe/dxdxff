"""Read a pickle cache in a short-lived process for memory-safe schema audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle


def inspect_cache(path: str | Path) -> dict[str, object]:
    cache = Path(path)
    with cache.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Window cache top-level payload must be a dictionary")
    names = list(map(str, payload.get("window_feature_names", [])))
    records = payload.get("run_records", [])
    patients = payload.get("patient_index", {})
    if not isinstance(records, list) or not isinstance(patients, dict):
        raise ValueError("Window cache must contain run_records and patient_index")
    return {
        "cache_version": payload.get("cache_version"),
        "feature_mode": payload.get("feature_mode"),
        "n_patients": len(patients),
        "n_runs": len(records),
        "window_feature_count": len(names),
        "window_feature_names": names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inspect_cache(args.cache)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
