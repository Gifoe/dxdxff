from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuroez_c.quality import QUALITY_LABELS, is_simple_quality_metadata, normalize_quality_label


QUALITY_FIELD_KEYWORDS = ("quality", "qc", "artifact", "review")


def _load_cache(cache_path: Path) -> dict[str, Any]:
    with cache_path.open("rb") as fin:
        payload = pickle.load(fin)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported cache payload in {cache_path}: expected dict.")
    if not isinstance(payload.get("run_records"), list):
        raise ValueError(f"Unsupported cache payload in {cache_path}: missing run_records list.")
    return payload


def _flatten_simple_fields(obj: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if key in {"window_features", "window_adjacency", "labels", "channel_names_norm"}:
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and is_simple_quality_metadata(value):
            out.update(_flatten_simple_fields(value, prefix=name))
        elif is_simple_quality_metadata(value):
            out[name] = value
    return out


def _record_field_values(record: Mapping[str, Any]) -> dict[str, Any]:
    values = _flatten_simple_fields(record)
    sample = record.get("sample")
    if isinstance(sample, Mapping):
        for key, value in _flatten_simple_fields(sample).items():
            values.setdefault(key, value)
            values.setdefault(f"sample.{key}", value)
    return values


def audit_quality_fields(cache_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    cache_path = Path(cache_path)
    payload = _load_cache(cache_path)
    run_records = payload["run_records"]
    field_counts: Counter[str] = Counter()
    normalized_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, Any] = {}
    candidate_fields: set[str] = set()

    for record in run_records:
        if not isinstance(record, Mapping):
            continue
        values = _record_field_values(record)
        for field, value in values.items():
            if value is None:
                continue
            field_counts[field] += 1
            examples.setdefault(field, value)
            lower_field = field.lower()
            looks_like_quality_name = any(token in lower_field for token in QUALITY_FIELD_KEYWORDS)
            try:
                normalized = normalize_quality_label(value)
                normalized_counts[field][normalized] += 1
                candidate_fields.add(field)
            except ValueError:
                normalized = ""
            if looks_like_quality_name and (normalized or isinstance(value, str)):
                candidate_fields.add(field)

    rows = []
    for field in sorted(field_counts):
        counts = normalized_counts.get(field, Counter())
        rows.append(
            {
                "field": field,
                "n_records_with_field": int(field_counts[field]),
                "example_value": str(examples.get(field, "")),
                "good": int(counts.get("good", 0)),
                "review": int(counts.get("review", 0)),
                "poor": int(counts.get("poor", 0)),
                "is_candidate": field in candidate_fields,
            }
        )
    result = {
        "cache_path": str(cache_path),
        "record_count": int(len(run_records)),
        "candidate_fields": sorted(candidate_fields),
        "quality_labels": list(QUALITY_LABELS),
        "field_count": len(rows),
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "edf_quality_field_audit.json").open("w", encoding="utf-8") as fout:
            json.dump(result, fout, indent=2, ensure_ascii=False, sort_keys=True)
        pd.DataFrame(rows).to_csv(output / "edf_quality_field_audit.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a NeuroEZ window cache for EDF quality label fields.")
    parser.add_argument("--cache_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    result = audit_quality_fields(args.cache_path, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if not result["candidate_fields"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
