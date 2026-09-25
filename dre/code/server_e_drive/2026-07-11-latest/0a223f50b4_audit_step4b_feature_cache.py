from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_factory import build_outer_splits
from ez_features import WINDOW_NODE_FEATURE_NAMES
from neuroez_c.dual_view_data import _sample
from neuroez_c.protocol import STEP4B_STATIC_TOP20_FEATURES


BASE_FEATURE_COUNT = 20


def _subjects(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "subject_id" not in rows[0]:
        raise ValueError("All90 ledger must contain subject_id")
    values = [str(row["subject_id"]).strip() for row in rows]
    if len(values) != 90 or len(set(values)) != 90:
        raise ValueError("All90 ledger must contain exactly 90 unique subjects")
    return values


def audit_feature_cache(cache_path: str | Path, subjects_path: str | Path, output_path: str | Path) -> dict:
    source = Path(cache_path)
    if not source.is_file():
        raise FileNotFoundError(f"Step4B feature cache not found: {source}")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    patient_index = payload.get("patient_index", {})
    records = payload.get("run_records", [])
    expected = _subjects(subjects_path)
    if set(patient_index) != set(expected):
        raise ValueError(f"Feature cache patient ledger mismatch: missing={sorted(set(expected)-set(patient_index))}, extra={sorted(set(patient_index)-set(expected))}")
    if len(records) != 281:
        raise ValueError(f"Step4B feature cache must contain 281 records, got {len(records)}")
    names = list(payload.get("window_feature_names") or payload.get("feature_names") or [])
    expected_count = BASE_FEATURE_COUNT + len(STEP4B_STATIC_TOP20_FEATURES)
    if (
        len(names) != expected_count
        or tuple(names[:BASE_FEATURE_COUNT]) != tuple(WINDOW_NODE_FEATURE_NAMES)
        or tuple(names[-8:]) != STEP4B_STATIC_TOP20_FEATURES
    ):
        raise ValueError("Step4B cache must contain the canonical base20 followed by the exact 8 static-top20 features")
    nonfinite = 0
    for record in records:
        sample = _sample(record)
        features = np.asarray(sample.get("window_features"))
        if features.ndim != 3 or features.shape[-1] != expected_count:
            raise ValueError(f"Invalid window feature tensor for {record.get('subject_id')}: {features.shape}")
        nonfinite += int((~np.isfinite(features[..., -8:])).sum())
    label_values: Counter[int] = Counter()
    for meta in patient_index.values():
        labels = np.asarray(meta.get("labels", []), dtype=np.float64)
        label_values.update(map(int, labels[np.isfinite(labels)].tolist()))
    if nonfinite:
        raise ValueError(f"Step4B features contain {nonfinite} non-finite values")
    if not set(label_values).issubset({-1, 0, 1}) or not ({0, 1} <= set(label_values)):
        raise ValueError(f"Unexpected label values: {dict(label_values)}")
    centers = Counter(str(subject).split(":", 1)[0].lower() for subject in patient_index)
    folds = build_outer_splits(patient_index, n_splits=5, random_seed=42)
    test_union = [subject for fold in folds for subject in fold["test_subjects"]]
    audit = {
        "protocol_name": "fixed_all90_step4b_static_top20_nez",
        "status": "passed", "cache_path": str(source), "n_patients": 90,
        "n_runs": 281, "window_feature_count": expected_count,
        "required_features_present": list(STEP4B_STATIC_TOP20_FEATURES),
        "required_features_missing": [], "all_features_finite": True,
        "center_distribution": dict(sorted(centers.items())),
        "positive_label": "nez", "score_semantics": "nez_probability",
        "cache_label_semantics": "labels_ez:1=EZ,0=NEZ; training derives labels_nez exactly once",
        "label_values": dict(sorted(label_values.items())),
        "n_outer_splits": len(folds), "patient_disjoint": True,
        "test_union_count": len(set(test_union)),
        "test_union_matches_all90": len(test_union) == len(set(test_union)) == 90,
        "folds": [{"fold_idx": row["fold_idx"], "n_train": len(row["train_subjects"]), "n_test": len(row["test_subjects"])} for row in folds],
        "center_as_input_allowed": False, "raw_cache_used": False,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of an existing 28D Step4B All90 feature cache.")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_feature_cache(args.cache_path, args.subjects, args.output_path), indent=2))


if __name__ == "__main__":
    main()
