"""Strict phase-0 ictal 60-second raw-to-A1 cohort alignment audit.

The public output contains only aggregate counts/hashes, no patient, channel,
seizure, label, waveform, or fold-row records. It does not train or evaluate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED = {
    "feature": "9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087",
    "raw": "011d7ffaa55469c9f34259be04b7136a987d6d95f0ce9cc443a06751557db733",
    "manifest": "fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("run_records"), list):
        raise RuntimeError("Cache schema lacks run_records list")
    return payload


def scalar_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    try:
        x, y = float(a), float(b)
        if np.isfinite(x) and np.isfinite(y):
            return abs(x-y) <= 1e-8
    except (TypeError, ValueError):
        pass
    return str(a) == str(b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    hashes = {name: sha256(getattr(args, name if name != "feature" else "feature_cache"))
              for name in ("feature", "manifest")}
    hashes["raw"] = sha256(args.raw_cache)
    if hashes != EXPECTED:
        raise RuntimeError(f"Frozen phase-0 input SHA mismatch: {hashes}")

    sys.path.insert(0, str(args.source_root))
    from neuroez_c.dual_view_data import _channel_names, _field, _record_key  # noqa: E402

    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        manifest = list(csv.DictReader(stream))
    if len(manifest) != 400 or {int(row["outer_fold"]) for row in manifest} != set(range(1, 6)):
        raise RuntimeError("Frozen five-fold manifest shape changed")
    fold_roles = {}
    subjects = set()
    for fold in range(1, 6):
        rows = [r for r in manifest if int(r["outer_fold"]) == fold]
        ids = [r["subject_id"] for r in rows]
        roles = Counter(r["split_role"] for r in rows)
        if (len(rows) != 80 or len(set(ids)) != 80 or
                set(roles) != {"fit", "validation", "test"} or
                roles["validation"] != 13 or not 15 <= roles["test"] <= 17 or
                roles["fit"] != 80 - roles["validation"] - roles["test"]):
            raise RuntimeError("Frozen fold membership/count mismatch")
        if subjects and set(ids) != subjects:
            raise RuntimeError("Different patient universe across frozen folds")
        subjects = set(ids)
        fold_roles[str(fold)] = dict(sorted(roles.items()))
    if len(subjects) != 80:
        raise RuntimeError("Historical cohort not 80 patients")

    feature_payload = load_cache(args.feature_cache)
    raw_payload = load_cache(args.raw_cache)
    feature_all = feature_payload["run_records"]
    raw_all = raw_payload["run_records"]
    patient_index = feature_payload.get("patient_index")
    if not isinstance(patient_index, dict) or not subjects.issubset(patient_index):
        raise RuntimeError("Frozen historical patient_index missing cohort members")
    canonical_channels = 0
    canonical_labeled_channels = 0
    canonical_positive_channels = 0
    canonical_by_patient = {}
    for subject in subjects:
        meta = patient_index[subject]
        names = list(meta["canonical_channels"])
        labels = np.asarray(meta["labels"])
        mask = np.asarray(meta.get("label_mask", np.ones(len(names), dtype=bool)), dtype=bool)
        if (labels.ndim != 1 or len(names) != len(labels) or len(mask) != len(names) or
                len(names) != len(set(names)) or not np.isin(labels, [-1, 0, 1]).all()):
            raise RuntimeError("Historical canonical patient label schema invalid")
        canonical_by_patient[subject] = {name: (labels[i], mask[i]) for i, name in enumerate(names)}
        canonical_channels += len(names)
        canonical_labeled_channels += int(np.sum(mask & (labels >= 0)))
        canonical_positive_channels += int(np.sum(mask & (labels == 1)))
    feature = [r for r in feature_all if str(r["subject_id"]) in subjects]
    raw = [r for r in raw_all if str(r["subject_id"]) in subjects]
    feature_index = {_record_key(r): r for r in feature}
    raw_index = {_record_key(r): r for r in raw}
    if len(feature_index) != len(feature) or len(raw_index) != len(raw):
        raise RuntimeError("Duplicate compound record key")
    if len(feature_index) != 256 or {k[0] for k in feature_index} != subjects:
        raise RuntimeError("Feature cache does not reproduce 80-patient/256-record A1 cohort")
    unmatched = set(feature_index) - set(raw_index)
    extra_raw = set(raw_index) - set(feature_index)
    counters = Counter()
    valid_durations = Counter()
    valid_window_counts = Counter()
    names_by_subject = defaultdict(dict)
    for key, fr in feature_index.items():
        rr = raw_index.get(key)
        if rr is None:
            continue
        fs = fr.get("sample", {})
        rs = rr.get("sample", {})
        fnames, rnames = _channel_names(fr), _channel_names(rr)
        flabels = np.asarray(fr.get("labels"))
        rlabels = np.asarray(rr.get("labels"))
        if flabels.ndim != 1 or rlabels.ndim != 1 or len(flabels) != len(fnames) or len(rlabels) != len(rnames):
            counters["invalid_label_shape_records"] += 1
            continue
        rmap = {name: idx for idx, name in enumerate(rnames)}
        if len(rmap) != len(rnames) or any(name not in rmap for name in fnames):
            counters["channel_alignment_failed_records"] += 1
            continue
        for idx, name in enumerate(fnames):
            counters["matched_record_channels"] += 1
            if name not in canonical_by_patient[key[0]]:
                counters["missing_canonical_patient_channels"] += 1
            if not scalar_equal(flabels[idx], rlabels[rmap[name]]):
                counters["label_mismatch_channels"] += 1
            prior = names_by_subject[key[0]].get(name)
            if prior is not None and not scalar_equal(prior, flabels[idx]):
                counters["cross_record_label_conflicts"] += 1
            names_by_subject[key[0]][name] = flabels[idx]
        for field in ("seizure_onset_sec", "source_seizure_id", "start_sec", "end_sec"):
            if not scalar_equal(_field(fr, field), _field(rr, field)):
                counters[f"{field}_mismatch_records"] += 1
        waveform = np.asarray(rs.get("raw_waveform"))
        rate = float(rs.get("raw_temporal_sfreq", 0) or 0)
        duration = float(rs.get("raw_temporal_duration_sec", 0) or 0)
        valid = int(rs.get("raw_valid_samples", -1))
        start = int(rs.get("raw_valid_start_sample", -1))
        if waveform.ndim != 2 or waveform.shape != (len(rnames), 15000) or rate != 250 or duration != 60:
            counters["invalid_raw_shape_rate_duration_records"] += 1
        if start < 0 or valid < 0 or start + valid > 15000:
            counters["invalid_valid_region_records"] += 1
        if valid != 15000 or start != 0:
            counters["incomplete_60_second_records"] += 1
        valid_durations[str(round(valid / rate, 3)) if rate else "invalid"] += 1
        if 0 <= start and 0 <= valid and start + valid <= 15000:
            n_valid_windows = sum(250 * w >= start and 250 * (w + 2) <= start + valid
                                  for w in range(59))
            valid_window_counts[n_valid_windows] += 1
        if not np.isfinite(waveform).all():
            counters["nonfinite_raw_records"] += 1
        counters["matched_records"] += 1

    counters["canonical_channels_absent_from_records"] = sum(
        len(set(canonical_by_patient[subject]) - set(names_by_subject[subject]))
        for subject in subjects)

    failures = {k: int(v) for k, v in counters.items() if
                k not in {"matched_records", "matched_record_channels", "incomplete_60_second_records",
                          "cross_record_label_conflicts"} and v}
    if unmatched:
        failures["missing_raw_records"] = len(unmatched)
    result = {
        "status": "STOP_RAW_ALIGNMENT" if failures else
                  ("PASS_WITH_MASKED_PARTIAL_RECORDS" if counters["incomplete_60_second_records"] else "PASS"),
        "reason": "Record/channel/label/onset mismatch" if failures else
                  ("Exact cohort alignment; source-short records require valid-window masking" if counters["incomplete_60_second_records"] else
                   "Exact alignment established"),
        "input_sha256": hashes,
        "manifest_subjects": len(subjects), "manifest_folds": 5,
        "fold_role_counts": fold_roles,
        "feature_cache_all_subjects": len({str(r["subject_id"]) for r in feature_all}),
        "feature_cache_all_records": len(feature_all),
        "raw_cache_all_subjects": len({str(r["subject_id"]) for r in raw_all}),
        "raw_cache_all_records": len(raw_all),
        "historical_a1_feature_records": len(feature_index),
        "historical_a1_raw_records": len(raw_index),
        "matched_records": int(counters["matched_records"]),
        "unmatched_feature_records": len(unmatched),
        "extra_raw_records_in_80_subjects": len(extra_raw),
        "matched_record_channels": int(counters["matched_record_channels"]),
        "unique_patient_channels": sum(len(v) for v in names_by_subject.values()),
        "historical_canonical_patient_channels": canonical_channels,
        "historical_canonical_labeled_channels": canonical_labeled_channels,
        "historical_canonical_positive_channels": canonical_positive_channels,
        "incomplete_60_second_records": int(counters["incomplete_60_second_records"]),
        "valid_raw_duration_seconds_distribution": dict(sorted(valid_durations.items(), key=lambda x: float(x[0]) if x[0] != "invalid" else -1)),
        "fully_valid_two_second_windows_distribution": dict(sorted(valid_window_counts.items())),
        "cross_record_source_label_conflicts_inherited_from_historical_cache": int(counters["cross_record_label_conflicts"]),
        "historical_canonical_patient_labels_required": True,
        "partial_record_policy": "Use 60-s tensor with exact raw_valid_start_sample/raw_valid_samples masks; never treat zero padding as observed EEG",
        "other_alignment_failures": failures,
        "compound_record_key_sha256": hashlib.sha256(json.dumps(sorted(feature_index), ensure_ascii=False).encode()).hexdigest(),
        "no_patient_seizure_channel_or_label_rows_exported": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
