"""Rebuild the exact historical B0 feature table in private server storage."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "feature-cache", "cohort-manifest", "fold-manifest", "split-manifest",
                 "feature-manifest", "b0-ledger", "replay-audit", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.source_root.resolve()))
    from outcome_hifos.cache_schema import load_cache_contract
    from task1_baselines.cache_io import task1_feature_records
    from task1_baselines.feature_aggregation import build_channel_feature_table
    from task1_baselines.fold_protocol import load_task1_sensitivity_protocol
    from task1_baselines.patient_controls import feature_columns

    a.output_dir.mkdir(parents=True, exist_ok=True)
    destination = a.output_dir / "FEATURE_TABLE_PRIVATE.pkl"
    if destination.exists():
        raise RuntimeError("Private feature table already exists; inspect and reuse it")
    folds, splits = load_task1_sensitivity_protocol(a.cohort_manifest, a.fold_manifest,
                                                     a.split_manifest, expected_subjects=80)
    contract = load_cache_contract(a.feature_cache)
    table, manifest = build_channel_feature_table(task1_feature_records(contract, set(folds.subject_id)),
                                                  profile="p2_matched_simple")
    expected = json.loads(a.feature_manifest.read_text(encoding="utf-8"))
    if manifest != expected or len(feature_columns(table)) != 88:
        raise RuntimeError("Historical 88-D feature definition changed")
    merged = table.merge(folds[["subject_id", "outer_fold"]], on="subject_id", how="inner", validate="many_to_one")
    ledger = pd.read_csv(a.b0_ledger)
    keys = ["subject_id", "outer_fold", "channel_name"]
    if len(merged) != 7635 or len(ledger) != 7635 or merged[keys].duplicated().any():
        raise RuntimeError("Historical channel count or key uniqueness changed")
    aligned = merged[keys + ["label_nez"]].merge(ledger[keys + ["label_nez"]], on=keys,
                                                  how="inner", validate="one_to_one", suffixes=("_table", "_b0"))
    if len(aligned) != 7635 or not (aligned.label_nez_table == aligned.label_nez_b0).all():
        raise RuntimeError("B0 channel keys or labels do not align")
    replay = json.loads(a.replay_audit.read_text(encoding="utf-8"))
    if replay.get("status") != "PASS" or len(replay.get("folds", [])) != 5:
        raise RuntimeError("Historical five-fold B0 checkpoint replay not PASS")
    merged.to_pickle(destination)
    audit = {"status": "PASS", "patients": 80, "channels": 7635, "feature_dimension": 88,
             "feature_profile": "p2_matched_simple", "outer_folds": 5,
             "feature_cache_sha256": digest(a.feature_cache), "feature_manifest_sha256": digest(a.feature_manifest),
             "b0_ledger_sha256": digest(a.b0_ledger), "split_manifest_sha256": digest(a.split_manifest),
             "private_feature_table_sha256": digest(destination),
             "b0_replay_max_abs_score_error": max(float(f["max_score_difference"]) for f in replay["folds"]),
             "split_role_counts": {str(f): {role: int(splits[(splits.outer_fold == f) & (splits.partition == role)].subject_id.nunique())
                                            for role in ("fit", "validation", "test")} for f in range(1, 6)}}
    (a.output_dir / "B0_REPLAY_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print("PREPARE_PASS " + json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
