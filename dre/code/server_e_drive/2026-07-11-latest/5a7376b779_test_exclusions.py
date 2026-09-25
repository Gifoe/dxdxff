import pandas as pd

from neuroez_c.task2.exclusions import apply_exclusions, build_exclusion_audit, exclusion_match


def _record(run_id):
    return {"subject_id": "lzu:tuoyongxiang", "run_id": run_id, "sample": {"sample_id": f"sample-{run_id}"}}


def test_declared_sz2_exclusion_matches_full_run_identifier_and_retains_patient():
    manifest = pd.DataFrame([{
        "patient_key": "lzu:tuoyongxiang", "seizure_id": "SZ2", "sample_id": "*",
        "action": "exclude_seizure", "reason": "ambiguous_duplicate_onset_317_vs_263",
    }])
    assert exclusion_match(_record("sub-x_run-SZ2_task-ictal"), manifest.iloc[0])
    feature = [_record("sub-x_run-SZ1"), _record("sub-x_run-SZ2_task-ictal"), _record("sub-x_run-SZ2_task-ictal")]
    raw = list(feature)
    kept, removed = apply_exclusions(feature, manifest)
    audit = build_exclusion_audit(manifest, feature, raw).iloc[0]
    assert len(kept) == 1 and removed[0] == 2
    assert audit["feature_records_removed"] == 2 and audit["raw_records_removed"] == 2
    assert audit["remaining_seizures"] == 1 and bool(audit["patient_retained"])
