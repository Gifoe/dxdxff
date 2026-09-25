from __future__ import annotations

import json
from pathlib import Path

from outcome_hifos.baselines.cohort_manifests import load_success_subject_manifest


def test_load_success_manifest_filters_csv_to_retained_subjects(tmp_path: Path) -> None:
    source = tmp_path / "success80.csv"
    source.write_text("subject_id,status\ns1,retained\ns2,excluded\ns3,retained\n", encoding="utf-8")
    assert load_success_subject_manifest(source) == {"s1", "s3"}


def test_load_success_manifest_reads_sensitivity_audit_json(tmp_path: Path) -> None:
    source = tmp_path / "sensitivity80.json"
    source.write_text(json.dumps({"retained_subjects": ["s1", "s2"]}), encoding="utf-8")
    assert load_success_subject_manifest(source) == {"s1", "s2"}
