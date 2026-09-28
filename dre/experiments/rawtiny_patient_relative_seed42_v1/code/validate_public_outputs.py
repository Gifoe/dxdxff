"""Read-only integrity checks for the compact public RawTiny Stage B bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_SHA = "028a951709f095459c39986f3823110fa9826078f24290d6afe9017c2ed621d2"
REQUIRED = (
    "FINAL_REPORT.md", "PROTOCOL_LOCK.json", "SOURCE_REPRODUCTION.json",
    "RAW_DATA_COVERAGE_AUDIT.json", "RAW_PREPROCESSING_AUDIT.json",
    "CHANNEL_ALIGNMENT_AUDIT.json", "MODEL_PARAMETER_COUNTS.csv",
    "FIT_HYPERPARAM_SELECTION.csv", "PRIMARY_MODEL_MATRIX.csv",
    "PATIENT_CLUSTER_BOOTSTRAP.csv", "FOLD_CONSISTENCY.csv",
    "CENTER_STRATIFIED_METRICS.csv", "REPRESENTATION_SEPARABILITY.csv",
    "PATIENT_RELATIVE_INTERVENTION.csv", "HYBRID_RAW_UTILIZATION.csv",
    "RAWTINY_GATES.json", "IMPLEMENTATION_AUDIT.md", "LABEL_USAGE_AUDIT.json",
    "SCORE_FREEZE_AUDIT.json",
)


def rows(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    assert data, name
    assert not {"subject_id", "patient_id", "sid", "channel_id", "label"}.intersection(data[0]), name
    return data


def document(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    assert all((ROOT / file).is_file() for file in REQUIRED)
    assert hashlib.sha256((ROOT / "PROTOCOL_LOCK.json").read_bytes()).hexdigest() == LOCK_SHA
    freeze = document("SCORE_FREEZE_AUDIT.json")
    assert freeze["pass"] and freeze["n_score_files"] == 450 and freeze["lock_sha256"] == LOCK_SHA
    coverage = document("RAW_DATA_COVERAGE_AUDIT.json")
    assert coverage["a1_raw_covered_patients"] == coverage["a1_manifest_patients"] == 80
    assert coverage["matched_runs"] == 256 and coverage["missing_runs"] == 0
    matrix = rows("PRIMARY_MODEL_MATRIX.csv")
    assert len(matrix) == 4
    assert all((int(r["n_cells"]), int(r["n_unique_patient_ids"]), int(r["n_repetitions"])) ==
               (65, 47, 1300) for r in matrix)
    indexed = {r["model"]: r for r in matrix}
    assert abs(float(indexed["M0_A1"]["ap"]) - 0.5767434626151353) < 1e-12
    assert len(rows("PATIENT_CLUSTER_BOOTSTRAP.csv")) == 5
    assert len(rows("FOLD_CONSISTENCY.csv")) == 20
    assert len(rows("REPRESENTATION_SEPARABILITY.csv")) == 8
    interventions = rows("PATIENT_RELATIVE_INTERVENTION.csv")
    assert len(interventions) == 4
    for row in interventions:
        assert abs(float(row["full_ap"]) - float(indexed[row["model"]]["ap"])) < 1e-12
        assert int(row["n_target_cells"]) == 65
    utilization = rows("HYBRID_RAW_UTILIZATION.csv")
    assert len(utilization) == 1
    assert abs(float(utilization[0]["full_ap"]) - float(indexed["M3_HYBRID_PR"]["ap"])) < 1e-12
    assert not document("RAWTINY_GATES.json")["outer_test_accessed"]
    assert not document("LABEL_USAGE_AUDIT.json")["outer_test_predictions_or_metrics_accessed"]
    print("RAWTINY_PUBLIC_VALIDATION_PASS")


if __name__ == "__main__":
    main()
