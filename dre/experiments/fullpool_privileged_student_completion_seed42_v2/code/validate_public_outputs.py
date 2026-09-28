"""Validate compact public v2 artifacts without private records or caches."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_SHA = "acfcbd600fbbd666ce5ccc8d480b50b6e131a7f7358da697860bb9d6db81fcb2"
REQUIRED = (
    "FINAL_REPORT.md", "PROTOCOL_LOCK.json", "V1_TEACHER_PROVENANCE.json",
    "TEACHER_HASH_REPLAY.json", "STUDENT_TRAINING_AUDIT.md",
    "FIT_KD_HYPERPARAM_SELECTION.csv", "D0_CONTINUED_CONTROL.csv",
    "STUDENT_VARIANT_MATRIX.csv", "PATIENT_CLUSTER_BOOTSTRAP.csv",
    "FOLD_CONSISTENCY.csv", "HEADROOM_TRANSFER.csv",
    "TEACHER_STUDENT_PREFERENCE_AGREEMENT.csv", "CORRECTION_STRATIFIED_AUDIT.csv",
    "TARGET_FAILURE_STRATIFIED_ANALYSIS.csv", "STUDENT_GATES.json",
    "LABEL_USAGE_AUDIT.json", "IMPLEMENTATION_AUDIT.md",
    "D4_FIT_ELIGIBILITY.json", "FIT_SELECTION_FREEZE_AUDIT.json",
    "TARGET_SCORE_FREEZE_AUDIT.json",
)


def read_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def read_csv(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, name
    forbidden = {"subject_id", "patient_id", "channel_id", "sid", "label", "y_true"}
    assert not forbidden.intersection(rows[0]), name
    return rows


def main() -> None:
    for name in REQUIRED:
        assert (ROOT / name).is_file(), name
    assert hashlib.sha256((ROOT / "PROTOCOL_LOCK.json").read_bytes()).hexdigest() == LOCK_SHA
    teacher = read_json("TEACHER_HASH_REPLAY.json")
    assert teacher["numerical_target_replay_pass"] and teacher["max_abs_teacher_score_replay_error"] == 0
    assert teacher["hash_exact_to_historical_per_file_manifest"] == "NOT_VERIFIABLE_NO_V1_MANIFEST"
    provenance = read_json("V1_TEACHER_PROVENANCE.json")
    assert provenance["TEACHER_AP_SIGNAL_ELIGIBLE_FOR_STUDENT"]
    assert provenance["v1_terminal_preserved"] == "PRIVILEGED_TEACHER_SIGNAL_GATE_FAILED"
    fit = read_json("FIT_SELECTION_FREEZE_AUDIT.json")
    score = read_json("TARGET_SCORE_FREEZE_AUDIT.json")
    assert fit["n_fold_models"] == score["n_fold_models"] == 30
    assert fit["all_models_fit_selected_before_target_outcomes"] and score["pass"]
    assert score["n_target_score_files"] == 30
    matrix = read_csv("STUDENT_VARIANT_MATRIX.csv")
    assert len(matrix) == 7
    assert all((int(r["n_cells"]), int(r["n_unique_patient_ids"]), int(r["n_repetitions"])) ==
               (65, 47, 1300) for r in matrix)
    baseline = next(r for r in matrix if r["model"] == "D0_ORIGINAL_A1")
    assert abs(float(baseline["ap"]) - 0.5767434626151353) < 1e-12
    assert len(read_csv("FIT_KD_HYPERPARAM_SELECTION.csv")) == 30
    assert len(read_csv("D0_CONTINUED_CONTROL.csv")) == 5
    assert len(read_csv("PATIENT_CLUSTER_BOOTSTRAP.csv")) == 11
    assert len(read_csv("FOLD_CONSISTENCY.csv")) == 35
    assert len(read_csv("HEADROOM_TRANSFER.csv")) == 6
    assert len(read_csv("TEACHER_STUDENT_PREFERENCE_AGREEMENT.csv")) == 6
    assert len(read_csv("CORRECTION_STRATIFIED_AUDIT.csv")) == 18
    assert len(read_csv("TARGET_FAILURE_STRATIFIED_ANALYSIS.csv")) == 18
    gates = read_json("STUDENT_GATES.json")
    assert not gates["PRIVILEGED_DISTILLATION_SUPPORTED"]
    assert gates["terminal"] == "STRONG_PRIVILEGED_SIGNAL_NOT_ZEROSHOT_DISTILLABLE"
    assert not gates["outer_test_accessed"] and not gates["student_target_used_for_training_or_selection"]
    label = read_json("LABEL_USAGE_AUDIT.json")
    assert not label["target_labels_before_student_model_and_score_freeze"]
    assert not label["target_labels_for_student_training_selection_or_calibration"]
    assert not label["outer_or_external_test_accessed"]
    print("PUBLIC_VALIDATION_PASS")


if __name__ == "__main__":
    main()
