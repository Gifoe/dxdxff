"""Validate the compact, protocol-stopped public bundle without private data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCK="43b7680413d360163eb9428bc03f0816c46e8a83fbc93465a10abd131aafbcb9"
REQUIRED=("FINAL_REPORT.md","PROTOCOL_LOCK.json","SOURCE_REPRODUCTION.json",
          "B0_IDENTITY_AUDIT.json","FULLPOOL_REFERENCE_REPRODUCTION.json",
          "TEACHER_CROSSFIT_AUDIT.json","FIT_OOF_TEACHER_METRICS.csv",
          "FIT_OOF_TEACHER_PATIENT_GAINS.csv","FIT_OOF_TEACHER_BOOTSTRAP.csv",
          "FIT_KD_HYPERPARAM_SELECTION.csv","STUDENT_VARIANT_MATRIX.csv",
          "CONTINUED_TRAINING_CONTROL.csv","PATIENT_CLUSTER_BOOTSTRAP.csv",
          "FOLD_CONSISTENCY.csv","TEACHER_STUDENT_PREFERENCE_AGREEMENT.csv",
          "CORRECTION_STRATIFIED_AUDIT.csv","TARGET_FAILURE_STRATIFIED_ANALYSIS.csv",
          "HEADROOM_TRANSFER.csv","DISTILLATION_GATES.json","IMPLEMENTATION_AUDIT.md",
          "LABEL_USAGE_AUDIT.json")


def rows(name):
    with (ROOT/name).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def main():
    if any(not (ROOT/name).is_file() for name in REQUIRED):raise RuntimeError("Required output absent")
    if hashlib.sha256((ROOT/"PROTOCOL_LOCK.json").read_bytes()).hexdigest()!=LOCK:
        raise RuntimeError("Protocol lock drift")
    audit=json.loads((ROOT/"TEACHER_CROSSFIT_AUDIT.json").read_text(encoding="utf-8"))
    gates=json.loads((ROOT/"DISTILLATION_GATES.json").read_text(encoding="utf-8"))
    usage=json.loads((ROOT/"LABEL_USAGE_AUDIT.json").read_text(encoding="utf-8"))
    if (audit["n_source_contexts"]!=17 or audit["n_fit_patient_contexts"]!=869 or
            audit["n_channel_crossfit_heads"]!=4345 or audit["PRIVILEGED_TEACHER_SIGNAL_VALID"] or
            audit["positive_outer_folds"]!=5 or audit["mean_delta_ap"]<=.03 or
            gates["terminal"]!="PRIVILEGED_TEACHER_SIGNAL_GATE_FAILED" or
            gates["student_training_started"] or gates["target_student_outcomes_accessed"] or
            usage["strict_target_label_sequencing"]):
        raise RuntimeError("Gate 0 provenance/status invalid")
    if len(rows("FIT_OOF_TEACHER_METRICS.csv"))!=23 or len(rows("FIT_OOF_TEACHER_PATIENT_GAINS.csv"))!=17:
        raise RuntimeError("Teacher metric coverage invalid")
    if len(rows("FIT_OOF_TEACHER_BOOTSTRAP.csv"))!=3 or len(rows("FOLD_CONSISTENCY.csv"))!=5:
        raise RuntimeError("FIT uncertainty/fold coverage invalid")
    if any(r["status"] not in ("HISTORICAL_REFERENCE_ONLY","NOT_RUN_GATE0") for r in rows("STUDENT_VARIANT_MATRIX.csv")):
        raise RuntimeError("Student rows cannot claim unrun outcomes")
    for name in ("FIT_KD_HYPERPARAM_SELECTION.csv","CONTINUED_TRAINING_CONTROL.csv",
                 "TEACHER_STUDENT_PREFERENCE_AGREEMENT.csv","CORRECTION_STRATIFIED_AUDIT.csv",
                 "TARGET_FAILURE_STRATIFIED_ANALYSIS.csv"):
        if rows(name)[0]["status"]!="NOT_RUN_TEACHER_GATE0_FAILED":
            raise RuntimeError("Unrun stage not marked")
    forbidden={".pt",".pth",".ckpt",".pkl",".npy",".npz",".log",".err"}
    if any(path.suffix.lower() in forbidden for path in ROOT.rglob("*") if path.is_file()):
        raise RuntimeError("Private/runtime payload in public experiment folder")
    print("PUBLIC_BUNDLE_VALIDATION_PASS")


if __name__=="__main__":main()
