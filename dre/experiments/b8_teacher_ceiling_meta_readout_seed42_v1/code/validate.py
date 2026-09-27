"""Read-only public aggregate and provenance validator; no private cache needed."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCK_SHA="6e9194da70cd850f29deceb1321e525ef08a18e61292f003221394b1210a20e2"
REQUIRED=("FINAL_REPORT.md","SOURCE_REPRODUCTION.json","IMPLEMENTATION_AUDIT.md",
    "LABEL_USAGE_AUDIT.json","FIT_B8_HYPERPARAM_SELECTION.csv","FULLPOOL_HYPERPARAM_SELECTION.csv",
    "TEACHER_VARIANT_MATRIX.csv","B8_TEACHER_COMPARISONS.csv","FULLPOOL_DIAGNOSTIC.csv",
    "FOLD_CONSISTENCY.csv","PATIENT_CLUSTER_BOOTSTRAP.csv","SUPPORT_CLASS_COMPOSITION.csv",
    "TEACHER_DIRECTION_ALIGNMENT.csv","TEACHER_CEILING_GATES.json","TEACHER_AP_VS_ORACLE_COSINE.png")


def rows(name):
    with (ROOT/name).open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if {x.lower() for x in reader.fieldnames}&{"sid","subject_id","patient_id","channel_id","repetition"}:
            raise RuntimeError(f"Patient/channel/repetition-level public header in {name}")
        return list(reader)


def main():
    assert hashlib.sha256((ROOT/"PROTOCOL_LOCK.json").read_bytes()).hexdigest()==LOCK_SHA
    for name in REQUIRED:
        if not (ROOT/name).is_file():raise RuntimeError(f"Missing aggregate artifact {name}")
    source=json.loads((ROOT/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    audit=json.loads((ROOT/"LABEL_USAGE_AUDIT.json").read_text(encoding="utf-8"))
    gates=json.loads((ROOT/"TEACHER_CEILING_GATES.json").read_text(encoding="utf-8"))
    assert source["pass"] and source["checkpoints"]==150 and source["max_grid_error"]<=1e-6
    assert source["R4_max_logit_replay_error"]<=1e-6 and source["current_b8_and_fullpool_max_ap_replay_error"]<=1e-8
    assert audit["target_cells"]==65 and audit["unique_patient_ids"]==47 and audit["repetitions"]==1300
    assert audit["target_query_labels_only_after_all_20_repetitions_all_variant_scores_frozen"]
    assert not audit["outer_predictions_metrics_selection"] and not audit["strict_no_outer_label_materialization"]
    matrix=rows("TEACHER_VARIANT_MATRIX.csv")
    assert len(matrix)==19 and len({r["variant"] for r in matrix})==19
    assert all(int(r["n_cells"])==65 and int(r["n_unique_patients"])==47 and int(r["n_repetitions"])==1300 for r in matrix)
    by={r["variant"]:r for r in matrix}
    assert abs(float(by["CURRENT_64D_B8"]["mean_ap"])-0.5996322681940147)<1e-12
    assert abs(float(by["CURRENT_64D_FULLPOOL"]["mean_ap"])-0.592851907369282)<1e-12
    assert len(rows("FOLD_CONSISTENCY.csv"))==95
    boot=rows("PATIENT_CLUSTER_BOOTSTRAP.csv")
    assert len(boot)==38 and all(int(r["n_patient_clusters"])==47 and int(r["resamples"])==10000 and int(r["seed"])==42 for r in boot)
    assert len(rows("FULLPOOL_DIAGNOSTIC.csv"))==8
    assert len(rows("B8_TEACHER_COMPARISONS.csv"))==19
    assert gates["best_b8_teacher"]==max((r for r in matrix if r["variant"].endswith("_B8") and r["variant"]!="ORACLE_BALANCED_B8"),key=lambda r:float(r["mean_ap"]))["variant"]
    for name in ROOT.iterdir():
        if name.is_file() and name.suffix.lower() in {".pt",".pth",".pkl",".npy",".npz"}:
            raise RuntimeError(f"Private/binary artifact in public root: {name.name}")
    print("PUBLIC_AGGREGATE_VALIDATION_PASS")


if __name__=="__main__":main()
