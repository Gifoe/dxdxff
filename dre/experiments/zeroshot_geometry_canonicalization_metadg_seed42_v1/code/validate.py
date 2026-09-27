"""Read-only validation of compact public study outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCK_SHA="80e028f57a6484f55273555fbdb76135551bc509ccece930006db85611958e5e"
AMENDMENT_SHA="765107a265c48d139430c97d95da2aeb513dbadff6a639984ff2b04afa69e0bf"
CSV_COUNTS={
    "FIT_HYPERPARAM_SELECTION.csv":160,
    "ZEROSHOT_VARIANT_MATRIX.csv":7,
    "MATCHED_B8_REFERENCE.csv":2,
    "PATIENT_CLUSTER_BOOTSTRAP.csv":7,
    "FOLD_CONSISTENCY.csv":35,
    "PATIENT_DIRECTION_DISPERSION.csv":7,
    "SHARED_VS_PATIENT_SPECIFIC_HEADROOM.csv":7,
    "DIRECTION_REVERSAL_AUDIT.csv":7,
}
JSONS=("SOURCE_REPRODUCTION.json","B0_IDENTITY_AUDIT.json",
       "LABEL_USAGE_AUDIT.json","GEOMETRY_MECHANISM_AUDIT.json","ZEROSHOT_GATES.json")


def rows(name):
    with (ROOT/name).open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        fields=set(reader.fieldnames or ())
        if any("subject_id" in k.lower() or "channel_id" in k.lower() for k in fields):
            raise RuntimeError(f"Public patient/channel identity column: {name}")
        return list(reader)


def main():
    assert hashlib.sha256((ROOT/"PROTOCOL_LOCK.json").read_bytes()).hexdigest()==LOCK_SHA
    assert hashlib.sha256((ROOT/"DIAGNOSTIC_AMENDMENT.json").read_bytes()).hexdigest()==AMENDMENT_SHA
    assert (ROOT/"FINAL_REPORT.md").is_file() and (ROOT/"IMPLEMENTATION_AUDIT.md").is_file()
    for name,count in CSV_COUNTS.items():
        data=rows(name)
        assert len(data)==count,(name,len(data),count)
    source=json.loads((ROOT/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    assert source["checkpoints"]==150 and source["max_grid_error"]<=1e-6
    assert source["R4_max_logit_replay_error"]<=1e-6
    assert abs(source["exact_a1_matched_query_ap"]-.5767434626151353)<=1e-6
    assert json.loads((ROOT/"B0_IDENTITY_AUDIT.json").read_text(encoding="utf-8"))["pass"]
    audit=json.loads((ROOT/"LABEL_USAGE_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["target_cells"]==65 and audit["unique_patient_ids"]==47 and audit["deployment_budget"]==0
    assert audit["strict_target_label_sequencing"] is False
    assert audit["strict_no_outer_label_materialization"] is False
    assert audit["diagnostic_amendment_sha"]==AMENDMENT_SHA
    assert audit["outer_predictions_metrics_selection"] is False
    for name in JSONS:assert (ROOT/name).is_file(),name
    variants={r["variant"] for r in rows("ZEROSHOT_VARIANT_MATRIX.csv")}
    assert len(variants)==7 and "Z0_A1" in variants
    for path in ROOT.rglob("*"):
        if path.is_file() and (path.suffix.lower() in {".pt",".pkl",".npy",".npz",".log",".err"}
                               or "PRIVATE" in path.name.upper()):
            raise RuntimeError(f"Private artifact under public root: {path}")
    print("ZEROSHOT_PUBLIC_VALIDATION_PASS")


if __name__=="__main__":main()
