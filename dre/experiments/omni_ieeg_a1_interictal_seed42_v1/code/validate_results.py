"""Independent compact-output validation; no raw signal or checkpoint access."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit"
OUT = ROOT / "outputs"


def close(a, b):
    return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-12))


def main():
    cohort = pd.read_csv(AUDIT / "OMNI_COHORT_AUDIT.csv")
    predictions = pd.read_csv(OUT / "CHANNEL_PREDICTIONS.csv")
    patients = pd.read_csv(OUT / "PATIENT_LEVEL_METRICS.csv")
    metrics = json.loads((OUT / "PRIMARY_METRICS.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "TEST_SCORE_FREEZE_AUDIT.json").read_text(encoding="utf-8"))
    access = json.loads((OUT / "TEST_ACCESS_AUDIT.json").read_text(encoding="utf-8"))
    split = pd.read_csv(AUDIT / "TRAIN_VAL_SPLIT.csv")
    train_ids = set(cohort.loc[cohort["official_split"] == "train", "patient"])
    test_eligible = cohort.loc[(cohort["official_split"] == "test") &
                               (cohort["valid_channels"] > 0)]
    test_ids = set(test_eligible["patient"])
    assert len(train_ids) == 151 and len(test_ids) == 94
    assert len(split) == 139 and split["role"].value_counts().to_dict() == {
        "inner_train": 111, "inner_val": 28}
    assert set(split["patient"]).isdisjoint(test_ids)
    assert len(predictions) == 9215 and predictions["patient"].nunique() == 94
    assert set(predictions["patient"]) == test_ids
    assert not predictions.duplicated(["patient", "channel"]).any()
    assert predictions["soz"].isin([0, 1]).all()
    assert np.isfinite(predictions["score_ez"]).all()
    assert predictions["score_ez"].between(0, 1).all()
    assert int(predictions["soz"].sum()) == metrics["soz_channels"] == 764
    y = predictions["soz"].to_numpy()
    score = predictions["score_ez"].to_numpy()
    assert close(average_precision_score(y, score), metrics["pooled_ez_ap"])
    assert close(roc_auc_score(y, score), metrics["pooled_auroc"])
    assert close(balanced_accuracy_score(y, score >= 0.5), metrics["balanced_accuracy"])
    assert close(patients["ap"].mean(), metrics["patient_equal_mean_ap"])
    assert int((patients["soz_channels"] == 0).sum()) == metrics["patients_without_soz"]
    strata = pd.read_csv(OUT / "DATASET_STRATIFIED_METRICS.csv")
    assert int(strata["patients"].sum()) == metrics["patients"]
    assert int(strata["channels"].sum()) == metrics["channels"]
    assert int(strata["soz_channels"].sum()) == metrics["soz_channels"]
    assert freeze["model_frozen_before_official_test"] is True
    assert access["model_frozen_before_test"] is True
    assert access["test_used_for_tuning"] is False
    assert access["final_checkpoint_sha256"] == freeze["final_checkpoint_sha256"]
    result = {"pass": True, "supervised_train_patients": 139,
              "supervised_test_patients": 94, "supervised_test_edfs": len(test_eligible),
              "test_patient_channels": len(predictions), "test_soz_channels": int(y.sum()),
              "test_used_for_tuning": False}
    (OUT / "VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
