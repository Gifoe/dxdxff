"""Synthetic, test-outcome-free checks of frozen threshold and metrics."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from audit_official_labels import official_pathology_label
from evaluate_v2 import bootstrap, classification_metrics, patient_metrics
from train_v2 import select_threshold


def main():
    assert official_pathology_label(1, 0, 1, 1) == 0  # official branch precedence
    assert official_pathology_label(0, 1, 1, 1) == 1
    assert official_pathology_label(0, 0, 0, 1) is None
    assert official_pathology_label(0, 1, 1, 0) is None
    frame = pd.DataFrame({
        "patient": ["p0"] * 3 + ["p1"] * 3,
        "dataset": ["synthetic"] * 6,
        "channel": ["a", "b", "c"] * 2,
        "pathology": [1, 0, 0, 0, 1, 0],
        "score_pathology": [0.85, 0.85, 0.10, 0.20, 0.70, 0.20],
    })
    with TemporaryDirectory() as tmp:
        frozen = select_threshold(frame, Path(tmp), "synthetic-checkpoint", "synthetic-protocol")
        scores = frame["score_pathology"].to_numpy()
        candidates = np.unique(scores)
        brute = max(candidates, key=lambda threshold: (
            f1_score(frame["pathology"], scores >= threshold,
                     labels=[0, 1], average="macro"),
            f1_score(frame["pathology"], scores >= threshold, pos_label=1),
            classification_metrics(frame, threshold)["balanced_accuracy"],
            -abs(threshold - 0.5), threshold))
        assert frozen["threshold"] == brute
        direct = classification_metrics(frame, frozen["threshold"])
        assert np.isclose(direct["macro_f1"], frozen["validation_macro_f1"])
    patient = pd.DataFrame([patient_metrics(part) for _, part in frame.groupby("patient")])
    ci = bootstrap(frame, patient, 0.5, draws=100)
    assert len(ci) == 8 and ci[["ci_lower", "ci_upper"]].notna().all().all()
    print("v2 synthetic label/threshold/bootstrap metrics PASS")


if __name__ == "__main__":
    main()
