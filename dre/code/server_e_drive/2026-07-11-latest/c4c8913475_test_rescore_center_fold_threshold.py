import pandas as pd

from scripts.rescore_a9v8_lcbo_nez_outputs import _metrics


def test_rescore_metrics_accepts_fold_specific_thresholds() -> None:
    frame = pd.DataFrame([
        {"subject_id": "p1", "true_nez": 1}, {"subject_id": "p1", "true_nez": 0},
        {"subject_id": "p2", "true_nez": 1}, {"subject_id": "p2", "true_nez": 0},
    ])
    scores = [0.7, 0.3, 0.7, 0.3]
    result = _metrics(frame, scores, [0.5, 0.5, 0.8, 0.8])
    assert result["threshold"] == 0.65
    assert result["patient_macro_f1"] < 1.0

