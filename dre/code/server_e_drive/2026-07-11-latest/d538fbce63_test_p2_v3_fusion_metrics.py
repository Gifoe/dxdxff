import pandas as pd

from neuroez_c.p2_v3_fusion_reporting import aggregate_patients, evaluate_probability_predictions


def test_metrics_are_patient_equal_not_channel_weighted():
    frame = pd.DataFrame({
        "subject_id": ["a", "a", "b", "b", "b", "b"], "center": ["hup"] * 6, "outer_fold": [1] * 6,
        "channel_name": list("ABCDEF"), "label_nez": [0, 1, 0, 1, 1, 1], "score": [.1, .9, .9, .9, .9, .9],
    })
    patients = evaluate_probability_predictions(frame, score_nez_column="score", threshold=.5)
    summary = aggregate_patients(patients).iloc[0]
    assert summary.patient_macro_f1 == patients.patient_macro_f1.mean()


def test_truek_is_explicitly_diagnostic():
    frame = pd.DataFrame({"subject_id": ["a", "a"], "center": ["hup", "hup"], "outer_fold": [1, 1], "channel_name": ["A", "B"], "label_nez": [0, 1], "score": [.8, .2]})
    result = evaluate_probability_predictions(frame, score_nez_column="score", truek=True)
    assert result.true_count_used_for_prediction.iloc[0]
