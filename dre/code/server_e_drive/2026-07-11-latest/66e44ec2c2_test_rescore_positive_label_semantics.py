import json

import pandas as pd

from scripts.rescore_a9v8_lcbo_outputs import rescore_run_dir


def test_rescore_fails_closed_without_resolved_args(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="cannot infer positive-label"):
        rescore_run_dir(tmp_path)


def test_rescore_maps_nez_positive_scores_to_ez_ranking(tmp_path) -> None:
    pd.DataFrame([
        {"fold_idx": 1, "subject_id": "p1", "center": "hup", "channel_name": "a", "true_ez": 1, "true_nez": 0, "score_eval": 0.1, "score_broad": 0.1, "score_core": 0.1},
        {"fold_idx": 1, "subject_id": "p1", "center": "hup", "channel_name": "b", "true_ez": 0, "true_nez": 1, "score_eval": 0.9, "score_broad": 0.9, "score_core": 0.9},
    ]).to_csv(tmp_path / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)
    (tmp_path / "resolved_run_args.json").write_text(json.dumps({"positive_label": "nez", "teacher_mode": "physiology_only"}), encoding="utf-8")
    summary, _, _ = rescore_run_dir(tmp_path)
    row = summary[summary["score_variant"] == "score_eval"].iloc[0]
    assert row["patient_macro_ez_f1"] == 1.0
