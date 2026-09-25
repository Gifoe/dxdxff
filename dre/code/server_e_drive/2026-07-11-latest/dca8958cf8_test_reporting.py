from __future__ import annotations

import pandas as pd

from task1_baselines.reporting import frame_to_markdown, summarize_task1


def test_markdown_report_has_no_optional_tabulate_dependency() -> None:
    text = frame_to_markdown(pd.DataFrame([{"model": "rbf", "score": 0.5}]))
    assert "| model | score |" in text
    assert "| rbf | 0.5 |" in text


def test_task1_reporting_writes_patient_bootstrap_vs_v3(tmp_path) -> None:
    rows = []
    reference = []
    for subject_index in range(6):
        for channel_index in range(2):
            label = channel_index
            rows.append({
                "model": "rbf_svm", "seed": 42, "subject_id": f"p{subject_index}", "center": "c1",
                "outer_fold": subject_index % 2 + 1, "channel_name": f"A{channel_index}", "label_nez": label,
                "score_nez_probability": 0.8 if label else 0.2, "predicted_nez": label,
            })
            reference.append({
                "subject_id": f"p{subject_index}", "channel_name": f"A{channel_index}",
                "true_ez": 1 - label, "predicted_ez": 1 - label,
            })
    summarize_task1(
        pd.DataFrame(rows), tmp_path, v3_reference=pd.DataFrame(reference), bootstrap_samples=50
    )
    result = pd.read_csv(tmp_path / "comparison" / "task1_bootstrap_vs_v3.csv")
    assert result.loc[0, "paired_status"] == "evaluated"
    assert result.loc[0, "paired_delta"] == 0.0
