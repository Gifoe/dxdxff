from __future__ import annotations

from pathlib import Path

import pandas as pd

from outcome_hifos.baselines.reporting import summarize_task2_baselines


def test_task2_reporting_writes_bootstrap_and_shortcut_outputs(tmp_path: Path) -> None:
    rows = []
    for model in ("majority", "elasticnet"):
        for index in range(20):
            outcome = index % 2
            probability = 0.5 if model == "majority" else (0.8 if outcome else 0.2)
            rows.append(
                {
                    "model": model,
                    "seed": 42,
                    "subject_id": f"p{index:02d}",
                    "center": f"c{index % 3}",
                    "outcome": outcome,
                    "outer_fold": index % 5 + 1,
                    "calibrated_probability": probability,
                    "predicted": int(probability >= 0.5),
                    "n_channels": 5 + index,
                    "n_seizures": 1 + index % 4,
                }
            )
    summarize_task2_baselines(pd.DataFrame(rows), tmp_path, pd.DataFrame(), bootstrap_samples=50)
    bootstrap = pd.read_csv(tmp_path / "comparison" / "task2_bootstrap.csv")
    shortcut = pd.read_csv(tmp_path / "shortcut_audit" / "task2_shortcut_metrics.csv")
    assert {"ci_lower", "ci_upper", "paired_vs_best_simple_delta"}.issubset(bootstrap)
    assert {"majority", "center_only", "channel_count_only", "seizure_count_only"}.issubset(set(shortcut["shortcut"]))
