from __future__ import annotations

import numpy as np
import pandas as pd

from task1_baselines.seegformer.multichannel_data import Task1MultichannelDataset, Task1MultichannelWindow
from task1_baselines.seegformer.runner import run_task1_seegformer_oof


def test_seegformer_runner_emits_grouped_channel_oof_and_checkpoints(tmp_path) -> None:
    rng = np.random.default_rng(7)
    items = []
    folds = []
    for subject_index in range(10):
        subject_id = f"p{subject_index:02d}"
        folds.append({"subject_id": subject_id, "outer_fold": subject_index % 5 + 1})
        for window_id in range(2):
            items.append(Task1MultichannelWindow(
                waveform=rng.normal(size=(2, 800)).astype(np.float32),
                label_nez=np.array([0, 1], dtype=np.float32),
                channel_names=("A1", "A2"), subject_id=subject_id, center="center_a",
                seizure_id=f"run_{subject_id}", window_id=window_id, modality="seeg", label_source="test",
            ))
    result = run_task1_seegformer_oof(
        Task1MultichannelDataset(items, []), pd.DataFrame(folds), model_name="seegformer", seed=42,
        inner_folds=2, max_epochs=1, batch_size=4, device="cpu", max_outer_folds=1,
        model_config={"embed_dim": 8, "num_heads": 2, "num_blocks": 1, "mlp_ratio": 1.0, "dropout": 0.0},
        checkpoint_root=tmp_path,
    )
    assert set(result.oof["outer_fold"]) == {1}
    assert result.oof["subject_id"].nunique() == 2
    assert {"score_nez_probability", "selected_threshold", "checkpoint_path"} <= set(result.oof)
    assert result.oof["checkpoint_path"].map(lambda path: bool(path)).all()
    assert (tmp_path / "outer_fold_1" / "seed_42" / "best_model.pt").exists()
    assert set(result.fit_subjects[1]).isdisjoint(set(result.oof["subject_id"]))
