from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from task1_baselines.token_data import Task1TokenDataset
from task1_baselines.training.token_runner import run_task1_token_oof


class TinyEncoder(torch.nn.Module):
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(8, 4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layer(values.squeeze(1))


def _tokens() -> tuple[Task1TokenDataset, pd.DataFrame]:
    rng = np.random.default_rng(4)
    values = []
    rows = []
    folds = []
    for patient in range(10):
        subject = f"p{patient:02d}"
        folds.append({"subject_id": subject, "center": "c", "outer_fold": patient % 5 + 1})
        for channel, label in (("A1", 0), ("A2", 1)):
            for window in range(2):
                values.append(rng.normal(label * 2, 0.1, 8))
                rows.append({"subject_id": subject, "center": "c", "seizure_id": "s1", "channel_name": channel, "window_id": window, "label_nez": label})
    return Task1TokenDataset(np.asarray(values, dtype=np.float32), pd.DataFrame(rows), ()), pd.DataFrame(folds)


def test_token_runner_produces_complete_channel_oof_without_patient_leakage() -> None:
    tokens, folds = _tokens()
    result = run_task1_token_oof(
        tokens, folds, model_name="tiny", seed=42, encoder_factory=TinyEncoder,
        frozen_backbone=False, inner_folds=2, max_epochs=1, batch_size=16, device="cpu",
    )
    assert len(result.oof) == 20
    assert not result.oof.duplicated(["subject_id", "channel_name"]).any()
    assert set(result.oof["outer_fold"]) == {1, 2, 3, 4, 5}
    assert set(result.oof["threshold_source"]) == {"inner_oof_patient_macro_f1"}
    for fold, subjects in result.fit_subjects.items():
        held_out = set(folds.loc[folds["outer_fold"] == fold, "subject_id"])
        assert not held_out & set(subjects)
