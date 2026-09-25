import pandas as pd
import torch

from neuroez_c.task2.tech_outcome_c1_aligned.data import AlignedCohort
from neuroez_c.task2.tech_outcome_c1_aligned.view_planner import (
    Capacity,
    DeterministicEvalViewPlanner,
    stable_hash,
    view_identity,
)


def _patient(n_seizures=5, n_channels=8, windows_per_phase=3):
    phase = torch.arange(3).repeat_interleave(windows_per_phase)
    time = torch.arange(len(phase), dtype=torch.float32)
    seizures = []
    for seizure_index in range(n_seizures):
        seizures.append(
            {
                "seizure_id": f"s{seizure_index}",
                "windows": torch.randn(n_channels, len(phase), 500),
                "window_mask": torch.ones(n_channels, len(phase), dtype=torch.bool),
                "phase_ids": phase,
                "relative_times_sec": time,
                "channel_names": [f"c{i}" for i in range(n_channels)],
            }
        )
    return {"patient_key": "p", "center": "x", "outcome_success": 1, "seizures": seizures}


def _cohort(tmp_path, patient, capacity):
    torch.save(patient, tmp_path / "p.pt")
    pd.DataFrame([{"patient_key": "p", "shard_path": "p.pt"}]).to_csv(
        tmp_path / "cache_manifest.csv", index=False
    )
    return AlignedCohort(tmp_path, seed=42, capacity=capacity)


def _assert_caps(view, capacity):
    assert len(view["seizures"]) <= capacity.max_seizures
    for seizure in view["seizures"]:
        assert len(seizure["channel_names"]) <= capacity.max_channels
        for channel in range(len(seizure["channel_names"])):
            for phase in range(3):
                count = (
                    seizure["window_mask"][channel]
                    & (seizure["phase_ids"][channel] == phase)
                ).sum()
                assert count <= capacity.max_windows_per_phase


def test_train_views_are_two_stable_distinct_capped_views(tmp_path):
    capacity = Capacity(2, 4, 2)
    cohort = _cohort(tmp_path, _patient(), capacity)
    first = cohort.train_views("p", outer_fold=1, epoch=3, n_views=2)
    second = cohort.train_views("p", outer_fold=1, epoch=3, n_views=2)
    assert len(first) == 2
    assert [view_identity(v) for v in first] == [view_identity(v) for v in second]
    assert view_identity(first[0]) != view_identity(first[1])
    for view in first:
        _assert_caps(view, capacity)
    assert stable_hash(42, 1, 3, "p", 0) != stable_hash(42, 1, 3, "p", 1)


def test_eight_eval_views_are_deterministic_capped_and_cover_data():
    patient = _patient()
    capacity = Capacity(2, 4, 2)
    planner = DeterministicEvalViewPlanner(42, capacity)
    first = planner.views(patient, outer_fold=1, n_views=8)
    second = planner.views(patient, outer_fold=1, n_views=8)
    assert len(first) == 8
    assert [view_identity(v) for v in first] == [view_identity(v) for v in second]
    for view in first:
        _assert_caps(view, capacity)
    coverage = planner.coverage(patient, outer_fold=1, n_views=8)
    assert coverage["seizure_coverage_fraction"] == 1.0
    assert coverage["channel_coverage_fraction"] == 1.0
    one_view_coverage = planner.coverage(patient, outer_fold=1, n_views=1)
    assert coverage["window_coverage_fraction"] > one_view_coverage["window_coverage_fraction"]
    assert coverage["window_coverage_fraction"] >= 0.80


def test_default_capacity_is_formal_protocol():
    assert Capacity() == Capacity(3, 96, 6)
