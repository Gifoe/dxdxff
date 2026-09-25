from pathlib import Path

import numpy as np
import torch

from data_factory import build_outer_splits
from neuroez_c.p23_seizure_tail import P23CrossSeizureTailEvidence, tail_reliability
from neuroez_c.p2_monotonic_shift_calibrator import apply_patient_shift, fit_patient_shift_calibrator
from neuroez_c.task1_cohort_contract import validate_task1_cohort
from neuroez_c.cane_path_cp_trainer import _selection_key


def _tail(logits: list[float]):
    module = P23CrossSeizureTailEvidence(1, robust_tail=True)
    module.scorer = torch.nn.Identity()
    embedding = torch.tensor(logits, dtype=torch.float32).reshape(1, len(logits), 1, 1)
    return module(embedding, torch.ones(1, len(logits), dtype=torch.bool), torch.ones(1, len(logits), 1, dtype=torch.bool))


def test_robust_tail_single_seizure_equals_mean_and_has_finite_gradients():
    embedding = torch.tensor([[[[1.5]]]], requires_grad=True)
    module = P23CrossSeizureTailEvidence(1, robust_tail=True)
    module.scorer = torch.nn.Identity()
    output = module(embedding, torch.tensor([[True]]), torch.tensor([[[True]]]))
    assert torch.allclose(output["seizure_nez_robust_tail_logit"], output["seizure_nez_logit_mean"])
    output["seizure_nez_robust_tail_logit"].sum().backward()
    assert torch.isfinite(embedding.grad).all()


def test_robust_tail_two_seizure_shrinks_and_is_permutation_invariant():
    left = _tail([0.0, 2.0])["seizure_nez_robust_tail_logit"].item()
    right = _tail([2.0, 0.0])["seizure_nez_robust_tail_logit"].item()
    mean = 1.0
    assert left < mean and np.isclose(left, right)


def test_tail_reliability_is_zero_for_one_seizure_and_bounded():
    value = tail_reliability(torch.tensor([[1, 2, 3]]), torch.tensor([[1.0, 0.25, 1.0]]))
    assert value[0, 0].item() == 0.0
    assert torch.all((value >= 0.0) & (value <= 1.0))
    assert value[0, 2].item() > value[0, 1].item()


def _record(subject: str, logits: tuple[float, float], labels: tuple[int, int]):
    return {
        "subject_id": subject, "center": "hup", "channel_mask": np.array([True, True]),
        "final_nez_logit": np.asarray(logits, dtype=np.float32), "labels_nez": np.asarray(labels, dtype=np.float32),
        "seizure_nez_robust_tail_probability": np.array([.5, .5], dtype=np.float32),
        "seizure_nez_tail_gap": np.array([.1, .1], dtype=np.float32),
        "seizure_agreement": np.array([.8, .8], dtype=np.float32), "valid_seizure_count": np.array([2., 2.], dtype=np.float32),
        "tail_reliability": np.array([.5, .5], dtype=np.float32),
    }


def test_shift_is_constant_within_patient_and_preserves_pairwise_order():
    records = [_record(f"p{i}", (1.0, -1.0), (1, 0)) for i in range(5)]
    calibrator, _ = fit_patient_shift_calibrator(records, seed=42, b_max=.12, lambda_grid=[.1])
    shifted, audit = apply_patient_shift(records, calibrator)
    for before, after in zip(records, shifted):
        delta = after["calibrated_nez_logit"] - before["final_nez_logit"]
        assert np.allclose(delta, delta[0])
        assert np.sign(np.diff(before["final_nez_logit"]))[0] == np.sign(np.diff(after["calibrated_nez_logit"]))[0]
    assert audit["ranking_pair_flip_fraction"] == 0.0


def test_primary90_and_sensitivity80_exact_contracts():
    root = Path(__file__).resolve().parents[1]
    ledger = root / "reference" / "all90_subjects.csv"
    exclusion = root / "configs" / "task1_sensitivity80_exclude_suspected_10.csv"
    subjects = [line.strip() for line in ledger.read_text(encoding="utf-8-sig").splitlines()[1:] if line.strip()]
    index = {subject: {} for subject in subjects}
    splits = build_outer_splits(index, n_splits=5, random_seed=42)
    primary = validate_task1_cohort(index, splits, cohort_name="primary90", all90_ledger_path=ledger, exclusion_path=None)
    sensitivity_index = {subject: {} for subject in subjects if subject not in {row.split(",")[0] for row in exclusion.read_text(encoding="utf-8-sig").splitlines()[1:]}}
    sensitivity_splits = []
    for split in splits:
        sensitivity_splits.append({"fold_idx": split["fold_idx"], "train_subjects": [s for s in split["train_subjects"] if s in sensitivity_index], "test_subjects": [s for s in split["test_subjects"] if s in sensitivity_index]})
    sensitivity = validate_task1_cohort(sensitivity_index, sensitivity_splits, cohort_name="sensitivity80", all90_ledger_path=ledger, exclusion_path=exclusion)
    assert primary["n_patients"] == 90
    assert sensitivity["n_patients"] == 80
    assert primary["outer_test_each_subject_once"] and sensitivity["outer_test_each_subject_once"]


def test_ez_aware_checkpoint_prefers_ez_on_macro_f1_tie():
    common = {
        "validation_patient_macro_f1": 0.60,
        "validation_patient_macro_auprc_ez": 0.55,
        "validation_patient_ranking_oracle_macro_f1": 0.70,
        "validation_worst_center_auprc_hmean": 0.50,
        "validation_balanced_accuracy": 0.60,
    }
    weak_ez = {
        **common,
        "validation_patient_nez_f1": 0.78,
        "validation_patient_ez_f1": 0.42,
        "validation_min_class_f1": 0.42,
    }
    strong_ez = {
        **common,
        "validation_patient_nez_f1": 0.66,
        "validation_patient_ez_f1": 0.54,
        "validation_min_class_f1": 0.54,
    }
    assert _selection_key(strong_ez, 4, "ez_aware_f1") > _selection_key(weak_ez, 3, "ez_aware_f1")
