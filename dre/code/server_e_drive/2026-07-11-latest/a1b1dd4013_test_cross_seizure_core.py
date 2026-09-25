from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import inspect
from pathlib import Path

from scripts.task1_cross_seizure.core import (
    FORMAL_THRESHOLD_SOURCE,
    LOCKED_BCR_WEIGHT,
    LOCKED_PRQ_WEIGHT,
    aggregate_metrics,
    build_subset_manifest,
    compare_all_seizure,
    discover_threshold_file,
    load_fit_subjects_by_seed,
    make_prediction_rows,
    paired_bootstrap,
    patient_metrics,
    patient_seizure_map,
    read_thresholds,
    sample_seizure_subset,
    validate_prediction_frame,
)
from scripts.task1_cross_seizure.audit_cross_seizure_inputs import _require_config


def _folds():
    return {1: {"p1"}, 2: {"p2"}, 3: {"p3"}, 4: {"p4"}, 5: {"p5"}}


def _prediction():
    prq = {"subject_id": "p1", "center": "hup", "channel_name": ["A 1", "A2"], "label_nez": np.array([1, 0]), "score_nez": np.array([.8, .2])}
    bcr = {"subject_id": "p1", "center": "hup", "channel_name": ["A 1", "A2"], "label_nez": np.array([1, 0]), "score_nez": np.array([.7, .3])}
    return make_prediction_rows(prq, bcr, seed=42, fold=1, mode="all", repeat_id=0, selected_ids=["r1", "r2"], total=2, thresholds={"prq": .5, "bcr": .5, "cdel": .5})


def test_fixed_80_patient_cohort_manifest_shape():
    values = {f"p{i}": [f"r{i}", f"r{i}b"] for i in range(80)}
    folds = {index: {f"p{i}" for i in range((index - 1) * 16, index * 16)} for index in range(1, 6)}
    manifest = build_subset_manifest(values, folds, [42], 10)
    assert manifest.subject_id.nunique() == 80


def test_result_table_name_is_not_a_roleless_fold_manifest(tmp_path):
    from scripts.task1_cross_seizure.core import load_subjects_and_folds
    rows = []
    for fold in range(1, 6):
        rows.extend({"subject_id": f"p{index}", "outer_fold": fold} for index in range((fold - 1) * 16, fold * 16))
    pd.DataFrame(rows).to_csv(tmp_path / "formal_by_patient.csv", index=False)
    with pytest.raises(FileNotFoundError):
        load_subjects_and_folds(tmp_path)


def test_fixed_partition_manifest_wins_over_oof_channel_ledgers(tmp_path):
    from scripts.task1_cross_seizure.core import load_subjects_and_folds
    rows = []
    for fold in range(1, 6):
        rows.extend(
            {"subject_id": f"p{index}", "outer_fold": fold, "partition": "test"}
            for index in range((fold - 1) * 16, fold * 16)
        )
    audit = tmp_path / "audit"
    output = tmp_path / "final_pooled" / "seed_42" / "bcr_boundary_coverage"
    audit.mkdir(parents=True)
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(audit / "fixed_partition_manifest.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "oof_channel_ledger.csv", index=False)
    subjects, folds, selected = load_subjects_and_folds(tmp_path)
    assert len(subjects) == 80
    assert set(folds) == {1, 2, 3, 4, 5}
    assert selected.name == "fixed_partition_manifest.csv"


def test_confirmatory_split_role_filters_fit_validation_and_test(tmp_path):
    from scripts.task1_cross_seizure.core import load_subjects_and_folds_by_seed
    rows = []
    subjects = [f"p{index}" for index in range(80)]
    for fold in range(1, 6):
        test = set(subjects[(fold - 1) * 16:fold * 16])
        validation = set(subjects[((fold % 5) * 16):((fold % 5) + 1) * 16])
        for subject in subjects:
            role = "test" if subject in test else "validation" if subject in validation else "fit"
            rows.append(
                {
                    "subject_id": subject,
                    "outer_fold": fold,
                    "split_role": role,
                }
            )
    audit = tmp_path / "audit"
    audit.mkdir()
    pd.DataFrame(rows).to_csv(audit / "fixed_partition_manifest.csv", index=False)
    cohort, fold_maps, _ = load_subjects_and_folds_by_seed(
        tmp_path,
        [42, 52, 62],
    )
    assert len(cohort) == 80
    assert fold_maps[42] == fold_maps[52] == fold_maps[62]
    assert all(len(values) == 16 for values in fold_maps[42].values())
    fit_maps = load_fit_subjects_by_seed(
        audit / "fixed_partition_manifest.csv",
        [42, 52, 62],
    )
    assert fit_maps[42] == fit_maps[52] == fit_maps[62]
    assert all(len(values) == 48 for values in fit_maps[42].values())


def test_fixed_partition_manifest_deduplicates_identical_training_seeds(tmp_path):
    from scripts.task1_cross_seizure.core import load_subjects_and_folds
    rows = []
    for training_seed in (42, 52, 62):
        for fold in range(1, 6):
            rows.extend(
                {
                    "training_seed": training_seed,
                    "subject_id": f"p{index}",
                    "outer_fold": fold,
                    "partition": "test",
                }
                for index in range((fold - 1) * 16, fold * 16)
            )
    audit = tmp_path / "audit"
    audit.mkdir()
    pd.DataFrame(rows).to_csv(audit / "fixed_partition_manifest.csv", index=False)
    subjects, folds, _ = load_subjects_and_folds(tmp_path)
    assert len(subjects) == 80
    assert sum(len(subjects_in_fold) for subjects_in_fold in folds.values()) == 80


def test_fixed_partition_manifest_supports_seed_specific_outer_folds(tmp_path):
    from scripts.task1_cross_seizure.core import load_subjects_and_folds_by_seed
    rows = []
    for training_seed, offset in ((42, 0), (52, 1), (62, 2)):
        for index in range(80):
            rows.append(
                {
                    "training_seed": training_seed,
                    "subject_id": f"p{index}",
                    "outer_fold": ((index + offset) % 5) + 1,
                    "partition": "test",
                }
            )
    audit = tmp_path / "audit"
    audit.mkdir()
    pd.DataFrame(rows).to_csv(audit / "fixed_partition_manifest.csv", index=False)
    subjects, fold_maps, _ = load_subjects_and_folds_by_seed(tmp_path, [42, 52, 62])
    assert len(subjects) == 80
    assert fold_maps[42] != fold_maps[52]
    assert all(sum(map(len, fold_maps[seed].values())) == 80 for seed in (42, 52, 62))


def test_fixed_partition_manifest_rejects_cross_seed_fold_conflict(tmp_path):
    from scripts.task1_cross_seizure.core import load_subjects_and_folds
    rows = []
    for fold in range(1, 6):
        rows.extend(
            {"subject_id": f"p{index}", "outer_fold": fold, "partition": "test"}
            for index in range((fold - 1) * 16, fold * 16)
        )
    rows.append({"subject_id": "p0", "outer_fold": 2, "partition": "test"})
    audit = tmp_path / "audit"
    audit.mkdir()
    pd.DataFrame(rows).to_csv(audit / "fixed_partition_manifest.csv", index=False)
    with pytest.raises(ValueError, match="multiple outer test folds"):
        load_subjects_and_folds(tmp_path)


def test_seizure_subset_determinism():
    assert sample_seizure_subset("p", ["b", "a", "c"], mode="two", repeat_id=1, subsample_seed=4) == sample_seizure_subset("p", ["c", "a", "b"], mode="two", repeat_id=1, subsample_seed=4)


def test_no_label_based_sampling_interface():
    assert all("label" not in name and "score" not in name for name in inspect.signature(sample_seizure_subset).parameters)


def test_all_mode_uses_all_seizures():
    assert sample_seizure_subset("p", ["r2", "r1"], mode="all", repeat_id=0, subsample_seed=0) == ["r1", "r2"]


def test_two_mode_requires_two_seizures():
    assert sample_seizure_subset("p", ["r1"], mode="two", repeat_id=0, subsample_seed=0) == []


def test_same_subset_for_prq_bcr_is_manifest_level():
    manifest = build_subset_manifest({"p1": ["r1", "r2"]}, {1: {"p1"}}, [42], 1)
    assert manifest[manifest.seizure_mode.eq("one")].selected_seizure_ids.nunique() == 1


def test_cdel_fixed_weight():
    assert LOCKED_PRQ_WEIGHT == .8 and LOCKED_BCR_WEIGHT == .2


def test_label_semantics_and_threshold_source():
    frame = _prediction(); validate_prediction_frame(frame)
    assert frame.label_ez.tolist() == [0, 1]
    assert frame.threshold_source.eq(FORMAL_THRESHOLD_SOURCE).all()


def test_ez_ranking_direction():
    metrics = patient_metrics(_prediction(), "CDEL")
    assert metrics.iloc[0].patient_ndcg_ez == 1.0


def test_patient_equal_metrics_not_pooled():
    frame = pd.concat([_prediction(), _prediction().assign(subject_id="p2", outer_fold=2)], ignore_index=True)
    metrics = patient_metrics(frame, "PRQ-Net")
    assert len(metrics) == 2


def test_prediction_alignment_rejects_channel_mismatch():
    prq = {"subject_id": "p1", "center": "hup", "channel_name": ["A"], "label_nez": np.array([1]), "score_nez": np.array([.7])}
    bcr = {"subject_id": "p1", "center": "hup", "channel_name": ["B"], "label_nez": np.array([1]), "score_nez": np.array([.7])}
    with pytest.raises(ValueError):
        make_prediction_rows(prq, bcr, seed=42, fold=1, mode="all", repeat_id=0, selected_ids=["r"], total=1, thresholds={"prq": .5, "bcr": .5, "cdel": .5})


def test_patient_seizure_map_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        patient_seizure_map([{"subject_id": "p", "run_id": "r"}, {"subject_id": "p", "run_id": "r"}], {"p"})


def test_reproduction_passes_exact_match():
    frame = _prediction()
    reference = frame.rename(columns={"prq_score_nez": "score_nez", "prq_threshold": "threshold"})[["training_seed", "outer_fold", "subject_id", "center", "channel_name", "label_nez", "threshold", "score_nez"]]
    result = compare_all_seizure(frame, reference, model="PRQ-Net")
    assert result["status"] == "PASS"


def test_reproduction_fails_score_drift():
    frame = _prediction()
    reference = frame.rename(columns={"prq_score_nez": "score_nez", "prq_threshold": "threshold"})[["training_seed", "outer_fold", "subject_id", "center", "channel_name", "label_nez", "threshold", "score_nez"]]
    reference.loc[0, "score_nez"] = .1
    assert compare_all_seizure(frame, reference, model="PRQ-Net")["status"] == "FAIL"


def test_reproduction_accepts_bounded_gpu_roundoff_without_prediction_change():
    frame = _prediction()
    reference = frame.rename(columns={"prq_score_nez": "score_nez", "prq_threshold": "threshold"})[["training_seed", "outer_fold", "subject_id", "center", "channel_name", "label_nez", "threshold", "score_nez"]]
    reference["score_nez"] = reference["score_nez"] + np.asarray([1.5e-5, 0.0])
    result = compare_all_seizure(frame, reference, model="PRQ-Net")
    assert result["status"] == "PASS"
    assert result["prediction_mismatch_count"] == 0


def test_reproduction_rejects_gpu_error_above_maximum_tolerance():
    frame = _prediction()
    reference = frame.rename(columns={"prq_score_nez": "score_nez", "prq_threshold": "threshold"})[["training_seed", "outer_fold", "subject_id", "center", "channel_name", "label_nez", "threshold", "score_nez"]]
    reference.loc[0, "score_nez"] += 2.1e-4
    assert compare_all_seizure(frame, reference, model="PRQ-Net")["status"] == "FAIL"


def test_paired_bootstrap_determinism():
    frame = _prediction()
    patient = pd.concat([patient_metrics(frame.assign(seizure_mode=mode), model) for mode in ("one", "two", "all") for model in ("PRQ-Net", "BCR-Net", "CDEL")], ignore_index=True)
    first = paired_bootstrap(patient, repeats=10, seed=42)
    second = paired_bootstrap(patient, repeats=10, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_primary_cohort_uses_two_seizure_eligibility():
    frame = pd.concat([patient_metrics(_prediction().assign(seizure_mode=mode), "PRQ-Net") for mode in ("one", "two", "all")])
    primary, _ = aggregate_metrics(frame, primary_subjects={"p1"})
    assert primary.cohort.eq("primary_matched").all()


def test_formal_prq_identity_does_not_depend_on_config_name():
    config = {
        "config_name": "",
        "positive_label": "nez",
        "use_p23_trn_nez": True,
        "p23_profile": "P2_TEMPORAL_Q10",
        "p23_direct_outer_only": True,
        "p23_regression_protocol": True,
        "p23_use_p2_loss": True,
        "use_cane_path_cp_nez": False,
        "use_patient_relative_z": True,
    }
    _require_config(
        config,
        {key: value for key, value in config.items() if key != "config_name"},
        Path("best_model.pt"),
    )


def test_formal_prq_identity_rejects_wrong_profile():
    with pytest.raises(ValueError, match="p23_profile"):
        _require_config(
            {"p23_profile": "P0_CURRENT_P2"},
            {"p23_profile": "P2_TEMPORAL_Q10"},
            Path("best_model.pt"),
        )


def test_prq_package_model_uses_p23_ranker_module():
    from P23_TRN_NEZ_80.neuroez_c.model import PatientChannelClassifier

    assert PatientChannelClassifier.__module__ == (
        "P23_TRN_NEZ_80.patient_channel_ranker"
    )


def test_selected_thresholds_are_filtered_by_model(tmp_path):
    root = tmp_path / "seed_42" / "p2_q10"
    table = root.parent / "thresholds" / "selected_thresholds.csv"
    table.parent.mkdir(parents=True)
    pd.DataFrame([
        {"outer_fold": fold, "model": model, "threshold": value}
        for fold in range(1, 6)
        for model, value in (("PRQ-Net", 0.4), ("BCR-Net", 0.5), ("CDEL", 0.45))
    ]).to_csv(table, index=False)
    selected = discover_threshold_file(root, 42, model="PRQ-Net")
    assert selected == table
    assert read_thresholds(selected, model="PRQ-Net") == {
        fold: 0.4 for fold in range(1, 6)
    }


def test_threshold_reader_rejects_duplicate_model_fold(tmp_path):
    path = tmp_path / "fold_thresholds.csv"
    pd.DataFrame([
        {"outer_fold": 1, "model": "PRQ-Net", "threshold": 0.4},
        {"outer_fold": 1, "model": "PRQ-Net", "threshold": 0.5},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate folds"):
        read_thresholds(path, model="PRQ-Net")
