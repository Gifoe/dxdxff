from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from task1_confirmatory.component_ablation import (
    BCR_PROFILES, PRQ_PROFILES, completion_matches, evaluate_branch, profile_definition,
    validate_component_inputs, _repair_missing_patient_accuracy,
)
from scripts.run_prq_component_ablation import _run_with_native_resume


def _ledger(*, ez_probability: bool) -> pd.DataFrame:
    labels = [1, 0, 1, 0]
    score_nez = [.9, .2, .8, .1]
    data = {
        "subject_id": ["hup:A", "hup:A", "lzu:B", "lzu:B"],
        "center": ["hup", "hup", "lzu", "lzu"],
        "outer_fold": [1, 1, 1, 1],
        "channel_name": ["A1", "B1", "A1", "B1"],
        "label_nez": labels,
    }
    if ez_probability:
        data["score_ez"] = [1 - value for value in score_nez]
    else:
        data["score_nez"] = score_nez
    return pd.DataFrame(data)


def test_profile_contracts_are_frozen() -> None:
    assert profile_definition("prq", "P2_TEMPORAL_Q10_NO_PATIENT_RELATIVE")["p23_profile"] == "P2_TEMPORAL_Q10"
    assert PRQ_PROFILES["P1_TEMPORAL"]["temporal"] is True
    assert BCR_PROFILES["BCR_BOUNDARY_ONLY"]["boundary_loss"] is True
    assert BCR_PROFILES["BCR_BOUNDARY_ONLY"]["coverage_loss"] is False
    assert BCR_PROFILES["BCR_COVERAGE_ONLY"]["boundary_loss"] is False
    assert BCR_PROFILES["BCR_COVERAGE_ONLY"]["coverage_loss"] is True


def test_bcr_is_converted_to_nez_probability_and_shared_evaluator(tmp_path: Path) -> None:
    root = tmp_path / "bcr"
    for role in ("val", "test"):
        path = root / "fold_1" / f"{role}_channel_predictions_neuroez_v2_fold_1.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        _ledger(ez_probability=True).to_csv(path, index=False)
    result = evaluate_branch(branch="bcr", profile="BCR_BOUNDARY_COVERAGE", seed=42, run_root=root, output_root=root, folds=[1])
    ledger = pd.read_csv(root / "metrics" / "oof_channel_ledger.csv")
    assert ledger.y_true.tolist() == [1, 0, 1, 0]
    assert ledger.p_pos.tolist() == pytest.approx([.9, .2, .8, .1])
    assert result["overall"]["patient_macro_f1"] == pytest.approx(1.0)
    threshold = pd.read_csv(root / "metrics" / "fold_thresholds.csv")
    assert threshold.threshold_source.iloc[0] == "validation_only"


def test_bcr_saved_probability_column_is_canonicalized(tmp_path: Path) -> None:
    root = tmp_path / "bcr_saved"
    for role in ("val", "test"):
        path = root / f"{role}_channel_predictions_neuroez_v2_fold_1.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = _ledger(ez_probability=True).rename(
            columns={"outer_fold": "fold_idx", "label_nez": "true_nez", "score_ez": "score_ez_probability"}
        )
        frame.to_csv(path, index=False)
    result = evaluate_branch(branch="bcr", profile="BCR_BC_ONLY", seed=42, run_root=root, output_root=root, folds=[1])
    ledger = pd.read_csv(root / "metrics" / "oof_channel_ledger.csv")
    assert ledger.p_pos.tolist() == pytest.approx([.9, .2, .8, .1])
    assert result["overall"]["patient_macro_f1"] == pytest.approx(1.0)


def test_manifest_rejects_incomplete_outer_test_union(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects.csv"; split = tmp_path / "split.csv"; outer = tmp_path / "outer.csv"
    pd.DataFrame({"subject_id": ["hup:A", "lzu:B"]}).to_csv(subjects, index=False)
    pd.DataFrame({"subject_id": ["hup:A", "lzu:B", "hup:A"], "outer_fold": [1, 1, 1], "partition": ["fit", "validation", "test"]}).to_csv(split, index=False)
    pd.DataFrame({"subject_id": ["hup:A"], "outer_fold": [1], "partition": ["test"]}).to_csv(outer, index=False)
    with pytest.raises(RuntimeError, match="misses allowed patients"):
        validate_component_inputs(subjects=subjects, split_manifest=split, outer_manifest=outer, require_n_patients=2)


def test_resume_marker_requires_matching_provenance(tmp_path: Path) -> None:
    marker = tmp_path / "completion.json"
    marker.write_text(json.dumps({"status": "complete", "profile": "BCR_BC_ONLY", "seed": 42}), encoding="utf-8")
    assert completion_matches(marker, expected={"profile": "BCR_BC_ONLY", "seed": 42})
    with pytest.raises(RuntimeError, match="mismatched provenance"):
        completion_matches(marker, expected={"profile": "BCR_BOUNDARY_COVERAGE", "seed": 42})


def test_native_resume_retries_only_access_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    def fake_run(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise __import__("subprocess").CalledProcessError(3221225477, ["python"])
    monkeypatch.setattr("scripts.run_prq_component_ablation.subprocess.run", fake_run)
    monkeypatch.setattr("scripts.run_prq_component_ablation.time.sleep", lambda _seconds: None)
    _run_with_native_resume(["python"], retries=1)
    assert len(calls) == 2


def test_report_backfill_recovers_accuracy_from_saved_oof_masks(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    patients = pd.DataFrame({
        "subject_id": ["hup:A", "lzu:B"], "center": ["hup", "lzu"], "outer_fold": [1, 1],
        "experiment": ["bcr:BCR_BC_ONLY", "bcr:BCR_BC_ONLY"], "branch": ["bcr", "bcr"],
        "profile": ["BCR_BC_ONLY", "BCR_BC_ONLY"], "seed": [42, 42],
        "patient_macro_f1": [0.5, 0.5], "patient_ez_f1": [0.5, 0.5], "patient_nez_f1": [0.5, 0.5],
        "patient_balanced_accuracy": [0.5, 0.5], "patient_ez_auprc": [0.5, 0.5], "patient_ez_auroc": [0.5, 0.5],
        "patient_ez_mrr": [0.5, 0.5], "top1_is_ez_rate": [0.5, 0.5],
        "predicted_ez_count_mae": [0.0, 0.0], "predicted_ez_fraction_mae": [0.0, 0.0],
        "truek_patient_macro_f1": [0.5, 0.5],
    })
    patients.to_csv(metrics / "patient_metrics.csv", index=False)
    pd.DataFrame({
        "subject_id": ["hup:A", "hup:A", "lzu:B", "lzu:B"],
        "label_nez": [1, 0, 1, 0], "predicted_nez": [1, 1, 1, 0],
    }).to_csv(metrics / "oof_channel_ledger.csv", index=False)
    _repair_missing_patient_accuracy(metrics)
    repaired = pd.read_csv(metrics / "patient_metrics.csv")
    assert repaired.patient_accuracy.tolist() == pytest.approx([0.5, 1.0])
    assert "patient_accuracy" in pd.read_csv(metrics / "overall_metrics.csv").columns


def test_report_backfill_survives_legacy_summaries_without_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    patients = pd.DataFrame({
        "subject_id": ["hup:A", "lzu:B"], "center": ["hup", "lzu"], "outer_fold": [1, 1],
        "experiment": ["bcr:BCR_BC_ONLY", "bcr:BCR_BC_ONLY"], "analysis_status": ["test", "test"],
        "branch": ["bcr", "bcr"], "profile": ["BCR_BC_ONLY", "BCR_BC_ONLY"], "seed": [42, 42],
        "n_channels": [2, 2], "true_ez_fraction": [0.5, 0.5], "predicted_ez_fraction": [0.5, 0.5],
        "patient_macro_f1": [0.5, 0.5], "patient_ez_f1": [0.5, 0.5], "patient_nez_f1": [0.5, 0.5],
        "patient_accuracy": [0.5, 1.0], "patient_balanced_accuracy": [0.5, 1.0],
        "patient_ez_auprc": [0.5, 1.0], "patient_ez_auroc": [0.5, 1.0],
        "patient_ez_mrr": [0.5, 1.0], "top1_is_ez_rate": [0.0, 1.0],
        "predicted_ez_count_mae": [0.0, 0.0], "predicted_ez_fraction_mae": [0.0, 0.0],
        "truek_patient_macro_f1": [0.5, 1.0],
    })
    patients.to_csv(metrics / "patient_metrics.csv", index=False)

    from task1_confirmatory import component_ablation as module
    current_summaries = module.summaries

    def legacy_summaries(frame: pd.DataFrame):
        outputs = current_summaries(frame)
        return tuple(output.drop(columns=["patient_accuracy"], errors="ignore") for output in outputs)

    monkeypatch.setattr(module, "summaries", legacy_summaries)
    module._repair_missing_patient_accuracy(metrics)

    overall = pd.read_csv(metrics / "overall_metrics.csv")
    assert overall.patient_accuracy.iloc[0] == pytest.approx(0.75)
