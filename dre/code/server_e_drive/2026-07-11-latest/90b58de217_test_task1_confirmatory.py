from __future__ import annotations

import json
import argparse
import importlib.util
from pathlib import Path
import pickle

import pandas as pd
import pytest

from task1_confirmatory.evaluate import evaluate_pair
from task1_confirmatory.final_summary import _mean_std_row
from task1_confirmatory.orchestrator import _enforce_branch_reuse, _p2_patient_batch_size
from task1_confirmatory.protocol import build_loco_manifests, build_reference_partition, validate_partition
from task1_confirmatory.provenance import (
    P2_METHOD_ID, audit_argument_coverage, audit_effective_config_diff,
    write_branch_provenance,
)
from task1_confirmatory.audit import _read_cache_schema_in_subprocess
from task1_confirmatory.statistics import CORE_METRICS, _normalize_model_column, summarize_statistics
from neuroez_c.v3_qbc_protocol import read_outer_fold_manifest


def _load_p2_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "task1_confirmatory" / "launch_p2.py"
    spec = importlib.util.spec_from_file_location("test_confirmatory_launch_p2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_v3_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_v3_qbc.py"
    spec = importlib.util.spec_from_file_location("test_confirmatory_run_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_final_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "task1_confirmatory" / "run_final_multiseed_loco.py"
    spec = importlib.util.spec_from_file_location("test_final_confirmatory_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_fresh_gate_reuses_only_complete_seed42_branch_markers(tmp_path: Path) -> None:
    runner = _load_final_runner()
    seed_root = tmp_path / "pooled_cv" / "seed_42"
    assert not runner._fresh_seed42_branches_exist(tmp_path)
    for branch in ("p2_q10", "v3_qbc"):
        marker = seed_root / branch / "training_provenance.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}", encoding="utf-8")
    assert runner._fresh_seed42_branches_exist(tmp_path)


def _cohort(tmp_path: Path) -> tuple[Path, list[str]]:
    subjects = [f"{center}:S{index}" for center in ("hup", "lzu", "multicenter", "pediatric") for index in range(3)]
    path = tmp_path / "subjects.csv"
    pd.DataFrame({"subject_id": subjects, "center": [value.split(":")[0] for value in subjects]}).to_csv(path, index=False)
    return path, subjects


def _fold_manifest(tmp_path: Path, subjects: list[str]) -> Path:
    rows = []
    for index, subject in enumerate(subjects): rows.append({"subject_id": subject, "outer_fold": index % 5 + 1, "partition": "test"})
    path = tmp_path / "folds.csv"; pd.DataFrame(rows).to_csv(path, index=False); return path


def _ledger(subjects: list[str], fold: int, role: str) -> pd.DataFrame:
    rows=[]
    for patient_index, subject in enumerate(subjects):
        for channel, label in (("A1", 1), ("B1", 0)):
            score = .8 if label else .2
            rows.append({"subject_id":subject,"center":subject.split(":")[0],"outer_fold":fold,"channel_name":channel,"label_nez":label,"score_nez":score})
    return pd.DataFrame(rows)


def test_loco_target_never_enters_fit_or_validation(tmp_path: Path) -> None:
    cohort, subjects = _cohort(tmp_path)
    report = build_loco_manifests(cohort_ledger=cohort, output_dir=tmp_path / "loco", split_seed=20260721)
    assert report["status"] == "passed"
    lzu = pd.read_csv(tmp_path / "loco" / "loco_heldout_lzu_manifest.csv")
    assert set(lzu.loc[lzu.center.eq("lzu"), "split_role"]) == {"test"}
    assert lzu.subject_id.nunique() == len(subjects)


def test_v3_outer_manifest_recognizes_split_role_for_loco(tmp_path: Path) -> None:
    cohort, subjects = _cohort(tmp_path)
    build_loco_manifests(cohort_ledger=cohort, output_dir=tmp_path / "loco", split_seed=20260721)
    manifest = tmp_path / "loco" / "loco_heldout_hup_manifest.csv"
    folds = read_outer_fold_manifest(manifest)
    expected = sorted(subject for subject in subjects if subject.startswith("hup:"))
    assert folds == {1: expected}


def test_partition_rejects_overlap() -> None:
    frame = pd.DataFrame([
        {"subject_id":"hup:A","center":"hup","outer_fold":1,"split_role":"fit"},
        {"subject_id":"hup:A","center":"hup","outer_fold":1,"split_role":"validation"},
        {"subject_id":"lzu:B","center":"lzu","outer_fold":1,"split_role":"test"},
    ])
    assert validate_partition(frame, {"hup:A", "lzu:B"}, expected_folds={1})["status"] == "failed"


def test_reference_partition_freezes_validation_ledgers(tmp_path: Path) -> None:
    cohort, subjects = _cohort(tmp_path); folds = _fold_manifest(tmp_path, subjects); root = tmp_path / "p2"; root.mkdir()
    manifest = pd.read_csv(folds)
    for fold in range(1, 6):
        test = manifest.loc[manifest.outer_fold.eq(fold), "subject_id"].tolist()
        validation = [subject for subject in subjects if subject not in test][:2]
        folder = root / f"fold_{fold}"; folder.mkdir()
        _ledger(validation, fold, "validation").to_csv(folder / f"val_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
        _ledger(test, fold, "test").to_csv(folder / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
    audit = build_reference_partition(cohort_ledger=cohort, outer_fold_manifest=folds, p2_reference_root=root, output_path=tmp_path / "partition.csv", require_n_patients=len(subjects))
    assert audit["status"] == "passed"
    assert len(pd.read_csv(tmp_path / "partition.csv")) == len(subjects) * 5


def test_locked_fusion_evaluation_is_validation_only(tmp_path: Path) -> None:
    p2, v3 = tmp_path / "p2", tmp_path / "v3"; p2.mkdir(); v3.mkdir()
    for fold in range(1, 6):
        subjects = [f"hup:S{fold}a", f"lzu:S{fold}b"]
        for root, perturbation in ((p2, 0.0), (v3, .05)):
            val = _ledger(subjects, fold, "validation"); test = _ledger(subjects, fold, "test")
            val.score_nez = (val.score_nez + perturbation).clip(0, 1); test.score_nez = (test.score_nez + perturbation).clip(0, 1)
            val.to_csv(root / f"val_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
            test.to_csv(root / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
    result = evaluate_pair(p2_root=p2, v3_root=v3, folds=range(1, 6), output_dir=tmp_path / "evaluation")
    assert result["status"] == "passed"
    selected = pd.read_csv(tmp_path / "evaluation" / "thresholds" / "selected_thresholds.csv")
    assert set(selected.model) == {"PRQ-Net", "BCR-Net", "CDEL"}
    ledger = pd.read_csv(tmp_path / "evaluation" / "ledgers" / "oof_channel_predictions.csv")
    assert ledger.p2_weight.eq(.8).all() and ledger.v3_weight.eq(.2).all()
    true_k = pd.read_csv(tmp_path / "evaluation" / "metrics" / "patient_level_truek_diagnostic.csv")
    assert true_k.true_count_used_for_prediction.all()
    assert (~true_k.formal_prediction).all()
    assert true_k.predicted_ez_count.eq(true_k.true_ez_count).all()
    assert true_k.analysis_status.eq("DIAGNOSTIC_ONLY_NOT_DEPLOYABLE").all()


def test_p2_launcher_does_not_emit_false_after_store_true() -> None:
    launcher = _load_p2_launcher()
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow_partial_fixed_test_manifest", action="store_true")
    action = parser._actions[-1]
    command: list[str] = []
    launcher._append_action(command, action, False)
    assert command == []
    launcher._append_action(command, action, True)
    assert command == ["--allow_partial_fixed_test_manifest"]


def test_true_p2_identity_is_not_cane_path() -> None:
    assert P2_METHOD_ID == "P23_TRN_NEZ_80:P2_TEMPORAL_Q10"
    assert "CANE" not in P2_METHOD_ID


def test_model_related_unknown_base_arg_fails_strict_coverage() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    report = audit_argument_coverage({"epochs": 40, "hidden_model_switch": True}, parser, strict=False)
    assert report["passed"] is False
    assert report["unknown_model_args"] == ["hidden_model_switch"]
    with pytest.raises(RuntimeError, match="strict argument coverage"):
        audit_argument_coverage({"hidden_model_switch": True}, parser, strict=True)


def test_effective_config_rejects_forbidden_model_difference() -> None:
    report = audit_effective_config_diff(
        {"loss_mode": "p2", "random_seed": 42},
        {"loss_mode": "cane", "random_seed": 52}, experiment_type="pooled_cv",
    )
    assert report["passed"] is False
    assert "loss_mode" in report["forbidden_differences"]
    assert "random_seed" in report["allowed_differences"]


def test_unidentified_p2_ledger_does_not_claim_p2_temporal_q10() -> None:
    from neuroez_c.p2_v3_fusion_protocol import canonicalize_p2_fusion_ledger

    frame = pd.DataFrame({
        "subject_id": ["hup:A", "hup:A"], "outer_fold": [1, 1],
        "channel_name": ["A1", "B1"], "label_nez": [1, 0], "score_nez": [.8, .2],
    })
    canonical = canonicalize_p2_fusion_ledger(frame, split_role="test")
    assert canonical.source_model.eq("UNVERIFIED_P2").all()


def test_resume_contract_rejects_mismatched_effective_hash(tmp_path: Path) -> None:
    root = tmp_path / "p2_q10"; root.mkdir()
    (root / "partial.txt").write_text("partial", encoding="utf-8")
    contract = tmp_path / "audit" / "p2_launch_contract.json"; contract.parent.mkdir()
    contract.write_text(json.dumps({"effective_config_sha256": "old"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="launch contract mismatch"):
        _enforce_branch_reuse(
            root, contract_path=contract, expected_contract={"effective_config_sha256": "new"},
            allow_resume=True,
        )


def test_final_summary_uses_seed_level_sample_std() -> None:
    rows = {
        42: {metric: .60 for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "overall_auroc", "patient_macro_f1", "patient_ez_f1", "patient_nez_f1")},
        52: {metric: .70 for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "overall_auroc", "patient_macro_f1", "patient_ez_f1", "patient_nez_f1")},
        62: {metric: .80 for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "overall_auroc", "patient_macro_f1", "patient_ez_f1", "patient_nez_f1")},
    }
    row = _mean_std_row("Pooled", rows, [42, 52, 62])
    assert row["patient_macro_f1_mean"] == pytest.approx(.70)
    assert row["patient_macro_f1_std"] == pytest.approx(.10)


def test_branch_provenance_accepts_existing_v3_checkpoint_name(tmp_path: Path) -> None:
    root = tmp_path / "v3_qbc" / "fold_1"
    root.mkdir(parents=True)
    checkpoint = root / "best_b0_pruned_model.pth"
    checkpoint.write_bytes(b"v3-checkpoint")
    payload = write_branch_provenance(
        root=root.parent, method_id="BCR_NET", profile_id="BCR_BOUNDARY_COVERAGE",
        effective_config_sha256="effective", cohort_sha256="cohort",
        partition_sha256="partition", feature_cache_sha256="cache", seed=42,
        repo=tmp_path, folds=[1],
    )
    assert payload["checkpoints"][0]["path"].endswith("best_b0_pruned_model.pth")
    assert checkpoint.with_suffix(".pth.metadata.json").is_file()


def test_cache_schema_probe_runs_outside_training_process(tmp_path: Path) -> None:
    cache = tmp_path / "cache.pkl"
    with cache.open("wb") as stream:
        pickle.dump({"cache_version": "test", "feature_mode": "test", "run_records": [{}], "patient_index": {"hup:A": {}}, "window_feature_names": ["f0"]}, stream)
    schema = _read_cache_schema_in_subprocess(cache, output=tmp_path / "schema.json")
    assert schema["n_patients"] == 1
    assert schema["window_feature_names"] == ["f0"]


def test_large_cache_audit_runs_in_isolated_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_final_runner()
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        audit = tmp_path / "audit"
        audit.mkdir(parents=True, exist_ok=True)
        (audit / "cache_provenance_audit.json").write_text(
            json.dumps({"status": "passed", "caches": [{"cache_path": "large.pkl"}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    report = runner._audit_cache_in_isolated_process({"feature_cache": "large.pkl"}, tmp_path)
    assert report["status"] == "passed"
    assert calls and calls[0][0][1].endswith("audit_cache_provenance.py")


def test_partial_fixed_test_union_is_loco_only(tmp_path: Path) -> None:
    from data_factory import read_fixed_split_manifest

    subjects = {"hup:A": {}, "lzu:B": {}, "pediatric:C": {}}
    manifest = tmp_path / "loco.csv"
    pd.DataFrame([
        {"subject_id": "hup:A", "outer_fold": 1, "split_role": "fit"},
        {"subject_id": "pediatric:C", "outer_fold": 1, "split_role": "validation"},
        {"subject_id": "lzu:B", "outer_fold": 1, "split_role": "test"},
    ]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="exactly one fixed outer-test fold"):
        read_fixed_split_manifest(manifest, subjects)
    splits = read_fixed_split_manifest(manifest, subjects, allow_partial_test_union=True)
    assert splits[0]["test_subjects"] == ["lzu:B"]


def test_v3_command_omits_false_store_true_switch() -> None:
    runner = _load_v3_runner()
    command = runner._command("python", {"allow_partial_fixed_test_manifest": False})
    assert "--allow_partial_fixed_test_manifest" not in command
    assert "--no-allow_partial_fixed_test_manifest" not in command
    command = runner._command("python", {"allow_partial_fixed_test_manifest": True})
    assert command[-1] == "--allow_partial_fixed_test_manifest"


def test_p2_center_balanced_batch_size_matches_available_centers() -> None:
    assert _p2_patient_batch_size("") == 4
    for center in ("hup", "lzu", "multicenter", "pediatric"):
        assert _p2_patient_batch_size(center) == 3


def test_statistics_normalizes_evaluator_experiment_column(tmp_path: Path) -> None:
    source = tmp_path / "patient_level.csv"
    frame = pd.DataFrame({"experiment": ["PRQ-Net", "BCR-Net"]})
    normalized = _normalize_model_column(frame, source=source)
    assert normalized["model"].tolist() == ["PRQ-Net", "BCR-Net"]
    assert "experiment" not in normalized.columns


def test_statistics_rejects_conflicting_model_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame({"model": ["PRQ-Net"], "experiment": ["BCR-Net"]})
    with pytest.raises(ValueError, match="Conflicting model"):
        _normalize_model_column(frame, source=tmp_path / "patient_level.csv")


def test_statistics_runs_on_evaluator_patient_schema(tmp_path: Path) -> None:
    models = ("PRQ-Net", "BCR-Net", "CDEL")
    for seed in (42, 52, 62):
        metrics_dir = tmp_path / "pooled_cv" / f"seed_{seed}" / "metrics"
        thresholds_dir = metrics_dir.parent / "thresholds"
        metrics_dir.mkdir(parents=True)
        thresholds_dir.mkdir()
        rows = []
        for patient_index, subject in enumerate(("hup:A", "lzu:B"), start=1):
            for model_index, model in enumerate(models):
                row = {
                    "experiment": model,
                    "subject_id": subject,
                    "center": subject.split(":")[0],
                    "outer_fold": patient_index,
                }
                row.update({metric: 0.5 + 0.01 * model_index for metric in CORE_METRICS})
                rows.append(row)
        pd.DataFrame(rows).to_csv(metrics_dir / "patient_level.csv", index=False)
        pd.DataFrame(
            {"outer_fold": [1, 1, 1], "model": models, "threshold": [0.5, 0.5, 0.5]}
        ).to_csv(thresholds_dir / "selected_thresholds.csv", index=False)
    result = summarize_statistics(
        output_root=tmp_path, seeds=[42, 52, 62], bootstrap_repeats=20, permutation_repeats=20,
    )
    assert result["status"] == "passed"
    assert result["n_patients"] == 2
    assert (tmp_path / "statistics" / "paired_patient_bootstrap.csv").is_file()
