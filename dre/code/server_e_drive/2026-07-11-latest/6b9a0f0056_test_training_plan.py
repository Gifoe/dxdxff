from __future__ import annotations

from pathlib import Path

import pytest

from task1_aaai.training_plan import (
    FINAL_BCR_PROFILE,
    FORMAL_SEEDS,
    LOCO_TRAINED_MODELS,
    bcr_command,
    timeconv_command,
    validate_bcr_runtime_contract,
    validate_plan,
)


def _plan() -> dict[str, object]:
    return {
        "repo_root": "E:/repo",
        "python": "E:/python.exe",
        "output_root": "D:/output",
        "confirmatory_config": "E:/repo/config.json",
        "prq_reuse_root": "D:/prq/pooled_cv",
        "bcr_base_args": "D:/bcr/run_args.json",
        "feature_cache": "D:/cache/features.pkl",
        "raw_cache": "D:/cache/raw.pkl",
        "cohort_manifest": "D:/manifests/cohort.csv",
        "outer_fold_manifest": "D:/manifests/folds.csv",
        "timeconv_entrypoint": "E:/repo/remote_task1_omni_baselines/run_baseline.py",
        "seeds": list(FORMAL_SEEDS),
        "bcr_profile": FINAL_BCR_PROFILE,
        "loco_models": list(LOCO_TRAINED_MODELS),
        "bcr_training": {
            "epochs": 40, "patience": 8, "min_epochs_before_early_stop": 18,
            "batch_size": 4, "patient_batch_size": 4, "num_workers": 0, "device": "cuda",
        },
    }


def test_training_plan_rejects_seegformer_schedule() -> None:
    plan = _plan()
    plan["loco_models"] = [*LOCO_TRAINED_MODELS, "SEEGformer"]
    with pytest.raises(ValueError, match="LOCO models"):
        validate_plan(plan)


def test_bcr_command_restores_final_training_budget() -> None:
    plan = _plan()
    validate_plan(plan)
    command = bcr_command(plan, seed=42, output_dir=Path("D:/output/bcr"), partition=Path("D:/manifest.csv"))
    assert "BCR_BOUNDARY_COVERAGE" in command
    assert command[command.index("--patience") + 1] == "8"
    assert command[command.index("--min-epochs-before-early-stop") + 1] == "18"
    assert "q10" not in " ".join(command).lower()


def test_timeconv_command_uses_explicit_split_and_never_schedules_seegformer() -> None:
    plan = _plan()
    command = timeconv_command(plan, seed=62, split_manifest=Path("D:/loco.csv"), output_dir=Path("D:/output/timeconv"))
    assert Path(command[command.index("--splits") + 1]) == Path("D:/loco.csv")
    assert command[command.index("--model") + 1] == "omni_timeconv_cnn"
    assert command[command.index("--patience") + 1] == "8"
    assert command[command.index("--min-epochs-before-early-stop") + 1] == "18"
    assert "seegformer" not in " ".join(command).lower()


def test_current_bcr_runtime_contract_is_complete() -> None:
    audit = validate_bcr_runtime_contract(Path(__file__).resolve().parents[2])
    assert audit["status"] == "passed"
    assert audit["q10_in_bcr"] is False


def test_bcr_runtime_contract_rejects_stale_profile_registry(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    required = (
        "neuroez_c/v3_qbc_profiles.py",
        "neuroez_c/model.py",
        "exp_ez_hybrid.py",
        "run_neuroez_c.py",
        "neuroez_c/v3_qbc_reporting.py",
        "scripts/run_v3_qbc.py",
    )
    for relative in required:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((repo / relative).read_text(encoding="utf-8"), encoding="utf-8")
    profile_path = tmp_path / "neuroez_c/v3_qbc_profiles.py"
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace(
            '"BCR_BOUNDARY_COVERAGE"', '"A3_QBC_FULL"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="out of sync"):
        validate_bcr_runtime_contract(tmp_path)
