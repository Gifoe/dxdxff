from pathlib import Path
import subprocess
import sys

from neuroez_c.task2.tech_outcome_c1_aligned import MODEL_VERSION, PROTOCOL
from neuroez_c.task2.tech_outcome_c1_aligned.audit import protocol_audit
from neuroez_c.task2.tech_outcome_c1.sampling import patient_batches


def test_protocol_audit_records_required_isolation_and_selection():
    audit = protocol_audit()
    assert MODEL_VERSION == "TECH_OUTCOME_C1_ALIGNED"
    assert PROTOCOL == "OUTER_5FOLD_TECH_C1_ALIGNED_MULTIVIEW"
    assert audit["uses_task1"] is False
    assert audit["uses_ez_nez_labels"] is False
    assert audit["uses_full_set_evaluation"] is False
    assert audit["checkpoint_monitor"] == "validation_auroc"
    assert audit["outer_test_inference_count_per_fold"] == 1
    assert audit["loss"] == "single_patient_bce_after_mean_view_logit"


def test_patient_sampler_has_no_replacement_and_no_drop_last():
    keys = [f"p{i}" for i in range(9)]
    labels = {key: i % 2 for i, key in enumerate(keys)}
    batches = patient_batches(keys, labels, batch_size=4, seed=42)
    flat = [key for batch in batches for key in batch]
    assert len(batches) == 3
    assert len(flat) == len(set(flat)) == len(keys)
    assert set(flat) == set(keys)


def test_runner_has_no_full_set_eval_and_loads_auroc_checkpoint_once():
    path = Path(__file__).parents[2] / "scripts" / "task2" / "run_tech_outcome_c1_aligned.py"
    source = path.read_text(encoding="utf8")
    assert "cohort.view(" not in source
    assert "best_val_bce.pt" not in source
    assert source.count("cp/'best_val_auroc.pt'") == 2  # one save, one final load
    assert source.count("f'Final test fold {fold}'") == 1
    assert "--gradient-clip-norm" in source
    assert "--checkpoint-monitor" in source
    assert "--checkpoint-mode" in source
    assert "Path(a.output_dir)/'_smoke' if a.smoke_test" in source
    assert "Task 1" not in source
    assert "EZ/NEZ" not in source


def test_oof_cardinality_contract():
    patient_rows = [{"patient_key": key} for key in ("a", "b")]
    view_rows = [
        {"patient_key": key, "view_index": index}
        for key in ("a", "b")
        for index in range(8)
    ]
    assert len({row["patient_key"] for row in patient_rows}) == len(patient_rows)
    for key in ("a", "b"):
        assert sum(row["patient_key"] == key for row in view_rows) == 8


def test_strict_mode_rejects_augmentation_before_loading_data(tmp_path):
    script = Path(__file__).parents[2] / "scripts" / "task2" / "run_tech_outcome_c1_aligned.py"
    command = [
        sys.executable,
        str(script),
        "--raw-cache", str(tmp_path / "raw.pkl"),
        "--outcome-table", str(tmp_path / "outcomes.csv"),
        "--fold-manifest", str(tmp_path / "folds.csv"),
        "--exclusion-manifest", str(tmp_path / "exclude.csv"),
        "--cache-dir", str(tmp_path / "cache"),
        "--output-dir", str(tmp_path / "output"),
        "--enable-augmentations",
        "--strict",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "strict aligned protocol rejects augmentation" in result.stderr
    assert not (tmp_path / "output").exists()
