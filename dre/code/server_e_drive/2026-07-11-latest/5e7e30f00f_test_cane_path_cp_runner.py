from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_cane_path_cp_ensemble import CP, KEY, _assert_alignment, aggregate


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_step4d_cane_path_cp_nez_80.ps1"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8").lower()


def _seed_ledger(seed: int, perturb: float = 0.0) -> pd.DataFrame:
    rows = []
    centers = ["hup", "lzu", "multicenter", "pediatric"]
    for fold in range(1, 6):
        for patient in range(4):
            subject = f"{centers[patient]}:f{fold}p{patient}"
            for channel in range(4):
                label_nez = int(channel < 2)
                score = (1.0 if label_nez else -1.0) + perturb
                row = {
                    "outer_fold": fold, "subject_id": subject, "channel_name": f"c{channel}", "channel_id": channel,
                    "center": centers[patient], "label_nez": label_nez, "label_ez": 1-label_nez, "valid": True,
                    "standardized_nez_logit": score, "predicted_patient_threshold": 0.0, "cp_feature_valid": True,
                }
                row.update({name: float(channel) / 4 for name in CP})
                rows.append(row)
    return pd.DataFrame(rows)


def _write_runs(root: Path) -> None:
    for seed, shift in zip((42, 43, 44), (0.0, 0.05, -0.05)):
        folder = root / f"seed_{seed}"; folder.mkdir(parents=True)
        frame = _seed_ledger(seed, shift)
        for fold, group in frame.groupby("outer_fold"):
            group.to_csv(folder / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)


def test_runner_declares_all_five_profiles() -> None:
    text = _runner_text()
    assert all(profile.lower() in text for profile in ("P0", "P1", "P2", "P3", "P4"))


def test_runner_uses_fixed_three_seed_ensemble() -> None:
    text = _runner_text()
    assert "@(42, 43, 44)" in text and "aggregate_cane_path_cp_ensemble.py" in text


def test_runner_allows_single_seed_screening() -> None:
    text = _runner_text()
    assert '[string]$seeds = ""' in text
    assert "$requestedseeds.count -gt 0" in text


def test_runner_uses_four_inner_folds() -> None:
    assert '"--inner-splits", "4"' in _runner_text()


def test_runner_exposes_direct_outer_only_mode() -> None:
    text = _runner_text()
    assert "[switch]$directouteronly" in text
    assert "[double]$directthreshold = 0.5" in text
    assert '"--cane-direct-outer-only"' in text
    assert '[string]$selectionobjective = "f1"' in text
    assert '"--cane-selection-objective"' in text


def test_runner_disables_four_center_sampler_for_outer_only_batch_two() -> None:
    text = _runner_text()
    assert "if ($patientbatchsize -ge 4)" in text
    assert '"--no-center-balanced-batches"' in text
    assert "nested cane-path requires patientbatchsize >= 4" in text


def test_runner_marks_sensitivity80_not_all90_primary() -> None:
    text = _runner_text()
    assert '"--cohort-mode", "sensitivity80"' in text and '"--require-n-patients", "80"' in text


def test_runner_uses_nez_probability_semantics() -> None:
    assert '"--positive-label", "nez"' in _runner_text()


def test_runner_disables_true_count_and_topk_paths() -> None:
    text = _runner_text()
    assert '"--no-use_hard_topk_loss"' in text and '"--no-use_cane_set_nez"' in text


def test_runner_sets_classification_threshold_only_for_direct_mode() -> None:
    text = _runner_text()
    direct_branch = text.split("if ($directouteronly)", 1)[1].split("}", 1)[0]
    assert '"--classification_threshold"' in direct_branch


def test_runner_raw_cache_is_offline_extraction_only() -> None:
    text = _runner_text()
    assert "build_causal_propagation_cache.py" in text and '"--no-use_n6_dual_view_ema"' in text


def test_ensemble_alignment_accepts_identical_ledgers() -> None:
    frames = [_seed_ledger(seed) for seed in (42, 43, 44)]
    _assert_alignment([frame.sort_values(KEY).reset_index(drop=True) for frame in frames])


def test_ensemble_alignment_rejects_channel_key_mismatch() -> None:
    frames = [_seed_ledger(seed).sort_values(KEY).reset_index(drop=True) for seed in (42, 43, 44)]
    frames[1].loc[0, "channel_name"] = "wrong"
    frames[1] = frames[1].sort_values(KEY).reset_index(drop=True)
    with pytest.raises(ValueError, match="keys"):
        _assert_alignment(frames)


def test_ensemble_alignment_rejects_label_mismatch() -> None:
    frames = [_seed_ledger(seed).sort_values(KEY).reset_index(drop=True) for seed in (42, 43, 44)]
    frames[2].loc[0, "label_nez"] = 1 - frames[2].loc[0, "label_nez"]
    with pytest.raises(ValueError, match="label_nez"):
        _assert_alignment(frames)


def test_aggregate_writes_required_outputs(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    output = tmp_path / "ensemble"
    summary = aggregate(tmp_path, output)
    required = ["heldout_summary_cane_path_cp.csv", "heldout_summary_cane_path_cp.json", "heldout_fold_summary.csv", "heldout_center_summary.csv", "heldout_patient_predictions.csv", "heldout_channel_predictions.csv", "bootstrap_ci.csv"]
    assert all((output / name).is_file() for name in required) and summary["ensemble_seeds"] == [42, 43, 44]


def test_aggregate_uses_equal_seed_weighting(tmp_path: Path) -> None:
    _write_runs(tmp_path); output = tmp_path / "ensemble"; aggregate(tmp_path, output)
    row = pd.read_csv(output / "heldout_channel_predictions.csv").iloc[0]
    assert row["ensemble_standardized_nez_logit"] == pytest.approx(1.0)


def test_aggregate_records_no_test_based_selection(tmp_path: Path) -> None:
    _write_runs(tmp_path); output = tmp_path / "ensemble"; summary = aggregate(tmp_path, output)
    assert summary["test_based_seed_selection"] is False and summary["test_label_ensemble_weighting"] is False
