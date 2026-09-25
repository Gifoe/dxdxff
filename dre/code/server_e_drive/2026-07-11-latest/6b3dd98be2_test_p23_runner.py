from pathlib import Path


def test_runner_exposes_required_profile_and_safe_defaults():
    script = Path("scripts/run_p23_trn_nez_80.ps1").read_text(encoding="utf-8")
    assert 'Profiles = "P5_FULL"' in script
    assert '[int]$MaxOuterFolds = 1' in script
    assert "--use-p23-trn-nez" in script
    assert "P6_FULL_WITH_CAUSAL" in script


def test_runner_only_enables_center_balanced_sampler_for_valid_batch_sizes():
    script = Path("scripts/run_p23_trn_nez_80.ps1").read_text(encoding="utf-8")
    assert "$PatientBatchSize -ge 4" in script
    assert "$PatientBatchSize % 4 -eq 0" in script
    assert "ordinary patient batches" in script


def test_runner_exposes_p23_direct_outer_only_without_enabling_it_by_default():
    script = Path("scripts/run_p23_trn_nez_80.ps1").read_text(encoding="utf-8")
    assert "[switch]$DirectOuterOnly" in script
    assert "if ($DirectOuterOnly) { $run += '--p23-direct-outer-only' }" in script
    assert "[int]$BatchSize = 4" in script
    assert "[int]$PatientBatchSize = 4" in script
