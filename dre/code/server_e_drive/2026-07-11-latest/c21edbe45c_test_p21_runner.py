from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runner_defaults_and_interfaces_are_explicit():
    text = (ROOT / "scripts" / "run_p21_v3_asrr_nez_80.ps1").read_text(encoding="utf-8")
    assert '[string]$Profiles = "R5_FULL"' in text
    assert '[string]$Seeds = "42"' in text
    assert '[switch]$DryRun' in text and '[switch]$SkipExisting' in text
    assert "p21_overall_summary.csv" in text and "p21_protocol_audit.json" in text
    assert "--use-p21-v3-asrr-nez" in text
    assert "--no-use_cane_path_cp_nez" in text
    assert "nested_fold_safe" in text


def test_bash_runner_and_reporter_exist():
    assert (ROOT / "scripts" / "run_p21_v3_asrr_nez_80.sh").is_file()
    assert (ROOT / "scripts" / "summarize_p21_v3_asrr.py").is_file()

