import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def _rows(subjects, fold):
    rows = []
    for subject in subjects:
        center = subject.split(":")[0]
        rows += [
            {"subject_id": subject, "center": center, "outer_fold": fold, "channel_name": "A", "label_nez": 0, "direct_nez_logit": -.4},
            {"subject_id": subject, "center": center, "outer_fold": fold, "channel_name": "B", "label_nez": 1, "direct_nez_logit": .4},
        ]
    return pd.DataFrame(rows)


def test_five_fold_audit_run_and_summary(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    p2, v3, out = tmp_path / "p2", tmp_path / "v3", tmp_path / "out"
    p2.mkdir(); v3.mkdir()
    (p2 / "run_args_p23.json").write_text('{"config_name":"P2_TEMPORAL_Q10"}')
    (v3 / "v3_qbc_protocol_audit.json").write_text('{"profile":"V3_QBC"}')
    subjects = [f"hup:s{i}" for i in range(10)]
    pd.DataFrame({"subject_id": subjects}).to_csv(tmp_path / "subjects.csv", index=False)
    folds = pd.DataFrame({"subject_id": subjects, "outer_fold": [index // 2 + 1 for index in range(10)]})
    folds.to_csv(tmp_path / "folds.csv", index=False)
    for fold in range(1, 6):
        test = [subject for subject, assigned in zip(subjects, folds.outer_fold) if assigned == fold]
        validation = [subject for subject, assigned in zip(subjects, folds.outer_fold) if assigned != fold][:2]
        p2_fold = p2 / f"fold_{fold}"; p2_fold.mkdir()
        for root, folder in ((p2, p2_fold), (v3, v3)):
            _rows(validation, fold).to_csv(folder / f"val_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
            _rows(test, fold).to_csv(folder / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
    audit = [sys.executable, str(repo / "scripts" / "audit_p2_v3_fusion_inputs.py"), "--p2_q10_root", str(p2), "--v3_qbc_root", str(v3), "--allowed_subjects_ledger", str(tmp_path / "subjects.csv"), "--fixed_fold_manifest", str(tmp_path / "folds.csv"), "--require_n_patients", "10", "--output_dir", str(out / "audit")]
    assert subprocess.run(audit, capture_output=True, text=True).returncode == 0
    run = [sys.executable, str(repo / "scripts" / "run_p2_v3_conservative_fusion.py"), "--p2_q10_root", str(p2), "--v3_qbc_root", str(v3), "--input_manifest", str(out / "audit" / "p2_v3_fusion_input_manifest.json"), "--allowed_subjects_ledger", str(tmp_path / "subjects.csv"), "--fixed_fold_manifest", str(tmp_path / "folds.csv"), "--require_n_patients", "10", "--output_dir", str(out / "fusion")]
    dry = subprocess.run(run + ["--dry_run"], capture_output=True, text=True)
    assert dry.returncode == 0 and json.loads(dry.stdout)["mode"] == "dry_run"
    assert subprocess.run(run, capture_output=True, text=True).returncode == 0
    summary = [sys.executable, str(repo / "scripts" / "summarize_p2_v3_conservative_fusion.py"), "--fusion_root", str(out / "fusion"), "--bootstrap_repeats", "10"]
    result = subprocess.run(summary, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ledger = pd.read_csv(out / "fusion" / "p2_v3_fusion_oof_channel_ledger.csv")
    assert len(ledger) == 20 and ledger.true_count_used_for_prediction.eq(False).all()
    assert (out / "fusion" / "p2_v3_fusion_paired_bootstrap.csv").is_file()
    assert (out / "fusion" / "p2_v3_fusion_by_fold.csv").is_file()
    assert (out / "fusion" / "p2_v3_fusion_by_center.csv").is_file()
    fused_manifest = json.loads((out / "fusion" / "p2_v3_fusion_input_manifest.json").read_text())
    assert all("selected_fusion_validation_threshold" in fold for fold in fused_manifest["folds"])
    assert (out / "fusion" / "P2_Q10_V3_CONSERVATIVE_FUSION_REPORT.md").is_file()
