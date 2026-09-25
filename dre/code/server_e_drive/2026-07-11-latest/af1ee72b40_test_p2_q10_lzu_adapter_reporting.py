import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import pandas as pd
from tests.test_p2_q10_lzu_adapter_protocol import frame
from neuroez_c.p2_q10_lzu_adapter_protocol import evaluate_channel_ledger


class ReportingTests(unittest.TestCase):
    def test_cli_dry_run_parses_optimization_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache.pkl"
            cache.touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"P2_profile": "P2_TEMPORAL_Q10", "per_fold": [
                {"outer_fold": fold, "status": "passed"} for fold in range(1, 6)
            ]}))
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "run_p2_q10_lzu_adapter.py"),
                       "--p2_q10_root", str(root), "--base_manifest", str(manifest),
                       "--window_cache_path", str(cache), "--allowed_subjects_ledger", "unused.csv",
                       "--fixed_fold_manifest", "unused.csv", "--output_dir", str(root / "out"),
                       "--p2_lzu_adapter_patient_batch_size", "2", "--p2_lzu_adapter_trunk_lr", "1e-3",
                       "--p2_lzu_adapter_output_lr", "3e-3", "--p2_lzu_adapter_min_optimizer_steps", "100",
                       "--dry_run"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["patient_batch_size"], 2)
            self.assertEqual(payload["min_optimizer_steps"], 100)
            self.assertEqual(payload["trunk_lr"], 1e-3)
            self.assertEqual(payload["output_lr"], 3e-3)

    def test_formal_and_truek_are_separate(self):
        data = frame()
        formal, summary, _, _ = evaluate_channel_ledger(data, logit_column="base_nez_logit")
        truek, diagnostic, _, _ = evaluate_channel_ledger(data, logit_column="base_nez_logit", truek=True)
        self.assertEqual(len(formal), 1); self.assertEqual(len(truek), 1)
        self.assertIn("patient_macro_f1", summary); self.assertIn("patient_macro_f1", diagnostic)

    def test_five_fold_synthetic_audit_and_adapter_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p2"; root.mkdir(); output = Path(directory) / "audit"; adapter_output = Path(directory) / "adapter"
            subjects = [f"lzu:s{i:02d}" for i in range(10)] + [f"hup:s{i:02d}" for i in range(10)]
            pd.DataFrame({"subject_id": subjects}).to_csv(Path(directory) / "subjects.csv", index=False)
            manifest_rows = [{"subject_id": subject, "outer_fold": index // 4 + 1} for index, subject in enumerate(subjects)]
            pd.DataFrame(manifest_rows).to_csv(Path(directory) / "folds.csv", index=False)
            (root / "run_args_b0_pruned.json").write_text(json.dumps({"config_name": "P2_TEMPORAL_Q10_seed42"}))
            def rows(ids, fold, threshold=.5):
                result = []
                for subject in ids:
                    center = subject.split(":")[0]
                    # Channel A is a borderline false NEZ at threshold 0.5, so a
                    # functioning adapter can produce an observable legal change.
                    for channel, label, logit in (("A", 0, .05), ("B", 1, .3)):
                        result.append({"subject_id":subject,"center":center,"channel_name":channel,"label_nez":label,"base_nez_logit":logit,"contextual_channel_embedding":"[0.1,0.2]","q10_nez_probability":.3,"anchor_distance_z":.1,"temporal_delta_norm":.2,"valid_seizure_count":1,"outer_fold":fold,"selected_threshold":threshold,"threshold_source":"outer_validation"})
                return pd.DataFrame(result)
            for fold in range(1, 6):
                folder = root / f"fold_{fold}"; folder.mkdir(); test_ids = [row["subject_id"] for row in manifest_rows if row["outer_fold"] == fold]; train_ids = [subject for subject in subjects if subject not in test_ids]
                lzu_train = [subject for subject in train_ids if subject.startswith("lzu:")]
                fit_ids = lzu_train[:4]; val_ids = lzu_train[4:5]
                # Include non-LZU outer-train patients in fit while keeping every split patient-disjoint.
                fit_ids += [subject for subject in train_ids if not subject.startswith("lzu:")][:2]
                rows(fit_ids, fold).to_csv(folder / f"fit_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
                rows(val_ids, fold).to_csv(folder / f"val_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
                rows(test_ids, fold).to_csv(folder / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
                torch.save({"model_state_dict": {"weight": torch.zeros(1)}}, folder / "best_model.pt")
            cache = Path(directory) / "cache.pkl"; cache.touch()
            audit = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "audit_p2_q10_for_adapter.py"), "--p2_q10_root", str(root), "--allowed_subjects_ledger", str(Path(directory) / "subjects.csv"), "--fixed_fold_manifest", str(Path(directory) / "folds.csv"), "--require_n_patients", "20", "--output_dir", str(output)]
            audit_result = subprocess.run(audit, capture_output=True, text=True)
            self.assertEqual(audit_result.returncode, 0, audit_result.stdout + audit_result.stderr)
            run = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "run_p2_q10_lzu_adapter.py"), "--p2_q10_root", str(root), "--base_manifest", str(output / "p2_q10_adapter_base_manifest.json"), "--window_cache_path", str(cache), "--allowed_subjects_ledger", str(Path(directory) / "subjects.csv"), "--fixed_fold_manifest", str(Path(directory) / "folds.csv"), "--require_n_patients", "20", "--output_dir", str(adapter_output), "--p2_lzu_adapter_epochs", "55", "--p2_lzu_adapter_patience", "5", "--p2_lzu_adapter_min_epochs", "10", "--p2_lzu_adapter_min_optimizer_steps", "100", "--p2_lzu_adapter_patient_batch_size", "2", "--max_outer_folds", "1"]
            subprocess.run(run, check=True, capture_output=True, text=True)
            self.assertTrue((adapter_output / "non_lzu_invariance_audit.json").is_file())
            selection=pd.read_csv(adapter_output/"p2_lzu_adapter_checkpoint_selection.csv")
            self.assertEqual(int(selection.selected_checkpoint.sum()),1)
            optimization=json.loads((adapter_output/"p2_lzu_adapter_optimization_audit.json").read_text())
            self.assertTrue(optimization["all_folds_optimization_passed"])
            self.assertGreaterEqual(optimization["folds"][0]["total_optimizer_steps"],100)
            self.assertGreaterEqual(optimization["folds"][0]["test_lzu_prediction_changes"],1)
            summarize = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "summarize_p2_q10_lzu_adapter.py"), "--base_root", str(output), "--adapter_root", str(adapter_output), "--bootstrap_repeats", "20", "--seed", "42"]
            summary_result = subprocess.run(summarize, capture_output=True, text=True)
            self.assertEqual(summary_result.returncode, 0, summary_result.stdout + summary_result.stderr)
            status = json.loads((adapter_output / "p2_lzu_adapter_final_status.json").read_text())
            self.assertEqual(status["optimization_status"], "OPTIMIZATION_EFFECTIVE")
            self.assertEqual(status["performance_status"], "OPTIMIZATION_EFFECTIVE_BUT_NO_PERFORMANCE_GAIN")
