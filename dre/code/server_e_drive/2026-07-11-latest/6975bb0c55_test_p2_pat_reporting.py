import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import torch

from tests.p2_pat_test_utils import channel_frame


class PATReportingTests(unittest.TestCase):
    def test_parser_help(self):
        root=Path(__file__).resolve().parents[1]
        for script in ("audit_p2_q10_pat_base.py","run_p2_q10_pat.py","summarize_p2_q10_pat.py"):
            result=subprocess.run([sys.executable,str(root/"scripts"/script),"--help"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_synthetic_audit_and_single_fold_decoder_smoke(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary=Path(directory); base=temporary/"p2"; base.mkdir(); audit=temporary/"audit"
            subjects=[f"hup:s{i:02d}" for i in range(5)]+[f"lzu:s{i:02d}" for i in range(5)]+[f"multicenter:s{i:02d}" for i in range(5)]+[f"pediatric:s{i:02d}" for i in range(5)]
            pd.DataFrame({"subject_id":subjects}).to_csv(temporary/"subjects.csv",index=False)
            manifest=pd.DataFrame({"subject_id":subjects,"outer_fold":[index//4+1 for index in range(20)]}); manifest.to_csv(temporary/"folds.csv",index=False)
            (base/"run_args_p23.json").write_text(json.dumps({"config_name":"P2_TEMPORAL_Q10_seed42"}))
            for fold in range(1,6):
                folder=base/f"fold_{fold}"; folder.mkdir(); test=manifest.loc[manifest.outer_fold.eq(fold),"subject_id"].tolist(); train=[subject for subject in subjects if subject not in test]; validation=train[:4]; fit=train[4:]
                for name,ids in (("fit",fit),("val",validation),("test",test)):
                    frame=channel_frame(tuple(ids),fold=fold,threshold=.45)
                    frame.to_csv(folder/f"{name}_channel_predictions_neuroez_v2_fold_{fold}.csv",index=False)
                torch.save({"model_state_dict":{"weight":torch.zeros(1)}},folder/"best_model.pt")
            audit_command=[sys.executable,str(repo/"scripts"/"audit_p2_q10_pat_base.py"),"--p2_q10_root",str(base),"--allowed_subjects_ledger",str(temporary/"subjects.csv"),"--fixed_fold_manifest",str(temporary/"folds.csv"),"--require_n_patients","20","--output_dir",str(audit)]
            result=subprocess.run(audit_command,capture_output=True,text=True); self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for profile in ("PAT0_GLOBAL","PAT1_MINIMAL","PAT2_EXTENDED"):
                output=temporary/profile
                command=[sys.executable,str(repo/"scripts"/"run_p2_q10_pat.py"),"--p2_q10_root",str(base),"--base_manifest",str(audit/"p2_pat_base_manifest.json"),"--p2_pat_profile",profile,"--output_dir",str(output),"--max_outer_folds","1","--random_seed","42"]
                result=subprocess.run(command,capture_output=True,text=True); self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertTrue((output/"p2_pat_oof_channel_ledger.csv").is_file())
                invariance=json.loads((output/"p2_pat_score_invariance_audit.json").read_text()); self.assertTrue(invariance["passed"])
                ledger=pd.read_csv(output/"p2_pat_oof_channel_ledger.csv"); self.assertFalse(ledger.true_count_used_for_prediction.astype(bool).any())
            summary_command=[sys.executable,str(repo/"scripts"/"summarize_p2_q10_pat.py"),"--root_dir",str(temporary),"--bootstrap_repeats","20","--seed","42"]
            result=subprocess.run(summary_command,capture_output=True,text=True); self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue((temporary/"p2_pat_ablation_summary.csv").is_file())
            self.assertTrue((temporary/"p2_pat_paired_bootstrap.csv").is_file())


if __name__=="__main__": unittest.main()
