import tempfile,unittest
from pathlib import Path
import pandas as pd
from neuroez_c.v3_rcc_protocol import validate_v3_rcc_protocol
class RCCProtocolTests(unittest.TestCase):
 def test_fixed_folds_pass(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);subjects=[f'p{i}' for i in range(10)];pd.DataFrame({'subject_id':subjects}).to_csv(root/'s.csv',index=False);pd.DataFrame({'subject_id':subjects,'outer_fold':[i%5+1 for i in range(10)]}).to_csv(root/'f.csv',index=False)
   self.assertEqual(validate_v3_rcc_protocol(patient_ids=subjects,allowed_subjects_path=root/'s.csv',outer_fold_manifest_path=root/'f.csv',require_n_patients=10,seed=42)['status'],'passed')
