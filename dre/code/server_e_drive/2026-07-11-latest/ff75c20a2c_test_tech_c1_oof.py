import pandas as pd,pytest
from neuroez_c.task2.tech_outcome_c1.evaluation import assert_oof
def test_duplicate_oof_fails():
 with pytest.raises(RuntimeError):assert_oof(pd.DataFrame({'patient_key':['p','p']}))
