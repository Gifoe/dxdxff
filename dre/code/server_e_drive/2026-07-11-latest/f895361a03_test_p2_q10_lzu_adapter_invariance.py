import unittest
import pandas as pd
from neuroez_c.p2_q10_lzu_adapter_protocol import non_lzu_invariance_audit


class InvarianceTests(unittest.TestCase):
    def test_non_lzu_is_bitwise_identical(self):
        frame = pd.DataFrame({"subject_id":["hup:a"], "center":["hup"], "base_nez_logit":[.2], "adapted_nez_logit":[.2], "base_pred_nez":[1], "adapted_pred_nez":[1]})
        self.assertTrue(non_lzu_invariance_audit(frame)["passed"])

    def test_non_lzu_change_fails(self):
        frame = pd.DataFrame({"subject_id":["hup:a"], "center":["hup"], "base_nez_logit":[.2], "adapted_nez_logit":[.21], "base_pred_nez":[1], "adapted_pred_nez":[1]})
        self.assertFalse(non_lzu_invariance_audit(frame)["passed"])

