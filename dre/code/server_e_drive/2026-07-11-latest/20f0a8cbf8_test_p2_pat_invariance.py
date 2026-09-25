import unittest
from neuroez_c.p2_pat_protocol import score_invariance_audit
from tests.p2_pat_test_utils import channel_frame


class PATInvarianceTests(unittest.TestCase):
    def test_identical_scores_pass(self): self.assertTrue(score_invariance_audit(channel_frame(),channel_frame())["passed"])
    def test_logit_change_fails(self):
        changed=channel_frame(); changed.loc[0,"base_nez_logit"]+=1e-4
        self.assertFalse(score_invariance_audit(channel_frame(),changed)["passed"])
    def test_threshold_change_does_not_change_ranking(self):
        changed=channel_frame(); changed["selected_threshold"]=.7
        self.assertTrue(score_invariance_audit(channel_frame(),changed)["passed"])
