import unittest
import numpy as np
from neuroez_c.v3_rcc_decoder import fit_raw_probability_threshold,apply_raw_probability_threshold,FORMAL_DECISION_RULE
class RCCDecoderTests(unittest.TestCase):
 def test_validation_only_threshold(self):
  rows=[{'score_nez':np.array([.1,.9]),'labels_nez':np.array([0,1]),'channel_mask':np.ones(2,bool),'center':'hup'}]
  fitted=fit_raw_probability_threshold(rows);self.assertFalse(fitted['test_labels_used']);self.assertEqual(apply_raw_probability_threshold(np.array([.2,.8]),fitted['threshold'])['decision_rule'],FORMAL_DECISION_RULE)
 def test_threshold_candidates_include_edges(self):
  from neuroez_c.v3_rcc_decoder import raw_threshold_candidates
  got=raw_threshold_candidates(np.array([.2,.8]));self.assertIn(0.,got);self.assertIn(1.,got)
