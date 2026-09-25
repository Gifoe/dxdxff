import unittest
from neuroez_c.p2_pat_protocol import canonicalize_pat_channel_frame,threshold_probability
from tests.p2_pat_test_utils import channel_frame


class PATProtocolTests(unittest.TestCase):
    def test_label_semantics(self): self.assertTrue((canonicalize_pat_channel_frame(channel_frame()).label_ez==1-canonicalize_pat_channel_frame(channel_frame()).label_nez).all())
    def test_duplicate_rejected(self):
        frame=channel_frame(); frame=frame._append(frame.iloc[0],ignore_index=True)
        with self.assertRaises(ValueError): canonicalize_pat_channel_frame(frame)
    def test_single_fold_threshold(self): self.assertEqual(threshold_probability(channel_frame(),1),.45)
    def test_multiple_thresholds_rejected(self):
        frame=channel_frame(); frame.loc[0,"selected_threshold"]=.4
        with self.assertRaises(RuntimeError): threshold_probability(frame,1)
    def test_center_is_diagnostic_only(self): self.assertIn("center",canonicalize_pat_channel_frame(channel_frame()).columns)
