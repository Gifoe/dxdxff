import unittest
import numpy as np
from neuroez_c.p2_pat_protocol import PAT1_FEATURES,PAT2_EXTRA_FEATURES,build_patient_features,canonicalize_pat_channel_frame
from tests.p2_pat_test_utils import channel_frame


class PATFeatureTests(unittest.TestCase):
    def test_pat1_has_exactly_seven_features(self): self.assertEqual(len(PAT1_FEATURES),7)
    def test_pat2_adds_exactly_seven_features(self): self.assertEqual(len(PAT2_EXTRA_FEATURES),7)
    def test_quantiles_and_iqr(self):
        frame=channel_frame(("hup:a",)); features=build_patient_features(frame,0.,"PAT1_MINIMAL").iloc[0]
        logits=frame.base_nez_logit.to_numpy(); self.assertAlmostEqual(features.logit_iqr,np.quantile(logits,.75)-np.quantile(logits,.25))
    def test_predicted_fraction_uses_no_labels(self):
        frame=channel_frame(("hup:a",)); first=build_patient_features(frame,0.,"PAT1_MINIMAL")
        frame.label_nez=1-frame.label_nez; frame.label_ez=1-frame.label_nez
        second=build_patient_features(frame,0.,"PAT1_MINIMAL"); self.assertTrue(np.allclose(first[list(PAT1_FEATURES)],second[list(PAT1_FEATURES)]))
    def test_entropy_finite(self): self.assertTrue(np.isfinite(build_patient_features(channel_frame(("hup:a",)),0.,"PAT1_MINIMAL").mean_binary_entropy).all())
    def test_channel_count(self): self.assertAlmostEqual(build_patient_features(channel_frame(("hup:a",)),0.,"PAT1_MINIMAL").log1p_n_channels.iloc[0],np.log(5))
    def test_pat2_aliases(self):
        frame=channel_frame(("hup:a",)).rename(columns={"q10_nez_probability":"seizure_nez_probability_q10","anchor_evidence":"negative_anchor_distance"})
        self.assertEqual(len(build_patient_features(frame,0.,"PAT2_EXTENDED")),1)
    def test_pat2_missing_field_fails(self):
        with self.assertRaises(ValueError): build_patient_features(channel_frame(("hup:a",)).drop(columns=["temporal_delta_norm"]),0.,"PAT2_EXTENDED")
    def test_center_not_a_feature(self): self.assertNotIn("center",PAT1_FEATURES+PAT2_EXTRA_FEATURES)
    def test_true_fraction_not_a_feature(self): self.assertFalse(any("true" in name for name in PAT1_FEATURES+PAT2_EXTRA_FEATURES))
