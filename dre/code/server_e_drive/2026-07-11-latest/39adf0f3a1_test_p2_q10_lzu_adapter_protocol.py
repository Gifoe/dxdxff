import unittest
import numpy as np
import pandas as pd
from neuroez_c.p2_q10_lzu_adapter_protocol import robust_patient_standardize, stable_ez_percentile_rank, canonicalize_channel_frame


def frame():
    return pd.DataFrame({"subject_id": ["lzu:a", "lzu:a"], "center": ["lzu", "lzu"], "channel_name": ["A", "B"], "label_nez": [0,1], "base_nez_logit": [-1.,1.], "contextual_channel_embedding": ["[0.1,0.2]", "[0.3,0.4]"], "q10_nez_probability": [.2,.8], "anchor_distance_z": [0.,0.], "temporal_delta_norm": [0.,.1], "valid_seizure_count": [1,1], "outer_fold": [1,1], "selected_threshold": [.5,.5]})


class AdapterProtocolTests(unittest.TestCase):
    def test_robust_z_fallback_and_constant(self):
        self.assertTrue(np.allclose(robust_patient_standardize(np.array([1., 1.])), 0))
        self.assertTrue(np.isfinite(robust_patient_standardize(np.array([1., 1., 2.]))).all())

    def test_rank_direction_and_ties(self):
        rank = stable_ez_percentile_rank(np.array([2., -2., -2.]))
        self.assertEqual(rank[0], 0); self.assertEqual(rank[1], rank[2]); self.assertGreater(rank[1], rank[0])

    def test_missing_required_field_fails_closed(self):
        bad = frame().drop(columns=["temporal_delta_norm"])
        with self.assertRaises(ValueError): canonicalize_channel_frame(bad)

