import unittest
import pandas as pd
from neuroez_c.p2_pat_oracle_target import compute_stable_patient_oracle_threshold_target
from neuroez_c.p2_pat_protocol import build_patient_features,run_lopo_selection
from tests.p2_pat_test_utils import channel_frame


class PATLOPOTests(unittest.TestCase):
    def setUp(self):
        self.channels=channel_frame(("hup:a","lzu:b","pediatric:c","multicenter:d"))
        self.features=build_patient_features(self.channels,0.,"PAT1_MINIMAL")
        self.targets=pd.DataFrame([compute_stable_patient_oracle_threshold_target(group.base_nez_logit.to_numpy(),group.label_nez.to_numpy(),0.,subject_id=subject,outer_fold=1) for subject,group in self.channels.groupby("subject_id")])
    def test_every_patient_held_out(self):
        _,rows,_=run_lopo_selection(self.channels,self.features,self.targets,profile="PAT1_MINIMAL"); self.assertEqual(rows.heldout_subject_id.nunique(),4)
    def test_fixed_grid_size(self):
        _,rows,_=run_lopo_selection(self.channels,self.features,self.targets,profile="PAT1_MINIMAL"); self.assertEqual(len(rows),48*4)
    def test_one_selected_config(self):
        _,rows,_=run_lopo_selection(self.channels,self.features,self.targets,profile="PAT1_MINIMAL"); self.assertEqual(rows[rows.selected_config][["ridge_alpha","shrinkage","max_prediction_residual"]].drop_duplicates().shape[0],1)
    def test_outer_test_not_required(self):
        selected,_,_=run_lopo_selection(self.channels,self.features,self.targets,profile="PAT1_MINIMAL"); self.assertIn("ridge_alpha",selected)
    def test_requires_three_validation_patients(self):
        subjects=["hup:a","lzu:b"]
        with self.assertRaises(ValueError): run_lopo_selection(self.channels[self.channels.subject_id.isin(subjects)],self.features[self.features.subject_id.isin(subjects)],self.targets[self.targets.subject_id.isin(subjects)],profile="PAT1_MINIMAL")
