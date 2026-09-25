import unittest

import numpy as np
import pandas as pd

from a12_vcsn.models.catboost_utility import CatBoostUtilityModel
from a12_vcsn.models.ensemble_utility import EnsembleUtilityModel
from a12_vcsn.models.siamese_utility_mlp import SiameseUtilityMLP
from a12_vcsn.models.trajectory_tcn import TrajectoryTCNEncoder, TrajectoryTCNUtilityModel


def training_pairs() -> pd.DataFrame:
    rows = []
    for subject_idx in range(8):
        for pair_idx in range(2):
            delta = 0.2 if (subject_idx + pair_idx) % 2 else -0.2
            trajectory = np.asarray([[[subject_idx + 1., pair_idx + 1.], [subject_idx + 2., pair_idx + 2.]]], dtype=np.float32)
            rows.append({"subject_id": f"p{subject_idx}", "f1": float(subject_idx), "f2": float(pair_idx), "beneficial_label": int(delta > 0), "harmful_label": int(delta < 0), "delta_patient_macro_f1": delta, "patient_pair_weight": 0.5, "eject_trajectory_values": trajectory, "add_trajectory_values": trajectory + .5, "eject_trajectory_mask": np.ones((1, 2), dtype=bool), "add_trajectory_mask": np.ones((1, 2), dtype=bool)})
    return pd.DataFrame(rows)


class A12ModelTests(unittest.TestCase):
    def _assert_model(self, model):
        pairs = training_pairs()
        model.fit(pairs, ["f1", "f2"])
        prediction = model.predict(pairs.iloc[:3])
        self.assertEqual(len(prediction), 3)
        self.assertTrue({"p_benefit", "p_harm", "pred_delta", "utility"}.issubset(prediction.columns))
        fitted = set(model.fit_subjects) | set(getattr(model, "validation_subjects", set()))
        self.assertEqual(fitted, set(pairs.subject_id))

    def test_catboost_dual_head(self):
        self._assert_model(CatBoostUtilityModel(iterations=20, random_seed=42))

    def test_siamese_dual_head(self):
        self._assert_model(SiameseUtilityMLP(epochs=8, random_seed=42))

    def test_ensemble_uses_three_utility_models(self):
        model = EnsembleUtilityModel(random_seed=42, cat_iterations=20, neural_epochs=8)
        self._assert_model(model)
        self.assertEqual(len(model.members), 3)

    def test_tcn_masked_encoder_and_utility(self):
        encoder = TrajectoryTCNEncoder(feature_dim=2, hidden_dim=8)
        embeddings, audit = encoder.encode(np.ones((2, 4, 2), dtype=np.float32), np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=bool))
        self.assertEqual(embeddings.shape, (2, 8))
        self.assertEqual(audit["padded_values_used"], 0)
        self._assert_model(TrajectoryTCNUtilityModel(epochs=8, random_seed=42))
