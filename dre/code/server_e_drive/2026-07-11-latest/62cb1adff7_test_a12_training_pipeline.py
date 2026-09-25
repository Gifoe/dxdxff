from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from a12_vcsn.config import A12Config
from a12_vcsn.models.siamese_utility_mlp import SiameseUtilityMLP
from a12_vcsn.models.training import PatientBalancedSampler, grouped_train_validation_split, resolve_device
from a12_vcsn.suite import _model_for_variant


def _pairs():
    return pd.DataFrame({
        "subject_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "eject_score_ez": [.9, .8, .7, .6, .4, .3, .2, .1],
        "add_score_ez": [.1, .2, .3, .4, .6, .7, .8, .9],
        "beneficial_label": [0, 0, 0, 0, 1, 1, 1, 1],
        "harmful_label": [1, 1, 1, 1, 0, 0, 0, 0],
        "delta_patient_macro_f1": [-.2, -.1, -.1, -.05, .05, .1, .1, .2],
        "patient_pair_weight": [.5] * 8,
    })


class A12TrainingPipelineTests(unittest.TestCase):
    def test_cuda_unavailable_strict_and_non_strict_are_audited(self):
        import torch
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                resolve_device("cuda", strict=True)
            resolution = resolve_device("cuda", strict=False)
            self.assertEqual(resolution.resolved_device, "cpu")
            self.assertIn("unavailable", resolution.fallback_reason)

    def test_grouped_train_validation_split(self):
        train, validation = grouped_train_validation_split(_pairs(), seed=7)
        self.assertFalse(set(_pairs().loc[train, "subject_id"]) & set(_pairs().loc[validation, "subject_id"]))

    def test_patient_balanced_sampler(self):
        sampler = PatientBalancedSampler(_pairs()["subject_id"], num_samples=400, seed=1)
        sampled = _pairs().iloc[list(iter(sampler))]["subject_id"].value_counts(normalize=True)
        self.assertLess(float(sampled.max() - sampled.min()), .12)

    def test_config_reaches_all_model_factory_members(self):
        config = A12Config(max_epochs=3, batch_size=5, patience=2, hidden_dim=11,
                           dropout=.3, learning_rate=.002, weight_decay=.003,
                           gradient_clip=.7, catboost_depth=4,
                           catboost_learning_rate=.02, catboost_l2_leaf_reg=5.,
                           catboost_max_iterations=9, strict_device=True)
        cat = _model_for_variant("A12-V1", 1, config, None)
        siamese = _model_for_variant("A12-V6", 1, config, None)
        ensemble = _model_for_variant("A12-V10", 1, config, object())
        self.assertEqual((cat.iterations, cat.depth, cat.learning_rate, cat.l2_leaf_reg), (9, 4, .02, 5.))
        self.assertEqual((siamese.epochs, siamese.batch_size, siamese.patience, siamese.hidden_dim), (3, 5, 2, 11))
        self.assertEqual(ensemble.members[0].depth, 4)
        self.assertEqual(ensemble.members[1].batch_size, 5)
        self.assertEqual(ensemble.members[2].trajectory_store, ensemble.trajectory_store)

    def test_siamese_minibatch_checkpoint_and_train_only_scaler(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = SiameseUtilityMLP(epochs=3, batch_size=2, patience=1,
                                       hidden_dim=8, random_seed=3, device="cpu",
                                       output_dir=tmp).fit(_pairs(), ["eject_score_ez", "add_score_ez"])
            self.assertTrue(Path(tmp, "seed_3_checkpoint.pt").exists())
            self.assertTrue(Path(tmp, "seed_3_training_log.csv").exists())
            self.assertTrue(Path(tmp, "seed_3_fit_manifest.json").exists())
            self.assertFalse(model.validation_subjects & model.fit_subjects)
            self.assertEqual(model.scalers.fit_subjects, model.fit_subjects)
            self.assertEqual(len(model.predict(_pairs())), len(_pairs()))


if __name__ == "__main__":
    unittest.main()
