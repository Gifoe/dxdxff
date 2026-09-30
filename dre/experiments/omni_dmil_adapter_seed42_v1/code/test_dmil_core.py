from __future__ import annotations

import unittest
from itertools import product

import numpy as np
from sklearn.metrics import f1_score

from dmil_core import (
    RobustScaler,
    choose_threshold,
    distribution_features,
    fit_adapter,
    logit,
    patient_equal_loss_and_gradient,
    sigmoid,
)


class DmilCoreTest(unittest.TestCase):
    def test_distribution_definition(self):
        values = np.asarray([0.1, 0.2, 0.3, 0.9, 0.8])
        result = distribution_features(values)
        self.assertAlmostEqual(result["mean_score"], 0.46)
        self.assertAlmostEqual(result["top20_mean"], 0.9)
        self.assertAlmostEqual(result["q90"], 0.86)
        self.assertAlmostEqual(result["heterogeneity"], float(values.std(ddof=0)))

    def test_zero_adapter_identity(self):
        probability = np.asarray([0.1, 0.25, 0.8])
        scaled = np.ones((3, 3))
        observed = sigmoid(logit(probability) + scaled @ np.zeros(3))
        np.testing.assert_allclose(observed, probability, atol=1e-12, rtol=0)

    def test_scaler_is_train_only_transform(self):
        fit = np.asarray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
        scaler = RobustScaler.fit(fit)
        np.testing.assert_allclose(scaler.median, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(scaler.iqr, [2.0, 2.0, 2.0])
        transformed = scaler.transform([[200.0, -200.0, 4.0]])
        np.testing.assert_allclose(transformed, [[5.0, -5.0, 0.0]])

    def test_only_active_coefficient_moves(self):
        base = np.zeros(6)
        features = np.asarray([[0.0, 0.0, 0.0], [1.0, 3.0, 4.0]] * 3)
        labels = np.asarray([0, 1] * 3)
        patients = np.asarray(["a", "a", "b", "b", "c", "c"])
        result = fit_adapter(base, features, labels, patients, active=(0,), max_epochs=20, patience=5)
        beta = result["beta"]
        self.assertGreater(beta[0], 0)
        self.assertEqual(beta[1], 0)
        self.assertEqual(beta[2], 0)

    def test_vectorized_patient_equal_gradient_matches_finite_difference(self):
        base = np.asarray([-0.4, 0.2, 0.8, -1.0])
        features = np.asarray([[0.2, -0.1, 0.0], [1.0, 0.5, -0.2], [-0.3, 0.7, 0.4], [0.1, -0.6, 0.9]])
        labels = np.asarray([0, 1, 1, 0])
        patients = np.asarray(["a", "a", "b", "b"])
        beta = np.asarray([0.2, -0.1, 0.05])
        loss, gradient = patient_equal_loss_and_gradient(base, features, labels, patients, beta)
        epsilon = 1e-6
        numerical = []
        for index in range(3):
            left = beta.copy(); left[index] -= epsilon
            right = beta.copy(); right[index] += epsilon
            left_loss, _ = patient_equal_loss_and_gradient(base, features, labels, patients, left)
            right_loss, _ = patient_equal_loss_and_gradient(base, features, labels, patients, right)
            numerical.append((right_loss - left_loss) / (2 * epsilon))
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-7)

    def test_patient_equal_weighted_bce_uses_channel_mean(self):
        base = np.zeros(4)
        features = np.zeros((4, 3))
        labels = np.asarray([1, 0, 0, 0])
        patients = np.asarray(["a", "a", "b", "b"])
        loss, gradient = patient_equal_loss_and_gradient(
            base, features, labels, patients, np.zeros(3)
        )
        expected_patient_a = (2.0 * np.log(2.0) + np.log(2.0)) / 2.0
        expected_patient_b = np.log(2.0)
        self.assertAlmostEqual(loss, (expected_patient_a + expected_patient_b) / 2.0)
        np.testing.assert_allclose(gradient, np.zeros(3), atol=0, rtol=0)

    def test_threshold_search_matches_exhaustive_definition(self):
        rng = np.random.default_rng(42)
        for size, tied in product((7, 23, 101), (False, True)):
            labels = rng.integers(0, 2, size=size, dtype=np.int8)
            labels[0], labels[1] = 0, 1
            scores = rng.random(size)
            if tied:
                scores = np.round(scores, 1)
            candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
            expected = max(
                candidates,
                key=lambda threshold: (
                    f1_score(labels, scores >= threshold, average="macro", zero_division=0),
                    -threshold,
                ),
            )
            actual = choose_threshold(labels, scores)
            self.assertEqual(actual["candidate_count"], len(candidates))
            self.assertEqual(actual["threshold"], expected)


if __name__ == "__main__":
    unittest.main()
