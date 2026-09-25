import unittest
import torch
from neuroez_c.p2_q10_lzu_adapter_loss import patient_balanced_bce, patient_mean_unweighted_bce, original_ez_pairwise_ranking_loss


class AdapterLossTests(unittest.TestCase):
    def test_patient_equal_weighting(self):
        logits = torch.zeros(5); labels = torch.tensor([0., 1., 0., 0., 0.]); patient = torch.tensor([0, 0, 1, 1, 1])
        self.assertAlmostEqual(float(patient_balanced_bce(logits, labels, patient)), float(patient_mean_unweighted_bce(logits, labels, patient)), places=6)

    def test_pairwise_prefers_true_ez_high_score(self):
        labels = torch.tensor([0., 1.]); patient = torch.tensor([0, 0])
        correct = original_ez_pairwise_ranking_loss(torch.tensor([-1., 1.]), labels, patient)
        wrong = original_ez_pairwise_ranking_loss(torch.tensor([1., -1.]), labels, patient)
        self.assertLess(float(correct), float(wrong))

    def test_pairwise_never_builds_cross_patient_pairs(self):
        labels = torch.tensor([0., 1., 0., 1.])
        patients = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor([-1., 1., 2., -2.])
        combined = original_ez_pairwise_ranking_loss(logits, labels, patients)
        separate = torch.stack([
            original_ez_pairwise_ranking_loss(logits[:2], labels[:2], torch.zeros(2, dtype=torch.long)),
            original_ez_pairwise_ranking_loss(logits[2:], labels[2:], torch.zeros(2, dtype=torch.long)),
        ]).mean()
        self.assertAlmostEqual(float(combined), float(separate), places=7)
