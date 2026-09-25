from __future__ import annotations

import torch

from neuroez_c.p21_center_alignment import clean_nez_center_moment_alignment


def test_alignment_uses_patient_pooled_clean_nez_only():
    embedding = torch.tensor([[[0., 0.], [99., 99.]], [[2., 2.], [-99., -99.]], [[4., 4.], [8., 8.]]], requires_grad=True)
    labels = torch.tensor([[1., 0.], [1., 0.], [1., 1.]])
    mask = torch.ones(3, 2, dtype=torch.bool); centers = torch.tensor([0, 1, 1])
    loss, diag = clean_nez_center_moment_alignment(embedding, labels, mask, centers)
    assert loss > 0
    assert diag["center_alignment_patient_count"] == 3
    altered = embedding.detach().clone(); altered[:, 1] = torch.where(labels[:, 1, None] == 0, torch.tensor(500.0), altered[:, 1])
    altered_loss, _ = clean_nez_center_moment_alignment(altered, labels, mask, centers)
    assert torch.allclose(loss, altered_loss)
    missing, _ = clean_nez_center_moment_alignment(embedding, labels, mask, None)
    assert missing == 0

