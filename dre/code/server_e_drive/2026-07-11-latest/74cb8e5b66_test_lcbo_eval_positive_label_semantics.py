from types import SimpleNamespace

import torch

from exp_ez_hybrid import compute_a9v8_lcbo_loss


def test_nez_lcbo_bce_uses_labels_nez_in_eval_and_train_loss() -> None:
    outputs = {
        "logits_broad": torch.tensor([[2.0, -2.0]]),
        "logits_core": torch.zeros(1, 2),
        "logits_eval": torch.zeros(1, 2),
        "score_broad": torch.sigmoid(torch.tensor([[2.0, -2.0]])),
        "score_core": torch.full((1, 2), 0.5),
        "score_eval": torch.full((1, 2), 0.5),
    }
    batch = {
        "labels_ez": torch.tensor([[0.0, 1.0]]),
        "labels_nez": torch.tensor([[1.0, 0.0]]),
        "channel_mask": torch.ones(1, 2, dtype=torch.bool),
        "subject_id": ["p1"],
        "canonical_channels": [["a", "b"]],
    }
    lookup = {
        ("p1", "a"): {"label_ez": 0.0, "label_nez": 1.0, "pseudo_core_q": 0.0, "target_semantics": "nez"},
        ("p1", "b"): {"label_ez": 1.0, "label_nez": 0.0, "pseudo_core_q": 0.0, "target_semantics": "nez"},
    }
    args = SimpleNamespace(positive_label="nez", lcbo_loss_mode="symmetric_lcbo", lambda_core_rank=0.0, lambda_soft_mrr=0.0, lambda_subset=0.0, lambda_core_distill=0.0, core_rank_margin=0.05, soft_mrr_tau=0.1, subset_eps=0.05, lcbo_hard_neg_top_frac=1.0, lcbo_min_core_mass=1.0, teacher_mode="physiology_only")
    _, parts = compute_a9v8_lcbo_loss(outputs, batch, lookup, args)
    assert parts["mean_broad_bce_loss"] < 0.2

