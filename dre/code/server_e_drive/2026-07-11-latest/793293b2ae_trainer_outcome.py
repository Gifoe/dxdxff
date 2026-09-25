"""Patient-level supervised stage helpers."""
from __future__ import annotations
import torch

def class_weight(labels):
    labels = list(labels); n0, n1 = labels.count(0), labels.count(1)
    return 1.0 if not n0 or not n1 else n0 / n1

def freeze_stage_b(model):
    for p in model.encoder.parameters(): p.requires_grad_(False)

def limited_unfreeze_stage_c(model):
    freeze_stage_b(model)
    for p in model.encoder.temporal.b[-1].parameters(): p.requires_grad_(True)
    for p in model.encoder.temporal.p.parameters(): p.requires_grad_(True)
    for p in model.encoder.window.f[-1].parameters(): p.requires_grad_(True)

def outcome_loss(logit, label, pos_weight):
    return torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), label.float(), pos_weight=torch.tensor(pos_weight, device=logit.device))
