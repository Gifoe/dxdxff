"""Synthetic FIT-only unit checks for KD losses and deterministic pairs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from student_core import VARIANTS, _pairs, kd_loss


def main() -> None:
    t = np.array([-1.0, -.5, .2, 1.4], dtype=np.float32)
    a = np.array([-.5, -1.0, 1.0, .5], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    data = SimpleNamespace(fold=1, teacher={"synthetic": [{"t": t, "a": a, "y": y, "weight": 1.0}]},
                           patient_weight={"synthetic": 1.2})
    batch = {"subject_id": ["synthetic"], "channel_mask": torch.ones((1, 4), dtype=torch.bool),
             "labels_ez": torch.tensor(y[None], dtype=torch.float32)}
    first = _pairs(100, 1, 2, 3, "synthetic")
    second = _pairs(100, 1, 2, 3, "synthetic")
    assert len(first[0]) == 128 and np.array_equal(first[0], second[0])
    for variant in VARIANTS[1:]:
        logits = torch.tensor([[.4, -.8, .3, -.2]], requires_grad=True)
        loss = kd_loss(data, variant, 1.0, batch, logits, 2, 3)
        assert torch.isfinite(loss).item() and loss.item() >= -1e-6
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all().item()
        assert logits.grad.abs().sum().item() > 0
    print("STUDENT_KD_UNIT_PASS")


if __name__ == "__main__":
    main()
