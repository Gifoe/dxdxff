"""Class-support and analytic loss invariance checks before any new training."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from common import EXPERIMENT, assert_no_outer_loader, build_fold, ensure_source, make_experiment, write_json
from losses import balanced_patient_losses


def score(logits, ez):
    logits = torch.as_tensor([logits], dtype=torch.float64)
    ez = torch.as_tensor([ez], dtype=torch.float64)
    nez = 1 - ez
    mask = torch.ones_like(ez, dtype=torch.bool)
    return balanced_patient_losses(logits, nez, ez, mask).mean()


def main():
    ensure_source()
    exp = make_experiment()
    folds = []
    for split in exp.outer_splits:
        fold, train_set, _, val_loader, test_loader, _ = build_fold(exp, split, "validation")
        assert_no_outer_loader(test_loader)
        item = {"fold": fold, "n_fit": len(train_set), "n_validation": len(val_loader.dataset), "single_class_fit": 0, "single_class_validation": 0}
        for name, dataset in (("fit", train_set), ("validation", val_loader.dataset)):
            for example in dataset.patient_examples:
                mask = np.asarray(example["channel_mask"], dtype=bool)
                classes = set(np.asarray(example["labels_ez"])[mask].tolist())
                item[f"single_class_{name}"] += int(classes != {0.0, 1.0})
        folds.append(item)
    if any(row["single_class_fit"] or row["single_class_validation"] or row["n_validation"] != 13 for row in folds):
        raise RuntimeError("Single-class patient or changed validation membership; stop before training")
    e, n = 0.7, -0.4
    expected = 0.5 * F.binary_cross_entropy_with_logits(torch.tensor(e), torch.tensor(0.0)) + 0.5 * F.binary_cross_entropy_with_logits(torch.tensor(n), torch.tensor(1.0))
    original = score([e, n], [1, 0])
    tests = {
        "equal_class_mass_1v1": bool(torch.allclose(original, expected.double(), atol=1e-7)),
        "duplicate_NEZ_10x_invariant": bool(torch.allclose(original, score([e] + [n] * 10, [1] + [0] * 10), atol=1e-12)),
        "duplicate_EZ_10x_invariant": bool(torch.allclose(original, score([e] * 10 + [n], [1] * 10 + [0]), atol=1e-12)),
        "equal_patient_weight": bool(torch.allclose((score([e,n],[1,0]) + score([e]*10+[n],[1]*10+[0]))/2, original, atol=1e-12)),
        "channel_permutation_invariant": bool(torch.allclose(original, score([n,e],[0,1]), atol=1e-12)),
    }
    x = torch.tensor([[e, n]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    balanced_patient_losses(x, 1-y, y, torch.ones_like(y, dtype=torch.bool)).mean().backward()
    tests["finite_gradient"] = bool(torch.isfinite(x.grad).all())
    payload = {"pass": all(tests.values()), "tests": tests, "class_support": folds, "outer_test_accessed": False}
    write_json(EXPERIMENT / "CLASS_BALANCED_LOSS_UNIT_TEST.json", payload)
    if not payload["pass"]:
        raise RuntimeError("Class-balanced loss unit test failed")
    print(json.dumps({"pass": True, "n_folds": len(folds)}), flush=True)


if __name__ == "__main__":
    main()
