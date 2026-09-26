"""Fail-closed synthetic tests of the new objective before any model training."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
from exp_ez_hybrid import _masked_bce_loss  # noqa: E402
from objectives import beta_at_epoch, patient_equal_weighted_bce_loss, patient_soft_macro_f1_loss, per_patient_weighted_bce  # noqa: E402


def require(name: str, condition: bool, details: dict, results: list[dict]) -> None:
    results.append({"test": name, "pass": bool(condition), **details})
    if not condition:
        raise RuntimeError(f"Objective unit test failed: {name}: {details}")


def main() -> None:
    torch.manual_seed(42)
    results: list[dict] = []
    labels_ez = torch.tensor([[1., 0., 0., 1., 0., 1.]])
    labels_nez = 1.0 - labels_ez
    logits = torch.tensor([[-0.8, 0.3, 1.2, -0.1, -0.4, 0.7]], requires_grad=True)
    mask = torch.tensor([[True, True, True, True, True, False]])
    a1 = patient_equal_weighted_bce_loss(logits, labels_nez, labels_ez, mask)
    a0 = _masked_bce_loss(logits, labels_nez, labels_ez, mask,
                          class_weight_mode="ez_negative", ez_negative_weight=torch.tensor(2.0))
    require("batch_one_equals_R0_masked_BCE", torch.allclose(a0, a1, rtol=0, atol=1e-7),
            {"R0": float(a0.detach()), "A1": float(a1.detach())}, results)

    original, _ = per_patient_weighted_bce(logits.detach(), labels_nez, labels_ez, mask)
    duplicate = patient_equal_weighted_bce_loss(
        logits[:, :5].repeat(1, 2), labels_nez[:, :5].repeat(1, 2),
        labels_ez[:, :5].repeat(1, 2), torch.ones((1, 10), dtype=torch.bool))
    require("duplicating_all_valid_channels_preserves_patient_loss",
            torch.allclose(original[0], duplicate, rtol=0, atol=1e-7),
            {"original": float(original[0]), "duplicated": float(duplicate)}, results)

    lengths = (50, 150)
    two_logits = torch.linspace(-2, 2, 300).reshape(2, 150).clone().detach().requires_grad_(True)
    two_ez = torch.zeros((2, 150))
    two_ez[0, :50:4] = 1
    two_ez[1, :150:5] = 1
    two_nez = 1 - two_ez
    two_mask = torch.zeros((2, 150), dtype=torch.bool)
    two_mask[0, :lengths[0]] = True
    two_mask[1, :lengths[1]] = True
    individual, active = per_patient_weighted_bce(two_logits, two_nez, two_ez, two_mask)
    combined = patient_equal_weighted_bce_loss(two_logits, two_nez, two_ez, two_mask)
    require("50_vs_150_channels_is_equal_patient_mean", active.tolist() == [True, True] and
            torch.allclose(combined, individual.mean(), rtol=0, atol=1e-7),
            {"A": float(individual[0]), "B": float(individual[1]), "A1": float(combined)}, results)

    permutation = torch.randperm(150)
    permuted = patient_equal_weighted_bce_loss(two_logits[:, permutation], two_nez[:, permutation],
                                                two_ez[:, permutation], two_mask[:, permutation])
    require("channel_permutation_invariant", torch.allclose(combined, permuted, rtol=0, atol=1e-7),
            {"original": float(combined), "permuted": float(permuted)}, results)

    soft_loss, soft_nez, soft_ez = patient_soft_macro_f1_loss(two_logits, two_nez, two_ez, two_mask)
    (combined + 0.1 * soft_loss).backward()
    require("finite_A1_A2_gradient", two_logits.grad is not None and torch.isfinite(two_logits.grad).all().item(),
            {"soft_loss": float(soft_loss), "soft_nez_f1": float(soft_nez), "soft_ez_f1": float(soft_ez)}, results)
    require("locked_beta_schedule", all(math.isclose(beta_at_epoch(epoch), expected, abs_tol=1e-12)
            for epoch, expected in zip((1, 5, 6, 7, 8, 9, 10, 11, 30),
                                       (0.0, 0.0, 0.0, 0.025, 0.05, 0.075, 0.1, 0.1, 0.1), strict=True)),
            {"epochs_1_5_6_7_8_9_10_11_30": [beta_at_epoch(i) for i in (1, 5, 6, 7, 8, 9, 10, 11, 30)]}, results)

    payload = {"pass": True, "tests": results}
    (EXPERIMENT / "A1_LOSS_UNIT_TEST.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
