from __future__ import annotations

import pytest
import torch

from outcome_hifos.models.unbalanced_ot import UnbalancedSinkhornTransport


def _solver() -> UnbalancedSinkhornTransport:
    return UnbalancedSinkhornTransport(epsilon=0.08, tau=0.8, iterations=80, tolerance=1e-7)


def test_identical_measure_cost_is_lower_than_crossed_measure_cost() -> None:
    mass = torch.tensor([[0.7, 0.3]], dtype=torch.float64)
    identical_cost = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]], dtype=torch.float64)
    crossed_cost = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float64)
    identical = _solver()(mass, mass, identical_cost)
    crossed = _solver()(mass, mass, crossed_cost)
    assert identical.cost.item() < crossed.cost.item()
    assert (identical.plan >= 0).all()


def test_small_and_zero_mass_are_finite_with_backward() -> None:
    source = torch.tensor([[1e-12, 0.8, 0.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.4, 1e-12, 0.0]], dtype=torch.float64, requires_grad=True)
    cost = torch.tensor([[[0.1, 1.0, 3.0], [1.0, 0.1, 3.0], [3.0, 3.0, 0.0]]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    result = _solver()(source, target, cost, source_mask=mask, target_mask=mask)
    assert torch.isfinite(result.cost).all()
    assert torch.isfinite(result.plan).all()
    assert torch.count_nonzero(result.plan[:, 2, :]).item() == 0
    assert torch.count_nonzero(result.plan[:, :, 2]).item() == 0
    result.cost.sum().backward()
    assert torch.isfinite(source.grad).all()
    assert torch.isfinite(target.grad).all()


def test_source_target_swap_has_symmetric_distance_for_symmetric_cost() -> None:
    source = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    target = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
    cost = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64)
    forward = _solver()(source, target, cost)
    backward = _solver()(target, source, cost.transpose(-1, -2))
    torch.testing.assert_close(forward.cost, backward.cost, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_results_are_close() -> None:
    source = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    target = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    cost = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
    cpu = _solver()(source, target, cost)
    gpu = _solver().cuda()(source.cuda(), target.cuda(), cost.cuda())
    torch.testing.assert_close(cpu.cost, gpu.cost.cpu(), atol=2e-4, rtol=2e-4)
