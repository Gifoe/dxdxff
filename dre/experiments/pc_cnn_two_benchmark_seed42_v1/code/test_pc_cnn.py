"""Fast topology/identity/gradient tests; no patient or outcome data."""

from __future__ import annotations

import torch
from torch import nn
from copy import deepcopy

from pc_cnn import PCCNN


class SmallOfficialTopology(nn.Module):
    """Official module names/call order with a cheap synthetic 32-D encoder."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, (3, 25), stride=(1, 25), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(16, 32, (3, 10), stride=(1, 10), padding=(1, 0)), nn.ReLU(),
        )
        self.cnn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(32, 32)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.LeakyReLU()
        self.fc_out = nn.Linear(16, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cnn(x)
        x = self.bn(self.relu(self.fc(x)))
        x = self.bn1(self.relu1(self.fc1(x)))
        return self.fc_out(x)


def test_exact_identity_and_trainable_projector():
    torch.manual_seed(42)
    pc = PCCNN(SmallOfficialTopology()).eval()
    image = torch.randn(1, 3, 1, 32, 500)
    descriptor = torch.randn(1, 3, 59, 36)
    mask = torch.ones(1, 3, dtype=torch.bool)
    available = torch.ones_like(descriptor)
    raw = pc.raw(image[0]).squeeze(-1)
    disabled = pc(image, descriptor, mask, available, physiology=False,
                  context=False)[0]
    initial = pc(image, descriptor, mask, available)[0]
    assert torch.equal(raw, disabled)
    assert torch.equal(raw, initial)
    assert all(abs(float(value.detach()) - 0.02) < 1e-7
               for value in pc.alphas().values())
    initial.sum().backward()
    assert pc.physiology[-1].weight.grad.abs().sum().item() > 0
    assert pc.context_out.weight.grad.abs().sum().item() > 0


def test_channel_permutation():
    torch.manual_seed(42)
    pc = PCCNN(SmallOfficialTopology()).eval()
    with torch.no_grad():
        pc.physiology[-1].weight.normal_(0, 0.01)
        pc.context_out.weight.normal_(0, 0.01)
    image = torch.randn(1, 3, 1, 32, 500)
    descriptor = torch.randn(1, 3, 59, 36)
    available = torch.ones_like(descriptor)
    valid = torch.ones(1, 3, dtype=torch.bool)
    order = torch.tensor([2, 0, 1])
    with torch.no_grad():
        a = pc(image, descriptor, valid, available)
        b = pc(image[:, order], descriptor[:, order], valid[:, order],
               available[:, order])
    assert torch.allclose(a[:, order], b, atol=1e-6, rtol=1e-6)


def test_streamed_record_identity_and_gradients():
    torch.manual_seed(42)
    pc = PCCNN(SmallOfficialTopology()).eval()
    waves = torch.randn(1, 3, 500)
    descriptor = torch.randn(1, 3, 59, 36)
    valid = torch.ones(1, 3, dtype=torch.bool)
    available = torch.ones_like(descriptor)

    def synthetic_spectrum(x, sampling_rate):
        assert sampling_rate == 250.0
        return x[:, None, None, :].expand(-1, 1, 32, -1).contiguous()

    image = synthetic_spectrum(waves[0], 250.0)[None]
    expected = pc(image, descriptor, valid, available)
    streamed = pc.forward_record(waves, 250.0, descriptor, valid, available,
                                 synthetic_spectrum, channel_chunk=2)
    assert torch.max(torch.abs(streamed - expected)).item() < 1e-6
    streamed.sum().backward()
    assert pc.physiology[-1].weight.grad.abs().sum().item() > 0
    assert pc.context_out.weight.grad.abs().sum().item() > 0


def test_reentrant_checkpoint_matches_uncheckpointed_gradients():
    """Check both frozen and trainable backbone cases without outcome data."""
    torch.manual_seed(42)
    def synthetic_spectrum(x, sampling_rate):
        return x[:, None, None, :].expand(-1, 1, 32, -1).contiguous()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        wave = torch.randn(1, 3, 500, device=device)
        descriptor = torch.randn(1, 3, 59, 36, device=device)
        valid = torch.ones(1, 3, dtype=torch.bool, device=device)
        available = torch.ones_like(descriptor)
        for frozen_backbone in (True, False):
            reference = PCCNN(SmallOfficialTopology()).to(device).train()
            candidate = deepcopy(reference)
            for model in (reference, candidate):
                model.set_batch_norm_running_state(True)
                if frozen_backbone:
                    for parameter in model.raw.parameters():
                        parameter.requires_grad_(False)
            plain = reference.forward_record(
                wave, 250.0, descriptor, valid, available,
                synthetic_spectrum, channel_chunk=2, checkpoint_backbone=False)
            checked = candidate.forward_record(
                wave, 250.0, descriptor, valid, available,
                synthetic_spectrum, channel_chunk=2, checkpoint_backbone=True)
            assert torch.equal(plain, checked)
            plain.square().sum().backward()
            checked.square().sum().backward()
            for (name, left), (_, right) in zip(reference.named_parameters(),
                                                candidate.named_parameters()):
                if not left.requires_grad:
                    continue
                assert left.grad is not None, name
                assert right.grad is not None, name
                assert torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-5), name
