"""SSS-inspired PatchTST encoding, patient-channel two-level MIL, D2 fusion.

Reference: xmootoo/sss-official, commit 00c5ee476df51b5f300ca7264ee818546a7d9188.
Independent minimal adaptation, not an exact published SSS reproduction.
"""
import math

import torch
from torch import nn


class SourceA0(nn.Module):
    """Exact archived 9,221-parameter source-conditioned A0 topology."""
    def __init__(self, seed=42):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(88), nn.Linear(88, 96),
                                     nn.GELU(), nn.Dropout(.15), nn.Linear(96, 1))
        gen = torch.Generator().manual_seed(42 + 7000 + seed)
        self.V = nn.Parameter(.01 * torch.randn(4, 96, generator=gen))
        self.u = nn.Parameter(torch.zeros(4, 4))
        self.b = nn.Parameter(torch.zeros(4))

    def head(self, h, g):
        base = self.network[4](h).squeeze(-1)
        idx = g.clamp(0, 3)
        delta = ((h @ self.V.T) * self.u[idx]).sum(-1) + self.b[idx]
        return base + torch.where((g >= 0) & (g < 4), delta, torch.zeros_like(delta))

    def forward(self, x, g):
        return self.head(self.network[:4](x), g)

    def regularizer(self, counts):
        weights = (counts.sum() / 4) / counts.clamp(min=1)
        return self.V.square().mean() + ((self.u.square().mean(1) + self.b.square()) * weights).mean()


class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(32, 32)
        # No endpoint replication: 31 measured-sample patches, not 32.
        t = torch.arange(31).float()[:, None]
        f = torch.exp(torch.arange(0, 32, 2).float() * (-math.log(10000) / 32))
        pos = torch.zeros(31, 32)
        pos[:, 0::2], pos[:, 1::2] = torch.sin(t * f), torch.cos(t * f)
        self.register_buffer('position', pos)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(32, 4, 128, .10, activation='gelu',
                                       batch_first=True, norm_first=False) for _ in range(2)])
        self.aux = nn.Linear(2, 32)

    def forward(self, wave, aux):
        assert wave.ndim == 3 and wave.shape[1:] == (1, 512)
        z = self.patch(wave[:, 0].unfold(-1, 32, 16)) + self.position
        for layer in self.layers:
            z = layer(z)
        return z.mean(1) + self.aux(aux)


class RawMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TemporalEncoder()
        self.attention = nn.Sequential(nn.Linear(33, 32), nn.Tanh(), nn.Linear(32, 1))

    def pool(self, z, mask, time):
        logits = self.attention(torch.cat([z, time.unsqueeze(-1)], -1)).squeeze(-1)
        any_valid = mask.any(-1, keepdim=True)
        safe = logits.masked_fill(~mask, -torch.inf)
        safe = torch.where(any_valid, safe, torch.zeros_like(safe))
        weight = torch.softmax(safe, -1) * mask
        seizure = (weight.unsqueeze(-1) * z).sum(-2)
        sm = mask.any(-1)
        channel = (seizure * sm.unsqueeze(-1)).sum(1) / sm.sum(1).clamp(min=1).unsqueeze(-1)
        return channel, sm.any(-1), weight

    def forward(self, wave, aux, mask, time, diagnostics=False):
        # Only measured windows enter encoder; padding neither informs BN nor attention.
        shape = mask.shape
        flat = mask.flatten()
        indices = flat.nonzero().squeeze(-1)
        z = torch.zeros((*shape, 32), device=wave.device, dtype=wave.dtype)
        if len(indices):
            encoded = self.encoder(wave.reshape(-1, 1, 512)[indices], aux.reshape(-1, 2)[indices])
            z = z.reshape(-1, 32).index_copy(0, indices, encoded).reshape(*shape, 32)
        u, available, weight = self.pool(z, mask, time)
        if diagnostics:
            return u, available, {'attention': weight, 'window_embedding': z}
        return u, available


class SSSMIL(nn.Module):
    def __init__(self, arm, seed=42):
        super().__init__()
        assert arm in ['S0', 'S1', 'S2']
        self.arm = arm
        # Identical S1/S2 RNG construction order and initial tensors.
        self.raw = RawMIL()
        if arm == 'S0':
            self.output = nn.Linear(32, 1)
        else:
            self.d2 = SourceA0(seed)
            self.project = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 96))
            self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, wave, aux, mask, time, x=None, g=None, disable=False, diagnostics=False):
        u, available, detail = self.raw(wave, aux, mask, time, True)
        if self.arm == 'S0':
            logit = self.output(u).squeeze(-1)
            detail.update(raw=u, available=available)
        else:
            h = self.d2.network[:4](x)
            contribution = self.gamma * self.project(u) * available.unsqueeze(-1)
            if disable:
                contribution = contribution * 0
            logit = self.d2.head(h + contribution, g)
            detail.update(raw=u, available=available, engineered=h, contribution=contribution)
        return (logit, detail) if diagnostics else logit
