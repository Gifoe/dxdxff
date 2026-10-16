"""Exact Raw-only forward; no physiological arrays or D1 are accepted."""
import torch
from torch import nn


class PureRaw(nn.Module):
    def __init__(self, compact=False):
        super().__init__()
        self.compact = compact
        self.cnn = nn.Sequential(nn.Conv1d(1, 12, 9, 4), nn.GELU(),
                                 nn.Conv1d(12, 24, 7, 4), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(96 if compact else 160, 64), nn.GELU(), nn.Linear(64, 1))

    def load_reference(self, state):
        expected = {'keep32', 'cnn.0.weight', 'cnn.0.bias', 'cnn.2.weight', 'cnn.2.bias',
                    'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias'}
        assert set(state) == expected and state['keep32'].shape == (32,)
        assert torch.count_nonzero(state['keep32']) == 0
        mapped = {k: v for k, v in state.items() if k != 'keep32'}
        if self.compact:
            mapped['head.0.weight'] = mapped['head.0.weight'][:, :96].clone()
        self.load_state_dict(mapped, strict=True)

    def normalized_windows(self, raw, mask, times):
        x = raw[mask]
        return ((x - x.mean(-1, keepdim=True)) /
                x.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6)).clamp(-8, 8)

    def aggregate(self, emb, mask, times):
        # Preserve the historical arithmetic, including its constant-coordinate
        # handling, rather than changing the reference while compacting its input.
        summaries, available = [], []
        for s in range(len(emb)):
            pre = mask[s] & (times[s][None] < 0)
            count, present = pre.sum(1), mask[s].any(1)
            assert (count[present] > 0).all()
            reference = (emb[s] * pre[..., None]).sum(1) / count.clamp_min(1)[:, None]
            latent = emb[s] - reference[:, None]
            stages, flags = [], []
            for low, high in [(0, 8), (8, 20)]:
                use = mask[s] & ((times[s] >= low) & (times[s] < high))[None]
                n = use.sum(1)
                stages.append((latent * use[..., None]).sum(1) / n.clamp_min(1)[:, None])
                flags.append(n > 0)
            summaries.append(torch.stack(stages, 1))
            available.append(torch.stack(flags, 1))
        summary = torch.stack(summaries)
        valid = torch.stack(available)[..., None].to(emb.dtype)
        n = valid.sum(0)
        assert (n > 0).all()
        mean = (summary * valid).sum(0) / n
        variance = ((summary - mean[None]).square() * valid).sum(0) / n
        sd = torch.where(variance > 0, variance.clamp_min(1e-30).sqrt(), torch.zeros_like(variance))
        vector = torch.stack([mean, sd], 2).reshape(emb.shape[1], 96).double()
        scale = vector.std(0, unbiased=False)
        return ((vector - vector.mean(0)) /
                torch.where(scale > 1e-8, scale, torch.ones_like(scale))).to(emb.dtype)

    def classify(self, z):
        if not self.compact:
            z = torch.cat([z, z.new_zeros(len(z), 64)], -1)
        return self.head(z).squeeze(-1)

    def forward(self, raw, mask, times):
        x = self.normalized_windows(raw, mask, times)
        h = torch.cat([self.cnn(q[:, None]).flatten(1) for q in x.split(4096)])
        dense = raw.new_zeros((*mask.shape, 24))
        dense[mask] = h
        return self.classify(self.aggregate(dense, mask, times))
