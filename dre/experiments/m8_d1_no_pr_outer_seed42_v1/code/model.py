"""Original M8 mathematics, with only the final cross-channel Z made optional.

The default and only trained ablation has patient_relative=False. True is used
only to prove equality to the immutable original implementation before training.
"""
import torch
from direct_correction import MorphCorrection
from dgm_net import transform


class M8NoPR(MorphCorrection):
    def __init__(self, base, patient_relative=False):
        super().__init__(base)
        self.patient_relative = patient_relative

    def features(self, raw, mask, times, delta, scale, mode='full', perms=None):
        S, C, W, L = raw.shape
        flat = raw[mask]
        v = ((flat-flat.mean(-1, keepdim=True)) /
             flat.std(-1, unbiased=False, keepdim=True).clamp_min(1e-6)).clamp(-8, 8)
        emb = raw.new_zeros(S, C, W, 24)
        emb[mask] = torch.cat([self.cnn(q[:, None]).flatten(1) for q in v.split(4096)])
        phys = transform(delta, mask, times, mode, perms)
        u = (phys/scale).clamp(-8, 8)
        if mode.startswith('neutral_'):
            u[..., int(mode.split('_')[1])] = 0
        summaries, available, activity = [], [], []
        for s in range(S):
            pre = mask[s] & (times[s][None] < 0)
            count = pre.sum(1)
            present = mask[s].any(1)
            assert (count[present] > 0).all()
            hpre = (emb[s]*pre[..., None]).sum(1)/count.clamp_min(1)[:, None]
            latent = emb[s]-hpre[:, None]
            post = mask[s] & ((times[s] >= 0) & (times[s] < 20))[None]
            g = torch.ones_like(latent) if mode == 'identity' else self.gate(u[s])
            activity.append(g[post])
            guided = g*latent
            st, av = [], []
            for low, high in [(0, 8), (8, 20)]:
                select = mask[s] & ((times[s] >= low) & (times[s] < high))[None]
                n = select.sum(1)
                st.append((guided*select[..., None]).sum(1)/n.clamp_min(1)[:, None])
                av.append(n > 0)
            summaries.append(torch.stack(st, 1))
            available.append(torch.stack(av, 1))
        summary = torch.stack(summaries)
        valid = torch.stack(available)[..., None].to(raw.dtype)
        n = valid.sum(0)
        assert (n > 0).all()
        mu = (summary*valid).sum(0)/n
        var = ((summary-mu[None]).square()*valid).sum(0)/n
        sd = torch.where(var > 0, torch.sqrt(var.clamp_min(1e-30)), torch.zeros_like(var))
        vector = torch.stack([mu, sd], 2).reshape(C, 96).double()
        if self.patient_relative:
            std = vector.std(0, unbiased=False)
            vector = (vector-vector.mean(0))/torch.where(std > 1e-8, std, torch.ones_like(std))
        return vector.to(raw.dtype), torch.cat(activity)
