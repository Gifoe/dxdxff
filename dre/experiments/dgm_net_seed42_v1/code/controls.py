"""Conditioning-only frozen diagnostics; deterministic and label independent."""
import hashlib
import json
import numpy as np
import torch

def permutations(patient, channels, plans, masks=None):
    output=[]; eligible=changed=singletons=0
    for s,rec in enumerate(plans):
        t=np.asarray(rec['times']); local=[]
        for c,channel in enumerate(channels):
            p=np.arange(len(t))
            for low, high in [(0,8),(8,20)]:
                genuine=np.ones(len(t),bool) if masks is None else masks[s,c]
                ix=np.flatnonzero((t>=low)&(t<high)&genuine)
                if len(ix)<2:
                    singletons += int(len(ix)==1)
                    continue
                seed=int.from_bytes(hashlib.sha256(json.dumps([42,str(patient),list(rec['compound_key']),str(channel),low,high],sort_keys=True).encode()).digest()[:8],'little')
                q=np.random.default_rng(seed).permutation(ix)
                if np.array_equal(q,ix):q=np.roll(q,1)
                p[ix]=q; eligible+=1; changed+=int(not np.array_equal(q,ix))
            local.append(p)
        output.append(np.stack(local))
    return output, dict(eligible=eligible,changed=changed,singletons=singletons)

def transform(delta, mask, times, mode='full', perms=None):
    assert not delta.requires_grad
    out=delta.clone()
    if mode=='static':
        for s in range(len(out)):
            for low,high in [(0,8),(8,20)]:
                m=mask[s]&((times[s]>=low)&(times[s]<high))[None]
                count=m.sum(1).clamp_min(1)[:,None]
                mean=(out[s]*m[...,None]).sum(1)/count
                out[s]=torch.where(m[...,None],mean[:,None],out[s])
    elif mode=='shuffle':
        assert perms is not None
        for s,p in enumerate(perms):
            p=torch.as_tensor(p,device=delta.device)
            assert p.shape==mask[s].shape
            moved=torch.gather(delta[s],1,p[...,None].expand(-1,-1,7))
            assert torch.equal(torch.gather(mask[s],1,p),mask[s]), 'Permutation crossed genuine support'
            out[s]=moved
    return out
