"""Read-only FIT replay of a finite-gradient failure; no epochs/results produced."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,r'D:\drechan_pooling_capacity_ablation_seed42_v1')
from common import *
binding,core,fac,records,rec,banks=setup(False);fold=1;arm='P2_TIME_ATTENTION';d=bank(fold,banks,arm);ss=scale(fold)
assert not (ROOT/'models_private'/arm/'seed42/fold1/LAST_PRIVATE.pt').exists()
inputs=store();m=make(core,fac,records,fold,arm,42);seed_all(42+1009*fold)
opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001)
tr=d['train'];pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))]
for ordinal,k in enumerate(np.random.default_rng(42+1009*fold+1).permutation(len(groups)),1):
    ii=groups[int(k)];pid=str(d['patient'][ii[0]]);channels=d['channel'][ii].astype(str).tolist()
    raw,mask,times,delta,_=inputs.get(pid,channels,'cuda')
    bad_times=int(((~torch.isfinite(times))[:,None,:]&~mask).sum())
    opt.zero_grad(set_to_none=True);logit,_=forward(m,d,ii,inputs,ss,fold,binding)
    loss=MODEL.objective(logit,torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32),pos)
    assert torch.isfinite(loss);loss.backward()
    bad=[n for n,p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    if bad:
        q=dict(status='REPRODUCED_FINITE_LOSS_NONFINITE_GRADIENT',masked_nonfinite_time_cells=bad_times,
               nonfinite_gradient_names=bad,formal_epochs_written=0,scientific_sources_unchanged=sources())
        write(ROOT/'GRADIENT_DIAGNOSIS_PRIVATE.json',q)
        print({k:v for k,v in q.items() if k!='scientific_sources_unchanged'},flush=True);break
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
else:raise AssertionError('Failure not reproduced; do not resume without diagnosis')
