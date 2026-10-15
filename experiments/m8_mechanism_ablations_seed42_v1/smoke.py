"""Synthetic CPU/GPU equality and isolated-mechanism checks; no cohort outputs."""
import copy
import numpy as np
import scipy
import sklearn
import torch
from torch import nn
import runner as R

torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
audits=[]
for device in ['cpu','cuda']:
    torch.manual_seed(42)
    base=nn.Sequential(nn.Linear(14,96),nn.GELU(),nn.Linear(96,1))
    original=R.P3.MorphCorrection(copy.deepcopy(base)).to(device)
    full=R.M8Mechanism(copy.deepcopy(base)).to(device);full.load_state_dict(original.state_dict())
    # Exercise upstream nonzero gradients instead of the zero-output identity case only.
    with torch.no_grad():
        original.head[-1].weight.normal_(std=.01);full.load_state_dict(original.state_dict())
    raw=torch.randn(2,6,5,500,device=device);mask=torch.ones(2,6,5,device=device,dtype=torch.bool)
    times=torch.tensor([[-4.,-2.,2.,6.,12.]]*2,device=device)
    delta=torch.randn(2,6,5,7,device=device);scale=torch.ones(7,device=device)
    old=original.features(raw,mask,times,delta,scale)[0];new=full.features(raw,mask,times,delta,scale)[0]
    error=float((old-new).abs().max().detach());assert error==0
    original.head(old).square().sum().backward();full.head(new).square().sum().backward()
    gradient_error=0.;old_grad={n:p.grad.detach().clone() for n,p in original.named_parameters() if p.grad is not None}
    for (name,p),(name2,q) in zip(original.named_parameters(),full.named_parameters()):
        assert name==name2
        if p.grad is not None:
            assert q.grad is not None;gradient_error=max(gradient_error,float((p.grad-q.grad).abs().max()))
    original.zero_grad(set_to_none=True)
    repeat=original.features(raw,mask,times,delta,scale)[0]
    original.head(repeat).square().sum().backward()
    self_error=max(float((p.grad-old_grad[n]).abs().max()) for n,p in original.named_parameters() if n in old_grad)
    if device=='cpu':assert gradient_error==0
    else:
        assert gradient_error<=max(2*self_error,1e-6)
        for n,p in full.named_parameters():
            if n in old_grad:torch.testing.assert_close(p.grad,old_grad[n],rtol=1e-5,atol=1e-6)
    print('GRADIENT_EQUIVALENCE',device,'new_vs_original',gradient_error,'original_self_replay',self_error,flush=True)
    identity=R.M8Mechanism(copy.deepcopy(base),identity_gate=True).to(device)
    assert sum(p.numel() for p in identity.parameters() if p.requires_grad)==8433
    with torch.inference_mode():
        a=identity.features(raw,mask,times,delta,scale)[0];b=identity.features(raw,mask,times,delta*0,scale)[0]
        assert torch.equal(a,b)
        nopr=R.M8Mechanism(copy.deepcopy(base),raw_pr=False).to(device)
        a=nopr.features(raw,mask,times,delta,scale)[0];b=nopr.features(raw[:,:-1],mask[:,:-1],times,delta[:,:-1],scale)[0]
        assert torch.allclose(a[:-1],b,atol=1e-5,rtol=0)
        nopre=R.M8Mechanism(copy.deepcopy(base),pre_reference=False).to(device)
        changed=raw.clone();changed[:,:,:2]=torch.randn_like(changed[:,:,:2])
        a=nopre.features(raw,mask,times,delta,scale)[0];b=nopre.features(changed,mask,times,delta,scale)[0]
        assert torch.equal(a,b)
        assert not torch.equal(full.features(raw,mask,times,delta,scale)[0],full.features(changed,mask,times,delta,scale)[0])
    audits.append(dict(device=device,status='PASS',feature_parity_max_error=error,gradient_max_error=gradient_error,original_self_replay_gradient_max_error=self_error,
        identity_gate_delta_independent=True,no_raw_PR_channel_subset_independent=True,no_raw_pre_reference_pre_waveform_independent=True))
    print('SYNTHETIC_MECHANISM_PARITY_PASS',device,flush=True)
R.OUT.mkdir(exist_ok=True);R.write(R.OUT/'public/SYNTHETIC_SMOKE_AUDIT.json',dict(status='PASS',tests=audits))
