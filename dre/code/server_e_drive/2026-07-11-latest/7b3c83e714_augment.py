from __future__ import annotations
import torch

NAMES=("identity","scale","jitter","temporal_mask","channel_mask")
def augment_seizure(windows,mask,generator=None,choice=None):
    generator=generator or torch.Generator(device=windows.device);choice=choice or NAMES[int(torch.randint(len(NAMES),(),generator=generator))];x=windows.clone();m=mask.bool().clone()
    if choice=="scale":x=x*(.9+.2*torch.rand((x.shape[0],1,1),generator=generator,device=x.device))
    elif choice=="jitter":
        robust=(x-x.median(-1,keepdim=True).values).abs().median(-1).values*1.4826;std=.015*torch.rand((x.shape[0],1),generator=generator,device=x.device)*robust;x=x+torch.randn(x.shape,generator=generator,device=x.device,dtype=x.dtype)*std[...,None]
    elif choice=="temporal_mask":
        length=max(1,x.shape[-1]//10);start=int(torch.randint(x.shape[-1]-length+1,(),generator=generator));x[...,start:start+length]=0
    elif choice=="channel_mask" and m.any(1).sum()>2:
        drop=(torch.rand(x.shape[0],generator=generator,device=x.device)<.1)&m.any(1);keep=max(2,int(m.any(1).sum())-int(drop.sum()));
        if keep>=2:m[drop]=False;x[drop]=0
    elif choice!="identity":raise ValueError(choice)
    return x,m,choice
