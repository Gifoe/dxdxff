from __future__ import annotations
import torch
def sparsemax(input,dim=-1):
    x=input.transpose(dim,-1); shape=x.shape; x=x.reshape(-1,shape[-1]); z,_=torch.sort(x,descending=True,dim=1); k=torch.arange(1,z.size(1)+1,device=x.device,dtype=x.dtype); support=1+k*z>z.cumsum(1); k_z=support.sum(1,keepdim=True).clamp_min(1); tau=(z.cumsum(1).gather(1,k_z-1)-1)/k_z; out=torch.clamp(x-tau,min=0); return out.reshape(shape).transpose(dim,-1)
