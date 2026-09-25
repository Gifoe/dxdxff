from __future__ import annotations
import torch
from torch import nn

class PostOnsetTemporalEncoder(nn.Module):
    def __init__(self,model_dim=32,early_pool_frac=.25,lse_pool_tau=1.):
        super().__init__(); self.frac=early_pool_frac; self.tau=lse_pool_tau; self.proj=nn.Linear(3*model_dim,model_dim)
    def forward(self,x,seizure_channel_mask,window_mask,window_centers):
        b,s,t,c,d=x.shape; valid=seizure_channel_mask[:,:,None,:]&window_mask[:,:,:,None]&torch.isfinite(x).all(-1); x=torch.nan_to_num(x); w=valid.float(); count=w.sum(2).clamp_min(1)
        mean=(x*w[...,None]).sum(2)/count[...,None]
        masked=x.masked_fill(~valid[...,None],float('-inf')); lse=self.tau*torch.logsumexp(masked/self.tau,2)-self.tau*torch.log(count[...,None]); lse=torch.nan_to_num(lse)
        post=valid&(window_centers[:,:,:,None]>=0); fallback=(valid.any(2)&~post.any(2)); chosen=torch.zeros_like(valid)
        for bi in range(b):
          for si in range(s):
            for ci in range(c):
              source=torch.where(post[bi,si,:,ci])[0]
              if not source.numel(): source=torch.where(valid[bi,si,:,ci])[0]
              if source.numel(): chosen[bi,si,source[:max(1,int(torch.ceil(torch.tensor(self.frac*source.numel())).item()))],ci]=True
        early=(x*chosen[...,None]).sum(2)/chosen.sum(2).clamp_min(1)[...,None]
        diagnostics={"post_onset_window_count_per_channel":post.sum(2),"pre_onset_window_count_per_channel":(valid&~post).sum(2),"post_onset_fallback_mask":fallback}
        return self.proj(torch.cat([mean,lse,early],-1))*seizure_channel_mask[...,None],chosen,diagnostics
__all__=["PostOnsetTemporalEncoder"]
