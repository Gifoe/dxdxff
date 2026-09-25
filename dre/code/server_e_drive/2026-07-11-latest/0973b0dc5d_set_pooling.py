from __future__ import annotations
import math,torch
def normalized_lse(h):return torch.logsumexp(h.float(),0)-math.log(h.shape[0])
def dual_statistics(h):
 if h.ndim!=2 or h.shape[0]<1:raise ValueError('set must be non-empty [N,48]')
 h=h.float();mean=h.mean(0);std=h.std(0,unbiased=False) if len(h)>1 else torch.zeros_like(mean);lse=normalized_lse(h)
 return torch.cat([mean,std,lse]),{'mean':mean,'std':std,'lse':lse}
