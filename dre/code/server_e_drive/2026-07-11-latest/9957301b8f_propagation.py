from __future__ import annotations
import math,torch
from torch import nn
import torch.nn.functional as F
VDR_SIDE_FEATURE_INDICES=(0,1,6,7)
DESCRIPTOR_DIM=199
class PropagationEscape(nn.Module):
 def __init__(self,enabled=True,descriptor_dim=DESCRIPTOR_DIM,use_side_features=True):
  super().__init__();self.enabled=enabled;self.descriptor_dim=descriptor_dim;self.use_side_features=use_side_features
  if enabled:self.scorer=nn.Sequential(nn.Linear(descriptor_dim,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,1))
 def forward(self,r,labels,side,mask):
  ez=labels==0;nez=labels==1
  if not self.enabled or not ez.any() or not nez.any():return None
  if side.shape[-1]<=max(VDR_SIDE_FEATURE_INDICES):raise ValueError(f'side feature dimension {side.shape[-1]} is too small for indices {VDR_SIDE_FEATURE_INDICES}')
  ez_pre=r['pre'][ez].mean(0);ez_on=r['onset'][ez].mean(0);ez_sp=r['spread'][ez].mean(0);ez_tau=r['tau'][ez].mean();ez_curve_members=r['curve'][ez];ez_curve=ez_curve_members.mean(0);ez_self=torch.nan_to_num(F.cosine_similarity(ez_curve_members,ez_curve.unsqueeze(0),dim=1)).mean()
  ez_side_per=(side[ez]*mask[ez].unsqueeze(-1)).sum(1)/mask[ez].sum(1).clamp_min(1).unsqueeze(-1);ez_side=ez_side_per.mean(0)[list(VDR_SIDE_FEATURE_INDICES)] if self.use_side_features else ez_pre.new_zeros(4);zero=torch.zeros_like(ez_pre);ez_desc=torch.cat([ez_pre,ez_on-ez_pre,ez_sp-ez_on,ez_sp-ez_pre,zero,zero,ez_tau.new_zeros(1),ez_self.reshape(1),ez_tau.new_zeros(1),ez_side]);assert ez_desc.shape==(DESCRIPTOR_DIM,)
  curves=r['curve'][nez];scores=[]
  for lag in range(5):
   a=ez_curve[:-lag or None];b=curves[:,lag:];scores.append(torch.nan_to_num(F.cosine_similarity(b,a.unsqueeze(0),dim=1)))
  corr=torch.stack(scores,1);lagw=torch.softmax(corr,1);strength=(lagw*corr).sum(1);delay=(lagw*torch.arange(5,device=corr.device,dtype=corr.dtype)).sum(1);zpre=r['pre'][nez];zon=r['onset'][nez];zsp=r['spread'][nez];side_mean=(side[nez]*mask[nez].unsqueeze(-1)).sum(1)/mask[nez].sum(1).clamp_min(1).unsqueeze(-1);nez_tau=r['tau'][nez];tau_diff=nez_tau-ez_tau;chosen_side=side_mean[:,list(VDR_SIDE_FEATURE_INDICES)] if self.use_side_features else zpre.new_zeros((len(zpre),4));nez_desc=torch.cat([zpre,zon-zpre,zsp-zon,zsp-zpre,zon-ez_on,zsp-ez_sp,tau_diff[:,None],strength[:,None],delay[:,None],chosen_side],1);assert nez_desc.shape[1]==DESCRIPTOR_DIM
  # The scorer is allowed to run under bf16 autocast, but ranking, probability
  # statistics and quantiles are deliberately float32.  torch.quantile rejects
  # bf16 tensors on CUDA/Windows, and these reductions are numerically sensitive.
  logits=self.scorer(nez_desc).squeeze(-1).float();n=len(logits);k=n if n<4 else min(12,max(4,math.ceil(.1*n)));idx=torch.topk(logits,k).indices;selected=torch.zeros(n,dtype=torch.bool,device=logits.device);selected[idx]=True;p=torch.sigmoid(logits);top=p[idx];entropy=-(torch.softmax(logits[idx],0)*torch.log_softmax(logits[idx],0)).sum()/max(math.log(max(k,2)),1e-6);stats=torch.stack([top.mean(),top.max(),torch.quantile(p,.9),p.mean(),p.std(unbiased=False),p.mean(),nez_tau[idx].float().min(),nez_tau.float().mean()-ez_tau.float(),entropy,logits.new_tensor(k/n)])
  return {'descriptor':nez_desc,'nez_descriptor':nez_desc,'ez_descriptor':ez_desc,'logits':logits,'probability':p,'selected':selected,'tau':nez_tau,'nez_tau':nez_tau,'ez_tau':ez_tau,'strength':strength,'delay':delay,'stats':torch.nan_to_num(stats),'nez_indices':torch.where(nez)[0],'ez_indices':torch.where(ez)[0],'ez_self_similarity':ez_self}
