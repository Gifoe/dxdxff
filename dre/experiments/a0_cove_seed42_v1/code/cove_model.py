"""Original shared backbone first; only 28 bounded input correction parameters."""
import numpy as np
import torch
from torch import nn

def constructor(original,arm,mp,stats,permutation,bank=None):
 assert arm in ['B0','B1','B2','B3']
 class COVE(nn.Module):
  def __init__(self,seed=42):
   super().__init__();self.network=original(seed).network;self.arm=arm
   if arm!='B0':self.alpha=nn.Parameter(torch.zeros(14));self.beta=nn.Parameter(torch.zeros(14))
   for key in ['abs','delta','ratio']:self.register_buffer(key+'_idx',torch.tensor([q[key] for q in mp['triples']],dtype=torch.long))
   self.register_buffer('dup_to',torch.tensor([q[0] for q in mp['duplicates']],dtype=torch.long));self.register_buffer('dup_from',torch.tensor([q[1] for q in mp['duplicates']],dtype=torch.long))
   self.register_buffer('permutation',torch.as_tensor(permutation,dtype=torch.long))
   for key in ['Dcoef','Dintercept','Rcoef','Rintercept']:self.register_buffer(key,torch.tensor(np.asarray([q[key] for q in stats['full']]),dtype=torch.float64))
   self.register_buffer('scale',torch.tensor(stats['scale'],dtype=torch.float64))
   if bank is not None:
    self.register_buffer('_indexed_x',torch.as_tensor(bank['x']),persistent=False);self.register_buffer('_indexed_e',torch.as_tensor(bank['e'],dtype=torch.float64),persistent=False)
   assert sum(p.numel() for p in self.parameters())==(8817 if arm=='B0' else 8845)

  def residuals(self,x):
   a=x[:,self.abs_idx].to(torch.float64);d=x[:,self.delta_idx].to(torch.float64);r=x[:,self.ratio_idx].to(torch.float64)
   ed=d-a*self.Dcoef-self.Dintercept;er=r-a*self.Rcoef[:,0]-d*self.Rcoef[:,1]-self.Rintercept
   return torch.stack([ed,er],dim=-1)/self.scale

  def transformed(self,x,e=None,intervention=None,group=None):
   if self.arm=='B0' or intervention=='M1_ZERO':return x
   e=self.residuals(x) if e is None else e;d=torch.tanh(e[:,:,0]);r=torch.tanh(e[:,:,1])
   if self.arm=='B1':qd,qr=d,r
   else:
    q=d*r[:,self.permutation if self.arm=='B2' or intervention=='M2_WRONG_PAIR' else torch.arange(14,device=x.device)];qd=qr=q
   ad=.25*torch.tanh(self.alpha);br=.25*torch.tanh(self.beta)
   if intervention=='M3_NO_DELTA_CORRECTION':ad=ad*0
   if intervention=='M4_NO_RATIO_CORRECTION':br=br*0
   if group is not None:
    keep=torch.ones(14,device=x.device);keep[group]=0;ad=ad*keep;br=br*keep
   out=x.clone();out[:,self.delta_idx]=x[:,self.delta_idx]+(qd*ad).to(x.dtype);out[:,self.ratio_idx]=x[:,self.ratio_idx]+(qr*br).to(x.dtype)
   out[:,self.dup_to]=out[:,self.dup_from];return out

  def forward(self,x,g=None,intervention=None,group=None):
   e=None
   if x.shape[1]==1:
    assert hasattr(self,'_indexed_x'),'Index adapter not available at target inference'
    ix=x[:,0].long();assert torch.equal(x[:,0],ix.to(x.dtype));x=self._indexed_x[ix];e=self._indexed_e[ix]
   assert x.shape[1]==88
   return self.network(self.transformed(x,e,intervention,group)).squeeze(-1)

  def regularizer(self,counts):return next(self.parameters()).new_zeros(())
 return COVE
