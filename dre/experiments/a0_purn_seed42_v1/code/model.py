"""One normalization replacement; exact original backbone and affine budget."""
import torch
from torch import nn


class UniqueFeatureNorm(nn.Module):
 def __init__(self,num_features=88,stat_indices=None,centered=False,eps=1e-5,elementwise_affine=True):
  super().__init__();assert num_features==88 and eps==1e-5 and elementwise_affine
  indices=list(range(88)) if stat_indices is None else list(stat_indices)
  assert len(indices) in [56,88] and len(set(indices))==len(indices) and all(0<=i<88 for i in indices)
  self.register_buffer('stat_indices',torch.tensor(indices,dtype=torch.long))
  self.centered=bool(centered);self.eps=eps
  self.weight=nn.Parameter(torch.ones(88));self.bias=nn.Parameter(torch.zeros(88))

 def forward(self,x):
  assert x.shape[-1]==88 and x.is_floating_point()
  u=x.to(torch.float64);v=u.index_select(-1,self.stat_indices)
  if self.centered:
   mean=v.mean(-1,keepdim=True);variance=(v-mean).square().mean(-1,keepdim=True)
   z=(u-mean)/torch.sqrt(variance+self.eps)
  else:z=u/torch.sqrt(v.square().mean(-1,keepdim=True)+self.eps)
  return (z*self.weight.to(torch.float64)+self.bias.to(torch.float64)).to(x.dtype)


def constructor(original,arm,mapping):
 assert arm in ['B0','B1','B2','B3']
 class Model(nn.Module):
  def __init__(self,seed=42):
   super().__init__();self.arm=arm;self.network=original(seed).network
   assert isinstance(self.network[0],nn.LayerNorm) and self.network[0].eps==1e-5
   if arm!='B0':
    old=self.network[0];indices=list(range(88)) if arm=='B1' else mapping['stat_indices']
    norm=UniqueFeatureNorm(stat_indices=indices,centered=arm=='B2')
    with torch.no_grad():norm.weight.copy_(old.weight);norm.bias.copy_(old.bias)
    self.network[0]=norm
   assert sum(p.numel() for p in self.parameters())==8817
   assert sum(p.numel() for p in self.network[0].parameters())==176

  def forward(self,x,g=None):return self.network(x).squeeze(-1)
  def regularizer(self,counts):return next(self.parameters()).new_zeros(())
 return Model
