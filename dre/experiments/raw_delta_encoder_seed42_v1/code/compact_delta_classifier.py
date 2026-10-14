"""Duplicate-aware algebraic reparameterization, never native LayerNorm14."""
import copy
import numpy as np
import torch
from torch import nn

def records_from_manifest(manifest):
 cols=manifest['columns'];assert len(cols)==88
 selected=[q for q in cols if q['view']=='delta'];assert len(selected)==26
 unique=[q for q in selected if not q['constant_zero_in_frozen_export'] and 'duplicate_of' not in q]
 assert len(unique)==14
 records=[]
 for q in unique:
  copies=[q['index']]+[r['index'] for r in selected if r.get('duplicate_of')==q['index']]
  records.append(dict(q,copies=copies,multiplicity=len(copies)))
 assert sum(r['multiplicity'] for r in records)==22
 return records

def fold_weights(network,records):
 ln,linear=network[0],network[1];assert ln.eps==1e-5
 w=linear.weight.detach().double();a=w*ln.weight.detach().double()
 active=[i for r in records for i in r['copies']];zero=[i for i in range(88) if i not in active];z=a[:,zero].sum(1)
 folded=torch.stack([a[:,r['copies']].sum(1)-z*r['multiplicity']/len(zero) for r in records],1)
 return folded,linear.bias.detach().double()+w@ln.bias.detach().double()

class CompactDelta(nn.Module):
 def __init__(self,network,records):
  super().__init__();self.register_buffer('indices',torch.tensor([r['index'] for r in records]));self.register_buffer('multiplicities',torch.tensor([r['multiplicity'] for r in records],dtype=torch.float64));self.n0=88-sum(r['multiplicity'] for r in records)
  w,b=fold_weights(network,records);self.first=nn.Linear(14,96,dtype=network[1].weight.dtype)
  with torch.no_grad():self.first.weight.copy_(w);self.first.bias.copy_(b)
  self.last=copy.deepcopy(network[4]);self.dropout=nn.Dropout(.15)
  assert sum(p.numel() for p in self.parameters())==1537
 def normalized(self,x):
  m=self.multiplicities.to(x.dtype);mu=(x*m).sum(-1,keepdim=True)/88
  var=(((x-mu).square()*m).sum(-1,keepdim=True)+self.n0*mu.square())/88
  return (x-mu)/torch.sqrt(var+1e-5)
 def forward(self,x,g=None):
  if x.shape[-1]==88:x=x.index_select(-1,self.indices)
  assert x.shape[-1]==14
  return self.last(self.dropout(torch.nn.functional.gelu(self.first(self.normalized(x))))).squeeze(-1)
 def regularizer(self,counts):return self.first.weight.new_zeros(())

def inflate(x,records):
 out=x.new_zeros((len(x),88))
 for j,r in enumerate(records):out[:,r['copies']]=x[:,j,None]
 return out
