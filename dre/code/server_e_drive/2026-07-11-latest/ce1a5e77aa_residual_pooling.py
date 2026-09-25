from __future__ import annotations
import torch
from torch import nn
def sparsemax(x,dim=-1):
 x=x.float();z=x-x.max(dim=dim,keepdim=True).values;zs,_=torch.sort(z,dim=dim,descending=True);r=torch.arange(1,z.shape[dim]+1,device=x.device,dtype=x.dtype);shape=[1]*x.ndim;shape[dim]=-1;r=r.reshape(shape);support=1+r*zs>zs.cumsum(dim);k=support.sum(dim,keepdim=True).clamp_min(1);tau=(zs.cumsum(dim).gather(dim,k-1)-1)/k;return (z-tau).clamp_min(0)
class ResidualRelation(nn.Module):
 def __init__(self):super().__init__();self.project=nn.Sequential(nn.LayerNorm(320),nn.Linear(320,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,32),nn.LayerNorm(32),nn.GELU());self.scorer=nn.Linear(32,1)
 def forward(self,ez,nez):
  if not len(ez) or not len(nez):raise ValueError('EZ/NEZ set cannot be empty')
  ez=ez.float();nez=nez.float();m=ez.mean(0);ref=m.expand(len(nez),-1);tok=self.project(torch.cat([nez,ref,nez-ref,(nez-ref).abs(),nez*ref],1));score=self.scorer(tok).squeeze(-1).float();weight=sparsemax(score);emb=(weight[:,None]*tok.float()).sum(0);p=torch.sigmoid(score);return emb,score,weight,{'ez_mean':m,'ez_max':ez.max(0).values,'ez_std':ez.std(0,unbiased=False) if len(ez)>1 else torch.zeros_like(m),'nez_mean':nez.mean(0),'nez_std':nez.std(0,unbiased=False) if len(nez)>1 else torch.zeros_like(m),'score_mean':score.mean(),'score_std':score.std(unbiased=False) if len(score)>1 else score.new_zeros(()),'score_max':score.max(),'burden':p.mean()}
