from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .propagation import DESCRIPTOR_DIM
def off_diagonal_mean(matrix):
 n=matrix.shape[0]
 if n<=1:return matrix.new_zeros(())
 mask=~torch.eye(n,dtype=torch.bool,device=matrix.device);return matrix[mask].mean()
class VirtualDisconnection(nn.Module):
 def __init__(self,descriptor_dim=DESCRIPTOR_DIM,enabled=True,edge_correction=False):
  super().__init__();self.enabled=enabled;self.edge_correction=edge_correction;self.true_ez_supernode=bool(enabled);self.register_buffer('direction_temperature',torch.tensor(1.))
  if enabled:
   self.node=nn.Sequential(nn.Linear(descriptor_dim,16),nn.LayerNorm(16),nn.GELU());self.alpha=nn.Parameter(torch.tensor(1.));self.beta=nn.Parameter(torch.tensor(0.))
   if edge_correction:self.edge=nn.Sequential(nn.Linear(34,16),nn.Tanh(),nn.Linear(16,1))
 def forward(self,prop):
  if not self.enabled or prop is None or not prop['selected'].any():return None
  ez_node=self.node(prop['ez_descriptor'].unsqueeze(0));nez_desc=prop['nez_descriptor'][prop['selected']];nez_nodes=self.node(nez_desc);nodes=torch.cat([ez_node,nez_nodes]);tau=torch.cat([prop['ez_tau'].reshape(1),prop['nez_tau'][prop['selected']]]);assert len(tau)==len(nodes);z=F.normalize(nodes,dim=1,eps=1e-6);sim=z@z.T;delta=tau[None,:]-tau[:,None];direction=torch.sigmoid(delta/self.direction_temperature);a=torch.sigmoid(self.alpha*sim+self.beta)*direction
  if self.edge_correction:
   ni=nodes[:,None,:].expand(-1,len(nodes),-1);nj=nodes[None,:,:].expand(len(nodes),-1,-1);extra=torch.stack([tau[:,None].expand_as(sim),tau[None,:].expand_as(sim)],-1);a=torch.sigmoid(torch.logit(a.clamp(1e-5,1-1e-5))+self.edge(torch.cat([ni,nj,extra],-1)).squeeze(-1))
  a=a.masked_fill(torch.eye(len(a),dtype=torch.bool,device=a.device),0);a2=a@a;an=a[1:,1:];an2=an@an;eps=1e-6;full1=off_diagonal_mean(a);full2=off_diagonal_mean(a2);res1=off_diagonal_mean(an);res2=off_diagonal_mean(an2);zero=a.new_zeros(());u=torch.ones(len(an),device=a.device,dtype=a.dtype);rho=zero
  if an.numel():
   u=F.normalize(u,dim=0,eps=1e-6)
   for _ in range(3):
    nxt=an@u
    if torch.linalg.vector_norm(nxt)<=1e-8:u=torch.zeros_like(u);break
    u=F.normalize(nxt,dim=0,eps=1e-6)
   rho=torch.linalg.vector_norm(an@u)
  ez_out=a[0,1:].mean() if len(a)>1 else zero;stats=torch.nan_to_num(torch.stack([ez_out,res1,res1/(full1+eps),res2/(full2+eps),rho]));selected_tau=prop['nez_tau'][prop['selected']]
  return {'adjacency':a,'stats':stats,'n_nodes':len(a),'n_nez_nodes':len(nez_nodes),'ez_tau':prop['ez_tau'],'selected_nez_tau':selected_tau,'selected_nez_indices':prop['nez_indices'][prop['selected']],'uses_true_ez_supernode':True,'ez_self_similarity':prop['ez_self_similarity']}
