import torch
from torch import nn
class B(nn.Module):
 def __init__(self):super().__init__();self.n=nn.Sequential(nn.Conv1d(48,48,3,1,1,groups=48),nn.Conv1d(48,48,1),nn.GroupNorm(6,48),nn.GELU(),nn.Dropout(.05))
 def forward(self,x):return self.n(x)
class Temporal(nn.Module):
 def __init__(self):super().__init__();self.phase=nn.Embedding(3,48);self.time=nn.Sequential(nn.Linear(1,16),nn.GELU(),nn.Linear(16,48));self.b=nn.ModuleList([B(),B()]);self.p=nn.Sequential(nn.Linear(384,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.LayerNorm(64),nn.GELU())
 def forward(self,z,m,phase,t):
  x=z.float()+self.phase(phase)[None]+self.time(t.float()[:,None])[None];valid=m[:,None];x=x.transpose(1,2)*valid
  for b in self.b:x=(x+b(x))*valid
  x=x.transpose(1,2);w=m.float().unsqueeze(-1)
  def avg(q):return (x*q.float().unsqueeze(-1)).sum(1)/q.float().sum(1).clamp_min(1).unsqueeze(-1)
  g=avg(m);sd=torch.sqrt((((x-g[:,None])**2)*w).sum(1)/w.sum(1).clamp_min(1));mx=x.masked_fill(~m[:,:,None],-float('inf')).amax(1).nan_to_num();a,b,c=[avg(m&(phase[None]==i)) for i in range(3)]
  return self.p(torch.cat([g,sd,mx,a,b,c,b-a,c-b],1)),x
