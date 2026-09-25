import torch
from torch import nn
class OutcomeModel(nn.Module):
 def __init__(self,encoder):super().__init__();self.encoder=encoder;self.seizure=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,96),nn.LayerNorm(96),nn.GELU());self.head=nn.Sequential(nn.LayerNorm(288),nn.Linear(288,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1))
 def forward(self,views):
  out=[]
  for v in views:
   u=[];ns=[]
   for s in v['seizures']:
    h,_=self.encoder.encode_seizure(s);m=h.mean(0);sd=h.std(0,unbiased=False) if len(h)>1 else torch.zeros_like(m);u.append(self.seizure(torch.cat([m,sd,h.max(0).values])));ns.append(len(h))
   u=torch.stack(u);m=u.mean(0);sd=u.std(0,unbiased=False) if len(u)>1 else torch.zeros_like(m);rep=torch.cat([m,sd,u.max(0).values]);out.append({'logit':self.head(rep.float()).squeeze(),'representation':rep,'n_seizures':len(u),'n_channels_mean':sum(ns)/len(ns)})
  return out
