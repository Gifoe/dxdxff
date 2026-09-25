from __future__ import annotations
import torch
from torch import nn
def masked_mean(x,mask,dim):return (x*mask.unsqueeze(-1)).sum(dim)/mask.sum(dim).clamp_min(1).unsqueeze(-1)
class RecruitmentDynamics(nn.Module):
 def __init__(self,enabled=True):
  super().__init__();self.enabled=enabled
  if enabled:self.temporal=nn.Sequential(nn.Conv1d(32,32,3,padding=1,groups=32),nn.Conv1d(32,32,1),nn.GELU(),nn.Conv1d(32,32,3,padding=1,groups=32));self.score=nn.Linear(32,1)
 def forward(self,h,mask,phase,times):
  z=self.temporal(h.transpose(1,2)).transpose(1,2) if self.enabled else h;z=z*mask.unsqueeze(-1);curve=self.score(z).squeeze(-1) if self.enabled else z.mean(-1);pre=(phase==0).unsqueeze(0)&mask;base=(curve*pre).sum(1)/pre.sum(1).clamp_min(1);centered=curve-base[:,None];active=((phase==1)|(phase==2)).unsqueeze(0)&mask;weights=torch.softmax(centered.masked_fill(~active,-1e9),1);tau=(weights*times.unsqueeze(0)).sum(1)
  emb=[masked_mean(z,((phase==p).unsqueeze(0)&mask),1) for p in range(3)];return {'sequence':z,'curve':centered,'tau':tau,'pre':emb[0],'onset':emb[1],'spread':emb[2]}
