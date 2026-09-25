from __future__ import annotations
import torch
import torch.nn.functional as F
def channel_loss(patient_seizures):
 """Equal EZ/NEZ weight within seizure, then seizure, then patient."""
 rows=[]
 for seizures in patient_seizures:
  per=[]
  for logits,labels in seizures:
   ez=logits[labels==0];nez=logits[labels==1]
   if not len(ez) or not len(nez):raise ValueError('channel loss requires EZ and NEZ')
   per.append(.5*F.binary_cross_entropy_with_logits(ez.float(),torch.zeros_like(ez))+.5*F.binary_cross_entropy_with_logits(nez.float(),torch.ones_like(nez)))
  rows.append(torch.stack(per).mean())
 return torch.stack(rows).mean()
def outcome_loss(logits,y,class_weight):return F.binary_cross_entropy_with_logits(logits.float(),y.float(),weight=torch.where(y>0,class_weight[1],class_weight[0]))
