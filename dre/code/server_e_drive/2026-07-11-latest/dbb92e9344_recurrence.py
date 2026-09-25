from __future__ import annotations
from collections import defaultdict
import math,torch
def recurrence_features(seizures,enabled=True):
 device=seizures[0]['logits'].device if seizures else 'cpu'
 if not enabled or len(seizures)<2:return torch.zeros(9,device=device),{'recurrence_present':0,'matched_channels':0}
 rows=defaultdict(list);sets=[]
 for s in seizures:
  selected=set()
  for name,score,tau,top in zip(s['names'],s['logits'],s['tau'],s['selected']):rows[name].append((score,tau,bool(top)));selected.add(name) if top else None
  sets.append(selected)
 matched={k:v for k,v in rows.items() if len(v)>=2}
 if not matched:return torch.zeros(9,device=device),{'recurrence_present':0,'matched_channels':0}
 means=[];variances=[];diffs=[];tauvars=[];freq=[]
 for vals in matched.values():
  x=torch.stack([v[0] for v in vals]);t=torch.stack([v[1] for v in vals]);means.append(x.mean());variances.append(x.var(unbiased=False));diffs.append(torch.pdist(x[:,None]).mean() if len(x)>1 else x.new_zeros(()));tauvars.append(t.var(unbiased=False));freq.append(x.new_tensor(sum(v[2] for v in vals)/len(seizures)))
 j=[]
 for i in range(len(sets)):
  for k in range(i+1,len(sets)):j.append(len(sets[i]&sets[k])/max(len(sets[i]|sets[k]),1))
 union=set().union(*sets);inter=set.intersection(*sets) if sets else set();f=torch.stack(freq);entropy=-(f.clamp_min(1e-6)*f.clamp_min(1e-6).log()).mean();features=torch.stack([torch.stack(means).mean(),torch.stack(means).max(),torch.stack(variances).mean(),(f>=.5).float().mean(),f.new_tensor(sum(j)/max(len(j),1)),f.new_tensor(len(union)/max(len(inter),1)),entropy,f.new_tensor(len(matched)/max(len(union),1)),f.new_tensor(1.)]);return features,{'recurrence_present':1,'matched_channels':len(matched),'topq10_jaccard':sum(j)/max(len(j),1),'recurrent_escape':float(features[0].detach())}
