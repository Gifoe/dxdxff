from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd,torch
from dataclasses import dataclass
@dataclass(frozen=True)
class ViewConfig:
 n_views:int=4;max_seizures:int=2;max_ez:int=24;max_nez:int=40;max_windows_per_phase:int=4
def _key(*x):return int.from_bytes(hashlib.sha256('|'.join(map(str,x)).encode()).digest()[:8],'big')
def _cyclic(items,n,offset):
 items=list(items)
 if len(items)<=n:return items
 return (items+items)[offset%len(items):offset%len(items)+n]
def make_view(patient,seed,view_id,cfg):
 seizures=patient['seizures'];order=sorted(range(len(seizures)),key=lambda i:_key(patient['patient_key'],seed,'seizure',i));chosen=_cyclic(order,cfg.max_seizures,view_id*cfg.max_seizures);out=[]
 for si in chosen:
  s=seizures[si];labels=s['channel_labels_nez'];ez=sorted(torch.where(labels==0)[0].tolist(),key=lambda i:_key(patient['patient_key'],seed,view_id,si,'ez',i));nez=sorted(torch.where(labels==1)[0].tolist(),key=lambda i:_key(patient['patient_key'],seed,view_id,si,'nez',i));channels=torch.tensor(_cyclic(ez,cfg.max_ez,view_id*cfg.max_ez)+_cyclic(nez,cfg.max_nez,view_id*cfg.max_nez),dtype=torch.long);times=[]
  for phase in range(3):times+=_cyclic(torch.where(s['phase_ids']==phase)[0].tolist(),cfg.max_windows_per_phase,view_id*cfg.max_windows_per_phase)
  times=torch.tensor(sorted(times),dtype=torch.long);out.append({'seizure_id':s['seizure_id'],'windows':s['windows'][channels][:,times].float(),'side_features':s['side_features'][channels][:,times].float(),'window_mask':s['window_mask'][channels][:,times],'phase_ids':s['phase_ids'][times],'relative_times_sec':s['relative_times_sec'][times],'channel_names':[s['channel_names'][i] for i in channels.tolist()],'channel_labels_nez':labels[channels]})
 return {'patient_key':patient['patient_key'],'center':patient['center'],'outcome_success':int(patient['outcome_success']),'view_id':view_id,'seizures':out}
class CachedCohort:
 def __init__(self,root,manifest,seed,cfg):
  frame=pd.read_csv(manifest) if not isinstance(manifest,pd.DataFrame) else manifest;self.paths={r.patient_key:Path(root)/r.shard_path for r in frame.itertuples()};self.seed=seed;self.cfg=cfg
 def keys(self):return list(self.paths)
 def load(self,key):return torch.load(self.paths[key],map_location='cpu',weights_only=False)
 def view(self,key,view):return make_view(self.load(key),self.seed,view,self.cfg)
