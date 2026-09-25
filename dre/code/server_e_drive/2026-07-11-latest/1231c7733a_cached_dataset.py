from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd,torch
from .schema import ViewConfig
def stable_int(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')
def _cyclic(items,count,offset):
 items=list(items)
 if len(items)<=count:return items
 start=offset%len(items);return (items+items)[start:start+count]
def make_view(patient,seed,view_id,cfg:ViewConfig):
 seizures=patient['seizures'];order=sorted(range(len(seizures)),key=lambda i:stable_int(patient['patient_key'],seed,'seizure',i));chosen=_cyclic(order,cfg.max_seizures,view_id*cfg.max_seizures);result=[]
 for si in chosen:
  s=seizures[si];labels=s['channel_labels_nez'];ez=torch.where(labels==0)[0].tolist();nez=torch.where(labels==1)[0].tolist();ez=sorted(ez,key=lambda x:stable_int(patient['patient_key'],seed,view_id,si,'ez',x));nez=sorted(nez,key=lambda x:stable_int(patient['patient_key'],seed,view_id,si,'nez',x));channels=_cyclic(ez,cfg.max_ez,view_id*cfg.max_ez)+_cyclic(nez,cfg.max_nez,view_id*cfg.max_nez);channels=torch.tensor(channels,dtype=torch.long)
  time_idx=[]
  for phase in range(3):
   ids=torch.where(s['phase_ids']==phase)[0].tolist();time_idx+=_cyclic(ids,cfg.max_windows_per_phase,view_id*cfg.max_windows_per_phase)
  time_idx=torch.tensor(sorted(time_idx),dtype=torch.long)
  result.append({'seizure_id':s['seizure_id'],'windows':s['windows'][channels][:,time_idx].float(),'window_mask':s['window_mask'][channels][:,time_idx],'phase_ids':s['phase_ids'][time_idx],'relative_times_sec':s['relative_times_sec'][time_idx],'side_features':s['side_features'][channels][:,time_idx],'channel_names':[s['channel_names'][i] for i in channels.tolist()],'channel_labels_nez':labels[channels],'channel_quality':s['channel_quality'][channels]})
 return {'patient_key':patient['patient_key'],'center':patient['center'],'outcome_success':patient['outcome_success'],'view_id':view_id,'seizures':result}
class CachedCohort:
 def __init__(self,cache_root,manifest,seed=42,view_config=ViewConfig()):
  frame=pd.read_csv(manifest) if not isinstance(manifest,pd.DataFrame) else manifest;self.paths={r.patient_key:Path(cache_root)/r.shard_path for r in frame.itertuples()};self.seed=seed;self.cfg=view_config
 def keys(self):return list(self.paths)
 def load(self,key):return torch.load(self.paths[key],map_location='cpu',weights_only=False)
 def view(self,key,view_id):return make_view(self.load(key),self.seed,view_id,self.cfg)
