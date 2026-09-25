from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
import torch
from .schema import ViewConfig,validate_patient

def stable_seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')
def _choose(values,limit,generator,randomized):
    values=list(values)
    if len(values)<=limit:return values
    if randomized:return [values[i] for i in torch.randperm(len(values),generator=generator)[:limit].tolist()]
    indexes=torch.linspace(0,len(values)-1,limit).round().long().tolist();return [values[i] for i in indexes]

class TeChC1PatientViewBuilder:
    def __init__(self,seed=42,config=ViewConfig()):self.seed,self.config=int(seed),config
    def build(self,patient,epoch=0,training=False):
        validate_patient(patient);g=torch.Generator().manual_seed(stable_seed(patient['patient_key'],self.seed,epoch));seizures=list(patient['seizures'])
        if training:seizures=_choose(seizures,self.config.max_seizures_train,g,True)
        output=[]
        for seizure in seizures:
            mask=seizure['window_mask'].bool();valid=torch.where(mask.any(1))[0].tolist();channels=_choose(valid,self.config.max_channels_train,g,training) if training else valid;selected=torch.zeros_like(mask)
            phase=seizure['phase_ids'];phase2=phase[None].expand_as(mask) if phase.ndim==1 else phase
            times=seizure['relative_times_sec'];times2=times[None].expand_as(mask) if times.ndim==1 else times
            for channel in channels:
                for phase_id in range(3):
                    candidates=torch.where(mask[channel]&(phase2[channel]==phase_id))[0].tolist();candidates=sorted(candidates,key=lambda i:float(times2[channel,i]));chosen=_choose(candidates,self.config.max_windows_per_phase_train,g,training) if training else candidates;selected[channel,chosen]=True
            keep_t=selected[channels].any(0);c=torch.tensor(channels);t=torch.where(keep_t)[0]
            if not len(t):continue
            output.append({'seizure_id':seizure['seizure_id'],'windows':seizure['windows'][c][:,t].float(),'window_mask':selected[c][:,t],'phase_ids':phase2[c][:,t].long(),'relative_times_sec':times2[c][:,t].float(),'channel_names':[seizure['channel_names'][i] for i in channels]})
        if not output:raise ValueError('patient view has no valid seizure')
        return {'patient_key':patient['patient_key'],'center':patient['center'],'seizures':output}

class CachedCohort:
    def __init__(self,cache_dir,seed=42,config=ViewConfig()):
        root=Path(cache_dir);manifest=pd.read_csv(root/'cache_manifest.csv');
        if manifest.patient_key.duplicated().any():raise ValueError('duplicate cache patient_key')
        self.root=root;self.paths={r.patient_key:root/r.shard_path for r in manifest.itertuples()};self.builder=TeChC1PatientViewBuilder(seed,config)
    def load(self,key):return torch.load(self.paths[key],map_location='cpu',weights_only=False)
    def view(self,key,epoch=0,training=False):return self.builder.build(self.load(key),epoch,training)
