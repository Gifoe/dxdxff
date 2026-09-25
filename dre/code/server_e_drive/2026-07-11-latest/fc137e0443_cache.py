from __future__ import annotations
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,torch

PHASES=((0,-10.,-1.),(1,0.,5.),(2,5.,15.))
def robust_window(x):
    x=np.asarray(x,dtype=np.float32);median=np.median(x);mad=np.median(np.abs(x-median));x=np.clip((x-median)/max(1.4826*mad,1e-6),-8,8);grid=np.linspace(0,len(x)-1,500);return np.interp(grid,np.arange(len(x)),x).astype(np.float32)
def build_cache(raw_records,outcomes,cache_dir):
    root=Path(cache_dir);patients_dir=root/'patients';patients_dir.mkdir(parents=True,exist_ok=True);grouped=defaultdict(list)
    for record in raw_records:
        key=str(record['subject_id'])
        if key not in outcomes:continue
        names=list(map(str,record['channel_names_norm']));signal=np.asarray(record['sample']['raw_waveform'],dtype=np.float32)
        if signal.shape[0]!=len(names):signal=signal.T
        fs=float(record['sfreq']);sample=record['sample'];onset=int(round((float(sample.get('seizure_onset_sec',0))-float(sample.get('start_sec',0)))*fs));windows=[];phase=[];times=[]
        for phase_id,lo,hi in PHASES:
            width=max(1,int(round(2*fs)));start=max(0,int(round(onset+lo*fs)));stop=min(signal.shape[1],int(round(onset+hi*fs)))
            for index in range(start,max(start,stop-width+1),max(1,int(round(fs)))):
                windows.append(np.stack([robust_window(channel[index:index+width]) for channel in signal]));phase.append(phase_id);times.append((index-onset)/fs)
        if not windows:continue
        array=np.stack(windows,1);grouped[key].append({'seizure_id':str(record['run_id']),'windows':torch.from_numpy(array),'window_mask':torch.ones(array.shape[:2],dtype=torch.bool),'phase_ids':torch.tensor(phase),'relative_times_sec':torch.tensor(times),'channel_names':names})
    rows=[]
    for key,seizures in grouped.items():
        patient={'patient_key':key,'center':key.split(':')[0],'outcome_success':int(outcomes[key]),'seizures':seizures};name=hashlib.sha256(key.encode()).hexdigest()[:16]+'.pt';path=patients_dir/name;torch.save(patient,path);rows.append({'patient_key':key,'center':patient['center'],'outcome_success':patient['outcome_success'],'n_seizures':len(seizures),'shard_path':str(path.relative_to(root))})
    pd.DataFrame(rows).to_csv(root/'cache_manifest.csv',index=False);return pd.DataFrame(rows)
