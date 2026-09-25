from __future__ import annotations
import hashlib,json,os,re,time
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd,torch
from . import CACHE_VERSION
from .preprocessing import TARGET_FS,resample_channel_once,reference,normalize_window
PHASES=((0,-10.,-1.),(1,0.,5.),(2,5.,15.));CODE_VERSION='trace_dre_cache_builder_v2_3_outcome_signature';NORM_VERSION='preictal_reference_v1';SIDE_VERSION='relative_side12_v1'
def file_identity(path):
 p=Path(path);s=p.stat();return {'path':str(p.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}
def make_signature(raw_cache,exclusion_manifest,outcome_table):return {'cache_version':CACHE_VERSION,'raw_cache':file_identity(raw_cache),'outcome_table':file_identity(outcome_table),'target_fs':TARGET_FS,'phases':PHASES,'window_sec':2.,'stride_sec':1.,'normalization_version':NORM_VERSION,'side_feature_version':SIDE_VERSION,'exclusion_manifest':file_identity(exclusion_manifest),'cache_code_version':CODE_VERSION}
def signature_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def safe_key(key):return re.sub(r'[^A-Za-z0-9_.-]+','_',key)+'_'+hashlib.sha256(key.encode()).hexdigest()[:10]
def atomic_torch(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf8');os.replace(tmp,path)
def _signal(record):
 x=np.asarray(record['sample']['raw_waveform']);n=len(record['channel_names_norm']);return x if x.shape[0]==n else x.T
def build_patient(patient,records,outcome,root,resume=True):
 shard=Path(root)/'patients'/(safe_key(patient)+'.pt')
 if resume and shard.exists():return shard,[],[],True
 seizures=[];reject_w=[];reject_c=[]
 for rec in records:
  sig=_signal(rec);sample=rec['sample'];fs=float(sample.get('raw_temporal_sfreq',rec['sfreq']));onset=(float(sample.get('seizure_onset_sec',0))-float(sample.get('start_sec',0)))*TARGET_FS;labels=np.asarray(rec['labels']);names=list(map(str,rec['channel_names_norm']));channel_windows=[];channel_sides=[];channel_masks=[];quality=[];kept_names=[];kept_labels=[];times=None;phase_ids=None
  for ci,(name,label) in enumerate(zip(names,labels)):
   if label not in (0,1):continue
   x=resample_channel_once(sig[ci],fs);pre=x[max(0,int(onset-10*TARGET_FS)):max(0,int(onset-1*TARGET_FS))]
   if len(pre)<TARGET_FS or np.isfinite(pre).mean()<.95 or np.std(np.nan_to_num(pre))<1e-8:reject_c.append({'patient_key':patient,'seizure_id':str(rec['run_id']),'channel_name':name,'reason':'invalid_preictal_reference'});continue
   ref=reference(np.nan_to_num(pre,nan=float(np.nanmedian(pre))));wins=[];sides=[];mask=[];local_times=[];local_phases=[]
   for phase,lo,hi in PHASES:
    for rel in np.arange(lo,hi-2+1e-6,1.):
     start=int(round(onset+rel*TARGET_FS));end=start+500
     if start<0 or end>len(x):z=side=None;reason='out_of_bounds'
     else:z,side,reason=normalize_window(x[start:end],ref)
     if z is None:z=np.zeros(500,np.float32);side=np.zeros(12,np.float32);mask.append(False);reject_w.append({'patient_key':patient,'seizure_id':str(rec['run_id']),'channel_name':name,'relative_start_sec':rel,'reason':reason})
     else:mask.append(True)
     wins.append(z);sides.append(side);local_times.append(float(rel+1));local_phases.append(phase)
   channel_windows.append(wins);channel_sides.append(sides);channel_masks.append(mask);quality.append([ref['scale'],ref['rms'],ref['ll'],float(np.mean(mask))]);kept_names.append(name);kept_labels.append(int(label));times=local_times;phase_ids=local_phases
  if kept_labels and 0 in kept_labels and 1 in kept_labels:seizures.append({'seizure_id':str(rec['run_id']),'windows':torch.tensor(np.asarray(channel_windows),dtype=torch.float16),'window_mask':torch.tensor(channel_masks,dtype=torch.bool),'phase_ids':torch.tensor(phase_ids),'relative_times_sec':torch.tensor(times),'side_features':torch.tensor(np.asarray(channel_sides),dtype=torch.float32),'channel_names':kept_names,'channel_labels_nez':torch.tensor(kept_labels),'channel_quality':torch.tensor(quality,dtype=torch.float32)})
 if not seizures:raise RuntimeError(f'{patient}: no valid seizure remains after TRACE-DRE preprocessing')
 obj={'cache_version':CACHE_VERSION,'patient_key':patient,'center':patient.split(':')[0],'outcome_success':int(outcome),'seizures':seizures};atomic_torch(shard,obj);return shard,reject_w,reject_c,False
def verify_cache(root,min_patients=10):
 files=sorted((Path(root)/'patients').glob('*.pt'));selected=files[:min(len(files),max(1,min_patients))];errors=[]
 for path in selected:
  p=torch.load(path,map_location='cpu',weights_only=False)
  if p.get('cache_version')!=CACHE_VERSION:errors.append(f'{path}:version')
  if p['outcome_success'] not in (0,1):errors.append(f'{path}:outcome')
  if not p['seizures']:errors.append(f'{path}:no_valid_seizure')
  for s in p['seizures']:
   if s['windows'].ndim!=3 or s['windows'].shape[-1]!=500:errors.append(f'{path}:shape')
   if not torch.isfinite(s['windows']).all() or s['windows'].abs().max()>8:errors.append(f'{path}:values')
   if not set(s['channel_labels_nez'].tolist())=={0,1}:errors.append(f'{path}:labels')
   for phase in range(3):
    if not (((s['phase_ids']==phase).unsqueeze(0)&s['window_mask']).any()):errors.append(f'{path}:missing_valid_phase_{phase}')
   if not torch.all(s['relative_times_sec'][1:]>=s['relative_times_sec'][:-1]):errors.append(f'{path}:order')
 report={'cache_version':CACHE_VERSION,'checked_patients':len(selected),'errors':errors,'valid':not errors};atomic_json(Path(root)/'cache_verify_report.json',report);return report
