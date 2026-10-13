"""Strict private waveform-only banks; historical compound alignment + valid bounds."""
import gc
import json
import pickle
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
from audit_sources import arguments,context,metadata


def support(centres,scales,sfreq,duration,total,valid_start,valid_samples):
 centres=np.asarray(centres,dtype=np.float64);scales=np.asarray(scales,dtype=np.float64)
 if scales.size!=len(centres):scales=np.full(len(centres),2.)
 assert np.isfinite(centres).all() and len(np.unique(np.round(centres,8)))==len(centres)
 assert sfreq==250 and duration==60 and total==15000 and 0<=valid_start<total and 0<valid_samples<=total-valid_start
 lengths=np.rint(scales*sfreq).astype(int);assert np.all(lengths==500)
 starts=np.rint((duration/2+centres)*sfreq).astype(int)-250
 assert len(np.unique(starts))==len(starts),'Duplicated physical window after sample rounding'
 valid=(starts>=valid_start)&(starts+500<=valid_start+valid_samples)&(starts>=0)&(starts+500<=total)
 return starts,valid


def clean_record(record,raw=False):
 allowed=['subject_id','run_id','sample_id','source_seizure_id','seizure_onset_sec','start_sec','end_sec','channel_names_norm','source_center','center',
  'window_relative_centers_sec','feature_scale_used_secs','raw_temporal_sfreq','raw_temporal_duration_sec','raw_valid_start_sample','raw_valid_samples']
 if raw:allowed.append('raw_waveform')
 return {**{k:record[k] for k in allowed if k in record},'sample':{k:v for k,v in record.get('sample',{}).items() if k in allowed}}


def main():
 a=arguments();lock,ref,c,core,original,trainer,helper,banks,manifest,bind=context(a)
 assert json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text())['binding']==bind
 out=a.runtime/'RAW_REGISTRY_PRIVATE.json';assert not out.exists()
 rawpath=Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl')
 featpath=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl')
 assert c.sha(rawpath)==lock['raw_cache_sha256'] and c.sha(featpath)==lock['window_cache_sha256']
 print('FULL_RAW_AND_WINDOW_CACHE_HASH_PASS',flush=True)
 meta=metadata(a);pid=meta['patient'].astype(str);assert len(np.unique(pid))==80
 with featpath.open('rb') as f:payload=pickle.load(f)
 records=[clean_record(r) for r in payload['run_records'] if str(r['subject_id']) in set(pid)]
 del payload;gc.collect();assert len(records)==256
 sys.path.insert(0,str(a.source));sys.path.insert(0,str(a.source/'neuroez_c'))
 from neuroez_c.dual_view_data import RawAlignmentStore,_record_key,_channel_names,_field,_window_centers
 from neuroez_c.raw_brainbert_data import normalize_channel_name
 store=RawAlignmentStore(records,feature_cache_path=featpath,raw_cache_path=rawpath,raw_target_samples=500,raw_target_sampling_rate=250)
 store.assert_formal_coverage(min_channel_match_rate=1,min_window_match_rate=1,expected_patients=80)
 store.raw_records={k:clean_record(r,raw=True) for k,r in store.raw_records.items()};gc.collect()
 plans=[];projected=0
 for p in sorted(set(pid)):
  channels=meta['channel'][pid==p].astype(str).tolist();norm=[normalize_channel_name(x) for x in channels];assert len(norm)==len(set(norm))
  rr=[r for r in records if str(r['subject_id'])==p];assert rr
  maxw=max(len(_window_centers(r)) for r in rr);shape=(1,len(rr),len(channels),maxw,500);projected+=int(np.prod(shape))*4
  plans.append((p,channels,norm,rr,shape))
 free=shutil.disk_usage(a.runtime).free;assert free>=projected+6*1024**3,'Insufficient private raw-bank disk reserve; no deletion allowed'
 files=[];recordwindows=channelwindows=run_channels=0;invalid_windows=0
 for n,(p,channels,norm,rr,shape) in enumerate(plans,1):
  raw=np.zeros(shape,np.float32);mask=np.zeros(shape[:-1],bool);lookup={k:i for i,k in enumerate(norm)}
  for s,record in enumerate(rr):
   key=_record_key(record);r=store.raw_records[key];centres=_window_centers(record);names=_channel_names(record)
   assert len(names)==len(set(names)) and set(names)<=set(norm)
   sample=r['sample'];wave=np.asarray(sample['raw_waveform']);assert wave.ndim==2 and len(wave)==len(_channel_names(r)) and np.isfinite(wave).all()
   starts,valid=support(centres,_field(record,'feature_scale_used_secs',[]),float(_field(r,'raw_temporal_sfreq')),float(_field(r,'raw_temporal_duration_sec')),wave.shape[-1],int(_field(r,'raw_valid_start_sample')),int(_field(r,'raw_valid_samples')))
   aligned,am,availability=store.aligned_windows(record);am[~valid]=False
   assert aligned.shape==(len(centres),len(names),500) and am.all(axis=1).sum()==valid.sum()
   rawnames=_channel_names(r);rawlookup={x:j for j,x in enumerate(rawnames)}
   for j,name in enumerate(names):
    target=lookup[name];raw[0,s,target,:len(centres)]=aligned[:,j];mask[0,s,target,:len(centres)]=am[:,j]
    for w in np.flatnonzero(valid):assert np.array_equal(aligned[w,j],wave[rawlookup[name],starts[w]:starts[w]+500].astype(np.float32))
   recordwindows+=int(valid.sum());channelwindows+=int(am.sum());run_channels+=len(names);invalid_windows+=int((~valid).sum())
  # Invalid/unavailable padded entries are not observations or model inputs.
  raw[~mask]=0.;assert mask.any(axis=(1,3)).all() and np.isfinite(raw).all()
  fname=c.digest(p)[:24]+'.pt';path=a.runtime/'raw_banks'/fname;assert not path.exists()
  core.torch_write(path,dict(raw=torch.from_numpy(raw),mask=torch.from_numpy(mask),channels=channels,patient=p,records=len(rr)))
  files.append(dict(patient=p,channels=channels,filename=fname,sha256=c.sha(path),records=len(rr),valid_windows=int(mask.sum())))
  print('RAW_PATIENT_BANK_COMPLETE',n,80,flush=True)
 assert recordwindows==15074 and channelwindows==1471965 and run_channels==24995 and len(files)==80
 c.write_json(out,dict(binding=c.sha(a.runtime/'EXPERIMENT_FREEZE.json'),raw_sha256=lock['raw_cache_sha256'],window_sha256=lock['window_cache_sha256'],files=files))
 c.write_json(a.runtime/'public/RAW_ALIGNMENT_AUDIT.json',dict(status='PASS',patients=80,canonical_channels=7635,seizures=256,run_channel_incidences=run_channels,
  valid_record_windows=recordwindows,valid_channel_windows=channelwindows,invalid_feature_windows=invalid_windows,all_sample_window_parity_exact=True,
  duplicate_physical_windows=0,raw_cache_sha256=lock['raw_cache_sha256'],window_cache_sha256=lock['window_cache_sha256'],registry_sha256=c.sha(out),
  raw_bank_disk_bytes=projected,free_disk_after_bytes=shutil.disk_usage(a.runtime).free,engineered_values_used_as_model_inputs=False,global_feature_y_member_loaded=False,
  legacy_pickle_materializes_other_fields_at_load=True,legacy_fields_discarded_without_label_indexing=True,onset_provenance='SOURCE_CODE_SUPPORTED_NOT_INDEPENDENTLY_EDF_CLOCK_CERTIFIED',sfreq=250,Nyquist=125))
 print('RAW_ALIGNMENT_ALL80_ALL256_ALL1471965_PASS',flush=True)


class RawBankStore:
 def __init__(self,runtime,c):
  self.runtime=Path(runtime);self.c=c;self.cache={};self.registry=json.loads((self.runtime/'RAW_REGISTRY_PRIVATE.json').read_text())
  audit=json.loads((self.runtime/'public/RAW_ALIGNMENT_AUDIT.json').read_text());assert audit['status']=='PASS' and self.c.sha(self.runtime/'RAW_REGISTRY_PRIVATE.json')==audit['registry_sha256']
  self.files={r['patient']:r for r in self.registry['files']}

 def get(self,patient,channels):
  patient=str(patient)
  if patient not in self.cache:
   rec=self.files[patient];path=self.runtime/'raw_banks'/rec['filename'];assert self.c.sha(path)==rec['sha256']
   d=torch.load(path,map_location='cpu',weights_only=False);assert d['patient']==patient and d['channels']==rec['channels'];self.cache[patient]=d
  d=self.cache[patient];assert set(channels)==set(d['channels']) and len(channels)==len(d['channels'])
  if list(channels)==d['channels']:return d['raw'],d['mask']
  ix=[d['channels'].index(str(k)) for k in channels]
  return d['raw'].index_select(2,torch.tensor(ix)),d['mask'].index_select(2,torch.tensor(ix))


if __name__=='__main__':main()
