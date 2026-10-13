"""Trusted-server metadata/provenance inventory. No PSD or clinical outcomes.

Only aggregate metadata leaves the server. Record identities and source paths
are saved exclusively to the private runtime. Trusted pickle label fields are
not inspected or used; full A0 export x/y are never materialized here.
"""
import gc
import hashlib
import json
import pickle
import re
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(r'C:\a0_fcpa_data_feasibility_seed42_v1_runtime')
SRC=Path(r'E:\DRE-nips\new-pipeline\7-11')
EXPORT=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
SPLIT=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
CANDIDATES={
 'A0_window':Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl'),
 'raw_primary':Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl'),
 'raw_task2_other':Path(r'D:\nips-temp\neuroez_c_four_center_caches_task2_primary149_raw_v1\all_window_cache.pkl'),
 'feature_success_failure_other':Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_feature_v1\all_window_cache.pkl')}


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=True,default=str),encoding='utf-8')


def norm(x):return ''.join(str(x).upper().split()).replace('-','').replace('_','')


def config_fields(x):
 keep={}
 names=['spectral_min_freq','spectral_max_freq','raw_temporal_sfreq','target_sfreq','reader_target_sfreq',
  'reader_bandpass_low','reader_bandpass_high','bandpass_low','bandpass_high','line_freq',
  'sliding_window_sec','sliding_step_sec','prepost_context_sec','source_code_sha256','code_sha256','cache_type','schema_version']
 if isinstance(x,dict):
  for key,value in x.items():
   if key in names and (value is None or isinstance(value,(str,int,float,bool))):keep[key]=value
   elif isinstance(value,dict) and key not in ['patient_index','channel_meta','metadata','sample']:
    inside=config_fields(value)
    if inside:keep[key]=inside
 return keep


def main():
 ROOT.mkdir(parents=True,exist_ok=True);pub=ROOT/'public';pub.mkdir(exist_ok=True)
 assert sha(EXPORT)=='1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd'
 assert sha(SPLIT)=='fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278'
 with np.load(EXPORT,allow_pickle=True) as f:
  patient=f['patient'].astype(str);channel=np.array([norm(s) for s in f['channel']]);centers=f['center'].astype(str)
  names=f['features'];sp=f['split_patient'];sr=f['split_role'];sf=f['split_fold']
 ids=set(patient);pairs=set(zip(patient,channel));assert len(ids)==80 and len(pairs)==len(patient)==7635
 reports=[];private={};expected=None
 for alias,path in CANDIDATES.items():
  if not path.exists():reports.append(dict(alias=alias,exists=False));continue
  digest=sha(path);print('INVENTORY_HASH_COMPLETE',alias,flush=True)
  with path.open('rb') as f:p=pickle.load(f)
  rr=p['run_records'];selected=[r for r in rr if str(r.get('subject_id')) in ids]
  ks=[];sample_keys=Counter();record_keys=Counter();meta_keys=Counter();channel_meta_keys=Counter();sfreq=Counter();sources=[]
  window_names=set();summary=[];matched_pairs=set()
  for r in selected:
   s=r.get('sample',{});pat=str(r['subject_id']);run=str(r['run_id']);sampleid=str(s.get('sample_id',run));nn=tuple(norm(n) for n in r['channel_names_norm'])
   key=(pat,run,sampleid,str(s.get('seizure_onset_sec')),str(s.get('start_sec')),str(s.get('end_sec')));ks.append(key)
   sample_keys.update(s.keys());record_keys.update(r.keys());meta_keys.update(r.get('metadata',{}).keys())
   mm=r.get('channel_meta',[])
   for cm in mm:channel_meta_keys.update(cm.keys())
   spaths=set(str(cm.get('source_path')) for cm in mm if cm.get('source_path'))
   sources.append(dict(key=key,paths=sorted(spaths),source_exists=sum(Path(x).exists() for x in spaths)))
   matched_pairs.update((pat,n) for n in nn);sfreq[str(s.get('raw_temporal_sfreq',r.get('sfreq')))]+=1
   window_names.update(map(str,s.get('window_feature_names',[])))
   summary.append(dict(key=key,channel_names=nn,sfreq=r.get('sfreq'),
      raw_sfreq=s.get('raw_temporal_sfreq'),raw_duration=s.get('raw_temporal_duration_sec'),
      raw_valid_start=s.get('raw_valid_start_sample'),raw_valid_samples=s.get('raw_valid_samples'),
      start_sec=s.get('start_sec'),end_sec=s.get('end_sec'),onset_sec=s.get('seizure_onset_sec'),
      waveform_sha256=hashlib.sha256(np.ascontiguousarray(s['raw_waveform']).tobytes()).hexdigest() if 'raw_waveform' in s else None,
      source_paths=sorted(spaths),source_channel_meta=[{k:cm[k] for k in cm if k in ['source_path','channel_name_orig','channel_name_norm','sfreq_original','onset_source','time_row_index','events_path','channels_path','raw_channel_name']} for cm in mm],
      center=pat.split(':')[0]))
  if alias=='A0_window':expected=set(ks)
  reports.append(dict(alias=alias,exists=True,sha256=digest,bytes=path.stat().st_size,total_records=len(rr),
    total_patients=len(set(str(r.get('subject_id')) for r in rr)),main_patients=len(set(str(r.get('subject_id')) for r in selected)),
    main_records=len(selected),main_compound_keys=len(set(ks)),matching_A0_keys=len(set(ks)&(expected or set())),
    canonical_union=len(matched_pairs),canonical_matches=len(matched_pairs&pairs),extra_channels=len(matched_pairs-pairs),
    raw_sfreq_counts=dict(sfreq),payload_keys=list(p.keys()),record_field_names=list(record_keys),
    sample_field_names=list(sample_keys),metadata_field_names=list(meta_keys),channel_meta_field_names=list(channel_meta_keys),
    config=config_fields(p),window_feature_names=sorted(window_names),source_records_with_existing_path=sum(q['source_exists']>0 for q in sources)))
  private[alias]=dict(path=str(path),sha256=digest,records=summary)
  del p,rr,selected;gc.collect();print('INVENTORY_ALIAS_COMPLETE',alias,flush=True)
 # Logs are scanned for explicit non-identifying command flags, never raw lines.
 flags=['reader-bandpass-low','reader-bandpass-high','reader-target-sfreq','spectral-min-freq','spectral-max-freq','raw-temporal-sfreq','prepost-context-sec','sliding-window-sec','sliding-step-sec','force-source-read']
 logs=[]
 for directory in Path(r'D:\nips-temp').glob('logs*'):
  for path in directory.glob('*raw*.log'):
   raw=path.read_bytes();text=raw.decode('utf-16' if raw[:2] in [b'\xff\xfe',b'\xfe\xff'] else 'utf-8',errors='replace')
   found={flag:re.findall(r'--'+flag+r'(?:[=\s]+)([-+\d.eE]+)',text) for flag in flags}
   logs.append(dict(alias=directory.name+'/'+path.name,sha256=sha(path),bytes=len(raw),encoding='utf16_BOM' if raw[:2] in [b'\xff\xfe',b'\xfe\xff'] else 'utf8',explicit_flags={k:sorted(set(v)) for k,v in found.items() if v},
    primary_cache_root_mentioned='neuroez_c_four_center_caches_success_failure_raw_v1' in text,
    reuse_source_checkpoint_mentions=text.count('reused source checkpoint'),build_mentions=text.count('wrote source checkpoint')))
 sourcefiles=['neuroez_c/evidence_views.py','ez_features.py','module2_preprocessing.py','build_neuroez_c_four_center_caches.py',
  'run_baseline_three_centers/build_center_caches_from_patient_records.py','bn_pdgs_ranker/data_readers/edf.py','bn_pdgs_ranker/data_readers/schemas.py',
  'bn_pdgs_ranker/data_readers/bids_loader.py','bn_pdgs_ranker/data_readers/lzu.py','neuroez_c/data/pediatric.py','neuroez_c/dual_view_data.py']
 write(ROOT/'DISCOVERY_PRIVATE.json',private)
 report=dict(status='METADATA_INVENTORY_COMPLETE',cache_inventory=reports,log_inventory=logs,
  source_hashes={p:sha(SRC/p) for p in sourcefiles if (SRC/p).exists()},A0_hash=sha(EXPORT),split_hash=sha(SPLIT),
  A0_feature_names=names.tolist(),A0_order_sha256=hashlib.sha256(json.dumps(names.tolist(),separators=(',',':')).encode()).hexdigest(),
  cohort=dict(patients=80,canonical_channels=7635),waveform_PSD_run=False,full_A0_x_y_materialized=False,
  labels_or_prediction_metrics_used=False)
 write(pub/'DISCOVERY_AGGREGATE.json',report)
 print('DISCOVERY_COMPLETE_METADATA_ONLY',flush=True)


if __name__=='__main__':main()
