"""Signal-only audit. Explicit allowlists; no ML imports or clinical scoring.

freeze: validate contracts, resolve EDFs/headers/annotations without signal PSD;
raw: valid-window PSD and named historical-window features;
edf: same-channel/time signal comparisons and fixed filtering controls;
finish: anonymous aggregates and the locked data gate.
"""
import argparse
import contextlib
import gc
import hashlib
import io
import json
import pickle
import re
import sys
import warnings
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import signal,stats
from discover import ROOT,SRC,EXPORT,SPLIT,CANDIDATES,sha,write,norm

HERE=Path(__file__).parents[1]


def atom(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial')
 with tmp.open('wb') as f:pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
 tmp.replace(p)


def key(r):
 s=r['sample'];return (str(r['subject_id']),str(r['run_id']),str(s.get('sample_id',r['run_id'])),
  str(s.get('seizure_onset_sec')),str(s.get('start_sec')),str(s.get('end_sec')))


def lock():return json.loads((HERE/'PROTOCOL_LOCK.json').read_text())


def bands(w,fs,p):
 w=np.asarray(w,dtype=np.float64);n=int(round(2*fs));assert w.shape[-1]==n
 f,psd=signal.welch(w,fs=fs,window='hann',nperseg=n,noverlap=n//2,nfft=n,detrend='constant',scaling='density',average='mean',axis=-1)
 ans={}
 for name,(lo,hi) in p['bands'].items():
  m=(f>=lo)&(f<=hi);assert m.sum()>=2
  ans[name]=np.trapezoid(psd[...,m],f[m],axis=-1)
 return ans


def header(path):
 with path.open('rb') as f:
  base=f.read(256);assert base[:8].strip()==b'0','Only16-bit EDF supported';ns=int(base[252:256]);h=f.read(ns*256)
 duration=float(base[244:252]);parts={};offset=0
 for name,width in [('label',16),('transducer',80),('unit',8),('pmin',8),('pmax',8),('dmin',8),('dmax',8),('prefilter',80),('samples',8),('reserved',32)]:
  b=h[offset:offset+ns*width];parts[name]=[b[i*width:(i+1)*width].decode('latin1').strip(' \x00') for i in range(ns)];offset+=ns*width
 parts['fs']=[float(n)/duration for n in parts['samples']]
 parts['header_bytes']=int(base[184:192]);parts['record_duration']=duration;parts['samples']=list(map(int,parts['samples']))
 parts['record_bytes']=2*sum(parts['samples']);parts['n_records']=int(base[236:244])
 if parts['n_records']<0:parts['n_records']=(path.stat().st_size-parts['header_bytes'])//parts['record_bytes']
 return parts


def filter_number(text,kind):
 match=re.search(r'(?i)\b'+kind+r'(?:ass)?\s*[:=]?\s*([\d.]+)',text)
 return float(match.group(1)) if match else None


def resolve(path):
 p=Path(path)
 if p.is_file():return p,'recorded_path'
 # Prefix relocation only; preserve complete patient/run suffix. No fuzzy IDs.
 candidates=[]
 for drive in ['E:/','F:/']:
  for root in Path(drive).iterdir():
   if root.is_dir() and 'seeg' in root.name.lower():
    q=root.joinpath(*p.parts[2:])
    if q.is_file():candidates.append(q)
 unique=list(dict.fromkeys(candidates))
 return (unique[0],'unique_exact_suffix_root_relocation') if len(unique)==1 else (None,'missing_or_ambiguous')


def quiet_edf(path):
 # Header-only original MNE default channel-name handling; no TAL signal scan.
 from mne._fiff.meas_info import _unique_channel_names
 h=header(path);sel=[i for i,n in enumerate(h['label']) if n not in ['EDF Annotations','BDF Annotations']]
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),warnings.catch_warnings():
  warnings.simplefilter('ignore')
  names=_unique_channel_names([h['label'][i] for i in sel])
 fs=max(h['fs'][i] for i in sel)
 return SimpleNamespace(info={'sfreq':fs},n_times=int(round(h['n_records']*h['record_duration']*fs)),ch_names=names,
  header_indices=sel,annotations=SimpleNamespace(onset=[],description=[]),close=lambda:None)


def native_data(path,picks,start,stop):
 """Exact16-bit digital-to-physical EDF calibration for bounded samples only."""
 h=header(path);fs=np.array(h['fs'])[picks];assert np.all(fs==fs[0]),'mixed_native_rates_not_silently_upsampled'
 ns=h['samples'][picks[0]];lo=start//ns;hi=(stop+ns-1)//ns;count=hi-lo
 assert 0<=lo<hi<=h['n_records']
 with path.open('rb') as f:
  f.seek(h['header_bytes']+lo*h['record_bytes']);blob=f.read(count*h['record_bytes'])
 assert len(blob)==count*h['record_bytes']
 buf=np.frombuffer(blob,dtype='<i2').reshape(count,sum(h['samples']));offsets=np.r_[0,np.cumsum(h['samples'])]
 arrays=[]
 for j in picks:
  units=h['unit'][j].lower().replace('\u00b5','u').replace('\u03bc','u').replace(' ','')
  factor={'v':1.,'mv':1e-3,'uv':1e-6,'nv':1e-9}.get(units);assert factor is not None,'unknown_EDF_physical_units'
  pmin,pmax,dmin,dmax=map(float,[h['pmin'][j],h['pmax'][j],h['dmin'][j],h['dmax'][j]])
  assert dmax>dmin and pmax>pmin
  digital=buf[:,offsets[j]:offsets[j+1]].reshape(-1).astype(float)
  arrays.append(((digital-dmin)*(pmax-pmin)/(dmax-dmin)+pmin)*factor)
 data=np.stack(arrays)[:,start-lo*ns:stop-lo*ns];assert data.shape==(len(picks),stop-start)
 return data,float(fs[0])


def freeze():
 p=lock();target=ROOT/'AUDIT_FREEZE_PRIVATE.json';assert not target.exists()
 disc=json.loads((ROOT/'public/DISCOVERY_AGGREGATE.json').read_text());private=json.loads((ROOT/'DISCOVERY_PRIVATE.json').read_text())
 inv={r['alias']:r for r in disc['cache_inventory']}
 assert sha(EXPORT)==p['A0_export_sha256'] and sha(SPLIT)==p['split_sha256']
 assert inv['A0_window']['sha256']==p['A0_window_sha256'] and inv['raw_primary']['sha256']==p['raw_primary_sha256'] and inv['raw_task2_other']['sha256']==p['other_raw_sha256']
 refs=json.loads((HERE/'INPUT_REFERENCE_HASHES.json').read_text())
 for r in refs:assert sha(ROOT/'input_references'/r['name'])==r['sha256']
 sources={name:sha(SRC/name) for name in disc['source_hashes']};assert sources==disc['source_hashes']
 with np.load(EXPORT,allow_pickle=True) as f:
  meta={n:f[n] for n in ['patient','channel','center','features','split_patient','split_fold','split_role']}
 assert len(set(meta['patient']))==80 and len(set(zip(meta['patient'],map(norm,meta['channel']))))==7635
 assert hashlib.sha256(json.dumps(meta['features'].tolist(),separators=(',',':')).encode()).hexdigest()==p['A0_feature_order_sha256']
 split=pd.read_csv(SPLIT,usecols=['subject_id','outer_fold','split_role'])
 assert set(zip(split.outer_fold,split.split_role.astype(str),split.subject_id.astype(str)))==set(zip(meta['split_fold'],meta['split_role'].astype(str),meta['split_patient'].astype(str)))
 assert all(set(split[(split.outer_fold==f)&(split.split_role==a)].subject_id).isdisjoint(split[(split.outer_fold==f)&(split.split_role==b)].subject_id) for f in range(1,6) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
 master={tuple(r['key']):r for r in private['A0_window']['records']};raw={tuple(r['key']):r for r in private['raw_primary']['records']};other={tuple(r['key']):r for r in private['raw_task2_other']['records']}
 assert len(master)==256 and set(master)==set(raw)==set(other)
 equal=sum(raw[k]['waveform_sha256']==other[k]['waveform_sha256'] for k in master)
 identical=sum(raw[k]['channel_names']==other[k]['channel_names'] and all(raw[k][v]==other[k][v] for v in ['raw_sfreq','raw_duration','raw_valid_start','raw_valid_samples','onset_sec','start_sec','end_sec']) for k in master)
 write(ROOT/'public/RAW_CACHE_IDENTITY_COMPARISON.json',dict(status='COMPLETE',records=256,identical_waveform_byte_hashes=equal,identical_time_channel_metadata=identical,other_raw_independent_lowfreq_source=False if equal==256 else None))
 plan=[];checked={};counts=defaultdict(Counter)
 for i,(k,r) in enumerate(master.items()):
  center=r['center'];row=dict(key=k,center=center,channels=r['channel_names'],onset=float(r['onset_sec']),source=None)
  paths=raw[k]['source_paths'];c=counts[center];c['records']+=1
  if len(paths)!=1:row['failure']='source_path_not_unique';c['source_path_not_unique']+=1;plan.append(row);continue
  path,method=resolve(paths[0]);row['resolution']=method
  if path is None:row['failure']='EDF_missing';c['EDF_missing']+=1;plan.append(row);continue
  try:
   if str(path) not in checked:
    h=header(path);e=quiet_edf(path);checked[str(path)]=(sha(path),h,float(e.info['sfreq']),int(e.n_times),list(e.ch_names),np.asarray(e.annotations.onset),list(e.annotations.description));e.close()
   digest,h,fs,n,enames,annotations,descriptions=checked[str(path)]
   hsel=[z for z,v in enumerate(h['label']) if v not in ['EDF Annotations','BDF Annotations']]
   cms=raw[k]['source_channel_meta'];picks=[];hp=[];lp=[];native=[];units=[]
   for name,cm in zip(r['channel_names'],cms):
    choices=[cm.get('raw_channel_name'),cm.get('channel_name_orig')]
    possible=[enames.index(v) for v in choices if v in enames]
    if not possible:
     possibles=[j for j,v in enumerate(enames) if norm(v)==name]
     possible=possibles if len(possibles)==1 else []
    assert possible and len(set(possible))==1,'channel_map_unproven';j=possible[0];picks.append(j)
    hj=hsel[j]
    hp.append(filter_number(h['prefilter'][hj],'HP'));lp.append(filter_number(h['prefilter'][hj],'LP'));native.append(h['fs'][hj]);units.append(h['unit'][hj])
   assert len(picks)==len(r['channel_names']) and len(set(picks))==len(picks)
   eventpaths=set(cm.get('events_path') for cm in cms if cm.get('events_path') and Path(cm['events_path']).is_file())
   channelpaths=set(cm.get('channels_path') for cm in cms if cm.get('channels_path') and Path(cm['channels_path']).is_file())
   onset_ok=False
   for ep in eventpaths:
    columns=pd.read_csv(ep,sep='\t',nrows=0).columns
    if 'onset' in columns:
     ev=pd.read_csv(ep,sep='\t',usecols=[s for s in ['onset','trial_type'] if s in columns]);tt=ev['trial_type'].astype(str).str.contains('seiz|sz|ictal|onset',case=False,regex=True) if 'trial_type' in ev else np.ones(len(ev),bool)
     onset_ok|=bool((np.abs(pd.to_numeric(ev.loc[tt,'onset'],errors='coerce')-row['onset'])<=p['paired_window_onset_tolerance_seconds']).any())
   onset_ok|=any(abs(float(t)-row['onset'])<=p['paired_window_onset_tolerance_seconds'] and re.search('seiz|sz|ictal|onset',str(s),re.I) for t,s in zip(annotations,descriptions))
   row.update(source=str(path),sha256=digest,sfreq=fs,n_samples=n,picks=picks,native_channel_sfreq=native,units=units,hp=hp,lp=lp,
      header_picks=[hsel[j] for j in picks],channels_tsv_available=bool(channelpaths),annotations_available=bool(eventpaths) or 'EDF Annotations' in h['label'],onset_independently_matched=bool(onset_ok))
   c['EDF_resolved']+=1;c['channels_mapped']+=len(picks);c['channels_tsv_records']+=int(bool(channelpaths));c['onset_matched']+=int(bool(onset_ok))
   c['hardware_filter_known_records']+=int(all(v is not None for v in hp+lp))
  except Exception as exc:row['failure']=type(exc).__name__;c['EDF_probe_failed']+=1
  plan.append(row)
  if (i+1)%16==0:print('EDF_METADATA_LOCK_PROGRESS',i+1,256,flush=True)
 atom(ROOT/'EDF_PLAN_PRIVATE.pkl',plan)
 rows=[dict(center=center,**v) for center,v in counts.items()];pd.DataFrame(rows).fillna(0).to_csv(ROOT/'public/EDF_METADATA_COVERAGE.csv',index=False)
 dependencies={'protocol':sha(HERE/'PROTOCOL_LOCK.json'),'references':sha(HERE/'INPUT_REFERENCE_HASHES.json'),'discovery':sha(ROOT/'DISCOVERY_PRIVATE.json'),'edf_plan':sha(ROOT/'EDF_PLAN_PRIVATE.pkl'),'export':sha(EXPORT),'split':sha(SPLIT),
  'code':{f.name:sha(f) for f in HERE.joinpath('code').glob('*.py')},'tests':{f.name:sha(f) for f in HERE.joinpath('tests').glob('*.py')},'source':sources}
 freeze=dict(status='PASS',dependencies=dependencies,cache_hashes={k:inv[k]['sha256'] for k in CANDIDATES},
   thresholds_frozen_before_PSD=True,full_A0_x_y_materialized=False,model_or_prediction_artifacts_opened=False,EDF_signal_PSD_run=False)
 if (ROOT/'RAW_PSD_PRIVATE.pkl').exists():
  original=ROOT/'initial_EDF_metadata_snapshot/AUDIT_FREEZE_PRIVATE.json'
  before=json.loads(original.read_text());assert before['dependencies']['protocol']==dependencies['protocol']
  freeze.update(metadata_only_engineering_repair=True,original_pre_PSD_freeze_sha256=sha(original),
   retained_raw_PSD_sha256=sha(ROOT/'RAW_PSD_PRIVATE.pkl'),raw_PSD_recomputed=False,
   EDF_annotation_TAL_not_parsed=True,EDF_channel_names='Exact original MNE default duplicate-name algorithm',
   EDF_native_reader='Bounded16-bit digital-to-physical calibration; synthetic MNE parity tested')
 write(target,freeze);write(ROOT/'public/AUDIT_FREEZE.json',freeze)
 print('FCPA_THRESHOLDS_SOURCE_AND_EDF_METADATA_FROZEN',flush=True)


def check():
 p=lock();fr=json.loads((ROOT/'AUDIT_FREEZE_PRIVATE.json').read_text());d=fr['dependencies']
 assert d['protocol']==sha(HERE/'PROTOCOL_LOCK.json') and d['references']==sha(HERE/'INPUT_REFERENCE_HASHES.json')
 assert d['code']=={f.name:sha(f) for f in HERE.joinpath('code').glob('*.py')} and d['tests']=={f.name:sha(f) for f in HERE.joinpath('tests').glob('*.py')}
 assert d['edf_plan']==sha(ROOT/'EDF_PLAN_PRIVATE.pkl') and d['discovery']==sha(ROOT/'DISCOVERY_PRIVATE.json')
 assert d['export']==sha(EXPORT) and d['split']==sha(SPLIT)
 assert all(sha(SRC/n)==v for n,v in d['source'].items())
 return p,fr


def raw():
 p,fr=check();assert not (ROOT/'RAW_PSD_PRIVATE.pkl').exists()
 with np.load(EXPORT,allow_pickle=True) as f:ids=set(f['patient'].astype(str));pairs=set(zip(f['patient'].astype(str),map(norm,f['channel'])))
 with CANDIDATES['A0_window'].open('rb') as f:fp=pickle.load(f)
 feature={key(r):r for r in fp['run_records'] if str(r['subject_id']) in ids}
 with CANDIDATES['raw_primary'].open('rb') as f:rp=pickle.load(f)
 rawmap={key(r):r for r in rp['run_records'] if key(r) in feature}
 assert len(feature)==len(rawmap)==256
 allrows=[];align=[];union=set();seen=set()
 for i,(k,r) in enumerate(feature.items()):
  rr=rawmap[k];s=rr['sample'];fs=float(s['raw_temporal_sfreq']);wave=np.asarray(s['raw_waveform']);names=tuple(norm(v) for v in r['channel_names_norm'])
  assert names==tuple(norm(v) for v in rr['channel_names_norm']) and len(names)==len(set(names))==wave.shape[0] and fs==250 and wave.shape[1]==15000
  start=int(s['raw_valid_start_sample']);end=start+int(s['raw_valid_samples']);assert 0<=start<end<=15000
  times=np.asarray(r['sample']['window_relative_centers_sec'],float);starts=np.rint((times+30)*fs).astype(int)-250
  valid=(starts>=start)&(starts+500<=end);indices=np.flatnonzero(valid);chunks=np.stack([wave[:,j:j+500] for j in starts[valid]],axis=1) if valid.any() else np.empty((len(names),0,500))
  powers=bands(chunks,fs,p) if valid.any() else {band:np.empty((len(names),0)) for band in p['bands']}
  fn=list(map(str,r['sample']['window_feature_names']));wf=np.asarray(r['sample']['window_features']);assert wf.shape==(len(times),len(names),len(fn)) and len(fn)==len(set(fn))
  named={name:np.expm1(wf[valid,:,fn.index('log_bp_'+name)].astype(float)).T for name in ['delta','theta','alpha']}
  grid=np.arange(0,14501,250);nominal_valid=(grid>=start)&(grid+500<=end)
  allrows.append(dict(key=k,patient=k[0],center=k[0].split(':')[0],names=names,times=times[valid],indices=indices,starts=starts[valid],powers=powers,old_named_power=named,
    valid_start=start,valid_end=end,old_feature_names=fn))
  align.append(dict(center=k[0].split(':')[0],patient=k[0],records=1,channels=len(names),feature_windows=len(times),complete_feature_windows=int(valid.sum()),
    invalid_feature_boundary_windows=int((~valid).sum()),nominal59_invalid_windows=int((~nominal_valid).sum()),padded_records=int(start>0 or end<15000)))
  union.update((k[0],n) for n in names);seen.add(k[0])
  if (i+1)%32==0:print('RAW_VALID_WINDOW_PSD_PROGRESS',i+1,256,flush=True)
 assert union==pairs and len(seen)==80
 atom(ROOT/'RAW_PSD_PRIVATE.pkl',allrows);atom(ROOT/'ALIGNMENT_PRIVATE.pkl',align)
 # Record-level data are private. Public statistics are aggregated later.
 assert sha(CANDIDATES['raw_primary'])==fr['cache_hashes']['raw_primary'] and sha(CANDIDATES['A0_window'])==fr['cache_hashes']['A0_window']
 write(ROOT/'public/RAW_PSD_STATUS.json',dict(status='COMPLETE',records=256,patients=80,canonical_channels=7635,Nyquist_hz=125,
   high_gamma80_150_reextracted=False,raw_PSD_sha256=sha(ROOT/'RAW_PSD_PRIVATE.pkl')))
 print('RAW_PSD_COMPLETE_NO_LABELS',flush=True)


def edf():
 p,fr=check()
 with (ROOT/'EDF_PLAN_PRIVATE.pkl').open('rb') as f:plan={tuple(r['key']):r for r in pickle.load(f)}
 with (ROOT/'RAW_PSD_PRIVATE.pkl').open('rb') as f:rawrows=pickle.load(f)
 import mne
 verified_sources=set()
 for i,r in enumerate(rawrows):
  path=ROOT/f'paired_private/record_{i:03d}.pkl'
  if path.exists():continue
  q=plan[r['key']];out=dict(key=r['key'],patient=r['patient'],center=r['center'],paired=False)
  if q.get('failure') or not q.get('source'):out['reason']=q.get('failure','EDF_missing');atom(path,out);continue
  try:
   if q['source'] not in verified_sources:
    assert sha(Path(q['source']))==q['sha256'];verified_sources.add(q['source'])
   h=header(Path(q['source']));rates=np.array(h['fs'])[q['header_picks']];assert np.all(rates==rates[0]),'mixed_native_rates_not_pairable'
   fs=float(rates[0]);absstarts=q['onset']+r['times']-1;absends=absstarts+2
   duration=h['n_records']*h['record_duration'];guard=p['EDF_filter_context_seconds'];valid=(absstarts>=guard)&(absends<=duration-guard)
   # Same-window subset defined solely by physical validity and filter context.
   if not valid.any():out['reason']='no_complete_guarded_matching_windows';atom(path,out);continue
   lo=max(0,int(np.floor((absstarts[valid].min()-guard)*fs)));hi=min(int(round(duration*fs)),int(np.ceil((absends[valid].max()+guard)*fs)))
   data,actual_fs=native_data(Path(q['source']),q['header_picks'],lo,hi);assert actual_fs==fs
   length=int(round(2*fs));ss=np.rint(absstarts[valid]*fs).astype(int)-lo
   arrays={'EDF_original':data}
   hfreq=min(100.,min(q['native_channel_sfreq'])/2-.5);assert hfreq>40
   with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for alias,hp in [('EDF_low05',.5),('EDF_reader1',1.),('EDF_negative10',10.)]:
     arrays[alias]=mne.filter.filter_data(data,fs,l_freq=hp,h_freq=hfreq,method='fir',phase='zero',fir_window='hamming',fir_design='firwin',verbose='ERROR',n_jobs=1)
   pow={alias:bands(np.stack([x[:,j:j+length] for j in ss],axis=1),fs,p) for alias,x in arrays.items()}
   out.update(paired=True,window_mask=valid,raw_powers={k:v[:,valid] for k,v in r['powers'].items()},EDF_powers=pow,old_alpha=r['old_named_power']['alpha'][:,valid],
      onset_independently_matched=q['onset_independently_matched'],hardware_hp=q['hp'],hardware_lp=q['lp'],channels_tsv_available=q['channels_tsv_available'],
      source_sha256=q['sha256'],native_channel_sfreq=q['native_channel_sfreq'])
   # Exact historical half-open alpha integral on the same native windows.
   wf=np.stack([data[:,j:j+length] for j in ss],axis=1)
   ff,ps=signal.welch(wf,fs=fs,nperseg=length,noverlap=length//2,detrend=False,axis=-1)
   mask=(ff>=10)&(ff<13);out['EDF_original_alpha10_halfopen']=np.trapezoid(ps[...,mask],ff[mask],axis=-1)
  except Exception as ex:out['reason']=type(ex).__name__;out['paired']=False
  atom(path,out)
  if (i+1)%16==0:print('EDF_PAIRED_SPECTRAL_PROGRESS',i+1,256,flush=True)
 write(ROOT/'public/EDF_PAIR_STATUS.json',dict(status='COMPLETE',records_examined=256,
  private_pair_files=256,correlations_are_signal_only=True,classifier_outputs_accessed=False))
 print('EDF_PAIRED_DIAGNOSTICS_COMPLETE',flush=True)


def summary(values,total,p):
 a=np.asarray(values,float).ravel();t=np.asarray(total,float).ravel();good=np.isfinite(a);v=a[good]
 ratio=a/np.maximum(t,p['denominator_epsilon'])
 return dict(candidate_values=len(a),finite_fraction=float(good.mean()) if len(a) else None,
  q25=float(np.percentile(v,25)) if len(v) else None,median=float(np.median(v)) if len(v) else None,
  q75=float(np.percentile(v,75)) if len(v) else None,maximum=float(v.max()) if len(v) else None,
  below_relative_operational_noise_floor_rate=float(np.mean(ratio[good]<=p['relative_operational_noise_floor'])) if len(v) else None,
  below_assumed_V2_floor_rate=float(np.mean(v<=p['assumed_volts_squared_operational_floor'])) if len(v) else None,
  median_band_fraction=float(np.nanmedian(ratio)) if len(v) else None)


def finish():
 p,fr=check();pub=ROOT/'public';assert not (pub/'DATA_AVAILABILITY_GATE.json').exists()
 with (ROOT/'RAW_PSD_PRIVATE.pkl').open('rb') as f:rr=pickle.load(f)
 with (ROOT/'ALIGNMENT_PRIVATE.pkl').open('rb') as f:alignment=pd.DataFrame(pickle.load(f))
 with (ROOT/'EDF_PLAN_PRIVATE.pkl').open('rb') as f:plan=pickle.load(f)
 pairs=[]
 for i in range(256):
  with (ROOT/f'paired_private/record_{i:03d}.pkl').open('rb') as f:pairs.append(pickle.load(f))
 rows=[];continuity=[];groupnames=['ALL']+sorted(set(r['center'] for r in rr))
 for group in groupnames:
  subset=[r for r in rr if group=='ALL' or r['center']==group]
  for band in ['delta','theta','alpha','beta','alpha_low','alpha_high']:
   power=np.concatenate([r['powers'][band].ravel() for r in subset]);total=np.concatenate([r['powers']['total'].ravel() for r in subset])
   rows.append(dict(center=group,aggregation='all_complete_channel_windows',band=band,patients=len(set(r['patient'] for r in subset)),records=len(subset),**summary(power,total,p)))
   med=[];tot=[]
   for patient in sorted(set(r['patient'] for r in subset)):
    part=[r for r in subset if r['patient']==patient];med.append(np.nanmedian(np.concatenate([r['powers'][band].ravel() for r in part])));tot.append(np.nanmedian(np.concatenate([r['powers']['total'].ravel() for r in part])))
   rows.append(dict(center=group,aggregation='equal_patient_medians',band=band,patients=len(med),records=len(subset),**summary(med,tot,p)))
  low=np.concatenate([r['powers']['alpha_low'].ravel()/2 for r in subset]);high=np.concatenate([r['powers']['alpha_high'].ravel()/3 for r in subset]);ratio=low/np.maximum(high,p['denominator_epsilon'])
  continuity.append(dict(center=group,ratio='mean_PSD8_10_over10_13',q25=np.nanpercentile(ratio,25),median=np.nanmedian(ratio),q75=np.nanpercentile(ratio,75),below_locked_floor_fraction=np.mean(ratio<p['continuity_low_to_high_mean_PSD_ratio_floor'])))
 pd.DataFrame(rows).to_csv(pub/'RAW_CACHE_BANDPOWER_SUMMARY.csv',index=False);pd.DataFrame(continuity).to_csv(pub/'SPECTRAL_CONTINUITY.csv',index=False)
 covered=[q for q in pairs if q['paired']];parity=[]
 for group in groupnames:
  subset=[q for q in covered if group=='ALL' or q['center']==group]
  for source in ['EDF_original','EDF_low05','EDF_reader1','EDF_negative10']:
   for band in ['delta','theta','alpha','beta']:
    if not subset:parity.append(dict(center=group,reference=source,band=band,status='NOT_RUN_NO_PAIRED_WINDOWS',pairs=0));continue
    a=np.concatenate([q['raw_powers'][band].ravel() for q in subset]);b=np.concatenate([q['EDF_powers'][source][band].ravel() for q in subset]);ok=np.isfinite(a)&np.isfinite(b)&(b>p['denominator_epsilon'])
    ratio=a[ok]/b[ok];af=np.concatenate([q['raw_powers'][band].ravel()/np.maximum(q['raw_powers']['total'].ravel(),p['denominator_epsilon']) for q in subset]);bf=np.concatenate([q['EDF_powers'][source][band].ravel()/np.maximum(q['EDF_powers'][source]['total'].ravel(),p['denominator_epsilon']) for q in subset])
    corr=stats.spearmanr(a[ok],b[ok]).statistic if ok.sum()>1 and np.ptp(a[ok]) and np.ptp(b[ok]) else np.nan
    parity.append(dict(center=group,reference=source,band=band,status='MEASURED_SIGNAL_ONLY',records=len(subset),pairs=len(a),finite_pair_fraction=float(ok.mean()),
       raw_over_EDF_q25=np.percentile(ratio,25),raw_over_EDF_median=np.median(ratio),raw_over_EDF_q75=np.percentile(ratio,75),spearman=corr,
       raw_band_fraction_median=np.nanmedian(af),EDF_band_fraction_median=np.nanmedian(bf),independent_onset_records=sum(q['onset_independently_matched'] for q in subset)))
 pd.DataFrame(parity).to_csv(pub/'RAW_VS_EDF_PARITY_SUMMARY.csv',index=False)
 alphas=[]
 for group in groupnames:
  subset=[q for q in covered if group=='ALL' or q['center']==group]
  if not subset:alphas.append(dict(center=group,status='NOT_RUN_NO_PAIRED_WINDOWS'));continue
  old=np.concatenate([q['old_alpha'].ravel() for q in subset]);partial=np.concatenate([q['EDF_original_alpha10_halfopen'].ravel() for q in subset]);full=np.concatenate([q['EDF_powers']['EDF_original']['alpha'].ravel() for q in subset]);valid=(partial>p['denominator_epsilon'])&(full>p['denominator_epsilon'])&np.isfinite(old)
  alphas.append(dict(center=group,status='MEASURED_NAME_INDEXED',pairs=len(old),old_window_alpha_over_EDF_full_median=np.median(old[valid]/full[valid]),old_window_alpha_over_EDF_10_halfopen_median=np.median(old[valid]/partial[valid]),
    old_vs_EDF_full_spearman=stats.spearmanr(old[valid],full[valid]).statistic,old_vs_EDF_10_halfopen_spearman=stats.spearmanr(old[valid],partial[valid]).statistic))
 pd.DataFrame(alphas).to_csv(pub/'NAMED_ALPHA_PARITY_SUMMARY.csv',index=False)
 alignout=[]
 for group in groupnames:
  sub=alignment if group=='ALL' else alignment[alignment.center==group];recordplans=[q for q in plan if group=='ALL' or q['center']==group];paired=[q for q in covered if group=='ALL' or q['center']==group]
  alignout.append(dict(center=group,patients=sub.patient.nunique(),records=len(sub),seizure_channel_incidences=int(sub.channels.sum()),feature_windows=int(sub.feature_windows.sum()),
    valid_complete_feature_windows=int(sub.complete_feature_windows.sum()),invalid_feature_boundary_windows=int(sub.invalid_feature_boundary_windows.sum()),
    nominal59_invalid_windows=int(sub.nominal59_invalid_windows.sum()),padded_records=int(sub.padded_records.sum()),EDF_resolved_records=sum(q.get('source') is not None and not q.get('failure') for q in recordplans),
    EDF_paired_records=len(paired),EDF_pair_record_rate=len(paired)/len(sub),EDF_onset_independently_matched=sum(q.get('onset_independently_matched',False) for q in recordplans)))
 pd.DataFrame(alignout).to_csv(pub/'ALIGNMENT_COVERAGE.csv',index=False)
 spec=[]
 for band in ['delta','theta','alpha']:
  vals=np.concatenate([r['old_named_power'][band].ravel() for r in rr]);spec.extend([
   dict(band=band,source='A0_88D',status='ABSENT_FROM_PRUNED' if band=='alpha' else 'HISTORICAL16_DELTA_THETA_COLUMNS_CONSTANT_ZERO',evidence='historical_name_manifest_and_frozen_export_hash'),
   dict(band=band,source='Window_cache',status='MEASURED_BY_NAME',feature_name='log_bp_'+band,nonzero_fraction=float(np.mean(vals!=0)),finite_fraction=float(np.isfinite(vals).mean()),
    definition='10<=f<13 candidate under historical spectral_min10; not full alpha' if band=='alpha' else 'zero compatible with no retained PSD bins, not evidence of absent raw signal'),
   dict(band=band,source='Raw_primary',status='MEASURED_WELCH_VALID_WINDOWS_PROVENANCE_UNPROVEN',frequency_lo=p['bands'][band][0],frequency_hi=p['bands'][band][1]),
   dict(band=band,source='EDF',status='PARTIAL_MATCHED_COVERAGE_AND_DEVICE_FILTER_AUDIT',paired_records=len(covered))])
 pd.DataFrame(spec).to_csv(pub/'FEATURE_SPEC_COMPARISON.csv',index=False)
 boundary=int(alignment.nominal59_invalid_windows.sum())
 # A count alone cannot identify or replay a historical mask. Only an exact
 # private mask ledger with matching record identity and input hashes may pass.
 historical_mask=ROOT/'HISTORICAL27_MASK_PRIVATE.pkl';maskok=False
 if historical_mask.exists():
  with historical_mask.open('rb') as f:hm=pickle.load(f)
  maskok=hm.get('raw_sha256')==p['raw_primary_sha256'] and hm.get('feature_sha256')==p['A0_window_sha256'] and hm.get('invalid_windows')==27 and hm.get('identity_replay_pass') is True
 fullmap=len(covered)==256 and all(q.get('onset_independently_matched') for q in plan)
 hwfull=all(q.get('hp') and all(h is not None and h<=.5 for h in q['hp']) and all(v is not None and v>=30 for v in q['lp']) for q in plan)
 hwalpha=all(q.get('hp') and all(h is not None and h<=8 for h in q['hp']) and all(v is not None and v>=30 for v in q['lp']) for q in plan)
 goodbands=True
 for band in ['delta','theta','alpha']:
  hits=[q for q in parity if q['center']=='ALL' and q['reference']=='EDF_low05' and q['band']==band]
  goodbands &= bool(hits and hits[0].get('spearman',-1)>=.9 and .5<=hits[0].get('raw_over_EDF_median',-1)<=2)
 disc=json.loads((pub/'DISCOVERY_AGGREGATE.json').read_text());inv=next(q for q in disc['cache_inventory'] if q['alias']=='raw_primary')
 cfg=inv.get('config',{})
 provenance=bool(cfg.get('reader_bandpass_low') is not None and cfg.get('reader_bandpass_high') is not None and cfg.get('source_code_sha256') in disc['source_hashes'].values())
 if fullmap and hwfull and maskok and goodbands and provenance:decision='GO_RAW_CACHE'
 elif fullmap and hwfull and maskok:decision='GO_EDF_REEXTRACT_ONLY'
 elif fullmap and hwalpha and maskok:decision='PARTIAL_ALPHA_ONLY'
 else:decision='STOP_DATA_NOT_IDENTIFIABLE'
 write(pub/'DATA_AVAILABILITY_GATE.json',dict(decision=decision,raw_exact_generation_provenance='VERIFIED' if provenance else 'PROVENANCE_NOT_PROVEN',
  complete_independent_EDF_time_channel_mapping=fullmap,full_lowfreq_hardware_filter_contract=hwfull,partial_alpha_hardware_contract=hwalpha,
  historical27_mask_reconciled=maskok,historical_mask_ledger_available=historical_mask.exists(),observed_nominal_boundary_invalid_windows=boundary,paired_delta_theta_alpha_numerical_gate=bool(goodbands),
  raw_identity_complete=True,raw_finite_rate_all_bands=float(min(q['finite_fraction'] for q in rows if q['aggregation']=='all_complete_channel_windows')),
  data_quality_patient_selection=False,thresholds_changed_after_PSD=False,training_performed=False,classifier_metrics_or_outer_results_used=False,
  A0_feature_export_unchanged=sha(EXPORT)==p['A0_export_sha256'],split_unchanged=sha(SPLIT)==p['split_sha256'],signals_and_individual_details_private=True))
 print('FCPA_DATA_AUDIT_COMPLETE',decision,flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','raw','edf','finish']);a=parser.parse_args()
 {'freeze':freeze,'raw':raw,'edf':edf,'finish':finish}[a.phase]()
