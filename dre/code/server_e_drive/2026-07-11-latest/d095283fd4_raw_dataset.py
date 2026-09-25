from __future__ import annotations
from collections import defaultdict
import numpy as np, torch
from ..functional_graph import normalize_channel_name
from .schema import PatientSample
from .preprocessing import preprocess_window
PHASES=((0,-10.,-1.),(1,0.,5.),(2,5.,15.))
def _signal(record):
 x=np.asarray(record['sample']['raw_waveform'],float); n=len(record['channel_names_norm']); return x if x.shape[0]==n else x.T
def build_patients(raw_records,outcomes):
 grouped=defaultdict(list); audits=[]
 for record in raw_records:
  patient=str(record['subject_id']);
  if patient not in outcomes: continue
  names=[normalize_channel_name(x) for x in record['channel_names_norm']]; labels=np.asarray(record.get('labels'),float)
  if len(names)!=len(labels): audits.append({'patient_key':patient,'seizure_id':str(record['run_id']),'valid':False,'reason':'label_channel_length_mismatch'}); continue
  valid=np.isin(labels,(0,1)); ez=(labels==0)&valid; nez=(labels==1)&valid
  if not ez.any() or not nez.any(): audits.append({'patient_key':patient,'seizure_id':str(record['run_id']),'valid':False,'reason':'missing_ez_or_nez'}); continue
  sample=record['sample']; onset=float(sample.get('seizure_onset_sec',0.)); start=float(sample.get('start_sec',0.)); fs=float(record['sfreq']); grouped[patient].append({'seizure_id':str(record['run_id']),'signal':_signal(record),'fs':fs,'onset_sample':int(round((onset-start)*fs)),'names':names,'labels':labels,'valid':valid}); audits.append({'patient_key':patient,'seizure_id':str(record['run_id']),'valid':True,'reason':''})
 result=[]
 for patient,runs in grouped.items():
  if runs: result.append(PatientSample(patient,runs[0].get('center',patient.split(':')[0]),int(outcomes[patient]),runs))
 return result,audits
def window_index(run):
 fs=run['fs']; onset=run['onset_sample']; n=run['signal'].shape[1]; result=[]
 for phase,lo,hi in PHASES:
  a=max(0,int(round(onset+lo*fs))); b=min(n,int(round(onset+hi*fs))); width=int(round(2*fs)); step=max(1,int(round(fs)))
  for s in range(a,max(a,b-width+1),step): result.append((phase,s,s+width))
 return result
def patient_tensors(patient,device='cpu',rng=None,max_seizures=None,max_ez=None,max_nez=None,max_windows=None):
 runs=list(patient.seizures)
 if rng is not None: rng.shuffle(runs)
 if max_seizures: runs=runs[:max_seizures]
 output=[]
 for run in runs:
  ez=np.where((run['labels']==0)&run['valid'])[0]; nez=np.where((run['labels']==1)&run['valid'])[0]
  if rng is not None:
   rng.shuffle(ez); rng.shuffle(nez)
  if max_ez: ez=ez[:max_ez]
  if max_nez: nez=nez[:max_nez]
  windows=[]; phases=[]; channels=[]; labels=[]
  for ci in np.r_[ez,nez]:
   per=defaultdict(list)
   for phase,a,b in window_index(run): per[phase].append((a,b))
   for phase,items in per.items():
    if rng is not None: rng.shuffle(items)
    if max_windows: items=items[:max_windows]
    for a,b in items:
     x,ok=preprocess_window(run['signal'][ci,a:b],run['fs'])
     if ok: windows.append(x); phases.append(phase); channels.append(int(ci)); labels.append(int(run['labels'][ci]))
  if windows: output.append((torch.tensor(np.asarray(windows))[:,None].to(device),torch.tensor(phases,device=device),torch.tensor(channels,device=device),torch.tensor(labels,device=device)))
 return output
