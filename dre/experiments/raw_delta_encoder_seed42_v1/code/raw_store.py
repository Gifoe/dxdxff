"""Reverify existing waveform-only banks against exact compound Raw source slices."""
import argparse
import gc
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import torch
from io_utils import sha,read,save,write
from temporal_shuffle import permutation

def prepare(source,runtime):
 source=Path(source);runtime=Path(runtime);d=read(runtime/'RECONSTRUCTION_PRIVATE.pkl');gate=json.loads((runtime/'results/STAGE_B_GATE.json').read_text());assert gate['pass_gate']
 historical=Path(r'D:\a0_compact_rawfirst_ablation_seed42_v1_runtime');regpath=historical/'RAW_REGISTRY_PRIVATE.json';audit=json.loads((historical/'public/RAW_ALIGNMENT_AUDIT.json').read_text());assert audit['status']=='PASS' and sha(regpath)==audit['registry_sha256']
 registry=json.loads(regpath.read_text());rawpath=Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl');assert sha(rawpath)==d['binding']['raw']==registry['raw_sha256']
 binding=dict(Stage_B=sha(runtime/'STAGE_B_MODEL_FREEZE_PRIVATE.json'),Stage_A=sha(runtime/'results/STAGE_A_GATE.json'),historical_raw_registry=sha(regpath),raw_sha256=sha(rawpath),code={f:sha(Path(__file__).parent/f) for f in ['raw_store.py','temporal_shuffle.py']})
 final=runtime/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'
 if final.exists():q=read(final);assert q['binding']==binding;print('DYNAMIC_RAW_STORE_ALREADY_VERIFIED',flush=True);return
 with rawpath.open('rb') as f:payload=pickle.load(f)
 sys.path.insert(0,str(source));from neuroez_c.dual_view_data import _record_key,_channel_names,_field
 from neuroez_c.raw_brainbert_data import normalize_channel_name
 rawrecords={_record_key(r):r for r in payload['run_records']};files={r['patient']:r for r in registry['files']};plans={};windows=eligible=changed=0
 for n,ledger in enumerate(d['ledgers'],1):
  pid=ledger['patient'];r=files[pid];bp=historical/'raw_banks'/r['filename'];assert sha(bp)==r['sha256'];b=torch.load(bp,map_location='cpu',weights_only=False)
  assert b['patient']==pid and b['channels']==ledger['channels'] and b['records']==len(ledger['records']);raw=b['raw'][0].numpy();mask=b['mask'][0].numpy();channels=b['channels'];norm=[normalize_channel_name(c) for c in channels];W=raw.shape[2];times=np.full((len(ledger['records']),W),np.nan);shuffles=[]
  for s,rec in enumerate(ledger['records']):
   source_rec=rawrecords[rec['compound_key']];wave=np.asarray(_field(source_rec,'raw_waveform'));assert wave.dtype==np.float32;lookup={c:j for j,c in enumerate(_channel_names(source_rec))};cn={c:j for j,c in enumerate(norm)};expected=np.zeros_like(mask[s]);t=rec['times'];times[s,:len(t)]=t
   for j,c in enumerate(rec['names']):
    target=cn[c];valid=rec['mask'][:,j];expected[target,:len(t)]=valid
    for w in np.flatnonzero(valid):assert np.array_equal(raw[s,target,w],wave[lookup[c],rec['starts'][w]:rec['starts'][w]+500])
   assert np.array_equal(mask[s],expected) and np.all(raw[s][~mask[s]]==0)
   post=np.flatnonzero(expected.any(0)&(times[s]>=0)&(times[s]<20));pre=expected.any(0)&(times[s]<0);assert pre.any() and len(post)>0
   pp=np.stack([permutation(pid,rec['compound_key'],c,times[s,post]) for c in channels]);shuffles.append(dict(post=post,perm=pp,compound_key=rec['compound_key']))
   for c in np.flatnonzero(expected.any(1)):
    assert expected[c,post].all() and expected[c,pre].all()
    for low,high in [(0,8),(8,20)]:
     ix=np.flatnonzero((times[s,post]>=low)&(times[s,post]<high))
     if len(ix)>=2:eligible+=1;changed+=int(not np.array_equal(pp[c,ix],ix))
   windows+=int(expected.sum())
  assert np.isfinite(raw).all();plans[pid]=dict(path=str(bp),sha256=sha(bp),channels=channels,times=times,shuffles=shuffles,records=len(ledger['records']))
  print('RAW_DYNAMIC_SOURCE_PARITY',n,'of80',flush=True);del b,raw,mask
 del payload,rawrecords;gc.collect();assert windows==1471965 and eligible==changed
 save(final,dict(binding=binding,plans=plans));write(runtime/'results/RAW_DYNAMIC_ALIGNMENT.json',dict(status='PASS',patients=80,canonical_channels=7635,seizures=256,valid_channel_windows=windows,whole_raw_source_hash_checked=True,every_existing_bank_sample_reverified=True,padded_entries_zero_and_masked=True,stage_permutations_eligible=eligible,stage_permutations_changed=changed,no_labels_used=True,no_Outer_inference=True,registry_sha256=sha(final)))

class Store:
 def __init__(self,runtime):self.runtime=Path(runtime);self.registry=read(self.runtime/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl');self.cache={}
 def get(self,patient,channels,device):
  patient=str(patient);r=self.registry['plans'][patient]
  if patient not in self.cache:
   bp=Path(r['path']);assert sha(bp)==r['sha256'];b=torch.load(bp,map_location='cpu',weights_only=False);assert b['patient']==patient and b['channels']==r['channels'];self.cache[patient]=b
  b=self.cache[patient];assert set(channels)==set(b['channels']) and len(channels)==len(b['channels']);ix=[b['channels'].index(str(c)) for c in channels];perms=[torch.tensor(s['perm'][ix],device=device) for s in r['shuffles']]
  index=torch.tensor(ix);return b['raw'][0].index_select(1,index).to(device),b['mask'][0].index_select(1,index).to(device),torch.tensor(r['times'],dtype=torch.float32,device=device),perms

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args();prepare(a.source,a.runtime)
