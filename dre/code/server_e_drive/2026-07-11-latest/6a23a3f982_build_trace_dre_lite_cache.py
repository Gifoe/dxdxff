from __future__ import annotations
import argparse,json,pickle,sys,time
from pathlib import Path
from collections import defaultdict,Counter
import pandas as pd
PROJECT=Path(__file__).resolve().parents[2];REPO=PROJECT.parent
for p in (REPO,PROJECT):sys.path.insert(0,str(p))
from neuroez_c.task2.data import load_cache,filtered_cache
from neuroez_c.task2.exclusions import load_exclusion_manifest
from neuroez_c.task2.outcomes import load_outcome_table
from neuroez_c.task2.trace_dre_lite.cache_builder import make_signature,signature_hash,build_patient,verify_cache,atomic_json
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-cache',required=True);p.add_argument('--outcome-table',required=True);p.add_argument('--exclusion-manifest',required=True);p.add_argument('--cache-dir',required=True);p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--verify-only',action='store_true');p.add_argument('--max-patients',type=int);a=p.parse_args();root=Path(a.cache_dir);root.mkdir(parents=True,exist_ok=True)
 if a.verify_only:print(json.dumps(verify_cache(root),indent=2));return 0
 start=time.time();sig=make_signature(a.raw_cache,a.exclusion_manifest,a.outcome_table);sig['signature_hash']=signature_hash(sig);sp=root/'cache_signature.json'
 if sp.exists() and json.loads(sp.read_text())['signature_hash']!=sig['signature_hash']:raise RuntimeError('TRACE-DRE cache signature mismatch; refusing reuse')
 atomic_json(sp,sig);source=load_cache(a.raw_cache);cache=filtered_cache(source,load_exclusion_manifest(a.exclusion_manifest));outcomes,_=load_outcome_table(a.outcome_table,cache=cache);labels=outcomes[outcomes.outcome_group.isin(['success','failure'])].set_index('patient_key').outcome_label.astype(int).to_dict();group=defaultdict(list)
 for r in cache['run_records']:
  if str(r['subject_id']) in labels:group[str(r['subject_id'])].append(r)
 keys=sorted(group)[:a.max_patients] if a.max_patients else sorted(group);manifest=[];reject_w=[];reject_c=[];hits=0
 for i,key in enumerate(keys,1):
  shard,rw,rc,hit=build_patient(key,group[key],labels[key],root,a.resume);hits+=hit;obj=__import__('torch').load(shard,map_location='cpu',weights_only=False);manifest.append({'patient_key':key,'center':obj['center'],'outcome_success':obj['outcome_success'],'n_seizures':len(obj['seizures']),'shard_path':str(shard.relative_to(root))});reject_w+=rw;reject_c+=rc;print(f'[TRACE-DRE cache] {i}/{len(keys)} {key} '+('HIT' if hit else 'BUILT'),flush=True)
 pd.DataFrame(manifest).to_csv(root/'cache_manifest.csv',index=False);pd.DataFrame(reject_w).to_csv(root/'rejected_windows.csv',index=False);pd.DataFrame(reject_c).to_csv(root/'rejected_channels.csv',index=False);n_channels=n_windows=0;phase_counts=Counter();center_counts=Counter();outcome_counts=Counter()
 for item in manifest:
  obj=__import__('torch').load(root/item['shard_path'],map_location='cpu',weights_only=False);center_counts[obj['center']]+=1;outcome_counts[str(obj['outcome_success'])]+=1
  for s in obj['seizures']:n_channels+=len(s['channel_names']);n_windows+=int(s['window_mask'].sum());phase_counts.update({str(p):int(((s['phase_ids']==p).unsqueeze(0)&s['window_mask']).sum()) for p in range(3)})
 elapsed=time.time()-start;audit={'n_patients':len(manifest),'n_seizures':sum(x['n_seizures'] for x in manifest),'n_channels':n_channels,'n_windows':n_windows,'invalid_windows':len(reject_w),'invalid_channels':len(reject_c),'cache_hits':hits,'elapsed_seconds':elapsed,'windows_per_second':n_windows/max(elapsed,1e-9),'cache_size_gb':sum(x.stat().st_size for x in (root/'patients').glob('*.pt'))/2**30,'center_counts':json.dumps(center_counts),'outcome_counts':json.dumps(outcome_counts),'phase_window_counts':json.dumps(phase_counts)};pd.DataFrame([audit]).to_csv(root/'cache_build_audit.csv',index=False);report=verify_cache(root,min(10,len(keys)));print(json.dumps({**audit,'verify':report},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
