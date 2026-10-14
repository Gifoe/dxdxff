"""Exact cached-window audit and label-free182D tables. No global y read."""
import argparse
from collections import defaultdict
import json
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from common import sha,digest,save,read,write_json
from validated_alignment import extract
from construct_patient_context import construct
from synchronization_controls import shift
from summarize_temporal_features import summarize,across_seizures,schema,STAGES

def options():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));p.add_argument('--phase',choices=['data','probe','evaluate','all'],default='all');return p.parse_args()

def setup(a):
 root=Path(__file__).parents[1];a.root=root;lock=json.loads((root/'PROTOCOL_LOCK.json').read_text());a.pub=a.runtime/'results';a.pub.mkdir(parents=True,exist_ok=True)
 paths=dict(cache=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl'),split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv'),archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))
 for k,l in [('cache','feature_cache_sha256'),('split','split_sha256'),('archive','feature_archive_sha256')]:assert sha(paths[k])==lock[l]
 assert sha(root/'code/validated_alignment.py')==lock['validated_alignment_sha256']==sha(a.source/'a0_temporal_cotar_seed42_v1/code/features.py')
 binding=dict(inputs={k:sha(v) for k,v in paths.items()},code={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*.py'))},protocol=sha(root/'PROTOCOL_LOCK.json'),historical_review=sha(root/'HISTORICAL_OVERLAP_AUDIT.md'))
 fr=a.runtime/'INPUT_FREEZE_PRIVATE.json'
 if fr.exists():
  original=json.loads(fr.read_text())['binding']
  if original!=binding:
   repair=json.loads((root/'ENGINEERING_REPAIR_LOCK.json').read_text())
   assert repair['scope']=='FIT_union_count_and_numpy_batch_shape_only'
   assert all(original[k]==binding[k] for k in ['inputs','protocol','historical_review'])
   assert set(original['code'])==set(binding['code'])
   changes={k for k in original['code'] if original['code'][k]!=binding['code'][k]}
   assert changes==set(repair['files'])
   for k in changes:assert repair['files'][k]==dict(before=original['code'][k],after=binding['code'][k])
   binding=original  # Keep all original table/model/prediction/threshold seals.
 else:write_json(fr,dict(binding=binding,status='PASS',no_labels_used=True))
 source_report=dict(status='PASS',binding=binding)
 if (root/'ENGINEERING_REPAIR_LOCK.json').exists():source_report['engineering_repair']=json.loads((root/'ENGINEERING_REPAIR_LOCK.json').read_text())
 write_json(a.pub/'SOURCE_HASHES.json',source_report)
 return lock,paths,binding

def aggregate(rows,keys,fields):
 df=pd.DataFrame(rows);out=[]
 for kk,g in df.groupby(keys,dropna=False):
  kk=kk if isinstance(kk,tuple) else (kk,);n=g.patient.nunique();out.append(dict(zip(keys,kk),rows=len(g),unique_patients=n,status='AGGREGATE' if n>=3 else 'SUPPRESSED_SMALL_GROUP',**{f:float(g[f].mean()) if n>=3 else None for f in fields}))
 return out

def data(a,lock,paths,binding):
 target=a.runtime/'CONTEXT_TABLES_PRIVATE.pkl'
 if target.exists():
  q=read(target);assert q['binding']==binding;return
 # Metadata members only; no global feature matrix or clinical label member.
 with np.load(paths['archive'],allow_pickle=True) as f:meta={k:f[k].astype(str) for k in ['patient','channel','center','split_patient','split_role']};meta['split_fold']=f['split_fold']
 assert len(meta['patient'])==7635 and len(set(meta['patient']))==80 and len(set(zip(meta['patient'],meta['channel'])))==7635
 fixed=pd.read_csv(paths['split']);assert set(fixed.subject_id.astype(str))==set(meta['patient']);folds={}
 for fold in range(1,6):
  take=meta['split_fold']==fold;roles={r:set(meta['split_patient'][take&(meta['split_role']==r)]) for r in ['fit','validation','test']}
  assert all(roles[u].isdisjoint(roles[v]) for u,v in [('fit','validation'),('fit','test'),('validation','test')])
  for r,ids in roles.items():assert ids==set(fixed[(fixed.outer_fold==fold)&(fixed.split_role==r)].subject_id.astype(str))
  assert len(roles['fit'])==lock['cohort']['FIT_counts'][fold-1];folds[fold]=roles
 sys.path.insert(0,str(a.source))
 with paths['cache'].open('rb') as f:payload=pickle.load(f)
 # Trusted supplied monolithic pickle may deserialize label/raw fields. No such
 # field is indexed or used; only named feature/identity/valid-range metadata.
 bypatient=defaultdict(list);keys=set()
 for rec in payload['run_records']:
  pid=str(rec['subject_id'])
  if pid not in set(meta['patient']):continue
  key=(pid,str(rec['run_id']),str(rec['sample']['sample_id']));assert key not in keys;keys.add(key);bypatient[pid].append(rec)
 assert len(keys)==256 and set(bypatient)==set(meta['patient'])
 tables={};coverage=[];disruption=[];ledger=[];source_hist=[];early_any=[];patient_info=[];summary_changes=[];maps=set()
 for pid in sorted(bypatient):
  ix=np.flatnonzero(meta['patient']==pid);canonical=meta['channel'][ix].tolist();center=meta['center'][ix[0]];records=sorted(bypatient[pid],key=lambda r:(str(r['run_id']),str(r['sample']['sample_id'])))
  base=[];own=[];real=[];wrong=[];seenearly=np.zeros(len(ix),bool);counts=[]
  channel_stratum='<=64' if len(ix)<=64 else ('65-128' if len(ix)<=128 else '>128');seizure_stratum=str(len(records)) if len(records)<3 else '3+'
  for rec in records:
   names=list(map(str,rec['channel_names_norm']));assert len(names)==len(set(names));sample=rec['sample'];fn=list(sample['window_feature_names']);fi=[fn.index(n) for n in lock['features']];maps.add(tuple((n,int(j)) for n,j in zip(lock['features'],fi)))
   _,legacy_t,legacy_mask,detail=extract(rec,names)
   t=np.asarray(sample['window_relative_centers_sec'],dtype=float);order=np.argsort(np.where(np.isfinite(t),t,np.inf),kind='stable');t=t[order];select=np.isfinite(t)&(t>=-10)&(t<20)
   x=np.asarray(sample['window_features'],dtype=np.float64)[order][select][...,fi];t=t[select];mask=legacy_mask.T
   assert x.shape==(len(t),len(names),7) and np.array_equal(np.broadcast_to(t.astype(np.float32),legacy_t.shape),legacy_t)
   ctx=construct(x,t,mask);shifted,us,diag=shift(ctx['d'],ctx['m'],t,names,[pid,str(rec['run_id']),str(sample['sample_id'])])
   dsum,dc=summarize(ctx['d'],t);rsum,uc=summarize(ctx['u'],t);ssum,sc=summarize(us,t);assert np.array_equal(uc,sc)
   rbase=np.full((len(ix),7),np.nan);rd=np.full((len(ix),42),np.nan);ru=rd.copy();rw=rd.copy();bc=np.zeros((len(ix),7),int);ddc=np.zeros((len(ix),2,7),int);uuc=ddc.copy();lookup={n:j for j,n in enumerate(names)};queryidx={lookup[n] for n in canonical if n in lookup}
   for j,n in enumerate(canonical):
    if n not in lookup:
     for stage,lo,hi in [('pre',-10,0),*STAGES]:
      for f in lock['features']:coverage.append(dict(patient=pid,center=center,stage=stage,feature=f,channel_count_stratum=channel_stratum,seizure_count_stratum=seizure_stratum,baseline_available=False,self_windows=0,context_windows=0,insufficient_other_windows=0,record_channel_observed=False))
     continue
    src=lookup[n];rbase[j]=ctx['b'][src];rd[j]=dsum[src];ru[j]=rsum[src];rw[j]=ssum[src];bc[j]=ctx['baseline_counts'][src];ddc[j]=dc[src];uuc[j]=uc[src];seenearly[j]|=bool((dc[src,0]>=2).all())
    for stage,lo,hi in [('pre',-10,0),*STAGES]:
     for k,f in enumerate(lock['features']):
      at=(t>=lo)&(t<hi);observed=np.isfinite(ctx['d'][at,src,k]);context=np.isfinite(ctx['m'][at,src,k]);coverage.append(dict(patient=pid,center=center,stage=stage,feature=f,channel_count_stratum=channel_stratum,seizure_count_stratum=seizure_stratum,baseline_available=bool(bc[j,k]>0),self_windows=int(observed.sum()),context_windows=int(context.sum()),insufficient_other_windows=int((observed&~context).sum()),record_channel_observed=True))
   for q in diag:
    if q['channel_index'] in queryidx:disruption.append(dict(q,patient=pid,center=center,feature=lock['features'][q['feature_index']],channel_count_stratum=channel_stratum,seizure_count_stratum=seizure_stratum))
   base.append(rbase);own.append(rd);real.append(ru);wrong.append(rw);counts.append(dict(baseline=bc,self=ddc,context=uuc));source_hist.append(dict(patient=pid,center=center,source_windows=detail['source_windows'],boundary_invalid=detail['invalid_boundary_windows'],invalid_time=detail['invalid_time_windows'],nonfinite=detail['nonfinite_feature_windows'],missing_canonical=sum(n not in lookup for n in canonical),source_channels=len(names),matched_channels=len(queryidx),noncanonical_reference_channels=len(names)-len(queryidx),spacing_min=detail['interval_min'],spacing_max=detail['interval_max']))
   ledger.append(dict(patient=pid,run=str(rec['run_id']),sample=str(sample['sample_id']),counts=counts[-1],time=t,query_channels=canonical,source_channels=names,other_counts=ctx['other_counts']))
  bb,bn=across_seizures(base);dd,dn=across_seizures(own);uu,un=across_seizures(real);ww,wn=across_seizures(wrong);assert np.array_equal(un,wn) and np.array_equal(np.isfinite(uu),np.isfinite(ww))
  shared=np.c_[bb,dd];tables[pid]=dict(channel=np.array(canonical),center=center,records=len(records),C0=np.c_[shared,np.zeros_like(uu)],C1=np.c_[shared,uu],C2=np.c_[shared,ww],count=dict(baseline=bn,self=dn,context=un))
  assert all(tables[pid][arm].shape==(len(ix),182) for arm in ['C0','C1','C2'])
  early_any.extend(seenearly.tolist());patient_info.append(dict(patient=pid,center=center,channels=len(ix),records=len(records)))
  summary_changes.append(dict(patient=pid,real=uu,shifted=ww))
  print('WINDOW_CONTEXT_PATIENT_COMPLETE',len(tables),'of80',flush=True)
 # Global label-free achieved disruption gate, not a predictive outcome gate.
 sh=pd.DataFrame(disruption);eligible=sh[sh.eligible];nonzero=float(eligible.shifted.mean());material=float(eligible.material_changed.mean());cor=float(eligible.correlation_reduction.mean());cyclic=float(eligible.cyclic_autocorrelation_drift.max())
 aa=np.concatenate([q['real'] for q in summary_changes]);bb=np.concatenate([q['shifted'] for q in summary_changes]);valid=np.isfinite(aa)&np.isfinite(bb)
 norm=np.sqrt(np.divide(np.nansum(aa*aa,0),valid.sum(0),out=np.zeros(84),where=valid.sum(0)>0));delta=np.sqrt(np.divide(np.nansum((aa-bb)**2,0),valid.sum(0),out=np.zeros(84),where=valid.sum(0)>0))/np.maximum(norm,1e-12);changed=float((delta>1e-6).mean())
 g=lock['control_gate'];conditions=dict(nonzero_shifts=nonzero>=g['nonzero_shift_fraction_eligible_min'],material_disruption=material>=g['materially_changed_fraction_all_eligible_min'],query_context_correlation_reduced=np.isfinite(cor) and cor>=g['mean_absolute_zero_lag_query_context_correlation_reduction_min'],cyclic_autocorrelation_preserved=not np.isfinite(cyclic) or cyclic<=g['cyclic_lag1_correlation_max_drift'],summary_disruption=changed>=g['changed_context_summary_coordinates_min_fraction'],mask_and_multiset_identity=True)
 passed=all(conditions.values());ctl=dict(status='PASS' if passed else 'SYNCHRONIZATION_CONTROL_NOT_IDENTIFIABLE',conditions=conditions,eligible_sequences=len(eligible),all_sequences=len(sh),nonzero_shift_fraction=nonzero,material_change_fraction=material,mean_absolute_query_context_correlation_reduction=cor,max_cyclic_autocorrelation_drift=cyclic,changed_context_summary_coordinate_fraction=changed,constant_sequences_retained=True)
 rr=pd.DataFrame(source_hist);assert rr.source_windows.value_counts().to_dict()=={59:253,51:1,40:1,56:1};assert rr.boundary_invalid.sum()==27 and rr.invalid_time.sum()==0
 assert rr.matched_channels.sum()==24995 and rr.missing_canonical.sum()==272 and len(early_any)==7635 and all(early_any)
 report=dict(status='PASS',patients=80,seizures=256,canonical_channels=7635,record_channel_slots=int(sum(q['channels']*q['records'] for q in patient_info)),observed_record_channels=int(rr.matched_channels.sum()),missing_record_channels=int(rr.missing_canonical.sum()),window_count_histogram=rr.source_windows.value_counts().to_dict(),boundary_rejected_windows=int(rr.boundary_invalid.sum()),invalid_time_windows=0,nonfinite_feature_channel_windows=int(rr.nonfinite.sum()),window_width_seconds=2,center_spacing_range=[rr.spacing_min.min(),rr.spacing_max.max()],canonical_early_supported_fraction=float(np.mean(early_any)),noncanonical_reference_channel_instances=int(rr.noncanonical_reference_channels.sum()),patients_by_seizure_count=pd.Series([q['records'] if q['records']<3 else '3+' for q in patient_info],dtype=object).value_counts().to_dict(),actual_feature_mapping=[list(q) for q in maps],label_fields_never_indexed=True,archive_label_member_never_loaded=True,cache_relative_timing_only=True,no_raw_or_EDF_processing=True,no_full_FIT_VAL_or_outer_model_bank_opened=True,incidental_supplied_pickle_metadata_deserialization_possible=True,source_binding=binding)
 cv=pd.DataFrame(coverage);pubrows=[]
 for (ce,st),q in cv.groupby(['center','stage']):pubrows.append(dict(center=ce,stage=st,unique_patients=q.patient.nunique(),record_channel_feature_rows=len(q),self_windows=int(q.self_windows.sum()),context_windows=int(q.context_windows.sum()),context_fraction_of_valid_self=q.context_windows.sum()/max(q.self_windows.sum(),1),baseline_reference_fraction=float(q.baseline_available.mean()),insufficient_other_windows=int(q.insufficient_other_windows.sum())))
 pd.DataFrame(pubrows).to_csv(a.pub/'REFERENCE_COVERAGE.csv',index=False)
 diagfields=['raw_mean_abs_change','relative_change','context_std','query_context_abs_correlation','shifted_query_context_abs_correlation','correlation_reduction','cyclic_autocorrelation_drift','ordinary_lag1_correlation','shifted_ordinary_lag1_correlation','ordinary_lag1_change','context_mean_drift','context_std_drift']
 grouped=aggregate(disruption,['center','stage','feature'],diagfields)
 for q in grouped:
  subset=sh[(sh.center==q['center'])&(sh.stage==q['stage'])&(sh.feature==q['feature'])];q['eligible_sequences']=int(subset.eligible.sum());q['shifted_fraction_eligible']=float(subset[subset.eligible].shifted.mean());q['material_fraction_eligible']=float(subset[subset.eligible].material_changed.mean())
 pd.DataFrame(grouped).to_csv(a.pub/'SHUFFLE_DISRUPTION.csv',index=False)
 hist=aggregate(coverage,['stage','channel_count_stratum','seizure_count_stratum'],['baseline_available','self_windows','context_windows','insufficient_other_windows']);pd.DataFrame(hist).to_csv(a.pub/'REFERENCE_COVERAGE_BY_STRATUM.csv',index=False)
 pd.DataFrame([dict(index=j,name=schema(lock['features'])[98+j],relative_RMS_real_shift_difference=delta[j],valid_channels=int(valid[:,j].sum())) for j in range(84)]).to_csv(a.pub/'CONTEXT_SUMMARY_DISRUPTION.csv',index=False)
 write_json(a.pub/'DATA_COVERAGE.json',report);write_json(a.pub/'SYNCHRONIZATION_CONTROL_GATE.json',ctl)
 save(target,dict(binding=binding,tables=tables,meta=meta,folds=folds,record_ledger=ledger,shift_ledger=disruption,patient_info=patient_info,control=ctl,schema=schema(lock['features'])))
 write_json(a.runtime/'PREPROBE_FREEZE_PRIVATE.json',dict(binding=binding,table_sha256=sha(target),control_sha256=sha(a.pub/'SYNCHRONIZATION_CONTROL_GATE.json'),protocol_sha256=sha(a.root/'PROTOCOL_LOCK.json'),labels_not_used=True))
 write_json(a.pub/'RUN_STATUS.json',dict(status='DATA_AND_CONTROL_PASS' if passed else 'SYNCHRONIZATION_CONTROL_NOT_IDENTIFIABLE',probe_models_fitted=0,no_new_outer=True));print('DATA_CONTROL_RESULT',ctl['status'],nonzero,material,cor,changed,flush=True)

if __name__=='__main__':
 a=options();lock,paths,binding=setup(a);data(a,lock,paths,binding)
