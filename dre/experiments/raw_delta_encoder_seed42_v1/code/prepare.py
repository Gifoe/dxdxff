"""Label-free real Raw-to-DELTA gate. No checkpoint/classifier/Outer loader."""
import argparse
from collections import defaultdict,Counter
import gc
import json
import pickle
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from io_utils import sha,write,save,read,load
from raw_alignment import support,exact_slices
from raw_physiological_features import Extractor,FEATURES,support_manifest
from construct_delta import window_delta,seizure_mean,aggregate,historical_record_mean
from feature_fidelity import rows,summarize,gate

CODE_FILES=['io_utils.py','raw_alignment.py','raw_physiological_features.py','construct_delta.py','feature_fidelity.py','prepare.py']

def parity(left,right):
 a=np.asarray(left,dtype=float);b=np.asarray(right,dtype=float);ulp=np.abs(np.spacing(np.abs(b).astype(np.float32)).astype(float));tol=8*ulp+2e-6*np.abs(b)
 return dict(pass_gate=bool(np.isfinite(a).all() and np.isfinite(b).all() and np.all(np.abs(a-b)<=tol)),max_absolute_difference=float(np.max(np.abs(a-b))),violating_coordinates=int((np.abs(a-b)>tol).sum()),coordinates=int(a.size))

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args();root=Path(__file__).parents[1];lock=json.loads((root/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));pub=a.runtime/'results';pub.mkdir(parents=True,exist_ok=True)
 paths=dict(raw=Path(r'D:\nips-temp\neuroez_c_four_center_caches_success_failure_raw_v1\all_window_cache.pkl'),window=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl'),archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv'))
 deps={}
 for k,path in paths.items():deps[k]=sha(path);assert deps[k]==lock['hashes'][k],k;print('EXACT_INPUT_HASH_PASS',k,flush=True)
 for rel,expected in lock['source_hashes'].items():assert sha(a.source/rel)==expected;deps['source_'+rel]=expected
 deps['protocol']=sha(root/'PROTOCOL_LOCK.json');deps['historical_review']=sha(root/'HISTORICAL_MECHANISM_AUDIT.md');deps['code']={f:sha(root/'code'/f) for f in CODE_FILES};deps['tests']=sha(root/'tests/test_stage_a.py')
 freeze=a.runtime/'STAGE_A_INPUT_FREEZE_PRIVATE.json'
 if freeze.exists():assert json.loads(freeze.read_text())==deps
 else:write(freeze,deps)
 if (pub/'STAGE_A_GATE.json').exists():assert json.loads((pub/'STAGE_A_GATE.json').read_text())['binding']==deps;print('STAGE_A_ALREADY_COMPLETE_NO_RECOMPUTATION',flush=True);return
 with np.load(paths['archive'],allow_pickle=True) as z:meta={k:z[k].astype(str) for k in ['patient','channel','center','features','split_patient','split_role']};meta['split_fold']=z['split_fold'];archived=z['x']  # Only label-blind original measurements, never y.
 pid=meta['patient'];ch=meta['channel'];center=meta['center'];ids=set(pid);assert len(ids)==80 and len(pid)==7635 and len(set(zip(pid,ch)))==7635 and archived.shape==(7635,88)
 indices=[list(meta['features']).index(f'b0_delta__{f}__seizure_{s}') for f in FEATURES for s in ['mean','std']];target=archived[:,indices];fitsets={fold:set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']=='fit')]) for fold in range(1,6)};fitids=set().union(*fitsets.values());assert len(fitids)==79
 fixed=pd.read_csv(paths['split']);assert set(zip(fixed.outer_fold,fixed.split_role.astype(str),fixed.subject_id.astype(str)))==set(zip(meta['split_fold'],meta['split_role'],meta['split_patient']))
 with paths['window'].open('rb') as f:payload=pickle.load(f)
 allowed=['subject_id','run_id','sample_id','source_seizure_id','seizure_onset_sec','start_sec','end_sec','channel_names_norm','source_center','center','window_relative_centers_sec','feature_scale_used_secs','raw_temporal_sfreq','raw_temporal_duration_sec','raw_valid_start_sample','raw_valid_samples','window_features','window_feature_names']
 records=[{**{k:r[k] for k in allowed if k in r},'sample':{k:v for k,v in r['sample'].items() if k in allowed}} for r in payload['run_records'] if str(r['subject_id']) in ids];del payload;gc.collect();assert len(records)==256
 sys.path.insert(0,str(a.source));from neuroez_c.dual_view_data import RawAlignmentStore,_record_key,_channel_names,_field,_window_centers
 from neuroez_c.raw_brainbert_data import normalize_channel_name
 store=RawAlignmentStore(records,feature_cache_path=paths['window'],raw_cache_path=paths['raw'],raw_target_samples=500,raw_target_sampling_rate=250);store.assert_formal_coverage(min_channel_match_rate=1,min_window_match_rate=1,expected_patients=80)
 # Trusted pickle may materialize extra metadata; never index clinical fields.
 rawkeep=allowed+['raw_waveform'];store.raw_records={k:{**{n:r[n] for n in rawkeep if n in r},'sample':{n:v for n,v in r['sample'].items() if n in rawkeep}} for k,r in store.raw_records.items()};gc.collect()
 original=load('rawdelta_historical_aggregation',a.source/'task1_baselines/feature_aggregation.py');extractor=Extractor(a.source,lock['source_hashes']['ez_features.py']);bypatient=defaultdict(list)
 for record in records:bypatient[str(record['subject_id'])].append(record)
 assert set(bypatient)==ids;E=np.full((7635,14),np.nan,np.float32);R=E.copy();historical=E.copy();ledgers=[];precomp=[];windowcomp=[];count=0;cw=rw=incidences=0;missing=0;histogram=Counter();missing_baselines=0;feature_metadata_disagreement=0
 for n,pid0 in enumerate(sorted(bypatient),1):
  ii=np.flatnonzero(pid==pid0);canonical=ch[ii].tolist();norm=[normalize_channel_name(x) for x in canonical];assert len(set(norm))==len(norm);lookup={name:i for i,name in enumerate(norm)};eng=[];raw=[];old=[];perrecord=[]
  cached=a.runtime/'patient_reconstruction_private'/f'{n:03d}.pkl'
  if cached.exists():
   q=read(cached);assert q['binding']==deps and q['patient']==pid0 and np.array_equal(q['indices'],ii)
   E[ii]=q['E'];R[ii]=q['R'];historical[ii]=q['H'];ledgers.append(q['ledger']);precomp.extend(q['precomp'])
   cw+=q['cw'];rw+=q['rw'];incidences+=q['incidences'];missing+=q['missing'];count+=q['count'];histogram.update(q['histogram']);missing_baselines+=q['missing_baselines'];feature_metadata_disagreement+=q['metadata_disagreement']
   print('VERIFIED_STAGE_A_PATIENT_RESUME',n,'of80',flush=True);continue
  before=(cw,rw,incidences,missing,count,missing_baselines,feature_metadata_disagreement,len(precomp));localhist=Counter()
  for record in bypatient[pid0]:
   key=_record_key(record);rec=store.raw_records[key];sample=record['sample'];names=_channel_names(record);rawnames=_channel_names(rec);assert set(names)<=set(norm);t=_window_centers(record);widths=_field(record,'feature_scale_used_secs');fs=float(_field(rec,'raw_temporal_sfreq'));wave=np.asarray(_field(rec,'raw_waveform'));starts,valid=support(t,fs,wave.shape[-1],int(_field(rec,'raw_valid_start_sample')),int(_field(rec,'raw_valid_samples')),widths)
   aligned,am,_=store.aligned_windows(record);am&=valid[:,None];assert am.shape==(len(t),len(names)) and (am[valid]).all();cw+=exact_slices(wave,names,rawnames,starts,valid,aligned);rw+=int(valid.sum());incidences+=len(names);missing+=len(ii)-len(names);histogram[len(t)]+=1;localhist[len(t)]+=1;count+=1
   raw_metadata_start=int(_field(rec,'raw_valid_start_sample'));raw_metadata_end=raw_metadata_start+int(_field(rec,'raw_valid_samples'));feature_start=int(_field(record,'raw_valid_start_sample'));feature_end=feature_start+int(_field(record,'raw_valid_samples'));feature_metadata_disagreement+=int((raw_metadata_start!=feature_start) or (raw_metadata_end!=feature_end))
   fn=list(sample['window_feature_names']);fi=[fn.index(f) for f in FEATURES];x=np.asarray(sample['window_features'],dtype=np.float32)[...,fi];assert x.shape==(*am.shape,7) and np.isfinite(x[am]).all()
   # Independent seven-family baseline reconstruction on genuine support.
   de,be,bc=window_delta(x,t,am);rf=np.full_like(x,np.nan);raw_all=np.full((*am.shape,len(extractor.module.BASE_SPECTRAL_FEATURE_NAMES)),np.nan,np.float32);flat=np.flatnonzero(am.reshape(-1));chunks=[]
   for ix in np.array_split(flat,max(1,int(np.ceil(len(flat)/4096)))):
    if len(ix):raw_all.reshape(-1,raw_all.shape[-1])[ix]=extractor.all(aligned.reshape(-1,500)[ix])
   rf=raw_all[...,extractor.indices]
   dr,br,brc=window_delta(rf,t,am);assert np.array_equal(bc,brc);missing_baselines+=int((bc==0).sum())
   # Original label-blind source helper, no fabricated clinical labels required.
   se,hn=historical_record_mean(original,sample['window_features'],fn,t,valid);sr,rn=historical_record_mean(original,raw_all,list(extractor.module.BASE_SPECTRAL_FEATURE_NAMES),t,valid);assert hn==rn
   hmean,_=historical_record_mean(original,sample['window_features'],fn,t,np.ones(len(t),bool));hj=[hn.index(f'b0_delta__{f}') for f in FEATURES]
   ee=np.full((len(ii),len(hn)),np.nan,np.float32);rr=ee.copy();oo=ee.copy()
   for j,name in enumerate(names):ee[lookup[name]]=se[j];rr[lookup[name]]=sr[j];oo[lookup[name]]=hmean[j]
   eng.append(ee);raw.append(rr);old.append(oo);perrecord.append(dict(compound_key=key,times=t,starts=starts,raw_valid=(raw_metadata_start,raw_metadata_end),feature_valid=(feature_start,feature_end),names=names,mask=am,baseline_counts=bc))
   for j,f in enumerate(FEATURES):
    ok=np.isfinite(be[:,j])&np.isfinite(br[:,j]);precomp.append(dict(patient=pid0,center=center[ii[0]],feature=f,baseline_channels=int(ok.sum()),median_absolute_reference_difference=float(np.median(np.abs(be[ok,j]-br[ok,j]))) if ok.any() else np.nan,engineered_reference_median=float(np.median(be[ok,j])) if ok.any() else np.nan,raw_reference_median=float(np.median(br[ok,j])) if ok.any() else np.nan))
   if count%16==0:print('RAW_RECORD_RECONSTRUCTION_COMPLETE',count,'of256',flush=True)
  allE,ec=aggregate(eng);allR,rc=aggregate(raw);allH,hc=aggregate(old);pick=[v for j in hj for v in [2*j,2*j+1]];E[ii]=allE[:,pick];R[ii]=allR[:,pick];historical[ii]=allH[:,pick];assert np.array_equal(ec,rc);ledgers.append(dict(patient=pid0,channels=canonical,records=perrecord,counts=ec[:,hj]));print('RAW_DELTA_PATIENT_COMPLETE',n,'of80',flush=True)
  save(cached,dict(binding=deps,patient=pid0,indices=ii,E=E[ii],R=R[ii],H=historical[ii],ledger=ledgers[-1],precomp=precomp[before[7]:],cw=cw-before[0],rw=rw-before[1],incidences=incidences-before[2],missing=missing-before[3],count=count-before[4],missing_baselines=missing_baselines-before[5],metadata_disagreement=feature_metadata_disagreement-before[6],histogram=dict(localhist)))
 coverage=dict(patients=len(ids),records=count,canonical_channels=7635,matched_record_channels=incidences,missing_record_channel_incidences=missing,valid_record_windows=rw,valid_channel_windows=cw,window_histogram=dict(histogram),missing_baseline_record_channels=missing_baselines,raw_feature_valid_range_metadata_disagreement_records=feature_metadata_disagreement,raw_sfreq=250,Nyquist=125,exact_raw_slice_parity=True,all80_retained=True,no_padded_Raw_samples_processed=True,new_Outer_accessed=False,clinical_y_member_loaded=False,legacy_pickle_extra_fields_may_materialize_but_never_indexed=True,binding=deps)
 identity=count==256 and incidences==24995 and missing==272 and cw==1471965 and rw==15074
 fullparity=parity(E,target);historicalparity=parity(historical,target);sourceparity=parity(E,historical);finite=bool(np.isfinite(E).all() and np.isfinite(R).all());fd=rows(E,R,pid,center,fitids);cg=gate(fd)
 summary=summarize(fd,['feature','category','stat']);pd.DataFrame(summary).to_csv(pub/'FEATURE_FIDELITY_BY_GROUP.csv',index=False);pd.DataFrame(summarize(fd,['center','feature','category','stat'])).to_csv(pub/'FEATURE_FIDELITY_BY_CENTER.csv',index=False);pd.DataFrame(summary).to_csv(pub/'RAW_DELTA_SCALE_AUDIT.csv',index=False)
 fitpre=pd.DataFrame(precomp);fitpre=fitpre[fitpre.patient.isin(fitids)];out=[]
 for (ce,feature),q in fitpre.groupby(['center','feature']):out.append(dict(center=ce,feature=feature,FIT_patients=q.patient.nunique(),record_cases=len(q),paired_reference_channels=int(q.baseline_channels.sum()),median_record_absolute_reference_difference=float(q.median_absolute_reference_difference.median()),median_engineered_reference=float(q.engineered_reference_median.median()),median_raw_reference=float(q.raw_reference_median.median())))
 pd.DataFrame(out).to_csv(pub/'PREICTAL_REFERENCE_FIDELITY.csv',index=False)
 foldout=[]
 for fold,fitset in fitsets.items():
  q=fd[fd.patient.isin(fitset)];foldout.extend([dict(fold=fold,**v) for v in summarize(q,['feature','category','stat'])])
 pd.DataFrame(foldout).to_csv(pub/'FIT_FIDELITY_BY_ORIGINAL_FOLD.csv',index=False)
 cols=[]
 for j,f in enumerate(FEATURES):cols.append(dict(feature=f,coordinate_indices=indices[2*j:2*j+2],genuine_vs_archive=parity(E[:,2*j:2*j+2],target[:,2*j:2*j+2]),pinned_historical_vs_archive=parity(historical[:,2*j:2*j+2],target[:,2*j:2*j+2])))
 write(pub/'RAW_COVERAGE.json',coverage);write(pub/'FEATURE_SUPPORT.json',support_manifest());write(pub/'E14_RECONSTRUCTION_PARITY.json',dict(genuine_vs_archive=fullparity,pinned_original_vs_archive=historicalparity,genuine_vs_pinned_original=sourceparity,features=cols,clinical_labels_used=False))
 checks=dict(complete_identity_and_support=identity,exact_window_parity=True,genuine_E14_reproduced=fullparity['pass_gate'],finite_inputs=finite,baseline_available=missing_baselines==0,Category_AB_fidelity=cg['pass_gate'],label_blind=True)
 passed=all(checks.values());terminal='STAGE_A_PASS' if passed else 'RAW_DELTA_FIDELITY_NOT_ESTABLISHED';gg=dict(binding=deps,pass_gate=passed,terminal=terminal,conditions=checks,correlation_gate=cg,FIT_unique_patients=len(fitids),cohort_alignment_denominator=80,no_labels_used=True,no_Outer_accessed=True)
 save(a.runtime/'RECONSTRUCTION_PRIVATE.pkl',dict(binding=deps,E14=E,R14=R,archived_E14=target,meta=meta,fitsets=fitsets,ledgers=ledgers,fidelity_patient_rows=fd.to_dict('records'),pre_reference_rows=precomp))
 write(pub/'STAGE_A_GATE.json',gg);write(pub/'RUN_STATUS.json',dict(status='STAGE_A_COMPLETE',terminal=terminal,models_trained=0,no_new_Outer=True));print('RAW_DELTA_STAGE_A_GATE',terminal,cg['passing_families'],'of5',flush=True)
 if not passed:
  metrics=['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr','sensitivity','specificity'];blank=[dict(model=m,status='NOT_RUN_STAGE_A_FIDELITY_FAILED',**{k:None for k in metrics}) for m in ['D0_REPLAY','E1','D1','D2','D3']]
  for name in ['DEVELOPMENT_SUMMARY.csv','DEVELOPMENT_BY_FOLD.csv','DEVELOPMENT_BY_CENTER.csv','DEVELOPMENT_UNIQUE_ID_SUMMARY.csv','PAIRED_BOOTSTRAP.csv','MODEL_SELECTION.csv','PARAMETER_AUDIT.csv','RAW_BRANCH_INTERVENTIONS.csv','ERROR_AND_RANKING_AUDIT.csv']:pd.DataFrame(blank).to_csv(pub/name,index=False)
  write(pub/'COMPACT_PARITY.json',dict(status='NOT_RUN_STAGE_A_FIDELITY_FAILED'));write(pub/'SCIENTIFIC_GATE.json',dict(terminal=terminal,stage='A',D1_D2_D3='NOT_RUN',no_rescue=True,no_Outer=True))

if __name__=='__main__':main()
