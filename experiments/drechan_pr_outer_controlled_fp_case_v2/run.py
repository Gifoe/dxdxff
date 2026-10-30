"""Frozen-score high-gain illustration, with a selection rule locked before use."""
import csv,hashlib,json,math,os,runpy,sys,zipfile
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
BASE=Path(r'D:\Codex_Research\experiments')
NAME='drechan_pr_outer_controlled_fp_case_v2'
SOURCE=BASE/NAME;RUNTIME=BASE/(NAME+'_runtime');PUBLIC=RUNTIME/'public';PRIVATE=RUNTIME/'private'
PARENT=BASE/'drechan_pr_outer_representative_cases_v1_runtime_run02'
sys.path.insert(0,str(BASE/'drechan_pr_representation_vs_score_v1'))
from path_adapter import install
install(r'D:\Codex_Research\_maintenance\cleanup_20261028\LEGACY_PATHS.json',[RUNTIME])
import numpy as np

def main():
    assert not RUNTIME.exists(),'Inspect existing state before any new execution'
    RUNTIME.mkdir();PUBLIC.mkdir();PRIVATE.mkdir()
    parent_source=BASE/'drechan_pr_outer_representative_cases_v1/run.py'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(parent_source)=='20b8c54c811f6b4157056862163acedfab9eb345180a0dfa6aad4733f5d02626'
    assert json.loads((PARENT/'public/STATUS.json').read_text())['status']=='COMPLETE_VERIFIED_FROZEN_OUTER_REPRESENTATIVE_CASES'
    g=runpy.run_path(str(parent_source),run_name='frozen_parent_math')
    parent_bind=g['jr'](PARENT/'private/INPUT_BINDING.json')
    protected=dict(parent_bind['protected'])
    for path in [BASE/'drechan_pr_outer_large_gain_case_v1_runtime/private/IDENTITY_MAP.json',PARENT/'private/INPUT_BINDING.json',PARENT/'private/SELECTED_CASES_FROZEN.json',PARENT/'private/IDENTITY_MAP.json',SOURCE/'run.py',SOURCE/'PROTOCOL_LOCK.json']:
        protected[str(path)]=sha(path)
    g['write'](PRIVATE/'PREANALYSIS_LOCK.json',dict(protocol=g['jr'](SOURCE/'PROTOCOL_LOCK.json'),source_commit=os.environ['CASE_SOURCE_COMMIT'],protected=protected))
    for p,h in protected.items():assert g['sha'](p)==h,'Protected input changed'
    protocol,meta,truth,roles,scores,config,gate,cell_audit=g['admission']()
    rows,detail,lookup,delta=g['compute'](meta,truth,roles,scores,config)
    ids=sorted({k[0] for k in lookup});excluded=set(g['jr'](PARENT/'private/IDENTITY_MAP.json').values())|set(g['jr'](BASE/'drechan_pr_outer_large_gain_case_v1_runtime/private/IDENTITY_MAP.json').values())
    def gains(p):return [delta[p,'DRECHANNET','MEAN_42_43_44'][m] for m in ['auroc','recall_at_10pct']]
    def joint(p):return g['stability'](p,delta,1)
    confusion={}
    for (pid,model,seed,cond),q in detail.items():
        y=q['y'].astype(bool);pred=q['pred'].astype(bool)
        tp=int(np.sum(y&pred));fp=int(np.sum(~y&pred));fn=int(np.sum(y&~pred));tn=int(np.sum(~y&~pred))
        independent=np.bincount(2*y.astype(int)+pred.astype(int),minlength=4)
        assert [tn,fp,fn,tp]==independent.tolist()
        confusion[pid,model,str(seed),cond]=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.0,sensitivity=tp/(tp+fn),fpr=fp/(fp+tn),predicted_positive=tp+fp)
    for pid in ids:
        for cond in ['NO_PR','PR']:
            rr=[confusion[pid,'DRECHANNET',str(seed),cond] for seed in [42,43,44]]
            confusion[pid,'DRECHANNET','MEAN_42_43_44',cond]={k:float(np.mean([r[k] for r in rr])) for k in rr[0]}
    def cd(pid,seed,col):return confusion[pid,'DRECHANNET',seed,'PR'][col]-confusion[pid,'DRECHANNET',seed,'NO_PR'][col]
    base_pool=[p for p in ids if p not in excluded and lookup[p,'SVM','42','NO_PR']['channels']>=20 and cd(p,'MEAN_42_43_44','tp')>1e-12 and sum(cd(p,str(s),'tp')>0 for s in [42,43,44])>=2]
    tiers=[[p for p in base_pool if all(cd(p,str(s),'fp')<=limit for s in [42,43,44])] for limit in [0,2]]
    write=g['write']
    write(PRIVATE/'CANDIDATE_TIER_COUNTS.json',dict(base_candidates=len(base_pool),tier_counts=list(map(len,tiers))))
    assert any(tiers),'No patient satisfies the locked TP/FP criteria; do not change rules after this outcome'
    tier=next(i for i,z in enumerate(tiers) if z);pool=tiers[tier]
    def key(p):
        a,b=gains(p);return (-cd(p,'MEAN_42_43_44','tp'),cd(p,'MEAN_42_43_44','fp'),-(a+b)/2,hashlib.sha256(('controlled_fp_case_v1|'+p).encode()).hexdigest())
    chosen=sorted(pool,key=key)[0]
    primary={p:cd(p,'MEAN_42_43_44','tp') for p in pool}
    assert abs(primary[chosen]-max(primary.values()))<1e-12
    candidate_count=len(pool)
    selected=g['jr'](PARENT/'private/SELECTED_CASES_FROZEN.json')
    assert chosen not in selected['cases'].values()
    ref=lookup[chosen,'SVM','42','NO_PR'];case='CASE-F01';metrics=[];channels=[];named=[];curves=[]
    out=g['csvout'];write=g['write']
    write(PRIVATE/'IDENTITY_MAP.json',{case:chosen})
    write(PRIVATE/'SELECTED_CASE_LOCK.json',dict(case_id=case,private_patient_id=chosen,candidates=candidate_count,primary_gain=primary[chosen],protocol_sha256=sha(SOURCE/'PROTOCOL_LOCK.json')))
    out(PUBLIC/'CASE_METADATA.csv',[dict(case_id=case,center=ref['source'],outer_fold=ref['fold'],total_channels=ref['channels'],ez_channels=ref['ez'],nez_channels=ref['channels']-ref['ez'],dre_joint_consistent_seeds=joint(chosen),tp_positive_seed_count=sum(cd(chosen,str(s),'tp')>0 for s in [42,43,44]),dre_auroc_consistent_seeds=sum(delta[chosen,'DRECHANNET',str(s)]['auroc']>1e-12 for s in [42,43,44]),dre_recall10_consistent_seeds=sum(delta[chosen,'DRECHANNET',str(s)]['recall_at_10pct']>1e-12 for s in [42,43,44]))])
    for model,seed in [('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]:
        for metric in g['METRICS']:
            no,pr=[lookup[chosen,model,seed,c][metric] for c in ['NO_PR','PR']];d=pr-no
            metrics.append(dict(case_id=case,model=model,seed=seed,metric=metric,no_pr=no,pr=pr,change=d,percentile_among_all80=g['pct']([delta[p,model,seed][metric] for p in ids],d),percentile_among_case_pool=g['pct']([delta[p,model,seed][metric] for p in pool],d)))
    alias={ch:f'CH{i:04d}' for i,ch in enumerate(sorted(detail[chosen,'SVM',42,'NO_PR']['channel'].tolist()),1)}
    for model,seeds in [('SVM',[42]),('DRECHANNET',[42,43,44])]:
        for seed in seeds:
            for cond in ['NO_PR','PR']:
                q=detail[chosen,model,seed,cond];n=len(q['y']);ez=int(q['y'].sum())
                order=sorted(range(n),key=lambda i:(-float(q['score'][i]),alias[q['channel'][i]]))
                assert [int(q['rank'][i]) for i in order]==list(range(1,n+1))
                for i in sorted(range(n),key=lambda i:alias[q['channel'][i]]):
                    r=dict(case_id=case,channel_id=alias[q['channel'][i]],truth_label='EZ' if q['y'][i] else 'NEZ',truth_ez=int(q['y'][i]),model=model,seed=seed,condition=cond,score_ez=float(q['score'][i]),rank_ez=int(q['rank'][i]),prediction_ez=int(q['pred'][i]))
                    channels.append(r);named.append(dict(r,private_patient_id=chosen,original_channel_name=str(q['channel'][i])))
                for k in range(n+1):
                    hits=sum(int(q['y'][i]) for i in order[:k]);assert hits==int(q['y'][q['rank']<=k].sum())
                    curves.append(dict(case_id=case,model=model,seed=seed,condition=cond,k=k,total_channels=n,review_fraction=k/n,ez_recovered=hits,ez_total=ez,recall=hits/ez,precision=hits/k if k else ''))
    out(PUBLIC/'CASE_METRICS.csv',metrics);out(PUBLIC/'CHANNEL_SCORES_ANONYMOUS.csv',channels);out(PUBLIC/'RECALL_AT_K.csv',curves)
    out(PRIVATE/'SELECTED_CHANNELS_ORIGINAL_NAMES.csv',named)
    cm_rows=[]
    for model,seed in [('SVM','42')]+[('DRECHANNET',s) for s in ['42','43','44','MEAN_42_43_44']]:
        for cond in ['NO_PR','PR']:cm_rows.append(dict(case_id=case,model=model,seed=seed,condition=cond,**confusion[chosen,model,seed,cond]))
    out(PUBLIC/'CASE_CONFUSION.csv',cm_rows)
    audit=dict(cohort_patients=80,candidates=candidate_count,base_candidates=len(base_pool),tier_counts=list(map(len,tiers)),selected_tier=tier+1,fp_increase_limit_per_seed=[0,2][tier],prior_cases_excluded=3,min_ez_count=None,minimum_total_channels=20,ranking='Maximize DRE mean TP increase; then minimize FP change; then ranking gain; then fixed hash',mean_tp_change=cd(chosen,'MEAN_42_43_44','tp'),mean_fp_change=cd(chosen,'MEAN_42_43_44','fp'),mean_precision_change=cd(chosen,'MEAN_42_43_44','precision'),tp_increase_seed_count=sum(cd(chosen,str(s),'tp')>0 for s in [42,43,44]),maximum_fp_change_across_seeds=max(cd(chosen,str(s),'fp') for s in [42,43,44]),independent_maximum_agreement=True,deliberate_outcome_based_selection=True)
    write(PUBLIC/'SELECTION_AUDIT.json',audit);write(PUBLIC/'CASE_SELECTION_REASON.json',dict(case_id=case,selection_kind='deliberately_controlled_false_positive_illustration',**audit))
    for name in ['ALL80_CHANGE_DISTRIBUTION.csv','COHORT_AGGREGATES.csv']:
        src=PARENT/'public'/name;protected[str(src)]=sha(src);(PUBLIC/name).write_bytes(src.read_bytes())
    write(PUBLIC/'ADMISSION.json',dict(status='PASS',patients=80,channels=7635,ez_channels=1743,score_cells=40,new_model_fits=0,new_model_forward_passes=0))
    for p,h in {**protected,**gate}.items():assert g['sha'](p)==h,'Parent changed'
    write(PUBLIC/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',score_cells=40,patient_condition_rows=len(rows),patient_change_rows=len(delta),endpoint_values=len(rows)*5,original_parent_patient_metric_replays=640,selected_case_count=1,case_metric_rows=len(metrics),anonymous_channel_score_rows=len(channels),recall_at_k_rows=len(curves),confusion_rows=len(cm_rows),all_patient_confusion_counts_verified=640,independent_metric_formulas=True,independent_selector_agreement=True,parent_hashes_unchanged=True,new_training=0,new_inference=0))
    write(PUBLIC/'PROVENANCE.json',dict(source_commit=os.environ['CASE_SOURCE_COMMIT'],source_sha256=sha(SOURCE/'run.py'),protocol_sha256=sha(SOURCE/'PROTOCOL_LOCK.json'),parent_source_sha256=sha(parent_source),private_export_location=str(PRIVATE),drechan='Original complete FULL vs WO_BOTH_PR, seeds42/43/44; not LOCO or no-guidance model',cell_audit=cell_audit,interpretation='Deliberately selected TP gain with controlled FP; retrospective illustration, not a typical patient or cohort-level evidence'))
    write(PUBLIC/'STATUS.json',dict(status='COMPLETE_VERIFIED_FROZEN_OUTER_CONTROLLED_FP_CASE_V2',new_training=0,new_inference=0))
    files=sorted(PUBLIC.iterdir());manifest=dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},real_patient_identifiers_included=False,original_contact_names_included=False,user_authorized_deidentified_channel_scores=True)
    archive=RUNTIME/'public_deidentified_case_export_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'public/'+p.name)
        z.writestr('PUBLIC_MANIFEST.json',json.dumps(manifest,sort_keys=True))
    print('CONTROLLED_FP_CASE_COMPLETE_VERIFIED',sha(archive),flush=True)

if __name__=='__main__':main()
