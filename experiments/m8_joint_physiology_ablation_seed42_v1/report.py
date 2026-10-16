"""CPU-only aggregate reports from validation or sealed Outer; never train/select Outer."""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score
from analysis import *
faulthandler.enable()
HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_joint_physiology_ablation_seed42_v1_runtime'); PUB=ROOT/'public'
ENGINE=Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py')
spec=importlib.util.spec_from_file_location('joint_metrics_engine',ENGINE);EV=importlib.util.module_from_spec(spec);spec.loader.exec_module(EV)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()

def jr(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.bool_,bool)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(clean(obj),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8');tmp.replace(path)
def read(path):
    path=Path(path);assert sha(path)==jr(path.with_suffix('.sha256.json'))['sha256']
    with path.open('rb') as f:return pickle.load(f)
def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:pickle.dump(obj,f,protocol=5)
    tmp.replace(path);write(path.with_suffix('.sha256.json'),dict(sha256=sha(path)))

def frames(stage):
    if stage=='development':return read(ROOT/'DEVELOPMENT_PREDICTIONS_PRIVATE.pkl')
    sp=ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';seal=jr(sp);fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert seal['freeze_sha256']==fh and seal['all80_cells_sealed_before_full_labels'] and len(seal['files'])==80
    context=read(ROOT/'OUTER_METADATA_PRIVATE.pkl');meta=context['meta']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==context['archive_sha256']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert len(y)==7635 and y.sum()==1743
    rows=[]
    for key,sh in sorted(seal['files'].items()):
        pp=ROOT/'outer_scores_private'/(key+'.pkl');assert sha(pp)==sh
        q=read(pp);assert q['freeze_sha256']==fh
        config,head,fold=key.split('_');ix=q['indices']
        rows.append(pd.DataFrame(dict(scope='outer',model=config,config=config,head=head,version=head,fold=int(fold),
            patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],
            probability=1-q['score_nez'].astype(float),prediction=q['score_nez']<q['threshold'])))
    return pd.concat(rows,ignore_index=True)

def patient_metrics(frame):
    # Historical engine is reused byte-for-byte. Independent sklearn/direct ranking below.
    rows=EV.patient_rows(frame)
    patients=pd.DataFrame(rows).rename(columns={'model':'config','version':'head'})
    patients['scope']=frame.scope.iloc[0]
    for (config,head,fold,pid),q in frame.groupby(['config','head','fold','patient'],sort=True):
        r=patients[(patients.config==config)&(patients['head']==head)&(patients.fold==fold)&(patients.patient==pid)].iloc[0]
        yy=q.y.to_numpy();prob=q.probability.to_numpy(dtype=np.float64);pred=q.prediction.to_numpy()
        checks=dict(macro_f1=f1_score(yy,pred,labels=[0,1],average='macro',zero_division=0),accuracy=accuracy_score(yy,pred))
        if yy.sum():checks.update(f1=f1_score(yy,pred,zero_division=0),auprc=average_precision_score(yy,prob))
        if (1-yy).sum():checks['nez_f1']=f1_score(1-yy,1-pred.astype(int),zero_division=0)
        if len(set(yy))==2:checks['auroc']=roc_auc_score(yy,prob)
        order=np.lexsort((q.channel.to_numpy(dtype=str),-prob));rank=yy[order]
        if yy.sum():checks.update(mrr=1/(np.flatnonzero(rank)[0]+1),hit_at_1=float(rank[0]))
        for key,v in checks.items():assert abs(float(r[key])-v)<1e-12,(config,head,key)
    return patients

def aggregate(frame,patients,out):
    out.mkdir(parents=True,exist_ok=True)
    summary=[];folds=[];sources=[];coverage=[];foldsd=[];stability=[];absolute=[]
    for (config,head),g in patients.groupby(['config','head'],sort=True):
        summary.append(dict(config=config,head=head,retained_groups=int(sum(keep(config))),appearances=len(g),unique_patients=g.patient.nunique(),
            channels=int(g.n.sum()),**{m:EV.nanmean(g[m]) for m in METRICS}))
        for col,target in [('fold',folds),('center',sources)]:
            for key,h in g.groupby(col,sort=True):target.append(dict(config=config,head=head,**{col:key},appearances=len(h),unique_patients=h.patient.nunique(),**{m:EV.nanmean(h[m]) for m in METRICS}))
        for m in METRICS:
            v=np.array([EV.nanmean(h[m]) for _,h in g.groupby('fold',sort=True)])
            coverage.append(dict(config=config,head=head,metric=m,finite_appearances=int(np.isfinite(g[m]).sum()),total_appearances=len(g)))
            foldsd.append(dict(config=config,head=head,metric=m,fold_mean=EV.nanmean(v),fold_sample_sd=float(np.nanstd(v,ddof=1)),valid_folds=int(np.isfinite(v).sum())))
    for name,rows in [('MAIN_RESULTS',summary),('FOLD_METRICS',folds),('SOURCE_METRICS',sources),('METRIC_COVERAGE',coverage),('FOLD_MEAN_SD',foldsd)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    g0=patients[(patients.config=='000')&(patients['head']=='D1')].sort_values(['fold','patient'])
    ids=g0.patient.to_numpy();order=list(zip(g0.fold,g0.patient));unique,counts,draw=bootstrap_counts(ids)
    matrix={}
    for (config,head),g in patients.groupby(['config','head'],sort=True):
        g=g.sort_values(['fold','patient']);assert list(zip(g.fold,g.patient))==order
        matrix[head,config]=g[METRICS].to_numpy(dtype=float)
    def evaluate(label,kind,weights,adjusted=False,**extra):
        mean,ci,n,dist=contrast(matrix,weights,ids,counts,unique)
        # Independent direct draws: cluster repeats stay together for Development.
        arr=np.stack([matrix[k] for k in weights]);valid=np.isfinite(arr).all(0)
        delta=sum(weights[k]*matrix[k] for k in weights);delta=np.where(valid,delta,np.nan)
        for i in range(32):
            ix=np.concatenate([np.flatnonzero(ids==unique[k]) for k in draw[i]])
            assert np.allclose(np.nanmean(delta[ix],axis=0),dist[i],atol=1e-12,rtol=0,equal_nan=True)
        for j,m in enumerate(METRICS):
            row=dict(contrast=label,kind=kind,metric=m,delta=float(mean[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),
                valid_appearances=int(n[j]),cluster_patients=len(unique),draws=10000,threshold_reoptimized=False,**extra)
            if adjusted and m in ['macro_f1','auroc','auprc']:
                lo,hi=np.nanpercentile(dist[:,j],[100*.05/84,100*(1-.05/84)])
                row.update(primary_family_contrasts=42,bonferroni_ci_low=float(lo),bonferroni_ci_high=float(hi))
            results.append(row)
    results=[]
    for head in ['D1','M8']:
        for config in CONFIGS:
            g=patients[(patients.config==config)&(patients['head']==head)].sort_values(['fold','patient'])
            u,ci,n,_=contrast(matrix,{(head,config):1},ids,counts,unique)
            absolute.extend(dict(config=config,head=head,metric=m,mean=float(u[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),valid_appearances=int(n[j]),draws=10000) for j,m in enumerate(METRICS))
            if config=='000':continue
            evaluate(head+'_'+config+'_minus_000','deletion',{(head,config):1,(head,'000'):-1},head=head,config=config)
            ref=patients[(patients.config=='000')&(patients['head']==head)].sort_values(['fold','patient'])
            for j,m in enumerate(METRICS):
                d=g[m].to_numpy()-ref[m].to_numpy();v=np.array([EV.nanmean(d[g.fold.to_numpy()==f]) for f in range(1,6)])
                stability.append(dict(config=config,head=head,metric=m,positive_folds=int((v>0).sum()),negative_folds=int((v<0).sum()),zero_folds=int((v==0).sum()),median_fold_delta=float(np.nanmedian(v)),worst_fold_delta=float(np.nanmin(v)),best_fold_delta=float(np.nanmax(v)),fold_delta_sample_sd=float(np.nanstd(v,ddof=1))))
        for factors in ['V','H','E','VH','VE','HE','VHE']:
            evaluate(head+'_'+factors,'factorial',{(head,c):w for c,w in factorial_weights(factors).items()},True,head=head,factors=factors)
        for factors,third in [('VH','E'),('VE','H'),('HE','V')]:
            for val in [0,1]:evaluate(head+'_'+factors+'_'+third+str(val),'conditional_interaction',
                {(head,c):w for c,w in factorial_weights(factors,{third:val}).items()},head=head,factors=factors,condition=third+str(val))
    for config in CONFIGS:
        evaluate('M8_minus_D1_'+config,'raw_increment',{('M8',config):1,('D1',config):-1},config=config)
        if config!='000':evaluate('RAW_COMPENSATION_'+config,'raw_compensation',{('M8',config):1,('M8','000'):-1,('D1',config):-1,('D1','000'):1},config=config)
    for factors in ['V','H','E','VH','VE','HE','VHE']:
        weights={}
        for c,w in factorial_weights(factors).items():weights['M8',c]=w;weights['D1',c]=-w
        evaluate('RAW_FACTORIAL_'+factors,'raw_factorial_compensation',weights,factors=factors)
    pd.DataFrame(results).to_csv(out/'PAIRED_CONTRASTS.csv',index=False)
    pd.DataFrame(absolute).to_csv(out/'ABSOLUTE_INTERVALS.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'FOLD_DELETION_STABILITY.csv',index=False)
    return len(results)

def main(stage):
    fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl')
    for name,h in fr['binding']['source'].items():assert sha(HERE/name)==h
    assert sha(ENGINE)==fr['binding']['metric_engine_sha256']
    frame=frames(stage);patients=patient_metrics(frame)
    assert len(patients)==(1040 if stage=='development' else 1280)
    for _,g in patients.groupby(['config','head']):
        assert (len(g),g.patient.nunique())==((65,47) if stage=='development' else (80,80))
        if stage=='outer':assert int(g.n.sum())==7635
    save(ROOT/(stage.upper()+'_PATIENT_METRICS_PRIVATE.pkl'),patients)
    contrasts=aggregate(frame,patients,PUB/stage)
    if stage=='development':
        rows,choices=candidate_decisions(patients,jr(HERE/'PROTOCOL_LOCK.json'))
        pd.DataFrame(rows).to_csv(PUB/stage/'CANDIDATE_STABILITY_GATE.csv',index=False)
        write(PUB/'DEVELOPMENT_SELECTION_FROZEN.json',dict(status='FROZEN_BEFORE_NEW_OUTER',choices=choices,
            model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),source=fr['binding']['source'],
            selection_scope='original_legal_validation_only_selection_conditioned',Outer_used=False,
            engineering_margins_not_clinical_noninferiority=True,eligible=rows))
    else:
        choice=jr(PUB/'DEVELOPMENT_SELECTION_FROZEN.json');freeze=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
        assert sha(PUB/'DEVELOPMENT_SELECTION_FROZEN.json')==freeze['candidate_decision_sha256']
        write(PUB/'COMPLETION_STATUS.json',dict(status='JOINT_2CUBED_D1_M8_OUTER_COMPLETE',patient_rows=1280,configurations=8,heads=2,folds=5,
            new_models=40,reused_models=40,choices_before_Outer=choice['choices'],Outer_selection=False,scientific_status='EXPLORATORY_REPEATED_TEST',
            seal_sha256=sha(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')))
    for p in (PUB/stage).glob('*.csv'):
        assert not set(pd.read_csv(p,nrows=0).columns)&{'patient','channel','probability','prediction','y','path','checkpoint','indices'}
    write(PUB/stage/'REPORT_VERIFICATION.json',dict(status='PASS',patient_rows=len(patients),all_sklearn_metric_and_ranking_checks=True,
        bootstrap_first32_draws_directly_verified=True,paired_contrast_metric_rows=contrasts,source_sha256=sha(__file__),metric_engine_sha256=sha(ENGINE),
        thresholds_reoptimized=False,scope=stage,python_runtime=os.sys.version,numpy=np.__version__,pandas=pd.__version__))
    print('JOINT_'+stage.upper()+'_AGGREGATES_AND_CONTRASTS_PASS',len(patients),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','outer']);main(p.parse_args().stage)
