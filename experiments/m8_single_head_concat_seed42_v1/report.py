"""CPU-only selected-candidate freeze and sealed-score statistical reporting."""
import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score
from selection import METRICS,STRICT,choose,counts,paired,check_target
HERE=Path(__file__).resolve().parent;ROOT=Path(r'D:\m8_single_head_concat_seed42_v1_runtime');PUB=ROOT/'public'
ENGINE=Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py')
spec=importlib.util.spec_from_file_location('single_head_metrics',ENGINE);EV=importlib.util.module_from_spec(spec);spec.loader.exec_module(EV)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def jr(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.bool_,bool)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(clean(x),sort_keys=True,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def save(p,x):
    p=Path(p);tmp=p.with_suffix('.partial')
    with tmp.open('wb') as f:pickle.dump(x,f,protocol=5)
    tmp.replace(p);write(p.with_suffix('.sha256.json'),dict(sha256=sha(p)))
def read(p):
    p=Path(p);assert sha(p)==jr(p.with_suffix('.sha256.json'))['sha256']
    with p.open('rb') as f:return pickle.load(f)

def frames(stage):
    if stage=='development':return read(ROOT/'DEVELOPMENT_PREDICTIONS_PRIVATE.pkl')
    seal=jr(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert seal['all80_scores_sealed_before_full_labels'] and len(seal['files'])==80 and seal['freeze_sha256']==fh
    context=read(ROOT/'OUTER_METADATA_PRIVATE.pkl');meta=context['meta']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==context['archive_sha256']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert len(y)==7635 and int(y.sum())==1743
    result=[]
    for key,h in sorted(seal['files'].items()):
        p=ROOT/'outer_scores_private'/(key+'.pkl');assert sha(p)==h;q=read(p);assert q['freeze_sha256']==fh
        model,fold=key.rsplit('_',1);ix=q['indices']
        result.append(pd.DataFrame(dict(scope='outer',model=model,version='seed42',fold=int(fold),patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],
            y=y[ix],probability=1-q['score_nez'].astype(float),prediction=q['score_nez']<q['threshold'])))
    return pd.concat(result,ignore_index=True)

def metrics(frame):
    p=pd.DataFrame(EV.patient_rows(frame));p['scope']=frame.scope.iloc[0]
    for (model,fold,pid),g in frame.groupby(['model','fold','patient'],sort=True):
        r=p[(p.model==model)&(p.fold==fold)&(p.patient==pid)].iloc[0];y=g.y.to_numpy();v=g.prediction.to_numpy();pr=g.probability.to_numpy(dtype=np.float64)
        checks=dict(accuracy=accuracy_score(y,v),macro_f1=f1_score(y,v,labels=[0,1],average='macro',zero_division=0))
        if y.sum():checks.update(f1=f1_score(y,v,zero_division=0),auprc=average_precision_score(y,pr))
        if (1-y).sum():checks['nez_f1']=f1_score(1-y,1-v.astype(int),zero_division=0)
        if len(set(y))==2:checks['auroc']=roc_auc_score(y,pr)
        order=np.lexsort((g.channel.to_numpy(dtype=str),-pr));rank=y[order]
        if y.sum():checks.update(mrr=1/(np.flatnonzero(rank)[0]+1),hit_at_1=float(rank[0]))
        for key,value in checks.items():assert abs(float(r[key])-value)<1e-12,(model,key)
    return p

def comparison_pairs(names,selected):
    pairs=[(f'S{i}','REF_M8') for i in range(5)]+[(a,'REF_M8_NO_VAR') for a in ['S2','S4']]
    pairs += [(f'S{i}','S0') for i in range(1,5)]+[('S2','S1'),('S4','S3'),('S3','S1'),('S4','S2'),('S0','REF_RAW_ONLY')]
    pairs += [(selected,r) for r in ['REF_D1','REF_RAW_ONLY','REF_LATE_FUSION']]
    pairs += [(name,name.split('__')[0]) for name in names if '__' in name]
    return sorted(set((a,b) for a,b in pairs if a in names and b in names))

def tables(frame,p,stage,selected):
    out=PUB/'development' if stage=='development' else PUB;out.mkdir(exist_ok=True)
    means=[];folds=[];sources=[];sd=[];coverage=[];absolute=[];intervals=[];distribution=[]
    for name,g in p.groupby('model',sort=True):
        means.append(dict(model=name,unit='patient_macro',appearances=len(g),unique_patients=g.patient.nunique(),channels=int(g.n.sum()),**{m:EV.nanmean(g[m]) for m in METRICS}))
        for col,target in [('fold',folds),('center',sources)]:
            for key,h in g.groupby(col,sort=True):target.append(dict(model=name,**{col:key},appearances=len(h),unique_patients=h.patient.nunique(),**{m:EV.nanmean(h[m]) for m in METRICS}))
        for m in METRICS:
            x=np.array([EV.nanmean(h[m]) for _,h in g.groupby('fold',sort=True)])
            sd.append(dict(model=name,metric=m,fold_mean=EV.nanmean(x),fold_sample_sd=float(np.nanstd(x,ddof=1))))
            coverage.append(dict(model=name,metric=m,finite_appearances=int(np.isfinite(g[m]).sum()),total_appearances=len(g)))
    main=[r for r in means if '__' not in r['model']];diag=[r for r in means if '__' in r['model']]
    for name,rows in [('MAIN_RESULTS',main),('DIAGNOSTIC_RESULTS',diag),('FOLD_RESULTS',folds),('SOURCE_RESULTS',sources),('FOLD_MEAN_SD',sd),('METRIC_COVERAGE',coverage)]:
        if rows:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    names=sorted(set(p.model));base=p[p.model=='REF_M8'].sort_values(['fold','patient']);ids=base.patient.to_numpy();order=list(zip(base.fold,base.patient))
    unique,multiplicity,draw=counts(ids);matrix={}
    for name in names:
        g=p[p.model==name].sort_values(['fold','patient']);assert list(zip(g.fold,g.patient))==order
        matrix[name]=g[METRICS].to_numpy(dtype=float)
        mean,ci,n,_=paired(matrix[name],np.zeros_like(matrix[name]),ids,unique,multiplicity)
        for j,m in enumerate(METRICS):absolute.append(dict(model=name,metric=m,mean=mean[j],ci_low=ci[0,j],ci_high=ci[1,j],valid_appearances=n[j],draws=10000))
    for a,b in comparison_pairs(names,selected):
        mean,ci,n,dist=paired(matrix[a],matrix[b],ids,unique,multiplicity)
        d=matrix[a]-matrix[b];valid=np.isfinite(matrix[a])&np.isfinite(matrix[b]);d[~valid]=np.nan
        for i in range(32):
            ix=np.concatenate([np.flatnonzero(ids==unique[k]) for k in draw[i]])
            assert np.allclose(np.nanmean(d[ix],axis=0),dist[i],atol=1e-12,rtol=0,equal_nan=True)
        for j,m in enumerate(METRICS):
            row=dict(model=a,reference=b,metric=m,delta=float(mean[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),valid_appearances=int(n[j]),cluster_patients=len(unique),draws=10000,threshold_reoptimized=False)
            if a==selected and b=='REF_M8' and m in STRICT:
                lo,hi=np.nanpercentile(dist[:,j],[100*.05/16,100*(1-.05/16)])
                row.update(selected8endpoint_bonferroni_ci_low=float(lo),selected8endpoint_bonferroni_ci_high=float(hi))
            intervals.append(row)
            if a in [f'S{k}' for k in range(5)] and b=='REF_M8':
                values=d[:,j];v=values[np.isfinite(values)];loss=-np.minimum(v,0);den=float(loss.sum())
                distribution.append(dict(model=a,reference=b,metric=m,improved_appearances=int((v>0).sum()),worse_appearances=int((v<0).sum()),equal_appearances=int((v==0).sum()),
                    q10=float(np.quantile(v,.1)),median=float(np.median(v)),q90=float(np.quantile(v,.9)),minimum=float(v.min()),maximum=float(v.max()),
                    worst10_share_of_all_negative_deltas=float(np.sort(loss)[-10:].sum()/den) if den else np.nan))
    pd.DataFrame(absolute).to_csv(out/'ABSOLUTE_INTERVALS.csv',index=False);pd.DataFrame(intervals).to_csv(out/'PAIRED_COMPARISONS.csv',index=False)
    pd.DataFrame(distribution).to_csv(out/'PATIENT_CHANGE_DISTRIBUTION.csv',index=False)
    return pd.DataFrame(main),intervals

def main(stage):
    prot=jr(HERE/'PROTOCOL_LOCK.json');fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');binding=fr['binding']
    for name,h in binding['source'].items():assert sha(HERE/name)==h
    assert sha(ENGINE)==binding['metric_engine_sha256']
    frame=frames(stage);p=metrics(frame);expected=520 if stage=='development' else 1280;assert len(p)==expected
    for name,g in p.groupby('model'):
        assert (len(g),g.patient.nunique())==((65,47) if stage=='development' else (80,80))
        if stage=='outer':assert int(g.n.sum())==7635
    save(ROOT/(stage.upper()+'_PATIENT_METRICS_PRIVATE.pkl'),p)
    if stage=='development':
        selected,rows=choose(p,prot)
        obj=dict(status='SELECTED_BEFORE_NEW_OUTER',selected_model=selected,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),
            selection_rule=prot['candidate_selection'],scores=rows,Outer_used=False,source=binding['source'])
        target=PUB/'CANDIDATE_FROZEN.json'
        if target.exists():assert jr(target)==clean(obj)
        else:write(target,obj)
        pd.DataFrame(rows).to_csv(PUB/'CANDIDATE_SELECTION.csv',index=False)
    else:
        decision=jr(PUB/'CANDIDATE_FROZEN.json');selected=decision['selected_model'];freeze=jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
        assert freeze['candidate_sha256']==sha(PUB/'CANDIDATE_FROZEN.json') and freeze['selected_model']==selected
    mainframe,intervals=tables(frame,p,stage,selected)
    if stage=='outer':
        indexed=mainframe.set_index('model');a=indexed.loc[selected];b=indexed.loc['REF_M8']
        for m,v in binding['fullprecision_baseline'].items():assert abs(float(b[m])-v)<1e-12
        comparisons,allpass,primarypass=check_target(a,b)
        write(PUB/'FINAL_DECISION.json',dict(status='PASS_ALL_METRICS_POINT_ESTIMATE' if allpass else 'FAIL_STRICT_M8_TARGET',selected_model=selected,
            all8_point_estimate_pass=allpass,all4_primary_classification_pass=primarypass,metrics=comparisons,
            failed_metrics=[r for r in comparisons if not r['passes']],candidate_selected_without_Outer=True,
            scientifically_confirmed_noninferiority=False,scientific_status='EXPLORATORY_REPEATED_TEST',
            failure_boundary=None if allpass else 'Independent D1 branch cannot yet be removed without verified performance loss.',
            freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),seal_sha256=sha(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')))
    out=PUB/'development' if stage=='development' else PUB
    for path in out.glob('*.csv'):
        assert not set(pd.read_csv(path,nrows=0).columns)&{'patient','channel','y','probability','prediction','path','checkpoint','indices'}
    write(out/'REPORT_VERIFICATION.json',dict(status='PASS',scope=stage,patient_metric_rows=len(p),all_sklearn_and_direct_ranking_checks=True,
        source_sha256=sha(__file__),source_metric_sha256=sha(ENGINE),bootstrap_draws=10000,threshold_reoptimized=False,selected_model=selected))
    print('SINGLE_HEAD_'+stage.upper()+'_REPORT_PASS',selected,len(p),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','outer']);main(p.parse_args().stage)
