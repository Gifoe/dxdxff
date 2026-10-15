"""FIT-only linear necessity screens and label-free numeric/redundancy audits.

Does not load Outer scores,predictions,metrics,seals or the full label archive.
Linear screens are preliminary evidence,not final nonlinear M8 ablations.
"""
import gc
import hashlib
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch

TASK=Path(__file__).resolve().parent
SOURCE=Path(r'E:\DRE-nips\new-pipeline\7-11')
BASELINE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
OUT=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
sys.path.insert(0,str(SOURCE/'raw_delta_encoder_seed42_v1/code'))
from io_utils import sha,read,save,write,load
from dependencies import context
EV=load('discovery_metrics',Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))


def jr(p):return json.loads(Path(p).read_text())
def setup():
    lock=jr(TASK/'AUDIT_PROTOCOL.json')
    c,core,fac,trainer,records,rec,banks,hist,deps=context(SOURCE,BASELINE)
    assert rec['R14'].shape==(7635,14) and np.isfinite(rec['R14']).all()
    assert len(set(rec['meta']['patient']))==80 and sum(len(q['records']) for q in rec['ledgers'])==256
    registry=Path(r'D:\dgm_net_seed42_v1_runtime\CONDITIONING_REGISTRY_PRIVATE.pkl')
    assert sha(registry)==lock['old_feature_registry_sha256']
    binding=dict(source={n:sha(TASK/n) for n in ['AUDIT_PROTOCOL.json','audit_old7.py']},dependencies=deps,
        reconstruction_sha256=sha(BASELINE/'RECONSTRUCTION_PRIVATE.pkl'),registry_sha256=sha(registry),
        metric_source_sha256=sha(Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py')),
        sklearn=sklearn.__version__,numpy=np.__version__)
    OUT.mkdir(exist_ok=True);(OUT/'public').mkdir(exist_ok=True)
    ip=OUT/'OLD7_INPUT_FREEZE_PRIVATE.json'
    if ip.exists():assert jr(ip)==binding
    else:write(ip,binding)
    return lock,core,rec,banks,binding


def weights(y,pid):
    ids,counts=np.unique(pid,return_counts=True);size=dict(zip(ids,counts))
    w=np.asarray([len(y)/(len(ids)*size[p]) for p in pid]);classes=np.bincount(y,minlength=2)
    assert (classes>0).all();return w*len(y)/(2*classes[y])


def numeric(lock,core,rec,banks):
    stat=[];corr=[]
    x=rec['R14'];meta=rec['meta']
    for fold,d in enumerate(banks,1):
        fit=np.asarray(sorted(set(d['patient'][d['train']])));assert set(fit).isdisjoint(d['roles']['validation']|d['roles']['test'])
        ids=[];rankpairs=[]
        for pid in fit:
            ix=meta['patient']==pid;raw=x[ix].astype(float);z=core.patient_z(raw,meta['patient'][ix])
            sd=raw.std(0);ranked=rankdata(raw,axis=0);ranked-=ranked.mean(0)
            norm=np.sqrt((ranked*ranked).sum(0));den=norm[:,None]*norm[None,:]
            rank=np.divide(ranked.T@ranked,den,out=np.full((14,14),np.nan),where=den>0)
            rankpairs.append(rank);ids.append((raw,sd,z))
        for k,name in enumerate(lock['old_feature_names']):
            for pos,suffix in [(2*k,'seizure_mean'),(2*k+1,'seizure_sd')]:
                values=np.concatenate([q[0][:,pos] for q in ids]);sds=np.asarray([q[1][pos] for q in ids])
                stat.append(dict(outer_fold=fold,scope='FIT_only',feature=name,coordinate=suffix,patients=len(fit),
                    channels=len(values),finite_fraction=float(np.isfinite(values).mean()),
                    constant_patients=int((sds==0).sum()),original_patientZ_fallback_fraction=float((sds<=1e-8).mean()),
                    median_patient_raw_sd=float(np.median(sds)),q25=float(np.quantile(values,.25)),median=float(np.median(values)),q75=float(np.quantile(values,.75))))
        for i in range(14):
            for j in range(i+1,14):
                v=np.asarray([r[i,j] for r in rankpairs]);v=v[np.isfinite(v)]
                corr.append(dict(outer_fold=fold,scope='FIT_only',left_feature=lock['old_feature_names'][i//2],left_coordinate=i%2,
                    right_feature=lock['old_feature_names'][j//2],right_coordinate=j%2,valid_patients=len(v),
                    median_spearman=float(np.median(v)) if len(v) else np.nan,
                    median_absolute_spearman=float(np.median(np.abs(v))) if len(v) else np.nan,
                    fraction_abs_r_ge_0_95=float((np.abs(v)>=.95).mean()) if len(v) else np.nan))
    pd.DataFrame(stat).to_csv(OUT/'public/OLD7_NUMERIC_FIT_AUDIT.csv',index=False)
    pd.DataFrame(corr).to_csv(OUT/'public/OLD7_REDUNDANCY_FIT_AUDIT.csv',index=False)


def probes(lock,core,rec,banks,binding):
    variants=[('FULL7',list(range(14)))]+[(f'ONLY_{k}',[2*k,2*k+1]) for k in range(7)]+[(f'DROP_{k}',[j for j in range(14) if j//2!=k]) for k in range(7)]
    rows=[];audits=[];patients=[]
    for fold,d in enumerate(banks,1):
        fit=d['train'];seed=42+1009*fold;pid=d['patient'];y=(d['y']==0).astype(int)
        x=core.patient_z(rec['R14'][d['archive_indices']],pid)
        cv=GroupKFold(n_splits=3,shuffle=True,random_state=seed)
        for inner,(it,iv) in enumerate(cv.split(x[fit],y[fit],groups=pid[fit])):
            remain=fit[it];held=fit[iv];names=np.asarray(sorted(set(pid[remain])))
            order=np.random.default_rng(seed+43*(inner+1)).permutation(names);ncal=max(1,int(np.ceil(.2*len(order))))
            cal=remain[np.isin(pid[remain],order[:ncal])];tr=remain[~np.isin(pid[remain],order[:ncal])]
            groups=[set(pid[q]) for q in [tr,cal,held]]
            assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
            assert set.union(*groups)==set(pid[fit])
            assert set.union(*groups).isdisjoint(d['roles']['validation']|d['roles']['test'])
            audits.append(dict(outer_fold=fold,inner_screen=inner+1,subfit_patients=len(groups[0]),calibration_patients=len(groups[1]),screen_patients=len(groups[2]),
                original_validation_excluded=True,own_outer_test_excluded=True,screen_not_used_for_threshold=True))
            for name,columns in variants:
                path=OUT/f'old7_probes_private/fold{fold}_screen{inner}_{name}.pkl'
                cb=dict(binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(),fold=fold,inner=inner,variant=name,columns=columns)
                if path.exists():q=read(path);assert q['binding']==cb
                else:
                    xx=x[:,columns];imp=SimpleImputer().fit(xx[tr]);zz=imp.transform(xx)
                    scaler=StandardScaler().fit(zz[tr]);zz=scaler.transform(zz)
                    model=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-7,random_state=seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');model.fit(zz[tr],y[tr],sample_weight=weights(y[tr],pid[tr]))
                    pc=model.predict_proba(zz[cal])[:,1];sel=EV.threshold(y[cal],pc,pid[cal]);ph=model.predict_proba(zz[held])[:,1]
                    frame=pd.DataFrame(dict(model=name,version='FIT_screen',fold=fold,patient=pid[held],channel=d['channel'][held],center=d['center'][held],
                        y=y[held],probability=ph,prediction=ph>=sel['threshold']))
                    q=dict(binding=cb,patients=EV.patient_rows(frame),threshold=sel['threshold'],screen_indices=held,
                        warnings=sorted(set(type(w.message).__name__ for w in caught)),iterations=int(model.n_iter_.max()))
                    save(path,q)
                group=pd.DataFrame(q['patients']);patients += [dict(r,outer_fold=fold,inner_screen=inner+1,variant=name) for r in q['patients']]
                rows.append(dict(outer_fold=fold,inner_screen=inner+1,variant=name,patients=len(group),channels=int(group.n.sum()),
                    **{m:EV.nanmean(group[m]) for m in ['auroc','auprc','macro_f1','f1','accuracy','mrr','hit_at_1']},iterations=q['iterations'],warnings=';'.join(q['warnings'])))
                print('FIT_ONLY_PROBE_COMPLETE',fold,inner+1,name,flush=True)
    assert len(rows)==225
    df=pd.DataFrame(patients);save(OUT/'OLD7_PROBE_PATIENTS_PRIVATE.pkl',df)
    summaries=[];delta=[]
    for (fold,name),g in df.groupby(['outer_fold','variant']):
        assert g.patient.nunique()==len(g)
        summaries.append(dict(outer_fold=fold,variant=name,patients=len(g),**{m:EV.nanmean(g[m]) for m in ['auroc','auprc','macro_f1','f1','accuracy','mrr','hit_at_1']}))
        if name!='FULL7':
            full=df[(df.outer_fold==fold)&(df.variant=='FULL7')].set_index('patient').loc[g.patient]
            for m in ['auroc','auprc','macro_f1','f1']:
                diff=g[m].to_numpy()-full[m].to_numpy()
                delta.append(dict(outer_fold=fold,variant=name,metric=m,paired_patients=int(np.isfinite(diff).sum()),delta=float(np.nanmean(diff))))
    pd.DataFrame(rows).to_csv(OUT/'public/OLD7_INNER_PROBE_METRICS.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'public/OLD7_OUTER_FIT_CROSSFIT_SUMMARY.csv',index=False)
    pd.DataFrame(delta).to_csv(OUT/'public/OLD7_PROBE_PAIRED_DIFFERENCES.csv',index=False)
    centers=[dict(outer_fold=f,variant=n,center=center,patients=len(g),**{m:EV.nanmean(g[m]) for m in ['auroc','auprc','macro_f1','f1']})
        for (f,n,center),g in df.groupby(['outer_fold','variant','center'])]
    pd.DataFrame(centers).to_csv(OUT/'public/OLD7_SCREEN_CENTER_METRICS.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'public/OLD7_PATIENT_ISOLATION_AUDIT.csv',index=False)
    write(OUT/'public/OLD7_AUDIT_STATUS.json',dict(status='OLD7_FIT_ONLY_INITIAL_AUDIT_COMPLETE',linear_probe_cells=225,
        original_validation_and_outer_not_used=True,nonlinear_M8_necessity_not_established=True,full_goal_complete=False,
        source=binding['source'],registry_sha256=binding['registry_sha256']))


if __name__=='__main__':
    lock,core,rec,banks,binding=setup();numeric(lock,core,rec,banks);probes(lock,core,rec,banks,binding)
    print('OLD7_FIT_ONLY_INITIAL_AUDIT_COMPLETE',flush=True)
