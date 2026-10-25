"""Aggregate only after every selected and permuted Outer score cell is sealed."""
from pathlib import Path
import numpy as np
import pandas as pd
import core as c

ALLOW=['main_results.csv','individual_feature_results.csv','all_results.csv','pr_improvement.csv',
 'paired_comparisons.csv','fold_results.csv','source_results.csv','negative_control.csv',
 'negative_control_summary.csv','coverage.csv','coordinate_variation.csv','model_selection.csv',
 'inner_cv.csv','fit_audit.csv','INPUT_AUDIT.json','ADMISSION.json','SOURCE_PUSH_BEFORE_FIT.json',
 'TRAIN_STATUS.json','FREEZE_STATUS.json','PREDICTION_STATUS.json','INTERNAL_VERIFICATION.json',
 'INDEPENDENT_VERIFICATION.json','STATUS.json']


def csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.partial');pd.DataFrame(rows).to_csv(tmp,index=False,float_format='%.17g');tmp.replace(path)


def summarize(frame,arm):
    row=dict(arm=arm,patients=len(frame),channels=int(frame.channels.sum()))
    for metric in c.METRICS:
        s=c.bootstrap(frame[metric]);row[metric]=s['mean'];row[metric+'_ci_low']=s['ci_low'];row[metric+'_ci_high']=s['ci_high']
    return row


def main(root):
    root=Path(root);pub=root/'public';d=c.load_data()
    assert c.jr(pub/'PREDICTION_STATUS.json')['status']=='ALL570_SEALED_BEFORE_METRICS'
    seal=c.jr(root/'private/OUTER_SCORES_SEAL.json');assert len(seal['files'])==570
    rows=[]
    for key,h in seal['files'].items():
        path=root/'private/scores'/(key+'.pkl');assert c.sha(path)==h;q=c.read(path);ix=q['indices']
        yy=d['y'][ix];prob=q['probability_ez'];pid=d['patient'][ix]
        for r in c.patient_metrics(yy,prob,pid,d['channel'][ix],q['threshold']):
            local=pid==r['patient'];rows.append(dict(**r,arm=q['arm'],fold=q['fold'],permutation=q['permutation'],
                source=str(d['source'][ix[local][0]]),channels=int(local.sum()),EZ_prevalence=float(yy[local].mean())))
    frame=pd.DataFrame(rows);assert len(frame)==9120 and not frame.duplicated(['arm','permutation','patient']).any()
    path=root/'private/PATIENT_METRICS.pkl'
    if path.exists():pd.testing.assert_frame_equal(c.read(path),frame)
    else:c.save(path,frame)
    main=frame[frame.permutation==-1];null=frame[frame.permutation>=0]
    assert len(main)==1120 and len(null)==8000 and np.isfinite(frame[c.METRICS]).all().all()
    summaries=[];folds=[];sources=[];coverage=[]
    for arm in c.ARMS:
        sub=main[main.arm==arm].sort_values('patient');assert len(sub)==80 and sub.patient.nunique()==80 and sub.channels.sum()==7635
        summaries.append(summarize(sub,arm));coverage.append(dict(arm=arm,patients=80,channels=7635,finite_patient_endpoints=640))
        for f,part in sub.groupby('fold'):
            for metric in c.METRICS:folds.append(dict(arm=arm,fold=int(f),metric=metric,patients=len(part),mean=float(part[metric].mean())))
        for z,part in sub.groupby('source'):
            for metric in c.METRICS:sources.append(dict(arm=arm,source=z,metric=metric,**c.bootstrap(part[metric])))
    csv(pub/'all_results.csv',summaries);csv(pub/'main_results.csv',[r for arm in c.MAIN for r in summaries if r['arm']==arm])
    csv(pub/'individual_feature_results.csv',[r for r in summaries if not r['arm'].startswith('ALL_SIX')])
    csv(pub/'fold_results.csv',folds);csv(pub/'source_results.csv',sources);csv(pub/'coverage.csv',coverage)
    full=main[main.arm=='ALL_SIX_PR'].set_index('patient').sort_index();comparisons=[];improvement=[]
    for arm in [f+'_PR' for f in c.FEATURES]+['ALL_SIX_NO_PR']:
        other=main[main.arm==arm].set_index('patient').loc[full.index]
        for metric in c.METRICS:
            delta=full[metric]-other[metric];r=dict(reference='ALL_SIX_PR',control=arm,metric=metric,**c.bootstrap(delta))
            r['difference_pp']=100*r['mean'];r['ci_low_pp']=100*r['ci_low'];r['ci_high_pp']=100*r['ci_high']
            r['primary_family_size']=28 if metric in c.PRIMARY else 0
            if metric in c.PRIMARY:
                s=c.bootstrap(delta,True);r.update(adjusted_ci_low=s['ci_low'],adjusted_ci_high=s['ci_high'],
                    adjusted_ci_low_pp=100*s['ci_low'],adjusted_ci_high_pp=100*s['ci_high'])
            else:r.update(adjusted_ci_low=np.nan,adjusted_ci_high=np.nan,adjusted_ci_low_pp=np.nan,adjusted_ci_high_pp=np.nan)
            comparisons.append(r)
    for name in c.FEATURES+['ALL_SIX']:
        a=main[main.arm==name+'_PR'].set_index('patient').sort_index()
        b=main[main.arm==name+'_NO_PR'].set_index('patient').loc[a.index]
        for metric in c.METRICS:
            r=dict(feature=name,metric=metric,**c.bootstrap(a[metric]-b[metric]));r.update(difference_pp=100*r['mean'],ci_low_pp=100*r['ci_low'],ci_high_pp=100*r['ci_high'])
            improvement.append(r)
    csv(pub/'paired_comparisons.csv',comparisons);csv(pub/'pr_improvement.csv',improvement)
    nullrows=[]
    for b,sub in null.groupby('permutation'):
        assert len(sub)==80 and sub.channels.sum()==7635
        nullrows.append(dict(permutation=int(b),patients=80,channels=7635,**{m:float(sub[m].mean()) for m in c.METRICS}))
    csv(pub/'negative_control.csv',nullrows);nr=pd.DataFrame(nullrows);ns=[]
    for metric in c.METRICS:
        vals=nr[metric];lo,hi=np.quantile(vals,[.025,.975])
        ns.append(dict(metric=metric,permutations=100,null_mean=float(vals.mean()),null_sample_SD=float(vals.std(ddof=1)),
            null_reference_low=float(lo),null_reference_high=float(hi),observed=float(full[metric].mean()),
            patient_equal_EZ_prevalence=float(full.EZ_prevalence.mean()),chance_AUROC=.5))
    csv(pub/'negative_control_summary.csv',ns)
    c.write(pub/'STATUS.json',dict(status='REPORTED_PENDING_INDEPENDENT_VERIFICATION',groups=14,main_groups=8,
        patients=80,main_and_supplementary_rows=1120,negative_control_patient_rows=8000,patient_endpoint_values=72960,
        formal_LR_fits=1410,selected_and_null_score_cells=570))
    print('PREDICTION_REPORT_COMPLETE 14groups,100nullrepeats,9120patientrows,72960endpoints',flush=True)
