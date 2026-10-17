"""Independent retained-model/CAL/role/reference/metric/statistics verification."""
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
import final_stage as f
from model import Model
from admit import evaluate
c=f.c


def main():
    binding,choices,_=f.setup();terminal=c.jr(f.PUBLIC/'FINAL_TERMINAL.json');assert terminal['status']=='COMPLETE_SIX_JOINT_RAW_PROCEDURES'
    fr=c.read(f.FINAL/'MODEL_FREEZE_PRIVATE.pkl');freeze=c.jr(f.FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert fr['input']==binding and freeze['input']==binding and freeze['model_freeze']==c.sha(f.FINAL/'MODEL_FREEZE_PRIVATE.pkl')
    assert binding['source']=={n:c.sha(c.HERE/n) for n in binding['source']}
    seal,meta,banks,y=f.sealed_labels();_,labeled=c.io.identity_and_banks(labels=True);store=c.RawStore();devchecks=[]
    for row in fr['rows']:
        if row['procedure']=='SELECTED' or row['kind']=='FIXED_REFERENCE':continue
        assert c.sha(row['checkpoint'])==row['checkpoint_sha256'];best=c.torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        assert best['binding']['input']==binding and best['binding']['arm']==row['arm'] and best['epoch']==row['epoch']>=1
        assert best['selection']['threshold']==row['threshold'];model=Model(row['arm'],row['fold']).cuda()
        assert c.state_hash(model.state_dict())==row['initial_state_sha256']==best['binding']['initial_state_sha256']
        model.load_state_dict(best['model']);d=labeled[row['fold']-1];assert np.array_equal(d['archive_indices'],banks[row['fold']-1]['archive_indices'])
        q=c.read(f.FINAL/'models_private'/f"fold{row['fold']}_screenfinal"/row['arm']/'HELD_PRIVATE.pkl')
        assert q['binding']==best['binding'] and np.array_equal(q['cal_archive_indices'],d['archive_indices'][d['val']]) and np.array_equal(q['archive_indices'],d['archive_indices'][d['val']])
        cl=evaluate(model,d,d['val'],store);error=float(np.max(np.abs(cl-q['cal_logit_nez'])));assert error<=1e-6
        grid=np.round(np.arange(0,1.0001,.005),3)
        assert np.array_equal(c.probability(cl)[:,None]<grid,c.probability(q['cal_logit_nez'])[:,None]<grid) and c.calibrate(d,d['val'],cl)==best['selection']
        devchecks.append(dict(fold=row['fold'],arm=row['arm'],maximum_Development_replay_error=error,all201_grid_decisions_exact=True,random_initial_state_exact=True))
        del model;gc.collect();c.torch.cuda.empty_cache()
    assert len(devchecks)==15
    for fold in range(1,6):assert len({r['initial_state_sha256'] for r in fr['rows'] if r['fold']==fold and r['procedure'] in c.TRAINED})==1
    rows=[];oldseal=c.jr(f.OLD/'OUTER_SCORE_SEAL_PRIVATE.json')
    for procedure in f.PROCEDURES:
        ids=set();indices=[]
        for fold in range(1,6):
            path=f.FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl';assert c.sha(path)==seal['cells'][f'{procedure}_{fold}'];q=c.read(path)
            row=next(r for r in fr['rows'] if r['fold']==fold and r['procedure']==procedure)
            assert q['configuration']==row and q['freeze']==seal['freeze'];ix=q['indices'];pn=q['score_nez'];pp=1-pn.astype(float);pred=pn<row['threshold']
            assert set(meta['patient'][ix])==banks[fold-1]['roles']['test'] and set(meta['patient'][ix]).isdisjoint(banks[fold-1]['roles']['fit']|banks[fold-1]['roles']['validation'])
            if procedure=='SELECTED':
                assert row['selected_arm']==choices[fold];other=c.read(f.FINAL/'outer_scores_private'/f'{choices[fold]}_{fold}.pkl')
                assert np.array_equal(pn,other['score_nez']) and row['threshold']==other['configuration']['threshold']
            elif row['kind']=='FIXED_REFERENCE':
                key=f"{row['old_procedure']}_{fold}";op=f.OLD/'outer_scores_private'/(key+'.pkl');assert c.sha(op)==oldseal['cells'][key]
                old=c.read(op);assert np.array_equal(pn,old['score_nez']) and np.array_equal(ix,old['indices']) and row['threshold']==old['threshold']
            indices.extend(ix)
            for pid in np.unique(meta['patient'][ix]):
                assert pid not in ids;ids.add(pid);j=meta['patient'][ix]==pid;ai=ix[j];yy=y[ai];p=pp[j];pr=pred[j];ch=meta['channel'][ai]
                order=sorted(range(len(yy)),key=lambda k:(-float(p[k]),str(ch[k])));first=next(k for k,i in enumerate(order) if yy[i])
                metric=dict(macro_f1=f1_score(yy,pr,average='macro',zero_division=0),f1=f1_score(yy,pr,zero_division=0),nez_f1=f1_score(yy,pr,pos_label=0,zero_division=0),
                    accuracy=accuracy_score(yy,pr),auroc=roc_auc_score(yy,p),auprc=average_precision_score(yy,p),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(pr.mean()-yy.mean()))
                rows.append(dict(procedure=procedure,fold=fold,patient=str(pid),center=str(meta['center'][ai][0]),**metric))
        assert len(ids)==80 and len(indices)==len(set(indices))==7635 and set(indices)==set(range(7635))
    frame=pd.DataFrame(rows);stored=pd.DataFrame(c.read(f.FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl'));keys=['procedure','fold','patient']
    a,b=frame.sort_values(keys).reset_index(drop=True),stored.sort_values(keys).reset_index(drop=True);assert a[keys].equals(b[keys])
    maximum=float(np.max(np.abs(a[c.METRICS].to_numpy()-b[c.METRICS].to_numpy())));assert maximum<1e-12
    public=pd.read_csv(f.PUBLIC/'OUTER_PATIENT_MEAN.csv').set_index('procedure');values={};ids=sorted(frame.patient.unique())
    for arm,g in frame.groupby('procedure'):
        g=g.sort_values('patient');assert g.patient.tolist()==ids;values[arm]=g[c.METRICS].to_numpy(float);assert np.isfinite(values[arm]).all()
        for m in c.METRICS:assert abs(float(g[m].mean())-float(public.loc[arm,m]))<1e-12
    draw=np.random.default_rng(42).integers(0,80,(10000,80));ci_checks=0;ci_error=0.
    def verify_ci(reported,x):
        nonlocal ci_checks,ci_error
        interval=np.quantile(x[draw].mean(1),[.025,.975],axis=0);mean=x.mean(0)
        for j,m in enumerate(c.METRICS):
            row=reported[reported.metric==m];assert len(row)==1;r=row.iloc[0]
            error=max(abs(float(r['mean'])-mean[j]),abs(float(r.low)-interval[0,j]),abs(float(r.high)-interval[1,j]))
            ci_error=max(ci_error,error);assert error<1e-12 and int(r.draws)==10000 and int(r.unique_patients)==80;ci_checks+=1
    absolute=pd.read_csv(f.PUBLIC/'OUTER_ABSOLUTE_CI.csv')
    for p in f.PROCEDURES:verify_ci(absolute[absolute.procedure==p],values[p])
    for (left,right),reported in pd.read_csv(f.PUBLIC/'OUTER_PAIRED_CI.csv').groupby(['left','right']):verify_ci(reported,values[left]-values[right])
    assert ci_checks==126
    actual_folds=frame.groupby(['procedure','fold'])[c.METRICS].mean().sort_index();public_folds=pd.read_csv(f.PUBLIC/'OUTER_FOLDS.csv').set_index(['procedure','fold']).sort_index()
    assert np.max(np.abs(actual_folds.to_numpy()-public_folds[c.METRICS].to_numpy()))<1e-12
    actual_sd=actual_folds.groupby('procedure').std(ddof=1).sort_index();public_sd=pd.read_csv(f.PUBLIC/'OUTER_FOLD_SD.csv').set_index('procedure').sort_index()
    assert np.max(np.abs(actual_sd.to_numpy()-public_sd[c.METRICS].to_numpy()))<1e-12
    sources=pd.read_csv(f.PUBLIC/'OUTER_SOURCE.csv');source_checks=0
    for (p,center),g in frame.groupby(['procedure','center']):
        r=sources[(sources.procedure==p)&(sources.center==center)].iloc[0];n=g.patient.nunique();assert int(r.patients)==n and bool(r.suppressed)==(n<5)
        if n<5:assert r[c.METRICS].isna().all()
        else:
            for m in c.METRICS:assert abs(float(r[m])-float(g[m].mean()))<1e-12
        source_checks+=1
    coverage=pd.read_csv(f.PUBLIC/'OUTER_METRIC_COVERAGE.csv');assert len(coverage)==54
    for _,r in coverage.iterrows():assert int(r.finite_patients)==int(r.total_patients)==80
    rawcoverage=c.jr(f.PUBLIC/'OUTER_STATUS.json')['coverage'];assert sum(r['valid_windows'] for r in rawcoverage)==1471965
    assert sum(r['patients'] for r in rawcoverage)==80 and sum(r['channels'] for r in rawcoverage)==7635
    c.write(f.PUBLIC/'INDEPENDENT_FINAL_VERIFICATION.json',dict(status='PASS',procedures=6,logical_cells=30,fresh_fits=15,patient_metric_rows=480,
        patients_per_procedure=80,channels_per_procedure=7635,valid_Raw250_windows=1471965,Development_checks=devchecks,
        all3_initializers_exact_per_fold=True,source_role_and_archive_identity_exact=True,selected_aliases_exact=True,reference_predictions_unchanged=True,
        all9_patient_metrics_independent=True,maximum_metric_error=maximum,independent_CI_metric_checks=ci_checks,maximum_bootstrap_difference=ci_error,
        fold_means_sampleSD_source_and_coverage_verified=True,source_groups_checked=source_checks,
        score_seal_sha256=c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),model_freeze_sha256=c.sha(f.FINAL/'MODEL_FREEZE_PRIVATE.pkl'),
        primary='SELECTED',primary_point_over_065=terminal['primary_point_over_065'],no_training_or_Outer_inference_in_verifier=True,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False))
    print('JOINT_RAW_INDEPENDENT_FINAL_VERIFICATION_PASS',terminal['patient_equal_macro_f1'],flush=True)


if __name__=='__main__':main()
