"""Independent frozen full80 endpoint/CI/count verification, no new prediction."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
from report_outer import FINAL,PUBLIC,load_rows


def main():
    f,_=load_rows();seal=jr(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');frozen=jr(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    context=read(FINAL/'METADATA_PRIVATE.pkl');meta=context['meta'];path=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(path)==context['archive_sha256']
    with np.load(path,allow_pickle=True) as z:y=1-z['y']
    numeric=0
    for key,h in sorted(seal['files'].items()):
        arm,fold=key.rsplit('_',1);fold=int(fold);q=read(FINAL/'scores_private'/(key+'.pkl'));ix=q['indices'];pn=q['score_nez'];pred=pn<q['threshold'];p=1-pn.astype(float)
        r=next(r for r in frozen['models'] if r['arm']==arm and r['fold']==fold)
        assert sha(r['checkpoint'])==r['checkpoint_sha256']==q['checkpoint'] and q['threshold']==r['threshold'] and r['epoch']>=1
        for pid in sorted(set(meta['patient'][ix])):
            j=meta['patient'][ix]==pid;yy=y[ix][j];pp=p[j];vv=pred[j];ch=meta['channel'][ix][j]
            order=np.lexsort((ch.astype(str),-pp));first=int(np.flatnonzero(yy[order])[0])
            values=dict(macro_f1=f1_score(yy,vv,labels=[0,1],average='macro',zero_division=0),f1=f1_score(yy,vv,zero_division=0),nez_f1=f1_score(1-yy,1-vv.astype(int),zero_division=0),
                auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(vv.mean()-yy.mean()))
            row=f[(f.arm==arm)&(f.fold==fold)&(f.patient==pid)].iloc[0]
            for m,v in values.items():assert abs(float(row[m])-v)<1e-12;numeric+=1
    main=pd.read_csv(PUBLIC/'MAIN_RESULTS.csv');folds=pd.read_csv(PUBLIC/'FOLD_RESULTS.csv');sd=pd.read_csv(PUBLIC/'FOLD_SD.csv');source=pd.read_csv(PUBLIC/'SOURCE_RESULTS.csv');coverage=pd.read_csv(PUBLIC/'METRIC_COVERAGE.csv')
    assert len(f)==240 and numeric==2160 and len(coverage)==27 and (coverage.finite_patients==80).all()
    matrix={}
    for arm,g in f.groupby('arm'):
        z=g.sort_values(['fold','patient']);matrix[arm]=z[METRICS].to_numpy(float)
        row=main[main.arm==arm].iloc[0];assert np.max(np.abs(row[METRICS].to_numpy(float)-matrix[arm].mean(0)))<1e-12
        ff=[]
        for fold,h in g.groupby('fold'):
            q=folds[(folds.arm==arm)&(folds.fold==fold)].iloc[0];v=h[METRICS].mean().to_numpy(float)
            assert np.max(np.abs(q[METRICS].to_numpy(float)-v))<1e-12;ff.append(v)
        row=sd[sd.arm==arm].iloc[0];assert np.max(np.abs(np.std(ff,axis=0,ddof=1)-row[METRICS].to_numpy(float)))<1e-12
        for center,h in g.groupby('center'):
            q=source[(source.arm==arm)&(source.center==center)].iloc[0];assert int(q.patients)==len(h)
            if len(h)>=5:assert np.max(np.abs(q[METRICS].to_numpy(float)-h[METRICS].mean().to_numpy(float)))<1e-12
    ci=pd.read_csv(PUBLIC/'INTERVALS.csv',keep_default_na=False);draw=np.random.default_rng(42).integers(0,80,(10000,80));maximum=0.;count=0
    # Row order is identical across arms and every patient occurs exactly once.
    order=f[f.arm=='C2_REFERENCE'].sort_values(['fold','patient'])[['fold','patient']].reset_index(drop=True)
    for arm in ARMS:assert f[f.arm==arm].sort_values(['fold','patient'])[['fold','patient']].reset_index(drop=True).equals(order)
    for (kind,left,right),rr in ci.groupby(['kind','left','right']):
        values=matrix[left]-(matrix[right] if kind=='paired' else np.zeros_like(matrix[left]))
        # cluster() sorts IDs, so reconstruct its exact prospective draw ordering independently.
        ids=sorted(order.patient.unique());position={pid:i for i,pid in enumerate(order.patient)};values=values[[position[pid] for pid in ids]]
        lohi=np.quantile(values[draw].mean(1),[.025,.975],axis=0)
        for j,m in enumerate(METRICS):
            row=rr[rr.metric==m].iloc[0];err=max(abs(float(row['mean'])-values[:,j].mean()),abs(float(row.low)-lohi[0,j]),abs(float(row.high)-lohi[1,j]))
            maximum=max(maximum,err);assert err<1e-12 and int(row.appearances)==int(row.unique_patients)==80 and int(row.draws)==10000;count+=1
    assert count==54
    census=pd.read_csv(PUBLIC/'RAW_COVERAGE.csv')
    for arm,g in census.groupby('arm'):assert len(g)==5 and g.patients.sum()==80 and g.channels.sum()==7635 and g.valid_windows.sum()==1471965
    stored=read(FINAL/'PATIENT_METRICS_PRIVATE.pkl');assert stored.sort_values(['arm','fold','patient'])[METRICS].equals(f.sort_values(['arm','fold','patient'])[METRICS])
    decision=jr(PUBLIC/'FINAL_DECISION.json');a=float(f[f.arm==frozen['selected']].macro_f1.mean());b=float(f[f.arm=='C2_REFERENCE'].macro_f1.mean())
    assert decision['selected']==frozen['selected'] and decision['patient_equal_Macro_F1']==a and decision['control_Macro_F1']==b
    target=bool(a>.64 and a>b);assert (decision['status']=='POINT_TARGET_MET')==target
    assert sha(BASE_SEAL)==frozen['old_reference']
    write(PUBLIC/'INDEPENDENT_OUTER_VERIFICATION.json',dict(status='PASS',patient_rows=240,patient_metric_checks=numeric,interval_checks=count,maximum_interval_difference=maximum,
        all3_complete80_verified=True,both_new_arms_full_Raw_census_verified=True,original7_features_unchanged=True,one_classification_head=True,
        candidate_selected_before_new_Outer=True,all_models_thresholds_frozen=True,old_scores_unchanged=True,no_new_training_or_inference=True))
    write(PUBLIC/'OUTER_STATUS.json',dict(status='COMPLETE_VERIFIED_FULL80_PATIENT_MACRO_TRAINING',selected=frozen['selected'],selected_patient_Macro_F1=a,control_patient_Macro_F1=b,
        target_met=target,features=7,patients=80,neural_inference_cells=10,reused_control_cells=5,scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False))
    print('PATIENT_MACRO_TRAINING_FULL80_INDEPENDENT_VERIFICATION_PASS',flush=True)


if __name__=='__main__':main()
