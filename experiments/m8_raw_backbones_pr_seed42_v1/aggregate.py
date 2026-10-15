"""Anonymous complete/partial aggregate tables, no model fitting or test tuning."""
from support import *

METRICS=EV.CLASSIFICATION+EV.RANKING

def means(g,metrics=METRICS):return {m:EV.nanmean(g[m]) for m in metrics}

def aggregate_results(runtime,lock):
    pub=runtime/'public';completed=[m for m in lock['models'] if (pub/f'{m}_OUTER_STATUS.json').exists()]
    assert completed
    all_frames=[];all_patients=[]
    for m in completed:
        status=jr(pub/f'{m}_OUTER_STATUS.json');assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
        assert status['predictions_sha256']==sha(runtime/m/'PREDICTIONS_PRIVATE.pkl')
        assert status['patient_metrics_sha256']==sha(runtime/m/'PATIENT_METRICS_PRIVATE.pkl')
        all_frames.append(read(runtime/m/'PREDICTIONS_PRIVATE.pkl'));all_patients.append(read(runtime/m/'PATIENT_METRICS_PRIVATE.pkl'))
    frame=pd.concat(all_frames,ignore_index=True);patients=pd.concat(all_patients,ignore_index=True)
    main=[];pooled=[];folds=[];centers=[];coverage=[];conf=[];ranks=[];deltas=[];intervals=[];foldstd=[]
    ids=np.asarray(sorted(patients.patient.unique()));assert len(ids)==80
    draw=np.random.default_rng(42).integers(0,80,size=(10000,80));counts=np.stack([(draw==j).sum(1) for j in range(80)],axis=1)
    for (model,version),g in patients.groupby(['model','version']):
        assert len(g)==g.patient.nunique()==80 and g.n.sum()==7635
        main.append(dict(model=model,version=version,unit='patient_macro',patients=80,channels=7635,**means(g)))
        ranks.append(dict(model=model,version=version,**means(g,EV.RANKING)))
        for m in METRICS:coverage.append(dict(model=model,version=version,metric=m,total_patients=80,valid_patients=int(np.isfinite(g[m]).sum()),undefined_patients=int((~np.isfinite(g[m])).sum())))
        for key,target in [('fold',folds),('center',centers)]:
            for v,q in g.groupby(key):target.append(dict(model=model,version=version,unit='patient_macro',**{key:v},patients=len(q),channels=int(q.n.sum()),**means(q)))
    for (model,version),g in frame.groupby(['model','version']):
        v=EV.classification(g.y,g.probability,g.prediction)
        pooled.append(dict(model=model,version=version,unit='pooled_channels',patients=80,channels=len(g),**v))
        conf.append(dict(model=model,version=version,scope='all',**{k:v[k] for k in EV.COUNTS}))
        for key,target in [('fold',folds),('center',centers)]:
            for kk,q in g.groupby(key):target.append(dict(model=model,version=version,unit='pooled_channels',**{key:kk},patients=q.patient.nunique(),channels=len(q),**EV.classification(q.y,q.probability,q.prediction)))
    main=pd.DataFrame(main);pooled=pd.DataFrame(pooled);folds=pd.DataFrame(folds)
    for (model,version,unit),g in folds.groupby(['model','version','unit']):
        for metric in (METRICS if unit=='patient_macro' else EV.CLASSIFICATION):foldstd.append(dict(model=model,version=version,unit=unit,metric=metric,fold_mean=EV.nanmean(g[metric]),fold_std_ddof1=float(np.nanstd(g[metric],ddof=1)),valid_folds=int(np.isfinite(g[metric]).sum())))
    for model in completed:
        left=patients[(patients.model==model)&(patients.version=='Original')].set_index('patient').loc[ids]
        right=patients[(patients.model==model)&(patients.version=='PR')].set_index('patient').loc[ids]
        assert np.array_equal(left.fold,right.fold) and np.array_equal(left.n,right.n)
        delta=right[METRICS].to_numpy()-left[METRICS].to_numpy();valid=np.isfinite(delta)
        num=counts@np.where(valid,delta,0);den=counts@valid.astype(int);dist=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        deltas.append(dict(model=model,unit='patient_macro',**{f'delta_{m}':EV.nanmean(delta[:,j]) for j,m in enumerate(METRICS)}))
        for j,m in enumerate(METRICS):intervals.append(dict(model=model,metric=m,delta=EV.nanmean(delta[:,j]),ci_low=float(np.nanpercentile(dist[:,j],2.5)),ci_high=float(np.nanpercentile(dist[:,j],97.5)),valid_paired_patients=int(valid[:,j].sum()),draws=10000,seed=42,threshold_reoptimized=False))
    main.to_csv(pub/'main_results.csv',index=False)
    pooled.to_csv(pub/'pooled_metrics.csv',index=False)
    pd.concat([main,pooled],ignore_index=True).to_csv(pub/'extended_metrics.csv',index=False)
    folds.to_csv(pub/'fold_metrics.csv',index=False);pd.DataFrame(foldstd).to_csv(pub/'fold_mean_std.csv',index=False)
    pd.DataFrame(centers).to_csv(pub/'center_metrics.csv',index=False);pd.DataFrame(coverage).to_csv(pub/'metric_coverage.csv',index=False)
    pd.DataFrame(conf).to_csv(pub/'confusion_matrices.csv',index=False);pd.DataFrame(ranks).to_csv(pub/'ranking_metrics.csv',index=False)
    pd.DataFrame(deltas).to_csv(pub/'pr_improvement.csv',index=False);pd.DataFrame(intervals).to_csv(pub/'paired_pr_intervals.csv',index=False)
    selections=pd.concat([pd.read_csv(pub/f'{m}_MODEL_SELECTION.csv') for m in completed],ignore_index=True)
    selections.to_csv(pub/'model_parameters.csv',index=False)
    # Same-cohort frozen M8/D1 references, no neural inference or model changes.
    oldroot=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime')
    assert sha(oldroot/'FROZEN_BEFORE_OUTER_PRIVATE.json')=='4882d62e0492618a277897e56e434578b8481b7a8192213d01cd3a5b5f420d5d'
    assert sha(oldroot/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')=='aa5ae8daeec58fb392e6587a89f24cb26ce34fc83a1125f8f6ae4a575ab81e02'
    assert jr(oldroot/'public/OUTER_INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    original=read(oldroot/'OUTER_METRICS_PRIVATE.pkl')
    old=pd.DataFrame(original);refs=[]
    for arm,g in old.groupby('arm'):
        assert len(g)==g.patient.nunique()==80 and g.channels.sum()==7635
        assert set(g.patient.astype(str))==set(patients.patient.astype(str))
        refs.append(dict(model='D1' if arm=='D1' else 'M8',unit='patient_macro',patients=80,channels=7635,auroc=g.ez_auroc.mean(),auprc=g.ez_auprc.mean(),macro_f1=g.macro_f1.mean(),f1=g.ez_f1.mean(),nez_f1=g.nez_f1.mean(),accuracy=g.accuracy.mean(),mrr=g.ez_mrr.mean(),hit_at_1=g.top1_is_ez.mean(),ez_fraction_bias=((g.FP-g.FN)/g.channels).mean(),comparison='same cohort/splits, different physiological context and objective'))
    pd.DataFrame(refs).to_csv(pub/'frozen_reference_metrics.csv',index=False)
    write(pub/'protocol_audit.json',dict(status='COMPLETE' if len(completed)==4 else 'PARTIAL_COMPLETE',completed_backbones=completed,completed_configurations=len(completed)*2,required_configurations=8,patients_per_configuration=80,channels_per_configuration=7635,seed=42,all_patient_disjoint_splits=True,thresholds_validation_only=True,full_patient_PR=True,no_handcrafted_feature_or_D1_input=True,no_window_subsampling=True,scientific_status='EXPLORATORY_REPEATED_TEST'))
