"""Anonymous tables only; individual identities/predictions stay private."""
from support import *

def means(g,keys=METRICS):return {m:EV.nanmean(g[m]) for m in keys}

def aggregate(a,frame,patients):
    pub=a.runtime/'public';main=[];pooled=[];folds=[];centers=[];coverage=[];rank=[];conf=[]
    for (model,version),g in patients.groupby(['model','version']):
        main.append(dict(model=model,version=version,unit='patient_macro',patients=len(g),channels=int(g.n.sum()),**means(g)))
        rank.append(dict(model=model,version=version,patients=len(g),**means(g,EV.RANKING),**{f'valid_{m}':int(np.isfinite(g[m]).sum()) for m in EV.RANKING}))
        for m in METRICS:coverage.append(dict(model=model,version=version,metric=m,total_patients=len(g),valid_patients=int(np.isfinite(g[m]).sum()),undefined_patients=int((~np.isfinite(g[m])).sum())))
    for (model,version),g in frame.groupby(['model','version']):
        pooled.append(dict(model=model,version=version,unit='pooled_channels',patients=g.patient.nunique(),channels=len(g),**EV.classification(g.y,g.probability,g.prediction)))
        conf.append(dict(model=model,version=version,scope='all',fold=None,center=None,**{k:EV.classification(g.y,g.probability,g.prediction)[k] for k in EV.COUNTS}))
    for key,target in [('fold',folds),('center',centers)]:
        for (model,version,value),g in patients.groupby(['model','version',key]):
            target.append(dict(model=model,version=version,**{key:value},unit='patient_macro',patients=len(g),channels=int(g.n.sum()),**means(g)))
        for (model,version,value),g in frame.groupby(['model','version',key]):
            values=EV.classification(g.y,g.probability,g.prediction)
            target.append(dict(model=model,version=version,**{key:value},unit='pooled_channels',patients=g.patient.nunique(),channels=len(g),**values))
            conf.append(dict(model=model,version=version,scope=key,**{key:value},**{k:values[k] for k in EV.COUNTS}))
    main=pd.DataFrame(main);pooled=pd.DataFrame(pooled);folds=pd.DataFrame(folds);centers=pd.DataFrame(centers)
    main.to_csv(pub/'main_results.csv',index=False);pooled.to_csv(pub/'pooled_metrics.csv',index=False)
    extended=pd.concat([main,pooled],ignore_index=True);extended.to_csv(pub/'extended_metrics.csv',index=False)
    folds.to_csv(pub/'outer_fold_metrics.csv',index=False);centers.to_csv(pub/'center_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(pub/'metric_coverage.csv',index=False);pd.DataFrame(rank).to_csv(pub/'ranking_metrics.csv',index=False);pd.DataFrame(conf).to_csv(pub/'confusion_matrices.csv',index=False)
    sd=[]
    for (model,version,unit),g in folds.groupby(['model','version','unit']):
        keys=METRICS if unit=='patient_macro' else EV.CLASSIFICATION+['predicted_ez_fraction']
        for m in keys:sd.append(dict(model=model,version=version,unit=unit,metric=m,folds_valid=int(np.isfinite(g[m]).sum()),fold_mean=EV.nanmean(g[m]),fold_std_ddof1=float(np.nanstd(g[m],ddof=1))))
    pd.DataFrame(sd).to_csv(pub/'outer_fold_mean_std.csv',index=False)
    deltas=[];intervals=[];rng=np.random.default_rng(42);ids=np.asarray(sorted(patients.patient.unique()));draw=rng.integers(0,len(ids),size=(10000,len(ids)));counts=np.stack([(draw==j).sum(1) for j in range(len(ids))],axis=1)
    for model in sorted(patients.model.unique()):
        left=patients[(patients.model==model)&(patients.version=='A')].set_index('patient').loc[ids]
        right=patients[(patients.model==model)&(patients.version=='B')].set_index('patient').loc[ids]
        assert np.array_equal(left.fold,right.fold) and np.array_equal(left.n,right.n)
        diff=right[METRICS].to_numpy()-left[METRICS].to_numpy();valid=np.isfinite(diff)
        num=counts@np.where(valid,diff,0);den=counts@valid.astype(int);dist=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        for j,m in enumerate(METRICS):
            intervals.append(dict(model=model,metric=m,paired_patient_delta=EV.nanmean(diff[:,j]),ci_low=float(np.nanpercentile(dist[:,j],2.5)) if np.isfinite(dist[:,j]).any() else np.nan,ci_high=float(np.nanpercentile(dist[:,j],97.5)) if np.isfinite(dist[:,j]).any() else np.nan,paired_valid_patients=int(valid[:,j].sum()),draws=10000,seed=42,lower_is_better=m in ['brier','log_loss','ece','fpr','fnr']))
        for unit,df in [('patient_macro',main),('pooled_channels',pooled)]:
            l=df[(df.model==model)&(df.version=='A')].iloc[0];r=df[(df.model==model)&(df.version=='B')].iloc[0]
            keys=METRICS if unit=='patient_macro' else EV.CLASSIFICATION+['predicted_ez_fraction']
            deltas.append(dict(model=model,unit=unit,**{f'delta_{m}':r[m]-l[m] for m in keys},MacroF1_improved_patients=int((diff[:,METRICS.index('macro_f1')]>0).sum()),MacroF1_improved_folds=int((right.macro_f1.groupby(right.fold).mean()-left.macro_f1.groupby(left.fold).mean()>0).sum())))
    pd.DataFrame(deltas).to_csv(pub/'pr_delta.csv',index=False)
    pd.DataFrame(intervals).to_csv(pub/'paired_pr_intervals.csv',index=False)

def reference(a,lock,meta,y):
    oldfr=jread(a.references/'FROZEN_BEFORE_OUTER_PRIVATE.json');oldseal=jread(a.references/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');status=jread(a.references/'public/OUTER_STATUS.json')
    assert sha(a.references/'FROZEN_BEFORE_OUTER_PRIVATE.json')==lock['reference_freeze_sha256']==oldseal['freeze_sha256']==status['freeze_sha256']
    assert sha(a.references/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==lock['reference_seal_sha256']==status['seal_sha256']
    assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
    frames=[]
    for r in oldfr['selections']:
        old=r['arm'];fold=r['fold'];p=a.references/f'scores_private/{old}_fold{fold}.pkl'
        assert sha(p)==oldseal['files'][f'{old}_{fold}'] and sha(Path(r['checkpoint']))==r['checkpoint_sha256']
        q=read(p);ix=q['indices'];assert q['freeze_sha256']==lock['reference_freeze_sha256'] and q['checkpoint_sha256']==r['checkpoint_sha256']
        assert np.array_equal(ix,np.flatnonzero(np.isin(meta['patient'],list(roles(meta,fold)['test']))))
        nez=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();prob=1-nez.astype(float);pred=nez<r['threshold']
        frames.append(pd.DataFrame(dict(model='D1' if old=='D1' else 'M8',version='frozen_reference',fold=fold,patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],prediction=pred,probability=prob)))
    frame=pd.concat(frames,ignore_index=True);patients=pd.DataFrame(EV.patient_rows(frame));save(a.runtime/'REFERENCE_PATIENT_METRICS_PRIVATE.pkl',patients)
    save(a.runtime/'REFERENCE_PREDICTIONS_PRIVATE.pkl',frame)
    out=[]
    for model,g in patients.groupby('model'):
        assert len(g)==g.patient.nunique()==80 and g.n.sum()==7635
        expected={'D1':[.732635663577925,.5271750948653131,.6165564790133051,.4085311473799952],'M8':[.7623966036157341,.5540672861984975,.6340971098987545,.43408686014555975]}[model]
        assert np.allclose(g[['auroc','auprc','macro_f1','f1']].mean(),expected,atol=1e-12,rtol=0)
        out.append(dict(model=model,version='frozen_reference',unit='patient_macro',patients=80,channels=7635,**means(g)))
    ref=pd.DataFrame(out);pub=a.runtime/'public';ref.to_csv(pub/'frozen_reference_metrics.csv',index=False)
    main=pd.read_csv(pub/'main_results.csv');pd.concat([main,ref],ignore_index=True).to_csv(pub/'reference_comparison.csv',index=False)
    write(pub/'REFERENCE_REUSE_AUDIT.json',dict(status='PASS',models=['D1','M8'],patients_per_model=80,channels_per_model=7635,original_primary_metrics_exact=True,expanded_metrics_recomputed_from_original_frozen_scores=True,no_checkpoint_inference_or_modification=True,original_freeze_sha256=lock['reference_freeze_sha256'],original_seal_sha256=lock['reference_seal_sha256']))
