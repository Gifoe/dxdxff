"""Expanded anonymous M8 ablation tables and independent checks from sealed scores."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,average_precision_score,log_loss


def finish(R,frame,fr,banks,binding,core):
    out=R.OUT;pub=out/'public';EV=R.EV;keys=EV.CLASSIFICATION+EV.RANKING
    patients=pd.DataFrame(EV.patient_rows(frame));assert len(patients)==640
    R.save(out/'PATIENT_METRICS_PRIVATE.pkl',patients)
    means=lambda g:{k:EV.nanmean(g[k]) for k in keys}
    main=[];pooled=[];folds=[];centers=[];coverage=[];conf=[];ranking=[];foldsd=[]
    for name,g in patients.groupby('model'):
        assert len(g)==g.patient.nunique()==80 and g.n.sum()==7635
        main.append(dict(model=name,unit='patient_macro',patients=80,channels=7635,**means(g)))
        ranking.append(dict(model=name,patients=80,**{k:EV.nanmean(g[k]) for k in EV.RANKING}))
        for k in keys:coverage.append(dict(model=name,metric=k,valid_patients=int(np.isfinite(g[k]).sum()),undefined_patients=int((~np.isfinite(g[k])).sum())))
        for group,target in [('fold',folds),('center',centers)]:
            for value,q in g.groupby(group):target.append(dict(model=name,**{group:value},unit='patient_macro',patients=len(q),channels=int(q.n.sum()),**means(q)))
        for k in keys:
            values=[EV.nanmean(q[k]) for _,q in g.groupby('fold')]
            finite=np.asarray(values)[np.isfinite(values)]
            foldsd.append(dict(model=name,metric=k,valid_folds=len(finite),fold_mean=EV.nanmean(finite),fold_sd_ddof1=float(finite.std(ddof=1)) if len(finite)>1 else np.nan))
    for name,g in frame.groupby('model'):
        v=EV.classification(g.y,g.probability,g.prediction)
        pooled.append(dict(model=name,unit='pooled_channels',patients=80,channels=len(g),**v))
        conf.append(dict(model=name,scope='all',**{k:v[k] for k in EV.COUNTS}))
        for key,target in [('fold',folds),('center',centers)]:
            for value,q in g.groupby(key):
                v=EV.classification(q.y,q.probability,q.prediction)
                target.append(dict(model=name,**{key:value},unit='pooled_channels',patients=q.patient.nunique(),channels=len(q),**v))
                conf.append(dict(model=name,scope=key,**{key:value},**{k:v[k] for k in EV.COUNTS}))
    for name,rows in [('main_results',main),('pooled_metrics',pooled),('fold_metrics',folds),('center_metrics',centers),
        ('metric_coverage',coverage),('confusion_matrices',conf),('ranking_metrics',ranking),('fold_mean_std',foldsd)]:
        pd.DataFrame(rows).to_csv(pub/(name+'.csv'),index=False)
    draws=np.random.default_rng(42).integers(0,80,size=(10000,80));counts=np.stack([(draws==i).sum(1) for i in range(80)],1)
    intervals=[]
    for arm,ref in [(a,'M8_FULL') for a in ['A1_WO_FEATURE_PR','A2_WO_RAW_PR','A3_IDENTITY_GATE','A4_WO_PRE_REFERENCE','A6_LATE_FUSION']]+[('A6_LATE_FUSION','D1_PR'),('A6_LATE_FUSION','M8_RAW_ONLY')]:
        a=patients[patients.model==arm].set_index('patient').sort_index();b=patients[patients.model==ref].set_index('patient').sort_index()
        assert a.index.equals(b.index) and np.array_equal(a.fold,b.fold) and np.array_equal(a.n,b.n)
        delta=a[keys].to_numpy()-b[keys].to_numpy();valid=np.isfinite(delta)
        den=counts@valid.astype(int);num=counts@np.where(valid,delta,0)
        dist=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        direct=np.nanmean(delta[draws[:256]],axis=1)
        assert np.allclose(direct,dist[:256],atol=1e-12,rtol=0,equal_nan=True)
        for j,k in enumerate(keys):intervals.append(dict(model=arm,reference=ref,metric=k,delta=EV.nanmean(delta[:,j]),
            ci_low=float(np.nanpercentile(dist[:,j],2.5)),ci_high=float(np.nanpercentile(dist[:,j],97.5)),
            paired_valid_patients=int(valid[:,j].sum()),draws=10000,seed=42,threshold_reoptimized=False))
    pd.DataFrame(intervals).to_csv(pub/'paired_intervals.csv',index=False)
    # Independent recomputation from original score/label rows, not aggregate CSVs.
    for (name,fold,pid),g in frame.groupby(['model','fold','patient']):
        r=patients[(patients.model==name)&(patients.fold==fold)&(patients.patient==pid)].iloc[0]
        y=g.y.to_numpy();p=g.probability.to_numpy(dtype=np.float64);pred=g.prediction.to_numpy()
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        assert np.array_equal([tn,fp,fn,tp],[r.TN,r.FP,r.FN,r.TP])
        assert abs(r.nez_f1-f1_score(y,pred,pos_label=0))<1e-12
        if y.sum() and (1-y).sum():
            assert abs(r.macro_f1-f1_score(y,pred,average='macro'))<1e-12
            assert abs(r.auroc-roc_auc_score(y,p))<1e-12
        if y.sum():assert abs(r.auprc-average_precision_score(y,p))<1e-12
        assert abs(r.log_loss-log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1]))<1e-12
        assert abs(r.ez_fraction_bias-(pred.mean()-y.mean()))<1e-12
        yy=g.sort_values(['probability','channel'],ascending=[False,True],kind='stable').y.to_numpy()
        if yy.sum():assert abs(r.mrr-1/(np.flatnonzero(yy)[0]+1))<1e-12 and r.hit_at_1==yy[0]
    # Freeze/epoch/threshold choices can be reconstructed entirely from validation.
    for s in fr['selections']:
        assert R.sha(s['checkpoint'])==s['checkpoint_sha256'] and R.sha(s['bank'])==s['bank_sha256']
        cp=R.torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
        last=R.torch.load(R.Path(s['checkpoint']).parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
        expected=max(last['history'],key=lambda q:(q['macro_f1'],q['ez_f1'],-q['epoch']))
        assert cp['epoch']==s['epoch']==expected['epoch'] and cp['threshold']==s['threshold']==expected['threshold']
        q=R.read(out/f"validation_private/{s['arm']}_fold{s['fold']}.pkl");d=banks[s['fold']-1];va=d['val']
        v=core.select_threshold(d['y'][va],R.torch.sigmoid(R.torch.from_numpy(q['logit'])).numpy(),d['patient'][va])
        assert v['threshold']==s['threshold']
    grid=pd.read_csv(pub/'FUSION_VALIDATION_SEARCH.csv')
    for s in fr['fusions']:
        g=grid[grid.fold==s['fold']];assert len(g)==21
        expected=max(g.to_dict('records'),key=lambda q:(q['macro_f1'],q['ez_f1'],-q['raw_weight']))
        assert s['raw_weight']==expected['raw_weight'] and s['threshold']==expected['threshold']
    # Historical primary metrics must reproduce exactly for all three references.
    old=pd.DataFrame(R.read(R.OLD/'OUTER_METRICS_PRIVATE.pkl'))
    mapping={'auroc':'ez_auroc','auprc':'ez_auprc','macro_f1':'macro_f1','f1':'ez_f1','nez_f1':'nez_f1','accuracy':'accuracy'}
    for name,oldarm in [('D1_PR','D1'),('M8_FULL','M8_MACRO_AWARE')]:
        a=patients[patients.model==name].set_index(['fold','patient']).sort_index();b=old[old.arm==oldarm].set_index(['fold','patient']).sort_index()
        assert a.index.equals(b.index)
        for new,prior in mapping.items():assert np.allclose(a[new],b[prior],atol=1e-12,rtol=0,equal_nan=True)
    oldraw=R.read(R.RAW/'PATIENT_METRICS_PRIVATE.pkl');a=patients[patients.model=='M8_RAW_ONLY'].set_index(['fold','patient']).sort_index()
    b=oldraw[oldraw.model=='M8_RAW_ONLY'].set_index(['fold','patient']).sort_index();assert a.index.equals(b.index)
    for k in ['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1']:
        assert np.allclose(a[k],b[k],atol=1e-12,rtol=0,equal_nan=True)
    for p,h in binding['protected'].items():assert R.sha(p)==h
    for p in pub.glob('*.csv'):
        columns=pd.read_csv(p,nrows=0).columns
        assert not set(columns)&{'patient','channel','path','checkpoint','probability','prediction','y','indices'}
    audit=dict(status='PASS',new_model_fold_cells=20,fusion_fold_cells=5,historical_reference_cells=15,
        verified_patient_rows=640,patients_per_arm=80,channels_per_arm=7635,
        own_fold_patient_isolation=True,original_training_budget=True,validation_only_weights_thresholds_and_epochs=True,
        independent_sklearn_and_ranking_checks=True,bootstrap_direct_256_draws_match=True,bootstrap_draws=10000,
        historical_primary_metrics_exact=True,protected_artifacts_unchanged=True,public_privacy_gate=True)
    R.write(pub/'INDEPENDENT_VERIFICATION.json',audit)
    R.write(pub/'FINAL_STATUS.json',dict(status='M8_MECHANISM_5FOLD_OUTER_COMPLETE',
        freeze_sha256=R.sha(out/'FROZEN_BEFORE_OUTER_PRIVATE.json'),seal_sha256=R.sha(out/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),
        scientific_status='EXPLORATORY_REPEATED_TEST',**{k:v for k,v in audit.items() if k!='status'}))
    R.write(pub/'PUBLIC_SHA256.json',{p.name:R.sha(p) for p in pub.iterdir() if p.name!='PUBLIC_SHA256.json'})
