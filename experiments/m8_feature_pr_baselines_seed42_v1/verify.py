"""Independent reconstruction from private CSVs; no refit or new predictions."""
from support import *
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,cohen_kappa_score,brier_score_loss,log_loss,confusion_matrix,roc_auc_score,average_precision_score

def check(g,row):
    y=np.asarray(g.y,dtype=int);p=np.asarray(g.probability,dtype=float);pred=np.asarray(g.prediction,dtype=bool)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    assert [tp,tn,fp,fn]==[row.TP,row.TN,row.FP,row.FN]
    vals=dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=np.nan),recall=recall_score(y,pred,zero_division=np.nan),specificity=recall_score(y,pred,pos_label=0,zero_division=np.nan),f1=f1_score(y,pred,zero_division=np.nan),macro_f1=f1_score(y,pred,average='macro',labels=[0,1],zero_division=np.nan),weighted_f1=f1_score(y,pred,average='weighted',labels=[0,1],zero_division=np.nan),macro_precision=precision_score(y,pred,average='macro',labels=[0,1],zero_division=np.nan),macro_recall=recall_score(y,pred,average='macro',labels=[0,1],zero_division=np.nan),mcc=matthews_corrcoef(y,pred) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) else np.nan,kappa=cohen_kappa_score(y,pred,labels=[0,1]),brier=brier_score_loss(y,p),log_loss=log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1]),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),npv=precision_score(y,pred,pos_label=0,zero_division=np.nan),fpr=fp/(tn+fp),fnr=fn/(tp+fn))
    b=np.digitize(p,np.arange(1,10)/10,right=False);n=np.bincount(b,minlength=10);positive=np.bincount(b,weights=y,minlength=10);prob=np.bincount(b,weights=p,minlength=10)
    vals['ece']=float(np.sum(np.abs(positive-prob))/len(y))
    for m,v in vals.items():assert np.isclose(v,row[m],atol=1e-12,rtol=0,equal_nan=True),m
    order=sorted(range(len(y)),key=lambda i:(-p[i],str(g.channel.iloc[i])));yy=y[order];positive_count=int(y.sum());k=min(5,len(y));gains=[1/np.log2(i+2) for i in range(k)]
    vals=dict(mrr=1/(list(yy).index(1)+1),hit_at_1=float(1 in yy[:1]),hit_at_3=float(1 in yy[:min(3,len(y))]),hit_at_5=float(1 in yy[:k]),precision_at_3=float(sum(yy[:min(3,len(y))])/min(3,len(y))),recall_at_3=float(sum(yy[:min(3,len(y))])/positive_count),ndcg_at_5=sum(v*w for v,w in zip(yy[:k],gains))/sum(gains[:min(positive_count,k)]))
    for m,v in vals.items():assert np.isclose(v,row[m],atol=1e-12,rtol=0),m

def run(a):
    lock,core,rec,banks,x,binding=setup(a);pub=a.runtime/'public'
    fr=jread(a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jread(a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');status=jread(pub/'OUTER_STATUS.json')
    assert fr['binding']==binding and status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
    assert seal['freeze_sha256']==sha(a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json')==status['freeze_sha256']
    assert sha(a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==status['seal_sha256']
    fp=a.runtime/'private/results/predictions.csv';pp=a.runtime/'private/results/per_patient_metrics.csv'
    assert sha(fp)==status['private_predictions_csv_sha256'] and sha(pp)==status['private_patient_csv_sha256']
    frame=pd.read_csv(fp,dtype={'patient':str,'channel':str},float_precision='round_trip');patients=pd.read_csv(pp,dtype={'patient':str},float_precision='round_trip')
    original=read(a.runtime/'PREDICTIONS_PRIVATE.pkl');assert len(frame)==len(original)==76350
    assert np.array_equal(frame.probability,original.probability) and np.array_equal(frame.prediction,original.prediction)
    assert len(patients)==800 and patients.patient.nunique()==80
    indexed=patients.set_index(['model','version','fold','patient'])
    for key,g in frame.groupby(['model','version','fold','patient']):check(g,indexed.loc[key])
    refs=read(a.runtime/'REFERENCE_PATIENT_METRICS_PRIVATE.pkl');refpred=read(a.runtime/'REFERENCE_PREDICTIONS_PRIVATE.pkl');ri=refs.set_index(['model','version','fold','patient'])
    for key,g in refpred.groupby(['model','version','fold','patient']):check(g,ri.loc[key])
    for r in fr['selections']:
        cp=Path(r['path']);assert sha(cp)==r['model_sha256'];q=read(cp);fold=r['fold'];d=banks[fold-1];raw=x[d['archive_indices']]
        z=transform_view(core,raw,d['patient'],r['version']=='B');tr=d['train'];imp=q['bundle']['pre']['imputer'];sc=q['bundle']['pre']['scaler']
        assert np.allclose(imp.statistics_,np.mean(z[tr],axis=0),rtol=0,atol=1e-12)
        if sc is not None:
            assert np.allclose(sc.mean_,imp.transform(z[tr]).mean(0),rtol=0,atol=1e-12)
            assert np.allclose(sc.var_,imp.transform(z[tr]).var(0),rtol=1e-12,atol=1e-12)
        assert q['bundle']['model'].n_features_in_==14
        out=a.runtime/f"outer_scores_private/{r['model']}_{r['version']}_fold{fold}.pkl"
        assert sha(out)==seal['files'][f"{r['model']}_{r['version']}_{fold}"]
        v=read(out);assert np.array_equal(v['indices'],np.flatnonzero(np.isin(rec['meta']['patient'],list(roles(rec['meta'],fold)['test']))))
    expected=patients.groupby(['model','version'])[METRICS].mean();main=pd.read_csv(pub/'main_results.csv').set_index(['model','version'])
    assert np.allclose(main.loc[expected.index,METRICS],expected,rtol=0,atol=1e-12,equal_nan=True)
    pooled=pd.read_csv(pub/'pooled_metrics.csv').set_index(['model','version'])
    for key,g in frame.groupby(['model','version']):
        vals=EV.classification(g.y,g.probability,g.prediction)
        for k,v in vals.items():assert np.isclose(v,pooled.loc[key,k],atol=1e-12,rtol=0,equal_nan=True)
    folds=pd.read_csv(pub/'outer_fold_metrics.csv');f=folds[folds.unit=='patient_macro'].set_index(['model','version','fold']);expect=patients.groupby(['model','version','fold'])[METRICS].mean()
    assert np.allclose(f.loc[expect.index,METRICS],expect,rtol=0,atol=1e-12,equal_nan=True)
    intervals=pd.read_csv(pub/'paired_pr_intervals.csv');ids=sorted(patients.patient.unique());sample=np.random.default_rng(42).choice(ids,size=(10000,80),replace=True)
    for model in sorted(patients.model.unique()):
        l=patients[(patients.model==model)&(patients.version=='A')].set_index('patient').loc[ids,METRICS]
        r=patients[(patients.model==model)&(patients.version=='B')].set_index('patient').loc[ids,METRICS];delta=r.to_numpy()-l.to_numpy()
        counts=np.stack([(sample==pid).sum(1) for pid in ids],axis=1);valid=np.isfinite(delta);num=counts@np.where(valid,delta,0);den=counts@valid
        dist=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        for j,m in enumerate(METRICS):
            q=intervals[(intervals.model==model)&(intervals.metric==m)].iloc[0]
            assert np.allclose([q.paired_patient_delta,q.ci_low,q.ci_high],[EV.nanmean(delta[:,j]),*np.nanpercentile(dist[:,j],[2.5,97.5])],atol=1e-12,rtol=0,equal_nan=True)
    for p in pub.glob('*.csv'):
        columns=pd.read_csv(p,nrows=0).columns
        assert not set(columns)&{'patient','channel','y','probability','prediction','balanced_accuracy','BA'}
    write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',new_model_fold_cells=50,groups=10,patients_per_group=80,channels_per_group=7635,private_csv_roundtrip_exact=True,patient_metric_rows_checked=960,private_predictions_sha256=sha(fp),private_patient_metrics_sha256=sha(pp),freeze_sha256=sha(a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json'),source_hashes=binding['code'],anonymous_aggregate_hashes={p.name:sha(p) for p in pub.glob('*.csv')},paired_bootstrap_independently_reproduced=True,no_forbidden_metric=True,feature_dimensions=14))
    write(pub/'RUN_STATUS.json',dict(status='COMPLETE_VERIFIED',groups=10,selected_cells=50,candidate_cells=190,patient_metric_rows=800,reference_patient_rows=160,scientific_status='EXPLORATORY_REPEATED_TEST',installed_skills_used=[]))
    print('M8_FEATURE_PR_BASELINES_INDEPENDENT_VERIFICATION_PASS',flush=True)
