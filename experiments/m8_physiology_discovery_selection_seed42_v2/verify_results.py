"""Independent score-only metric/table/bootstrap replay. Never fit or predict."""
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from common import *
from report import frames


def close(a,b):
    assert np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=0,atol=2e-12,equal_nan=True),(a,b)


def independent(g):
    y=g.y.to_numpy(dtype=bool);p=g.probability.to_numpy(dtype=float);v=g.prediction.to_numpy(dtype=bool)
    tp=int((y&v).sum());tn=int((~y&~v).sum());fp=int((~y&v).sum());fn=int((y&~v).sum())
    pos=tp+fn;neg=tn+fp;n=len(y)
    f1=2*tp/(2*tp+fp+fn) if pos else np.nan
    f0=2*tn/(2*tn+fp+fn) if neg else np.nan
    auc=((rankdata(p)[y].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else np.nan
    rank=y[np.lexsort((g.channel.to_numpy(dtype=str),-p))]
    return dict(TP=tp,TN=tn,FP=fp,FN=fn,n=n,positives=pos,negatives=neg,accuracy=(tp+tn)/n,
        auroc=auc,auprc=average_precision_score(y,p) if pos else np.nan,macro_f1=np.nanmean([f1,f0]),
        f1=f1,nez_f1=f0,mrr=1/(np.flatnonzero(rank)[0]+1) if pos else np.nan,
        hit_at_1=float(rank[0]) if pos else np.nan,ez_fraction_bias=(fp-fn)/n)


def boot(a,ids):
    # Independent multinomial-count implementation; identical seed and unit.
    unique=np.unique(ids);n=len(unique);indices=[np.flatnonzero(ids==u) for u in unique]
    sums=np.array([np.nansum(a[ix],0) for ix in indices]);counts=np.array([np.isfinite(a[ix]).sum(0) for ix in indices])
    random=np.random.default_rng(42).integers(n,size=(10000,n))
    weights=np.array([np.bincount(row,minlength=n) for row in random],dtype=float)
    den=weights@counts;dist=np.divide(weights@sums,den,out=np.full(den.shape,np.nan),where=den>0)
    return np.nanmean(a,0),np.nanpercentile(dist,[2.5,97.5],axis=0),dist


def verify_scope(scope,expected):
    frame=frames(scope);rows=[]
    for (name,fold,pid),g in frame.groupby(['model','fold','patient']):
        rows.append(dict(model=name,fold=fold,patient=pid,center=g.center.iloc[0],**independent(g)))
    pt=pd.DataFrame(rows);assert len(pt)==expected
    saved=read(ROOT/(scope.upper()+'_PATIENT_METRICS_PRIVATE.pkl')).sort_values(['model','fold','patient'])
    pt=pt.sort_values(['model','fold','patient']);assert list(zip(pt.model,pt.fold,pt.patient))==list(zip(saved.model,saved.fold,saved.patient))
    close(pt[METRICS+['TP','TN','FP','FN','n','positives','negatives']],saved[METRICS+['TP','TN','FP','FN','n','positives','negatives']])
    directory=PUB/scope
    tables={n:pd.read_csv(directory/(n+'.csv')) for n in ['MAIN_RESULTS','FOLD_RESULTS','SOURCE_RESULTS','FOLD_STABILITY','METRIC_COVERAGE','PAIRED_COMPARISONS','ABSOLUTE_INTERVALS','CONFUSION_COUNTS']}
    matrices={};ids=None
    for name,g in pt.groupby('model'):
        g=g.sort_values(['fold','patient']);a=g[METRICS].to_numpy(float);matrices[name]=a
        if ids is None:ids=g.patient.to_numpy();keys=list(zip(g.fold,g.patient))
        assert list(zip(g.fold,g.patient))==keys
        r=tables['MAIN_RESULTS'].query('model == @name').iloc[0];close(r[METRICS],np.nanmean(a,0));assert r.appearances==len(g) and r.unique_patients==g.patient.nunique() and r.channels==g.n.sum()
        for col,file in [('fold','FOLD_RESULTS'),('center','SOURCE_RESULTS')]:
            for key,h in g.groupby(col):
                r=tables[file];r=r[(r.model==name)&(r[col]==key)].iloc[0];close(r[METRICS],np.nanmean(h[METRICS],0));assert r.unique_patients==h.patient.nunique()
        for m in METRICS:
            r=tables['FOLD_STABILITY'];r=r[(r.model==name)&(r.metric==m)].iloc[0]
            values=g.groupby('fold')[m].mean().to_numpy();close([r.fold_equal_mean,r.fold_sample_sd],[np.nanmean(values),np.nanstd(values,ddof=1)])
            r=tables['METRIC_COVERAGE'];r=r[(r.model==name)&(r.metric==m)].iloc[0];assert r.finite_appearances==np.isfinite(g[m]).sum() and r.total_appearances==len(g)
        mu,ci,_=boot(a,ids)
        for j,m in enumerate(METRICS):
            r=tables['ABSOLUTE_INTERVALS'];r=r[(r.model==name)&(r.metric==m)].iloc[0];close([r['mean'],r.ci_low,r.ci_high],[mu[j],ci[0,j],ci[1,j]])
    for (name,ref),t in tables['PAIRED_COMPARISONS'].groupby(['model','reference']):
        d=matrices[name]-matrices[ref];mu,ci,dist=boot(d,ids)
        for j,m in enumerate(METRICS):
            r=t[t.metric==m].iloc[0];close([r.delta,r.ci_low,r.ci_high],[mu[j],ci[0,j],ci[1,j]])
            assert r.valid_appearances==np.isfinite(d[:,j]).sum() and r.draws==10000 and not r.threshold_reoptimized
            if name=='M8__SELECTED' and ref=='M8__OLD7' and m!='ez_fraction_bias':
                close([r.selected8endpoint_bonferroni_ci_low,r.selected8endpoint_bonferroni_ci_high],np.nanpercentile(dist[:,j],[.3125,99.6875]))
    for _,r in tables['CONFUSION_COUNTS'].iterrows():
        g=pt[pt.model==r.model]
        if r.scope=='fold':g=g[g.fold==int(r['group'])]
        elif r.scope=='center':g=g[g.center==r['group']]
        close(r[['TP','TN','FP','FN','n','positives','negatives']],g[['TP','TN','FP','FN','n','positives','negatives']].sum())
    print('INDEPENDENT_FINAL_SCOPE_PASS',scope,len(pt),flush=True)
    return len(pt),len(tables['PAIRED_COMPARISONS'])


def main():
    assert jr(PUB/'REPORT_VERIFICATION.json')['status']=='PASS'
    assert jr(PUB/'INDEPENDENT_SELECTION_VERIFICATION.json')['status']=='PASS'
    before=sha(ROOT/'SELECTED_FEATURES_PRIVATE.json');assert before==jr(PUB/'SCREEN_STATUS.json')['feature_freeze_sha256']
    counts={scope:verify_scope(scope,n) for scope,n in [('development',1105),('outer',1680)]}
    a=read(ROOT/'ASSESSMENT_PATIENTS_PRIVATE.pkl');t=pd.read_csv(PUB/'CARDINALITY_CURVE.csv')
    for k,g in a.groupby('k'):
        mu,ci,_=boot(g[METRICS].to_numpy(float),g.patient.to_numpy())
        for j,m in enumerate(METRICS):
            r=t[(t.k==k)&(t.metric==m)].iloc[0];close([r['mean'],r.ci_low,r.ci_high],[mu[j],ci[0,j],ci[1,j]])
    # Public feature sets and model allocations cannot be altered by outcomes.
    chosen=jr(ROOT/'SELECTED_FEATURES_PRIVATE.json')
    models=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl')['selections'];support=read(ROOT/'SUPPORT_MODEL_FREEZE_PRIVATE.pkl')['selections']
    assert len(models)==80 and len(support)==5
    for r in models:assert r['groups']==([] if r['recipe']=='RAW_ONLY' else chosen['folds'][str(r['fold'])][r['recipe']])
    for r in support:assert r['groups']==[] and r['recipe']=='RAW_SUPPORT30'
    seal=jr(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');assert len(seal['files'])==105
    for key,value in seal['files'].items():assert sha(ROOT/'outer_scores_private'/(key+'.pkl'))==value
    assert before==sha(ROOT/'SELECTED_FEATURES_PRIVATE.json')
    write(PUB/'INDEPENDENT_RESULTS_VERIFICATION.json',dict(status='PASS',scope_patient_and_paired_rows=counts,
        all9_metrics_independently_recomputed=True,all_aggregate_tables_replayed=True,all10000_draw_intervals_replayed=True,
        all9_cardinalities_replayed=True,all85_model_feature_sets_still_frozen=True,all105_score_files_intact=True,
        no_training_prediction_or_threshold_optimization=True,source_sha256=sha(HERE/'verify_results.py')))
    print('INDEPENDENT_FINAL_RESULTS_ALL_TABLES_AND_BOOTSTRAPS_PASS',flush=True)


if __name__=='__main__':main()
