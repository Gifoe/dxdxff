"""Sealed-score CPU reporting, patient-cluster uncertainty and cardinality tables."""
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score,f1_score
from common import *
from probes import metric_rows


def frames(stage):
    pool=read(ROOT/'POOL_PRIVATE.pkl');meta=pool['meta'];allframes=[]
    if stage=='development':
        selections=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl')['selections']+read(ROOT/'SUPPORT_MODEL_FREEZE_PRIVATE.pkl')['selections']
        for row in selections:
            d=torch.load(ROOT/f"fold{row['fold']}/BANK_PRIVATE.pt",map_location='cpu',weights_only=False);q=read(row['validation']);va=d['val']
            assert sha(row['checkpoint'])==q['checkpoint_sha256']
            pn=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();ix=q['archive_indices']
            assert np.array_equal(ix,d['archive_indices'][va])
            allframes.append(pd.DataFrame(dict(model=row['family']+'__'+row['recipe'],version='seed42',fold=row['fold'],patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=(d['y'][va]==0).astype(int),probability=1-pn.astype(float),prediction=pn<row['threshold'])))
    else:
        seal=jr(ROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');assert seal['all105_score_cells_sealed_before_full_labels'] and len(seal['files'])==105
        assert seal['freeze_sha256']==sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
        archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
        rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');assert sha(archive)==rec['binding']['archive']
        with np.load(archive,allow_pickle=True) as z:
            y=1-z['y'];assert np.array_equal(z['patient'],meta['patient']) and np.array_equal(z['channel'],meta['channel'])
        assert len(y)==7635 and int(y.sum())==1743
        for key,h in sorted(seal['files'].items()):
            path=ROOT/'outer_scores_private'/(key+'.pkl');assert sha(path)==h;q=read(path);assert q['freeze_sha256']==seal['freeze_sha256']
            name,fold=key.rsplit('_',1);ix=q['indices'];pn=q['score_nez']
            allframes.append(pd.DataFrame(dict(model=name,version='seed42',fold=int(fold),patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],probability=1-pn.astype(float),prediction=pn<q['threshold'])))
    return pd.concat(allframes,ignore_index=True)


def verify_rows(frame,engine):
    result=pd.DataFrame(engine.patient_rows(frame))
    for (name,fold,pid),g in frame.groupby(['model','fold','patient']):
        r=result[(result.model==name)&(result.fold==fold)&(result.patient==pid)].iloc[0]
        y=g.y.to_numpy();p=g.probability.to_numpy();v=g.prediction.to_numpy();ch=g.channel.to_numpy()
        expected=dict(accuracy=accuracy_score(y,v),macro_f1=f1_score(y,v,labels=[0,1],average='macro',zero_division=0))
        if y.sum():expected.update(f1=f1_score(y,v,zero_division=0),auprc=average_precision_score(y,p))
        if (1-y).sum():expected['nez_f1']=f1_score(1-y,1-v.astype(int),zero_division=0)
        if len(set(y))==2:expected['auroc']=roc_auc_score(y,p)
        rank=y[np.lexsort((ch.astype(str),-p))]
        if y.sum():expected.update(mrr=1/(np.flatnonzero(rank)[0]+1),hit_at_1=float(rank[0]))
        for k,val in expected.items():assert abs(float(r[k])-val)<1e-12,(name,k)
    return result


def distribution(a,ids):
    ids=np.asarray(ids);unique=sorted(set(ids));rng=np.random.default_rng(42);draw=rng.integers(0,len(unique),(10000,len(unique)))
    count=np.stack([(draw==j).sum(1) for j in range(len(unique))],1)
    num=np.stack([np.nansum(a[ids==p],axis=0) for p in unique]);den=np.stack([np.isfinite(a[ids==p]).sum(0) for p in unique])
    n=count@den;dist=np.divide(count@num,n,out=np.full(n.shape,np.nan),where=n>0)
    return np.nanmean(a,axis=0),np.nanpercentile(dist,[2.5,97.5],axis=0),dist


def pairs(names):
    result=[]
    for family in ['D1','M8','Concat']:
        for arm in ['SELECTED','BEST_SWAP','RANDOM_A','RANDOM_B']:
            result.append((family+'__'+arm,family+'__OLD7'))
        for arm in ['BEST_SWAP','RANDOM_A','RANDOM_B']:
            result.append((family+'__SELECTED',family+'__'+arm))
    for family in ['M8','Concat']:
        for arm in ['SELECTED','OLD7']:
            for raw in ['RAW_ONLY','RAW_SUPPORT30']:result.append((family+'__'+arm,'Concat__'+raw))
    result += [('Concat__RAW_SUPPORT30','Concat__RAW_ONLY'),('M8__SELECTED','Concat__SELECTED'),('M8__SELECTED','D1__SELECTED'),('M8__OLD7','D1__OLD7')]
    result += [('M8__SELECTED',n) for n in ['HIST_M8','HIST_D1','HIST_RAW','HIST_SINGLE_HEAD_S4']]
    return sorted(set((a,b) for a,b in result if a in names and b in names))


def tables(patient,scope):
    out=PUB/scope;out.mkdir(exist_ok=True)
    means=[];folds=[];sources=[];sd=[];coverage=[];intervals=[];absolute=[];confusion=[]
    for name,g in patient.groupby('model'):
        means.append(dict(model=name,unit='patient_macro',appearances=len(g),unique_patients=g.patient.nunique(),channels=int(g.n.sum()),**{m:float(np.nanmean(g[m])) for m in METRICS}))
        for col,target in [('fold',folds),('center',sources)]:
            for key,h in g.groupby(col):target.append(dict(model=name,**{col:key},patient_appearances=len(h),unique_patients=h.patient.nunique(),**{m:float(np.nanmean(h[m])) for m in METRICS}))
        for m in METRICS:
            a=np.array([np.nanmean(h[m]) for _,h in g.groupby('fold')]);sd.append(dict(model=name,metric=m,fold_equal_mean=float(np.nanmean(a)),fold_sample_sd=float(np.nanstd(a,ddof=1))))
            coverage.append(dict(model=name,metric=m,finite_appearances=int(np.isfinite(g[m]).sum()),total_appearances=len(g)))
        for col in ['all','fold','center']:
            groups=[('all',g)] if col=='all' else g.groupby(col)
            for key,h in groups:confusion.append(dict(model=name,scope=col,group=key,patients=len(h),**{v:int(h[v].sum()) for v in ['TP','TN','FP','FN','n','positives','negatives']}))
    order=patient[patient.model==sorted(set(patient.model))[0]].sort_values(['fold','patient']);ids=order.patient.to_numpy();keys=list(zip(order.fold,order.patient));mat={}
    for name,g in patient.groupby('model'):
        g=g.sort_values(['fold','patient']);assert list(zip(g.fold,g.patient))==keys
        mat[name]=g[METRICS].to_numpy(dtype=float);mu,ci,dist=distribution(mat[name],ids)
        for j,m in enumerate(METRICS):absolute.append(dict(model=name,metric=m,mean=mu[j],ci_low=ci[0,j],ci_high=ci[1,j],cluster_patients=len(set(ids)),draws=10000))
    for a,b in pairs(mat):
        d=mat[a]-mat[b];mu,ci,dist=distribution(d,ids)
        for j,m in enumerate(METRICS):
            row=dict(model=a,reference=b,metric=m,delta=mu[j],ci_low=ci[0,j],ci_high=ci[1,j],valid_appearances=int(np.isfinite(d[:,j]).sum()),cluster_patients=len(set(ids)),draws=10000,threshold_reoptimized=False)
            if a=='M8__SELECTED' and b=='M8__OLD7' and m!='ez_fraction_bias':
                lo,hi=np.nanpercentile(dist[:,j],[100*.05/16,100*(1-.05/16)]);row.update(selected8endpoint_bonferroni_ci_low=lo,selected8endpoint_bonferroni_ci_high=hi)
            intervals.append(row)
    for filename,data in [('MAIN_RESULTS',means),('FOLD_RESULTS',folds),('SOURCE_RESULTS',sources),('FOLD_STABILITY',sd),('METRIC_COVERAGE',coverage),('PAIRED_COMPARISONS',intervals),('ABSOLUTE_INTERVALS',absolute),('CONFUSION_COUNTS',confusion)]:
        pd.DataFrame(data).to_csv(out/(filename+'.csv'),index=False)
    return pd.DataFrame(means),pd.DataFrame(intervals)


def cardinality():
    frame=read(ROOT/'ASSESSMENT_PATIENTS_PRIVATE.pkl');rows=[]
    for k,g in frame.groupby('k'):
        a=g[METRICS].to_numpy(dtype=float);mu,ci,dist=distribution(a,g.patient.to_numpy())
        for j,m in enumerate(METRICS):rows.append(dict(k=int(k),metric=m,mean=mu[j],ci_low=ci[0,j],ci_high=ci[1,j],patient_appearances=len(g),unique_patients=g.patient.nunique(),draws=10000,scope='nested_FIT_assessment_not_Outer',threshold_reoptimized=False))
    pd.DataFrame(rows).to_csv(PUB/'CARDINALITY_CURVE.csv',index=False)
    selected=jr(ROOT/'SELECTED_FEATURES_PRIVATE.json');frequency=[]
    for g,name in enumerate(NAMES):
        count=sum(g in r['SELECTED'] for r in selected['folds'].values());frequency.append(dict(feature=name,selected_folds=count,total_folds=5,frequency=count/5,consensus_not_applied_to_Outer=True))
    pd.DataFrame(frequency).to_csv(PUB/'FINAL_FEATURE_FREQUENCY.csv',index=False)
    stability=pd.read_csv(PUB/'SELECTION_STABILITY.csv');summary=[]
    for k,g in stability.groupby('k'):
        summary.append(dict(k=int(k),overlapping_path_pairs=len(g),jaccard_mean=float(g.jaccard.mean()),jaccard_q25=float(g.jaccard.quantile(.25)),jaccard_q75=float(g.jaccard.quantile(.75)),chance_adjusted_overlap_mean=float(g.chance_adjusted_overlap.mean()),not_independent_pair_replicates=True))
    pd.DataFrame(summary).to_csv(PUB/'STABILITY_SUMMARY.csv',index=False)


def main():
    assert jr(PUB/'OUTER_STATUS.json')['status']=='ALL105_FROZEN_SCORE_CELLS_SEALED'
    engine=ev();counts={}
    for stage in ['development','outer']:
        frame=frames(stage);patient=verify_rows(frame,engine);expected=1105 if stage=='development' else 1680
        assert len(patient)==expected
        save(ROOT/(stage.upper()+'_PATIENT_METRICS_PRIVATE.pkl'),patient);means,paired=tables(patient,stage);counts[stage]=len(patient)
        if stage=='outer':
            original=pd.read_csv(Path(r'D:\m8_single_head_concat_seed42_v1_runtime\public\MAIN_RESULTS.csv'))
            for new,old in [('HIST_M8','REF_M8'),('HIST_D1','REF_D1'),('HIST_RAW','REF_RAW_ONLY'),('HIST_SINGLE_HEAD_S4','S4')]:
                a=means[means.model==new].iloc[0];b=original[original.model==old].iloc[0]
                assert np.allclose(a[METRICS].to_numpy(dtype=float),b[METRICS].to_numpy(dtype=float),rtol=0,atol=1e-12)
            primary=paired[(paired.model=='M8__SELECTED')&(paired.reference=='M8__OLD7')].to_dict('records')
            write(PUB/'FINAL_FACTS.json',dict(status='COMPLETE_PHYSIOLOGICAL_SELECTION_STUDY_EXPLORATORY',primary_procedure='M8__SELECTED',global_best_subset_claim=False,
                independent_validation_claim=False,per_fold_selected_cardinality={f:r['k'] for f,r in jr(ROOT/'SELECTED_FEATURES_PRIVATE.json')['folds'].items()},
                selected_vs_matched_OLD7=primary,model_choice_changed_after_Outer=False,feature_selection_uses_Outer=False))
    cardinality()
    write(PUB/'REPORT_VERIFICATION.json',dict(status='PASS',patient_metric_rows=counts,independent_sklearn_and_direct_ranking=True,all4_history_references_reproduced=True,
        bootstrap_draws=10000,source_sha256=sha(HERE/'report.py'),metric_source_sha256=sha(Path(engine.__file__)),no_refit_or_threshold_reoptimization=True))
    print('PHYSIOLOGY_SELECTION_REPORT_COMPLETE',counts,flush=True)


if __name__=='__main__':main()
