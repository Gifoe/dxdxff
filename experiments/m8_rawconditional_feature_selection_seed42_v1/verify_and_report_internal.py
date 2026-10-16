"""Full canonical head/selection replay, own-FIT statistics; never read Outer y."""
import itertools
from sklearn.linear_model import Ridge
from scipy.special import expit
from conditional_io import *

DEST=Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_dense_reference_runtime')


def independently_prepare(x,train):
    a=np.asarray(x,float);counts=np.isfinite(a[train]).sum(0)
    mu=np.divide(np.nansum(a[train],axis=0),counts,out=np.zeros(a.shape[1]),where=counts>0)
    fill=np.where(np.isfinite(a),a,mu);mean=fill[train].mean(0);sd=np.std(fill[train],axis=0,ddof=0);sd[sd==0]=1
    return (fill-mean)/sd,dict(imputer=mu,mean=mean,scale=sd)


def bootstrap(a,ids):
    ids=np.asarray(ids);unique=np.unique(ids);draw=np.random.default_rng(42).integers(len(unique),size=(10000,len(unique)))
    weights=np.array([np.bincount(row,minlength=len(unique)) for row in draw],float)
    sums=np.array([np.nansum(a[ids==who],axis=0) for who in unique]);n=np.array([np.isfinite(a[ids==who]).sum(0) for who in unique])
    den=weights@n;dist=np.divide(weights@sums,den,out=np.full(den.shape,np.nan),where=den>0)
    return np.nanmean(a,0),np.nanpercentile(dist,[2.5,97.5],axis=0)


def main():
    common,data,training,support,p=old_modules();_,core,_,_,_,_,banks,_=common.legal();engine=common.ev()
    met=common.METRICS;pub=DEST/'public';b=jr(DEST/'INPUT_FREEZE_PRIVATE.json')
    assert jr(pub/'SCREEN_STATUS.json')['status']=='COMPLETE_INTERNAL_RAW_CONDITIONAL_SCREEN'
    assert jr(pub/'ALL20_ACTUAL_RIDGE_ADMISSION.json')['status']=='PASS'
    assert jr(pub/'CANONICAL_ADMISSION_CONTEXT.json')['status']=='PASS'
    for name,h in b['numerical_engine'].items():assert sha(HERE/name)==h
    assert bindings()==b['original_scientific_binding']
    registry=read(DEST/'ENCODER_REGISTRY_PRIVATE.pkl');assert registry['binding']==b
    pool=read(OLDROOT/'POOL_PRIVATE.pkl');names=pool['names'];records=pd.read_csv(pub/'SEARCH_RECORD.csv')
    pts=read(DEST/'ASSESSMENT_PATIENT_METRICS_PRIVATE.pkl');final=jr(DEST/'SELECTED_FEATURES_PRIVATE.json')['folds']
    boots=pd.read_csv(pub/'WEIGHT_BOOTSTRAP_RESELECTION.csv');replayed=assessed=0;maximum=0.
    for fold,d in enumerate(banks,1):
        fit=d['train'];pid=d['patient'][fit];y=(d['y'][fit]==0).astype(int);ch=d['channel'][fit];center=d['center'][fit]
        for screen in range(4):
            q=read(DEST/f'fold{fold}_SCREEN_PRIVATE.pkl')['paths'][screen]
            ent=q['raw_entry'];assert sha(Path(ent['path']))==ent['sha256'];raw=read(Path(ent['path']))
            assert raw['binding']['input']==b['original_scientific_binding']
            assert raw['binding']['role_sha256']==b['original_scientific_binding']['original_role_files'][f'{fold}_{screen}']
            assert np.array_equal(raw['archive_indices'],d['archive_indices'][fit])
            rr=q['roles'];sets=[set(pid[rr[k]]) for k in rr]
            assert all(sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i))
            assert set.union(*sets)==d['roles']['fit'] and set.union(*sets).isdisjoint(d['roles']['validation']|d['roles']['test'])
            tr,cal,choice=rr['train'],rr['cal'],rr['choice'];x=np.c_[raw['raw'],core.patient_z(pool['summary'][d['archive_indices'][fit]],pid)]
            xx,pre=independently_prepare(x,tr)
            for key,value in pre.items():np.testing.assert_array_equal(value,q['pre'][key])
            base=p.weights(y[tr],pid[tr]);options={};own=records[(records.fold==fold)&(records.screen==screen)]
            perturb={};train_ids=np.unique(pid[tr]);rng=np.random.default_rng(42+1009*fold+830003)
            for repeat in range(10):
                sample=rng.choice(train_ids,size=len(train_ids),replace=True)
                perturb[repeat]=np.array([(sample==who).sum() for who in pid[tr]])
            for row in own.itertuples(index=False):
                groups=[] if pd.isna(row.groups) else [int(g) for g in str(row.groups).split(',')]
                cols=np.r_[np.arange(192),192+np.array([2*g+j for g in groups for j in [0,1]],int)]
                weight=base.copy()
                if row.kind.startswith('weight_bootstrap_'):weight*=perturb[int(row.kind.rsplit('_',1)[1])]
                weight=weight*len(weight)/weight.sum();m=Ridge(alpha=1.,fit_intercept=True,solver='cholesky')
                m.fit(xx[tr][:,cols],2*y[tr]-1,sample_weight=weight);score=expit(m.predict(xx[:,cols]))
                theta=p.threshold(y[cal],score[cal],pid[cal]);rows=p.metric_rows(y[choice],score[choice],pid[choice],ch[choice],center[choice],theta)
                utility=float(np.mean([np.mean([r[k] for r in rows]) for k in ['auroc','auprc','macro_f1','f1']]))
                error=abs(utility-row.choice_utility);assert error<1e-12;maximum=max(maximum,error);replayed+=1
                for key in met:assert abs(np.mean([r[key] for r in rows])-getattr(row,key))<1e-12
                options.setdefault((row.kind,int(row.k)),[]).append((utility,tuple(-g for g in groups),groups))
            for step in q['path']:
                best=max(options['greedy',step['k']],key=lambda r:(r[0],r[1]));assert best[2]==step['groups'] and abs(best[0]-step['utility'])<1e-12
            if screen:
                refit=np.r_[tr,choice];zz,_=independently_prepare(x,refit);w=p.weights(y[refit],pid[refit]);w=w*len(w)/w.sum();held=rr['held']
                for recipe,step in [('path',r) for r in q['path']]+[('old7',dict(k=7,groups=list(range(7))))]:
                    cols=np.r_[np.arange(192),192+np.array([2*g+j for g in step['groups'] for j in [0,1]],int)]
                    m=Ridge(alpha=1.,fit_intercept=True,solver='cholesky');m.fit(zz[refit][:,cols],2*y[refit]-1,sample_weight=w)
                    score=expit(m.predict(zz[:,cols]));theta=p.threshold(y[cal],score[cal],pid[cal])
                    frame=pd.DataFrame(dict(model=recipe,version='internal',fold=fold,patient=pid[held],channel=ch[held],center=center[held],y=y[held],probability=score[held],prediction=score[held]>=theta))
                    actual=pd.DataFrame(engine.patient_rows(frame)).sort_values('patient')
                    saved=pts[(pts.fold==fold)&(pts.screen==screen)&(pts.recipe==recipe)&(pts.k==step['k'])].sort_values('patient')
                    assert list(actual.patient)==list(saved.patient) and np.allclose(actual[met],saved[met],rtol=0,atol=1e-12)
                    assessed+=1
            else:
                subset=final[str(fold)]['SELECTED'];k=final[str(fold)]['k'];assert subset==q['path'][k]['groups']
                if k:
                    best=max(options['single_swap',k],key=lambda r:(r[0],r[1]));assert best[2]==final[str(fold)]['BEST_SWAP']
                for repeat in range(10):
                    expected=[] if not k else max(options[f'weight_bootstrap_{repeat}',k],key=lambda r:(r[0],r[1]))[2]
                    r=boots[(boots.fold==fold)&(boots.replicate==repeat)].iloc[0]
                    observed=[] if pd.isna(r.groups) else [int(g) for g in str(r.groups).split(',')];assert expected==observed
        own=pts[(pts.fold==fold)&(pts.recipe=='path')];scores=[]
        for k in range(10):
            g=own[own.k==k];u=g[['auroc','auprc','macro_f1','f1']].mean(axis=1)
            scores.append(float(u.mean()-.1*np.std([u[g.center==c].mean() for c in sorted(set(g.center))],ddof=1)))
        chosen=min(k for k,v in enumerate(scores) if v>=max(scores)-.005);assert chosen==final[str(fold)]['k']
        print('CANONICAL_ALL_HEADS_AND_ASSESSMENTS_REPLAYED',fold,replayed,assessed,flush=True)
    assert replayed==len(records) and assessed==165 and replayed+assessed==jr(pub/'SCREEN_STATUS.json')['actual_ridge_fits']
    write(pub/'INTERNAL_REPLAY_VERIFICATION.json',dict(status='PASS',search_head_evaluations_replayed=replayed,
        assessment_head_fits_replayed=assessed,assessment_patient_rows=len(pts),maximum_choice_utility_error=maximum,
        canonical_API_replay_not_independent_numerical_replication=True,independent_assessment_patient_metrics=True,
        all_role_preprocessing_cardinality_swap_bootstrap_checks_pass=True,source_sha256=sha(Path(__file__))))
    # Fixed-score descriptive inference, selection-conditioned, never new Outer.
    views=[]
    for fold in range(1,6):
        own=pts[pts.fold==fold];k=final[str(fold)]['k']
        for model,recipe,kk in [('SELECTED','path',k),('RAW_ONLY','path',0),('OLD7','old7',7)]:
            views.append(own[(own.recipe==recipe)&(own.k==kk)].assign(model=model))
    frame=pd.concat(views,ignore_index=True);main=[];interval=[];source=[];folds=[]
    ordered={name:g.sort_values(['fold','patient']) for name,g in frame.groupby('model')}
    for name,g in ordered.items():
        main.append(dict(model=name,patient_appearances=len(g),unique_patients=g.patient.nunique(),selection_conditioned=True,**{m:float(g[m].mean()) for m in met}))
        for col,target in [('center',source),('fold',folds)]:
            for key,h in g.groupby(col):target.append(dict(model=name,**{col:key},patient_appearances=len(h),unique_patients=h.patient.nunique(),**{m:float(h[m].mean()) for m in met}))
    a=ordered['SELECTED'];ids=a.patient.to_numpy()
    for name in ['RAW_ONLY','OLD7']:
        ref=ordered[name];assert list(zip(a.fold,a.patient))==list(zip(ref.fold,ref.patient))
        delta=a[met].to_numpy(float)-ref[met].to_numpy(float);mu,ci=bootstrap(delta,ids)
        for j,m in enumerate(met):interval.append(dict(model='SELECTED',reference=name,metric=m,delta=mu[j],ci_low=ci[0,j],ci_high=ci[1,j],draws=10000,selection_conditioned=True,independent_confirmation=False))
    for name,rows in [('INTERNAL_MAIN_RESULTS',main),('INTERNAL_PAIRED_INTERVALS',interval),('INTERNAL_SOURCE_RESULTS',source),('INTERNAL_FOLD_RESULTS',folds)]:pd.DataFrame(rows).to_csv(pub/(name+'.csv'),index=False)
    stability=[]
    for fold,g in boots.groupby('fold'):
        sets=[set() if pd.isna(v) else set(str(v).split(',')) for v in g.groups]
        overlap=[len(a&b)/len(a|b) if a|b else 1. for a,b in itertools.combinations(sets,2)]
        for j,n in enumerate(names):stability.append(dict(fold=fold,feature=n,k=final[str(fold)]['k'],selected_replicates=sum(str(j) in s for s in sets),replicates=10,mean_pairwise_jaccard=float(np.mean(overlap)),encoder_and_preprocessing_fixed=True))
    pd.DataFrame(stability).to_csv(pub/'CONDITIONAL_SELECTION_STABILITY.csv',index=False)
    assert sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==b['original_scientific_binding']['parent_Outer_seal_protection']
    write(pub/'INTERNAL_REPORT_STATUS.json',dict(status='COMPLETE_INTERNAL_ONLY_CONDITIONAL_SCREEN_VERIFIED',
        original_validation_or_Outer_used=False,new_neural_candidate_training=False,goal_completion_or_final_stable_minimum_claim=False,
        all20_Raw_encoders_reused=True,canonical_math_hyperparameters_and_input_unchanged=True,
        original_failed_fast_path_remains_FAIL=True,primary_intervals_selection_conditioned=True))


if __name__=='__main__':
    main()
