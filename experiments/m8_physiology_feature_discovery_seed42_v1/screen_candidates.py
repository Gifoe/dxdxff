"""390 fixed linear probes, exclusively inside each original outer FIT cohort."""
from audit_old7 import *

stage=jr(TASK/'PROBE_STAGE_C_LOCK.json')
lock,core,rec,banks,oldbinding=setup()
assert jr(OUT/'public/CANDIDATE_EXTRACTION_STATUS.json')['status']==stage['required_extraction_status']
fp=OUT/'CANDIDATE_FEATURES_PRIVATE.pkl';assert sha(fp)==stage['feature_cache_sha256']
cache=read(fp);assert np.array_equal(cache['original14'],rec['R14']) and cache['candidate22'].shape==(7635,22)
xall=np.concatenate([rec['R14'],cache['candidate22']],axis=1)
assert np.isfinite(xall).all()
binding=dict(source={n:sha(TASK/n) for n in ['screen_candidates.py','PROBE_STAGE_C_LOCK.json','AUDIT_PROTOCOL.json','audit_old7.py']},
    oldbinding=oldbinding,feature_cache_sha256=sha(fp))
freeze=OUT/'CANDIDATE_SCREEN_FREEZE_PRIVATE.json'
if freeze.exists():assert jr(freeze)==binding
else:write(freeze,binding)
oldcols=list(range(14))
variants=[('FULL7',oldcols)]+[(f'ONLY_NEW_{k}',[14+2*k,15+2*k]) for k in range(11)]+[(f'ADD_NEW_{k}',oldcols+[14+2*k,15+2*k]) for k in range(11)]
variants += [('ALL18',list(range(36))),('ADD_SHAPE5',oldcols+[14+2*k+j for k in [1,2,3,9,10] for j in [0,1]]),
    ('ADD_SPECTRAL5',oldcols+[14+2*k+j for k in [4,5,6,7,8] for j in [0,1]])]
assert len(variants)==26
metrics=['auroc','auprc','macro_f1','f1','accuracy','mrr','hit_at_1'];rows=[];patient_rows=[]
for fold,d in enumerate(banks,1):
    fit=d['train'];seed=42+1009*fold;pid=d['patient'];y=(d['y']==0).astype(int)
    x=core.patient_z(xall[d['archive_indices']],pid)
    for inner,(it,iv) in enumerate(GroupKFold(n_splits=3,shuffle=True,random_state=seed).split(x[fit],y[fit],groups=pid[fit])):
        remain=fit[it];held=fit[iv];ids=np.asarray(sorted(set(pid[remain])))
        order=np.random.default_rng(seed+43*(inner+1)).permutation(ids);ncal=max(1,int(np.ceil(.2*len(order))))
        cal=remain[np.isin(pid[remain],order[:ncal])];tr=remain[~np.isin(pid[remain],order[:ncal])]
        groups=[set(pid[q]) for q in [tr,cal,held]]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
        assert set.union(*groups)==set(pid[fit]) and set.union(*groups).isdisjoint(d['roles']['validation']|d['roles']['test'])
        for name,columns in variants:
            path=OUT/f'candidate_probes_private/fold{fold}_screen{inner}_{name}.pkl'
            cb=dict(binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(),fold=fold,inner=inner,variant=name,columns=columns)
            if path.exists():q=read(path);assert q['binding']==cb
            else:
                xx=x[:,columns];imp=SimpleImputer().fit(xx[tr]);z=imp.transform(xx);scaler=StandardScaler().fit(z[tr]);z=scaler.transform(z)
                model=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-7,random_state=seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model.fit(z[tr],y[tr],sample_weight=weights(y[tr],pid[tr]))
                pc=model.predict_proba(z[cal])[:,1];sel=EV.threshold(y[cal],pc,pid[cal]);p=model.predict_proba(z[held])[:,1]
                frame=pd.DataFrame(dict(model=name,version='FIT_screen',fold=fold,patient=pid[held],channel=d['channel'][held],center=d['center'][held],
                    y=y[held],probability=p,prediction=p>=sel['threshold']))
                q=dict(binding=cb,patients=EV.patient_rows(frame),frame=frame,threshold_selection=sel,model=model,imputer=imp,scaler=scaler,
                    train_indices=tr,calibration_indices=cal,screen_indices=held,calibration_probability=pc,
                    warnings=sorted(set(type(w.message).__name__ for w in caught)),iterations=int(model.n_iter_.max()))
                save(path,q)
            g=pd.DataFrame(q['patients'])
            # Independent recomputation from retained score rows and validation probabilities.
            independent=pd.DataFrame(EV.patient_rows(q['frame']))
            assert np.allclose(independent[metrics],g[metrics],atol=0,rtol=0,equal_nan=True)
            assert EV.threshold(y[q['calibration_indices']],q['calibration_probability'],pid[q['calibration_indices']])==q['threshold_selection']
            patient_rows += [dict(r,outer_fold=fold,inner_screen=inner+1,variant=name) for r in q['patients']]
            rows.append(dict(outer_fold=fold,inner_screen=inner+1,variant=name,patients=len(g),**{m:EV.nanmean(g[m]) for m in metrics},
                iterations=q['iterations'],warnings=';'.join(q['warnings'])))
            print('CANDIDATE_FIT_PROBE_COMPLETE',fold,inner+1,name,flush=True)
assert len(rows)==390
df=pd.DataFrame(patient_rows);save(OUT/'CANDIDATE_PROBE_PATIENTS_PRIVATE.pkl',df)
summary=[];deltas=[];eligibility=[]
for (fold,name),g in df.groupby(['outer_fold','variant']):
    assert len(g)==g.patient.nunique()
    summary.append(dict(outer_fold=fold,variant=name,patients=len(g),**{m:EV.nanmean(g[m]) for m in metrics}))
    if name=='FULL7':continue
    full=df[(df.outer_fold==fold)&(df.variant=='FULL7')].set_index('patient').loc[g.patient]
    delta={m:EV.nanmean(g[m].to_numpy()-full[m].to_numpy()) for m in metrics}
    deltas.append(dict(outer_fold=fold,variant=name,**{f'delta_{m}':delta[m] for m in metrics}))
    if name.startswith('ADD_NEW_'):
        support=0
        for screen,h in g.groupby('inner_screen'):
            ref=df[(df.outer_fold==fold)&(df.variant=='FULL7')&(df.inner_screen==screen)]
            a=EV.nanmean(h.auroc)-EV.nanmean(ref.auroc);b=EV.nanmean(h.auprc)-EV.nanmean(ref.auprc);c=EV.nanmean(h.macro_f1)-EV.nanmean(ref.macro_f1)
            support+=int((a>=.005 or b>=.005) and min(a,b)>=-.002 and c>=0)
        eligible=(delta['auroc']>=.005 or delta['auprc']>=.005) and min(delta['auroc'],delta['auprc'])>=-.002 and delta['macro_f1']>=0 and support>=2
        k=int(name.rsplit('_',1)[1]);eligibility.append(dict(outer_fold=fold,candidate=cache['names'][k],
            provisional_linear_eligible=bool(eligible),supporting_screens=support,delta_auroc=delta['auroc'],delta_auprc=delta['auprc'],delta_macro_f1=delta['macro_f1'],
            nonlinear_and_center_checks_pending=True,final_feature_decision=False))
pd.DataFrame(rows).to_csv(OUT/'public/CANDIDATE_INNER_PROBE_METRICS.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'public/CANDIDATE_FIT_CROSSFIT_SUMMARY.csv',index=False)
pd.DataFrame(deltas).to_csv(OUT/'public/CANDIDATE_PROBE_DIFFERENCES.csv',index=False)
pd.DataFrame(eligibility).to_csv(OUT/'public/CANDIDATE_PROVISIONAL_ELIGIBILITY.csv',index=False)
centers=[dict(outer_fold=f,variant=n,center=center,patients=len(g),**{m:EV.nanmean(g[m]) for m in metrics})
    for (f,n,center),g in df.groupby(['outer_fold','variant','center'])]
pd.DataFrame(centers).to_csv(OUT/'public/CANDIDATE_SCREEN_CENTER_METRICS.csv',index=False)
write(OUT/'public/CANDIDATE_SCREEN_STATUS.json',dict(status='ALL390_FIT_ONLY_LINEAR_CANDIDATE_PROBES_COMPLETE',source=binding['source'],
    original_validation_and_Outer_excluded=True,all_scores_and_thresholds_retained_and_recomputed=True,
    nonlinear_and_center_heldout_confirmation_pending=True,M8_integration_and_final_validation_pending=True,full_goal_complete=False))
print('ALL390_FIT_ONLY_LINEAR_CANDIDATE_PROBES_COMPLETE',flush=True)
