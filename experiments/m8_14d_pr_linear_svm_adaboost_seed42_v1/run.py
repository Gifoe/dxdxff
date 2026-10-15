"""Linear-kernel SVC and stump AdaBoost, exact DELTA14 with/without PR.

All private identities, feature arrays, fitted estimators and scores stay on server.
A=14D without PR, B=14D with original patient Z. No raw/GPU processing.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
PARENT = Path(r'D:\m8_feature_pr_baselines_seed42_v1')
OLD = Path(r'D:\m8_feature_pr_baselines_seed42_v1_runtime')
sys.path.insert(0, str(PARENT))
from support import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score, log_loss

MD = load('linear_ada_parent_models', PARENT/'models.py')
MET = load('linear_ada_expanded_metrics', Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))
REPORT = load('linear_ada_parent_aggregate', PARENT/'aggregate.py')
REPORT.EV = MET
REPORT.METRICS = MET.CLASSIFICATION + MET.RANKING
KEYS = REPORT.METRICS


def initialize(a):
    protocol = jread(TASK_ROOT/'PROTOCOL_LOCK.json')
    oldlock, core, rec, banks, x, parent_binding = setup(a)
    assert sha(OLD/'FROZEN_BEFORE_OUTER_PRIVATE.json') == protocol['parent_outer_freeze_sha256']
    assert sha(OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json') == protocol['parent_outer_seal_sha256']
    assert jread(OLD/'public/OUTER_STATUS.json')['status'] == 'ONE_FROZEN_OUTER_PASS_COMPLETE'
    protected = {}
    for p in [OLD/'FROZEN_BEFORE_OUTER_PRIVATE.json', OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json',
              OLD/'MODEL_SELECTION_PRIVATE.pkl', OLD/'PREDICTIONS_PRIVATE.pkl',
              OLD/'PATIENT_METRICS_PRIVATE.pkl', PARENT/'aggregate.py',
              Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py')]:
        protected[str(p)] = sha(p)
    oldselections = read(OLD/'MODEL_SELECTION_PRIVATE.pkl')['selections']
    seal = jread(OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    for fold in range(1, 6):
        for version in ['A', 'B']:
            s = [r for r in oldselections if r['model']=='B2' and r['version']==version and r['fold']==fold]
            assert len(s)==1 and s[0]['config']['C']==protocol['linear_svm_C_by_fold'][str(fold)]
            p = Path(s[0]['path']); assert sha(p)==s[0]['model_sha256']; protected[str(p)]=sha(p)
            p = OLD/f'outer_scores_private/B2_{version}_fold{fold}.pkl'
            assert sha(p)==seal['files'][f'B2_{version}_{fold}']; protected[str(p)]=sha(p)
    binding = dict(source={p.name:sha(p) for p in TASK_ROOT.iterdir() if p.suffix in ['.py','.json']},
                   parent_binding=parent_binding, protected=protected)
    path=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if path.exists(): assert jread(path)==binding, 'Changed source/data/config: refuse resume'
    else: write(path,binding)
    audit=[]
    for fold,d in enumerate(banks,1):
        z=transform_view(core,x[d['archive_indices']],d['patient'],True)
        zz,_=MD.preprocessing(z,d['train'],True)
        bd=torch.load(a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert np.array_equal(zz.astype(np.float32),bd['x'])
        audit.append(dict(fold=fold,fit_patients=len(set(d['patient'][d['train']])),
            validation_patients=len(set(d['patient'][d['val']])),PR_exact_original=True,
            scaled_PR_exact_D1_bank=True,test_patients_excluded=True,dimension=14))
    write(a.runtime/'public/INPUT_AUDIT.json',dict(status='PASS',patients=80,channels=7635,seizures=256,
        folds=audit,source_sha256=digest(binding['source']),feature_sha256=parent_binding['feature_sha256'],
        versions=parent_binding['versions'],no_raw_or_gpu_or_new_extraction=True))
    print('INPUT_GATES_PASS_80_7635_256_BOTH_VIEWS',flush=True)
    return protocol,oldlock,core,rec,banks,x,binding


def fit(model,config,x,y,pid,tr,seed):
    z,pre=MD.preprocessing(x,tr,model=='Linear_SVM')
    cal=None; audit=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if model=='Linear_SVM':
            def estimator():
                return SVC(kernel='linear',C=config['C'],probability=False,cache_size=512,random_state=seed)
            oof=np.full(len(tr),np.nan)
            cv=GroupKFold(n_splits=3,shuffle=True,random_state=seed)
            for k,(it,iv) in enumerate(cv.split(x[tr],y[tr],groups=pid[tr])):
                ti,vi=tr[it],tr[iv]; assert set(pid[ti]).isdisjoint(set(pid[vi]))
                zx,_=MD.preprocessing(x,ti,True); m=estimator()
                m.fit(zx[ti],y[ti],sample_weight=MD.patient_weights(y[ti],pid[ti]))
                assert m.fit_status_==0
                oof[iv]=m.decision_function(zx[vi])
                audit.append(dict(split=k,fit_patients=len(set(pid[ti])),held_out_patients=len(set(pid[vi])),
                    patient_exclusion=True,preprocessing_fit_subtrain_only=True))
            assert np.isfinite(oof).all()
            cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=seed)
            cal.fit(oof[:,None],y[tr],sample_weight=MD.patient_weights(y[tr],pid[tr],False))
            m=estimator()
        else:
            m=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,random_state=seed),
                random_state=seed,**config)
        m.fit(z[tr],y[tr],sample_weight=MD.patient_weights(y[tr],pid[tr]))
    assert np.array_equal(m.classes_,[0,1])
    if model=='Linear_SVM': assert m.fit_status_==0
    return dict(model=m,pre=pre,calibration=cal,model_id=model,calibration_audit=audit,
        warning_categories=sorted(set(type(w.message).__name__ for w in caught)),
        iterations=int(np.asarray(m.n_iter_).max()) if model=='Linear_SVM' else len(m.estimators_))


def predict(bundle,x):
    z=MD.apply_pre(x,bundle['pre']);m=bundle['model'];decision=m.decision_function(z)
    p=(bundle['calibration'].predict_proba(decision[:,None]) if bundle['calibration'] is not None else m.predict_proba(z))[:,1]
    assert p.shape==(len(x),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p.astype(np.float64),np.asarray(decision,dtype=np.float64)


def train(a,protocol,core,banks,x,binding):
    fp=a.runtime/'MODEL_SELECTION_PRIVATE.pkl'
    if fp.exists():
        q=read(fp);assert q['binding']==binding
        for r in q['selections']:assert sha(r['path'])==r['model_sha256']
        print('ALL20_ALREADY_SELECTED_NO_REFIT',flush=True);return q['selections']
    selections=[];search=[];calibration=[]
    for fold,d in enumerate(banks,1):
        y=(d['y']==0).astype(int);pid=d['patient'];tr=d['train'];va=d['val'];seed=42+1009*fold
        views={v:transform_view(core,x[d['archive_indices']],pid,v=='B') for v in ['A','B']}
        for model in protocol['models']:
            configs=([dict(C=protocol['linear_svm_C_by_fold'][str(fold)])] if model=='Linear_SVM' else protocol['adaboost_grid'])
            candidates=[]
            for ci,config in enumerate(configs):
                pair=[]
                for version in ['A','B']:
                    cp=a.runtime/f'candidates_private/{model}/fold{fold}/c{ci}_{version}.pkl'
                    cb=dict(binding_sha256=digest(binding),fold=fold,model=model,version=version,config=config,seed=seed)
                    if cp.exists():q=read(cp);assert q['binding']==cb
                    else:
                        start=time.perf_counter();bundle=fit(model,config,views[version],y,pid,tr,seed)
                        p,decision=predict(bundle,views[version][va]);selected=EV.threshold(y[va],p,pid[va])
                        fr=pd.DataFrame(dict(model=model,version=version,fold=fold,patient=pid[va],channel=d['channel'][va],
                            center=d['center'][va],y=y[va],probability=p,prediction=p>=selected['threshold']))
                        rows=pd.DataFrame(MET.patient_rows(fr))
                        q=dict(binding=cb,bundle=bundle,validation_probability=p,selected=selected,
                            validation_means={m:MET.nanmean(rows[m]) for m in MET.CLASSIFICATION},seconds=time.perf_counter()-start)
                        save(cp,q)
                    pair.append(dict(path=str(cp),model_sha256=sha(cp),model=model,version=version,fold=fold,candidate=ci,
                        config=config,seed=seed,**q['selected'],validation_auroc=q['validation_means']['auroc'],
                        validation_auprc=q['validation_means']['auprc']))
                    search.append(dict(model=model,version=version,fold=fold,candidate=ci,config_json=json.dumps(config,sort_keys=True),
                        **q['selected'],validation_auroc=q['validation_means']['auroc'],validation_auprc=q['validation_means']['auprc'],
                        iterations=q['bundle']['iterations'],warning_categories=';'.join(q['bundle']['warning_categories']),seconds=q['seconds']))
                    for row in q['bundle']['calibration_audit']:calibration.append(dict(model=model,version=version,fold=fold,**row))
                    print('CANDIDATE_COMPLETE',model,version,fold,ci,'seconds',round(q['seconds'],3),flush=True)
                key=tuple(np.mean([r[k] for r in pair]) for k in ['validation_macro_f1','validation_auprc','validation_auroc'])+(-ci,)
                candidates.append((key,pair))
            selections+=max(candidates,key=lambda q:q[0])[1]
            write(a.runtime/'public/RUN_STATUS.json',dict(status='TRAINING',selected_cells=len(selections),total_cells=20,candidates_complete=len(search)))
    assert len(selections)==20 and len(search)==50 and len(calibration)==30
    save(fp,dict(binding=binding,selections=selections))
    public=[]
    for r in selections:
        s={k:v for k,v in r.items() if k not in ['path','config']};s['config_json']=json.dumps(r['config'],sort_keys=True);public.append(s)
    pd.DataFrame(public).to_csv(a.runtime/'public/MODEL_SELECTION.csv',index=False)
    pd.DataFrame(search).to_csv(a.runtime/'public/VALIDATION_SEARCH.csv',index=False)
    pd.DataFrame(calibration).to_csv(a.runtime/'public/SVM_CALIBRATION_AUDIT.csv',index=False)
    print('ALL20_MODELS_SELECTED',flush=True)
    return selections


def freeze(a,selections,binding):
    for r in selections:assert sha(r['path'])==r['model_sha256']
    obj=dict(binding_sha256=digest(binding),selections=selections,all20_selected_before_new_outer=True)
    fp=a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if fp.exists():assert jread(fp)==obj
    else:write(fp,obj)
    write(a.runtime/'public/OUTER_FREEZE_AUDIT.json',dict(status='PASS',models=20,freeze_sha256=sha(fp),
        matched_hyperparameters_across_PR=True,patient_disjoint_validation=True,validation_only_thresholds=True))
    return sha(fp)


def outer(a,oldlock,core,rec,x,selections,fh):
    if (a.runtime/'PREDICTIONS_PRIVATE.pkl').exists():
        assert jread(a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')['freeze_sha256']==fh
        return read(a.runtime/'PREDICTIONS_PRIVATE.pkl')
    meta=rec['meta'];cells=[];files={}
    for r in selections:
        fold,model,version=r['fold'],r['model'],r['version']
        ix=np.flatnonzero(np.isin(meta['patient'],list(roles(meta,fold)['test'])))
        fp=a.runtime/f'outer_scores_private/{model}_{version}_fold{fold}.pkl'
        if fp.exists():q=read(fp)
        else:
            z=transform_view(core,x[ix],meta['patient'][ix],version=='B')
            p,decision=predict(read(r['path'])['bundle'],z)
            q=dict(indices=ix,probability=p,decision_score=decision,prediction=p>=r['threshold'],
                model=model,version=version,fold=fold,threshold=r['threshold'],freeze_sha256=fh,model_sha256=r['model_sha256'])
            save(fp,q)
        assert q['freeze_sha256']==fh and q['model_sha256']==r['model_sha256'] and np.array_equal(q['indices'],ix) and 'y' not in q
        cells.append(q);files[f'{model}_{version}_{fold}']=sha(fp)
        print('OUTER_CELL_SEALED',model,version,fold,flush=True)
    obj=dict(freeze_sha256=fh,files=files,all20_sealed_before_new_full_labels=True)
    sp=a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'
    if sp.exists():assert jread(sp)==obj
    else:write(sp,obj)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==oldlock['archive_sha256']
    with np.load(archive,allow_pickle=True) as z:y=(z['y']==0).astype(int)
    assert len(y)==7635 and y.sum()==1743
    frames=[]
    for q in cells:
        ix=q['indices'];frames.append(pd.DataFrame(dict(model=q['model'],version=q['version'],fold=q['fold'],
            patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],global_index=ix,
            y=y[ix],probability=q['probability'],prediction=q['prediction'],threshold=q['threshold'])))
    # Verify and reuse old sealed RBF scores, never retrain or infer the old model.
    oldsel=read(OLD/'MODEL_SELECTION_PRIVATE.pkl')['selections'];oldseal=jread(OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    for r in oldsel:
        if r['model']!='B2':continue
        v,f=r['version'],r['fold'];fp=OLD/f'outer_scores_private/B2_{v}_fold{f}.pkl'
        assert sha(fp)==oldseal['files'][f'B2_{v}_{f}'];q=read(fp);ix=q['indices']
        assert q['freeze_sha256']==oldseal['freeze_sha256'] and q['model_sha256']==r['model_sha256']
        assert np.array_equal(ix,np.flatnonzero(np.isin(meta['patient'],list(roles(meta,f)['test']))))
        frames.append(pd.DataFrame(dict(model='RBF_SVM',version=v,fold=f,patient=meta['patient'][ix],
            channel=meta['channel'][ix],center=meta['center'][ix],global_index=ix,y=y[ix],
            probability=q['probability'],prediction=q['prediction'],threshold=q['threshold'])))
    frame=pd.concat(frames,ignore_index=True)
    for _,g in frame.groupby(['model','version']):assert len(g)==g.global_index.nunique()==7635 and g.patient.nunique()==80
    save(a.runtime/'PREDICTIONS_PRIVATE.pkl',frame)
    return frame


def verify_and_report(a,frame,banks,selections,binding,fh):
    patients=pd.DataFrame(MET.patient_rows(frame));assert len(patients)==480
    save(a.runtime/'PATIENT_METRICS_PRIVATE.pkl',patients)
    REPORT.aggregate(a,frame,patients)
    # Independent sklearn recomputation of confusion/F1/AUC/AP/logloss for every patient.
    for _,g in frame.groupby(['model','version','fold','patient']):
        row=patients[(patients.model==g.model.iloc[0])&(patients.version==g.version.iloc[0])&(patients.patient==g.patient.iloc[0])].iloc[0]
        tn,fp,fn,tp=confusion_matrix(g.y,g.prediction,labels=[0,1]).ravel()
        assert np.array_equal([tn,fp,fn,tp],[row.TN,row.FP,row.FN,row.TP])
        assert abs(row.nez_f1-f1_score(g.y,g.prediction,pos_label=0))<1e-12
        if g.y.nunique()==2:
            assert abs(row.macro_f1-f1_score(g.y,g.prediction,average='macro'))<1e-12
            assert abs(row.auroc-roc_auc_score(g.y,g.probability))<1e-12
        if g.y.sum():assert abs(row.auprc-average_precision_score(g.y,g.probability))<1e-12
        p=np.clip(g.probability.to_numpy(dtype=np.float64),1e-15,1-1e-15)
        assert abs(row.log_loss-log_loss(g.y,p,labels=[0,1]))<1e-12
        assert abs(row.ez_fraction_bias-(g.prediction.mean()-g.y.mean()))<1e-12
    # Recheck every selected validation threshold and paired candidate argmax without test data.
    hist=pd.read_csv(a.runtime/'public/VALIDATION_SEARCH.csv')
    for r in selections:
        q=read(r['path']);d=banks[r['fold']-1];va=d['val']
        assert EV.threshold((d['y'][va]==0).astype(int),q['validation_probability'],d['patient'][va])==q['selected']
        assert r['threshold']==q['selected']['threshold'] and sha(r['path'])==r['model_sha256']
        h=hist[(hist.model==r['model'])&(hist.fold==r['fold'])]
        scores=[]
        for ci,g in h.groupby('candidate'):
            assert set(g.version)=={'A','B'}
            scores.append((tuple(g[k].mean() for k in ['validation_macro_f1','validation_auprc','validation_auroc'])+(-int(ci),),int(ci)))
        assert r['candidate']==max(scores)[1]
    old=read(OLD/'PATIENT_METRICS_PRIVATE.pkl');maximum=0.
    for version in ['A','B']:
        l=old[(old.model=='B2')&(old.version==version)].set_index('patient').sort_index()
        r=patients[(patients.model=='RBF_SVM')&(patients.version==version)].set_index('patient').loc[l.index]
        for m in ['auroc','auprc','macro_f1','f1','accuracy']:
            maximum=max(maximum,float(np.nanmax(np.abs(l[m]-r[m]))))
    assert maximum<1e-12
    # Paired patient bootstrap: PR effect for each model and new-vs-RBF in both views.
    ids=sorted(patients.patient.unique());rng=np.random.default_rng(42);draw=rng.integers(0,80,size=(10000,80))
    counts=np.stack([(draw==i).sum(axis=1) for i in range(80)],axis=1);out=[]
    pairs=[((m,'A'),(m,'B')) for m in ['Linear_SVM','AdaBoost','RBF_SVM']]
    pairs += [(('RBF_SVM',v),(m,v)) for v in ['A','B'] for m in ['Linear_SVM','AdaBoost']]
    for left,right in pairs:
        l=patients[(patients.model==left[0])&(patients.version==left[1])].set_index('patient').loc[ids]
        r=patients[(patients.model==right[0])&(patients.version==right[1])].set_index('patient').loc[ids]
        assert np.array_equal(l.fold,r.fold) and np.array_equal(l.n,r.n)
        diff=r[KEYS].to_numpy()-l[KEYS].to_numpy();valid=np.isfinite(diff)
        num=counts@np.where(valid,diff,0);den=counts@valid.astype(int)
        dist=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        direct=np.nanmean(diff[draw[:128]],axis=1)
        assert np.allclose(direct,dist[:128],atol=1e-12,rtol=0,equal_nan=True)
        for j,m in enumerate(KEYS):out.append(dict(reference_model=left[0],reference_version=left[1],model=right[0],version=right[1],
            metric=m,delta=MET.nanmean(diff[:,j]),ci_low=float(np.nanpercentile(dist[:,j],2.5)),
            ci_high=float(np.nanpercentile(dist[:,j],97.5)),paired_valid_patients=int(valid[:,j].sum()),draws=10000,seed=42))
    pd.DataFrame(out).to_csv(a.runtime/'public/paired_comparisons.csv',index=False)
    for p,h in binding['protected'].items():assert sha(p)==h
    for p in (a.runtime/'public').glob('*.csv'):
        df=pd.read_csv(p);assert not set(df.columns)&{'patient','channel','global_index','path','probability','prediction','y'}
    verification=dict(status='PASS',new_model_fold_cells=20,candidate_cells=50,new_patient_rows=320,reference_patient_rows=160,
        patients_per_group=80,channels_per_group=7635,all_six_groups=True,independent_sklearn_metrics=True,
        validation_only_selection_reverified=True,matched_PR_hyperparameters=True,reference_primary_max_error=maximum,
        bootstrap_draws=10000,bootstrap_direct_128_draws_max_tolerance=1e-12,protected_artifacts_unchanged=True,
        privacy_columns_check=True,freeze_sha256=fh)
    write(a.runtime/'public/INDEPENDENT_VERIFICATION.json',verification)
    write(a.runtime/'public/SOURCE_HASHES.json',dict(source=binding['source'],parent_binding_sha256=digest(binding['parent_binding']),
        parent_outer_freeze_sha256=sha(OLD/'FROZEN_BEFORE_OUTER_PRIVATE.json'),parent_outer_seal_sha256=sha(OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')))
    status=dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',groups=4,new_models=20,reference_groups=2,
        patients_per_group=80,channels_per_group=7635,freeze_sha256=fh,
        seal_sha256=sha(a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),scientific_status='EXPLORATORY_REPEATED_TEST',
        public_sha256={p.name:sha(p) for p in (a.runtime/'public').iterdir() if p.name not in ['OUTER_STATUS.json','RUN_STATUS.json']})
    write(a.runtime/'public/OUTER_STATUS.json',status)
    write(a.runtime/'public/RUN_STATUS.json',dict(status='COMPLETE',selected_cells=20,total_cells=20))
    print('ALL4_NEW_GROUPS_5FOLD_OUTER_AND_INDEPENDENT_VERIFY_PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\m8_14d_pr_linear_svm_adaboost_seed42_v1_runtime'))
    a=p.parse_args();a.baseline=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
    a.features=Path(r'D:\d1_m8_progressive_ablation_seed42_v1_runtime');a.references=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime')
    protocol,oldlock,core,rec,banks,x,binding=initialize(a)
    selections=train(a,protocol,core,banks,x,binding);fh=freeze(a,selections,binding)
    frame=outer(a,oldlock,core,rec,x,selections,fh)
    verify_and_report(a,frame,banks,selections,binding,fh)
