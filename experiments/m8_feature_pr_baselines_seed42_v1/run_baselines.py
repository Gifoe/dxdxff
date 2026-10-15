"""Five classifiers x two PR settings; matched inner selection, frozen Outer."""
import time
from support import *
MODELS=load('feature_pr_classifiers',ROOT/'models.py')
REPORT=load('feature_pr_aggregate',ROOT/'aggregate.py')

def gate(a):
    lock,core,rec,banks,x,binding=setup(a);pub=a.runtime/'public'
    audit=[]
    for fold,d in enumerate(banks,1):
        r=x[d['archive_indices']];z=transform_view(core,r,d['patient'],True)
        scaled,_=MODELS.preprocessing(z,d['train'],True)
        bp=a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt'
        bd=torch.load(bp,map_location='cpu',weights_only=False)
        assert np.array_equal(scaled.astype(np.float32),bd['x'])
        independent=r.copy().astype(float)
        for pid in np.unique(d['patient']):
            ix=d['patient']==pid;s=np.std(r[ix].astype(float),axis=0)
            independent[ix]=(r[ix].astype(float)-r[ix].astype(float).mean(0))/np.where(s>1e-8,s,1.)
        assert np.array_equal(z,independent)
        audit.append(dict(fold=fold,fit_patients=len(d['roles']['fit']),validation_patients=len(d['roles']['validation']),test_patients_excluded=True,PatientZ_exact_original=True,scaled_PR_exact_D1_legal_bank=True,dimension=14))
    obj=dict(binding=binding,folds=audit,cohort_patients=80,cohort_channels=7635,seizures=256,no_raw_or_EDF_processing=True)
    fp=a.runtime/'INPUT_FREEZE_PRIVATE.json'
    if fp.exists():assert jread(fp)==obj
    else:write(fp,obj)
    write(pub/'INPUT_AND_PR_AUDIT.json',dict(status='PASS',**obj))
    write(pub/'RUN_STATUS.json',dict(status='INPUT_GATES_PASS',selected_cells=0,total_cells=50))
    print('FEATURE_PR_INPUT_GATES_PASS',flush=True)

def train(a):
    lock,core,rec,banks,x,binding=setup(a);pub=a.runtime/'public'
    assert jread(a.runtime/'INPUT_FREEZE_PRIVATE.json')['binding']==binding
    fp=a.runtime/'MODEL_SELECTION_PRIVATE.pkl'
    if fp.exists():
        q=read(fp);assert q['binding']==binding
        for r in q['selections']:assert sha(Path(r['path']))==r['model_sha256']
        print('TRAINING_ALREADY_COMPLETE_NO_REFIT',flush=True);return
    selections=[];search=[];calibration=[];timings=[]
    for fold,d in enumerate(banks,1):
        ix=d['archive_indices'];y=(d['y']==0).astype(int);pid=d['patient'];va=d['val'];tr=d['train'];seed=42+1009*fold
        views={'A':transform_view(core,x[ix],pid,False),'B':transform_view(core,x[ix],pid,True)}
        for model,configs in lock['grids'].items():
            candidates=[]
            for ci,config in enumerate(configs):
                pair=[]
                for version in ['A','B']:
                    cp=a.runtime/f'candidates_private/{model}/fold{fold}/c{ci}_{version}.pkl'
                    cb=dict(binding=binding,fold=fold,model=model,version=version,config=config,seed=seed,legal_bank_sha256=binding['dependencies'][f'legal_bank{fold}'])
                    if cp.exists():q=read(cp);assert q['binding']==cb
                    else:
                        start=time.perf_counter();bundle=MODELS.fit(model,config,views[version],y,pid,tr,seed)
                        p,decision=MODELS.predict(bundle,views[version][va]);selected=EV.threshold(y[va],p,pid[va])
                        frame=pd.DataFrame(dict(model=model,version=version,fold=fold,patient=pid[va],channel=d['channel'][va],center=d['center'][va],y=y[va],probability=p,prediction=p>=selected['threshold']))
                        rows=pd.DataFrame(EV.patient_rows(frame))
                        q=dict(binding=cb,bundle=bundle,validation_probability=p,validation_decision_score=decision,selected=selected,validation_patient_mean={m:EV.nanmean(rows[m]) for m in EV.CLASSIFICATION},wall_seconds=time.perf_counter()-start)
                        save(cp,q)
                    assert q['validation_probability'].shape==(len(va),)
                    pair.append(dict(path=str(cp),model_sha256=sha(cp),**q['selected'],**{f'validation_{m}':q['validation_patient_mean'][m] for m in ['auroc','auprc']},version=version,config=config))
                    search.append(dict(model=model,version=version,fold=fold,candidate=ci,config_json=json.dumps(config,sort_keys=True),**q['selected'],validation_auroc=q['validation_patient_mean']['auroc'],validation_auprc=q['validation_patient_mean']['auprc'],model_sha256=sha(cp),iterations=q['bundle']['iterations'],warning_categories=';'.join(q['bundle']['warning_categories']),wall_seconds=q['wall_seconds']))
                    for v in q['bundle']['calibration_audit']:calibration.append(dict(model=model,version=version,fold=fold,candidate=ci,**v))
                    print('CANDIDATE_COMPLETE',model,fold,version,ci+1,'of',len(configs),flush=True)
                key=(np.mean([r['validation_macro_f1'] for r in pair]),np.mean([r['validation_auprc'] for r in pair]),np.mean([r['validation_auroc'] for r in pair]),-ci)
                candidates.append((key,pair,ci))
            _,chosen,ci=max(candidates,key=lambda item:item[0])
            for r in chosen:selections.append(dict(r,model=model,fold=fold,candidate=ci,seed=seed))
            write(pub/'RUN_STATUS.json',dict(status='TRAINING',selected_cells=len(selections),total_cells=50,candidate_cells_complete=len(search)))
            print('MATCHED_PAIR_SELECTED',model,fold,'config',ci,flush=True)
    assert len(selections)==50 and len(search)==190
    save(fp,dict(binding=binding,selections=selections))
    public=[{k:v for k,v in r.items() if k!='path'} for r in selections]
    for r in public:r['config_json']=json.dumps(r.pop('config'),sort_keys=True)
    pd.DataFrame(public).to_csv(pub/'MODEL_SELECTION.csv',index=False)
    pd.DataFrame(search).to_csv(pub/'VALIDATION_SEARCH.csv',index=False)
    pd.DataFrame(calibration).to_csv(pub/'SVM_CALIBRATION_AUDIT.csv',index=False)
    write(pub/'RUN_STATUS.json',dict(status='TRAINING_COMPLETE',selected_cells=50,candidate_cells=190,extra_seeds=0))
    print('ALL50_SELECTED_CELLS_TRAINING_COMPLETE',flush=True)

def freeze(a,lock,core,rec,banks,binding):
    selected=read(a.runtime/'MODEL_SELECTION_PRIVATE.pkl');assert selected['binding']==binding
    for r in selected['selections']:assert sha(Path(r['path']))==r['model_sha256']
    for model in lock['grids']:
        for fold in range(1,6):
            pair=[r for r in selected['selections'] if r['model']==model and r['fold']==fold]
            assert len(pair)==2 and pair[0]['config']==pair[1]['config'] and pair[0]['seed']==pair[1]['seed']
    obj=dict(binding=binding,selections=selected['selections'],full_new_test_y_unread=True,all_50_models_and_preprocessors_and_thresholds_locked=True)
    fp=a.runtime/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if fp.exists():assert jread(fp)==obj
    else:write(fp,obj)
    write(a.runtime/'public/OUTER_FREEZE_AUDIT.json',dict(status='PASS',freeze_sha256=sha(fp),models=50,matched_pair_hyperparameters=True,own_fold_test_excluded=True,validation_only_thresholds=True))
    return selected['selections'],sha(fp)

def outer(a):
    lock,core,rec,banks,x,binding=setup(a);pub=a.runtime/'public'
    selections,fh=freeze(a,lock,core,rec,banks,binding)
    if (pub/'OUTER_STATUS.json').exists():
        assert jread(pub/'OUTER_STATUS.json')['freeze_sha256']==fh;print('OUTER_ALREADY_COMPLETE_NO_NEW_INFERENCE',flush=True);return
    meta=rec['meta'];files={};cells=[]
    for r in selections:
        model,version,fold=r['model'],r['version'],r['fold'];test=roles(meta,fold)['test'];ix=np.flatnonzero(np.isin(meta['patient'],list(test)))
        pp=a.runtime/f'outer_scores_private/{model}_{version}_fold{fold}.pkl'
        if pp.exists():q=read(pp)
        else:
            z=transform_view(core,x[ix],meta['patient'][ix],version=='B');saved=read(Path(r['path']))
            probability,decision=MODELS.predict(saved['bundle'],z)
            q=dict(indices=ix,probability=probability,decision_score=decision,prediction=probability>=r['threshold'],threshold=r['threshold'],model=model,version=version,fold=fold,freeze_sha256=fh,model_sha256=r['model_sha256'])
            save(pp,q)
        assert q['freeze_sha256']==fh and q['model_sha256']==r['model_sha256'] and np.array_equal(q['indices'],ix) and 'y' not in q
        files[f'{model}_{version}_{fold}']=sha(pp);cells.append(q)
        print('FROZEN_OUTER_CELL_SEALED',model,version,fold,len(ix),flush=True)
    sp=a.runtime/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';seal=dict(freeze_sha256=fh,files=files,all50_scores_sealed_before_new_full_y=True)
    if sp.exists():assert jread(sp)==seal
    else:write(sp,seal)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==lock['archive_sha256']
    with np.load(archive,allow_pickle=True) as source:y=(source['y']==0).astype(int)
    assert len(y)==7635 and y.sum()==1743
    frames=[]
    for q in cells:
        ix=q['indices'];frames.append(pd.DataFrame(dict(model=q['model'],version=q['version'],fold=q['fold'],patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],global_index=ix,y=y[ix],prediction=q['prediction'],probability=q['probability'],decision_score=q['decision_score'],threshold=q['threshold'],config_sha256=q['model_sha256'])))
    frame=pd.concat(frames,ignore_index=True)
    for _,q in frame.groupby(['model','version']):assert len(q)==7635 and q.patient.nunique()==80 and q.global_index.nunique()==7635
    private=a.runtime/'private/results';private.mkdir(parents=True,exist_ok=True)
    frame.to_csv(private/'predictions.csv',index=False,float_format='%.17g')
    patients=pd.DataFrame(EV.patient_rows(frame));patients.to_csv(private/'per_patient_metrics.csv',index=False,float_format='%.17g')
    save(a.runtime/'PREDICTIONS_PRIVATE.pkl',frame);save(a.runtime/'PATIENT_METRICS_PRIVATE.pkl',patients)
    REPORT.aggregate(a,frame,patients)
    REPORT.reference(a,lock,meta,y)
    write(pub/'OUTER_STATUS.json',dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',freeze_sha256=fh,seal_sha256=sha(sp),groups=10,model_fold_cells=50,patients_per_group=80,channels_per_group=7635,all50_sealed_before_full_new_test_y=True,scientific_status='EXPLORATORY_REPEATED_TEST',private_predictions_csv_sha256=sha(private/'predictions.csv'),private_patient_csv_sha256=sha(private/'per_patient_metrics.csv')))
    print('ALL10_GROUPS_OUTER_COMPLETE',flush=True)

if __name__=='__main__':
    a=arguments()
    if a.phase=='gate':gate(a)
    elif a.phase=='train':train(a)
    elif a.phase=='outer':outer(a)
    else:
        verify=load('feature_pr_independent_verify',ROOT/'verify.py');verify.run(a)
