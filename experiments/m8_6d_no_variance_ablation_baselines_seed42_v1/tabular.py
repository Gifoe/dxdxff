"""Seven fixed parent classifiers, actual12D; paired PR, no new hyperparameter search."""
from common import *
import time
P5=Path(r'D:\m8_feature_pr_baselines_seed42_v1')
EXTRA=Path(r'D:\m8_14d_pr_linear_svm_adaboost_seed42_v1')
MD=module('sixd_classifiers',P5/'models.py')
BL=module('sixd_linear_ada',EXTRA/'run.py')
IDS={'LogisticRegression':'B1','RBF_SVM':'B2','RandomForest':'B3','XGBoost':'B4','MLP':'B5'}
CONFIG=jr(HERE/'BASELINE_CONFIGS.json')

def view(core,z,patient,pr):
    a=np.asarray(z)[:,KEEP12]
    out=core.patient_z(a,patient) if pr else a.astype(np.float64)
    expected=core.patient_z(z,patient)[:,KEEP12] if pr else np.asarray(z,dtype=np.float64)[:,KEEP12]
    assert out.shape==(len(z),12) and np.array_equal(out,expected) and np.isfinite(out).all()
    return out

def admission(binding,core,rec,banks):
    import xgboost,sklearn
    assert sklearn.__version__=='1.7.2' and xgboost.__version__=='3.0.5'
    rows=[]
    for f in range(1,6):
        d=bank(f,banks,'FULL_F0');r=rec['R14'][d['archive_indices']]
        for pr in [False,True]:
            z=view(core,r,d['patient'],pr);scaled,_=MD.preprocessing(z,d['train'],True)
            reference_bank=bank(f,banks,'FULL_F0' if pr else 'WO_BOTH_PR')
            assert np.array_equal(scaled.astype(np.float32),reference_bank['x'][:,KEEP12])
            rows.append(dict(fold=f,PR=pr,active_dimension=12,exact_matched_FIT_bank=True,test_patients_excluded=True))
    for classifier in MODEL.CLASSIFIERS:
        for f in range(1,6):
            config=CONFIG['configs'][classifier][str(f)]
            if classifier in IDS:MD.estimator(IDS[classifier],config,42+1009*f)
    write(PUB/'TABULAR_ADMISSION.json',dict(status='PASS',rows=rows,models=7,configs_frozen=True,new_search=False,
        sklearn=sklearn.__version__,xgboost=xgboost.__version__,no_raw_or_new_extraction=True))
    print('SIXD_TABULAR_12D_BOTH_VIEWS_ADMISSION_PASS',flush=True)

def fit_model(classifier,config,x,y,pid,tr,seed):
    return MD.fit(IDS[classifier],config,x,y,pid,tr,seed) if classifier in IDS else BL.fit(classifier,config,x,y,pid,tr,seed)

def scores(classifier,bundle,x):
    return (MD.predict(bundle,x) if classifier in IDS else BL.predict(bundle,x))[0]

def train():
    binding,core,fac,records,rec,banks=setup()
    assert jr(PUB/'ADMISSION.json')['status']=='PASS'
    assert jr(PUB/'SOURCE_PUSH_BEFORE_TRAIN.json')['source']==sources()
    cells=[];rows=[];cal=[]
    for f in range(1,6):
        d=bank(f,banks,'FULL_F0');ix=d['archive_indices'];y=(d['y']==0).astype(int);seed=42+1009*f
        for classifier in MODEL.CLASSIFIERS:
            config=CONFIG['configs'][classifier][str(f)]
            for pr in [False,True]:
                arm=classifier+('_PR' if pr else '_NO_PR')
                z=view(core,rec['R14'][ix],d['patient'],pr)
                path=ROOT/'tabular_models_private'/arm/f'fold{f}.pkl'
                cb=dict(input=binding,arm=arm,fold=f,seed=seed,config=config,features=legacy.array_sha(z),train=legacy.array_sha(d['train']),validation=legacy.array_sha(d['val']))
                if path.exists():q=read(path);assert q['binding']==cb
                else:
                    tick=time.perf_counter();bundle=fit_model(classifier,config,z,y,d['patient'],d['train'],seed)
                    p=scores(classifier,bundle,z[d['val']]);sel=BL.EV.threshold(y[d['val']],p,d['patient'][d['val']])
                    q=dict(binding=cb,bundle=bundle,validation_probability=p,selected=sel,seconds=time.perf_counter()-tick)
                    save(path,q)
                r=dict(arm=arm,fold=f,checkpoint=str(path),checkpoint_sha256=sha(path),validation=str(path),validation_sha256=sha(path),
                    threshold=q['selected']['threshold'],threshold_class='EZ_GE',reused=False,classifier=classifier,PR=pr,config=config)
                cells.append(r);rows.append(dict(arm=arm,fold=f,active_dimension=12,PR=pr,threshold=r['threshold'],config_json=json.dumps(config,sort_keys=True),
                    checkpoint_sha256=r['checkpoint_sha256'],iterations=q['bundle']['iterations'],seconds=q['seconds'],warning_categories=';'.join(q['bundle']['warning_categories'])))
                for a in q['bundle']['calibration_audit']:cal.append(dict(arm=arm,fold=f,**a))
                csv_out('TABULAR_TRAIN_CELLS',rows)
                write(PUB/'TABULAR_TRAIN_STATUS.json',dict(status='TRAINING',completed_cells=len(cells),total_cells=70))
                print('SIXD_TABULAR_CELL_COMPLETE',arm,f,len(cells),70,flush=True)
    assert len(cells)==70
    path=ROOT/'TABULAR_FREEZE_PRIVATE.pkl';obj=dict(input=binding,cells=cells)
    if path.exists():assert read(path)==obj
    else:save(path,obj)
    csv_out('TABULAR_CALIBRATION_AUDIT',cal)
    write(PUB/'TABULAR_TRAIN_STATUS.json',dict(status='ALL70_TABULAR_MODELS_FROZEN',models=70,freeze_sha256=sha(path),new_hyperparameter_search=False))
    print('ALL70_TABULAR_12D_MODELS_FROZEN',flush=True)

def verify_selected(binding,core,rec,banks):
    q=read(ROOT/'TABULAR_FREEZE_PRIVATE.pkl');assert q['input']==binding and len(q['cells'])==70
    checks=[]
    for r in q['cells']:
        d=bank(r['fold'],banks,'FULL_F0');saved=read(r['checkpoint']);config=CONFIG['configs'][r['classifier']][str(r['fold'])]
        assert r['config']==config and sha(r['checkpoint'])==r['checkpoint_sha256']
        z=view(core,rec['R14'][d['archive_indices']],d['patient'],r['PR'])
        assert legacy.array_sha(z)==saved['binding']['features']
        p=scores(r['classifier'],saved['bundle'],z[d['val']])
        assert np.array_equal(p,saved['validation_probability'])
        sel=BL.EV.threshold((d['y'][d['val']]==0).astype(int),p,d['patient'][d['val']])
        assert sel==saved['selected'] and sel['threshold']==r['threshold']
        if r['classifier'] in ['Linear_SVM','RBF_SVM']:
            a=saved['bundle']['calibration_audit'];assert len(a)==3
            assert all(v['patient_exclusion'] and v['preprocessing_fit_subtrain_only'] for v in a)
        checks.append(dict(arm=r['arm'],fold=r['fold'],maximum_replay_error=0.,threshold_verified=True,FIT_only=True,active_dimension=12))
    csv_out('TABULAR_VALIDATION_CHECKS',checks);return len(checks)

def outer():
    binding,core,fac,records,rec,banks=setup();fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    tab=read(ROOT/'TABULAR_FREEZE_PRIVATE.pkl');seal={};coverage=[];meta=rec['meta']
    for r in tab['cells']:
        f=r['fold'];d=bank(f,banks,'FULL_F0');ix=np.flatnonzero(np.isin(meta['patient'],list(d['roles']['test'])))
        path=ROOT/'scores_private'/f"{r['arm']}_{f}.pkl"
        if path.exists():q=read(path)
        else:
            z=view(core,rec['R14'][ix],meta['patient'][ix],r['PR']);saved=read(r['checkpoint'])
            p=scores(r['classifier'],saved['bundle'],z)
            q=dict(freeze=fh,indices=ix,probability_ez=p,score_nez=1-p,prediction_ez=p>=r['threshold'],threshold=r['threshold'],checkpoint=r['checkpoint_sha256'])
            save(path,q)
        assert q['freeze']==fh and q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold'] and np.array_equal(ix,q['indices'])
        assert np.array_equal(q['prediction_ez'],q['probability_ez']>=r['threshold']) and 'y' not in q
        seal[f"{r['arm']}_{f}"]=sha(path)
        coverage.append(dict(arm=r['arm'],fold=f,patients=len(d['roles']['test']),channels=len(ix),active_dimension=12,raw_consumed=False))
        print('SIXD_TABULAR_OUTER_CELL_SEALED',r['arm'],f,len(seal),70,flush=True)
    freeze(ROOT/'TABULAR_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,files=seal))
    csv_out('TABULAR_COVERAGE',coverage)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','outer']);args=p.parse_args()
    train() if args.phase=='train' else outer()

