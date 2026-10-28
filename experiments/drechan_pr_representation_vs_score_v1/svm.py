"""Sub-FIT re-estimation of normalization, preprocessing and Platt calibration."""
import time,warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from task_io import *
from math_core import KEEP12,fit_feature,query_feature

def predict(bundle,raw,pid):
    z=query_feature(raw,pid,bundle['pre']);margin=bundle['model'].decision_function(z)
    logit=bundle['calibration'].decision_function(margin[:,None]);prob=bundle['calibration'].predict_proba(margin[:,None])[:,1]
    assert np.isfinite(logit).all() and np.isfinite(prob).all()
    return prob,logit

def fit(p,condition,fold,binding):
    cell=ROOT/'models_private'/'LINEAR_SVM'/condition/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True)
    done=cell/'COMPLETE.json'
    if done.exists():
        q=jr(done);assert q['binding']==binding and sha(cell/'BEST.pkl')==q['checkpoint_sha256'];return q['selection']
    d=p.bank(fold);raw=p.rec['R14'][d['archive_indices']][:,KEEP12];y=1-d['y'];pid=d['patient'];tr=d['train'];va=d['val'];seed=42+1009*fold
    config=p.svm_config[str(fold)]
    def model():return SVC(kernel='linear',C=config['C'],probability=False,cache_size=512,random_state=seed)
    oof=np.full(len(tr),np.nan);audit=[];tick=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for k,(it,iv) in enumerate(GroupKFold(n_splits=3,shuffle=True,random_state=seed).split(raw[tr],y[tr],groups=pid[tr])):
            a,b=tr[it],tr[iv];assert set(pid[a]).isdisjoint(set(pid[b]))
            z,pre=fit_feature(raw,pid,a,condition,False);m=model();m.fit(z[a],y[a],sample_weight=p.md.patient_weights(y[a],pid[a]));assert m.fit_status_==0
            oof[iv]=m.decision_function(z[b]);audit.append(dict(split=k,fit_patients=len(set(pid[a])),held_patients=len(set(pid[b])),statistics_subfit_only=True))
        assert np.isfinite(oof).all()
        cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=seed)
        cal.fit(oof[:,None],y[tr],sample_weight=p.md.patient_weights(y[tr],pid[tr],False))
        z,pre=fit_feature(raw,pid,tr,condition,False);m=model();m.fit(z[tr],y[tr],sample_weight=p.md.patient_weights(y[tr],pid[tr]));assert m.fit_status_==0
    bundle=dict(model=m,pre=pre,calibration=cal,condition=condition,config=config,calibration_audit=audit)
    prob,ll=predict(bundle,raw[va],pid[va]);sel=p.ev.threshold(y[va],prob,pid[va])
    save(cell/'BEST.pkl',dict(binding=binding,bundle=bundle,warnings=sorted(set(type(w.message).__name__ for w in caught))))
    save(cell/'VALIDATION.pkl',dict(logit_ez=ll,probability_ez=prob,indices=d['archive_indices'][va],selection=sel))
    out=dict(model='LINEAR_SVM',condition=condition,fold=fold,checkpoint=str(cell/'BEST.pkl'),checkpoint_sha256=sha(cell/'BEST.pkl'),validation=str(cell/'VALIDATION.pkl'),validation_sha256=sha(cell/'VALIDATION.pkl'),threshold=sel['threshold'],orientation='EZ_GE',config=config,new_oof_fits=3,reused=False)
    write(done,dict(binding=binding,checkpoint_sha256=out['checkpoint_sha256'],selection=out))
    print('PR_REPR_SVM_COMPLETE',condition,fold,'seconds',round(time.perf_counter()-tick,2),flush=True)
    return out
