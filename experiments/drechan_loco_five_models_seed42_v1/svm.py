"""Fixed linear SVC, including unit-scale fitting inside every OOF subFIT."""
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from common import feature_fit,feature_query,KEEP

def weights(y,pid,class_weight=True):
    ids,n=np.unique(pid,return_counts=True);size=dict(zip(ids,n))
    w=np.asarray([len(y)/(len(ids)*size[p]) for p in pid])
    if class_weight:
        counts=np.bincount(y.astype(int),minlength=2);assert (counts>0).all();w*=len(y)/(2*counts[y.astype(int)])
    return w

def fit(d,k):
    tr=d['train'];y=(1-d['y']).astype(int);seed=42+1009*k;oof=np.full(len(tr),np.nan);audit=[]
    def estimator():return SVC(kernel='linear',C=1.,probability=False,cache_size=512,random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cv=GroupKFold(n_splits=3,shuffle=True,random_state=seed)
        for j,(it,iv) in enumerate(cv.split(d['raw'][tr],y[tr],d['patient'][tr]),1):
            ti,vi=tr[it],tr[iv];x,pre=feature_fit(d['raw'],d['patient'],ti)
            assert set(pre['fit_patient_ids'])==set(d['patient'][ti]) and set(d['patient'][ti]).isdisjoint(d['patient'][vi])
            m=estimator();m.fit(x[ti][:,KEEP],y[ti],sample_weight=weights(y[ti],d['patient'][ti]));assert m.fit_status_==0
            oof[iv]=m.decision_function(x[vi][:,KEEP])
            audit.append(dict(split=j,fit_patients=len(set(d['patient'][ti])),held_patients=len(set(d['patient'][vi])),patient_disjoint=True,all_preprocessing_subFIT_only=True))
        assert np.isfinite(oof).all()
        cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=seed)
        cal.fit(oof[:,None],y[tr],sample_weight=weights(y[tr],d['patient'][tr],False))
        m=estimator();m.fit(d['x'][tr][:,KEEP],y[tr],sample_weight=weights(y[tr],d['patient'][tr]));assert m.fit_status_==0
    return dict(model=m,calibration=cal,pre=d['pre'],calibration_audit=audit,OOF=oof,
                warning_categories=sorted(set(type(w.message).__name__ for w in caught)))

def predict(bundle,raw,pid):
    x=feature_query(raw,pid,bundle['pre']);decision=bundle['model'].decision_function(x[:,KEEP])
    p=bundle['calibration'].predict_proba(decision[:,None])[:,1]
    assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all();return p
