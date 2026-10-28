"""Fixed-C linear SVC, nested patient-held-out Platt fitting, and score replay."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

def weights(y,patient,balanced=True):
    ids,count=np.unique(patient,return_counts=True);size=dict(zip(ids,count))
    w=np.array([len(y)/(len(ids)*size[p]) for p in patient],float)
    if balanced:w*=len(y)/(2*np.bincount(y,minlength=2)[y])
    return w

def preprocess(x,indices):
    imputer=SimpleImputer().fit(x[indices]);z=imputer.transform(x)
    scaler=StandardScaler().fit(z[indices]);return scaler.transform(z),dict(imputer=imputer,scaler=scaler)

def transformed(bundle,x):return bundle['pre']['scaler'].transform(bundle['pre']['imputer'].transform(x))

def fit_fixed(x,y,pid,train,C,seed):
    z,pre=preprocess(x,train);oof=np.full(len(train),np.nan);audit=[]
    def estimator():return SVC(kernel='linear',C=C,probability=False,cache_size=512,random_state=seed)
    for split,(a,b) in enumerate(GroupKFold(3,shuffle=True,random_state=seed).split(x[train],y[train],pid[train])):
        ti,vi=train[a],train[b];assert set(pid[ti]).isdisjoint(set(pid[vi]))
        zz,_=preprocess(x,ti);m=estimator();m.fit(zz[ti],y[ti],sample_weight=weights(y[ti],pid[ti]))
        assert m.fit_status_==0;oof[b]=m.decision_function(zz[vi])
        audit.append(dict(split=split,fit_patients=len(set(pid[ti])),held_patients=len(set(pid[vi])),patient_exclusion=True))
    assert np.isfinite(oof).all()
    cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=seed)
    cal.fit(oof[:,None],y[train],sample_weight=weights(y[train],pid[train],False))
    m=estimator();m.fit(z[train],y[train],sample_weight=weights(y[train],pid[train]));assert m.fit_status_==0
    return dict(model=m,pre=pre,calibration=cal,calibration_audit=audit)

def logits(bundle,x):
    decision=bundle['model'].decision_function(transformed(bundle,x))
    return np.asarray(bundle['calibration'].decision_function(decision[:,None]),float)

def probabilities(bundle,x):
    decision=bundle['model'].decision_function(transformed(bundle,x))
    return np.asarray(bundle['calibration'].predict_proba(decision[:,None])[:,1],float)

def crossfit(x,y,pid,train,C,seed):
    out=np.full(len(x),np.nan);audit=[]
    for split,(a,b) in enumerate(GroupKFold(3,shuffle=True,random_state=seed).split(x[train],y[train],pid[train])):
        ti,vi=train[a],train[b];assert set(pid[ti]).isdisjoint(set(pid[vi]))
        bundle=fit_fixed(x,y,pid,ti,C,seed+10007*(split+1));out[vi]=logits(bundle,x[vi])
        audit.append(dict(split=split,fit_patients=len(set(pid[ti])),held_patients=len(set(pid[vi])),
                          held_channels=len(vi),patient_exclusion=True,nested_calibration_splits=3))
    assert np.isfinite(out[train]).all();return out,audit
