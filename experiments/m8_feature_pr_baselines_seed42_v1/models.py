"""Five conventional classifiers, matched candidate grids; FIT-only SVM calibration."""
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

def patient_weights(y,patient,class_weight=True):
    y=np.asarray(y,dtype=int);patient=np.asarray(patient);ids,counts=np.unique(patient,return_counts=True)
    sizes=dict(zip(ids,counts));w=np.array([len(y)/(len(ids)*sizes[p]) for p in patient],float)
    if class_weight:
        n=np.bincount(y,minlength=2);assert (n>0).all();w*=len(y)/(2*n[y])
    return w

def preprocessing(x,indices,scaled):
    imp=SimpleImputer();imp.fit(x[indices]);z=imp.transform(x)
    sc=StandardScaler() if scaled else None
    if sc is not None:sc.fit(z[indices]);z=sc.transform(z)
    assert z.shape==x.shape and np.isfinite(z).all()
    return z,dict(imputer=imp,scaler=sc)

def apply_pre(x,pre):
    z=pre['imputer'].transform(x)
    return pre['scaler'].transform(z) if pre['scaler'] is not None else z

def estimator(model,config,seed):
    if model=='B1':return LogisticRegression(C=config['C'],solver='lbfgs',max_iter=2000,tol=1e-7,random_state=seed)
    if model=='B2':return SVC(C=config['C'],gamma=config['gamma'],kernel='rbf',probability=False,cache_size=512,random_state=seed)
    if model=='B3':return RandomForestClassifier(n_estimators=300,max_depth=config['max_depth'],min_samples_leaf=config['min_samples_leaf'],max_features='sqrt',bootstrap=True,n_jobs=2,random_state=seed)
    if model=='B4':return XGBClassifier(n_estimators=config['n_estimators'],max_depth=config['max_depth'],learning_rate=.05,min_child_weight=5,subsample=1.,colsample_bytree=1.,reg_lambda=1.,objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=2,random_state=seed)
    if model=='B5':return MLPClassifier(hidden_layer_sizes=tuple(config['hidden']),alpha=config['alpha'],activation='relu',solver='adam',learning_rate_init=.001,batch_size=128,max_iter=400,tol=1e-5,n_iter_no_change=20,early_stopping=False,random_state=seed)
    raise ValueError(model)

def fit(model,config,x,y,patient,tr,seed):
    """Receives legal FIT/VAL bank only; the estimator and Platt fitting see FIT only."""
    scaled=model in ['B1','B2','B5'];z,pre=preprocessing(x,tr,scaled);w=patient_weights(y[tr],patient[tr]);calibration=None;cal_audit=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if model=='B2':
            oof=np.full(len(tr),np.nan)
            cv=GroupKFold(n_splits=3,shuffle=True,random_state=seed)
            for k,(it,iv) in enumerate(cv.split(x[tr],y[tr],groups=patient[tr])):
                fitix,holdix=tr[it],tr[iv]
                assert set(patient[fitix]).isdisjoint(set(patient[holdix]))
                zz,_=preprocessing(x,fitix,True)
                m=estimator(model,config,seed);m.fit(zz[fitix],y[fitix],sample_weight=patient_weights(y[fitix],patient[fitix]))
                oof[iv]=m.decision_function(zz[holdix])
                cal_audit.append(dict(split=k,fit_patients=len(set(patient[fitix])),held_out_patients=len(set(patient[holdix])),patient_exclusion=True,preprocessing_fit_subtrain_only=True))
            assert np.isfinite(oof).all()
            calibration=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=seed)
            calibration.fit(oof[:,None],y[tr],sample_weight=patient_weights(y[tr],patient[tr],False))
        m=estimator(model,config,seed);m.fit(z[tr],y[tr],sample_weight=w)
    assert np.array_equal(m.classes_,[0,1])
    return dict(model=m,pre=pre,calibration=calibration,model_id=model,config=config,calibration_audit=cal_audit,warning_categories=sorted(set(type(w.message).__name__ for w in caught)),iterations=int(np.asarray(m.n_iter_).max()) if model in ['B1','B5'] else None)

def predict(bundle,x):
    z=apply_pre(x,bundle['pre']);m=bundle['model']
    if bundle['model_id']=='B2':
        decision=m.decision_function(z);p=bundle['calibration'].predict_proba(decision[:,None])[:,1]
    else:
        p=m.predict_proba(z)[:,1];decision=m.decision_function(z) if hasattr(m,'decision_function') else p.copy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return np.asarray(p,dtype=float),np.asarray(decision,dtype=float)
