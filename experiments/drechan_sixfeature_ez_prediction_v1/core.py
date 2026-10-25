"""Frozen CPU-only six-feature prediction math and trusted-cache access."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import warnings
import numpy as np
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score

HERE=Path(__file__).resolve().parent
REC=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime\RECONSTRUCTION_PRIVATE.pkl')
LABEL=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
PR_PARENT=Path(r'D:\drechan_f0_complete_ablation_3seed_v1\feature_pr.py')
A_ROOT=Path(r'D:\drechan_sixfeature_ez_association_v1_runtime\public')
KEEP=[0,1,2,3,4,5,6,7,10,11,12,13]
FEATURES=['BETA','LOW_GAMMA','HIGH_GAMMA','RMS','LINE_LENGTH','SPECTRAL_ENTROPY']
MAIN=[f+'_PR' for f in FEATURES]+['ALL_SIX_NO_PR','ALL_SIX_PR']
ARMS=MAIN+[f+'_NO_PR' for f in FEATURES]
METRICS=['AUROC','AP','Macro_F1','EZ_F1','NEZ_F1','Accuracy','MRR','Top_1']
PRIMARY=['AUROC','AP','Macro_F1','Top_1']
CS=[.01,.1,1.,10.]
SCIENCE=['PROTOCOL.json','core.py','run.py','report.py','verify.py','test_math.py']
EXPECTED={
 'reconstruction':'73095eefbb3035200894560de3c3cfdb6259a4f28c343ecd3d7195be5b94f609',
 'labels':'1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd',
 'PR_parent':'3b9823db47382cc3d4d66d226b4041f05efbc65041095fccaf298f459dc8148f',
 'A_input':'ac9decf436a87dcf58f35ba96b4df9dd914c1ce1dbfddd97aad5acb95c36ed7d',
 'A_verification':'77dcf4ba5e885d0383962fcefed1e8abe5821874f471442d3bd50f5278b55aa1',
 'split':'fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def array_sha(a):
    a=np.asarray(a)
    if a.dtype.kind in 'OUS':return digest(a.astype(str).tolist())
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def clean(q):
    if isinstance(q,np.ndarray):return q.tolist()
    if isinstance(q,np.generic):return q.item()
    if isinstance(q,Path):return str(q)
    if isinstance(q,dict):return {str(k):clean(v) for k,v in q.items()}
    if isinstance(q,(list,tuple)):return [clean(v) for v in q]
    return q


def digest(q):return hashlib.sha256(json.dumps(clean(q),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def jr(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,q):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(clean(q),indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def save(path,q):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),('PRESERVE_EXISTING_ARTIFACT',path)
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('wb') as f:pickle.dump(q,f,protocol=5)
    tmp.replace(path)


def read(path):
    with Path(path).open('rb') as f:return pickle.load(f)


def sources():return {name:sha(HERE/name) for name in SCIENCE}


def cpu():
    import psutil
    psutil.Process().cpu_affinity([2])
    assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    assert 'torch' not in sys.modules,'No neural runtime is needed'


def load_data():
    assert sha(REC)==EXPECTED['reconstruction'] and sha(LABEL)==EXPECTED['labels']
    assert sha(PR_PARENT)==EXPECTED['PR_parent']
    assert sha(A_ROOT/'INPUT_AUDIT.json')==EXPECTED['A_input']
    assert sha(A_ROOT/'INDEPENDENT_VERIFICATION.json')==EXPECTED['A_verification']
    assert jr(A_ROOT/'INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    r=read(REC);m=r['meta'];raw=r['R14']
    assert r['binding']['archive']==EXPECTED['labels'] and r['binding']['split']==EXPECTED['split']
    with np.load(LABEL,allow_pickle=True) as z:
        for key in ['patient','channel','center']:assert np.array_equal(z[key].astype(str),m[key])
        y=(z['y']==0).astype(np.int64)
    assert raw.shape==(7635,14) and np.isfinite(raw).all() and y.sum()==1743
    assert len(set(m['patient']))==80 and len(set(zip(m['patient'],m['channel'])))==7635
    assert sum(len(z['records']) for z in r['ledgers'])==256
    assert sum(int(z['mask'].sum()) for a in r['ledgers'] for z in a['records'])==1471965
    assert dict(zip(*np.unique(m['center'],return_counts=True))) # channel counts, not patient counts
    roles=[];visited=[]
    for k in range(1,6):
        rr={}
        for name in ['fit','validation','test']:
            ids=sorted(set(m['split_patient'][(m['split_fold']==k)&(m['split_role']==name)]))
            rr[name]=np.flatnonzero(np.isin(m['patient'],ids))
            expected={'fit':[51,51,50,52,51][k-1],'validation':13,'test':[16,16,17,15,16][k-1]}
            assert len(ids)==expected[name]
        assert set(rr['fit']).isdisjoint(rr['validation']) and set(rr['fit']).isdisjoint(rr['test']) and set(rr['validation']).isdisjoint(rr['test'])
        assert sorted(np.concatenate(list(rr.values())).tolist())==list(range(7635))
        assert len(rr['test'])==[1631,1498,1746,1425,1335][k-1]
        visited.extend(rr['test'].tolist());roles.append(rr)
    assert sorted(visited)==list(range(7635))
    for pid in sorted(set(m['patient'])):
        yy=y[m['patient']==pid];assert yy.any() and (1-yy).any()
    x=raw[:,KEEP].astype(np.float64)
    return dict(x=x,raw14=raw,y=y,patient=m['patient'],channel=m['channel'],source=m['center'],roles=roles)


def columns(arm):
    if arm.startswith('ALL_SIX'):return np.arange(12)
    name=arm.removesuffix('_NO_PR').removesuffix('_PR');j=FEATURES.index(name)
    return np.array([2*j,2*j+1])


def relative(x,patient,units):
    x=np.asarray(x,np.float64);patient=np.asarray(patient)
    out=np.full_like(x,np.nan)
    for pid in np.unique(patient):
        ix=patient==pid;v=x[ix]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            mu=np.nanmean(v,axis=0);sd=np.nanstd(v,axis=0,ddof=0)
        supported=np.isfinite(sd)&(sd>1e-8*units)
        z=np.divide(v-mu,sd,out=np.zeros_like(v),where=supported[None,:]);z[np.isnan(v)]=np.nan
        out[ix]=z
    return out


def fit_pre(x,patient,pr):
    """Receives FIT (or sub-FIT) only. All operations are coordinate-wise."""
    x=np.asarray(x,float);assert not np.isinf(x).any()
    imp=SimpleImputer(strategy='mean',keep_empty_features=True).fit(x)
    base=imp.transform(x);unit=StandardScaler().fit(base)
    z=relative(x,patient,unit.scale_) if pr else x
    zi=SimpleImputer(strategy='mean',keep_empty_features=True).fit(z)
    sc=StandardScaler().fit(zi.transform(z))
    return dict(PR=bool(pr),units=unit.scale_.copy(),raw_imputer=imp.statistics_.copy(),
        imputer=zi.statistics_.copy(),mean=sc.mean_.copy(),scale=sc.scale_.copy(),
        fit_patients=sorted(set(patient)),fit_input_sha=array_sha(x))


def transform(x,patient,pre):
    x=np.asarray(x,float);assert not np.isinf(x).any()
    z=relative(x,patient,pre['units']) if pre['PR'] else x
    z=((np.where(np.isnan(z),pre['imputer'],z)-pre['mean'])/pre['scale']).astype(np.float32)
    assert np.isfinite(z).all()
    return z.astype(np.float64) # exact original float32 feature bank, then sklearn float64


def weights(y,patient):
    """Mean weight1; every patient equal and each within-patient class half."""
    y=np.asarray(y,int);patient=np.asarray(patient);ids=np.unique(patient);w=np.empty(len(y),float)
    for pid in ids:
        ix=patient==pid;n=np.bincount(y[ix],minlength=2);assert (n>0).all()
        w[ix]=len(y)/(2*len(ids)*n[y[ix]])
    assert abs(w.mean()-1)<1e-12
    return w


def inner_splits(patient,fold):
    ids=np.unique(patient);order=np.random.default_rng(42+1009*fold).permutation(ids)
    out=[]
    for held in np.array_split(order,3):
        va=np.flatnonzero(np.isin(patient,held));tr=np.flatnonzero(~np.isin(patient,held))
        assert set(patient[tr]).isdisjoint(patient[va]);out.append((tr,va))
    assert sorted(np.concatenate([v for _,v in out]).tolist())==list(range(len(patient)))
    return out


def fit_lr(x,y,patient,C):
    w=weights(y,patient)
    model=LogisticRegression(C=C,solver='lbfgs',max_iter=2000,tol=1e-6,penalty='l2',class_weight=None,random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(x,y,sample_weight=w)
    assert np.array_equal(model.classes_,[0,1]) and np.isfinite(model.coef_).all()
    state=dict(C=float(C),iterations=int(model.n_iter_[0]),warnings=sorted(set(type(w.message).__name__ for w in caught)),
        converged=not any(type(w.message).__name__=='ConvergenceWarning' for w in caught),
        training_labels_sha=array_sha(y),sample_weights_sha=array_sha(w),training_input_sha=array_sha(x))
    return model,state


def predict(model,x):
    p=model.predict_proba(x)[:,1]
    np.testing.assert_allclose(p,expit(x@model.coef_[0]+model.intercept_[0]),atol=2e-15,rtol=0)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();return p


def endpoints(y,p,pred,channel):
    y=np.asarray(y,int);pred=np.asarray(pred,bool);p=np.asarray(p,float)
    tp=int(((y==1)&pred).sum());tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
    assert y.any() and (1-y).any()
    f1=2*tp/(2*tp+fp+fn);f0=2*tn/(2*tn+fp+fn)
    order=np.lexsort((np.asarray(channel).astype(str),-p));first=int(np.flatnonzero(y[order])[0])
    return dict(AUROC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),
        Macro_F1=(f0+f1)/2,EZ_F1=f1,NEZ_F1=f0,Accuracy=(tp+tn)/len(y),MRR=1/(first+1),Top_1=float(first==0))


def patient_metrics(y,p,patient,channel,threshold):
    return [dict(patient=pid,**endpoints(y[patient==pid],p[patient==pid],p[patient==pid]>=threshold,channel[patient==pid])) for pid in np.unique(patient)]


def patient_auc(y,p,patient):
    return float(np.mean([roc_auc_score(y[patient==pid],p[patient==pid]) for pid in np.unique(patient)]))


def choose_c(scores):
    best=max(scores.values());return min(c for c,s in scores.items() if s>=best-1e-12)


def threshold(y,p,patient):
    grid=np.round(np.arange(201)*.005,3);b=p[:,None]>=grid[None,:];mf=[];ez=[]
    for pid in np.unique(patient):
        ix=patient==pid;truth=y[ix,None];z=b[ix]
        tp=((truth==1)&z).sum(0);tn=((truth==0)&~z).sum(0);fp=((truth==0)&z).sum(0);fn=((truth==1)&~z).sum(0)
        a=2*tp/(2*tp+fp+fn);c=2*tn/(2*tn+fp+fn);mf.append((a+c)/2);ez.append(a)
    macro=np.mean(mf,axis=0);pos=np.mean(ez,axis=0)
    j=max(range(201),key=lambda i:(round(float(macro[i]),12),round(float(pos[i]),12),-abs(float(grid[i])-.5),-float(grid[i])))
    return dict(threshold=float(grid[j]),CAL_Macro_F1=float(macro[j]),CAL_EZ_F1=float(pos[j]))


def permuted_labels(y,patient,fold,b):
    rng=np.random.default_rng([42,20261025,fold,b]);out=np.asarray(y).copy()
    for pid in np.unique(patient):
        ix=np.flatnonzero(patient==pid);out[ix]=rng.permutation(y[ix]);assert out[ix].sum()==y[ix].sum()
    return out


def bootstrap(values,adjusted=False):
    a=np.asarray(values,float);assert a.ndim==1 and np.isfinite(a).all()
    ids=np.random.default_rng(42).integers(len(a),size=(10000,len(a)))
    samples=a[ids].mean(1);alpha=.05/28 if adjusted else .05
    lo,hi=np.quantile(samples,[alpha/2,1-alpha/2])
    return dict(mean=float(a.mean()),ci_low=float(lo),ci_high=float(hi),patients=len(a))
