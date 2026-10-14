import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def digest(q):return hashlib.sha256(json.dumps(q,sort_keys=True,default=str).encode()).hexdigest()

def clean(q):
 if isinstance(q,dict):return {str(k):clean(v) for k,v in q.items()}
 if isinstance(q,(list,tuple,np.ndarray)):return [clean(v) for v in q]
 if isinstance(q,(np.integer,)):return int(q)
 if isinstance(q,(bool,np.bool_)):return bool(q)
 if isinstance(q,(float,np.floating)):return float(q) if np.isfinite(q) else None
 return q

def replace(tmp,path):
 for n in range(20):
  try:tmp.replace(path);return
  except PermissionError:
   if n==19:raise
   time.sleep(.05)

def write_json(path,q):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(clean(q),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8');replace(tmp,path)

def save(path,q):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
 with tmp.open('wb') as f:pickle.dump(q,f,protocol=5)
 replace(tmp,path);write_json(path.with_suffix('.sha256.json'),dict(sha256=sha(path)))

def read(path):
 path=Path(path);assert sha(path)==json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
 with path.open('rb') as f:return pickle.load(f)

def patient_weights(ids):
 return np.array([1/np.sum(ids==p) for p in ids],dtype=float)

def metric(y,p,tau):
 # Original y=0 is EZ; probe probability is already EZ-positive.
 from sklearn.metrics import roc_auc_score,average_precision_score
 z=np.asarray(y)==0;b=np.asarray(p)>=tau
 tp=int((z&b).sum());fp=int((~z&b).sum());tn=int((~z&~b).sum());fn=int((z&~b).sum())
 ef=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;nf=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
 se=tp/(tp+fn) if tp+fn else np.nan;sp=tn/(tn+fp) if tn+fp else np.nan
 order=np.argsort(-np.asarray(p),kind='stable');pos=np.flatnonzero(z[order])
 return dict(ez_auroc=roc_auc_score(z,p) if z.any() and (~z).any() else np.nan,ez_auprc=average_precision_score(z,p) if z.any() else np.nan,
  macro_f1=(ef+nf)/2,ez_f1=ef,nez_f1=nf,balanced_accuracy=float(np.nanmean([se,sp])),sensitivity=se,specificity=sp,ez_mrr=1/(pos[0]+1) if len(pos) else 0.,top1_is_ez=float(z[order[0]]),accuracy=float((z==b).mean()),predicted_ez_fraction=float(b.mean()),TP=tp,FP=fp,TN=tn,FN=fn)
