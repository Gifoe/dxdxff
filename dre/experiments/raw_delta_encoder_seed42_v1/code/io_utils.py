import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import sys
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(np.ndarray,list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.bool_,bool)):return bool(x)
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 return x

def write(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(clean(x),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8');tmp.replace(path)

def save(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
 with tmp.open('wb') as f:pickle.dump(x,f,protocol=5)
 tmp.replace(path);write(path.with_suffix('.sha256.json'),dict(sha256=sha(path)))

def read(path):
 path=Path(path);assert sha(path)==json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
 with path.open('rb') as f:return pickle.load(f)

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
