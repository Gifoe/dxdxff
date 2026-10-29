"""CPU-only adapter for read-only pinned parents; private trusted outputs."""
import os,sys,json,pickle,csv,hashlib
from pathlib import Path
SOURCE=Path(__file__).resolve().parent.parent
RUNTIME=Path(r'D:\Codex_Research\experiments\drechan_pr_ranking_linear_residual_abc_v1_runtime')
PUB=RUNTIME/'public';MAIN=Path(r'D:\Codex_Research\experiments\drechan_pr_representation_vs_score_v1_runtime')
FROZEN_SOURCE=Path(r'D:\Codex_Research\experiments\drechan_pr_representation_vs_score_v1')
PREVIEW=MAIN/'partial_outer_20261028_v2'
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
sys.dont_write_bytecode=True
sys.path.insert(0,str(FROZEN_SOURCE))
from path_adapter import install
install(r'D:\Codex_Research\_maintenance\cleanup_20261028\LEGACY_PATHS.json',[RUNTIME])
import numpy as np
import torch
assert not torch.cuda.is_available()
torch.nn.Module.cuda=lambda self,*a,**k:self.cpu()
torch.Tensor.cuda=lambda self,*a,**k:self.cpu()
_rand,_tensor=torch.rand,torch.tensor
def _cpu_rand(*a,**k):
    if k.get('device')=='cuda':k['device']='cpu'
    return _rand(*a,**k)
def _cpu_tensor(*a,**k):
    if k.get('device')=='cuda':k['device']='cpu'
    return _tensor(*a,**k)
torch.rand=_cpu_rand;torch.tensor=_cpu_tensor
from parents import Parents,RAW_ROOT
import task_io as frozen_io

def parents():
    p=Parents();torch.rand=_rand;torch.set_num_threads(1);return p

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):
    with Path(path).open('rb') as f:return pickle.load(f)

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:pickle.dump(value,f,protocol=5)
    tmp.replace(path)

def jr(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2),encoding='utf-8');tmp.replace(path)

def csv_out(name,rows,private=False):
    target=(RUNTIME/'private' if private else PUB)/(name+'.csv');target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+'.partial')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(target)

def source_hashes():return {str(p.relative_to(SOURCE)).replace('\\','/'):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.suffix in ['.py','.md','.json'] and not {'results','figures','tables','logs'}&set(p.relative_to(SOURCE).parts)}

def source_map(p):
    sizes={36:'hup',21:'lzu',15:'multicenter',8:'pediatric'};mapping={}
    for c in set(p.meta['center']):
        n=len(set(p.meta['patient'][p.meta['center']==c]));mapping[str(c)]=sizes[n]
    return {str(pid):mapping[str(p.meta['center'][p.meta['patient']==pid][0])] for pid in set(p.meta['patient'])}
