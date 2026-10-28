"""Trusted-server paths and immutable input binding; no private output export."""
import os,sys,json,hashlib,pickle,importlib.util,gc,random
from pathlib import Path
import numpy as np
import scipy,sklearn
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('PR2_RUNTIME',r'D:\Codex_Research\experiments\drechan_pr2_quick_ictal_v1_runtime'))
PUBLIC=ROOT/'public';KEEP=np.array([0,1,2,3,4,5,6,7,10,11,12,13])
PARENT=Path(r'D:\drechan_f0_complete_ablation_3seed_v1')
SVMROOT=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1_runtime')
_context=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def load(path):
    with Path(path).open('rb') as f:return pickle.load(f)
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,protocol=5)
    os.replace(tmp,path)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str),encoding='utf-8');os.replace(tmp,path)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(value,tmp);os.replace(tmp,path)
def source_hashes():
    # Reporting/packaging presentation may be added after fitting; it cannot alter
    # this complete, explicit numerical/source contract or any frozen endpoint.
    files=['PROTOCOL_LOCK.json','reference.py','pr2.py','losses.py','svm_adapter.py','drechan_adapter.py',
           'runtime.py','metrics.py','training.py','admit.py','train_svm_plugin.py','train_drechan_joint.py',
           'evaluate.py','verify.py','run_all.ps1','tests/test_pr2.py']
    return {name:sha(HERE/name) for name in files}
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
def context():
    global _context
    if _context is None:
        ROOT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(exist_ok=True)
        import psutil
        psutil.Process().cpu_affinity(list(range(16,32)));torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        spec=importlib.util.spec_from_file_location('pr2_frozen_parent',PARENT/'common.py')
        parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
        binding,core,fac,records,rec,banks=parent.setup(False)
        _context=dict(parent=parent,binding=binding,core=core,fac=fac,records=records,rec=rec,banks=banks,inputs=parent.store())
    return _context
def fold_bank(fold):
    c=context();d=c['parent'].bank(fold,c['banks'],'FULL')
    assert set(d['patient'][d['train']])==d['roles']['fit']
    assert set(d['patient'][d['val']])==d['roles']['val'] if 'val' in d['roles'] else set(d['patient'][d['val']]).isdisjoint(d['roles']['fit']|d['roles']['test'])
    return d
def subset(data,indices):return {k:np.asarray(data[k])[indices] for k in ['patient','channel','source','y','raw12','archive_indices'] if k in data}
def bank_data(fold,outer=False,labels=False):
    c=context();d=fold_bank(fold);rec=c['rec'];meta=rec['meta']
    if outer:
        ix=np.flatnonzero(np.isin(meta['patient'],list(d['roles']['test'])))
        result=dict(patient=np.asarray(meta['patient'])[ix],channel=np.asarray(meta['channel'])[ix],
                    source=np.asarray(meta.get('center',meta.get('source')))[ix],archive_indices=ix,
                    raw12=np.asarray(rec['R14'][ix][:,KEEP],np.float32))
        result['x']=c['parent'].query_features(rec['R14'][ix],result['patient'],d['pre'],'FULL',c['core'])
        if labels:
            assert (ROOT/'B_FREEZE.json').exists() or (ROOT/'A_FREEZE.json').exists()
            archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
            assert sha(archive)==rec['binding']['archive']
            with np.load(archive,allow_pickle=False) as q:result['y']=(1-q['y'][ix]).astype(int)
        return result
    return dict(patient=d['patient'],channel=d['channel'],source=np.asarray(meta.get('center',meta.get('source')))[d['archive_indices']],
                archive_indices=d['archive_indices'],raw12=np.asarray(rec['R14'][d['archive_indices']][:,KEEP],np.float32),
                x=d['x'],y=(d['y']==0).astype(int),train=d['train'],val=d['val'])
def base_model(fold):
    c=context();return c['parent'].make(c['core'],c['fac'],c['records'],fold,'FULL',42)
def raw_arguments(data,ii,fold):
    c=context();pid=str(data['patient'][ii[0]]);channels=data['channel'][ii].astype(str).tolist()
    raw,mask,times,delta,_=c['inputs'].get(pid,channels,'cuda')
    return (torch.as_tensor(data['x'][ii],device='cuda'),raw,mask,times,delta,c['parent'].scale(fold),
            torch.as_tensor(data['raw12'][ii],device='cuda'))
def own_freeze(stage,cells):
    value=dict(source=source_hashes(),input=read_json(PUBLIC/'ADMISSION.json')['input'],cells=cells,
               source_commit=os.environ['PR2_SOURCE_COMMIT'])
    path=ROOT/(stage+'_FREEZE.json')
    if path.exists():assert read_json(path)==value
    else:write(path,value)
    return value
def check_admission():
    a=read_json(PUBLIC/'ADMISSION.json');assert a['status']=='PASS' and a['source']==source_hashes()
    assert len(os.environ.get('PR2_SOURCE_COMMIT',''))==40
    return a
