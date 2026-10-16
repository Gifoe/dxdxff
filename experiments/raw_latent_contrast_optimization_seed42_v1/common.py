"""Raw/identity-only access. No dependency on physiology model/data modules."""
import hashlib
import importlib.util
import json
import pickle
import random
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_latent_contrast_optimization_seed42_v1_runtime')
PUB=ROOT/'public'
BASE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
OLD=Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_runtime')
EXPECTED_REC='73095eefbb3035200894560de3c3cfdb6259a4f28c343ecd3d7195be5b94f609'
EXPECTED_REG='48fb162c51ae34c3e29e4803a87577535a0f9e01ea09276acc0da8c8ef4c40f6'
EXPECTED_ENCODERS='2b81e6460a1fcf64a550eb90224bf7205024d022b066269abea8cda175456cba'
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','metrics.py','stage1.py']
METRICS=['auroc','auprc','f1','macro_f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def jr(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,(int,np.integer)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(clean(x),sort_keys=True,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def freeze(path,x):
    path=Path(path)
    if path.exists():assert jr(path)==clean(x), f'Frozen state changed: {path.name}'
    else:write(path,x)


def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f:pickle.dump(x,f,protocol=5)
    temp.replace(path);write(path.with_suffix('.sha256.json'),dict(sha256=sha(path)))


def read(path):
    path=Path(path);assert sha(path)==jr(path.with_suffix('.sha256.json'))['sha256'],path.name
    with path.open('rb') as f:return pickle.load(f)


def atomic_torch(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
    torch.save(x,temp);temp.replace(path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    assert sha(BASE/'RECONSTRUCTION_PRIVATE.pkl')==EXPECTED_REC
    assert sha(BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')==EXPECTED_REG
    assert sha(OLD/'ENCODER_REGISTRY_PRIVATE.pkl')==EXPECTED_ENCODERS
    audit=jr(BASE/'results/RAW_DYNAMIC_ALIGNMENT.json')
    assert audit['status']=='PASS' and audit['valid_channel_windows']==1471965 and audit['registry_sha256']==EXPECTED_REG
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},raw_registry=EXPECTED_REG,
                 identity_reconstruction=EXPECTED_REC,encoders=EXPECTED_ENCODERS,
                 torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'STAGE1_INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return binding


def identity_and_banks(labels=False):
    rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');meta=rec['meta']
    # No feature array is returned or used. Legacy bundles mix identity and measurements.
    result={k:np.asarray(meta[k]).copy() for k in ['patient','channel','center','split_patient','split_role','split_fold']}
    assert len(result['patient'])==7635 and len(set(result['patient']))==80
    assert len(set(zip(result['patient'],result['channel'])))==7635
    assert sum(len(q['records']) for q in rec['ledgers'])==256
    del rec
    amendment=Path(r'E:\DRE-nips\new-pipeline\7-11\d2_ictal_outer_seed42_v1\PROTOCOL_AMENDMENT.json')
    assert sha(amendment)=='452a4f6957d18125a9ea1b32d827c5f50fbc81c22d9d41100173bddcd2e22503'
    hashes=jr(amendment)['bank_sha256'];banks=[]
    lookup={tuple(v):i for i,v in enumerate(zip(result['patient'].astype(str),result['channel'].astype(str)))}
    for fold in range(1,6):
        path=Path(r'C:\a0_source_posterior_decoder_seed42_runtime')/f'fold{fold}/BANK_PRIVATE.pt'
        assert sha(path)==hashes[fold-1]
        old=torch.load(path,map_location='cpu',weights_only=False)
        d={k:np.asarray(old[k]).copy() for k in ['patient','channel','train','val']}
        if labels:d['y']=np.asarray(old['y']).copy()
        del old
        roles={r:set(result['split_patient'][(result['split_fold']==fold)&(result['split_role']==r)]) for r in ['fit','validation','test']}
        assert all(roles[a].isdisjoint(roles[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
        assert set(d['patient'][d['train']])==roles['fit'] and set(d['patient'][d['val']])==roles['validation']
        assert set(d['patient']).isdisjoint(roles['test'])
        assert len(roles['fit'])==[51,51,50,52,51][fold-1]
        d['archive_indices']=np.array([lookup[(str(p),str(c))] for p,c in zip(d['patient'],d['channel'])])
        d['center']=result['center'][d['archive_indices']];d['roles']=roles;banks.append(d)
    return result,banks


class RawStore:
    def __init__(self):
        self.plans=read(BASE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')['plans'];self.cache=OrderedDict()

    def get(self,pid,channels,device='cuda'):
        pid=str(pid);plan=self.plans[pid]
        if pid not in self.cache:
            assert sha(plan['path'])==plan['sha256']
            b=torch.load(plan['path'],map_location='cpu',weights_only=False)
            assert b['patient']==pid and b['channels']==plan['channels']
            self.cache[pid]=b
            while len(self.cache)>2:self.cache.popitem(last=False)
        else:self.cache.move_to_end(pid)
        b=self.cache[pid];channels=list(map(str,channels))
        assert len(channels)==len(set(channels))==len(b['channels']) and set(channels)==set(b['channels'])
        ix=torch.tensor([b['channels'].index(v) for v in channels])
        raw=b['raw'][0].index_select(1,ix).to(device)
        mask=b['mask'][0].index_select(1,ix).to(device)
        times=torch.tensor(plan['times'],dtype=torch.float32,device=device)
        assert raw.shape[-1]==500 and mask.shape==raw.shape[:-1] and torch.isfinite(raw).all()
        return raw,mask,times


def private_key(pid):return hashlib.sha256(str(pid).encode()).hexdigest()[:24]
