"""Only the immutable Raw/identity reader and own-FIT role metadata are reused."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\raw_information_retention_internal_seed42_v1_runtime')
PUB=ROOT/'public'
PARENT=Path(r'D:\raw_latent_contrast_optimization_seed42_v1')
PARENT_RUNTIME=Path(r'D:\raw_latent_contrast_optimization_seed42_v1_runtime')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


io=module('retention_immutable_raw_reader',PARENT/'common.py')
assert io.sha(PARENT/'common.py')=='c5dd9fb82fab45dd3b1e4ae0911f642a221fbf5bd039a5a642f838a3a5ffa499'
assert io.sha(PARENT/'model.py')=='2a837032d9f2f2bc7a7b42c6d71c87863920a37ec5e35313c7ed1a6b54b20605'
assert io.sha(PARENT/'metrics.py')=='028a51ae271706c028aed60e55e02b2e06bf867a6d3ebd9b22d8758ede0388fa'
metrics=module('retention_immutable_metrics',PARENT/'metrics.py')
sha,read,save,write,freeze,jr,digest=io.sha,io.read,io.save,io.write,io.freeze,io.jr,io.digest
atomic_torch,seed_all,state_hash=io.atomic_torch,io.seed_all,io.state_hash
RawStore,private_key=io.RawStore,io.private_key
METRICS=io.METRICS
CRITICAL=['PROTOCOL_LOCK.json','common.py','model.py','engine.py','audit_inputs.py','train_internal.py','summarize.py']


def setup():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    protected={n:sha(PARENT_RUNTIME/n) for n in ['MODEL_FREEZE_PRIVATE.json','FROZEN_BEFORE_OUTER_PRIVATE.json','OUTER_SCORE_SEAL_PRIVATE.json']}
    assert protected['MODEL_FREEZE_PRIVATE.json']=='ef751fbbd66dcf825f1157bc5bf029039f489647033f90368958c0063af5cc9a'
    assert protected['FROZEN_BEFORE_OUTER_PRIVATE.json']=='92f8d0adc186cf79dde4a5f25171ad115ae36ddc7effd1127a38df7cc60fcbc5'
    assert protected['OUTER_SCORE_SEAL_PRIVATE.json']=='da2424a9bbc5e2314f3ba15d8b68d142d1e19e2beefcaa79bfe5160809da8b19'
    binding=dict(source={n:sha(HERE/n) for n in CRITICAL},protected_previous_study=protected,
        raw_registry=io.EXPECTED_REG,reconstruction=io.EXPECTED_REC,inner_roles=io.EXPECTED_ENCODERS,
        parent_source={n:sha(PARENT/n) for n in ['common.py','model.py','metrics.py']},
        torch=torch.__version__,numpy=np.__version__,sklearn=__import__('sklearn').__version__)
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return binding


def legal_fit_banks(labels):
    meta,old=io.identity_and_banks(labels=labels);banks=[]
    for original in old:
        ix=original['train']
        names=['patient','channel','center','archive_indices']+(['y'] if labels else [])
        d={k:original[k][ix].copy() for k in names}
        assert set(d['patient'])==original['roles']['fit']
        assert set(d['patient']).isdisjoint(original['roles']['validation']|original['roles']['test'])
        banks.append(d)
    return banks


def role_paths(banks):
    assert sha(io.OLD/'ENCODER_REGISTRY_PRIVATE.pkl')==io.EXPECTED_ENCODERS
    registry=read(io.OLD/'ENCODER_REGISTRY_PRIVATE.pkl');out={}
    for entry in registry['entries']:
        if entry['screen']==0:continue
        assert sha(entry['path'])==entry['sha256']
        old=read(entry['path']);d=banks[entry['fold']-1]
        assert np.array_equal(old['archive_indices'],d['archive_indices'])
        roles={k:np.asarray(old['roles'][k]).copy() for k in ['train','cal','choice','held']}
        sets=[set(d['patient'][ix]) for ix in roles.values()]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i))
        assert set.union(*sets)==set(d['patient'])
        out[(int(entry['fold']),int(entry['screen']))]=dict(roles=roles,role_sha256=entry['sha256'])
    assert len(out)==15;return out
