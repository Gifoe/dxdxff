"""Pinned v2 sources and private-only role/representation I/O for a new study."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\m8_rawconditional_feature_selection_seed42_v1_runtime')
PUB=ROOT/'public'
OLD=Path(r'D:\m8_physiology_discovery_selection_seed42_v2')
OLDROOT=Path(r'D:\m8_physiology_discovery_selection_seed42_v2_runtime')
sys.path.insert(0,r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code')
from io_utils import read,save,write,sha,load


def jr(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def freeze(p,value):
    if p.exists():assert jr(p)==value,p.name
    else:write(p,value)


def old_modules():
    audit=jr(OLDROOT/'public/EXECUTION_AUDIT.json')
    for name,value in audit['main_scientific_source'].items():assert sha(OLD/name)==value
    for name,value in audit['frozen_evaluation_sources'].items():assert sha(OLD/name)==value
    sys.path.insert(0,str(OLD))
    common=load('common',OLD/'common.py')
    load('model',OLD/'model.py')
    data=load('model_io',OLD/'model_io.py')
    trainer=load('conditional_v1_unchanged_raw_trainer',OLD/'train.py')
    support=load('conditional_v1_unchanged_pool_support',OLD/'support_control.py')
    primitives=load('conditional_v1_original_metric_primitives',OLD/'probes.py')
    trainer.PUB=PUB  # Execution output root only; never write v2 progress/state.
    return common,data,trainer,support,primitives


def bindings():
    names=['PROTOCOL_LOCK.json','conditional_io.py','conditional_ridge.py','prepare_encoders.py','select_conditional.py','admit.py','test_conditional_ridge.py']
    return dict(source={n:sha(HERE/n) for n in names},parent_protocol=sha(OLD/'PROTOCOL_LOCK.json'),
        parent_pool=sha(OLDROOT/'POOL_PRIVATE.pkl'),parent_registry=sha(OLDROOT/'POOL_REGISTRY_PRIVATE.pkl'),
        parent_Outer_seal_protection=sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'),
        original_role_files={f'{f}_{s}':sha(OLDROOT/'screens_private'/(f'fold{f}_screen{s}.pkl' if s else f'fold{f}_final_path.pkl')) for f in range(1,6) for s in range(4)})


def roles(fold,screen,d):
    q=read(OLDROOT/'screens_private'/(f'fold{fold}_screen{screen}.pkl' if screen else f'fold{fold}_final_path.pkl'))
    names=['train','cal','choice']+(['held'] if screen else [])
    sets=[set(d['patient'][q[n]]) for n in names]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i))
    assert set.union(*sets)==d['roles']['fit'] and set.union(*sets).isdisjoint(d['roles']['validation']|d['roles']['test'])
    return q
