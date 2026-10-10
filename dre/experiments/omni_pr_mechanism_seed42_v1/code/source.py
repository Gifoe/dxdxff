"""Pinned historical dependencies; read-only bank and FIT-only clinical API."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
CODE=Path(__file__).resolve().parent
PROTOCOL=CODE.parent/'PROTOCOL_LOCK.json'
ROOT=Path(r'C:\omni_pr_mechanism_seed42_runtime')
HIST=Path(r'C:\omni_d2_patient_source_seed42_runtime')
HCODE=Path(r'E:\DRE-nips\new-pipeline\7-11\omni_d2_patient_source_seed42_v1\code')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def cfg():return json.loads(PROTOCOL.read_text())
assert sha(HCODE.parent/'PROTOCOL_LOCK.json')==cfg()['historical_protocol_sha256']
assert sha(HIST/'FROZEN_BEFORE_HEAD_OUTER_PRIVATE.json')==cfg()['historical_freeze_sha256']
FREEZE=json.loads((HIST/'FROZEN_BEFORE_HEAD_OUTER_PRIVATE.json').read_text())
for name,h in FREEZE['code'].items():assert sha(HCODE/name)==h,'Historical executable changed'
sys.path.insert(0,str(HCODE))
import common as historical
import head_data as historical_data
from model import Head
from train_heads import train as authoritative_train
SOURCES=historical.SOURCES;RANK=historical.RANK;CLASS=historical.CLASS;METRICS=historical.METRICS
key=historical.key;digest=historical.digest;seeded=historical.seeded
sigmoid=historical.sigmoid;metric=historical.metric;classification=historical.classification
threshold=historical.threshold;normalized=historical.normalized;manifest=historical.manifest
write_json=historical.write_json;save_npz=historical.save_npz
state_hash=historical_data.state_hash;save_torch=historical_data.save_torch;plan=historical_data.plan
def code_hashes():return {p.name:sha(p) for p in CODE.glob('*.py')}
def csv(name,rows):pd.DataFrame(rows).to_csv(ROOT/'public'/name,index=False)
def selected(fold,method):
    arm='C1' if method=='C1-PR' else 'C0'
    return next(r for r in FREEZE['selections'] if r['fold']==fold and r['method']==arm)
def pr_head(fold):
    path=HIST/f'fold_{fold}/C1/SELECTED_HEAD_PRIVATE.pt';s=selected(fold,'C1-PR')
    assert sha(path)==s['checkpoint_sha256'];v=torch.load(path,map_location='cpu',weights_only=False)
    assert v['threshold']==s['threshold'] and v['selected_epoch']==s['selected_epoch']
    head=Head('C1');head.load_state_dict(v['model']);head.eval();return head
def bank(fold,patients,fit_labels=False):
    mf=manifest()
    if fit_labels:assert set(patients)<=set(mf.loc[mf.fold!=fold,'patient_name']), 'Outer-held labels forbidden'
    gate=HIST/'BANK_GATE_PRIVATE.json';assert sha(gate)==cfg()['historical_bank_gate_sha256']
    hashes=json.loads(gate.read_text())['banks'];result={}
    for p in patients:
        path=HIST/f'embeddings/fold_{fold}/{key(p)}.npz';assert sha(path)==hashes[str(path)]
        with np.load(path,allow_pickle=False) as d:
            assert str(d['checkpoint_sha256'].item())==historical.lock()['backbone_checkpoints'][str(fold)]
            item={'channels':d['channels'].astype(str),'h':d['embedding'].copy(),'base':d['logits'].copy(),'bank_sha':sha(path)}
        assert item['h'].shape==(len(item['base']),32) and np.isfinite(item['h']).all()
        if fit_labels:
            with np.load(HIST/'labels'/f'{key(p)}.npz',allow_pickle=False) as v:
                assert np.array_equal(v['channels'],item['channels']);item['y']=v['labels'].copy()
        result[p]=item
    return result
def predict(head,data,patients,source):
    return historical_data.predict(head,data,patients,source)
def aggregate(frame,keys,columns):
    result=[]
    for k,g in frame.groupby(keys,sort=True):
        if not isinstance(k,tuple):k=(k,)
        result.append({**dict(zip(keys,k)),'patients':len(g),**{m:float(g[m].mean()) for m in columns}})
    return result
