"""Trusted inputs remain server-side; immutable parent assignment and new sources."""
import importlib.util
import hashlib
import json
import os
import pickle
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'drechan_pr_statistical_mechanism_v1'
SCIENCE=['PROTOCOL_LOCK.json','PROTOCOL_LOCK.md','math_core.py','contract.py','pipeline.py','stats_report.py','independent.py','tests/test_reference.py']
PAYLOADS=['BACKGROUND.csv','CONTRASTS.csv','ANOMALY.csv','SUPPORT.csv','PERMUTATION.csv','PERMUTATION_SUMMARY.csv','COVERAGE.json','ADMISSION.json','FREEZE.json','SEAL.json','INDEPENDENT_VERIFICATION.json','STATUS.json']

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

parent=module('reference_transfer_parent_io',PARENT/'trusted_io.py')
geometry=module('reference_transfer_parent_geometry',PARENT/'mechanism_math.py')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def sources():
    return {**{n:sha(HERE/n) for n in SCIENCE},**{'parent/'+n:sha(PARENT/n) for n in ['trusted_io.py','mechanism_math.py']}}
def clean(x):return parent.clean(x)
def digest(x):return parent.digest(x)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):return pickle.loads(Path(p).read_bytes())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p.name
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p.name
    t=p.with_suffix(p.suffix+'.partial');t.write_bytes(pickle.dumps(x,protocol=5));t.replace(p)
def inputs():
    data,binding=parent.load_inputs()
    assert binding['new_assignment_sha256']=='878b0a83431b99390e47e46be7069e7aaeec6853965f1c45e57f4e98ed8f1350'
    rec=parent.read(parent.path('reconstruction'))
    assert sum(len(p['records'])==1 for p in rec['ledgers'])==7
    for d in data:d['channel']=rec['meta']['channel'][d['archive_indices']]
    binding['single_seizure_patients']=7
    del rec
    return data,binding
def checked(root):
    a=read(Path(root)/'private/LOCK.json');assert a['sources']==sources(),'Frozen source changed'
    data,binding=inputs();assert digest(binding)==a['binding_sha256'],'Input/assignment changed'
    return data,binding
def commit():
    import re
    value=os.environ.get('PRREF_SOURCE_COMMIT','');assert re.fullmatch('[a-f0-9]{40}',value)
    return value
