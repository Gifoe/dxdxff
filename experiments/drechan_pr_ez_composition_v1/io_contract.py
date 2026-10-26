"""Frozen parent reuse. All individual inputs/outputs stay in trusted runtime."""
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PRREF=HERE.parent/'drechan_pr_reference_transfer_v1'
SCIENCE=['PROTOCOL_LOCK.json','PROTOCOL_LOCK.md','core.py','io_contract.py','pipeline.py','stats_report.py','independent.py','tests/test_composition.py','tests/test_end_to_end.py']
TABLES=['E1_CLASS_LOSS','E1_GAIN','E1_H','E2_LOSS','E2_GAIN','E2_H','E2_ANOMALY','E2_SUPPORT','REFERENCE_COMPOSITION','COMPOSITION_SLOPE','PRIMARY','SECONDARY','H_PERMUTATION','H_PERMUTATION_SUMMARY']
JSONS=['COVERAGE','ADMISSION','FREEZE','SEAL','INDEPENDENT_VERIFICATION','STATUS']
PAYLOADS=[n+'.csv' for n in TABLES]+[n+'.json' for n in JSONS]+['LEGACY_ANOMALY.csv']

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
parent=module('composition_parent_contract',PRREF/'contract.py')
def sha(p):return parent.sha(p)
def clean(x):return parent.clean(x)
def digest(x):return parent.digest(x)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):return pickle.loads(Path(p).read_bytes())
def write(p,x):return parent.write(p,x)
def save(p,x):return parent.save(p,x)
def protocol():return read(HERE/'PROTOCOL_LOCK.json')
def sources():
    return {**{n:sha(HERE/n) for n in SCIENCE},**{'reference_parent/'+k:v for k,v in parent.sources().items()}}
def commit():
    import re
    s=os.environ.get('PRCOMP_SOURCE_COMMIT','');assert re.fullmatch('[a-f0-9]{40}',s);return s
def parent_root():return Path(os.environ['PRCOMP_PARENT_RUNTIME'])

def inputs():
    p=parent_root();lock=protocol();f=read(p/'public/FREEZE.json');s=read(p/'public/SEAL.json')
    assert read(p/'public/STATUS.json')['status']=='COMPLETE_VERIFIED_PR_REFERENCE_TRANSFER_80'
    assert f['sources']==s['sources']==parent.sources(),'Parent source parity failed'
    assert f['references_sha256']==lock['parent_reference_sha256']==sha(p/'private/REFERENCES.pkl')
    assert s['sha256']==lock['parent_endpoint_sha256']==sha(p/'private/ENDPOINTS.pkl')
    refs=load(p/'private/REFERENCES.pkl');old=load(p/'private/ENDPOINTS.pkl');data,binding=parent.inputs()
    assert len(refs)==len(old)==80 and sum(len(q['x']) for q in refs)==7635
    labels={}
    for d in data:
        for patient in d['roles']['assessment']:
            ix=np.flatnonzero(d['patient']==patient);labels[patient]=d['y'][ix]
            ref=next(q for q in refs if q['patient']==patient)
            assert ref['fold']==d['fold'] and np.array_equal(ref['archive_indices'],d['archive_indices'][ix])
            assert np.array_equal(ref['channels'],d['channel'][ix])
    for a,b in zip(refs,old):assert a['patient']==b['patient']
    rec=parent.parent.read(parent.parent.path('reconstruction'))
    anatomy_keys=sorted(set(rec['meta'])&{'xyz','coordinates','anatomy','region','regions','anatomical_region','electrode_positions','channel_positions'})
    del rec
    binding.update(parent_reference_sha256=f['references_sha256'],parent_endpoint_sha256=s['sha256'],existing_anatomical_metadata_fields=anatomy_keys)
    return refs,old,labels,binding

def checked(root):
    lock=read(root/'private/LOCK.json');assert lock['sources']==sources(),'Frozen science edited'
    a,b,y,binding=inputs();assert digest(binding)==lock['binding_sha256']
    return a,b,y,binding

def frozen(root):
    checked(root);summary=read(root/'public/FREEZE.json');manifest=read(root/'private/REFERENCE_MANIFEST.json')
    assert digest(manifest)==summary['reference_manifest_sha256']
    for n,h in manifest['files'].items():assert sha(root/'private'/n)==h
    return manifest
