import csv
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('PR_REPR_RUNTIME',r'D:\Codex_Research\experiments\drechan_pr_representation_vs_score_v1_runtime'))
PUB=ROOT/'public'
MODELS=['LINEAR_SVM','TIMECONV','DRECHANNET']
ARMS=['NO_PR','GLOBAL','CENTER','SCALE','FULL_PR','SCORE_CENTER','SCORE_Z']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while True:
            b=f.read(1024*1024)
            if not b:break
            h.update(b)
    return h.hexdigest()
def digest(q):return hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def jr(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
    with Path(p).open('rb') as f:return pickle.load(f)
def write(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(q,sort_keys=True,indent=2),encoding='utf-8');t.replace(p)
def save(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.partial')
    with t.open('wb') as f:pickle.dump(q,f,protocol=5)
    t.replace(p)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sources():return {p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json','.ps1']}
def csv_out(name,rows):
    assert rows and all(list(q)==list(rows[0]) for q in rows)
    p=PUB/(name+'.csv');p.parent.mkdir(exist_ok=True);t=p.with_name(p.name+'.partial')
    with t.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    t.replace(p)
def key(model,arm,fold):return f'{model}_{arm}_fold{fold}'
def status(stage,**kw):write(PUB/'RUN_STATUS.json',dict(stage=stage,**kw))

