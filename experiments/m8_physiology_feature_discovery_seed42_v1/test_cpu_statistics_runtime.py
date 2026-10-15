"""Outcome-free classification/ranking/bootstrap parity between reporting runtimes."""
import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
sys.path.insert(0,r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code')
from io_utils import load,save,read,write,sha
OUT=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
EV=load('synthetic_cpu_metrics',Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))
parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args()
rng=np.random.default_rng(42);rows=[]
for n in [1,2,7,64,127,256,512]:
    for kind in ['random','ties','equal','all_nez_prediction','all_ez_prediction']:
        y=rng.integers(0,2,n);p=rng.random(n)
        if kind=='ties':p=np.round(p,1)
        if kind=='equal':p[:]=.5
        pred=p>=.5
        if kind=='all_nez_prediction':pred[:]=False
        if kind=='all_ez_prediction':pred[:]=True
        channel=np.asarray([f'SYNTHETIC_{i:04d}' for i in rng.permutation(n)])
        rows.append(dict(n=n,kind=kind,**EV.classification(y,p,pred),**EV.ranking(y,p,channel)))
draw=rng.integers(0,80,size=(10000,80));x=rng.normal(size=(80,5));dist=x[draw].mean(1)
record=dict(rows=rows,bootstrap_quantiles=np.quantile(dist,[.025,.975],axis=0),python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__)
path=OUT/('SYNTHETIC_STATS_'+a.tag+'_PRIVATE.pkl');assert not path.exists();save(path,record)
reference=OUT/'SYNTHETIC_STATS_py313_PRIVATE.pkl'
if a.tag!='py313':
    other=read(reference);maximum=0.
    for left,right in zip(rows,other['rows']):
        assert left.keys()==right.keys()
        for key in left:
            if key=='kind':assert left[key]==right[key];continue
            assert np.isclose(left[key],right[key],atol=1e-12,rtol=0,equal_nan=True),(key,left[key],right[key])
            if np.isfinite(left[key]):maximum=max(maximum,abs(float(left[key])-float(right[key])))
    assert np.allclose(record['bootstrap_quantiles'],other['bootstrap_quantiles'],atol=1e-12,rtol=0)
    write(OUT/'public/CPU_STATISTICS_RUNTIME_PARITY.json',dict(status='PASS',synthetic_only=True,cases=len(rows),
        all_classification_and_ranking_metrics=True,bootstrap_draws=10000,maximum_metric_difference=maximum,absolute_tolerance=1e-12,
        reference_versions={k:other[k] for k in ['python','numpy','pandas','sklearn']},new_versions={k:record[k] for k in ['python','numpy','pandas','sklearn']},
        no_clinical_labels_or_predictions_read=True,source_sha256=sha(__file__)))
print('SYNTHETIC_CPU_STATS_RUNTIME_PARITY_PASS',a.tag,len(rows),flush=True)
