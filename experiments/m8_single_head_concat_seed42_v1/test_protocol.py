import json
from pathlib import Path
import numpy as np
import pandas as pd
from selection import *

def test_selector_and_outer_rejection():
    protocol=json.loads((Path(__file__).parent/'PROTOCOL_LOCK.json').read_text());rows=[]
    for arm in ['S0','S1','S2','S3','S4']:
        for fold in range(5):
            for k in range(13):rows.append(dict(scope='development',model=arm,fold=fold+1,patient=str((fold*13+k)%47),**{m:.6 for m in METRICS}))
    frame=pd.DataFrame(rows);selected,_=choose(frame,protocol);assert selected=='S2'
    frame.loc[frame.model=='S3',METRICS]=.7;selected,_=choose(frame,protocol);assert selected=='S3'
    frame['scope']='outer'
    try:choose(frame,protocol)
    except AssertionError:pass
    else:raise AssertionError('No Outer selection')

def test_strict8_no_rounding_or_average_rescue():
    reference={m:.6 for m in STRICT};candidate={m:.7 for m in STRICT}
    rows,ok,primary=check_target(candidate,reference);assert ok and primary
    candidate['hit_at_1']=.6-1e-12;rows,ok,primary=check_target(candidate,reference)
    assert not ok and primary and len([r for r in rows if not r['passes']])==1
    candidate['auroc']=.5;assert not check_target(candidate,reference)[2]

def test_patient_cluster_bootstrap():
    ids=np.array(['a','a','b']);unique,mult,draw=counts(ids)
    a=np.array([[1.,2.],[3.,np.nan],[5.,8.]]);mean,ci,n,dist=paired(a,np.zeros_like(a),ids,unique,mult)
    assert np.allclose(mean,[3,5]) and list(n)==[3,2]
    for dd,expected in zip(draw[:32],dist[:32]):
        ix=np.concatenate([np.flatnonzero(ids==unique[j]) for j in dd]);assert np.allclose(np.nanmean(a[ix],axis=0),expected)
