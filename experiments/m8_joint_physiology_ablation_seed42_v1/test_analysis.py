import numpy as np
from analysis import *

def test_masks():
    assert keep('111')==[1,1,0,1,0,1,0]
    assert sum(keep('110'))==5 and sum(keep('101'))==5 and sum(keep('011'))==5
    assert len({tuple(keep(c)) for c in CONFIGS})==8

def test_factorial_polynomial():
    # y=3+2V+4H+8E+5VH+7VE+11HE+13VHE
    y={c:3+2*int(c[0])+4*int(c[1])+8*int(c[2])+5*int(c[0])*int(c[1])+7*int(c[0])*int(c[2])+11*int(c[1])*int(c[2])+13*np.prod(list(map(int,c))) for c in CONFIGS}
    effect=lambda f,condition=None:sum(y[c]*w for c,w in factorial_weights(f,condition).items())
    assert effect('VH')==11.5 and effect('VHE')==13
    assert effect('VH',{'E':0})==5 and effect('VH',{'E':1})==18
    assert effect('V')==11.25
    assert all(sum(factorial_weights(f).values())==0 for f in ['V','H','E','VH','VE','HE','VHE'])

def test_bootstrap_cluster_and_missing():
    ids=np.array(['a','a','b']); unique,counts,draw=bootstrap_counts(ids,200)
    x=np.array([[1.,2.],[3.,np.nan],[5.,8.]])
    mean,ci,n,dist=contrast({'a':x,'b':x*0},{'a':1,'b':-1},ids,counts,unique)
    assert np.allclose(mean,[3.,5.]) and list(n)==[3,2]
    for i in range(200):
        rows=np.concatenate([np.flatnonzero(ids==unique[k]) for k in draw[i]])
        assert np.allclose(dist[i],np.nanmean(x[rows],axis=0))

def test_no_outer_selection():
    import pandas as pd
    try: candidate_decisions(pd.DataFrame({'scope':['outer']}),{})
    except AssertionError: pass
    else: raise AssertionError('Outer must never enter candidate selector')

def test_fixed_candidate_rule():
    import pandas as pd
    import json
    from pathlib import Path
    protocol=json.loads((Path(__file__).parent/'PROTOCOL_LOCK.json').read_text())
    rows=[]
    for head in ['D1','M8']:
        for config in CONFIGS:
            for fold in range(1,6):
                for i in range(4):
                    gain=.02 if config=='111' else 0
                    rows.append(dict(scope='development',head=head,config=config,fold=fold,patient=str(4*fold+i),center='source',
                        **{m:.6+gain for m in METRICS}))
    frame=pd.DataFrame(rows)
    _,choices=candidate_decisions(frame,protocol)
    assert choices=={'D1':'111','M8':'111'}
    frame.loc[frame.config!='000',METRICS]=.1
    _,choices=candidate_decisions(frame,protocol)
    assert choices=={'D1':'000','M8':'000'}
