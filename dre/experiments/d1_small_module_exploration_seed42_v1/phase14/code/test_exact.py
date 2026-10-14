import numpy as np
from sklearn.metrics import f1_score
from exact import select_exact

def test_exact_plateau_against_bruteforce():
    y=np.array([0,0,1,1,0,1,1,1]);p=np.array([.13,.27,.25,.82,.24,.37,.28,.69]);patient=np.array(['a']*4+['b']*4);q=select_exact(y,p,patient);scores=[]
    for tau in np.linspace(0,1,10001):scores.append(np.mean([f1_score(y[patient==pid],p[patient==pid]>=tau,labels=[0,1],average='macro',zero_division=0) for pid in ['a','b']]))
    assert np.isclose(q['macro_f1'],max(scores))

def test_constant_and_single_class_finite():
    q=select_exact(np.array([1,1,1]),np.array([.3,.3,.3]),np.array(['a']*3));assert q['macro_f1']==.5 and np.isfinite(q['threshold'])
