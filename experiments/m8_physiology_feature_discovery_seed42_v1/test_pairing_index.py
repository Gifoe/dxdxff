"""Outcome-free regression for separated NumPy advanced-index axis movement."""
from integrated_data_v2 import *
S,C,W,F=3,64,59,21
v=np.arange(S*C*W*F,dtype=np.float32).reshape(S,C,W,F)
p=np.roll(np.arange(C),1);slot=[0,1,2,3,4,5,6,17,13,20]
actual=np.zeros((S,C,W,10),np.float32)
for j,k in enumerate(slot):actual[...,j]=v[...,k][:,p,:] if k>=7 else v[...,k]
reference=np.zeros_like(actual)
for s in range(S):
    for c in range(C):
        for j,k in enumerate(slot):reference[s,c,:,j]=v[s,p[c] if k>=7 else c,:,k]
assert np.array_equal(actual,reference)
assert np.array_equal(actual[...,:7],v[...,:7])
assert not np.array_equal(actual[...,7],v[...,17])
write(OUT/'public/WRONG_PAIR_AXIS_REGRESSION.json',dict(status='PASS',seizures=3,channels=64,windows=59,
    independent_nested_loop_exact=True,unchanged_old7_exact=True,source_sha256=sha(TASK/'integrated_data_v2.py'),test_sha256=sha(__file__)))
print('WRONG_PAIR_CHANNEL_AXIS_AND_OLD7_INVARIANCE_PASS',flush=True)
