from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'outer_code'))
from run import plan

def test_final_fit_selection_split():
 for fold,n in enumerate([51,51,50,52,51],1):
  ids=[f'synthetic_{j}' for j in range(n)];a=plan(ids,fold);b=plan(ids[::-1],fold)
  assert a==b and set(a['train']).isdisjoint(a['select'])
  assert set(a['train']+a['select'])==set(ids)
  assert len(a['select'])==int(np.ceil(.2*n))

def test_numpy2_batched_vector_ols_shape():
 a=np.stack([np.eye(7)]*5);b=np.arange(35.).reshape(5,7)
 result=np.linalg.solve(a,b[:,:,None])[...,0]
 assert np.array_equal(result,b)
