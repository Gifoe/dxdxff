"""Synthetic-only frozen preprocessing, label orientation and80-ID paired bootstrap."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from verify import independent
from bootstrap import independently_verify


def main():
 y=np.array([0,0,1,1]);score=np.array([.1,.4,.3,.9]);r=independent(y,score,.35)
 assert r['TP']==r['FP']==r['TN']==r['FN']==1 and r['macro_f1']==.5 and r['ez_auroc']==.75
 rows=[dict(fold=i%5,patient='synthetic'+str(i),value=float(i)/100) for i in range(80)]
 left=pd.DataFrame(rows);right=left.copy();right['value']=0.
 calc=independently_verify(left,right,['value'])['value']
 order=left.set_index(['fold','patient']).sort_index();values=order.value.to_numpy();pid=np.asarray(order.index.get_level_values('patient'));ids=np.unique(pid)
 draws=np.random.default_rng(42).integers(0,80,size=(10000,80));data=np.array([values[pid==p][0] for p in ids]);direct=data[draws].mean(1)
 assert np.allclose(calc,[values.mean(),*np.percentile(direct,[2.5,97.5])],atol=1e-12,rtol=0)
 print('PURN_OUTER_SYNTHETIC_CONTRACT_PASS',flush=True)


if __name__=='__main__':main()
