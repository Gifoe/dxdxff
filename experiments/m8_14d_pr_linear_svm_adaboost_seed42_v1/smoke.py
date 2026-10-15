"""Synthetic-only API/calibration admission; no cohort or Outer artifacts read."""
import numpy as np
import run

rng=np.random.default_rng(42)
pid=np.repeat(np.arange(12),10)
x=rng.normal(size=(120,14))
y=(x[:,0]+.2*rng.normal(size=120)>0).astype(int)
tr=np.flatnonzero(pid<9)
for model,config in [('Linear_SVM',dict(C=1)),('AdaBoost',dict(n_estimators=100,learning_rate=.05))]:
    b=run.fit(model,config,x,y,pid,tr,1051)
    p,d=run.predict(b,x)
    assert p.shape==(120,) and np.isfinite(d).all()
    assert all(t['patient_exclusion'] for t in b['calibration_audit'])
    assert (len(b['calibration_audit'])==3)==(model=='Linear_SVM')
    print('SYNTHETIC_SMOKE_PASS',model,b['iterations'],flush=True)
