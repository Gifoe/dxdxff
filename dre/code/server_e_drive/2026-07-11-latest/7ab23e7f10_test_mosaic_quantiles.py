import numpy as np
from neuroez_c.task2.mosaic.quantile_features import distribution_quantiles,phase_region_quantiles,pool_named_rows

def test_quantile_count_and_values():
    q=distribution_quantiles(np.arange(1,11)); assert len(q)==10 and q["q100"]==10
    times=np.array([-5,1,6]); x=np.tile(np.arange(4),(3,1)); f=phase_region_quantiles(x,times,np.array([1,1,0,0],bool))
    assert len(f)==60

def test_pool_ignores_audit_identifiers():
    pooled=pool_named_rows([{"patient_key":"p1","seizure_id":"s1","score":1.}, {"patient_key":"p1","seizure_id":"s2","score":3.}])
    assert "patient_key__mean" not in pooled and pooled["score__mean"]==2.
