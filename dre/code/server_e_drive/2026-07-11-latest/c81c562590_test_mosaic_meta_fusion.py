import numpy as np, pandas as pd
from neuroez_c.task2.mosaic.meta_fusion import META_FEATURES,fit_meta

def test_meta_has_exactly_eight_inputs():
    frame=pd.DataFrame({"patient_key":[f"p{i}" for i in range(8)],"outcome_true":[0,1]*4,**{x:np.arange(8)+j for j,x in enumerate(META_FEATURES)}})
    model=fit_meta(frame); assert model.model.coef_.shape[1]==8 and set(META_FEATURES)==set(frame)-{"patient_key","outcome_true"}
