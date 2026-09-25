import numpy as np
import pandas as pd
import pytest

from neuroez_c.task2.ngbr.outcome_model import RIDGE_C,fit_outcome_model
from neuroez_c.task2.ngbr.preprocessing import NGBRPreprocessor,hfo_fold_eligible
from neuroez_c.task2.protocol import assert_checkpoint_safe


def test_p2_manifest_overlap_fails():
    with pytest.raises(ValueError):assert_checkpoint_safe({"p1","p2"},{"p2","p3"})


def test_preprocessor_uses_train_medians_and_fixed_ridge():
    train=pd.DataFrame({"x":[0.,1.,np.nan,2.],"outcome_true":[0,1,0,1],"n_hfo_valid_seizures":[0,0,0,0]});pre=NGBRPreprocessor.fit(train,["x"],1)
    assert pre.medians["x"]==1
    fitted=fit_outcome_model(train,train.outcome_true,["x"],outer_fold=1,seed=42,hfo_enabled=False)
    assert fitted.model.C==RIDGE_C


def test_hfo_gate_uses_availability_counts_only():
    frame=pd.DataFrame({"n_hfo_valid_seizures":[1]*30,"outcome_true":[0]*10+[1]*20})
    assert hfo_fold_eligible(frame)[0]
