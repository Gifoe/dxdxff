import numpy as np
import pandas as pd
import pytest
from neuroez_c.task2.cop.preprocessing import COPPreprocessor
from neuroez_c.task2.protocol import ProtocolError,assert_checkpoint_safe


def test_preprocessor_statistics_are_outer_train_only():
    train=pd.DataFrame({"feature__x":[0.,1.,np.nan],"feature__y":[1.,2.,3.]});test=pd.DataFrame({"feature__x":[1000.],"feature__y":[1000.]});prep=COPPreprocessor.fit(train,list(train),1)
    assert prep.medians["feature__x"]==.5;prep.transform(test);assert prep.medians["feature__x"]==.5


def test_checkpoint_overlap_rejected():
    with pytest.raises(ProtocolError):assert_checkpoint_safe(["p1","p2"],["p2"])
