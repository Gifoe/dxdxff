import numpy as np
from neuroez_c.task2.mosaic.expert_preprocessing import ExpertPreprocessor

def test_pca_fit_partition_is_recorded():
    x=np.arange(24,dtype=float).reshape(6,4); prep=ExpertPreprocessor(use_pca=True).fit(x,[f"p{i}" for i in range(6)])
    assert prep.fit_patient_keys==tuple(f"p{i}" for i in range(6)) and prep.pca.n_components_<=6
