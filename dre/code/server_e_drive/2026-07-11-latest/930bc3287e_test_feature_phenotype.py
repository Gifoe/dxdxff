import numpy as np
from neuroez_c.task2.cop.feature_groups import match_feature_families
from neuroez_c.task2.cop.feature_phenotype import build_feature_phenotype
from neuroez_c.task2.cop.schema import FeatureRunRecord


def test_feature_family_balance_and_patient_dimension():
    names=["high_gamma","line_length_per_sec","rms","spectral_entropy"];groups,audit=match_feature_families(names);records=[]
    for seizure in range(2):
        centers=np.array([-5.,1.,5.,12.,20.]);values=np.random.default_rng(seizure).normal(size=(5,6,4)).astype("float32");target=np.array([1,1,0,0,0,0],dtype=bool)
        records.append(FeatureRunRecord("p","c",f"s{seizure}",[f"A{i}" for i in range(6)],centers-.5,centers+.5,values,names,np.ones((5,6),dtype=bool),target))
    table,definition,_=build_feature_phenotype(records,groups);features=[c for c in table if c.startswith("feature__")]
    assert len(features)==44 and len(definition)==44 and table.loc[0,"n_valid_seizures"]==2
