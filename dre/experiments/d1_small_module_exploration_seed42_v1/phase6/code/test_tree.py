import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from run import FrozenM8Init,weights,prune_fit

def test_frozen_initializer_and_equal_patient_weights():
    x=np.array([[-2.,1.],[0.,1.],[1.,1.],[3.,1.]]);y=np.array([0,1,0,1]);m=FrozenM8Init().fit(x,y)
    assert np.allclose(m.predict_proba(x).sum(1),1)
    w=weights(np.array(['a','a','b','b']),y);assert np.isclose(w[:2].sum(),w[2:].sum())
    tree=GradientBoostingClassifier(init=FrozenM8Init(),n_estimators=3,max_depth=1,random_state=42);tree.fit(x,y,sample_weight=w)
    assert np.isfinite(tree.predict_proba(x)).all()

def test_fit_only_pruning_does_not_observe_validation():
    x=np.array([[1,2,2,5],[2,3,3,5],[3,4,9,6]],float);keep,removed=prune_fit(x,np.array([0,1]),['base','a','dup','constant'])
    assert keep.tolist()==[0,1] and len(removed)==2
