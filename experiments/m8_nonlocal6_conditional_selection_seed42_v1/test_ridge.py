import numpy as np
from sklearn.linear_model import Ridge
from ridge import fit,prepare,choose,subsets


def test_full_dense_solution_and_empty_intercept():
    rng=np.random.default_rng(42);x=rng.normal(size=(101,18));tr=np.arange(60);target=rng.choice([-1.,1.],101);w=rng.uniform(.1,2,60)
    z,_=prepare(x,tr);a=fit(z,6,(0,2,5),target,w,tr)
    direct=Ridge(alpha=1.,fit_intercept=True,solver='cholesky').fit(z[tr][:,a['columns']],target[tr],sample_weight=w*len(w)/w.sum())
    np.testing.assert_array_equal(a['score'],direct.predict(z[:,a['columns']]))
    empty=fit(z,0,(),target,w,tr);np.testing.assert_allclose(empty['score'],np.average(target[tr],weights=w))


def test_sparse_choice_ties_and_train_only_scaler():
    rows=[dict(k=2,groups=[0,1],choice_score=.71),dict(k=1,groups=[2],choice_score=.708),dict(k=1,groups=[1],choice_score=.708),dict(k=0,groups=[],choice_score=.70)]
    assert choose(rows)['groups']==[1] and len(subsets(list(range(6))))==64
    x=np.arange(24,dtype=float).reshape(8,3);tr=np.arange(4);a,pre=prepare(x,tr);x[4:]=1e10;b,post=prepare(x,tr)
    for key in pre:np.testing.assert_array_equal(pre[key],post[key])
